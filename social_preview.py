"""Screenshot the rendered map -> docs/social-preview.png (1200 x 630 link card).

The card is the interactive map itself rather than a separate static plot, so a
shared link previews what the reader will actually get. The map is WebGL, so this
drives the installed Chrome through Playwright (no bundled browser is downloaded).
The page fetches its data from files beside it, so it's served over http.

    uv run render.py
    uv run social_preview.py
"""

import argparse
import functools
import http.server
import io
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

from common import write_bytes_atomic
from render import DOCS, OUTPUT, SOCIAL_PREVIEW, ZOOM_SPEED

OG_SIZE = (1200, 630)
# deck.gl zooms by a function of delta * speed, so scaling the wheel delta against
# the page's zoom speed keeps --zoom meaning the same framing if that speed changes.
# The defaults below were chosen with 50 px ticks at deck.gl's own speed, 0.01.
WHEEL_DELTA = 50 * 0.01 / ZOOM_SPEED
SCALE = 2  # capture at 2x and downsample, so labels and points stay crisp
SETTLE_MS = 8000

# Controls that do nothing in a still image.
HIDE_CONTROLS = """
for (const id of ['search-container', 'colormap-selector-container']) {
  const el = document.getElementById(id);
  if (el) (el.closest('.container-box') || el).style.display = 'none';
}
"""

# Link previews print og:title under the image, so the card can spend the space
# on the map instead.
HIDE_TITLE = """
const t = document.getElementById('title-container');
if (t) (t.closest('.container-box') || t).style.display = 'none';
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    # Wheel ticks at the centre: the card is wider than the map's default framing,
    # which otherwise leaves the cloud small in the middle. Chosen by eye from 0-6:
    # 4 still fits the whole cloud and shows four region names; 3 or fewer show
    # two, and 5 crops the top and bottom.
    parser.add_argument("--zoom", type=int, default=4)
    parser.add_argument("--pan", type=int, default=0, help="drag the map down N px")
    parser.add_argument("--title", action="store_true", help="keep the title panel")
    parser.add_argument("--out", type=Path, default=SOCIAL_PREVIEW)
    args = parser.parse_args()

    handler = functools.partial(QuietHandler, directory=str(DOCS))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/{OUTPUT.name}"
    w, h = OG_SIZE
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome")
            page = browser.new_page(
                viewport={"width": w, "height": h}, device_scale_factor=SCALE
            )
            page.goto(url, wait_until="load", timeout=120_000)
            page.wait_for_load_state("networkidle", timeout=45_000)
            page.wait_for_timeout(SETTLE_MS)
            for _ in range(args.zoom):
                page.mouse.move(w / 2, h / 2)
                page.mouse.wheel(0, -WHEEL_DELTA)
                page.wait_for_timeout(150)
            if args.pan:
                page.mouse.move(w / 2, h / 2)
                page.mouse.down()
                page.mouse.move(w / 2, h / 2 + args.pan, steps=8)
                page.mouse.up()
            page.mouse.move(0, 0)  # park the cursor off the points: no hover card
            page.evaluate(HIDE_CONTROLS)
            if not args.title:
                page.evaluate(HIDE_TITLE)
            page.wait_for_timeout(4000)
            shot = page.screenshot()
            browser.close()
    finally:
        server.shutdown()

    with Image.open(io.BytesIO(shot)) as im:
        card = im.convert("RGB").resize(OG_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    card.save(buf, format="PNG", optimize=True)
    write_bytes_atomic(buf.getvalue(), args.out)
    print(f"wrote {args.out} {card.size} ({args.out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
