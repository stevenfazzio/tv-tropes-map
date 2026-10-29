"""Download one namespace of the June 2023 TvTroper dump without the rest of the zip.

The dump is a single 20.7 GB zip with one jsonl member per namespace. Its central
directory is read with HTTP range requests, then only the requested member's
compressed bytes are downloaded into a .part file. Resume is by .part size. The
CRC is checked against the zip's own record before the file is moved into place,
so a file at the final path is a complete, verified member. Nothing is fetched
from tvtropes.org.

    uv run fetch_dump.py            # Main/ (trope pages, 2.2 GB)
    uv run fetch_dump.py --list     # the largest members, with sizes
    uv run fetch_dump.py Laconic
"""

import argparse
import io
import json
import os
import struct
import time
import zipfile
import zlib

import requests

from common import DUMP_URL, dump_member_path, dump_meta_path, write_bytes_atomic

CHUNK = 1 << 20
MAX_ATTEMPTS = 8

session = requests.Session()
session.headers["User-Agent"] = "tv-tropes-map/0.1 (personal research project)"


class HttpRangeFile(io.RawIOBase):
    """Just enough of a seekable file for zipfile to read the central directory."""

    def __init__(self, url: str):
        resp = session.head(url, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        self.url = url
        self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        resp = session.get(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60
        )
        resp.raise_for_status()
        self.pos += len(resp.content)
        return resp.content


def data_offset(remote: HttpRangeFile, info: zipfile.ZipInfo) -> int:
    # The local header repeats the name and carries its own extra field, whose length
    # can differ from the central directory's, so it has to be read to find the data.
    remote.seek(info.header_offset)
    header = remote.read(30)
    assert header[:4] == b"PK\x03\x04", f"no local header at {info.header_offset}"
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_len + extra_len


def download_range(start: int, length: int, part_path) -> None:
    for attempt in range(MAX_ATTEMPTS):
        have = part_path.stat().st_size if part_path.exists() else 0
        assert have <= length, (
            f"{part_path} is larger than the member ({have} > {length})"
        )
        if have == length:
            return
        print(
            f"downloading {(length - have) / 1e9:.2f} GB ({have / 1e9:.2f} GB already on disk)"
        )
        try:
            headers = {"Range": f"bytes={start + have}-{start + length - 1}"}
            with session.get(
                DUMP_URL, headers=headers, stream=True, timeout=60
            ) as resp:
                assert resp.status_code == 206, (
                    f"expected 206 Partial Content, got {resp.status_code}"
                )
                next_report = have + length // 20
                with open(part_path, "ab") as f:
                    for chunk in resp.iter_content(CHUNK):
                        f.write(chunk)
                        have += len(chunk)
                        if have >= next_report:
                            print(
                                f"  {have / 1e9:.2f} / {length / 1e9:.2f} GB",
                                flush=True,
                            )
                            next_report += length // 20
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            wait = min(2**attempt * 5, 120)
            print(f"  attempt {attempt + 1}: {e}, resuming in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts; rerun to resume")


def verify(part_path, info: zipfile.ZipInfo) -> None:
    """Decompress the whole member and compare its size and CRC with the zip's record."""
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    crc, size = 0, 0
    with open(part_path, "rb") as f:
        while chunk := f.read(1 << 24):
            out = decompressor.decompress(chunk)
            crc, size = zlib.crc32(out, crc), size + len(out)
    out = decompressor.flush()
    crc, size = zlib.crc32(out, crc), size + len(out)
    assert size == info.file_size, (
        f"decompressed {size} bytes, zip records {info.file_size}"
    )
    assert crc == info.CRC, f"CRC {crc:08x} != recorded {info.CRC:08x}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("namespace", nargs="?", default="Main")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    remote = HttpRangeFile(DUMP_URL)
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
    print(f"dump: {remote.size / 1e9:.2f} GB, {len(infos)} members")

    if args.list:
        for info in sorted(infos, key=lambda i: -i.compress_size)[:60]:
            print(
                f"  {info.filename:60s} {info.compress_size / 1e6:9.1f} MB zipped  {info.file_size / 1e6:9.1f} MB raw"
            )
        return

    final = dump_member_path(args.namespace)
    if final.exists():
        print(f"{final} already downloaded and verified")
        return

    name = f"pages/ns_{args.namespace}_scraped.jsonl"
    info = next((i for i in infos if i.filename == name), None)
    assert info is not None, f"no member {name}; try --list"
    assert info.compress_type == zipfile.ZIP_DEFLATED, (
        f"unsupported compression {info.compress_type}"
    )

    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_suffix(".deflate.part")
    download_range(data_offset(remote, info), info.compress_size, part)

    print("verifying CRC")
    verify(part, info)
    meta = {
        "source": DUMP_URL,
        "member": info.filename,
        "compressed_bytes": info.compress_size,
        "uncompressed_bytes": info.file_size,
        "crc32": f"{info.CRC:08x}",
        "member_date": "-".join(f"{x:02d}" for x in info.date_time[:3]),
    }
    write_bytes_atomic(
        json.dumps(meta, indent=2).encode(), dump_meta_path(args.namespace)
    )
    os.replace(part, final)
    print(f"wrote {final} ({info.compress_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
