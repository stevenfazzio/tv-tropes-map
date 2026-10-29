"""Shared paths and helpers for the TV Tropes datamap pipeline."""

import json
import os
import stat
import tempfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

# Development corpus: the June 2023 TvTroper dump (raw HTML of ~651k pages, one zip
# member per namespace). The published map will come from a fresh crawl, parsed by
# the same code. The dump is labelled Apache-2.0 on Hugging Face, but its content is
# TV Tropes text and stays CC BY-NC-SA 3.0.
DUMP_URL = "https://huggingface.co/datasets/RyokoExtra/TvTroper/resolve/main/pages.zip"
DUMP_SNAPSHOT = "2023-06"

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW_DUMP = DATA / "raw" / f"tvtroper-{DUMP_SNAPSHOT}"
MAIN_PAGES_PARQUET = DATA / "main_pages.parquet"
TROPES_PARQUET = DATA / "tropes.parquet"


def dump_member_path(namespace: str) -> Path:
    """A namespace's compressed bytes, exactly as stored in the zip (raw deflate)."""
    return RAW_DUMP / f"ns_{namespace}.deflate"


def dump_meta_path(namespace: str) -> Path:
    return RAW_DUMP / f"ns_{namespace}.json"


def iter_dump_pages(namespace: str) -> Iterator[tuple[str, str]]:
    """Yield (url, html) for each page of a downloaded dump namespace.

    The member is decompressed on the fly, so the 9 GB of HTML behind Main/ never
    has to sit on disk. Each jsonl line is a two-element list, [url, html].
    """
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    pending = b""
    with open(dump_member_path(namespace), "rb") as f:
        while chunk := f.read(1 << 22):
            pending += decompressor.decompress(chunk)
            *lines, pending = pending.split(b"\n")
            for line in lines:
                if line:
                    url, html = json.loads(line)
                    yield url, html
    pending += decompressor.flush()
    if pending.strip():
        url, html = json.loads(pending)
        yield url, html


def _match_or_default_mode(tmp_path: Path, output_path: Path) -> None:
    # mkstemp creates 0600 and os.replace preserves the source mode.
    if output_path.exists():
        os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
    else:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)


def write_bytes_atomic(data: bytes, output_path: Path) -> None:
    """Write so output_path only ever holds a complete file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _match_or_default_mode(Path(tmp), output_path)
        os.replace(tmp, output_path)
    except BaseException:
        os.unlink(tmp)
        raise


def write_parquet_safely(df, output_path: Path) -> None:
    """Write df to a temp file, verify rows and schema from the footer, then swap in."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        _match_or_default_mode(Path(tmp), output_path)
        meta = pq.read_metadata(tmp)
        assert meta.num_rows == len(df), f"row count {meta.num_rows} != {len(df)}"
        # to_arrow_schema(): the Parquet schema's .names are leaf names, so list
        # columns would show up as "element" rather than their top-level name.
        missing = set(df.columns) - set(meta.schema.to_arrow_schema().names)
        assert not missing, f"columns missing on disk: {missing}"
        os.replace(tmp, output_path)
    except BaseException:
        os.unlink(tmp)
        raise
