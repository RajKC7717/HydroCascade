"""HTTP download helper: resumable, verified, provenance-aware.

Every open-source fetcher in this package goes through :func:`download`. It is
deliberately boring — the interesting property is that it never leaves a
half-written file behind that a later run would mistake for a complete one.
"""

from __future__ import annotations

from pathlib import Path

import requests

from ..logging_setup import get

log = get("ingest.fetch")

_HEADERS = {"User-Agent": "HydroCascade/0.2 (VIT Pune CS-C1-04; research)"}
CHUNK = 1 << 20


def download(url: str, dest: Path, *, force: bool = False, timeout: int = 120,
             session: requests.Session | None = None) -> Path:
    """Download ``url`` to ``dest``, skipping if already present.

    Writes to ``dest.part`` and renames on success, so an interrupted download
    can never be mistaken for a complete file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and dest.stat().st_size > 0 and not force:
        log.info("cached  %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    sess = session or requests.Session()
    log.info("GET     %s", url)

    with sess.get(url, stream=True, timeout=timeout, headers=_HEADERS) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        step = max(total // 10, 1) if total else 50 * CHUNK
        next_mark = step
        with open(part, "wb") as fh:
            for block in r.iter_content(CHUNK):
                if not block:
                    continue
                fh.write(block)
                done += len(block)
                if done >= next_mark:
                    if total:
                        log.info("        %5.1f%%  %.0f/%.0f MB", 100 * done / total,
                                 done / 1e6, total / 1e6)
                    else:
                        log.info("        %.0f MB", done / 1e6)
                    next_mark += step

    if total and part.stat().st_size != total:
        part.unlink(missing_ok=True)
        raise IOError(f"short read for {url}: got {part.stat().st_size}, expected {total}")

    part.replace(dest)
    log.info("saved   %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def head_ok(url: str, timeout: int = 30) -> bool:
    """True if the URL is reachable. Used to fail fast with a clear message."""
    try:
        r = requests.head(url, timeout=timeout, headers=_HEADERS, allow_redirects=True)
        return r.status_code < 400
    except requests.RequestException:
        return False


def unzip(archive: Path, dest_dir: Path, *, members: list[str] | None = None) -> list[Path]:
    """Extract ``archive`` into ``dest_dir``; returns the extracted paths."""
    import zipfile

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        names = members if members is not None else zf.namelist()
        for name in names:
            if name.endswith("/"):
                continue
            target = dest_dir / Path(name).name
            if not target.is_file():
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            out.append(target)
    log.info("unzip   %s -> %d files", archive.name, len(out))
    return out
