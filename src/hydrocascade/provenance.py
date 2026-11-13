"""Provenance records, written AT DOWNLOAD TIME.

Doc 10 §8: every raw download folder carries a ``PROVENANCE.md`` recording the
URL, the exact query, the date, the person and the licence. Doc 04 §2 makes
this non-negotiable, and FR-04 requires an automated check that 100% of catalog
rows carry a source URL and an accessor.

The template here matches Doc 10 §8 field for field so a reviewer can diff them.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

TEMPLATE = """\
Source:            {source}
URL:               {url}
Exact query:       {query}
Downloaded:        {downloaded}
By:                {by}
Licence:           {licence}
Citation:          {citation}
File checksum:     {checksums}
Notes:             {notes}
"""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 so multi-GB rasters do not have to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def default_accessor() -> str:
    """Who is running this. Set ``HYDROCASCADE_USER`` to your name."""
    return os.environ.get("HYDROCASCADE_USER") or os.environ.get("USERNAME") or "unknown"


@dataclass
class Provenance:
    """One download, fully sourced."""

    source: str
    url: str
    query: str
    licence: str
    citation: str = ""
    notes: str = ""
    by: str = field(default_factory=default_accessor)
    downloaded: str = field(default_factory=lambda: date.today().isoformat())

    def write(self, folder: Path, files: list[Path] | None = None) -> Path:
        """Write ``PROVENANCE.md`` into ``folder``, checksumming ``files``."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        if files is None:
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.name != "PROVENANCE.md")
        lines = [f"sha256:{sha256(p)}  {p.name}" for p in files if p.is_file()]
        checksums = ("\n" + " " * 19).join(lines) if lines else "(no files)"

        text = TEMPLATE.format(
            source=self.source,
            url=self.url,
            query=self.query,
            downloaded=self.downloaded,
            by=self.by,
            licence=self.licence,
            citation=self.citation or "(none recorded)",
            checksums=checksums,
            notes=self.notes or "(none)",
        )
        out = folder / "PROVENANCE.md"
        out.write_text(text, encoding="utf-8")
        return out


def check_raw_tree(raw_root: Path) -> list[Path]:
    """Return download folders that are missing a ``PROVENANCE.md``.

    A download folder is any directory holding at least one regular file.
    Used by ``hydrocascade check`` so a missing record fails loudly and early.
    """
    offenders: list[Path] = []
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        return offenders
    for folder in raw_root.rglob("*"):
        if not folder.is_dir():
            continue
        has_files = any(p.is_file() and p.name != "PROVENANCE.md" for p in folder.iterdir())
        if has_files and not (folder / "PROVENANCE.md").is_file():
            offenders.append(folder)
    return offenders
