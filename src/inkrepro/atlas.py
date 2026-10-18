"""Stream one surface's published ink predictions from the atlas.

The eligible-scroll atlas already publishes, for every surface, the output of
three ink checkpoints run at the same settings. Measuring how much those three
agree therefore needs no GPU and no inference: it is a download, a correlation,
and a delete.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RENDERS_REPO = "rodriguescarson/eligible-scroll-atlas-renders"
_BASE = f"https://huggingface.co/datasets/{RENDERS_REPO}/resolve/main"
_TREE = f"https://huggingface.co/api/datasets/{RENDERS_REPO}/tree/main"

# "PHerc0813-z4704_w080-20250821151723-ink9um-s43_060k_reverse.tif"
_PRED = re.compile(
    r"^(?P<scroll>[^-]+)-(?P<mesh>[^-]+)-(?P<volume>\d+)-ink9um-"
    r"(?P<tag>.+?)(?P<reverse>_reverse)?\.tif$"
)


@dataclass(frozen=True)
class Prediction:
    """One stored ink map: which checkpoint, which reading direction."""

    tag: str
    reverse: bool
    url: str

    @property
    def name(self) -> str:
        return f"{self.tag}{'_reverse' if self.reverse else ''}"


@dataclass(frozen=True)
class SurfaceFiles:
    scroll: str
    mesh: str
    predictions: tuple[Prediction, ...]
    mask_url: str | None


def _get_json(url: str, *, timeout: float = 60.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def list_surface(scroll: str, mesh: str, *, timeout: float = 60.0) -> SurfaceFiles:
    """Ask the Hub which predictions exist for one surface."""

    entries = _get_json(f"{_TREE}/{scroll}/{mesh}/ink-detection", timeout=timeout)
    preds: list[Prediction] = []
    mask_url = None
    for entry in entries:
        if entry.get("type") != "file":
            continue
        path = entry["path"]
        base = path.rsplit("/", 1)[-1]
        if base == "valid_mask.tif":
            mask_url = f"{_BASE}/{path}"
            continue
        # the -ds8.jpg previews are lossy; they must never reach a correlation
        if not base.endswith(".tif"):
            continue
        match = _PRED.match(base)
        if match is None:
            continue
        preds.append(
            Prediction(
                tag=match.group("tag"),
                reverse=bool(match.group("reverse")),
                url=f"{_BASE}/{path}",
            )
        )
    preds.sort(key=lambda p: (p.tag, p.reverse))
    return SurfaceFiles(scroll, mesh, tuple(preds), mask_url)


def download(url: str, dest: Path, *, timeout: float = 600.0) -> Path:
    """Fetch one file, writing through a temporary name.

    A truncated download must not be left somewhere a later run can mistake for
    a complete one, so the file only takes its real name once it is closed.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(
            fd, "wb"
        ) as handle:
            try:
                declared = int(response.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                declared = 0
            shutil.copyfileobj(response, handle)
        got = tmp.stat().st_size
        # Only a positive declared length is worth checking; comparing against
        # a missing header means comparing against zero, which always passes.
        if declared and got != declared:
            raise OSError(
                f"{url}: wrote {got} bytes, server declared {declared}"
            )
        if got == 0:
            raise OSError(f"{url}: wrote an empty file")
        tmp.replace(dest)
        return dest
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
