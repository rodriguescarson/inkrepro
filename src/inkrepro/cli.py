"""inkrepro: how much of an ink prediction survives changing the checkpoint?"""

from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from .atlas import download, list_surface
from .metrics import agreement, align, to_unit

FIELDS = (
    "scroll",
    "mesh",
    "run_a",
    "run_b",
    "comparison",
    "n_valid",
    "pearson_r",
    "iou",
    "dice",
    "frac_ink_a",
    "frac_ink_b",
    "mean_abs_diff",
)


def _load(path: Path) -> np.ndarray:
    return to_unit(tifffile.imread(path))


def _classify(a: str, b: str) -> str:
    """Name what differs between two runs, so rows can be grouped honestly.

    A forward run against its own reverse is not a second opinion -- it is the
    same checkpoint reading the same layers in the other direction -- so it is
    labelled separately and must not be averaged in with cross-checkpoint rows.
    """

    tag_a, rev_a = a.removesuffix("_reverse"), a.endswith("_reverse")
    tag_b, rev_b = b.removesuffix("_reverse"), b.endswith("_reverse")
    if tag_a == tag_b:
        return "direction" if rev_a != rev_b else "identical"
    return "checkpoint" if rev_a == rev_b else "checkpoint+direction"


def _compare_dir(
    directory: Path,
    scroll: str,
    mesh: str,
    *,
    mask: np.ndarray | None,
    threshold: float,
) -> list[dict]:
    preds = sorted(p for p in directory.glob("*.tif") if p.name != "valid_mask.tif")
    rows = []
    for left, right in itertools.combinations(preds, 2):
        a, b = _load(left), _load(right)
        arrays = [a, b] if mask is None else [a, b, mask]
        aligned = align(*arrays)
        a, b = aligned[0], aligned[1]
        m = aligned[2] if mask is not None else None
        got = agreement(a, b, mask=m, threshold=threshold)
        name_a, name_b = left.stem, right.stem
        rows.append(
            {
                "scroll": scroll,
                "mesh": mesh,
                "run_a": name_a,
                "run_b": name_b,
                "comparison": _classify(name_a, name_b),
                **{
                    k: v
                    for k, v in got.as_dict().items()
                    if k in FIELDS
                },
            }
        )
    return rows


def _surface_rows(
    scroll: str,
    mesh: str,
    workdir: Path,
    *,
    tags: set[str] | None,
    threshold: float,
) -> list[dict]:
    files = list_surface(scroll, mesh)
    wanted = [
        p
        for p in files.predictions
        if tags is None or p.tag in tags
    ]
    if len(wanted) < 2:
        raise RuntimeError(
            f"{scroll}/{mesh}: need two predictions to compare, found {len(wanted)}"
        )
    workdir.mkdir(parents=True, exist_ok=True)
    for pred in wanted:
        download(pred.url, workdir / f"{pred.name}.tif")
    mask = None
    if files.mask_url is not None:
        mask_path = workdir / "valid_mask.tif"
        download(files.mask_url, mask_path)
        mask = tifffile.imread(mask_path)
    return _compare_dir(
        workdir, scroll, mesh, mask=mask, threshold=threshold
    )


def _write(rows: list[dict], out: Path | None) -> None:
    stream = sys.stdout if out is None else out.open("w", newline="")
    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    finally:
        if out is not None:
            stream.close()


def cmd_pair(args: argparse.Namespace) -> int:
    a, b = _load(args.a), _load(args.b)
    mask = tifffile.imread(args.mask) if args.mask else None
    arrays = [a, b] if mask is None else [a, b, mask]
    aligned = align(*arrays)
    got = agreement(
        aligned[0],
        aligned[1],
        mask=aligned[2] if mask is not None else None,
        threshold=args.threshold,
    )
    width = max(len(k) for k in got.as_dict())
    for key, value in got.as_dict().items():
        if isinstance(value, float):
            print(f"{key:<{width}}  {value:.6f}")
        else:
            print(f"{key:<{width}}  {value}")
    if mask is None:
        print(
            "\nnote: no --mask given, so pixels where either run wrote 0 were "
            "dropped.\n      A genuine 0.0 prediction is indistinguishable "
            "from background here."
        )
    return 0


def cmd_surface(args: argparse.Namespace) -> int:
    tags = set(args.tags.split(",")) if args.tags else None
    workdir = Path(args.workdir) if args.workdir else None
    temp = None
    if workdir is None:
        temp = tempfile.mkdtemp(prefix="inkrepro-")
        workdir = Path(temp)
    try:
        rows = _surface_rows(
            args.scroll, args.mesh, workdir, tags=tags, threshold=args.threshold
        )
    finally:
        if temp is not None and not args.keep:
            shutil.rmtree(temp, ignore_errors=True)
    _write(rows, Path(args.out) if args.out else None)
    return 0


def cmd_population(args: argparse.Namespace) -> int:
    """Stream every surface in a manifest: download, measure, delete."""

    with open(args.manifest, newline="") as handle:
        surfaces = [(r["scroll"], r["mesh"]) for r in csv.DictReader(handle)]
    if args.limit:
        surfaces = surfaces[: args.limit]
    tags = set(args.tags.split(",")) if args.tags else None

    out = Path(args.out)
    done: set[tuple[str, str]] = set()
    if out.exists() and args.resume:
        with out.open(newline="") as handle:
            done = {(r["scroll"], r["mesh"]) for r in csv.DictReader(handle)}
        print(f"resuming: {len(done)} surfaces already measured", file=sys.stderr)

    new = not out.exists() or not args.resume
    stream = out.open("w" if new else "a", newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    if new:
        writer.writeheader()
        stream.flush()

    failures = 0
    try:
        for index, (scroll, mesh) in enumerate(surfaces, 1):
            if (scroll, mesh) in done:
                continue
            workdir = Path(tempfile.mkdtemp(prefix="inkrepro-"))
            try:
                rows = _surface_rows(
                    scroll, mesh, workdir, tags=tags, threshold=args.threshold
                )
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in FIELDS})
                stream.flush()
                print(
                    f"[{index}/{len(surfaces)}] {scroll}/{mesh}: {len(rows)} pairs",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001 - one bad surface must not stop the run
                failures += 1
                print(
                    f"[{index}/{len(surfaces)}] {scroll}/{mesh}: FAILED {exc}",
                    file=sys.stderr,
                )
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
    finally:
        stream.close()
    print(f"done, {failures} surfaces failed", file=sys.stderr)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inkrepro",
        description=(
            "Measure how much an ink prediction depends on which checkpoint "
            "produced it."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="probability at or above which a pixel counts as ink (default 0.5)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pair = sub.add_parser("pair", help="compare two prediction TIFFs")
    pair.add_argument("a", type=Path)
    pair.add_argument("b", type=Path)
    pair.add_argument("--mask", type=Path, help="valid_mask.tif for this surface")
    pair.set_defaults(func=cmd_pair)

    surface = sub.add_parser(
        "surface", help="compare every published prediction for one surface"
    )
    surface.add_argument("scroll")
    surface.add_argument("mesh")
    surface.add_argument("--tags", help="comma separated checkpoint tags to keep")
    surface.add_argument("--workdir", help="keep downloads here instead of a temp dir")
    surface.add_argument("--keep", action="store_true", help="do not delete downloads")
    surface.add_argument("--out", help="write CSV here instead of stdout")
    surface.set_defaults(func=cmd_surface)

    population = sub.add_parser(
        "population", help="stream a whole manifest, one surface at a time"
    )
    population.add_argument("manifest", help="manifest.csv with scroll,mesh columns")
    population.add_argument("--out", required=True)
    population.add_argument("--tags")
    population.add_argument("--limit", type=int)
    population.add_argument(
        "--resume",
        action="store_true",
        help="skip surfaces already present in --out",
    )
    population.set_defaults(func=cmd_population)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
