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


def cmd_figure(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    from .figures import BOTH, NEITHER, ONLY_A, ONLY_B, build_panels, densest_ink_window

    a, b = _load(args.a), _load(args.b)
    mask = (
        tifffile.imread(args.mask)
        if args.mask
        else ((a > 0) & (b > 0)).astype(np.uint8)
    )
    a, b, mask = align(a, b, mask)
    panels = build_panels(a, b, mask, threshold=args.threshold)
    got = agreement(a, b, mask=mask, threshold=args.threshold)

    zoom = args.zoom
    zy, zx = densest_ink_window(a, b, mask, zoom, threshold=args.threshold)
    za, zb, zmask = (
        a[zy : zy + zoom, zx : zx + zoom],
        b[zy : zy + zoom, zx : zx + zoom],
        mask[zy : zy + zoom, zx : zx + zoom],
    )
    zpanels = build_panels(za, zb, zmask, threshold=args.threshold, target_long_edge=zoom)

    label_a = args.label_a or Path(args.a).stem
    label_b = args.label_b or Path(args.b).stem
    total_ink = panels.n_both + panels.n_only_a + panels.n_only_b
    share = (panels.n_both / total_ink) if total_ink else float("nan")

    # A surface strip can be 1:4 or wider. Side-by-side panels would then be
    # unreadable slivers, so wide surfaces get one full-width row each.
    height, width = panels.a.shape
    wide = width / max(height, 1) > 3.0
    strip_h = 16.0 * height / max(width, 1)

    if wide:
        fig = plt.figure(figsize=(16.0, 3 * strip_h + 6.4))
        grid = fig.add_gridspec(
            4,
            3,
            height_ratios=[strip_h, strip_h, strip_h, 5.6],
            hspace=0.28,
            wspace=0.06,
        )
        ax_a = fig.add_subplot(grid[0, :])
        ax_b = fig.add_subplot(grid[1, :])
        ax_g = fig.add_subplot(grid[2, :])
    else:
        fig = plt.figure(figsize=(16.0, 11.4))
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.16, wspace=0.06)
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_g = fig.add_subplot(grid[0, 2])
    ax_za = fig.add_subplot(grid[-1, 0])
    ax_zb = fig.add_subplot(grid[-1, 1])
    ax_zg = fig.add_subplot(grid[-1, 2])

    for ax, img, title in (
        (ax_a, panels.a, label_a),
        (ax_b, panels.b, label_b),
        (ax_za, zpanels.a, f"{label_a} — detail"),
        (ax_zb, zpanels.b, f"{label_b} — detail"),
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=11)

    ax_g.imshow(panels.agreement, interpolation="nearest")
    ax_g.set_title(
        f"agreement over the whole surface — IoU {got.iou:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax_zg.imshow(zpanels.agreement, interpolation="nearest")
    ax_zg.set_title(f"detail, {zoom}x{zoom} px, native resolution", fontsize=11)

    # mark the detail window on the full-surface agreement panel
    f = panels.factor
    ax_g.add_patch(
        Rectangle(
            (zx / f, zy / f),
            zoom / f,
            zoom / f,
            fill=False,
            edgecolor="#f0a202",
            linewidth=2.0,
        )
    )
    for spine in ax_zg.spines.values():
        spine.set_edgecolor("#f0a202")
        spine.set_linewidth(2.0)

    for ax in (ax_a, ax_b, ax_g, ax_za, ax_zb, ax_zg):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(
        handles=[
            Patch(facecolor=BOTH, label=f"both called ink ({panels.n_both:,} px)"),
            Patch(facecolor=ONLY_A, label=f"only {label_a} ({panels.n_only_a:,})"),
            Patch(facecolor=ONLY_B, label=f"only {label_b} ({panels.n_only_b:,})"),
            Patch(facecolor=NEITHER, edgecolor="#bbb", label="neither"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.032),
        ncol=4,
        fontsize=10,
        frameon=False,
    )

    headline = args.title or (
        f"Same surface, same settings, different checkpoint — "
        f"only {share:.0%} of the ink is called by both"
    )
    fig.suptitle(headline, fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.012,
        f"ink = p >= {args.threshold};  {got.n_valid:,} valid px;  "
        f"Pearson r {got.pearson_r:.3f};  IoU {got.iou:.3f};  Dice {got.dice:.3f};  "
        f"both/(both+either) = {share:.3f}"
        + (";  mask: valid_mask.tif" if args.mask else ";  mask: both runs nonzero"),
        ha="center",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(top=0.93, bottom=0.085, left=0.012, right=0.988)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(
        f"  both {panels.n_both:,} | only {label_a} {panels.n_only_a:,} "
        f"| only {label_b} {panels.n_only_b:,} | reproducible share {share:.4f}"
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

    figure = sub.add_parser(
        "figure", help="render what two checkpoints disagree about"
    )
    figure.add_argument("a", type=Path)
    figure.add_argument("b", type=Path)
    figure.add_argument("--mask", type=Path)
    figure.add_argument("--out", required=True)
    figure.add_argument("--label-a")
    figure.add_argument("--label-b")
    figure.add_argument("--title")
    figure.add_argument("--zoom", type=int, default=1024)
    figure.add_argument("--dpi", type=int, default=130)
    figure.set_defaults(func=cmd_figure)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
