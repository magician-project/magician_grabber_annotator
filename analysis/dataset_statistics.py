#!/usr/bin/env python3
"""
Scan colorFrame*.json files across directories and report:
  - Total frames / annotated frames per directory (and aggregate)
  - Defect class x severity frequency table
  - Tile statistics: total tiles per frame, dirty tiles per class/severity,
    clean tiles (total - dirty)

Tiling assumptions
------------------
  * Image resolution is fixed at IMAGE_W x IMAGE_H pixels (default 2464x2056).
  * Tiles are extracted with a sliding window of TILE_SIZE x TILE_SIZE pixels
    and stride TILE_STEP pixels.
  * Tiles per frame = floor((W - TILE_SIZE) / TILE_STEP + 1)
                    x floor((H - TILE_SIZE) / TILE_STEP + 1)
    With defaults (2464x2056, 48x48 step 3) this is 806 x 670 = 540,020 tiles.
  * A defect point dirties the tile whose top-left corner is the largest valid
    grid position <= the point coordinates.
  * Multiple defect points falling in the same tile count as ONE dirty tile
    for that (class, severity) pair — deduplicated within each frame.

Usage
-----
    python defect_stats.py [options] <dir1> [<dir2> ...]

Options:
    --tile-size N     Tile size in pixels        (default: 48)
    --tile-step S     Tile stride in pixels       (default: 3)
    --img-width  W    Image width  in pixels      (default: 2464)
    --img-height H    Image height in pixels      (default: 2056)
"""

import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path

IGNORED_CLASSES = {"Unknown", "Suspicious"}


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

def tiles_per_axis(length: int, tile_size: int, tile_step: int) -> int:
    if length < tile_size:
        return 0
    return (length - tile_size) // tile_step + 1


def tiles_per_frame(img_w: int, img_h: int, tile_size: int, tile_step: int) -> int:
    return tiles_per_axis(img_w, tile_size, tile_step) * \
           tiles_per_axis(img_h, tile_size, tile_step)


def dirty_tiles_for_points(
    points: list,
    img_w: int, img_h: int,
    tile_size: int, tile_step: int,
) -> set:
    """Return the deduplicated set of tile origins (tx, ty) hit by the given points."""
    max_tx = (tiles_per_axis(img_w, tile_size, tile_step) - 1) * tile_step
    max_ty = (tiles_per_axis(img_h, tile_size, tile_step) - 1) * tile_step
    dirty = set()
    for px, py in points:
        tx = min((int(px) // tile_step) * tile_step, max_tx)
        ty = min((int(py) // tile_step) * tile_step, max_ty)
        dirty.add((tx, ty))
    return dirty


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def parse_file(json_path: Path) -> dict | None:
    """Return parsed defect data from one JSON file, or None on error."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read {json_path}: {e}", file=sys.stderr)
        return None

    classes    = data.get("pointClasses",    [])
    severities = data.get("pointSeverities", [])
    points     = data.get("pointClicks",     [])

    defects = []
    for i, cls in enumerate(classes):
        if cls in IGNORED_CLASSES:
            continue
        sev = (severities[i] if i < len(severities) else None) or "<no severity>"
        pt  = tuple(points[i]) if i < len(points) else None
        defects.append((cls, sev, pt))

    return {"annotated": len(defects) > 0, "defects": defects}


def scan_directory(
    dir_path: Path,
    img_w: int, img_h: int,
    tile_size: int, tile_step: int,
) -> dict:
    n_tiles = tiles_per_frame(img_w, img_h, tile_size, tile_step)

    total_frames     = 0
    annotated_frames = 0
    counts  = defaultdict(lambda: defaultdict(int))   # {cls: {sev: point_count}}
    dirty   = defaultdict(lambda: defaultdict(int))   # {cls: {sev: dirty_tile_count}}

    for json_path in sorted(dir_path.glob("colorFrame*.json")):
        result = parse_file(json_path)
        if result is None:
            continue

        total_frames += 1
        if result["annotated"]:
            annotated_frames += 1

        # accumulate point counts and dirty tiles, deduplicating per (cls, sev) per frame
        groups = defaultdict(list)
        for cls, sev, pt in result["defects"]:
            counts[cls][sev] += 1
            if pt is not None:
                groups[(cls, sev)].append(pt)

        for (cls, sev), pts in groups.items():
            dirty[cls][sev] += len(
                dirty_tiles_for_points(pts, img_w, img_h, tile_size, tile_step)
            )

    return {
        "total_frames":     total_frames,
        "annotated_frames": annotated_frames,
        "tiles_per_frame":  n_tiles,
        "total_tiles":      n_tiles * total_frames,
        "counts":           counts,
        "dirty":            dirty,
    }


def merge_stats(a: dict, b: dict) -> dict:
    def merge_nested(x, y):
        m = defaultdict(lambda: defaultdict(int))
        for src in (x, y):
            for k, inner in src.items():
                for kk, n in inner.items():
                    m[k][kk] += n
        return m

    assert a["tiles_per_frame"] == b["tiles_per_frame"], \
        "Cannot merge stats computed with different tile parameters."

    return {
        "total_frames":     a["total_frames"]     + b["total_frames"],
        "annotated_frames": a["annotated_frames"] + b["annotated_frames"],
        "tiles_per_frame":  a["tiles_per_frame"],
        "total_tiles":      a["total_tiles"]      + b["total_tiles"],
        "counts":           merge_nested(a["counts"], b["counts"]),
        "dirty":            merge_nested(a["dirty"],  b["dirty"]),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _col(value, width: int) -> str:
    return str(value).ljust(width)


def print_stats(
    stats: dict,
    img_w: int, img_h: int,
    tile_size: int, tile_step: int,
    indent: int = 4,
):
    pad = " " * indent

    total_frames     = stats["total_frames"]
    annotated_frames = stats["annotated_frames"]
    tiles_per_frm    = stats["tiles_per_frame"]
    total_tiles      = stats["total_tiles"]
    counts           = stats["counts"]
    dirty            = stats["dirty"]

    print(f"{pad}Frames      : {total_frames:,}  "
          f"|  Annotated: {annotated_frames:,}  "
          f"|  Unannotated: {total_frames - annotated_frames:,}")
    print(f"{pad}Image size  : {img_w} x {img_h} px")
    print(f"{pad}Tile grid   : {tile_size}x{tile_size} step {tile_step}  "
          f"-> {tiles_per_axis(img_w, tile_size, tile_step)} x "
          f"{tiles_per_axis(img_h, tile_size, tile_step)} "
          f"= {tiles_per_frm:,} tiles/frame")
    print(f"{pad}Total tiles : {total_tiles:,}  ({total_frames:,} frames x {tiles_per_frm:,})")
    print()

    if not counts:
        print(f"{pad}(no defects found)")
        return

    all_sevs = sorted({sev for sevs in counts.values() for sev in sevs})

    cls_w = max(len("Class"), max(len(c) for c in counts)) + 2
    num_w = max(14, max(len(s) for s in all_sevs) + 2)

    header = (
        f"{pad}{_col('Class', cls_w)}"
        + "".join(_col(s, num_w) for s in all_sevs)
        + _col("Total pts", num_w)
        + _col("Dirty tiles", num_w)
        + "Clean tiles"
    )
    sep = pad + "-" * (len(header) - indent)
    print(header)
    print(sep)

    grand_pts   = 0
    grand_dirty = 0
    tot_sev_pts = defaultdict(int)
    tot_sev_dty = defaultdict(int)

    for cls, sevs in sorted(counts.items()):
        row_pts   = sum(sevs.values())
        row_dirty = sum(dirty.get(cls, {}).get(s, 0) for s in all_sevs)
        row_clean = total_tiles - row_dirty
        grand_pts   += row_pts
        grand_dirty += row_dirty

        row = f"{pad}{_col(cls, cls_w)}"
        for sev in all_sevs:
            n = sevs.get(sev, 0)
            tot_sev_pts[sev] += n
            tot_sev_dty[sev] += dirty.get(cls, {}).get(sev, 0)
            row += _col(n, num_w)
        row += _col(row_pts, num_w)
        row += _col(row_dirty, num_w)
        row += f"{row_clean:,}"
        print(row)

    print(sep)
    totals_row = f"{pad}{_col('TOTAL', cls_w)}"
    for sev in all_sevs:
        totals_row += _col(tot_sev_pts[sev], num_w)
    totals_row += _col(grand_pts, num_w)
    totals_row += _col(grand_dirty, num_w)
    totals_row += f"{total_tiles - grand_dirty:,}"
    print(totals_row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Report defect and tile statistics from colorFrame JSON files."
    )
    parser.add_argument("directories", nargs="+", help="Directories to scan")
    parser.add_argument("--tile-size",  type=int, default=48,   metavar="N",
                        help="Tile size in pixels (default: 48)")
    parser.add_argument("--tile-step",  type=int, default=3,    metavar="S",
                        help="Tile stride in pixels (default: 3)")
    parser.add_argument("--img-width",  type=int, default=1232, metavar="W", #2464
                        help="Image width in pixels (default: 1232)")
    parser.add_argument("--img-height", type=int, default=1028, metavar="H",
                        help="Image height in pixels (default: 1028)")
    args = parser.parse_args()

    tile_size = args.tile_size
    tile_step = args.tile_step
    img_w     = args.img_width
    img_h     = args.img_height

    aggregate = None

    for dir_str in args.directories:
        dir_path = Path(dir_str)
        if not dir_path.is_dir():
            print(f"Warning: '{dir_path}' is not a valid directory, skipping.",
                  file=sys.stderr)
            continue

        stats = scan_directory(dir_path, img_w, img_h, tile_size, tile_step)
        aggregate = stats if aggregate is None else merge_stats(aggregate, stats)

        print(f"\n{'='*72}")
        print(f"  Directory: {dir_path.resolve()}")
        print(f"{'='*72}")
        print_stats(stats, img_w, img_h, tile_size, tile_step)

    if aggregate is not None and len(args.directories) > 1:
        print(f"\n{'='*72}")
        print(f"  AGGREGATE (all directories)")
        print(f"{'='*72}")
        print_stats(aggregate, img_w, img_h, tile_size, tile_step)


if __name__ == "__main__":
    main()
