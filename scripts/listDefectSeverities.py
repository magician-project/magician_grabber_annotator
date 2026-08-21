#!/usr/bin/env python3
"""
List every annotated defect point of a raw dataset as  frame / defect / severity.

Written to hunt annotation inconsistencies, e.g. the same defect class carrying
different severities across frames (FORTH_PORTAPISW_WELDPOSNEG_650 has
"Negative Dent" marked both Class A and Class C).

Usage
-----
    python3 listDefectSeverities.py <dataset_dir> [<dataset_dir> ...]
    python3 listDefectSeverities.py --csv <dataset_dir> > defects.csv

Reads the colorFrame*.json files written by mga/wx_annotator.py; nothing is modified.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def collect(directory):
    """Return one (frame, defect, severity, x, y, source) row per annotated point."""
    rows = []
    for path in sorted(Path(directory).glob("color*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print("Could not read %s : %s" % (path, e), file=sys.stderr)
            continue

        classes    = data.get("pointClasses")     or []
        severities = data.get("pointSeverities")  or []
        points     = data.get("pointClicks")      or []
        sources    = data.get("pointSources")     or []

        for i, defect in enumerate(classes):
            # The lists are written in lockstep, but a hand-edited json can be
            # short — flag that rather than silently dropping the point.
            severity = severities[i] if i < len(severities) else "MISSING"
            point    = points[i]     if i < len(points)     else [-1, -1]
            source   = sources[i]    if i < len(sources)    else "?"
            rows.append((path.stem, defect, severity, point[0], point[1], source))
    return rows


def report(directory, rows):
    print("")
    print("=" * 78)
    print("%s : %d annotated points" % (directory, len(rows)))
    print("=" * 78)

    header = "%-24s %-18s %-10s %9s %9s  %s" % \
             ("FRAME", "DEFECT", "SEVERITY", "X", "Y", "SOURCE")
    print(header)
    print("-" * 78)
    for frame, defect, severity, x, y, source in sorted(rows, key=lambda r: (r[1], r[2], r[0])):
        print("%-24s %-18s %-10s %9.1f %9.1f  %s" % (frame, defect, severity, x, y, source))

    # The point of the exercise: which classes disagree with themselves.
    perClass = defaultdict(lambda: defaultdict(int))
    for _, defect, severity, _, _, _ in rows:
        perClass[defect][severity] += 1

    print("")
    print("Severity spread per defect class")
    print("-" * 78)
    for defect in sorted(perClass):
        counts = perClass[defect]
        breakdown = ", ".join("%s=%d" % (s, counts[s]) for s in sorted(counts))
        flag = "  <-- INCONSISTENT" if len(counts) > 1 else ""
        print("%-18s %s%s" % (defect, breakdown, flag))


def main():
    args = sys.argv[1:]
    asCsv = "--csv" in args
    dirs = [a for a in args if a != "--csv"]

    if not dirs:
        print(__doc__)
        return 1

    for directory in dirs:
        if not Path(directory).is_dir():
            print("Error: %s is not a directory" % directory, file=sys.stderr)
            return 1

    if asCsv:
        writer = csv.writer(sys.stdout)
        writer.writerow(["dataset", "frame", "defect", "severity", "x", "y", "source"])
        for directory in dirs:
            for row in collect(directory):
                writer.writerow([directory] + list(row))
    else:
        for directory in dirs:
            report(directory, collect(directory))

    return 0


if __name__ == "__main__":
    sys.exit(main())
