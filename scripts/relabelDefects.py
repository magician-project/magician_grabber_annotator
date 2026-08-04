#!/usr/bin/env python3
"""
Relabel annotated defect points in a raw dataset, in place, with backups.

Companion to listDefectSeverities.py — that one finds the inconsistencies, this
one fixes them.  Every point whose class/severity matches the --from filters is
rewritten with the --to values.  Frames whose points all match are still only
touched once.

Usage
-----
    python3 relabelDefects.py <dataset_dir> \\
        --fromClass "Positive Dent" --fromSeverity "Class A" \\
        --toClass   "Positive Dent" --toSeverity   "Class C" \\
        [--start 3] [--end 567] [--dry-run]

Omitting --fromClass or --fromSeverity matches any value for that field.
Omitting --start / --end leaves that end of the frame range open.
At least one of --toClass / --toSeverity must be given.

Before a file is modified it is copied to  <stem>_<YYYYmmdd-HHMMSS>.bak  next to
the original, so a botched run can be undone by moving the .bak back over the
.json.  --dry-run reports what would change and writes nothing at all.
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

FRAME_NUMBER = re.compile(r"(\d+)$")


def frameNumber(stem):
    """Frame index out of a colorFrame_0_00042 style stem, or None."""
    match = FRAME_NUMBER.search(stem)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Relabel defect class/severity in annotator json files.")
    parser.add_argument("directory", help="raw dataset directory")
    parser.add_argument("--start", type=int, default=None, help="first frame to touch (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="last frame to touch (inclusive)")
    parser.add_argument("--fromClass", default=None, help="only points of this class (default: any)")
    parser.add_argument("--fromSeverity", default=None, help="only points of this severity (default: any)")
    parser.add_argument("--toClass", default=None, help="new class")
    parser.add_argument("--toSeverity", default=None, help="new severity")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args()

    if args.toClass is None and args.toSeverity is None:
        parser.error("nothing to do: give --toClass and/or --toSeverity")

    directory = Path(args.directory)
    if not directory.is_dir():
        print("Error: %s is not a directory" % directory, file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pointsChanged = 0
    filesChanged = 0
    filesSkippedByRange = 0
    changesPerFrame = []

    for path in sorted(directory.glob("color*.json")):
        number = frameNumber(path.stem)
        if number is not None:
            if args.start is not None and number < args.start:
                filesSkippedByRange += 1
                continue
            if args.end is not None and number > args.end:
                filesSkippedByRange += 1
                continue

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print("Could not read %s : %s" % (path, e), file=sys.stderr)
            continue

        classes    = data.get("pointClasses")    or []
        severities = data.get("pointSeverities") or []

        hits = 0
        for i, defect in enumerate(classes):
            severity = severities[i] if i < len(severities) else None
            if args.fromClass is not None and defect != args.fromClass:
                continue
            if args.fromSeverity is not None and severity != args.fromSeverity:
                continue
            if args.toClass is not None:
                classes[i] = args.toClass
            if args.toSeverity is not None:
                if i < len(severities):
                    severities[i] = args.toSeverity
                else:
                    # Truncated severity list — listDefectSeverities.py shows these
                    # as MISSING.  Leave it alone rather than silently misaligning
                    # the list against pointClasses.
                    print("  %s point %d has no severity entry, class changed only"
                          % (path.name, i), file=sys.stderr)
            hits += 1

        if hits == 0:
            continue

        pointsChanged += hits
        filesChanged += 1
        changesPerFrame.append((path.name, hits))

        if not args.dry_run:
            backup = path.with_name("%s_%s.bak" % (path.stem, stamp))
            shutil.copy2(path, backup)
            data["pointClasses"] = classes
            data["pointSeverities"] = severities
            with open(path, "w") as f:
                json.dump(data, f, sort_keys=False)

    print("")
    print("=" * 70)
    print("%s%s" % (directory, "   (DRY RUN - nothing written)" if args.dry_run else ""))
    print("  %s / %s  ->  %s / %s" % (args.fromClass or "<any class>",
                                      args.fromSeverity or "<any severity>",
                                      args.toClass or "<unchanged>",
                                      args.toSeverity or "<unchanged>"))
    if args.start is not None or args.end is not None:
        print("  frame range: %s .. %s  (%d files outside it)" %
              (args.start if args.start is not None else "start",
               args.end if args.end is not None else "end",
               filesSkippedByRange))
    print("=" * 70)
    for name, hits in changesPerFrame:
        print("  %-28s %d point%s" % (name, hits, "" if hits == 1 else "s"))
    print("-" * 70)
    print("records altered : %d points in %d files" % (pointsChanged, filesChanged))
    if pointsChanged and not args.dry_run:
        print("backups written : <frame>_%s.bak" % stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
