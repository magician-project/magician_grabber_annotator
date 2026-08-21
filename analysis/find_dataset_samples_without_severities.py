#!/usr/bin/env python3
"""
Find colorFrame*.json files that contain defect points with missing severity.
Ignores points with class "Unknown" or "Suspicious".

Usage:
    python find_missing_severities.py /path/to/dir1 /path/to/dir2 ...
"""

import json
import sys
from pathlib import Path

IGNORED_CLASSES = {"Unknown", "Suspicious"}


def get_classes_missing_severity(json_path: Path) -> list[str]:
    """Return a deduplicated list of defect classes that have at least one point
    with a missing severity, excluding classes in IGNORED_CLASSES."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read {json_path}: {e}", file=sys.stderr)
        return []

    classes    = data.get("pointClasses", [])
    severities = data.get("pointSeverities", [])

    missing_classes = []
    for i, cls in enumerate(classes):
        if cls in IGNORED_CLASSES:
            continue

        # Missing if the severity list is shorter than the class list,
        # or the entry is None / empty string.
        severity = severities[i] if i < len(severities) else None
        if not severity and cls not in missing_classes:
            missing_classes.append(cls)

    return missing_classes


def find_files_missing_severities(directories: list[str]) -> list[tuple[Path, list[str]]]:
    results = []
    for dir_str in directories:
        dir_path = Path(dir_str)
        if not dir_path.is_dir():
            print(f"Warning: '{dir_path}' is not a valid directory, skipping.", file=sys.stderr)
            continue

        for json_path in sorted(dir_path.glob("colorFrame*.json")):
            missing_classes = get_classes_missing_severity(json_path)
            if missing_classes:
                results.append((json_path.resolve(), missing_classes))

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_missing_severities.py <dir1> [<dir2> ...]")
        sys.exit(1)

    missing = find_files_missing_severities(sys.argv[1:])

    if not missing:
        print("No files with missing severities found.")
    else:
        for path, classes in missing:
            print(f"{path}  [{', '.join(classes)}]")


if __name__ == "__main__":
    main()
