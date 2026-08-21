#!/bin/bash
# Usage: ./countDefectsAndSeverities.sh <dataset_dir> [<dataset_dir> ...] [-- <extra dataset_statistics args>]
# Example: ./countDefectsAndSeverities.sh /media/ammar/games2/Datasets/Magician/altinay_test_data
# Example: ./countDefectsAndSeverities.sh ds_a ds_b -- --tile-size 64 --tile-step 4
#
# Counts defect points per class x severity across the colorFrame*.json files of
# one or more raw datasets, by invoking analysis.dataset_statistics from the repo root.
# With more than one directory an extra AGGREGATE table is printed at the end.
# Read-only: nothing in the datasets is modified.
#
# Note: the point counts are exact, but the "dirty/clean tiles" columns depend on
# the --img-width/--img-height/--tile-* values. Pass them after -- if this dataset
# was not captured at the analysis.dataset_statistics defaults.

if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_dir> [<dataset_dir> ...] [-- <extra dataset_statistics args>]"
  echo "Example: $0 /path/to/dataset"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATS="$REPO_ROOT/analysis/dataset_statistics.py"

if [ ! -f "$STATS" ]; then
  echo "Error: could not find $STATS"
  exit 1
fi

# Prefer the project venv, fall back to the system interpreter.
PYTHON="$REPO_ROOT/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3

DIRS=()
EXTRA=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift
    EXTRA=("$@")
    break
  fi
  if [ ! -d "$1" ]; then
    echo "Error: $1 is not a valid directory"
    exit 1
  fi
  if ! ls "$1"/color*.json >/dev/null 2>&1; then
    echo "Error: no color*.json files found in $1 (is this a raw dataset?)"
    exit 1
  fi
  DIRS+=("$1")
  shift
done

if [ ${#DIRS[@]} -eq 0 ]; then
  echo "Error: no dataset directories given"
  exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON" -m analysis.dataset_statistics "${DIRS[@]}" "${EXTRA[@]}"
