#!/bin/bash
# Usage: ./compressAllDatasets.sh /media/ammar/FastDatasets/Magician/CameraV2Datasets/
#
# Finds every directory under the given path that still contains raw *.pnm files
# and runs compressDataset.py on each (in-place). compressDataset.py skips any
# directory that is already compressed, so re-running is safe.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$DIR/.." && pwd )"

if [ -z "$1" ]; then
  echo "Usage: $0 <datasets_root_directory>"
  exit 1
fi

ROOT="$1"

if [ ! -d "$ROOT" ]; then
  echo "Error: $ROOT is not a valid directory"
  exit 1
fi

cd "$REPO_ROOT"

find "$ROOT" -name '*.pnm' -printf '%h\n' | sort -u | while read -r d; do
  python3 compressDataset.py "$d"
done
