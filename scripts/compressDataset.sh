#!/bin/bash
# Usage: ./compressDataset.sh <dataset_dir> [output_dir]
#
# Compresses one raw dataset in place (.pnm -> lossless RGBA .png), or into
# <output_dir> if given. Wraps `python3 -m mga.core.compress_dataset`, which
# must run from the repo root, and picks a Python interpreter that has the
# dependencies (cv2/numpy) — so callers outside this repo do not need to know
# where the package lives or which venv to use.
#
# Intended for recording scripts in sibling repos, e.g. magician_grabber's
# recordDatasetFORTH.sh:
#
#   ../magician_grabber_annotator/scripts/compressDataset.sh "$OUTPUT"
#
# Safe to re-run: a directory with no .pnm files is reported as already
# compressed and left untouched.

if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_dir> [output_dir]"
  echo "Example: $0 mySession_5000"
  exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$DIR/.." && pwd )"

# Resolve to absolute paths BEFORE cd'ing to the repo root, so relative
# arguments stay relative to the caller's working directory.
if [ ! -d "$1" ]; then
  echo "Error: $1 is not a valid directory"
  exit 1
fi
INPUT="$( cd "$1" && pwd )"

OUTPUT=""
if [ -n "$2" ]; then
  mkdir -p "$2" || { echo "Error: could not create output directory $2"; exit 1; }
  OUTPUT="$( cd "$2" && pwd )"
fi

# Prefer a venv that actually has cv2/numpy: the project venv, then the
# classifier's, then the old classifier checkout, then the system interpreter.
pick_python() {
  local candidates=(
    "$REPO_ROOT/venv/bin/python3"
    "$REPO_ROOT/../magician_vision_classifier/venv/bin/python3"
    "/home/ammar/Documents/Programming/Magician/src/python/classifier/venv/bin/python3"
    "$(command -v python3)"
  )
  for p in "${candidates[@]}"; do
    if [ -x "$p" ] && "$p" -c "import cv2, numpy" >/dev/null 2>&1; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

PYTHON="$(pick_python)"
if [ -z "$PYTHON" ]; then
  echo "Error: no Python interpreter with opencv-python and numpy was found."
  echo "Tried the project venv, the classifier venv and the system python3."
  echo "Install them with: $REPO_ROOT/venv/bin/pip install opencv-python numpy"
  exit 1
fi

cd "$REPO_ROOT" || exit 1
exec "$PYTHON" -m mga.core.compress_dataset "$INPUT" ${OUTPUT:+"$OUTPUT"}
