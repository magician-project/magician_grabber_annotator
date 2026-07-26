#!/bin/bash
# Usage: ./renameSeveritiesInDatasetRaw.sh <dataset_dir> "<from_severity>" "<to_severity>"
# Example: ./renameSeveritiesInDatasetRaw.sh /media/ammar/games2/Datasets/Magician/altinay_test_data "Class A" "Class C"
#
# Bulk-rewrites a defect SEVERITY label across all raw color*.json annotation
# files in a dataset (the "pointSeverities" values). Matches the quoted JSON
# string exactly (e.g. "Class A") so defect class names are never touched.
# A timestamped backup of the color*.json files is created before editing.

if [ -z "$3" ]; then
  echo "Usage: $0 <dataset_dir> \"<from_severity>\" \"<to_severity>\""
  echo "Example: $0 /path/to/dataset \"Class A\" \"Class C\""
  exit 1
fi

DATASET="$1"
FROM="$2"
TO="$3"

if [ ! -d "$DATASET" ]; then
  echo "Error: $DATASET is not a valid directory"
  exit 1
fi

cd "$DATASET" || exit 1

if ! ls color*.json >/dev/null 2>&1; then
  echo "Error: no color*.json files found in $DATASET (is this a raw dataset?)"
  exit 1
fi

BEFORE=$(grep -rho "\"$FROM\"" color*.json | wc -l)
if [ "$BEFORE" -eq 0 ]; then
  echo "Nothing to do: \"$FROM\" not found in any color*.json"
  exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
BACKUP="color_json_backup_${TS}.tar.gz"
tar czf "$BACKUP" color*.json
echo "Backup: $DATASET/$BACKUP ($(ls -1 color*.json | wc -l) files)"

sed -i "s/\"$FROM\"/\"$TO\"/g" color*.json

AFTER=$(grep -rho "\"$TO\"" color*.json | wc -l)
echo "Renamed \"$FROM\" -> \"$TO\": $BEFORE occurrence(s) changed ($AFTER now \"$TO\")"
echo "Rollback with: cd \"$DATASET\" && tar xzf \"$BACKUP\""
