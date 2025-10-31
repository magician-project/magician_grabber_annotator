#!/bin/bash
# Usage: ./download_apache_dir.sh http://ammar.gr/magician/datasets/NDA_3_A_T1/

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..


# Check if URL argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <apache_directory_url>"
    exit 1
fi

# Get the URL and folder name
FOLDER="$1"
URL="http://ammar.gr/magician/datasets/$1"

# Remove trailing slash from folder name if it exists
FOLDER="${FOLDER%/}"

# Create the folder if it doesn't exist
mkdir -p "$FOLDER"

# Use wget to download all files (not recursing into subdirs)
wget -r -np -nH --cut-dirs=3 -P "$FOLDER" -R "index.html*" "$URL"

echo "✅ All files downloaded to: $FOLDER/"

