#!/usr/bin/env bash

DIR1="$1"
DIR2="$2"

if [[ -z "$DIR1" || -z "$DIR2" ]]; then
  echo "Usage: $0 <dir1> <dir2>"
  exit 1
fi

# Create sorted lists of image filenames (basename only)
tmp1=$(mktemp)
tmp2=$(mktemp)

find "$DIR1" -type f \( \
  -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.bmp" \
\) -printf "%f\n" | sort > "$tmp1"

find "$DIR2" -type f \( \
  -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.bmp" \
\) -printf "%f\n" | sort > "$tmp2"

# Print files common to both
comm -12 "$tmp1" "$tmp2"

only_in_dir2=$(comm -23 "$tmp2" "$tmp1")
[[ -z "$only_in_dir2" ]] && echo "DIR2 is a subset of DIR1" || echo "DIR2 has files not in DIR1"

rm "$tmp1" "$tmp2"

