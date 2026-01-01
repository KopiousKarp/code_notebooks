#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-.}"

for f in "$DIR"/*.jpg; do
  base=$(basename "$f")

  # Match: <tag>_YYYY-MM-DD_HH-MM-SS.jpg
  if [[ "$base" =~ ^(.+)_([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{2})-([0-9]{2})-([0-9]{2})\.jpg$ ]]; then
    tag="${BASH_REMATCH[1]}"
    date="${BASH_REMATCH[2]}"
    hour="${BASH_REMATCH[3]}"
    min="${BASH_REMATCH[4]}"
    sec="${BASH_REMATCH[5]}"

    new="${tag}_${date}_${hour}_${min}_${sec}.jpg"

    echo "RENAME: $base -> $new"
    mv -- "$f" "$DIR/$new"
  fi
done

