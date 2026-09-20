#!/bin/bash
#
# convert_webp.sh — generate a .webp sibling for every JPEG under images/.
#
# The original JPEGs are kept as <picture> fallbacks, so nothing breaks on
# browsers without WebP support. A WebP that comes out larger than its JPEG is
# deleted rather than shipped. Idempotent: a .webp newer than its source is
# skipped, so re-runs are cheap.
#
# Requires cwebp (brew install webp).
# Usage: ./scripts/convert_webp.sh [images-dir]

set -euo pipefail

IMAGES_DIR="${1:-images}"
QUALITY=80

if ! command -v cwebp >/dev/null; then
  echo "convert_webp: cwebp not found — install it with 'brew install webp'" >&2
  exit 1
fi

jpg_total=0
served_total=0
converted=0
skipped=0
dropped=0

while IFS= read -r -d '' img; do
  webp="${img%.*}.webp"
  jpg_size=$(stat -f%z "$img")

  if [ -f "$webp" ] && [ "$webp" -nt "$img" ]; then
    skipped=$((skipped + 1))
  else
    cwebp -quiet -q "$QUALITY" -m 4 "$img" -o "$webp"
    # A WebP that lost to its own JPEG is worse than no WebP: <picture> would
    # serve the bigger file. Drop it and let the JPEG stand. (Re-encoded on the
    # next run — only a handful of high-entropy photos hit this.)
    if [ "$(stat -f%z "$webp")" -ge "$jpg_size" ]; then
      rm -f "$webp"
      dropped=$((dropped + 1))
    else
      converted=$((converted + 1))
    fi
  fi

  jpg_total=$((jpg_total + jpg_size))
  # count what a browser would actually be served
  if [ -f "$webp" ]; then
    served_total=$((served_total + $(stat -f%z "$webp")))
  else
    served_total=$((served_total + jpg_size))
  fi
done < <(find "$IMAGES_DIR" \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0)

echo "----------------------------------------"
echo "Converted: $converted   Up-to-date: $skipped   Dropped(WebP lost to JPEG): $dropped"
echo "JPEG     total: $((jpg_total / 1024 / 1024)) MB"
echo "Served   total: $((served_total / 1024 / 1024)) MB"
if [ "$jpg_total" -gt 0 ]; then
  echo "Saving: $(( 100 - served_total * 100 / jpg_total ))%"
fi
