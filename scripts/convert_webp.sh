#!/bin/bash
#
# convert_webp.sh — generate a .webp sibling for every JPEG under images/.
#
# The original JPEGs are kept as <picture> fallbacks, so nothing breaks on
# browsers without WebP support. Idempotent: a .webp newer than its source
# is skipped, so re-runs are cheap.
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
webp_total=0
converted=0
skipped=0

while IFS= read -r -d '' img; do
  webp="${img%.*}.webp"

  if [ -f "$webp" ] && [ "$webp" -nt "$img" ]; then
    skipped=$((skipped + 1))
  else
    cwebp -quiet -q "$QUALITY" -m 4 "$img" -o "$webp"
    converted=$((converted + 1))
  fi

  jpg_total=$((jpg_total + $(stat -f%z "$img")))
  webp_total=$((webp_total + $(stat -f%z "$webp")))
done < <(find "$IMAGES_DIR" \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0)

echo "----------------------------------------"
echo "Converted: $converted   Skipped(up-to-date): $skipped"
echo "JPEG  total: $((jpg_total / 1024 / 1024)) MB"
echo "WebP  total: $((webp_total / 1024 / 1024)) MB"
if [ "$jpg_total" -gt 0 ]; then
  echo "Saving: $(( 100 - webp_total * 100 / jpg_total ))% when WebP is served"
fi
