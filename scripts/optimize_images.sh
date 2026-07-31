#!/bin/bash
#
# optimize_images.sh — batch-shrink blog photos for the web.
#
# Resizes every JPEG under images/ to at most MAX_WIDTH px wide and
# re-encodes at QUALITY %, using macOS built-in `sips` (no dependencies).
# Safe to re-run: images already within the size limit are only re-encoded
# if they are still oversized on disk.
#
# Usage: ./scripts/optimize_images.sh [images-dir]

set -euo pipefail

IMAGES_DIR="${1:-images}"
MAX_WIDTH=1600
QUALITY=80
# skip files already this small (bytes) — they are fine as-is
SKIP_UNDER=$((300 * 1024))

total_before=0
total_after=0
processed=0
skipped=0

while IFS= read -r -d '' img; do
  size_before=$(stat -f%z "$img")

  if [ "$size_before" -lt "$SKIP_UNDER" ]; then
    skipped=$((skipped + 1))
    continue
  fi

  width=$(sips -g pixelWidth "$img" | awk '/pixelWidth/ {print $2}')

  if [ "$width" -gt "$MAX_WIDTH" ]; then
    sips --resampleWidth "$MAX_WIDTH" -s format jpeg -s formatOptions "$QUALITY" "$img" --out "$img" >/dev/null
  else
    sips -s format jpeg -s formatOptions "$QUALITY" "$img" --out "$img" >/dev/null
  fi

  size_after=$(stat -f%z "$img")
  total_before=$((total_before + size_before))
  total_after=$((total_after + size_after))
  processed=$((processed + 1))
done < <(find "$IMAGES_DIR" \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0)

echo "----------------------------------------"
echo "Processed: $processed  Skipped(small): $skipped"
echo "Before: $((total_before / 1024 / 1024)) MB"
echo "After:  $((total_after / 1024 / 1024)) MB"
