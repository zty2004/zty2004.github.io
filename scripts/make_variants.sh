#!/bin/bash
#
# make_variants.sh — generate downscaled WebP tiers (-200 / -480 / -800) for
# every photo under images/, so a post can serve a file that matches the slot
# the photo actually got instead of always the 1600px original.
#
# The full-size JPEG plus its .webp sibling (convert_webp.sh) remain the top
# tier. Idempotent: a tier newer than its source is skipped, and a tier wider
# than the source is not made at all.
#
# Requires cwebp (brew install webp) and sips (macOS).
# Usage: ./scripts/make_variants.sh [images-dir]

set -euo pipefail

IMAGES_DIR="${1:-images}"
QUALITY=80
TIERS="200 480 800"

if ! command -v cwebp >/dev/null; then
  echo "make_variants: cwebp not found — install it with 'brew install webp'" >&2
  exit 1
fi

made=0
skipped=0
toosmall=0
added_bytes=0

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

while IFS= read -r -d '' img; do
  src_w=$(sips -g pixelWidth "$img" | awk '/pixelWidth/ {print $2}')
  base="${img%.*}"

  for w in $TIERS; do
    out="$base-$w.webp"

    if [ -f "$out" ] && [ "$out" -nt "$img" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    if [ "$src_w" -le "$w" ]; then
      toosmall=$((toosmall + 1))   # the full-size file already covers this slot
      continue
    fi

    sips --resampleWidth "$w" "$img" --out "$tmp/r.jpg" >/dev/null \
      || { echo "make_variants: resample failed on $img" >&2; exit 1; }
    cwebp -quiet -q "$QUALITY" -m 4 "$tmp/r.jpg" -o "$out"
    made=$((made + 1))
    added_bytes=$((added_bytes + $(stat -f%z "$out")))
  done
done < <(find "$IMAGES_DIR" \( -iname '*.jpg' -o -iname '*.jpeg' \) \
           -not -iname 'thumb.*' -print0)

echo "----------------------------------------"
echo "Tiers made: $made   up-to-date: $skipped   source narrower than tier: $toosmall"
echo "Added (this run): $((added_bytes / 1024 / 1024)) MB"
