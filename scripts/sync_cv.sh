#!/bin/bash
#
# sync_cv.sh — refresh the local CV copy from the CV repo's latest release.
#
# The profile page links to assets/pdf/cv.pdf rather than straight to the
# release asset: GitHub sends release downloads with
# `Content-Disposition: attachment`, which makes the browser download the file
# instead of displaying it. GitHub Pages serves the same PDF inline, so the
# link opens in a new tab and just shows the CV.
#
# Run this after the CV repo publishes a new build, then commit the result.
#
# Usage: ./scripts/sync_cv.sh

set -euo pipefail

SOURCE="https://github.com/zty2004/CV/releases/download/latest/cv.pdf"
DEST="assets/pdf/cv.pdf"

if [ ! -f _config.yml ]; then
  echo "sync_cv: run this from the repository root" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
tmp="$(mktemp -t cv)"
trap 'rm -f "$tmp"' EXIT

curl -fsSL "$SOURCE" -o "$tmp"

# refuse to install anything that is not actually a PDF (404 pages, etc.)
if [ "$(head -c 5 "$tmp")" != "%PDF-" ]; then
  echo "sync_cv: downloaded file is not a PDF — is the release published?" >&2
  exit 1
fi

if [ -f "$DEST" ] && cmp -s "$tmp" "$DEST"; then
  echo "sync_cv: already up to date ($(wc -c < "$DEST" | tr -d ' ') bytes)"
  exit 0
fi

cp "$tmp" "$DEST"
echo "sync_cv: updated $DEST ($(wc -c < "$DEST" | tr -d ' ') bytes)"
echo "next: git add $DEST && git commit -m 'chore. refresh CV' && git push"
