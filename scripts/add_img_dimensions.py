#!/usr/bin/env python3
"""Backfill width/height attributes on <img> tags in _posts/*.md.

Reads pixel dimensions with macOS `sips`. Idempotent: tags that already
carry a width attribute are left untouched; external/missing images are
skipped with a warning.

Usage: python3 scripts/add_img_dimensions.py   (run from repo root)
"""

import glob
import re
import subprocess
import sys

IMG_RE = re.compile(r'<img\b[^>]*>')
SRC_RE = re.compile(r'src="([^"]+)"')


def sips_dims(path):
    out = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", path],
        capture_output=True, text=True,
    ).stdout
    w = h = None
    for line in out.splitlines():
        if "pixelWidth" in line:
            w = int(line.split()[-1])
        elif "pixelHeight" in line:
            h = int(line.split()[-1])
    return w, h


def local_path(src):
    """Map a src URL to a repo-relative file path, or None if external."""
    src = src.replace("{{site.baseurl}}", "").replace("{{ site.baseurl }}", "")
    if src.startswith(("http://", "https://", "data:")):
        return None
    return src.lstrip("/")


def process(md_file):
    text = open(md_file, encoding="utf-8").read()
    changed = 0

    def repl(m):
        nonlocal changed
        tag = m.group(0)
        if " width=" in tag:
            return tag
        src_m = SRC_RE.search(tag)
        if not src_m:
            return tag
        path = local_path(src_m.group(1))
        if path is None:
            return tag
        try:
            w, h = sips_dims(path)
        except FileNotFoundError:
            w = h = None
        if not w or not h:
            print(f"  [warn] cannot read dimensions: {path}", file=sys.stderr)
            return tag
        changed += 1
        return tag[:-1] + f' width="{w}" height="{h}">'

    new_text = IMG_RE.sub(repl, text)
    if changed:
        open(md_file, "w", encoding="utf-8").write(new_text)
    print(f"{md_file}: {changed} tags updated")


if __name__ == "__main__":
    for f in sorted(glob.glob("_posts/*.md")):
        process(f)
