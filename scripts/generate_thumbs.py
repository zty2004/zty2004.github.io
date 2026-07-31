#!/usr/bin/env python3
"""Generate index-card thumbnails for posts and wire them into front matter.

For each post whose body references a local image, creates
images/<dir>/thumb.jpg (480px wide, q70) from the FIRST image and adds
`thumb: /images/<dir>/thumb.jpg` to the front matter. Idempotent.

Usage: python3 scripts/generate_thumbs.py   (run from repo root)
"""

import glob
import os
import re
import subprocess

SRC_RE = re.compile(r'<img\b[^>]*?src="([^"]+)"')
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)


def local_path(src):
    src = src.replace("{{site.baseurl}}", "").replace("{{ site.baseurl }}", "")
    if src.startswith(("http://", "https://", "data:")):
        return None
    return src.lstrip("/")


def process(md_file):
    text = open(md_file, encoding="utf-8").read()
    fm = FM_RE.match(text)
    if not fm:
        return
    if "thumb:" in fm.group(1):
        print(f"{md_file}: thumb already set, skip")
        return

    first = None
    for m in SRC_RE.finditer(text):
        p = local_path(m.group(1))
        if p and os.path.exists(p) and not p.endswith("thumb.jpg"):
            first = p
            break
    if not first:
        print(f"{md_file}: no local image, skip")
        return

    thumb = os.path.join(os.path.dirname(first), "thumb.jpg")
    if not os.path.exists(thumb):
        subprocess.run(
            ["sips", "--resampleWidth", "480", "-s", "format", "jpeg",
             "-s", "formatOptions", "70", first, "--out", thumb],
            capture_output=True,
        )
    fm_new = fm.group(1) + f"\nthumb: /{thumb}"
    open(md_file, "w", encoding="utf-8").write(
        text.replace(fm.group(1), fm_new, 1)
    )
    size = os.path.getsize(thumb) // 1024
    print(f"{md_file}: thumb -> /{thumb} ({size} KB)")


if __name__ == "__main__":
    for f in sorted(glob.glob("_posts/*.md")):
        process(f)
