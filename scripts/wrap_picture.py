#!/usr/bin/env python3
"""Wrap post <img> tags in <picture> so browsers prefer the WebP sibling.

    <img src="....jpg" ...>
    ->
    <picture><source srcset="....webp" type="image/webp"><img src="....jpg" ...></picture>

Only local images that actually have a .webp sibling are wrapped. Idempotent:
tags already inside a <picture> are left alone.

Usage: python3 scripts/wrap_picture.py   (run from repo root)
"""

import glob
import os
import re

# match either a whole existing <picture> block (skip it) or a bare <img>
TOKEN_RE = re.compile(r'<picture\b.*?</picture>|<img\b[^>]*>', re.S)
SRC_RE = re.compile(r'src="([^"]+)"')


def local_path(src):
    src = src.replace("{{site.baseurl}}", "").replace("{{ site.baseurl }}", "")
    if src.startswith(("http://", "https://", "data:")):
        return None
    return src.lstrip("/")


def process(md_file):
    text = open(md_file, encoding="utf-8").read()
    wrapped = 0
    out = []
    pos = 0

    for m in TOKEN_RE.finditer(text):
        token = m.group(0)
        out.append(text[pos:m.start()])
        pos = m.end()

        # already a <picture> block — leave untouched (idempotent)
        if token.startswith("<picture"):
            out.append(token)
            continue

        src_m = SRC_RE.search(token)
        path = local_path(src_m.group(1)) if src_m else None
        if not path:
            out.append(token)
            continue

        webp_path = os.path.splitext(path)[0] + ".webp"
        if not os.path.exists(webp_path):
            out.append(token)
            continue

        webp_src = os.path.splitext(src_m.group(1))[0] + ".webp"
        out.append(
            f'<picture><source srcset="{webp_src}" type="image/webp">{token}</picture>'
        )
        wrapped += 1

    out.append(text[pos:])
    if wrapped:
        open(md_file, "w", encoding="utf-8").write("".join(out))
    print(f"{md_file}: {wrapped} images wrapped in <picture>")


if __name__ == "__main__":
    for f in sorted(glob.glob("_posts/*.md")):
        process(f)
