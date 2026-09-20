#!/usr/bin/env python3
"""Single source of truth for post image markup: several resolutions per photo.

    <img src="D/N.jpg" ...>            (bare, or an older single-tier <picture>)
    ->
    <picture><source type="image/webp"
        srcset="D/N-200.webp 200w, D/N-480.webp 480w, D/N-800.webp 800w, D/N.webp 1600w"
        sizes="(max-width: 900px) 45vw, 235px"
      ><img src="D/N.jpg" ... data-full-webp="D/N.webp"></picture>

Which tiers appear follows what is actually on disk: make_variants.sh skips a
tier wider than its source, and convert_webp.sh deletes a full-size WebP that
lost to its JPEG. Safe to re-run at any point in the pipeline; idempotent.

The `sizes` value is only the pre-JS default — gallery.js rewrites it to the
cell's real width on every layout and zoom step.

Usage: python3 scripts/responsive_images.py   (run from repo root)
"""

import glob
import os
import re
import sys

TIERS = (200, 480, 800)

TOKEN_RE = re.compile(r"<picture\b.*?</picture>|<img\b[^>]*>", re.S)
IMG_IN_PICTURE_RE = re.compile(r"<img\b[^>]*>", re.S)
SRC_RE = re.compile(r'\bsrc="([^"]+)"')
WIDTH_RE = re.compile(r'\bwidth="(\d+)"')
# attributes this script owns — stripped before re-emitting so re-runs are clean
OWNED_RE = re.compile(
    r'\s*\b(?:srcset|sizes|fetchpriority|data-full-webp|decoding)="[^"]*"'
    r'|\s*\bloading="[^"]*"'
)

# photo walls show ~3 across on desktop and ~2 on a phone; prose posts run the
# full 720px reading column
SIZES_GALLERY = "(max-width: 900px) 45vw, 235px"
SIZES_PROSE = "(max-width: 900px) calc(100vw - 40px), 720px"

# above-the-fold tiles must not wait for lazy-loading; the first one is the LCP
EAGER_COUNT = 3


def local_path(src):
    src = src.replace("{{site.baseurl}}", "").replace("{{ site.baseurl }}", "")
    if src.startswith(("http://", "https://", "data:")):
        return None
    return src.lstrip("/")


def is_gallery(text):
    m = re.search(r"^gallery:\s*true\s*$", text[:2000], re.M)
    return bool(m)


def build(img_token, src, index, gallery):
    """Return the <picture> block for one image, or None to leave it alone."""
    path = local_path(src)
    if not path:
        return None
    disk = os.path.splitext(path)[0]          # images/D/N
    prefix = src[: len(src) - len(os.path.basename(src))]  # keeps {{site.baseurl}}

    width_m = WIDTH_RE.search(img_token)
    if not width_m:
        print(f"  [warn] no width attribute, tiers skipped: {src}", file=sys.stderr)
        return None
    full_w = int(width_m.group(1))

    candidates = []
    for w in TIERS:
        if os.path.exists(f"{disk}-{w}.webp"):
            candidates.append(f"{prefix}{os.path.basename(disk)}-{w}.webp {w}w")
    full_webp = f"{disk}.webp"
    if os.path.exists(full_webp):
        candidates.append(f"{prefix}{os.path.basename(disk)}.webp {full_w}w")

    clean = OWNED_RE.sub("", img_token)
    clean = re.sub(r"\s*/?>$", "", clean.rstrip())

    # a photo wall's first images are the LCP element; prose posts keep lazy
    if gallery and index < EAGER_COUNT:
        loading = ' loading="eager"'
        priority = ' fetchpriority="high"' if index == 0 else ""
    else:
        loading = ' loading="lazy"'
        priority = ""

    data_full = (
        f' data-full-webp="{prefix}{os.path.basename(disk)}.webp"'
        if os.path.exists(full_webp)
        else ""
    )

    img_out = f"{clean}{loading} decoding=\"async\"{priority}{data_full}>"
    if not candidates:
        return img_out                        # nothing on disk to offer

    srcset = ", ".join(candidates)
    sizes = SIZES_GALLERY if gallery else SIZES_PROSE
    return (
        f'<picture><source type="image/webp" srcset="{srcset}" sizes="{sizes}">'
        f"{img_out}</picture>"
    )


def process(md_file):
    with open(md_file, encoding="utf-8") as f:
        text = f.read()
    gallery = is_gallery(text)
    out = []
    pos = 0
    index = 0
    changed = 0

    for m in TOKEN_RE.finditer(text):
        token = m.group(0)
        out.append(text[pos:m.start()])
        pos = m.end()

        if token.startswith("<picture"):
            inner = IMG_IN_PICTURE_RE.search(token)
            if not inner:
                out.append(token)
                continue
            img_token = inner.group(0)
        else:
            img_token = token

        src_m = SRC_RE.search(img_token)
        if not src_m or local_path(src_m.group(1)) is None:
            out.append(token)
            continue

        rebuilt = build(img_token, src_m.group(1), index, gallery)
        if rebuilt is None:
            out.append(token)      # not ours to manage; don't spend an eager slot
            continue
        index += 1
        if rebuilt == token:
            out.append(token)
            continue
        out.append(rebuilt)
        changed += 1

    out.append(text[pos:])
    if changed:
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("".join(out))
    print(f"{md_file}: {changed} image tags rewritten ({index} total, gallery={gallery})")


if __name__ == "__main__":
    for f in sorted(glob.glob("_posts/*.md")):
        process(f)
