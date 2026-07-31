#!/usr/bin/env python3
"""Subset the Great Vibes cursive font down to the signature glyphs.

The font is only used for one line — the site tagline in the masthead — so
shipping the full 447KB face (or pulling it from Google Fonts) is wasteful.
This keeps just the characters that line needs, in woff2.

Requires: pip3 install fonttools brotli
Usage: python3 scripts/subset_font.py <source.ttf> <out.woff2> "<text>"
"""

import sys
from fontTools import subset


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    src, out, text = sys.argv[1], sys.argv[2], sys.argv[3]

    args = [
        src,
        f"--output-file={out}",
        "--flavor=woff2",
        f"--text={text}",
        "--layout-features=kern,liga,calt",  # cursive joins need these
        "--desubroutinize",
        "--no-hinting",
        "--drop-tables+=DSIG",
    ]
    subset.main(args)
    print(f"subset written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
