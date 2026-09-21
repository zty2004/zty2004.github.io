#!/usr/bin/env python3
"""Replace timestamped photo filenames with opaque, content-derived ids.

Phone exports land in images/ as IMG_20240101_142614.jpg, which publishes the
capture time to the second in every page URL. This renames each photo group to
the first 16 hex digits of the SHA-256 of its JPEG, keeping the -200/-480/-800
WebP tiers on the same stem so srcset keeps working, and rewrites the
references in _posts/ to match.

Idempotent by construction: only names matching IMG_<8 digits>_<6 digits> are
ever touched, so an already-anonymised photo is left alone even if its content
is later re-encoded.

Usage: python3 scripts/anonymize_filenames.py [images-dir] [--check|--dry-run]
  --check    exit 1 if any timestamped filename remains (rename nothing)
  --dry-run  print the planned renames, change nothing
"""

import hashlib
import os
import re
import sys

TS_RE = re.compile(r"^(IMG_\d{8}_\d{6}.*?)(?:-(\d+))?(\.[A-Za-z0-9]+)$")
POSTS_DIR = "_posts"
STEM_LEN = 16


def groups_under(root):
    """Map directory -> {old stem: {suffix: filename}} for timestamped photos."""
    out = {}
    for dirpath, _, filenames in os.walk(root):
        found = {}
        for name in filenames:
            m = TS_RE.match(name)
            if m:
                found.setdefault(m.group(1), {})[(m.group(2), m.group(3))] = name
        if found:
            out[dirpath] = found
    return out


def new_stem(dirpath, members):
    """Hash the JPEG of the group, falling back to whatever member exists."""
    for suffix, name in sorted(members.items(), key=lambda kv: kv[1]):
        if suffix[1].lower() in (".jpg", ".jpeg"):
            preferred = name
            break
    else:
        preferred = sorted(members.values())[0]
    with open(os.path.join(dirpath, preferred), "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return digest[:STEM_LEN]


def rewrite_posts(mapping):
    """Return {post path: (old text, new text)} for every post that changes."""
    changed = {}
    # longest first, so a bare stem is never substituted inside a tier filename
    ordered = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    for name in sorted(os.listdir(POSTS_DIR)):
        if not name.endswith(".md"):
            continue
        path = os.path.join(POSTS_DIR, name)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        new_text = text
        for old, new in ordered:
            if old in new_text:
                new_text = new_text.replace(old, new)
        if new_text != text:
            changed[path] = (text, new_text)
    return changed


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    root = args[1] if len(args) > 1 else "images"
    check = "--check" in argv
    dry = "--dry-run" in argv

    groups = groups_under(root)
    total_files = sum(len(m) for g in groups.values() for m in g.values())

    if check:
        print(f"timestamped filenames remaining: {total_files} in {sum(len(g) for g in groups.values())} photos")
        return 1 if total_files else 0

    renames = []       # (old path, new path)
    mapping = {}       # old filename -> new filename, for post rewriting
    stems = {}         # generated stem -> old stem, to catch a truncated-hash clash
    for dirpath in sorted(groups):
        for stem in sorted(groups[dirpath]):
            members = groups[dirpath][stem]
            base = new_stem(dirpath, members)
            if base in stems:
                sys.exit(f"anonymize: hash prefix clash between {stems[base]} and {stem}")
            stems[base] = stem
            for (tier, ext), old_name in sorted(members.items(),
                                                key=lambda kv: (kv[0][0] or "", kv[0][1])):
                new_name = f"{base}-{tier}{ext}" if tier else f"{base}{ext}"
                if new_name != old_name:
                    renames.append((os.path.join(dirpath, old_name),
                                    os.path.join(dirpath, new_name)))
                    mapping[old_name] = new_name

    posts = rewrite_posts(mapping)

    if dry:
        for old, new in renames[:20]:
            print(f"  {old} -> {os.path.basename(new)}")
        print(f"  ... {len(renames)} file renames, {len(posts)} posts to rewrite")
        return 0

    for old, new in renames:
        os.rename(old, new)
    for path, (_, new_text) in posts.items():
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        os.replace(tmp, path)

    print(f"renamed {len(renames)} files across {len(groups)} directories, "
          f"rewrote {len(posts)} posts")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
