#!/usr/bin/env python3
"""Strip identity/location metadata from published images.

Removes EXIF, XMP and IPTC from every JPEG/PNG/WebP under a directory, in
place and without re-encoding: the metadata segments are dropped and the
compressed image data is copied through untouched. JFIF (APP0) and ICC colour
profiles are kept, so dimensions and colours are unchanged.

Usage: python3 scripts/strip_metadata.py [images-dir] [--check]
  --check  report what would be removed, change nothing (exit 1 if not clean)
"""

import os
import sys

# APPn segments that can carry a device fingerprint, a timestamp, a caption or
# coordinates. APP0 (JFIF) and APP2 (ICC) are deliberately absent — they hold
# nothing personal and dropping them shifts colours or breaks strict decoders.
JPEG_STRIP = {0xE1, 0xEC, 0xED}

PNG_STRIP = (b"eXIf", b"iTXt", b"zTXt", b"tIME")
WEBP_STRIP = (b"EXIF", b"XMP ")


def iter_jpeg_segments(buf):
    """Yield (marker, start, end) up to and including SOS.

    Entropy-coded data after SOS contains stuffed 0xFF00 bytes and restart
    markers, so it cannot be scanned as segments — it is opaque from there on.
    """
    if buf[:2] != b"\xff\xd8":
        return
    i, n = 2, len(buf)
    while i + 4 <= n:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:
            break
        length = int.from_bytes(buf[i + 2:i + 4], "big")
        yield marker, i, i + 2 + length
        if marker == 0xDA:
            break
        i += 2 + length


def strip_jpeg(buf):
    """Return (new_buf, removed_markers) with metadata segments dropped."""
    removed = []
    out = bytearray(b"\xff\xd8")
    sos_at = None
    for marker, start, end in iter_jpeg_segments(buf):
        if marker in JPEG_STRIP:
            removed.append(marker)
            continue
        if marker == 0xDA:
            sos_at = start
            break
        out += buf[start:end]
    if sos_at is None:
        return buf, []          # truncated or unusual file: leave it alone
    return bytes(out) + buf[sos_at:], removed


def strip_riff(buf, kinds):
    """Return (new_buf, removed_fourccs) for a RIFF container."""
    if buf[:4] != b"RIFF" or buf[8:12] != b"WEBP":
        return buf, []
    removed = []
    body = bytearray()
    i = 12
    while i + 8 <= len(buf):
        fourcc = buf[i:i + 4]
        size = int.from_bytes(buf[i + 4:i + 8], "little")
        end = i + 8 + size + (size & 1)
        if fourcc in kinds:
            removed.append(fourcc)
        else:
            body += buf[i:end]
        i = end
    if not removed:
        return buf, []
    return b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WEBP" + bytes(body), removed


def strip_png(buf):
    if buf[:8] != b"\x89PNG\r\n\x1a\n":
        return buf, []
    removed = []
    out = bytearray(buf[:8])
    i = 8
    while i + 12 <= len(buf):
        size = int.from_bytes(buf[i:i + 4], "big")
        kind = buf[i + 4:i + 8]
        end = i + 12 + size
        if kind in PNG_STRIP:
            removed.append(kind)
        else:
            out += buf[i:end]
        i = end
    if not removed:
        return buf, []
    return bytes(out), removed


def strip_bytes(buf):
    if buf[:2] == b"\xff\xd8":
        return strip_jpeg(buf)
    if buf[:4] == b"RIFF":
        return strip_riff(buf, WEBP_STRIP)
    if buf[:8] == b"\x89PNG\r\n\x1a\n":
        return strip_png(buf)
    return buf, []


def has_metadata(buf):
    _, removed = strip_bytes(buf)
    return removed


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    check = "--check" in argv
    root = args[1] if len(args) > 1 else "images"

    scanned = stripped = dirty = 0
    saved = 0
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "rb") as fh:
                buf = fh.read()
            scanned += 1
            new, removed = strip_bytes(buf)
            if not removed:
                continue
            dirty += 1
            saved += len(buf) - len(new)
            if check:
                print(f"{path}: would drop {[hex(m) if isinstance(m, int) else m.decode() for m in removed]}")
                continue
            # sibling temp then rename: an interrupted write must not leave a
            # half-stripped image that looks newer than its source
            st = os.stat(path)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(new)
            os.replace(tmp, path)
            # restore mtime: the WebP tiers decide freshness with `-nt`, and
            # re-encoding 1756 derivatives for a metadata-only edit is pure churn
            os.chmod(path, st.st_mode & 0o7777)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
            stripped += 1

    if check:
        print(f"scanned {scanned}, carrying metadata {dirty}")
        return 1 if dirty else 0
    print(f"scanned {scanned}, stripped {stripped}, freed {saved / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
