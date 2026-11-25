#!/usr/bin/env python3
"""Low-level reader for Radioss binary time-history files (*T01, /TFILE/4 format).

The file is a sequence of records: [int32 len][len bytes][int32 len], big-endian.
`records(path)` yields raw payloads; `dump(path)` prints a structure summary.
"""
import struct, sys


def records(path):
    with open(path, "rb") as f:
        data = f.read()
    pos, n = 0, len(data)
    while pos + 4 <= n:
        (ln,) = struct.unpack(">i", data[pos:pos + 4])
        if ln < 0 or pos + 8 + ln > n:
            raise ValueError(f"bad record length {ln} at byte {pos}")
        payload = data[pos + 4:pos + 4 + ln]
        (ln2,) = struct.unpack(">i", data[pos + 4 + ln:pos + 8 + ln])
        if ln2 != ln:
            raise ValueError(f"record marker mismatch at byte {pos}: {ln} vs {ln2}")
        yield payload
        pos += 8 + ln


def as_text(b):
    try:
        s = b.decode("ascii")
    except UnicodeDecodeError:
        return None
    return s if all(32 <= ord(c) < 127 for c in s.rstrip("\x00")) else None


def dump(path, limit=200):
    for k, r in enumerate(records(path)):
        if k >= limit:
            print("..."); break
        t = as_text(r)
        nw = len(r) // 4
        ints = struct.unpack(f">{nw}i", r[:4 * nw])[:12]
        flts = struct.unpack(f">{nw}f", r[:4 * nw])[:8]
        print(f"#{k:4d} len={len(r):7d} " + (f"TEXT {t[:90]!r}" if t else f"I {ints} F {tuple(round(x, 5) for x in flts)}"))


if __name__ == "__main__":
    dump(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
