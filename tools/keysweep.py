"""Padding-agnostic AES key sweep.

Filter is entropy of the first 512 decrypted bytes: a correct key yields
structured data (H ~4-6), a wrong key yields noise (H ~7.6 at this sample size).
Candidate keys come from IL2CPP string literals (with common derivations) and
from sliding windows over the metadata default-value blob, where
`byte[] key = new byte[]{...}` initializers are stored verbatim.
"""
import hashlib, pathlib, struct, sys
import numpy as np
from Crypto.Cipher import AES
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

data = pathlib.Path("data/raw/textassets/EncryptedMainGroup").read_bytes()
CHUNK = data[:512]
CHUNK_OFF = data[16:528]
THRESH = 6.5

def entropy(buf):
    c = np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256)
    p = c[c > 0] / len(buf)
    return float(-(p * np.log2(p)).sum())

def test(k, tag, out):
    for mode in ("ECB", "CBC0", "CBCF"):
        try:
            if mode == "ECB":
                pt = AES.new(k, AES.MODE_ECB).decrypt(CHUNK)
            elif mode == "CBC0":
                pt = AES.new(k, AES.MODE_CBC, b"\0" * 16).decrypt(CHUNK)
            else:
                pt = AES.new(k, AES.MODE_CBC, data[:16]).decrypt(CHUNK_OFF)
        except Exception:
            continue
        H = entropy(pt)
        if H < THRESH:
            out.append((H, tag, mode, k, pt))
            print(f"HIT H={H:.2f} {mode:5s} {tag} key={k.hex()[:64]}\n    {pt[:64]!r}")

hits = []
# source 1: string literals
lits = [s for s in pathlib.Path("data/raw/string_literals.txt").read_text(encoding="utf-8").split("\n") if 4 <= len(s) <= 96]
for s in lits:
    b = s.encode("utf-8", "ignore")
    for name, k in (("utf8-32", b.ljust(32, b"\0")[:32]), ("utf8-16", b.ljust(16, b"\0")[:16]),
                    ("md5", hashlib.md5(b).digest()), ("sha256", hashlib.sha256(b).digest()),
                    ("sha1-16", hashlib.sha1(b).digest()[:16])):
        test(k, f"lit:{name}:{s[:32]}", hits)
print(f"literals swept ({len(lits)}), hits so far {len(hits)}")

# source 2: sliding window over the default-value blob
blob = pathlib.Path("data/raw/default_values.bin").read_bytes()
for size in (32, 16, 24):
    for i in range(len(blob) - size + 1):
        test(blob[i:i + size], f"dv:{size}@{i}", hits)
    print(f"  default-values size {size} swept, hits {len(hits)}")

print(f"\nTOTAL hits {len(hits)}")
if hits:
    hits.sort(key=lambda h: h[0])
    H, tag, mode, k, _ = hits[0]
    print(f"BEST H={H:.2f} {mode} {tag} key={k.hex()}")
