"""Try the candidate AES key/IV/mode combinations against an encrypted TextAsset."""
import itertools, os, pathlib, sys, binascii
from Crypto.Cipher import AES

BLOB = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/textassets/EncryptedMainGroup")
data = BLOB.read_bytes()
ASCII_KEY = (os.environ.get("DESPOT_PASSPHRASE") or "").encode()
if not ASCII_KEY:
    sys.exit("set DESPOT_PASSPHRASE; see notes/datamining.md")
HEXES = ["13de8260017e9e6419a4251b25fd6a4c", "3799bc5938d848f18ce2f7b78e9aeb6d",
         "77cc6a0a9cf4544fc860cadc64da2543", "8d3674321cd1bc040a730a78d6df5794"]

keys = {"ascii32": ASCII_KEY, "ascii16": ASCII_KEY[:16], "ascii24": ASCII_KEY[:24]}
for h in HEXES:
    keys[f"hexraw32:{h[:8]}"] = h.encode()            # the hex string itself as 32 ascii bytes
    keys[f"hexbin16:{h[:8]}"] = binascii.unhexlify(h)  # decoded 16 bytes
ivs = {"zero": b"\x00" * 16, "keyhead": ASCII_KEY[:16]}
for h in HEXES:
    ivs[f"hexbin:{h[:8]}"] = binascii.unhexlify(h)

def score(pt):
    """How plausible is this as the decrypted payload?"""
    if not pt: return 0
    pad = pt[-1]
    ok_pad = 1 <= pad <= 16 and pt[-pad:] == bytes([pad]) * pad
    body = pt[:-pad] if ok_pad else pt
    head = body[:400]
    printable = sum(c in b"\t\r\n" or 32 <= c < 127 for c in head) / max(1, len(head))
    s = printable
    if ok_pad: s += 0.5
    if body[:1] in (b"{", b"["): s += 2
    if body[:2] in (b"\x1f\x8b", b"PK", b"\x78\x9c", b"\x78\x01", b"\x78\xda"): s += 2  # gzip/zip/zlib
    return s

results = []
for (kn, k), (ivn, iv) in itertools.product(keys.items(), ivs.items()):
    for mode, off in (("ECB", 0), ("CBC", 0), ("CBC-skip16", 16)):
        payload = data[off:]
        if len(payload) % 16: continue
        try:
            c = AES.new(k, AES.MODE_ECB) if mode == "ECB" else AES.new(k, AES.MODE_CBC, data[:16] if off else iv)
            pt = c.decrypt(payload)
        except Exception:
            continue
        tag = f"{kn:22s} iv={ivn:16s} {mode:11s}"
        results.append((score(pt), tag, pt))
results.sort(key=lambda r: -r[0])
for s, tag, pt in results[:6]:
    print(f"score={s:5.2f}  {tag}  head={pt[:60]!r}")
best = results[0]
if best[0] >= 2:
    out = pathlib.Path("data/extracted") / (BLOB.name + ".bin")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(best[2])
    print(f"\nWROTE {out} ({len(best[2])} bytes) via {best[1]}")
