"""Brute-force the StringCipher passphrase over the IL2CPP literal table."""
import hashlib, pathlib, sys, time
sys.path.insert(0, "tools")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rijndael256 import cbc_decrypt

blob = pathlib.Path("data/raw/textassets/EncryptedMainGroup").read_bytes()
salt, iv, ct = blob[:32], blob[32:64], blob[64:]
lits = [s for s in pathlib.Path("data/raw/string_literals.txt").read_text(encoding="utf-8").split("\n") if s]
print(f"{len(lits)} candidates; salt={salt[:8].hex()}... iv={iv[:8].hex()}...")

t0, hits = time.time(), []
for n, p in enumerate(lits):
    key = hashlib.pbkdf2_hmac("sha1", p.encode("utf-8", "ignore"), salt, 1000, 32)
    pt = cbc_decrypt(ct, key, iv, limit=64)
    if all(c in b"\t\r\n" or 32 <= c < 127 for c in pt[:32]):
        hits.append((p, pt))
        print(f"\nHIT passphrase={p!r}\n  {pt[:96]!r}")
    if n % 2000 == 0 and n:
        print(f"  {n}/{len(lits)}  {time.time()-t0:.0f}s")
print(f"\ndone in {time.time()-t0:.0f}s, {len(hits)} hits")
if hits:
    pathlib.Path("data/extracted/passphrase.txt").write_text(hits[0][0], encoding="utf-8")
