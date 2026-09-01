"""Sweep every IL2CPP string literal as an AES passphrase under common derivations.
Only the first two blocks are decrypted per candidate, so the sweep stays cheap."""
import hashlib, pathlib, sys
from Crypto.Cipher import AES
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

data = pathlib.Path("data/raw/textassets/EncryptedMainGroup").read_bytes()
head, iv_from_file, head_after_iv = data[:32], data[:16], data[16:48]
lits = [s for s in pathlib.Path("data/raw/string_literals.txt").read_text(encoding="utf-8").split("\n")
        if 4 <= len(s) <= 96]
print(f"{len(lits)} candidate literals")

def derivations(s):
    b = s.encode("utf-8", "ignore")
    yield "utf8-32", b.ljust(32, b"\0")[:32]
    yield "utf8-16", b.ljust(16, b"\0")[:16]
    yield "utf8-24", b.ljust(24, b"\0")[:24]
    yield "md5", hashlib.md5(b).digest()
    yield "sha256", hashlib.sha256(b).digest()
    yield "sha1-16", hashlib.sha1(b).digest()[:16]

def plausible(pt):
    """All 32 bytes printable: ~1e-14 false-positive rate against random data."""
    if not all(c in b"\t\r\n" or 32 <= c < 127 for c in pt):
        return False
    return pt[:1] in (b"{", b"[", b'"') or pt[:1].isalpha()

hits = tested = 0
for s in lits:
    for dname, k in derivations(s):
        for mode in ("ECB", "CBC0", "CBCF"):
            tested += 1
            try:
                if mode == "ECB":
                    pt = AES.new(k, AES.MODE_ECB).decrypt(head)
                elif mode == "CBC0":
                    pt = AES.new(k, AES.MODE_CBC, b"\0" * 16).decrypt(head)
                else:
                    pt = AES.new(k, AES.MODE_CBC, iv_from_file).decrypt(head_after_iv)
            except Exception:
                continue
            if plausible(pt):
                hits += 1
                print(f"HIT  {dname:8s} {mode:5s}  lit={s[:60]!r}  -> {pt[:48]!r}")
print(f"\ntested {tested} combos, {hits} hits")
