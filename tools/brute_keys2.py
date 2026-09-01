"""Format-agnostic key sweep: PKCS7 padding validity on the final block is the
primary filter (works even if the payload is compressed before encryption)."""
import hashlib, pathlib, sys
from Crypto.Cipher import AES
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

data = pathlib.Path("data/raw/textassets/EncryptedMainGroup").read_bytes()
lits = [s for s in pathlib.Path("data/raw/string_literals.txt").read_text(encoding="utf-8").split("\n")
        if 4 <= len(s) <= 96]
MAGIC = (b"\x1f\x8b", b"PK", b"\x78\x01", b"\x78\x9c", b"\x78\xda", b"\x04\x22", b"BZh", b"\xfd7zX")

def derivations(s):
    b = s.encode("utf-8", "ignore")
    yield "utf8-32", b.ljust(32, b"\0")[:32]
    yield "utf8-16", b.ljust(16, b"\0")[:16]
    yield "md5", hashlib.md5(b).digest()
    yield "sha256", hashlib.sha256(b).digest()
    yield "sha1-16", hashlib.sha1(b).digest()[:16]

def good_pad(block):
    n = block[-1]
    return 1 <= n <= 16 and block[-n:] == bytes([n]) * n

def head_ok(pt):
    if pt.startswith(MAGIC):
        return True
    return all(c in b"\t\r\n" or 32 <= c < 127 for c in pt[:16])

hits = tested = padhits = 0
for s in lits:
    for dname, k in derivations(s):
        for mode in ("ECB", "CBC0", "CBCF"):
            tested += 1
            try:
                if mode == "ECB":
                    last = AES.new(k, AES.MODE_ECB).decrypt(data[-16:])
                    if not good_pad(last): continue
                    head = AES.new(k, AES.MODE_ECB).decrypt(data[:16])
                elif mode == "CBC0":
                    last = AES.new(k, AES.MODE_CBC, data[-32:-16]).decrypt(data[-16:])
                    if not good_pad(last): continue
                    head = AES.new(k, AES.MODE_CBC, b"\0" * 16).decrypt(data[:16])
                else:
                    last = AES.new(k, AES.MODE_CBC, data[-32:-16]).decrypt(data[-16:])
                    if not good_pad(last): continue
                    head = AES.new(k, AES.MODE_CBC, data[:16]).decrypt(data[16:32])
            except Exception:
                continue
            padhits += 1
            if head_ok(head):
                hits += 1
                print(f"HIT {dname:8s} {mode:5s} lit={s[:60]!r} head={head[:24]!r} pad={last[-1]}")
print(f"\ntested {tested}, padding-valid {padhits}, full hits {hits}")
