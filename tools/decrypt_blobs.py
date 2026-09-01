"""Decrypt the game's encrypted TextAssets.

Scheme (from EncryptedAssetProvider.InternalOp.ActionComplete):
    StringCipher.Decrypt(bytes, <passphrase>) -> FS.Unzip
StringCipher is the well-known snippet: Rijndael-256 CBC, PKCS7,
key = PBKDF2-HMAC-SHA1(passphrase, salt, 1000, 32), layout salt[32]||iv[32]||ct.
FS.Unzip turns out to be gzip.
"""
import gzip, hashlib, io, os, pathlib, sys, zipfile
sys.path.insert(0, "tools")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rijndael256 import cbc_decrypt

PASS = os.environ.get("DESPOT_PASSPHRASE")
if not PASS:
    sys.exit("set DESPOT_PASSPHRASE to the 32-char literal from your own copy "
             "of the game; notes/datamining.md says where it lives in the binary")
SRC = pathlib.Path("data/raw/textassets")
OUT = pathlib.Path("data/extracted/gamedata")
OUT.mkdir(parents=True, exist_ok=True)

def decrypt(blob):
    salt, iv, ct = blob[:32], blob[32:64], blob[64:]
    key = hashlib.pbkdf2_hmac("sha1", PASS.encode(), salt, 1000, 32)
    pt = cbc_decrypt(ct, key, iv)
    pad = pt[-1]
    if 1 <= pad <= 32 and pt[-pad:] == bytes([pad]) * pad:
        pt = pt[:-pad]
    return pt

for name in ("metadata", "EncryptedMainGroup", "EncryptedMainTasksGroup",
             "EncryptedLocalizationsGroup", "EncryptedDLCGroup"):
    f = SRC / name
    if not f.exists():
        continue
    pt = decrypt(f.read_bytes())
    if pt[:2] == b"\x1f\x8b":
        pt = gzip.decompress(pt)
    print(f"=== {name}: {len(pt)} bytes  head={pt[:70]!r}")
    if pt[:2] == b"PK":
        d = OUT / name
        d.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(pt)) as z:
            for i in z.infolist():
                print(f"    {i.file_size:9d}  {i.filename}")
                tgt = d / i.filename
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_bytes(z.read(i))
    else:
        (OUT / f"{name}.txt").write_bytes(pt)
