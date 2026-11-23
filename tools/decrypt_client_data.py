"""Extract the balance data the game actually plays: its server copy.

The tables in the install (the `EncryptedMainGroup` TextAsset and friends,
`tools/decrypt_blobs.py`) are only where the game starts. `C_ClientData` keeps
the data the server hands out in a persistent model, `clientData.bin` in the
player's `LocalLow/Konfa Games/Despot's Game`, and that is what a run uses
(notes/reference-sim.md, "The live data, found").

The file is two layers, both `StringCipher` (Rijndael-256, `tools/rijndael256.py`)
then gzip:

  * the persistent model: `C_PersistentModel<object>.Load` decrypts it with a
    32-character literal of its own and unzips it, giving JSON with `version`,
    `data`, `hash` and `date`;
  * `data`: base64 of the shipped tables' own scheme, the asset passphrase,
    whose plain text has their `path\\n<body>\\n` records.

Neither passphrase is in this repository. Set them from your own copy of the
game, where notes/datamining.md and the function above say they live:

    DESPOT_PASSPHRASE       the asset literal
    DESPOT_SAVE_PASSPHRASE  the persistent-model literal

    python tools/decrypt_client_data.py [path/to/clientData.bin]

Writes data/extracted/live/: clientdata.json (version, date, hash),
gamedata/EncryptedMainGroup.txt (the payload) and json/EncryptedMainGroup/...
(one file per record, as tools/split_gamedata.py lays out the install's).
`sim.data.load_ruleset(source="live")` reads it.
"""
import base64
import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rijndael256 import cbc_decrypt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "extracted" / "live"
DEFAULT_SRC = (pathlib.Path.home() / "AppData" / "LocalLow" / "Konfa Games"
               / "Despot's Game" / "clientData.bin")
EXT = re.compile(r"\.(json|csv)$", re.I)
BS = chr(92)


def string_cipher(blob: bytes, passphrase: str) -> bytes:
    """`StringCipher.Decrypt`: salt[32] || iv[32] || ciphertext, the key
    PBKDF2-HMAC-SHA1 over 1000 rounds, Rijndael-256 CBC, PKCS7."""
    salt, iv, ct = blob[:32], blob[32:64], blob[64:]
    key = hashlib.pbkdf2_hmac("sha1", passphrase.encode(), salt, 1000, 32)
    pt = cbc_decrypt(ct, key, iv)
    pad = pt[-1]
    if 1 <= pad <= 32 and pt[-pad:] == bytes([pad]) * pad:
        pt = pt[:-pad]
    return pt


def gunzip(b: bytes) -> bytes:
    if b[:2] != b"\x1f\x8b":
        raise ValueError("expected gzip after decryption; is the passphrase right?")
    return gzip.GzipFile(fileobj=io.BytesIO(b)).read()


def unescape(s: str) -> str:
    return (s.replace(BS + "r" + BS + "n", "\n").replace(BS + "n", "\n")
             .replace(BS + "t", "\t").replace(BS + '"', '"'))


def main() -> None:
    asset_pass = os.environ.get("DESPOT_PASSPHRASE")
    save_pass = os.environ.get("DESPOT_SAVE_PASSPHRASE")
    if not asset_pass or not save_pass:
        sys.exit("set DESPOT_PASSPHRASE and DESPOT_SAVE_PASSPHRASE (see the docstring)")
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    model = json.loads(gunzip(string_cipher(src.read_bytes(), save_pass)).decode("utf-8-sig"))
    payload = gunzip(string_cipher(base64.b64decode(model["data"]), asset_pass)).decode("utf-8")

    (OUT / "gamedata").mkdir(parents=True, exist_ok=True)
    (OUT / "gamedata" / "EncryptedMainGroup.txt").write_text(payload, encoding="utf-8")
    meta = {k: model.get(k) for k in ("version", "date", "hash")}
    (OUT / "clientdata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    group = OUT / "json" / "EncryptedMainGroup"
    lines, i, n = payload.split("\n"), 0, 0
    while i < len(lines):
        name = lines[i].strip()
        if EXT.search(name) and i + 1 < len(lines):
            body, tgt = lines[i + 1], group / name
            tgt.parent.mkdir(parents=True, exist_ok=True)
            if name.lower().endswith(".csv"):
                tgt.write_text(unescape(body), encoding="utf-8")
            else:
                try:
                    tgt.write_text(json.dumps(json.loads(body), indent=1, ensure_ascii=False),
                                   encoding="utf-8")
                except json.JSONDecodeError:
                    tgt.write_text(body, encoding="utf-8")
            n += 1
            i += 2
        else:
            i += 1
    print(f"version {meta['version']}, dated {meta['date']}: {n} files -> {group}")


if __name__ == "__main__":
    main()
