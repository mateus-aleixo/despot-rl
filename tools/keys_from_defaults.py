"""Slide an AES key window over IL2CPP's field/parameter default-value blob.
`byte[] key = new byte[]{...}` initializers are stored there verbatim."""
import pathlib, struct, sys
from Crypto.Cipher import AES
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MD = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\Despot's Game_Data\il2cpp_data\Metadata\global-metadata.dat")
b = MD.read_bytes()
dvOff, dvSize = struct.unpack_from("<ii", b, 72)
blob = b[dvOff:dvOff + dvSize]
print(f"fieldAndParameterDefaultValueData @{dvOff} size={dvSize}")
pathlib.Path("data/raw/default_values.bin").write_bytes(blob)

data = pathlib.Path("data/raw/textassets/EncryptedMainGroup").read_bytes()
MAGIC = (b"\x1f\x8b", b"PK", b"\x78\x01", b"\x78\x9c", b"\x78\xda", b"BZh", b"\xfd7zX", b"\x04\x22")

def good_pad(x):
    n = x[-1]
    return 1 <= n <= 16 and x[-n:] == bytes([n]) * n

def head_ok(pt):
    return pt.startswith(MAGIC) or all(c in b"\t\r\n" or 32 <= c < 127 for c in pt)

hits = tested = pad_ok = 0
for size in (32, 16, 24):
    for i in range(0, len(blob) - size + 1):
        k = blob[i:i + size]
        for mode in ("ECB", "CBC0", "CBCF"):
            tested += 1
            try:
                if mode == "ECB":
                    if not good_pad(AES.new(k, AES.MODE_ECB).decrypt(data[-16:])): continue
                    head = AES.new(k, AES.MODE_ECB).decrypt(data[:16])
                else:
                    if not good_pad(AES.new(k, AES.MODE_CBC, data[-32:-16]).decrypt(data[-16:])): continue
                    iv = b"\0" * 16 if mode == "CBC0" else data[:16]
                    src = data[:16] if mode == "CBC0" else data[16:32]
                    head = AES.new(k, AES.MODE_CBC, iv).decrypt(src)
            except Exception:
                continue
            pad_ok += 1
            if head_ok(head):
                hits += 1
                print(f"HIT keysize={size} off={i} mode={mode} key={k.hex()}\n    head={head!r}")
    print(f"  ...size {size} done, tested={tested}, pad_ok={pad_ok}, hits={hits}")
print(f"\ntested {tested}, padding-valid {pad_ok}, hits {hits}")
