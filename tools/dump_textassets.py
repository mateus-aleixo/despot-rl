"""Dump every TextAsset out of the build; these often carry balance tables."""
import pathlib
import UnityPy

GAME = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\Despot's Game_Data")
OUT = pathlib.Path("data/raw/textassets"); OUT.mkdir(parents=True, exist_ok=True)
PATHS = ([GAME / "resources.assets"] + sorted(GAME.glob("sharedassets*.assets"))
         + sorted(GAME.glob("level*")) + sorted((GAME / "StreamingAssets/aa/StandaloneWindows64").glob("*.bundle")))

def as_bytes(v):
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, str):
        return v.encode("utf-8", "surrogateescape")
    return bytes(bytearray(v))

n = 0
for p in PATHS:
    try:
        env = UnityPy.load(str(p))
    except Exception:
        continue
    for obj in env.objects:
        if obj.type.name != "TextAsset":
            continue
        ta = obj.read()
        try:
            raw = as_bytes(ta.m_Script)
        except Exception as e:
            print(f"!! {ta.m_Name}: {type(e).__name__}: {e}"); continue
        name = (ta.m_Name or f"unnamed_{obj.path_id}").replace("/", "_")
        (OUT / name).write_bytes(raw)
        printable = sum(c in b"\t\r\n" or 32 <= c < 127 for c in raw[:400])
        kind = "text" if raw[:400] and printable / max(1, len(raw[:400])) > 0.9 else "binary"
        print(f"{len(raw):9d}  {kind:6s}  {name:38s} [{p.name[:18]}]")
        if kind == "text":
            print("        " + raw[:200].decode("utf-8", "replace").replace("\n", " ")[:190])
        n += 1
print(f"\n{n} TextAssets -> {OUT}")
