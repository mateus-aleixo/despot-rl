"""Build a VA -> symbol map from Il2CppDumper's script.json."""
import json, pathlib, pickle
src = pathlib.Path("data/extracted/il2cpp/script.json")
d = json.loads(src.read_text(encoding="utf-8"))
print("top-level keys:", list(d.keys()))
m = {}
for key in d:
    entries = d[key]
    if not isinstance(entries, list): continue
    n = 0
    for e in entries:
        if isinstance(e, dict) and "Address" in e and ("Name" in e or "Signature" in e):
            m[int(e["Address"])] = e.get("Name") or e.get("Signature")
            n += 1
    print(f"  {key}: {len(entries)} entries, {n} addressed")
pathlib.Path("data/extracted/symmap.pkl").write_bytes(pickle.dumps(m))
print(f"\n{len(m)} addresses mapped")
