"""Print a ripped prefab's MonoBehaviour components with script names resolved."""
import pathlib, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path("data/extracted/ripped/ExportedProject/Assets")
GUID = {}
for m in ROOT.rglob("*.cs.meta"):
    t = m.read_text(encoding="utf-8", errors="replace")
    g = re.search(r"guid:\s*([a-f0-9]{32})", t)
    if g:
        GUID[g.group(1)] = m.name[:-8]

target = pathlib.Path(sys.argv[1])
blocks = re.split(r"^--- !u!(\d+) &(\d+)", target.read_text(encoding="utf-8", errors="replace"), flags=re.M)
want = set(a for a in sys.argv[2:]) or None
for i in range(1, len(blocks), 3):
    cls, fid, body = blocks[i], blocks[i + 1], blocks[i + 2]
    if cls != "114":       # MonoBehaviour
        continue
    g = re.search(r"m_Script:.*guid:\s*([a-f0-9]{32})", body)
    name = GUID.get(g.group(1), g.group(1)) if g else "?"
    if want and name not in want:
        continue
    lines = [l for l in body.splitlines() if l.strip() and not re.match(r"\s*(m_ObjectHideFlags|m_CorrespondingSourceObject|m_PrefabInstance|m_PrefabAsset|m_GameObject|m_Enabled|m_EditorHideFlags|m_Script|m_Name|m_EditorClassIdentifier):", l)]
    print(f"\n===== {name}  (fileID {fid}) =====")
    print("\n".join(lines[:60]))
