"""Hunt for the balance tables: TextAssets and stat-shaped ScriptableObject classes."""
import re, pathlib, collections
import UnityPy

GAME = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\Despot's Game_Data")
PATHS = ([GAME / "resources.assets"] + sorted(GAME.glob("sharedassets*.assets"))
         + sorted(GAME.glob("level*")) + sorted((GAME / "StreamingAssets/aa/StandaloneWindows64").glob("*.bundle")))
KEY = re.compile(r"config|data|balance|stat|weapon|item|mutat|unit|class|preset|table|settings|shop|enemy|boss|squad|level",
                 re.I)
SKIP = re.compile(r"^(UnityEngine|TMPro|SuperTiled2Unity|DarkTonic|Rewired|Pathfinding|NodeCanvas|Spine)\.", re.I)

texts, classes = [], collections.Counter()
for p in PATHS:
    try:
        env = UnityPy.load(str(p))
    except Exception:
        continue
    for obj in env.objects:
        if obj.type.name == "TextAsset":
            try:
                ta = obj.read()
                raw = bytes(ta.m_Script) if not isinstance(ta.m_Script, (bytes, bytearray)) else ta.m_Script
                texts.append((p.name, ta.m_Name, len(raw), raw[:120]))
            except Exception as e:
                texts.append((p.name, f"<fail {type(e).__name__}>", 0, b""))
        elif obj.type.name == "MonoBehaviour":
            try:
                mb = obj.read(check_read=False)
                s = getattr(mb, "m_Script", None)
                if not s:
                    continue
                ms = s.read()
                cls = ms.m_ClassName
                if SKIP.search(f"{ms.m_Namespace}.{cls}") or not KEY.search(cls):
                    continue
                name = getattr(mb, "m_Name", "") or ""
                classes[cls] += 1
                if classes[cls] <= 3 and name:
                    print(f"  eg {cls}: {name}")
            except Exception:
                pass

print("\n== TextAssets ==")
for f, n, sz, head in texts:
    print(f"{sz:9d}  {n:40s} [{f[:18]}]  {head[:70]!r}")
print("\n== candidate ScriptableObject classes ==")
for c, n in classes.most_common(50):
    print(f"{n:6d}  {c}")
