"""Survey a Unity IL2CPP build: what object types exist, and which
ScriptableObject (MonoBehaviour) classes are present, by name."""
import sys, collections, pathlib
import UnityPy

GAME = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\Despot's Game_Data")

def survey(paths):
    types = collections.Counter()
    scripts = collections.Counter()   # class name -> count of MonoBehaviours
    per_file = {}
    for p in paths:
        try:
            env = UnityPy.load(str(p))
        except Exception as e:
            print(f"  !! {p.name}: {type(e).__name__}: {e}")
            continue
        local = collections.Counter()
        for obj in env.objects:
            tname = obj.type.name
            types[tname] += 1
            local[tname] += 1
            if tname == "MonoBehaviour":
                try:
                    mb = obj.read(check_read=False)
                    sref = getattr(mb, "m_Script", None)
                    cls = "?"
                    if sref:
                        try:
                            ms = sref.read()
                            cls = f"{ms.m_Namespace + '.' if ms.m_Namespace else ''}{ms.m_ClassName}"
                        except Exception:
                            cls = "<unresolved script>"
                    scripts[cls] += 1
                except Exception as e:
                    scripts[f"<read failed: {type(e).__name__}>"] += 1
        per_file[p.name] = local
    return types, scripts, per_file

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "bundles"
    if which == "bundles":
        paths = sorted((GAME / "StreamingAssets/aa/StandaloneWindows64").glob("*.bundle"))
    else:
        paths = [GAME / "resources.assets"] + sorted(GAME.glob("sharedassets*.assets")) + sorted(GAME.glob("level*"))
    print(f"scanning {len(paths)} files\n")
    types, scripts, per_file = survey(paths)
    print("== object types ==")
    for t, c in types.most_common(30):
        print(f"{c:7d}  {t}")
    print("\n== MonoBehaviour script classes ==")
    for s, c in scripts.most_common(60):
        print(f"{c:7d}  {s}")
    print("\n== per file ==")
    for f, c in per_file.items():
        print(f"{f}: {sum(c.values())} objs, top={c.most_common(4)}")
