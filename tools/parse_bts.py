"""Parse the exported NodeCanvas behaviour trees and inventory their node types."""
import collections, json, pathlib, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BT = pathlib.Path("data/extracted/ripped/ExportedProject/Assets/Behavior Trees")
OUT = pathlib.Path("data/extracted/bt"); OUT.mkdir(parents=True, exist_ok=True)

actions, conditions, composites, trees = collections.Counter(), collections.Counter(), collections.Counter(), {}
for f in sorted(BT.glob("*.asset")):
    text = f.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"_serializedGraph:\s*'(.*?)'\n\s*_objectReferences", text, re.S)
    if not m:
        m = re.search(r"_serializedGraph:\s*'(.*)'", text, re.S)
    if not m:
        print(f"  !! no graph in {f.name}")
        continue
    raw = m.group(1).replace("''", "'")
    try:
        g = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  !! {f.name}: {e}")
        continue
    trees[f.stem] = g
    (OUT / f"{f.stem}.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
    for n in g.get("nodes", []):
        t = n.get("$type", "?").split(".")[-1]
        composites[t] += 1
        if "_action" in n and isinstance(n["_action"], dict):
            actions[n["_action"].get("$type", "?")] += 1
        if "_condition" in n and isinstance(n["_condition"], dict):
            conditions[n["_condition"].get("$type", "?")] += 1

print(f"parsed {len(trees)} trees -> {OUT}\n")
print("== node kinds ==")
for k, v in composites.most_common(20):
    print(f"{v:5d}  {k}")
print(f"\n== actions ({len(actions)}) ==")
for k, v in actions.most_common(60):
    print(f"{v:5d}  {k}")
print(f"\n== conditions ({len(conditions)}) ==")
for k, v in conditions.most_common(60):
    print(f"{v:5d}  {k}")
