"""Print a parsed NodeCanvas behaviour tree as an indented outline."""
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BT = pathlib.Path("data/extracted/bt")


def load(name):
    return json.loads((BT / f"{name}.json").read_text(encoding="utf-8"))


def children_map(graph):
    """Child ids per source id, in connection order (that is BT child order)."""
    kids = {}
    for c in graph.get("connections", []):
        src = c.get("_sourceNode", {}).get("$ref")
        tgt = c.get("_targetNode", {}).get("$ref")
        kids.setdefault(src, []).append(tgt)
    return kids


def label(node):
    t = node.get("$type", "?").split(".")[-1]
    for key in ("_action", "_condition"):
        inner = node.get(key)
        if isinstance(inner, dict):
            extra = {k: v for k, v in inner.items()
                     if not k.startswith("$") and not k.startswith("_")}
            name = inner.get("$type", "?")
            return f"{t}<{name}>" + (f" {json.dumps(extra)}" if extra else "")
    flags = []
    if node.get("dynamic"):
        flags.append("dynamic")
    for k in ("_repeat", "_times", "_policy", "_mode"):
        if k in node:
            flags.append(f"{k}={node[k]}")
    return t + (f" [{', '.join(flags)}]" if flags else "")


def show(name, max_depth=12):
    g = load(name)
    nodes = {n["$id"]: n for n in g["nodes"]}
    kids = children_map(g)
    print(f"\n===== {name}  ({len(nodes)} nodes, repeat={g.get('derivedData', {}).get('repeat')}) =====")

    seen = set()

    def walk(nid, depth):
        if depth > max_depth or nid not in nodes:
            return
        if nid in seen:
            print("  " * depth + f"- (revisit {nid})")
            return
        seen.add(nid)
        print("  " * depth + f"- {label(nodes[nid])}")
        for k in kids.get(nid, []):
            walk(k, depth + 1)

    walk("0", 0)
    orphans = [i for i in nodes if i not in seen]
    if orphans:
        print(f"  (unreached nodes: {[label(nodes[i]) for i in orphans]})")


if __name__ == "__main__":
    for n in sys.argv[1:] or ["NC-BaseUnit"]:
        show(n)
