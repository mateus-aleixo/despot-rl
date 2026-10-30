"""Dump the NodeCanvas dialogue trees out of the build.

The level-entry dialogs are a JSON table (`EncryptedMainGroup/dialogs.json`), but
the quest dialogs are not in it, nor anywhere under the encrypted groups. They
are NodeCanvas `DialogueTree` graphs bound to the quest prefabs, serialized into
`sharedassets3.assets` as each controller's `_serializedGraph` string. The
classes are stripped of their type trees, so the string is read by hand: a
MonoBehaviour's fixed header (`m_GameObject`, `m_Enabled`, `m_Script`, an empty
`m_Name`) is followed by the graph as a length-prefixed UTF-8 string.

    python tools/dump_dialogue_trees.py

Writes one file per graph to `data/extracted/dialogue_trees/`, named
`<prefab root>.<GameObject>` after the object that carries it, plus `index.json` mapping each file to the quest
its text keys belong to (`Quest.<Name>.*`).
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import struct
import sys

import UnityPy

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GAME = pathlib.Path(r"C:\Program Files (x86)\Steam\steamapps\common\Despot's Game\Despot's Game_Data")
OUT = pathlib.Path("data/extracted/dialogue_trees")
MARK = b'{"type":"NodeCanvas.DialogueTrees.DialogueTree"'
KEY = re.compile(r'"Quest\.([A-Za-z]+)\.')


def graph_of(raw: bytes) -> str | None:
    at = raw.find(MARK)
    if at < 4:
        return None
    n = struct.unpack_from("<i", raw, at - 4)[0]
    return raw[at:at + n].decode("utf-8")


def owner_of(raw: bytes, objects: dict) -> tuple[str, str]:
    """The GameObject carrying the graph, and the root of its prefab."""
    path_id = struct.unpack_from("<q", raw, 4)[0]
    go = objects.get(path_id)
    if go is None:
        return f"object-{path_id}", ""
    go = go.read()
    name, root = go.m_Name or f"object-{path_id}", go.m_Name
    for _ in range(32):
        transform = next((c.read() for c in _components(go)
                          if c.type.name in ("Transform", "RectTransform")), None)
        if transform is None or not transform.m_Father.path_id:
            break
        father = objects.get(transform.m_Father.path_id)
        if father is None:
            break
        go = father.read().m_GameObject.read()
        root = go.m_Name
    return name, root


def _components(go):
    for c in go.m_Components:
        ptr = getattr(c, "component", c)
        try:
            yield ptr.deref()
        except Exception:
            continue


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = ([GAME / "resources.assets"] + sorted(GAME.glob("sharedassets*.assets"))
             + sorted(GAME.glob("level[0-9]*")))
    index, seen = {}, collections.Counter()
    for p in paths:
        env = UnityPy.load(str(p))
        objects = {o.path_id: o for o in env.objects}
        for obj in env.objects:
            if obj.type.name != "MonoBehaviour":
                continue
            raw = obj.get_raw_data()
            text = graph_of(raw)
            if text is None:
                continue
            owner, root = owner_of(raw, objects)
            stem = f"{root}.{owner}" if root and root != owner else owner
            seen[stem] += 1
            stem = stem if seen[stem] == 1 else f"{stem}-{seen[stem]}"
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
            graph = json.loads(text)
            (OUT / f"{stem}.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")
            quests = collections.Counter(KEY.findall(text))
            index[stem] = {"file": p.name, "path_id": obj.path_id, "root": root,
                           "nodes": len(graph.get("nodes", [])),
                           "quest": quests.most_common(1)[0][0] if quests else None}
    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    for stem, row in sorted(index.items(), key=lambda kv: (str(kv[1]["quest"]), kv[0])):
        print(f"{stem:50s} {row['file']:22s} nodes {row['nodes']:3d}  quest {row['quest']}")
    print(f"{len(index)} dialogue trees -> {OUT}")


if __name__ == "__main__":
    main()
