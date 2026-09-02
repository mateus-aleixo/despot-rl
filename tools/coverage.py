"""What the game ships against what this sim reads.

The scope of this project has been discovered by accident: the `RoomType` column
in `EnemyPacks.json` sat unused for weeks, `ChipChoice/Squads.json` was never
loaded at all, and the starting squad was wrong three separate times because
nobody had enumerated what the data actually contains. This enumerates it.

    python tools/coverage.py              # every table, every column
    python tools/coverage.py --classes    # also the C_/M_ classes in dump.cs
    python tools/coverage.py --missing    # only what the sim never mentions

A column counts as read if its name appears anywhere in `sim/` or `rl/`. That
over-counts -- a name can appear in a comment, and reading a key is not the same
as implementing what it means -- so treat a hit as "not obviously missing" and a
miss as hard evidence. The point is the miss list.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = pathlib.Path("data/extracted/json/EncryptedMainGroup/DB")
DUMP = pathlib.Path("data/extracted/il2cpp/dump.cs")


def sim_text() -> str:
    """Everything the simulator and the RL side actually say."""
    files = subprocess.run(["git", "-c", "safe.directory=*", "ls-files", "*.py"],
                           capture_output=True, text=True).stdout.split()
    out = []
    for f in files:
        if f.startswith(("sim/", "rl/")):
            out.append(pathlib.Path(f).read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def columns(obj, depth: int = 0, cap: int = 2) -> set[str]:
    """Every key name in a shipped table, a couple of levels down."""
    found: set[str] = set()
    if depth > cap:
        return found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                found.add(k)
            found |= columns(v, depth + 1, cap)
    elif isinstance(obj, list):
        for v in obj[:200]:
            found |= columns(v, depth + 1, cap)
    return found


def tables() -> dict[str, object]:
    out = {}
    for f in sorted(DB.rglob("*.json")):
        try:
            out[str(f.relative_to(DB))] = json.loads(
                f.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            out[str(f.relative_to(DB))] = f"<unreadable: {type(exc).__name__}>"
    return out


def game_classes() -> dict[str, list[str]]:
    """`C_` controllers and `M_` models by family, from the IL2CPP dump."""
    if not DUMP.exists():
        return {}
    text = DUMP.read_text(encoding="utf-8", errors="replace")
    names = set(re.findall(r"^public (?:sealed |abstract )?class ([CM]_[A-Za-z0-9_]+)",
                           text, re.M))
    fams = collections.defaultdict(list)
    for n in sorted(names):
        kind = n[:2]
        fams[kind].append(n)
    return fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", action="store_true",
                    help="also report the C_/M_ classes in dump.cs")
    ap.add_argument("--missing", action="store_true",
                    help="print only the names the sim never mentions")
    args = ap.parse_args()

    src = sim_text()
    data = tables()
    total_cols = miss_cols = 0
    unread_tables = []

    print(f"{'table':40s} {'cols':>5s} {'read':>5s} {'missing':>7s}")
    print("-" * 62)
    detail: dict[str, list[str]] = {}
    for name, obj in sorted(data.items()):
        if isinstance(obj, str):
            print(f"{name:40s} {obj}")
            continue
        cols = sorted(c for c in columns(obj) if len(c) > 2)
        hit = [c for c in cols if c in src]
        gone = [c for c in cols if c not in src]
        total_cols += len(cols)
        miss_cols += len(gone)
        detail[name] = gone
        stem = pathlib.Path(name).stem
        if stem not in src:
            unread_tables.append(name)
        print(f"{name:40s} {len(cols):5d} {len(hit):5d} {len(gone):7d}"
              + ("   TABLE NEVER NAMED" if stem not in src else ""))

    print(f"\n{len(data)} tables, {total_cols} distinct column names, "
          f"{miss_cols} never mentioned in sim/ or rl/ "
          f"({miss_cols / max(1, total_cols):.0%})")
    if unread_tables:
        print(f"\ntables this sim never names at all ({len(unread_tables)}):")
        for t in unread_tables:
            print(f"   {t}")

    if args.missing:
        print("\ncolumns never mentioned, by table:")
        for name, gone in sorted(detail.items()):
            if gone:
                print(f"\n  {name}")
                for i in range(0, len(gone), 6):
                    print("     " + "  ".join(f"{c:22s}" for c in gone[i:i + 6]))

    if args.classes:
        fams = game_classes()
        print()
        for kind, names in sorted(fams.items()):
            named = [n for n in names if n in src]
            print(f"{kind} classes in the dump: {len(names)}, "
                  f"mentioned in sim/ or rl/: {len(named)}")


if __name__ == "__main__":
    main()
