"""What the game ships against what this sim reads.

The scope of this project has been discovered by accident: the `RoomType` column
in `EnemyPacks.json` sat unused for weeks, `ChipChoice/Squads.json` was never
loaded at all, and the starting squad was wrong three separate times because
nobody had enumerated what the data actually contains. This enumerates it.

    python tools/coverage.py              # every table, every column
    python tools/coverage.py --classes    # also the C_/M_ classes in dump.cs
    python tools/coverage.py --missing    # only what the sim never mentions

A column counts as read if its name appears in `sim/` or `rl/`, **or** if it
matches an f-string the source builds. That second half matters: the first
version of this tool did a plain substring scan and reported `Skills.json` as 23
columns of which the sim read one, because every `Param1Name` is reached through
`row.get(f"Param{i}Name")`. It reads all of them. The same held for `Q7Prob`
(`f"Q{q}Prob"`), `DamagePerLevel` (`f"{key}PerLevel"`) and `Skill3`.

It still over-counts reads: a name can appear in a comment, and reading a key is
not the same as implementing what it means. So a hit means "not obviously
missing" and a miss is hard evidence. The point is the miss list.
"""
from __future__ import annotations

import argparse
import ast
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


def py_files() -> list[str]:
    """The simulator and the RL side, as tracked paths."""
    files = subprocess.run(["git", "-c", "safe.directory=*", "ls-files", "*.py"],
                           capture_output=True, text=True).stdout.split()
    return [f for f in files if f.startswith(("sim/", "rl/"))]


def sim_text() -> str:
    return "\n".join(pathlib.Path(f).read_text(encoding="utf-8", errors="replace")
                     for f in py_files())


def dynamic_patterns() -> list[re.Pattern]:
    """Column families the source builds with an f-string.

    Walks the AST rather than the text, so `f"Param{i}Name"` becomes
    `^Param.+Name$` and every `Param1Name` in the shipped data counts as read.
    An f-string that is nothing but a hole is skipped, since it would match
    every column name there is.
    """
    out: list[re.Pattern] = []
    for path in py_files():
        try:
            tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8",
                                                          errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            parts, hole = [], False
            for piece in node.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    parts.append(re.escape(piece.value))
                else:
                    parts.append(".+")
                    hole = True
            pat = "".join(parts)
            if not hole or pat.strip(".+") == "":
                continue
            out.append(re.compile("^" + pat + "$"))
    return out


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
    """`C_` controllers and `M_` models, from the IL2CPP dump."""
    if not DUMP.exists():
        return {}
    text = DUMP.read_text(encoding="utf-8", errors="replace")
    names = set(re.findall(
        r"^public (?:sealed |abstract )?class ([CM]_[A-Za-z0-9_]+)", text, re.M))
    fams = collections.defaultdict(list)
    for n in sorted(names):
        fams[n[:2]].append(n)
    return fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", action="store_true")
    ap.add_argument("--missing", action="store_true")
    args = ap.parse_args()

    src = sim_text()
    pats = dynamic_patterns()

    def reads(col: str) -> bool:
        return col in src or any(p.match(col) for p in pats)

    data = tables()
    total = miss = 0
    never_named = []
    detail: dict[str, list[str]] = {}

    print(f"{'table':40s} {'cols':>5s} {'read':>5s} {'missing':>7s}")
    print("-" * 62)
    for name, obj in sorted(data.items()):
        if isinstance(obj, str):
            print(f"{name:40s} {obj}")
            continue
        cols = sorted(c for c in columns(obj) if len(c) > 2)
        gone = [c for c in cols if not reads(c)]
        total += len(cols)
        miss += len(gone)
        detail[name] = gone
        stem = pathlib.Path(name).stem
        if stem not in src:
            never_named.append(name)
        print(f"{name:40s} {len(cols):5d} {len(cols) - len(gone):5d} {len(gone):7d}"
              + ("   TABLE NEVER NAMED" if stem not in src else ""))

    print(f"\n{len(data)} tables, {total} distinct column names, "
          f"{miss} never mentioned ({miss / max(1, total):.0%})")
    if never_named:
        print(f"\ntables this sim never names at all ({len(never_named)}):")
        for t in never_named:
            print(f"   {t}")

    if args.missing:
        print("\ncolumns never mentioned, by table:")
        for name, gone in sorted(detail.items()):
            if gone:
                print(f"\n  {name}")
                for i in range(0, len(gone), 5):
                    print("     " + "  ".join(f"{c:24s}" for c in gone[i:i + 5]))

    if args.classes:
        print()
        for kind, names in sorted(game_classes().items()):
            named = [n for n in names if n in src]
            print(f"{kind} classes in the dump: {len(names)}, "
                  f"mentioned in sim/ or rl/: {len(named)}")


if __name__ == "__main__":
    main()
