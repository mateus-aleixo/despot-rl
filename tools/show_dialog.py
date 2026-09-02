"""Render a level-entry dialog with its text resolved.

The table is EncryptedMainGroup/dialogs.json, which sits beside DB/ rather than
inside it. Every string in it is a localization key; this resolves them against
the chosen language so a dialog can be read as the player sees it.

    python tools/show_dialog.py                 list the pool
    python tools/show_dialog.py WheelOfFortune  render one
    python tools/show_dialog.py --all           render the whole pool
"""
import json, pathlib, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path("data/extracted/json")
DIALOGS = ROOT / "EncryptedMainGroup/dialogs.json"
GAME = ROOT / "EncryptedMainGroup/DB/Game.json"
LANG = ROOT / "EncryptedLocalizationsGroup/Languages"


def load(lang="en"):
    entries = json.loads(DIALOGS.read_text(encoding="utf-8"))
    table = {e["title"][len("Dialog."):-len(".Title")]: e for e in entries}
    pool = json.loads(GAME.read_text(encoding="utf-8"))["Dialogs"]
    loc = json.loads((LANG / f"{lang}.json").read_text(encoding="utf-8"))
    return table, pool, loc


def render(node, loc, indent=0):
    pad = " " * indent
    text = loc.get(node.get("text"), node.get("text"))
    if text:
        print(f"{pad}{text}")
    for e in node.get("events", []):
        params = json.dumps(e.get("parameters", {}), ensure_ascii=False)
        print(f"{pad}  [{e['className']} {params}]")
    for choice in node.get("choices", []):
        print(f"{pad}  > {loc.get(choice['button'], choice['button'])}")
        for e in choice.get("events", []):
            params = json.dumps(e.get("parameters", {}), ensure_ascii=False)
            print(f"{pad}      [{e['className']} {params}]")
        outcomes = choice.get("outcomes", [])
        total = sum(o.get("weight", 0) for o in outcomes)
        for o in outcomes:
            w = o.get("weight")
            share = f"  ({w:g}/{total:g})" if w is not None and total else ""
            if len(outcomes) > 1 or share:
                print(f"{pad}    --{share}")
            render(o, loc, indent + 6)


def main():
    args = [a for a in sys.argv[1:]]
    table, pool, loc = load()
    if not args:
        print(f"{len(pool)} dialogs in the Default pool, {len(table)} in the table\n")
        for name in pool:
            print(f"  {name:<22} {loc.get(f'Dialog.{name}.Title', '')}")
        return
    names = pool if args[0] == "--all" else args
    for name in names:
        if name not in table:
            sys.exit(f"no dialog named {name!r}; run with no arguments to list the pool")
        title = loc.get(f"Dialog.{name}.Title", name)
        print(f"=== {name}: {title}")
        render(table[name]["body"], loc)
        print()


if __name__ == "__main__":
    main()
