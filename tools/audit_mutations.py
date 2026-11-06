"""Which parameters each implemented mutation actually reads: Batch 2's audit.

Three handlers once looked up a key their rows do not have (`vampirism` read
`percent` where the rows say `value`, then `ExplodeProjectile`, then
`EventualKnockback`) and did nothing while counting as implemented. This runs
every implemented mutation row of the Default ruleset down its whole path --
`apply_to_specs`, `apply_class_skills`, `apply_to_agents`, the core's packer,
and a stretch of fight in the Python oracle so the on-attack, on-damaged,
on-cast and on-death hooks fire -- with the row's parameters in a dict that
records every key looked up. Per row it reports:

    unread   a parameter the row sets that nothing looked at
    missing  a key looked up that the row does not set (so a default stood in)

An unread parameter is not a bug by itself (`VisibleSkill` is for the UI), but
each needs a reason, and a `missing` key beside an unread look-alike is how all
three earlier bugs looked. A row whose dict was iterated (copied or walked)
counts every key as read and is marked `*`.

    python tools/audit_mutations.py [--seconds 20] [--only Name] [--all]
"""
import argparse
import random
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dataclasses

import sim.mutations as M
from sim.assumptions import DEFAULT
from sim.battle import Battle, apply_class_skills, deploy
from sim.data import items_by_name, load_ruleset, parse_room_layouts
from sim.nav import Grid
from sim.spec import build_enemy_pack, build_player_squad
from sim.unit_skills import ACTIVE, PASSIVE

READS: dict[int, set] = {}
ITERATED: set[int] = set()

# Parameters a row may leave unread, and why. Anything else unread is open.
EXPLAINED = {
    "VisibleSkill": "UI only: whether the skill shows on the unit's card",
    "mechanic": "`M_StatBonusStatus.mechanic` is read only by `Qualify` (purges); "
                "a stat mutation is a standing spec change here",
    "kind": "read from the row itself by `kind_applies`",
}
# Rows whose numbers follow the run (food, experience): their run-layer half is
# `RunState.feed_cost` and `RunState.gain_experience`, outside a fight.
RUN_LAYER = {"UnderFeed": {"value"}, "Intelligence": {"value"}}


class RecordingDict(dict):
    """A params dict that notes every key looked up, under its row's ID."""

    def __init__(self, data, tag):
        super().__init__(data)
        self.tag = tag

    def _note(self, key):
        READS.setdefault(self.tag, set()).add(key)

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._note(key)
        return super().__contains__(key)

    def _walk(self):
        ITERATED.add(self.tag)

    def __iter__(self):
        self._walk()
        return super().__iter__()

    def keys(self):
        self._walk()
        return super().keys()

    def items(self):
        self._walk()
        return super().items()

    def values(self):
        self._walk()
        return super().values()


_original_params = M.mutation_params


def recording_params(m):
    return RecordingDict(_original_params(m), m.get("ID"))


def pick_class(row, item_class):
    classes = M.offer_classes(row)
    if classes == ["All"]:
        return "Warrior"
    for c in ("Warrior", *classes):
        if c in classes and c in item_class:
            return c
    return next((c for c in classes if c in item_class), None)


def exercise(tables, row, cls, item, seconds, grid, layout, other_item):
    """The row down every path a fight takes, for `seconds` of oracle time."""
    held = M.for_class(row, cls)
    rng = random.Random(1)
    # ClassDiversity counts another class, so the squad carries two of it.
    p = _original_params(row)
    other = p.get("otherClass") or p.get("targetClass")
    loadout = [(item, 1)] * 2 + [(other_item.get(other, item), 1)] * 2
    specs = build_player_squad(tables, loadout)
    # gold 0 and hunger 2 switch HoboFury and HungerFury on
    specs = M.apply_to_specs(specs, [held], rng, gold=0.0, hunger=2)
    pack = next(p for p in tables["EnemyPacks"] if p["Class"] == "Mancrack")
    enemies = build_enemy_pack(tables, pack)[:6]
    t0 = deploy(grid, layout, specs, 0, "p", rng)
    t1 = deploy(grid, layout, enemies, 1, "e1", rng)
    apply_class_skills(tables, t0)
    M.apply_to_agents(t0, [held], random.Random(1))
    for agent in t0:
        M.boost_agent_chances(agent, M.chance_bonuses([held], agent.spec.cls))
    M.underdog(t0, len(t1), [held])
    a = dataclasses.replace(DEFAULT, max_fight_seconds=seconds)
    b = Battle(grid, t0 + t1, assumptions=a, seed=1, tables=tables, build_trees=True)
    try:
        from sim.fast import pack_all
        pack_all(b.agents, tables)
    except Exception:   # outside the core's envelope: the oracle still ran it
        pass
    b.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--only", default=None, help="one mutation Name")
    ap.add_argument("--all", action="store_true", help="also print rows with nothing to flag")
    args = ap.parse_args()

    M.mutation_params = recording_params
    tables = load_ruleset("Default", "default", strict=True)
    layout = parse_room_layouts(tables["RoomLayouts"])[0]
    grid = Grid.from_layout(layout)
    item_class = {}
    for name, r in items_by_name(tables).items():
        item_class.setdefault(r.get("Class"), name)

    rows = [m for m in tables["SimpleMutations"]
            if M.handler_for(m["Name"]).kind in (PASSIVE, ACTIVE)
            and (args.only is None or m["Name"] == args.only)]
    flagged = 0
    print(f"{len(rows)} implemented mutation rows\n")
    print(f"{'ID':>4} {'Name':24s} {'class':9s} unread / missing")
    for row in rows:
        cls = pick_class(row, item_class)
        if cls is None:
            print(f"{row['ID']:>4} {row['Name']:24s} {'-':9s} no item for {row.get('Class')}")
            continue
        try:
            exercise(tables, row, cls, item_class[cls], args.seconds, grid, layout, item_class)
        except Exception as exc:  # noqa: BLE001
            print(f"{row['ID']:>4} {row['Name']:24s} {cls:9s} ERROR {type(exc).__name__}: {exc}")
            flagged += 1
            continue
        own = {row[f"Param{i}Name"] for i in range(1, 10) if row.get(f"Param{i}Name")}
        read = READS.get(row["ID"], set())
        star = row["ID"] in ITERATED
        skip = set(EXPLAINED) | RUN_LAYER.get(row["Name"], set())
        unread = sorted(own - read - skip) if not star else []
        missing = sorted(k for k in read - own if isinstance(k, str) and k != "damageType")
        # A missing key matters beside an unread one: that is a misspelling.
        missing = missing if unread else []
        if unread or missing or args.all:
            flagged += bool(unread or missing)
            print(f"{row['ID']:>4} {row['Name']:24s} {cls:9s}{'*' if star else ' '}"
                  f"unread {unread or '-'} / missing {missing or '-'}")
    print(f"\n{flagged} rows to look at")


if __name__ == "__main__":
    main()
