"""Which parameters each unit skill actually reads: the Skills.json half of the audit.

The same check as `tools/audit_mutations.py`, for the skills the units carry
(`Units.json` Skill1..Skill8, resolved by `apply_unit_skills`). Every unit class
that carries a skill the sim implements fights a small squad in the Python oracle
for a stretch, with each skill's parameters in a dict that records every key
looked up, and the core's packer runs over the same agents. Per skill row:

    unread   a parameter the row sets that nothing looked at

A skill several units carry is checked once per unit and its reads pooled.

    python tools/audit_skills.py [--seconds 20] [--only CSClass]
"""
import argparse
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dataclasses

import sim.model_defaults as MD
from audit_mutations import EXPLAINED, ITERATED, READS, RecordingDict
from sim.assumptions import DEFAULT
from sim.battle import Battle, apply_class_skills, deploy
from sim.data import load_ruleset, parse_room_layouts, skills_by_id, units_by_class
from sim.nav import Grid
from sim.spec import build_player_squad, build_unit
from sim.unit_skills import ACTIVE, PASSIVE, REGISTRY

_original = MD.with_model_defaults
# (CSClass, parameters) -> every row with exactly those (45 and 124 are twins)
_TAGS: dict[tuple, list] = {}


def _key(cs, params):
    return (cs, tuple(sorted((str(k), str(v)) for k, v in params.items())))


def recording_defaults(name, params):
    """`apply_unit_skills` builds each row's dict and passes it here; the row is
    found again by its CSClass and parameters."""
    out = _original(name, params)
    tags = _TAGS.get(_key(name, params))
    return SharedRecordingDict(out, tuple(tags)) if tags else out


class SharedRecordingDict(RecordingDict):
    """A read counts for every row that shares the dict's parameters."""

    def _note(self, key):
        for tag in self.tag:
            READS.setdefault(tag, set()).add(key)

    def _walk(self):
        ITERATED.update(self.tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--only", default=None, help="one CSClass")
    args = ap.parse_args()

    tables = load_ruleset("Default", "default", strict=True)
    sb = skills_by_id(tables)
    implemented = {}
    for sid, row in sb.items():
        cs = row.get("CSClass")
        if cs in REGISTRY and REGISTRY[cs].kind in (PASSIVE, ACTIVE) \
                and (args.only is None or cs == args.only):
            p = {row[f"Param{i}Name"]: row.get(f"Param{i}Value")
                 for i in range(1, 11) if row.get(f"Param{i}Name")}
            _TAGS.setdefault(_key(cs, p), []).append(sid)
            implemented[sid] = (cs, set(p))
    MD.with_model_defaults = recording_defaults

    layout = parse_room_layouts(tables["RoomLayouts"])[0]
    grid = Grid.from_layout(layout)
    ubc = units_by_class(tables)
    carriers = {}
    for cls, levels in ubc.items():
        spec = build_unit(ubc, cls, 1)
        for sid in spec.skills:
            if sid in implemented:
                carriers.setdefault(sid, []).append(cls)
    print(f"{len(implemented)} implemented skill rows, {len(carriers)} carried by a unit\n")

    a = dataclasses.replace(DEFAULT, max_fight_seconds=args.seconds)
    fought = set()
    for sid, classes in sorted(carriers.items()):
        for cls in classes[:2]:
            if cls in fought:
                continue
            fought.add(cls)
            rng = random.Random(1)
            enemy = [build_unit(ubc, cls, 1)]
            squad = build_player_squad(tables, [("broadsword", 1), ("medbag", 1)])
            t0 = deploy(grid, layout, squad, 0, "p", rng)
            t1 = deploy(grid, layout, enemy, 1, "e1", rng)
            apply_class_skills(tables, t0)
            try:
                b = Battle(grid, t0 + t1, assumptions=a, seed=1, tables=tables)
                try:
                    from sim.fast import pack_all
                    pack_all(b.agents, tables)
                except Exception:
                    pass
                b.run()
            except Exception as exc:  # noqa: BLE001
                print(f"{cls}: ERROR {type(exc).__name__}: {exc}")

    flagged = 0
    print(f"{'ID':>4} {'CSClass':22s} {'carried by':24s} unread")
    for sid in sorted(implemented):
        cs, own = implemented[sid]
        if sid not in carriers:
            continue
        read = READS.get(sid, set())
        if sid in ITERATED:
            continue
        unread = sorted(own - read - set(EXPLAINED))
        if unread:
            flagged += 1
            print(f"{sid:>4} {cs:22s} {','.join(carriers[sid][:2]):24s} {unread}")
    never = sorted(sid for sid in implemented if sid not in carriers)
    print(f"\n{flagged} skill rows to look at; implemented but carried by no unit: {never}")


if __name__ == "__main__":
    main()
