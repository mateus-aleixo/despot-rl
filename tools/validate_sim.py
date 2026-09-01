"""Sanity checks on the reference battle sim.

These are internal-consistency checks, not fidelity checks: they prove the sim
does what this code says it does. Fidelity against the real game still needs
in-game fights to compare against.
"""
import dataclasses
import random
import statistics
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import collections

from sim.battle import Battle, apply_class_skills, apply_damage, deploy
from sim.data import items_by_name, load_ruleset, parse_room_layouts, units_by_class
from sim.nav import Grid, astar
from sim.spec import build_enemy_pack, build_player_squad, build_unit

TABLES = load_ruleset(strict=False)
LAYOUTS = parse_room_layouts(TABLES["RoomLayouts"])
LAYOUT = LAYOUTS[0]
GRID = Grid.from_layout(LAYOUT)

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def fight(loadout, pack_class, n_enemies, seed):
    squad = build_player_squad(TABLES, loadout)
    pack = next(p for p in TABLES["EnemyPacks"] if p["Class"] == pack_class)
    enemies = build_enemy_pack(TABLES, pack)[:n_enemies]
    rng = random.Random(seed)
    agents = deploy(GRID, LAYOUT, squad, 0, "p", rng) + deploy(GRID, LAYOUT, enemies, 1, "e1", rng)
    return Battle(GRID, agents, seed=seed).run()


print("== damage formula (verified against CS_Damage.Apply) ==")
check("flat armor subtraction", apply_damage(60, 5, 0) == 55.0)
check("floors at 1 damage", apply_damage(3, 5, 0) == 1.0)
check("never raises damage below 1", apply_damage(0.5, 5, 0) == 0.5)
check("resistance is multiplicative", apply_damage(60, 0, 0.25, magical=True) == 45.0)
check("zero armor is a no-op", apply_damage(60, 0, 0) == 60.0)

print("\n== stat composition (verified against the Swordsman prefab) ==")
ubc, items = units_by_class(TABLES), items_by_name(TABLES)
sw = build_unit(ubc, "Novice", 1, items["broadsword"])
check("speed 80 + 20 = 100 (UnitMovement.speed)", sw.speed == 100.0, f"got {sw.speed}")
check("health 70 + 160 = 230", sw.health == 230.0, f"got {sw.health}")
check("damage 20 + 60 = 80", sw.damage == 80.0, f"got {sw.damage}")
check("armor 1 + 5 = 6", sw.armor == 6.0, f"got {sw.armor}")
check("melee flagged", sw.melee)

gun = build_unit(ubc, "Novice", 1, items["gun"])
check("gun is ranged", gun.is_ranged)
check("gun reach is 10 tiles = 60", gun.range_world == 60.0, f"got {gun.range_world}")

print("\n== level scaling ==")
# One level, applied to both halves. The class row is re-read at the new level
# (`CS_Units.LevelUp`) and the item's bonus is recomputed there
# (`CS_Item.ApplyBonus`), so a level-3 human is a level-3 Novice carrying a
# level-3 broadsword. The per-level term compounds as a percentage rather than
# adding: `CS_Item.GetBonus` is `value * powf(1 + perLevel/100, level - 1)`.
l1 = build_unit(ubc, "Novice", 1, items["broadsword"])
l3 = build_unit(ubc, "Novice", 3, items["broadsword"])
_bs = items["broadsword"]
check("the class row is re-read at the new level",
      l3.damage - l1.damage
      == (ubc["Novice"][3]["Damage"] - ubc["Novice"][1]["Damage"])
      + _bs["Damage"] * ((1 + _bs["DamagePerLevel"] / 100) ** 2 - 1),
      f"got {l3.damage - l1.damage:.3f}")
check("DamagePerLevel compounds, it does not add",
      abs((l3.damage - ubc["Novice"][3]["Damage"]) - 60 * 1.1 ** 2) < 1e-9,
      f"item part {l3.damage - ubc['Novice'][3]['Damage']:.3f}, "
      f"additive would be {60 + 2 * 10}")
check("Speed has no per-level term", l3.speed == l1.speed, f"{l1.speed} -> {l3.speed}")

print("\n== pathfinding ==")
p = astar(GRID, (0, 0), (GRID.rows - 1, GRID.cols - 1))
check("finds a path across the room", len(p) > 0, f"{len(p)} cells")
check("path starts and ends right", p[0] == (0, 0) and p[-1] == (GRID.rows - 1, GRID.cols - 1))
check("path is 8-connected", all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
                                 for a, b in zip(p, p[1:])))

print("\n== determinism ==")
r1 = fight([("broadsword", 1)] * 6, "Mancrack", 6, seed=7)
r2 = fight([("broadsword", 1)] * 6, "Mancrack", 6, seed=7)
check("same seed gives same result",
      (r1.winner, r1.ticks, r1.total_damage) == (r2.winner, r2.ticks, r2.total_damage))
r3 = fight([("broadsword", 1)] * 6, "Mancrack", 6, seed=8)
check("different seed changes something", (r1.ticks, r1.total_damage) != (r3.ticks, r3.total_damage))

print("\n== fights terminate and scale sensibly ==")
res = [fight([("broadsword", 1)] * 6, "Mancrack", 6, s) for s in range(12)]
decisive = sum(1 for r in res if r.winner is not None)
check("all fights reach a decision", decisive == len(res), f"{decisive}/{len(res)}")
wins = sum(1 for r in res if r.winner == 0)
print(f"      6 broadswords vs 6 Mancrack: player wins {wins}/{len(res)}, "
      f"median {statistics.median(r.seconds for r in res):.1f}s")

big = [fight([("broadsword", 1)] * 6, "Mancrack", 16, s) for s in range(6)]
big_wins = sum(1 for r in big if r.winner == 0)
check("being outnumbered 16v6 hurts", big_wins < wins, f"player wins {big_wins}/{len(big)}")

print("\n== ranged behaviour ==")
rr = [fight([("gun", 1)] * 6, "Mancrack", 6, s) for s in range(6)]
rwins = sum(1 for r in rr if r.winner == 0)
print(f"      6 guns vs 6 Mancrack: player wins {rwins}/{len(rr)}, "
      f"median {statistics.median(r.seconds for r in rr):.1f}s")
check("ranged squad also resolves", all(r.winner is not None for r in rr))

print("\n== behaviour trees ==")
import collections
import json
import pathlib

from sim.bt import build_tree
from sim.bt_leaves import NOT_MODELLED, BTContext, UnknownLeaf, _PREFIX_NOT_MODELLED

BT_FILES = sorted(pathlib.Path("data/extracted/bt").glob("*.json"))
built, build_fail = 0, []
for f in BT_FILES:
    try:
        build_tree(json.loads(f.read_text(encoding="utf-8")))
        built += 1
    except Exception as e:
        build_fail.append(f"{f.stem}: {e}")
check("every shipped tree builds", built == len(BT_FILES), f"{built}/{len(BT_FILES)}")
if build_fail:
    for m in build_fail:
        print("        " + m)

# Every leaf must be either implemented or explicitly declared unmodelled.
leaves = collections.Counter()
for f in BT_FILES:
    for n in json.loads(f.read_text(encoding="utf-8"))["nodes"]:
        for key in ("_action", "_condition"):
            v = n.get(key)
            if isinstance(v, dict):
                leaves[(key, v.get("$type", "?"))] += 1
                for sub in v.get("conditions", []) or []:
                    if isinstance(sub, dict):
                        leaves[("_condition", sub.get("$type", "?"))] += 1

_rng = random.Random(0)
_squad = build_player_squad(TABLES, [("broadsword", 1)] * 2)
_pack = next(p for p in TABLES["EnemyPacks"] if p["Class"] == "Mancrack")
_ags = (deploy(GRID, LAYOUT, _squad, 0, "p", _rng)
        + deploy(GRID, LAYOUT, build_enemy_pack(TABLES, _pack)[:2], 1, "e1", _rng))
_b = Battle(GRID, _ags, seed=0)
_ags[0].target = _ags[2]
_ctx = BTContext(_ags[0], _b)
# Leaves act on the action bound by the enclosing ActionScope, so the probe has
# to bind one the way the tree would.
_ctx.action = _ags[0].actions[0]

unknown = []
for (kind, name), cnt in leaves.items():
    if name in NOT_MODELLED or name.startswith(_PREFIX_NOT_MODELLED):
        continue
    if name == "NodeCanvas.Framework.ConditionList":
        continue
    try:
        _ctx._condition(name, None) if kind == "_condition" else _ctx._action(name, None)
    except UnknownLeaf:
        unknown.append(name)
check("no undeclared leaf types", not unknown, f"{len(leaves)} types, unknown: {unknown}")

print("\n== attack cadence follows NC-DefaultAttack ==")
from sim.assumptions import DEFAULT as A

_sq = build_player_squad(TABLES, [("broadsword", 1)])
_en = build_enemy_pack(TABLES, _pack)[:1]
_rng2 = random.Random(5)
_a = deploy(GRID, LAYOUT, _sq, 0, "p", _rng2) + deploy(GRID, LAYOUT, _en, 1, "e1", _rng2)
# dataclasses.replace, not mutation: A is the shared DEFAULT instance and
# mutating it would leak into every later check in this run.
_bt = Battle(GRID, _a, assumptions=dataclasses.replace(A, max_fight_seconds=20.0), seed=5)
hits = []
_orig = _bt.hit
# team 0 only: both sides engage simultaneously, so pooling teams would show
# spurious zero-length gaps between two different units swinging on one tick.
_bt.hit = lambda team, tgt, raw, magical=False, attacker=None: (
    hits.append(_bt.tick_count) if team == 0 else None,
    _orig(team, tgt, raw, magical, attacker))
_bt.run()
gaps = [(b_ - a_) / A.tick_hz for a_, b_ in zip(hits, hits[1:])]
expected = max(_sq[0].attack_period, A.attack_anim_s + A.recovery_anim_s)
check("swing interval matches the cycle",
      bool(gaps) and all(abs(g - expected) < 0.06 for g in gaps),
      f"expected {expected:.2f}s, got {[round(g,2) for g in gaps]}")

print("\n== interruptor aborts a swing when the target dies ==")
# Tested at the engine level. Through a live battle the Interruptor cannot be
# reached: step() re-targets a dead target before the tree ticks, so
# IsStillApplicable always sees a live one, and with a single enemy
# Team2IsDying gates off the whole fight branch anyway.
from sim.bt import Status as _S
from sim.bt import build_tree as _bt_build

_tree = _bt_build(json.loads(pathlib.Path("data/extracted/bt/NC-DefaultAttack.json")
                             .read_text(encoding="utf-8")))


class _FakeCtx:
    """Drives the real tree with scripted leaf results."""

    def __init__(self):
        self.guards = {}
        self.applicable = True
        self.interrupts = 0
        self.damage = 0
        self.anim = 0

    def eval_condition(self, spec, node):
        name = (spec or {}).get("$type", "")
        val = {"BT.DefaultAttack.ReadyToAct": True,
               "BT.DefaultAttack.IsStillApplicable": self.applicable}.get(name, True)
        return (not val) if (spec or {}).get("_invert") else val

    def run_action(self, spec, node):
        name = (spec or {}).get("$type", "")
        if name == "WaitForAnimation":
            self.anim -= 1
            return _S.RUNNING if self.anim > 0 else _S.SUCCESS
        if name == "BT.DefaultAttack.StartExecution":
            self.anim = 3
            return _S.SUCCESS
        if name == "BT.DefaultAttack.AfterAttackAnimation":
            self.damage += 1
            self.anim = 3
            return _S.SUCCESS
        if name == "BT.DefaultAttack.StopIfRunning":
            return _S.FAILURE
        return _S.SUCCESS

    def switch_index(self, raw):
        return 0

    def on_interrupt(self, node):
        self.interrupts += 1


_ctx = _FakeCtx()
_tree.tick(_ctx)                       # StartExecution, animation running
_tree.tick(_ctx)
check("no damage while the attack animation runs", _ctx.damage == 0, f"damage={_ctx.damage}")
_ctx.applicable = False                # target stops being valid mid-swing
_tree.tick(_ctx)
check("interruptor fired", _ctx.interrupts == 1, f"interrupts={_ctx.interrupts}")
check("no damage landed after the interrupt", _ctx.damage == 0, f"damage={_ctx.damage}")

_ctx2 = _FakeCtx()                     # uninterrupted swing does land damage
for _ in range(12):
    _tree2 = _tree
    _tree2.tick(_ctx2)
check("an uninterrupted swing lands damage", _ctx2.damage >= 1, f"damage={_ctx2.damage}")

print("\n== item stat composition ==")
_lance = build_unit(ubc, "Novice", 1, items["lance"], 1)
_grenade = build_unit(ubc, "Novice", 1, items["throwing-grenade"], 1)
_opm = build_unit(ubc, "Novice", 1, items["opm-costume"], 1)
# Item AttackSpeed is a percentage: Novice 0.5 with lance +40% -> 0.7/s.
check("attack speed composes as a percentage",
      abs(_lance.attack_period - 1 / 0.7) < 1e-6, f"got {_lance.attack_period:.3f}s")
check("negative attack speed slows rather than zeroing",
      abs(_grenade.attack_period - 2.5) < 1e-6, f"got {_grenade.attack_period:.3f}s")
check("heavy weapons swing slower", _opm.attack_period > _lance.attack_period,
      f"opm {_opm.attack_period:.2f}s vs lance {_lance.attack_period:.2f}s")
check("dps stays in a sane range",
      all(1 < build_unit(ubc, "Novice", 1, i, 1).damage /
          build_unit(ubc, "Novice", 1, i, 1).attack_period < 500
          for i in TABLES["Items"] if i["Damage"]),
      "every item between 1 and 500 dps")

print("\n== reach vs body size ==")
# The in-range test is centre-to-centre, so a weapon whose reach is under the
# minimum separation of two bodies can never land a hit. This is what silently
# disabled the entire Dodger line when the agent radius was 6.
#
# octopus-claws is a real unresolved case, not an accepted one: its reach is
# exactly 6.0 against a minimum separation of exactly 6.0, and the verified
# test is a strict `>`, so it lands nothing. Either the radius is slightly
# smaller than Size/2 tiles, or bodies overlap a little in practice. It is
# pinned here so it stays visible and so any NEW unreachable item still fails.
KNOWN_UNREACHABLE = {"octopus-claws"}

_unreachable = set()
for i in TABLES["Items"]:
    if not i["Damage"]:
        continue
    u = build_player_squad(TABLES, [(i["Name"], 1)])[0]
    if u.range_world <= 2 * u.radius:
        _unreachable.add(i["Name"])
check("no new item is unable to reach a same-size target",
      _unreachable == KNOWN_UNREACHABLE,
      f"unreachable: {sorted(_unreachable)}, expected: {sorted(KNOWN_UNREACHABLE)}")

print("\n== class skills ==")
from sim.actions import (PASSIVE_SKILLS, build_actions, resolve_class_skills,
                         skill_for_class)

_classes = sorted({i["Class"] for i in TABLES["Items"]})
check("every player class is in Meta.Classes",
      all(c in TABLES["Meta"]["Classes"] for c in _classes), f"{len(_classes)} classes")

# level selection follows HumansRequired
_lv = {n: resolve_class_skills(TABLES, {"Warrior": n}).get("Warrior", {}).get("level")
       for n in range(1, 6)}
check("class skill level follows HumansRequired",
      _lv == {1: None, 2: 1, 3: 2, 4: 3, 5: 4}, f"{_lv}")

_rep = {}
for i in TABLES["Items"]:
    _rep.setdefault(i["Class"], i["Name"])

_no_skill = [c for c in _classes if not skill_for_class(TABLES, c)]
check("only Plant has no class skill", _no_skill == ["Plant"], f"{_no_skill}")

_fired, _missing = [], []
for cls in _classes:
    name = skill_for_class(TABLES, cls)
    if not name:
        continue
    squad = build_player_squad(TABLES, [(_rep[cls], 1)] * 5)
    rng = random.Random(3)
    team0 = deploy(GRID, LAYOUT, squad, 0, "p", rng)
    resolved = apply_class_skills(TABLES, team0)
    agents = team0 + deploy(GRID, LAYOUT, build_enemy_pack(TABLES, _pack)[:6], 1, "e1", rng)
    b = Battle(GRID, agents, seed=3, tables=TABLES)
    r = b.run()
    if name in PASSIVE_SKILLS:
        got = (team0[0].crit_chance > 0 or team0[0].attack_speed_pct > 0
               or team0[0].dodge_cooldown > 0)
    else:
        got = sum(v for k, v in r.casts.items() if k != "attack") > 0
    (_fired if got else _missing).append(f"{cls}:{name}")
check("every class skill takes effect", not _missing,
      f"{len(_fired)} fired, missing: {_missing}")

# summons actually appear
_sq = build_player_squad(TABLES, [(_rep["Cultist"], 1)] * 5)
_rng4 = random.Random(2)
_t0 = deploy(GRID, LAYOUT, _sq, 0, "p", _rng4)
apply_class_skills(TABLES, _t0)
_b4 = Battle(GRID, _t0 + deploy(GRID, LAYOUT, build_enemy_pack(TABLES, _pack)[:6], 1, "e1", _rng4),
             seed=2, tables=TABLES)
_b4.run()
check("Cultist summons a real unit",
      any(a.summoned for a in _b4.agents),
      f"{sum(1 for a in _b4.agents if a.summoned)} summoned")

print("\n== per-unit skills ==")
from sim.unit_skills import ACTIVE, REGISTRY, coverage, handler_for

_cov = coverage(TABLES)
check("every CSClass in use is registered", not _cov["unregistered"],
      f"unregistered: {_cov['unregistered']}")
_tot = sum(_cov["by_kind"].values())
_handled = _cov["by_kind"].get("noop", 0) + _cov["by_kind"].get(
    "passive", 0) + _cov["by_kind"].get("active", 0)
print(f"      {_tot} skill uses: " + ", ".join(
    f"{k} {v}" for k, v in sorted(_cov['by_kind'].items(), key=lambda x: -x[1])))
check("most skill uses are handled", _handled / _tot > 0.75,
      f"{_handled}/{_tot} = {_handled/_tot*100:.0f}%")

# Resistance is a percentage in the table and a fraction in the formula.
_mr = build_unit(ubc, "MagicResistant", 1)
check("resistance is scaled to a fraction", 0.0 < _mr.resistance <= 1.0,
      f"got {_mr.resistance}")
check("magical damage is reduced, not inverted",
      0 < apply_damage(100, 0, _mr.resistance, magical=True) < 100,
      f"got {apply_damage(100, 0, _mr.resistance, magical=True):.1f}")

# Melee must be able to touch a larger body: reach is built from BOTH radii.
_small = build_player_squad(TABLES, [("broadsword", 1)])[0]
_bigcls = next(c for c in units_by_class(TABLES)
               if int((TABLES["Meta"]["Classes"].get(c) or {}).get("Size") or 1) >= 2)
_big = build_unit(ubc, _bigcls, 1,
                  size=int(TABLES["Meta"]["Classes"][_bigcls]["Size"]))
_rngX = random.Random(1)
_t0 = deploy(GRID, LAYOUT, [_small] * 3, 0, "p", _rngX)
apply_class_skills(TABLES, _t0)
_bX = Battle(GRID, _t0 + deploy(GRID, LAYOUT, [_big], 1, "e1", _rngX),
             assumptions=dataclasses.replace(A, max_fight_seconds=30.0),
             seed=1, tables=TABLES)
_rX = _bX.run()
check(f"melee can damage a Size-{_big.size} target ({_bigcls})",
      _rX.total_damage[0] > 0, f"dealt {_rX.total_damage[0]:.0f}")

# SplashAround: Mancrack's skill 18 should spread damage across the squad.
_rngS = random.Random(4)
_t0s = deploy(GRID, LAYOUT, build_player_squad(TABLES, [("broadsword", 1)] * 6), 0, "p", _rngS)
apply_class_skills(TABLES, _t0s)
_bs = Battle(GRID, _t0s + deploy(GRID, LAYOUT, build_enemy_pack(TABLES, _pack)[:4], 1, "e1", _rngS),
             seed=4, tables=TABLES)
_bs.run()
_hurt = sum(1 for a in _t0s if a.hp < a.spec.health)
check("SplashAround damages more than one target", _hurt > 1,
      f"{_hurt}/6 players took damage")

print("\n== control effects ==")


def _duel(seed=3, secs=12.0):
    rng = random.Random(seed)
    t0 = deploy(GRID, LAYOUT, build_player_squad(TABLES, [("broadsword", 1)]), 0, "p", rng)
    e = deploy(GRID, LAYOUT, build_enemy_pack(TABLES, _pack)[:1], 1, "e1", rng)
    b = Battle(GRID, t0 + e, assumptions=dataclasses.replace(A, max_fight_seconds=secs),
               seed=seed, tables=TABLES)
    return b, t0[0], e[0]


# a stunned unit deals no damage
_b1, _p1, _e1 = _duel()
for _ in range(int(12 * A.tick_hz)):
    _p1.apply_status("stun", 1.0)
    _b1.step()
check("a permanently stunned unit deals no damage", _b1.damage_done[0] == 0.0,
      f"dealt {_b1.damage_done[0]:.0f}")

# an unstunned one does
_b2, _p2, _e2 = _duel()
_b2.run()
check("the same duel without stun does damage", _b2.damage_done[0] > 0,
      f"dealt {_b2.damage_done[0]:.0f}")

# silence blocks skills but not the plain attack
_rngZ = random.Random(6)
_sq = build_player_squad(TABLES, [("ring-green", 1)] * 5)   # Mage -> Mage class skill
_t0z = deploy(GRID, LAYOUT, _sq, 0, "p", _rngZ)
apply_class_skills(TABLES, _t0z)
_bz = Battle(GRID, _t0z + deploy(GRID, LAYOUT, build_enemy_pack(TABLES, _pack)[:4], 1, "e1", _rngZ),
             assumptions=dataclasses.replace(A, max_fight_seconds=25.0), seed=6, tables=TABLES)
for _ in range(int(25 * A.tick_hz)):
    if not any(x.alive for x in _t0z):
        break
    for x in _t0z:
        x.apply_status("silence", 1.0)
    _bz.step()
_silenced_casts = sum(v for k, v in _bz.casts.items() if k != "attack")
check("silence blocks skills", _silenced_casts == 0, f"{_silenced_casts} casts while silenced")
check("silence leaves the plain attack alone", _bz.casts.get("attack", 0) > 0,
      f"{_bz.casts.get('attack', 0)} attacks")

# knockback displaces
_b3, _p3, _e3 = _duel(seed=8)
_b3.step()
_x0, _y0 = _e3.x, _e3.y
_b3.push(_e3, _p3.x, _p3.y, speed=200.0, seconds=0.3)
for _ in range(20):
    _b3.step()
import math as _math
check("knockback moves the target away",
      _math.hypot(_e3.x - _x0, _e3.y - _y0) > 1.0,
      f"moved {_math.hypot(_e3.x - _x0, _e3.y - _y0):.1f}")

print("\n== merge strategies ==")
from sim.data import MergeNotImplemented, _merge_grid, load_ruleset

_strict_ok, _strict_fail = [], []
for _chip in ("default", "easy", "hard"):
    for _wf in (False, True):
        try:
            load_ruleset("Default", _chip, without_food=_wf, strict=True)
            _strict_ok.append((_chip, _wf))
        except Exception as e:
            _strict_fail.append(f"{_chip}/wf={_wf}: {type(e).__name__}: {e}")
check("every chip loads strictly", not _strict_fail, f"{len(_strict_ok)} ok, {_strict_fail}")

_STRICT = load_ruleset("Default", "default", strict=True)

# __remove really removes: Default/WithoutFood ships exactly 41 __remove rows.
_wf_tables = load_ruleset("Default", "default", without_food=True, strict=True)
_removed = len(_STRICT["MutationsByLevel"]) - len(_wf_tables["MutationsByLevel"])
check("__remove drops rows from MutationsByLevel", _removed == 41,
      f"1296 -> {len(_wf_tables['MutationsByLevel'])}, dropped {_removed}")

# mergeGrid keeps base keys the override does not mention, and merges
# CombinedMutations by ID rather than replacing the list.
_base = {"Width": 5, "Mutations": {"a": 1},
         "CombinedMutations": [{"ID": 1, "x": 1}, {"ID": 2, "x": 2}]}
_over = {"CombinedMutations": [{"ID": 2, "x": 99}, {"ID": 3, "x": 3}]}
_m = _merge_grid(_base, _over)
check("mergeGrid keeps untouched keys", _m["Width"] == 5 and _m["Mutations"] == {"a": 1})
_ids = {r["ID"]: r["x"] for r in _m["CombinedMutations"]}
check("mergeGrid merges CombinedMutations by ID", _ids == {1: 1, 2: 99, 3: 3}, f"{_ids}")

print("\n== mutations ==")
from sim.mutations import (apply_to_agents, apply_to_specs, coverage as mut_coverage,
                           is_timed, mutation_params, offered_at_level)

_mc = mut_coverage(_STRICT)
check("every mutation Name is registered", not _mc["unregistered"],
      f"unregistered: {_mc['unregistered']}")
_mt = sum(_mc["by_kind"].values())
_mh = _mt - _mc["by_kind"].get("unimplemented", 0)
print(f"      {_mt} definitions: " + ", ".join(
    f"{k} {v}" for k, v in sorted(_mc['by_kind'].items(), key=lambda x: -x[1])))
print(f"      handled {_mh}/{_mt} = {_mh/_mt*100:.0f}%  (mutations are mostly bespoke)")

check("mutations are offered per level", len(offered_at_level(_STRICT, 1)) > 0
      and len(offered_at_level(_STRICT, 12)) > len(offered_at_level(_STRICT, 1)),
      f"L1 {len(offered_at_level(_STRICT, 1))}, L12 {len(offered_at_level(_STRICT, 12))}")

# a permanent StatBonus changes the spec; a timed one does not
_dmg = next(m for m in _STRICT["SimpleMutations"]
            if m["Name"] == "StatBonus" and m.get("Class") == "All"
            and mutation_params(m).get("stat") == "Damage")
_specs = build_player_squad(_STRICT, [("broadsword", 1)] * 3)
_after = apply_to_specs(_specs, [_dmg], random.Random(0))
check("a permanent StatBonus changes the stat", _after[0].damage > _specs[0].damage,
      f"{_specs[0].damage} -> {_after[0].damage}")

_timed = [m for m in _STRICT["SimpleMutations"]
          if m["Name"] in ("StatBonus", "OraStatBonus") and is_timed(mutation_params(m))]
_after2 = apply_to_specs(_specs, _timed, random.Random(0))
check("a timed StatBonus does not change the spec",
      all(a.damage == b.damage and a.attack_speed == b.attack_speed
          for a, b in zip(_specs, _after2)),
      f"{len(_timed)} timed definitions")

# and it becomes a castable action instead
_rngM = random.Random(2)
_monks = build_player_squad(_STRICT, [("gloves", 1)] * 3)      # Monk class
_tm = deploy(GRID, LAYOUT, _monks, 0, "p", _rngM)
apply_class_skills(_STRICT, _tm)
_before_actions = len(_tm[0].actions)
apply_to_agents(_tm, _timed, random.Random(2))
check("a timed StatBonus becomes an action",
      len(_tm[0].actions) > _before_actions,
      f"{_before_actions} -> {len(_tm[0].actions)} actions")


print("\n== agent-level mutation passives ==")
from sim.data import items_by_name as _ibn
from sim.mutations import ON_ATTACK, ON_CAST, ON_DAMAGED, ON_DEATH, STANDING

_MBY = {m["ID"]: m for m in _STRICT["SimpleMutations"]}
_ITEM_CLASS = {n: r.get("Class") for n, r in _ibn(_STRICT).items()}


def _item_for(cls):
    return next(n for n, c in _ITEM_CLASS.items() if c == cls)


def _mut_fight(mut_ids, item="broadsword", n=4, seed=1, enemy="Mancrack", foes=6):
    """A squad carrying some mutations, against a pack, ready to step."""
    specs = build_player_squad(_STRICT, [(item, 1)] * n)
    muts = [_MBY[i] for i in mut_ids]
    if muts:
        specs = apply_to_specs(specs, muts, random.Random(seed))
    pack = next(p for p in _STRICT["EnemyPacks"] if p["Class"] == enemy)
    enemies = build_enemy_pack(_STRICT, pack)[:foes]
    rng = random.Random(seed)
    t0 = deploy(GRID, LAYOUT, specs, 0, "p", rng)
    t1 = deploy(GRID, LAYOUT, enemies, 1, "e1", rng)
    apply_class_skills(_STRICT, t0)
    if muts:
        apply_to_agents(t0, muts, random.Random(seed))
    return t0, t1, Battle(GRID, t0 + t1, seed=seed, tables=_STRICT)


def _until(battle, pred, ticks=4000):
    """Step until `pred` holds, and say whether it ever did."""
    for _ in range(ticks):
        battle.step()
        if pred():
            return True
    return False


def _status_ticks(mut_ids, status, item="broadsword", seed=0, ticks=2500):
    t0, t1, b = _mut_fight(mut_ids, item=item, seed=seed)
    seen = 0
    for _ in range(ticks):
        if not any(a.alive for a in t0) or not any(a.alive for a in t1):
            break
        b.step()
        seen += sum(1 for a in b.agents if a.has(status))
    return seen


# Every hook the game hangs a C_PassiveSkillMutation off has to actually fire,
# or the mutation shop is scenery again.

# On-attack: C_PassiveSkill.OnDamageCreated
# Both roll: PassiveStun 15% for 1.5 s, FearsomeAttack 10% for 3 s. One fight
# can easily see neither, so this is summed over seeds.
_stun = sum(_status_ticks([224], "stun", seed=s) for s in range(8))
_panic = sum(_status_ticks([14], "panic", seed=s) for s in range(8))
_none = sum(_status_ticks([], "stun", seed=s) + _status_ticks([], "panic", seed=s)
            for s in range(8))
check("on-attack passives apply their status", _stun > 0 and _panic > 0 and _none == 0,
      f"stun {_stun}, panic {_panic}, baseline {_none}")

# Panic is enforced in the loop rather than by a leaf -- NC-Fear is a tree of
# its own and is not inside NC-BaseUnit -- so it is worth pinning that it does
# anything at all.
_t0, _t1, _b = _mut_fight([])
for _ in range(400):
    _b.step()
    for _a in _t1:
        _a.apply_status("panic", 99.0)
check("a panicked unit stops fighting", _b.damage_done[1] == 0.0,
      f"team1 dealt {_b.damage_done[1]:.0f}")

# BuffAttack's two-stat debuff lands on the victim (castTarget: Target)
_t0, _t1, _b = _mut_fight([226])
check("BuffAttack debuffs the target it hits",
      _until(_b, lambda: any(a.buffs for a in _t1)),
      f"{[s for a in _t1 if a.buffs for s in a.buffs[0].stats]}")

# and on the attacker when the row says castTarget: Source
_t0, _t1, _b = _mut_fight([318])
check("BuffAttack buffs the attacker when castTarget is Source",
      _until(_b, lambda: any(a.buffs for a in _t0)))

# On-damaged: C_DamageReactionSkill, which reacts onto the attacker
_craggy = _status_ticks([317], "stun")
check("Craggy stuns whoever hit it", _craggy > 0, f"{_craggy} stunned ticks")

_t0, _t1, _b = _mut_fight([240], item=_item_for("Shooter"))
check("Untouchable slows whoever hit it",
      _until(_b, lambda: any(a.buffs for a in _t1)))

# On-cast: C_OnSkillCastedSkill
_t0, _t1, _b = _mut_fight([351], item=_item_for("Mage"))
check("BuffOnCasted buffs the caster on every cast",
      _until(_b, lambda: any(a.buffs for a in _t0)))

# On-death: C_BaseOnDeathSkill
_t0, _t1, _b = _mut_fight([303], seed=2)       # StickyBlood: enemies in radius
check("StickyBlood slows enemies around the corpse",
      _until(_b, lambda: any(a.buffs for a in _t1)))
_t0, _t1, _b = _mut_fight([353], seed=2)       # BuffOnDeath: the whole team
check("BuffOnDeath buffs the surviving team",
      _until(_b, lambda: any(a.buffs for a in _t0)))

# ResurrectionChance brings a unit back rather than letting it die
_res = 0
for _s in range(6):
    _t0, _t1, _b = _mut_fight([245], seed=_s)
    _b.run()
    _res += sum(1 for a in _t0 if a.resurrected and a.alive)
check("ResurrectionChance revives some of the fallen", _res > 0, f"{_res} revived")

# Compensation is recomputed every tick from how hurt the squad is
_t0, _t1, _b = _mut_fight([374], item=_item_for("Monk"))
check("Compensation scales a stat with the allies below its threshold",
      _until(_b, lambda: any(a.standing_buff is not None for a in _t0)))

# ModifyDamage reaches only the damage type it names
_dm = [_mut_fight([355], item=_item_for("Mage"), seed=s)[2].run().total_damage[0]
       for s in range(6)]
_d0 = [_mut_fight([], item=_item_for("Mage"), seed=s)[2].run().total_damage[0]
       for s in range(6)]
check("ModifyDamage raises the damage type it names",
      sum(_dm) > sum(_d0), f"{sum(_d0)/6:.0f} -> {sum(_dm)/6:.0f}")

# ClassDiversity: one unit, bonus x count(otherClass)
_ubc = units_by_class(_STRICT)
_mixed = (build_player_squad(_STRICT, [("broadsword", 1)] * 2)
          + [build_unit(_ubc, "Novice", 1, name="Novice") for _ in range(3)])
_cd = apply_to_specs(_mixed, [_MBY[365]], random.Random(0))   # Warrior <- Novice, +300
_moved = [a.damage - b.damage for a, b in zip(_cd, _mixed)]
check("ClassDiversity pays bonus x count(otherClass), to one unit",
      sorted(_moved)[-1] == 900.0 and sum(1 for d in _moved if d) == 1,
      f"{_moved}")
check("ClassDiversity pays nothing without the other class",
      not any(a.damage != b.damage for a, b in
              zip(apply_to_specs(_mixed[:2], [_MBY[365]], random.Random(0)), _mixed[:2])))

# CSLinkStatBonus rides on the Monk's class-skill cast
_mix = build_player_squad(_STRICT, [(_item_for("Monk"), 1)] * 3
                          + [(_item_for("Shooter"), 1)] * 2)
_pack = next(p for p in _STRICT["EnemyPacks"] if p["Class"] == "Mancrack")
_rngC = random.Random(0)
_t0 = deploy(GRID, LAYOUT, _mix, 0, "p", _rngC)
_t1 = deploy(GRID, LAYOUT, build_enemy_pack(_STRICT, _pack)[:6], 1, "e1", _rngC)
apply_class_skills(_STRICT, _t0)
apply_to_agents(_t0, [_MBY[379]], random.Random(0))     # Monk -> Shooter, +30% Damage
_b = Battle(GRID, _t0 + _t1, seed=0, tables=_STRICT)
check("CSLinkStatBonus buffs targetClass on the class-skill cast",
      _until(_b, lambda: any(a.spec.cls == "Shooter" and a.buffs for a in _t0)))

# Splash must not trigger any of it: DamageType.Secondary is gated out
from sim.battle import DT_PHYSICAL, DT_SECONDARY
from sim.mutations import _gate as _pgate
check("Secondary damage triggers no passive",
      not _pgate(_b, {"chance": 100}, DT_PHYSICAL | DT_SECONDARY)
      and _pgate(_b, {"chance": 100}, DT_PHYSICAL))
check("a damageType gate is a subset test",
      _pgate(_b, {"damageType": "Physical"}, DT_PHYSICAL)
      and not _pgate(_b, {"damageType": "Magical"}, DT_PHYSICAL))

# The hooks are ported, so a fight carrying them stays in the fast envelope --
# and the Rust core has to agree with the oracle about what they did. At 1v1
# neither engine's float width has had time to separate them, so the damage
# error is the proof that every roll landed on the same swing.
from sim.fast import UnsupportedFight as _Unsupported
from sim.fast import available as _core_available
from sim.fast import fast_battle as _fast_battle
from sim.fast import choice_probe as _choice_probe
from sim.fast import pack_all as _pack_all
from sim.fast import pack_passives as _pack_passives
from sim.fast import supported as _fast_supported

_t0, _t1, _b = _mut_fight([226])
check("a fight carrying an agent-level passive stays in the fast envelope",
      _fast_supported(_t0 + _t1, _STRICT)[0],
      _fast_supported(_t0 + _t1, _STRICT)[1])
check("every wired passive has a core kind", len(_pack_passives(_t0)) == len(_t0),
      f"{len(_pack_passives(_t0))} rows for {len(_t0)} agents")


def _core_vs_oracle(mut_ids, item="broadsword", n=1, foes=1, seeds=10):
    """Run the same fight in both engines. Returns (winners agreeing, damage error)."""
    agree, err, runs = 0, [], 0
    for seed in range(seeds):
        # two independent builds of the same fight: `_mut_fight` returns a
        # Battle whose __init__ has already attached the per-unit skills, so
        # re-wrapping its agents would attach them twice.
        oracle = _mut_fight(mut_ids, item=item, n=n, seed=seed, foes=foes)[2]
        core = _mut_fight(mut_ids, item=item, n=n, seed=seed, foes=foes)[2]
        res = oracle.run()
        try:
            fr = _fast_battle(GRID, core.agents, seed=seed, tables=_STRICT)
        except _Unsupported as exc:
            return -1, str(exc)
        runs += 1
        agree += int(res.winner == fr["winner"])
        if res.total_damage[0]:
            err.append(abs(fr["damage"][0] - res.total_damage[0]) / res.total_damage[0])
    return agree == runs, (sum(err) / len(err) * 100.0 if err else 0.0)


if _core_available():
    _EXACT = [("BuffAttack", [226]), ("PassiveStun", [224]),
              ("FearsomeAttack", [14]), ("Craggy", [317]), ("ManaBreak", [19]),
              ("StickyBlood", [303]), ("BuffOnDeath", [353]),
              ("ResurrectionChance", [245]), ("ClassDiversity", [365]),
              ("Vampirism", [23]), ("Evasion", [29]),
              ("ExplodeProjectile", [200])]
    for _name, _ids in _EXACT:
        _ok, _err = _core_vs_oracle(_ids)
        check(f"the core reproduces {_name} exactly at 1v1",
              _ok is True and _err < 0.01, f"damage error {_err:.2f}%")
    # These three need a caster, so they run on a Mage or a Monk instead.
    for _name, _ids, _item in (("BuffOnCasted", [351], _item_for("Mage")),
                               ("MultiCast", [301], _item_for("Mage")),
                               ("ModifyDamage", [355], _item_for("Mage")),
                               ("CSLinkStatBonus", [379], _item_for("Monk")),
                               ("Compensation", [374], _item_for("Monk"))):
        _ok, _err = _core_vs_oracle(_ids, item=_item)
        check(f"the core reproduces {_name} exactly at 1v1",
              _ok is True and _err < 0.01, f"damage error {_err:.2f}%")
else:
    print("      (rust core not built; skipping the differential checks)")


print("\n== vampirism and evasion ==")
from sim.battle import DT_CANT_BE_EVADED as _DT_NOEVADE
from sim.battle import DT_MAGICAL as _DT_MAG
from sim.battle import DT_PHYSICAL as _DT_PHY
from sim.battle import DT_SECONDARY as _DT_SEC
from sim.battle import JUMP_BACK_SPEED as _JB_SPEED
from sim.unit_skills import evasion_entry as _evasion_entry
from sim.unit_skills import vampirism_entry as _vampirism_entry

# Every shipped row names the number `value`, never `percent`. Reading it as
# `percent` is why vampirism healed exactly nothing, in the oracle and in the
# core, from the day it was "ported" until this check existed.
check("a Vampirism row reads its value, not a percent",
      _vampirism_entry({"value": 5}) == (5.0, True),
      f"{_vampirism_entry({'value': 5})}")
check("FixedVampirism is a flat heal",
      _vampirism_entry({"value": 15, "percentage": "false"}) == (15.0, False))
check("an Evasion row carries chance, threshold and jumpBack",
      _evasion_entry({"chance": 35, "healthThreshold": 30}) == (35.0, 30.0, False)
      and _evasion_entry({"chance": 40, "jumpBack": "true"})[2] is True,
      f"{_evasion_entry({'chance': 35, 'healthThreshold': 30})}")
check("an absent healthThreshold leaves the skill always on",
      _evasion_entry({"chance": 20})[1] >= 100.0)

_t0, _t1, _b = _mut_fight([23])          # Vampirism, a 5% share of the damage
check("a Vampirism mutation reaches the agents",
      all(a.vampirisms == [(5.0, True)] for a in _t0), f"{_t0[0].vampirisms}")
_b.run()
check("vampirism heals the squad that carries it", _b.healing_done[0] > 0.0,
      f"{_b.healing_done[0]:.1f} healed")
check("vampirism never heals past the ceiling",
      all(a.hp <= a.max_hp + 1e-3 for a in _t0))
_t0n, _t1n, _bn = _mut_fight([])
_bn.run()
check("the same fight without it heals nothing", _bn.healing_done[0] == 0.0)

# `Battle._evaded` is the whole gate in one call, so it can be asked directly
# rather than inferred from a win rate.
_t0, _t1, _b = _mut_fight([29])
_ev, _foe = _t0[0], _t1[0]
_ev.evasions = [(100.0, 100.0, False)]
check("a certain evasion evades physical damage", _b._evaded(_ev, _foe, _DT_PHY))
check("splash cannot be evaded",
      not _b._evaded(_ev, _foe, _DT_PHY | _DT_SEC)
      and not _b._evaded(_ev, _foe, _DT_PHY | _DT_NOEVADE))
check("magical damage cannot be evaded", not _b._evaded(_ev, _foe, _DT_MAG))

_ev.evasions = [(100.0, 30.0, False)]
check("a threshold row is off at full health", not _b._evaded(_ev, _foe, _DT_PHY))
_ev.hp = 0.2 * _ev.max_hp
check("and on once the unit is hurt", _b._evaded(_ev, _foe, _DT_PHY))

_ev.hp, _ev.knockback = _ev.max_hp, None
_ev.evasions = [(100.0, 100.0, True)]
_b._evaded(_ev, _foe, _DT_PHY)
_kb = _ev.knockback
check("jumpBack pushes the dodger away from the attacker",
      _kb is not None
      and abs((_kb[0] ** 2 + _kb[1] ** 2) ** 0.5 - _JB_SPEED) < 1e-3
      and (_kb[0] * (_ev.x - _foe.x) + _kb[1] * (_ev.y - _foe.y)) > 0.0,
      f"{_kb}")


print("\n== explode and knockback ==")
from sim.unit_skills import explode_entry as _explode_entry
from sim.unit_skills import knockback_entry as _knockback_entry

# Both rows name their numbers the way their model class does, and both readers
# used to look up names no row has, so every row silently took the default.
check("an ExplodeProjectile row reads damage, radius and chance",
      _explode_entry({"damage": 15, "radius": 16}) == (15.0, 16.0, 100.0, 0)
      and _explode_entry({"damage": 80, "radius": 20, "Chance": 30})[2] == 30.0,
      f"{_explode_entry({'damage': 15, 'radius': 16})}")
check("a knockback row reads speed and acceleration, either spelling",
      _knockback_entry({"chance": 20, "speed": 240, "acceleration": -300})
      == (20.0, 0.0, 240.0, -300.0, 0)
      and _knockback_entry({"knockbackSpeed": 180, "knockbackAcceleration": -350})
      == (100.0, 0.0, 180.0, -350.0, 0),
      f"{_knockback_entry({'chance': 20, 'speed': 240, 'acceleration': -300})}")

# ID 200 is ExplodeProjectile at 100 damage in a 16 radius, on the melee classes.
_t0, _t1, _b = _mut_fight([200])
check("an ExplodeProjectile mutation reaches the agents",
      all(a.explodes == [(100.0, 16.0, 100.0, 0)] for a in _t0), f"{_t0[0].explodes}")
_r = _b.run()
_t0n, _t1n, _bn = _mut_fight([])
_rn = _t0n and _bn.run()
check("the explosion adds damage the same fight does not have",
      _r.total_damage[0] > _bn.damage_done[0],
      f"{_r.total_damage[0]:.0f} against {_bn.damage_done[0]:.0f}")

# The explosion is Magical, so it neither chains into itself nor is evadable.
_t0, _t1, _b = _mut_fight([200])
_ex, _foe = _t0[0], _t1[0]
_foe.evasions = [(100.0, 100.0, False)]
_hp = _foe.hp
_b.hit(0, _foe, 10.0, magical=True, attacker=_ex)
check("magical damage still lands on a certain evader", _foe.hp < _hp,
      f"{_hp} -> {_foe.hp}")

# The knockback gate: physical, not splash, and the chance is honoured.
_t0, _t1, _b = _mut_fight([])
_att, _vic = _t0[0], _t1[0]
_att.knockbacks = [(100.0, 0.0, 200.0, -300.0, 0)]
_vic.knockback = None
_b.hit(0, _vic, 1.0, attacker=_att)
check("a knockback fires on a physical hit", _vic.knockback is not None)
_vic.knockback = None
_b.hit(0, _vic, 1.0, attacker=_att, dtype=_DT_PHY | _DT_SEC)
check("splash does not knock anybody back", _vic.knockback is None)
_vic.knockback = None
_b.hit(0, _vic, 1.0, magical=True, attacker=_att)
check("magical damage does not knock anybody back", _vic.knockback is None)

_att.knockbacks = [(0.0, 0.0, 200.0, -300.0, 0)]
_vic.knockback = None
_b.hit(0, _vic, 1.0, attacker=_att)
check("a zero chance never knocks back", _vic.knockback is None)

# Radius sends the victim's whole team, not just the victim.
_t0, _t1, _b = _mut_fight([], foes=6)
_att = _t0[0]
_att.knockbacks = [(100.0, 400.0, 200.0, -300.0, 0)]
for _o in _t1:
    _o.knockback = None
_b.hit(0, _t1[0], 1.0, attacker=_att)
check("a knockback with a radius moves the victim's whole team",
      sum(1 for o in _t1 if o.knockback is not None) == len(_t1),
      f"{sum(1 for o in _t1 if o.knockback is not None)}/{len(_t1)}")


print("\n== the per-unit on-death skills, in both engines ==")
from sim.battle import place_at as _place_at

_DEATH_UNITS = ["FireHead", "Firestarter", "Fish", "Ostrich", "PoringMedium",
                "OstrichRider2"]
_CELLS = LAYOUT.zone("p") or [(0, 0)]


def _touching(seed, enemy, players=3, enemy_hp=None, full_mana=False):
    """Both sides placed already in contact, so nothing has to walk.

    Walking is what separates the two engines: an approach is hundreds of ORCA
    steps and f32 and f64 disagree about them long before they disagree about
    anything else. Removing it is what makes a death effect measurable on its
    own.
    """
    specs = build_player_squad(_STRICT, [("broadsword", 1)] * players)
    size = int((_STRICT["Meta"]["Classes"].get(enemy) or {}).get("Size") or 1)
    foe = [build_unit(units_by_class(_STRICT), enemy, 1, name=enemy, size=size)]
    t0 = _place_at(GRID, specs, _CELLS[:players], team=0)
    t1 = _place_at(GRID, foe, _CELLS[:1], team=1)
    if enemy_hp is not None:
        t1[0].hp = enemy_hp
    if full_mana:
        # Mana arrives through the damage pipeline, so a caster in a two-body
        # fight often never banks its cost -- the Silencer needs 150. Handing it
        # the mana is what makes the cast the thing under test.
        for a in t1:
            a.mana = a.spec.mana
    return Battle(GRID, t0 + t1, seed=seed, tables=_STRICT)


# every on-death carrier has to reach the core at all
for _u in _DEATH_UNITS:
    _b = _touching(0, _u)
    _ok, _why = _fast_supported(_b.agents, _STRICT)
    check(f"{_u}'s on-death skill is inside the fast envelope", _ok, _why)

if _core_available():
    def _death_agreement(enemy, players=3, seeds=24, enemy_hp=1.0,
                         full_mana=False):
        """Compare the two engines on seeds where neither has drifted.

        A tick-count match means no swing has moved, so any damage difference
        after it is the death effect itself and nothing else.
        """
        exact = matched = 0
        for seed in range(seeds):
            py = _touching(seed, enemy, players, enemy_hp, full_mana).run()
            core = _touching(seed, enemy, players, enemy_hp, full_mana)
            rs = _fast_battle(GRID, core.agents, seed=seed, tables=_STRICT)
            if py.ticks != rs["ticks"]:
                continue
            matched += 1
            exact += int(abs(py.total_damage[0] - rs["damage"][0]) < 0.5
                         and abs(py.total_damage[1] - rs["damage"][1]) < 0.5)
        return exact, matched

    # The three damage-on-death skills. FireheadDeath has no radius, so it picks
    # its victim with `random.choice` -- getrandbits and a rejection loop, not a
    # float -- which is the one place the core needs CPython's `_randbelow`.
    for _u in ("FireHead", "Firestarter", "Fish"):
        _e, _m = _death_agreement(_u)
        check(f"the core reproduces {_u}'s death damage exactly",
              _m >= 20 and _e == _m, f"{_e}/{_m} tick-matched seeds")

    # A transform is a summon, and a summon's spawn offset is two `uniform`
    # draws, so the two engines place it a float-width apart -- enough to shift
    # an approach by a tick. PoringMedium spawns its three in contact, so it is
    # the one that stays exact.
    _e, _m = _death_agreement("PoringMedium", players=1, seeds=40)
    check("the core reproduces TransformAfterDeath exactly when nothing walks",
          _m >= 30 and _e == _m, f"{_e}/{_m} tick-matched seeds")

    # and the summon is a real unit, not a stat block: it carries its class's
    # own skills, which is how a transform chain keeps going
    _b = _touching(0, "Ostrich2", players=1, enemy_hp=1.0)
    _packed = _pack_all(_b.agents, _STRICT)
    check("a summon template carries its own actions and passives",
          len(_packed.tmpl_specs) >= 2 and len(_packed.tmpl_passives) > 0,
          f"{len(_packed.tmpl_specs)} templates, "
          f"{len(_packed.tmpl_actions)} action rows, "
          f"{len(_packed.tmpl_passives)} passive rows")

    # `random.choice` is `_randbelow`, which is getrandbits with a rejection
    # loop; how many words it eats depends on what it draws, so a near-miss
    # implementation desynchronises the stream rather than picking wrong.
    _lens = [1, 2, 3, 5, 7, 10, 3, 2, 1, 64, 63, 100, 1, 1, 9]
    _all_match = True
    for _seed in (0, 1, 42, 12345, 987654321987):
        _r = random.Random(_seed)
        if [_r.choice(list(range(n))) for n in _lens] != _choice_probe(_seed, _lens):
            _all_match = False
    check("the core's random.choice matches CPython's", _all_match,
          f"{len(_lens)} draws over 5 seeds")


print("\n== status casts and blinks ==")
from sim.unit_skills import STATUS_OF as _STATUS_OF

# `EmptyAction` is NodeCanvas's no-op and its OnUpdate sets the node's status to
# Success. It used to sit in NOT_MODELLED, which returns FAILURE, and NC-Skill's
# cycle ends on it -- so every skill used to yield its tick to the default
# attack the moment it finished.
from sim.bt import Status as _Status
from sim.bt_leaves import NOT_MODELLED as _NOT_MODELLED
check("EmptyAction is not treated as unmodelled", "EmptyAction" not in _NOT_MODELLED)

_sil = _touching(0, "Silencer", players=1, enemy_hp=None, full_mana=True)
check("a status cast reaches the core", _fast_supported(_sil.agents, _STRICT)[0],
      _fast_supported(_sil.agents, _STRICT)[1])

# Fear is a panic and WatcherDisable is a silence: the skill's name is not the
# status's name, and the packer has to use the same table the oracle does.
check("the status a cast applies comes from STATUS_OF",
      _STATUS_OF["Fear"] == "panic" and _STATUS_OF["WatcherDisable"] == "silence")

# each one actually lands
_CAST_CASES = [("Silencer", "silence"), ("Orb", "stun"), ("Watcher", "silence"),
               ("Rat", "panic")]
for _u, _st in _CAST_CASES:
    _b = _touching(0, _u, players=1, enemy_hp=None, full_mana=True)
    _seen = False
    for _ in range(1200):
        _b.step()
        if any(a.has(_st) for a in _b.agents):
            _seen = True
            break
    check(f"{_u} applies {_st}", _seen)

# A blink moves the caster directly away from its target, so the two need to
# start on different cells: with the target exactly on top of it there is no
# direction to move along and the blink is a no-op -- in both engines, since
# both take `hypot(dx, dy) or 1.0` and then multiply by a zero offset.
_bl_specs = build_player_squad(_STRICT, [("broadsword", 1)])
_bl_foe = [build_unit(units_by_class(_STRICT), "Mushroom", 1, name="Mushroom", size=1)]
_bl_t0 = _place_at(GRID, _bl_specs, _CELLS[:1], team=0)
_bl_t1 = _place_at(GRID, _bl_foe, _CELLS[1:2] or _CELLS[:1], team=1)
_bl_t1[0].mana = _bl_t1[0].spec.mana
_b = Battle(GRID, _bl_t0 + _bl_t1, seed=0, tables=_STRICT)
_caster, _foe = _bl_t1[0], _bl_t0[0]
_gap = lambda: (_caster.x - _foe.x) ** 2 + (_caster.y - _foe.y) ** 2
_before, _moved = _gap(), False
for _ in range(1200):
    _b.step()
    if _gap() > _before + 100.0:
        _moved = True
        break
    if not (_caster.alive and _foe.alive):
        break
check("a blink puts distance between the caster and its target", _moved,
      f"{_before ** 0.5:.1f} -> {_gap() ** 0.5:.1f}")

if _core_available():
    # Both engines, both sides in contact. A tick match means no swing has
    # moved, so any damage difference after it is the cast itself.
    for _u in ("Silencer", "Orb", "Watcher", "Rat", "PoringSmall2", "Mushroom"):
        _e, _m = _death_agreement(_u, players=1, enemy_hp=None, full_mana=True)
        check(f"the core reproduces {_u}'s cast exactly",
              _m >= 20 and _e == _m, f"{_e}/{_m} tick-matched seeds")

    # A blinker teleports and walks back over and over, so the two float widths
    # end the fight a few ticks apart. The damage is what says the blink agrees.
    for _u in ("RangeBlinker", "RangeBlinker2"):
        _err = []
        for _s in range(16):
            _py = _touching(_s, _u, players=1, enemy_hp=None, full_mana=True).run()
            _rs = _fast_battle(GRID, _touching(_s, _u, players=1, enemy_hp=None,
                                               full_mana=True).agents,
                               seed=_s, tables=_STRICT)
            _a, _c = _py.total_damage[0], _rs["damage"][0]
            _err.append(abs(_a - _c) / max(1.0, _a, _c))
        check(f"the core deals the same damage as the oracle for {_u}",
              max(_err) < 0.005, f"worst {max(_err)*100:.2f}%")

    # and with those two kinds in, nothing in the roster is outside any more
    _out = []
    for _cls in sorted(units_by_class(_STRICT)):
        _lv = sorted(units_by_class(_STRICT)[_cls])[0]
        _size = int((_STRICT["Meta"]["Classes"].get(_cls) or {}).get("Size") or 1)
        _foes = [build_unit(units_by_class(_STRICT), _cls, _lv, name=_cls, size=_size)
                 for _ in range(2)]
        _rng = random.Random(0)
        _sq = build_player_squad(_STRICT, [("broadsword", 1)] * 2)
        _bb = Battle(GRID, deploy(GRID, LAYOUT, _sq, 0, "p", _rng)
                     + deploy(GRID, LAYOUT, _foes, 1, "e1", _rng), seed=0, tables=_STRICT)
        _ok, _why = _fast_supported(_bb.agents, _STRICT)
        if not _ok:
            _out.append(f"{_cls}: {_why}")
    check("every enemy class is inside the fast envelope", not _out,
          f"{len(_out)} outside" + (f": {_out[:3]}" if _out else ""))

print("\n== run layer ==")
from sim.run import Food, RoomMap, RunState

_rm = RoomMap.from_table(_STRICT["Rooms"])
check("room map parses", len(_rm.rooms) > 10, f"{len(_rm.rooms)} rooms")
check("start and boss exist", _rm.start in _rm.rooms and _rm.boss in _rm.rooms,
      f"start={_rm.start} boss={_rm.boss}")

# the boss must be reachable from the start, or a run can never finish
_seen, _stack = {_rm.start}, [_rm.start]
while _stack:
    _cur = _stack.pop()
    for _n in _rm.neighbours(_cur):
        if _n not in _seen:
            _seen.add(_n); _stack.append(_n)
check("boss is reachable from the start", _rm.boss in _seen,
      f"{len(_seen)}/{len(_rm.rooms)} rooms reachable")
check("adjacency is symmetric",
      all(a in _rm.neighbours(b) for a in _rm.rooms for b in _rm.neighbours(a)))

# generated maps: one per level, sized by that level's own row. The sim used to
# reuse the fixed map above for all twelve levels, which is 22 rooms and six food
# shops against a row asking for 7 and one.
_bad_count, _bad_shops, _broken, _same = [], [], [], 0
for _seed in range(40):
    _rng = random.Random(_seed)
    for _row in _STRICT["Levels"]:
        _m = RoomMap.generate(_row, _rng)
        if not (_row["MinRooms"] <= len(_m.rooms) <= _row["MaxRooms"]):
            _bad_count.append((_row["ID"], len(_m.rooms)))
        _kinds = collections.Counter(r.kind for r in _m.rooms.values())
        for _col, _kind in (("ItemShops", "item_shop"), ("FoodShops", "food_shop"),
                            ("Shrines", "mutation")):
            if _kinds[_kind] != int(_row.get(_col) or 0):
                _bad_shops.append((_row["ID"], _kind, _kinds[_kind], _row.get(_col)))
        if len(_m.to_boss) != len(_m.rooms):
            _broken.append((_row["ID"], len(_m.to_boss), len(_m.rooms)))
        _same += _m.start == _m.boss
check("a generated level holds the rooms its row asks for", not _bad_count,
      f"{_bad_count[:4]}")
check("a generated level holds the shops its row asks for", not _bad_shops,
      f"{_bad_shops[:4]}")

# the three treasure rooms `ChooseTreasureRooms` places, and what they are:
# a shrine on level 1, a mutation shop on every level after it, and a talent
# shop never -- `StatShops` is 0 on all twelve rows, which is why a Default run
# never sees a consumable (`C_Room.InitShop` builds the consumable shop only in
# a TalentShop room, and `C_ConsumableShop.Buy` is the only caller of
# `C_Consumables.Add`).
_treasure = []
for _seed in range(30):
    _rng = random.Random(700 + _seed)
    for _row in _STRICT["Levels"]:
        _m = RoomMap.generate(_row, _rng)
        _k = collections.Counter(r.kind for r in _m.rooms.values())
        for _col, _kind in (("Shrines", "mutation"),
                            ("RerollShrines", "mutation_shop"),
                            ("StatShops", "talent_shop")):
            if _k[_kind] != int(_row.get(_col) or 0):
                _treasure.append((_row["ID"], _kind, _k[_kind], _row.get(_col)))
check("a level holds the treasure rooms its row asks for", not _treasure,
      f"{_treasure[:4]}")
check("no Default level asks for a talent shop, so consumables never appear",
      all(int(r.get("StatShops") or 0) == 0 for r in _STRICT["Levels"]))

# the mutation shop: a shelf of `ShowCount`, `BuyCount` takes, no reroll
from sim.run import MUTATION_SHOP as _MS

_stm = RunState.new(_STRICT, seed=21)
_stm.level = 3
_stm.rooms = RoomMap.generate(_stm.level_row, _stm.rng)
_shop = next((r for r in _stm.rooms.rooms.values() if r.kind == "mutation_shop"), None)
check("a level past the first has a mutation shop", _shop is not None)
if _shop is not None:
    _stm.room = _shop.id
    _stm.ensure_stock(_shop)
    check("the shop shows ShowCount mutations",
          len(_shop.offer_mutations) == _MS["ShowCount"], f"{len(_shop.offer_mutations)}")
    check("the shelf holds no duplicates",
          len({id(m) for m in _shop.offer_mutations}) == len(_shop.offer_mutations))
    _gold_before, _taken = _stm.gold, 0
    while any(a[0] == "buy_mutation" for a in _stm.legal_actions()) and _taken < 20:
        _stm.apply(next(a for a in _stm.legal_actions() if a[0] == "buy_mutation"))
        _taken += 1
    check("the shop allows exactly BuyCount takes", _taken == _MS["BuyCount"],
          f"{_taken} takes")
    check("a mutation costs nothing", _stm.gold == _gold_before, f"{_stm.gold}")
    check("the takes land in the run's mutations", len(_stm.mutations) == _taken)

# what a mutation on the shelf would do to this squad
from sim.mutations import EFFECT_STATS as _EFFECT_STATS
from sim.mutations import effect_on as _effect_on
from sim.mutations import offered_at_level as _offered

_stmm = RunState.new(_STRICT, seed=41)
_specs_now = _stmm.specs_cached()
from sim.run import Human as _HumanRow

check("the specs cache tracks the squad it was built for",
      _stmm.specs_cached() is _specs_now
      and (_stmm.squad.append(_HumanRow(item=None)) or
           _stmm.specs_cached() is not _specs_now))
_stmm.squad.pop()

# a stat bonus moves the stat it names and leaves Power alone, which is the
# whole reason this is not a Power delta: `UnitSpec.power` is a stored number
# level 5 is the first that offers one. Its `Class` list names the classes it
# touches, so the squad has to field one for the effect to be visible at all --
# which is itself the point of the `fraction` this reports.
_bonus = next(m for m in _offered(_STRICT, 5)
              if m["Name"] == "StatBonus" and m.get("Class") == "Random")
_e = _effect_on(_stmm.specs_cached(), _bonus, random.Random(0))
check("a stat bonus reports as modelled stats", _e["stats"] and _e["implemented"],
      f"{_bonus['Param1Value']} {_e['kind']}")
check("a stat bonus moves the stat it names",
      any(abs(_e[s]) > 0 for s in _EFFECT_STATS) and _e["fraction"] > 0,
      f"{ {s: round(_e[s], 3) for s in _EFFECT_STATS if _e[s]} }")
# and moves no Power, which is why the shelf is described by stats rather than
# by a Power delta: `UnitSpec.power` is `Units.Power` plus the item's, stored.
from sim.mutations import apply_to_specs as _apply_specs

_after = _apply_specs(_stmm.specs_cached(), [_bonus], random.Random(0))
check("a stat bonus moves no Power at all",
      sum(s.power for s in _after) == sum(s.power for s in _stmm.specs_cached()),
      f"{sum(s.power for s in _after):.0f}")
# a mutation aimed at classes the squad does not field touches nobody
_narrow = next(m for m in _offered(_STRICT, 5)
               if m["Name"] == "StatBonus" and m.get("Class") != "Random")
check("a mutation for a class the squad lacks touches nobody",
      _effect_on(_stmm.specs_cached(), _narrow, random.Random(0))["fraction"] == 0.0,
      f"{_narrow.get('Class')}")

# an unimplemented mutation reports as such rather than as a zero effect
_unimpl = next(m for m in _offered(_STRICT, 3)
               if _effect_on(_stmm.specs_cached(), m, random.Random(0))["kind"]
               == "unimplemented")
_eu = _effect_on(_stmm.specs_cached(), _unimpl, random.Random(0))
check("an unimplemented mutation says so", not _eu["implemented"] and not _eu["stats"],
      f"{_unimpl['Name']}")

# How much of what the shops show this sim actually does. It was a quarter
# before the agent-level passives went in and is over half now, which is the
# number the mutation paired test has to be read against: a shelf that mostly
# does nothing cannot be worth shopping at.
_pool = [m for _lv in range(1, 13) for m in _offered(_STRICT, _lv)]
_impl = sum(1 for m in _pool
            if _effect_on(_stmm.specs_cached(), m, random.Random(0))["implemented"])
check("over half the mutation pool is implemented", _impl / len(_pool) > 0.5,
      f"{_impl} of {len(_pool)} implemented")

# gold is a level total split by room type, not a flat payment per room
_stg = RunState.new(_STRICT, seed=31)
_row1 = _stg.level_row
_want = float(_row1["GoldPerRoom"]) * (len(_stg.rooms.rooms) - 1)
_have = sum(_stg.room_gold(r.kind) for r in _stg.rooms.rooms.values())
check("a level pays (rooms - 1) x GoldPerRoom in total", abs(_have - _want) < 1e-3,
      f"{_have:.1f} against {_want:.1f}")
check("a shop pays more than a corridor and the start pays nothing",
      _stg.room_gold("item_shop") > _stg.room_gold("fight") > 0
      and _stg.room_gold("mutation") > _stg.room_gold("fight")
      and _stg.room_gold("start") == 0.0,
      f'shop {_stg.room_gold("item_shop"):.1f}, room {_stg.room_gold("fight"):.1f}')
check("every room on a generated map is reachable", not _broken, f"{_broken[:4]}")
check("the start is not the boss room", _same == 0, f"{_same}")

# the shape rules `LevelGenerator` bounds a map with, over the same 40 seeds
_deg4, _loose, _big, _cut = [], [], [], []
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
for _seed in range(40):
    _rng = random.Random(1000 + _seed)
    for _row in _STRICT["Levels"]:
        _m = RoomMap.generate(_row, _rng)
        _cells = {(r.row, r.col) for r in _m.rooms.values()}
        _deg = {p: sum(((p[0] + d[0], p[1] + d[1]) in _cells) for d in _DIRS)
                for p in _cells}
        # `CheckNeighborCount`: no room may have four room-neighbours
        if max(_deg.values()) > 3:
            _deg4.append((_row["ID"], max(_deg.values())))
        # `CheckSquares` / `CheckMultiSquares`: 2x2 blocks are rationed
        _num = len(_cells)
        _sq = sum(1 for (r, c) in _cells
                  if {(r + 1, c), (r, c + 1), (r + 1, c + 1)} <= _cells)
        if _sq > _num // 10 + 1:
            _big.append((_row["ID"], _sq, _num // 10 + 1))
        # and the map is one piece
        if len(_m.to_boss) != len(_m.rooms):
            _cut.append(_row["ID"])
check("no room has four neighbours (CheckNeighborCount)", not _deg4, f"{_deg4[:4]}")
check("2x2 blocks stay inside maxSquares", not _big, f"{_big[:4]}")
check("a generated map is one connected piece", not _cut, f"{_cut[:4]}")

# the boss room is what `CouldBeFinishRoom` allows: never an articulation point,
# and either a dead end or a room with exactly two neighbours
_arts, _shape = [], []
for _seed in range(30):
    _rng = random.Random(2000 + _seed)
    for _row in _STRICT["Levels"][:6]:
        _m = RoomMap.generate(_row, _rng)
        _cells = {(r.row, r.col) for r in _m.rooms.values()}
        _boss = _m.rooms[_m.boss]
        _n = sum((((_boss.row + d[0], _boss.col + d[1]) in _cells) for d in _DIRS))
        if _n > 2:
            _shape.append((_row["ID"], _n))
        # cutting the boss room must not split the map, which is what
        # "not an articulation point" means
        _rest = _cells - {(_boss.row, _boss.col)}
        if _rest:
            _seen = {next(iter(_rest))}
            _q = collections.deque(_seen)
            while _q:
                _cur = _q.popleft()
                for _d in _DIRS:
                    _nb = (_cur[0] + _d[0], _cur[1] + _d[1])
                    if _nb in _rest and _nb not in _seen:
                        _seen.add(_nb)
                        _q.append(_nb)
            if len(_seen) != len(_rest):
                _arts.append(_row["ID"])
check("the boss room is a dead end or has two neighbours", not _shape, f"{_shape[:4]}")
check("the boss room is not an articulation point", not _arts, f"{_arts[:4]}")

# the start should be the far end of the map from the boss, which is the
# strongest reading of `LevelGenerator`'s `minFinishDistance`
_m = RoomMap.generate(_STRICT["Levels"][3], random.Random(4))
check("the start is the furthest room from the boss",
      _m.to_boss[_m.start] == max(_m.to_boss.values()),
      f"start {_m.to_boss[_m.start]} of {max(_m.to_boss.values())}")

# adjacency is symmetric on a generated map too
check("generated adjacency is symmetric",
      all(a in _m.neighbours(b) for a in _m.rooms for b in _m.neighbours(a)))

# and the level's food supply is now the row's own expression, because the map
# really does hold `FoodShops` food shops
_stf = RunState.new(_STRICT, seed=11)
_row1 = _stf.level_row
check("a food shop's budget is TotalFood / FoodShops",
      _stf.food_shop_budget == _row1["TotalFood"] / _row1["FoodShops"],
      f"{_stf.food_shop_budget}")

# hunger model. `moves` is a list of thresholds on a `maxMoves` reserve, not an
# allowance per stage, and moving feeds the team rather than costing a move.
_f = Food.from_game(_STRICT["Game"])
check("maxMoves is moves[0]", _f.max_moves == _f.moves[0] == _f.moves_left,
      f"max {_f.max_moves}, moves {_f.moves}, left {_f.moves_left}")
check("food starts unhungry", _f.hunger_level == 0 and _f.damage_penalty == 0)

# with food in the larder, a move eats and spends no reserve
_before = _f.amount
check("a move with food in hand feeds instead of moving",
      _f.spend_move(5.0) == "fed" and _f.amount == _before - 5.0
      and _f.moves_left == _f.max_moves, f"amount {_f.amount}, left {_f.moves_left}")

# broke: the reserve drains and hunger is read off the thresholds [6, 3]
_f.amount = 0.0
_want = []
for _ in range(_f.max_moves):
    _f.spend_move(5.0)
    _want.append((_f.moves_left, _f.hunger_level))
check("hunger is derived from the reserve, not incremented",
      _want == [(5, 1), (4, 1), (3, 1), (2, 2), (1, 2), (0, 2)], f"{_want}")
check("hunger applies a damage penalty", _f.damage_penalty == 60, f"{_f.damage_penalty}%")
check("an empty reserve and an empty larder blocks moving",
      not _f.can_move(5.0) and _f.can_move(0.0), f"left {_f.moves_left}")

_f.amount = 20.0
check("food in the larder unblocks moving even at zero reserve", _f.can_move(5.0))
_f.spend_move(5.0)
check("eating restores the reserve and clears hunger in one step",
      _f.hunger_level == 0 and _f.moves_left == _f.max_moves)

# a soft-locked run is over, not merely stuck: the environment only charges its
# terminal penalty on `finished`, so a run that ends any other way is free.
_st9 = RunState.new(_STRICT, seed=17)
_st9.squad = _st9.squad[:1]
_st9.food.amount, _st9.food.moves_left = 0.0, 0
_st9.gold = 0.0
check("a soft-locked run has no legal action", not _st9.legal_actions())
# The invariant, over real play rather than a hand-built state: a run with no
# legal action must already be over. Only `finished` is charged the terminal
# penalty, so a run that stops any other way is cheaper than losing, and an
# agent trained long enough learns to aim for exactly that.
_stuck = []
for _s in range(12):
    _r = RunState.new(_STRICT, seed=500 + _s)
    for _ in range(200):
        _acts = _r.legal_actions()
        if not _acts:
            if not _r.finished:
                _stuck.append((_s, _r.level, len(_r.squad)))
            break
        _r.apply(_r.rng.choice(_acts))
        if _r.finished:
            break
check("a run with no legal action is already finished", not _stuck, f"{_stuck}")

# enemy budget grows with level
_st = RunState.new(_STRICT, seed=1)
_p1 = sum(s.power for s in _st.enemy_specs("fight"))
_st.level = 6
_p6 = sum(s.power for s in _st.enemy_specs("fight"))
check("enemy power scales with level", _p6 > _p1, f"L1 {_p1:.0f} -> L6 {_p6:.0f}")

# shops run out of stock
_st2 = RunState.new(_STRICT, seed=3)
_shop = next((r for r in _st2.rooms.rooms.values() if r.kind == "item_shop"), None)
if _shop is not None:
    _st2.room = _shop.id
    _st2.gold = 9999
    _bought = 0
    while any(a[0] == "buy_item" for a in _st2.legal_actions()) and _bought < 50:
        _st2.apply(next(a for a in _st2.legal_actions() if a[0] == "buy_item"))
        _bought += 1
    check("item shops run out of stock", _bought == _st2.shop_quantity(),
          f"bought {_bought}, shop holds {_st2.shop_quantity()}")

# the food shop: packs, the stocking loop, and what a purchase costs
from sim.run import determine_quantities as _dq
from sim.run import food_packs as _packs

_sizes, _costs = _packs(_STRICT)
check("the food shop sells five packs", len(_sizes) == len(_costs) == 5,
      f"sizes {_sizes}, costs {_costs}")
# [food, gold], not [gold, food]: `C_FoodShop.Buy` spends index 1 and calls
# `AddFood` with index 0. Reading them the other way round would price a 7-food
# pack at 7 gold and hand over 2 food.
check("packs are [food, gold]", all(s > c for s, c in zip(_sizes, _costs))
      and _sizes == [7.0, 11.0, 37.0, 120.0, 250.0]
      and _costs == [2.0, 3.0, 9.0, 27.0, 55.0], f"{list(zip(_sizes, _costs))}")
check("the big packs are the better rate",
      [round(s / c, 2) for s, c in zip(_sizes, _costs)]
      == sorted(round(s / c, 2) for s, c in zip(_sizes, _costs)),
      f"{[round(s / c, 2) for s, c in zip(_sizes, _costs)]}")

# `DetermineQuantities` spends the budget on `costs`, so a shelf costs exactly
# the budget it was stocked to whenever the budget is reachable by the prices.
_q50 = _dq(_sizes, _costs, 50.0)
check("a 50-gold budget stocks the level-5 shelf", _q50 == [4, 2, 1, 1, 0],
      f"{_q50}")
check("the shelf is priced at its budget",
      sum(c * n for c, n in zip(_costs, _q50)) == 50.0
      and sum(s * n for s, n in zip(_sizes, _q50)) == 207.0, f"{_q50}")
# pack 0 is bought whether or not the budget covers it: it is what terminates
# the loop, and it is why a food shop is never empty.
check("pack 0 is stocked even below its price", _dq(_sizes, _costs, 1.0)
      == [1, 0, 0, 0, 0], f"{_dq(_sizes, _costs, 1.0)}")
check("a zero budget still stocks one pack", _dq(_sizes, _costs, 0.0) == [0] * 5
      and _dq(_sizes, _costs, 0.01) == [1, 0, 0, 0, 0])

_st10 = RunState.new(_STRICT, seed=5)
_st10.level = 5
_fs = next(r for r in _st10.rooms.rooms.values() if r.kind == "food_shop")
_st10.room, _st10.gold = _fs.id, 9999.0
_st10.food.amount = 0.0
_st10.ensure_stock(_fs)
_stocked = list(_fs.food_stock)
_g0, _f0 = _st10.gold, _st10.food.amount
_st10.apply(("buy_food", 2))
check("buying a pack charges its gold and adds its food",
      _st10.gold == _g0 - _costs[2] and _st10.food.amount == _f0 + _sizes[2]
      and _fs.food_stock[2] == _stocked[2] - 1,
      f"gold {_g0} -> {_st10.gold}, food {_f0} -> {_st10.food.amount}")

# a shop runs dry and stays dry: `C_FoodShop..ctor` is the only caller of
# `DetermineQuantities` in play, so leaving and coming back is not a restock.
_spent = 0
while any(a[0] == "buy_food" for a in _st10.legal_actions()) and _spent < 200:
    _st10.apply(next(a for a in _st10.legal_actions() if a[0] == "buy_food"))
    _spent += 1
check("a food shop runs out of stock", sum(_fs.food_stock) == 0
      and _spent == sum(_stocked) - 1, f"{_spent} buys, stock {_fs.food_stock}")
_st10.room = next(iter(_st10.rooms.neighbours(_fs.id)))
_st10.room = _fs.id
_st10.ensure_stock(_fs)
check("a sold-out food shop does not restock on re-entry",
      not any(a[0] == "buy_food" for a in _st10.legal_actions()),
      f"{_fs.food_stock}")

# a level hands out no food. `TotalFood` is the shops' gold budget, and
# `C_Levels.WinLevel` touches the larder nowhere.
_st11 = RunState.new(_STRICT, seed=6)
_st11.food.amount = 3.0
_st11.next_level()
check("a new level grants no free food",
      _st11.food.amount == 3.0 and _st11.level == 2, f"{_st11.food.amount}")

print("\n== progression ==")
from sim.data import items_by_quality as _by_quality
from sim.run import MAX_UNIT_LEVEL as _MAX_LV
from sim.run import Human as _Human
from sim.spec import _num as _num

_ITEMS = items_by_name(_STRICT)
_NOVICE = units_by_class(_STRICT)["Novice"]

# The item price is the item's own Cost, not the shop row's Price. The two used
# to be swapped, which made every item cost the same and made quality free.
_st4 = RunState.new(_STRICT, seed=11)
_costs = {n: _st4.item_cost(n) for n in ("broadsword", "stone-sword", "guts-sword")}
check("an item costs its own Cost", _costs == {"broadsword": 4.0, "stone-sword": 12.0,
                                               "guts-sword": 38.0}, f"{_costs}")
check("the shop upgrade charges the next level's Price",
      _st4.upgrade_cost == 3.0 and _st4.shop_level == 1, f"{_st4.upgrade_cost}")

# quality weights come from the row for the shop's level, and a zero weight
# cannot be drawn: at level 1 only quality 1 and 2 exist.
check("shop level 1 offers only quality 1-2", set(_st4.quality_weights()) == {1, 2},
      f"{_st4.quality_weights()}")
check("shop level 5 offers every quality",
      set(_st4.quality_weights(5)) == {1, 2, 3, 4, 5}, f"{_st4.quality_weights(5)}")
_st4.shop_level = 5
_drawn = collections.Counter(_ITEMS[_st4.roll_item()]["Quality"] for _ in range(400))
check("a level-5 shop draws every quality", set(_drawn) == {1, 2, 3, 4, 5}, f"{dict(_drawn)}")
_st4.shop_level = 1
_drawn1 = {_ITEMS[_st4.roll_item()]["Quality"] for _ in range(200)}
check("a level-1 shop never draws above quality 2", _drawn1 <= {1, 2}, f"{sorted(_drawn1)}")

# never-given items are quest props with no stats; the shop must not sell them
check("the shop never offers a NeverGiven item",
      not ({"plant-leaf", "leaflet", "cube-part", "comic-book", "rat-flute",
            "mik-helmet"} & {n for names in _by_quality(_STRICT).values()
                             for n in names}))

# levelling: thresholds, the split, and the cap
_st5 = RunState.new(_STRICT, seed=12)
_st5.squad = [_Human(item="broadsword") for _ in range(4)]
check("a human starts at level 1", all(h.level == 1 for h in _st5.squad))
_st5.gain_experience(4 * 50.0)            # exactly one level each
check("experience is split evenly and levels the squad",
      [h.level for h in _st5.squad] == [2, 2, 2, 2], f"{[h.level for h in _st5.squad]}")
check("the threshold is subtracted, not zeroed",
      all(abs(h.experience) < 1e-9 for h in _st5.squad))
_st5.gain_experience(4 * 10_000.0)
check("levelling stops at 5", [h.level for h in _st5.squad] == [_MAX_LV] * 4)

# the item scales with the level, and it compounds rather than adding
_l1 = build_player_squad(_STRICT, [("opm-costume", 1)])[0]
_l5 = build_player_squad(_STRICT, [("opm-costume", 5)])[0]
_costume = _ITEMS["opm-costume"]
_want = (_num(_NOVICE[5], "Damage")
         + _costume["Damage"] * (1 + _costume["DamagePerLevel"] / 100) ** 4)
check("an item's damage compounds with the unit level",
      abs(_l5.damage - _want) < 1e-3,
      f"L1 {_l1.damage:.1f} -> L5 {_l5.damage:.1f}, want {_want:.1f}")
check("a level-5 human is worth more Power than a level-1 one",
      _l5.power > _l1.power, f"{_l1.power:.0f} -> {_l5.power:.0f}")

# a room pays its gold once. Paid per move, this was an unbounded fountain: an
# agent paced two cleared rooms for 400 steps and ended holding 1,115 gold.
_st8 = RunState.new(_STRICT, seed=15)
_a, _b = _st8.room, _st8.rooms.neighbours(_st8.room)[0]
_st8.apply(("move", _b))
_st8.apply(("move", _a))          # the start room is only cleared once entered
_st8.squad = _st8.squad or [_Human(item="guts-sword")]     # survive the first fight
_gold8 = _st8.gold
for _ in range(6):
    _st8.apply(("move", _a))
    _st8.apply(("move", _b))
check("pacing cleared rooms earns nothing", _st8.gold == _gold8,
      f"{_gold8} -> {_st8.gold} over 12 moves")

# buying experience: 1 gold for 400, split over the squad
_st7 = RunState.new(_STRICT, seed=14)
_shop7 = next(r for r in _st7.rooms.rooms.values() if r.kind == "item_shop")
_st7.room, _st7.gold = _shop7.id, 5.0
_st7.squad = [_Human(item="broadsword") for _ in range(5)]
check("buying experience is a legal shop action",
      ("buy_exp", None) in _st7.legal_actions())
_st7.apply(("buy_exp", None))
check("buying experience costs ExperienceCost and levels the squad",
      _st7.gold == 4.0 and [h.level for h in _st7.squad] == [2, 2, 2, 2, 2],
      f"gold {_st7.gold}, levels {[h.level for h in _st7.squad]}")
_st7.squad = [_Human(item="broadsword", level=_MAX_LV) for _ in range(5)]
check("a maxed squad cannot buy experience",
      ("buy_exp", None) not in _st7.legal_actions())


# a fight pays out the experience of the enemies it killed, and nothing else
def _total_exp(st, h):
    """Experience earned, undoing the thresholds that levelling subtracted."""
    return h.experience + sum(st.exp_thresholds[:h.level - 1])


# A squad strong enough to take a level-1 room without losses, so the payout can
# be compared against what the survivors actually banked.
_st6 = _res6 = _before6 = None
for _seed6 in range(20):
    _cand = RunState.new(_STRICT, seed=_seed6)
    _cand.squad = [_Human(item="guts-sword") for _ in range(6)]
    _snap = {id(h): _total_exp(_cand, h) for h in _cand.squad}
    _r = _cand.fight(next(r for r in _cand.rooms.rooms.values() if r.kind == "fight"))
    if _r.get("won") and len(_cand.squad) == 6:
        _st6, _res6, _before6 = _cand, _r, _snap
        break
check("a fight to compare against was found", _st6 is not None)
if _st6 is not None:
    _delta = sum(_total_exp(_st6, h) - _before6[id(h)] for h in _st6.squad)
    check("a fight pays out exactly the dead enemies' ExpReward",
          abs(_delta - _res6["exp"]) < 1e-6,
          f"paid {_res6['exp']:.1f}, squad banked {_delta:.1f}")
    check("clearing a room pays some experience", _res6["exp"] > 0, f"{_res6['exp']:.1f}")

# a full run terminates, and playing better gets further
def _play(policy, seed, max_steps=400):
    st = RunState.new(_STRICT, seed=seed)
    steps = 0
    while not st.finished and steps < max_steps:
        acts = st.legal_actions()
        if not acts:
            break
        st.apply(policy(st))
        steps += 1
    return st, steps


def _random_policy(st):
    return st.rng.choice(st.legal_actions())


def _greedy_policy(st):
    """The shared run-level baseline; see `rl.heuristic`."""
    from rl.heuristic import heuristic_action
    return heuristic_action(st)


_st3, _steps3 = _play(_random_policy, 0)
check("a run terminates", _st3.finished or _steps3 < 400, f"{_steps3} steps, level {_st3.level}")

_rand_levels = [_play(_random_policy, s)[0].level for s in range(8)]
_greedy_levels = [_play(_greedy_policy, s)[0].level for s in range(8)]
_ra = sum(_rand_levels) / len(_rand_levels)
_ga = sum(_greedy_levels) / len(_greedy_levels)
check("playing better reaches deeper levels", _ga > _ra,
      f"random mean {_ra:.2f}, greedy mean {_ga:.2f}")

print("\n== throughput ==")
t0 = time.perf_counter()
n = 20
runs = [fight([("broadsword", 1)] * 6, "Mancrack", 6, s) for s in range(n)]
el = time.perf_counter() - t0
ticks = sum(r.ticks for r in runs)
print(f"      {n} battles, {ticks} ticks in {el:.2f}s -> {ticks/el:,.0f} ticks/s, "
      f"{el/n*1000:.0f} ms per battle")

print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
