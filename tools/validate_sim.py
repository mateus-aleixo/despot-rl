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

# The checks below test the map, the shops, the fights and the rest of the run
# layer, each by acting on a fresh run. A fresh run now opens on a modal
# level-entry dialog, so they build their runs with the dialogs off unless they
# ask; the dialogs have their own section.
_RunState_new = RunState.new.__func__
RunState.new = classmethod(
    lambda cls, *a, **kw: _RunState_new(cls, *a, **{"dialogs": False, **kw}))

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
        # `num`: the secret rooms are made on top of it
        _num = len(_m.rooms) - len(_m.secret_rooms)
        if not (_row["MinRooms"] <= _num <= _row["MaxRooms"]):
            _bad_count.append((_row["ID"], _num))
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
check("no Default level asks for a talent shop",
      all(int(r.get("StatShops") or 0) == 0 for r in _STRICT["Levels"]))

# the mutation shop, as `C_Room..ctor` builds it on a generated level: a shelf
# of 3, one take, two free rerolls
from sim.run import MUTATION_SHOP as _MS

check("a generated mutation shop is the constructor's defaults",
      (_MS["ShowCount"], _MS["BuyCount"], _MS["RollCount"], _MS["RollCost"])
      == (3, 1, 2, 0.0), f"{_MS}")

_stm = RunState.new(_STRICT, seed=21)
_stm.level = 3
_stm.rooms = RoomMap.generate(_stm.level_row, _stm.rng)
_shop = next((r for r in _stm.rooms.rooms.values() if r.kind == "mutation_shop"), None)
check("a level past the first has a mutation shop", _shop is not None)
if _shop is not None:
    _stm.room = _shop.id
    _shop.cleared = True          # `AfterFight` is what opens a shop
    _stm.ensure_stock(_shop)
    check("the shop shows ShowCount mutations",
          len(_shop.offer_mutations) == _MS["ShowCount"], f"{len(_shop.offer_mutations)}")
    check("the shelf holds no duplicates",
          len({id(m) for m in _shop.offer_mutations}) == len(_shop.offer_mutations))
    # a reroll is a whole new shelf, free, and there are RollCount of them
    _rolled, _shelves = 0, [[m["ID"] for m in _shop.offer_mutations]]
    _gold_before = _stm.gold
    while ("reroll_mutations", None) in _stm.legal_actions() and _rolled < 10:
        _stm.apply(("reroll_mutations", None))
        _rolled += 1
        _shelves.append([m["ID"] for m in _shop.offer_mutations])
    check("the shelf rerolls RollCount times, for nothing",
          _rolled == _MS["RollCount"] and _stm.gold == _gold_before
          and len(_shelves[-1]) == _MS["ShowCount"],
          f"{_rolled} rerolls, shelves {_shelves}")
    check("and a reroll draws the shelf again",
          any(a != b for a, b in zip(_shelves, _shelves[1:])), f"{_shelves}")
    _gold_before, _taken = _stm.gold, 0
    while any(a[0] == "buy_mutation" for a in _stm.legal_actions()) and _taken < 20:
        _stm.apply(next(a for a in _stm.legal_actions() if a[0] == "buy_mutation"))
        _taken += 1
    check("the shop allows exactly BuyCount takes", _taken == _MS["BuyCount"],
          f"{_taken} takes")
    check("a mutation costs nothing", _stm.gold == _gold_before, f"{_stm.gold}")
    check("the takes land in the run's mutations", len(_stm.mutations) == _taken)
    # `OnLevelChange` reserves what the run holds, so it is never offered again
    _held = {m["ID"] for m in _stm.mutations}
    _again = 0
    for _ in range(30):
        _shop.offer_mutations = None
        _stm.ensure_stock(_shop)
        _again += sum(1 for m in _shop.offer_mutations if m["ID"] in _held)
    check("a mutation the run holds is never on a shelf", _held and _again == 0,
          f"held {sorted(_held)}, offered again {_again} times in 30 shelves")

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
# `Class` is a comma-separated list and the starting squad now fields seven of
# them off `ChipChoice/Squads.json`, where five bare Novices fielded almost
# none, so no shelf mutation misses the whole squad any more. The claim under
# test is unchanged: a mutation aimed at classes nobody in the room belongs to
# touches nobody. So pick the mutation first and test it against the units it
# does not name.
_narrow = next(m for m in _offered(_STRICT, 5)
               if m["Name"] == "StatBonus" and m.get("Class") not in ("Random", None))
_named = set(str(_narrow["Class"]).split(","))
_outsiders = [s for s in _stmm.specs_cached() if s.cls not in _named]
check("a mutation for a class the squad lacks touches nobody",
      not _outsiders
      or _effect_on(_outsiders, _narrow, random.Random(0))["fraction"] == 0.0,
      f"{len(_outsiders)} units outside {_narrow.get('Class')}")

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
        # Rooms with four room-neighbours do occur: `CheckNeighborCount` is in
        # the binary with no caller, and the Pit quest's room allows all four doors.
        if max(_deg.values()) > 3:
            _deg4.append((_row["ID"], max(_deg.values())))
        # `CalculateSquares` / `CheckMultiSquares`: 2x2 blocks are rationed while
        # rooms are made; `CloseLoops` writes rooms directly and can add one.
        _num = len(_cells)
        _sq = sum(1 for (r, c) in _cells
                  if {(r + 1, c), (r, c + 1), (r + 1, c + 1)} <= _cells)
        if _sq > _num // 10 + 2:
            _big.append((_row["ID"], _sq, _num // 10 + 1))
        # and the map is one piece
        if len(_m.to_boss) != len(_m.rooms):
            _cut.append(_row["ID"])
check("rooms with four neighbours occur, nothing forbidding them", len(_deg4) > 0,
      f"{len(_deg4)} maps")
check("2x2 blocks stay inside maxSquares, but for what CloseLoops adds", not _big, f"{_big[:4]}")
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

# `PositionRooms` draws the finish and the start uniformly from the rooms that
# qualify. This used to check that the start was the furthest room from the
# boss, which is what the port did until `ChooseStartRoom` was read: it takes
# `RND.Element` of the rooms at least `d` from the finish, `d` counting down
# from `minFinishDistance = num / 6 + 2` until one qualifies.
from sim.mapgen import (MIN_PORTAL_DISTANCE, Generator, min_finish_distance,
                        portals_count, start_is_portal)


def _graph(m):
    """Door distances between every pair of rooms, and the articulation points.
    Secret rooms have no door and are left out."""
    _adj = {rid: list(m.doors(rid)) for rid in m.rooms if rid not in m.secret_rooms}
    _d = {}
    for _src in _adj:
        _out, _q = {_src: 0}, collections.deque([_src])
        while _q:
            _c = _q.popleft()
            for _n in _adj[_c]:
                if _n not in _out:
                    _out[_n] = _out[_c] + 1
                    _q.append(_n)
        _d[_src] = _out
    return _d, Generator(7, random.Random(0)).articulation_points(_adj)


_short, _start_far, _start_short, _below, _levels = [], [], 0, [], 0
for _seed in range(40):
    _rng = random.Random(3000 + _seed)
    for _row in _STRICT["Levels"]:
        _m = RoomMap.generate(_row, _rng)
        _d, _arts = _graph(_m)
        _need = min_finish_distance(len(_m.rooms) - len(_m.secret_rooms))
        # `ChooseFinishRoom`: the boss can see at least `minFinishDistance`
        if max(_d[_m.boss].values()) < _need:
            _short.append((_row["ID"], max(_d[_m.boss].values()), _need))
        # `ChooseStartRoom`: the start is at least the largest `d <= need` that
        # some eligible room clears, and is not always the furthest of them
        _eligible = [r for r in _d if r not in (_m.boss,) and r not in _arts]
        _best = max(_d[r][_m.boss] for r in _eligible)
        if _d[_m.start][_m.boss] < min(_need, _best):
            _below.append((_row["ID"], _d[_m.start][_m.boss], min(_need, _best)))
        _start_short += _d[_m.start][_m.boss] < _best
        _levels += 1
check("the boss room clears minFinishDistance = num / 6 + 2", not _short,
      f"{_short[:4]}")
check("the start clears the highest distance threshold any room can",
      not _below, f"{_below[:4]}")
check("the start is a draw, not the far end of the map",
      0.2 < _start_short / _levels < 1.0,
      f"short of the furthest eligible room on {_start_short} of {_levels} levels")

# `or eax, 5` in `ChooseStartRoom`: every level after the first opens in an item
# shop. The second half gives it teeth: the shop actually sells there.
_lv = [RoomMap.generate(_row, random.Random(17)) for _row in _STRICT["Levels"][:3]]
check("the start room is an item shop on every level but the first",
      [m.rooms[m.start].has("item_shop") for m in _lv] == [False, True, True]
      and all(m.rooms[m.start].kind == "start" for m in _lv))
_sh = RunState.new(_STRICT, seed=5)
_sh.next_level()
_acts = {a[0] for a in _sh.legal_actions()}
check("and the shop is open the moment the level starts",
      _sh.level == 2 and _sh.rooms.rooms[_sh.room].has("item_shop")
      and "reroll" in _acts and _sh.room_gold("start") == 0.0,
      f"level {_sh.level}, actions {sorted(_acts)}")

# -- portals ----------------------------------------------------------------
# `C_Rooms.Generate` sets `portalsCount = num / 7 + 1` and `minPortalDistance =
# 3`; `ChoosePortals` places them on spread-out dead ends, adds one central
# room, undoes the lot if fewer than two land, and makes the start a portal on
# levels 5 and 9.
_counts, _close, _bad_room, _forced, _unforced = collections.Counter(), [], [], [], []
for _seed in range(40):
    _rng = random.Random(4000 + _seed)
    for _row in _STRICT["Levels"]:
        _m = RoomMap.generate(_row, _rng)
        _d, _ = _graph(_m)
        _p = [p for p in _m.portals if p not in _m.secret_rooms]
        _counts[len(_p)] += 1
        _num = len(_m.rooms) - len(_m.secret_rooms)
        _most = (max(portals_count(_num), 2) if _m.secret_rooms else portals_count(_num)) + 1
        if _p and not (2 <= len(_p) <= _most):
            _bad_room.append((_row["ID"], "count", len(_p)))
        if _m.boss in _p:
            _bad_room.append((_row["ID"], "boss"))
        for _i, _a in enumerate(_p):
            for _b in _p[_i + 1:]:
                if _d[_a][_b] < MIN_PORTAL_DISTANCE:
                    _close.append((_row["ID"], _a, _b, _d[_a][_b]))
        if _p and start_is_portal(_row["ID"]) != (_m.start in _p):
            (_forced if start_is_portal(_row["ID"]) else _unforced).append(_row["ID"])
check("every generated level has none (the step undone) or between 2 and "
      "portalsCount + 1 portals, none on the boss", not _bad_room,
      f"{dict(sorted(_counts.items()))} over 480 levels, {_bad_room[:4]}")
check("portals are at least minPortalDistance apart", not _close, f"{_close[:4]}")
check("the start is a portal on levels 5 and 9 and on no other",
      not _forced and not _unforced, f"missing {_forced[:4]}, extra {_unforced[:4]}")

# A teleport: free, legal only from a portal, only to a portal on the map, and
# arriving is the same as walking in.
from sim.run import UNEXPLORED as _UNEXPLORED, UNKNOWN

def _on_portal(seed):
    """Seed `seed`'s first level, standing on a cleared portal that is not the
    start, with another portal still unrevealed. Deterministic in `seed`."""
    _s = RunState.new(_STRICT, seed=seed)
    _m = _s.rooms
    _src = next((p for p in _m.portals if p != _m.start), None)
    if _src is None:
        return None
    _s.room = _src
    _m.set_current(_src)
    _m.rooms[_src].cleared = True
    _dst = next(p for p in _m.portals if p != _src)
    if _m.rooms[_dst].state != UNKNOWN or _dst in _m.doors(_src):
        return None
    return _s, _src, _dst


# the first seed whose teleport fight is won, so the checks after it run on a
# squad that is still standing
for _seed in range(200):
    _tp = _on_portal(_seed)
    if _tp is None:
        continue
    _tp[0].rooms.rooms[_tp[2]].state = _UNEXPLORED
    _tp[0].rooms._refresh_known()
    if _tp[0].apply(("teleport", _tp[2])).get("won"):
        break
_s, _src, _dst = _on_portal(_seed)
check("an unrevealed portal is not a teleport target",
      ("teleport", _dst) not in _s.legal_actions())
_s.rooms.rooms[_dst].state = _UNEXPLORED
_s.rooms._refresh_known()
check("a revealed one is, without having been visited",
      ("teleport", _dst) in _s.legal_actions() and not _s.rooms.rooms[_dst].cleared)
_plain = next(r for r in _s.rooms.rooms if r not in _s.rooms.portals)
check("a room that is not a portal teleports nowhere",
      _s.rooms.teleports(_plain) == [])
_food, _left = _s.food.amount, _s.food.moves_left
_res = _s.apply(("teleport", _dst))
check("a teleport spends no food and no move",
      _s.food.amount == _food and _s.food.moves_left == _left and "food" not in _res,
      f"food {_food} -> {_s.food.amount}, moves {_left} -> {_s.food.moves_left}")
check("teleporting into a room nobody has entered fights it",
      "won" in _res and _s.room == _dst, f"{_res}")
check("and reveals its doors the way walking in does",
      all(_s.rooms.rooms[n].state != UNKNOWN for n in _s.rooms.doors(_dst)))
_s.food.amount, _s.food.moves_left = 0.0, 0
_acts = _s.legal_actions()
check("a starving squad on a portal can still teleport",
      not any(a[0] == "move" for a in _acts)
      and any(a[0] == "teleport" for a in _acts), f"{_acts}")
# with every room cleared and nothing to buy, a teleport leads nowhere
for _r in _s.rooms.rooms.values():
    _r.cleared = True
_s.squad = _s.squad[:1]
_s.gold = 0.0
check("only teleports into cleared rooms is stranded", _s.stranded(),
      f"{_s.legal_actions()}")
_s.rooms.rooms[_src].cleared = False
check("one into a room nobody has cleared is not", not _s.stranded())

# adjacency is symmetric on a generated map too
check("generated adjacency is symmetric",
      all(a in _m.neighbours(b) for a in _m.rooms for b in _m.neighbours(a)))

# -- fog of war ------------------------------------------------------------
# `C_Rooms.SetCurrent` sets the room being left to `EXPLORED`, the room being
# entered to `CURRENT`, and every door neighbour still at `UNKNOWN` to
# `UNEXPLORED`. These check that it does, and that the observation is built on
# it rather than on the whole map.
from sim.run import CURRENT, EXPLORED, UNEXPLORED, UNKNOWN

_fs = RunState.new(_STRICT, seed=7)
_fm = _fs.rooms
check("a level starts mostly hidden",
      len(_fm.known()) < len(_fm.rooms),
      f"{len(_fm.known())} of {len(_fm.rooms)} rooms on the map")
check("the start room is the only CURRENT room",
      [r for r in _fm.rooms.values() if r.state == CURRENT]
      == [_fm.rooms[_fm.start]])
check("the reveal is exactly the start room's door neighbours",
      {rid for rid in _fm.known() if rid != _fm.start}
      == set(_fm._ortho[_fm.start]),
      f"revealed {sorted(set(_fm.known()) - {_fm.start})}, "
      f"doors {sorted(_fm._ortho[_fm.start])}")

# walking uncovers more of the level, and never uncovers less
_seen_counts, _regressed = [len(_fm.known())], []
_before = {rid: r.state for rid, r in _fm.rooms.items()}
for _ in range(40):
    _moves = [a for a in _fs.legal_actions() if a[0] == "move"]
    if not _moves or _fs.finished:
        break
    _lvl = _fs.level
    _fs.apply(min(_moves, key=lambda a: not _fs.rooms.rooms[a[1]].cleared))
    if _fs.level != _lvl:            # a new level starts its own fog
        break
    for _rid, _st in _before.items():
        if _fs.rooms.rooms[_rid].state == UNKNOWN and _st != UNKNOWN:
            _regressed.append(_rid)
    _before = {rid: r.state for rid, r in _fs.rooms.rooms.items()}
    _seen_counts.append(len(_fs.rooms.known()))
check("walking uncovers more of the level",
      _seen_counts[-1] > _seen_counts[0],
      f"{_seen_counts[0]} -> {_seen_counts[-1]} rooms")
check("a revealed room is never hidden again", not _regressed, f"{_regressed[:4]}")

# the reveal follows doors, not portals. The shipped fixed map is the only map
# with portals in it, so it is the only place this can be checked at all.
_pm = RoomMap.from_table(_STRICT["Rooms"])
_p0 = _pm.portals[0]
_pm.set_current(_p0)
check("standing on a portal does not reveal the portals it links to",
      all(_pm.rooms[_p].state == UNKNOWN
          for _p in _pm.portals if _p != _p0 and _p not in _pm._ortho[_p0]),
      f"portals {_pm.portals}, "
      f"states {[_pm.rooms[_p].state for _p in _pm.portals]}")
check("but the portals it links to stay one step away in the distance graph",
      set(_pm.portals) - {_p0} <= set(_pm.neighbours(_p0)))
check("and they are not teleport targets until they are on the map",
      _pm.teleports(_p0) == [p for p in _pm.portals
                             if p != _p0 and _pm.rooms[p].state != UNKNOWN])

# `known_to_boss` is the fair distance, and it is empty until the boss is found
_fb = RunState.new(_STRICT, seed=3).rooms
check("the distance to the boss is unavailable until the boss is found",
      not _fb.boss_found() and _fb.known_to_boss == {},
      f"boss_found={_fb.boss_found()}, {len(_fb.known_to_boss)} distances")
for _r in _fb.rooms.values():         # reveal the whole level by hand
    _r.state = EXPLORED
_fb._refresh_known()
check("with the whole level revealed it agrees with the true distance",
      _fb.known_to_boss == _fb.to_boss,
      f"{len(_fb.known_to_boss)} of {len(_fb.to_boss)}")

# `C_BossVisionMutation.OnNewLevel`, mutation 188
_bv = RunState.new(_STRICT, seed=5).rooms
check("BossVision puts the boss on the map", _bv.reveal_boss() and _bv.boss_found())
check("BossVision is worth nothing once the boss is already found",
      not _bv.reveal_boss())

# and the observation is built on what has been revealed. Rewriting every room
# the squad cannot see must not move a single number in the encoding: that is
# the whole claim, and it is what the old `to_boss` and `len(rooms)` features
# failed. The second half gives the check teeth by showing a *visible* room
# still moves it.
from rl.env import DespotRunEnv

_leaks, _blind, _rewritten, _tested = [], [], 0, 0
for _seed in range(12):
    _env = DespotRunEnv(tables=_STRICT, seed=_seed)
    _env.reset(seed=_seed)
    # a few points along a run, not only the first room of the first level
    for _hop in range(6):
        _hidden = [r for r in _env.state.rooms.rooms.values() if r.state == UNKNOWN]
        if _hidden:
            _tested += 1
            _rewritten += len(_hidden)
            _o1 = _env._encode(_env.state).copy()
            _was = [(r.kind, r.cleared) for r in _hidden]
            for _r in _hidden:
                _r.kind = "boss" if _r.kind != "boss" else "fight"
                _r.cleared = not _r.cleared
            # and every hidden room swaps whether it is a portal, which a slot
            # order over all of the level's portals would give away
            _rm = _env.state.rooms
            _portals_were = list(_rm.portals)
            _rm.portals = sorted(set(_rm.portals) ^ {r.id for r in _hidden})
            _rm._portal_set = frozenset(_rm.portals)
            if (_o1 != _env._encode(_env.state)).any():
                _leaks.append((_seed, _hop))
            _rm.portals = _portals_were
            _rm._portal_set = frozenset(_portals_were)
            for _r, (_k, _c) in zip(_hidden, _was):
                _r.kind, _r.cleared = _k, _c
        _moves = [a for a in _env.state.legal_actions() if a[0] == "move"]
        if not _moves or _env.state.finished:
            break
        _env.state.apply(_moves[0])
check("the observation ignores rooms the squad has not seen", not _leaks,
      f"{_tested} states, {_rewritten} hidden rooms rewritten, leaked at {_leaks[:4]}")

# and the same rewrite on a room the squad *can* see does move it, which is what
# gives the check above its teeth
_env = DespotRunEnv(tables=_STRICT, seed=0)
_env.reset(seed=0)
_o2 = _env._encode(_env.state).copy()
_visible = [r for r in _env.state.rooms.rooms.values()
            if r.state == UNEXPLORED and r.id in _env.state.rooms._ortho[_env.state.room]]
for _r in _visible:
    _r.kind = "food_shop" if _r.kind != "food_shop" else "item_shop"
_o3 = _env._encode(_env.state)
check("the observation does follow a room the squad can see",
      bool(_visible) and (_o2 != _o3).any(),
      f"{len(_visible)} visible neighbours rewritten, "
      f"{int((_o2 != _o3).sum())} features moved")
# and the portal half of the rewrite has teeth of its own: making a visible
# door neighbour a portal moves it, through the neighbour's portal bit and the
# teleport slot it now fills
_rm = _env.state.rooms
_was = list(_rm.portals)
_rm.portals = sorted(set(_rm.portals) ^ {r.id for r in _visible})
_rm._portal_set = frozenset(_rm.portals)
_rm._refresh_known()
_o4 = _env._encode(_env.state)
check("and whether a room it can see is a portal",
      bool(_visible) and (_o3 != _o4).any(), f"{int((_o3 != _o4).sum())} features moved")
_rm.portals, _rm._portal_set = _was, frozenset(_was)

# and the level's food supply is now the row's own expression, because the map
# really does hold `FoodShops` food shops
_stf = RunState.new(_STRICT, seed=11)
_row1 = _stf.level_row
check("a food shop's budget is TotalFood / FoodShops",
      _stf.food_shop_budget == _row1["TotalFood"] / _row1["FoodShops"],
      f"{_stf.food_shop_budget}")

# -- wounds carry from fight to fight ----------------------------------------
# In Default nothing heals a unit between fights (`C_Unit.BeforeFight` restores
# health only in KingOfTheHill and an Arcade boss room), and a level or an item
# rescales health as `totalMaxHealth * ratio`. These check the share is carried,
# kept, and actually handed to the next fight.
import sim.run as _sim_run
from sim.run import Human as _Hum

_starts = []


class _ProbeBattle(_sim_run.Battle):
    """Record what each player unit starts a fight with."""

    def __init__(self, grid, agents, *a, **kw):
        _starts.append([(ag.hp, ag.max_hp) for ag in agents if ag.team == 0])
        super().__init__(grid, agents, *a, **kw)


_real_battle = _sim_run.Battle
_sim_run.Battle = _ProbeBattle
try:
    _sw = RunState.new(_STRICT, seed=13)
    _hurt_seen = None
    for _ in range(30):
        _moves = [a for a in _sw.legal_actions() if a[0] == "move"
                  and not _sw.rooms.rooms[a[1]].cleared]
        if not _moves or _sw.finished:
            break
        _sw.apply(_moves[0])
        if any(h.hp < 1.0 for h in _sw.squad):
            _hurt_seen = [h.hp for h in _sw.squad]
            break
    check("a fight leaves its wounds on the squad",
          _hurt_seen is not None and all(0.0 < x <= 1.0 for x in _hurt_seen),
          f"{_hurt_seen}")
    _carried = [h.hp for h in _sw.squad]
    _starts.clear()
    _next = [a for a in _sw.legal_actions() if a[0] == "move"
             and not _sw.rooms.rooms[a[1]].cleared]
    if _next:
        _sw.apply(_next[0])
    _first = _starts[0] if _starts else []
    check("and the next fight starts with them",
          bool(_first) and all(abs(hp - mx * c) < 1e-3
                               for (hp, mx), c in zip(_first, _carried)),
          f"carried {[round(c, 3) for c in _carried]}, "
          f"started {[round(hp / mx, 3) for hp, mx in _first]}")
finally:
    _sim_run.Battle = _real_battle

# a level and an item both keep the share; a bought human is whole
_sk = RunState.new(_STRICT, seed=2)
_sk.squad[0].hp = 0.5
_sk.gain_experience(10_000.0, targets=[_sk.squad[0]])
check("a level-up keeps the share of health", _sk.squad[0].level > 1
      and _sk.squad[0].hp == 0.5, f"level {_sk.squad[0].level}, hp {_sk.squad[0].hp}")
check("a new human arrives at full health", _Hum(item=None).hp == 1.0)
# the pit takes the most wounded
_sk.squad[1].hp = 0.1
_before_ids = [id(h) for h in _sk.squad]
_wounded = id(_sk.squad[1])
_sk.apply(("sacrifice", None))
check("the pit takes the most wounded human",
      _wounded not in {id(h) for h in _sk.squad} and len(_sk.squad) == len(_before_ids) - 1)
# carry_hp off is the old sim: everyone whole after every fight
_so = RunState.new(_STRICT, seed=13)
_so.carry_hp = False
for _ in range(30):
    _moves = [a for a in _so.legal_actions() if a[0] == "move"
              and not _so.rooms.rooms[a[1]].cleared]
    if not _moves or _so.finished:
        break
    _so.apply(_moves[0])
check("with carry_hp off every unit stays whole", all(h.hp == 1.0 for h in _so.squad))

# Mutation 356, `FoodPackHeal`: buying food heals `cost * koefficient` percent
_sf = RunState.new(_STRICT, seed=11)
_fp = next(m for m in _STRICT["SimpleMutations"] if m["Name"] == "FoodPackHeal")
_sf.mutations.append(_fp)
_shopf = next(r for r in _sf.rooms.rooms.values() if r.kind == "food_shop")
_sf.room = _shopf.id
_shopf.cleared = True
_sf.ensure_stock(_shopf)
_sf.gold = 100.0
for _h in _sf.squad:
    _h.hp = 0.5
_sf.apply(("buy_food", 1))
_cost1 = _sf.food_packs[1][1]
check("FoodPackHeal heals the squad by what the pack cost",
      all(abs(h.hp - (0.5 + _cost1 / 100.0)) < 1e-9 for h in _sf.squad),
      f"pack cost {_cost1}, hp {[h.hp for h in _sf.squad]}")

# -- the boss room, the pipe, and consumables ---------------------------------
from sim.run import (CONSUMABLE_SHELF as _SHELF, CONSUMABLE_SLOTS as _SLOTS,
                     Consumable as _Cons, consumable_family as _fam)

_cons_rows = {c["ID"]: c for c in _STRICT["Consumables"]}


def _won_boss(seed, level=1):
    """A run standing in a boss room it has just won."""
    s = RunState.new(_STRICT, seed=seed)
    if level > 1:
        s.level = level
        s.rooms = RoomMap.generate(s.level_row, s.rng)
    s.room = s.rooms.boss
    s.rooms.set_current(s.room)
    s.rooms.rooms[s.room].cleared = True
    s.win_level()
    return s


_bw = _won_boss(31)
check("beating the boss does not change the level", _bw.level == 1 and _bw.level_won
      and ("descend", None) in _bw.legal_actions())
_bw.apply(("descend", None))
check("the pipe does", _bw.level == 2 and not _bw.level_won)
_bl = _won_boss(32, level=len(_STRICT["Levels"]))
check("the last boss ends the run, won", _bl.finished and _bl.won and not _bl.level_won)

_bs = _won_boss(33)
_bs.ensure_stock(_bs.rooms.rooms[_bs.room])
_shelf_c = _bs.rooms.rooms[_bs.room].consumable_offer
_lvl1 = {r["Consumable"] for r in _STRICT["ConsumablesByLevel"] if r["Level"] == 1}
check("a boss room shows three consumables from its level's list, none twice",
      len(_shelf_c) == _SHELF and {c["ID"] for c in _shelf_c} <= _lvl1
      and len({c["ID"] for c in _shelf_c}) == _SHELF, f"{[c['Name'] for c in _shelf_c]}")
_g0 = _bs.gold
while any(a[0] == "take_consumable" for a in _bs.legal_actions()):
    _bs.apply(next(a for a in _bs.legal_actions() if a[0] == "take_consumable"))
check("they are free, and the run carries two", len(_bs.consumables) == _SLOTS
      and _bs.gold == _g0, f"{len(_bs.consumables)} carried, gold {_g0} -> {_bs.gold}")
_bs.apply(("sell_consumable", 0))
check("selling one pays SellCost", _bs.gold == _g0 + _bs.consumable_sell_price
      and _bs.consumable_sell_price == 5.0 and len(_bs.consumables) == _SLOTS - 1)
# what the run carries is never on the next shelf
_bs.rooms.rooms[_bs.room].consumable_offer = None
_carried_ids = {c.row["ID"] for c in _bs.consumables}
_again = 0
for _ in range(30):
    _bs.rooms.rooms[_bs.room].consumable_offer = None
    _bs.ensure_stock(_bs.rooms.rooms[_bs.room])
    _again += sum(1 for c in _bs.rooms.rooms[_bs.room].consumable_offer if c["ID"] in _carried_ids)
check("nothing the run carries is on a shelf", _again == 0)

# The long-run reward: `C_Levels.WinLevel` on level 4 sets a persistent flag,
# and the next run's level-1 `C_Rooms.Generate` takes it and makes the start
# room `Start | ConsumableShop`, two slots wide.
_lr = RunState.new(_STRICT, seed=34, long_run_reward=True)
_lr_start = _lr.rooms.rooms[_lr.rooms.start]
_lr_takes = [a for a in _lr.legal_actions() if a[0] == "take_consumable"]
check("the long-run reward opens level 1 in a two-slot consumable shop",
      _lr_start.has("consumable_shop") and _lr.room == _lr.rooms.start
      and _lr_takes == [("take_consumable", 0), ("take_consumable", 1)]
      and {c["ID"] for c in _lr_start.consumable_offer} <= _lvl1, f"{_lr_takes}")
_lr.apply(("take_consumable", 1))
check("and what it shows is free to take", len(_lr.consumables) == 1
      and _lr_start.consumable_offer[1] is None)
_plain = RunState.new(_STRICT, seed=34)
check("a run without it opens with no shop and the same map",
      not any(a[0] == "take_consumable" for a in _plain.legal_actions())
      and not _plain.rooms.rooms[_plain.rooms.start].has("consumable_shop")
      and set(_plain.rooms.rooms) == set(_lr.rooms.rooms))
_crazy = RunState.new(_STRICT, seed=34, chip="crazy", long_run_reward=True)
check("the crazy chip leaves the flag for a later run",
      not _crazy.long_run_reward_used
      and not _crazy.rooms.rooms[_crazy.rooms.start].has("consumable_shop"))
_lr.level = 2
_lr.generate_level()
check("only level 1 gets it", not _lr.rooms.rooms[_lr.rooms.start].has("consumable_shop"))
_w3, _w4 = _won_boss(35, level=3), _won_boss(35, level=4)
check("winning level 4 sets the flag for the next run, level 3 does not",
      _w4.long_run_reward_earned and not _w3.long_run_reward_earned)
from rl.env import DespotRunEnv as _Env
_ce = _Env(tables=_STRICT, long_run_reward="carry", quests=False, dialogs=False)
_ce.reset(seed=36)
_c0 = _ce.state.long_run_reward
_ce.state.long_run_reward_earned = True
_ce.reset(seed=37)
_c1 = _ce.state.long_run_reward
_ce.reset(seed=38)
check("the env carries it to the next episode only, the way the save does",
      not _c0 and _c1 and not _ce.state.long_run_reward)


def _give(s, cid):
    s.consumables = [_Cons(row=_cons_rows[cid], charges=int(_cons_rows[cid]["Count"]))]


# heals: the potion makes the most wounded whole, the scroll gives 30% to all
_hs = RunState.new(_STRICT, seed=5)
for _h, _v in zip(_hs.squad, (0.2, 0.6, 0.9, 1.0)):
    _h.hp = _v
_give(_hs, 1)
_hs.apply(("use_consumable", 0))
check("a Heal potion makes the most wounded unit whole",
      [h.hp for h in _hs.squad][:4] == [1.0, 0.6, 0.9, 1.0]
      and _hs.consumables[0].charges == _cons_rows[1]["Count"] - 1)
_give(_hs, 2)
_hs.apply(("use_consumable", 0))
check("a Heal scroll gives every unit 30%",
      all(abs(a - b) < 1e-9 for a, b in zip([h.hp for h in _hs.squad][:4], [1.0, 0.9, 1.0, 1.0])),
      f"{[h.hp for h in _hs.squad]}")

# a cleaner wins the next room it is carried into, with no fight and its experience
_built = []


class _CountBattle(_sim_run.Battle):
    def __init__(self, *a, **kw):
        _built.append(1)
        super().__init__(*a, **kw)


_sim_run.Battle = _CountBattle
try:
    _cs = RunState.new(_STRICT, seed=6)
    _give(_cs, 3)
    _cs.apply(("use_consumable", 0))
    _tgt = next(a for a in _cs.legal_actions() if a[0] == "move"
                and not _cs.rooms.rooms[a[1]].cleared
                and _cs.rooms.rooms[a[1]].kind != "boss")
    _exp0 = sum(h.experience + 1000 * h.level for h in _cs.squad)
    _built.clear()
    _r = _cs.apply(_tgt)
    check("a RoomCleaner wins the next room without a fight",
          _r.get("cleaned") and _cs.rooms.rooms[_tgt[1]].cleared and not _built
          and sum(h.experience + 1000 * h.level for h in _cs.squad) > _exp0, f"{_r}")
finally:
    _sim_run.Battle = _real_battle

# reserves: three Novices join for good; an ally fights the next fight and goes
_rs = RunState.new(_STRICT, seed=7)
_n0 = len(_rs.squad)
_give(_rs, 4)
_rs.apply(("use_consumable", 0))
check("a Reserve of Novices adds three humans", len(_rs.squad) == _n0 + 3
      and all(h.item is None for h in _rs.squad[_n0:]))
_starts.clear()
_sim_run.Battle = _ProbeBattle
try:
    _ra = RunState.new(_STRICT, seed=7)
    _give(_ra, 5)
    _ra.apply(("use_consumable", 0))
    _n1 = len(_ra.squad)
    _mv = next(a for a in _ra.legal_actions() if a[0] == "move"
               and not _ra.rooms.rooms[a[1]].cleared)
    _ra.apply(_mv)
finally:
    _sim_run.Battle = _real_battle
check("a summoned ally fights the next fight and does not join the squad",
      _starts and len(_starts[0]) == _n1 + 1 and len(_ra.squad) <= _n1
      and not _ra.armed.get("allies"), f"{len(_starts[0]) if _starts else 0} fielded")

# resurrection brings back the dead lying in this room
_rz = RunState.new(_STRICT, seed=8)
_dead = _rz.squad.pop()
_here = _rz.rooms.rooms[_rz.room]
_here.corpses.append(_dead)
_give(_rz, 8)
check("Resurrection is usable where corpses lie",
      ("use_consumable", 0) in _rz.legal_actions())
_rz.apply(("use_consumable", 0))
check("and brings them back", _dead in _rz.squad and not _here.corpses and _dead.hp == 1.0)

# teleport: a random room nobody has seen, no food spent
_tz = RunState.new(_STRICT, seed=9)
_hidden_before = {rid for rid, r in _tz.rooms.rooms.items() if r.state == UNKNOWN}
_give(_tz, 9)
_food0 = (_tz.food.amount, _tz.food.moves_left)
_tz.apply(("use_consumable", 0))
check("a Teleport lands in a room that was hidden, for no food",
      _tz.room in _hidden_before and (_tz.food.amount, _tz.food.moves_left) == _food0)

# AntiHunger lifts the hunger penalties for one fight
_az = RunState.new(_STRICT, seed=10)
_az.food.amount, _az.food.moves_left = 0.0, 0
_az.food.spend_move(_az.feed_cost)
check("a starving squad fights with its penalties", _az.food.damage_penalty > 0
      and _az.specs()[0].damage < _az.specs(no_hunger=True)[0].damage)
_give(_az, 11)
_az.apply(("use_consumable", 0))
check("AntiHunger waits for the next fight", _az.armed.get("anti_hunger"))

# Reroll: a free new shelf in the shop the squad stands in
_rr = RunState.new(_STRICT, seed=2)
_rr.next_level()                      # level 2 opens in an item shop
_rr.ensure_stock(_rr.rooms.rooms[_rr.room])
_before_offer, _g1 = list(_rr.offer), _rr.gold
_give(_rr, 14)
_rr.apply(("use_consumable", 0))
check("a Reroll rolls the item shop for free",
      _rr.gold == _g1 and len(_rr.offer) == len(_before_offer), f"{_before_offer} -> {_rr.offer}")
check("every consumable but DamageAllEnemies has an effect",
      {c["Name"] for c in _STRICT["Consumables"] if _fam(c) == "unmodelled"}
      == {"DamageAllEnemies"})

# DejaVu: taken, it saves; a wipe while it is carried rewinds to the save
_dj = _won_boss(40)
_dj.ensure_stock(_dj.rooms.rooms[_dj.room])
_dj.rooms.rooms[_dj.room].consumable_offer = [_cons_rows[10]]
_dj.apply(("take_consumable", 0))
_dj_n, _dj_gold = len(_dj.squad), _dj.gold
_dj.squad = []
_rw = _dj.apply(("buy_human", None))
check("a wipe with a DejaVu carried rewinds to its save",
      _rw.get("dejavu") and not _dj.finished and len(_dj.squad) == _dj_n
      and _dj.gold == _dj_gold and not _dj.holds_dejavu(), f"{_rw}")
_dj.squad = []
_dj.apply(("buy_human", None))
check("and a second wipe ends the run", _dj.finished and not _dj.won)

# -- layouts: the pack chooses the room ------------------------------------------
from sim.data import parse_room_layouts as _prl

_lay_by_id = {lay.id: lay for lay in _prl(_STRICT["RoomLayouts"])}
_lay_of = {r["Pack"]: r["Layout"] for r in _STRICT["LayoutsByPack"]}
_lay_bad = []
for _row in _STRICT["EnemyPacks"]:
    if _row["ID"] >= 1001:              # 1001 to 1011 are no Default level's
        continue
    _lid = _lay_of.get(_row["ID"])
    if _lid is None or not _lay_by_id[_lid].cells(_row.get("Zone") or "e1"):
        _lay_bad.append((_row["ID"], _row.get("Zone"), _lid))
check("every pack has a layout holding every zone its rows name", not _lay_bad,
      f"{_lay_bad[:4]}")
_lf = RunState.new(_STRICT, seed=3)
_lay_seen = []
for _ in range(8):
    _mv = [a for a in _lf.legal_actions() if a[0] == "move"
           and not _lf.rooms.rooms[a[1]].cleared]
    if not _mv or _lf.finished:
        break
    _lf.apply(_mv[0])
    _lay_seen.append((_lf._last_pack[0], _lf._last_layout))
check("a fight is fought in its pack's own layout",
      _lay_seen and all(_lay_of.get(p) == l for p, l in _lay_seen), f"{_lay_seen}")

# -- the level-entry dialogs ----------------------------------------------------
from sim.dialogs import Engine as _Engine, pool as _dpool, table as _dtable

_names = _dpool(_STRICT)
check("the Default pool is Game.Dialogs, all 46 in the table",
      len(_names) == 46 and all(n in _dtable() for n in _names), f"{len(_names)}")

# every pool dialog walks to its end under random choices
_bad_walk = []
for _n in _names:
    for _seed in range(6):
        _wd = RunState.new(_STRICT, seed=_seed)
        _wd.dialog = None
        _wd.gold = 40.0
        _eng = _Engine(_wd)
        if not _eng.can_start(_n):
            continue
        try:
            _wd.dialog = _eng.start(_n)
            _wd._settle_dialog()
            _wr = random.Random(_seed)
            for _ in range(60):
                if _wd.dialog is None:
                    break
                _wd.apply(_wr.choice(_wd.legal_actions()))
            if _wd.dialog is not None:
                _bad_walk.append((_n, "open"))
        except Exception as _ex:
            _bad_walk.append((_n, repr(_ex)[:60]))
check("every pool dialog runs to its end", not _bad_walk, f"{_bad_walk[:4]}")

# a run opens with one and every level change brings another, never twice
_dr = RunState.new(_STRICT, seed=4, dialogs=True)
_seen = [n for n in _names if n not in _dr.dialog_pool]
_dr.dialog = None
_dr.next_level()
_seen2 = [n for n in _names if n not in _dr.dialog_pool]
check("the run opens with a dialog and the next level brings another",
      len(_seen) == 1 and len(_seen2) == 2 and _seen[0] in _seen2, f"{_seen} -> {_seen2}")

# gates: `Stores` needs 5 gold to start, and its 5-gold purchase needs 5 gold
_dg = RunState.new(_STRICT, seed=1)
_dg.dialog = None
_dg.gold = 0.0
_eg = _Engine(_dg)
_closed_poor = _eg.can_start("Stores")
_dg.gold = 5.0
check("a dialog whose own gates fail does not start", not _closed_poor
      and _eg.can_start("Stores"))
_dg.dialog = _eg.start("Stores")
_dg._settle_dialog()
_dg.apply(("choose", 0))                       # the "Steam" door
_rich = list(_eg.legal(_dg.dialog))
_dg.gold = 4.0
_poor = list(_eg.legal(_dg.dialog))
check("a choice that costs more than the purse holds is not offered",
      0 in _rich and 0 not in _poor and 1 in _poor, f"5 gold: {_rich}, 4 gold: {_poor}")

# `OneHandBandit` defines `robotsGold` and tests it in the same node
check("a node's variables hold for its own gates", _eg.can_start("OneHandBandit"))

# `WheelOfFortune`'s first prize is mutation 4
_dw = RunState.new(_STRICT, seed=0)
_dw.dialog = None
_ew = _Engine(_dw)
_got = set()
for _seed in range(20):
    _dw.rng = random.Random(_seed)
    _dw.mutations = []
    _dw.dialog = _ew.start("WheelOfFortune")
    _dw._settle_dialog()
    _dw.apply(("choose", 1))                   # spin the wheel
    while _dw.dialog is not None:
        _dw.apply(("choose", _ew.legal(_dw.dialog)[0]))
    _got |= {m["ID"] for m in _dw.mutations}
check("GiveMutation hands over the mutation it names", _got == {4, 5}, f"{sorted(_got)}")

# an Ambush makes the room the squad stands in fight after all
_built.clear()
_sim_run.Battle = _CountBattle
try:
    _da = RunState.new(_STRICT, seed=2)
    _da.dialog = None
    _da.gold = 0.0
    _da.squad[0].item = "stone-sword"
    _ea = _Engine(_da)
    _da.dialog = _ea.start("GuidingStone")
    _da._settle_dialog()
    _da.apply(("choose", 1))                   # "Go left"
    while _da.dialog is not None:
        _da.apply(("choose", _ea.legal(_da.dialog)[0]))
finally:
    _sim_run.Battle = _real_battle
check("an Ambush makes the start room fight", _built and _da.gold >= 10.0,
      f"{len(_built)} battles, gold {_da.gold}")

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
    _shop.cleared = True          # `AfterFight` is what opens a shop
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
_fs.cleared = True                # `AfterFight` is what opens a shop
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

# -- quests ------------------------------------------------------------------
print("\n== quests ==")
import sim.quests as _Q
from sim.mapgen import DIRECTION_STEP as _STEP
from sim.run import UNKNOWN as _UNKNOWN
from sim.run import Human as _QH


def _quest_level(name, seed=0, squad=None):
    """A run put on level 2 with the quest `name`, standing in the start room."""
    st = RunState.new(_STRICT, seed=seed)
    st.level = {"Sci": 2, "Pit": 2, "Cube": 2, "Rat": 2, "Reaper": 6, "SecretRoom": 6,
                "Spider": 6, "Bot": 6, "Nurgle": 10, "Flight": 10, "Sword": 10}[name]
    draw = _Q.draw_quest
    _Q.draw_quest = lambda row, rng: name
    try:
        st.generate_level()
    finally:
        _Q.draw_quest = draw
    st.room = st.rooms.start
    st.rooms.set_current(st.room)
    st.rooms.rooms[st.room].cleared = True
    if squad is not None:
        st.squad = squad
    return st


def _walk_in(st, rid, talk=True):
    """Arrive in `rid` the way a move does, without the food or the adjacency.
    A dialog that opens is answered with its first choice each time, unless
    `talk` is off, which leaves it on screen."""
    res = {}
    st._arrive(rid, res)
    if talk:
        while st.quest_dialog is not None and st.legal_actions():
            st.apply(st.legal_actions()[0])
    return res


def _sides(rooms, rid):
    r = rooms[rid]
    taken = {(x.row, x.col) for x in rooms.values()}
    return {d for d, (dr, dc) in _STEP.items() if (r.row + dr, r.col + dc) in taken}


# The level's quest is drawn from its shortlist, and every name on it comes up.
_drawn = collections.Counter(RunState.new(_STRICT, seed=s).tables and
                             _Q.draw_quest(next(r for r in _STRICT["Levels"] if r["ID"] == 2),
                                           random.Random(s)) for s in range(400))
check("level 2 draws each of its four quests",
      set(_drawn) == {"Sci", "Pit", "Cube", "Rat"} and min(_drawn.values()) > 70, f"{dict(_drawn)}")
check("levels without a shortlist have no quest",
      all(_Q.draw_quest(r, random.Random(0)) is None for r in _STRICT["Levels"]
          if r["ID"] not in (2, 6, 10)))

# Placement: the quest room's doors, the extra rooms' shape, the path from the
# start, and no portal on any of them.
_bad = []
for _name in ("Sci", "Pit", "Cube", "Rat", "Reaper", "SecretRoom", "Spider", "Bot",
              "Nurgle", "Flight", "Sword"):
    _spec = _Q.quest_spec(_STRICT, _name)
    for _s in range(12):
        _st = _quest_level(_name, seed=_s)
        _rm = _st.rooms
        _qr = _rm.quest_room
        _sides_q = _sides(_rm.rooms, _qr)
        if not (_sides_q <= set(_spec.get("Doors") or [])
                and set(_spec.get("GuaranteedDoors") or []) <= _sides_q):
            _bad.append((_name, _s, "doors", _sides_q))
        if len(_rm.quest_extras) != int(_spec.get("ExtraRooms") or 0):
            _bad.append((_name, _s, "extras"))
        _dq = _rm._distances(_qr)
        for _x in _rm.quest_extras:
            if _dq[_x] <= 1 or len(_rm.doors(_x)) not in (1, 2):
                _bad.append((_name, _s, "extra shape", _x))
            if _name == "Flight":
                _xr, _qq = _rm.rooms[_x], _rm.rooms[_qr]
                if not (_xr.col < _qq.col and abs(_xr.row - _qq.row) <= 3):
                    _bad.append((_name, _s, "flight extra", _x))
        if any(_rm.is_portal(r) for r in [_qr] + _rm.quest_extras):
            _bad.append((_name, _s, "portal"))
        # a way in from the start past neither the boss nor an extra room
        _block = {_rm.boss, *_rm.quest_extras}
        _seen_q, _todo = {_rm.start}, [_rm.start]
        _reach = False
        while _todo:
            _p = _todo.pop()
            for _n in _rm.doors(_p):
                if _n == _qr:
                    _reach = True
                elif _n not in _seen_q and _n not in _block:
                    _seen_q.add(_n)
                    _todo.append(_n)
        if not _reach:
            _bad.append((_name, _s, "path"))
check("every quest's rooms sit where ChooseQuestRoom and ChooseQuestExtraRoom put them",
      not _bad, f"{_bad[:4]}")

# The economy: quest rooms carry no weight, a quest room pays nothing, an extra
# room pays the plain share (nothing on Flight and Bot) and fights its own pack.
_qc = _quest_level("Cube", seed=3)
_plain = sum(1 for r in _qc.rooms.rooms.values()
             if r.kind not in ("start", "quest", "quest_extra"))
check("quest rooms are left out of CalculatePower's weights",
      _qc.room_mult("quest") == 0.0 and _qc.room_mult("quest_extra") == 0.0)
_gold_x = _qc.room_gold("quest_extra")
_want_x = (float(_qc.level_row["GoldPerRoom"]) * (len(_qc.rooms.rooms) - 1)
           * float(_qc.level_row["DefaultMult"]) / _qc.level_weight)
check("a quest room pays nothing and an extra room the plain share",
      _qc.room_gold("quest") == 0.0 and abs(_gold_x - _want_x) < 1e-6, f"{_gold_x:.2f}")
check("Flight's and Bot's extra rooms pay nothing",
      _quest_level("Flight", seed=1).room_gold("quest_extra") == 0.0
      and _quest_level("Bot", seed=1).room_gold("quest_extra") == 0.0)
_cx = _qc.rooms.rooms[_qc.rooms.quest_extras[0]]
_ens, _lay = _qc.quest_enemies(_cx)
check("the extra room fights its own pack in its own layout",
      len(_ens) == 6 and all(e.name == "Dalek" for e in _ens) and _lay is not None
      and _lay.id == "cube-quest-extra", f"{len(_ens)} {_lay.id if _lay else None}")
check("the quest room has nothing to fight",
      _qc.quest_enemies(_qc.rooms.rooms[_qc.rooms.quest_room]) == ([], None))

# The fog: walking past an extra room does not reveal it; accepting does.
_qr_room = _qc.rooms.rooms[_qc.rooms.quest_room]
for _n in _qc.rooms.doors(_cx.id):
    _qc.rooms.set_current(_n)
check("walking next to an extra room leaves it off the map", _cx.state == _UNKNOWN)
_qc.room = _qc.rooms.doors(_cx.id)[0]
_qc.rooms.rooms[_qc.room].cleared = True
check("and its door does not let the squad in (CheckDirection wants it off Unknown)",
      ("move", _cx.id) not in _qc.legal_actions())
_qc.rooms.set_current(_qc.rooms.start)
_qc.room = _qc.rooms.start
_walk_in(_qc, _qc.rooms.quest_room)
check("the Cube quest is accepted in its room and reveals the extra room",
      _qc.quest.status == _Q.ACCEPTED and _cx.state != _UNKNOWN and _qc.quest_dialog is None,
      f"status {_qc.quest.status}")

# Cube, through: the part on a won extra room, then the choice.
_strong = [_QH(item="broadsword", level=5) for _ in range(6)]
_qc.squad = [dataclasses.replace(h) for h in _strong]
_walk_in(_qc, _cx.id)
check("winning the extra room hands over the cube part",
      _cx.cleared and _qc.quest.status == _Q.ACQUIRED and "cube-part" in _qc.quest.items,
      f"status {_qc.quest.status}")
_g0 = _qc.gold
_walk_in(_qc, _qc.rooms.quest_room, talk=False)
_opts = _qc.legal_actions()
check("back with the part, the quest room asks: tokens or the secret",
      _opts == [("choose", 0), ("choose", 1)], f"{_opts}")
_qc.apply(("choose", 0))
check("tokens pay outcome 1, 25 gold, and finish the quest",
      _qc.gold == _g0 + 25 and _qc.quest.status == _Q.FINISHED
      and "cube-part" not in _qc.quest.items, f"{_qc.gold - _g0}")
_walk_in(_qc, _qc.rooms.quest_room)
check("a finished Cube quest says nothing more", _qc.quest_dialog is None)

# Pit: five Novices and nobody else jump and the run ends; with someone else
# they jump and the Pit pays mutation 306; tokens are 30 gold.
_qp = _quest_level("Pit", seed=2, squad=[_QH(item=None) for _ in range(5)])
_walk_in(_qp, _qp.rooms.quest_room)
check("five Novices alone jump into the Pit and the run is lost",
      _qp.finished and not _qp.won and not _qp.squad)
_qp = _quest_level("Pit", seed=2,
                   squad=[_QH(item=None) for _ in range(5)] + [_QH(item="broadsword")])
_walk_in(_qp, _qp.rooms.quest_room, talk=False)
_opts = _qp.legal_actions()
_qp.apply(("choose", 0))
check("five Novices with company: letting them go buys mutation 306",
      _opts == [("choose", 0), ("choose", 1)] and len(_qp.squad) == 1
      and any(m.get("ID") == 306 for m in _qp.mutations)
      and _qp.quest.status == _Q.FINISHED, f"{len(_qp.squad)} left")
_qp = _quest_level("Pit", seed=2, squad=[_QH(item="broadsword") for _ in range(3)])
_g0 = _qp.gold
_walk_in(_qp, _qp.rooms.quest_room, talk=False)
_qp.apply(("choose", 1))
check("without five Novices, tokens thrown come back as 30 gold",
      _qp.gold == _g0 + 30 and _qp.quest.status == _Q.FINISHED, f"{_qp.gold - _g0}")
_qp = _quest_level("Pit", seed=2, squad=[_QH(item="broadsword") for _ in range(3)])
_walk_in(_qp, _qp.rooms.quest_room, talk=False)
_qp.apply(("choose", 0))
_walk_in(_qp, _qp.rooms.quest_room, talk=False)
check("going to look for Novices leaves the Pit waiting",
      _qp.quest.status == _Q.ACCEPTED and _qp.legal_actions() == [("choose", 0), ("choose", 1)])

# Sci: the lab's fight turns the weakest human into a level-1 Zombie, and the
# scientist pays for it alive.
_qs = _quest_level("Sci", seed=4, squad=[dataclasses.replace(h) for h in _strong]
                   + [_QH(item=None, level=1)])
_walk_in(_qs, _qs.rooms.quest_room)
_qs_ok = _qs.quest.status == _Q.ACCEPTED
_weak = _qs.squad[-1]
_walk_in(_qs, _qs.quest.extras[0])
_z = [h for h in _qs.squad if h.cls == "Zombie"]
check("the Sci lab turns the weakest human into a level-1 Zombie",
      _qs_ok and _qs.quest.status == _Q.ACQUIRED and len(_z) == 1 and _z[0].level == 1
      and all(h is not _weak for h in _qs.squad) and len(_qs.squad) == 7,
      f"{[(h.item, h.cls) for h in _qs.squad]}")
_g0 = _qs.gold
_walk_in(_qs, _qs.rooms.quest_room, talk=False)
_qs.apply(("choose", 1))
check("the scientist takes the Zombie and pays 25 gold",
      _qs.gold == _g0 + 25 and not any(h.cls for h in _qs.squad)
      and _qs.quest.status == _Q.FINISHED)
_qs = _quest_level("Sci", seed=4, squad=[dataclasses.replace(h) for h in _strong])
_walk_in(_qs, _qs.quest.extras[0])
_walk_in(_qs, _qs.rooms.quest_room)
_walk_in(_qs, _qs.quest.extras[0])
check("a lab won before the quest was accepted never gives a Zombie",
      _qs.quest.status == _Q.ACCEPTED and not any(h.cls for h in _qs.squad))
_qs = _quest_level("Sci", seed=4, squad=[_QH(item="broadsword", level=5)])
_walk_in(_qs, _qs.rooms.quest_room)
_walk_in(_qs, _qs.quest.extras[0])
check("alone, the squad gets a new Zombie instead",
      len(_qs.squad) == 2 and _qs.squad[1].cls == "Zombie" and _qs.quest.status == _Q.ACQUIRED)
_qs.squad = [h for h in _qs.squad if not h.cls]
_g0 = _qs.gold
_walk_in(_qs, _qs.rooms.quest_room)
check("with the Zombie dead, the scientist pays nothing",
      _qs.gold == _g0 and _qs.quest.status == _Q.FINISHED and _qs.quest_dialog is None)

# Rat: the flute on accepting, the Rat on the lab's win, the reward for it alive.
_qr = _quest_level("Rat", seed=5, squad=[dataclasses.replace(h) for h in _strong])
_walk_in(_qr, _qr.rooms.quest_room)
check("accepting the Rat quest hands over the flute",
      _qr.quest.status == _Q.ACCEPTED and "rat-flute" in _qr.quest.items)
_walk_in(_qr, _qr.quest.extras[0])
check("winning the Rat's room adds the Rat to the team",
      _qr.quest.status == _Q.ACQUIRED and sum(1 for h in _qr.squad if h.cls == "Rat") == 1)
_g0 = _qr.gold
_walk_in(_qr, _qr.rooms.quest_room, talk=False)
_opts = _qr.legal_actions()
_qr.apply(("choose", 1))
check("the trumpeter takes the Rat back and pays 25 gold",
      _opts == [("choose", 0), ("choose", 1)] and _qr.gold == _g0 + 25
      and not any(h.cls for h in _qr.squad) and _qr.quest.status == _Q.FINISHED
      and "rat-flute" not in _qr.quest.items)

# Level 6. The walks below are over rooms marked won, so what is checked is the
# quest and not whether a squad survives level-6 fights on the way.
def _calm(st):
    for _r in st.rooms.rooms.values():
        _r.cleared = True
    return st


# Reaper: the hater runs from the Reaper's room and from every arrival until
# his five moves are spent, then talks.
_qa = _calm(_quest_level("Reaper", seed=1, squad=[dataclasses.replace(h) for h in _strong]))
_walk_in(_qa, _qa.rooms.quest_room)
_first = (_qa.quest.status, _qa.quest.moves, _qa.quest.hater_room)
_ran_ok = (_first[0] == _Q.ACCEPTED and _first[1] == 1 and _first[2] is not None
           and _first[2] in _qa.rooms.doors(_qa.rooms.quest_room))
for _ in range(10):
    if _qa.quest_dialog is not None or _qa.quest.hater_room is None:
        break
    _was = _qa.quest.hater_room
    _walk_in(_qa, _was, talk=False)
    if _qa.quest.hater_room not in (_was, None):
        _ran_ok = _ran_ok and _qa.quest.hater_room in _qa.rooms.doors(_was) \
            and _qa.quest.hater_room != _qa.prev_room and _qa.quest.hater_room != _qa.rooms.quest_room
check("the hater runs from the Reaper and from the squad, never back to either",
      _ran_ok, f"{_first}")
check("after five moves he stays and offers to pay",
      _qa.quest_dialog is not None and _qa.quest.moves <= 5
      and _qa.legal_actions() == [("choose", 0), ("choose", 1)], f"moves {_qa.quest.moves}")
_qa.apply(("choose", 1))
check("refusing his money is a fight the squad wins, and the quest is Acquired",
      _qa.quest.status == _Q.ACQUIRED and len(_qa.squad) == len(_strong))
_walk_in(_qa, _qa.rooms.quest_room)
check("the Reaper pays mutation 230 for him",
      _qa.quest.status == _Q.FINISHED and any(m.get("ID") == 230 for m in _qa.mutations))

# Spider: three leaflets, one per new room, then the monster pays.
_qd = _calm(_quest_level("Spider", seed=1, squad=[dataclasses.replace(h) for h in _strong]))
_walk_in(_qd, _qd.rooms.quest_room)
_three = [r for r in _qd.rooms.rooms if r not in (_qd.rooms.quest_room, _qd.rooms.start)][:3]
_walk_in(_qd, _three[0])
_walk_in(_qd, _three[0])
_one = (_qd.quest.status, list(_qd.quest.posted))
for _r in _three[1:]:
    _walk_in(_qd, _r)
check("a leaflet goes up in each new room, the third finishing the round",
      _one == (_Q.ACCEPTED, [_three[0]]) and _qd.quest.status == _Q.ACQUIRED, f"{_one}")
_g0 = _qd.gold
_walk_in(_qd, _qd.rooms.quest_room, talk=False)
_qd.apply(("choose", 1))
check("the monster pays 50 gold for the leaflets",
      _qd.gold == _g0 + 50 and _qd.quest.status == _Q.FINISHED)

# Bot: the pay option needs the money; paid, the bot joins; beaten, it joins too.
_qb = _calm(_quest_level("Bot", seed=1, squad=[dataclasses.replace(h) for h in _strong]))
_qb.gold = 20.0
_walk_in(_qb, _qb.rooms.quest_room)
_walk_in(_qb, _qb.quest.bot_room, talk=False)
_rich = _qb.legal_actions()
_qb.apply(("choose", 1))
check("with 5 gold the bot can be paid, and it joins the team",
      _rich == [("choose", 0), ("choose", 1), ("choose", 2)] and _qb.gold == 15.0
      and [h.cls for h in _qb.squad if h.cls] == ["Bot"] and _qb.quest.status == _Q.ACQUIRED)
_g0 = _qb.gold
_walk_in(_qb, _qb.rooms.quest_room, talk=False)
_qb.apply(("choose", 1))
check("the gamer takes the bot back and pays 50 gold",
      _qb.gold == _g0 + 50 and not any(h.cls for h in _qb.squad) and _qb.quest.status == _Q.FINISHED)
_qb = _calm(_quest_level("Bot", seed=2, squad=[dataclasses.replace(h) for h in _strong]))
_qb.gold = 0.0
_walk_in(_qb, _qb.rooms.quest_room)
_walk_in(_qb, _qb.quest.bot_room, talk=False)
_poor = _qb.legal_actions()
_qb.apply(("choose", 0))
for _ in range(8):
    if _qb.quest.bot_room is None or _qb.quest.status == _Q.ACQUIRED:
        break
    _walk_in(_qb, _qb.quest.bot_room, talk=False)
check("broke, the bot is threatened, runs, is caught, fought and joins",
      _poor == [("choose", 0), ("choose", 2)] and _qb.quest.status == _Q.ACQUIRED
      and [h.cls for h in _qb.squad if h.cls] == ["Bot"] and _qb.quest.moves <= 3,
      f"{_poor} status {_qb.quest.status} moves {_qb.quest.moves}")

# Level 10. Nurgle: a leaf per enemy killed while Accepted, five for the reward,
# and an unfinished quest goes on into the next level from its Overrides.
_qn = _calm(_quest_level("Nurgle", seed=1, squad=[dataclasses.replace(h) for h in _strong]))
_walk_in(_qn, _qn.rooms.quest_room)
_qn.quest.killed(_qn, 4)
_walk_in(_qn, _qn.rooms.quest_room, talk=False)
_short4 = _qn.quest_dialog is None and _qn.quest.status == _Q.ACCEPTED
_qn.quest.killed(_qn, 1)
_g0 = _qn.gold
_walk_in(_qn, _qn.rooms.quest_room, talk=False)
_qn.apply(("choose", 0))
check("Nurgle wants five dead enemies' leaves, and pays 35 gold for them",
      _short4 and _qn.gold == _g0 + 35 and _qn.quest.status == _Q.FINISHED
      and _qn.quest.leaves == 0)
_qn = _calm(_quest_level("Nurgle", seed=1, squad=[dataclasses.replace(h) for h in _strong]))
_walk_in(_qn, _qn.rooms.quest_room)
_qn_q = _qn.quest
_qn.next_level()
_calm(_qn)
check("an unfinished Nurgle quest carries into the next level, in its second room",
      _qn.level == 11 and _qn.quest is _qn_q and _qn.quest.room is not None
      and _qn.quest.spec["RoomParams"]["room"]["Layout"] == "nurgle-quest-2")
_qn.quest.killed(_qn, 5)
_walk_in(_qn, _qn.quest.room, talk=False)
_qn.apply(("choose", 0))
_qn.next_level()
check("and is paid there, and goes no further", _qn_q.status == _Q.FINISHED and _qn.quest is None)

# Flight: on foot, the extra room is revealed and its dialog plays there; by
# cannon the squad lands in it with 10% of its health gone.
_qf = _calm(_quest_level("Flight", seed=1, squad=[dataclasses.replace(h) for h in _strong]))
_walk_in(_qf, _qf.rooms.quest_room, talk=False)
_qf.apply(("choose", 1))
check("the cannon lands the squad in the extra room, 10% lighter, holding the helm",
      _qf.room == _qf.quest.extras[0] and _qf.quest.status == _Q.ACQUIRED
      and all(abs(h.hp - 0.9) < 1e-9 for h in _qf.squad) and "mik-helmet" in _qf.quest.items)
_g0 = _qf.gold
_walk_in(_qf, _qf.rooms.quest_room, talk=False)
_qf.apply(("choose", 0))
check("Mike's friend pays 35 gold for the helm", _qf.gold == _g0 + 35 and _qf.quest.status == _Q.FINISHED)
_qf = _calm(_quest_level("Flight", seed=2, squad=[dataclasses.replace(h) for h in _strong]))
_walk_in(_qf, _qf.rooms.quest_room, talk=False)
_qf.apply(("choose", 0))
_xs = _qf.rooms.rooms[_qf.quest.extras[0]].state
_walk_in(_qf, _qf.quest.extras[0])
check("on foot, the extra room is marked, and reaching it finds the helm unhurt",
      _xs != _UNKNOWN and _qf.quest.status == _Q.ACQUIRED and all(h.hp == 1.0 for h in _qf.squad))

# Sword: taken, it joins; the twin joins the boss; after the boss, the reward;
# too late, nothing.
_qw = _calm(_quest_level("Sword", seed=1, squad=[dataclasses.replace(h) for h in _strong]))
_qw.rooms.rooms[_qw.rooms.boss].cleared = False
_walk_in(_qw, _qw.rooms.quest_room, talk=False)
_qw.apply(("choose", 1))
_boss = _qw.rooms.rooms[_qw.rooms.boss]
check("the sword joins the team, and its twin will stand with the boss",
      [h.cls for h in _qw.squad if h.cls] == ["TalkingSword"]
      and _qw.quest.extra_enemies(_qw, _boss) == [("DeathSword", 1)])
_qw.room = _qw.rooms.boss
_stem = _qw.quest.arrive(_qw, _qw.rooms.boss, True, True)
_qw.open_quest_dialog(_stem)
_g0 = _qw.gold
_qw.apply(("choose", 0))
check("after the boss falls the sword pays 35 gold and leaves",
      _qw.gold == _g0 + 35 and _qw.quest.status == _Q.FINISHED and not any(h.cls for h in _qw.squad))
_qw = _calm(_quest_level("Sword", seed=1, squad=[dataclasses.replace(h) for h in _strong]))
_walk_in(_qw, _qw.rooms.quest_room, talk=False)
check("met after the boss is dead, the sword has nothing to give",
      _qw.quest.status == _Q.FINISHED and _qw.quest_dialog is None and not any(h.cls for h in _qw.squad))

# The secret room. On level 7 it touches no room, is a portal, and two other
# portals switch it on; on the SecretRoom quest's level a sacrifice opens it.
_ls = RunState.new(_STRICT, seed=3)
_ls.level = 7
_ls.generate_level()
_ls.room = _ls.rooms.start
_ls.rooms.set_current(_ls.room)
_calm(_ls)
_sid = _ls.rooms.secret_rooms[0] if _ls.rooms.secret_rooms else None
check("a level-7 map has one secret room, a portal with no door",
      _sid is not None and _ls.rooms.doors(_sid) == [] and _ls.rooms.is_portal(_sid)
      and _ls.rooms.rooms[_sid].kind == "secret" and len(_ls.rooms.activators) == 2,
      f"{_sid} {_ls.rooms.activators}")
_a1, _a2 = _ls.rooms.activators
_walk_in(_ls, _a1)
_half = _ls.rooms.rooms[_sid].state
_walk_in(_ls, _a2)
check("standing in both activators puts it on the map, where a teleport reaches it",
      _half == _UNKNOWN and _ls.rooms.rooms[_sid].state != _UNKNOWN
      and _sid in _ls.rooms.teleports(_a2))
_g0 = _ls.gold
_walk_in(_ls, _sid)
check("the secret room fights nothing and pays nothing",
      _ls.gold == _g0 and _ls.rooms.rooms[_sid].cleared and not _ls.finished)
check("secret rooms are left out of the level's room count",
      _ls.num_rooms == len(_ls.rooms.rooms) - 1)

_qx = _calm(_quest_level("SecretRoom", seed=4,
                         squad=[_QH(item="broadsword", level=5) for _ in range(10)]))
_walk_in(_qx, _qx.rooms.quest_room)
_init = _qx.quest.status
_qx.apply(("sacrifice", None))
check("a sacrifice in the mysterious room is a trip to the Moon and back, quest in hand",
      _init == _Q.INITIALIZED and _qx.quest.status == _Q.ACCEPTED
      and _qx.room == _qx.rooms.quest_room
      and _qx.rooms.rooms[_qx.quest.extras[0]].state != _UNKNOWN,
      f"status {_qx.quest.status}")
_qx.rooms.rooms[_qx.quest.extras[0]].cleared = False
_walk_in(_qx, _qx.quest.extras[0])
check("the maze's fight hands over the book",
      _qx.quest.status == _Q.ACQUIRED and "comic-book" in _qx.quest.items,
      f"status {_qx.quest.status}, squad {len(_qx.squad)}")
_walk_in(_qx, _qx.rooms.quest_room)
_g0 = _qx.gold
_qx.apply(("sacrifice", None))
_opts = _qx.legal_actions()
_qx.apply(("choose", 0))
check("a second sacrifice takes the book to the Doctor, who pays 35 gold",
      _opts == [("choose", 0), ("choose", 1)] and _qx.gold == _g0 + 35
      and _qx.quest.status == _Q.FINISHED and _qx.room == _qx.rooms.quest_room,
      f"{_opts}")

# A companion is no human: no item goes to it, and it fights as its own row.
_qz = _quest_level("Sci", seed=4, squad=[_QH(item=None, cls="Zombie")])
check("a companion takes no item", _qz.best_item_target("broadsword") is None)
check("a companion fights as its class row",
      _qz.specs()[0].name == "Zombie" and _qz.specs()[0].health > 0)

# The quests switch: off generates the same levels with no quest at all.
_qo = RunState.new(_STRICT, seed=7, quests=False)
_qo.next_level()
check("quests off leaves level 2 without a quest or its rooms",
      _qo.quest is None and not any(r.kind in ("quest", "quest_extra")
                                    for r in _qo.rooms.rooms.values()))

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

# Inside a quality the draw is `DynamicWeightedList` with factor 2: every item
# starts at weight 1, and each pick halves an item's weight against the
# least-picked one, so over a run the shop cycles through a quality's items.
_st6 = RunState.new(_STRICT, seed=13)
_q1 = _by_quality(_STRICT)[1]
_st6.item_picks = {1: {_q1[0]: 2, _q1[1]: 1}}
_w = _st6.item_weights(1, _q1)
check("an item picked k times more than the least-picked weighs 1 / 2^k",
      _w[0] == 0.25 and _w[1] == 0.5 and all(x == 1.0 for x in _w[2:]), f"{_w[:3]}")
_st6.item_picks, _st6.shop_level = {}, 1
for _ in range(300):
    _st6.roll_item()
_spread = {q: max(c.get(n, 0) for n in _by_quality(_STRICT)[q])
           - min(c.get(n, 0) for n in _by_quality(_STRICT)[q])
           for q, c in _st6.item_picks.items()}
check("every roll is a pick, and 300 of them leave a quality's counts within 6 "
      "(a uniform draw spreads them about 15 apart)",
      sum(sum(c.values()) for c in _st6.item_picks.values()) == 300
      and max(_spread.values()) <= 6, f"{_spread}")

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
_shop7.cleared = True             # `AfterFight` is what opens a shop
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
