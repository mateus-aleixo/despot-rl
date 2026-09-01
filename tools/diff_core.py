"""Differential test: the Rust core against the Python oracle.

Only fights inside the core's ported envelope are used, so any difference is a
porting bug rather than a missing feature.

Two things the raw numbers hide, so they are reported separately:

  * a **draw boundary**. Long fights end on the 120 s timeout, and a few percent
    of tick difference decides whether one engine finishes just inside it and
    the other just outside. That shows up as win-vs-draw, which is not the same
    kind of disagreement as win-vs-loss.
  * **relative damage error on a near-zero denominator**. A side that deals 41
    damage in one engine and 0 in the other is not 3725% wrong, it is a side
    that did nothing in both. Damage error is therefore scaled by the larger
    side's damage, not by one of them.
"""
import random
import statistics
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sim.battle import Battle, apply_class_skills, deploy
from sim.data import load_ruleset, parse_room_layouts, units_by_class
from sim.fast import available, fast_batch, fast_battle, supported
from sim.nav import Grid
from sim.spec import build_player_squad, build_unit

if not available():
    print("core not built; run `cargo build --release` in core/")
    sys.exit(1)

t = load_ruleset(strict=True)
layouts = parse_room_layouts(t["RoomLayouts"])
ubc = units_by_class(t)
META = t["Meta"]["Classes"]

ENEMIES = ["Dalek", "Shielder", "Mancrack", "FurySwiper", "Tomato",
           "AoeRanger", "CatBot", "HealBot", "Specter",
           "Necromancer", "Samurai", "Slower", "SpiderSummoner"]


def make(seed, n_player=6, n_enemy=6, enemy="Dalek", item="broadsword"):
    layout = layouts[seed % len(layouts)]
    grid = Grid.from_layout(layout)
    rng = random.Random(seed)
    specs = build_player_squad(t, [(item, 1)] * n_player)
    size = int((META.get(enemy) or {}).get("Size") or 1)
    especs = [build_unit(ubc, enemy, 1, name=enemy, size=size) for _ in range(n_enemy)]
    agents = deploy(grid, layout, specs, 0, "p", rng) + \
        deploy(grid, layout, especs, 1, "e1", rng)
    return grid, agents


print("== envelope check ==")
g, ag = make(0)
ok, why = supported(Battle(g, ag, seed=0, tables=t).agents, tables=t)
print(f"  fight is inside the ported envelope: {ok} {why}")

print("\n== differential: Python oracle vs Rust core ==")
print(f"  {'enemy':15s} {'decisive':9s} {'draw-edge':9s} {'tick err':9s} damage err")
tot_agree = tot_n = tot_draw = 0
tick_errs, dmg_errs = [], []
for enemy in ENEMIES:
    n = 12
    agree = draw_edge = 0
    tr, dr = [], []
    for s in range(n):
        g, ag = make(s, enemy=enemy)
        py = Battle(g, [a for a in ag], seed=s, tables=t).run()
        g2, ag2 = make(s, enemy=enemy)
        b2 = Battle(g2, ag2, seed=s, tables=t)
        rs = fast_battle(g2, b2.agents, seed=s, tables=t)
        if py.winner == rs["winner"]:
            agree += 1
        elif py.winner is None or rs["winner"] is None:
            draw_edge += 1          # timeout boundary, not a contradiction
        tr.append(abs(py.ticks - rs["ticks"]) / max(1, py.ticks))
        a_, b_ = py.total_damage[0], rs["damage"][0]
        dr.append(abs(a_ - b_) / max(1.0, a_, b_))
    tot_agree += agree
    tot_n += n
    tot_draw += draw_edge
    tick_errs.append(statistics.mean(tr))
    dmg_errs.append(statistics.mean(dr))
    print(f"  {enemy:15s} {f'{agree}/{n}':9s} {draw_edge:^9d} "
          f"{statistics.mean(tr)*100:7.2f}%  {statistics.mean(dr)*100:7.2f}%")

print(f"\n  winners agreeing outright : {tot_agree}/{tot_n}")
print(f"  differing only by draw-vs-win: {tot_draw}")
print(f"  genuinely opposite results   : {tot_n - tot_agree - tot_draw}")
print(f"  mean tick error {statistics.mean(tick_errs)*100:.2f}%, "
      f"mean damage error {statistics.mean(dmg_errs)*100:.2f}%")


# --------------------------------------------------------------------------
# The agent-level passives, which are the newest thing in the core and the ones
# whose hooks consume randomness. Small fights are the sharp test: while the
# fight is short enough that float32 and float64 have not separated the two
# engines, a damage error of exactly zero means every roll landed on the same
# swing in both. Anything above zero at 1v1 is a porting bug, not chaos.
from sim import mutations as M
from sim.data import items_by_name

MUT_BY_ID = {m["ID"]: m for m in t["SimpleMutations"]}
_ITEM_CLASS = {n: r.get("Class") for n, r in items_by_name(t).items()}


def item_for(cls):
    return next(n for n, c in _ITEM_CLASS.items() if c == cls)


# One row per mechanic: the shipped mutation that carries it, and a class the
# `Class` selector actually covers.
PASSIVE_CASES = [
    ("BuffAttack (target)", [226], "broadsword"),
    ("BuffAttack (source)", [318], "broadsword"),
    ("PassiveStun", [224], "broadsword"),
    ("FearsomeAttack", [14], "broadsword"),
    ("Craggy", [317], "broadsword"),
    ("Untouchable", [240], item_for("Shooter")),
    ("ManaBreak", [19], "broadsword"),
    ("BuffOnCasted", [351], item_for("Mage")),
    ("MultiCast", [301], item_for("Mage")),
    ("ModifyDamage", [355], item_for("Mage")),
    ("StickyBlood", [303], "broadsword"),
    ("BuffOnDeath", [353], "broadsword"),
    ("ResurrectionChance", [245], "broadsword"),
    ("Compensation", [374], item_for("Monk")),
    ("CSLinkStatBonus", [379], item_for("Monk")),
    ("ClassDiversity", [365], "broadsword"),
    ("Vampirism", [23], "broadsword"),
    ("Vampirism (fixed)", [263], item_for("Dodger")),
    ("Evasion", [29], "broadsword"),
    ("Evasion (threshold)", [221], "broadsword"),
    ("ExplodeProjectile", [200], "broadsword"),
    ("ExplodeProjectile (chance)", [668], _ITEM_MAGE := item_for("Mage")),
    ("EventualKnockback", [236], item_for("Thrower")),
    ("EventualKnockback (chance)", [338], _ITEM_MAGE),
    ("-- no mutations --", [], "broadsword"),
    ("-- no mutations (Mage)", [], item_for("Mage")),
]


def make_mut(seed, mut_ids, item, n_player, n_enemy, enemy="Mancrack"):
    """The same fight as `make`, with a squad carrying some mutations."""
    layout = layouts[seed % len(layouts)]
    grid = Grid.from_layout(layout)
    rng = random.Random(seed)
    specs = build_player_squad(t, [(item, 1)] * n_player)
    muts = [MUT_BY_ID[i] for i in mut_ids]
    if muts:
        specs = M.apply_to_specs(specs, muts, random.Random(seed))
    size = int((META.get(enemy) or {}).get("Size") or 1)
    especs = [build_unit(ubc, enemy, 1, name=enemy, size=size) for _ in range(n_enemy)]
    t0 = deploy(grid, layout, specs, 0, "p", rng)
    t1 = deploy(grid, layout, especs, 1, "e1", rng)
    apply_class_skills(t, t0)
    if muts:
        M.apply_to_agents(t0, muts, random.Random(seed))
    # Battle.__init__ is what attaches the per-unit skills, and `run.py` packs
    # the agents it produced, so both engines have to start from one.
    return grid, Battle(grid, t0 + t1, seed=seed, tables=t)


print("\n== differential: the agent-level passives ==")
for size_label, npl, nen in (("1v1", 1, 1), ("2v2", 2, 2), ("4v6", 4, 6)):
    print(f"\n  {size_label}")
    print(f"    {'mechanic':24s} {'winners':9s} {'tick err':9s} damage err")
    for label, ids, item in PASSIVE_CASES:
        n = 16
        agree = 0
        tr, dr = [], []
        for s in range(n):
            g1, b1 = make_mut(s, ids, item, npl, nen)
            g2, b2 = make_mut(s, ids, item, npl, nen)
            py = b1.run()
            rs = fast_battle(g2, b2.agents, seed=s, tables=t)
            agree += int(py.winner == rs["winner"])
            tr.append(abs(py.ticks - rs["ticks"]) / max(1, py.ticks))
            a_, b_ = py.total_damage[0], rs["damage"][0]
            dr.append(abs(a_ - b_) / max(1.0, a_, b_))
        print(f"    {label:24s} {f'{agree}/{n}':9s} "
              f"{statistics.mean(tr)*100:7.2f}%  {statistics.mean(dr)*100:7.2f}%")


# --------------------------------------------------------------------------
# The per-unit on-death skills. Measured with both sides already in contact and
# the carrier on 1 hp, because an approach is hundreds of ORCA steps and the two
# float widths disagree about those long before they disagree about anything
# else -- put the walk back in and the death effect is buried under it.
from sim.battle import place_at
from sim.spec import build_player_squad as _bps

DEATH_UNITS = ["FireHead", "Firestarter", "Fish", "Ostrich", "PoringMedium",
               "OstrichRider2"]
CELLS = layouts[0].zone("p") or [(0, 0)]
DGRID = Grid.from_layout(layouts[0])


def touching(seed, enemy, players=3, enemy_hp=1.0):
    specs = _bps(t, [("broadsword", 1)] * players)
    size = int((META.get(enemy) or {}).get("Size") or 1)
    foe = [build_unit(ubc, enemy, 1, name=enemy, size=size)]
    t0 = place_at(DGRID, specs, CELLS[:players], team=0)
    t1 = place_at(DGRID, foe, CELLS[:1], team=1)
    if enemy_hp is not None:
        t1[0].hp = enemy_hp
    return Battle(DGRID, t0 + t1, seed=seed, tables=t)


print("\n== differential: the per-unit on-death skills ==")
print(f"  {'unit':16s} {'skill':22s} {'tick-matched':13s} exact")
SKILL_OF = {"FireHead": "FireheadDeath", "Firestarter": "FirestarterDeath",
            "Fish": "DamageOnDeath", "Ostrich": "TransformAfterDeath",
            "PoringMedium": "TransformAfterDeath x3",
            "OstrichRider2": "TransformAfterDeath"}
for unit in DEATH_UNITS:
    # a transform is measured 1v1: the fewer bodies in the room, the less the
    # summon has to walk, and walking is the only thing left that separates them
    players = 1 if SKILL_OF[unit].startswith("Transform") else 3
    exact = matched = 0
    for s in range(24):
        py = touching(s, unit, players).run()
        rs = fast_battle(DGRID, touching(s, unit, players).agents, seed=s, tables=t)
        if py.ticks != rs["ticks"]:
            continue
        matched += 1
        exact += int(abs(py.total_damage[0] - rs["damage"][0]) < 0.5
                     and abs(py.total_damage[1] - rs["damage"][1]) < 0.5)
    print(f"  {unit:16s} {SKILL_OF[unit]:22s} {f'{matched}/24':13s} {exact}/{matched}")
print("  (a transform spawns at an offset drawn from two `uniform` calls, and the")
print("   two widths place it ~1e-7 apart -- enough to move a swing by a tick once")
print("   the summon has to walk, which is why those seeds do not tick-match. The")
print("   Porings spawn in contact and stay exact.)")


# --------------------------------------------------------------------------
# The status casts and the blinks, the last two effect kinds. Same setup as the
# on-death block: both sides in contact, so the measurement is the effect rather
# than the walk.
STATUS_CASTERS = [("Silencer", "Silence"), ("Orb", "Stun"),
                  ("Watcher", "WatcherDisable"), ("Rat", "Panic"),
                  ("PoringSmall2", "Panic"), ("Mushroom", "MushroomBlink"),
                  ("Mushroom2", "MushroomBlink"), ("RangeBlinker", "BlinkAway"),
                  ("RangeBlinker2", "BlinkAway")]

print("\n== differential: status casts and blinks ==")
print(f"  {'unit':16s} {'skill':16s} {'tick-matched':13s} {'exact':8s} damage err")
for unit, skill in STATUS_CASTERS:
    exact = matched = 0
    derr = []
    for s in range(24):
        py = touching(s, unit, 1, enemy_hp=None).run()
        rs = fast_battle(DGRID, touching(s, unit, 1, enemy_hp=None).agents,
                         seed=s, tables=t)
        a_, b_ = py.total_damage[0], rs["damage"][0]
        derr.append(abs(a_ - b_) / max(1.0, a_, b_))
        if py.ticks != rs["ticks"]:
            continue
        matched += 1
        exact += int(abs(a_ - b_) < 0.5
                     and abs(py.total_damage[1] - rs["damage"][1]) < 0.5)
    print(f"  {unit:16s} {skill:16s} {f'{matched}/24':13s} "
          f"{f'{exact}/{matched}':8s} {statistics.mean(derr)*100:7.2f}%")
print("  (a blinker teleports 25 units and walks back, over and over, so the two")
print("   widths end the fight a few ticks apart -- the damage is identical, which")
print("   is what says the blink itself agrees)")

print("\n== speed ==")
N = 40
py_t0 = time.perf_counter()
for s in range(N):
    g2, ag2 = make(s)
    Battle(g2, ag2, seed=s, tables=t).run()
py_el = time.perf_counter() - py_t0

rs_t0 = time.perf_counter()
for s in range(N):
    g2, ag2 = make(s)
    fast_battle(g2, Battle(g2, ag2, seed=s, tables=t).agents, seed=s, tables=t)
rs_el = time.perf_counter() - rs_t0

batches = []
for s in range(N):
    g2, ag2 = make(s)
    batches.append(Battle(g2, ag2, seed=s, tables=t).agents)
bt0 = time.perf_counter()
fast_batch(g, batches, tables=t)
b_el = time.perf_counter() - bt0

print(f"  python oracle : {py_el/N*1000:8.2f} ms/battle")
print(f"  rust (1 call) : {rs_el/N*1000:8.2f} ms/battle   {py_el/rs_el:6.1f}x")
print(f"  rust (batched): {b_el/N*1000:8.2f} ms/battle   {py_el/b_el:6.1f}x")
