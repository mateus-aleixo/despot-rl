"""Checks on the RL environment and the placement policies."""
import random, sys
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from rl.env import DespotRunEnv, NON_MOVE_ACTIONS
from rl.placement import POLICIES, frontline_placement, zone_cells
from sim.data import load_ruleset, parse_room_layouts
from sim.nav import Grid
from sim.spec import build_player_squad

failures = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond: failures.append(name)

TABLES = load_ruleset(strict=True)

print("== env shape ==")
env = DespotRunEnv(tables=TABLES)
obs, info = env.reset(seed=0)
check("observation matches declared dim", obs.shape == (env.obs_dim,), f"{obs.shape}")
check("observation is finite", np.isfinite(obs).all())
check("action space covers the moves plus the rest",
      env.n_actions == env.n_moves + len(NON_MOVE_ACTIONS), f"{env.n_actions}")
# A move is a direction now, so the same index has to mean the same direction
# wherever the squad stands, and a direction with no room behind it has to be
# masked out rather than decoded to a neighbour.
_targets = env._move_targets(env.state)
check("every legal move decodes to an adjacent room",
      all(t in env.state.rooms.neighbours(env.state.room)
          for t in _targets if t is not None), f"{_targets}")
check("a direction with no room is masked",
      all(not env.action_mask()[i] for i, t in enumerate(_targets) if t is None),
      f"{_targets}")
check("mask starts non-empty", info["action_mask"].any(),
      f"{int(info['action_mask'].sum())} legal")

print("\n== masking ==")
mask = env.action_mask()
legal = {(k, a) for k, a in env.state.legal_actions()}
decoded = {env._decode(i) for i in np.flatnonzero(mask)}
check("mask matches legal_actions exactly", decoded == legal,
      f"{len(decoded)} vs {len(legal)}")

illegal = int(np.flatnonzero(~mask)[0])
before = (env.state.level, env.state.room, env.state.gold)
o, r, term, trunc, inf = env.step(illegal)
after = (env.state.level, env.state.room, env.state.gold)
check("an illegal action does not change state", before == after)
check("an illegal action is penalised, not fatal", r < 0 and not term, f"reward {r}")

# `action_mask` memoises, invalidating on `reset` and on the `apply` inside
# `step`. The window that could go stale is the one the trainer actually uses:
# the mask carried from the previous step's info dict, after `_encode` has run
# and called `ensure_stock`. So check it there rather than in isolation, and
# check that a fresh scan agrees with what the cache hands back.
_stale = _cached_checked = 0
for _seed in range(4):
    _e = DespotRunEnv(tables=TABLES, seed=_seed, placement_policy=POLICIES["frontline"],
                      fast_core=True)
    _o, _i = _e.reset(seed=_seed)
    _m, _n = _i["action_mask"], 0
    _rng = random.Random(_seed)
    while _m.any() and _n < _e.max_steps:
        _held = _e.action_mask()
        _e._mask_cache = None
        _cached_checked += 1
        if not np.array_equal(_held, _e.action_mask()) or not np.array_equal(_m, _held):
            _stale += 1
        _o, _r, _t, _tr, _i = _e.step(_rng.choice(list(np.flatnonzero(_m))))
        _m, _n = _i["action_mask"], _n + 1
        if _t or _tr:
            break
check("the memoised mask always equals a fresh scan", _stale == 0,
      f"{_cached_checked} masks over 4 runs")

print("\n== teleports ==")
# A teleport slot names a portal that is on the map, is legal only while the
# squad stands on a portal, and keeps naming the same room as more of the level
# comes into view: the slots follow the order portals were revealed in, not
# their names, which would say how many unseen portals sort before a seen one.
from sim.run import UNEXPLORED, UNKNOWN
from rl.env import MOVES, TELEPORT_SLOTS

_te = DespotRunEnv(tables=TABLES, dialogs=False)
for _seed in range(200):
    _te.reset(seed=_seed)
    # level 1 is 7 rooms and always holds exactly two portals; level 2 holds
    # three nearly every time
    _te.state.next_level()
    _te._mask_cache = None
    _m = _te.state.rooms
    _hidden = [p for p in _m.portals
               if _m.rooms[p].state == UNKNOWN and p != _m.start]
    if len(_m.portals) >= 3 and len(_hidden) >= 3:
        break
_tm = _te.state.rooms
_here = _te.state.room
_on = _tm.is_portal(_here)
_mask = _te.action_mask()
check("off a portal every teleport slot is masked",
      _on or not _mask[len(MOVES):len(MOVES) + TELEPORT_SLOTS].any())
# stand on a portal and bring the others onto the map one at a time
_src = next(p for p in _tm.portals if p != _tm.start)
_te.state.room = _src
_tm.set_current(_src)
_tm.rooms[_src].cleared = True
_te._mask_cache = None
_order = _tm.revealed_portals()
_slots_seen = []
for _p in sorted((p for p in _tm.portals if _tm.rooms[p].state == UNKNOWN),
                 reverse=True):
    _tm.rooms[_p].state = UNEXPLORED
    _tm._refresh_known()
    _te._mask_cache = None
    _targets = _te._move_targets(_te.state)[len(MOVES):]
    _slots_seen.append(list(_targets))
_final = _te._move_targets(_te.state)[len(MOVES):]
_mask = _te.action_mask()
check("each slot keeps its room as more portals are revealed",
      len(_slots_seen) >= 2
      and all(s[i] in (None, _final[i]) for s in _slots_seen for i in range(len(s))),
      f"{_slots_seen} -> {_final}")
check("on a portal, every other revealed portal is a legal teleport slot",
      sorted(t for t in _final if t is not None)
      == sorted(p for p in _tm.portals if p != _src)
      and all(_mask[len(MOVES) + i] == (t is not None) for i, t in enumerate(_final)),
      f"{_final}")
check("a teleport slot decodes to a teleport",
      all(_te._decode(len(MOVES) + i) == ("teleport", t)
          for i, t in enumerate(_final) if t is not None))

print("\n== no legal action is a no-op ==")
# A legal action that changes nothing is free real estate for a greedy policy.
# One trained agent looped on `feed` at hunger 0 for a whole 400-step episode
# instead of playing, scoring exactly 400 x -0.02 = -8.00.
import copy

from sim.run import RunState


def snapshot(st):
    return (st.level, st.room, st.gold, st.food.amount, st.food.hunger_level,
            st.food.moves_left, len(st.squad), len(st.mutations),
            st.shop_level, tuple(st.offer), st.level_won,
            (st.dialog.name, st.dialog.path, st.dialog.steps) if st.dialog else None,
            tuple((c.name, c.charges) for c in st.consumables),
            tuple(sorted((k, str(v)) for k, v in st.armed.items())),
            tuple((h.item, h.level, round(h.experience, 6), round(h.hp, 6))
                  for h in st.squad),
            tuple(sorted((r.id, r.cleared, r.stock, tuple(r.food_stock or ()),
                          tuple(m["ID"] if m else None for m in r.offer_mutations or ()),
                          r.takes_left, r.rolls_left, len(r.corpses),
                          tuple(c["ID"] if c else None for c in r.consumable_offer or ()))
                         for r in st.rooms.rooms.values())))


noops = []
for s in range(5):
    st = RunState.new(TABLES, seed=s)
    for _ in range(30):
        acts = st.legal_actions()
        if not acts:
            break
        for a in acts:
            probe = copy.deepcopy(st)
            before = snapshot(probe)
            probe.apply(a)
            if snapshot(probe) == before:
                noops.append(a[0])
        st.apply(random.choice(acts))
check("every legal action changes something", not noops,
      f"no-ops found: {sorted(set(noops))}")

print("\n== episodes terminate ==")
# Seeded, because this used to draw from numpy's global RNG and so passed or
# failed depending on what had run before it.
from rl.heuristic import heuristic as _heuristic

_pick = random.Random(0)
lens, levels = [], []
for s in range(6):
    obs, info = env.reset(seed=100 + s)
    m, n = info["action_mask"], 0
    while m.any() and n < env.max_steps:
        a = _pick.choice(list(np.flatnonzero(m)))
        obs, r, term, trunc, info = env.step(int(a))
        m = info["action_mask"]; n += 1
        if term or trunc: break
    lens.append(n); levels.append(env.state.level)
check("every episode ends within max_steps", all(n <= env.max_steps for n in lens),
      f"lengths {lens}")

# Progress is measured with a policy that plays, not with a random one. The run
# starts with five unarmed humans against an 800-Power room, so uniform-random
# play mostly wipes on level 1 and "max level > 1" was a coin flip.
levels = []
for s in range(6):
    obs, info = env.reset(seed=200 + s)
    m, n = info["action_mask"], 0
    while m.any() and n < env.max_steps:
        obs, r, term, trunc, info = env.step(_heuristic(env, obs, m))
        m = info["action_mask"]; n += 1
        if term or trunc: break
    levels.append(env.state.level)
check("a policy that plays makes progress", max(levels) > 1, f"levels {levels}")

print("\n== determinism ==")
def rollout(seed):
    e = DespotRunEnv(tables=TABLES)
    o, i = e.reset(seed=seed)
    rng = random.Random(0)
    out = []
    m = i["action_mask"]
    for _ in range(25):
        if not m.any(): break
        a = rng.choice(list(np.flatnonzero(m)))
        o, r, term, trunc, i = e.step(int(a))
        out.append(round(float(r), 6))
        m = i["action_mask"]
        if term or trunc: break
    return out
check("same seed gives the same trajectory", rollout(7) == rollout(7))

print("\n== the to_boss fork's arms ==")
# Each arm differs from the shipped observation in information only: the same
# run, step for step, with its own entries held at zero (or, for `lights_on`,
# the whole level shown) and nothing else moved, the mask included.
def paired_obs(seed, steps=40, **arm):
    """The shipped env and an arm, driven by the same actions."""
    base, other = DespotRunEnv(tables=TABLES), DespotRunEnv(tables=TABLES, **arm)
    o1, i1 = base.reset(seed=seed)
    o2, i2 = other.reset(seed=seed)
    rng = random.Random(seed)
    pairs = [(o1, o2, i1["action_mask"], i2["action_mask"])]
    for _ in range(steps):
        m = i1["action_mask"]
        if not m.any():
            break
        a = int(rng.choice(list(np.flatnonzero(m))))
        o1, _, t1, _, i1 = base.step(a)
        o2, _, t2, _, i2 = other.step(a)
        pairs.append((o1, o2, i1["action_mask"], i2["action_mask"]))
        if t1 or t2:
            break
    return pairs, base, other

for arm in ({"blind_boss": True}, {"blind_frontier": True}, {"lights_on": True}):
    name = next(iter(arm))
    moved, masks_same, zeroed = set(), True, True
    for seed in (3, 11, 29):
        pairs, base, other = paired_obs(seed, **arm)
        masks_same &= all((m1 == m2).all() for _, _, m1, m2 in pairs)
        for o1, o2, _, _ in pairs:
            idx = set(np.flatnonzero(o1 != o2).tolist())
            moved |= idx
            if name != "lights_on":
                zeroed &= all(o2[i] == 0.0 for i in idx)
        same_run = (base.state.level == other.state.level
                    and base.state.room == other.state.room
                    and len(base.state.squad) == len(other.state.squad))
        masks_same &= same_run
    check(f"{name}: the same run and the same mask, step for step", masks_same)
    if name == "lights_on":
        check("lights_on: shows more than the fog does", len(moved) > 0,
              f"{len(moved)} entries")
    else:
        check(f"{name}: what moves is only zeroed, and something does", zeroed and moved,
              f"{sorted(moved)[:12]}")
_e = DespotRunEnv(tables=TABLES, lights_on=True)
_e.reset(seed=5)
from sim.run import UNKNOWN as _UNKNOWN
_hidden = sum(1 for r in _e.state.rooms.rooms.values() if r.state == _UNKNOWN)
check("lights_on lights the view and leaves the run in the fog", _hidden > 0,
      f"{_hidden} rooms still unknown to the run")

print("\n== placement policies ==")
layout = parse_room_layouts(TABLES["RoomLayouts"])[0]
grid = Grid.from_layout(layout)
specs = build_player_squad(TABLES, [("broadsword", 1), ("gun", 1), ("broadsword", 1)])
zone = set(zone_cells(layout))
enemy_probe = []
for name, pol in POLICIES.items():
    cells = pol(grid, layout, specs, random.Random(0), enemy_probe)
    check(f"{name}: one cell per unit", len(cells) == len(specs), f"{len(cells)}")
    check(f"{name}: all cells inside the player zone", all(c in zone for c in cells))

cells = frontline_placement(grid, layout, specs, random.Random(0), enemy_probe)
melee_cols = [c for (r, c), s in zip(cells, specs) if s.melee]
ranged_cols = [c for (r, c), s in zip(cells, specs) if not s.melee]
check("frontline puts melee ahead of ranged",
      min(melee_cols) > max(ranged_cols), f"melee {melee_cols} ranged {ranged_cols}")

print("\n== placement environment ==")
from rl.place_env import PlacementEnv, sample_scenarios, scale_power, score

scns = sample_scenarios(TABLES, n=12, seed=42)
check("scenarios come out of real runs", len(scns) == 12 and
      all(s.n_units > 0 and s.enemy_specs for s in scns))
check("every scenario carries a seed pool", all(len(s.seeds) > 0 for s in scns))

penv = PlacementEnv(TABLES, scns, seed=0)
pobs, pinfo = penv.reset(scns[0])
check("placement observation matches declared dim",
      pobs.shape == (penv.obs_dim,), f"{pobs.shape}")
check("placement observation is finite", np.isfinite(pobs).all())
check("the action space covers the widest player zone",
      penv.n_actions == max(len(l.zone("p"))
                            for l in parse_room_layouts(TABLES["RoomLayouts"])))

steps, seen = 0, []
m = pinfo["action_mask"]
while True:
    idx = pinfo["cell_index"]
    a = int(np.flatnonzero(m)[0])
    seen.append(a)
    pobs, r, term, trunc, pinfo = penv.step(a)
    steps += 1
    m = pinfo["action_mask"]
    if term or trunc:
        break
    check_taken = not m[seen[-1]]
    if not check_taken:
        break
check("an episode is exactly one step per unit", steps == scns[0].n_units,
      f"{steps} vs {scns[0].n_units}")
check("a used cell is masked out afterwards", check_taken)
check("the terminal reward is finite", np.isfinite(r))
check("cell_index only points at real cells",
      bool(((idx >= 0) & (idx < penv.rows * penv.cols)).any()))

# The paired reward is a difference against the heuristic on the same seed, so
# giving the policy the heuristic's own cells has to score exactly zero.
from rl.placement import frontline_placement as _front
from sim.battle import Agent as _Ag

scn = scns[0]
layout0 = parse_room_layouts(TABLES["RoomLayouts"])[scn.layout_index]
grid0 = Grid.from_layout(layout0)
penv.reset(scn)
foes = [_Ag(spec=s, team=1, x=x, y=y, hp=s.health, mana=0.0)
        for s, (x, y) in zip(scn.enemy_specs, scn.enemy_xy)]
want = _front(grid0, layout0, scn.specs, random.Random(penv.battle_seed), foes)
last = 0.0
for rc in want:
    i = penv.zone.index(rc) if rc in penv.zone else int(np.flatnonzero(penv.action_mask())[0])
    if not penv.action_mask()[i]:
        i = int(np.flatnonzero(penv.action_mask())[0])
    _, last, term, trunc, _ = penv.step(i)
    if term or trunc:
        break
check("the heuristic's own cells score zero against the heuristic",
      abs(last) < 1e-6, f"{last:+.6f}")

harder = scale_power(scns[0], 2.0)
check("scaling power only touches the enemy side",
      [s.health for s in harder.specs] == [s.health for s in scns[0].specs]
      and all(abs(a.health - 2.0 * b.health) < 1e-3
              for a, b in zip(harder.enemy_specs, scns[0].enemy_specs)))
check("a win scores above a loss",
      score({"won": True, "ally_frac": 0.0, "foe_frac": 0.0})
      > score({"won": False, "ally_frac": 1.0, "foe_frac": 0.0}))

print("\n== rust core ==")
from sim.fast import available as core_available

if not core_available():
    print("  SKIP  core not built (run `cargo build --release` in core/)")
else:
    import ctypes as _ct

    from sim.fast import load as core_load

    # The core runs CPython's MT19937, so crit and evasion consume the same
    # stream as the oracle. Without that, RNG fights could only be compared
    # statistically instead of exactly.
    lib = core_load()
    lib.despot_rng_probe.argtypes = [_ct.c_uint64, _ct.c_int32,
                                     _ct.POINTER(_ct.c_double)]
    lib.despot_rng_probe.restype = None
    exact = True
    for _seed in (0, 1, 5, 42, 12345, 2 ** 31 + 7):
        buf = (_ct.c_double * 8)()
        lib.despot_rng_probe(_seed, 8, buf)
        ref = random.Random(_seed)
        exact &= all(abs(buf[i] - ref.random()) < 1e-15 for i in range(8))
    check("the core's RNG matches CPython's exactly", exact)

print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
