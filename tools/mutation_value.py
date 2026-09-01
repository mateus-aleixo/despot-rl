"""Is the mutation shop worth anything? The paired test from `notes/rl.md`.

The heuristic plays the same run seeds twice: once able to take mutations, once
with `buy_mutation` and `take_mutation` struck out of its action mask.
Everything else -- the map, the shops, the packs, the fight seeds -- is drawn
from the run seed, so the two halves face an identical run and the difference is
paired. An unpaired comparison over 60 runs cannot see a tenth of a level.

The first time this was run, 823 of the 1,094 offers across the twelve levels
were `unimplemented` and every free mutation a run could collect was worth
+0.07 levels, 95% CI [-0.09, +0.23]: nothing distinguishable from zero. That was
a statement about the sim rather than about the game, and re-running it after the
agent-level passives went in is the point of this file.

    python tools/mutation_value.py [--runs 60] [--no-fast-core]
"""
import argparse
import statistics
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from rl.env import NON_MOVE_ACTIONS, DespotRunEnv
from rl.heuristic import heuristic as heuristic_policy
from rl.placement import POLICIES
from sim.data import load_ruleset
from sim.mutations import disable_agent_passives as disable_passives

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=60)
ap.add_argument("--no-fast-core", action="store_true")
ap.add_argument("--without-passives", action="store_true",
                help="put the agent-level passives back to unimplemented, so the "
                     "same seeds can be scored against the shelf as it was")
ap.add_argument("--compare", action="store_true",
                help="run both configurations over the same seeds and report the "
                     "difference of the two paired differences, with a CI")
args = ap.parse_args()

if args.without_passives:
    # The shelf still shows the same offers -- what is offered comes from
    # MutationsByLevel, not from the handler -- but nothing hangs off a hook,
    # which is the state this test first ran in.
    disable_passives()

TABLES = load_ruleset(strict=True)
PLACEMENT = POLICIES["frontline"]


def _mutation_indices(env) -> np.ndarray:
    """Where the two mutation action families sit in the fixed layout."""
    return np.array([env.n_moves + i for i, name in enumerate(NON_MOVE_ACTIONS)
                     if name.startswith("buy_mutation_") or name == "take_mutation"])


def play(seed: int, mutations: bool) -> dict:
    env = DespotRunEnv(tables=TABLES, seed=seed, placement_policy=PLACEMENT,
                       fast_core=not args.no_fast_core)
    obs, info = env.reset(seed=seed)
    blocked = _mutation_indices(env)
    mask, total, steps = info["action_mask"], 0.0, 0
    if not mutations:
        mask = mask.copy()
        mask[blocked] = False
    while mask.any() and steps < env.max_steps:
        obs, r, term, trunc, info = env.step(heuristic_policy(env, obs, mask))
        total += r
        steps += 1
        mask = info["action_mask"]
        if not mutations:
            mask = mask.copy()
            mask[blocked] = False
        if term or trunc:
            break
    return {"level": env.state.level, "return": total,
            "mutations": len(env.state.mutations), "steps": steps}


def ci95(xs) -> tuple[float, float]:
    if len(xs) < 2:
        return (0.0, 0.0)
    half = 1.96 * statistics.stdev(xs) / len(xs) ** 0.5
    return (statistics.mean(xs) - half, statistics.mean(xs) + half)


seeds = list(range(args.runs))
rows = {}
for label, take in (("takes them", True), ("ignores them", False)):
    t0 = time.perf_counter()
    rows[label] = [play(s, take) for s in seeds]
    print(f"  {label:14s} {args.runs} runs in {time.perf_counter() - t0:.0f}s")

print()
print(f"{'':16s} {'mean level':>10s} {'mean return':>12s} {'mutations a run':>16s}")
for label, runs in rows.items():
    print(f"{label:16s} {statistics.mean(r['level'] for r in runs):10.2f} "
          f"{statistics.mean(r['return'] for r in runs):12.2f} "
          f"{statistics.mean(r['mutations'] for r in runs):16.1f}")

for field, unit in (("level", "level"), ("return", "return")):
    diff = [a[field] - b[field] for a, b in zip(rows["takes them"], rows["ignores them"])]
    lo, hi = ci95(diff)
    print(f"\npaired difference in {unit}: {statistics.mean(diff):+.2f}, "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]")
    if field == "level":
        print(f"same level reached on {sum(1 for d in diff if d == 0)} "
              f"of {len(diff)} seeds")

if args.compare and not args.without_passives:
    # What the passives themselves are worth: the same seeds, scored with the
    # hooks live and then with them inert, and the difference of the two paired
    # differences. Doing it per seed rather than comparing two intervals is the
    # only way to get a CI on the change itself.
    with_on = [a["level"] - b["level"]
               for a, b in zip(rows["takes them"], rows["ignores them"])]
    disable_passives()
    print("\n  re-scoring the same seeds with the passives inert")
    t0 = time.perf_counter()
    off_take = [play(s, True) for s in seeds]
    off_skip = [play(s, False) for s in seeds]
    print(f"  {'passives off':14s} {args.runs} runs in {time.perf_counter() - t0:.0f}s")
    with_off = [a["level"] - b["level"] for a, b in zip(off_take, off_skip)]
    delta = [a - b for a, b in zip(with_on, with_off)]
    lo, hi = ci95(delta)
    # The two mean levels the notes table quotes, which the paired differences
    # above do not carry: "passives inert, takes them" against "ignores them".
    print(f"\npassives inert  takes them "
          f"{statistics.mean(r['level'] for r in off_take):.2f}"
          f"   ignores them {statistics.mean(r['level'] for r in off_skip):.2f}")
    print(f"shelf worth with the passives : {statistics.mean(with_on):+.2f} levels")
    print(f"shelf worth without them      : {statistics.mean(with_off):+.2f} levels")
    print(f"what the passives added       : {statistics.mean(delta):+.2f}, "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]")
