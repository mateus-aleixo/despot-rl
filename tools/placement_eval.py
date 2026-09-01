"""How much does placement matter in the fights a real run actually produces?

`tools/placement_compare.py` answers a narrower question: on one hand-picked
enemy pack, does the front-line heuristic beat a random spread? It does, clearly.
This tool asks the question the RL side needs answered instead: over fights
sampled from real runs, how much of the outcome is placement able to move at
all?

Every policy is measured on the same scenarios and the same battle seeds, so the
comparison is paired and the fight draw cancels. Three references and one bound:

    random      the spread the sim used before any policy existed
    frontline   the hand-written heuristic
    learned     a trained PlacementNet, if a checkpoint is given
    best-of-8   the best of eight random placements *on that seed*

`best-of-8` is not a policy: it picks after seeing the outcome, which no agent
can do. It is the ceiling. If it barely beats `frontline`, then placement has
almost nothing left to give on that fight, and a learned policy failing to beat
the heuristic is a fact about the game rather than about the learner.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rl.place_env import (cached_scenarios, resolve, sample_close_scenarios,
                          sample_scenarios, score)
from rl.placement import POLICIES
from sim.battle import Agent
from sim.data import load_ruleset, parse_room_layouts
from sim.nav import Grid


def enemies_of(scn):
    return [Agent(spec=s, team=1, x=x, y=y, hp=s.health, mana=0.0)
            for s, (x, y) in zip(scn.enemy_specs, scn.enemy_xy)]


def mean_ci(xs) -> tuple[float, float]:
    """Mean and half-width of the 95% interval, normal approximation."""
    if len(xs) < 2:
        return (statistics.mean(xs) if xs else 0.0), 0.0
    return statistics.mean(xs), 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=8, help="battle seeds per fight")
    ap.add_argument("--checkpoint", default="runs/placement_final.pt")
    ap.add_argument("--no-learned", action="store_true")
    ap.add_argument("--close", action="store_true",
                    help="measure on fights rescaled until placement decides them")
    ap.add_argument("--cache", default="",
                    help="reuse a pickled scenario set (the trainer writes these)")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    layouts = parse_room_layouts(tables["RoomLayouts"])
    if args.cache:
        # the trainer stores (train, held-out); measuring uses the held-out half
        scns = cached_scenarios(args.cache, lambda: (None, None))[1][:args.scenarios]
    elif args.close:
        scns = sample_close_scenarios(tables, n=args.scenarios, seed=700_000)
    else:
        scns = sample_scenarios(tables, n=args.scenarios, seed=700_000)

    policies = {"random": POLICIES["random"], "frontline": POLICIES["frontline"]}
    if not args.no_learned:
        try:
            from rl.place_policy import load_placement
            policies["learned"] = load_placement(args.checkpoint, greedy=True)
        except (FileNotFoundError, OSError):
            print(f"no checkpoint at {args.checkpoint}, skipping the learned policy")

    # scores[policy][i] is the mean score of that policy on scenario i
    scores: dict[str, list[float]] = {k: [] for k in policies}
    wins: dict[str, list[float]] = {k: [] for k in policies}
    best_scores, best_wins = [], []
    front_p: list[float] = []

    for scn in scns:
        layout = layouts[scn.layout_index]
        grid = Grid.from_layout(layout)
        seeds = list(scn.seeds[:args.seeds]) or [0]
        per_policy = {k: [] for k in policies}
        per_policy_win = {k: [] for k in policies}
        best_s, best_w = [], []

        for seed in seeds:
            for name, pol in policies.items():
                cells = pol(grid, layout, scn.specs, random.Random(seed), enemies_of(scn))
                out = resolve(scn, cells, tables, layouts, seed)
                per_policy[name].append(score(out))
                per_policy_win[name].append(1.0 if out["won"] else 0.0)

            # the selection bound: eight random placements, keep the best
            rolls = []
            for k in range(8):
                cells = POLICIES["random"](grid, layout, scn.specs,
                                           random.Random(seed * 31 + k), enemies_of(scn))
                out = resolve(scn, cells, tables, layouts, seed)
                rolls.append((score(out), 1.0 if out["won"] else 0.0))
            top = max(rolls, key=lambda x: x[0])
            best_s.append(top[0])
            best_w.append(top[1])

        for name in policies:
            scores[name].append(statistics.mean(per_policy[name]))
            wins[name].append(statistics.mean(per_policy_win[name]))
        best_scores.append(statistics.mean(best_s))
        best_wins.append(statistics.mean(best_w))
        front_p.append(statistics.mean(per_policy_win["frontline"]))

    n = len(scns)
    kind = ("fights rescaled until placement decides them" if args.close or args.cache
            else "fights sampled from real runs")
    print(f"{n} {kind}, {args.seeds} battle seeds each "
          f"(mean squad {statistics.mean(s.n_units for s in scns):.1f}, "
          f"mean enemies {statistics.mean(len(s.enemy_specs) for s in scns):.1f})\n")

    print(f"{'policy':12s} {'win rate':>9s} {'mean score':>11s} "
          f"{'vs frontline':>14s} {'95% CI':>16s}")
    ref = scores["frontline"]
    rows = list(policies.items()) + [("best-of-8", None)]
    for name, _ in rows:
        sc = best_scores if name == "best-of-8" else scores[name]
        wr = best_wins if name == "best-of-8" else wins[name]
        diff = [a - b for a, b in zip(sc, ref)]
        m, hw = mean_ci(diff)
        print(f"{name:12s} {statistics.mean(wr):9.3f} {statistics.mean(sc):11.3f} "
              f"{m:+14.3f} {f'[{m - hw:+.3f}, {m + hw:+.3f}]':>16s}")

    # Placement can only matter where the fight is not already decided.
    print("\nby how close the fight is (frontline win rate over its seeds):")
    buckets = {"always lost (0)": lambda p: p == 0.0,
               "close (0 < p < 1)": lambda p: 0.0 < p < 1.0,
               "always won (1)": lambda p: p == 1.0}
    print(f"{'bucket':20s} {'fights':>6s} {'random':>8s} {'frontline':>10s} "
          f"{'learned':>8s} {'best-of-8':>10s}")
    for label, pred in buckets.items():
        idx = [i for i, p in enumerate(front_p) if pred(p)]
        if not idx:
            continue
        cells = [f"{statistics.mean([wins[k][i] for i in idx]):8.3f}"
                 if k in wins else " " * 8 for k in ("random", "frontline", "learned")]
        b = statistics.mean([best_wins[i] for i in idx])
        print(f"{label:20s} {len(idx):6d} {cells[0]} {cells[1]:>10s} "
              f"{cells[2]} {b:10.3f}")


if __name__ == "__main__":
    main()
