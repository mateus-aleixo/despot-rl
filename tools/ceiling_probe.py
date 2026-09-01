"""How much room is there above the baseline on a generated level?

The trained agents reach 2.67 where the heuristic reaches 2.50, against +0.5 to
+0.9 on the old oversized map. This asks whether that is the agent failing or
the environment having no room left in it:

  * per-seed agreement, and what a *lucky* run of the same policy reaches
    (best of N randomised-heuristic rollouts on the same seed), which bounds how
    much of the outcome the dice own;
  * where each gold piece goes, and how much squad Power it buys.

    python tools/ceiling_probe.py [agent.pt ...] [--runs 30] [--rollouts 12]

`tools/power_curve.py` is the other half of the answer: it prints squad Power
against the room budget level by level.
"""
from __future__ import annotations

import argparse
import collections
import random
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from rl.env import DespotRunEnv
from rl.heuristic import heuristic_action
from rl.placement import POLICIES
from sim.data import load_ruleset
from tools.hierarchy_eval import heuristic, make_ppo


def play(tables, policy, seed, placement):
    """One run, with the economy instrumented."""
    env = DespotRunEnv(tables=tables, seed=seed, placement_policy=placement,
                       fast_core=True)
    obs, info = env.reset(seed=seed)
    mask, n = info["action_mask"], 0
    gold_in = 0.0
    spent = collections.Counter()
    bought = collections.Counter()
    peak = 0.0
    while mask.any() and n < env.max_steps:
        st = env.state
        before, power_before = st.gold, st.squad_power()
        action = policy(env, obs, mask)
        kind, _ = env._decode(action)
        legal = mask[action]
        obs, _, term, trunc, info = env.step(action)
        if legal:
            gold_in += max(0.0, env.state.gold - before)
            paid = max(0.0, before - env.state.gold)
            if kind != "move" and paid:
                spent[kind] += paid
                bought[kind] += env.state.squad_power() - power_before
            peak = max(peak, env.state.squad_power())
        mask, n = info["action_mask"], n + 1
        if term or trunc:
            break
    return dict(level=env.state.level, steps=n, gold_in=gold_in, peak=peak,
                spent=spent, bought=bought)


def noisy_heuristic(eps, rng):
    """The baseline with `eps` of its choices taken at random.

    Best-of-N over this is a hindsight bound of the cheapest kind: same policy,
    same map, different dice and the occasional different decision. It says how
    much of a run's outcome was never the policy's to decide.
    """
    def policy(env, obs, mask):
        if rng.random() < eps:
            return int(rng.choice(np.flatnonzero(mask)))
        legal = [env._decode(i) for i in np.flatnonzero(mask)]
        pick = heuristic_action(env.state, legal)
        return env._index(pick) if pick is not None else 0
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs="*")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--placement", default="frontline")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    placement = POLICIES[args.placement]
    seeds = list(range(30_000, 30_000 + args.runs))

    runs = {"heuristic": [play(tables, heuristic, s, placement) for s in seeds]}
    for path in args.agents:
        pol = make_ppo(path)
        runs[path] = [play(tables, pol, s, placement) for s in seeds]

    print(f"{'policy':26s} {'level':>6s} {'gold in':>8s} {'spent':>7s} "
          f"{'peak power':>11s}")
    for name, rs in runs.items():
        mean = lambda k: statistics.mean(r[k] for r in rs)
        out = statistics.mean(sum(r["spent"].values()) for r in rs)
        print(f"{name:26s} {mean('level'):6.2f} {mean('gold_in'):8.1f} "
              f"{out:7.1f} {mean('peak'):11,.0f}")

    base = [r["level"] for r in runs["heuristic"]]
    for name, rs in runs.items():
        if name == "heuristic":
            continue
        other = [r["level"] for r in rs]
        same = sum(1 for x, y in zip(base, other) if x == y)
        print(f"\n{name} against the heuristic: same level on {same}/{len(base)} "
              f"seeds, higher on {sum(1 for x, y in zip(base, other) if y > x)}, "
              f"lower on {sum(1 for x, y in zip(base, other) if y < x)}, "
              f"correlation {np.corrcoef(base, other)[0, 1]:.2f}")

    best, mean_roll = [], []
    for s in seeds:
        rng = random.Random(s)
        levels = [play(tables, noisy_heuristic(args.epsilon, rng), s, placement)["level"]
                  for _ in range(args.rollouts)]
        best.append(max(levels))
        mean_roll.append(statistics.mean(levels))
    print(f"\nbest of {args.rollouts} noisy-heuristic rollouts per seed: "
          f"mean best {statistics.mean(best):.2f}, "
          f"mean rollout {statistics.mean(mean_roll):.2f} "
          f"-- the dice are worth {statistics.mean(best) - statistics.mean(mean_roll):+.2f} levels")

    for name, rs in runs.items():
        spent = collections.Counter()
        bought = collections.Counter()
        for r in rs:
            spent.update(r["spent"])
            bought.update(r["bought"])
        total = sum(spent.values()) or 1.0
        print(f"\n=== {name} ===  {total / len(rs):.1f} gold spent per run")
        for kind, gold in spent.most_common():
            print(f"  {kind:14s} {gold / len(rs):6.1f} gold/run ({100 * gold / total:3.0f}%)"
                  f"   Power bought {bought[kind] / len(rs):8,.0f}"
                  f"  ({bought[kind] / max(1.0, gold):6,.0f} per gold)")


if __name__ == "__main__":
    main()
