"""Does upgrading the item shop pay?

The 150k agent never upgrades: peak shop level 1.00 across 60 runs, so it buys
quality-1 and quality-2 items all run and never opens the quality-5 pool, where
an item is worth ten times as much. That is either a gap in the agent or a
correct read of the mechanic, and the two need separating before either is
called a mistake.

This runs the same baseline with the upgrade moved in its priority order --
never, only when nothing on the shelf is worth buying, or before anything else
-- over identical run seeds, and reports the per-seed paired difference against
`never` with a 95% interval. Paired, because run seeds vary far more than the
policies do: an unpaired comparison of 60 runs cannot see a tenth of a level.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rl.env import DespotRunEnv
from rl.heuristic import heuristic_action
from rl.placement import POLICIES
from sim.data import load_ruleset

ORDERS = ("never", "last", "first")


def policy_for(order: str):
    def policy(env, obs, mask):
        legal = [env._decode(i) for i in mask.nonzero()[0]]
        pick = heuristic_action(env.state, legal, upgrade=order)
        return env._index(pick) if pick is not None else 0
    return policy


def run(tables, policy, placement, seeds, max_steps=400):
    out = []
    for s in seeds:
        env = DespotRunEnv(tables=tables, seed=s, placement_policy=placement,
                           fast_core=True, max_steps=max_steps)
        obs, info = env.reset(seed=s)
        mask, n, total = info["action_mask"], 0, 0.0
        peak_shop, peak_power = 1, 0.0
        while mask.any() and n < max_steps:
            obs, r, term, trunc, info = env.step(policy(env, obs, mask))
            total += r
            mask = info["action_mask"]
            n += 1
            st = env.state
            peak_shop = max(peak_shop, st.shop_level)
            if st.squad:
                peak_power = max(peak_power, st.squad_power())
            if term or trunc:
                break
        out.append({"level": env.state.level, "ret": total, "steps": n,
                    "shop": peak_shop, "power": peak_power})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--placement", default="frontline", choices=sorted(POLICIES))
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    seeds = list(range(30_000, 30_000 + args.runs))
    placement = POLICIES[args.placement]

    rows = {o: run(tables, policy_for(o), placement, seeds) for o in ORDERS}
    ref = [r["level"] for r in rows["never"]]

    print(f"{args.runs} runs per row, identical seeds, {args.placement} placement.\n"
          f"The last column is the per-seed paired difference in level against "
          f"`never`.\n")
    print(f"{'upgrade':8s} {'mean level':>10s} {'5+':>5s} {'mean return':>12s} "
          f"{'peak shop':>10s} {'peak power':>11s} {'vs never':>22s}")
    for order in ORDERS:
        lv = [r["level"] for r in rows[order]]
        diff = [a - b for a, b in zip(lv, ref)]
        spread = statistics.stdev(diff) if len(diff) > 1 else 0.0
        m = statistics.mean(diff)
        hw = 1.96 * spread / math.sqrt(len(diff)) if spread > 0 else 0.0
        print(f"{order:8s} {statistics.mean(lv):10.2f} "
              f"{sum(1 for x in lv if x >= 5):5d} "
              f"{statistics.mean(r['ret'] for r in rows[order]):12.2f} "
              f"{statistics.mean(r['shop'] for r in rows[order]):10.2f} "
              f"{statistics.mean(r['power'] for r in rows[order]):11,.0f} "
              f"{f'{m:+.2f} [{m - hw:+.2f}, {m + hw:+.2f}]':>22s}")


if __name__ == "__main__":
    main()
