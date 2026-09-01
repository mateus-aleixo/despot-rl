"""Where does a run's food actually go?

`tools/run_autopsy.py` reports `buy_food` as a share of actions, and that share
falls from 17% to 3% between 150k and 600k steps, which reads as an agent that
stops buying food. It is not: an action share has the episode length in its
denominator, and episodes triple over the same budget.

This counts the things a share cannot show -- how often the shop is reachable,
how often the agent takes the offer, whether it was ever priced out, and how the
run's food balance closes:

    python tools/food_probe.py runs/food_shop_s1.600000.pt [more...]
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from rl.env import DespotRunEnv
from rl.placement import POLICIES
from sim.data import load_ruleset
from tools.hierarchy_eval import heuristic, make_ppo


def probe(tables, policy, seeds, placement, max_steps=400):
    n = collections.Counter()
    declined_gold, taken_gold, shelf_left = [], [], []
    for s in seeds:
        env = DespotRunEnv(tables=tables, seed=s, placement_policy=placement,
                           fast_core=True, max_steps=max_steps)
        obs, info = env.reset(seed=s)
        mask, steps = info["action_mask"], 0
        while mask.any() and steps < max_steps:
            st = env.state
            here = st.rooms.rooms[st.room]
            sizes, costs = st.food_packs
            if here.kind == "food_shop":
                st.ensure_stock(here)
                stock = here.food_stock or []
                n["in_shop"] += 1
                n["in_shop_with_stock"] += any(q > 0 for q in stock)
                # Priced out: stock on the shelf and not enough gold for any of
                # it. This is the reading "it stopped buying" implies, so it is
                # the one worth counting separately from declining.
                if any(q > 0 for q in stock) and not any(
                        q > 0 and st.gold >= costs[i] for i, q in enumerate(stock)):
                    n["priced_out"] += 1

            offered = any(env._decode(i)[0] == "buy_food"
                          for i in np.flatnonzero(mask))
            action = policy(env, obs, mask)
            kind, arg = env._decode(action)
            if mask[action]:
                if offered:
                    n["offered"] += 1
                    if kind == "buy_food":
                        n["taken"] += 1
                        taken_gold.append(st.gold)
                    else:
                        declined_gold.append(st.gold)
                if kind == "buy_food":
                    n["food_bought"] += sizes[int(arg)]
                elif kind == "move":
                    n["moves"] += 1
                    if st.food.can_feed(st.feed_cost):
                        n["food_eaten"] += st.feed_cost
                    else:
                        n["moves_on_reserve"] += 1

            obs, _, term, trunc, info = env.step(action)
            mask, steps = info["action_mask"], steps + 1
            if term or trunc:
                break

        st = env.state
        sizes, _ = st.food_packs
        shelf_left.append(sum(
            sum(sz * q for sz, q in zip(sizes, r.food_stock))
            for r in st.rooms.rooms.values()
            if r.kind == "food_shop" and r.food_stock))
        n["levels"] += st.level
    return n, declined_gold, taken_gold, shelf_left


def report(name, runs, n, declined, taken, shelf):
    mean = lambda xs: statistics.mean(xs) if xs else float("nan")
    rate = 100 * n["taken"] / n["offered"] if n["offered"] else 0.0
    print(f"\n=== {name} ===")
    print(f"  offered on {n['offered']:4d} steps, taken {n['taken']:4d} ({rate:.0f}%), "
          f"priced out {n['priced_out']:3d}")
    print(f"  gold in hand {mean(taken):5.1f} buying / {mean(declined):5.1f} declining;"
          f"  food left on the level's shelves {mean(shelf):5.1f}")
    print(f"  per run: in a food shop {n['in_shop'] / runs:5.2f} steps "
          f"({n['in_shop_with_stock'] / runs:.2f} with stock), "
          f"moves {n['moves'] / runs:5.1f} "
          f"({100 * n['moves_on_reserve'] / max(1, n['moves']):.0f}% on the reserve)")
    print(f"  per run: food bought {n['food_bought'] / runs:6.1f}, "
          f"eaten {n['food_eaten'] / runs:6.1f}, "
          f"level reached {n['levels'] / runs:4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs="*")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--placement", default="frontline")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    seeds = list(range(30_000, 30_000 + args.runs))
    placement = POLICIES[args.placement]

    report("heuristic", args.runs,
           *probe(tables, heuristic, seeds, placement))
    for path in args.agents:
        report(path, args.runs,
               *probe(tables, make_ppo(path), seeds, placement))


if __name__ == "__main__":
    main()
