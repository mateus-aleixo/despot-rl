"""How do runs actually end, and what is the agent doing with its steps?

The 300k-step agent plateaus at level 4 and spends a hundred actions getting
there. Before reshaping the reward it is worth knowing which of the three ways a
run can stop is actually stopping it -- wiped, starved into a corner, or simply
truncated -- and where the actions go.
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from rl.env import NON_MOVE_ACTIONS, DespotRunEnv
from rl.placement import POLICIES
from rl.train import ActorCritic
from sim.data import load_ruleset
from tools.hierarchy_eval import heuristic, make_ppo


def autopsy(tables, policy, seeds, placement, max_steps=400):
    rows = []
    actions = collections.Counter()
    for s in seeds:
        env = DespotRunEnv(tables=tables, seed=s, placement_policy=placement,
                           fast_core=True, max_steps=max_steps)
        obs, info = env.reset(seed=s)
        mask, n, total = info["action_mask"], 0, 0.0
        fights = wins = losses = 0
        # The peak, not the end: every run in this sample wipes, and a wiped
        # squad has no level, no items and no Power to report.
        peak = {"unit_level": 1.0, "shop": 1, "power": 0.0, "armed": 0}
        while mask.any() and n < max_steps:
            a = policy(env, obs, mask)
            kind, _ = env._decode(a)
            if mask[a]:
                actions[kind if kind != "move" else "move"] += 1
            else:
                actions["illegal"] += 1
            obs, r, term, trunc, info = env.step(a)
            total += r
            res = info.get("result") or {}
            if "won" in res:
                fights += 1
                wins += 1 if res["won"] else 0
                losses += 0 if res["won"] else 1
            mask = info["action_mask"]
            n += 1
            st = env.state
            if st.squad:
                peak["unit_level"] = max(peak["unit_level"],
                                         sum(h.level for h in st.squad) / len(st.squad))
                peak["armed"] = max(peak["armed"], sum(1 for h in st.squad if h.item))
                peak["power"] = max(peak["power"], st.squad_power())
            peak["shop"] = max(peak["shop"], st.shop_level)
            if term or trunc:
                break

        st = env.state
        if st.won:
            why = "won the run"
        elif not st.squad:
            why = "wiped"
        elif st.finished:
            why = "finished, lost"
        elif n >= max_steps:
            why = "out of steps"
        elif not mask.any():
            why = "no legal action"
        else:
            why = "?"
        rows.append({
            "why": why, "level": st.level, "steps": n, "ret": total,
            "squad": len(st.squad),
            "gold": st.gold, "food": st.food.amount,
            "hunger": st.food.hunger_level, "mutations": len(st.mutations),
            "fights": fights, "wins": wins, "losses": losses,
            "armed": sum(1 for h in st.squad if h.item),
            "shop": peak["shop"], "unit_level": peak["unit_level"],
            "power": peak["power"], "peak_armed": peak["armed"],
        })
    return rows, actions


def report(name, rows, actions):
    n = len(rows)
    lv = [r["level"] for r in rows]
    print(f"\n=== {name} ===")
    print(f"  mean level {statistics.mean(lv):4.2f}  median {statistics.median(lv):3.1f}  "
          f"max {max(lv)}  reached 5+ {sum(1 for x in lv if x >= 5)}/{n}  "
          f"mean return {statistics.mean(r['ret'] for r in rows):6.2f}")
    why = collections.Counter(r["why"] for r in rows)
    for w, c in why.most_common():
        lv = statistics.mean([r["level"] for r in rows if r["why"] == w])
        stp = statistics.mean([r["steps"] for r in rows if r["why"] == w])
        print(f"  {w:16s} {c:3d}/{n}  mean level {lv:4.2f}  mean steps {stp:5.1f}")

    f = lambda k: statistics.mean(r[k] for r in rows)
    print(f"  at the end: squad {f('squad'):4.1f} ({f('armed'):.1f} armed), "
          f"gold {f('gold'):5.1f}, food {f('food'):5.1f}, "
          f"hunger {f('hunger'):.2f}, mutations {f('mutations'):.1f}")
    print(f"  fights {f('fights'):4.1f} per run, "
          f"{f('wins'):.1f} won / {f('losses'):.1f} lost")
    print(f"  peak: unit level {f('unit_level'):.2f}, shop level {f('shop'):.2f}, "
          f"{f('peak_armed'):.1f} armed, squad power {f('power'):,.0f}")

    total = sum(actions.values()) or 1
    # Counted by the action's kind, not by its index: the shop is one index per
    # slot, so `NON_MOVE_ACTIONS` holds buy_item_0..6 while `_decode` gives back
    # a plain "buy_item". Listing the indices here silently reported 0% items.
    order = ["move", "buy_item"] + [a for a in NON_MOVE_ACTIONS
                                    if not a.startswith("buy_item_")] + ["illegal"]
    order += [k for k in sorted(actions) if k not in order]
    print("  actions: " + "  ".join(f"{k} {100 * actions[k] / total:.0f}%"
                                    for k in order if actions[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--agents", default="runs/prog_a_baseline.pt")
    ap.add_argument("--placement", default="frontline")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    seeds = list(range(30_000, 30_000 + args.runs))
    placement = POLICIES[args.placement]

    report("heuristic", *autopsy(tables, heuristic, seeds, placement))
    for path in args.agents.split(","):
        if not path.strip():
            continue
        report(path.strip(), *autopsy(tables, make_ppo(path.strip()), seeds, placement))


if __name__ == "__main__":
    main()
