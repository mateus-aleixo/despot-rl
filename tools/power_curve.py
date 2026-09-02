"""Does the squad get stronger as fast as the rooms do?

Every reward variant tried lands at level 4, which is a hint that the wall is not
in the reward at all. This measures the two curves that decide whether a run can
continue: the Power budget a room is filled to, which comes straight from
`Levels.json`, and the Power the squad actually has when it arrives at that
level.
"""
from __future__ import annotations

import argparse
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from rl.env import DespotRunEnv
from rl.placement import POLICIES
from sim.data import load_ruleset
from tools.hierarchy_eval import StaleCheckpoint, heuristic, make_ppo


def curve(tables, policy, seeds, placement, max_steps=400):
    by_level: dict[int, list[float]] = {}
    squads: dict[int, list[int]] = {}
    for s in seeds:
        env = DespotRunEnv(tables=tables, seed=s, placement_policy=placement,
                           fast_core=True, max_steps=max_steps)
        obs, info = env.reset(seed=s)
        mask, n, seen = info["action_mask"], 0, set()
        while mask.any() and n < max_steps:
            st = env.state
            if st.level not in seen:
                seen.add(st.level)
                by_level.setdefault(st.level, []).append(st.squad_power())
                squads.setdefault(st.level, []).append(len(st.squad))
            obs, r, term, trunc, info = env.step(policy(env, obs, mask))
            mask = info["action_mask"]
            n += 1
            if term or trunc:
                break
    return by_level, squads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--agent", default="runs/prog_a_baseline.pt")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    seeds = list(range(30_000, 30_000 + args.runs))
    placement = POLICIES["frontline"]

    env = DespotRunEnv(tables=tables, fast_core=True)
    env.reset(seed=0)

    policies = [("heuristic", heuristic)]
    if args.agent and args.agent.lower() != "none":
        try:
            policies.append((args.agent, make_ppo(args.agent, env)))
        except (FileNotFoundError, StaleCheckpoint) as exc:
            print(f"skipping {args.agent}: {exc}")

    for name, pol in policies:
        by_level, squads = curve(tables, pol, seeds, placement)
        print(f"\n=== {name} ===")
        print(f"{'level':>5s} {'runs':>5s} {'squad':>6s} {'squad power':>12s} "
              f"{'room power':>11s} {'ratio':>7s} {'ratio vs L1':>12s}")
        first = None
        for lvl in sorted(by_level):
            st = env.state
            st.level = lvl
            room = st.room_power()
            power = statistics.mean(by_level[lvl])
            ratio = power / room
            first = first if first is not None else ratio
            print(f"{lvl:5d} {len(by_level[lvl]):5d} "
                  f"{statistics.mean(squads[lvl]):6.1f} {power:12.0f} "
                  f"{room:11.0f} {ratio:7.1f} {ratio / first:12.2f}")


if __name__ == "__main__":
    main()
