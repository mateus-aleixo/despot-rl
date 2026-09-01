"""Random vs hand-written heuristic vs trained PPO, under identical conditions.

The agent is a command-line argument rather than a hard-coded path, because a
saved agent stops loading the moment the observation or the action space
changes and this file used to crash on import when that happened.
"""
import argparse, statistics, sys
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from rl.env import DespotRunEnv
from rl.heuristic import heuristic as policy_heuristic
from rl.placement import POLICIES
from sim.data import load_ruleset
from tools.hierarchy_eval import StaleCheckpoint, make_ppo

ap = argparse.ArgumentParser()
ap.add_argument("--agent", default="runs/prog_a_baseline.pt")
ap.add_argument("--runs", type=int, default=30)
args = ap.parse_args()

t = load_ruleset(strict=True)
PLACEMENT = POLICIES["frontline"]
SEEDS = list(range(30_000, 30_000 + args.runs))


# Seeded, so the random row is reproducible rather than drifting a tenth of a
# level between runs of the same command.
_RNG = np.random.default_rng(0)


def policy_random(env, obs, mask):
    return int(_RNG.choice(np.flatnonzero(mask)))


def run(policy, seeds):
    levels, returns, steps_ = [], [], []
    for s in seeds:
        env = DespotRunEnv(tables=t, seed=s, placement_policy=PLACEMENT,
                           fast_core=True)
        obs, info = env.reset(seed=s)
        mask, total, n = info["action_mask"], 0.0, 0
        while mask.any() and n < env.max_steps:
            obs, r, term, trunc, info = env.step(policy(env, obs, mask))
            total += r; mask = info["action_mask"]; n += 1
            if term or trunc:
                break
        levels.append(env.state.level); returns.append(total); steps_.append(n)
    return levels, returns, steps_


agents = [("random", policy_random), ("heuristic", policy_heuristic)]
try:
    agents.append((args.agent,
                   make_ppo(args.agent, DespotRunEnv(tables=t, fast_core=True))))
except (FileNotFoundError, StaleCheckpoint) as exc:
    print(f"skipping {args.agent}: {exc}\n")

print(f"{'agent':28s} {'mean level':>10s} {'median':>7s} {'max':>4s} "
      f"{'mean return':>12s} {'mean steps':>11s}")
for name, pol in agents:
    lv, rt, stp = run(pol, SEEDS)
    print(f"{name:28s} {statistics.mean(lv):10.2f} {statistics.median(lv):7.1f} "
          f"{max(lv):4d} {statistics.mean(rt):12.2f} {statistics.mean(stp):11.1f}")
