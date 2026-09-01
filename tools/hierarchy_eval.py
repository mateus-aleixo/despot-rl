"""Both levels together: does the placement policy change how a run goes?

The low level is measured one fight at a time in `tools/placement_eval.py`. This
runs whole runs, crossing each run-level agent with each placement policy on the
same seeds, which is the only measurement that says whether the hierarchy is
worth having.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from rl.env import DespotRunEnv
from rl.heuristic import heuristic
from rl.placement import POLICIES
from rl.train import ActorCritic
from sim.data import load_ruleset


class StaleCheckpoint(ValueError):
    """A saved agent whose shapes no longer match the environment."""


def make_ppo(path: str, env: DespotRunEnv | None = None):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if env is not None and (ck["obs_dim"] != env.obs_dim
                            or ck["n_actions"] != env.n_actions):
        raise StaleCheckpoint(
            f"{path} is {ck['obs_dim']}x{ck['n_actions']}, the environment is "
            f"{env.obs_dim}x{env.n_actions} -- it predates a change to the "
            f"observation or the action space and has to be retrained")
    model = ActorCritic(ck["obs_dim"], ck["n_actions"])
    model.load_state_dict(ck["model"])
    model.eval()

    def policy(env, obs, mask):
        with torch.no_grad():
            d, _ = model(torch.as_tensor(obs[None]), torch.as_tensor(mask[None]))
            return int(d.probs.argmax())
    return policy


def run(tables, policy, placement, seeds, fast=True):
    levels, returns, steps, wins = [], [], [], 0
    for s in seeds:
        env = DespotRunEnv(tables=tables, seed=s, placement_policy=placement,
                           fast_core=fast)
        obs, info = env.reset(seed=s)
        mask, total, n = info["action_mask"], 0.0, 0
        while mask.any() and n < env.max_steps:
            obs, r, term, trunc, info = env.step(policy(env, obs, mask))
            total += r
            mask = info["action_mask"]
            n += 1
            if term or trunc:
                break
        levels.append(env.state.level)
        returns.append(total)
        steps.append(n)
        wins += 1 if env.state.won else 0
    return levels, returns, steps, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--run-agent", default="runs/prog_a_baseline.pt")
    ap.add_argument("--placement", default="runs/placement_final.pt")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    seeds = list(range(30_000, 30_000 + args.runs))

    placements = {"random": POLICIES["random"], "frontline": POLICIES["frontline"]}
    try:
        from rl.place_policy import load_placement
        placements["learned"] = load_placement(args.placement, greedy=True)
    except (FileNotFoundError, OSError):
        print(f"no placement checkpoint at {args.placement}")

    agents = {"heuristic": heuristic}
    if args.run_agent and args.run_agent.lower() != "none":
        try:
            agents["PPO"] = make_ppo(args.run_agent,
                                     DespotRunEnv(tables=tables, fast_core=True))
        except (FileNotFoundError, StaleCheckpoint) as exc:
            print(f"skipping {args.run_agent}: {exc}\n")

    print(f"{args.runs} runs per cell, identical seeds. The last column is the "
          f"per-seed paired\ndifference in level against the same agent on "
          f"`frontline`, with its 95% interval.\n")
    print(f"{'run agent':10s} {'placement':11s} {'mean level':>10s} {'median':>7s} "
          f"{'max':>4s} {'mean return':>12s} {'mean steps':>11s} {'won':>4s} "
          f"{'vs frontline':>22s}")
    for aname, apol in agents.items():
        rows, ref = [], None
        for pname, ppol in placements.items():
            lv, rt, st, wins = run(tables, apol, ppol, seeds)
            rows.append((pname, lv, rt, st, wins))
            if pname == "frontline":
                ref = lv
        for pname, lv, rt, st, wins in rows:
            diff = [a - b for a, b in zip(lv, ref)]
            spread = statistics.stdev(diff) if len(diff) > 1 else 0.0
            m = statistics.mean(diff)
            hw = 1.96 * spread / math.sqrt(len(diff)) if spread > 0 else 0.0
            print(f"{aname:10s} {pname:11s} {statistics.mean(lv):10.2f} "
                  f"{statistics.median(lv):7.1f} {max(lv):4d} "
                  f"{statistics.mean(rt):12.2f} {statistics.mean(st):11.1f} "
                  f"{wins:4d} {f'{m:+.2f} [{m - hw:+.2f}, {m + hw:+.2f}]':>22s}")


if __name__ == "__main__":
    main()
