"""The two reference policies, random and the heuristic, at a batch boundary.

Both play under the conditions an agent is trained and scored under
(`frontline` placement, the Rust core, the eight squads cycled by seed), so a
trained agent's level reads against these directly. Each random episode draws
from its own generator, seeded by the episode, so a slice of the seeds gives
the same levels in any process.

    python tools/baseline.py --seeds 2400 --start 40000 --jobs 8
"""
import argparse
import collections
import multiprocessing
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from rl.env import DespotRunEnv
from rl.heuristic import heuristic
from rl.placement import POLICIES
from sim.data import load_ruleset


def _play(env, seed, policy):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    mask, n = info["action_mask"], 0
    while mask.any() and n < env.max_steps:
        a = (int(rng.choice(np.flatnonzero(mask))) if policy == "random"
             else heuristic(env, obs, mask))
        obs, _, term, trunc, info = env.step(a)
        mask, n = info["action_mask"], n + 1
        if term or trunc:
            break
    return env.state.level, env.squad_name


def _chunk(seeds):
    env = DespotRunEnv(tables=load_ruleset(strict=True), fast_core=True,
                       placement_policy=POLICIES["frontline"])
    return [(s, policy, *_play(env, s, policy))
            for s in seeds for policy in ("random", "heuristic")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2400)
    ap.add_argument("--start", type=int, default=40000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    seeds = list(range(args.start, args.start + args.seeds))
    parts = [seeds[i::args.jobs] for i in range(args.jobs)]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            rows = [r for part in pool.map(_chunk, parts) for r in part]
    else:
        rows = _chunk(seeds)

    for policy in ("random", "heuristic"):
        levels = [lv for _, p, lv, _ in rows if p == policy]
        m, sd = statistics.mean(levels), statistics.stdev(levels)
        half = 1.96 * sd / len(levels) ** 0.5
        print(f"  {policy:9s} {m:.3f} [{m - half:.3f}, {m + half:.3f}]  (sd {sd:.3f}) "
              f"over {len(levels)} seeds from {args.start}")
    by_squad = collections.defaultdict(list)
    for _, p, lv, squad in rows:
        if p == "heuristic":
            by_squad[squad].append(lv)
    print("  heuristic by squad: " + ", ".join(
        f"{sq} {statistics.mean(v):.2f}" for sq, v in
        sorted(by_squad.items(), key=lambda kv: -statistics.mean(kv[1]))))
    reached = collections.Counter(lv for _, p, lv, _ in rows if p == "heuristic")
    print("  heuristic, runs ending on each level: " + ", ".join(
        f"{lv}: {n}" for lv, n in sorted(reached.items())))


if __name__ == "__main__":
    main()
