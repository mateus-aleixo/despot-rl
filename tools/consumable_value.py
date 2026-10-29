"""What the boss room's consumable shop is worth to the baseline.

Every boss room but the last builds a consumable shop (`C_Room..ctor`, from the
`ConsumableShop` bit `ChooseFinishRoom` sets), and it is free. The heuristic
plays the same seeds twice, once able to take from it and once with
`take_consumable` struck out of its mask; everything else is identical until
the first boss, so the difference is paired.

This is a check that the feature changes something, not a re-baseline. RL
numbers are taken at a batch boundary; see `notes/roadmap.md`.

    python tools/consumable_value.py --seeds 240 --fast
"""
import argparse
import collections
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from rl.env import NON_MOVE_ACTIONS, DespotRunEnv
from rl.heuristic import heuristic
from sim.data import load_ruleset


def play(env, seed, take):
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    struck = np.array([env.n_moves + i for i, name in enumerate(NON_MOVE_ACTIONS)
                       if name.startswith("take_consumable_")])
    used = collections.Counter()
    while True:
        if not take:
            mask = mask.copy()
            mask[struck] = False
        if not mask.any():
            break
        action = heuristic(env, obs, mask)
        obs, _, term, trunc, info = env.step(action)
        result = info.get("result") or {}
        if result.get("used"):
            used[result["family"]] += 1
        mask = info["action_mask"]
        if term or trunc:
            break
    return env.state.level, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=240)
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--fast", action="store_true", help="resolve fights in the Rust core")
    args = ap.parse_args()

    env = DespotRunEnv(tables=load_ruleset(strict=True), fast_core=args.fast)
    seeds = range(args.start, args.start + args.seeds)
    arms = {}
    for name, take in (("takes them", True), ("ignores them", False)):
        out = [play(env, s, take) for s in seeds]
        arms[name] = [o[0] for o in out]
        uses = collections.Counter()
        for o in out:
            uses.update(o[1])
        per_run = ", ".join(f"{k} {v / len(out):.2f}" for k, v in uses.most_common())
        print(f"  {name:<13} {statistics.mean(arms[name]):.3f}  "
              f"(sd {statistics.pstdev(arms[name]):.3f})   uses a run: {per_run or 'none'}")
    diff = [a - b for a, b in zip(arms["takes them"], arms["ignores them"])]
    m = statistics.mean(diff)
    se = statistics.stdev(diff) / len(diff) ** 0.5
    print(f"\n  consumables, paired by seed: {m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}], "
          f"better on {sum(1 for d in diff if d > 0)} of {len(diff)}, "
          f"worse on {sum(1 for d in diff if d < 0)}")


if __name__ == "__main__":
    main()
