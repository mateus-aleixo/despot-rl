"""What the level-entry dialogs change, and what choosing well in them is worth.

Three arms over the same seeds: no dialogs at all (the sim before them), the
dialogs with the heuristic choosing, and the dialogs with a random choice at
every node. The dialogs draw from the run's random stream from the first level
on, so the arms are paired by seed only, not step for step.

This is a check that the feature changes something, not a re-baseline. RL
numbers are taken at a batch boundary; see `notes/roadmap.md`.

    python tools/dialog_value.py --seeds 240 --fast
"""
import argparse
import random
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from rl.env import DespotRunEnv
from rl.heuristic import heuristic
from sim.data import load_ruleset


def play(env, seed, random_choices):
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    pick = random.Random(seed)
    choices = 0
    while mask.any():
        if random_choices and env.state.dialog is not None:
            action = int(pick.choice(np.flatnonzero(mask)))
        else:
            action = heuristic(env, obs, mask)
        choices += env.state.dialog is not None
        obs, _, term, trunc, info = env.step(action)
        mask = info["action_mask"]
        if term or trunc:
            break
    return env.state.level, choices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=240)
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--fast", action="store_true", help="resolve fights in the Rust core")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    seeds = range(args.start, args.start + args.seeds)
    arms = {}
    for name, dialogs, rnd in (("no dialogs", False, False),
                               ("heuristic chooses", True, False),
                               ("random choices", True, True)):
        env = DespotRunEnv(tables=tables, fast_core=args.fast, dialogs=dialogs)
        out = [play(env, s, rnd) for s in seeds]
        arms[name] = [o[0] for o in out]
        print(f"  {name:<18} {statistics.mean(arms[name]):.3f}  "
              f"(sd {statistics.pstdev(arms[name]):.3f})   "
              f"dialog choices a run {statistics.mean(o[1] for o in out):.2f}")

    def paired(label, a, b):
        diff = [x - y for x, y in zip(arms[a], arms[b])]
        m = statistics.mean(diff)
        se = statistics.stdev(diff) / len(diff) ** 0.5
        print(f"  {label}: {m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}]")

    print()
    paired("dialogs, heuristic against none", "heuristic chooses", "no dialogs")
    paired("choosing, heuristic against random", "heuristic chooses", "random choices")


if __name__ == "__main__":
    main()
