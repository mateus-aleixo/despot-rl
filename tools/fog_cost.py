"""What the fog of war costs a player, measured by playing with it off.

The env's `lights_on` reveals the whole level at an unchanged `obs_dim`, which
is the observation this project had before `C_Rooms.SetCurrent` was modelled. So
the same policy on the same seeds, once each way, is a paired measurement of the
information itself rather than of anything else.

This is a check that the feature changes something, not a re-baseline. RL
numbers are taken at a batch boundary; see `notes/roadmap.md`.

    python tools/fog_cost.py --seeds 60
"""
import argparse
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rl.env import DespotRunEnv
from rl.heuristic import heuristic
from sim.data import load_ruleset


def play(env, seed):
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    while mask.any():
        obs, _, term, trunc, info = env.step(heuristic(env, obs, mask))
        mask = info["action_mask"]
        if term or trunc:
            break
    return env.state.level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    arms = {}
    for name, lit in (("fog", False), ("lights on", True)):
        env = DespotRunEnv(tables=tables, lights_on=lit)
        arms[name] = [play(env, s) for s in range(args.seeds)]
        print(f"  {name:<10} {statistics.mean(arms[name]):.3f}  "
              f"(sd {statistics.pstdev(arms[name]):.3f})")

    fog, lit = arms["fog"], arms["lights on"]
    diff = [a - b for a, b in zip(lit, fog)]
    print(f"\n  paired {statistics.mean(diff):+.3f} levels of free information, "
          f"lights on better on {sum(1 for d in diff if d > 0)} of {len(diff)}, "
          f"worse on {sum(1 for d in diff if d < 0)}")


if __name__ == "__main__":
    main()
