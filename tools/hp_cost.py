"""What carrying wounds between fights costs the baseline.

In Default a unit keeps the damage it took until something heals it; the sim
used to rebuild every unit at full health for every fight (see `Human.hp`).
`RunState.carry_hp` switches between the two, so the same heuristic over the
same seeds, once each way, measures what the free heal was worth. The maps and
the first fight of every run are identical; the arms part at the second fight.

This is a check that the feature changes something, not a re-baseline. RL
numbers are taken at a batch boundary; see `notes/roadmap.md`.

    python tools/hp_cost.py --seeds 240 --fast
"""
import argparse
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rl.env import DespotRunEnv
from rl.heuristic import heuristic
from sim.data import load_ruleset


def play(env, seed, carry):
    obs, info = env.reset(seed=seed)
    env.state.carry_hp = carry
    mask = info["action_mask"]
    worst = 1.0
    while mask.any():
        obs, _, term, trunc, info = env.step(heuristic(env, obs, mask))
        mask = info["action_mask"]
        if env.state.squad:
            worst = min(worst, min(h.hp for h in env.state.squad))
        if term or trunc:
            break
    return env.state.level, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=240)
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--fast", action="store_true", help="resolve fights in the Rust core")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    env = DespotRunEnv(tables=tables, fast_core=args.fast)
    seeds = range(args.start, args.start + args.seeds)
    arms = {}
    for name, carry in (("wounds carried", True), ("healed every fight", False)):
        out = [play(env, s, carry) for s in seeds]
        arms[name] = [o[0] for o in out]
        print(f"  {name:<20} {statistics.mean(arms[name]):.3f}  "
              f"(sd {statistics.pstdev(arms[name]):.3f})   "
              f"lowest health a run, mean {statistics.mean(o[1] for o in out):.2f}")
    diff = [a - b for a, b in zip(arms["wounds carried"], arms["healed every fight"])]
    m = statistics.mean(diff)
    se = statistics.stdev(diff) / len(diff) ** 0.5
    print(f"\n  wounds, paired by seed: {m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}], "
          f"better on {sum(1 for d in diff if d > 0)} of {len(diff)}, "
          f"worse on {sum(1 for d in diff if d < 0)}")


if __name__ == "__main__":
    main()
