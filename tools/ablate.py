"""Turn one `RunState` switch off and measure the heuristic, paired by seed.

The switches are the faithful behaviours that kept their old form for exactly
this: `carry_hp` (wounds between fights), `pack_layouts` (each pack in its own
layout, by zone), `dialogs_on` (the level-entry dialogs) and `quests_on` (the
quests, their rooms included). The heuristic plays the same seeds with the
switch on and off.

This is a check that a feature changes something, not a re-baseline. RL numbers
are taken at a batch boundary; see `notes/roadmap.md`.

    python tools/ablate.py pack_layouts --seeds 240 --fast
"""
import argparse
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rl.env import DespotRunEnv
from rl.heuristic import heuristic
from sim.data import load_ruleset

SWITCHES = ("carry_hp", "pack_layouts", "dialogs_on", "quests_on")


def play(env, seed, switch, value):
    if switch == "dialogs_on":
        env.dialogs = value
    if switch == "quests_on":
        env.quests = value
    obs, info = env.reset(seed=seed)
    setattr(env.state, switch, value)
    mask = info["action_mask"]
    while mask.any():
        obs, _, term, trunc, info = env.step(heuristic(env, obs, mask))
        mask = info["action_mask"]
        if term or trunc:
            break
    return env.state.level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("switch", choices=SWITCHES)
    ap.add_argument("--seeds", type=int, default=240)
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--fast", action="store_true", help="resolve fights in the Rust core")
    args = ap.parse_args()

    env = DespotRunEnv(tables=load_ruleset(strict=True), fast_core=args.fast)
    seeds = range(args.start, args.start + args.seeds)
    on = [play(env, s, args.switch, True) for s in seeds]
    off = [play(env, s, args.switch, False) for s in seeds]
    print(f"  {args.switch} on   {statistics.mean(on):.3f}  (sd {statistics.pstdev(on):.3f})")
    print(f"  {args.switch} off  {statistics.mean(off):.3f}  (sd {statistics.pstdev(off):.3f})")
    diff = [a - b for a, b in zip(on, off)]
    m = statistics.mean(diff)
    se = statistics.stdev(diff) / len(diff) ** 0.5
    print(f"\n  on against off, paired by seed: {m:+.3f} [{m - 1.96 * se:+.3f}, "
          f"{m + 1.96 * se:+.3f}], better on {sum(1 for d in diff if d > 0)} of "
          f"{len(diff)}, worse on {sum(1 for d in diff if d < 0)}")


if __name__ == "__main__":
    main()
