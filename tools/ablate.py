"""Turn one `RunState` switch off and measure the heuristic, paired by seed.

The switches are the faithful behaviours that kept their old form for exactly
this: `carry_hp` (wounds between fights), `pack_layouts` (each pack in its own
layout, by zone), `dialogs_on` (the level-entry dialogs) and `quests_on` (the
quests, their rooms included). The heuristic plays the same seeds with the
switch on and off.

This is a check that a feature changes something, not a re-baseline. RL numbers
are taken at a batch boundary; see `notes/roadmap.md`.

    python tools/ablate.py pack_layouts --seeds 240 --fast
    python tools/ablate.py quests_on --seeds 2400 --start 40000 --fast --jobs 8

240 seeds is too few for an effect under about 0.2 levels: on `quests_on`,
seeds 30000 to 30239 read -0.358 [-0.515, -0.202] where 2400 seeds from 40000
read -0.072 [-0.124, -0.020], and ten 240-seed windows of those 2400 spread
from 0.000 to -0.154. `--jobs` splits the seeds over processes, so the larger
sample costs about a minute with `--fast`.
"""
import argparse
import multiprocessing
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


def _chunk(job):
    """Both arms over one slice of the seeds, in a process of its own."""
    switch, seeds, fast = job
    env = DespotRunEnv(tables=load_ruleset(strict=True), fast_core=fast)
    on = [play(env, s, switch, True) for s in seeds]
    off = [play(env, s, switch, False) for s in seeds]
    return on, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("switch", choices=SWITCHES)
    ap.add_argument("--seeds", type=int, default=240)
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--fast", action="store_true", help="resolve fights in the Rust core")
    ap.add_argument("--jobs", type=int, default=1, help="processes to split the seeds over")
    args = ap.parse_args()

    seeds = list(range(args.start, args.start + args.seeds))
    # Every run starts from its own seed, so a slice gives the same levels in
    # any process and in any order.
    jobs = [(args.switch, seeds[i::args.jobs], args.fast) for i in range(args.jobs)]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            parts = pool.map(_chunk, jobs)
    else:
        parts = [_chunk(jobs[0])]
    on = [x for part in parts for x in part[0]]
    off = [x for part in parts for x in part[1]]
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
