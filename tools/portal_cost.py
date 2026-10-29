"""What the portals, and the faithful `PositionRooms`, change for the baseline.

Three arms, the same heuristic, the same seeds:

  * `faithful`: levels as `sim/mapgen.py` now places them, portals included.
  * `no portals`: the same levels with the portals taken off. `ChoosePortals`
    draws nothing from the random stream, so dropping its output leaves every
    other room exactly where it was; the two arms differ in the portals and in
    nothing else until the first teleport.
  * `old placement` (with `--old PATH`): the `sim/mapgen.py` from before the
    port, which maximised the boss's average distance and put the start at the
    far end, spread shops against every placed room, placed no portals and gave
    no start room a shop. Its random stream differs, so it is paired by seed
    only.

This is a check that the features change something, not a re-baseline. RL
numbers are taken at a batch boundary; see `notes/roadmap.md`.

    python tools/portal_cost.py --seeds 120
    git show <rev>:sim/mapgen.py > old_mapgen.py
    python tools/portal_cost.py --seeds 120 --old old_mapgen.py
"""
import argparse
import importlib.util
import statistics
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sim.mapgen as mapgen
from rl.env import DespotRunEnv
from rl.heuristic import heuristic
from sim.data import load_ruleset

FAITHFUL = mapgen.generate


def no_portals(level_row, rng, chip="default"):
    level = FAITHFUL(level_row, rng, chip)
    level["portals"] = []
    return level


def play(env, seed):
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    teleports = steps = 0
    while mask.any():
        action = heuristic(env, obs, mask)
        teleports += env._decode(action)[0] == "teleport"
        steps += 1
        obs, _, term, trunc, info = env.step(action)
        mask = info["action_mask"]
        if term or trunc:
            break
    return env.state.level, teleports, steps


def arm(name, generate, tables, seeds, fast):
    mapgen.generate = generate
    try:
        env = DespotRunEnv(tables=tables, fast_core=fast)
        out = [play(env, s) for s in seeds]
    finally:
        mapgen.generate = FAITHFUL
    levels = [o[0] for o in out]
    print(f"  {name:<14} {statistics.mean(levels):.3f}  "
          f"(sd {statistics.pstdev(levels):.3f})   "
          f"teleports a run {statistics.mean(o[1] for o in out):.2f}, "
          f"in {sum(1 for o in out if o[1])} of {len(out)} runs   "
          f"steps a run {statistics.mean(o[2] for o in out):.1f}")
    return levels


def paired(label, a, b):
    diff = [x - y for x, y in zip(a, b)]
    m = statistics.mean(diff)
    se = statistics.stdev(diff) / len(diff) ** 0.5 if len(diff) > 1 else 0.0
    print(f"  {label}: {m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}], "
          f"better on {sum(1 for d in diff if d > 0)} of {len(diff)}, "
          f"worse on {sum(1 for d in diff if d < 0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--old", help="a copy of an earlier sim/mapgen.py")
    ap.add_argument("--fast", action="store_true", help="resolve fights in the Rust core")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    seeds = range(args.start, args.start + args.seeds)
    faithful = arm("faithful", FAITHFUL, tables, seeds, args.fast)
    stripped = arm("no portals", no_portals, tables, seeds, args.fast)
    print()
    paired("portals, paired", faithful, stripped)
    if args.old:
        spec = importlib.util.spec_from_file_location("old_mapgen", args.old)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        legacy = arm("old placement", lambda row, rng, chip="default": old.generate(row, rng),
                     tables, seeds, args.fast)
        paired("faithful against the old placement, by seed", faithful, legacy)


if __name__ == "__main__":
    main()
