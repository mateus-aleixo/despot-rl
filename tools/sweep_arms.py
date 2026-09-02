"""The twelve-seed sweep: two arms of twelve runs, paired on the init seed.

    python tools/sweep_arms.py train [--sweep passives] [--steps 600000]
                                     [--jobs 8] [--tag arm2] [--checkpoint-every N]
    python tools/sweep_arms.py score [--sweep passives] [--runs 240] [--tag arm2]

`--sweep passives` (the default, and the original) varies whether the mutation
hooks are live *during training*: `--without-passives` is held for the whole run
of the inert arm, and **both arms are graded on the live environment**, so they
differ in what they were trained against and never in what they are scored on.
That is the right grading because the arms differ in the environment's dynamics.

`--sweep shelf` varies only the *information*: `--blind-shelf` holds the five
description floats and the passive bit at zero for every mutation slot, leaving
`present`, the takes left and the shrine bit -- the vector the agents saw before
`mutshelf` described the shelf, at an unchanged 195 dims and an unchanged first
layer. Here the dynamics are identical in both arms, so **each arm is graded on
the vector it trained on**: handing the blind agents a description they have
never seen would measure a distribution shift rather than the value of the
information. This is the `--blind-shrine` protocol, applied to the feature the
shrine bit was a footnote to.

Init seed is shared across the arms, so the twelve differences are paired.

`train` writes runs/<tag>_<arm>_s0..s11.pt and refuses to overwrite an existing
file unless --overwrite is passed: the previous sweep's agents are the only copy
of a number that took 18 minutes and cannot be reproduced once the environment
moves under them. Pass `--checkpoint-every 600000` on a 2M sweep so the budget
curve comes off one run per seed with no init or sampling difference in it.
"""
import argparse
import concurrent.futures as cf
import os
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEEDS = range(12)
# arms, the train flag that defines each, and the environment each is scored in.
# The empty env dict is the plain live environment; see the module docstring for
# why `shelf` scores its arms differently from `passives`.
SWEEPS = {
    "passives": {
        "arms": ("live", "inert"),
        "flags": {"live": [], "inert": ["--without-passives"]},
        "env": {"live": {}, "inert": {}},
    },
    "shelf": {
        "arms": ("described", "blind"),
        "flags": {"described": [], "blind": ["--blind-shelf"]},
        "env": {"described": {}, "blind": {"blind_shelf": True}},
    },
    # One arm, because the question is the budget rather than a difference
    # between two environments: twelve seeds of the plain live environment,
    # scored at each `--checkpoint-every` mark with `score --at`. Nothing in
    # `rl/train.py` is scheduled off `--steps` (fixed LR, fixed entropy), so a
    # long run's checkpoint is the same trajectory a short run would have had at
    # that mark, and the curve carries no init or sampling difference.
    "budget": {
        "arms": ("long",),
        "flags": {"long": []},
        "env": {"long": {}},
    },
}
PY_EXE = sys.executable

# One torch thread per worker. The net here is a two-layer MLP on batches of a
# few hundred, so intra-op parallelism buys nothing and every worker otherwise
# claims all sixteen cores: six concurrent runs measured 3,302 steps/s
# aggregate at the default against **6,206 with this set**, and a lone run 1,815
# against 2,077. The updates are identical either way -- same mean level and
# return at every update -- so this is speed only, and it also keeps `--jobs 6`
# meaning six busy cores rather than the whole machine.
WORKER_ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def out_path(tag: str, arm: str, seed: int, at: int = 0) -> pathlib.Path:
    """The run's final agent, or its checkpoint at `at` steps.

    `rl/train.py --checkpoint-every N` writes `<out>.<steps>.pt` alongside the
    final `<out>.pt`, so `score --at 1000000` grades the budget curve off the
    same runs rather than a second set of shorter ones.
    """
    stem = f"{tag}_{arm}_s{seed}"
    return pathlib.Path("runs") / (f"{stem}.{at}.pt" if at else f"{stem}.pt")


def train_one(tag: str, sweep: str, arm: str, seed: int, steps: int,
              checkpoint_every: int = 0) -> tuple[str, float, str]:
    out = out_path(tag, arm, seed)
    cmd = [PY_EXE, "rl/train.py", "--steps", str(steps), "--fast-core",
           "--shaping", "none", "--seed", str(seed), "--out", str(out)]
    if checkpoint_every:
        cmd += ["--checkpoint-every", str(checkpoint_every)]
    cmd += SWEEPS[sweep]["flags"][arm]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=WORKER_ENV)
    dt = time.perf_counter() - t0
    tail = (p.stdout or "").strip().splitlines()
    note = tail[-1] if p.returncode == 0 and tail else (p.stderr or "").strip()[-300:]
    return f"{arm}_s{seed}", dt, ("" if p.returncode == 0 else f"FAILED: {note}")


def do_train(args) -> None:
    arms = SWEEPS[args.sweep]["arms"]
    jobs = [(arm, s) for s in SEEDS for arm in arms]
    if args.resume:
        # A run is finished when its unsuffixed .pt exists; the intermediate
        # `<out>.<steps>.pt` checkpoints do not count, because a run killed at
        # 1.8M has them and is still a run that has to be redone from zero.
        done = [(a, s) for a, s in jobs if out_path(args.tag, a, s).exists()]
        jobs = [j for j in jobs if j not in done]
        print(f"resuming: {len(done)} of {len(done) + len(jobs)} already saved")
    if not args.overwrite:
        clash = [str(out_path(args.tag, a, s)) for a, s in jobs
                 if out_path(args.tag, a, s).exists()]
        if clash:
            sys.exit(f"{len(clash)} of these already exist, starting with "
                     f"{clash[0]} -- pass --overwrite or pick another --tag")
    print(f"{len(jobs)} runs of {args.steps:,} steps, {args.jobs} concurrent")
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(train_one, args.tag, args.sweep, a, s, args.steps,
                               args.checkpoint_every)
                   for a, s in jobs]
        for done, f in enumerate(cf.as_completed(futures), 1):
            name, dt, err = f.result()
            print(f"  [{done:2d}/{len(jobs)}] {name:10s} {dt / 60:5.1f} min  {err}")
    print(f"\nwall clock {(time.perf_counter() - t0) / 60:.1f} min for "
          f"{len(jobs) * args.steps / 1e6:.1f}M steps")


def do_score(args) -> None:
    # Imported here so `train` does not pay for a ruleset load it never uses,
    # and so the scoring environment is built once, live, for every agent.
    from rl.env import MOVES, NON_MOVE_ACTIONS, DespotRunEnv
    from rl.heuristic import heuristic as heuristic_policy
    from rl.placement import POLICIES
    from sim.data import load_ruleset
    from tools.hierarchy_eval import StaleCheckpoint, make_ppo

    # The shrine's free mutation, by fixed index: the denominator for uptake is
    # the steps where taking it was legal, which is what the heuristic scores
    # 100% on and every trained arm has scored 0.2% to 29.2% on.
    TAKE_MUTATION = len(MOVES) + NON_MOVE_ACTIONS.index("take_mutation")

    tables = load_ruleset(strict=True)
    placement = POLICIES["frontline"]
    seeds = list(range(30_000, 30_000 + args.runs))
    arms = SWEEPS[args.sweep]["arms"]
    arm_env = SWEEPS[args.sweep]["env"]

    def play(policy, env_kwargs: dict | None = None) -> tuple[list[int], dict]:
        """Levels per seed, and what the policy did with the mutation shop.

        The level difference between two arms is unreadable until both arms
        engage with the thing they differ in: at 600k the passives sweep held
        0.34 against 0.38 mutations and its null said nothing about mutations.
        `offered` counts the steps where `take_mutation` was legal, so `taken`
        over it is uptake on the same denominator the heuristic scores 100% on.
        """
        levels, held, any_mut = [], [], 0
        offered = taken = 0
        for s in seeds:
            env = DespotRunEnv(tables=tables, seed=s, placement_policy=placement,
                               fast_core=True, **(env_kwargs or {}))
            obs, info = env.reset(seed=s)
            mask, n, got = info["action_mask"], 0, 0
            while mask.any() and n < env.max_steps:
                a = policy(env, obs, mask)
                kind, _ = env._decode(a)
                if mask[TAKE_MUTATION]:
                    offered += 1
                    taken += 1 if a == TAKE_MUTATION else 0
                if kind in ("take_mutation", "buy_mutation") and mask[a]:
                    got += 1
                obs, r, term, trunc, info = env.step(a)
                mask, n = info["action_mask"], n + 1
                if term or trunc:
                    break
            levels.append(env.state.level)
            held.append(len(env.state.mutations))
            any_mut += 1 if got else 0
        return levels, {"held": statistics.mean(held),
                        "uptake": taken / offered if offered else float("nan"),
                        "any": any_mut / len(seeds)}

    t0 = time.perf_counter()
    base, base_beh = play(heuristic_policy)
    print(f"heuristic  {statistics.mean(base):.3f} over {len(seeds)} seeds "
          f"({time.perf_counter() - t0:.0f}s)  held {base_beh['held']:.2f}, "
          f"take_mutation {base_beh['uptake']:.1%}, "
          f"any mutation {base_beh['any']:.1%}")

    probe = DespotRunEnv(tables=tables, fast_core=True)
    means: dict[str, dict[int, float]] = {a: {} for a in arms}
    per_seed: dict[str, dict[int, list[int]]] = {a: {} for a in arms}
    behaviour: dict[str, list[dict]] = {a: [] for a in arms}
    for seed in SEEDS:
        for arm in arms:
            path = out_path(args.tag, arm, seed, args.at)
            try:
                policy = make_ppo(str(path), probe)
            except (FileNotFoundError, StaleCheckpoint) as exc:
                print(f"  skipping {path}: {exc}")
                continue
            lv, beh = play(policy, arm_env[arm])
            per_seed[arm][seed] = lv
            means[arm][seed] = statistics.mean(lv)
            behaviour[arm].append(beh)
            print(f"  {arm:9s} s{seed:<2d} {means[arm][seed]:.3f}  "
                  f"held {beh['held']:.2f}, take {beh['uptake']:.1%}, "
                  f"any {beh['any']:.1%}")

    # A one-arm sweep (`--sweep budget`) has nothing to pair against: the
    # comparison there is between checkpoints of the same runs, made by scoring
    # the tag at several `--at` marks, so print the per-seed column and skip the
    # paired block rather than unpacking two arms that do not exist.
    paired: list[float] = []
    if len(arms) >= 2:
        a, b = arms[0], arms[1]
        paired = [means[a][s] - means[b][s] for s in SEEDS
                  if s in means[a] and s in means[b]]
        print(f"\n| init seed | {a} | {b} | {a} minus {b} |")
        print("|---|---|---|---|")
        for s in SEEDS:
            if s in means[a] and s in means[b]:
                print(f"| {s} | {means[a][s]:.2f} | {means[b][s]:.2f} | "
                      f"{means[a][s] - means[b][s]:+.2f} |")
    else:
        only = arms[0]
        print(f"\n| init seed | {only} |")
        print("|---|---|")
        for s in SEEDS:
            if s in means[only]:
                print(f"| {s} | {means[only][s]:.3f} |")

    def ci95(xs):
        half = 1.96 * statistics.stdev(xs) / len(xs) ** 0.5
        return statistics.mean(xs) - half, statistics.mean(xs) + half

    print()
    for arm in arms:
        if means[arm]:
            print(f"{arm + ' arm':11s} {statistics.mean(means[arm].values()):.3f}")
    print(f"{'heuristic':11s} {statistics.mean(base):.3f}")
    if len(paired) > 1:
        lo, hi = ci95(paired)
        print(f"\npaired difference, per training seed   "
              f"{statistics.mean(paired):+.2f} [{lo:+.2f}, {hi:+.2f}]")
        # Per eval seed: the same difference resolved seed by seed, which is a
        # tighter interval on a sample that is not independent -- twelve agents
        # an arm share it -- so it is quoted as the lower bound it is.
        by_eval = [statistics.mean(per_seed[a][s][i] - per_seed[b][s][i]
                                   for s in SEEDS if s in per_seed[a])
                   for i in range(len(seeds))]
        flat = [per_seed[a][s][i] - per_seed[b][s][i]
                for s in SEEDS if s in per_seed[a] for i in range(len(seeds))]
        lo, hi = ci95(flat)
        print(f"per eval seed (n={len(flat):,}, not independent) "
              f"{statistics.mean(flat):+.2f} [{lo:+.2f}, {hi:+.2f}]")

    print(f"\n| arm | mutations held | `take_mutation` | any mutation |")
    print("|---|---|---|---|")
    for arm in list(arms) + ["heuristic"]:
        b = base_beh if arm == "heuristic" else None
        if b is None:
            if not behaviour[arm]:
                continue
            b = {k: statistics.mean(x[k] for x in behaviour[arm])
                 for k in ("held", "uptake", "any")}
        print(f"| {arm} | {b['held']:.2f} | {b['uptake']:.1%} | {b['any']:.1%} |")

    every = [m for arm in arms for m in means[arm].values()]
    if len(every) > 1:
        print(f"\nstandard deviation across {len(every)} identically configured "
              f"agents: {statistics.stdev(every):.3f} levels "
              f"({min(every):.2f} to {max(every):.2f})")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
t = sub.add_parser("train")
t.add_argument("--sweep", default="passives", choices=sorted(SWEEPS))
t.add_argument("--steps", type=int, default=600_000)
t.add_argument("--jobs", type=int, default=8)
t.add_argument("--tag", default="arm2")
t.add_argument("--checkpoint-every", type=int, default=0)
t.add_argument("--overwrite", action="store_true")
t.add_argument("--resume", action="store_true",
               help="skip seeds whose final .pt is already saved, so a sweep "
                    "stopped part way does not retrain what it finished")
t.set_defaults(fn=do_train)
c = sub.add_parser("score")
c.add_argument("--sweep", default="passives", choices=sorted(SWEEPS))
c.add_argument("--runs", type=int, default=240)
c.add_argument("--tag", default="arm2")
c.add_argument("--at", type=int, default=0,
               help="score the `<out>.<steps>.pt` checkpoint at this many "
                    "steps instead of the final agent, for a budget curve off "
                    "one run per seed")
c.set_defaults(fn=do_score)
args = ap.parse_args()
args.fn(args)
