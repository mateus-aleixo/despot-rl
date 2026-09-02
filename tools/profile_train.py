"""Where does training wall clock actually go?

The Rust core exports a batched entry point (`despot_battle_batch`, bound as
`sim.fast.fast_batch`) that `tools/diff_core.py` measures at 0.29 ms a battle
against 1.9 ms for one call at a time. Training never uses it: `rl.train.rollout`
steps its envs in a Python loop, so every fight is its own call. Making training
batch its fights means letting `env.step` defer a fight and resume after the
batch returns, which is a change to the env contract, so it is worth knowing
what share of the clock the fights are before paying for it.

This wraps three call sites and then runs the real `rl.train.main`, so it
profiles the actual training loop rather than a reimplementation of it:

    rollout            `rl.train.rollout`, one call per update
    env step           `DespotRunEnv.step`, inside a rollout only
    core               `sim.fast.fast_battle`, inside a rollout only

Everything between one rollout ending and the next starting is the PPO update
plus its bookkeeping. The accounting window runs from the first rollout to the
last, so it holds whole (rollout, update) cycles and never a half one; the
warm-up baseline and the final evaluation are outside it by construction.

    python tools/profile_train.py --steps 100000

Timer overhead is a `perf_counter` pair per call, nanoseconds against the
microseconds being measured, so unlike cProfile it does not inflate the Python
side against the single ctypes call into the core.
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rl.env as env_mod
import rl.train as train_mod
import sim.fast as fast_mod
import sim.run as run_mod

PER: list[dict] = []
CUR: dict | None = None


def _inner(name: str, func):
    """Time a method into `CUR[name]`, ignoring calls outside a rollout."""
    def wrapped(*a, **kw):
        if CUR is None:
            return func(*a, **kw)
        t0 = time.perf_counter()
        try:
            return func(*a, **kw)
        finally:
            CUR[name] += time.perf_counter() - t0
    return wrapped


def _wrap():
    real_rollout = train_mod.rollout
    real_step = env_mod.DespotRunEnv.step
    real_battle = fast_mod.fast_battle

    # Inside `env.step`: the observation, the legality scan (twice a step, once
    # to check the action and once for the info dict) and the state transition.
    env_mod.DespotRunEnv._encode = _inner("encode", env_mod.DespotRunEnv._encode)
    env_mod.DespotRunEnv.action_mask = _inner(
        "mask", env_mod.DespotRunEnv.action_mask)
    run_mod.RunState.apply = _inner("apply", run_mod.RunState.apply)

    def rollout(*a, **kw):
        global CUR
        CUR = {"env": 0.0, "core": 0.0, "fights": 0,
               "encode": 0.0, "mask": 0.0, "apply": 0.0}
        start = time.perf_counter()
        try:
            return real_rollout(*a, **kw)
        finally:
            CUR["start"], CUR["end"] = start, time.perf_counter()
            PER.append(CUR)
            CUR = None

    def step(self, action):
        if CUR is None:
            return real_step(self, action)
        t0 = time.perf_counter()
        try:
            return real_step(self, action)
        finally:
            CUR["env"] += time.perf_counter() - t0

    def battle(*a, **kw):
        if CUR is None:
            return real_battle(*a, **kw)
        t0 = time.perf_counter()
        try:
            return real_battle(*a, **kw)
        finally:
            CUR["core"] += time.perf_counter() - t0
            CUR["fights"] += 1

    train_mod.rollout = rollout
    env_mod.DespotRunEnv.step = step
    fast_mod.fast_battle = battle


def report():
    if len(PER) < 3:
        sys.exit("too few updates to report on; raise --steps")
    # Whole cycles only: from the first rollout's start to the last one's, so
    # every rollout counted has its update counted after it.
    window = PER[-1]["start"] - PER[0]["start"]
    done = PER[:-1]
    roll = sum(p["end"] - p["start"] for p in done)
    envt = sum(p["env"] for p in done)
    core = sum(p["core"] for p in done)
    fights = sum(p["fights"] for p in done)

    policy = roll - envt          # torch inference plus buffer writes
    py_env = envt - core          # the run layer: shops, economy, encode, mapgen
    update = window - roll        # PPO epochs, GAE, checkpointing, logging

    encode = sum(p["encode"] for p in done)
    mask = sum(p["mask"] for p in done)
    apply_ = sum(p["apply"] for p in done) - core   # apply() contains the fight
    rest = py_env - encode - mask - apply_

    rows = [
        ("rollout", roll, None),
        ("  policy forward + buffers", policy, None),
        ("  env.step, Python run layer", py_env, None),
        ("    _encode, the observation", encode, None),
        ("    action_mask, twice a step", mask, None),
        ("    RunState.apply, minus fight", apply_, None),
        ("    the rest of step()", rest, None),
        ("  env.step, Rust battle core", core, f"{fights:,} fights"),
        ("PPO update + bookkeeping", update, None),
    ]
    print(f"\n{len(done)} complete update cycles, {window:.1f}s of wall clock\n")
    print(f"{'part':32s} {'seconds':>9s} {'share':>7s}")
    for name, secs, note in rows:
        print(f"{name:32s} {secs:9.1f} {secs / window:7.1%}"
              + (f"   {note}" if note else ""))

    per_fight = core / fights * 1000 if fights else float("nan")
    print(f"\nmean {per_fight:.2f} ms per fight in the core")

    # What batching could buy, at the 0.29ms vs 1.9ms the core's own benchmark
    # measured, applied only to the part of the clock that is fights.
    for factor in (2.0, 6.5):
        saved = core * (1 - 1 / factor)
        print(f"a {factor:.1f}x faster fight would cut {saved / window:.1%} of "
              f"the loop  ->  {window / (window - saved):.2f}x overall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/_profile_scratch.pt")
    ap.add_argument("--train-args", nargs=argparse.REMAINDER, default=[],
                    help="anything else to hand rl/train.py verbatim")
    args = ap.parse_args()

    _wrap()
    sys.argv = ["rl/train.py", "--steps", str(args.steps), "--fast-core",
                "--shaping", "none", "--seed", str(args.seed),
                "--out", args.out] + args.train_args
    train_mod.main()
    report()


if __name__ == "__main__":
    main()
