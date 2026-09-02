"""Does the same command with the same seed produce the same agent?

`shrine2m_s*` and `shelf2m_described_s*` are the same command with the same
seeds, and ten of twelve diverge (see notes/rl.md, "Training does not reproduce
from `--seed` alone"). The hypothesis on record is that the divergence depends
on how many runs are training at once. This is the test.

Two conditions, same seed, same flags, same `OMP_NUM_THREADS=1` the sweep
workers use:

    alone        n runs of one seed, one at a time
    concurrent   n runs of the same seed, all at once

If `alone` reproduces and `concurrent` does not, the nondeterminism is
concurrency-dependent. If neither reproduces, it is unconditional and every
paired comparison in this project is pairing on the init draw alone. If both
reproduce, the original divergence came from something other than the seed and
the launch, and the two sweeps were not the same command after all.

    python tools/determinism_probe.py --steps 200000 --copies 4

Runs are written to `runs/det_<condition>_<i>.pt` and left there; they are small
and the comparison is worth being able to repeat.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

# Exactly what `tools/sweep_arms.py` gives its workers, so the test reproduces
# the conditions the divergence was observed under rather than new ones.
WORKER_ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
PY_EXE = sys.executable


def train(out: pathlib.Path, seed: int, steps: int, fast_core: bool) -> float:
    cmd = [PY_EXE, "rl/train.py", "--steps", str(steps), "--shaping", "none",
           "--seed", str(seed), "--out", str(out)]
    if fast_core:
        cmd.append("--fast-core")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=WORKER_ENV)
    if p.returncode != 0:
        raise SystemExit(f"{out} failed:\n{(p.stderr or '')[-600:]}")
    return time.perf_counter() - t0


def weights(path: pathlib.Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def compare(paths: list[pathlib.Path]) -> tuple[int, float]:
    """How many distinct agents came out, and the largest weight difference."""
    ws = [weights(p) for p in paths]
    groups: list[int] = []
    worst = 0.0
    for i, w in enumerate(ws):
        for g in groups:
            if all(torch.equal(w[k], ws[g][k]) for k in w):
                break
        else:
            groups.append(i)
        for j in range(i):
            d = max(float((w[k] - ws[j][k]).abs().max()) for k in w)
            worst = max(worst, d)
    return len(groups), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--copies", type=int, default=4,
                    help="runs of the same seed per condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fast-core", action="store_true",
                    help="run the Python oracle instead, to see whether the "
                         "Rust core is what differs")
    args = ap.parse_args()
    fast_core = not args.no_fast_core
    tag = "oracle" if args.no_fast_core else "rust"

    print(f"{args.copies} copies of seed {args.seed}, {args.steps:,} steps, "
          f"{'Rust core' if fast_core else 'Python oracle'}\n")

    alone = [pathlib.Path(f"runs/det_{tag}_alone_{i}.pt")
             for i in range(args.copies)]
    conc = [pathlib.Path(f"runs/det_{tag}_conc_{i}.pt")
            for i in range(args.copies)]

    t0 = time.perf_counter()
    for i, out in enumerate(alone):
        dt = train(out, args.seed, args.steps, fast_core)
        print(f"  alone      {i}  {dt / 60:4.1f} min")
    print(f"  -> {(time.perf_counter() - t0) / 60:.1f} min\n")

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.copies) as pool:
        futures = [pool.submit(train, out, args.seed, args.steps, fast_core)
                   for out in conc]
        for i, f in enumerate(cf.as_completed(futures)):
            print(f"  concurrent {i}  {f.result() / 60:4.1f} min")
    print(f"  -> {(time.perf_counter() - t0) / 60:.1f} min\n")

    n_alone, d_alone = compare(alone)
    n_conc, d_conc = compare(conc)
    n_all, d_all = compare(alone + conc)

    print(f"{'condition':12s} {'distinct agents':>16s} {'max weight delta':>18s}")
    print(f"{'alone':12s} {n_alone:>10d} of {args.copies:<3d} {d_alone:>18.4g}")
    print(f"{'concurrent':12s} {n_conc:>10d} of {args.copies:<3d} {d_conc:>18.4g}")
    print(f"{'both':12s} {n_all:>10d} of {2 * args.copies:<3d} {d_all:>18.4g}")

    print()
    if n_all == 1:
        print("Reproducible in both conditions: the seed and the command fully "
              "determine the agent, so the shrine2m/shelf2m divergence came "
              "from something else that changed between those two launches.")
    elif n_alone == 1 and n_conc > 1:
        print("Concurrency-dependent: identical alone, divergent when run "
              "together. The pairing in every sweep pairs the init draw only.")
    else:
        print("Unconditional: the same command with the same seed does not "
              "reproduce even one at a time.")


if __name__ == "__main__":
    main()
