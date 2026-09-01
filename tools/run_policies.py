"""Does the run layer reward playing better? Random vs a simple heuristic."""
import statistics, sys, time
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rl.heuristic import heuristic_action
from sim.data import load_ruleset
from sim.run import RunState

t = load_ruleset(strict=True)


def random_policy(st):
    return st.rng.choice(st.legal_actions())


def heuristic_policy(st):
    """The shared run-level baseline; see `rl.heuristic`."""
    return heuristic_action(st)


def play(policy, seed, max_steps=600):
    st = RunState.new(t, seed=seed)
    steps = fights = wins = 0
    while not st.finished and steps < max_steps:
        acts = st.legal_actions()
        if not acts:
            break
        r = st.apply(policy(st))
        if "won" in r:
            fights += 1; wins += bool(r["won"])
        steps += 1
    return st, fights, wins


N = 30
for name, pol in (("random", random_policy), ("heuristic", heuristic_policy)):
    t0 = time.perf_counter()
    levels, survived, fights, wins = [], 0, 0, 0
    for s in range(N):
        st, f, w = play(pol, s)
        levels.append(st.level); survived += bool(st.squad); fights += f; wins += w
    el = time.perf_counter() - t0
    print(f"{name:10s} level reached: mean {statistics.mean(levels):.2f} "
          f"median {statistics.median(levels):.1f} max {max(levels)}  "
          f"squads alive {survived}/{N}  fights {fights} won {wins} "
          f"({wins/max(1,fights)*100:.0f}%)  {el/N:.2f}s/run")
