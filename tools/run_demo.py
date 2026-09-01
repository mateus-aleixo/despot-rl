"""Drive a full run with a random policy and report what happened."""
import collections, random, sys, time
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sim.data import load_ruleset
from sim.run import RoomMap, RunState

t = load_ruleset(strict=True)
rm = RoomMap.from_table(t["Rooms"])
print(f"room map: {len(rm.rooms)} rooms, start={rm.start} boss={rm.boss} portals={rm.portals}")
kinds = collections.Counter(r.kind for r in rm.rooms.values())
print("room kinds:", dict(kinds))
print("neighbours of start:", rm.neighbours(rm.start))

def play(seed, max_steps=400, verbose=False):
    st = RunState.new(t, seed=seed)
    steps = 0
    fights = wins = 0
    while not st.finished and steps < max_steps:
        acts = st.legal_actions()
        if not acts:
            break
        a = st.rng.choice(acts)
        r = st.apply(a)
        if "won" in r:
            fights += 1; wins += bool(r["won"])
        if verbose and steps < 14:
            print(f"   {steps:3d} {a[0]:14s} lvl={st.level} room={st.room:3s} "
                  f"gold={st.gold:5.0f} food={st.food.amount:5.0f} "
                  f"hunger={st.food.hunger_level} moves={st.food.moves_left} "
                  f"squad={len(st.squad)} muts={len(st.mutations)} {r if len(r)>1 else ''}")
        steps += 1
    return st, steps, fights, wins

print("\n--- one narrated run (seed 1) ---")
st, steps, fights, wins = play(1, verbose=True)
print(f"   ... ended: level={st.level} finished={st.finished} won={st.won} "
      f"squad={len(st.squad)} steps={steps} fights={fights} won={wins}")

print("\n--- 25 random-policy runs ---")
t0 = time.perf_counter()
levels, sq, fought, alive = [], [], 0, 0
for s in range(25):
    st, steps, fights, wins = play(s)
    levels.append(st.level); sq.append(len(st.squad)); fought += fights; alive += bool(st.squad)
el = time.perf_counter() - t0
print(f"   reached level: min {min(levels)} median {sorted(levels)[len(levels)//2]} max {max(levels)}")
print(f"   squads surviving: {alive}/25   total fights {fought}")
print(f"   {el:.1f}s for 25 runs -> {el/25:.2f}s per run")
