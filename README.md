# despot-rl

[![ci](https://github.com/mateus-aleixo/despot-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/mateus-aleixo/despot-rl/actions/workflows/ci.yml)

Reinforcement learning against a headless simulator of *Despot's Game: Dystopian Battle
Simulator*, built to match the game as closely as possible from the balance tables the game
itself ships with.

![The trained agent playing a run](docs/run.gif)

*An agent trained for 4M steps playing `Squad2` on the corrected environment, where every
room has a fight and its shop only opens once that fight is won. The full 100-second run is
in [`docs/run.mp4`](docs/run.mp4), rendered by [`tools/render_run.py`](tools/render_run.py):
194 decisions and 30 fights, reaching level 6 against the arm's mean of 3.918, so it is a
good run and not a typical one.*

## What this is

The project has three parts, and they depend on each other.

**A simulator that aims to reproduce the game exactly.** The unit, item, mutation and level
tables are the ones the game ships, read out of its encrypted asset bundles. Combat is
continuous-space steering (grid A*, ORCA local avoidance and the game's own NodeCanvas
behaviour trees) at a fixed 50 Hz tick, instead of moves on a grid. Every number that is
still a guess lives in one file, `sim/assumptions.py`, so no unexplained constants are
spread through the code.

**A hierarchical agent.** The high level plays the run: where to move or teleport, what to
buy, when to upgrade the shop, which mutation to take, and which of the eight starting squads
it is playing. It sees a 258-dimensional observation and chooses among 36 masked actions, under
the same fog of war the game gives a player: the map is revealed one room at a time, and
nothing the policy sees is computed from rooms it has not yet passed. The low level places
the squad on the grid before each fight. Both levels are trained with PPO. Fights are
resolved inside the environment, so each run-level decision is judged by a fight that
actually took place.

**A habit of measuring.** An agent will exploit any difference between the simulator and
the game, and most of what this project has learned came from an agent finding something
cheaper in the simulator than in the game. The `notes/` directory is the running record of
that, including results that were later retracted.

## Results

A run has twelve levels, and nothing here finishes one yet.

| | mean level reached |
|---|---|
| Random policy | 1.429 |
| Hand-written heuristic | 3.946 |

Both are measured over 240 fixed evaluation seeds, cycling through the eight starting
squads. No trained agent has been measured on this environment yet: it was corrected on
2026-09-02, and agents trained before that no longer load.

Both numbers were measured before the fog of war was added, later the same day. Over 60
seeds, the same heuristic scores 3.433 with the fog of war and 3.983 with the map fully
revealed, so about half a level of the 3.946 came from seeing the whole map. A new baseline
will be measured once the rest of the run layer is in place (see `notes/roadmap.md`).

There are no PPO numbers here at the moment because the simulator differed from the game
in two places. In the game, every room has a fight, and its shop, shrine, doors and gold
only open once that fight is won. A run also starts from one of eight squads in
`ChipChoice/Squads.json`, not from `Game.Team.Packs`. The simulator had fights in only
about a third of its rooms, and it started every run with five Novices without equipment,
at 750 Power, a squad the game never gives anyone; the game's eight squads have 3,783 to
7,176 Power. Both differences are now fixed, so the numbers that depend on them have to be
measured again.

For reference, on the old environment: the heuristic scored 2.283, and PPO scored 2.561 at
600k steps and 2.932 at 2M over twelve seeds, with a standard deviation of 0.098 across
identically configured 2M agents. The methodology below still applies; those levels do
not.

The largest effect measured so far is the starting squad. The eight squads span a whole
level, from 3.467 to 4.492 under the heuristic, against a tenth of a level between
identically configured 2M agents. The squad is now part of the observation and is cycled
across episodes, so the agent learns to play all eight.

Before that, the largest effect was the training budget, and four environment features were
designed, built and measured before this became clear. The numbers come from one run per
seed with checkpoints, so differences in initialisation or architecture play no part
(measured on the old environment):

| budget | mean level | mutations held | free mutation taken |
|---|---|---|---|
| 600k | 2.561 | 0.34 | 0.2% |
| 1.2M | 2.810 | 1.30 | 22.9% |
| 1.8M | 2.886 | 1.97 | 28.4% |
| 2M | 2.932 | 2.14 | 29.2% |

2M minus 600k, paired within seed: +0.370 levels, 95% CI [+0.310, +0.431], positive on 12
of 12 seeds. At 600k steps, the behaviour under study (taking a free mutation) occurred in
0.2% of cases, so every A/B comparison at that budget compared two copies of a policy that
did not use the feature being tested.

## What the measurements showed

These findings apply beyond this game:

- **An A/B test of an environment feature means nothing until both arms have trained long
  enough to use the feature.** Check how often the behaviour occurs, or how much of the
  feature the agent holds, before reading any difference in score.
- **Changing the observation changes the architecture.** The first layer is sized from
  `obs_dim`, so adding a feature also changes the number of parameters, the random
  initialisation and the optimisation path. Here, five features in a row made no difference
  in their own ablation: with the entry held at zero but kept in the vector, the score did
  not change. The most recent one, which describes the mutation shelf, measures +0.01
  levels [-0.06, +0.08] against a blinded control at 2M.
- **A control that really is inert gives a wider spread.** In one sweep, the control arm
  was never actually inert, because the disable path changed a registry entry that a
  fallback lookup bypassed. After the fix, the standard deviation across identically
  configured agents rose from 0.132 to 0.183.
- **Four times, an agent found an action that was cheaper in the simulator than in the
  game**, and each time it looked like a strategy rather than a bug: a squad that started
  fully equipped, a room that paid out gold on every visit, free food, and a stranded run
  that scored better than losing the whole squad.

## The simulator

`sim/` is the readable, slow reference implementation. It is written so that it can be
compared line by line with the decompiled game, and speed is not a goal.

    sim/data.py         ruleset loading and the game's own override layering
    sim/spec.py         unit + item -> resolved stats
    sim/nav.py          room grid, A*, waypoint following
    sim/orca.py         RVO2 / ORCA local avoidance
    sim/bt.py           NodeCanvas behaviour tree executor
    sim/actions.py      the default attack and the class skills
    sim/unit_skills.py  per-unit skills from Skills.json
    sim/mutations.py    run-level mutations
    sim/mapgen.py       a port of the game's LevelGenerator
    sim/battle.py       the fixed-timestep loop
    sim/run.py          levels, rooms, economy, shops, progression
    sim/assumptions.py  every number that is still a guess

Levels are generated for each run by a port of the game's `LevelGenerator`, including the
position weighting, the 2x2 block rations and the cap on neighbours, so a level has the
same web of corridors the game builds. Three parameters could not be recovered from the
binary and are recorded as assumptions.

## The Rust core

`core/` is a port of the battle loop as a `cdylib` with a plain C ABI, called through
`ctypes`. Its only purpose is to run the same fights faster. The Python implementation
remains the reference, and `sim/fast.py` rejects any fight outside the scope of the port,
so it never returns numbers that silently differ.

| six broadswords against six enemies | ms per battle | speed-up |
|---|---|---|
| Python reference | 276 | 1x |
| Rust, one call at a time | 1.9 | 142x |
| Rust, batched across threads | 0.29 | 957x |

`tools/diff_core.py` runs the same fights through both engines. It caught a real porting
bug straight away: melee reach is measured between both bodies, so an attacker with radius
3 needs 10 units to touch a target with radius 6, not the 7 it needs against a target of its
own size. Agreement on the winner, for fights within the port's scope, went from 24 of 60 to
60 of 60.

## Layout

    sim/      the reference simulator
    core/     the Rust battle core
    rl/       Gymnasium env, placement policies, the heuristic baseline, PPO
    tools/    datamining, validation, evaluation, rendering
    tests/    the pytest suite; most of it runs without game data
    notes/    the running record: datamining, reference sim, Rust core, RL

The files in `notes/` hold the numbers and the mechanics, including what has been
retracted, and are worth reading before working anything out again.
[`notes/roadmap.md`](notes/roadmap.md) is the plan for the rest of the project. The goal is
every mode and every feature, and the roadmap defines what "done" can mean, given that
bit-exact behaviour can be shown to be out of reach.

## Running it

Python 3.12 or newer.

    pip install -e .                 # the simulator and the RL side
    pip install -e ".[render]"       # plus the video renderer (ffmpeg on PATH)
    pip install -e ".[datamining]"   # plus the extraction pipeline

    python tools/validate_sim.py        # the sim's check suite
    python tools/validate_rl.py         # the RL side's check suite
    python tools/run_demo.py            # a narrated run under a random policy
    python rl/train.py --steps 2000000 --checkpoint-every 600000 --seed 0
    python tools/compare_agents.py --agent runs/<checkpoint>.pt
    python tools/render_run.py --agent runs/<checkpoint>.pt --seed 30039

Everything after the installs reads the extracted balance tables, so the data step (see
Data, below) has to come first.

Build the Rust core with `cargo build --release` in `core/`. Without it everything still
runs, about 140x slower per fight. Training runs on the CPU by default; the network is a
two-layer MLP, and a GPU does not make it faster.

## Tests

    pip install -e ".[test]"
    pytest                         # everything; about two minutes with data/
    pytest -m "not needs_data"     # only what needs no game data, in seconds

`tests/` checks what can be checked without the game's tables: A* against a reference
Dijkstra on random grids, ORCA's no-collision guarantee on random encounters, the semantics
of the behaviour-tree nodes, the override layering on a synthetic ruleset, the shape rules
of generated levels, Rijndael against the FIPS-197 vectors at the AES block size, and the
Rust core's random number generator against CPython's, bit for bit. CI runs that part on
every push, with the core built. With `data/` present, `pytest` also runs
`tools/validate_sim.py` and `tools/validate_rl.py`, the full suites against the real tables.

## Data

No game data is redistributed here, and none is needed to read the code. To run anything,
you need your own copy of the game: `tools/` extracts and decrypts its asset bundles into
`data/`, which is gitignored.

The bundles are encrypted with Rijndael-256 (not AES, and the 256-bit block is why no
mainstream crypto library can open them), gzipped, with the key derived by PBKDF2 from a
passphrase stored in the binary. That passphrase is not in this repository. The tools read
it from `DESPOT_PASSPHRASE`, and `notes/datamining.md` describes where it is stored and how
it was found, including the searches that did not find it.
