"""The low level as a learning problem: one fight, one cell per unit.

The run environment (`rl/env.py`) decides *what* to fight with; this decides
*where* to stand. An episode here is a single fight: the policy is shown the
already-deployed enemies and picks a cell for each of its own units in turn,
then the battle resolves and the outcome is the reward.

Two choices are worth stating up front.

**Fights are sampled from real runs, not invented.** `sample_scenarios` plays
actual `RunState` runs with a heuristic policy and captures every fight it walks
into, so the squad compositions, enemy packs, mutations and hunger penalties are
the ones a run-level agent actually produces. A hand-made distribution of fights
would train a placement policy for fights that never happen.

**Reward is paired against the baseline.** The same scenario is resolved twice
on the same battle seed, once with the policy's cells and once with
`frontline_placement`, and the reward is the difference. Fight difficulty varies
enormously between scenarios and individual outcomes near a decision boundary
are close to coin flips (see `notes/rust-core.md`), so an unpaired reward is
mostly noise about which fight was drawn. The paired difference cancels that and
asks the only question that matters: is this better than the heuristic, here?
"""
from __future__ import annotations

import pathlib
import pickle
import random
from dataclasses import dataclass, field, replace

import numpy as np

from sim.battle import Agent, Battle, apply_class_skills, place_at
from sim.data import parse_room_layouts
from sim.fast import UnsupportedFight, available, fast_battle
from sim.nav import Grid
from sim.run import RunState
from sim.spec import UnitSpec

# The observation is a stack of planes over the room rectangle. Layouts are not
# all the same shape (69 of the 77 are 7x20 with a 49-cell player zone, the rest
# are taller or have a bigger zone), so the planes are drawn on one canvas big
# enough for the largest and the action space is the largest zone, masked down
# to whatever the current layout actually offers.
PLANES = ("ally", "ally_melee", "ally_ranged", "ally_hp",
          "enemy", "enemy_melee", "enemy_ranged", "enemy_hp", "enemy_dps",
          "zone")
N_PLANES = len(PLANES)

# Per-unit features for the unit currently being placed, plus squad context.
UNIT_FEATURES = 12


@dataclass
class Scenario:
    """One fight, captured mid-run and replayable without the run.

    `uid` and `seeds` exist so the baseline half of the paired reward can be
    cached. Battle seeds are drawn from a small fixed pool per scenario rather
    than freshly at random, which makes `(uid, seed)` a key that repeats: the
    heuristic's outcome on that pair is computed once and reused for every
    episode after, halving the battles the trainer runs.
    """
    level: int
    layout_index: int
    specs: list[UnitSpec]
    enemy_specs: list[UnitSpec]
    enemy_xy: list[tuple[float, float]]
    mutations: list = field(default_factory=list)
    uid: int = 0
    seeds: tuple[int, ...] = ()

    @property
    def n_units(self) -> int:
        return len(self.specs)


SEEDS_PER_SCENARIO = 8


def sample_scenarios(tables: dict, n: int = 200, seed: int = 0,
                     max_steps: int = 400) -> list[Scenario]:
    """Play runs with a simple heuristic and capture the fights they produce."""
    layouts = parse_room_layouts(tables["RoomLayouts"])
    index_of = {id(l): i for i, l in enumerate(layouts)}
    out: list[Scenario] = []
    rid = seed

    while len(out) < n:
        st = RunState.new(tables, seed=rid)
        st.use_fast_core = available()
        rid += 1
        captured: list[Scenario] = []

        def capture(grid, layout, specs, rng, enemies):
            captured.append(Scenario(
                level=st.level,
                layout_index=index_of[id(layout)],
                specs=list(specs),
                enemy_specs=[a.spec for a in enemies],
                enemy_xy=[(a.x, a.y) for a in enemies],
                mutations=list(st.mutations),
            ))
            from rl.placement import frontline_placement
            return frontline_placement(grid, layout, specs, rng, enemies)

        st.placement_policy = capture
        rng = random.Random(rid)
        for _ in range(max_steps):
            if st.finished:
                break
            legal = st.legal_actions()
            if not legal:
                break
            st.apply(_heuristic_action(st, legal, rng))
        out.extend(captured)

    out = out[:n]
    for i, scn in enumerate(out):
        scn.uid = seed * 1_000_003 + i
        pool = random.Random(scn.uid)
        scn.seeds = tuple(pool.randrange(1 << 30) for _ in range(SEEDS_PER_SCENARIO))
    return out


def _heuristic_action(st: RunState, legal, rng: random.Random):
    """The shared run-level baseline; see `rl.heuristic`."""
    from .heuristic import heuristic_action
    return heuristic_action(st, legal, rng)


# -- difficulty -------------------------------------------------------------
def scale_power(scn: Scenario, mult: float) -> Scenario:
    """The same fight with the enemy side's health and damage scaled.

    Enemy *count* was the first knob tried and it is too coarse: adding one unit
    takes a fight from always won to always lost, so bisecting on it almost
    never lands anywhere close. A continuous multiplier does, at the cost of
    being a probe rather than a fight the game would deal you.
    """
    specs = [replace(sp, health=sp.health * mult, damage=sp.damage * mult)
             for sp in scn.enemy_specs]
    return replace(scn, enemy_specs=specs,
                   uid=scn.uid * 1000 + int(mult * 100) % 1000)


def win_rate(scn: Scenario, tables: dict, layouts, policy, seeds) -> float:
    """How often `policy` wins this fight, over the given battle seeds."""
    from sim.battle import Agent as _Agent
    layout = layouts[scn.layout_index]
    grid = Grid.from_layout(layout)
    won = 0
    for seed in seeds:
        enemies = [_Agent(spec=sp, team=1, x=x, y=y, hp=sp.health, mana=0.0)
                   for sp, (x, y) in zip(scn.enemy_specs, scn.enemy_xy)]
        cells = policy(grid, layout, scn.specs, random.Random(seed), enemies)
        won += 1 if resolve(scn, cells, tables, layouts, seed)["won"] else 0
    return won / len(seeds)


def placement_win_rate(scn: Scenario, tables: dict, layouts,
                       n_placements: int = 6, seeds=(1, 2, 3)) -> float:
    """Win rate across *placements*, which is the thing the low level controls.

    Tuning difficulty by the heuristic's win rate over battle seeds was the
    first attempt and it finds almost nothing: given a squad and a multiplier,
    the fight is close to deterministic, so seed noise flips it only inside a
    razor-thin band. Placement is a decision, not noise, and a fight can be
    perfectly deterministic and still be decided by where the units stand. So
    the probe spreads over random placements instead.
    """
    from rl.placement import random_placement
    layout = layouts[scn.layout_index]
    grid = Grid.from_layout(layout)
    enemies = [Agent(spec=sp, team=1, x=x, y=y, hp=sp.health, mana=0.0)
               for sp, (x, y) in zip(scn.enemy_specs, scn.enemy_xy)]
    won = n = 0
    for k in range(n_placements):
        for seed in seeds:
            cells = random_placement(grid, layout, scn.specs,
                                     random.Random(seed * 977 + k), enemies)
            won += 1 if resolve(scn, cells, tables, layouts, seed)["won"] else 0
            n += 1
    return won / max(1, n)


def make_close(scn: Scenario, tables: dict, layouts, steps: int = 7,
               lo: float = 0.25, hi: float = 4.0,
               tol: float = 0.25) -> Scenario | None:
    """Scale the enemy side until placement decides the fight about half the time.

    A policy trained on fights that are won or lost whatever it does is learning
    from noise. This is the knob that produces the fights where the low level
    has something to say. Returns None if no multiplier in range lands within
    `tol` of even.
    """
    best, best_gap = None, 1e9
    for _ in range(steps):
        mid = (lo + hi) / 2
        cand = scale_power(scn, mid)
        p = placement_win_rate(cand, tables, layouts)
        gap = abs(p - 0.5)
        if gap < best_gap:
            best, best_gap = cand, gap
        if p > 0.5:
            lo = mid              # winning too easily: tougher enemies
        else:
            hi = mid
    return best if best_gap <= tol else None


def sample_close_scenarios(tables: dict, n: int = 200, seed: int = 0,
                           pool: int = 0) -> list[Scenario]:
    """Real fights, rescaled to the point where the outcome is still open."""
    layouts = parse_room_layouts(tables["RoomLayouts"])
    raw = sample_scenarios(tables, n=pool or n * 2, seed=seed)
    out = []
    for scn in raw:
        tuned = make_close(scn, tables, layouts)
        if tuned is not None:
            rng = random.Random(tuned.uid)
            tuned.seeds = tuple(rng.randrange(1 << 30)
                                for _ in range(SEEDS_PER_SCENARIO))
            out.append(tuned)
        if len(out) >= n:
            break
    return out


# -- resolving a placement -------------------------------------------------
def resolve(scn: Scenario, cells, tables: dict, layouts, seed: int) -> dict:
    """Run `scn` with the player units on `cells`. Fast core, oracle fallback."""
    layout = layouts[scn.layout_index]
    grid = Grid.from_layout(layout)
    # Specs are never mutated in place -- `sim.mutations` returns replacements --
    # so the agents can share them and skip a deep copy per fight.
    team0 = place_at(grid, scn.specs, cells, team=0)
    apply_class_skills(tables, team0)
    if scn.mutations:
        from sim.mutations import apply_to_agents
        apply_to_agents(team0, scn.mutations, random.Random(seed))
    team1 = [Agent(spec=s, team=1, x=x, y=y, hp=s.health, mana=0.0)
             for s, (x, y) in zip(scn.enemy_specs, scn.enemy_xy)]

    agents = team0 + team1
    if available():
        b = Battle(grid, agents, seed=seed, tables=tables, build_trees=False)
        try:
            fr = fast_battle(grid, b.agents, seed=seed, tables=tables)
            return _outcome(b.agents, fr["winner"], fr["hp"])
        except UnsupportedFight:
            agents = team0 + team1
    b = Battle(grid, agents, seed=seed, tables=tables)
    res = b.run()
    hp = [a.hp for a in b.agents]
    return _outcome(b.agents, res.winner, hp)


def _outcome(agents, winner, hp) -> dict:
    ally_hp = ally_max = foe_hp = foe_max = 0.0
    alive = 0
    for a, h in zip(agents, hp):
        h = max(0.0, float(h))
        if a.team == 0:
            ally_hp += h
            ally_max += a.spec.health
            alive += 1 if h > 0 else 0
        else:
            foe_hp += h
            foe_max += a.spec.health
    return {
        "won": winner == 0,
        "ally_frac": ally_hp / ally_max if ally_max else 0.0,
        "foe_frac": foe_hp / foe_max if foe_max else 0.0,
        "survivors": alive,
    }


def score(outcome: dict) -> float:
    """Win dominates; surviving HP and damage dealt break ties inside it."""
    return ((1.0 if outcome["won"] else -1.0)
            + outcome["ally_frac"] - outcome["foe_frac"])


# -- the environment -------------------------------------------------------
class PlacementEnv:
    """One fight per episode, one action per unit.

    `step` takes an index into the sorted player-zone cells. Cells already taken
    are masked out, so a placement is always a permutation-free assignment. The
    reward arrives on the last unit, as the paired difference against the
    baseline placement on the same battle seed.
    """

    def __init__(self, tables: dict, scenarios: list[Scenario], seed: int = 0,
                 paired: bool = True, baseline_cache: dict | None = None):
        self.tables = tables
        # shared across envs when the trainer passes one in
        self.baseline_cache = {} if baseline_cache is None else baseline_cache
        self.layouts = parse_room_layouts(tables["RoomLayouts"])
        self.scenarios = scenarios
        self.rng = random.Random(seed)
        self.paired = paired

        self.rows = max(l.size[0] for l in self.layouts)
        self.cols = max(l.size[1] for l in self.layouts)
        self.n_actions = max(len(l.zone("p")) for l in self.layouts)
        self.obs_dim = N_PLANES * self.rows * self.cols + UNIT_FEATURES

        self.zone: list[tuple[int, int]] = []

        self.scn: Scenario | None = None
        self.cells: list[tuple[int, int]] = []
        self.taken: set[tuple[int, int]] = set()
        self.unit = 0
        self.battle_seed = 0

    # -- episode -----------------------------------------------------------
    def reset(self, scenario: Scenario | None = None,
              battle_seed: int | None = None):
        self.scn = scenario or self.rng.choice(self.scenarios)
        layout = self.layouts[self.scn.layout_index]
        self.zone = sorted(layout.zone("p"))
        self.cells, self.taken, self.unit = [], set(), 0
        if battle_seed is not None:
            self.battle_seed = battle_seed
        else:
            self.battle_seed = (self.rng.choice(self.scn.seeds) if self.scn.seeds
                                else self.rng.randrange(1 << 30))
        return self._encode(), {"action_mask": self.action_mask(),
                                "cell_index": self.cell_index()}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        for i, rc in enumerate(self.zone):
            mask[i] = rc not in self.taken
        if not mask.any():          # more units than cells: allow stacking
            mask[:len(self.zone)] = True
        return mask

    def step(self, action: int):
        scn = self.scn
        cell = self.zone[int(action) % len(self.zone)]
        self.cells.append(cell)
        self.taken.add(cell)
        self.unit += 1

        if self.unit < scn.n_units:
            return (self._encode(), 0.0, False, False,
                    {"action_mask": self.action_mask(), "cell_index": self.cell_index()})

        mine = resolve(scn, self.cells, self.tables, self.layouts, self.battle_seed)
        reward = score(mine)
        info = {"action_mask": np.zeros(self.n_actions, dtype=bool),
                "cell_index": self.cell_index(),
                "won": mine["won"], "raw": reward}
        if self.paired:
            base = self._baseline(scn)
            reward -= score(base)
            info["baseline_won"] = base["won"]
        return self._encode(), float(reward), True, False, info

    def _baseline(self, scn: Scenario) -> dict:
        """The heuristic's outcome on this scenario and seed, computed once."""
        key = (scn.uid, self.battle_seed)
        hit = self.baseline_cache.get(key)
        if hit is not None:
            return hit
        from rl.placement import frontline_placement
        layout = self.layouts[scn.layout_index]
        grid = Grid.from_layout(layout)
        enemies = [Agent(spec=s, team=1, x=x, y=y, hp=s.health, mana=0.0)
                   for s, (x, y) in zip(scn.enemy_specs, scn.enemy_xy)]
        cells = frontline_placement(grid, layout, scn.specs,
                                    random.Random(self.battle_seed), enemies)
        out = resolve(scn, cells, self.tables, self.layouts, self.battle_seed)
        self.baseline_cache[key] = out
        return out

    # -- observation -------------------------------------------------------
    def cell_index(self) -> np.ndarray:
        """Action index -> flat cell on the padded canvas, -1 where unused.

        The network scores every cell of the canvas; this is what turns that map
        into the action space of the layout currently in play.
        """
        out = np.full(self.n_actions, -1, dtype=np.int64)
        for i, (r, c) in enumerate(self.zone):
            out[i] = r * self.cols + c
        return out

    def _encode(self) -> np.ndarray:
        scn = self.scn
        return encode_state(
            self.rows, self.cols, self.zone, scn.specs, self.cells, self.unit,
            scn.enemy_specs, scn.enemy_xy,
            Grid.from_layout(self.layouts[scn.layout_index]).tile)


def encode_state(rows: int, cols: int, zone, specs, placed_cells, unit: int,
                 enemy_specs, enemy_xy, tile: float) -> np.ndarray:
    """The planes plus the current unit's features, as one flat float vector.

    Shared by the environment and by `LearnedPlacement`, so a policy sees the
    same encoding when it is driving a real run as it did in training.
    """
    planes = np.zeros((N_PLANES, rows, cols), dtype=np.float32)
    idx = {name: i for i, name in enumerate(PLANES)}

    for r, c in zone:
        planes[idx["zone"], r, c] = 1.0

    for spec, (r, c) in zip(specs, placed_cells):
        planes[idx["ally"], r, c] += 1.0
        planes[idx["ally_melee" if spec.melee else "ally_ranged"], r, c] += 1.0
        planes[idx["ally_hp"], r, c] += spec.health / 500.0

    for spec, (x, y) in zip(enemy_specs, enemy_xy):
        r, c = int(y // tile), int(x // tile)
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        planes[idx["enemy"], r, c] += 1.0
        planes[idx["enemy_melee" if spec.melee else "enemy_ranged"], r, c] += 1.0
        planes[idx["enemy_hp"], r, c] += spec.health / 500.0
        planes[idx["enemy_dps"], r, c] += spec.damage * spec.attack_speed / 100.0

    n_units = len(specs)
    spec = specs[min(unit, n_units - 1)]
    ally_hp = sum(s.health for s in specs) or 1.0
    foe_hp = sum(s.health for s in enemy_specs) or 1.0
    feats = np.asarray([
        1.0 if spec.melee else 0.0,
        spec.range_world / 100.0,
        spec.health / 500.0,
        spec.damage / 100.0,
        spec.attack_speed,
        spec.speed / 50.0,
        spec.armor / 50.0,
        spec.size / 3.0,
        unit / max(1, n_units),
        n_units / 12.0,
        len(enemy_specs) / 12.0,
        foe_hp / (ally_hp + foe_hp),
    ], dtype=np.float32)

    return np.concatenate([planes.reshape(-1), feats]).astype(np.float32)


def cached_scenarios(path: str | None, build):
    """Whatever `build` returns, pickled once and reused after.

    A close-fight set costs a few hundred battles per scenario to find; caching
    turns "sample the fights" from minutes into a file read, which matters when
    the same set is reused across training runs and measurements.
    """
    if not path:
        return build()
    f = pathlib.Path(path)
    if f.exists():
        with f.open("rb") as fh:
            return pickle.load(fh)
    out = build()
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("wb") as fh:
        pickle.dump(out, fh)
    return out
