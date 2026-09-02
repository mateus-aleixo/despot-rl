"""Gymnasium environment over the run layer.

This is the high level of the hierarchy: the policy decides where to move, what
to buy and which mutation to take. Battles are resolved inside `RunState.apply`,
with unit placement delegated to whatever placement policy is installed (the low
level, see `rl/placement.py`). There is no `feed`: moving feeds the team when
the larder covers it, so food is the price of a room rather than a decision.

The action space is fixed and masked rather than variable-length. A move is a
**direction** -- north, south, east, west, or through a portal -- because the map
is generated per level from that level's own room count (`sim/mapgen.py`), so
room ids are not stable across levels and a per-room action index would mean a
different room on every one of them. `action_mask()` marks which moves are
currently legal, and stepping an illegal action is a no-op with a small penalty
rather than an exception, so a policy that has not learned masking yet still
trains.

The shop is one action per slot for the same reason: slot 3 is always slot 3,
so "the expensive one" is learnable, where a single `buy_item` could only ever
mean "an item". The food shop is one action per pack on the same argument: the
packs are fixed at [7, 2] ... [250, 55], so pack 4 always means the 250-food
one. Who receives the item is not in the action space -- that is a
second choice on top of a choice, the same shape as placement -- so `RunState`
picks the human who gains the most Power by it. See `best_item_target`.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # the env is usable without gymnasium installed
    gym = None
    spaces = None

from sim.data import load_ruleset
from sim.run import MAX_UNIT_LEVEL, RunState, squad_names

# The widest shop is `max(ItemShopData.Quantity)` = 7 slots, at shop level 5.
# Checked against the table in `__init__` rather than trusted.
SHOP_SLOTS = 7

# `Rooms.Shops.Food.Packs` holds five [food, gold] pairs. Also checked in
# `__init__`, for the same reason.
FOOD_PACKS = 5

# A mutation shop shows `MUTATION_SHOP["ShowCount"]` mutations and lets the run
# take `BuyCount` of them. One action per slot, for the same reason the item
# shop has one per slot.
MUTATION_SLOTS = 10

# Fixed action layout: room moves first, then the non-move actions.
NON_MOVE_ACTIONS = (tuple(f"buy_item_{i}" for i in range(SHOP_SLOTS))
                    + tuple(f"buy_food_{i}" for i in range(FOOD_PACKS))
                    + tuple(f"buy_mutation_{i}" for i in range(MUTATION_SLOTS))
                    + ("reroll", "upgrade_shop", "buy_exp",
                       "buy_human", "take_mutation", "sacrifice"))
ROOM_KINDS = ("start", "fight", "boss", "item_shop", "food_shop", "mutation",
              "mutation_shop", "talent_shop", "empty")

# A move is a direction, not a room. (row, col) deltas, then the portal, which
# is not a direction but is a move: `RoomMap.neighbours` links portal rooms to
# each other, and the shipped fixed map has three of them.
MOVES = (("north", (-1, 0)), ("south", (1, 0)),
         ("west", (0, -1)), ("east", (0, 1)), ("portal", None))

# Player classes, for the squad-composition part of the observation.
PLAYER_CLASSES = ("Warrior", "Tank", "Shooter", "Thrower", "Medic", "Mage",
                  "Monk", "Scientist", "Dodger", "Cultist", "Plant")


class DespotRunEnv(gym.Env if gym is not None else object):
    """One run per episode."""

    metadata = {"render_modes": []}

    def __init__(self, tables: dict | None = None, seed: int = 0,
                 max_steps: int = 400, placement_policy=None,
                 fast_core: bool = False, shaping: str = "none",
                 step_cost: float = 0.02, gamma: float = 0.99,
                 level_reward: str = "flat",
                 free_mutation_steps: bool = False,
                 blind_shrine: bool = False,
                 blind_shelf: bool = False,
                 squad: str | None = None,
                 blind_squad: bool = False):
        self.tables = tables if tables is not None else load_ruleset(strict=True)
        self.max_steps = max_steps
        self.placement_policy = placement_policy
        self.fast_core = fast_core
        self.shaping = shaping
        self.step_cost = step_cost
        # Exempt the two mutation actions from the per-step tax, leaving every
        # other action taxed. Lowering `step_cost` globally does not test "a
        # mutation costs a step now and pays later": it also unbraces the item
        # shop, where the tax turns out to be the only thing that ends a reroll
        # streak (79% of them end broke at `step_cost` 0, against 16% at 0.02).
        # This isolates the mutation step from every other step.
        self.free_mutation_steps = free_mutation_steps
        # Hold the shrine bit at zero while leaving it in the vector, so an arm
        # trained without the information is architecturally identical to one
        # trained with it. Dropping the entry instead would change `obs_dim`,
        # which resizes the first layer and confounds the comparison -- the
        # mistake every observation change in `notes/rl.md` made before this.
        self.blind_shrine = blind_shrine
        # The same control for the described shelf. `mutshelf` grew the
        # observation from 134 to 184 by describing each offer, and the null it
        # measured was taken at 600k, where `buy_mutation` runs at 0-4% and the
        # behaviour under study barely happens. Repeating it needs an arm that
        # differs only in the information: this holds the five description
        # floats per slot at zero -- implemented, fraction, damage, health, the
        # largest other delta -- plus the `passive` bit that was added later,
        # and leaves `present`, the takes left and the shrine bit alone, which
        # is exactly the vector the pre-`mutshelf` agents saw, at 195 dims.
        self.blind_shelf = blind_shelf
        # Which squad a run opens with. `ChipChoice/Squads.json` ships eight and
        # they are worth about a level of spread between them, more than any
        # environment feature measured here, so this is a decision rather than a
        # constant. `squad=None` cycles them by episode seed, which is uniform
        # and reproducible and gives an even split over any block of seeds; a
        # name pins one, which is how a per-squad score is taken.
        self.squads = squad_names(self.tables)
        self.squad = squad
        self.squad_name: str | None = None
        # The same control every other observation feature gets: hold the entry
        # at zero and keep it in the vector, so `obs_dim` and the first layer do
        # not move and the comparison is information-only.
        self.blind_squad = blind_squad
        # "flat": +10 a level. "rising": +10 x the level reached, so the fifth
        # level is worth five times the first. Levels get exponentially harder
        # (a room is 800 Power at level 1 and 24,600 by level 6) while a flat
        # bonus plus discounting makes the deep ones worth less, not more.
        self.level_reward = level_reward
        self.gamma = gamma
        self._seed = seed
        # `step` asks for the mask twice: once against the pre-action state to
        # check the action, and once afterwards for the info dict. The first of
        # those recomputes exactly what the caller was handed at the end of the
        # previous step, and `legal_actions` is 7.7% of training wall clock
        # (see notes/rust-core.md), so the last mask is kept and invalidated at
        # the two places the state moves: `reset`, and `apply` inside `step`.
        self._mask_cache: np.ndarray | None = None

        probe = RunState.new(self.tables, seed=0)
        self.n_moves = len(MOVES)
        self.n_actions = self.n_moves + len(NON_MOVE_ACTIONS)
        widest = max(r["Quantity"] for r in self.tables["ItemShopData"])
        if widest > SHOP_SLOTS:
            raise ValueError(f"ItemShopData sells up to {widest} slots, "
                             f"the action space has {SHOP_SLOTS}")
        from sim.run import MUTATION_SHOP
        if MUTATION_SHOP["ShowCount"] > MUTATION_SLOTS:
            raise ValueError(f"a mutation shop shows {MUTATION_SHOP['ShowCount']}, "
                             f"the action space has {MUTATION_SLOTS}")
        self.food_sizes, self.food_costs = probe.food_packs
        if len(self.food_sizes) > FOOD_PACKS:
            raise ValueError(f"the food shop sells {len(self.food_sizes)} packs, "
                             f"the action space has {FOOD_PACKS}")
        # `Items.Class` and `Items.Quality`, hoisted out of `_encode`: it runs
        # once per environment step and was rebuilding this table every time.
        self.item_class = {i["Name"]: i["Class"] for i in self.tables["Items"]}
        self.item_quality = {i["Name"]: int(i.get("Quality") or 1)
                             for i in self.tables["Items"]}

        self.state: RunState | None = None
        self.steps = 0
        obs_dim = len(self._encode(probe))
        self.obs_dim = obs_dim

        if spaces is not None:
            self.action_space = spaces.Discrete(self.n_actions)
            self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                                shape=(obs_dim,), dtype=np.float32)

    # -- encoding ----------------------------------------------------------
    def _encode(self, st: RunState) -> np.ndarray:
        f = st.food
        # Progression, which the observation had no way to see: how far the
        # squad's levels have come, how strong it is against the room it is
        # standing in, and what the shop is currently offering. Without these
        # the policy cannot tell a level-1 squad from a level-5 one, so it
        # cannot learn when to stop shopping and push.
        levels = [h.level for h in st.squad] or [1]
        toward = [min(1.0, h.experience / st.max_experience(h.level))
                  for h in st.squad if h.level < MAX_UNIT_LEVEL]
        room = st.room_power("boss" if st.rooms.rooms[st.room].kind == "boss" else "fight")
        scalars = [
            st.level / 12.0,
            np.log1p(max(0.0, st.gold)) / 6.0,
            np.log1p(max(0.0, f.amount)) / 6.0,
            f.hunger_level / max(1, len(f.damage_penalties) - 1),
            f.moves_left / max(1, max(f.moves)),
            len(st.squad) / 20.0,
            len(st.mutations) / 20.0,
            f.damage_penalty / 100.0,
            self.steps / self.max_steps,
            sum(levels) / len(levels) / MAX_UNIT_LEVEL,
            min(levels) / MAX_UNIT_LEVEL,
            (sum(toward) / len(toward)) if toward else 1.0,
            st.shop_level / max(1, st.max_shop_level),
            np.log1p(st.squad_power()) / 12.0,
            float(np.clip(np.log1p(st.squad_power() / max(1.0, room)) / 4.0, -10.0, 10.0)),
        ]

        # squad composition by class
        counts = dict.fromkeys(PLAYER_CLASSES, 0.0)
        for h in st.squad:
            cls = self.item_class.get(h.item)
            if cls in counts:
                counts[cls] += 1.0
        comp = [counts[c] / 10.0 for c in PLAYER_CLASSES]

        # what is on the shelf: quality and whether it is affordable, per slot
        shelf = np.zeros(2 * SHOP_SLOTS, dtype=np.float32)
        if st.rooms.rooms[st.room].kind == "item_shop":
            for i, name in enumerate(st.offer[:SHOP_SLOTS]):
                if name is None:
                    continue
                shelf[2 * i] = self.item_quality.get(name, 1) / 5.0
                shelf[2 * i + 1] = 1.0 if st.gold >= st.item_cost(name) else 0.0

        # what the food shop is holding: stock and affordability, per pack. The
        # packs are fixed ([7, 2] ... [250, 55]), so the sizes and prices are
        # constants the network can learn; what changes is which are left and
        # which the purse covers.
        larder = np.zeros(2 * FOOD_PACKS, dtype=np.float32)
        if st.rooms.rooms[st.room].kind == "food_shop":
            st.ensure_stock(st.rooms.rooms[st.room])
            held = st.rooms.rooms[st.room].food_stock or []
            for i, count in enumerate(held[:FOOD_PACKS]):
                larder[2 * i] = min(1.0, count / 4.0)
                larder[2 * i + 1] = 1.0 if st.gold >= self.food_costs[i] else 0.0

        # What the mutation shop is showing, slot by slot. Ten presence bits
        # were not enough to choose between ten free mutations: three quarters
        # of the pool is `unimplemented` in this sim and does nothing at all,
        # and a mutation aimed at a class the squad does not field does nothing
        # either, so the two facts a policy needs first are whether the slot is
        # modelled and what share of the squad it touches. `sim.mutations.
        # effect_on` supplies both, and the stat deltas where there are any --
        # which is only `StatBonus`, 32 offers of 1,094.
        #
        # `passive` was added after the shelf was measured against a control
        # that actually strips the hooks: with them inert the shelf is worth
        # +0.00 levels rather than +0.05, so the agent-level passives are the
        # whole +0.13 and the stat deltas below are worth nothing. `implemented`
        # does not stand in for it, because 96 of the 603 modelled offers attach
        # no passive and are exactly the ones measuring at zero.
        per_m = 7
        shelf_m = np.zeros(MUTATION_SLOTS * per_m + 2, dtype=np.float32)
        room_now = st.rooms.rooms[st.room]
        if room_now.kind == "mutation_shop":
            st.ensure_stock(room_now)
            from sim.run import MUTATION_SHOP
            for i, m in enumerate((room_now.offer_mutations or [])[:MUTATION_SLOTS]):
                if m is None:
                    continue
                base = i * per_m
                shelf_m[base] = 1.0
                if self.blind_shelf:
                    continue
                e = st.mutation_effect(m)
                shelf_m[base + 1] = 1.0 if e["implemented"] else 0.0
                shelf_m[base + 2] = e["fraction"]
                shelf_m[base + 3] = float(np.clip(e["damage"], -1.0, 1.0))
                shelf_m[base + 4] = float(np.clip(e["health"], -1.0, 1.0))
                shelf_m[base + 5] = float(np.clip(
                    max((abs(e[s]) for s in ("armor", "attack_speed", "speed",
                                             "range", "resistance")), default=0.0),
                    0.0, 1.0))
                shelf_m[base + 6] = 1.0 if e["passive"] else 0.0
            shelf_m[MUTATION_SLOTS * per_m] = (
                room_now.takes_left / max(1, int(MUTATION_SHOP["BuyCount"])))
        elif room_now.kind == "mutation":
            # The shrine's one free mutation. Without this a stocked shrine and
            # a spent one encode identically and only the action mask separates
            # them, so the logit for `take_mutation` is computed from evidence
            # that cannot distinguish the two cases and the association is
            # unlearnable. That is why every agent took the shop's mutations up
            # to 60% of the time and the free one 0 to 1%, against a heuristic
            # that takes it every time. `ensure_stock` is deterministic here
            # (`stock = 1`, no rng) and `legal_actions` calls it every step
            # anyway, so reading it costs nothing and shifts no random stream.
            st.ensure_stock(room_now)
            shelf_m[MUTATION_SLOTS * per_m + 1] = (
                0.0 if self.blind_shrine
                else 1.0 if (room_now.stock or 0) > 0 else 0.0)

        # The room the squad is standing in, one-hot over the kinds. With a
        # per-level map there is no stable per-room index to carry this, and the
        # policy still has to tell a shop from a fight.
        here_kind = np.zeros(len(ROOM_KINDS), dtype=np.float32)
        if st.rooms.rooms[st.room].kind in ROOM_KINDS:
            here_kind[ROOM_KINDS.index(st.rooms.rooms[st.room].kind)] = 1.0

        # What lies one step away, per direction: whether there is a room there,
        # whether it is cleared, whether it is nearer the boss, and what kind it
        # is. This replaced the four per-room vectors, and it is the whole local
        # view the policy gets of a map it has not seen before.
        per = 3 + len(ROOM_KINDS)
        around = np.zeros(self.n_moves * per, dtype=np.float32)
        far = max(1, max(st.rooms.to_boss.values(), default=1))
        d_here = st.rooms.to_boss.get(st.room, far)
        for i, rid in enumerate(self._move_targets(st)):
            if rid is None:
                continue
            room = st.rooms.rooms[rid]
            base = i * per
            around[base] = 1.0
            around[base + 1] = 1.0 if room.cleared else 0.0
            around[base + 2] = 1.0 if st.rooms.to_boss.get(rid, far) < d_here else 0.0
            if room.kind in ROOM_KINDS:
                around[base + 3 + ROOM_KINDS.index(room.kind)] = 1.0

        # and where this room sits in the level as a whole
        rooms = list(st.rooms.rooms.values())
        pantries = [r for r in rooms if r.kind == "food_shop"]
        level_view = [
            d_here / far,
            sum(1 for r in rooms if r.cleared) / max(1, len(rooms)),
            len(rooms) / 20.0,
            sum(1 for r in pantries
                if r.food_stock is None or any(q > 0 for q in r.food_stock))
            / max(1, len(pantries)),
        ]

        # Which squad this run is playing. The policy has to know: the eight
        # differ in size, in whether they field a frontline and in whether they
        # field anything ranged, and a two-unit squad pays half the food a
        # four-unit one does for the same move.
        who = np.zeros(len(self.squads), dtype=np.float32)
        if not self.blind_squad and self.squad_name in self.squads:
            who[self.squads.index(self.squad_name)] = 1.0

        return np.concatenate([
            np.asarray(scalars, dtype=np.float32),
            np.asarray(level_view, dtype=np.float32),
            np.asarray(comp, dtype=np.float32),
            here_kind, shelf, larder, shelf_m, around, who,
        ]).astype(np.float32)

    def _move_targets(self, st: RunState) -> list:
        """The room each move leads to from where the squad stands, or None.

        A portal move goes to the linked portal room nearest the boss, so the
        action still means one thing when a room links to more than one.
        """
        here = st.rooms.rooms[st.room]
        by_pos = {(r.row, r.col): rid for rid, r in st.rooms.rooms.items()}
        near = set(st.rooms.neighbours(st.room))
        out = []
        for _, delta in MOVES:
            if delta is None:
                links = [p for p in st.rooms.portals if p != st.room and p in near]
                out.append(min(links, key=lambda p: st.rooms.to_boss.get(p, 99))
                           if links else None)
                continue
            rid = by_pos.get((here.row + delta[0], here.col + delta[1]))
            out.append(rid if rid in near else None)
        return out

    # -- reward shaping ----------------------------------------------------
    def potential(self, st: RunState) -> float:
        """The shaping potential, or 0 when shaping is off.

        Potential-based shaping (`gamma * phi(s') - phi(s)`) is the form that
        leaves the optimal policy alone: whatever it does to the learning
        signal, it cannot make a worse policy look better over a whole episode.
        That matters here because the reward is the thing under test -- a
        hand-added bonus for, say, buying items would be a way of writing the
        strategy into the reward and then congratulating the agent for finding
        it.

        Two terms, both from the autopsy rather than from taste. Runs end with
        the squad wiped while holding 115 gold and 130 food, so the agent is
        dying rich: the potential rises with **squad power actually bought**
        (`Power` is the game's own strength statistic, and an item is worth
        thousands where a bare human is worth tens, so this ranks item buying
        far above body buying) and with **food banked** up to three feedings,
        after which more larder is not more safety.
        """
        if self.shaping == "none" or st is None:
            return 0.0
        power = st.squad_power()
        food = min(1.0, st.food.amount / (3.0 * st.feed_cost))
        return 3.0 * np.log1p(power / 1000.0) + 1.0 * food

    # -- action plumbing ---------------------------------------------------
    def _is_mutation_action(self, index: int) -> bool:
        """`buy_mutation_*` or `take_mutation`, by index into the fixed layout."""
        if index < self.n_moves:
            return False
        name = NON_MOVE_ACTIONS[index - self.n_moves]
        return name.startswith("buy_mutation_") or name == "take_mutation"

    def _decode(self, index: int):
        if index < self.n_moves:
            # A direction only names a room once you know where you stand, so
            # this decodes against the live state rather than a fixed table.
            target = self._move_targets(self.state)[index] if self.state else None
            return ("move", target)
        name = NON_MOVE_ACTIONS[index - self.n_moves]
        if name.startswith("buy_item_"):
            # (slot, target); a target of None lets the run layer choose.
            return ("buy_item", (int(name.rsplit("_", 1)[1]), None))
        if name.startswith("buy_food_"):
            return ("buy_food", int(name.rsplit("_", 1)[1]))
        if name.startswith("buy_mutation_"):
            return ("buy_mutation", int(name.rsplit("_", 1)[1]))
        return (name, None)

    def _index(self, action) -> int | None:
        """The action's fixed index, or None if it has none."""
        kind, arg = action
        if kind == "move":
            targets = self._move_targets(self.state)
            return targets.index(arg) if arg in targets else None
        if kind == "buy_item":
            slot = arg[0] if isinstance(arg, tuple) else int(arg or 0)
            if slot >= SHOP_SLOTS:
                return None
            kind = f"buy_item_{slot}"
        if kind == "buy_food":
            pack = int(arg or 0)
            if pack >= FOOD_PACKS:
                return None
            kind = f"buy_food_{pack}"
        if kind == "buy_mutation":
            slot = int(arg or 0)
            if slot >= MUTATION_SLOTS:
                return None
            kind = f"buy_mutation_{slot}"
        return self.n_moves + NON_MOVE_ACTIONS.index(kind)

    def action_mask(self) -> np.ndarray:
        # A copy, not the cached array itself: two calls used to hand back two
        # independent arrays and nothing should start aliasing because this is
        # memoised. The copy is 33 bools; the scan it skips is not.
        if self._mask_cache is not None:
            return self._mask_cache.copy()
        mask = np.zeros(self.n_actions, dtype=bool)
        if self.state is None or self.state.finished:
            return mask
        for action in self.state.legal_actions():
            i = self._index(action)
            if i is not None:
                mask[i] = True
        self._mask_cache = mask
        return mask.copy()

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._seed = seed
        self.squad_name = (self.squad if self.squad is not None
                           else self.squads[self._seed % len(self.squads)])
        self.state = RunState.new(self.tables, seed=self._seed,
                                  squad_name=self.squad_name)
        self.state.use_fast_core = self.fast_core
        if self.placement_policy is not None:
            self.state.placement_policy = self.placement_policy
        self._seed += 1
        self.steps = 0
        self._mask_cache = None
        return self._encode(self.state), {"action_mask": self.action_mask()}

    def step(self, action: int):
        st = self.state
        mask = self.action_mask()
        reward = 0.0

        if not mask.any():
            return self._encode(st), 0.0, True, False, {"action_mask": mask}

        if not mask[action]:
            # Illegal: a small penalty and no state change, so the policy learns
            # the mask without the episode dying on a bad index.
            reward -= 0.05
            self.steps += 1
            done = self.steps >= self.max_steps
            return self._encode(st), reward, done, False, {"action_mask": mask,
                                                           "illegal": True}

        level_before = st.level
        squad_before = len(st.squad)
        phi_before = self.potential(st)
        result = st.apply(self._decode(action))
        self._mask_cache = None          # the state moved; the mask is stale
        self.steps += 1

        if self.level_reward == "rising":
            reward += 10.0 * sum(range(level_before + 1, st.level + 1))
        else:
            reward += 10.0 * (st.level - level_before)      # progress dominates
        if "won" in result:
            reward += 1.0 if result["won"] else -1.0
        if not (self.free_mutation_steps and self._is_mutation_action(action)):
            reward -= self.step_cost                         # mild step cost
        lost = squad_before - len(st.squad)
        if lost > 0:
            reward -= 0.5 * lost

        terminated = st.finished or not st.squad
        if terminated and not st.won:
            reward -= 2.0
        if st.won:
            reward += 50.0
        truncated = self.steps >= self.max_steps

        # gamma * phi(s') - phi(s), with phi = 0 wherever the episode stops.
        # The step cap counts as stopping: the trainer does not bootstrap past a
        # truncation, so leaving the potential standing there would pay an agent
        # to dither until the cap and keep its potential rather than spend it.
        if self.shaping != "none":
            phi_after = 0.0 if (terminated or truncated) else self.potential(st)
            reward += self.gamma * phi_after - phi_before

        info = {"action_mask": self.action_mask(), "level": st.level,
                "squad": len(st.squad), "result": result}
        return self._encode(st), reward, bool(terminated), bool(truncated), info


def make_env(**kwargs) -> DespotRunEnv:
    return DespotRunEnv(**kwargs)
