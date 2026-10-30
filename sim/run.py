"""The run layer: levels, rooms, economy, shops and progression.

A run is 12 levels. Each level is generated from its own `Levels.json` row --
`MinRooms`..`MaxRooms` rooms, `ItemShops` item shops, `FoodShops` food shops,
`Shrines` shrines, see `sim/mapgen.py`. You move between orthogonally adjacent
rooms or teleport between portal rooms, fight what is in them, spend gold in
shops and food to keep moving, and finish at the boss. The shipped fixed map in `Rooms.json` is still parsed, by
`RoomMap.from_table`, and is what every level used to reuse.

This is the layer the hierarchical agent acts on: which room to move to, what to
buy, which mutation to take. Battles are resolved by `Battle`.

Verified against the shipped data:
  * `Game.Food` is the hunger model, and it is not shaped the way the JSON
    reads -- see `Food`. Moving feeds the team when the larder covers one food
    per human; `moves` is a list of thresholds on a six-move reserve that only
    drains once the larder does not.
  * `Levels.json` gives `PowerPerRoom` as the enemy budget, scaled by
    `DefaultMult` for normal rooms and `BossMult` for the boss.
  * `Game.Team.Packs` is the starting squad, `Game.Gold` and `Game.Food.amount`
    the starting purse and larder.
  * The food shop sells `Rooms.Shops.Food.Packs`, five [food, gold] pairs, and
    each room stocks itself once from a gold budget -- see `food_packs` and
    `determine_quantities`. Nothing else fills the larder: a level grants no
    food on arrival, whatever the `TotalFood` column looks like.

## Progression

This is what turns gold into strength, and it was the piece that was missing.
`buy_item` used to overwrite a random human's item with a random item, item
levels never moved, and the squad started already armed, so squad Power sat
flat at ~20,000 from level 1 to level 4 while the room budget grew 800 ->
7,700. It was flat because it began at the ceiling.

Read off the binary (`C_ItemShop`, `CS_Units.LevelUp`, `C_Team.GetExperience`,
`C_Unit.Die`):

  * **The item shop is one model for the whole run** -- `M_Session._itemShop`,
    and `C_Rooms..ctor` builds its controller once. Its level, its stock and
    its prices persist between rooms.
  * **Stock** is `ItemShopData.Quantity` slots at the shop's level. Each slot
    draws a quality with weight `Q<n>Prob` from the row for that level, then an
    item of that quality.
  * **An item costs its own `Items.Cost`**, not the shop row's `Price`. `Price`
    is the *upgrade* cost, and `C_ItemShop..ctor` reads `prices[level]`, the
    next level's, so going 1 -> 2 costs 3 and 4 -> 5 costs 12.
  * **Upgrading** raises the shop level to at most `levelCount` (5), which
    widens the quality weights and adds slots (3 at level 1, 7 at level 5).
  * **Experience** arrives two ways: `C_Unit.Die` hands a dead enemy's
    `ExpReward` to the other team, and the item shop sells `ExperienceAmount`
    (400) for `ExperienceCost` (1 gold). Either way `C_Team.GetExperience`
    splits it evenly over the units in the room.
  * **A unit levels** while `experience >= Game.ExperienceForLevels[level-1]`,
    up to level 5, subtracting the threshold each time. The level re-reads the
    class row *and* rescales the item, so it is one number, not two.
  * **The run starts unarmed.** `Game.Team.Packs` is five bare `Novice` rows,
    which is why the first item shop is the run's first real decision. See
    `starting_squad`.

One thing here is not measured and is marked as such: which slot the shop
refills and when (modelled as: empty slots refill free the first time you enter
a shop room). The anti-repeat weighting inside one quality, which used to be
the second, is read and modelled: see `RunState.roll_item`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .battle import Battle, BattleResult, apply_class_skills, deploy, place_at
from .data import (items_by_name, items_by_quality,
                   parse_room_layouts, units_by_class)
from .nav import Grid
from .spec import MAX_UNIT_LEVEL, UnitSpec, build_player_squad, build_unit

EMPTY = "  "

# Shops.Matrix codes -> room kind. The digits distinguish which stock table an
# item shop draws from, so the leading letter is what matters here.
ROOM_KIND = {"i": "item_shop", "f": "food_shop", "m": "mutation", "e": "fight", " ": "empty"}

# A generated mutation shop, read out of `C_Room..ctor`. `C_Rooms.FromGeneratedData`
# hands the constructor its level data only for a food shop, so every other
# shop is built from the constructor's own defaults: a shelf of 3 (4 in a secret
# room, 6 in Arcade), 1 take, 2 rerolls (none in a secret room) at a roll cost of
# 0. This used to be `Rooms.Shops.Mutations.m1`, `{ShowCount 10, BuyCount 2,
# RollCount 0}`, which is what the shipped fixed map's one mutation room carries
# and is three times the shelf and twice the takes a generated level offers.
MUTATION_SHOP = {"ShowCount": 3, "BuyCount": 1, "RollCost": 0.0, "RollCount": 2}

# `C_MutationShop.Roll` decrements `rollCount` only below this; 90 or more is a
# shop that never runs out of rerolls.
UNLIMITED_ROLLS = 90

# The run carries two consumables: `M_Consumables..cctor` builds `_size` as
# `new ReactiveProperty<int>(2)` and nothing in the Default path writes it.
CONSUMABLE_SLOTS = 2

# A consumable shop's shelf, `C_Room..ctor`: `quantity = 2 + (~type & 1)`, so
# three in a boss room and two in a start room (the long-run reward's).
CONSUMABLE_SHELF = 3


@dataclass
class Consumable:
    """One consumable the run carries: its `Consumables.json` row and the
    charges left of its `Count`. `C_Consumable.Use` spends one a use, and the
    item goes when they run out."""
    row: dict
    charges: int

    @property
    def name(self) -> str:
        return self.row.get("Name") or ""


def consumable_params(row: dict) -> dict:
    """`Param1Name`/`Param1Value` and so on, as a dict."""
    return {row.get(f"Param{i}Name"): row.get(f"Param{i}Value")
            for i in range(1, 6) if row.get(f"Param{i}Name")}


def consumable_family(row: dict) -> str:
    """What a consumable does, in the terms the sim models it in.

    Read off each controller's `Use`: `C_HealConsumable` heals one unit when
    the row is `Direct` and every unit otherwise; `C_ReserveConsumable` makes
    team members when the row's `kind` carries `TeamMember` and temporary
    allies when it does not (`Kinds.TeamMember = 64`, and `C_Team.AfterFight`
    removes temporary units). `Event` opens a pool dialog
    (`C_EventConsumable.Use` calls `C_Levels.StartDialog`). `DejaVu` is a
    second life, never used by hand: see `RunState.dejavu_save`.
    `DamageAllEnemies` is on no Default level's list, so it is carried and
    sold but has no effect here.
    """
    name = row.get("Name")
    if name == "Heal":
        return "heal_one" if row.get("Direct") else "heal_all"
    if name == "Reserve":
        kind = str(consumable_params(row).get("kind") or "")
        return "reserve_humans" if "TeamMember" in kind else "reserve_ally"
    return {"RoomCleaner": "cleaner", "Resurrection": "resurrection",
            "Teleport": "teleport", "AntiHunger": "anti_hunger",
            "Reroll": "reroll", "Event": "event",
            "DejaVu": "dejavu"}.get(name, "unmodelled")


CONSUMABLE_FAMILIES = ("heal_one", "heal_all", "cleaner", "reserve_humans",
                       "reserve_ally", "resurrection", "teleport", "anti_hunger",
                       "reroll", "event", "dejavu", "unmodelled")

# `RoomState`, straight from the enum. `C_Rooms.SetCurrent` is the only thing
# that writes it, and the disassembly says exactly what it writes: the room being
# left is set to `EXPLORED`, the room being entered to `CURRENT`, and then the
# `M_Room.neighbors` walk sets every neighbour still at `UNKNOWN` to
# `UNEXPLORED`. So a level is uncovered a room at a time and the map is
# discovered, not given.
UNKNOWN, UNEXPLORED, EXPLORED, CURRENT = 0, 1, 2, 4

# What the reveal skips. The neighbour walk tests `bt eax, 0xa` on the type and
# jumps past the room when the bit is set: bit 10 is `RoomType.QuestExtra =
# 1024`, so an extra quest room is never revealed by walking next to it, and a
# room still at `UNKNOWN` cannot be walked into (see `legal_actions`). Accepting
# the quest is what puts it on the map (`sim/quests.py`).
NEVER_REVEALED = frozenset({"quest_extra"})


def room_kind(code: str) -> str:
    code = (code or " ").strip()
    return ROOM_KIND.get(code[:1], "empty") if code else "empty"


@dataclass
class Room:
    id: str
    row: int
    col: int
    kind: str
    cleared: bool = False
    # `M_Room.state`, one of the four `RoomState` values above. Everything the
    # policy is allowed to see is gated on this: a room at `UNKNOWN` is not on
    # the player's minimap at all.
    state: int = UNKNOWN
    # A shrine is one-use. Item shops do not use this: their stock is
    # `RunState.offer`, one shared shop for the whole run. Food shops do not
    # either -- they stock packs, and their stock is `food_stock`.
    stock: int | None = None
    # A mutation shop's shelf: the mutations it is showing, `None` where one has
    # been taken, how many takes are left of `MUTATION_SHOP["BuyCount"]` and
    # how many rerolls of `MUTATION_SHOP["RollCount"]`.
    offer_mutations: list | None = None
    takes_left: int = 0
    rolls_left: int = 0
    # A food shop's stock: how many of each `Rooms.Shops.Food.Packs` entry this
    # room holds, from `C_FoodShop.DetermineQuantities`. One list per room,
    # filled on arrival and never refilled: `C_FoodShop..ctor` is called from
    # `C_Room..ctor` and is the only caller of `DetermineQuantities` in play
    # (`Roll` has no call site), so a food shop is stocked once per level.
    food_stock: list[int] | None = None
    # Whether this room has already refilled the shop's empty slots. Set on
    # arrival so that walking out and back in is not a free reroll.
    refilled: bool = False
    # A boss room's consumable shop: what it is showing, `None` where a slot
    # has been taken. See `RunState.draw_consumables`.
    consumable_offer: list | None = None
    # The humans who died in this room's fights. `C_Room.CreateCorpse` leaves
    # one per dead unit, and a Resurrection used in the room brings them back.
    corpses: list = field(default_factory=list)
    # Other kinds the same room is as well. `RoomType` is a flags enum, and the
    # start room of every level after the first is `Start | ItemShop`: it keeps
    # the start's place in the economy (`CalculatePower` tests `Start` first and
    # pays it nothing) and sells items. `kind` is the role the economy and the
    # pack chooser read; `extras` is what else the room offers.
    extras: tuple = ()

    def has(self, kind: str) -> bool:
        """Whether the room is a `kind`, as its main role or as an extra one."""
        return self.kind == kind or kind in self.extras


class RoomMap:
    """One level's rooms, their adjacency, and the distance to the boss.

    Two ways in. `generate` builds the level the level row asks for -- its own
    room count and its own shop counts, see `sim/mapgen.py` -- and is what a run
    uses. `from_table` reads the fixed `Rooms.json` map, which is the one the
    game ships for a fixed level and which every level in this sim used to
    reuse; it is kept because it is real shipped data and the map the earlier
    measurements were made on.
    """

    def __init__(self, rooms: dict[str, Room], start: str, boss: str,
                 portals: list[str] | None = None):
        self.rooms = rooms
        self.start = start
        self.boss = boss
        self.portals: list[str] = list(portals or [])
        self._portal_set = frozenset(self.portals)
        if self.boss in self.rooms:
            self.rooms[self.boss].kind = "boss"
        if self.start in self.rooms:
            self.rooms[self.start].kind = "start"
        # Adjacency is fixed once the map is built, and `legal_actions` asks for
        # it on every step, so it is built once rather than scanned per call.
        by_pos = {(r.row, r.col): rid for rid, r in self.rooms.items()}
        self._adj: dict[str, list[str]] = {}
        # Doors only, which is what the reveal follows: `M_Room.neighbors` is a
        # `Dictionary<Direction, M_Room>`, so a portal link reveals nothing and
        # teleporting is `C_Rooms.TeleportTo`, a separate path. `_adj` is doors
        # plus portal links, which is what movement follows.
        self._ortho: dict[str, list[str]] = {}
        for rid, room in self.rooms.items():
            near = [by_pos[p] for p in ((room.row - 1, room.col), (room.row + 1, room.col),
                                        (room.row, room.col - 1), (room.row, room.col + 1))
                    if p in by_pos]
            self._ortho[rid] = sorted(set(near))
            if rid in self.portals:
                near += [p for p in self.portals if p != rid and p in self.rooms]
            self._adj[rid] = sorted(set(near))
        # The whole-map distance. This is ground truth for the tools and the
        # checks; **nothing the policy sees may read it**, because a player has
        # no way to know it before walking the level. `known_to_boss` is the
        # version that is fair to show, and it is empty until the boss is found.
        self.to_boss = self._distances(self.boss)
        self.current: str | None = None
        self.known_to_boss: dict[str, int] = {}
        # The quest's rooms on a generated level, see `sim/quests.py`.
        self.quest_room: str | None = None
        self.quest_extras: list[str] = []
        # The secret rooms (`Secret | Portal`, off the door graph) and the two
        # portals flagged `activatesSecret`, with the ones stood in so far.
        self.secret_rooms: list[str] = []
        self.activators: list[str] = []
        self.activated: set = set()
        # The portals on the player's map, in the order they came onto it. See
        # `revealed_portals`.
        self._seen_portals: list[str] = []

    @classmethod
    def from_table(cls, rooms_table: dict) -> "RoomMap":
        """The shipped fixed map: `Matrix` for the rooms, `Shops.Matrix` for the
        kinds."""
        matrix = rooms_table["Matrix"]
        shops = (rooms_table.get("Shops") or {}).get("Matrix") or []
        rooms: dict[str, Room] = {}
        for r, row in enumerate(matrix):
            for c, rid in enumerate(row):
                if rid == EMPTY or not rid.strip():
                    continue
                code = shops[r][c] if r < len(shops) and c < len(shops[r]) else " "
                rooms[rid] = Room(id=rid, row=r, col=c, kind=room_kind(code))
        return cls(rooms, rooms_table["Start"], rooms_table["Boss"],
                   rooms_table.get("Portals"))

    @classmethod
    def generate(cls, level_row: dict, rng: random.Random,
                 chip: str = "default", quest: dict | None = None,
                 long_run_reward: bool = False) -> "RoomMap":
        """The level the row asks for: see `sim/mapgen.py` for what is the
        game's and what is not. `quest` is the level's quest row, see
        `sim/quests.py`; `long_run_reward` makes the start a consumable shop."""
        from .mapgen import generate as _generate
        spec = _generate(level_row, rng, chip, quest, long_run_reward)
        extras = spec.get("extras") or {}
        rooms = {rid: Room(id=rid, row=r, col=c, kind=kind,
                           extras=tuple(extras.get(rid, ())))
                 for rid, (r, c, kind) in spec["rooms"].items()}
        out = cls(rooms, spec["start"], spec["boss"], spec.get("portals"))
        out.quest_room = spec.get("quest_room")
        out.quest_extras = list(spec.get("quest_extras") or [])
        out.secret_rooms = list(spec.get("secret_rooms") or [])
        out.activators = list(spec.get("activators") or [])
        return out

    def _distances(self, source: str, known_only: bool = False) -> dict[str, int]:
        import collections
        if source not in self.rooms:
            return {}
        if known_only and self.rooms[source].state == UNKNOWN:
            return {}
        out = {source: 0}
        q = collections.deque([source])
        while q:
            cur = q.popleft()
            for n in self._adj[cur]:
                if n in out or (known_only and self.rooms[n].state == UNKNOWN):
                    continue
                out[n] = out[cur] + 1
                q.append(n)
        return out

    def neighbours(self, rid: str) -> list[str]:
        """Orthogonally adjacent rooms, plus portal-to-portal links.

        What the distances are measured over. A move goes through a door, see
        `doors`, and a teleport between portals, see `teleports`.
        """
        return self._adj[rid]

    def doors(self, rid: str) -> list[str]:
        """The rooms one door away, which is all a move can reach."""
        return self._ortho[rid]

    def is_portal(self, rid: str) -> bool:
        return rid in self._portal_set

    def teleports(self, rid: str) -> list[str]:
        """Where a teleport from `rid` can land: `C_Room.CanBeTeleportedTo`.

        Read off the binary, every room the squad is not standing in, whose
        state is not `UNKNOWN`, and that is a portal, from a room that is itself
        a portal and whose fight has a result. A room's fight always has one by
        the time the squad stands in it here (`AfterFight` writes `2 * win - 1`,
        so a lost fight counts too). A target only has to be on the map, not
        visited: teleporting into a room nobody has entered fights it.
        """
        if rid not in self._portal_set:
            return []
        return [p for p in self.portals
                if p != rid and self.rooms[p].state != UNKNOWN]

    # -- fog of war --------------------------------------------------------
    def set_current(self, rid: str) -> None:
        """`C_Rooms.SetCurrent`: enter a room and reveal what it touches.

        The order is the game's. It sets the room it is leaving to `EXPLORED`,
        the room it is entering to `CURRENT`, then walks that room's doors and
        promotes each neighbour that is still `UNKNOWN` to `UNEXPLORED`. Two
        conditions guard the promotion and both are in the disassembly: the
        neighbour's type must not carry the `QuestExtra` bit, and its state must
        be exactly `UNKNOWN`, so nothing already seen is rewritten.
        """
        if self.current is not None and self.current in self.rooms:
            self.rooms[self.current].state = EXPLORED
        self.current = rid
        self.rooms[rid].state = CURRENT
        self.maybe_activate_secret_room(rid)
        for n in self._ortho[rid]:
            room = self.rooms[n]
            if room.kind not in NEVER_REVEALED and room.state == UNKNOWN:
                room.state = UNEXPLORED
        self._refresh_known()

    def maybe_activate_secret_room(self, rid: str) -> None:
        """`C_Rooms.MaybeActivateSecretRoom`: standing in a room flagged
        `activatesSecret` counts it, and at `roomsToActivateSecret` (2) every
        secret room that is not a quest's switches on (`secretIsActive`) and
        goes on the map, where a teleport can reach it."""
        if rid not in self.activators:
            return
        self.activated.add(rid)
        if len(self.activated) >= 2:
            for sid in self.secret_rooms:
                room = self.rooms[sid]
                if room.state == UNKNOWN:
                    room.state = UNEXPLORED

    def reveal_boss(self) -> bool:
        """`C_BossVisionMutation.OnNewLevel`, mutation 188 `BossVision`.

        It walks every room reading `get_type` and `get_state`, so it reveals a
        room **by type**: the boss, not a random one. Returns whether anything
        changed, which is False when the boss is already on the map.
        """
        room = self.rooms.get(self.boss)
        if room is None or room.state != UNKNOWN:
            return False
        room.state = UNEXPLORED
        self._refresh_known()
        return True

    def _refresh_known(self) -> None:
        """Recompute the distances the policy is allowed to see.

        Done here rather than on demand because state changes only in the two
        methods above, and `_encode` asks for these several times a step.
        """
        self.known_to_boss = self._distances(self.boss, known_only=True)
        fresh = sorted(p for p in self.portals
                       if self.rooms[p].state != UNKNOWN and p not in self._seen_portals)
        self._seen_portals += fresh

    def revealed_portals(self) -> list[str]:
        """The portals on the player's map, in the order they were revealed.

        Not sorted by name: a portal's place in the whole level's sorted list
        says how many portals the squad has not found yet sort before it, which
        no player can know. The order a room came onto the map is fixed once it
        has, since a revealed room is never hidden again, so an index into this
        names one room for the rest of the level.
        """
        return list(self._seen_portals)

    def known(self) -> list[str]:
        """The rooms that are on the player's map, in any state but `UNKNOWN`."""
        return [rid for rid, r in self.rooms.items() if r.state != UNKNOWN]

    def boss_found(self) -> bool:
        return (self.boss in self.rooms
                and self.rooms[self.boss].state != UNKNOWN)


@dataclass
class Food:
    """`Game.Food`, matching `C_Food`.

    The shape of this is not what it looks like from the JSON, and the sim had
    it wrong in three ways. From `C_Food.Move`, `C_Food.Feed` and
    `C_Food.OnMovesLeftChanged`:

        maxMoves  = moves[0]
        canFeed   = amount >= needed
        canMove   = canFeed or movesLeft > 0

        Move():   if canFeed: Feed(); return       # no move is spent
                  movesLeft -= 1
                  hungerLevel = 0
                  for i from len(moves)-1 down to 0:
                      if movesLeft < moves[i]: hungerLevel = i + 1; break

        Feed():   amount -= needed
                  movesLeft = maxMoves
                  hungerLevel = 0

    So: **moving feeds you** whenever the larder covers one food per human, and
    that is the whole cost of a room. `moves` is not an allowance per hunger
    stage, it is a list of *thresholds on `movesLeft`*, and `movesLeft` is a
    six-move reserve that only drains once you are broke -- which is why hunger
    can fall again as well as rise. `C_Food.Feed` has no UI call site at all
    (`C_Food.Move`, a mutation, and the sacrifice handler are the three
    callers), so feeding is not a player action and this sim used to offer it
    as one.
    """
    amount: float
    moves: list[int]
    damage_penalties: list[float]
    armor_penalties: list[float]
    per_sacrifice: float
    hunger_level: int = 0
    moves_left: int = 0
    max_moves: int = 0

    @classmethod
    def from_game(cls, game: dict) -> "Food":
        f = game["Food"]
        out = cls(amount=float(f["amount"]), moves=list(f["moves"]),
                  damage_penalties=list(f["damagePenalties"]),
                  armor_penalties=list(f["armorPenalties"]),
                  per_sacrifice=float(f.get("foodPerSacrifice") or 0))
        out.max_moves = out.moves[0]
        out.moves_left = out.max_moves
        return out

    @property
    def damage_penalty(self) -> float:
        i = min(self.hunger_level, len(self.damage_penalties) - 1)
        return self.damage_penalties[i]

    @property
    def armor_penalty(self) -> float:
        i = min(self.hunger_level, len(self.armor_penalties) - 1)
        return self.armor_penalties[i]

    def can_feed(self, needed: float) -> bool:
        """`M_Food.canFeed`: the larder covers one food per human."""
        return self.amount >= needed

    def can_move(self, needed: float) -> bool:
        """`M_Food.canMove = canFeed || movesLeft > 0`.

        Once both fail the run cannot leave the room, and there is no way back
        except sacrificing a human for food.
        """
        return self.can_feed(needed) or self.moves_left > 0

    def spend_move(self, needed: float) -> str:
        """`C_Food.Move`. Returns "fed" or "hungry" for the caller's log."""
        if self.can_feed(needed):
            self.feed(needed)
            return "fed"
        self.moves_left -= 1
        # hungerLevel is *derived* from movesLeft against the thresholds, not
        # incremented, so eating restores the penalties in one step.
        self.hunger_level = 0
        for i in range(len(self.moves) - 1, -1, -1):
            if self.moves_left < self.moves[i]:
                self.hunger_level = i + 1
                break
        return "hungry"

    def feed(self, cost: float) -> bool:
        if self.amount < cost:
            return False
        self.amount -= cost
        self.hunger_level = 0
        self.moves_left = self.max_moves
        return True


def food_packs(tables: dict) -> tuple[list[float], list[float]]:
    """`Rooms.Shops.Food.Packs` split into sizes (food) and costs (gold).

    `C_FoodShop..ctor` reads `Rooms["Shops"]["Food"]["Packs"]` as a JArray and
    puts each entry's index 0 into `sizes` and index 1 into `costs`, and
    `C_FoodShop.Buy` spends `costs[i]` of `M_Team.gold` and hands
    `C_Food.AddFood(sizes[i])` to the larder. So the pairs are **[food, gold]**:
    `[[7, 2], [11, 3], [37, 9], [120, 27], [250, 55]]` is 3.5 to 4.55 food per
    gold, the big packs being the better rate.
    """
    food = ((tables["Rooms"].get("Shops") or {}).get("Food") or {})
    packs = food.get("Packs") or []
    sizes = [float(p[0]) for p in packs]
    costs = [float(p[1]) for p in packs]
    return sizes, costs


def determine_quantities(sizes: list[float], costs: list[float],
                         budget: float) -> list[int]:
    """`C_FoodShop.DetermineQuantities`: how many of each pack a shop stocks.

    Read off the binary, and it is not the shape the column name suggests. The
    budget is `M_FoodShop.totalFood`, which the constructor sets to the level
    row's `TotalFood / FoodShops` -- but the loop spends it on **`costs`**, the
    gold price, never on `sizes`. So `TotalFood` is a gold budget for the
    level's food stock, not an amount of food, and the gold a shop's whole shelf
    costs is exactly its share of it (a 50-gold budget stocks 4x7 + 2x11 + 37 +
    120 = 207 food, priced at 4x2 + 2x3 + 9 + 27 = 50).

    The loop sweeps the packs in order, buying one of each it can still afford,
    and repeats while anything is left. Pack 0 is the exception: `j == 0` is
    bought whether or not the budget covers it, which both guarantees a shop is
    never empty and is what stops the loop -- the budget goes negative on the
    last pass.
    """
    n = min(len(sizes), len(costs))
    qty = [0] * n
    # `costs[0] <= 0` would spin forever. The shipped table has 2, and this is
    # the only thing keeping the port honest against a table that changes.
    if n == 0 or costs[0] <= 0.0:
        return qty
    remaining = float(budget)
    while remaining > 0.0:
        for i in range(n):
            if remaining >= costs[i] or i == 0:
                qty[i] += 1
                remaining -= costs[i]
    return qty


@dataclass
class Human:
    """One squad member: an item, a level, and experience toward the next.

    One level, not two. `M_Unit` has a single `level` and `CS_Units.LevelUp`
    re-reads the class row at the new level *and* re-applies the item's bonus
    at it, so the same number scales the Novice base row and the item. The
    separate `item_level` this used to carry has no counterpart in the game.
    """
    item: str | None
    level: int = 1
    experience: float = 0.0
    # The share of its health it carries into the next fight. In Default a
    # unit is not healed between fights: `C_Unit.BeforeFight` restores health
    # only in KingOfTheHill and in an Arcade boss room, `C_Unit.AfterFight`
    # never writes it, and `UnitExtensions.Heal` is reached only from a heal
    # consumable, a heal skill and `HealAfterFight` (the "Regeneration" perk of
    # the meta-progression grid, not a run mutation). `CS_Units.LevelUp` and
    # `CS_Item.ApplyBonus` both rescale health as `totalMaxHealth * ratio`, so
    # a level or an item keeps the share, which is why this is a share and not
    # a number of hit points.
    hp: float = 1.0
    # A unit of a fixed class that a quest put in the team: the Sci quest's
    # Zombie, the Rat quest's Rat. Not a human: it holds no item, is not a
    # Novice, and fights as its own `Units.json` row. None for a human.
    cls: str | None = None


def squad_names(tables: dict) -> list[str]:
    """Every startable squad, the default first.

    `ChipChoice/Squads.json` ships eight. Only `Squad1` has no
    `unlockCondition`, so it is what a fresh save starts with; the rest are
    behind Cultist, Scientist, Mage and so on, which an account that has played
    a while will have. Which one is best is not written down anywhere and is
    worth finding out rather than assuming.
    """
    squads = tables.get("Squads") or []
    free = [s["name"] for s in squads if not s.get("unlockCondition")]
    rest = [s["name"] for s in squads if s.get("unlockCondition")]
    return free + rest


def starting_squad(tables: dict, name: str | None = None) -> tuple[list[Human], list]:
    """The squad a run opens with, and the cells it opens in.

    **Not `Game.Team.Packs`.** That reads as five bare `Novice` rows with every
    entry of `Team.Items` at 0, and this sim opened on it for a long time, but a
    run starts from a squad chosen out of `ChipChoice/Squads.json`. The default
    `Squad1` is four units, three of them armed: a `stone-sword`, a `crossbow`,
    one bare and a `shield`, each with its own cell. A player reported exactly
    that roster, which is what sent us looking.

    The composition is the part that matters. Five identical unarmed bodies have
    no frontline and no ranged unit, so every early-fight number this project
    measured was measured on a squad the game never hands anyone.

    `Game.Team.Packs` stays as the fallback for a ruleset with no squad table.
    """
    squads = {s["name"]: s for s in (tables.get("Squads") or [])}
    if squads:
        chosen = squads.get(name) or squads[squad_names(tables)[0]]
        units = chosen.get("units") or []
        return ([Human(item=u.get("item")) for u in units],
                [u.get("cells") for u in units])
    team = (tables.get("Game") or {}).get("Team") or {}
    packs = team.get("Packs") or []
    squad = [Human(item=p.get("Item"), level=int(p.get("Level") or 1))
             for p in packs for _ in range(int(p.get("Number") or 1))]
    if not squad:
        squad = [Human(item=None) for _ in range(int(team.get("Humans") or 5))]
    return squad, []


@dataclass
class RunState:
    # Food spent per feeding is `M_Food.needed`, and `C_Food.SetUnitCount`
    # writes the squad's unit count straight into it: feeding costs one food per
    # human. `C_Food.Feed` is `amount -= needed`, then movesLeft = maxMoves and
    # hungerLevel = 0. This used to be a flat 10, which made a large squad free
    # to keep -- and a shaped agent went and bought seventy humans.

    tables: dict
    level: int = 1
    room: str = ""
    gold: float = 0.0
    food: Food | None = None
    squad: list[Human] = field(default_factory=list)
    mutations: list[dict] = field(default_factory=list)
    rooms: RoomMap | None = None
    rng: random.Random = field(default_factory=random.Random)
    finished: bool = False
    won: bool = False
    log: list[str] = field(default_factory=list)
    # The item shop, which is one model for the whole run (`M_Session._itemShop`)
    # rather than one per room. `offer` is its stock: one entry per slot, None
    # where a slot has been bought out.
    shop_level: int = 1
    offer: list[str | None] = field(default_factory=list)
    # How often each item has been drawn this run, per quality: the
    # `pickCount`s of `M_ItemData.weightedTemplatesByQuality`, which
    # `RootLoader.LoadData` rebuilds for every new game. See `roll_item`.
    item_picks: dict = field(default_factory=dict)
    # The low level of the hierarchy. Called as
    #   policy(grid, layout, specs, rng) -> [(row, col), ...]
    # one cell per spec. None means the default spread over the player zone.
    placement_policy: Any = None
    # Resolve fights in the Rust core when the fight is inside its envelope,
    # falling back to the Python oracle otherwise. Off by default so the
    # reference path stays the default.
    use_fast_core: bool = False
    # (item, level) -> Power, filled lazily by `item_power`
    _power_cache: dict = field(default_factory=dict)
    # (sizes, costs) of the food shop's packs, filled lazily by `food_packs`
    _food_packs: tuple[list[float], list[float]] | None = None
    # (key, specs) for `specs_cached`, and (key, mutation id) -> effect
    _specs_cache: tuple | None = None
    _mutation_cache: dict = field(default_factory=dict)
    # (level, room type) -> the eligible packs and their Power bands
    _pack_cache: dict = field(default_factory=dict)
    # The cells the chosen squad opens in, one list per unit.
    start_cells: list = field(default_factory=list)
    # The chip `load_ruleset` was given. The tables do not carry it, and level
    # generation reads it: `ChoosePortals` forces a start portal on different
    # levels under `shortpvp`.
    chip: str = "default"
    # Carry each unit's health from one fight to the next, as the game does in
    # Default (see `Human.hp`). Off reproduces the sim as it was, every fight
    # at full health, and exists only so the difference can be measured.
    carry_hp: bool = True
    # The consumables the run carries, at most `CONSUMABLE_SLOTS` of them.
    consumables: list = field(default_factory=list)
    # The boss is dead and the pipe is open, see `win_level`.
    level_won: bool = False
    # Consumable effects waiting on the next room or the next fight:
    # "cleaner" (a RoomCleaner), "anti_hunger" (AntiHunger), "allies" (the
    # temporary units a Reserve summons, as (class, level) pairs).
    armed: dict = field(default_factory=dict)
    # The level-entry dialogs not yet seen this run, and the one on screen.
    # See `sim/dialogs.py`. While a dialog is open it is the only thing the
    # run can do: the dialog screen is modal.
    dialog_pool: list | None = None
    dialog: Any = None
    # Off plays a run with no dialogs at all, for measuring what they change.
    dialogs_on: bool = True
    # Fight each pack in its own layout, zone by zone. Off is the old draw of
    # any layout with every enemy in e1, kept for measuring.
    pack_layouts: bool = True
    # (pack id, zone per enemy) of the pack `enemy_specs` last drew, and the
    # layout the last fight was fought in.
    _last_pack: tuple | None = None
    _last_layout: int | None = None
    # A DejaVu's save, see `dejavu_save`.
    _dejavu: dict | None = None
    # The level's quest controller (`sim/quests.py`), None on a level without
    # one, and the quest dialog on screen, which is modal like a level dialog.
    quest: Any = None
    quest_dialog: Any = None
    # Off plays a run with no quests: no quest rooms on the map at all, for
    # measuring what they change.
    quests_on: bool = True
    # The room the squad stood in before this one, which `C_ReaperQuest`
    # records on every move (`OnRoomChange(room, prevRoom)`).
    prev_room: str | None = None
    # Quest dialogs waiting for the one on screen to end: the Bot's
    # after-fight dialog, which its controller creates once the fight that the
    # before-fight dialog started is over.
    quest_queue: list = field(default_factory=list)
    # The persistent `longRunReward` flag as this run found it. `C_Levels.
    # WinLevel` sets it when level 4 is won in Default, and `C_Rooms.Generate`
    # reads it once, on level 1 in Default under any chip but `crazy`: it sets
    # `GenerationParams.longRunReward`, which makes the start room a consumable
    # shop, and clears the flag. So a run that beats level 4 opens the next one
    # with a two-slot consumable shop. Carrying it between runs is the
    # environment's business (`DespotRunEnv(long_run_reward="carry")`).
    long_run_reward: bool = False
    # Whether this run has set the flag for the next one.
    long_run_reward_earned: bool = False

    # -- setup -------------------------------------------------------------
    @classmethod
    def new(cls, tables: dict, seed: int = 0,
            squad_name: str | None = None, chip: str = "default",
            dialogs: bool = True, quests: bool = True,
            long_run_reward: bool = False) -> "RunState":
        game = tables["Game"]
        rng = random.Random(seed)
        squad, cells = starting_squad(tables, squad_name)
        st = cls(tables=tables, gold=float(game.get("Gold") or 0),
                 food=Food.from_game(game), squad=squad, rng=rng, chip=chip,
                 dialogs_on=dialogs, quests_on=quests,
                 long_run_reward=long_run_reward)
        st.generate_level()
        st.room = st.rooms.start
        st.rooms.set_current(st.room)
        # Level 1 opens the same way every later level does: see `next_level`.
        st.rooms.rooms[st.room].cleared = True
        # `C_ItemShop..ctor` fills the shop the moment the run starts, at level
        # 1, so the first shop room already has stock waiting.
        st.start_cells = cells
        st.offer = [st.roll_item() for _ in range(st.shop_quantity())]
        # `C_Session.Begin` reaches `C_Levels.ChangeLevel`, so the first level
        # opens with a dialog as every later one does.
        st.start_level_dialog()
        return st

    def generate_level(self) -> None:
        """Draw the level's quest, then build its map around it.

        `LevelData.Load` draws every level's quest when the data loads and
        `C_Rooms.Generate` reads it from the level data, so the quest is fixed
        before the map is built and the map is built to fit it.
        """
        from .quests import carry_on, draw_quest, make_quest
        name = draw_quest(self.level_row, self.rng) if self.quests_on else None
        if name is None and self.quest is not None and carry_on(self.quest):
            # `AdditionalLevels`: `CreateQuestModel` keeps the model and the
            # next level places the room again, from the row's `Overrides`.
            pass
        else:
            self.quest = make_quest(self.tables, name)
        self.quest_dialog = None
        self.quest_queue = []
        self.prev_room = None
        self.rooms = RoomMap.generate(self.level_row, self.rng, self.chip,
                                      self.quest.spec if self.quest else None,
                                      self.long_run_reward_used and self.level == 1)
        if self.quest is not None:
            self.quest.room = self.rooms.quest_room
            self.quest.extras = list(self.rooms.quest_extras)
            if hasattr(self.quest, "secret"):
                self.quest.secret = (self.rooms.secret_rooms or [None])[0]

    @property
    def long_run_reward_used(self) -> bool:
        """Whether `C_Rooms.Generate` takes the flag: Default (the only mode
        here) and any chip but `crazy`, which leaves it set for a later run."""
        return self.long_run_reward and self.chip != "crazy"

    def consumable_shelf(self, room: Room) -> int:
        """How many slots the consumable shop open in `room` has, 0 if none.

        `C_Room..ctor` builds one in a room typed `ConsumableShop`, with
        `quantity = 2 + (~type & 1)`. Every boss but the last is one, open once
        the boss is dead (`C_Room.AfterFight`). A start room the long-run
        reward typed is one too, open at construction, and two slots wide
        because `Start` is bit 1.
        """
        if room.kind == "boss" and self.level_won:
            return CONSUMABLE_SHELF
        if room.has("consumable_shop") and room.cleared:
            return CONSUMABLE_SHELF - 1
        return 0

    @property
    def level_row(self) -> dict:
        levels = self.tables["Levels"]
        return next((l for l in levels if l["ID"] == self.level), levels[-1])

    # -- squad -------------------------------------------------------------
    def specs(self, no_hunger: bool = False) -> list[UnitSpec]:
        """Resolve the squad to UnitSpecs, applying hunger penalties unless an
        AntiHunger is holding them off."""
        # One spec per squad member, in squad order: `fight` maps survivors back
        # by position, so the two lists must stay aligned.
        ubc = units_by_class(self.tables)
        specs = []
        for h in self.squad:
            if h.cls:
                specs.append(self.companion_spec(h))
            elif h.item:
                specs.append(build_player_squad(self.tables, [(h.item, h.level)])[0])
            else:
                specs.append(build_unit(ubc, "Novice", h.level, name="Novice"))
        if (self.food and not no_hunger
                and (self.food.damage_penalty or self.food.armor_penalty)):
            import dataclasses
            dm = 1.0 - self.food.damage_penalty / 100.0
            am = 1.0 - self.food.armor_penalty / 100.0
            specs = [dataclasses.replace(s, damage=s.damage * dm, armor=s.armor * am)
                     for s in specs]
        if self.mutations:
            from .mutations import apply_to_specs
            specs = apply_to_specs(specs, self.mutations, self.rng, gold=self.gold,
                                   hunger=self.food.hunger_level if self.food else 0)
        return specs

    @property
    def feed_cost(self) -> float:
        """`M_Food.needed`: one food per human in the squad."""
        return float(max(1, len(self.squad)))

    def item_power(self, item: str | None, level: int) -> float:
        """Power of one human holding `item` at `level`.

        Memoised because a shaped reward asks for the squad's Power on every
        step, and the shop asks for a would-be Power once per candidate target
        per purchase.
        """
        key = (item, level)
        hit = self._power_cache.get(key)
        if hit is None:
            if item:
                hit = build_player_squad(self.tables, [(item, level)])[0].power
            else:
                hit = build_unit(units_by_class(self.tables), "Novice", level,
                                 name="Novice").power
            self._power_cache[key] = hit
        return hit

    def _specs_key(self) -> tuple:
        """What the squad's resolved specs depend on, for the cache below."""
        from .mutations import hobo_fury_active
        return (tuple((h.item, h.level, h.cls) for h in self.squad),
                self.food.hunger_level if self.food else 0,
                len(self.mutations), hobo_fury_active(self.mutations, self.gold))

    def specs_cached(self) -> list[UnitSpec]:
        """`specs()`, memoised for the length of one squad state.

        `specs` rebuilds every human from the tables, and the mutation shelf
        wants the squad's stats once per shelf rather than once per slot.
        """
        key = self._specs_key()
        if self._specs_cache is None or self._specs_cache[0] != key:
            self._specs_cache = (key, self.specs())
        return self._specs_cache[1]

    def mutation_effect(self, mutation: dict) -> dict:
        """What a mutation on the shelf would do to this squad.

        Stat by stat and relative, from `sim.mutations.effect_on` -- see there
        for why this is not a Power delta.
        """
        key = (self._specs_key(), mutation.get("ID"))
        hit = self._mutation_cache.get(key)
        if hit is None:
            from .mutations import effect_on
            hit = effect_on(self.specs_cached(), mutation,
                            random.Random(int(mutation.get("ID") or 0)))
            self._mutation_cache[key] = hit
        return hit

    def squad_power(self) -> float:
        """Sum of the squad's Power, the game's own strength number.

        `Levels.json` sizes a room by the same statistic, so this over the room
        budget is a like-for-like readiness ratio.
        """
        return sum(self.companion_spec(h).power if h.cls else self.item_power(h.item, h.level)
                   for h in self.squad)

    def companion_spec(self, h: Human) -> UnitSpec:
        """A quest companion as the unit its class row at its level makes."""
        meta = self.tables["Meta"]["Classes"]
        size = int((meta.get(h.cls) or {}).get("Size") or 1)
        return build_unit(units_by_class(self.tables), h.cls, h.level, name=h.cls,
                          size=size)

    def make_companion(self, cls: str, level: int = 1) -> Human:
        """A unit of `cls` for a quest to put in the team (`C_Player.CreateUnit`
        or `DownGradeUnit` with that class)."""
        return Human(item=None, level=int(level), cls=cls)

    # -- the room-weighted economy (`C_Rooms.CalculatePower`) --------------
    # Which level-row multiplier weights a room of each kind. The three treasure
    # kinds -- shrine, mutation shop, talent shop -- share `TreasureMult`, which
    # is what `ChooseTreasureRooms` groups them by.
    ROOM_MULT = {"boss": "BossMult", "item_shop": "ItemsMult",
                 "food_shop": "FoodMult", "mutation": "TreasureMult",
                 "mutation_shop": "TreasureMult", "talent_shop": "TreasureMult"}

    # Which `EnemyPacks.RoomType` a room draws its enemies from. `M_EnemyPacks`
    # indexes packs as `Dictionary<string, Dictionary<RoomType, List<M_EnemyPack>>>`
    # and `RoomType` is a flags enum in which `Shrine = 192`, `MutationShop =
    # 320` and `TalentShop = 4160` all carry the `Treasure = 64` bit, so the
    # three treasure rooms share one pack list. The start room has no row and
    # never fights.
    PACK_ROOM_TYPE = {"boss": "Boss", "item_shop": "ItemShop",
                      "food_shop": "FoodShop", "mutation": "Treasure",
                      "mutation_shop": "Treasure", "talent_shop": "Treasure"}

    def room_mult(self, kind: str) -> float:
        """The weight a room of `kind` carries, `DefaultMult` for a plain room.

        The start room carries none: `CalculatePower` subtracts 2 from the room
        count for the start and the boss, and then adds the boss back in with
        `BossMult`, so the room the squad opens on has no enemies and no gold
        in it. The quest rooms carry none either: the total subtracts
        `questCount` and `extraQuestRoomCount` as well, and the per-room loop
        gives both kinds an `expectedPower` of 0 (an extra room's gold is the
        exception, see `room_gold`).
        """
        if kind in ("start", "quest", "quest_extra", "secret"):
            return 0.0
        lvl = self.level_row
        return float(lvl.get(self.ROOM_MULT.get(kind, "DefaultMult")) or 0.0)

    @property
    def level_weight(self) -> float:
        """`CalculatePower`'s normaliser: the level's weights, summed.

        Read off `C_Rooms.CalculatePower`: the level holds `(num - 1)` rooms'
        worth of Power and of gold, and every room takes the share its own
        multiplier buys of the total. That is why `ItemsMult` and `FoodMult`
        exist at 1.1 to 1.3 against a `DefaultMult` of 0.5 to 0.8 -- **a shop is
        a bigger fight and a bigger payout than a plain room**, not a peaceful
        stop, which is what this sim used to make of them.
        """
        return sum(self.room_mult(r.kind) for r in self.rooms.rooms.values()) or 1.0

    @property
    def num_rooms(self) -> int:
        """`num`: the level's rooms, the secret ones (made on top of it) left
        out."""
        return len(self.rooms.rooms) - len(self.rooms.secret_rooms)

    def room_share(self, column: str, kind: str) -> float:
        """One room's share of `(num - 1) * <column>`, by weight."""
        total = float(self.level_row.get(column) or 0.0) * (self.num_rooms - 1)
        return total * self.room_mult(kind) / self.level_weight

    def room_power(self, room_kind_: str = "fight") -> float:
        """The Power budget a room of this kind is filled to.

        This is `M_Room.expectedPower`, the level's Power split by the same
        weights as the gold. It used to be a flat `PowerPerRoom * mult` over
        `BossMult` or `DefaultMult` only, on the reading that shops are peaceful
        and `expectedPower` had no consumer.

        Both halves of that were wrong, and the binary says so plainly:
        `M_EnemyPacks.packsByType` is keyed by `RoomType` and `M_EnemyPack`
        carries `minPower`/`maxPower`, so a room's `expectedPower` is what picks
        its pack. `EnemyPacks.json` ships 47 ItemShop, 32 FoodShop and 53
        Treasure rows that this sim never touched. See
        `notes/reference-sim.md`, "Every room has a fight".

        The old objection, that a 1,389-Power food shop against a 750-Power
        squad makes level 1 unplayable, compared a shop's fight against a
        baseline of no fight at all. Against the level's own budget it is one
        room's share among several: at level 1 a plain room is ~700 where the
        flat reading gave 800, so the ordinary fight is *easier* and only the
        shops are harder.
        """
        return self.expected_power(room_kind_)

    def expected_power(self, kind: str) -> float:
        """`M_Room.expectedPower`: this room's share of the level's Power.

        The start room is the one place Power and gold part ways:
        `CalculatePower` tests `Start` first and gives it `ItemsMult` of Power
        and no gold. It only matters when the start room fights at all, which
        is an `Ambush` from a dialog.
        """
        if kind == "start":
            total = float(self.level_row.get("PowerPerRoom") or 0.0) * (self.num_rooms - 1)
            return total * float(self.level_row.get("ItemsMult") or 0.0) / self.level_weight
        return self.room_share("PowerPerRoom", kind)

    def room_gold(self, room_kind_: str = "fight") -> float:
        """`M_Room.goldReward`: what a room pays the first time it is cleared.

        `C_Rooms.CalculatePower` hands the level `(num - 1) * GoldPerRoom` and
        splits it by room type -- `ItemsMult` for an item shop, `FoodMult` for a
        food shop, `TreasureMult` for any of the three treasure rooms,
        `BossMult` for the boss, `DefaultMult` for everything else, and nothing
        at all for the room the run starts in. A shop is worth two to three
        plain rooms, which is the opposite of the flat `GoldPerRoom` this used
        to pay.
        """
        if room_kind_ == "quest_extra":
            # `CalculatePower` leaves both quest kinds out of its weight total
            # and pays a quest room nothing; an extra room is paid the plain
            # room's share, except on Flight and Bot (`quest == 9 or 11`).
            if self.quest is None or self.quest.name in ("Flight", "Bot"):
                return 0.0
            total = float(self.level_row.get("GoldPerRoom") or 0.0) * (self.num_rooms - 1)
            return total * float(self.level_row.get("DefaultMult") or 0.0) / self.level_weight
        return self.room_share("GoldPerRoom", room_kind_)

    # -- experience --------------------------------------------------------
    @property
    def exp_thresholds(self) -> list[float]:
        """`Game.ExperienceForLevels`: what level *n* costs, for n = 1..4."""
        return [float(x) for x in (self.tables["Game"].get("ExperienceForLevels") or [])]

    def max_experience(self, level: int) -> float:
        """What a human at `level` needs before the next one.

        `CS_Units.LevelUp` sets `maxExperience = experienceForLevels[level]`
        after raising the level, which is the new level's own threshold at a
        zero-based index -- so level *n* wants `ExperienceForLevels[n - 1]`.
        """
        t = self.exp_thresholds
        return t[level - 1] if 1 <= level <= len(t) else float("inf")

    def gain_experience(self, amount: float, targets: list[Human] | None = None) -> int:
        """`C_Team.GetExperience`: split evenly, then level up as far as it goes.

        The game divides the amount by the number of eligible units and gives
        every one of them the same share, then loops `LevelUp` while the unit
        is over its threshold and under level 5. Returns how many levels the
        squad gained, which is only for logging.
        """
        group = self.squad if targets is None else targets
        if amount <= 0.0 or not group:
            return 0
        share = amount / len(group)
        gained = 0
        for h in group:
            h.experience += self.experience_gain(h, share)
            while h.level < MAX_UNIT_LEVEL and h.experience >= self.max_experience(h.level):
                h.experience -= self.max_experience(h.level)
                h.level += 1
                gained += 1
        return gained

    def experience_gain(self, h: Human, gain: float) -> float:
        """What `gain` experience comes to for `h`, through the held
        Intelligence mutations.

        `C_IntelligenceMutation.Modify(coming, current)` sits on each covered
        unit's experience: a rise of `g` becomes `g * (1 + value / 100)` when
        `percentage` (the model's default) and `g + value` otherwise, never
        below zero, so bought experience grows as well as earned. A "Random"
        class is the one `mutation_effect` shows for the mutation, drawn from
        the squad's classes off the mutation's id: the run layer has no
        offered class to keep (see `mutation_pool_by_id_all_unlocked`).
        """
        from .mutations import applies_to, mutation_params
        smart = [m for m in self.mutations if m["Name"] == "Intelligence"]
        if not smart or gain <= 0.0:
            return gain
        cls = h.cls or self.item_class(h.item)
        for m in smart:
            random_class = None
            if m.get("Class") == "Random":
                classes = sorted({x.cls or self.item_class(x.item) for x in self.squad})
                random_class = random.Random(int(m.get("ID") or 0)).choice(classes)
            if not applies_to(m, cls, None, random_class):
                continue
            p = mutation_params(m)
            value = float(p.get("value") or 0.0)
            pct = str(p.get("percentage", "true")).lower() == "true"
            gain = max(0.0, gain * (1.0 + value / 100.0) if pct else gain + value)
        return gain

    # -- the item shop -----------------------------------------------------
    @property
    def shop_rows(self) -> list[dict]:
        return sorted(self.tables["ItemShopData"], key=lambda r: int(r.get("Level") or 1))

    @property
    def max_shop_level(self) -> int:
        """`M_ItemShopData.levelCount`, which caps `C_ItemShop.Upgrade`."""
        return len(self.shop_rows)

    def shop_row(self, level: int | None = None) -> dict:
        rows = self.shop_rows
        i = min(max((self.shop_level if level is None else level) - 1, 0), len(rows) - 1)
        return rows[i]

    def shop_quantity(self, level: int | None = None) -> int:
        """Slots at this shop level: `quantities[level - 1]`."""
        return int(self.shop_row(level).get("Quantity") or 3)

    @property
    def roll_cost(self) -> float:
        """`rollCosts[level - 1]`, the price of re-rolling the stock."""
        return float(self.shop_row().get("RollCost") or 0)

    @property
    def upgrade_cost(self) -> float:
        """`prices[level]` -- the *next* level's Price, not the current one.

        `C_ItemShop..ctor` reads `prices[1]` while the shop is at level 1, and
        `Upgrade` re-reads `prices[level]` after raising it, so `prices[0]`
        (which is 0) is never charged for anything.
        """
        rows = self.shop_rows
        if self.shop_level >= len(rows):
            return float("inf")
        return float(rows[self.shop_level].get("Price") or 0)

    @property
    def _shop_config(self) -> dict:
        """`Rooms.Shops.Items`, which `C_ItemShop..ctor` reads its costs from."""
        return (self.tables["Rooms"].get("Shops") or {}).get("Items") or {}

    @property
    def exp_cost(self) -> float:
        """`ExperienceCost`: what one `BuyExperience` costs in gold."""
        return float(self._shop_config.get("ExperienceCost") or 0)

    @property
    def exp_amount(self) -> float:
        """`ExperienceAmount`: the pool one purchase adds, before the split."""
        return float(self._shop_config.get("ExperienceAmount") or 0)

    # -- the food shop -----------------------------------------------------
    @property
    def food_packs(self) -> tuple[list[float], list[float]]:
        """The shop's shelf: `(sizes, costs)`, one entry per pack.

        Cached because `legal_actions` asks for it on every step.
        """
        if self._food_packs is None:
            self._food_packs = food_packs(self.tables)
        return self._food_packs

    @property
    def food_shop_budget(self) -> float:
        """What one food shop on this level has to stock itself with.

        `C_FoodShop..ctor` is `M_FoodShop.totalFood = TotalFood / FoodShops`,
        and the level now really has `FoodShops` food shops on it, so this is
        the game's own expression. While every level reused the shipped 22-room
        map it could not be: that map has six food shops where level 1 asks for
        one, and using the row's divisor would have put a whole level's food
        supply in each of six rooms.
        """
        shops = float(self.level_row.get("FoodShops") or 0)
        return float(self.level_row.get("TotalFood") or 0) / max(1.0, shops)

    def quality_weights(self, level: int | None = None) -> dict[int, float]:
        """`probabilities[level - 1]` keyed by quality, zeroes dropped.

        `C_ItemShop.<FillWeightedList>b__4_0(quality)` returns exactly
        `probabilities[shopLevel - 1, quality - 1]`, and a zero weight can
        never be drawn.
        """
        row = self.shop_row(level)
        out = {}
        for q in items_by_quality(self.tables):
            w = row.get(f"Q{q}Prob")
            if w:
                out[q] = float(w)
        return out

    def roll_item(self) -> str | None:
        """One slot: draw a quality by weight, then an item of that quality.

        `C_ItemShop.Roll` takes the quality from `weightedQualities`, a plain
        `WeightedList<int>` whose weights stay the shop level's
        probabilities, and the item from that quality's
        `DynamicWeightedList<JToken>` (`ItemData.Load` builds one per quality
        with no weight function, so every item starts at 1, and factor 2). It
        draws the index and then calls the list's own `GetElement(index)`,
        which counts the pick and runs `AdjustWeights`: every item's weight
        becomes `1 / 2^(picks - fewest picks in the quality)`. So an item the
        shop has shown is half as likely next time as one it has shown less,
        whether or not it was bought, for the rest of the run.
        """
        weights = self.quality_weights()
        if not weights:
            return None
        pool = items_by_quality(self.tables)
        qualities = sorted(weights)
        q = self.rng.choices(qualities, weights=[weights[x] for x in qualities])[0]
        names = pool.get(q) or []
        if not names:
            return None
        picks = self.item_picks.setdefault(q, {})
        pick = self.rng.choices(names, weights=self.item_weights(q, names))[0]
        picks[pick] = picks.get(pick, 0) + 1
        return pick

    def item_weights(self, quality: int, names: list[str]) -> list[float]:
        """`DynamicWeightedList.AdjustWeights` over one quality's items:
        `1 / factor^(pickCount - minPickCount)`, factor 2. The shipped weight
        function is none, so before any pick every item weighs 1, which the
        same formula gives."""
        picks = self.item_picks.get(quality, {})
        fewest = min((picks.get(n, 0) for n in names), default=0)
        return [1.0 / 2.0 ** (picks.get(n, 0) - fewest) for n in names]

    def item_cost(self, name: str) -> float:
        """What the shop charges: the item's own `Cost`.

        `C_ItemShop.Buy` compares `M_Team.gold` against `M_Item.cost` and
        subtracts that. The shop row's `Price` is the upgrade cost and was
        being charged here instead, which made every item the same price and
        made quality free.
        """
        row = items_by_name(self.tables).get(name)
        return float((row or {}).get("Cost") or 0)

    def refill_offer(self) -> None:
        """Fill empty slots, free. `C_ItemShop.Roll(free, onlyEmpty)`.

        When this happens is the one part of the shop not read off the binary:
        `Roll` is only ever called through the UI, so there is no call site to
        follow. It is modelled as happening once when you walk into a shop
        room, which is why `Room.refilled` exists -- otherwise leaving and
        coming back would be a free reroll.
        """
        want = self.shop_quantity()
        if len(self.offer) < want:
            self.offer += [None] * (want - len(self.offer))
        for i, slot in enumerate(self.offer):
            if slot is None:
                self.offer[i] = self.roll_item()

    def best_item_target(self, name: str) -> int | None:
        """Which human gains the most Power from `name`.

        The game lets the player drop a bought item onto any human, so `apply`
        takes an explicit target. The run-level action space does not enumerate
        targets -- that is a second choice on top of a choice, the same shape
        as placement -- so when none is given the item goes to whoever the
        game's own strength statistic says gains most by it. An unarmed human
        wins that by default, since a bare Novice gains the item's whole Power.

        Note the gain can be negative: buying a worse item than the squad
        already holds downgrades whoever loses least by it. That is what
        dropping the item on someone does, and a policy that buys at a loss
        should be charged for it.
        """
        humans = [i for i, h in enumerate(self.squad) if not h.cls]
        if not humans:
            return None
        gains = {i: self.item_power(name, self.squad[i].level)
                 - self.item_power(self.squad[i].item, self.squad[i].level)
                 for i in humans}
        return max(humans, key=lambda i: (gains[i], -i))

    # -- enemies -----------------------------------------------------------
    def _eligible_packs(self, want: str) -> list[list[dict]]:
        """The level's packs for one room type, each grouped into its rows.

        A pack is several `EnemyPacks.json` rows sharing an `ID`, one per zone:
        `M_EnemyPack` holds `enemiesByZone`, and pack 4 is `Mancrack e1 1-20`
        with `RangeBlinker e2 1-1`. `Min` and `Max` are that row's unit count,
        which `M_EnemyPack.Create` draws between while it places units of
        `Class` at `Level` with `Item` into the layout's zone.
        """
        lvl = self.level_row
        allowed = {r["Pack"] for r in self.tables["PacksByStyle"]
                   if lvl["MinStyleID"] <= r["Style"] <= lvl["MaxStyleID"]}
        groups: dict = {}
        for row in self.tables["EnemyPacks"]:
            if row.get("RoomType") in (want, None) and row.get("ID") in allowed:
                groups.setdefault(row["ID"], []).append(row)
        if not groups:
            for row in self.tables["EnemyPacks"]:
                if row.get("RoomType") == want:
                    groups.setdefault(row["ID"], []).append(row)
        return list(groups.values())

    def _pack_units(self, rows: list[dict]) -> list[tuple[UnitSpec, int, int, str]]:
        """One pack as (unit, min count, max count, zone) per row."""
        ubc = units_by_class(self.tables)
        meta = self.tables["Meta"]["Classes"]
        out = []
        for row in rows:
            cls = row["Class"]
            if cls not in ubc:
                continue
            size = int((meta.get(cls) or {}).get("Size") or 1)
            unit = build_unit(ubc, cls, int(row.get("Level") or 1), name=cls,
                              size=size)
            lo, hi = int(row.get("Min") or 1), int(row.get("Max") or 1)
            out.append((unit, min(lo, hi), max(lo, hi), row.get("Zone") or "e1"))
        return out

    def _pack_table(self, want: str) -> list[tuple[list, float, float]]:
        """Every eligible pack with the Power band it can come out at.

        `M_EnemyPack` carries `minPower` and `maxPower`, which are not columns in
        the JSON, so they are the band the pack's own counts span. Cached per
        (level, room type): building the units costs more than the fight does.
        """
        key = (self.level, want)
        cached = self._pack_cache.get(key)
        if cached is not None:
            return cached
        table = []
        for rows in self._eligible_packs(want):
            units = self._pack_units(rows)
            if not units:
                continue
            lo = sum(u.power * a for u, a, _, _ in units)
            hi = sum(u.power * b for u, _, b, _ in units)
            table.append((units, lo, hi, rows[0].get("ID")))
        self._pack_cache[key] = table
        return table

    def enemy_specs(self, room_kind_: str) -> list[UnitSpec]:
        """The pack this room draws, instantiated.

        **`expectedPower` picks a pack; it is not a budget to fill.**
        `M_EnemyPacks` indexes packs by room type, `M_EnemyPack` carries a
        `minPower`/`maxPower` band, and neither `C_Rooms.ChoosePacks` nor
        `M_EnemyPack.Create` compares a running total against the room: `Create`
        reads `Class`, `Level`, `Item`, `Size`, `Min` and `Max` and places that
        many units, and errors with "There are no places of size {0} in layout
        ID={1}" when the layout cannot hold them.

        This used to draw packs repeatedly until their summed Power reached the
        room's budget, which is a different game: with the corrected budgets a
        level-1 item shop came out at 1,682 Power against a 750-Power starting
        squad and no run got past level 1.
        """
        table = self._pack_table(self.PACK_ROOM_TYPE.get(room_kind_, "Default"))
        if not table:
            return []
        budget = self.room_power(room_kind_)
        # The pack whose band the room's Power falls in, else the nearest band.
        inside = [t for t in table if t[1] <= budget <= t[2]]
        if inside:
            units, _, _, pack_id = self.rng.choice(inside)
        else:
            units, _, _, pack_id = min(table, key=lambda t: min(abs(budget - t[1]),
                                                       abs(budget - t[2])))
        # Counts start at each row's `Min` and are raised toward the room's
        # Power, never past that row's `Max`. That the count is bounded by the
        # row rather than free is the part the data shows; the exact rule for
        # picking inside the bounds is not recovered, so this fills the cheapest
        # way that respects both ends. Drawing uniformly in [Min, Max] instead
        # puts 14 units and 23,000 Power in a level-1 item shop, because `Max`
        # runs to 20 and 40 on some rows.
        counts = [lo for _, lo, _, _ in units]
        power = sum(u.power * c for (u, _, _, _), c in zip(units, counts))
        room = [i for i, (_, lo, hi, _) in enumerate(units) if hi > lo]
        while room:
            # Add the unit that lands the room closest to its Power, and stop
            # when no addition gets closer. Filling while `power < budget`
            # instead overshoots by a whole unit, which on a level-1 item shop
            # is 2x the room: one Mancrack is 1,642 Power against a budget of
            # 1,682, and the second takes it to 3,284.
            here = abs(budget - power)
            best, gain = None, here
            for i in room:
                d = abs(budget - (power + units[i][0].power))
                if d < gain:
                    best, gain = i, d
            if best is None:
                break
            unit, _, hi, _ = units[best]
            counts[best] += 1
            power += unit.power
            if counts[best] >= hi:
                room.remove(best)
        out: list[UnitSpec] = []
        zones: list[str] = []
        for (unit, _, _, zone), c in zip(units, counts):
            out += [unit] * c
            zones += [zone] * c
        # Where they stand, for `fight`: the pack's layout and each unit's zone.
        self._last_pack = (pack_id, zones)
        return out

    # -- actions -----------------------------------------------------------
    # -- dialogs -------------------------------------------------------------
    def _dialog_engine(self):
        from .dialogs import Engine
        return Engine(self)

    def start_level_dialog(self, from_consumable: bool = False) -> bool:
        """`C_Levels.StartDialog`: a random pool dialog that can start, once a
        run. Returns whether one started."""
        if not self.dialogs_on:
            return False
        from .dialogs import pool
        if self.dialog_pool is None:
            self.dialog_pool = pool(self.tables)
        engine = self._dialog_engine()
        ready = [n for n in self.dialog_pool if engine.can_start(n, from_consumable)]
        if not ready:
            return False
        name = self.rng.choice(ready)
        self.dialog_pool.remove(name)
        self.dialog = engine.start(name, from_consumable)
        self._settle_dialog()
        return True

    def _settle_dialog(self) -> None:
        """Walk through nodes with one choice, close a dialog that has run out
        of choices, and fight the ambush a closed dialog left behind.

        A node with a single choice is a "Continue": taking it is not a
        decision, so it is taken here rather than spending one of the
        policy's steps.
        """
        engine = self._dialog_engine()
        while self.dialog is not None:
            legal = engine.legal(self.dialog)
            if len(legal) == 1:
                if not engine.choose(self.dialog, legal[0]):
                    self._close_dialog()
                continue
            if not legal:
                self._close_dialog()
            break

    def _close_dialog(self) -> None:
        d, self.dialog = self.dialog, None
        if d is not None and d.ambush and self.squad:
            # `Ambush.Apply` resets the room's `fightResult` and closes its
            # doors: the room the squad stands in fights after all.
            room = self.rooms.rooms[self.room]
            room.cleared = False
            result = self.fight(room)
            if result.get("won", True):
                room.cleared = True
        if not self.squad:
            self.finished, self.won = True, False

    # -- DejaVu ---------------------------------------------------------------
    _NOT_SAVED = ("tables", "_power_cache", "_specs_cache", "_mutation_cache",
                  "_pack_cache", "_food_packs", "placement_policy", "_dejavu")

    def dejavu_save(self) -> None:
        """`C_DejaVuConsumable.PrepareData`: save the run, for a rewind.

        The controller saves when the consumable is taken and again whenever
        a level starts while it is held, subscribes to `C_Session.OnLoss`, and
        on a wipe opens the `DejaVu` dialog, whose one choice loads the save
        (`Loader.LoadGameFromData`). Rows from the tables are shared, not
        copied: nothing writes to them.
        """
        import copy
        memo = {id(self.tables): self.tables}
        for m in self.mutations:
            memo[id(m)] = m
        for c in self.consumables:
            memo[id(c.row)] = c.row
        state = {k: v for k, v in self.__dict__.items() if k not in self._NOT_SAVED}
        self._dejavu = copy.deepcopy(state, memo)

    def holds_dejavu(self) -> bool:
        return any(consumable_family(c.row) == "dejavu" for c in self.consumables)

    def dejavu_rewind(self) -> bool:
        """A wipe with a DejaVu carried: back to its save, the DejaVu spent.

        Whether the reloaded save still carries the DejaVu is not read
        (`sim/assumptions.py`); it is taken out, so it saves one run once. The
        random stream is not rewound with the rest: a reloaded game does not
        replay its fights bit for bit (`notes/rust-core.md` measured a 1e-4
        nudge flipping one fight in twelve), and rewinding it would make a
        deterministic policy die in the same place again.
        """
        if self._dejavu is None or not self.holds_dejavu():
            return False
        import copy
        save, self._dejavu = self._dejavu, None
        rng = self.rng
        self.__dict__.update(copy.deepcopy(save, {id(self.tables): self.tables}))
        self.rng = rng
        for c in list(self.consumables):
            if consumable_family(c.row) == "dejavu":
                self.consumables.remove(c)
                break
        self.finished, self.won = False, False
        self.log.append("dejavu")
        return True

    def item_class(self, item: str | None) -> str:
        """The class a human holding `item` fights as; a bare one is a Novice."""
        if not item:
            return "Novice"
        return (items_by_name(self.tables).get(item) or {}).get("Class") or "Novice"

    def mutation_by_id(self, mid: int) -> dict | None:
        from .mutations import by_id
        return by_id(self.tables).get(mid)

    def add_human(self, item: str | None = None, level: int = 1) -> Human:
        h = Human(item=item, level=max(1, int(level)))
        self.squad.append(h)
        return h

    def give_item(self, name: str) -> None:
        """An item handed over outside a shop, to whoever gains most by it.

        `GiveItem` puts it in the team's items for the player to drop on a
        human; the run layer picks the human, as it does for a bought item.
        """
        if name not in items_by_name(self.tables) or not self.squad:
            return
        target = self.best_item_target(name)
        if target is not None:
            self.squad[target].item = name

    def take_item(self, prefer: str | None = None) -> float:
        """`C_Player.DownGradeUnit` on a human holding an item, one of class
        `prefer` first. Returns the item's cost, which `RemoveItem` hands the
        dialog as `itemCost`."""
        armed = [h for h in self.squad if h.item]
        if not armed:
            return 0.0
        pick = next((h for h in armed if self.item_class(h.item) == prefer), armed[0])
        cost = self.item_cost(pick.item)
        pick.item = None
        return cost

    def legal_actions(self) -> list[tuple]:
        if self.finished:
            return []
        if self.quest_dialog is not None:
            return [("choose", i) for i in self.quest_dialog.options(self)]
        if self.dialog is not None:
            return [("choose", i) for i in self._dialog_engine().legal(self.dialog)]
        acts: list[tuple] = []
        here = self.rooms.rooms[self.room]
        if self.food.can_move(self.feed_cost):
            # `RoomTransition.CheckDirection` wants the door open and the room
            # behind it off `Unknown`. Entering a room reveals every neighbour
            # but a quest's extra room, so that is the one door this holds shut
            # until the quest puts the room on the map.
            acts += [("move", rid) for rid in self.rooms.doors(self.room)
                     if self.rooms.rooms[rid].state != UNKNOWN]
        # A teleport is not a move and costs no food. `C_Food.Move` has two
        # callers, the walk through a door (`RoomTransition.OnTransitBlocked`)
        # and `C_Rooms.InstantMove`; the minimap click that teleports
        # (`V_MinimapRoom.OnLeftClick` -> `C_Rooms.TeleportTo`) reaches neither,
        # and `M_Food.canMove` is read on the door path only. So a starving
        # squad standing on a portal can still teleport. This used to be a
        # portal "move" that charged a feeding, on generated maps that had no
        # portals at all.
        acts += [("teleport", rid) for rid in self.rooms.teleports(self.room)]
        shop = self._shop_config
        self.ensure_stock(here)
        # Nothing in a room is usable until its fight is won: `C_Room.AfterFight`
        # is what calls `M_Shop.set_open` and `V_Shrine.Open`, and `InitShop`
        # decides the shop's opening state from `M_Room.fightResult`. `RoomType`
        # even carries an inactive variant of every shop for this state
        # (`InactiveItemShop = 8196` is `Inactive | ItemShop`). A room the squad
        # walked into and did not win therefore offers nothing but a way out.
        open_ = here.cleared
        if open_ and here.has("item_shop"):
            # One action per slot, so the slot index is stable and the policy
            # can learn "the expensive one" rather than "an item".
            for i, name in enumerate(self.offer):
                if name is not None and self.gold >= self.item_cost(name):
                    acts.append(("buy_item", (i, None)))
            if self.gold >= self.roll_cost:
                acts.append(("reroll", None))
            if self.shop_level < self.max_shop_level and self.gold >= self.upgrade_cost:
                acts.append(("upgrade_shop", None))
            # Buying experience for a squad that is already all level 5 spends
            # gold for nothing. The state would still change, so it is not the
            # no-op trap that `feed` was, but it is a pure sink and there is no
            # reason to leave it in the mask.
            if (self.gold >= self.exp_cost
                    and any(h.level < MAX_UNIT_LEVEL for h in self.squad)):
                acts.append(("buy_exp", None))
        if open_ and here.kind == "food_shop":
            # One action per pack, the same reason the item shop is one action
            # per slot: pack 4 is always the 250-food one, so "the big pack" is
            # learnable where a single `buy_food` could only mean "some food".
            _, costs = self.food_packs
            for i, held in enumerate(here.food_stock or []):
                if held > 0 and self.gold >= costs[i]:
                    acts.append(("buy_food", i))
        if open_ and here.kind == "mutation" and here.stock > 0:
            acts.append(("take_mutation", None))
        if open_ and here.kind == "mutation_shop" and here.takes_left > 0:
            # One action per slot, like the item shop: which mutation is on
            # offer is the whole decision, so the slot index has to be stable.
            for i, m in enumerate(here.offer_mutations or []):
                if m is not None:
                    acts.append(("buy_mutation", i))
            # `C_MutationShop.Roll` refuses when the purse is short of
            # `rollCost` or no rerolls are left. Once the take is spent the
            # shop is closed and the reroll goes with it.
            if here.rolls_left > 0 and self.gold >= float(MUTATION_SHOP["RollCost"]):
                acts.append(("reroll_mutations", None))
        # The boss room once the boss is dead: the pipe down, and on every level
        # but the last a consumable shop. `C_ConsumableShop.Buy` never reads
        # the purse and every shipped consumable costs 0, so the only limit is
        # room to carry it.
        if open_ and here.kind == "boss" and self.level_won:
            acts.append(("descend", None))
        if open_ and self.consumable_shelf(here) and len(self.consumables) < CONSUMABLE_SLOTS:
            for i, c in enumerate(here.consumable_offer or []):
                if c is not None:
                    acts.append(("take_consumable", i))
        # What the run carries can be used where its controller allows, and
        # sold anywhere for `Rooms.Shops.Consumables.SellCost`.
        for k, c in enumerate(self.consumables):
            if self.can_use(c, here):
                acts.append(("use_consumable", k))
            acts.append(("sell_consumable", k))
        # Humans are bought at the item shop, not anywhere on the map:
        # `C_ItemShop.BuyHuman` is the only caller, and it checks `M_Team.gold`
        # against `M_ItemShop.humanCost` (a flat 2, never raised after a sale).
        # This was unrestricted here, and a trained agent found it: 73% of its
        # actions were `buy_human` and one run ended with 277 humans.
        if open_ and here.has("item_shop") and self.gold >= float(shop.get("HumanCost") or 2):
            acts.append(("buy_human", None))
        if len(self.squad) > 1 and self.food.per_sacrifice:
            acts.append(("sacrifice", None))
        # There is no `feed`. `C_Food.Feed` has no UI call site: moving feeds
        # the team when the larder covers it, so food is the price of a room
        # rather than something a policy decides to spend. Offering it as an
        # action made food nearly free and let an agent hoard it.
        return acts

    def apply(self, action: tuple) -> dict:
        kind, arg = action
        shop = self._shop_config
        result: dict = {"action": kind}

        if kind == "choose":
            if self.quest_dialog is not None:
                i = int(arg or 0)
                if i in self.quest_dialog.options(self):
                    result["quest"] = self.quest_dialog.stem
                    self.quest_dialog.choose(self, i)
                    self._settle_quest_dialog()
            elif self.dialog is not None:
                engine = self._dialog_engine()
                i = int(arg or 0)
                if i in engine.legal(self.dialog):
                    result["dialog"] = self.dialog.name
                    if not engine.choose(self.dialog, i):
                        self._close_dialog()
                    self._settle_dialog()
        elif kind in ("move", "teleport"):
            # `C_Rooms.TeleportTo` runs `SetCurrent` and `C_Room.AfterMoveInto`
            # exactly as a move does, so arriving is the same either way. Only
            # the door walk feeds the squad.
            if kind == "move":
                result["food"] = self.food.spend_move(self.feed_cost)
            self._arrive(arg, result)
        elif kind == "descend":
            # `V_Pipes.<MoveInternal>` is the only gameplay caller of
            # `C_Levels.ChangeLevel`, and the pipe opens on the boss room's
            # `fightResult`. Until the squad takes it the run is still on this
            # level, doors open, shops still selling.
            if self.level_won and self.rooms.rooms[self.room].kind == "boss":
                self.next_level()
        elif kind == "buy_item":
            # arg is (slot, target). A target of None means "whoever the game's
            # own Power statistic says gains most", see `best_item_target`.
            slot, target = arg if isinstance(arg, tuple) else (arg, None)
            name = self.offer[slot] if 0 <= slot < len(self.offer) else None
            price = self.item_cost(name) if name else 0.0
            if name is not None and self.gold >= price and self.squad:
                if target is None:
                    target = self.best_item_target(name)
                if target is not None and 0 <= target < len(self.squad):
                    self.gold -= price
                    self.offer[slot] = None
                    self.squad[target].item = name
                    result["bought"] = name
                    result["target"] = target
        elif kind == "reroll":
            # `C_ItemShop.Roll(bool free = False, bool onlyEmpty = False)`: the
            # paid reroll takes both defaults, so it re-rolls the whole shelf
            # and fills it back to `quantity`. `onlyEmpty` exists for the free
            # refill, which is the other caller. Bounded by gold, which a room
            # now pays once.
            if self.gold >= self.roll_cost:
                self.gold -= self.roll_cost
                self.offer = [self.roll_item() for _ in range(self.shop_quantity())]
                result["rerolled"] = True
        elif kind == "upgrade_shop":
            cost = self.upgrade_cost
            if self.shop_level < self.max_shop_level and self.gold >= cost:
                self.gold -= cost
                self.shop_level += 1
                # A level adds slots, and `Upgrade` re-rolls into them.
                self.refill_offer()
                result["shop_level"] = self.shop_level
        elif kind == "buy_exp":
            if self.gold >= self.exp_cost and self.squad:
                self.gold -= self.exp_cost
                result["levels"] = self.gain_experience(self.exp_amount)
        elif kind == "buy_food":
            # `C_FoodShop.Buy(index)`: refuse a sold-out pack, refuse one the
            # purse does not cover, then gold -= costs[i] and
            # `C_Food.AddFood(sizes[i])`. This used to sell 5 food for a flat 1
            # gold three times a shop, which is both a better rate than any real
            # pack and a fifteen-food ceiling on a whole shop -- harmless while
            # food was free, load-bearing now that a room costs one per human.
            here = self.rooms.rooms[self.room]
            self.ensure_stock(here)
            sizes, costs = self.food_packs
            i = int(arg or 0)
            stock = here.food_stock or []
            if 0 <= i < len(stock) and stock[i] > 0 and self.gold >= costs[i]:
                stock[i] -= 1
                self.gold -= costs[i]
                self.food.amount += sizes[i]
                result["bought_food"] = sizes[i]
                result["cost"] = costs[i]
                # Mutation 356, `FoodPackHeal`: each unit gets back `cost *
                # koefficient` percent of its health, up to full. The
                # controller watches the shop's stock and turns what was bought
                # into gold spent, so the heal is by price, not by pack size.
                for m in self.mutations:
                    if m.get("Name") == "FoodPackHeal":
                        from .mutations import mutation_params
                        from .unit_skills import param
                        k = param(mutation_params(m), "koefficient", default=1.0)
                        for h in self.squad:
                            h.hp = min(1.0, h.hp + costs[i] * k / 100.0)
        elif kind == "buy_human":
            here = self.rooms.rooms[self.room]
            cost = float(shop.get("HumanCost") or 2)
            if here.has("item_shop") and self.gold >= cost:
                self.gold -= cost
                self.squad.append(Human(item=None))
        elif kind == "sacrifice":
            # The player drags a human of their choosing into the pit. Who is
            # not in the action space, the same way the target of a bought item
            # is not, so it is the most wounded human, the last one on a tie:
            # with wounds carried between fights that is the one losing least.
            worst = min(range(len(self.squad)),
                        key=lambda i: (self.squad[i].hp, -i))
            self.squad.pop(worst)
            self.food.amount += self.food.per_sacrifice
            if self.quest is not None:
                self.quest.sacrificed(self)
        elif kind == "buy_mutation":
            # `BaseMutationShop.Buy`: refuse one already taken, refuse one the
            # purse does not cover, then gold -= cost and the mutation is the
            # team's. Nothing in `Mutations.json` carries a price and
            # `M_Mutation.cost` is left at zero for a shop mutation, so this is
            # free and bounded by `BuyCount` rather than by gold.
            here = self.rooms.rooms[self.room]
            self.ensure_stock(here)
            i = int(arg or 0)
            shelf = here.offer_mutations or []
            if 0 <= i < len(shelf) and shelf[i] is not None and here.takes_left > 0:
                m = shelf[i]
                shelf[i] = None
                here.takes_left -= 1
                self.mutations.append(m)
                result["mutation"] = m["Name"]
                # `C_MutationShop.Buy`: with takes left and nothing unbought on
                # the shelf, a free `Roll` fills it again.
                if here.takes_left > 0 and all(x is None for x in shelf):
                    here.offer_mutations = self.draw_mutations(here)
        elif kind == "reroll_mutations":
            # `C_MutationShop.Roll(free: false)`: pay `rollCost`, spend a
            # reroll (unless the shop has 90 or more), and build a whole new
            # shelf. `UnregisterMutations` puts the old shelf back in the pool
            # first, so a reroll can show some of the same mutations again.
            here = self.rooms.rooms[self.room]
            cost = float(MUTATION_SHOP["RollCost"])
            if (here.kind == "mutation_shop" and here.takes_left > 0
                    and here.rolls_left > 0 and self.gold >= cost):
                self.gold -= cost
                if here.rolls_left < UNLIMITED_ROLLS:
                    here.rolls_left -= 1
                here.offer_mutations = self.draw_mutations(here)
                result["rerolled"] = True
        elif kind == "take_consumable":
            here = self.rooms.rooms[self.room]
            shelf = here.consumable_offer or []
            i = int(arg or 0)
            if (self.consumable_shelf(here) and 0 <= i < len(shelf)
                    and shelf[i] is not None
                    and len(self.consumables) < CONSUMABLE_SLOTS):
                row = shelf[i]
                shelf[i] = None
                self.consumables.append(
                    Consumable(row=row, charges=int(row.get("Count") or 1)))
                result["consumable"] = row.get("Name")
                # `C_DejaVuConsumable.Take` enables it with a fresh save.
                if consumable_family(row) == "dejavu":
                    self.dejavu_save()
        elif kind == "use_consumable":
            k = int(arg or 0)
            here = self.rooms.rooms[self.room]
            if 0 <= k < len(self.consumables) and self.can_use(self.consumables[k], here):
                result.update(self.use_consumable(k))
        elif kind == "sell_consumable":
            # `C_Consumable.Sell`: out of the inventory, `sellCost` into the purse.
            k = int(arg or 0)
            if 0 <= k < len(self.consumables):
                result["sold"] = self.consumables.pop(k).name
                self.gold += self.consumable_sell_price
        elif kind == "take_mutation":
            here = self.rooms.rooms[self.room]
            pool = self.mutation_pool() if (here.stock or 0) > 0 else []
            if pool:
                here.stock -= 1
                m = self.rng.choice(pool)
                self.mutations.append(m)
                result["mutation"] = m["Name"]

        if not self.squad:
            self.finished, self.won = True, False
            # `C_DejaVuConsumable.OnLoss` catches the loss and rewinds.
            if self.dejavu_rewind():
                result["dejavu"] = True
        elif self.stranded():
            # Soft-locked: no food, no moves in reserve, and too small a squad
            # to sacrifice from, so `M_Food.canMove` is false and the run
            # cannot continue. Ending it here is what stops a stranded run from
            # being *cheaper* than a wipe -- the environment only charges its
            # terminal penalty on `finished`, so a soft-lock used to end the
            # episode at reward zero, and an agent trained long enough learned
            # to aim for it.
            self.finished, self.won = True, False
        return result

    def _arrive(self, rid: str, result: dict) -> None:
        """Walk or teleport into `rid`: reveal, fight if it was never won, pay."""
        self.prev_room = self.room
        self.room = rid
        # The reveal happens on entry, before the fight, so the neighbours
        # of a room the squad dies in are still uncovered.
        self.rooms.set_current(rid)
        room = self.rooms.rooms[rid]
        first_visit = not room.cleared
        # A cleaner armed before this step is spent on this room whatever it
        # is, the way using one in a room that has already fought spends it.
        cleaner = self.armed.pop("cleaner", False)
        if first_visit:
            if cleaner and room.kind != "boss":
                result.update(self.clean_room(room))
            else:
                # **Every room fights.** `C_Rooms.ChoosePacks` walks every room
                # in the level, looks it up in the pack dictionary and logs
                # "No enemy pack is set for room " on a miss, so a room without
                # enemies is a bug in the game's own view. This used to fight
                # only in `fight`, `boss` and `start` rooms and to treat every
                # shop as peaceful, which left 132 of the 212 rows in
                # `EnemyPacks.json` unused: the ItemShop, FoodShop and Treasure
                # ones. See `notes/reference-sim.md`.
                result.update(self.fight(room))
        # `C_Room.AfterFight(win)` is what opens the doors, opens the shrine,
        # opens the shop and pays `goldReward`, so none of that happens on a
        # room that was not won.
        if first_visit and result.get("won", True):
            room.cleared = True
        # A room pays `GoldPerRoom` once. `C_Room.AfterFight` reads
        # `M_Room.goldReward`, skips the whole block when it is already 0, and
        # sets it to 0 after paying -- so walking back through a cleared room
        # pays nothing. This was paid on every move, which is an unbounded gold
        # fountain: a trained agent walked two rooms back and forth for 400
        # steps and finished holding 1,115 gold.
        if first_visit and room.cleared:
            # `GoldPerRoom` is a level total too, `(num - 1)` of them shared
            # out by the same weights, so a shop pays more than a corridor.
            self.gold += self.room_gold(room.kind)
        if first_visit and room.cleared and room.kind == "boss":
            self.win_level()
        # `V_Rooms.MoveCamera` launches the room's quest on every arrival, and
        # the quest controllers watch the extra rooms' fights.
        if self.quest is not None and not self.finished:
            stem = self.quest.arrive(self, rid, first_visit, result.get("won", True))
            if stem is not None:
                self.open_quest_dialog(stem)

    def open_quest_dialog(self, stem: str) -> None:
        """Start a quest dialog and walk it to its first real choice."""
        from .quests import TreeRun, trees
        if stem not in trees():
            return
        self.quest_dialog = TreeRun(self.quest, stem)
        self.quest_dialog.advance(self)
        self._settle_quest_dialog()

    def _settle_quest_dialog(self) -> None:
        """Take a choice that is the only one shown, and drop a dialog that has
        ended, the way `_settle_dialog` does for the level dialogs."""
        run = self.quest_dialog
        while run is not None and not run.closed:
            options = run.options(self)
            if len(options) != 1:
                break
            run.choose(self, options[0])
        if run is not None and (run.closed or not run.options(self)):
            self.quest_dialog = None
            if self.quest is not None and not self.finished:
                self.quest.dialog_ended(self)
            if self.quest_dialog is None and self.quest_queue and not self.finished:
                self.open_quest_dialog(self.quest_queue.pop(0))

    def quest_flight(self, rid: str, hp_loss_percent: float) -> None:
        """The Flight quest's cannon: into `rid` without a door, so without
        `C_Food.Move`, and `hpLossPercent` of each unit's health gone on
        landing (`WaitForArrival`: `health * (100 - loss) / 100`)."""
        self._arrive(rid, {})
        keep = max(0.0, 1.0 - hp_loss_percent / 100.0)
        for h in self.squad:
            h.hp = max(0.01, h.hp * keep)

    def quest_teleport(self, rid: str) -> None:
        """A teleport a quest makes (the SecretRoom quest's sacrifice and its
        way back): into `rid` as `C_Rooms.TeleportTo` does, with no food."""
        self._arrive(rid, {})

    def queue_quest_dialog(self, stem: str) -> None:
        """Open `stem` once the quest dialog on screen has ended."""
        self.quest_queue.append(stem)

    def win_level(self) -> None:
        """`C_Levels.WinLevel`, which `C_Fight.AfterFight` calls on a boss win.

        It ends the run only on the last level. Anywhere else it marks the level
        won and moves nobody: the next level is the pipe's business, see
        `descend`. The sim used to start the next level the moment the boss
        fell, which skipped the boss room's consumable shop and any walk back
        to a shop with the boss's gold.
        """
        # On level 4 in Default it also sets the persistent flag behind the
        # next run's `longRunReward`, see `long_run_reward`.
        if self.level == 4:
            self.long_run_reward_earned = True
        if self.level >= len(self.tables["Levels"]):
            self.finished, self.won = True, True
            return
        self.level_won = True

    def stranded(self) -> bool:
        """Whether nothing the run can still do leads anywhere.

        No legal action at all, or only teleports, none of which lands in a room
        nobody has entered or in a food shop holding a pack the purse covers. A
        teleport is free, so without the second clause a starving squad on a
        portal could hop between cleared rooms for ever, and an agent would
        learn to do that rather than take the loss.
        """
        legal = self.legal_actions()
        if any(a[0] != "teleport" for a in legal):
            return False
        _, costs = self.food_packs
        for _, rid in legal:
            room = self.rooms.rooms[rid]
            if not room.cleared:
                return False
            if room.kind == "food_shop":
                self.ensure_stock(room)
                if any(held > 0 and self.gold >= costs[i]
                       for i, held in enumerate(room.food_stock or [])):
                    return False
        return True

    def ensure_stock(self, room: Room) -> None:
        """Per-room stock, plus the shop's one free refill for this room."""
        if room.has("item_shop") and not room.refilled:
            room.refilled = True
            self.refill_offer()
        if room.consumable_offer is None and self.consumable_shelf(room):
            room.consumable_offer = self.draw_consumables(self.consumable_shelf(room))
        if room.kind == "mutation_shop":
            if room.offer_mutations is None:
                room.takes_left = int(MUTATION_SHOP["BuyCount"])
                room.rolls_left = int(MUTATION_SHOP["RollCount"])
                room.offer_mutations = self.draw_mutations(room)
            return
        if room.kind == "food_shop":
            if room.food_stock is None:
                sizes, costs = self.food_packs
                room.food_stock = determine_quantities(sizes, costs,
                                                       self.food_shop_budget)
            return
        if room.stock is not None:
            return
        if room.kind == "mutation":
            room.stock = 1
        else:
            room.stock = 0

    def mutation_pool(self, besides: Room | None = None) -> list[dict]:
        """What a shelf or a shrine can draw from on this level.

        `C_Mutations.OnLevelChange` rebuilds the available mutations for the
        level and then reserves every one the run already holds and every one
        already on a shelf in the level's shops, and `ReserveMutation` takes a
        reserved mutation out of the pool. So nothing the run holds is offered
        again, and two shelves on one level never show the same mutation.
        `besides` is the shelf being drawn for, which does not count against
        itself: `Roll` unreserves the old shelf before drawing the new one.
        """
        from .mutations import offered_at_level
        taken = {m.get("ID") for m in self.mutations}
        for room in self.rooms.rooms.values():
            if room is besides or not room.offer_mutations:
                continue
            taken.update(m.get("ID") for m in room.offer_mutations if m is not None)
        return [m for m in offered_at_level(self.tables, self.level)
                if m.get("ID") not in taken]

    def draw_mutations(self, room: Room) -> list:
        """`C_MutationShop.CreateMutations`: a fresh shelf of `ShowCount`."""
        pool = self.mutation_pool(besides=room)
        show = int(MUTATION_SHOP["ShowCount"])
        return self.rng.sample(pool, min(show, len(pool))) if pool else []

    # -- consumables --------------------------------------------------------
    @property
    def consumable_sell_price(self) -> float:
        """`M_Consumables.sellCost`, from `Rooms.Shops.Consumables.SellCost`."""
        shops = self.tables["Rooms"].get("Shops") or {}
        return float((shops.get("Consumables") or {}).get("SellCost") or 0)

    def consumable_pool(self) -> list[dict]:
        """What a consumable shop can draw from on this level.

        `C_ConsumableShop.Roll` weighs the level's `ConsumablesByLevel` rows,
        so a consumable listed twice is twice as likely, and leaves out what
        the run already carries. The draw itself is a `WeightedList` with
        history, whose anti-repeat weighting is not modelled.
        """
        rows = {c["ID"]: c for c in self.tables.get("Consumables") or []}
        held = {c.row.get("ID") for c in self.consumables}
        return [rows[r["Consumable"]] for r in self.tables.get("ConsumablesByLevel") or []
                if r.get("Level") == self.level and r.get("Consumable") in rows
                and r["Consumable"] not in held]

    def draw_consumables(self, n: int) -> list:
        """A shelf of `n`, no consumable twice."""
        pool, out = self.consumable_pool(), []
        while pool and len(out) < n:
            pick = self.rng.choice(pool)
            out.append(pick)
            pool = [p for p in pool if p["ID"] != pick["ID"]]
        return out

    def can_use(self, c: Consumable, here: Room) -> bool:
        """Whether using `c` here does anything.

        Mostly the controllers' own conditions. Two are this port's: a heal is
        offered only to a squad with a wound to heal, and a cleaner or an
        AntiHunger only while one is not already waiting.
        """
        fam = consumable_family(c.row)
        if fam in ("heal_one", "heal_all"):
            return any(h.hp < 1.0 for h in self.squad)
        if fam == "cleaner":
            return not self.armed.get("cleaner")
        if fam == "anti_hunger":
            return not self.armed.get("anti_hunger")
        if fam in ("reserve_humans", "reserve_ally"):
            return True
        if fam == "resurrection":
            # `C_ResurrectionConsumable.Use` revives only in a room whose fight
            # has a result, and only the corpses lying in it.
            return here.cleared and bool(here.corpses)
        if fam == "teleport":
            return len(self.rooms.rooms) > 1
        if fam == "event":
            if not self.dialogs_on:
                return False
            from .dialogs import pool
            remaining = self.dialog_pool if self.dialog_pool is not None else pool(self.tables)
            engine = self._dialog_engine()
            return any(engine.can_start(n) for n in remaining)
        if fam == "reroll":
            # `RoomType.ShopWithChoices = 16652`: an item, food, mutation or
            # consumable shop, open.
            return here.cleared and (
                here.has("item_shop") or here.kind == "food_shop"
                or (here.kind == "mutation_shop" and here.takes_left > 0)
                or (self.consumable_shelf(here) > 0
                    and here.consumable_offer is not None))
        return False

    def use_consumable(self, k: int) -> dict:
        """`C_<Name>Consumable.Use`, then `C_Consumable.Use`'s charge."""
        from .unit_skills import param
        c = self.consumables[k]
        fam = consumable_family(c.row)
        p = consumable_params(c.row)
        here = self.rooms.rooms[self.room]
        out = {"used": c.name, "family": fam}
        if fam in ("heal_one", "heal_all"):
            # A `Direct` heal is dragged onto one unit; the run layer picks the
            # most wounded, the way the pit does. The others heal the team.
            amount = param(p, "amount")
            percent = str(p.get("percentage", "false")).lower() == "true"
            targets = range(len(self.squad))
            if fam == "heal_one":
                targets = [min(targets, key=lambda i: (self.squad[i].hp, i))]
            specs = None if percent else self.specs_cached()
            for i in targets:
                gain = (amount / 100.0 if percent
                        else amount / max(1.0, specs[i].health))
                self.squad[i].hp = min(1.0, self.squad[i].hp + gain)
        elif fam == "cleaner":
            self.armed["cleaner"] = True
        elif fam == "anti_hunger":
            self.armed["anti_hunger"] = True
        elif fam == "reserve_humans":
            n = int(param(p, "amount", default=1))
            level = int(param(p, "level", default=1))
            self.squad += [Human(item=None, level=level) for _ in range(n)]
            out["joined"] = n
        elif fam == "reserve_ally":
            n = int(param(p, "amount", default=1))
            level = int(param(p, "level", default=1))
            self.armed.setdefault("allies", []).extend(
                [(str(p.get("class")), level)] * n)
        elif fam == "resurrection":
            back = list(here.corpses)
            here.corpses.clear()
            for h in back:
                h.hp = 1.0
            self.squad += back
            out["revived"] = len(back)
        elif fam == "teleport":
            # `C_TeleportConsumable.Use` collects the rooms still `UNKNOWN`,
            # takes `RND.Element` of them, reveals it and `TeleportTo`s there.
            hidden = sorted(rid for rid, r in self.rooms.rooms.items()
                            if r.state == UNKNOWN and rid != self.room)
            pool = hidden or sorted(rid for rid in self.rooms.rooms if rid != self.room)
            dest = self.rng.choice(pool)
            self.rooms.rooms[dest].state = UNEXPLORED
            self._arrive(dest, out)
        elif fam == "reroll":
            self.reroll_shop(here)
        elif fam == "event":
            out["dialog"] = self.start_level_dialog()
        c.charges -= 1
        if c.charges <= 0 and c in self.consumables:
            self.consumables.remove(c)
        return out

    def reroll_shop(self, here: Room) -> None:
        """A free `Roll` of the shop the squad stands in, every slot."""
        if here.has("item_shop"):
            self.offer = [self.roll_item() for _ in range(self.shop_quantity())]
        elif here.kind == "food_shop":
            sizes, costs = self.food_packs
            here.food_stock = determine_quantities(sizes, costs, self.food_shop_budget)
        elif here.kind == "mutation_shop":
            here.offer_mutations = self.draw_mutations(here)
        elif self.consumable_shelf(here):
            here.consumable_offer = self.draw_consumables(self.consumable_shelf(here))

    def clean_room(self, room: Room) -> dict:
        """A RoomCleaner on a room that has not fought: `C_Room.AfterFight(true)`
        with no fight, and the room's experience to the squad.

        `CalculateExperience` reads the room's enemies; the pack is drawn the
        way a fight would draw it and nobody fights it.
        """
        enemies = self.enemy_specs(room.kind)
        exp = sum(s.exp_reward for s in enemies)
        levels = self.gain_experience(exp)
        return {"won": True, "cleaned": True, "exp": exp, "levels": levels}

    # -- fights ------------------------------------------------------------
    def fight(self, room: Room, enemies: list | None = None) -> dict:
        # An AntiHunger used before this fight holds the hunger penalties off
        # for it: `M_AntiHungerStatus` wraps each unit's hunger status until
        # the fight ends. Temporary allies from a Reserve fight once and go.
        no_hunger = bool(self.armed.pop("anti_hunger", False))
        allies = self.armed.pop("allies", [])
        specs = self.specs(no_hunger=no_hunger)
        quest_layout = None
        if enemies is not None:
            # A fight a quest starts (`quest_fight`): its units, everyone in e1,
            # in whatever layout the fallback below draws.
            self._last_pack = (None, ["e1"] * len(enemies))
        elif room.kind in ("quest", "quest_extra"):
            enemies, quest_layout = self.quest_enemies(room)
        elif room.kind == "secret":
            # `CalculatePower` gives it no Power, so `GenerateRoom` gives it
            # its layout (`regular-secret-room`, or the quest's) and no pack.
            enemies = []
        else:
            enemies = self.enemy_specs(room.kind)
            # `OnEnemyTeamCreation`: the Sword quest's twin joins the boss.
            extra = self.quest.extra_enemies(self, room) if self.quest is not None else []
            if extra and enemies:
                more = self.ally_specs(extra)
                pack_id, zones = self._last_pack or (None, [])
                enemies = enemies + more
                self._last_pack = (pack_id, list(zones) + ["e1"] * len(more))
        if not specs:
            self.finished, self.won = True, False
            return {"won": False, "reason": "no squad"}
        if not enemies:
            return {"won": True, "reason": "empty room"}

        layouts = parse_room_layouts(self.tables["RoomLayouts"])
        # The pack chooses the room: `M_EnemyPack.ChooseLayout` takes one of the
        # pack's own `LayoutsByPack` layouts, and every shipped pack lists
        # exactly one. This used to be any of the 77 layouts at random, 35 of
        # which no pack uses.
        pack_id, zones = self._last_pack or (None, [])
        by_id = {lay.id: lay for lay in layouts}
        own = self.pack_layout(pack_id) if self.pack_layouts else None
        layout = by_id.get(own) if own is not None else None
        if quest_layout is not None:
            layout = quest_layout
        if layout is None:
            layout = layouts[self.rng.randrange(len(layouts))]
            zones = []
        self._last_layout = layout.id
        grid = Grid.from_layout(layout)
        rng = random.Random(self.rng.randrange(1 << 30))
        # Enemies deploy first so a placement policy can see what it is facing,
        # which is what the game shows the player before a fight starts. Each
        # pack row names its zone (`EnemyPacks.Zone`, e1 to e3), and
        # `M_EnemyPack.Create` places the row's units there.
        if zones and len(zones) == len(enemies):
            team1 = []
            for zone in dict.fromkeys(zones):
                group = [s for s, z in zip(enemies, zones) if z == zone]
                team1 += deploy(grid, layout, group, 1, zone, rng)
        else:
            team1 = deploy(grid, layout, enemies, 1, "e1", rng)
        # The player zone is the roster: one unit per cell, and a squad bigger
        # than the zone leaves the rest out of the fight. `C_Player.FindCell`
        # walks the layout for an empty cell through `C_UnitLayout.IsCellEmpty`,
        # so there is nowhere to put unit 50 on a 49-cell layout. Without this,
        # bodies stack on one cell and buying humans is unboundedly good.
        fielded, _ = self.split_by_zone(layout, specs)
        # Summoned allies take the cells after the humans, while there are any.
        n_humans = len(fielded)
        room_left = max(0, (len(layout.zone("p")) or 1) - n_humans)
        fielded = fielded + self.ally_specs(allies)[:room_left]
        if self.placement_policy is not None:
            cells = self.placement_policy(grid, layout, fielded, rng, team1)
            team0 = place_at(grid, fielded, cells, team=0)
        else:
            team0 = deploy(grid, layout, fielded, 0, "p", rng)
        apply_class_skills(self.tables, team0)
        if self.mutations:
            from .mutations import apply_to_agents
            apply_to_agents(team0, self.mutations, rng)
        # Wounds carry over: each unit starts the fight with the share of its
        # health it ended the last one with. `team0` starts with the squad's
        # first `n_humans` humans, in order; allies arrive whole.
        if self.carry_hp:
            for agent, human in zip(team0[:n_humans], self.squad):
                agent.hp = agent.max_hp * human.hp
        seed = rng.randrange(1 << 30)
        b = Battle(grid, team0 + team1, seed=seed, tables=self.tables,
                   build_trees=not self.use_fast_core)

        if self.use_fast_core:
            from .fast import UnsupportedFight, available, fast_battle
            if available():
                try:
                    fr = fast_battle(grid, b.agents, seed=seed, tables=self.tables)
                    for agent, hp in zip(b.agents, fr["hp"]):
                        agent.hp = float(hp)
                    res = BattleResult(
                        winner=fr["winner"], ticks=fr["ticks"], seconds=fr["seconds"],
                        survivors=fr["survivors"], total_damage=fr["damage"],
                        healing={0: 0.0, 1: 0.0}, casts={},
                    )
                    return self._after_fight(team0[:n_humans], team1, res, room)
                except UnsupportedFight:
                    # outside the envelope: rebuild with trees and use the oracle
                    b = Battle(grid, team0 + team1, seed=seed, tables=self.tables)
        res = b.run()

        return self._after_fight(team0[:n_humans], team1, res, room)

    def quest_fight(self, units: list) -> dict:
        """A fight a quest starts in the room the squad stands in, against
        `units` as (class, level) pairs: the Reaper's hater, the runaway Bot.
        `OnEnemyTeamCreation` adds the unit to the enemy team of the fight that
        `PrepareToFight` starts there."""
        return self.fight(self.rooms.rooms[self.room], enemies=self.ally_specs(units))

    def quest_enemies(self, room: Room) -> tuple[list, Any]:
        """A quest room's own pack and layout, from the quest's `RoomParams`.

        `C_GameMode.GenerateRoom` gives the room `RoomParams[name].Layout` from
        `SpecialRoomLayouts.csv` and, when the block has `Packs`, creates
        exactly that pack (`EnemyPacks.Create`); with no `Packs` the room has
        nothing to fight. Every shipped quest pack has `Min == Max`.
        """
        if self.quest is None:
            return [], None
        key = "room" if room.kind == "quest" else "extra"
        params = (self.quest.spec.get("RoomParams") or {}).get(key) or {}
        packs = params.get("Packs") or []
        if not packs:
            return [], None
        ubc = units_by_class(self.tables)
        meta = self.tables["Meta"]["Classes"]
        out: list = []
        zones: list = []
        for row in packs:
            cls = row.get("Class")
            if cls not in ubc:
                continue
            size = int((meta.get(cls) or {}).get("Size") or 1)
            unit = build_unit(ubc, cls, int(row.get("Level") or 1), name=cls, size=size)
            lo, hi = int(row.get("Min") or 1), int(row.get("Max") or 1)
            n = lo if hi <= lo else self.rng.randint(lo, hi)
            out += [unit] * n
            zones += [row.get("Zone") or "e1"] * n
        self._last_pack = (packs[0].get("ID"), zones)
        layouts = parse_room_layouts(self.tables.get("SpecialRoomLayouts") or "")
        layout = next((lay for lay in layouts if lay.id == params.get("Layout")), None)
        return out, layout

    def pack_layout(self, pack_id) -> int | None:
        """The layout `LayoutsByPack` gives a pack, or None."""
        for row in self.tables.get("LayoutsByPack") or []:
            if row.get("Pack") == pack_id:
                return row.get("Layout")
        return None

    def ally_specs(self, allies: list) -> list:
        """The temporary units a Reserve summoned, as specs, `(class, level)`."""
        if not allies:
            return []
        ubc = units_by_class(self.tables)
        meta = self.tables["Meta"]["Classes"]
        out = []
        for cls, level in allies:
            if cls in ubc:
                size = int((meta.get(cls) or {}).get("Size") or 1)
                out.append(build_unit(ubc, cls, int(level), name=cls, size=size))
        return out

    def split_by_zone(self, layout, specs: list) -> tuple[list, list]:
        """Split the squad into who fits on the layout and who sits it out."""
        room = len(layout.zone("p")) or 1
        return specs[:room], specs[room:]

    def _after_fight(self, team0, team1, res, room: Room | None = None):
        # survivors carry over; the dead are lost. Anyone who could not be
        # fielded is untouched -- they were not in the room. `team0` is the
        # humans only: summoned allies were removed before this was called.
        survivors = {id(a) for a in team0 if a.alive}
        kept = [h for h, a in zip(self.squad, team0) if id(a) in survivors]
        reserve = list(self.squad[len(team0):])
        # The dead stay in the room as corpses, for a Resurrection.
        if room is not None:
            room.corpses += [h for h, a in zip(self.squad, team0) if id(a) not in survivors]
        # What each survivor walks out with. Measured against the unit's own
        # health rather than `max_hp`, which carries whatever buffs were still
        # running when the fight ended; they end with it, and the game clamps
        # health to the new maximum.
        if self.carry_hp:
            for h, a in zip(self.squad, team0):
                if id(a) in survivors:
                    h.hp = min(1.0, max(0.0, a.hp / max(1.0, a.spec.health)))

        # `C_Unit.Die` gives a dead enemy's `expReward` to the other team, and
        # `C_Team.GetExperience` splits it evenly over the units in the room.
        # This used to be `total_damage / squad size`, which is not a number the
        # game has anywhere. Reserves get nothing: they were not in the room.
        #
        # The game levels a unit up the moment the kill lands, so a long fight
        # can be won by a squad that grew during it. Here the whole fight
        # resolves first and the levels land after, which is the one place the
        # timing differs.
        exp = sum(a.spec.exp_reward for a in team1 if not a.alive)
        if self.quest is not None:
            self.quest.killed(self, sum(1 for a in team1 if not a.alive))
        # A quest companion is not levelled: its class has rows only at the
        # levels the quest creates it at (see `sim/assumptions.py`).
        levels = self.gain_experience(exp, targets=[h for h in kept if not h.cls])

        self.squad = kept + reserve
        won = res.winner == 0
        if not won and not self.squad:
            self.finished, self.won = True, False
        return {"won": won, "seconds": res.seconds, "survivors": len(self.squad),
                "enemies": len(team1), "exp": exp, "levels": levels}

    def next_level(self) -> None:
        if self.level >= len(self.tables["Levels"]):
            self.finished, self.won = True, True
            return
        self.level += 1
        self.level_won = False
        self.generate_level()
        self.room = self.rooms.start
        self.rooms.set_current(self.room)
        # `C_BossVisionMutation.OnNewLevel` fires here, on the new level, and
        # puts the boss on the map without walking to it. It was registered as
        # "UI only", which was true only while this sim handed the whole map
        # over for free.
        if any(m.get("Name") == "BossVision" for m in self.mutations):
            self.rooms.reveal_boss()
        # `FightInFirst`. `C_Rooms.Init` looks the key up in the level data and,
        # when it is missing or false, calls `C_Room.AfterFight(win: true)` on
        # the room the level opens in, which resolves it as won without a fight
        # and so opens its doors and anything in it. Only the shipped fixed map
        # sets the key; no generated level row carries it, so a generated
        # level's first room is free. The start room has no pack and no gold
        # share either (`room_mult` is 0 for it), so this only marks it clear.
        self.rooms.rooms[self.room].cleared = True
        # No free larder on arrival. This used to add the level row's
        # `TotalFood`, on the reading that the column was the food a level hands
        # out; it is not. `C_FoodShop..ctor` is the only thing that reads
        # `TotalFood`, as the gold budget its shelf is stocked to, and the only
        # callers of `C_Food.AddFood` are the food shop and two dialog events --
        # `C_Levels.WinLevel` and `ChangeLevel` touch food nowhere. So the food
        # for a level is bought from its shops, and this was handing over up to
        # 250 free food a level, which is 40 rooms for a six-human squad.
        #
        # A DejaVu carried saves again as the level starts, then `ChangeLevel`
        # ends in `C_Levels.StartDialog`.
        if self.holds_dejavu():
            self.dejavu_save()
        self.start_level_dialog()
