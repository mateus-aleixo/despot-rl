"""One level map per level, generated the way `LevelGenerator` generates one.

The sim used to reuse the single `Rooms.json` map for all twelve levels -- 22
rooms with six food shops where level 1 asks for 7 rooms and one -- and then, for
one revision, grew a compact blob of the right size with the typed rooms spread
over it. This is the real thing: the growth rule, the constraints that bound the
shape, and the placement predicates, read out of `GameAssembly.dll`.

## The parameters

`C_Rooms.Generate` fills a `GenerationParams` from the level row and hands it to
`LevelGenerator`:

    num                 the room count, from `M_Levels` (see `room_count`)
    maxSquares          num / 10 + 1
    maxSquaresAtOnce    1 + (num >= 15)
    foodShopCount       the row's `FoodShops`, and likewise `ItemShops`,
                        `Shrines`, `RerollShrines`, `StatShops`, `SecretRoom`
    minFinishDistance   `IC_GameMode.GetRoomsMinFinishDistance(num)`, which
                        `C_DefaultMode` implements as num / 6 + 2
    maxDeadEnds         never written in the Default path, so 0
    portalsCount        num / 7 + 1
    minPortalDistance   3, a literal

The last three were read as unrecoverable for a while. `C_Rooms.Generate`
keeps the struct at `rsp + 0x30`, so `portalsCount` (offset 0x24) is the store
to `rsp + 0x54` and `minPortalDistance` (0x40) the store of 3 to `rsp + 0x70`;
both come after `OnBeforeGeneration` has run, so the `RoomCount` mutation's
extra rooms feed into them. The divisions are the compiler's: `0x92492493`
with an add is a signed divide by 7, `0x2aaaaaab` a divide by 6.

## The growth

`Generate` seeds one room, then repeats: `WeighPositions`, draw an open position
from the weighted list, `MaybeMakeRoom`. `<WeighPositions>b__60_0` is

    weight = snCount^5 + nCount^10 + 2^(maxDistance - distance) * (maxDistance - distance) + 2

where `nCount` counts adjacent **rooms** and `snCount` adjacent **nodes** (a node
is any position the generator has touched, room or not; `RoomNode.CalculateNeighbors`
counts them with `HasRoom` and `HasNode`). The tenth power is not a typo: a
position with three room-neighbours outweighs one with two by about fifty to one,
so growth fills its own concavities before it ever grows a limb.

`MaybeMakeRoom` sets the node to `Yes`, recounts, and reverts it to `No` --
permanently, that position is never offered again -- unless both hold:

  * `CalculateSquares() <= maxSquares`, the number of 2x2 blocks of rooms on the
    whole map, and
  * every room's own `squares <= maxSquaresAtOnce`.

Those two are what stop the weight function from producing a rectangle. A level
is a thin, winding shape because 2x2 blocks are rationed to `num / 10 + 1`.

`Adjust` then removes dead ends outright -- `deadends[0].state = Never` -- while
there are more than `maxDeadEnds` of them, and `CloseLoops` runs afterwards with
an iteration cap of 10. `CheckNeighborCount` separately keeps any room from
having four room-neighbours.

`maxDeadEnds` is the one parameter this port could not recover. `C_Rooms.Generate`
never writes that field, which reads as a zero, and zero is impossible: a level
with **no** dead ends must be a union of cycles, and an exhaustive search says no
7-room shape satisfies that together with the neighbour ceiling and
`maxSquares = 1` -- while level 1 is exactly 7 rooms (`MinRooms == MaxRooms == 7`).
So something outside `C_Rooms.Generate` sets it. Until that is found, `Adjust`
here prunes dead ends only while the map is over its room count, which keeps the
count exact and lets the growth rule decide how many dead ends are left. See
`sim/assumptions.py`.

## The placement

`PositionRooms` runs five steps in order, and every one of them draws
**uniformly** from the rooms that qualify (`RND.Element`, inlined as
`list[rnd.Next(list.Count)]`). The port used to maximise instead: the boss by
average distance and the start as the room furthest from it, which made every
level as long as its shape allowed.

  * `ChooseFinishRoom`: untyped, not an articulation point, a dead end or
    exactly two neighbours, and `maxDistance >= minFinishDistance`.
  * `ChooseStartRoom`: untyped, not an articulation point, at least `d` from
    the finish, with `d` counting down from `minFinishDistance` until something
    qualifies. On every level after the first the start room is also an item
    shop: the type is or-ed with 5, `Start | ItemShop`.
  * `ChooseDiversePositions`, once per shop type: the type with more rooms goes
    first, item shops on a tie. See `choose_diverse_positions`.
  * `ChoosePortals`, when `portalsCount > 1`. See `choose_portals`.
  * `ChooseTreasureRooms`: the rooms typed exactly `Default` or `Portal`, drawn
    without replacement for talent shops, then mutation shops, then shrines.

A step that finds nothing fails the whole generation, and `C_Rooms.Generate`
builds a new `LevelGenerator` with the same parameters and tries again, for as
long as it takes. `generate` does the same.

## Quests

On a level with a quest, `PositionRooms` places it between the start and the
shops; see `choose_quest_room`, `choose_quest_extra_room` and `quest_reachable`.
A quest that does not fit fails the generation like any other step.

## What is still not ported

`MaybeMakeSecretRoom`, `MaybeFlip` (a mirror, which changes nothing about the
graph) and multi-square rooms: a room here covers exactly one square, so
`maxSquares` and `maxSquaresAtOnce` are read as constraints on the layout rather
than as a budget for large rooms. `StatShops` is counted and left as a fight
room, because the sim does not model the talent shop. Two more room bits are
read and not modelled: `ChooseFinishRoom` makes the boss `Boss | ConsumableShop`
on every level but the last (`or edx, 0x4002`), and the final boss
`Boss | FinalBoss`, plus an item shop unless the chip is `hard`.
"""
from __future__ import annotations

import collections
import random
import string

# The two shops `ChooseDiversePositions` spreads out, by level-row column.
TYPED_ROOMS = (("ItemShops", "item_shop"),
               ("FoodShops", "food_shop"))

# The three `ChooseTreasureRooms` places instead, in its order, with the
# `RoomType` each one is or-ed with. They all carry the Treasure bit (64), which
# is what the method groups and what `TreasureMult` weights:
#   Shrine       192  = 128 | 64   a free mutation
#   MutationShop 320  = 256 | 64   the mutation shop, `C_MutationShop`
#   TalentShop  4160 = 4096 | 64   which `C_Room.InitShop` gives a
#                                  `C_ConsumableShop` -- and `StatShops` is 0 on
#                                  every Default level, so no run ever sees one
TREASURE_ROOMS = (("StatShops", "talent_shop"),
                  ("RerollShrines", "mutation_shop"),
                  ("Shrines", "mutation"))

# `LevelGenerator.roomTypesWithNoPortal = 34307`, which is
# `Start | Boss | Quest | QuestExtra | Secret`. Shops are not in it, so a shop can
# carry a portal. Secret rooms are not generated here.
NO_PORTAL = frozenset({"start", "boss", "quest", "quest_extra"})

# `Direction` as a step in (row, col). The game's rows count down from the top
# (`RoomNode.GetName` prints `topBorder - pos.y`), so `Up`, which is `y + 1` in
# `DirectionExtensions.Neighbor`, is a row less here.
DIRECTION_STEP = {"Right": (0, 1), "Left": (0, -1), "Up": (-1, 0), "Down": (1, 0)}

# `C_Rooms.Generate` writes the literal 3 into `minPortalDistance`.
MIN_PORTAL_DISTANCE = 3

# How many times `generate` rebuilds a level whose placement failed before it
# gives up. The game has no cap; this is only so a bad row raises instead of
# hanging.
MAX_ATTEMPTS = 200

# `LevelGenerator.MakeRoom`
MAYBE, NO, YES, NEVER = 0, 1, 2, 3

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def room_name(row: int, col: int) -> str:
    """`LevelGenerator.RoomNode.GetName`: a column letter and a row number."""
    letters = string.ascii_lowercase
    tag, c = "", col
    while True:
        tag = letters[c % 26] + tag
        c = c // 26 - 1
        if c < 0:
            break
    return f"{tag}{row + 1}"


class Generator:
    """`LevelGenerator`, over one-square rooms."""

    def __init__(self, num: int, rng: random.Random, min_finish_distance: int = 0,
                 max_dead_ends: int | None = None):
        self.rng = rng
        self.num = num
        self.max_squares = num // 10 + 1
        self.max_squares_at_once = 1 + (num >= 15)
        self.max_dead_ends = max_dead_ends
        self.min_finish_distance = min_finish_distance
        # pos -> state. A position is a "node" once it is in here at all.
        self.state: dict[tuple[int, int], int] = {(0, 0): YES}
        for d in DIRS:
            self.state[d] = MAYBE

    # -- the grid ----------------------------------------------------------
    def rooms(self) -> list[tuple[int, int]]:
        return [p for p, s in self.state.items() if s == YES]

    def n_count(self, pos) -> int:
        """`RoomNode.nCount`: adjacent rooms (`HasRoom`)."""
        return sum(self.state.get((pos[0] + d[0], pos[1] + d[1])) == YES for d in DIRS)

    def sn_count(self, pos) -> int:
        """`RoomNode.snCount`: adjacent nodes (`HasNode`), room or not."""
        return sum((pos[0] + d[0], pos[1] + d[1]) in self.state for d in DIRS)

    def squares_at(self, pos) -> int:
        """`RoomNode.get_squares`: 2x2 blocks of rooms this room belongs to."""
        r, c = pos
        out = 0
        for dr, dc in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
            corner = ((r + dr, c + dc), (r + dr + 1, c + dc),
                      (r + dr, c + dc + 1), (r + dr + 1, c + dc + 1))
            out += all(self.state.get(p) == YES for p in corner)
        return out

    def calculate_squares(self) -> int:
        """`LevelGenerator.CalculateSquares`: 2x2 blocks on the whole map."""
        rooms = set(self.rooms())
        return sum(1 for (r, c) in rooms
                   if {(r + 1, c), (r, c + 1), (r + 1, c + 1)} <= rooms)

    def check_neighbour_count(self) -> bool:
        """`CheckNeighborCount`: no room may have four room-neighbours."""
        return all(self.n_count(p) <= 3 for p in self.rooms())

    def adjacency(self) -> dict:
        rooms = set(self.rooms())
        return {p: [(p[0] + d[0], p[1] + d[1]) for d in DIRS
                    if (p[0] + d[0], p[1] + d[1]) in rooms]
                for p in rooms}

    def anchor(self):
        """Any room to measure from. `Adjust` can cut the seed room itself."""
        rooms = self.rooms()
        return (0, 0) if (0, 0) in rooms else (rooms[0] if rooms else (0, 0))

    def distances_from(self, source, adj=None) -> dict:
        adj = adj if adj is not None else self.adjacency()
        if source not in adj:
            return {}
        out = {source: 0}
        q = collections.deque([source])
        while q:
            cur = q.popleft()
            for n in adj[cur]:
                if n not in out:
                    out[n] = out[cur] + 1
                    q.append(n)
        return out

    # -- growth ------------------------------------------------------------
    def weigh(self, pos, depth: dict, max_distance: int) -> float:
        """`<WeighPositions>b__60_0`, exactly."""
        far = max_distance - depth.get(pos, max_distance)
        return (self.sn_count(pos) ** 5 + self.n_count(pos) ** 10
                + 2.0 ** far * far + 2.0)

    def maybe_make_room(self, pos) -> bool:
        """`MaybeMakeRoom`: commit the position, or refuse it for good."""
        self.state[pos] = YES
        ok = (self.calculate_squares() <= self.max_squares
              and all(self.squares_at(p) <= self.max_squares_at_once
                      for p in self.rooms())
              and self.check_neighbour_count())
        if not ok:
            self.state[pos] = NO
            return False
        for d in DIRS:
            n = (pos[0] + d[0], pos[1] + d[1])
            if n not in self.state:
                self.state[n] = MAYBE           # `AddOpenPosition`
        return True

    def grow(self, target: int, budget: int = 4000) -> None:
        """Draw weighted open positions until the map holds `target` rooms."""
        while len(self.rooms()) < target and budget > 0:
            budget -= 1
            open_positions = [p for p, s in self.state.items()
                              if s == MAYBE and self.n_count(p) > 0]
            if not open_positions:
                return
            # `distance` is measured over the rooms placed so far, which is what
            # the generator's own BFS maintains as it grows.
            depth = self.distances_from(self.anchor())
            max_distance = max(depth.values(), default=0)
            near = {}
            for p in open_positions:
                d = [depth[(p[0] + dd[0], p[1] + dd[1])] for dd in DIRS
                     if (p[0] + dd[0], p[1] + dd[1]) in depth]
                near[p] = min(d) + 1 if d else max_distance
            weights = [self.weigh(p, near, max_distance) for p in open_positions]
            self.maybe_make_room(self.rng.choices(open_positions, weights=weights)[0])

    def dead_ends(self) -> list:
        """`GetDeadends`: rooms with exactly one room-neighbour."""
        return [p for p in self.rooms() if self.n_count(p) == 1]

    def close_loops(self) -> int:
        """`CloseLoops`: give a dead end a second neighbour, up to ten passes.

        A dead end is closed by making a room out of a free position that
        touches both it and something else, so the limb becomes part of a
        cycle. This is the counterpart of `Adjust`: between them a level ends
        with no room that has fewer than two neighbours, which with
        `CheckNeighborCount`'s ceiling of three is why a level reads as a web of
        loops rather than as a tree of corridors. The game's own version is a
        larger routine and its iteration cap of ten is the one thing about it
        read directly.
        """
        added = 0
        for _ in range(10):
            ends = self.dead_ends()
            # Closing a loop adds a room, so it stops at the level's room count
            # rather than overshooting it; `Adjust` is what trims, and it only
            # cuts dead ends.
            if not ends or len(self.rooms()) >= self.num:
                break
            progress = False
            for end in ends:
                if len(self.rooms()) >= self.num:
                    break
                options = []
                for d in DIRS:
                    p = (end[0] + d[0], end[1] + d[1])
                    if self.state.get(p, MAYBE) in (MAYBE, NO) and self.n_count(p) >= 2:
                        options.append(p)
                if options and self.maybe_make_room(self.rng.choice(options)):
                    added += 1
                    progress = True
            if not progress:
                break
        return added

    def adjust(self) -> None:
        """`Adjust`: cut dead ends, `deadends[0].state = Never`.

        With `maxDeadEnds` unrecovered (see the module docstring) the budget is
        the room count instead: prune while the map is both over its count and
        holding a dead end. A cut room becomes `Never`, so the map cannot regrow
        the limb it just lost, which is what makes prune-and-regrow terminate.
        """
        while True:
            ends = self.dead_ends()
            budget = (len(self.rooms()) - self.num if self.max_dead_ends is None
                      else len(ends) - self.max_dead_ends)
            if not ends or budget <= 0 or len(self.rooms()) <= 3:
                return
            # The deepest first: a shallow dead end is likely to be the neck of
            # something worth keeping.
            depth = self.distances_from(self.anchor())
            self.state[max(ends, key=lambda p: (depth.get(p, 0), p))] = NEVER

    def build(self) -> None:
        """`Generate`'s growth phase: grow, `CloseLoops`, `Adjust`, repeat.

        The game runs the growth loop once and then `Adjust` and `CloseLoops`,
        which between them change the room count -- `Adjust` cuts and
        `CloseLoops` adds -- so the count and the dead-end rule cannot both be
        satisfied in one pass. Repeating the three until they agree is this
        port's reading; the alternative, that the game accepts whatever count
        falls out, cannot be told apart from the outside because
        `C_Levels.CalculateLevelsRoomsCount` is what the run actually reads.
        """
        for _ in range(40):
            self.grow(self.num)
            self.close_loops()
            self.adjust()
            if len(self.rooms()) == self.num:
                return
            if not [p for p, s in self.state.items()
                    if s == MAYBE and self.n_count(p) > 0]:
                return

    # -- the graph the placement rules read --------------------------------
    def articulation_points(self, adj) -> set:
        """`AP`/`APUtil`, which is Tarjan's algorithm."""
        disc, low, parent, out = {}, {}, {}, set()
        time = [0]
        for root in adj:
            if root in disc:
                continue
            stack = [(root, iter(adj[root]))]
            disc[root] = low[root] = time[0]
            time[0] += 1
            children = 0
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    stack.pop()
                    if stack:
                        up = stack[-1][0]
                        low[up] = min(low[up], low[node])
                        if up != root and low[node] >= disc[up]:
                            out.add(up)
                    continue
                if nxt not in disc:
                    parent[nxt] = node
                    if node == root:
                        children += 1
                    disc[nxt] = low[nxt] = time[0]
                    time[0] += 1
                    stack.append((nxt, iter(adj[nxt])))
                elif nxt != parent.get(node):
                    low[node] = min(low[node], disc[nxt])
            if children > 1:
                out.add(root)
        return out


def portals_count(num: int) -> int:
    """`portalsCount`, which `C_Rooms.Generate` sets to `num / 7 + 1`."""
    return num // 7 + 1


def min_finish_distance(num: int) -> int:
    """`C_DefaultMode.GetRoomsMinFinishDistance(roomsCount)`: `num / 6 + 2`.

    `C_Rooms.Generate` calls it through `IC_GameMode` slot 4 with the room count.
    Arcade overrides it as `roomsCount - 1`; Arcade is not modelled.
    """
    return num // 6 + 2


def shop_spread(kind: str, num: int) -> int:
    """The `distanceThreshold` `PositionRooms` hands `ChooseDiversePositions`.

    Item shops `num >= 18 ? num / 6 : 2`, food shops `num >= 16 ? num / 4 : 3`.
    """
    if kind == "item_shop":
        return num // 6 if num >= 18 else 2
    return num // 4 if num >= 16 else 3


def start_is_portal(level: int, chip: str = "default") -> bool:
    """Whether `ChoosePortals` makes the start room a portal before it picks any.

    It does in Default mode on levels 5 and 9, and on levels 3, 5 and 9 when the
    chip name starts with `shortpvp`. The start room is then the first entry of
    the chosen list, so it takes one of the `portalsCount` places.
    """
    if chip.startswith("shortpvp"):
        return level in (3, 5, 9)
    return level in (5, 9)


def choose_diverse_positions(cells: list, kinds: dict, dist: dict, count: int,
                             threshold: int, rng: random.Random) -> list:
    """`LevelGenerator.ChooseDiversePositions(type, count, distanceThreshold, null)`.

    Draws `count` untyped rooms one at a time, each uniformly from the rooms at
    least `threshold` from the ones **this call** has already drawn. Rooms of
    other types, the start and the boss included, do not count. A draw that
    finds nothing throws the partial set away and starts again; ten of those and
    the threshold comes down by one.

    The game loops forever when there are fewer untyped rooms than `count`, so
    `count` is capped at that here. No shipped row reaches the cap.
    """
    free = [p for p in cells if p not in kinds]
    count = min(count, len(free))
    chosen: list = []
    while len(chosen) < count:
        for _ in range(10):
            for _k in range(count):
                pool = [p for p in free if p not in chosen
                        and all(dist[p][c] >= threshold for c in chosen)]
                if not pool:
                    chosen = []
                    break
                chosen.append(rng.choice(pool))
            if len(chosen) == count:
                break
        threshold -= 1
    return chosen


def choose_portals(cells: list, kinds: dict, dist: dict, avg_distance: dict,
                   dead_end: dict, boss, start, count: int, threshold: int,
                   forced_start: bool) -> list:
    """`LevelGenerator.ChoosePortals(portalsCount, distanceThreshold)`.

    Read off the binary step by step:

      1. With `forced_start`, the start room becomes a portal and is the first
         room chosen.
      2. The candidates are the dead ends that may carry a portal (anything but
         the start, the boss and the quest rooms here) at least one room from
         the finish.
      3. Too few of them, and the room list is sorted by average distance,
         largest first; either way it is walked to top the candidates up to
         `2 * count` with rooms that may carry a portal.
      4. Each candidate scores its mean distance to the candidates, itself
         included, and they are sorted by that, largest first: the most
         spread-out first.
      5. Walking that order, a candidate at least `threshold` from every room
         chosen so far is chosen, until `count` are.
      6. Then **one more**: the most central room (smallest average distance)
         that may carry a portal, is not chosen and clears `threshold` from all
         of them, so a level asking for two portals can get three.
      7. Fewer than two portals in all and the whole step is undone.

    `List.Sort` is an introsort, so the order among equal keys is not
    recoverable; a stable sort stands in for it.
    """
    chosen = [start] if forced_start else []
    candidates = [p for p in cells if kinds.get(p) not in NO_PORTAL
                  and dist[p][boss] >= 1 and dead_end[p]]
    order = list(cells)
    if len(candidates) < count:
        order.sort(key=lambda p: -avg_distance[p])
    for p in order:
        if len(candidates) >= 2 * count:
            break
        if kinds.get(p) in NO_PORTAL or p in candidates or dist[p][boss] < 1:
            continue
        candidates.append(p)
    spread = {b: sum(dist[b][c] for c in candidates) / len(candidates)
              for b in candidates}
    candidates.sort(key=lambda b: -spread[b])
    for b in candidates:
        if all(dist[b][a] >= threshold for a in chosen):
            chosen.append(b)
        if len(chosen) == count:
            break
    portals = chosen[:count]
    for p in sorted(cells, key=lambda p: avg_distance[p]):
        if kinds.get(p) in NO_PORTAL or p in chosen or dist[p][boss] < 1:
            continue
        if all(dist[p][a] >= threshold for a in chosen):
            portals.append(p)
            break
    return portals if len(portals) >= 2 else []


def directions(p, adj: dict) -> set:
    """`RoomNode.neighborDirections`: which sides of `p` have a room."""
    out = set()
    for name, (dr, dc) in DIRECTION_STEP.items():
        if (p[0] + dr, p[1] + dc) in adj:
            out.add(name)
    return out


def choose_quest_room(cells: list, kinds: dict, adj: dict, quest: dict,
                      rng: random.Random):
    """`LevelGenerator.ChooseQuestRoom`, or None when nothing qualifies.

    The candidates are the untyped rooms whose doors all lie in the quest's
    `Doors` (`questRoomAllowedDoors`) and that have every one of its
    `GuaranteedDoors`: the quest prefab only has doors on those sides. One is
    drawn uniformly, except for `Flight`, whose list is sorted by `pos.x`
    (`<ChooseQuestRoom>b__5_0`) and whose first entry, the leftmost room, is
    taken.
    """
    allowed = set(quest.get("Doors") or [])
    guaranteed = set(quest.get("GuaranteedDoors") or [])
    pool = [p for p in cells if p not in kinds
            and directions(p, adj) <= allowed and guaranteed <= directions(p, adj)]
    if not pool:
        return None
    if quest.get("Name") == "Flight":
        return min(pool, key=lambda p: p[1])
    return rng.choice(pool)


def choose_quest_extra_room(cells: list, kinds: dict, dist: dict, arts: set,
                            n_count, quest_room, quest: dict, rng: random.Random):
    """`LevelGenerator.ChooseQuestExtraRoom(questRoom)`, or None.

    The candidates are the untyped rooms that are not articulation points, are
    a dead end or have exactly two neighbours, and lie more than one room from
    the quest room. One is drawn uniformly, except for `Flight`: its list is
    sorted by `pos.x`, largest first (`<ChooseQuestExtraRoom>b__6_0`), and the
    first entry is taken only when it lies left of the quest room and at most
    three rows from it. The type is or-ed with `QuestExtra`, 1024.
    """
    pool = [p for p in cells if p not in kinds and p not in arts
            and n_count(p) in (1, 2) and dist[p][quest_room] > 1]
    if not pool:
        return None
    if quest.get("Name") == "Flight":
        best = max(pool, key=lambda p: p[1])
        if best[1] < quest_room[1] and abs(best[0] - quest_room[0]) <= 3:
            return best
        return None
    return rng.choice(pool)


def quest_reachable(adj: dict, start, quest_room, blocked: set) -> bool:
    """`CountValidPaths(start, quest, QuestRoomCondition) > 0`.

    `<CountValidPaths>g__Count|23_0` enumerates every simple path from the
    start; one that reaches a neighbour of the quest room is counted when
    `QuestRoomCondition` passes it, which is when no room on it carries `Boss`
    or `QuestExtra` (`test type, 0x402`). Such a path exists exactly when the
    quest room can be reached from the start without stepping on those rooms,
    so a search answers it without the enumeration.
    """
    if start in blocked:
        return False
    seen, todo = {start}, [start]
    while todo:
        p = todo.pop()
        for n in adj[p]:
            if n == quest_room:
                return True
            if n in seen or n in blocked:
                continue
            seen.add(n)
            todo.append(n)
    return False


def room_count(level_row: dict, rng: random.Random) -> int:
    """`num`, drawn inside the row's `MinRooms`..`MaxRooms`.

    `C_Levels.CalculateLevelsRoomsCount` draws per level and then balances the
    draws across the whole run, which is a property of a run rather than of a
    level and is not reproduced here.
    """
    lo = int(level_row.get("MinRooms") or 7)
    hi = int(level_row.get("MaxRooms") or lo)
    return rng.randint(min(lo, hi), max(lo, hi))


def position_rooms(gen: Generator, level_row: dict, rng: random.Random,
                   chip: str = "default", quest: dict | None = None) -> dict | None:
    """`LevelGenerator.PositionRooms`, or None where the game's would fail.

    See the module docstring for the order. `quest` is the level's quest row
    with its `Name` (`sim/quests.py`), or None on a level without one.
    """
    cells = gen.rooms()
    adj = gen.adjacency()
    arts = gen.articulation_points(adj)
    dist = {p: gen.distances_from(p, adj) for p in cells}
    max_distance = {p: max(dist[p].values()) for p in cells}
    # `RoomNode.InitRoom`: the mean over `distances`, which holds the room
    # itself at 0, and `deadEnd = neighbors.Count == 1`.
    avg_distance = {p: sum(dist[p].values()) / len(dist[p]) for p in cells}
    dead_end = {p: gen.n_count(p) == 1 for p in cells}
    level = int(level_row.get("ID") or 1)
    num = gen.num
    kinds: dict[tuple[int, int], str] = {}
    extras: dict[tuple[int, int], tuple[str, ...]] = {}

    # `ChooseFinishRoom` over its inline predicate: untyped, not an
    # articulation point, a dead end or exactly two neighbours, and far enough
    # from somewhere. Uniform over what qualifies.
    finish_pool = [p for p in cells if p not in arts
                   and gen.n_count(p) in (1, 2)
                   and max_distance[p] >= gen.min_finish_distance]
    if not finish_pool:
        return None
    boss = rng.choice(finish_pool)
    kinds[boss] = "boss"

    # `ChooseStartRoom(finish)` over `CouldBeStartRoom(room, d)`: untyped, not
    # an articulation point, at least `d` from the finish, `d` counting down
    # from `minFinishDistance`.
    start = None
    for d in range(gen.min_finish_distance, 0, -1):
        pool = [p for p in cells if p not in kinds and p not in arts
                and dist[p][boss] >= d]
        if pool:
            start = rng.choice(pool)
            break
    if start is None:
        return None
    kinds[start] = "start"
    # `or eax, 5` on any level but the first: the run opens every later level
    # standing in an item shop.
    if level >= 2:
        extras[start] = ("item_shop",)

    # The quest, when the level has one: its room, its extra rooms, and a way
    # from the start to it that passes neither the boss nor an extra room.
    quest_room, quest_extras = None, []
    if quest:
        quest_room = choose_quest_room(cells, kinds, adj, quest, rng)
        if quest_room is None:
            return None
        kinds[quest_room] = "quest"
        for _ in range(int(quest.get("ExtraRooms") or 0)):
            extra = choose_quest_extra_room(cells, kinds, dist, arts, gen.n_count,
                                            quest_room, quest, rng)
            if extra is None:
                return None
            kinds[extra] = "quest_extra"
            quest_extras.append(extra)
        if not quest_reachable(adj, start, quest_room, {boss, *quest_extras}):
            return None

    # The shop type with more rooms is spread first; item shops on a tie.
    shops = [(kind, int(level_row.get(column) or 0)) for column, kind in TYPED_ROOMS]
    if shops[0][1] < shops[1][1]:
        shops.reverse()
    for kind, count in shops:
        for p in choose_diverse_positions(cells, kinds, dist, count,
                                          shop_spread(kind, num), rng):
            kinds[p] = kind

    # `ChoosePortals`, only when more than one is asked for. With a secret room
    # the game retries at shrinking distances and fails the level unless
    # `secretRoomCount + 2` portals land; secret rooms are not generated here.
    portals: list = []
    if portals_count(num) > 1:
        portals = choose_portals(cells, kinds, dist, avg_distance, dead_end,
                                 boss, start, portals_count(num),
                                 MIN_PORTAL_DISTANCE, start_is_portal(level, chip))

    # `ChooseTreasureRooms`: the rooms typed exactly `Default` or `Portal`,
    # which here is every room without a kind yet, drawn without replacement.
    for column, kind in TREASURE_ROOMS:
        for _ in range(int(level_row.get(column) or 0)):
            free = [p for p in cells if p not in kinds]
            if not free:
                break
            kinds[rng.choice(free)] = kind

    for p in cells:
        kinds.setdefault(p, "fight")

    lo_r = min(r for r, _ in cells)
    lo_c = min(c for _, c in cells)

    def name(p):
        return room_name(p[0] - lo_r, p[1] - lo_c)

    return {"rooms": {name(p): (p[0] - lo_r, p[1] - lo_c, kinds[p]) for p in cells},
            "start": name(start),
            "boss": name(boss),
            "portals": [name(p) for p in portals],
            "extras": {name(p): e for p, e in extras.items()},
            "quest_room": name(quest_room) if quest_room is not None else None,
            "quest_extras": [name(p) for p in quest_extras]}


def generate(level_row: dict, rng: random.Random, chip: str = "default",
             quest: dict | None = None) -> dict:
    """One level, as `position_rooms` returns it.

    The room count is drawn once and kept across attempts: `C_Rooms.Generate`
    reads it from the run's precomputed list and retries with the same
    parameters, so only the shape and the placement are drawn again.
    """
    num = room_count(level_row, rng)
    for _ in range(MAX_ATTEMPTS):
        gen = Generator(num, rng, min_finish_distance=min_finish_distance(num))
        gen.build()
        if len(gen.rooms()) < 3:
            continue
        level = position_rooms(gen, level_row, rng, chip, quest)
        if level is not None:
            return level
    raise RuntimeError(f"no level placed for row {level_row.get('ID')} "
                       f"after {MAX_ATTEMPTS} attempts")
