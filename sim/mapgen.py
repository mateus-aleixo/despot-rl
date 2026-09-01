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
    minFinishDistance   a game-mode call, not a column -- see the assumptions
    maxDeadEnds         never written in the Default path, so 0
    portalsCount        likewise 0, which is why no portals are placed

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

`PositionRooms` picks the finish first. `CouldBeFinishRoom` wants a room that is
untyped, **not an articulation point**, either a dead end or with exactly two
neighbours, and whose `maxDistance` clears `minFinishDistance`; `ChooseFinishRoom`
takes the best of those by average distance. `ChooseStartRoom` then wants an
untyped non-articulation room far from the finish, and `ChooseDiversePositions`
places each shop type at least a threshold apart, lowering the threshold when
nothing qualifies.

## What is still not ported

`MaybeMakeSecretRoom`, quests (`ChooseQuestRoom`, `ChooseQuestExtraRoom`,
`CountValidPaths`), treasure rooms (`ChooseTreasureRooms`), portals (`portalsCount`
is 0 in this path anyway), `MaybeFlip` (a mirror, which changes nothing about the
graph) and multi-square rooms: a room here covers exactly one square, so
`maxSquares` and `maxSquaresAtOnce` are read as constraints on the layout rather
than as a budget for large rooms. `RerollShrines` and `StatShops` are counted and
left as fight rooms, because the sim models neither shop.
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


def room_count(level_row: dict, rng: random.Random) -> int:
    """`num`, drawn inside the row's `MinRooms`..`MaxRooms`.

    `C_Levels.CalculateLevelsRoomsCount` draws per level and then balances the
    draws across the whole run, which is a property of a run rather than of a
    level and is not reproduced here.
    """
    lo = int(level_row.get("MinRooms") or 7)
    hi = int(level_row.get("MaxRooms") or lo)
    return rng.randint(min(lo, hi), max(lo, hi))


def generate(level_row: dict, rng: random.Random) -> dict:
    """One level: `{name: (row, col, kind)}` plus the start and boss names."""
    num = room_count(level_row, rng)
    for _ in range(20):
        gen = Generator(num, rng)
        gen.build()
        if len(gen.rooms()) >= 3:
            break

    cells = gen.rooms()
    adj = gen.adjacency()
    arts = gen.articulation_points(adj)
    dist = {p: gen.distances_from(p, adj) for p in cells}
    max_distance = {p: max(dist[p].values()) for p in cells}
    avg_distance = {p: sum(dist[p].values()) / max(1, len(cells) - 1) for p in cells}
    kinds: dict[tuple[int, int], str] = {}

    def untyped():
        return [p for p in cells if p not in kinds]

    # `ChooseFinishRoom` over `CouldBeFinishRoom`: untyped, not an articulation
    # point, a dead end or with exactly two neighbours, far enough away -- best
    # by average distance. The predicate is relaxed rather than failed if
    # nothing qualifies, where the game would fail the whole generation and let
    # `C_Rooms` try again.
    finish_pool = [p for p in untyped()
                   if p not in arts
                   and (gen.n_count(p) == 1 or gen.n_count(p) == 2)
                   and max_distance[p] >= gen.min_finish_distance]
    finish_pool = finish_pool or [p for p in untyped() if p not in arts] or untyped()
    boss = max(finish_pool, key=lambda p: (avg_distance[p], rng.random()))
    kinds[boss] = "boss"

    # `ChooseStartRoom(finish)`: untyped, not an articulation point, as far from
    # the finish as the threshold demands.
    start_pool = [p for p in untyped() if p not in arts] or untyped()
    start = max(start_pool, key=lambda p: (dist[boss].get(p, 0), rng.random()))
    kinds[start] = "start"

    # `ChooseDiversePositions(type, count, distanceThreshold, excludes)`: each
    # room of a type at least `distanceThreshold` from the ones already placed,
    # the threshold coming down until something qualifies.
    for column, kind in TYPED_ROOMS:
        for _ in range(int(level_row.get(column) or 0)):
            free = untyped()
            if not free:
                break
            placed = list(kinds)
            for threshold in range(max(max_distance.values(), default=1), 0, -1):
                pool = [p for p in free
                        if min((dist[p][q] for q in placed if q in dist[p]),
                               default=99) >= threshold]
                if pool:
                    kinds[rng.choice(pool)] = kind
                    break
            else:
                kinds[rng.choice(free)] = kind

    # `ChooseTreasureRooms` is not `ChooseDiversePositions`: it collects every
    # room still untyped (or a portal) and takes `RND.Element` of them, so the
    # shrines land uniformly at random rather than spread apart.
    for column, kind in TREASURE_ROOMS:
        for _ in range(int(level_row.get(column) or 0)):
            free = untyped()
            if not free:
                break
            kinds[rng.choice(free)] = kind

    for p in cells:
        kinds.setdefault(p, "fight")

    lo_r = min(r for r, _ in cells)
    lo_c = min(c for _, c in cells)
    rooms = {room_name(r - lo_r, c - lo_c): (r - lo_r, c - lo_c, kinds[(r, c)])
             for (r, c) in cells}
    return {"rooms": rooms,
            "start": room_name(start[0] - lo_r, start[1] - lo_c),
            "boss": room_name(boss[0] - lo_r, boss[1] - lo_c)}
