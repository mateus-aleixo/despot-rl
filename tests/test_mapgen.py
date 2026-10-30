"""Level generation (sim/mapgen.py): the shape rules every generated level obeys."""
import collections
import random

import pytest

from sim.mapgen import (DIRS, MIN_PORTAL_DISTANCE, Generator, choose_diverse_positions,
                        choose_portals, choose_quest_extra_room, choose_quest_room,
                        directions, generate, min_finish_distance, portals_count,
                        quest_reachable, room_count, room_name, shop_spread,
                        start_is_portal)

# Shaped like `M_Levels` rows: level 1 is exactly 7 rooms; later levels draw a range.
SMALL = {"ID": 1, "MinRooms": 7, "MaxRooms": 7, "ItemShops": 1, "FoodShops": 1,
         "Shrines": 1}
LARGE = {"ID": 5, "MinRooms": 15, "MaxRooms": 18, "ItemShops": 2, "FoodShops": 2,
         "Shrines": 1, "RerollShrines": 1}


def neighbours(cells: set, pos):
    return [(pos[0] + dr, pos[1] + dc) for dr, dc in DIRS
            if (pos[0] + dr, pos[1] + dc) in cells]


def test_room_names_count_columns_like_a_spreadsheet():
    assert room_name(0, 0) == "a1"
    assert room_name(4, 25) == "z5"
    assert room_name(0, 26) == "aa1"
    assert room_name(2, 27) == "ab3"


def test_room_count_stays_inside_the_rows_range():
    rng = random.Random(0)
    assert {room_count(LARGE, rng) for _ in range(400)} == {15, 16, 17, 18}
    assert room_count({"MinRooms": 9, "MaxRooms": 5}, rng) in range(5, 10)
    assert room_count({}, rng) == 7


def test_same_seed_same_level():
    assert generate(LARGE, random.Random(11)) == generate(LARGE, random.Random(11))
    assert generate(LARGE, random.Random(11)) != generate(LARGE, random.Random(12))


@pytest.mark.parametrize("row", [SMALL, LARGE], ids=["7 rooms", "15-18 rooms"])
@pytest.mark.parametrize("seed", range(25))
def test_level_shape(row, seed):
    level = generate(row, random.Random(seed))
    rooms = level["rooms"]
    cells = {(r, c) for r, c, _ in rooms.values()}
    kinds = collections.Counter(kind for _, _, kind in rooms.values())

    assert row["MinRooms"] <= len(rooms) <= row["MaxRooms"]
    assert all(name == room_name(r, c) for name, (r, c, _) in rooms.items())
    assert min(r for r, _ in cells) == 0 and min(c for _, c in cells) == 0

    # One connected web: every room reachable from the start.
    start = rooms[level["start"]][:2]
    seen, todo = {start}, [start]
    while todo:
        for n in neighbours(cells, todo.pop()):
            if n not in seen:
                seen.add(n)
                todo.append(n)
    assert seen == cells

    # The 2x2 rationing that keeps levels thin. `MaybeMakeRoom` holds growth and
    # `Adjust` to it; `CloseLoops` writes rooms directly and can add a block.
    # Nothing bounds a room's neighbours (`CheckNeighborCount` has no caller).
    squares = sum(1 for r, c in cells if {(r + 1, c), (r, c + 1), (r + 1, c + 1)} <= cells)
    assert squares <= len(rooms) // 10 + 2

    assert rooms[level["start"]][2] == "start" and rooms[level["boss"]][2] == "boss"
    assert level["start"] != level["boss"]
    assert kinds["start"] == kinds["boss"] == 1
    assert kinds["item_shop"] == row.get("ItemShops", 0)
    assert kinds["food_shop"] == row.get("FoodShops", 0)
    assert kinds["mutation"] == row.get("Shrines", 0)
    assert kinds["mutation_shop"] == row.get("RerollShrines", 0)

    # Portals: none (a step that lands fewer than two is undone, which a compact
    # small level can force) or between two and `portalsCount + 1`, never on
    # the boss, all at least `minPortalDistance` apart by doors, and the start
    # one exactly when the level is 5 or 9.
    dist = door_distances(cells)
    portals = [rooms[p][:2] for p in level["portals"]]
    assert not portals or 2 <= len(portals) <= portals_count(len(rooms)) + 1
    assert rooms[level["boss"]][:2] not in portals
    assert all(dist[a][b] >= MIN_PORTAL_DISTANCE
               for i, a in enumerate(portals) for b in portals[i + 1:])
    assert not portals or (level["start"] in level["portals"]) == start_is_portal(row["ID"])

    # The finish can see `minFinishDistance`; every level after the first
    # opens in an item shop.
    assert max(dist[rooms[level["boss"]][:2]].values()) >= min_finish_distance(len(rooms))
    assert (level["extras"].get(level["start"]) == ("item_shop",)) == (row["ID"] >= 2)


def door_distances(cells: set) -> dict:
    out = {}
    for src in cells:
        seen, todo = {src: 0}, [src]
        while todo:
            cur = todo.pop(0)
            for n in neighbours(cells, cur):
                if n not in seen:
                    seen[n] = seen[cur] + 1
                    todo.append(n)
        out[src] = seen
    return out


def test_the_generation_parameters_read_off_the_binary():
    # `C_Rooms.Generate`: num / 7 + 1 portals, `C_DefaultMode`: num / 6 + 2
    assert [portals_count(n) for n in (6, 7, 13, 14, 20, 21)] == [1, 2, 2, 3, 3, 4]
    assert [min_finish_distance(n) for n in (7, 11, 12, 18)] == [3, 3, 4, 5]
    # `PositionRooms`' spreads: item shops num >= 18 ? num / 6 : 2, food
    # shops num >= 16 ? num / 4 : 3
    assert [shop_spread("item_shop", n) for n in (7, 17, 18, 24)] == [2, 2, 3, 4]
    assert [shop_spread("food_shop", n) for n in (7, 15, 16, 20)] == [3, 3, 4, 5]
    assert [lvl for lvl in range(1, 13) if start_is_portal(lvl)] == [5, 9]
    assert [lvl for lvl in range(1, 13) if start_is_portal(lvl, "shortpvp")] == [3, 5, 9]


# A corridor of seven rooms, 0 .. 6, with a spur 7 off room 3:
#
#     0 - 1 - 2 - 3 - 4 - 5 - 6
#                 |
#                 7
CORRIDOR = {i: [j for j in (i - 1, i + 1) if 0 <= j <= 6] for i in range(7)}
CORRIDOR[3].append(7)
CORRIDOR[7] = [3]


def bfs(adj, src):
    out, todo = {src: 0}, [src]
    while todo:
        cur = todo.pop(0)
        for n in adj[cur]:
            if n not in out:
                out[n] = out[cur] + 1
                todo.append(n)
    return out


def test_choose_portals_spreads_dead_ends_then_adds_the_middle():
    cells = sorted(CORRIDOR)
    dist = {p: bfs(CORRIDOR, p) for p in cells}
    avg = {p: sum(dist[p].values()) / len(dist[p]) for p in cells}
    dead = {p: len(CORRIDOR[p]) == 1 for p in cells}
    kinds = {6: "boss", 0: "start"}
    # The dead ends are 0, 6 and 7, and 0 is the start and 6 the boss, so 7 is
    # the only candidate. One is fewer than two, so the rooms are sorted by
    # average distance, largest first (0 and 6, then 1, 5 and 7, then 2 and 4,
    # then 3), and the walk tops the candidates up to four: 7, 1, 5, 2. By mean
    # distance to the candidates 5 is the most spread out (2.5), then 7 and 1
    # (2.0), then 2. 5 is taken, then 7, three doors from it. The most central
    # room three or more from both is 1 (four from 5, three from 7).
    got = choose_portals(cells, kinds, dist, avg, dead, boss=6, start=0, count=2,
                         threshold=3, forced_start=False)
    assert got == [5, 7, 1]
    # With the start forced it is chosen first and takes one of the two places:
    # 0, then 5 (five doors away), then 7 as the central one.
    forced = choose_portals(cells, kinds, dist, avg, dead, boss=6, start=0,
                            count=2, threshold=3, forced_start=True)
    assert forced == [0, 5, 7]
    # Nothing can be three apart from anything on a three-room corridor, and a
    # single portal is undone.
    tiny = {0: [1], 1: [0, 2], 2: [1]}
    tdist = {p: bfs(tiny, p) for p in tiny}
    tavg = {p: sum(tdist[p].values()) / 3 for p in tiny}
    assert choose_portals([0, 1, 2], {2: "boss"}, tdist, tavg,
                          {0: True, 1: False, 2: True}, boss=2, start=1,
                          count=2, threshold=3, forced_start=False) == []


def test_choose_diverse_positions_spreads_only_against_its_own_draws():
    cells = sorted(CORRIDOR)
    dist = {p: bfs(CORRIDOR, p) for p in cells}
    kinds = {0: "start", 6: "boss"}
    for seed in range(50):
        got = choose_diverse_positions(cells, kinds, dist, 2, 4, random.Random(seed))
        assert len(got) == 2 and not set(got) & set(kinds)
        # Four doors apart is only 1 and 5, so ten draws can miss it and the
        # threshold comes down to three, which every first draw but 3 can meet.
        # The start and the boss are not measured against at all.
        assert dist[got[0]][got[1]] >= 3
    # More than there is room for comes back as many as fit, not a hang.
    assert len(choose_diverse_positions(cells, kinds, dist, 9, 2,
                                        random.Random(0))) == 6


def test_articulation_points():
    gen = Generator(7, random.Random(0))
    path = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    assert gen.articulation_points(path) == {"b"}
    ring = {"a": ["b", "d"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c", "a"]}
    assert gen.articulation_points(ring) == set()
    # Two rings sharing one room: only the shared room splits the level.
    bowtie = {"x": ["a", "b", "c", "d"], "a": ["x", "b"], "b": ["a", "x"],
              "c": ["x", "d"], "d": ["c", "x"]}
    assert gen.articulation_points(bowtie) == {"x"}


# A T on the grid: a row of four rooms with one room hanging under the second.
#
#     (0,0) (0,1) (0,2) (0,3)
#           (1,1)
T_SHAPE = {(0, 0): [(0, 1)], (0, 1): [(0, 0), (0, 2), (1, 1)],
           (0, 2): [(0, 1), (0, 3)], (0, 3): [(0, 2)], (1, 1): [(0, 1)]}


def test_quest_room_doors_must_fit_the_prefab():
    cells = sorted(T_SHAPE)
    # Rows count down from the top, so the room under (0, 1) is `Down` of it.
    assert directions((0, 1), T_SHAPE) == {"Left", "Right", "Down"}
    assert directions((1, 1), T_SHAPE) == {"Up"}
    # Doors only Up and Left (Sci's prefab): the room hanging below, and (0, 3),
    # whose one door is on its left. Never the hub, which has three.
    sci = {"Name": "Sci", "Doors": ["Up", "Left"]}
    got = {choose_quest_room(cells, {}, T_SHAPE, sci, random.Random(s)) for s in range(40)}
    assert got == {(1, 1), (0, 3)}
    # A guaranteed door must be there: Rat wants `Down`, which only the hub has.
    rat = {"Name": "Rat", "Doors": ["Left", "Right", "Down"], "GuaranteedDoors": ["Down"]}
    assert choose_quest_room(cells, {}, T_SHAPE, rat, random.Random(0)) == (0, 1)
    # A typed room is never a candidate, and none fitting fails the step.
    assert choose_quest_room(cells, {(0, 1): "start"}, T_SHAPE, rat, random.Random(0)) is None
    # Flight takes the leftmost candidate, not a draw.
    flight = {"Name": "Flight", "Doors": ["Left", "Right", "Up", "Down"]}
    assert all(choose_quest_room(cells, {}, T_SHAPE, flight, random.Random(s)) == (0, 0)
               for s in range(10))


def test_quest_extra_room_is_a_far_dead_end_or_corridor():
    cells = sorted(T_SHAPE)
    dist = {p: bfs(T_SHAPE, p) for p in cells}
    gen = Generator(7, random.Random(0))
    arts = gen.articulation_points(T_SHAPE)
    n_count = lambda p: len(T_SHAPE[p])
    kinds = {(1, 1): "quest"}
    # Not an articulation point, one or two doors, more than one room from the
    # quest room: (0, 0) and (0, 3) are two away; (0, 1) is an articulation
    # point and next door anyway, (0, 2) is an articulation point.
    got = {choose_quest_extra_room(cells, kinds, dist, arts, n_count, (1, 1),
                                   {"Name": "Cube"}, random.Random(s)) for s in range(40)}
    assert got == {(0, 0), (0, 3)}
    # Flight takes the rightmost candidate, and only when it lies left of the
    # quest room. From (1, 1) the rightmost is (0, 3), to its right: nothing.
    # On a corridor with the quest room at the right end, (0, 1) is the
    # rightmost room far enough away, and it lies left of it.
    flight = {"Name": "Flight"}
    assert choose_quest_extra_room(cells, {(1, 1): "quest"}, dist, arts, n_count,
                                   (1, 1), flight, random.Random(0)) is None
    wide = {(0, 0): [(0, 1)], (0, 1): [(0, 0), (0, 2)], (0, 2): [(0, 1), (0, 3)],
            (0, 3): [(0, 2)]}
    wdist = {p: bfs(wide, p) for p in wide}
    assert choose_quest_extra_room(sorted(wide), {(0, 3): "quest"}, wdist, set(),
                                   lambda p: len(wide[p]), (0, 3), flight,
                                   random.Random(0)) == (0, 1)


def test_quest_needs_a_way_in_past_the_boss_and_the_extra_rooms():
    # The quest room (0, 3) is reached from the start (0, 0) only through (0, 1)
    # and (0, 2).
    assert quest_reachable(T_SHAPE, (0, 0), (0, 3), set())
    assert not quest_reachable(T_SHAPE, (0, 0), (0, 3), {(0, 2)})
    assert not quest_reachable(T_SHAPE, (1, 1), (0, 3), {(0, 1)})
    # Next door to the start is always reachable.
    assert quest_reachable(T_SHAPE, (0, 2), (0, 3), {(0, 1)})


def test_a_quest_level_places_its_rooms():
    quest = {"Name": "Cube", "Doors": ["Up", "Down", "Left", "Right"], "ExtraRooms": 1}
    for seed in range(10):
        level = generate(LARGE, random.Random(seed), quest=quest)
        kinds = collections.Counter(k for _, _, k in level["rooms"].values())
        assert kinds["quest"] == 1 and kinds["quest_extra"] == 1
        assert level["quest_room"] in level["rooms"]
        assert level["quest_extras"] and level["quest_extras"][0] in level["rooms"]
        assert not set(level["portals"]) & {level["quest_room"], *level["quest_extras"]}
    # No quest, no quest rooms.
    plain = generate(LARGE, random.Random(0))
    assert plain["quest_room"] is None and plain["quest_extras"] == []
