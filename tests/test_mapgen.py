"""Level generation (sim/mapgen.py): the shape rules every generated level obeys."""
import collections
import random

import pytest

from sim.mapgen import DIRS, Generator, generate, room_count, room_name

# Shaped like `M_Levels` rows: level 1 is exactly 7 rooms; later levels draw a range.
SMALL = {"MinRooms": 7, "MaxRooms": 7, "ItemShops": 1, "FoodShops": 1, "Shrines": 1}
LARGE = {"MinRooms": 15, "MaxRooms": 18, "ItemShops": 2, "FoodShops": 2,
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

    # `CheckNeighborCount`, and the 2x2 rationing that keeps levels thin.
    assert all(len(neighbours(cells, p)) <= 3 for p in cells)
    squares = sum(1 for r, c in cells if {(r + 1, c), (r, c + 1), (r + 1, c + 1)} <= cells)
    assert squares <= len(rooms) // 10 + 1

    assert rooms[level["start"]][2] == "start" and rooms[level["boss"]][2] == "boss"
    assert level["start"] != level["boss"]
    assert kinds["start"] == kinds["boss"] == 1
    assert kinds["item_shop"] == row.get("ItemShops", 0)
    assert kinds["food_shop"] == row.get("FoodShops", 0)
    assert kinds["mutation"] == row.get("Shrines", 0)
    assert kinds["mutation_shop"] == row.get("RerollShrines", 0)


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
