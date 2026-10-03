"""Grid pathfinding and waypoint following (sim/nav.py)."""
import heapq
import math
import random

import pytest

from sim.assumptions import TILE
from sim.data import RoomLayout
from sim.nav import DIAG, NEIGHBOURS, Grid, PathFollower, astar


def grid_from(rows: list[str]) -> Grid:
    """'.' is floor, '#' is blocked."""
    return Grid(rows=len(rows), cols=len(rows[0]),
                walkable=[[ch != "#" for ch in row] for row in rows])


def legal_step(grid: Grid, a, b) -> bool:
    """One 8-connected move with A* Pathfinding Project's no-corner-cutting rule."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    if max(abs(dr), abs(dc)) != 1 or not grid.is_walkable(*b):
        return False
    if dr and dc:
        return grid.is_walkable(a[0] + dr, a[1]) and grid.is_walkable(a[0], a[1] + dc)
    return True


def path_cost(path) -> float:
    return sum(DIAG if a[0] != b[0] and a[1] != b[1] else 1.0
               for a, b in zip(path, path[1:]))


def dijkstra_cost(grid: Grid, start, goal):
    """Reference shortest-path cost under the same movement rules, or None."""
    best = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        g, cur = heapq.heappop(heap)
        if cur == goal:
            return g
        if g > best[cur]:
            continue
        for dr, dc in NEIGHBOURS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not legal_step(grid, cur, nxt):
                continue
            ng = g + (DIAG if dr and dc else 1.0)
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                heapq.heappush(heap, (ng, nxt))
    return None


OPEN = grid_from(["....."] * 5)


def test_straight_line():
    assert astar(OPEN, (2, 0), (2, 4)) == [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]


def test_diagonal_across_open_floor():
    assert astar(OPEN, (0, 0), (3, 3)) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_start_is_goal():
    assert astar(OPEN, (1, 1), (1, 1)) == [(1, 1)]


def test_blocked_or_unreachable_goal_gives_empty_path():
    walled = grid_from(["..#..",
                        "..#..",
                        "..#.."])
    assert astar(walled, (0, 0), (0, 4)) == []
    assert astar(walled, (0, 0), (1, 2)) == []


def test_no_corner_cutting():
    # Both orthogonal cells blocked: the diagonal is not a move at all.
    assert astar(grid_from([".#", "#."]), (0, 0), (1, 1)) == []
    # One blocked: the path has to go round the corner.
    assert astar(grid_from([".#", ".."]), (0, 0), (1, 1)) == [(0, 0), (1, 0), (1, 1)]


def test_routes_through_the_only_gap():
    grid = grid_from(["...#...",
                      "...#...",
                      ".......",
                      "...#...",
                      "...#..."])
    path = astar(grid, (0, 0), (0, 6))
    assert (2, 3) in path
    assert all(legal_step(grid, a, b) for a, b in zip(path, path[1:]))


def test_matches_a_reference_dijkstra_on_random_grids():
    rng = random.Random(1234)
    for _ in range(300):
        rows, cols = rng.randint(3, 9), rng.randint(3, 11)
        grid = Grid(rows=rows, cols=cols,
                    walkable=[[rng.random() > 0.3 for _ in range(cols)] for _ in range(rows)])
        floor = [(r, c) for r in range(rows) for c in range(cols) if grid.walkable[r][c]]
        if len(floor) < 2:
            continue
        start, goal = rng.sample(floor, 2)
        want = dijkstra_cost(grid, start, goal)
        path = astar(grid, start, goal)
        if want is None:
            assert path == []
            continue
        assert path[0] == start and path[-1] == goal
        assert all(legal_step(grid, a, b) for a, b in zip(path, path[1:]))
        assert path_cost(path) == pytest.approx(want, abs=1e-9)


def test_world_and_cell_coordinates_round_trip():
    grid = grid_from(["...."] * 3)
    assert grid.to_world(0, 0) == (TILE / 2, TILE / 2)
    for r in range(grid.rows):
        for c in range(grid.cols):
            assert grid.to_cell(*grid.to_world(r, c)) == (r, c)


def test_clamp_keeps_points_inside_the_room():
    grid = grid_from(["...."] * 3)
    x, y = grid.clamp_world(-50.0, 1e9)
    assert 0.0 <= x < grid.cols * grid.tile
    assert 0.0 <= y < grid.rows * grid.tile
    assert grid.to_cell(x, y) == (grid.rows - 1, 0)


def test_grid_from_layout_is_all_floor():
    layout = RoomLayout(id=1, type=0, tags=[], grid=[["p", "", "e1"], ["", "s", ""]])
    grid = Grid.from_layout(layout)
    assert (grid.rows, grid.cols) == (2, 3)
    assert all(all(row) for row in grid.walkable)


def test_follower_heads_for_the_next_waypoint():
    follower = PathFollower(OPEN, pick_next_dist=12.0)
    follower.set_path([(0, 0), (0, 4)])
    # Standing on the first waypoint advances past it, toward the second.
    assert follower.desired_direction(*OPEN.to_world(0, 0)) == pytest.approx((1.0, 0.0))
    # On the last waypoint there is nowhere left to go.
    assert follower.desired_direction(*OPEN.to_world(0, 4)) == (0.0, 0.0)


def test_follower_walks_a_path_to_its_end():
    path = astar(OPEN, (0, 0), (4, 2))
    follower = PathFollower(OPEN, pick_next_dist=12.0)
    follower.set_path(path)
    x, y = OPEN.to_world(*path[0])
    goal = OPEN.to_world(*path[-1])
    for _ in range(500):
        if math.dist((x, y), goal) < 0.5:
            break
        dx, dy = follower.desired_direction(x, y)
        assert math.hypot(dx, dy) == pytest.approx(1.0)
        x, y = x + 0.4 * dx, y + 0.4 * dy
    assert math.dist((x, y), goal) < 0.5
    assert follower.index == len(path) - 1
