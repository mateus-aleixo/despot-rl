"""Room geometry, grid pathfinding and path following.

The walkability grid comes from the room layout for now. The shipped A* graph
caches carry the true obstacle map (three grids per room shape, one per unit
collision size); parsing their node records is the next fidelity step, and
`Grid.from_astar_cache` is where that will land. Everything above the grid --
A*, waypoint following -- is written against the grid interface, so swapping
the source does not touch it.

Distances are world units. One tile is 6 world units, from the graph caches'
`"nodeSize": 6`.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .assumptions import TILE

# 8-connected, matching A* Pathfinding Project's default grid neighbours.
NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
DIAG = math.sqrt(2.0)


@dataclass
class Grid:
    rows: int
    cols: int
    walkable: list[list[bool]]
    tile: float = TILE

    @classmethod
    def from_layout(cls, layout) -> "Grid":
        """Every cell of the room rectangle is floor.

        The layout's tokens mark spawn zones (`p`, `e1`, `e2`) and the door
        (`s`), not obstacles, so nothing here is blocked. Real obstacles come
        from the A* cache.
        """
        rows, cols = layout.size
        return cls(rows=rows, cols=cols, walkable=[[True] * cols for _ in range(rows)])

    @classmethod
    def from_astar_cache(cls, _cache_bytes: bytes) -> "Grid":
        raise NotImplementedError(
            "A* node record layout is not decoded yet: uint32 count then 22 bytes "
            "per node, with a coordinate field stepping by 6000 (nodeSize 6 at "
            "Int3's 1000-unit fixed point). See notes/datamining.md.")

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.walkable[r][c]

    def to_world(self, r: int, c: int) -> tuple[float, float]:
        """Cell centre in world coordinates."""
        return ((c + 0.5) * self.tile, (r + 0.5) * self.tile)

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(y // self.tile), int(x // self.tile))

    def clamp_world(self, x: float, y: float) -> tuple[float, float]:
        return (min(max(x, 0.0), self.cols * self.tile - 1e-3),
                min(max(y, 0.0), self.rows * self.tile - 1e-3))


def astar(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Octile-heuristic A* over the grid. Returns cells start..goal, or []."""
    if start == goal:
        return [start]
    if not grid.is_walkable(*goal):
        return []

    def h(a, b):
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dr + dc) + (DIAG - 2) * min(dr, dc)

    open_heap = [(h(start, goal), 0.0, start)]
    came: dict = {start: None}
    cost: dict = {start: 0.0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        if g > cost.get(cur, float("inf")):
            continue
        for dr, dc in NEIGHBOURS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not grid.is_walkable(nr, nc):
                continue
            if dr and dc:  # no corner cutting, as A* PP does by default
                if not (grid.is_walkable(cur[0] + dr, cur[1]) and grid.is_walkable(cur[0], cur[1] + dc)):
                    continue
            step = DIAG if dr and dc else 1.0
            ng = g + step
            if ng < cost.get((nr, nc), float("inf")):
                cost[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heapq.heappush(open_heap, (ng + h((nr, nc), goal), ng, (nr, nc)))
    return []


class PathFollower:
    """Walks a cell path, advancing when within `pick_next_dist` of a waypoint.

    `pick_next_dist` is 12 on the shipped Swordsman prefab
    (`UnitMovement.pickNextWaypointDist`).
    """

    def __init__(self, grid: Grid, pick_next_dist: float = 12.0):
        self.grid = grid
        self.pick_next_dist = pick_next_dist
        self.waypoints: list[tuple[float, float]] = []
        self.index = 0

    def set_path(self, cells: list[tuple[int, int]]) -> None:
        self.waypoints = [self.grid.to_world(r, c) for r, c in cells]
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def desired_direction(self, x: float, y: float) -> tuple[float, float]:
        """Unit vector toward the current waypoint, advancing past reached ones."""
        while self.index < len(self.waypoints):
            wx, wy = self.waypoints[self.index]
            dx, dy = wx - x, wy - y
            d = math.hypot(dx, dy)
            if d <= self.pick_next_dist and self.index < len(self.waypoints) - 1:
                self.index += 1
                continue
            if d < 1e-6:
                return (0.0, 0.0)
            return (dx / d, dy / d)
        return (0.0, 0.0)
