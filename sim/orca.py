"""ORCA / RVO2 local collision avoidance in 2D.

This is the algorithm behind `Pathfinding.RVO.RVOController`, which every unit
prefab carries. Parameters come off the shipped prefabs (Swordsman):
`radius: 6`, `agentTimeHorizon: 0.5`, `maxNeighbours: 10`, `priority: 0.5`.

Structure follows the reference RVO2 implementation: build one half-plane
constraint per neighbour, then solve the 2D linear program for the velocity
closest to the preferred one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EPS = 1e-5


@dataclass(frozen=True)
class Line:
    px: float
    py: float
    dx: float
    dy: float


def det(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _norm(x: float, y: float) -> tuple[float, float]:
    n = math.hypot(x, y)
    return (0.0, 0.0) if n < EPS else (x / n, y / n)


def orca_line(pos, vel, radius, other_pos, other_vel, other_radius,
              time_horizon: float, dt: float, share: float) -> Line:
    """The ORCA half-plane this agent must respect for one neighbour.

    `share` is how much of the correction this agent takes; 0.5 is the
    reciprocal split RVO2 uses when both agents run the algorithm.
    """
    rpx, rpy = other_pos[0] - pos[0], other_pos[1] - pos[1]
    rvx, rvy = vel[0] - other_vel[0], vel[1] - other_vel[1]
    dist_sq = rpx * rpx + rpy * rpy
    r = radius + other_radius
    r_sq = r * r

    if dist_sq > r_sq:
        inv_h = 1.0 / time_horizon
        wx, wy = rvx - inv_h * rpx, rvy - inv_h * rpy
        w_len_sq = wx * wx + wy * wy
        dot1 = wx * rpx + wy * rpy
        if dot1 < 0.0 and dot1 * dot1 > r_sq * w_len_sq:
            # closest point is on the cut-off circle
            w_len = math.sqrt(w_len_sq)
            ux, uy = wx / w_len, wy / w_len
            dirx, diry = uy, -ux
            f = r * inv_h - w_len
            ux, uy = f * ux, f * uy
        else:
            # closest point is on one of the legs
            leg = math.sqrt(max(dist_sq - r_sq, 0.0))
            if det(rpx, rpy, wx, wy) > 0.0:
                dirx = (rpx * leg - rpy * r) / dist_sq
                diry = (rpx * r + rpy * leg) / dist_sq
            else:
                dirx = -(rpx * leg + rpy * r) / dist_sq
                diry = -(-rpx * r + rpy * leg) / dist_sq
            dot2 = rvx * dirx + rvy * diry
            ux, uy = dot2 * dirx - rvx, dot2 * diry - rvy
    else:
        # already overlapping: push apart over one timestep
        inv_dt = 1.0 / dt
        wx, wy = rvx - inv_dt * rpx, rvy - inv_dt * rpy
        w_len = math.hypot(wx, wy)
        if w_len < EPS:
            return Line(vel[0], vel[1], 1.0, 0.0)
        ux, uy = wx / w_len, wy / w_len
        dirx, diry = uy, -ux
        f = r * inv_dt - w_len
        ux, uy = f * ux, f * uy

    return Line(vel[0] + share * ux, vel[1] + share * uy, dirx, diry)


def _lp1(lines: list[Line], i: int, radius: float, opt: tuple[float, float],
         direction_opt: bool, result: tuple[float, float]):
    """Optimise along line `i` subject to lines[:i]. Returns the new velocity or None."""
    ln = lines[i]
    dot = ln.px * ln.dx + ln.py * ln.dy
    disc = dot * dot + radius * radius - (ln.px * ln.px + ln.py * ln.py)
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq

    for j in range(i):
        lj = lines[j]
        denom = det(ln.dx, ln.dy, lj.dx, lj.dy)
        numer = det(lj.dx, lj.dy, ln.px - lj.px, ln.py - lj.py)
        if abs(denom) <= EPS:
            if numer < 0.0:
                return None
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None

    if direction_opt:
        t = t_right if (opt[0] * ln.dx + opt[1] * ln.dy) > 0.0 else t_left
    else:
        t = ln.dx * (opt[0] - ln.px) + ln.dy * (opt[1] - ln.py)
        t = min(max(t, t_left), t_right)
    return (ln.px + t * ln.dx, ln.py + t * ln.dy)


def _lp2(lines: list[Line], radius: float, opt: tuple[float, float], direction_opt: bool):
    """Returns (velocity, index of first violated line or len(lines))."""
    if direction_opt:
        result = (opt[0] * radius, opt[1] * radius)
    elif opt[0] * opt[0] + opt[1] * opt[1] > radius * radius:
        nx, ny = _norm(*opt)
        result = (nx * radius, ny * radius)
    else:
        result = opt

    for i, ln in enumerate(lines):
        if det(ln.dx, ln.dy, ln.px - result[0], ln.py - result[1]) > 0.0:
            new = _lp1(lines, i, radius, opt, direction_opt, result)
            if new is None:
                return result, i
            result = new
    return result, len(lines)


def _lp3(lines: list[Line], num_obst: int, begin: int, radius: float,
         result: tuple[float, float]) -> tuple[float, float]:
    """Fallback when the program is infeasible: minimise the worst violation."""
    distance = 0.0
    for i in range(begin, len(lines)):
        li = lines[i]
        if det(li.dx, li.dy, li.px - result[0], li.py - result[1]) <= distance:
            continue
        proj = list(lines[:num_obst])
        for j in range(num_obst, i):
            lj = lines[j]
            d = det(li.dx, li.dy, lj.dx, lj.dy)
            if abs(d) <= EPS:
                if li.dx * lj.dx + li.dy * lj.dy > 0.0:
                    continue
                px, py = 0.5 * (li.px + lj.px), 0.5 * (li.py + lj.py)
            else:
                t = det(lj.dx, lj.dy, li.px - lj.px, li.py - lj.py) / d
                px, py = li.px + t * li.dx, li.py + t * li.dy
            dx, dy = _norm(lj.dx - li.dx, lj.dy - li.dy)
            proj.append(Line(px, py, dx, dy))
        new, idx = _lp2(proj, radius, (-li.dy, li.dx), True)
        if idx >= len(proj):
            result = new
        distance = det(li.dx, li.dy, li.px - result[0], li.py - result[1])
    return result


def new_velocity(pos, vel, radius, max_speed, pref_vel, neighbours,
                 time_horizon: float = 0.5, dt: float = 0.02,
                 max_neighbours: int = 10, share: float = 0.5):
    """Collision-free velocity closest to `pref_vel`.

    `neighbours` is [(pos, vel, radius), ...]; the nearest `max_neighbours` are
    used, matching `RVOController.maxNeighbours`.
    """
    if len(neighbours) > max_neighbours:
        neighbours = sorted(
            neighbours,
            key=lambda n: (n[0][0] - pos[0]) ** 2 + (n[0][1] - pos[1]) ** 2,
        )[:max_neighbours]

    lines = [orca_line(pos, vel, radius, npos, nvel, nrad, time_horizon, dt, share)
             for npos, nvel, nrad in neighbours]

    result, idx = _lp2(lines, max_speed, pref_vel, False)
    if idx < len(lines):
        result = _lp3(lines, 0, idx, max_speed, result)
    return result
