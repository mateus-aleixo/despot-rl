"""ORCA local avoidance (sim/orca.py).

The property that matters: when two agents each take a velocity from their own
ORCA half-plane, they do not collide within the time horizon. That is what the
reciprocal construction guarantees, so it is checked directly by moving both
agents along their new velocities.
"""
import math
import random

import pytest

from sim.orca import new_velocity

R = 6.0       # `radius: 6` on the shipped unit prefabs
TAU = 0.5     # `agentTimeHorizon: 0.5`


def closest_approach(pa, va, pb, vb, horizon: float) -> float:
    """Smallest distance between two points moving linearly over [0, horizon]."""
    rx, ry = pb[0] - pa[0], pb[1] - pa[1]
    wx, wy = vb[0] - va[0], vb[1] - va[1]
    ww = wx * wx + wy * wy
    t = 0.0 if ww == 0.0 else min(max(-(rx * wx + ry * wy) / ww, 0.0), horizon)
    return math.hypot(rx + t * wx, ry + t * wy)


def step_pair(pa, va, pb, vb, max_speed):
    """Both agents run ORCA against each other, preferring their current velocity."""
    na = new_velocity(pa, va, R, max_speed, va, [(pb, vb, R)], time_horizon=TAU)
    nb = new_velocity(pb, vb, R, max_speed, vb, [(pa, va, R)], time_horizon=TAU)
    return na, nb


def test_alone_keeps_its_preferred_velocity():
    assert new_velocity((0, 0), (0, 0), R, 100.0, (30.0, 40.0), []) == (30.0, 40.0)


def test_preferred_speed_is_capped():
    vx, vy = new_velocity((0, 0), (0, 0), R, 100.0, (300.0, 400.0), [])
    assert (vx, vy) == pytest.approx((60.0, 80.0))


def test_a_distant_neighbour_changes_nothing():
    far = ((1000.0, 0.0), (0.0, 0.0), R)
    assert new_velocity((0, 0), (50, 0), R, 100.0, (50.0, 0.0), [far]) == pytest.approx((50.0, 0.0))


def test_head_on_pair_does_not_collide():
    pa, pb = (0.0, 0.0), (30.0, 0.0)
    va, vb = (50.0, 0.0), (-50.0, 0.0)
    # Unchanged, they would meet inside the horizon.
    assert closest_approach(pa, va, pb, vb, TAU) < 2 * R
    na, nb = step_pair(pa, va, pb, vb, max_speed=100.0)
    assert closest_approach(pa, na, pb, nb, TAU) >= 2 * R - 1e-6
    assert math.hypot(*na) <= 100.0 + 1e-9 and math.hypot(*nb) <= 100.0 + 1e-9


def test_random_encounters_do_not_collide():
    rng = random.Random(7)
    checked = 0
    for _ in range(500):
        pa = (rng.uniform(-40, 40), rng.uniform(-40, 40))
        pb = (rng.uniform(-40, 40), rng.uniform(-40, 40))
        if math.dist(pa, pb) <= 2 * R + 0.5:
            continue
        va = (rng.uniform(-60, 60), rng.uniform(-60, 60))
        vb = (rng.uniform(-60, 60), rng.uniform(-60, 60))
        # A speed cap well above the current speeds keeps both programs
        # feasible, which is the case the guarantee is stated for.
        na, nb = step_pair(pa, va, pb, vb, max_speed=500.0)
        assert closest_approach(pa, na, pb, nb, TAU) >= 2 * R - 1e-6
        checked += 1
    assert checked > 300


def test_mirrored_scene_gives_mirrored_velocity():
    pos, vel, pref = (0.0, 0.0), (40.0, 5.0), (40.0, 5.0)
    other = ((25.0, 3.0), (-40.0, 0.0), R)
    vx, vy = new_velocity(pos, vel, R, 100.0, pref, [other])
    mirrored = ((25.0, -3.0), (-40.0, 0.0), R)
    mx, my = new_velocity((0.0, 0.0), (40.0, -5.0), R, 100.0, (40.0, -5.0), [mirrored])
    assert (mx, my) == pytest.approx((vx, -vy), abs=1e-9)
    # And the avoidance is real: the agent leaves its preferred velocity.
    assert (vx, vy) != pytest.approx(pref)


def test_overlapping_agents_push_apart():
    vx, _ = new_velocity((0.0, 0.0), (0.0, 0.0), R, 100.0, (0.0, 0.0),
                         [((5.0, 0.0), (0.0, 0.0), R)])
    assert vx < 0.0


def test_only_the_nearest_neighbours_count():
    near = ((20.0, 0.0), (-40.0, 0.0), R)
    far = ((80.0, 1.0), (-40.0, 0.0), R)
    pos, vel = (0.0, 0.0), (40.0, 0.0)
    only_near = new_velocity(pos, vel, R, 100.0, vel, [near], max_neighbours=1)
    assert new_velocity(pos, vel, R, 100.0, vel, [far, near], max_neighbours=1) == only_near
    assert new_velocity(pos, vel, R, 100.0, vel, [near, far], max_neighbours=1) == only_near
