"""The low level of the hierarchy: where to stand before a fight.

The player zone is a 7x7 block of `p` cells in the room layout, so placement is
a choice of one cell per unit inside that block. A placement policy is any
callable `(grid, layout, specs, rng, enemies) -> [(row, col), ...]`, where
`enemies` are the already-deployed opposing agents: the game shows them before
the fight starts, so a policy is allowed to react to them.

Two reference policies live here: the random spread the sim used before, and a
front-line heuristic that puts melee at the enemy-facing edge and ranged behind.
The heuristic is the baseline a learned policy has to beat.
"""
from __future__ import annotations

import random


def zone_cells(layout) -> list[tuple[int, int]]:
    """The player deployment block, ordered left-to-right then top-to-bottom."""
    return sorted(layout.zone("p"))


def random_placement(grid, layout, specs, rng: random.Random, enemies=None):
    cells = zone_cells(layout)
    if not cells:
        return [(0, 0)] * len(specs)
    picks = list(cells)
    rng.shuffle(picks)
    return [picks[i % len(picks)] for i in range(len(specs))]


def frontline_placement(grid, layout, specs, rng: random.Random, enemies=None):
    """Melee at the column nearest the enemy, ranged as far back as possible.

    Enemies spawn in the `e1`/`e2` zones, which sit to the right of the player
    block in every shipped layout, so "front" is the highest column index.
    """
    cells = zone_cells(layout)
    if not cells:
        return [(0, 0)] * len(specs)
    by_col = sorted({c for _, c in cells})
    front, back = by_col[-1], by_col[0]

    front_cells = [rc for rc in cells if rc[1] == front]
    back_cells = [rc for rc in cells if rc[1] == back]
    mid_cells = [rc for rc in cells if rc[1] not in (front, back)] or cells

    out = []
    fi = bi = mi = 0
    for s in specs:
        if s.melee:
            pool, i = front_cells, fi
            fi += 1
        elif s.range_world > 40:
            pool, i = back_cells, bi
            bi += 1
        else:
            pool, i = mid_cells, mi
            mi += 1
        pool = pool or cells
        out.append(pool[i % len(pool)])
    return out


POLICIES = {"random": random_placement, "frontline": frontline_placement}
