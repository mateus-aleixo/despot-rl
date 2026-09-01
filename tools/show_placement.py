"""Print what a placement policy actually does, next to the heuristic.

Numbers say a policy is better; this says what it does differently. Melee are
`M`, ranged `R`, enemies `x`, and the player zone is dotted.
"""
from __future__ import annotations

import argparse
import random
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rl.place_env import cached_scenarios, resolve, sample_close_scenarios, score
from rl.placement import POLICIES
from sim.battle import Agent
from sim.data import load_ruleset, parse_room_layouts
from sim.nav import Grid


def render(scn, layout, grid, cells) -> list[str]:
    rows, cols = layout.size
    canvas = [["  " for _ in range(cols)] for _ in range(rows)]
    for r, c in layout.zone("p"):
        canvas[r][c] = " ."
    for spec, (x, y) in zip(scn.enemy_specs, scn.enemy_xy):
        r, c = int(y // grid.tile), int(x // grid.tile)
        if 0 <= r < rows and 0 <= c < cols:
            canvas[r][c] = " x" if canvas[r][c] in ("  ", " .") else " X"
    for spec, (r, c) in zip(scn.specs, cells):
        if 0 <= r < rows and 0 <= c < cols:
            canvas[r][c] = " M" if spec.melee else " R"
    return ["".join(row) for row in canvas]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/placement_final.pt")
    ap.add_argument("--fights", type=int, default=4)
    ap.add_argument("--cache", default="runs/scenarios_close_600_200_0.pkl")
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    layouts = parse_room_layouts(tables["RoomLayouts"])
    _, scns = cached_scenarios(args.cache, lambda: (
        sample_close_scenarios(tables, n=1, seed=0),
        sample_close_scenarios(tables, n=args.fights, seed=500_000)))

    from rl.place_policy import load_placement
    policies = {"frontline": POLICIES["frontline"],
                "learned": load_placement(args.checkpoint, greedy=True)}

    for scn in scns[:args.fights]:
        layout = layouts[scn.layout_index]
        grid = Grid.from_layout(layout)
        enemies = [Agent(spec=s, team=1, x=x, y=y, hp=s.health, mana=0.0)
                   for s, (x, y) in zip(scn.enemy_specs, scn.enemy_xy)]
        melee = sum(1 for s in scn.specs if s.melee)
        print(f"\n=== {scn.n_units} units ({melee} melee) vs "
              f"{len(scn.enemy_specs)} enemies, layout {scn.layout_index} ===")

        views = {}
        for name, pol in policies.items():
            cells = pol(grid, layout, scn.specs, random.Random(scn.uid), enemies)
            wins = sum(1 for s in scn.seeds
                       if resolve(scn, cells, tables, layouts, s)["won"])
            sc = sum(score(resolve(scn, cells, tables, layouts, s))
                     for s in scn.seeds) / len(scn.seeds)
            views[name] = (render(scn, layout, grid, cells), wins, sc)

        width = max(len(r) for rows, _, _ in views.values() for r in rows)
        head = "  ".join(f"{n} ({w}/{len(scn.seeds)} wins, score {s:+.2f})".ljust(width)
                         for n, (_, w, s) in views.items())
        print(head)
        rowsets = [v[0] for v in views.values()]
        for line in zip(*rowsets):
            print("  ".join(l.ljust(width) for l in line))


if __name__ == "__main__":
    main()
