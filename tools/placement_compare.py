"""Does placement matter? Same squads, same enemies, different starting cells."""
import random, statistics, sys
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sim.battle import Battle, apply_class_skills, place_at
from sim.data import load_ruleset, parse_room_layouts
from sim.nav import Grid
from sim.spec import build_enemy_pack, build_player_squad
from rl.placement import POLICIES

t = load_ruleset(strict=True)
layouts = parse_room_layouts(t["RoomLayouts"])
pack = next(p for p in t["EnemyPacks"] if p["Class"] == "Mancrack")

SQUADS = {
    "3 melee + 3 ranged": [("broadsword", 1)] * 3 + [("gun", 1)] * 3,
    "2 melee + 4 ranged": [("broadsword", 1)] * 2 + [("crossbow", 1)] * 4,
    "all ranged":         [("gun", 1)] * 6,
}

print(f"{'squad':22s} {'placement':11s} {'wins':7s} {'survivors':9s} {'dmg taken':9s}")
for label, loadout in SQUADS.items():
    for pname, pol in POLICIES.items():
        wins = surv = taken = 0
        n = 24
        for s in range(n):
            layout = layouts[s % len(layouts)]
            grid = Grid.from_layout(layout)
            specs = build_player_squad(t, loadout)
            rng = random.Random(s)
            cells = pol(grid, layout, specs, rng)
            team0 = place_at(grid, specs, cells, team=0)
            apply_class_skills(t, team0)
            from sim.battle import deploy
            team1 = deploy(grid, layout, build_enemy_pack(t, pack)[:8], 1, "e1", rng)
            b = Battle(grid, team0 + team1, seed=s, tables=t)
            r = b.run()
            wins += (r.winner == 0)
            surv += r.survivors[0]
            taken += r.total_damage[1]
        print(f"{label:22s} {pname:11s} {wins:2d}/{n:<4d} {surv/n:9.2f} {taken/n:9.0f}")
