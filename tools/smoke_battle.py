"""Run one battle end to end and print what happened."""
import random, sys, time
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sim.data import load_ruleset, parse_room_layouts
from sim.spec import build_player_squad, build_enemy_pack
from sim.nav import Grid
from sim.battle import Battle, deploy, apply_damage, apply_class_skills

t = load_ruleset(strict=False)
layouts = parse_room_layouts(t["RoomLayouts"])
layout = layouts[0]
grid = Grid.from_layout(layout)
print(f"layout id={layout.id} size={layout.size} grid={grid.rows}x{grid.cols} "
      f"world={grid.cols*grid.tile:.0f}x{grid.rows*grid.tile:.0f}")

squad = build_player_squad(t, [("broadsword", 1)] * 6)
print(f"\nplayer squad ({len(squad)}):")
s = squad[0]
print(f"  {s.name}: hp={s.health} dmg={s.damage} armor={s.armor} spd={s.speed} "
      f"range={s.range_world:.1f} atk_period={s.attack_period}s ranged={s.is_ranged}")

pack = next(p for p in t["EnemyPacks"] if p["Class"] == "Mancrack")
enemies = build_enemy_pack(t, pack)
e = enemies[0]
print(f"enemy pack {pack['Class']} x{len(enemies)}: hp={e.health} dmg={e.damage} "
      f"armor={e.armor} spd={e.speed} range={e.range_world:.1f}")

rng = random.Random(1)
team0 = deploy(grid, layout, squad, 0, "p", rng)
resolved = apply_class_skills(t, team0)
print('class skills:', {k: (v['name'], v['level']) for k, v in resolved.items()})
agents = team0 + deploy(grid, layout, enemies[:6], 1, "e1", rng)
print(f"\ndeployed {len(agents)} agents")

t0 = time.perf_counter()
b = Battle(grid, agents, seed=1, tables=t)
res = b.run()
el = time.perf_counter() - t0
print(f"\nwinner={res.winner} ticks={res.ticks} sim_seconds={res.seconds:.1f} "
      f"survivors={res.survivors} damage={ {k: round(v,1) for k,v in res.total_damage.items()} }")
print(f"wall time {el:.2f}s -> {res.ticks/el:,.0f} ticks/s, {el/res.seconds:.3f} s wall per sim-second")

print("\n-- damage formula checks (verified against CS_Damage.Apply) --")
print(f"  60 dmg vs 5 armor        -> {apply_damage(60, 5, 0)}")
print(f"  3 dmg vs 5 armor (floor) -> {apply_damage(3, 5, 0)}")
print(f"  0.5 dmg vs 5 armor       -> {apply_damage(0.5, 5, 0)}")
print(f"  60 dmg vs 0.25 resist    -> {apply_damage(60, 0, 0.25, magical=True)}")
