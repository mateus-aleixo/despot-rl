"""Run one squad per player class and report what its class skill did."""
import random, sys
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sim.data import load_ruleset, parse_room_layouts
from sim.spec import build_player_squad, build_enemy_pack
from sim.nav import Grid
from sim.battle import Battle, deploy, apply_class_skills
from sim.actions import PASSIVE_SKILLS

t = load_ruleset(strict=False)
L = parse_room_layouts(t["RoomLayouts"])[0]
g = Grid.from_layout(L)
pack = next(p for p in t["EnemyPacks"] if p["Class"] == "Mancrack")

# one representative item per class
rep = {}
for i in t["Items"]:
    rep.setdefault(i["Class"], i["Name"])

print(f"{'class':10s} {'skill':15s} {'lvl':3s} {'casts':6s} {'dmg':7s} {'heal':6s} "
      f"{'summ':5s} {'win':4s} {'secs':5s}")
for cls in sorted(rep):
    wins = 0; casts = dmg = heal = summ = 0.0; secs = []
    for seed in range(6):
        squad = build_player_squad(t, [(rep[cls], 1)] * 5)
        rng = random.Random(seed)
        team0 = deploy(g, L, squad, 0, "p", rng)
        resolved = apply_class_skills(t, team0)
        agents = team0 + deploy(g, L, build_enemy_pack(t, pack)[:6], 1, "e1", rng)
        b = Battle(g, agents, seed=seed, tables=t)
        r = b.run()
        wins += (r.winner == 0)
        casts += sum(v for k, v in r.casts.items() if k != "attack")
        dmg += r.total_damage[0]; heal += r.healing[0]
        summ += sum(1 for a in b.agents if a.summoned)
        secs.append(r.seconds)
    sk = next(iter(resolved.values()), None)
    name = sk["name"] if sk else "-"
    lvl = sk["level"] if sk else "-"
    tag = f"{name}{'*' if name in PASSIVE_SKILLS else ''}"
    print(f"{cls:10s} {tag:15s} {str(lvl):3s} {casts/6:6.1f} {dmg/6:7.0f} {heal/6:6.0f} "
          f"{summ/6:5.1f} {wins:d}/6  {sum(secs)/6:5.1f}")
print("\n* = passive skill, applied as a modifier rather than cast")
