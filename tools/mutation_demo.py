"""Apply mutations to a squad and show what changes."""
import dataclasses, random, sys
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sim.assumptions import DEFAULT
from sim.battle import Battle, apply_class_skills, deploy
from sim.data import load_ruleset, parse_room_layouts
from sim.mutations import (apply_to_agents, apply_to_specs, by_id, coverage,
                           handler_for, mutation_params, offered_at_level)
from sim.nav import Grid
from sim.spec import build_enemy_pack, build_player_squad
from sim.unit_skills import PASSIVE

t = load_ruleset(strict=True)
L = parse_room_layouts(t["RoomLayouts"])[0]
G = Grid.from_layout(L)
pack = next(p for p in t["EnemyPacks"] if p["Class"] == "Mancrack")

c = coverage(t); tot = sum(c["by_kind"].values())
print("mutation coverage:", dict(sorted(c["by_kind"].items(), key=lambda x: -x[1])),
      f"-> handled {tot - c['by_kind'].get('unimplemented', 0)}/{tot}")

defs = by_id(t)
lvl1 = offered_at_level(t, 1)
stat_muts = [m for m in lvl1 if m["Name"] in ("StatBonus", "OraStatBonus")]
print(f"\nlevel 1 offers {len(lvl1)} mutations, {len(stat_muts)} of them stat bonuses")
for m in stat_muts[:6]:
    print(f"   ID={m['ID']:4d} Class={str(m.get('Class'))[:28]:28s} {mutation_params(m)}")


def run(muts, seeds=10, label=""):
    wins, dmg = 0, 0.0
    for s in range(seeds):
        specs = build_player_squad(t, [("broadsword", 1)] * 6)
        specs = apply_to_specs(specs, muts, random.Random(s))
        rng = random.Random(s)
        team0 = deploy(G, L, specs, 0, "p", rng)
        apply_class_skills(t, team0)
        apply_to_agents(team0, muts, random.Random(s))
        b = Battle(G, team0 + deploy(G, L, build_enemy_pack(t, pack)[:10], 1, "e1", rng),
                   seed=s, tables=t)
        r = b.run()
        wins += (r.winner == 0); dmg += r.total_damage[0]
    print(f"  {label:36s} wins {wins:2d}/{seeds}  avg dmg {dmg/seeds:7.0f}")
    return wins


print("\n6 broadswords vs 10 Mancrack:")
run([], label="no mutations")
warrior_dmg = [m for m in t["SimpleMutations"]
               if m["Name"] == "StatBonus"
               and mutation_params(m).get("stat") == "Damage"
               and str(mutation_params(m).get("percentage")).lower() == "true"][:1]
print(f"  (applying {mutation_params(warrior_dmg[0])} to Class={warrior_dmg[0].get('Class')})")
run(warrior_dmg, label="+1 damage StatBonus")

hp = [m for m in t["SimpleMutations"]
      if m["Name"] == "StatBonus" and mutation_params(m).get("stat") == "Health"
      and mutation_params(m).get("bonus", 0) > 0][:1]
run(hp, label=f"+health StatBonus {mutation_params(hp[0]).get('bonus')}")
run(warrior_dmg + hp, label="both")

# a passive that routes through the skill registry
ev = [m for m in t["SimpleMutations"] if m["Name"] == "Evasion"][:1]
print(f"  (Evasion mutation params: {mutation_params(ev[0])}, Class={ev[0].get('Class')})")
run(ev, label="Evasion mutation")
