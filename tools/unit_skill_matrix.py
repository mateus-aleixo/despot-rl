"""Per-unit skill probe.

For every unit class carrying an implemented active skill, run a controlled
fight and report whether the skill fired. When it does not, say why -- a skill
can legitimately fail to fire because the fight ended first, the unit never
banked enough mana, or the target was never in range. That distinction is the
whole point: "never cast" alone cannot tell a bug from correct behaviour.
"""
import dataclasses
import random
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sim.assumptions import DEFAULT
from sim.battle import Battle, apply_class_skills, deploy
from sim.data import load_ruleset, parse_room_layouts, skills_by_id, units_by_class
from sim.nav import Grid
from sim.spec import build_player_squad, build_unit
from sim.unit_skills import ACTIVE, handler_for

t = load_ruleset(strict=False)
L = parse_room_layouts(t["RoomLayouts"])[0]
GRID = Grid.from_layout(L)
S = skills_by_id(t)
ubc = units_by_class(t)
META = t["Meta"]["Classes"]

targets = []
for cls, lv in ubc.items():
    row = lv[min(lv)]
    ids = [int(row[f"Skill{i}"]) for i in range(1, 9) if row.get(f"Skill{i}")]
    for i in ids:
        s = S.get(i)
        if s and handler_for(s["CSClass"]).kind == ACTIVE:
            targets.append((cls, s["CSClass"]))
            break


def probe(cls, seed=1, secs=60.0):
    """One enemy of `cls` against a weak squad, so the fight lasts."""
    size = int((META.get(cls) or {}).get("Size") or 1)
    rng = random.Random(seed)
    team0 = deploy(GRID, L, build_player_squad(t, [("broadsword", 1)] * 3), 0, "p", rng)
    apply_class_skills(t, team0)
    enemy = deploy(GRID, L, [build_unit(ubc, cls, 1, name=cls, size=size)], 1, "e1", rng)
    # dataclasses.replace, not mutation: `DEFAULT` is shared by every battle.
    a = dataclasses.replace(DEFAULT, max_fight_seconds=secs)
    b = Battle(GRID, team0 + enemy, assumptions=a, seed=seed, tables=t)
    e = enemy[0]

    peak_mana = 0.0
    ever_in_range = False
    while b.tick_count < int(secs * a.tick_hz):
        if not e.alive or not any(x.alive for x in team0):
            break
        b.step()
        peak_mana = max(peak_mana, e.mana)
        if e.target is not None:
            d2 = (e.target.x - e.x) ** 2 + (e.target.y - e.y) ** 2
            for act in e.actions:
                if act.name != "attack" and act.range_world ** 2 > d2:
                    ever_in_range = True
    skill_act = next((x for x in e.actions if x.name != "attack"), None)
    casts = sum(v for k, v in b.casts.items() if k != "attack")
    return b, e, skill_act, casts, peak_mana, ever_in_range


print(f"{len(targets)} unit classes carry an implemented active skill\n")
print(f"{'unit class':22s} {'skill':20s} {'casts':5s} {'summ':5s} why-not")
fired, explained, unexplained = 0, 0, []
for cls, skill in sorted(targets):
    b, e, act, casts, peak, in_range = probe(cls)
    summ = sum(1 for x in b.agents if x.summoned)
    why = ""
    if casts == 0:
        if act is None:
            why = "no action built"
        elif peak < act.mana_cost:
            why = f"mana: peaked {peak:.0f} < cost {act.mana_cost:.0f}"
        elif not in_range:
            why = f"never in range (reach {act.range_world:.0f})"
        elif not e.alive:
            why = f"died at {b.tick_count / DEFAULT.tick_hz:.1f}s before it could cast"
        else:
            why = "UNEXPLAINED"
    if casts:
        fired += 1
    elif why == "UNEXPLAINED":
        unexplained.append(f"{cls}:{skill}")
    else:
        explained += 1
    print(f"{cls:22s} {skill:20s} {casts:5d} {summ:5d} {why}")

print(f"\n{fired} fired, {explained} explained by a gate, {len(unexplained)} unexplained")
if unexplained:
    print("unexplained:", unexplained)
