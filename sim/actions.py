"""Actions: the default attack and the class skills.

The behaviour trees do not hardcode "attack". NC-DefaultAttack, NC-Skill and
NC-HealCSkill all drive the same `BT.DefaultAttack.*` leaves; what differs is
the action those leaves are bound to. So a unit is a list of actions, each with
its own range, cooldown, mana cost and effect, and each gets its own subtree
inside the base tree's dynamic Selector -- which is the priority list the
runtime fills via `C_Unit.TryResolveAction`.

Class skills come from `Meta.Classes[cls].Skill` and `ClassSkills.json`, whose
level is chosen by how many humans of that class are in the squad
(`HumansRequired`). Their parameters are exact, read straight from the table;
the `comment1..6` columns name them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .assumptions import DEFAULT, TILE

# Class skills that modify the unit rather than being cast. Applied once at
# setup instead of running as an action.
PASSIVE_SKILLS = {"CriticalStrike", "ASAura", "Dodge"}

# Class skills that target the caster, so they need no approach.
SELF_CAST = {"Tank", "Cultist", "Scientist", "SpiritLink"}

# A mutation-granted self buff also needs no approach.
SELF_CAST_PREFIX = "Mutation:"

# Class skill -> the unit class it summons. Both are real Units.json classes,
# and the "tentacle level" / "tower level" parameter is the summon's Level.
SUMMONS = {"Cultist": "Tentacle", "Scientist": "SciTower"}


@dataclass
class Action:
    """One thing a unit can do, with its own gates and timers."""
    name: str
    tree_name: str
    range_world: float
    cooldown: float
    mana_cost: float
    effect: Callable[["Action", Any, Any], None] | None = None
    params: dict = field(default_factory=dict)
    priority: int = 0
    # per-unit runtime state
    cooldown_left: float = 0.0
    anim_timer: float = 0.0
    swinging: bool = False

    @property
    def is_self_cast(self) -> bool:
        return self.name in SELF_CAST or self.name.startswith(SELF_CAST_PREFIX)


# --------------------------------------------------------------- class skills

def class_skill_rows(tables: dict) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for r in tables["ClassSkills"]:
        rows.setdefault(r["Name"], []).append(r)
    for v in rows.values():
        v.sort(key=lambda r: r["HumansRequired"])
    return rows


def named_params(row: dict) -> dict[str, Any]:
    """Collapse Param1..6 onto the names in comment1..6."""
    out = {}
    for i in range(1, 7):
        name = row.get(f"comment{i}")
        if name:
            out[name] = row.get(f"Param{i}")
    return out


def skill_for_class(tables: dict, cls: str) -> str | None:
    return (tables["Meta"]["Classes"].get(cls) or {}).get("Skill")


def resolve_class_skills(tables: dict, class_counts: dict[str, int]) -> dict[str, dict]:
    """`class -> {name, level, params}` for the squad's composition.

    The level is the highest ClassSkills row whose `HumansRequired` is met by
    the number of humans of that class.
    """
    rows = class_skill_rows(tables)
    out: dict[str, dict] = {}
    for cls, count in class_counts.items():
        name = skill_for_class(tables, cls)
        if not name or name not in rows:
            continue
        eligible = [r for r in rows[name] if r["HumansRequired"] <= count]
        if not eligible:
            continue
        row = eligible[-1]
        out[cls] = {"name": name, "level": row["Level"], "params": named_params(row)}
    return out


# ------------------------------------------------------------------- effects

def effect_attack(action, agent, battle):
    """The default attack."""
    battle.land_attack(agent, action)


def effect_heal(action, agent, battle):
    """Medic: heal the most damaged ally in range.

    `C_HealCSkill.GetMostDamagedTarget` is the real selection rule.
    """
    amount = float(action.params.get("heal") or 0)
    best, worst = None, 0.0
    for o in battle.agents:
        if o.team != agent.team or not o.alive:
            continue
        missing = o.max_hp - o.hp
        if missing > worst:
            best, worst = o, missing
    if best is not None:
        battle.heal(best, amount, agent.team)


def effect_magic(action, agent, battle):
    """Mage: magical damage to the current target (resistance, not armor)."""
    tgt = agent.target
    if tgt is None or not tgt.alive:
        return
    battle.hit(agent.team, tgt, float(action.params.get("damage") or 0), magical=True,
               attacker=agent)


def effect_bomb(action, agent, battle):
    """Thrower: AoE at the target, `radius` from the table."""
    tgt = agent.target
    if tgt is None or not tgt.alive:
        return
    dmg = float(action.params.get("damage") or 0)
    radius = float(action.params.get("radius") or 0)
    for o in battle.agents:
        if o.team == agent.team or not o.alive:
            continue
        if (o.x - tgt.x) ** 2 + (o.y - tgt.y) ** 2 <= radius * radius:
            battle.hit(agent.team, o, dmg, attacker=agent)


def effect_tank(action, agent, battle):
    """Tank: timed self buff, +armor% and a flat speed penalty."""
    battle.add_buff(agent,
                    duration=float(action.params.get("duration") or 0),
                    armor_pct=float(action.params.get("armor bonus (%)") or 0),
                    speed_flat=float(action.params.get("speed nerf (abs.)") or 0))


def effect_spirit_link(action, agent, battle):
    """Monk: share a percentage of damage across nearby allies for a duration."""
    battle.add_spirit_link(
        agent,
        duration=float(action.params.get("duration") or 0),
        share_pct=float(action.params.get("share percent") or 0),
        extra_allies=int(action.params.get("additional allied units") or 0),
    )


def effect_summon(action, agent, battle):
    """Cultist / Scientist: summon a unit of the table's level next to the caster."""
    cls = SUMMONS[action.name]
    level = int(action.params.get("tentacle level") or action.params.get("tower level") or 1)
    battle.summon(agent, cls, level)


EFFECTS = {
    "attack": effect_attack,
    "Heal": effect_heal,
    "Mage": effect_magic,
    "Bomb": effect_bomb,
    "Tank": effect_tank,
    "SpiritLink": effect_spirit_link,
    "Cultist": effect_summon,
    "Scientist": effect_summon,
}


class UnsupportedClassSkill(NotImplementedError):
    pass


def build_actions(spec, skill: dict | None) -> list[Action]:
    """The unit's action list: default attack first, then its class skill.

    Priority order matches the base tree's dynamic Selector: the class skill is
    listed before the default attack so it is tried first when its gates pass,
    which is what makes cooldown-gated skills fire at all.
    """
    attack = Action(
        name="attack", tree_name="NC-DefaultAttack",
        range_world=spec.range_world,
        cooldown=spec.attack_period, mana_cost=0.0,
        effect=effect_attack, priority=0,
    )
    if not skill or skill["name"] in PASSIVE_SKILLS:
        return [attack]

    name, params = skill["name"], skill["params"]
    effect = EFFECTS.get(name)
    if effect is None:
        raise UnsupportedClassSkill(name)

    # Only Bomb carries an explicit range. For the others the real value comes
    # from M_CSGroup.get_range, which has not been extracted; self-cast skills
    # need none, and the rest fall back to the unit's own reach.
    if name in SELF_CAST:
        rng = 0.0
    elif params.get("range") is not None:
        rng = float(params["range"])
    else:
        rng = spec.range_world

    skill_action = Action(
        name=name, tree_name="NC-Skill",
        range_world=rng,
        cooldown=float(params.get("cooldown") or 0),
        mana_cost=float(params.get("mana") or params.get("manacost") or 0),
        effect=effect, params=params, priority=1,
    )
    return [skill_action, attack]
