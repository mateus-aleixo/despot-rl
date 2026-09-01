"""Per-unit skills from `Skills.json`.

`Units.json` rows carry up to eight `SkillN` ids into `Skills.json`, whose
`CSClass` names the mechanic and whose `ParamNName`/`ParamNValue` pairs give its
numbers. `Meta.ActiveSkills` and `Meta.PassiveSkills` say which are cast and
which are always on.

Every CSClass in use is registered here with an explicit kind, so coverage is a
measurable number rather than an impression:

    NOOP           real, but nothing for this sim to do (see each entry's note)
    PASSIVE        an always-on modifier or hook
    ACTIVE         cast on cooldown, driven by the generic NC-Skill tree
    UNIMPLEMENTED  declared, not implemented; the unit simply will not use it

An unregistered CSClass raises, so a new skill cannot slip in silently.
"""
from __future__ import annotations

from dataclasses import dataclass

NOOP = "noop"
PASSIVE = "passive"
ACTIVE = "active"
UNIMPLEMENTED = "unimplemented"


class UnknownSkill(NotImplementedError):
    pass


@dataclass(frozen=True)
class Handler:
    kind: str
    effect: str = ""      # for ACTIVE: which generic effect to build
    note: str = ""


# --------------------------------------------------------------------- registry

REGISTRY: dict[str, Handler] = {
    # -- nothing to do ----------------------------------------------------
    # Every unit carrying this has a nonzero Resistance stat and every unit
    # without it has zero, so the skill is the marker and the stat is the
    # value. The stat is already applied in the damage formula.
    "Resistance": Handler(NOOP, note="marker for the Resistance stat"),
    "Text": Handler(NOOP, note="UI label only"),
    "EnemyAnimationPrefix": Handler(NOOP, note="picks an animation set"),
    "DespotStartDialogue": Handler(NOOP, note="dialogue trigger"),
    "DespotMidDialogue": Handler(NOOP, note="dialogue trigger"),
    # Flying matters for walls and ground-only effects. The room grid has no
    # obstacles yet and no ground-only effects are modelled, so it is inert.
    "Flying": Handler(NOOP, note="no obstacles or ground-only effects modelled"),

    # -- passives ---------------------------------------------------------
    "SplashAround": Handler(PASSIVE, note="attacks splash for percent% in radius"),
    "Evasion": Handler(PASSIVE, note="Chance% to evade an incoming hit"),
    "StatusEvasion": Handler(PASSIVE, note="evades statuses; treated as evasion"),
    "CriticalStrike": Handler(PASSIVE, note="chance% for mult x damage"),
    "Regeneration": Handler(PASSIVE, note="+value of a stat every period"),
    "Vampirism": Handler(PASSIVE, note="heals the attacker for a share of damage"),
    "FurySwipe": Handler(PASSIVE, note="stacking attack-speed gain per hit"),
    "TransformAfterDeath": Handler(PASSIVE, note="spawns Class x amount on death"),
    "FireheadDeath": Handler(PASSIVE, note="on death, damages a random enemy"),
    "DamageOnDeath": Handler(PASSIVE, note="on death, damages nearby enemies"),
    "FirestarterDeath": Handler(PASSIVE, note="on death, damages nearby enemies"),

    # -- actives ----------------------------------------------------------
    "Clap": Handler(ACTIVE, "direct"),
    "DespotClap": Handler(ACTIVE, "direct"),
    "Charge": Handler(ACTIVE, "direct"),
    "DespotCharge": Handler(ACTIVE, "direct"),
    "Bruh": Handler(ACTIVE, "direct"),
    "ShiftingEffect": Handler(ACTIVE, "direct"),
    "ElectroWave": Handler(ACTIVE, "direct"),
    "HipThrow": Handler(ACTIVE, "direct"),
    "Blurp": Handler(ACTIVE, "aoe"),
    "FirePunch": Handler(ACTIVE, "aoe"),
    "DespotBlast": Handler(ACTIVE, "aoe"),
    "AoeBurn": Handler(ACTIVE, "aoe"),
    "Venom": Handler(ACTIVE, "direct"),
    "Swoop": Handler(ACTIVE, "direct"),
    "SkeletonHandToss": Handler(ACTIVE, "direct"),
    "ClusterBombing": Handler(ACTIVE, "aoe"),
    "Juggernaut": Handler(ACTIVE, "direct"),
    "HealBot": Handler(ACTIVE, "heal"),
    "HealChannel": Handler(ACTIVE, "heal"),
    "HealBotDrain": Handler(ACTIVE, "drain"),
    "ManaBurn": Handler(ACTIVE, "manaburn"),
    "Animate": Handler(ACTIVE, "summon"),
    "SpiderSummon": Handler(ACTIVE, "summon"),
    "LeechSummon": Handler(ACTIVE, "summon"),
    "Summon": Handler(ACTIVE, "summon"),
    "DespotClawSummon": Handler(ACTIVE, "summon"),
    "StatBonusCast": Handler(ACTIVE, "buff_self"),
    "TankWithBlock": Handler(ACTIVE, "buff_self"),
    "SciCast": Handler(ACTIVE, "buff_self"),
    "ReduceStatAround": Handler(ACTIVE, "debuff_around"),

    # -- declared, not implemented ---------------------------------------
    # Each needs a mechanic this sim does not have: displacement, control
    # effects, channels, or bespoke targeting.
    "Dash": Handler(UNIMPLEMENTED, note="displacement"),
    "BlinkAway": Handler(ACTIVE, "blink"),
    "MushroomBlink": Handler(ACTIVE, "blink"),
    "Toss": Handler(UNIMPLEMENTED, note="displacement"),
    "PassiveKnockback": Handler(PASSIVE, note="knocks the target back on hit"),
    "EventualKnockback": Handler(PASSIVE, note="knocks the target back on hit"),
    "Stun": Handler(ACTIVE, "status"),
    "Silence": Handler(ACTIVE, "status"),
    "Panic": Handler(ACTIVE, "status"),
    "Fear": Handler(ACTIVE, "status"),
    "WatcherDisable": Handler(ACTIVE, "status"),
    "Devour": Handler(UNIMPLEMENTED, note="grab and channel"),
    "CollectLeeches": Handler(UNIMPLEMENTED, note="bespoke targeting"),
    "Dodge": Handler(PASSIVE, note="negates one hit every Cooldown seconds"),
    "Punisher": Handler(UNIMPLEMENTED, note="multi-shot burst"),
    "AntiHeal": Handler(UNIMPLEMENTED, note="heal inversion"),
    "AntiAntiHeal": Handler(UNIMPLEMENTED, note="heal inversion"),
    "Feast": Handler(UNIMPLEMENTED, note="corpse consumption"),
    "Reflection": Handler(PASSIVE, note="reflects percent% of damage taken"),
    "Damaged": Handler(UNIMPLEMENTED, note="on-damaged trigger"),
    "BuffAttack": Handler(PASSIVE, note="on-attack timed stat status"),
    "BuffOnCasted": Handler(PASSIVE, note="timed self stat status on every cast"),
    "BuffOnForeignDeath": Handler(UNIMPLEMENTED, note="on-ally-death buff"),
    "LoveOfPain": Handler(UNIMPLEMENTED, note="on-damaged buff"),
    "Craggy": Handler(PASSIVE, note="on-damaged stun or silence on the attacker"),
    "Compensation": Handler(PASSIVE, note="stat x allies below a health threshold"),
    "HunterMark": Handler(UNIMPLEMENTED, note="target marking"),
    "Splash": Handler(PASSIVE, note="projectile splashes for percent% in radius"),
    "ExplodeProjectile": Handler(PASSIVE, note="projectile splashes on impact"),
    "ExplodeOnDeath": Handler(UNIMPLEMENTED, note="death explosion with a projectile"),
    "ResurrectionChance": Handler(PASSIVE, note="chance to revive at healthValue"),
    "PassiveBurn": Handler(UNIMPLEMENTED, note="damage over time"),
    "DespotMassSummon": Handler(UNIMPLEMENTED, note="styled mass summon"),
    "SummonSimple": Handler(UNIMPLEMENTED, note="bespoke summon"),
    "OgriMagi": Handler(UNIMPLEMENTED, note="bespoke animation-driven attack"),
    "OctopusCast": Handler(UNIMPLEMENTED, note="bespoke buff/debuff cast"),
    "SciTowerManaAura": Handler(UNIMPLEMENTED, note="mana aura"),
}


def handler_for(cs_class: str) -> Handler:
    h = REGISTRY.get(cs_class)
    if h is None:
        raise UnknownSkill(cs_class)
    return h


def unit_skill_rows(tables: dict, unit_row: dict) -> list[dict]:
    """The Skills.json rows a unit references through Skill1..Skill8."""
    from .data import skills_by_id
    by_id = skills_by_id(tables)
    ids = [int(unit_row[f"Skill{i}"]) for i in range(1, 9) if unit_row.get(f"Skill{i}")]
    return [by_id[i] for i in ids if i in by_id]


def param(params: dict, *names, default=0.0) -> float:
    """Skills.json is inconsistent about capitalisation (damage vs Damage)."""
    for n in names:
        if n in params and params[n] is not None:
            v = params[n]
            if isinstance(v, str):
                if v.lower() in ("true", "false"):
                    return 1.0 if v.lower() == "true" else 0.0
                try:
                    return float(v)
                except ValueError:
                    return default
            return float(v)
    return default


def str_param(params: dict, *names, default=None):
    for n in names:
        if params.get(n) is not None:
            return params[n]
    return default


def evasion_entry(params: dict) -> tuple[float, float, bool]:
    """One `M_EvasionSkill` row: chance, healthThreshold and jumpBack.

    Both extra properties carry `[DefaultValueAttribute]` and most rows omit
    them. `OnHealthChanged` computes `_active = healthThreshold >= health%`,
    so the default has to be at or above 100 for a plain evasion to work at
    full health; `evasion_health_threshold_default` says so and why.
    """
    from .assumptions import DEFAULT
    return (param(params, "chance", "Chance"),
            param(params, "healthThreshold", "HealthThreshold",
                  default=DEFAULT.evasion_health_threshold_default),
            str(str_param(params, "jumpBack", "JumpBack", default="")).lower() == "true")


def vampirism_entry(params: dict) -> tuple[float, bool]:
    """One `M_VampirismSkill` row: `value`, and whether it is a percentage.

    Every row in `Skills.json` and `Mutations.json` names the number `value`,
    never `percent`: reading it as `percent` is how this fired nowhere at all.
    Only the FixedVampirism rows carry `percentage: false`, so the default is
    a percentage share (`vampirism_percentage_default`).
    """
    from .assumptions import DEFAULT
    pct = str_param(params, "percentage", "Percentage")
    return (param(params, "value", "Value", default=0.0),
            DEFAULT.vampirism_percentage_default if pct is None
            else str(pct).lower() == "true")


def explode_entry(params: dict) -> tuple[float, float, float, int]:
    """One `M_ExplodeProjectileSkill` row: damage, radius, chance and the
    damageType gate. Its parent `M_ExplodeSkill` names the number `damage` and
    it is absolute, not a share of the hit -- `ExplodeAddon.OnApply` passes
    `get_damage()` straight to `M_Damage.Get`, with nothing multiplied by the
    triggering hit. Reading it as `percent` gave every row 100."""
    from .mutations import _mask
    return (param(params, "damage", "Damage", default=0.0),
            param(params, "radius", "Radius", default=0.0),
            param(params, "chance", "Chance", default=100.0),
            _mask(params))


def knockback_entry(params: dict) -> tuple[float, float, float, float, int]:
    """One `KnockbackDamageAddon`: chance, radius, speed and acceleration.

    `M_EventualKnockbackSkill` names them `speed` and `acceleration`; the older
    `PassiveKnockback` row spells the same two `knockbackSpeed` and
    `knockbackAcceleration`, so both spellings are read. The defaults were what
    every row silently fell back to while only the second pair was looked up.
    `knockbackType` (Push against Fly) is not modelled, see
    `knockback_type_is_uniform`.
    """
    from .mutations import _mask
    return (param(params, "chance", "Chance", default=100.0),
            param(params, "radius", "Radius", default=0.0),
            param(params, "speed", "Speed", "knockbackSpeed", default=180.0),
            param(params, "acceleration", "Acceleration",
                  "knockbackAcceleration", default=-350.0),
            _mask(params))


def coverage(tables: dict) -> dict:
    """How much of what units actually reference is implemented, by uses."""
    from .data import skills_by_id
    by_id = skills_by_id(tables)
    counts: dict[str, int] = {}
    unknown: set[str] = set()
    for u in tables["Units"]:
        for i in range(1, 9):
            sid = u.get(f"Skill{i}")
            if not sid:
                continue
            row = by_id.get(int(sid))
            if row is None:
                continue
            cs = row.get("CSClass")
            try:
                kind = handler_for(cs).kind
            except UnknownSkill:
                unknown.add(cs)
                kind = "UNREGISTERED"
            counts[kind] = counts.get(kind, 0) + 1
    return {"by_kind": counts, "unregistered": sorted(unknown)}


# ------------------------------------------------------------------- effects
# Generic active effects. Skills.json is inconsistent about capitalisation, so
# every lookup accepts both spellings.

def _targets_in_radius(battle, x, y, radius, team, enemies=True):
    out = []
    r2 = radius * radius
    for o in battle.agents:
        if not o.alive:
            continue
        if (o.team != team) != enemies:
            continue
        if (o.x - x) ** 2 + (o.y - y) ** 2 <= r2:
            out.append(o)
    return out


def _magical(params) -> bool:
    return str_param(params, "damageType", "DamageType", default="") == "Magical"


def make_direct(params):
    dmg = param(params, "damage", "Damage")
    magical = _magical(params)

    def effect(action, agent, battle):
        tgt = agent.target
        if tgt is not None and tgt.alive and dmg:
            battle.hit(agent.team, tgt, dmg, magical=magical, attacker=agent)
    return effect


def make_aoe(params):
    dmg = param(params, "damage", "Damage")
    radius = param(params, "radius", "Radius")
    magical = _magical(params)

    def effect(action, agent, battle):
        tgt = agent.target
        if tgt is None or not tgt.alive or not dmg:
            return
        for o in _targets_in_radius(battle, tgt.x, tgt.y, radius, agent.team):
            battle.hit(agent.team, o, dmg, magical=magical, attacker=agent)
    return effect


def make_heal(params):
    amount = param(params, "amount", "Amount", "healPerSecond")
    radius = param(params, "radius", "Radius", default=0.0)

    def effect(action, agent, battle):
        pool = (_targets_in_radius(battle, agent.x, agent.y, radius, agent.team, enemies=False)
                if radius else [agent])
        best, worst = None, 0.0
        for o in pool:
            missing = o.max_hp - o.hp
            if missing > worst:
                best, worst = o, missing
        if best is not None:
            battle.heal(best, amount, agent.team)
    return effect


def make_drain(params):
    dmg = param(params, "damage", "Damage")
    magical = _magical(params)

    def effect(action, agent, battle):
        tgt = agent.target
        if tgt is None or not tgt.alive:
            return
        before = tgt.hp
        battle.hit(agent.team, tgt, dmg, magical=magical, attacker=agent)
        drained = max(0.0, before - tgt.hp)
        battle.heal(agent, drained, agent.team)
    return effect


def make_manaburn(params):
    amount = param(params, "amount", "Amount")
    radius = param(params, "radius", "Radius")

    def effect(action, agent, battle):
        for o in _targets_in_radius(battle, agent.x, agent.y, radius, agent.team):
            o.mana = max(0.0, o.mana - amount)
    return effect


def make_summon(params):
    cls = str_param(params, "class", "Class", "summonClass")
    level = int(param(params, "level", "Level", default=1) or 1)
    amount = int(param(params, "amount", "Amount", default=1) or 1)

    def effect(action, agent, battle):
        if not cls:
            return
        for _ in range(amount):
            battle.summon(agent, cls, level)
    return effect


def make_buff_self(params):
    duration = param(params, "duration", "Duration")
    stat = str_param(params, "stat", "Stat", default="")
    value = param(params, "value", "Value")
    percentage = param(params, "percentage", default=1.0) >= 1.0
    armor_bonus = param(params, "armorBonus", "ArmorBuff")
    speed_bonus = param(params, "speedBonus")

    def effect(action, agent, battle):
        battle.add_buff(
            agent, duration=duration,
            armor_pct=armor_bonus,
            speed_flat=speed_bonus,
            attack_speed_pct=value if (stat == "AttackSpeed" and percentage) else 0.0,
        )
    return effect


def make_debuff_around(params):
    duration = param(params, "duration", "Duration")
    amount = param(params, "amount", "Amount")
    stat = str_param(params, "stat", "Stat", default="")
    percentage = param(params, "percentage", "Percentage", default=0.0) >= 1.0
    radius = param(params, "radius", "Radius", default=60.0)

    def effect(action, agent, battle):
        for o in _targets_in_radius(battle, agent.x, agent.y, radius, agent.team):
            if stat == "Armor":
                battle.add_buff(o, duration=duration,
                                armor_pct=-amount if percentage else 0.0,
                                armor_flat=0.0 if percentage else -amount)
            elif stat == "AttackSpeed":
                battle.add_buff(o, duration=duration, attack_speed_pct=-amount)
    return effect


EFFECT_BUILDERS = {
    "direct": make_direct,
    "aoe": make_aoe,
    "heal": make_heal,
    "drain": make_drain,
    "manaburn": make_manaburn,
    "summon": make_summon,
    "buff_self": make_buff_self,
    "debuff_around": make_debuff_around,
}


# --------------------------------------------------------- applying to a unit

def apply_unit_skills(agent, tables: dict) -> list:
    """Attach a unit's Skills.json entries. Returns extra Actions to run.

    Passives become fields on the agent; actives become Actions driven by the
    generic NC-Skill tree, exactly like class skills.
    """
    from .actions import Action
    from .data import skills_by_id

    by_id = skills_by_id(tables)
    extra: list = []
    for prio, sid in enumerate(agent.spec.skills):
        row = by_id.get(sid)
        if row is None:
            continue
        cs = row.get("CSClass")
        h = handler_for(cs)
        p = {}
        for i in range(1, 11):
            n = row.get(f"Param{i}Name")
            if n:
                p[n] = row.get(f"Param{i}Value")

        if h.kind in (NOOP, UNIMPLEMENTED):
            continue

        if h.kind == PASSIVE:
            if cs == "SplashAround":
                agent.splash = (param(p, "percent", "Percent", default=100.0),
                                param(p, "radius", "Radius", default=0.0))
            elif cs in ("Evasion", "StatusEvasion"):
                agent.evasions.append(evasion_entry(p))
            elif cs == "CriticalStrike":
                agent.crit_chance = max(agent.crit_chance, param(p, "chance", "Chance"))
                agent.crit_mult = max(agent.crit_mult, param(p, "mult", "value", default=1.0))
            elif cs == "Regeneration":
                agent.regen = (str_param(p, "stat", default="Mana"),
                               param(p, "period", default=10.0),
                               param(p, "value", default=0.0))
            elif cs == "Vampirism":
                agent.vampirisms.append(vampirism_entry(p))
            elif cs == "FurySwipe":
                agent.fury_per_stack = param(p, "amount", default=0.0)
            elif cs == "Reflection":
                agent.reflect_pct = param(p, "percent", "Percent")
            elif cs == "Dodge":
                agent.unit_dodge_cooldown = param(p, "Cooldown", "cooldown")
            elif cs == "Splash":
                agent.projectile_splash = (
                    param(p, "percent", "Percent", default=100.0),
                    param(p, "radius", "Radius", default=0.0))
            elif cs == "ExplodeProjectile":
                agent.explodes.append(explode_entry(p))
            elif cs in ("PassiveKnockback", "EventualKnockback"):
                agent.knockbacks.append(knockback_entry(p))
            elif cs in ("TransformAfterDeath", "FireheadDeath", "DamageOnDeath",
                        "FirestarterDeath"):
                agent.on_death.append((cs, p))
            else:
                # The hook passives: the same mechanics the mutation table
                # names, reached here when a unit carries one as a skill.
                from .mutations import ON_ATTACK, ON_CAST, ON_DAMAGED, STANDING
                if cs in ON_ATTACK:
                    agent.on_attack.append((cs, p))
                elif cs in ON_DAMAGED:
                    agent.on_damaged.append((cs, p))
                elif cs in ON_CAST:
                    agent.on_cast.append((cs, p))
                elif cs in STANDING:
                    agent.standing.append((cs, p))
                elif cs == "ResurrectionChance":
                    agent.resurrect = (
                        param(p, "chance", "Chance"),
                        param(p, "healthValue", "HealthValue"),
                        str(p.get("percentage", "false")).lower() == "true")
            continue

        # ACTIVE
        builder = EFFECT_BUILDERS.get(h.effect)
        if builder is None:
            continue
        eff = builder(p, cs) if h.effect == "status" else builder(p)
        rng = param(p, "range", "Range", default=0.0) or agent.spec.range_world
        extra.append(Action(
            name=cs, tree_name="NC-Skill",
            range_world=rng,
            cooldown=param(p, "cooldown", "Cooldown", default=0.0),
            mana_cost=param(p, "manacost", "ManaCost", default=0.0),
            effect=eff, params=p, priority=2 + prio,
        ))
    return extra


def run_death_effects(agent, battle) -> None:
    """Fire a unit's on-death skills once."""
    for cs, p in agent.on_death:
        if cs == "TransformAfterDeath":
            cls = str_param(p, "Class", "class")
            amount = int(param(p, "amount", "Amount", default=1) or 1)
            if cls:
                for _ in range(amount):
                    battle.summon(agent, cls, int(param(p, "level", default=1) or 1))
        elif cs in ("FireheadDeath", "DamageOnDeath", "FirestarterDeath"):
            dmg = param(p, "value", "Value", "damage", "Damage")
            radius = param(p, "radius", "Radius", default=0.0)
            if not dmg:
                continue
            if radius:
                for o in _targets_in_radius(battle, agent.x, agent.y, radius, agent.team):
                    battle.hit(agent.team, o, dmg)
            else:
                foes = [o for o in battle.agents if o.team != agent.team and o.alive]
                if foes:
                    battle.hit(agent.team, battle.rng.choice(foes), dmg)


# Status names each CSClass applies, and the field its duration comes from.
STATUS_OF = {
    "Stun": "stun",
    "Silence": "silence",
    "Panic": "panic",
    "Fear": "panic",
    "WatcherDisable": "silence",
}


def make_status(params, cs_class=""):
    duration = param(params, "duration", "Duration")
    radius = param(params, "radius", "Radius", default=0.0)
    dmg = param(params, "damage", "Damage")
    status = STATUS_OF.get(cs_class, "stun")
    magical = _magical(params)

    def effect(action, agent, battle):
        if radius:
            victims = _targets_in_radius(battle, agent.x, agent.y, radius, agent.team)
        else:
            victims = [agent.target] if agent.target is not None and agent.target.alive else []
        for o in victims:
            o.apply_status(status, duration)
            if dmg:
                battle.hit(agent.team, o, dmg, magical=magical, attacker=agent)
    return effect


def make_blink(params):
    """Teleport away from the current target (range blinkers, mushrooms)."""
    radius = param(params, "radius", "Radius", "Radius", default=25.0)
    duration = param(params, "duration", "Duration", default=0.0)

    def effect(action, agent, battle):
        battle.blink(agent, agent.target, radius)
        if duration:
            agent.apply_status("blinked", duration)
    return effect


EFFECT_BUILDERS["status"] = make_status
EFFECT_BUILDERS["blink"] = make_blink
