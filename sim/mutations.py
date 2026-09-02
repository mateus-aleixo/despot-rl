"""Mutations: the run-level upgrades applied to a squad.

`Mutations.json` holds the definitions (ID, Name, Class, Param1..8 pairs);
`MutationGrid.json` holds the shop grid, the always-on `SimpleMutations` and the
`CombinedMutations`; `MutationsByLevel.json` says which ids are offered at each
level.

`Name` reuses the same effect vocabulary as `Skills.json`'s `CSClass`, so 23 of
the names route straight to the handlers in `unit_skills`. The rest are
mutation-only, dominated by `StatBonus`.

`Class` selects who a mutation applies to: "All", "Random", a single class, or a
comma-separated list. Player humans carry their item's class (Warrior, Medic,
...), which is what these names refer to.

As with skills, every Name in use is registered with an explicit kind so
coverage is measurable, and an unregistered one raises.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any

from .unit_skills import NOOP, PASSIVE, ACTIVE, UNIMPLEMENTED, Handler, param, str_param

# Stats StatBonus can name, mapped onto UnitSpec fields.
STAT_FIELDS = {
    "Damage": "damage",
    "Health": "health",
    "AttackSpeed": "attack_speed",
    "Armor": "armor",
    "Resistance": "resistance",
    "Speed": "speed",
    "Mana": "mana",
    "Range": "range_world",
}


class UnknownMutation(NotImplementedError):
    pass


# Mutation-only names, i.e. those with no matching skill CSClass.
MUTATION_REGISTRY: dict[str, Handler] = {
    "StatBonus": Handler(PASSIVE, note="flat or percentage change to one stat"),
    "BuffAttack": Handler(PASSIVE, note="on-attack timed stat status"),
    "BuffOnCasted": Handler(PASSIVE, note="timed self stat status on every cast"),
    "Craggy": Handler(PASSIVE, note="on-damaged stun or silence on the attacker"),
    "Compensation": Handler(PASSIVE, note="stat x allies below a health threshold"),
    "ModifyDamage": Handler(PASSIVE, note="adds to damage of one damage type"),
    "ResurrectionChance": Handler(PASSIVE, note="chance to revive at healthValue"),
    "OraStatBonus": Handler(PASSIVE, note="stat bonus variant"),
    "CSLinkStatBonus": Handler(PASSIVE, note="targetClass allies buffed by the Monk cast"),
    "ClassSkill": Handler(UNIMPLEMENTED, note="grants or upgrades a class skill"),
    "ClassDiversity": Handler(PASSIVE, note="one unit gains stat x count(otherClass)"),
    "ClassDiversityMagicalDamage": Handler(PASSIVE, note="magical damage x count(otherClass)"),
    "SwapAttack": Handler(UNIMPLEMENTED, note="replaces the default attack"),
    "BeforeFightSummon": Handler(UNIMPLEMENTED, note="summons before the fight starts"),
    "FearsomeAttack": Handler(PASSIVE, note="on-attack fear"),
    "ChanceIncrease": Handler(UNIMPLEMENTED, note="raises another mutation's chance"),
    "PassiveStun": Handler(PASSIVE, note="on-attack stun"),
    "UnderFeed": Handler(UNIMPLEMENTED, note="food economy"),
    "Intelligence": Handler(UNIMPLEMENTED, note="mana economy"),
    "ManaBreak": Handler(PASSIVE, note="mana burn on hit"),
    "RailProjectile": Handler(UNIMPLEMENTED, note="piercing projectile"),
    "RubberProjectile": Handler(UNIMPLEMENTED, note="bouncing projectile"),
    # Not a fight effect, so it is still NOOP to the battle layer, but it is not
    # nothing: `C_BossVisionMutation.OnNewLevel` reveals the boss room, and the
    # run layer applies it in `RunState.next_level` via `RoomMap.reveal_boss`.
    "BossVision": Handler(NOOP, note="run layer: reveals the boss room on a new level"),
    "Afk": Handler(NOOP, note="idle-progress meta, outside a fight"),
    "SuperPunch": Handler(UNIMPLEMENTED, note="bespoke attack replacement"),
    "CooldownReduction": Handler(PASSIVE, note="scales every action's cooldown"),
    "CastSpeed": Handler(UNIMPLEMENTED, note="scales cast animation speed"),
    "HungerFury": Handler(UNIMPLEMENTED, note="food economy"),
    "HoboFury": Handler(UNIMPLEMENTED, note="gold economy"),
    "CSRange": Handler(UNIMPLEMENTED, note="class-skill range"),
    "ChainLightning": Handler(UNIMPLEMENTED, note="chaining projectile"),
    "PlagueAntiHeal": Handler(UNIMPLEMENTED, note="swaps the Medic class skill"),
    "Untouchable": Handler(PASSIVE, note="on-damaged stat debuff on the attacker"),
    "Purification": Handler(UNIMPLEMENTED, note="status cleanse"),
    "Sustain": Handler(UNIMPLEMENTED, note="conditional healing"),
    "Revival": Handler(UNIMPLEMENTED, note="resurrection"),
    "DeferredDamage": Handler(UNIMPLEMENTED, note="delayed damage pool"),
    "RandomDamage": Handler(UNIMPLEMENTED, note="damage variance"),
    "MedicRetreat": Handler(UNIMPLEMENTED, note="repositioning"),
    "Underdog": Handler(UNIMPLEMENTED, note="scales with squad deficit"),
    "Shield": Handler(UNIMPLEMENTED, note="absorb pool"),
    "Tank": Handler(UNIMPLEMENTED, note="mutation form of the Tank skill"),
    "Fireball": Handler(UNIMPLEMENTED, note="bespoke projectile"),
    "MultiCast": Handler(PASSIVE, note="chance to repeat a cast, refunding its cost"),
    "StickyBlood": Handler(PASSIVE, note="on death, slows enemies in radius"),
    "CasualitiesEating": Handler(UNIMPLEMENTED, note="corpse consumption"),
    "GetItemBack": Handler(NOOP, note="shop economy, outside a fight"),
    "AdditionalRerolls": Handler(NOOP, note="shop economy, outside a fight"),
    "RoomCount": Handler(NOOP, note="level generation, outside a fight"),
    "GiveGold": Handler(NOOP, note="economy, outside a fight"),
    "ItemDiscount": Handler(NOOP, note="shop economy, outside a fight"),
    "CheapReroll": Handler(NOOP, note="shop economy, outside a fight"),
    # Class-skill variants and auras: each modifies a specific class skill or
    # adds an aura, which needs per-skill plumbing this sim does not have.
    "CSBoostAura": Handler(UNIMPLEMENTED, note="class-skill aura"),
    "CSDodgeDash": Handler(UNIMPLEMENTED, note="class-skill variant"),
    "CSDoubleAura": Handler(UNIMPLEMENTED, note="class-skill aura"),
    "CSHealMana": Handler(UNIMPLEMENTED, note="class-skill variant"),
    "CSHealPurify": Handler(UNIMPLEMENTED, note="class-skill variant"),
    "CSLinkDamage": Handler(UNIMPLEMENTED, note="class-skill variant"),
    "CSTankTauntedStat": Handler(UNIMPLEMENTED, note="class-skill variant"),
    "CSTauntTime": Handler(UNIMPLEMENTED, note="class-skill variant"),
    "ImprovedSciCast": Handler(UNIMPLEMENTED, note="class-skill variant"),
    "SeparateSpiritLinker": Handler(UNIMPLEMENTED, note="class-skill variant"),
    "DodgeAura": Handler(UNIMPLEMENTED, note="aura"),
    # Taunt and threat: no threat model in this sim.
    "MasterTaunt": Handler(UNIMPLEMENTED, note="threat/taunt"),
    "ManaBurnTaunt": Handler(UNIMPLEMENTED, note="threat/taunt"),
    "FearTentacle": Handler(UNIMPLEMENTED, note="threat/taunt"),
    # Triggers and conditional effects.
    "BuffOnDeath": Handler(PASSIVE, note="on death, buffs the surviving team"),
    # Verified as an on-death buff to allies in radius, but its *stat* lives on
    # the status prefab rather than in the shipped data, so there is no number
    # to implement it from.
    "Revenge": Handler(UNIMPLEMENTED, note="on-death ally buff, stat not in the data"),
    "Rage": Handler(UNIMPLEMENTED, note="conditional buff"),
    "Overheal": Handler(UNIMPLEMENTED, note="excess healing becomes a pool"),
    "PassiveBlock": Handler(UNIMPLEMENTED, note="conditional mitigation"),
    "BlindDefense": Handler(UNIMPLEMENTED, note="conditional mitigation"),
    "Impostor": Handler(UNIMPLEMENTED, note="bespoke"),
    "MineSweeper": Handler(UNIMPLEMENTED, note="bespoke"),
    "HumanBlinkAway": Handler(UNIMPLEMENTED, note="displacement"),
    "BlindingLightning": Handler(UNIMPLEMENTED, note="bespoke projectile"),
    "LightningChance": Handler(UNIMPLEMENTED, note="bespoke projectile"),
    "FoodPackHeal": Handler(UNIMPLEMENTED, note="food economy, outside a fight"),
    "Combined": Handler(UNIMPLEMENTED, note="composite of child mutations"),
}


def handler_for(name: str) -> Handler:
    """Skill handlers first, since the vocabularies overlap."""
    from .unit_skills import REGISTRY as SKILLS
    h = MUTATION_REGISTRY.get(name) or SKILLS.get(name)
    if h is None:
        raise UnknownMutation(name)
    return h


def mutation_params(m: dict) -> dict[str, Any]:
    return {m[f"Param{i}Name"]: m[f"Param{i}Value"]
            for i in range(1, 9) if m.get(f"Param{i}Name")}


def by_id(tables: dict) -> dict[int, dict]:
    return {m["ID"]: m for m in tables["SimpleMutations"]}


def offered_at_level(tables: dict, level: int) -> list[dict]:
    """Mutation definitions offered at a given run level."""
    ids = {r["Mutation"] for r in tables["MutationsByLevel"] if r.get("Level") == level}
    defs = by_id(tables)
    return [defs[i] for i in sorted(ids) if i in defs]


def applies_to(mutation: dict, cls: str, rng: random.Random | None = None,
               random_class: str | None = None) -> bool:
    """Whether a mutation's `Class` selector covers a unit class."""
    spec = mutation.get("Class")
    if not spec or spec == "All":
        return True
    if spec == "Random":
        # "Random" picks one class for the run; the caller supplies which.
        return random_class is None or cls == random_class
    return cls in {c.strip() for c in spec.split(",")}


# ---------------------------------------------------------------- application

def is_timed(params: dict) -> bool:
    """A StatBonus carrying a duration is a timed cast, not a permanent change.

    Only one shipped definition is like this (ID 222, the Monk attack-speed
    cast); the other 19 are permanent and use `bonus` rather than `value`.
    """
    return any(k in params for k in ("duration", "Duration"))


def apply_stat_bonus(spec, params: dict):
    """Return a UnitSpec with one stat changed. Specs are frozen, so replace."""
    stat = str_param(params, "stat", "Stat", default="")
    field = STAT_FIELDS.get(stat)
    if field is None:
        return spec
    bonus = param(params, "bonus", "Bonus", "value", "Value")
    percentage = str(params.get("percentage", params.get("Percentage", "false"))).lower() == "true"
    current = getattr(spec, field)
    new = current * (1.0 + bonus / 100.0) if percentage else current + bonus
    if field in ("health", "damage", "attack_speed", "speed", "mana", "range_world"):
        new = max(0.0, new)
    elif field == "resistance":
        new = min(1.0, max(0.0, new))
    return dataclasses.replace(spec, **{field: new})


def apply_to_specs(specs: list, mutations: list[dict],
                   rng: random.Random | None = None) -> list:
    """Apply the stat-changing mutations to a squad's specs.

    Returns new specs; the originals are untouched. Everything that is a hook
    rather than a standing number is handled at agent level by
    `apply_to_agents`.

    Two shapes land here. `StatBonus`/`OraStatBonus` change one stat on
    everyone the `Class` selector covers. `ClassDiversity` changes one stat on
    a single randomly chosen member of its class, by `bonus x count(otherClass)`
    -- the game applies it at fight start and re-picks when that unit dies, so
    reading it as a standing pre-fight change is the approximation here.
    """
    rng = rng or random.Random(0)
    classes = sorted({s.cls for s in specs})
    out = list(specs)
    for m in mutations:
        name = m["Name"]
        if handler_for(name).kind != PASSIVE:
            continue
        p = mutation_params(m)
        if name in ("StatBonus", "OraStatBonus"):
            if is_timed(p):
                continue      # handled as a cast in apply_to_agents
            random_class = (rng.choice(classes)
                            if m.get("Class") == "Random" and classes else None)
            out = [apply_stat_bonus(s, p) if applies_to(m, s.cls, rng, random_class) else s
                   for s in out]
        elif name == "ClassDiversity":
            i = diversity_target(out, m, rng)
            bonus = class_diversity_bonus(out, p)
            if i >= 0 and bonus:
                out[i] = apply_stat_bonus(out[i], dict(p, bonus=bonus))
    return out


# The stats a mutation can move that this sim resolves into a spec, and what a
# reasonable size for each one is, so a caller can normalise them against each
# other. Everything else a mutation might do lands in `apply_to_agents` or in
# `UNIMPLEMENTED`, and shows up here as a zero.
EFFECT_STATS = ("damage", "health", "armor", "attack_speed", "range",
                "speed", "resistance")


def effect_on(specs: list, mutation: dict,
              rng: random.Random | None = None) -> dict:
    """What one mutation would do to this squad, stat by stat.

    Returns the **relative** change in each stat, summed over the squad -- the
    squad's total Damage rises 8%, say -- plus the share of the squad it
    touches and whether the sim models it at all.

    Not a Power delta. `UnitSpec.power` is the game's own strength statistic and
    it is a *stored* number, `Units.Power` plus the item's, so a mutation that
    raises Damage by 15% moves no Power at all and a Power delta would read zero
    for every mutation in the table. What a mutation changes is the stats, so
    that is what this reports.

    The stat deltas only ever fire for `StatBonus` and `OraStatBonus`, which is
    32 of the 1,094 offers across the twelve levels. The other numbers are what
    carries the rest: `kind` is the handler's, and 823 of those 1,094 offers are
    `unimplemented` -- this sim does nothing at all with three quarters of what a
    mutation shop shows, so `implemented` is the honest first thing to tell a
    policy about a slot.
    """
    rng = rng or random.Random(0)
    out = {"kind": UNIMPLEMENTED, "implemented": False, "stats": False,
           "passive": False, "fraction": 0.0,
           **{s: 0.0 for s in EFFECT_STATS}}
    if not specs:
        return out
    try:
        handler = handler_for(mutation["Name"])
    except UnknownMutation:
        return out
    out["kind"] = handler.kind
    out["implemented"] = handler.kind in (PASSIVE, ACTIVE)
    out["passive"] = (out["implemented"]
                      and mutation["Name"] in PASSIVE_ATTACHING)

    classes = sorted({s.cls for s in specs})
    random_class = (rng.choice(classes)
                    if mutation.get("Class") == "Random" and classes else None)
    touched = [applies_to(mutation, s.cls, rng, random_class) for s in specs]
    out["fraction"] = sum(touched) / len(specs)

    params = mutation_params(mutation)
    if (handler.kind != PASSIVE
            or mutation["Name"] not in ("StatBonus", "OraStatBonus")
            or is_timed(params)):
        # Not a standing stat bonus: a cast, an agent-level passive, or nothing
        # this sim implements. `kind` and `fraction` are all there is to say.
        return out
    after = [apply_stat_bonus(s, params) if hit else s
             for s, hit in zip(specs, touched)]
    out["stats"] = True
    for stat in EFFECT_STATS:
        before_total = sum(getattr(s, stat, 0.0) or 0.0 for s in specs)
        after_total = sum(getattr(s, stat, 0.0) or 0.0 for s in after)
        if before_total:
            out[stat] = (after_total - before_total) / before_total
        elif after_total:
            out[stat] = 1.0
    return out


# --------------------------------------------------- agent-level passives
#
# The game hangs each of these off a named hook, and the hooks are what this
# section reproduces. Read out of `GameAssembly.dll`:
#
#   C_PassiveSkill.OnDamageCreated      fires while this unit's damage is being
#                                       built, so before it lands
#   C_DamageReactionSkill.OnDamageInternal
#                                       subscribes to the victim's
#                                       C_Unit.OnDamage and re-fires OnDamage
#                                       with the *attacker* as its argument
#   C_OnSkillCastedSkill.OnSkillCasted  fires on a skill cast, not on a swing
#   C_BaseOnDeathSkill.OnDeath          fires once, on this unit's death
#
# `C_PassiveSkillMutation<TMutation, TSkill>` is the game's own name for the
# category: a mutation that installs one of these on every unit it covers.
#
# Both damage hooks share one gate, verified in OnDamageCreated and again in
# OnDamageInternal:
#
#   (damage.type & skill.damageType) == skill.damageType   a subset test, so an
#                                       absent damageType matches everything
#   damage.type & Secondary            -> never triggers (splash and knock-on)
#   the other unit is alive, and for the reaction hook, an enemy
#   PseudoRandom.Get(chance)           -> the roll; an absent chance is always

# Which hook each Name hangs off. `apply_to_agents` files the mutation onto the
# matching per-agent list, and the hook reads it back.
ON_ATTACK = ("BuffAttack", "FearsomeAttack", "PassiveStun", "ManaBreak")
ON_DAMAGED = ("Craggy", "Untouchable")
ON_CAST = ("BuffOnCasted", "MultiCast")
ON_DEATH = ("BuffOnDeath", "StickyBlood")
STANDING = ("Compensation",)

# Every mutation that lands as an agent-level passive rather than a stat. The
# first line is handled in `apply_to_agents` directly; the rest resolve through
# the unit-skill registry, which `handler_for` falls back to, and are listed
# here so "passives off" means all of them and not most of them.
AGENT_PASSIVES = ("ClassDiversity", "ClassDiversityMagicalDamage",
                  "CSLinkStatBonus", "ModifyDamage", "ResurrectionChance",
                  "Evasion", "StatusEvasion", "Vampirism", "CriticalStrike",
                  "FurySwipe", "ExplodeProjectile", "EventualKnockback",
                  "CooldownReduction")

# The union of everything `disable_agent_passives` strips, which is the same
# thing as everything the shelf is worth: with these inert the shelf measures
# +0.00 levels over 2,000 paired seeds against +0.13 with them live. `effect_on`
# reports membership because `implemented` does not separate it -- 507 of the
# 1,094 offers attach one of these, and another 96 are implemented and attach
# nothing, so a policy told only "this slot is modelled" cannot tell the two
# apart and 96 of the offers it rates are worth nothing.
PASSIVE_ATTACHING = frozenset(
    ON_ATTACK + ON_DAMAGED + ON_CAST + ON_DEATH + STANDING + AGENT_PASSIVES)


def disable_agent_passives() -> None:
    """Put every hook back to `unimplemented`, leaving the offers untouched.

    The shadowing is the point: `handler_for` reads `MUTATION_REGISTRY` before
    it falls back to the unit-skill registry, so a name that only exists in the
    skill registry has to be *inserted* here rather than rewritten in place. An
    earlier version guarded on `name in MUTATION_REGISTRY` and so left evasion,
    vampirism, crit, fury swipe and the two projectile shapes live in the arm
    that was supposed to be the control. Class skills are untouched either way:
    units resolve theirs through `unit_skills.handler_for`, which never looks
    here, so this strips what a *mutation* grants and nothing else.
    """
    from .unit_skills import REGISTRY as SKILLS
    for name in ON_ATTACK + ON_DAMAGED + ON_CAST + ON_DEATH + STANDING + AGENT_PASSIVES:
        handler = MUTATION_REGISTRY.get(name) or SKILLS.get(name)
        if handler is not None:
            MUTATION_REGISTRY[name] = Handler(UNIMPLEMENTED, note=handler.note)


def _mask(params: dict) -> int:
    from .battle import damage_type_mask
    return damage_type_mask(str_param(params, "damageType", "DamageType"))


def _is_pct(params: dict, *names) -> bool:
    for n in names:
        v = params.get(n)
        if v is not None:
            return str(v).lower() == "true"
    return False


# Where each mechanic keeps its roll. Everything implements `IProbable`, but
# `FearsomeAttack` names the property `percent` and its explicit
# `IProbable.get_chance` is a straight `jmp` to `get_percent`, so the two are
# the same number under different names.
CHANCE_PARAM = {"FearsomeAttack": ("percent", "Percent")}


def chance_of(name: str, params: dict) -> float:
    """The roll a passive gates on. Absent means it always fires."""
    return param(params, *CHANCE_PARAM.get(name, ("chance", "Chance")), default=100.0)


def _gate(battle, params: dict, dtype: int, name: str = "") -> bool:
    """The shared damage gate, plus the chance roll."""
    from .battle import DT_SECONDARY
    mask = _mask(params)
    if mask and (dtype & mask) != mask:
        return False
    if dtype & DT_SECONDARY:
        return False
    chance = chance_of(name, params)
    return chance >= 100.0 or battle.rng.random() * 100.0 < chance


def status_stats(params: dict, numbered: bool = False) -> tuple:
    """The (stat, amount, percentage) triples a status carries.

    `BuffAttack` numbers its pairs -- stat1/value1/percentage1, stat2/... --
    while every other status uses a single unnumbered set. `StickyBlood` names
    several stats at once in one comma-separated `stats`.
    """
    from .battle import STAT_TO_BUFF
    out = []
    if numbered:
        for i in ("1", "2", "3"):
            field = STAT_TO_BUFF.get(str_param(params, "stat" + i, "Stat" + i))
            if field is None:
                continue
            out.append((field, param(params, "value" + i, "Value" + i),
                        _is_pct(params, "percentage" + i, "percentage")))
        if out:
            return tuple(out)
    names = str_param(params, "stat", "Stat", "stats", "Stats", default="") or ""
    pct = _is_pct(params, "percentage", "Percentage")
    amount = param(params, "value", "Value", "bonus", "Bonus", "amount", "Amount")
    for name in str(names).split(","):
        field = STAT_TO_BUFF.get(name.strip())
        if field is not None:
            out.append((field, amount, pct))
    return tuple(out)


def fire_on_attack(battle, attacker, target, dtype: int) -> None:
    """`C_PassiveSkill.OnDamageCreated` for every passive this unit carries."""
    for name, p in attacker.on_attack:
        if not _gate(battle, p, dtype, name):
            continue
        if name == "BuffAttack":
            # `castTarget` picks which end of the damage the status lands on:
            # Source is the attacker (the armour-stacking rows), Target the
            # victim (the slows and the armour shreds).
            who = (attacker if str_param(p, "castTarget", "CastTarget") == "Source"
                   else target)
            battle.add_stat_buff(who, param(p, "duration", "Duration"),
                                 status_stats(p, numbered=True))
        elif name == "FearsomeAttack":
            target.apply_status("panic", param(p, "Duration", "duration"))
        elif name == "PassiveStun":
            target.apply_status("stun", param(p, "duration", "Duration"))
        elif name == "ManaBreak":
            target.mana = max(0.0, target.mana - param(p, "amount", "Amount"))


def fire_on_damaged(battle, victim, attacker, dtype: int) -> None:
    """`C_DamageReactionSkill`: the reaction lands on the attacker."""
    if not attacker.alive:
        return
    for name, p in victim.on_damaged:
        if not _gate(battle, p, dtype, name):
            continue
        duration = param(p, "duration", "Duration")
        if name == "Craggy":
            # DebuffType is None/Stun/Silence, and the property carries a
            # [DefaultValue] this dump does not give. The shipped row without
            # one is a 30% reaction to physical damage, which would do nothing
            # at all if the default were None, so it is read as Stun.
            debuff = str_param(p, "debuff", "Debuff", default="Stun")
            attacker.apply_status("silence" if debuff == "Silence" else "stun",
                                  duration)
        elif name == "Untouchable":
            battle.add_stat_buff(attacker, duration, status_stats(p))


def fire_on_cast(battle, agent, action) -> None:
    """`C_OnSkillCastedSkill.OnSkillCasted`, after a skill's effect resolves."""
    for name, p in agent.on_cast:
        if name == "BuffOnCasted":
            battle.add_stat_buff(agent, param(p, "duration", "Duration"),
                                 status_stats(p))
        elif name == "MultiCast":
            chance = param(p, "chance", "Chance")
            if chance <= 0.0 or battle.rng.random() * 100.0 >= chance:
                continue
            # A repeat cast, refunding the shares the row names. The shipped
            # row saves 100% of both, so the repeat is entirely free.
            agent.mana = min(agent.spec.mana, agent.mana
                             + action.mana_cost * param(p, "saveManaPercent") / 100.0)
            action.cooldown_left = max(
                0.0, action.cooldown_left
                - action.cooldown * param(p, "saveCooldownPercent") / 100.0)
            if action.effect is not None:
                action.effect(action, agent, battle)


def run_death_passives(agent, battle) -> None:
    """`C_BaseOnDeathSkill.OnDeath`, once, after the unit is confirmed dead."""
    for name, p in agent.on_death_passives:
        duration = param(p, "duration", "Duration")
        stats = status_stats(p)
        if not stats:
            continue
        if name == "BuffOnDeath":
            # OnDeath walks the dead unit's own team, with no radius term.
            for o in battle.agents:
                if o.team == agent.team and o.alive:
                    battle.add_stat_buff(o, duration, stats)
        elif name == "StickyBlood":
            # ApplyStatus takes an `enemy`, inside `radius` of the corpse.
            radius = param(p, "radius", "Radius")
            r2 = radius * radius
            for o in battle.agents:
                if o.team == agent.team or not o.alive:
                    continue
                if (o.x - agent.x) ** 2 + (o.y - agent.y) ** 2 <= r2:
                    battle.add_stat_buff(o, duration, stats)


def try_resurrect(battle, agent) -> bool:
    """`ResurrectionChance`: one roll, once, back at `healthValue`."""
    if agent.resurrected:
        return False
    chance, value, percentage = agent.resurrect
    agent.resurrected = True
    if battle.rng.random() * 100.0 >= chance:
        return False
    agent.hp = agent.max_hp * value / 100.0 if percentage else value
    agent.statuses.clear()
    return agent.hp > 0.0


def refresh_standing(battle, agent) -> None:
    """`Compensation`: a bonus recomputed from the squad, every tick.

    `SetBonus` counts the allies under `healthThreshold` percent of their
    maximum and multiplies the row's bonus by that count, so the buff grows as
    the squad is worn down. It is one status that is replaced, not stacked.
    """
    stats = []
    for name, p in agent.standing:
        if name != "Compensation":
            continue
        threshold = param(p, "healthThreshold", "HealthThreshold") / 100.0
        hurt = sum(1 for o in battle.agents
                   if o.team == agent.team and o.alive and o.hp < o.max_hp * threshold)
        if not hurt:
            continue
        for field, amount, pct in status_stats(p):
            stats.append((field, amount * hurt, pct))
    if agent.standing_buff is not None:
        # by identity: two buffs can hold identical numbers, and `remove` would
        # drop whichever compared equal first.
        old = agent.standing_buff
        agent.buffs = [b for b in agent.buffs if b is not old]
    agent.standing_buff = None
    if stats:
        from .battle import Buff
        agent.standing_buff = Buff(remaining=float("inf"), stats=tuple(stats))
        agent.buffs.append(agent.standing_buff)


def class_diversity_bonus(specs: list, params: dict) -> float:
    """`bonus x count(otherClass)`, the multiplier `TryFindNewTarget` applies.

    Verified: the method looks up the `otherClass` group, takes its unit count,
    converts it to a float and multiplies the row's bonus by it, then applies
    the result to **one randomly chosen unit** of the mutation's own Class.
    """
    other = str_param(params, "otherClass", "OtherClass", default="")
    count = sum(1 for s in specs if getattr(s, "cls", None) == other)
    return param(params, "bonus", "Bonus", "value", "Value") * count


def diversity_target(specs: list, mutation: dict, rng) -> int:
    """Which squad member a ClassDiversity row lands on, or -1 for none.

    `TryFindNewTarget` takes `RND.Element` of the mutation Class's group, so it
    is one unit, chosen at random, not the whole class. The game re-picks when
    that unit dies; this picks once, before the fight.
    """
    hits = [i for i, s in enumerate(specs) if applies_to(mutation, s.cls, rng)]
    return rng.choice(hits) if hits else -1


def apply_to_agents(agents: list, mutations: list[dict],
                    rng: random.Random | None = None) -> dict[str, int]:
    """Apply agent-level mutation passives. Returns a count per kind."""
    rng = rng or random.Random(0)
    classes = sorted({a.spec.cls for a in agents})
    specs = [a.spec for a in agents]
    counts: dict[str, int] = {}
    for m in mutations:
        name = m["Name"]
        h = handler_for(name)
        counts[h.kind] = counts.get(h.kind, 0) + 1
        p = mutation_params(m)
        if name in ("StatBonus", "OraStatBonus") and is_timed(p):
            # A timed stat mutation becomes an action, like a class skill.
            from .actions import Action
            from .unit_skills import EFFECT_BUILDERS
            random_class = (rng.choice(classes)
                            if m.get("Class") == "Random" and classes else None)
            for a in agents:
                if not applies_to(m, a.spec.cls, rng, random_class):
                    continue
                a.actions.append(Action(
                    name=f"Mutation:{name}", tree_name="NC-Skill",
                    range_world=0.0,
                    cooldown=param(p, "cooldown", "Cooldown"),
                    mana_cost=param(p, "manaCost", "ManaCost", "manacost"),
                    effect=EFFECT_BUILDERS["buff_self"](p), params=p, priority=3,
                ))
            continue
        if h.kind != PASSIVE or name in ("StatBonus", "OraStatBonus"):
            continue
        if name == "ClassDiversity":
            continue          # a standing stat change, done in apply_to_specs
        if name == "ClassDiversityMagicalDamage":
            # Same rule as ClassDiversity, but the bonus is to magical damage,
            # which is not a UnitSpec field, so it lands on the agent instead.
            from .battle import DT_MAGICAL
            i = diversity_target(specs, m, rng)
            bonus = class_diversity_bonus(specs, p)
            if i >= 0 and bonus:
                agents[i].damage_bonus.append(
                    (DT_MAGICAL, bonus, _is_pct(p, "percentage", "Percentage")))
            continue
        random_class = rng.choice(classes) if m.get("Class") == "Random" and classes else None
        for a in agents:
            if not applies_to(m, a.spec.cls, rng, random_class):
                continue
            if name in ON_ATTACK:
                a.on_attack.append((name, p))
            elif name in ON_DAMAGED:
                a.on_damaged.append((name, p))
            elif name in ON_CAST:
                a.on_cast.append((name, p))
            elif name in ON_DEATH:
                a.on_death_passives.append((name, p))
            elif name in STANDING:
                a.standing.append((name, p))
            elif name == "ModifyDamage":
                a.damage_bonus.append((
                    _mask(p), param(p, "Value", "value", "amount", "Amount"),
                    _is_pct(p, "percentage", "Percentage")))
            elif name == "ResurrectionChance":
                a.resurrect = (param(p, "chance", "Chance"),
                               param(p, "healthValue", "HealthValue"),
                               _is_pct(p, "percentage", "Percentage"))
            elif name == "CSLinkStatBonus":
                # `C_CSLinkStatBonusSkill.OnCasted` buffs the cast's targets of
                # `targetClass` for the class skill's own duration, so it rides
                # along on the caster's class-skill action.
                a.cs_link_bonus.append(p)
            elif name in ("Evasion", "StatusEvasion"):
                from .unit_skills import evasion_entry
                a.evasions.append(evasion_entry(p))
            elif name == "Vampirism":
                from .unit_skills import vampirism_entry
                a.vampirisms.append(vampirism_entry(p))
            elif name == "CriticalStrike":
                a.crit_chance = max(a.crit_chance, param(p, "chance", "Chance"))
                a.crit_mult = max(a.crit_mult, param(p, "mult", "value", default=1.0))
            elif name == "FurySwipe":
                a.fury_per_stack = max(a.fury_per_stack, param(p, "amount"))
            elif name == "ExplodeProjectile":
                from .unit_skills import explode_entry
                a.explodes.append(explode_entry(p))
            elif name in ("EventualKnockback", "PassiveKnockback"):
                from .unit_skills import knockback_entry
                a.knockbacks.append(knockback_entry(p))
            elif name == "CooldownReduction":
                factor = 1.0 - param(p, "percent", "value", default=0.0) / 100.0
                for act in a.actions:
                    act.cooldown *= max(0.0, factor)
    return counts


def coverage(tables: dict) -> dict:
    """How many mutation definitions are handled, by kind."""
    counts: dict[str, int] = {}
    unknown: set[str] = set()
    for m in tables["SimpleMutations"]:
        try:
            kind = handler_for(m["Name"]).kind
        except UnknownMutation:
            unknown.add(m["Name"])
            kind = "UNREGISTERED"
        counts[kind] = counts.get(kind, 0) + 1
    return {"by_kind": counts, "unregistered": sorted(unknown)}
