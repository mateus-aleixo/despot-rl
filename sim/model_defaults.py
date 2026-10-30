"""What a skill or mutation row gets for a field it leaves out.

A row's parameters fill a model (`M_VampirismSkill`, `M_BuffAttackMutation`,
...), and a field the row does not name keeps the value its property was
built with. `Model.CreateSettings` sets `DefaultValueHandling.Ignore`, so the
JSON never populates anything; the value comes from the model's static
constructor, which builds each property with the same constant its
`[DefaultValueAttribute]` carries (`M_VampirismSkill..cctor` makes
`_damageType` as `new ReactiveProperty<DamageType>(1)`, and the attribute's
generator passes 1). Every attribute generator in `dump.cs` was read for the
values below; a property with no attribute starts at the type's zero.

Only `damageType` is collected here. It gates a passive the same way
everywhere it was read (`C_VampirismSkill.OnDamageCreated`,
`C_BuffAttackSkill.OnDamageCreated`,
`C_DamageReactionSkill.OnDamageInternal`): the passive fires only when
`(damage.type & damageType) == damageType` and the damage is not
`Secondary`. The port read an absent `damageType` as "any", which is the
type's zero (`Empty`) and right for models without a default, and wrong for
these: a Vampirism row without one heals off physical hits only, and Blurp
deals magical damage.

The other defaulted fields the port reads were assumptions in
`sim/assumptions.py` until the same read settled them, each at the value it
had been argued to: they are below.
"""
from __future__ import annotations

# `M_EvasionSkill.healthThreshold` (and `M_EvasionMutation`'s): 100, so a plain
# evasion is on at full health (`OnHealthChanged` sets `_active =
# healthThreshold >= health%`).
EVASION_HEALTH_THRESHOLD = 100.0
# `M_VampirismSkill.percentage` (and the mutation's): true, a share of the
# damage; only the FixedVampirism rows set it false.
VAMPIRISM_PERCENTAGE = True
# `M_CraggySkill.debuff` (and the mutation's): `DebuffType` 1, Stun.
CRAGGY_DEBUFF = "Stun"
# `chance` on `M_DamageReactionSkill`, `M_BuffAttackSkill`, `M_BurnAttackSkill`,
# `M_StatusEvasionSkill` and the matching mutations: 100, always fires.
CHANCE = 100.0

# Skill or mutation name -> its model's `damageType` default, as the name the
# rows use for the flag. Craggy, Untouchable and LoveOfPain inherit
# `M_DamageReactionSkill`'s (and `M_DamageReactionMutation`'s); PassiveBlock is
# Magical as a skill and as a mutation.
DAMAGE_TYPE = {
    "Vampirism": "Physical",
    "BuffAttack": "Physical",
    "PassiveStun": "Physical",
    "Craggy": "Physical",
    "Untouchable": "Physical",
    "LoveOfPain": "Physical",
    "PassiveBurn": "Physical",
    "PassiveKnockback": "Physical",
    "AggressiveDodge": "Magical",
    "ReinforcedDamage": "Magical",
    "PassiveBlock": "Magical",
    "Blurp": "Magical",
}


def with_model_defaults(name: str | None, params: dict) -> dict:
    """`params` with the model's `damageType` default added where the row
    leaves the field out. The row's own value, under either spelling, wins."""
    default = DAMAGE_TYPE.get(name or "")
    if default is None or any(str(k).lower() == "damagetype" for k in params):
        return params
    return {**params, "damageType": default}
