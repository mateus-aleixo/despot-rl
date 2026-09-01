"""Every unverified number in the battle sim, in one place.

Anything here is a guess that has not been read out of the game binary or its
data. They are collected so calibration has a single surface to attack, and so
nothing silently hardens into a "fact". Each entry says what it is, why it is
uncertain, and how to settle it.

Facts that HAVE been verified live in the code that uses them, with a comment
pointing at the source (the decompiled method, or the data file).
"""
from dataclasses import dataclass


@dataclass
class Assumptions:
    # Unity's default fixed timestep. The game may override it in
    # ProjectSettings/TimeManager, which the bundle export did not include.
    # Settle by: reading TimeManager.asset from a full-game AssetRipper export.
    tick_hz: float = 50.0

    # Units.json AttackSpeed is 0.3-2.0, clustered hard at 0.5. Read here as
    # attacks per second, so 0.5 means one swing every 2 s. It could instead be
    # a cooldown in seconds (0.5 s per swing), which is 4x faster.
    # Settle by: timing a known unit's swings in game against both readings.
    attack_speed_is_rate: bool = True

    # Agent radius, as a fraction of a tile per point of Meta.Size (1, 2 or 3).
    # Size 1 -> 6 world units across -> radius 3.
    #
    # This CONTRADICTS the shipped prefab, where Swordsman (a Size 1 class)
    # has RVOController.radius 6. It is set from Size anyway because radius 6
    # is empirically impossible: two such agents can never be closer than 12
    # apart, and the in-range test is centre-to-centre, so every one of the
    # five shipped Dodger weapons (reach 9), octopus-claws (reach 6) and 29
    # unit rows including both Despot boss forms could never land a single
    # hit. Those classes plainly work in the real game.
    # Settle by: checking whether RVOController.radius is scaled at runtime, or
    # whether the attack range test uses surface distance for these units.
    agent_radius_per_size: float = 3.0

    # Extra reach beyond touching, for a unit whose Range stat is 0.
    melee_margin: float = 1.0

    # 88 of 168 unit rows have Range 0, and melee weapons add none, yet
    # BT.DefaultAttack.ReachedTarget tests `SkillRange^2 > sqrDistance` on plain
    # centre-to-centre distance, which range 0 can never satisfy. So melee gets
    # its reach from somewhere this sim has not found yet.
    #
    # Two RVO agents of radius 6 cannot come closer than 12 apart, so any melee
    # reach below 12 means melee units never land a hit at all. The value here
    # is therefore contact distance plus a small margin -- physically the right
    # shape, but it is a guess, not a read value. Skills.json's melee row
    # carries Range 0.5 (3 world units), which cannot be the whole story for
    # exactly this reason.
    # Narrowed, not closed. What is now known:
    #   * M_Unit has BOTH get_baseRange and get_range, so range is a composed
    #     reactive property, not the raw stat.
    #   * M_Unit has `pixelSize`, a settable per-unit body radius, and
    #     `GetFreePosition(float pixelSize, ...)` uses it for placement.
    #   * DistanceUtils has two checks: SqrDistance (centre-to-centre, what
    #     BT.DefaultAttack.ReachedTarget uses) and InRadius, which DOES call
    #     get_pixelSize. So the game has a radius-aware path.
    #   * Meta.Size (1/2/3) tracks the "Big" attribute exactly.
    # So the pieces for a radius-aware reach all exist; what has not been found
    # is the composition that feeds `range`, because it is assembled at runtime
    # from registered contributions rather than written in one place.
    # Settle by: finding what registers a contribution into the unit's range
    # property, or what seeds pixelSize, or by timing an in-game fight.

    # Target selection. Nearest living enemy, re-evaluated when the current
    # target dies. The real rule comes from the behaviour trees and may weight
    # threat, class, or current attacker.
    # Settle by: reading BT.DefaultAttack.ReadyToAct's target resolution.
    targeting: str = "nearest"

    # The default-attack tree is a three-phase cycle: StartExecution plays the
    # attack animation, AfterAttackAnimation lands the damage, then a recovery
    # animation plays and AfterRecoveryAnimation starts the cooldown. Both
    # phases are WaitForAnimation nodes, so the true durations are per-unit
    # animation clip lengths, not constants.
    #
    # The clip lengths ARE extractable -- 312 of them came out of the export
    # with real m_StopTime values -- but they are effect and status clips. The
    # unit attack clips live in a bundle that has not been ripped yet, so these
    # stay constants for now. The 0.417 / 0.750 pairing mirrors the cast/
    # cast-recovery clips that did come out, which is suggestive, not evidence.
    # Settle by: ripping the bundle holding the unit animation clips and
    # resolving each unit's animator overrides to its attack/recovery pair.
    attack_anim_s: float = 0.417
    recovery_anim_s: float = 0.750
    spawn_anim_s: float = 0.5

    # When the cooldown starts. `M_Action` carries both `cooldown` and
    # `elapsedCooldown` -- elapsed time since the cast, not a countdown begun
    # after recovery -- so the sim starts it at StartExecution and lets it run
    # through both animations. The cycle is then
    # max(attack_period, attack_anim + recovery_anim), which for the common
    # AttackSpeed 0.5 is a clean 2 s.
    #
    # The alternative reading, starting it at AfterRecoveryAnimation, would
    # make the cycle their sum (3.17 s) and would mean AttackSpeed does not
    # correspond to the observed swing rate at all. That is why this reading
    # was chosen, but it has not been confirmed in the binary:
    # BT.DefaultAttack.AfterRecoveryAnimation.OnUpdate only chains to the base
    # BTAction, so whatever sets the cooldown lives in the action controller.
    # Settle by: finding what writes M_Action.cooldown for a default attack,
    # and whether elapsedCooldown advances during a cast.
    cooldown_starts_at_swing: bool = True

    # Projectile flight speed for ranged attacks, world units/s. Real values
    # live on the V_Projectile prefabs.
    # Settle by: reading the projectile prefabs' speed fields.
    projectile_speed: float = 300.0

    # Mana. The mechanism is verified: CS_Damage.InflictDamage tests
    # DamageType.Mana (0x20) and routes that branch to get_mana/set_mana, so
    # mana is delivered through the damage pipeline rather than regenerating.
    # What is NOT known is who creates that Mana-type damage and how much, so
    # these two rates are guesses. Units.json `Mana` is read as the maximum.
    #
    # Impact is limited for most class skills because their cooldowns (11-30 s)
    # bind well before mana does; it matters most for the cheap, short-cooldown
    # skills.
    # Settle by: finding what creates a DamageType.Mana M_Damage and with what
    # amount (M_ManaAttackStatus.Setup is the likely place).
    mana_per_damage_dealt: float = 0.5
    mana_per_damage_taken: float = 0.5

    # `GenerationParams.maxDeadEnds`, the budget `LevelGenerator.Adjust` prunes
    # dead ends down to. `C_Rooms.Generate` never writes the field, which reads
    # as zero, and zero cannot be right: a map with no dead ends is a union of
    # cycles, and an exhaustive search over 7-cell shapes finds none that also
    # satisfies the neighbour ceiling of 3 and `maxSquares = 1` -- while level 1
    # is exactly 7 rooms. So something outside `C_Rooms.Generate` sets it.
    # `sim/mapgen.py` prunes only while the map is over its room count instead,
    # which keeps the count exact and leaves the growth rule to decide how many
    # dead ends survive (about 2.8 a map).
    # Settle by: finding the writer -- the `BeforeGenerationDelegate` hook
    # (`C_RoomCountMutation.OnBeforeGeneration` is one subscriber) or whatever
    # initialises the struct before `C_Rooms.Generate` fills it.
    map_max_dead_ends: int | None = None

    # Rooms covering more than one grid square. `maxSquares = num / 10 + 1` and
    # `maxSquaresAtOnce = 1 + (num >= 15)` are read as constraints on how many
    # 2x2 blocks of rooms a layout may contain, which is what `CalculateSquares`
    # and `RoomNode.get_squares` count. If they instead budget *large rooms*,
    # the constraint is the same shape but the rooms it applies to are not.
    # Settle by: reading `CreateRooms` and how `M_Room` stores its squares.
    map_rooms_are_one_square: bool = True

    # `M_Room.expectedPower`. `C_Rooms.CalculatePower` gives every room a share
    # of `(num - 1) * PowerPerRoom` by the same weights it splits the gold with,
    # including the shops -- but what reads that field was not found, and using
    # it as the enemy budget makes level 1 unplayable (the first food shop would
    # hold 1,389 Power against a starting squad of 750). So the enemy budget
    # stays `PowerPerRoom * DefaultMult|BossMult` and shops hold no enemies,
    # which is also what `EnemyPacks.RoomType` supports: it has `Default` and
    # `Boss` rows and no shop row.
    # Settle by: finding the read of `M_Room.expectedPower`.
    shops_hold_no_enemies: bool = True

    # A mutation shop's shelf. `Rooms.Shops.Mutations` ships `m1` as
    # `{RollCost 0, RollCount 0, BuyCount 2, ShowCount 10}` and leaves the
    # generic `m` entry null, so what a generated shop shows is a code default
    # that is not in the data. `m1` is the only shipped example and is used for
    # every shop; `RollCount 0` is why there is no reroll action.
    # Settle by: reading `M_MutationShop`'s defaults out of `C_MutationShop..ctor`.
    mutation_shop_from_m1: bool = True

    # Portals, secret rooms and quests are not placed at all. `portalsCount` and
    # `minPortalDistance` are `GenerationParams` fields with no `Levels.json`
    # column behind them, so there is no number to place them from; the shipped
    # fixed map has three portals. (Treasure rooms *are* placed now -- they are
    # the shrine, the mutation shop and the talent shop.)
    # Settle by: finding what writes portalsCount in `C_Rooms.Generate`.
    map_has_portals: bool = False

    # `minFinishDistance`, which `CouldBeFinishRoom` tests the boss room's
    # `maxDistance` against. `C_Rooms.Generate` fills it from a game-mode call
    # (`IC_GameMode`, argument 4 and the room count) rather than from a column,
    # so the number is not recoverable from the data. Read as 0 here, which
    # makes the predicate vacuous; the boss still lands at the far end of the
    # map because `ChooseFinishRoom` maximises average distance.
    # Settle by: resolving the IC_GameMode vtable slot for the Default mode.
    map_min_finish_distance: int = 0

    # A fight is called a draw if neither side dies within this long. The game
    # has some real timeout or sudden-death rule.
    # Settle by: finding the fight timeout in the fight controller.
    max_fight_seconds: float = 120.0

    # `Craggy`'s debuff when the row does not name one. `M_CraggySkill.debuff`
    # is a `DebuffType` (None/Stun/Silence) carrying a [DefaultValue] whose
    # value the dump does not give. Read as Stun: the one shipped row without a
    # `debuff` is a 30% reaction to physical damage, which would do nothing
    # whatever if the default were None, and a shipped mutation that does
    # nothing is the less likely reading.
    # Settle by: resolving the [DefaultValue] attribute on M_CraggySkill.debuff.
    craggy_default_debuff: str = "Stun"

    # A damage reaction with no `chance` in its row always fires.
    # `M_DamageReactionSkill.chance` carries a [DefaultValue] the dump does not
    # give; Untouchable ships without one, and a reaction that never fires is
    # the less likely reading of a shipped mutation. Same argument as the line
    # above, and the two would be settled by the same read.
    # Settle by: resolving the [DefaultValue] attribute on
    # M_DamageReactionSkill.chance.
    reaction_default_chance: float = 100.0

    # `ClassDiversity` is applied once, before the fight. The game subscribes to
    # fight start and re-picks its target when that unit dies, so a squad that
    # loses the buffed unit gets the bonus back on someone else; here it is lost
    # with them. Cheap to change if it turns out to matter -- it is one call to
    # `diversity_target` per death.
    # Settle by: nothing to settle; this is a modelling choice, not a guess at a
    # number.
    class_diversity_repicks_on_death: bool = False

    # A Health stat bonus raises the ceiling without granting the hit points.
    # `M_StatBonusStatus` is a stat contribution and nothing found says the
    # current value follows the maximum up, so a unit given +200 Health gains
    # room to be healed into rather than 200 hp.
    # Settle by: finding what M_Unit.health does when totalMaxHealth changes.
    health_bonus_heals: bool = False

    # A panicked unit stops acting and stays where it is. NC-Fear is a tree of
    # its own that the runtime overlays, and it is not inside NC-BaseUnit, so
    # building the base tree never reaches `PanicInPlace` or `PanicFollowPath`;
    # panic is enforced in `Battle.step` instead, with the fleeing left out.
    #
    # The fleeing is probably the point. Both shipped `Panic` rows carry
    # `speedBonus 50` -- a number only a fleeing unit would use -- and both
    # carry `Duration 99999`, which under this model removes the target from the
    # fight permanently rather than sending it running and letting it come back.
    # So Rat and PoringSmall2 are stronger here than they are in the game.
    # Settle by: modelling NC-Fear's panic target, or timing a Rat in game.
    panic_flees: bool = False

    # `M_EvasionSkill.healthThreshold` carries `[DefaultValueAttribute]`, whose
    # value the dump does not print, and only two of the seven shipped rows set
    # it. `C_EvasionSkill.OnHealthChanged` sets `_active = healthThreshold >=
    # health / totalMaxHealth * 100`, so a default below 100 would switch plain
    # evasion off at full health, which it plainly is not in game. 100 is the
    # smallest value that keeps every other row always-on.
    # Settle by: reading the DefaultValue attribute's blob out of the metadata.
    evasion_health_threshold_default: float = 100.0

    # `M_VampirismSkill.percentage` is likewise defaulted. Only the rows the
    # game's Meta calls FixedVampirism set it, and they set it to false, so the
    # unset majority reads as a percentage share of the damage.
    # Settle by: the same metadata read.
    vampirism_percentage_default: bool = True

    # `C_VampirismSkill.Addon.OnApply` divides the heal by one plus the length
    # of a list carried on the damage and heals each of its members as well as
    # the attacker -- the Healers vampirism variant. Nothing in this sim fills
    # that list, so the attacker keeps the whole heal.
    # Settle by: finding what fills `M_Damage`'s list at 0x28.
    vampirism_shares_with_healers: bool = False

    # `KnockbackDamageAddon` carries a `KnockbackType`, Push or Fly, and the
    # sim gives both the same arc: away from the attacker, decelerating. Fly
    # probably crosses units and terrain where Push shoves along the ground.
    # Settle by: reading C_Unit.ApplyKnockBack's use of the type.
    knockback_type_is_uniform: bool = True

    # A knockback with a Radius knocks the victim's whole team back, and this
    # pushes each of them away from the attacker. The game may instead push
    # them away from the impact point, which differs for a unit standing
    # between the two.
    # Settle by: reading the direction M_Knockback.Get is given per unit.
    knockback_radius_pushes_from_attacker: bool = True

    # `ExplodeAddon.OnApply` centres its radius check on a position taken from
    # the damage; this sim uses the victim's, which is the same point unless a
    # projectile is resolved somewhere other than where its target stands.
    # Settle by: identifying the field the addon reads for the centre.
    explosion_centred_on_the_victim: bool = True

    # Every chance in the game is `PseudoRandom.Get`, a shuffle bag that spaces
    # successes out; this sim rolls uniformly. Over a fight the rates match, but
    # a bag cannot miss five times in a row where a uniform draw can.
    # Settle by: porting PseudoRandom, which needs its internal state layout.
    pseudo_random_is_uniform: bool = True


DEFAULT = Assumptions()

# Verified constants, kept here only because several modules need them.
# Source: the A* graph caches, "nodeSize": 6, and Int3's 1000-unit fixed point.
TILE = 6.0
