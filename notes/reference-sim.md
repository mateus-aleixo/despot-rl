# The reference battle sim

`sim/` is the readable, slow implementation. Its job is to be the oracle the
fast Rust core is differential-tested against, so it is written to be diffed
against the decompiled game rather than to run quickly.

    sim/data.py         ruleset loading and the game's own override layering
    sim/spec.py         unit + item -> resolved stats
    sim/nav.py          room grid, A*, waypoint following
    sim/orca.py         RVO2 / ORCA local avoidance
    sim/bt.py           NodeCanvas behaviour tree executor
    sim/bt_leaves.py    the trees' conditions and actions, bound to sim state
    sim/actions.py      the default attack and the class skills
    sim/unit_skills.py  per-unit skills from Skills.json
    sim/mutations.py    run-level mutations
    sim/battle.py       the fixed-timestep loop
    sim/run.py          levels, rooms, economy, shops, progression
    sim/assumptions.py  every number that is still a guess

Run `tools/validate_sim.py` for the check suite and `tools/smoke_battle.py` for
a single narrated fight.

## What is verified against the game

Each of these was read out of `GameAssembly.dll` or the shipped data, not
inferred:

- **Damage.** `CS_Damage.Apply` runs resistance, then armor.
  - resistance: `amount = (1 - resistance) * amount`
  - armor: `amount = max(amount - (armor + bonusArmor) * armorMult, min(max(amount, 0), 1.0))`
  - So armor is a flat subtraction with a floor of 1 damage, and resistance is a
    multiplicative fraction. `DamageType` is a flags enum (Physical 1, Magical 2,
    Direct 4, Heal 8, Reflected 16, Mana 32, CantBeEvaded 128, ...).
- **In-range test.** `BT.DefaultAttack.ReachedTarget` computes
  `SkillRange^2 > sqrDistance`, on plain centre-to-centre distance:
  `DistanceUtils.SqrDistance` is an ordinary squared distance with no radius
  terms.
- **What SkillRange is.** `C_Action.get_SkillRange` returns the class-skill
  group's range if there is one, otherwise `M_Unit.get_range`, i.e. the unit's
  own composed Range stat.
- **Stat composition.** Novice Speed 80 + broadsword Speed 20 = 100, which is
  exactly `UnitMovement.speed` on the shipped Swordsman prefab. An item's base
  value adds; its `*PerLevel` term does **not**, except for Range.
  `CS_Item.GetBonus` is `value * powf(perLevel/100 + 1, level - 1)`, a
  compounding percentage, and `CS_Item` is one Apply function per stat with
  five different rules between them. The table is in the progression section
  below. This was modelled as a flat `+ perLevel * (level - 1)` for every stat,
  which is right for none of them.
- **Scales.** Range is in tiles and a tile is 6 world units (the graph caches'
  `"nodeSize": 6`). Speed is world units per second.
- **Steering parameters.** From the prefabs: `agentTimeHorizon 0.5`,
  `maxNeighbours 10`, `priority 0.5`, `UnitMovement.pickNextWaypointDist 12`.
  The prefab's `RVOController.radius: 6` is *not* used -- see the class-skill
  section for why.
- **Class skill mapping.** `Meta.Classes[cls].Skill` names each class's skill
  and `ClassSkills.json` gives its parameters per level, selected by
  `HumansRequired` against the squad's count of that class.
- **Mana is delivered through the damage pipeline.**
  `CS_Damage.InflictDamage` tests `DamageType.Mana` (0x20) and routes that
  branch to `get_mana`/`set_mana`.

## What is still a guess

All of it is in `sim/assumptions.py`, with the reason and the way to settle it.
The ones that matter most:

- **Melee reach, and body radius with it.** This is the sharp one. 88 of 168
  unit rows have Range 0 and melee weapons add none, but the in-range test is
  centre-to-centre, which range 0 can never satisfy. Melee reach is therefore
  set to touching plus a margin. Worse, the radius that defines "touching" is
  itself contested: the shipped prefab says `RVOController.radius: 6`, but at
  that value five shipped Dodger weapons and 29 unit rows can never land a hit,
  so radius is taken from `Meta.Classes[cls].Size` instead. See the class-skill
  section below for the evidence.
- **AttackSpeed's meaning.** Read as attacks per second, so the common 0.5
  becomes one swing every 2 s. It could be a cooldown in seconds instead, which
  would be 4x faster.
- **Tick rate.** Unity's 50 Hz default; the game may override it.
- **Targeting.** Nearest living enemy. The real rule is in the behaviour trees.
- **Attack and recovery animation lengths, and projectile speed.** Placeholders;
  the real values are animation clip lengths and projectile prefab fields. See
  the behaviour-tree section below.
- **When the cooldown starts.** At the swing, not after recovery. Reasoned from
  `M_Action.elapsedCooldown`, not read. See below.
- **Whether a unit re-targets mid-swing.** This sim re-targets every tick, which
  is its own logic and not something the trees do.

## Fidelity gaps beyond the assumptions

- **No obstacles.** Walkability comes from the room layout, where every cell of
  the rectangle is floor. The A* caches hold the true obstacle map; their node
  records are `uint32` count then 22 bytes per node, with a coordinate field
  stepping by 6000 (nodeSize 6 at Int3's 1000-unit fixed point). Field order is
  not decoded, so `Grid.from_astar_cache` raises rather than guessing.
- **The long tail.** All 10 class skills, 92% of per-unit skill uses and the
  mutation tables are in, the last of it in the agent-level passives section at
  the end of this file. What remains is bespoke: 20 skill uses (Devour, Feast,
  heal inversion, target marking) and 70 of 155 mutation definitions -- the
  economy, taunt and threat, the projectile shapes, and the class-skill
  variants. `remover` and `cloner` are still the two merge strategies
  `sim/data.py` refuses to fake, but no file in the Default ruleset uses them.
- **The bespoke skill trees are unused.** NC-MageCSkill, NC-SwoopSkill,
  NC-DodgeCSkill and friends build and are leaf-complete, but class skills run
  through the generic NC-Skill tree instead, since they share its leaves.

## Behaviour so far

Fights terminate, are deterministic per seed, and reconcile arithmetically:
6 broadswords vs 6 Mancrack deals 1242 damage, which is exactly 18 hits of
(80 - 11 armor). Composition already produces a real gradient rather than a
degenerate one:

| squad | vs 6 Mancrack | vs 10 Mancrack |
|---|---|---|
| 6 broadsword | 12/12 | 12/12 |
| 6 gun | 0/12 | 0/12 |
| 3 broadsword + 3 gun | 12/12 | 0/12 |
| 4 broadsword + 2 crossbow | 12/12 | 5/12 |

The gun squads lose because a gun human is 70 hp and 1 armor -- the gun adds no
health -- against Mancrack's 200 hp and 11 armor. They do fire first. Shooters
being glass cannons that need a front line is a genuine consequence of the data,
not a sim bug, and it is the sort of thing the placement policy should discover.

## Throughput, and why the Rust core is needed

About 2,000 ticks/s single-threaded, roughly 150 ms per battle, after the
behaviour trees and class skills. A full 12-level
run is on the order of 80 fights, so ~9 s of wall time per run. Hierarchical RL
over a million runs would be around 100 days on one core. That is the entire
argument for the native core; this implementation stays as the oracle.

## Behaviour trees

`sim/bt.py` executes the game's own NodeCanvas trees; `sim/bt_leaves.py` binds
their leaves to sim state. Units run **NC-BaseUnit with NC-DefaultAttack
injected** into the empty dynamic Selector under `BinarySelector<CanFight>` --
the slot `C_Unit.TryResolveAction` fills at runtime with the unit's resolved
actions.

**All 27 shipped trees build**, and node coverage is complete: Sequencer,
Selector (both with `dynamic`), SwitchSequence, Parallel, Switch, Guard,
Interruptor, ConditionalEvaluator, BinarySelector, Optional, ConditionNode,
ActionNode, SubTree.

Leaf coverage, checked by `tools/validate_sim.py` so it cannot rot:

| | types | uses |
|---|---|---|
| implemented | 28 | 207 |
| declared unmodelled (knockback, stun, panic, shop flow, class skills) | 39 | 52 |
| **unknown** | **0** | **0** |

An undeclared leaf raises rather than silently doing nothing, so a tree that
quietly stops behaving cannot be mistaken for one that works.

### What the trees changed

The hand-written attack loop was wrong in shape. NC-DefaultAttack is:

    Sequencer
      Sequencer [dynamic]
        Selector [dynamic]      ReadyToAct, else FollowPath
        Selector                StopIfRunning, else IsNotOnCooldownContinuous
      Guard "AttackGuard"
        SwitchSequence
          StartExecution
          Sequencer  Interruptor<!IsStillApplicable> -> WaitForAnimation, then AfterAttackAnimation
          Sequencer  WaitForAnimation, then AfterRecoveryAnimation

So a swing is three phases -- attack animation, damage on
`AfterAttackAnimation`, then a recovery animation -- not a single windup. The
outer `dynamic` Sequencer re-checks range every tick, so a unit whose target
walks away drops back to FollowPath, while the inner `SwitchSequence` runs
under a `Guard` and is *not* re-evaluated, so a swing already begun completes
unless the Interruptor aborts it.

### The cooldown question

`AfterRecoveryAnimation` looked like where the cooldown starts, which would make
the cycle `cooldown + attack + recovery` = 3.17 s and leave AttackSpeed with no
relation to the observed swing rate. `M_Action` carries both `cooldown` and
**`elapsedCooldown`** -- elapsed time since the cast, not a countdown begun
after recovery -- so the sim starts it at `StartExecution` and lets it run
through both animations. The cycle is then
`max(attack_period, attack_anim + recovery_anim)`, which for the common
AttackSpeed 0.5 is exactly 2 s, and the measured swing interval is 2.00 s.

This is reasoning from a field name, not a read value:
`BT.DefaultAttack.AfterRecoveryAnimation.OnUpdate` only chains to the base
`BTAction`, so whatever writes the cooldown lives in the action controller and
has not been found. It is recorded in `assumptions.py`.

### Animation lengths

Still constants. 312 clip lengths did come out of the export with real
`m_StopTime` values (`data/extracted/anim_lengths.json`), but they are effect
and status clips -- the unit attack clips live in a bundle that has not been
ripped. The 0.417 / 0.750 defaults mirror the cast / cast-recovery pairs that
did come out, which is suggestive, not evidence.

### Two test traps worth remembering

Both cost a debugging round and neither was a sim bug:

- Pooling both teams' damage events makes the swing interval look like
  `[0.0, 2.0, 0.0, 2.0]`, because both sides engage simultaneously and swing on
  the same tick. Filter by team.
- The Interruptor cannot be exercised through a live battle: `step()` re-targets
  a dead target before the tree ticks, so `IsStillApplicable` always sees a live
  one, and with a single enemy `Team2IsDying` gates off the whole fight branch
  first. It is tested at the engine level with a scripted context instead.

The second one is a real fidelity question in disguise: re-targeting every tick
is this sim's own logic, not something the trees do. Whether a real unit
switches target mid-swing is unresolved.

## Class skills

`sim/actions.py` resolves them; the battle loop applies them. All 10 player
class skills are implemented and all 10 demonstrably take effect.

The mapping is exact, not guessed: `Meta.Classes[cls].Skill` names the skill,
and `ClassSkills.json` gives its parameters per level with `comment1..6` naming
the columns. The level is the highest row whose `HumansRequired` the squad's
count of that class meets, so 5 Warriors give CriticalStrike L4 (55% chance,
4x damage). Plant is the only class with no skill, which matches its items:
all six have zero stats, so a "Plant human" is a naked Novice.

| class | skill | kind | what it does |
|---|---|---|---|
| Warrior | CriticalStrike | passive | chance% to deal value x damage |
| Shooter | ASAura | passive | +value% attack speed, side-wide |
| Dodger | Dodge | passive | negate one hit every `cooldown` s |
| Medic | Heal | active | heal the most damaged ally (`GetMostDamagedTarget`) |
| Mage | Mage | active | magical damage, so resistance not armor |
| Thrower | Bomb | active | AoE at the target, `radius` from the table |
| Tank | Tank | active | timed self buff, +armor% and a speed penalty |
| Monk | SpiritLink | active | share share% of damage across N nearby allies |
| Cultist | Cultist | active | summon a Tentacle of `tentacle level` |
| Scientist | Scientist | active | summon a SciTower of `tower level` |

### Actions, not a bolted-on system

The trees made this straightforward. NC-DefaultAttack, NC-Skill and
NC-HealCSkill all drive the *same* `BT.DefaultAttack.*` leaves -- the leaves are
generic and what differs is the action bound to them. So a unit is now a list of
actions, each with its own range, cooldown, mana cost and effect, and each gets
its own subtree inside the base tree's dynamic Selector. That Selector is the
priority list `C_Unit.TryResolveAction` fills, and `ActionScope` binds the
action the way the runtime binds an `M_Action`. No new leaves were needed.

### Mana

Verified: `CS_Damage.InflictDamage` tests `DamageType.Mana` (0x20) and routes
that branch to `get_mana`/`set_mana`, so mana is delivered through the damage
pipeline rather than regenerating. What creates that Mana-type damage, and how
much, is not known -- the two rates in `assumptions.py` are guesses. Most class
skills are cooldown-bound (11-30 s) rather than mana-bound, so the exposure is
limited; the cheap short-cooldown skills (Heal at 2 s) are where it bites.

## Two bugs the class-skill work surfaced

Both were mine, both were silent, and both changed real numbers.

### Item AttackSpeed is a percentage, not an addition

Item `AttackSpeed` values run -40..+70 and the heaviest weapons carry negative
ones (opm-costume: -40% at 300 damage). Composing them additively onto the
unit's 0.5 gave `lance` a 0.02 s period (6075 dps) and `throwing-grenade` an
infinite one (it never attacked at all). Corroborated by the ASAura class skill,
which uses the same convention explicitly (`percentage=true`, values 10-70).
Fixed, and DPS now tracks item Cost sensibly: broadsword 4 gold -> 40 dps,
chainsaw 18 -> 121, guts-sword 38 -> 136, plant-leaf 1 -> 10.

### Agent radius made whole classes unable to attack

The in-range test is centre-to-centre, so a weapon whose reach is below the
minimum separation of two bodies can never land a hit. At the prefab's
`RVOController.radius: 6`, that minimum is 12, which silently disabled:

- all five shipped Dodger weapons (reach 9)
- octopus-claws (reach 6)
- 29 unit rows, including both Despot boss forms, DeathKnight and Samurai

The Dodger squad was doing 394 damage and losing 0/6 over 23 s. Radius is now
taken from `Meta.Classes[cls].Size` (1, 2 or 3 tiles, so Size 1 -> radius 3),
after which Dodger does 1668 damage and wins 6/6 in 3.8 s.

This contradicts the shipped prefab value and is recorded as such in
`assumptions.py`. The empirical argument is what decided it: those classes
plainly work in the real game, so radius 6 cannot be what the range test sees.
`octopus-claws` still cannot connect -- reach exactly 6.0 against separation
exactly 6.0, against a strict `>` -- and is pinned as a known exception in the
check suite so a new unreachable item still fails.

## Per-unit skills

`sim/unit_skills.py`. `Units.json` rows carry up to eight `SkillN` ids into
`Skills.json`, whose `CSClass` names the mechanic and whose `ParamNName` /
`ParamNValue` pairs carry the numbers.

Every CSClass in use is registered with an explicit kind, so coverage is a
number rather than an impression. Across the 259 skill uses units actually
reference:

| kind | uses | share |
|---|---|---|
| NOOP (real, but nothing for this sim to do) | 100 | 39% |
| ACTIVE (cast on cooldown via the generic NC-Skill tree) | 76 | 29% |
| PASSIVE (always-on modifier or hook) | 57 | 22% |
| UNIMPLEMENTED (declared, unit will not use it) | 26 | 10% |
| **unregistered** | **0** | |

So 90% of uses are handled or correctly inert. An unregistered CSClass raises.

The largest NOOP is worth spelling out: **`Resistance` (66 uses) is purely a
marker.** Every unit carrying it has a nonzero `Resistance` stat and every unit
without it has zero, so the skill is the icon and the stat is the value, which
the damage formula already applies.

The 26 still-unimplemented uses are a long tail of at most 3 uses each:
Devour's grab-and-channel, Feast's corpse consumption, heal inversion
(AntiHeal), on-damaged and on-cast triggers, conditional mitigation (Craggy,
Compensation), resurrection, target marking, and DespotMassSummon's styled
spawn.

`tools/unit_skill_matrix.py` probes every class carrying an implemented active
skill: **47 fire, 5 are blocked by a real gate** (four cannot bank enough mana
-- SpiderSummon costs 260, LeechSummon 210 -- and CatBot dies at 0.9 s), **0
unexplained**. Reporting *why* a skill did not fire is the point; "never cast"
on its own cannot tell a bug from correct behaviour, and it hid two real bugs
below.

## Three more bugs, all mine, all silent

### Resistance was a percentage read as a fraction

`Units.json` Resistance is 5..90, but `ApplyResistance` computes
`(1 - resistance) * amount` with it as a fraction. Passing the raw stat meant
a unit with Resistance 80 took `(1 - 80) x 100 = -7900` from a 100-damage
magical hit -- a massive heal. Nothing caught it because only the Mage class
skill deals magical damage and the test enemy has Resistance 0. Now scaled by
100; that hit correctly does 20.

### Melee could not touch anything bigger than itself

Melee reach was built from the attacker's own radius (`2 x radius + margin`),
but two bodies cannot be closer than the **sum** of their radii. A Size 1
attacker (radius 3, reach 7) against a Size 2 target (radius 6) needs to close
to 9 and never can. **51 of 121 classes are Size 2 and 9 are Size 3**, so melee
did nothing at all to half the roster: a 3-broadsword squad dealt literally 0
damage to a Necromancer over a full 60 s fight.

Reach is now evaluated per target pair, in `Battle.effective_range`:
`attacker.radius + target.radius + margin`. The same fight now deals 578.

This is the third distinct appearance of the melee-reach problem (first: no
melee could hit anything; second: the Dodger line; third: anything larger than
Size 1). All three come from the same root -- the verified in-range test is
centre-to-centre while the real source of melee reach is still unknown.

### Two test harnesses mutated the shared Assumptions

`b.a.max_fight_seconds = 40.0` mutates the `DEFAULT` dataclass instance that
every other battle shares, so one probe silently changed the timeout for every
later check in the same run. Both now use `dataclasses.replace`. Worth
remembering: `DEFAULT` is a singleton, not a template.


## Control effects, displacement and the last passives

A second pass took per-unit skill coverage from 82% to **90% of uses**, and the
matrix from 39 classes firing to **47**.

**Statuses are now real, and the behaviour trees already had the leaves for
them.** `IsStunned`, `Stunned`, `IsKnockedBack`, `OnPushKnockback`,
`IsPanicked`, `PanicInPlace` and `PanicFollowPath` were all sitting in
NOT_MODELLED; they are now wired to a per-agent `statuses` dict. A stunned unit
neither steers nor acts; a knocked-back one is carried by its push vector and
cannot steer.

Silence has no tree leaf of its own, so it is enforced in the range gate: a
silenced unit fails `ReadyToAct` for any action that is not the plain attack.
That keeps the distinction the game draws -- silence stops casting, not
swinging -- and the check suite pins both halves.

Also added: Reflection (percent of damage taken returned raw to the attacker),
the unit-level Dodge (distinct from the Dodger class skill -- it negates one hit
every `Cooldown` seconds), and projectile splash (Splash, ExplodeProjectile),
which reuses the existing splash path at the moment of impact.

Knockback derives its duration from the data rather than a constant:
`PassiveKnockback` gives `knockbackSpeed 180` and `knockbackAcceleration -350`,
so the push lasts `speed / |acceleration|` seconds.

## Mutations, and the merge strategies that were blocking them

`sim/data.py` previously **raised** on `mergeGrid` and `mergeByMutationAndLevel`
rather than merging wrongly, which blocked every mutation table. Both are now
implemented, read out of the game rather than guessed.

`Loader` holds six merge functions as fields -- `jsonMerger`, `mergeByID`,
`mergeByMutationAndLevel`, `mergeGrid`, `remover`, `cloner` -- and they compile
to the lambdas `<.cctor>b__70_0` .. `b__70_5` in declaration order.

- **`mergeByMutationAndLevel`** references exactly the string literals
  `"Mutation"` and `"Level"` and builds a dictionary keyed by them. So it
  overlays rows by that pair.
- **`mergeGrid`** references only `"CombinedMutations"`: it takes that property
  off *both* sides and `Remove()`s it, merges what is left with the ordinary
  merger, then merges that one key separately. Its rows carry `ID` (10000+), so
  they key by ID.
- **`__remove`** is a general marker, not a quirk: it appears 134 times across
  MutationGrid, MutationsByLevel and Consumables overrides. A row carrying it
  deletes the base entry instead of overlaying it.

Array handling could not be read directly (the `JsonMergeSettings` setter is
inlined), but the data settles it: the Default chip's `SimpleMutations` is
byte-identical to the base list, and Newtonsoft's *default* is `Concat`, which
would produce two `AdditionalRerolls`. Mutagens also swaps `GetItemBack` for
`RoomCount` in place. So it is Replace.

The `__remove` implementation verifies cleanly against the data:
`Chips/Default/WithoutFood/MutationsByLevel.json` ships exactly 41 `__remove`
rows, and MutationsByLevel goes **1296 -> 1255**.

All six chip/WithoutFood combinations now load with `strict=True`.

### The mutations themselves

`sim/mutations.py`. 155 definitions, 90 distinct `Name`s. `Name` reuses the same
vocabulary as `Skills.json`'s `CSClass`, so 23 names route straight to the
handlers already written for per-unit skills. `Class` selects who is affected:
"All", "Random", a single class, or a comma-separated list.

| kind | definitions |
|---|---|
| UNIMPLEMENTED | 105 |
| PASSIVE | 40 |
| ACTIVE | 8 |
| NOOP (economy and UI, outside a fight) | 2 |
| **unregistered** | **0** |

Handled is **50/155 = 32%**, much lower than skills, and that is the honest
shape of the data rather than a shortfall of effort: mutations are largely
bespoke one-off mechanics (taunt and threat, absorb pools, resurrection,
conditional immunity, class-skill variants, chaining and bouncing projectiles),
each appearing once or twice. `StatBonus` alone is 19 of the 20 handled stat
definitions and is what most of the offered pool is built from.

One subtlety worth recording: **a `StatBonus` carrying a `duration` is a timed
cast, not a permanent change.** Exactly one shipped definition is like this
(ID 222, `OraStatBonus`, the Monk attack-speed buff -- note the different Name,
which is why a filter on `StatBonus` alone misses it). Permanent ones use
`bonus`; the timed one uses `value`. It is routed to an Action driven by the
generic NC-Skill tree, like a class skill, rather than being folded into the
spec.

Applying mutations moves outcomes as expected: 6 broadswords vs 10 Mancrack go
from 3284 average damage to 3803 with a single +15% damage StatBonus, and from
9/10 to 10/10 wins with a +50 health one.

Order matters when using this: `apply_to_specs` before deploying (specs are
frozen, so it returns new ones), then `apply_to_agents` **before** constructing
the `Battle`, since that is when each agent's behaviour tree is built from its
action list.

## The run layer

`sim/run.py`. A run is 12 levels; each level is a 6x6 room map from
`Rooms.json`, where `Matrix` holds the room ids and a parallel `Shops.Matrix`
gives each room its type (`i` item shop, `f` food shop, `m` mutation shrine,
`e` fight, ' ' none). You move between orthogonally adjacent rooms, plus
portal-to-portal links, from `Start` to `Boss`.

This is the surface the hierarchical agent acts on. `legal_actions()` returns
move / buy_item / reroll / upgrade_shop / buy_exp / buy_food / buy_human /
take_mutation / sacrifice / feed, and `apply()` advances the state, resolving
any fight through `Battle`.

**The hunger model was read off the JSON, and the JSON does not say what it
looks like.** `Game.Food` gives `moves`, `damagePenalties`, `armorPenalties` and
`foodPerSacrifice`, and the obvious reading -- a move allowance per hunger
stage, with feeding as something the player chooses -- is wrong on all three
counts. `C_Food.Move`, `C_Food.Feed` and `C_Food.OnMovesLeftChanged` say:

    maxMoves  = moves[0]                       # C_Food..ctor
    canFeed   = amount >= needed
    canMove   = canFeed or movesLeft > 0

    Move():   if canFeed: Feed(); return       # no move is spent
              movesLeft -= 1
              hungerLevel = 0
              for i from len(moves)-1 down to 0:
                  if movesLeft < moves[i]: hungerLevel = i + 1; break

    Feed():   amount -= needed
              movesLeft = maxMoves
              hungerLevel = 0

So **moving feeds you**. Entering a room costs `needed` food, which
`C_Food.SetUnitCount` sets to the squad's unit count, and spends no move at all.
`moves` is not an allowance: it is a list of thresholds on `movesLeft`, a
six-move reserve that only drains once the larder cannot cover a room, and
`hungerLevel` is *derived* by comparing the reserve against those thresholds
rather than incremented -- so hunger falls again the moment you eat, in one
step. With `moves = [6, 3]` a broke squad gets six more rooms: three at 30%
penalties, three at 60%, and then `canMove` is false.

**There is no feed button.** `C_Food.Feed` has exactly three callers --
`C_Food.Move`, a mutation, and the sacrifice handler -- and none of them is UI.
The sim offered `feed` as a run-level action, which made food nearly free: a
policy paid only when it chose to, so it hoarded (runs ended holding 130 food)
and food never constrained anything. Food is now the price of a room, which is
what makes a large squad expensive to walk around.

Enemy strength comes from `Levels.json`: `PowerPerRoom` scaled by `DefaultMult`
for a normal room and `BossMult` for the boss, filled from packs eligible for
the level's `MinStyleID..MaxStyleID` via `PacksByStyle`. The ramp is steep and
real: level 1 fills a room to ~1,500 Power, level 6 to ~24,600.

### What the policy comparison found

A random policy and a simple heuristic (feed when hungry, take what a room
offers, otherwise head for the boss) are worth comparing because the gap is the
signal an agent has to learn:

| policy | mean level | max | s/run |
|---|---|---|---|
| random | 2.27 | 5 | 1.3 |
| heuristic | 4.60 | 6 | 4.6 |

Twice as deep for a policy that is a dozen lines. That is a healthy gradient to
hand to RL, and neither policy finishes 12 levels, so there is a lot of room
above.

Building that comparison also **found a modelling gap rather than a policy bug**:
the heuristic first scored *worse* than random, reaching level 1 with zero
fights, because item shops had unlimited stock and the level-1 `Price` is 0, so
it stood in the first shop buying forever. Shops now carry stock from
`ItemShopData.Quantity` (3), food shops 3, and a mutation shrine is one-use. The
lesson is that an agent will find any unbounded action, so every action needs
its real limit before training, not after.

### Three run-layer rules an RL agent found before I did

Each of these was a place the run layer was *cheaper* than the game, and each was
found the same way: by looking at what a trained agent actually did with its
actions (`tools/run_autopsy.py`) rather than at its score.

**Humans were buyable anywhere.** `legal_actions` offered `buy_human` in every
room, so a 300k-step agent spent 73% of its actions on it and one run ended with
277 humans. In the game the only caller is `C_ItemShop.BuyHuman`, which compares
`M_Team.gold` against `M_ItemShop.humanCost` -- a flat 2 that is never raised
after a sale, and is only reachable inside an item shop. Now gated on the room.

**The deployment grid was not a limit.** `place_at` cycled the player zone's
cells, so a squad of 200 stacked onto 49 cells and all of them fought.
`C_Player.FindCell` walks the layout through `C_UnitLayout.IsCellEmpty` and
returns nothing when the zone is full, so the zone *is* the roster: the sim now
fields as many units as there are cells and leaves the rest out of the fight,
untouched.

**Feeding cost a flat 10.** It is `M_Food.needed`, and `C_Food.SetUnitCount`
writes the squad's unit count straight into it (`C_Food.OnCountChanged` sums the
units and calls it), while `C_Food.Feed` is `amount -= needed` followed by
`movesLeft = maxMoves` and `hungerLevel = 0`. So feeding costs **one food per
human**, which is precisely what makes a large crowd expensive rather than free.
With a flat cost, a shaped agent bought 70 humans and fed them for the price of
ten.

The pattern is the one already recorded twice in `notes/rl.md`: an agent will
find any action that is cheaper in the sim than in the game, and it will look
like a strategy rather than a bug.

### Progression: what gold is for

The run layer had no way to turn gold into strength, which is why squad Power
sat flat at ~20,000 for four levels while the rooms grew 800 -> 7,700, and why
four different reward functions all landed on level 3.7 (`notes/rl.md`). Four
things were wrong, and the fourth was the one that mattered.

**The squad started fully armed, and that was a misread.** `RunState.new` took
the *keys* of `Game.Team.Items` as a pool and gave every human a random weapon
out of it, opening at ~20,000 Power against an 800-Power first room. But
`Team.Items` and `Team.Classes` are inventory maps whose every value is 0, and
`C_Session.LoadPlayerTeam` does not read them: it reads `Game.Team.Packs`, which
in the Default ruleset is five `Novice` rows with **no `Item` field at all**. The
run starts with five bare humans. The Power curve was not flat because nothing
made the squad stronger; it was flat because the squad began at the ceiling.

**An item cost the wrong number.** `C_ItemShop.Buy` compares `M_Team.gold`
against `M_Item.cost` and subtracts that, so an item costs its own `Items.Cost`
-- 4 for a broadsword, 43 for a casket. The sim charged `ItemShopData.Price`
instead, which is the *shop upgrade* cost (`C_ItemShop..ctor` reads
`prices[level]`, the next level's). Everything cost the same, so quality was
free.

**The shop had no quality and no identity.** It is one model for the whole run
(`M_Session._itemShop`; `C_Rooms..ctor` builds its controller once), holding
`ItemShopData.Quantity` slots at its level. Each slot draws a quality with weight
`Q<n>Prob` from that level's row -- `C_ItemShop.<FillWeightedList>b__4_0(q)` is
literally `probabilities[level - 1, q - 1]` -- and then an item of that quality.
A level-1 shop can only sell quality 1 and 2; a level-5 shop sells all five, and
quality 5 runs to 11,444 Power against quality 1's ~600. Upgrading costs
`prices[level]` and also widens the shop from 3 slots to 7.

**Nothing consumed experience.** `M_Unit` has one `level`, and
`CS_Units.LevelUp` re-reads the class row at the new level *and* re-applies the
item's bonus there, so the sim's separate `item_level` had no counterpart in the
game. Experience arrives from `C_Unit.Die`, which hands a dead enemy's
`ExpReward` to the other team, and from the shop, which sells
`ExperienceAmount` (400) for `ExperienceCost` (1 gold);
`C_Team.GetExperience` splits either evenly over the units in the room and loops
`LevelUp` while the unit is over `ExperienceForLevels[level - 1]` and under
level 5. The sim was crediting `total_damage / squad size`, a number the game
does not have anywhere.

With those four fixed, the same heuristic's squad Power goes 750 -> 3,453 ->
6,754 -> 10,425 across levels 1 to 4, and its readiness against the room it is
standing in *rises* through level 3 instead of collapsing:

| level | squad | squad Power | room Power | ratio | ratio vs level 1 |
|---|---|---|---|---|---|
| 1 | 5.0 | 750 | 800 | 0.9 | 1.00 |
| 2 | 2.7 | 4,015 | 2,450 | 1.6 | 1.75 |
| 3 | 2.9 | 7,044 | 3,920 | 1.8 | 1.92 |
| 4 | 2.9 | 9,657 | 7,700 | 1.3 | 1.34 |

#### The item bonus compounds; it does not add

`CS_Item` is one `Apply*` function per stat and they do not agree, so the sim's
single additive rule was wrong for most of them:

| stat | law |
|---|---|
| Health, Damage, Armor, Mana | `unit += item * (1 + perLevel/100) ** (level-1)` |
| Range | `unit += item + perLevel * (level-1)` |
| Speed | `unit += item` — no per-level term exists |
| AttackSpeed | `unit *= 1 + (item + perLevel*(level-1)) / 100` |
| Resistance | `unit = u + (1-u)*r`, `r = (item + perLevel*(level-1))/100` |

`CS_Item.GetBonus(value, perLevel, level)` is `value * powf(perLevel/100 + 1,
level - 1)`, and only the four compounding stats route through it. In the
Default ruleset only `DamagePerLevel` and `HealthPerLevel` are ever non-zero, so
the visible difference is there: an opm-costume at level 5 is 423 damage, not
the 336 the additive rule gave; a broadsword is 88, not 100.

#### Two things here are marked as assumptions, not facts

`C_ItemShop.Roll` is only ever reached through the UI, so there is no call site
saying *when* the shop refills a bought-out slot. It is modelled as one free
refill of empty slots the first time you enter a shop room, tracked per room so
that leaving and coming back is not a free reroll. And within one quality the
game draws from a `DynamicWeightedList` with factor 2, which biases against
repeating a recent draw; the sim draws uniformly.

The pool itself is measured, not assumed: `M_Item.neverGiven` comes from
`Meta.Items[name].NeverGiven`, true for exactly six rows (plant-leaf, leaflet,
cube-part, comic-book, rat-flute, mik-helmet), all with 0 Damage, 0 Health and
0 Power. Ten items per quality remain.

### The food shop

Read off `C_FoodShop` (`..ctor` at RVA 0x71B070, `DetermineQuantities` at
0x71BBD0, `Buy` at 0x71BEA0). It used to be a stub selling 5 food for 1 gold,
three times a shop, which was harmless while feeding was optional and became
load-bearing the moment a room started costing one food per human.

**The shelf is `Rooms.Shops.Food.Packs`**, `[[7, 2], [11, 3], [37, 9],
[120, 27], [250, 55]]`. The constructor reads index 0 into `sizes` and index 1
into `costs`, and `Buy(i)` spends `costs[i]` of `M_Team.gold` and calls
`C_Food.AddFood(sizes[i])`, so the pairs are **[food, gold]**: 3.5 food per gold
at the bottom of the shelf and 4.55 at the top. The sizes are the real
difference from the stub, not the rate -- a 250-food pack is forty rooms for a
six-human squad, where the stub capped a whole shop at fifteen food.

**`TotalFood` is a gold budget, not an amount of food.** `..ctor` sets
`M_FoodShop.totalFood = levelData["TotalFood"] / levelData["FoodShops"]` and
tail-jumps into `DetermineQuantities`, which sweeps the packs in order buying one
of each it can still afford and repeating while anything is left -- subtracting
**`costs[i]`**, the gold price, never `sizes[i]`. So a shelf costs exactly the
budget it was stocked to: 50 gold buys 4x7 + 2x11 + 37 + 120 = 207 food, priced
at 4x2 + 2x3 + 9 + 27 = 50. Pack 0 is the exception, bought whether or not the
budget covers it (`if remaining >= costs[j] || j == 0`), which is both why a
shop is never empty and what terminates the loop.

**A shop is stocked once and never restocks.** `C_Room..ctor` is the only caller
of `C_FoodShop..ctor`, so there is one controller per room, and
`DetermineQuantities` has no other call site: `Roll` reaches it but has no
callers at all, being UI-only like `C_ItemShop.Roll`. Walking out and back in
buys nothing. `CheckEmptiness` closes the shop once every pack is gone.

**One map per level, generated the way the game generates one.** Every level used to reuse the
shipped `Rooms.json` map, which is 22 rooms with six food shops, three item shops
and three mutation shrines against a level-1 row asking for 7 rooms, one food
shop, two item shops and one shrine. That put two to three times the rooms and
three to six times the shops under a food budget written for the row, and it is
why the food shop's budget had to be divided by the shops on the map rather than
by the row's `FoodShops`. `sim/mapgen.py` now generates the level the row asks
for, and `food_shop_budget` is the game's own `TotalFood / FoodShops` again.

The shape is `LevelGenerator`'s, not a stand-in for it. Growth draws open
positions by `WeighPositions`'s own weight,

    snCount^5 + nCount^10 + 2^(maxDistance - distance) * (maxDistance - distance) + 2

(`nCount` counts adjacent rooms, `snCount` adjacent nodes), and `MaybeMakeRoom`
refuses a position for good -- `state = No` -- unless the map still satisfies
`CalculateSquares() <= maxSquares` and every room's `squares <=
maxSquaresAtOnce`. Those two rations of 2x2 blocks are what keep the tenth power
from filling in a rectangle, and they are why a level comes out a thin winding
web rather than a blob: mean degree 2.17, about 2.8 dead ends, and no room with
four neighbours (`CheckNeighborCount`). `Adjust` cuts dead ends and `CloseLoops`
closes them into cycles. The boss room is what `CouldBeFinishRoom` allows --
never an articulation point, either a dead end or a room with exactly two
neighbours -- chosen by `ChooseFinishRoom`'s average distance, with the start
furthest from it and the shops spread by `ChooseDiversePositions`' relaxing
distance threshold.

**A level hands out no food.** The sim used to add the level row's `TotalFood`
to the larder on arrival, on the reading that the column was a level's food
allowance. It is not: `C_FoodShop..ctor` is the only thing that reads it, and
the only callers of `C_Food.AddFood` are the food shop and the `PlusFood` /
`MinusFood` dialog events -- `C_Levels.WinLevel` and `ChangeLevel` touch the
larder nowhere. That grant was up to 250 free food a level, forty rooms for a
six-human squad, and it is gone. Food for a level is bought from that level's
shops.

### Treasure rooms, and the consumables that are not reachable

`LevelGenerator.ChooseTreasureRooms` is misleadingly named: it places the three
room types that carry the `Treasure` bit (64), one per level-row column, and it
places them **uniformly at random** over the still-untyped rooms (`RND.Element`)
rather than spread out the way `ChooseDiversePositions` places the item and food
shops:

| column | `RoomType` or-ed in | what it is |
|---|---|---|
| `Shrines` | 192, `Shrine` | a free mutation -- one, on level 1 only |
| `RerollShrines` | 320, `MutationShop` | the mutation shop -- one, levels 2 to 12 |
| `StatShops` | 4160, `TalentShop` | the consumable shop -- **0 on every level** |

`TreasureMult` weights all three the same in `C_Rooms.CalculatePower`, which is
the only place it appears.

**The mutation shop.** `C_Room.InitShop` builds a `C_MutationShop` for a room of
type 320. `BaseMutationShop.Buy(index)` refuses a mutation already taken, checks
`M_Mutation.cost` against the team's gold, subtracts it and decrements
`buyCount`; `Roll` charges `M_Shop.rollCost` and is bounded by `rollCount`.
Nothing in `Mutations.json` carries a price and the shipped `m1` config is
`{RollCost 0, RollCount 0, BuyCount 2, ShowCount 10}`, so a shop shows ten
mutations, hands over two of them for nothing, and cannot be rerolled. The sim
had one mutation shrine per level of the fixed map and nothing else; a generated
run now gets one free mutation on level 1 and two more at each level after it,
which is 2.1 mutations a run for the heuristic where the old map gave 0.6.

**Consumables cannot happen in a Default run.** `C_Consumables.Add` has exactly
one caller, `C_ConsumableShop.Buy`; `C_ConsumableShop` is only constructed by
`C_Room.InitShop` for a room of type 4160 (`TalentShop`); and `StatShops` is 0
on all twelve rows of `Levels.json`. So the fourteen rows of `Consumables.json`
and the 124 of `ConsumablesByLevel.json` are unreachable content in this mode,
and the sim is not missing income by leaving them out. `tools/validate_sim.py`
asserts the `StatShops` half of that, so a ruleset that turns them on will fail
the check rather than pass silently.

### Gold is a level total, split by room type

`C_Rooms.CalculatePower` reads `PowerPerRoom` and `GoldPerRoom`, multiplies each
by `(num - 1)`, and divides by the level's total weight

    shrines+mutationShops+talentShops x TreasureMult + BossMult
      + itemShops x ItemsMult + foodShops x FoodMult
      + (everything else) x DefaultMult

before walking the rooms and giving each one `mult(type)` of the result:
`M_Room.expectedPower` and `M_Room.goldReward`, with the start room getting
Power but **no gold**. So `GoldPerRoom` is not what a room pays; it is what a
room pays *on average*, and a shop pays two to three times a corridor. At level
1: 10.5 gold for an item shop or a shrine, 6.6 for a food shop, 17.5 for the
boss, 4.4 for a plain room, and 60 for the level as a whole where the flat rule
this replaced paid 70.

That changes what walking is worth. The old rule paid the same for any room, so
a run was paid to wander; the game pays for the rooms that are worth entering
anyway, which is most of the gold in the two or three shops on a level.

### Deliberate simplifications, to revisit

These are places the run layer is thinner than the game, and each is visible in
`sim/run.py` rather than hidden:

- A generated room covers one grid square. `maxSquares` and `maxSquaresAtOnce`
  are honoured as constraints on 2x2 blocks of rooms, which is what
  `CalculateSquares` and `RoomNode.get_squares` count, rather than as a budget
  for rooms that cover several squares.
- `maxDeadEnds` could not be recovered -- see `sim/mapgen.py` and the
  assumptions -- so `Adjust` prunes only while the map is over its room count.
- Portals, treasure rooms, secret rooms and quests are not placed. `portalsCount`
  and `minPortalDistance` are `GenerationParams` fields with no `Levels.json`
  column behind them, so there is no number to place them from.
- A talent shop is placed when a row asks for one, and no Default row does, so
  the consumable shop it would carry never appears. See the treasure section.
- The per-shop predefined pools in `Shops.Items` (`i1`, `i2`, `i3`, which pin
  the first shops to a fixed list) are parsed but not used; every item shop
  rolls from the general pool.
- A bought item goes to whoever gains the most Power by it. The game lets the
  player drop it on anyone, and `RunState.apply` takes an explicit target, but
  the run-level action space does not enumerate targets.
- The game levels a unit the instant a kill lands, so a long fight can be won by
  a squad that grew during it. Here the fight resolves first and the levels land
  after.
- Consumables, quests, shrines other than mutation, and the `Squads.json`
  starting presets are not wired in.


## The agent-level passives

The mutation table's long tail is not a pile of one-off mechanics after all.
The game has a name for most of it: `C_PassiveSkillMutation<TMutation, TSkill>`,
a mutation that installs a `C_PassiveSkill` on every unit its `Class` selector
covers. 54 mutation types derive from it. Each of those skills hangs off one of
four hooks, and the hooks are what `sim/mutations.py` now reproduces:

| hook | when | reaches |
|---|---|---|
| `C_PassiveSkill.OnDamageCreated` | this unit's damage is being built | the victim, or the attacker |
| `C_DamageReactionSkill.OnDamageInternal` | this unit took damage | whoever hit it |
| `C_OnSkillCastedSkill.OnSkillCasted` | this unit cast a skill | itself |
| `C_BaseOnDeathSkill.OnDeath` | this unit died | its team, or the enemies nearby |

**Both damage hooks share one gate**, and it is the same code in
`OnDamageCreated` and in `OnDamageInternal`:

    (damage.type & skill.damageType) == skill.damageType   a SUBSET test, so an
                                        absent damageType matches everything
    damage.type & Secondary            -> never fires
    the other unit is alive; for the reaction hook it must also be an enemy
    PseudoRandom.Get(chance)           the roll; an absent chance is always

`Secondary` is `DamageType` bit 0x40. **Splash and knock-on damage trigger no
passive at all** -- which is worth stating because the obvious implementation
gets it wrong and doubles every on-hit effect on a splashing unit.

### What each one does, and where the number came from

- **BuffAttack** (76 of the 1,094 offers, the largest single entry on the
  shelf). A timed `M_StatBonusStatus` on `castTarget`: `Target` for the slows
  and armour shreds, `Source` for the rows that stack armour on the attacker.
  It is the only status that numbers its stat pairs -- `stat1`/`value1`/
  `percentage1`, `stat2`/... -- which is why a reader written for the
  unnumbered shape sees nothing.
- **ClassDiversity** (48) and **ClassDiversityMagicalDamage** (24).
  `TryFindNewTarget` takes `RND.Element` of the mutation Class's group, so the
  bonus lands on **one randomly chosen unit**, not on the class; and it is
  `bonus x count(otherClass)` -- the method converts the other group's unit
  count to a float and multiplies. Warrior/Novice at +300 flat is +900 Damage on
  one Warrior in a squad with three Novices. The game re-picks when that unit
  dies; this applies it once, before the fight.
- **FearsomeAttack** (24) fears, **PassiveStun** (24) stuns. Both statuses
  already existed.
- **CSLinkStatBonus** (24). Not "scales with class-skill level", which was the
  old guess. `C_CSLinkStatBonusSkill.OnCasted` walks the class skill's targets,
  keeps the ones whose class is `targetClass`, and buffs them for the skill's
  own duration -- so it rides on the Monk's spirit link.
- **Craggy** (20). A damage reaction that applies `DebuffType` to the attacker.
  The type is Stun or Silence; the shipped row without one is a 30% reaction to
  physical damage, which would do nothing whatever if the default were `None`,
  so it is read as Stun and recorded in `assumptions.py`.
- **BuffOnCasted** (20), **MultiCast** (12). On a cast: a self buff, or a
  `chance%` repeat that refunds `saveManaPercent` and `saveCooldownPercent` of
  what the cast charged. The shipped MultiCast row refunds 100% of both.
- **ManaBreak** (12) burns mana off the target. **Untouchable** (12) is a damage
  reaction, not "conditional immunity": its effect address is
  `Effects/Prefabs/slower-debuff` and it slows whoever hit it.
- **BuffOnDeath** (12) walks the dead unit's own team with no radius term;
  **StickyBlood** (8) takes an `enemy` inside `radius` of the corpse.
  **ResurrectionChance** (12) rolls once and comes back at `healthValue`.
- **Compensation** (8). `SetBonus` counts the allies under `healthThreshold`
  percent of their maximum and multiplies the row's bonus by that count, so it
  grows as the squad is worn down. Recomputed every tick, held as one status
  that is replaced rather than stacked.
- **ModifyDamage** (8) adds to one damage type only.

**Revenge is verified and still not implemented**, which is the useful kind of
gap to record: `OnDeath` picks the allies inside `radius` and applies a
`M_StatBonusStatus` with `amount`, but `M_RevengeSkill` carries only `amount`
and `radius` -- the *stat* lives on the status prefab, not in the shipped data,
so there is no number to implement it from. `PlagueAntiHeal` turned out not to
be heal inversion either: its `Init` swaps `M_Unit.classSkill`, so it is
class-skill plumbing.

### Two silent bugs this found, both about who owns a hit

Both were invisible while nothing keyed off the attacker.

**A projectile did not carry its source.** `Battle._step_projectiles` called
`hit()` with no attacker, so a ranged unit's vampirism, fury stacks and
knockback had never fired -- and every on-attack passive would have been dead
for Shooter and Mage, which is most of the classes the BuffAttack rows name.
`M_Damage` carries its source unit whatever delivered the hit, so `Projectile`
now carries one. The Rust core had the same hole and got the same field.

**Two class skills did not credit their caster.** `effect_magic` and
`effect_bomb` called `hit()` with no attacker while the Rust core's `K_DIRECT`
and `K_AOE` both passed the caster index -- so the two engines had been
disagreeing about the Mage and the Thrower all along. Python now matches the
core.

### Two more, found by porting the hooks to the Rust core

Both were in the oracle, and both were invisible until something else had to
reproduce them exactly.

**Panic did nothing.** `IsPanicked`, `PanicInPlace` and `PanicFollowPath` live
in **NC-Fear**, which is a tree of its own that the runtime overlays on a
panicked unit. It is not inside NC-BaseUnit, so `build_unit_tree` never reaches
those leaves: applying panic to every enemy for 600 ticks changed their damage
by exactly nothing. It is enforced in `Battle.step` now, next to where the base
tree stops a stunned unit, doing what NC-Fear does with the fleeing left out.

**FearsomeAttack had no chance roll.** The shared gate reads `chance` with a
default of 100, and `M_FearsomeAttackSkill` names its property `percent`, so a
10% fear fired on every single hit. The model implements `IProbable` explicitly
and its `IProbable.get_chance` is a bare `jmp` to `get_percent`, so the two are
the same number under two names. `mutations.CHANCE_PARAM` records the mapping
rather than special-casing it at the call site.

### And one in the Rust core, from porting the on-death skills

The core took `living` once at the top of a tick, so an agent killed by an
earlier agent in the same tick was still walked through the action loop and got
**one free swing from beyond the grave**. The oracle stops it in the tree --
NC-BaseUnit's first branch is `ShouldDie -> Die`, which sets the intent to stop
and never reaches the action Selector.

It had been there since the action system was ported and nothing noticed,
because it only ever moves a fight by one swing and a swing is inside the noise
of a 6v6. A death effect is what made it legible: three broadswords in contact
with a 1 hp FireHead, both engines finishing on the same tick, the oracle
reporting 194 damage from team 1 and the core 299 -- exactly one FireHead swing
apart. Details in `notes/rust-core.md`.

### And `EmptyAction`, which had been failing since the trees were wired

NC-Skill's cycle ends on `BinarySelector<IsRunningOut> -> EmptyAction`, and
`EmptyAction` sat in `NOT_MODELLED`, which returns FAILURE. So the whole skill
subtree failed on the tick it finished, the base tree's Selector fell through,
and **every skill in the sim yielded a tick to the default attack the moment it
completed**. NC-DefaultAttack has no such tail, so only skills did it.

`EmptyAction` is NodeCanvas's own no-op: `EmptyAction.OnUpdate` sets the node's
status to Success. It succeeds here now, and a skill whose cooldown is shorter
than its animation cycle simply repeats -- which is what a Rat with a 0-cooldown
Panic should do rather than alternating panic and a 1-damage bite.

It surfaced from porting the status casts, where the two engines disagreed about
exactly that Rat. Fixing it cut the oracle-versus-core tick error from 15.96% to
13.82% and the damage error from 12.66% to 10.92%.

### Coverage now

| | offers on the shelf | share |
|---|---|---|
| before | 271 of 1,094 | 25% |
| after | **603 of 1,094** | **55%** |

344 offers moved, and mutation *definitions* went 50/155 to **85/155**. The
same names exist in `Skills.json`, so per-unit skill coverage moved with them:
**92% of uses** (was 90%), and `tools/unit_skill_matrix.py` reports 48 classes
firing, 4 explained by a real gate, 0 unexplained.

What is left outside is now a coherent list rather than a tail: the economy
mutations (food, gold, rerolls), taunt and threat, the projectile shapes
(bouncing, chaining, piercing), and the class-skill variants -- plus `ClassSkill`
and `SwapAttack`, which grant or replace an action rather than modifying a unit.


## Vampirism and Evasion, read out of the binary

Both were registered as `passive` and counted as implemented, and one of them
did nothing at all. They are worth writing up together because the failure and
the fix are the same shape: the model class names its properties, and the sim
was reading names the data never uses.

### Vampirism healed exactly nothing

`M_VampirismSkill` carries three properties: `damageType`, `value` and
`percentage`. Every row that uses it names the number **`value`**: the four
mutation rows (23, 263, 291, 377) and both `Skills.json` rows (114 on Tomato2,
140). The sim read `percent`/`Percentage`, which no row has, so
`param(..., default=0.0)` returned zero and the heal never fired -- in the
oracle, and therefore in the Rust core, which had faithfully ported a field
that was always 0.

`C_VampirismSkill.Addon.OnApply` is the whole mechanic:

    damage is non-zero, the victim is alive, the attacker is alive,
    and neither carries an excluded trait          (traits masks 0x401, 0x2c09)
    percentage -> heal = value * damage / 100, capped at the victim's health
    otherwise  -> heal = value                     (the FixedVampirism rows)
    heal /= 1 + len(damage's healer list); each of them and the attacker is
    healed that share

Only row 263 sets `percentage: false`, and the game's `Meta` calls that variant
FixedVampirism, so the unset majority is a percentage share: that is the reading
`vampirism_percentage_default` records. Nothing in this sim fills the healer
list, so the attacker keeps the whole heal (`vampirism_shares_with_healers`).

The victim-health cap is real and worth keeping: a 20% vampirism swing into a
target with 3 hp left heals 3, not 20% of the damage rolled.

### Evasion had a chance and nothing else

`M_EvasionSkill` has `chance`, `healthThreshold` and `jumpBack`, and
`C_EvasionSkill` splits them across two methods.

`OnHealthChanged` is four instructions of arithmetic and one `setae`:

    _active = healthThreshold >= health / totalMaxHealth * 100

so a row with a threshold is **off until the unit is hurt**. Two shipped rows
use it -- mutation 221 (35% under 30% health, the variant `Meta` calls
LastSecondChance) and the Novice row -- and the sim had been giving both of them
away at full health. `healthThreshold` is a `[DefaultValueAttribute]` the dump
does not print; 100 is the only value that leaves a plain evasion always on,
which is `evasion_health_threshold_default`.

`OnTryToEvade` gates on the damage itself before it rolls:

    the damage is not already evaded
    _active
    damage.type & Physical                        (`test al, 1`)
    damage.type & (Secondary | CantBeEvaded) == 0 (`test eax, 0xc0`)
    PseudoRandom.Get(chance)

**Splash cannot be dodged, and neither can magical damage.** The sim used to
roll evasion against everything, including its own secondary splash damage.

On a success, `jumpBack` sends the dodger away from the attacker with
`M_Knockback.Get`, whose two literals read out of the DLL as **speed 150,
acceleration -280** -- the same shape as EventualKnockback, so it reuses
`Battle.push`.

### One controller per source, so they stack

Both are `C_PassiveSkill`s, and a unit gets one controller per skill: a squad
carrying two evasion mutations rolls twice, and two vampirism sources heal
twice. The sim kept a single `max()`-ed number for each, which is the wrong
shape for both. `Agent.evasions` and `Agent.vampirisms` are lists now, and the
Rust core takes them as passive rows (`P_EVASION`, `P_VAMPIRISM`) rather than
spec columns for the same reason.

### What they are worth

84 of the 1,094 shelf offers are these two (48 Evasion, 28 Vampirism, 8
StatusEvasion), and 11 unit rows carry them as unit skills. In
`tools/mutation_value.py` at 2,000 paired seeds the shelf is worth +0.13 levels
[+0.09, +0.16] and the agent-level passives now account for **+0.08 of it
[+0.06, +0.09]**, against +0.06 before -- part of which is the control itself
widening, since "passives off" now also strips evasion, vampirism, crit, fury
and the projectile shapes instead of only the mutation hooks.

The heuristic baseline moved with the change: 2.325 to **2.283** over 240 seeds,
because the enemies collect these too and evasion on the enemy side is worth
more than vampirism on ours.

### The same bug was in two other passives, and both are fixed

Auditing every scalar branch of `apply_to_agents` for the same failure (a reader
naming a key the data does not use) found two more. `CriticalStrike`,
`FurySwipe` and `CooldownReduction` were checked the same way and read their
rows correctly.

**ExplodeProjectile** named its share `damage` in every row (100, 15, 80 in the
mutations, 40 on Dalek2) and the sim read `percent`, whose default is 100 -- so
rows 314 and 668 ran 6.7x and 1.25x too strong, and row 668's `Chance: 30` was
ignored outright. Worse than the numbers, the *shape* was wrong: it had been
folded in with `Splash` as a percentage of the triggering hit.
`ExplodeAddon.OnApply` passes `M_ExplodeSkill.get_damage()` straight to
`M_Damage.Get` with nothing multiplied by anything, so the explosion is a **flat
damage of its own**, dealt to every enemy within `radius` of the victim, the
victim included. `M_Damage.Get` is called with type 2, Magical -- the same first
argument that is 8 in the vampirism heal -- so the explosion cannot be evaded
and cannot chain into another explosion, but it does feed the on-attack
passives. `C_ExplodeProjectileSkill.OnDamageCreated` gates it on Physical,
non-Secondary damage and then rolls `chance`.

**EventualKnockback** named its numbers `speed` and `acceleration` (200 and 240)
and the sim read `knockbackSpeed`/`knockbackAcceleration`, so every row fell
back to the 180 / -350 defaults; its `chance` (100 and 20) was ignored, so the
20% row fired every time. The older `PassiveKnockback` row does use the
`knockback`-prefixed spelling, which is presumably where the reader came from,
so both spellings are read now. `M_EventualKnockbackSkill` also carries a
`radius`, and `KnockbackDamageAddon.OnApply` walks the victim's whole team when
it is set -- skill 27, on BladeMailer, has radius 30 and had been knocking back
one unit instead of a group. `KnockbackType` (Push against Fly) is still not
distinguished; that is `knockback_type_is_uniform`.

Both are per-source lists now, for the same reason evasion and vampirism are:
one controller per skill.

Where they appear: 31 of the 1,094 shelf offers (16 ExplodeProjectile, 15
EventualKnockback), plus Dalek2's explode, BladeMailer's knockback and both
Despot2 forms'. The run-level baseline did not move at all -- the heuristic
stays at 2.283 over 240 seeds and the mutation shelf at +0.13 levels -- because
those are rare rows in a run that mostly ends on level 2 or 3. The fight-level
effect is real; the run-level one is below measurement.
