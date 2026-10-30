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
repeating a recent draw; the sim draws uniformly. (Read and modelled since,
see "The anti-repeat weighting, read (2026-10-30)" below.)

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
closes them into cycles. (Corrected 2026-10-30: `CheckNeighborCount` has no
caller, so rooms with four neighbours are allowed and common, and `Adjust` and
`CloseLoops` work differently from this description; see "The generator's last
two steps" below.) The boss room is what `CouldBeFinishRoom` allows --
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
  for rooms that cover several squares. (Settled 2026-10-30: `RoomNode` and
  `M_Room` each hold a single grid position, `CreateRooms` makes one node per
  `Yes` cell and links it to the cells beside it, and `get_squares` counts the
  2x2 blocks of rooms around its one cell. A room is one square in the game
  too, and the assumption is gone.)
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
`vampirism_percentage_default` recorded, and the model's default confirmed it
(see "The model defaults, read" at the end of this file). Nothing in this sim fills the healer
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
which is what `evasion_health_threshold_default` assumed and the model's
default turned out to be.

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


## Every room has a fight, and the shop opens after you win it

Read out of the binary on 2026-09-02, prompted by a player saying the sim did
not match the game. It does not, and this is the largest fidelity gap the
project has found: **`sim/run.py` fights only in `fight` and `boss` rooms and
treats every shop as peaceful.** The game fights in all of them.

### The evidence

`M_Room` carries, on every room:

    RoomType type        Quest questType      RoomState state
    int fightResult      bool wonMainFight    M_Shop shop, shopStorage
    float expectedPower  float goldReward     M_Shrine shrine
    bool activatesSecret bool secretIsActive  Biom biom

`wonMainFight` sitting next to `shop` is the shape of the thing. Then
`C_Room.AfterFight(bool win)` (`off=0x6b5b40`) calls, in order:

    M_Room.set_fightResult
    C_Room.OpenAll                 the doors open
    V_Shrine.Open                  the shrine opens
    M_Room.get_goldReward -> C_ResLog.RoomReward -> M_Room.set_goldReward
    M_Room.get_shop, get_wonMainFight, get_type -> M_Shop.set_open
    SubscriptionToken.Unsubscribe

So winning the room's fight is what opens its doors, its shrine and its shop,
and what pays its gold. `C_Room.InitShop` reads `M_Room.get_fightResult` when it
decides the shop's initial `set_open`, which is the same gate from the other
side, and `RoomType` carries an inactive variant of every shop for exactly this:

    Inactive = 8192   InactiveItemShop = 8196    InactiveFoodShop = 8200
    InactiveTreasure = 8256   InactiveShrine = 8384
    InactiveMutationShop = 8512   InactiveTalentShop = 12352
    InactiveConsumableShop = 24576

each being `Inactive` OR'd with the shop's own bit. `RoomType` is a flags enum
throughout, which is also why `Shrine = 192`, `MutationShop = 320` and
`TalentShop = 4160` all carry the `Treasure = 64` bit.

`C_Rooms.ChoosePacks` (`off=0x6bd6a0`) then settles it. It walks **every** room
in `M_Rooms.rooms`, looks the room up in a `Dictionary<string, M_EnemyPack>`,
and on a miss logs the error string **"No enemy pack is set for room "**. A room
without enemies is a bug in the game's own view.

### The consumer of `expectedPower`, found at last

`M_EnemyPacks` holds `packsByID`, `packsByRoom` and
**`packsByType: Dictionary<string, Dictionary<RoomType, List<M_EnemyPack>>>`**,
and `M_EnemyPack` has `ID`, `minPower`, `maxPower`, `enemiesByZone`. So a room's
`expectedPower` selects among the packs registered for that room's type. This
retires the note in `sim/assumptions.py` that `expectedPower` is set for every
room including shops and its consumer could not be found: the consumer is the
pack chooser, and it is set for shops because shops fight.

The old objection to using it, that a first food shop of 1,389 Power against a
750-Power squad makes level 1 unplayable, was reasoning from the wrong premise.
It is not an extra fight bolted onto a peaceful room; it is the room's own fight,
and the level's Power budget is spread over rooms accordingly.

### What we are throwing away

`EnemyPacks.json` ships 212 rows over 135 packs, by room type:

| RoomType | rows |
|---|---|
| Boss | 55 |
| Treasure | 53 |
| ItemShop | 47 |
| FoodShop | 32 |
| Default | 25 |

`RunState.enemy_specs` filters `p.get("RoomType") in (want, None)` with `want`
being `Boss` or `Default`, so **132 of the 212 rows, 62% of the shipped enemy
packs, have never been used by this project**, precisely the ones for the rooms
we decided were peaceful. There is no Start pack, which is consistent with the
start room not fighting.

Rows sharing an ID differ in `Min`/`Max` (78 of 135 packs are uniform, and pack
4 is `Mancrack e1 1-20` with `RangeBlinker e2 1-1`), so those are per-class
counts per zone, which is how `enemy_specs` already reads them. That part is
right; only the room-type filter is wrong.

### Also found, and unrelated to rooms

`C_Fight` carries a fight-escalation schedule the sim does not model at all:

    _MIDDLE_FIGHT_TIME = 120        _MIDDLE_FIGHT_TIME_FINAL_BOSS = 1200
    _LONG_FIGHT_TIME = 180          _LONG_FIGHT_TIME_FINAL_BOSS = 1320
    _LONG_TIME_WITHOUT_DAMAGE = 8   _LONG_TIME_WITHOUT_DAMAGE_FINAL_BOSS = 12
    DELAY_SINGLE = 0.5              DELAY_KOH = 3.5

with `_enhanceStatusController` over `M_DamageBonusOverTimeStatus` and
`_exhaustStatusController` over `M_DyingDotStatus`. So a long fight is escalated,
first by a damage bonus over time and then by a dying damage-over-time, rather
than being cut off. `sim/assumptions.py` caps a fight at
`max_fight_seconds = 120` and calls it a draw, which is the same number wearing
the wrong meaning: 120s is where the game starts *pushing*, not where it stops.

### Quests

`C_Rooms.CreateQuest` walks the rooms, reads each `type`, resolves a quest name
through `EnumUtils.ToEnum<Quest>` and reflection over `"M_" + Name` and
`"C_" + Name + "Quest"`, reads an `"outcomes"` array, and sets
`M_Room.questType`. The `Quest` enum has eleven entries (Sci, Reaper, Nurgle,
Pit, SecretRoom, Cube, Spider, Rat, Flight, Sword, Bot) with a `QuestStatus` of
Accepted, Declined, Acquired or Finished, and `Quests.json` ships alongside.
None of this is modelled. Whether a quest room fights is still open: there is no
Quest pack in `EnemyPacks.json`, which suggests not.

### Still open after this pass

- Whether the first room of a level after level 1 is a shop, and whether a level
  transition always runs an event. Both are player-visible claims that this pass
  did not settle.
- `RoomType.Portal`, `Secret`, `PermanentShop`, `ConsumableShop`, `QuestExtra`
  and `FinalBoss`, none of them modelled.
- `activatesSecret` / `secretIsActive`, with `C_Rooms.roomsToActivateSecret = 2`
  and `MaybeActivateSecretRoom`.
- `C_RoomCleanerConsumable.Use` calls `AfterFight` directly, i.e. it clears a
  room without fighting it.

## Phase 2: the level rows, the first room, and the level-entry dialog

### The multipliers say the shops fight, independently of Phase 1

| level | rooms | item | food | shrines | reroll | gold | power | I / F / T / D / B | quest | secret |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 | 2 | 1 | 1 | 0 | 10 | 1,600 | 1.2 / 0.75 / 1.2 / 0.5 / 2 | | |
| 2 | 10-12 | 3 | 2 | 0 | 1 | 11 | 3,500 | 1.3 / 0.8 / 1.2 / 0.7 / 2 | Sci, Pit, Cube, Rat | |
| 5 | 10-15 | 3 | 2 | 0 | 1 | 12 | 14,000 | 1.1 / 0.9 / 1.2 / 0.8 / 1.6 | | |
| 6 | 10-15 | 3 | 2 | 0 | 1 | 13 | 28,000 | 1.1 / 0.9 / 1.2 / 0.8 / 1.6 | Reaper, SecretRoom, Spider | |
| 7 | 10-14 | 3 | 2 | 0 | 1 | 13 | 50,000 | 1.1 / 0.9 / 1.2 / 0.8 / 1.6 | | **true** |
| 10 | 10-14 | 3 | 2 | 0 | 1 | 15 | 85,000 | 1.1 / 0.9 / 1.2 / 0.8 / 1.6 | Nurgle, Flight, Sword | |
| 12 | 14-18 | 6 | 4 | 0 | 1 | 18 | 140,000 | 1.1 / 0.9 / 1.2 / 0.8 / 3 | | |

**`ItemsMult` is 1.1 to 1.3 and `TreasureMult` is 1.2, against a `DefaultMult`
of 0.5 to 0.8.** A shop or treasure room is a *harder* fight than a plain room,
by a factor of about two at level 1. We have been reading those columns as gold
weights only. They are the room's share of the level's Power as well, which is
what `PowerPerRoom` is for, and no reading of them makes sense if those rooms are
peaceful.

Also in the rows and not modelled: `Quest` is a shortlist per level and only
levels 2, 6 and 10 have one; `SecretRoom` is true only on level 7; `Shrines` is 1
at level 1 and 0 after, while `RerollShrines` is 1 from level 2 on, so the shrine
family after level 1 is the **reroll** shrine, which we conflate with the plain
one. `StatShops` is 0 on all twelve, which is the earlier finding that
consumables are unreachable in Default play.
(**Wrong, 2026-10-29**: every boss room but the last builds a consumable
shop, and the mutation shop's placement is right; see "What `C_Room..ctor`
builds" at the end of this file.)

### `FightInFirst`, and the first room of a level

`Rooms.json` carries a key we never read: **`FightInFirst: true`**, alongside
`Explored: false` and `InstantTransitions: false`. It is not in
`GenerationParams` and neither `FromFixedData` nor `FromGeneratedData` mentions
it, so it took a data xref to find. `tools/data_xrefs.py` is new for this: it
finds rip-relative references to a string literal, which `tools/xrefs.py` cannot
do because it only follows `call rel32`.

Both references are in `C_Rooms.<Init>d__21.MoveNext`, and the code reads:

    mov  rdx, [rip + ...]      ; "FightInFirst"
    call ...                   ; look it up in the level data
    test al, al
    je   0x180a62c94           ; not set -> the no-fight path
    ...                        ; set -> "HideLayout", V_Room.ShowLayout
  0x180a62c94:
    mov  dl, 1                 ; win = true
    call C_Room.AfterFight     ; resolve the room as already won
    call C_Rooms.SetCurrent

So when `FightInFirst` is absent or false, **the level's first room is marked won
without a fight**, which by Phase 1 opens its doors, its shrine and its shop and
pays its gold. When it is set, the room is shown and fought normally.

The shipped fixed map sets it true. No generated level row carries the key at
all, so a generated level falls down the no-fight path and opens its first room
for free. That is a per-level data switch, not a universal rule, and it matches
the player report that the first room of a level is not a fight, with level 1 on
the fixed map as the exception.

### The level-entry dialog is real, and it is not flavour

`C_Levels` owns the transition: `ChangeLevel`, `WinLevel`, `IsLastLevel`,
`LoadLevelRooms`, `DetermineLevelStyles`, and **`StartDialog(bool externalCall)`**
with an `OnDialogsClosed` beside it. `StartDialog` reads the session mode, builds
an `RND`, takes `Services.Dialogs`, picks one and **`RemoveAt`s it**, so a dialog
is used once per run, and falls back to a `"NoDialogsPlaceholder"` when the pool
runs out. `C_ResLog.NewLevel(level, dialog, roomCount, quest)` logs the chosen
dialog by name as part of a `NEW_LEVEL` event, which is how we know the dialog
belongs to the level change rather than to a room.

`M_Dialog` is not a cutscene:

    float weight       string title      string text
    string[] choices   M_Dialog[][] outcomes
    M_Event[] events   IList<M_UnlockEntry> unlocks
    bool[] enableds    bool[] hiddens    float[] weightSums, float[][] cSums

Weighted choice, branching outcomes, attached events and unlocks. So **every
level change presents a weighted choice with gameplay consequences**, and the
sim models none of it.

### The dialog table, found

It is `EncryptedMainGroup/dialogs.json`, 115 KB, **beside `DB/` rather than
inside it**, which is the whole reason the earlier sweep of
`EncryptedMainGroup/DB` came up empty. It is a flat list of 96 entries, each
`{title, body}`, where `title` is a localization key of the form
`Dialog.<Name>.Title`. Every string in the file is a localization key; the text
resolves against `EncryptedLocalizationsGroup/Languages/en.json` (2,944 keys)
and reads correctly, which is how we know it is the right table.

**The pool is `Game.Dialogs`**, and it names 46 of the 96 by bare name. All 46
resolve to an entry, with nothing missing in either direction. The remaining 50
are menu, sign-in, win, loss, Arcade and King of the Hill dialogs. `WinDialog`
is `"Win"` and `LossDialog` is `"Loss"` in every ruleset that sets them.

Per-ruleset pools, which matter for batch 3:

| ruleset | pool |
|---|---|
| Default and the 21 other rulesets that set it | the same 46 |
| `Default/WithoutFood`, `ShortPvP/WithoutFood` | 31, the same 15 dropped from both |
| `Arcade`, `KingOfTheHill` | empty, so no level-entry dialog at all |

Three entries, `Chess`, `PlayersChoice` and `Supermarket`, are in no pool and
are referenced by no other table or binary string. Treat them as dead content.
`DejaVu` is not dead and not a level-entry dialog either: it belongs to
consumable ID 10, `DejaVu`, so **the dialog system has a second entry point
through consumables**, which batch 1's consumables item has to account for.

The serialized grammar, counted over all 96 entries:

    body     text, choices, events(22), comment(1)
    choice   button, outcomes(179), events(80), link(30)
    outcome  text, choices, events(123), weight(67), comment(48)
    event    className, parameters(172)

`choices` and `outcomes` alternate, and the deepest chain is five levels below
the body. `unlocks` appears nowhere in the shipped file, so the `M_Dialog` field
is unused in practice.

**Weights are clean.** Of the 179 choices that carry outcomes, 149 have exactly
one outcome and no weight, and the other 30 carry two or three outcomes with a
weight on every one, never a mix. The weights sum to 100 in 23 of those and to
150 in the other 7, so they are raw numbers normalized by their own sum, which
is what `weightSums` and `cSums` compute at runtime. A single unweighted outcome
is a deterministic continuation, not a coin flip.

**The event vocabulary is 26 classNames** across the 46 pool dialogs (63 across
all 96, but the other 37 are menu, cheat and platform actions). By what they
touch:

| group | classNames |
|---|---|
| resources | `PlusGold`, `MinusGold`, `PlusFood`, `MinusFood`, `GiveExperience` |
| roster | `GiveUnit`, `RemoveUnit`, `CloneUnit`, `GetNames`, `RemoveHealth` |
| inventory | `GiveItem`, `RemoveItem` |
| mutations | `GiveMutation` |
| variables | `DefineVariables`, `AddToVariable`, `MultiplyVariable`, `RemoveVariables`, `CompareVariable` |
| gates | `CheckFood`, `CheckGold`, `CheckItem`, `HasClass`, `HasSlot`, `HasSlotForClones`, `Disable` |
| fight | `Ambush` |

The four heaviest are `GiveMutation` (30 uses), `PlusGold` (29), `GetNames` (23)
and `PlusFood` (16). `GiveMutation` takes a bare `id` into `Mutations.json`, so
a dialog can hand out a mutation outside the shop entirely, and 30 uses of it
across a 46-dialog pool is not flavour.

Gates are what make a choice conditional: `CompareVariable` carries `name`,
`value`, `op` and `hideIfDisabled`, and `HasClass` carries `invert`, so the
runtime evaluates them to fill the `enableds` and `hiddens` arrays before the
choice is shown. Variables are dialog-local state, declared by
`DefineVariables` (`robotsGold`, `numberOfUnitsToGive`, `foodToRemove` are the
three seen) and cleared by `RemoveVariables`.

**One parsing trap:** numeric parameters are inconsistently typed in the
shipped file, so a reader has to coerce rather than assume. `PlusGold.amount` is
an int 19 times and a string 11 times, `PlusFood.amount` a string 14 times and
an int 3, and `MinusGold`, `MinusFood` and even `GiveMutation.id` are mixed the
same way. Every one of those strings parses as an integer, so a plain `int()`
covers the whole file. The single exception is `fromVariable`, which
`PlusGold` takes instead of `amount` in `GuidingStone` and whose value
(`"itemCost"`) is a variable name.

**Dialog text interpolates the same variables.** Bodies carry `{goldLoss}`,
`{foodGain}`, `{chance}` and `{name}` placeholders, plus a pluralizer
(`{goldLoss:token|tokens}`) and a nested localization lookup
(`{class:loc:Classes.{}.Title}`). Nothing in the sim needs to render text, but
it confirms that the variables an event writes and the text a player reads share
one namespace, so a variable is not purely internal bookkeeping.

`tools/show_dialog.py` renders any dialog by name with the text resolved, which
is the fastest way to read one.

### The generation parameters we still ignore

`GenerationParams` confirms `maxDeadEnds` and `minFinishDistance` are real fields
whose values we could not recover, and lists a lot we never implemented:
`portalsCount`, `minPortalDistance`, `rerollShrinesCount`, `questCount`,
`extraQuestRoomCount`, `secretRoomCount`, `questRoomAllowedDoors`,
`questRoomGuaranteedDoors`, `finalLevel`, `mode`, `chip`, `quest`,
`longRunReward`.

### Scorecard against the player report

| claim | verdict |
|---|---|
| a fight in every room, whatever its type | **confirmed**, `ChoosePacks` errors on a room without a pack |
| the shop or shrine appears only after the fight is won | **confirmed**, `AfterFight` calls `M_Shop.set_open` |
| an event on entering every level | **confirmed**, `C_Levels.StartDialog` on the level change |
| no fight in the first room of a level | **confirmed and data-driven**, `FightInFirst` |
| that first room is a shop | **confirmed from level 2 on**: `ChooseStartRoom` makes it `Start | ItemShop` (2026-10-29) |
| quest rooms have no fight | **likely**: there is no Quest pack in `EnemyPacks.json`, but not proven |

## The run does not start from `Game.Team.Packs`

A player reported starting a run with **four** humans, one with a crossbow, one
with a shield, one with a sword and one bare. The sim starts five bare Novices
at 750 Power, on `Game.Team.Packs`, which really does read as five `Novice` rows
with `Number: 1` and every entry of `Team.Items` at 0.

The starting squad is chosen, and the choices are in a file this project never
loaded: `ChipChoice/Squads.json`.

| squad | units | unlock |
|---|---|---|
| **Squad1** | `stone-sword@4,2`, `crossbow@3,4`, unarmed`@2,3`, `shield@5,3` | **none** |
| Squad2 | `hood-dark`, unarmed, `stone-sword`, unarmed, `shaman-mask` | Cultist |
| Squad3 | `football`, `paddle-ball`, `gloves` | Scientist |
| Squad4 | `ring-green`, `lance`, `shield` | Mage |
| Squad5 | `plague-mask`, `chainsaw` | Squad5 |
| Squad6 | `burning-ring`, three unarmed | Squad6 |
| Squad7 | `pretzel`, `bed`, unarmed | Squad7 |
| Squad8 | `gun`, `mad-claws`, `gloves`, unarmed | Squad8 |

`Squad1` is the only one with no `unlockCondition`, so it is the default start,
and it is exactly what the player described. Every unit also carries its own
`cells`, so the game ships the starting **placement** as well as the roster.

**Why this matters more than four against five.** The composition is the point:
`Squad1` has a shield in front, a crossbow behind and a sword, where the sim
fields five identical unarmed bodies. Any early-fight result measured against
five bare Novices is measuring a squad the game never gives anyone, and this is
the third different answer this project has had for the opening roster. It first
read `Team.Items`' keys as a draw pool and opened at ~20,000 Power, then was
corrected to five bare Novices at 750, and both were wrong.

It also bears directly on the room-fight work on the `room-fights` branch, where
every baseline finished at level 1.000: a squad with no frontline and no ranged
unit is the worst possible test of whether a room's fight is survivable.

**How to apply.** `load_ruleset` already takes a `chip` argument and the modes
carry `Chips` (`Crazy`, `Default`, `Easy`, `Hard`, `Hunger`, `Mutagens`,
`ShortPvP`), so the loader has the shape for this; what is missing is reading
`ChipChoice/Squads.json` and starting from `Squad1` with its cells.

Two smaller confirmations from the same report. The pit that takes a human for
food is in every room, and `Game.Food.foodPerSacrifice` is **4**, which is what
`RunState.sacrifice` already pays: that mechanic is right. And `Game.json`
carries `Dialogs`, `WinDialog`, `LossDialog`, `FixedLevel` and `GenerateRooms`
at the top level, none of which this sim reads.

## Fog of war, teleports, and what the observation is quietly leaking

Three player reports, all three checked against the binary. Two are real and
unmodelled, and the third is real but not quite what was remembered.

### The map is discovered, not given

`RoomState` is `Unknown = 0`, `Unexplored = 1`, `Explored = 2`, `Current = 4`,
and `Rooms.json` ships `Explored: false`. `C_Rooms.SetCurrent` is where it
happens: it sets the entered room's `state`, then walks `M_Room.get_neighbors`
reading each neighbour's `get_state`, which is the reveal step. So a level is
uncovered a room at a time, exactly as reported.

**This sim does not model it, and the observation actively hands the policy the
whole map.** `rl/env.py:_encode` carries:

    d_here / far          distance to the boss, from `RoomMap.to_boss`
    len(rooms) / 20       the level's room count
    cleared / len(rooms)  the fraction of the level cleared
    around[base + 2]      whether a neighbour is nearer the boss

`to_boss` is a BFS over the **entire** graph, computed in `RoomMap.__init__`
before a single room is entered, so the agent knows from its first step how far
the boss is from anywhere, how big the level is, and which direction closes the
distance. A player knows none of that. `_move_targets` leans on it too, picking
which portal to expose by `min(..., key=to_boss)`.

This is the first fidelity gap found here that makes the task **easier** than the
game rather than cheaper to exploit, and it is on the agent's side of the line:
every navigation number in `notes/rl.md` was measured by a policy playing with
the lights on. Modelling it means a per-room `state`, an observation restricted
to what has been revealed, and `to_boss` computed over the known subgraph with
some convention for the unknown, which is a design decision rather than a
lookup.

### Teleport rooms exist and no generated level has one

`RoomType.Portal = 32`, `C_Room.CanBeTeleportedTo` / `CanBeTeleportedFrom`,
`C_Rooms.TeleportTo`, and `V_Room.ActivateTeleportCorners` called from
`SetCurrent`. `GenerationParams` carries `portalsCount` and `minPortalDistance`.

`RoomMap` already supports them: `__init__` links every portal room to every
other, and `from_table` reads the shipped `Portals` list. But `sim/mapgen.py`
sets `portalsCount` to 0 and places none, so **the fixed map has three portals
and every level a run actually plays has zero**. The `portal` move exists in the
action space and is never legal.

**Superseded 2026-10-29.** `portalsCount` is written, as `num / 7 + 1`, and
generated levels now carry two to four portals; see "Portals, and
`PositionRooms` read end to end" at the end of this file.

### The vision mutation is `BossVision`, and it reveals the boss

`Mutations.json` has one map-related entry, ID 188 `BossVision`
(`Quest: 1`, comment `RedStringsBar`), and `C_BossVisionMutation.OnNewLevel`
walks every room reading `get_type` and `get_state`. So it fires on each new
level and reveals a room by type: the **boss**, not a random one. The other
room-ish mutation is ID 673 `RoomCount` (`SubType: EmptyRooms`, `rooms: 1`),
which adds an empty room rather than revealing anything.

A mutation that reveals the boss is worth nothing to an agent that already knows
where the boss is, which is another reason the fog matters: it is not only a
difficulty question, it decides whether a whole mutation has any value.

### Implemented, 2026-09-02

The reveal rule is `C_Rooms.SetCurrent` read instruction by instruction, and it
is smaller than it sounds:

    previous current -> EXPLORED
    entered room     -> CURRENT
    for each door neighbour:
        if type carries QuestExtra: skip        # bt eax, 0xa
        if state is not UNKNOWN:    skip        # test eax, eax
        state -> UNEXPLORED

Two details are worth keeping. The neighbour walk is over
`Dictionary<Direction, M_Room>`, so it follows **doors only** and a portal link
reveals nothing, which is why `RoomMap` now keeps `_ortho` beside `_adj`. And
the `QuestExtra` skip means an extra quest room is never revealed by walking
past it, so those have to be found some other way.

**A revealed room shows what it is, and that was checked rather than assumed.**
`V_MinimapRoom` holds `iconsByState`, a `Dictionary<RoomState,
Dictionary<RoomType, string>>`, and its static constructor builds nine icons
under `unexplored` for the nine it builds under `explored`:
`unexplored-shop`, `unexplored-food-shop`, `unexplored-static-shop`,
`unexplored-altar`, `unexplored-inactive-altar`, `unexplored-quest`,
`unexplored-boss`, `unexplored-final-boss`, and the plain `unexplored`
background. There is no `unknown-` icon of any kind. So the fog hides a room's
**existence**, not its type: once a door reveals a room the player can see it is
a shop, and the observation is allowed to say so. That settles what would
otherwise have been a guess about how much the neighbour vector may carry.

The design decision the earlier note left open, what `to_boss` becomes, resolved
this way. `RoomMap.to_boss` stays exactly as it was, a BFS over the whole map,
because the checks and the map generator want ground truth. Beside it is
`known_to_boss`, the same BFS restricted to revealed rooms, **empty until the
boss is found**, recomputed in `set_current` rather than on demand. The
observation reads only the second, and carries a `boss_found` flag next to the
distance, because without it a distance of 0 means "standing on the boss" and
"no idea where the boss is" at once and the two want opposite moves. The
level-wide features became level-so-far features: rooms seen instead of rooms,
cleared over seen instead of cleared over all, pantries among the seen ones, and
a new frontier fraction, which is the only sense of "how much is left" that
survives hiding the level size. `_move_targets` picks its portal by
`known_to_boss` too.

`rl/heuristic.py` navigates on `known_to_boss` as well. A baseline that can see
the boss from the start room is not a baseline for a policy that cannot, and the
heuristic is what every trained agent is scored against.

**The check found a second leak, which is the point of writing checks that way.**
The test rewrites the kind and cleared flag of every room the squad has not seen
and asserts the encoding does not move; it moved by one feature. That feature was
the squad-power-against-this-room scalar, which read `st.room_power()`, and
`room_power` is `CalculatePower`'s share: a total scaled by `len(rooms)` divided
by `level_weight`, a sum over **every** room's multiplier. Faithful for the sim,
and a back door into the size and the shop density of an unwalked level for the
observation. The sim keeps `room_power`; the observation now scales by the level
row's own `PowerPerRoom` against this room's multiplier, which says the same
thing about difficulty and nothing about the unseen map. The check sweeps twelve
seeds and six hops each, 69 states and 217 hidden rooms rewritten, and the
paired check in the other direction confirms a *visible* neighbour still moves
the encoding.

`BossVision`, mutation 188, is now real: `RoomMap.reveal_boss` puts the boss on
the map and `RunState.next_level` calls it when the run holds the mutation. Its
registry note said "UI only", which was true only while this sim handed the
whole map over for free.

`tools/render_run.py` draws the map the agent is actually playing with: unknown
rooms are not drawn and edges into them are not drawn, while the bounds still
come from the whole level so the picture does not jump about as it is uncovered.

**What the fog is worth, measured.** `tools/fog_cost.py` plays the heuristic
over the same 60 seeds twice, once under the fog and once with `--lights-on`,
which reveals the level at an unchanged `obs_dim`:

    fog        3.433  (sd 1.101)
    lights on  3.983  (sd 0.922)
    paired    +0.550, lights on better on 24 of 60, worse on 12

So free sight of the map is worth **half a level** to a policy that navigates on
it. Two things follow. The lights-on arm at 3.983 sits on top of the 3.946
recorded for this heuristic over 240 seeds, which is the cross-check that
`lights_on` really does reproduce the environment as it was, and it means
**every baseline in this repo was measured with the lights on**. And the fog
widens the spread as well as lowering the mean, 0.922 to 1.101, which is what a
harder navigation problem should do.

**Not re-baselined here.** Batch 1 is the interface-breaking batch and the
measurement rule holds: RL numbers are taken at a batch boundary, not per
feature. The above is a heuristic check that the feature changes something, and
it costs minutes rather than hours. `obs_dim` moved from 203 to 210, so every
existing checkpoint is against a different observation and the `roomfight*`
numbers do not carry over.

## What is left, and in what order

"Implement every feature" is the right direction and the wrong target, for
three reasons this project has already paid for.

**Some features provably cannot fire in Default play.** `StatShops` is 0 on all
twelve level rows, `C_Consumables.Add` has exactly one caller
(`C_ConsumableShop.Buy`), and that shop is only built in a `TalentShop` room, so
consumables are unreachable. Implementing them would add code that can never
execute and a maintenance surface that can never be tested. That is not
fidelity.
(**Wrong, 2026-10-29**: the shop is also built in every boss room but the
last, from the `ConsumableShop` bit `ChooseFinishRoom` sets.)

**Every environment change invalidates every agent and every number.** This has
happened four times in a day. The cost is not the implementation, it is the
re-baselining, so features that touch the same measurement should land together
rather than one at a time.

**The thing worth being faithful about is the decision.** A gap matters if it
changes what the agent chooses between or how a fight resolves. Sorted that way:

**Tier 1, navigation. Land as one change.** Fog of war and portals. Portals are
a navigation option and fog is what makes navigation a decision at all; done
separately they cost two invalidations for one result. This also removes an
advantage the agent currently has over a player, which no other item on this
list does.

**Tier 2, fight outcomes.** `C_Fight`'s escalation schedule: a damage bonus over
time at 120s, a dying DoT at 180s, and `_LONG_TIME_WITHOUT_DAMAGE = 8`. Our
`max_fight_seconds = 120` is that number wearing the wrong meaning, and it
changes how every long fight ends. Alongside it, mutation coverage: 259 of 1,094
offers are implemented, which is what caps the shelf at +0.13 and therefore
caps every mutation result this project has measured.

**Tier 3, the decision layer that does not exist yet.** Quest rooms and the
level-entry dialog. `M_Dialog` carries `choices`, branching `outcomes`, `events`
and `unlocks`, and `C_Levels.StartDialog` fires one on every level change.
**The agent currently makes zero choices per level transition where a player
makes one**, and there are eleven quests in `Quests.json` on top. In decision
terms this is the largest hole in the sim, and it is a bigger build than
everything above it put together.

**Tier 4, completeness.** Secret rooms (`activatesSecret`,
`roomsToActivateSecret = 2`, level 7 only), `PermanentShop`, `TalentShop`,
`QuestExtra`, `FinalBoss`, and the three `GenerationParams` fields whose values
could not be recovered. Low decision impact, worth doing when the tiers above
are done.

**Consumables, corrected.** An earlier version of this list said not to
implement them because `StatShops` is 0 and they cannot fire. That was a
decision-impact judgment smuggled into a fidelity backlog: the question for a
faithful sim is whether the game has the feature, not whether the mode we happen
to train on reaches it. `StatShops` is 0 across all eight Default chips
(default, easy, hard, hunger, mutagens, crazy, shortpvp, arcade), but
`KingOfTheHill`, `Arcade` and `Tasks` are separate modes that have not been
checked, and a `TalentShop` room in any of them makes consumables live.
Implement the mechanism, then record where it fires.

### The project goal is completeness, and what that can and cannot mean

The target is every mode and every feature: `Common`, `Default`,
`KingOfTheHill`, `Arcade` and `Tasks`, with Default's eight chips underneath it.
`load_ruleset(mode, chip)` already layers them, so the plumbing exists; what is
missing is the mechanics each one turns on. `crazy` alone is two levels, one of
3 rooms and one of 100.

Two precisions, so "done" is a thing that can actually be reached.

**Bit-exact behavioural equality is out of reach, and this repository already
measured why.** Nudging every start position by `1e-4` and re-running the *same*
engine flips the winner in about 1 fight in 12 with splash, and the cross-engine
error is the same order (see "The residual error is chaos, not logic" in
`notes/rust-core.md`). float32 against float64 over a 120-second fight is a far
larger perturbation than `1e-4`. So identical inputs will not give identical
outcomes, ever, in any implementation that is not the original binary. The
reachable target is **every mechanism present and correct, with outcome
distributions matching** -- which is a real, checkable standard, and the one the
differential testing already works to.

**Some constants are unrecoverable.** `maxDeadEnds`, `minFinishDistance`,
whether a room may cover several squares, the anti-repeat weighting inside a
quality. The mechanism can be exact while the number is a choice. A complete sim
still has `sim/assumptions.py` in it; that file is the honest boundary of the
work, not a to-do list that ever empties.

**The real risk is unverifiable work, not wasted work.** With completeness as
the goal the failure mode is forty features implemented and none of them tested,
which is exactly how `vampirism` sat reading `percent` where every row says
`value`, healing nothing, while being counted as implemented. Every feature
needs a check that it *changes something*, not that it parses. Completeness
without that is completeness on paper.

### The design decision fog forces

`to_boss` is a BFS over the whole graph and four observation features read it.
Under fog there is no whole graph, so something has to replace it, and the
choice changes what the agent knows:

- drop the boss-distance features and let the policy navigate on the local view
  and the room-kind one-hot alone;
- compute distance over the **revealed** subgraph, which needs a convention for
  unreached rooms (optimistic, pessimistic, or a separate "unknown" flag);
- expose a frontier instead: how many revealed rooms are still unentered, and in
  which directions.

These are different agents, not different implementations of one agent, so the
choice wants measuring rather than picking. `obs_dim` moves either way, so it
gets the `--blind-*` treatment: hold the replacement features at zero and keep
them in the vector.

## Portals, and `PositionRooms` read end to end (2026-10-29)

Portals were next in Batch 1, with `portalsCount` noted as never written in the
Default path. It is written. Finding the write led through the rest of
`LevelGenerator.PositionRooms`, and four of its five steps turned out to differ
from the port.

### The parameters

`C_Rooms.Generate` keeps its `GenerationParams` at `rsp + 0x30` (it invokes the
`BeforeGenerationDelegate` with `lea rdx, [rsp + 0x30]`), so every field is a
fixed stack slot and its writes can be read off directly:

    field               slot         value
    portalsCount        rsp + 0x54   num / 7 + 1     0x92492493, add, sar 2: a signed divide by 7
    minPortalDistance   rsp + 0x70   3               a literal
    minFinishDistance   rsp + 0x48   IC_GameMode.GetRoomsMinFinishDistance(num)
    maxDeadEnds         rsp + 0x44   0               zeroed by a movups over rsp + 0x40, never written

`GetRoomsMinFinishDistance` is `IC_GameMode` slot 4. `C_DefaultMode` implements
it as `num / 6 + 2` (`0x2aaaaaab` is a divide by 6) and `C_ArcadeMode` as
`num - 1`. All three were in `sim/assumptions.py` as unrecoverable. They are set
after the delegate runs, so the `RoomCount` mutation's extra rooms feed into
them. `maxDeadEnds` stays open, with its obvious writers ruled out: the one
subscriber to the delegate, `C_RoomCountMutation.OnBeforeGeneration`, adds to
`num` and the shop counts and writes nothing else.

When `LevelGenerator.Generate` returns false, `C_Rooms.Generate` disposes of it
and builds another with the same parameters, with no cap. `Generate` returns
what `PositionRooms` returns, so a placement step that finds nothing rebuilds
the level. The port used to relax the predicate instead.

### The five steps

Every chooser draws uniformly from the rooms that qualify. `RND.Element` is
inlined as `list[rnd.Next(list.Count)]`, so it shows up as a virtual call on
the generator's `RND` followed by an indexed load.

1. `ChooseFinishRoom`: untyped, not an articulation point, a dead end or exactly
   two neighbours, `maxDistance >= minFinishDistance`. The port took the best of
   these by average distance.
2. `ChooseStartRoom` over `CouldBeStartRoom(room, d)`: untyped, not an
   articulation point, at least `d` from the finish, with `d` counting down from
   `minFinishDistance` until a room qualifies. The port took the furthest room.
   On every level after the first the start is `Start | ItemShop` (`or eax, 5`),
   so each later level opens in an item shop. `CalculatePower` tests `Start`
   before `ItemShop` and pays the start nothing, and its weight sum is built from
   the row's counts, so this shop adds nothing to the level's gold or Power.
3. `ChooseDiversePositions(type, count, threshold, null)` once per shop type,
   the type with more rooms first and item shops on a tie. Each draw is at least
   `threshold` from the rooms this call has already drawn; the start, the boss
   and the other shop type are not measured against. A draw that finds nothing
   discards the partial set, and after ten of those the threshold drops by one.
   The thresholds are `num >= 18 ? num / 6 : 2` for item shops and
   `num >= 16 ? num / 4 : 3` for food shops. The port started at the map's whole
   width and measured against every room placed so far.
4. `ChoosePortals(portalsCount, minPortalDistance)`, only when
   `portalsCount > 1` (with a secret room the game retries at shrinking
   distances and needs `secretRoomCount + 2`; secret rooms are not generated).
   The candidates are dead ends that may carry a portal (`roomTypesWithNoPortal
   = 34307` is `Start | Boss | Quest | QuestExtra | Secret`, so shops may),
   topped up to `2 * portalsCount` from the room list, sorted by mean distance
   to each other, and taken greedily at least `minPortalDistance` apart. Then one
   more goes on the most central room that clears the same distance, and if
   fewer than two land the step is undone. On Default levels 5 and 9 (levels 3,
   5 and 9 on a `shortpvp` chip) the start room is made a portal first and
   counts as one of them. The three comparators were read too:
   `maxAverageDistance` and `maxTmpValue` sort descending, `minAverageDistance`
   ascending.
5. `ChooseTreasureRooms`, over the rooms typed exactly `Default` or `Portal`:
   `StatShops` get `TalentShop` (`0x1040`), `RerollShrines` `MutationShop`
   (`0x140`), `Shrines` the shrine. The port had this one right.

Over 300 seeds a level, no shipped row fails placement once, and the portal
counts come out as: level 1 (7 rooms) always exactly 2, levels 2 to 11 mostly 3
with a sixth to a third at 4 where the room count reaches 14, level 12 mostly 4.
The walk from start to boss shortens from 4.2 to 8.3 rooms under the old
placement to 3.4 to 5.7, before any teleport.

### Teleporting

`C_Room.CanBeTeleportedTo` is the whole rule: the target's state is not
`UNKNOWN`, it is not the current room, the current room's `fightResult` is set,
and both rooms carry the `Portal` bit. A target only has to be on the map, so a
teleport can land in a room nobody has entered. `AfterFight` writes
`2 * win - 1`, so a lost fight counts as set.

A teleport costs nothing. `C_Food.Move` has two callers, the door walk
(`RoomTransition.OnTransitBlocked`) and `C_Rooms.InstantMove`, and the minimap
click (`V_MinimapRoom.OnLeftClick`, which calls `C_Rooms.TeleportTo`) reaches
neither. `M_Food.canMove` is read on the door path only, so a starving squad on
a portal can still teleport. `TeleportTo` runs `SetCurrent` and
`C_Room.AfterMoveInto` as a move does, so arriving reveals the room's doors and
fights a room that has not been fought.

The minimap marks portals in every revealed state (`teleportsByState` has
`unexplored-teleport`, `explored-teleport` and `current-teleport`), so the
observation may say which revealed rooms are portals.

### In the sim

- `sim/mapgen.py` runs the five steps in the game's order and regenerates on a
  failed step. `RoomMap.portals` is now filled on generated levels, and a room
  carries `extras` for a second kind (`Room.has`).
- `RunState`: a move goes through a door (`RoomMap.doors`); a teleport is its
  own action, `("teleport", rid)`, free, from `RoomMap.teleports`. A portal link
  used to be a move that charged a feeding. `stranded()` replaces "no legal
  action": a run whose only actions are teleports into cleared rooms with
  nothing to buy is lost, since otherwise free teleports would let a starving
  squad (and an agent) put off a loss for ever.
- `rl/env.py`: four teleport slots after the four directions, in the order the
  portals were revealed (`RoomMap.revealed_portals`). Ordering by name would
  have leaked: a revealed portal's place among all of a level's portals says how
  many unseen ones sort before it. Each door neighbour and slot has a portal
  bit, the room kinds are multi-hot, and a flag says whether the squad stands on
  a portal. `obs_dim` 210 to 258, actions 33 to 36, so every checkpoint is
  stale.
- `rl/heuristic.py` teleports only to an unentered room, to a room strictly
  nearer the known boss, or, when the larder cannot pay for a door, to a stocked
  food shop. On a tie a door wins.

The no-leak check now also flips whether each hidden room is a portal, and has
its own positive control: flipping a visible neighbour moves the encoding.

### What it is worth

`tools/portal_cost.py`, the heuristic over seeds 30,000 to 30,239 on the Rust
core:

    faithful       3.663  (sd 0.903)   teleports 0.80 a run, in 125 of 240 runs
    no portals     3.425  (sd 1.006)   the same levels with the portals taken off
    old placement  3.425  (sd 1.097)   sim/mapgen.py as of da2e518

    portals, paired                 +0.237 [+0.115, +0.360], better on 55, worse on 23
    faithful against old, by seed   +0.237 [+0.057, +0.418]

The Python oracle over the first 120 of those seeds agrees:

    faithful       3.675  (sd 0.950)   teleports 0.81 a run, in 71 of 120 runs
    no portals     3.450  (sd 0.973)
    old placement  3.408  (sd 1.061)

    portals, paired                 +0.225 [+0.045, +0.405], better on 28, worse on 13
    faithful against old, by seed   +0.267 [+0.012, +0.522]

`ChoosePortals` draws nothing from the random stream, so the first pair differs
in the portals and nothing else until the first teleport. The rest of the port
(uniform draws, the shop spreads, a shop at every later start) nets to about
zero for this heuristic: 0.000 on the core and +0.042 on the oracle, unpaired.
This is a check that the features change something, not a baseline; Batch 1 is
still open.

### Read and not modelled

- `ChooseFinishRoom` makes the boss `Boss | ConsumableShop` on every level but
  the last (`or edx, 0x4002`) and the final boss `Boss | FinalBoss`, plus an item
  shop unless the chip is `hard`. `C_Room.InitShop` builds its controller from
  the class of the room's shop model (`M_ConsumableShop` gives a
  `C_ConsumableShop`), so whether that bit produces a consumable shop is decided
  where the models are made, which is unread (`C_Rooms.FromGeneratedData` is the
  likely place). **The note above that consumables cannot be reached in Default
  play rested on `InitShop` alone and is open again.**
- With `longRunReward` set, the level-1 start room is a `ConsumableShop` too.
  (Modelled since, see "The long-run reward" under the boss room section.)
- The quest choosers between the start and the shops, the secret-room branch of
  the portal step, and Arcade's own predicates for the finish and the start.

## What `C_Room..ctor` builds, and what nothing heals (2026-10-29, later)

Reading `C_Rooms.FromGeneratedData` to settle the consumable shop question led
into `C_Room..ctor`, which is where every room's shop model is made, and from
there to three corrections.

### The constructor dispatches on the room bits

`FromGeneratedData` builds an `M_Room` per generator node, copies the node's
type onto it, and constructs a `C_Room(room, levelData)`, handing it the level
data only when the type carries `FoodShop`. The constructor then takes the
first branch that matches:

    type & 8                 FoodShop        C_FoodShop, budget from the level data
    type & 4                 ItemShop        the run's one C_ItemShop
    bt type, 14              ConsumableShop  M_Shop<M_Consumable>, quantity 2 + (~type & 1)
    (type & 0x140) == 0x140  MutationShop    see below
    (type & 0x1040) == 0x1040 TalentShop     C_TalentShop

`ChooseFinishRoom` made every boss but the last `Boss | ConsumableShop`
(`0x4002`), and that is neither a food shop nor an item shop, so **every boss
room before the last builds a consumable shop**: three slots (two in a start
room, where `type & 1` is set), roll cost 0, opened at construction. The note
earlier in this file that consumables cannot be reached in Default play was
wrong.

The run also does not leave the boss room when the boss dies. `C_Fight.AfterFight`
calls `C_Levels.WinLevel`, which marks the level won (and, on level 4 in
Default, sets the persistent flag behind `longRunReward`) and ends the session
only on the last level. The only gameplay caller of `C_Levels.ChangeLevel` is
`V_Pipes.<MoveInternal>`, the pipe in the boss room. So the squad stays on the
level until it takes the pipe, with the consumable shop open and the doors
open behind it; the sim moves to the next level the moment the boss falls.
Both are still to be modelled.

### A generated mutation shop is 3, 1 and 2

The mutation branch reads `ShowCount`, `BuyCount`, `RollCount` and `RollCost`
from the level data when there is any, which for a generated room there never
is, and otherwise uses its own defaults: a shelf of 3 (4 in a secret room, 6 in
Arcade), 1 take, 2 rerolls (0 in a secret room), roll cost 0, and a pool named
`default` (or `secret`). The sim had been using `Rooms.Shops.Mutations.m1`,
`{ShowCount 10, BuyCount 2, RollCount 0}`, which belongs to the shipped fixed
map's one mutation room: three times the shelf and twice the takes.

`C_MutationShop.Roll` needs `gold >= rollCost` and `rollCount > 0`, pays,
decrements `rollCount` unless it is 90 or more, and builds a whole new shelf
(`UnregisterMutations`, then `CreateMutations`). `BaseMutationShop.Buy` refuses
a bought item, a short purse or `buyCount <= 0`, decrements `buyCount` and
closes the shop at zero; with takes left and the shelf bought out, `Buy` rolls a
fresh shelf for free.

What a shelf is drawn from: `C_Mutations.OnLevelChange` rebuilds the available
mutations for the level, per mutation and per class, filtered by the player's
unlocks, and finishes by reserving every mutation the run holds and every one
already on a shelf in the level's shops. `ReserveMutation` takes an (id, class)
out of the pool. So nothing the run holds is offered again. The sim now draws
by id with the same two exclusions; the per-class granularity and the unlock
state are in `sim/assumptions.py`.

The shop has a `reroll_mutations` action, and the env four mutation slots
instead of ten. `tools/mutation_value.py --runs 600`, heuristic taking against
ignoring: **+0.04 [-0.02, +0.10]** levels, 2.2 mutations a run. On the old
shelf the same test gave +0.13.

### Nothing heals a unit between fights

`C_Unit.BeforeFight` sets `health = min(desiredHealth, totalMaxHealth)` only
when the mode is KingOfTheHill, or Arcade in a boss room for team 1.
`C_Unit.AfterFight` refills mana and resets traits, actions and passives, and
never writes health. `M_Unit.set_health` has 22 callers and the `_health`
property is written directly through `ModelExtensions.Set<float>` in 17 more
places (a scan for functions that load `M_Unit`'s statics at `+0x68` and call
`Set<float>`); none of them is a routine heal. Health comes back only from
`UnitExtensions.Heal` (a heal consumable, the Medic's channelled heal, and
`HealAfterFight`, which is the "Regeneration" perk of the meta-progression
grid in `MutationGrid.json`, not a run mutation), from in-fight healing,
resurrection, and mutation 356 `FoodPackHeal`. `CS_Units.LevelUp` and
`CS_Item.ApplyBonus` both rescale it as `totalMaxHealth * ratio`.

So in Default **wounds carry over**, and the sim had been rebuilding every unit
at full health for every fight. `Human.hp` is now the share of health a unit
carries, set from the survivors after each fight and handed to the next;
`RunState.carry_hp = False` is the old behaviour, kept for measuring. A share
rather than a number of hit points is exactly what level-ups and item swaps
preserve. The pit takes the most wounded human. `FoodPackHeal` is implemented:
`UpdateHealth` heals every unit `goldSpent * koefficient / 100` of its health
when a pack is bought.

`tools/hp_cost.py --seeds 240 --fast`, the heuristic on seeds 30,000 to
30,239:

    wounds carried       2.688  (sd 0.746)   lowest health in a run, mean 0.12
    healed every fight   3.688  (sd 0.903)

    paired -1.000 [-1.105, -0.895], better on 1 of 240, worse on 169

A full level, the largest effect any single change has had in this project, and
in the direction of a harder game: the sim was handing out a free full heal
after every fight. It also changes what several things are worth. Heal
consumables, the Medic, vampirism, `FoodPackHeal` and resurrection all heal
something that now stays hurt.

The observation carries the squad's mean and lowest health share; `obs_dim`
217 to 219.

### The boss room, the pipe and the consumables, implemented

**The pipe.** A boss win calls `RunState.win_level`, which ends the run on the
last level and otherwise sets `level_won`. The squad stays where it is, doors
open; `descend` (legal in the won boss room) is the pipe, and calls
`next_level`. Before this the next level started the moment the boss fell.

**The shop.** On every level but the last the won boss room shows three
consumables, drawn from `ConsumablesByLevel` for the level (a consumable
listed twice weighs twice), no consumable twice and nothing the run carries.
`C_ConsumableShop.Buy` never reads the purse and every shipped row costs 0, so
taking one is free and the only limit is room: `M_Consumables..cctor` builds
the inventory size as `new ReactiveProperty<int>(2)`. `C_Consumable.Sell`
removes one and pays `Rooms.Shops.Consumables.SellCost`, 5 gold.

**The fourteen**, from each controller's `Use` (`C_Consumable.Use` spends a
charge of the row's `Count` and the item goes when they run out):

    1  Heal, Direct      one unit +100%                    C_HealConsumable, target = a unit
    2  Heal              every unit +30%                   range 0, so the whole team
    3  RoomCleaner       an unfought, non-boss room won     C_Room.AfterFight(true), the room's exp
                         without a fight
    4  Reserve           three Novices join                kind TeamMember|Human
    5-7 Reserve          one temporary ally for a fight    no TeamMember bit; C_Team.AfterFight
                         (Dalek, FireHead, SwoopBird)      removes temporary units
    8  Resurrection      the corpses in this room revive   C_Room.CreateCorpse, after the fight only
    9  Teleport          a random room nobody has seen     collects UNKNOWN rooms, RND.Element
    10 DejaVu            a dialog                          not modelled (no dialogs yet)
    11 AntiHunger        hunger penalties held off         M_AntiHungerStatus wraps hunger
    12 DamageAllEnemies  -                                 on no Default level's list
    13 Event             a dialog                          not modelled
    14 Reroll            a free reroll of this shop        RoomType.ShopWithChoices

RoomCleaner and Resurrection both check the room: the cleaner does nothing
(and still spends its charge) in a room whose fight has a result or in a boss
room, and the resurrection only works in one whose fight has. The sim has no
stage between entering a room and fighting it, so a cleaner is used beforehand
and spent on the next room entered; `sim/assumptions.py` lists that and the
other choices.

Corpses are kept per room (`Room.corpses`), the observation shows the two
carried consumables with their charges, the shelf, and what is waiting on the
next fight, and the env has `descend`, three `take_consumable`, two
`use_consumable` and two `sell_consumable` actions: `obs_dim` 219 to 281,
actions 31 to 39. The heuristic takes the shelf best first, sells what it has
no rule for, heals below half health (one unit) or 70% (the team), resurrects
where corpses lie, calls reserves at once, and arms a cleaner or an AntiHunger
just before walking into a fight.

`tools/consumable_value.py --seeds 240 --fast`, the shop against the same
seeds with `take_consumable` struck out:

    takes them    3.367  (sd 1.080)   uses a run: heal_one 1.35, heal_all 1.12,
                                      reserve_humans 1.12, reserve_ally 0.90,
                                      cleaner 0.59, resurrection 0.42, anti_hunger 0.36
    ignores them  2.646  (sd 0.704)

    paired +0.721 [+0.597, +0.844], better on 116 of 240, worse on 8

Most of that is healing: with wounds carried between fights, a free heal in
every boss room is the only routine way to get health back.

### The long-run reward (2026-10-30)

`C_Levels.WinLevel` checks `M_Levels.currentLevel == 4` and the mode, and in
Default it sets `M_PersistentData._longRunReward`, a saved flag, to true.
`C_Rooms.Generate` reads it back when it fills `GenerationParams`: in Default,
on level 1 (`params.level == 1`), with any chip but `crazy` (an inlined string
compare), a set flag sets `params.longRunReward` and is written back to false.
`ChooseStartRoom` then ors `ConsumableShop` (0x4000) into the start room, and
`C_Room..ctor` builds the shop with `2 + (~type & 1)` = 2 slots, since `Start`
is bit 1, open from construction. So a run that beats level 4 opens the next
one standing in a two-slot consumable shop, drawn from level 1's list, and the
reward is spent.

In the sim, `RunState.long_run_reward` is the flag as the run found it,
`win_level` on level 4 sets `long_run_reward_earned`, and the boss room's shop
code now asks `consumable_shelf(room)`, which answers for either shop. Carrying
the flag between runs is the environment's: `DespotRunEnv(long_run_reward=...)`
is False by default (a fresh save, and every seed's run independent of the
episodes before it), True for every run, or "carry" for the save's behaviour.

`tools/ablate.py long_run_reward --seeds 2400 --start 40000 --fast --jobs 8`:

    with the reward     3.363  (sd 1.117)
    without             3.216  (sd 1.157)
    paired +0.147 [+0.090, +0.205], better on 915 of 2400, worse on 748

The heuristic beats level 4 in 16% of runs (384 of 2400), so under "carry" a
heuristic player would hold the reward about one run in six and gain about
+0.02 levels from it on average.

## The level-entry dialogs, implemented (2026-10-30)

`sim/dialogs.py` runs the 46 dialogs of `Game.Dialogs` from
`EncryptedMainGroup/dialogs.json`. What was read for it, beyond the schema
section earlier in this file:

- **The draw.** `C_Levels.StartDialog` loops over random picks from the pool,
  asks `C_Dialogs.StartDialog` to start each, and removes only the one that
  starts. A dialog whose gates fail stays in the pool for a later level. It
  runs on every level change, the first included, since `C_Session.Begin`
  calls `ChangeLevel`.
- **The gates.** `C_Dialogs.PrepareEvents(M_Dialog)` walks each choice's
  outcomes and prepares their events with the choice index as `num`, folding
  `C_Event.GetEnabledState` into `enableds` and `hiddens`. So a choice is gated
  by what it leads to: `MinusGold` is a `CheckGold`, which is why "buy for 5"
  greys out on a short purse. The body's own gates decide whether the dialog
  starts (`Stores` needs 5 gold, `Mushrooms` 20 food). Events are prepared in
  order and `DefineVariables` defines while preparing, so a node can define a
  variable and test it (`OneHandBandit`'s `robotsGold`).
- **Links** name a path in the localization-key scheme; a path that ends on a
  choice with several outcomes means taking that choice again, which is how
  `RedStringsBar` and `OneHandBandit` loop back to a coin flip. `#` is the body.
- **The events.** `GetNames` picks random units into the dialog's `units`,
  which `RemoveUnit`, `CloneUnit` and `RemoveHealth` act on by 1-based `id`;
  `RemoveHealth` multiplies health by `1 - percent / 100`, every unit when it
  names none; `RemoveItem` is `C_Player.DownGradeUnit` and leaves the item's
  cost in `itemCost`; `HasSlot` counts the team's cells against 49; `Ambush`
  resets the current room's `fightResult` and closes its doors, so the start
  room's pack attacks (and `CalculatePower` gives a start room `ItemsMult` of
  Power, though no gold). A `MinusFood` with no amount spends the dialog's own
  `CheckFood` amount. `GiveUnit` gives `numberOfUnitsToGive` units when the
  dialog defines it (`Reproductor`: half the team).
- **Consumable 13**, `Event`, opens a pool dialog (`C_EventConsumable.Use`
  calls `C_Levels.StartDialog`). `DejaVu` (10) is a rewind to a save after a
  wipe and is still not modelled.

In the run, a dialog is modal: while one is open the only legal actions are
`("choose", i)`. Nodes with a single choice are taken without spending a step.
The env has four `choose` actions and the observation a flag plus a one-hot over
the 69 decision nodes; `obs_dim` 281 to 356, actions 39 to 43. The heuristic
takes the choice with the best expected immediate value, one node deep, in
guessed gold equivalents (`rl/heuristic.py`, `_EVENT_GOLD`).

A random walk of every dialog from varied states runs each to its end with no
error; 44 of the 46 change gold, food, the squad, the mutations or the items on
some path. `tools/dialog_value.py --seeds 240 --fast`:

    no dialogs         3.367  (sd 1.080)
    heuristic chooses  3.562  (sd 1.146)   5.03 dialog choices a run
    random choices     3.408  (sd 1.162)

    dialogs, heuristic against none     +0.196 [+0.021, +0.370]
    choosing, heuristic against random  +0.154 [+0.012, +0.296]

The no-dialog arm reproduces `tools/consumable_value.py`'s 3.367 to the last
digit, which is the check that turning the dialogs off leaves the rest alone.
The choice matters: about five decisions a run, worth a sixth of a level
between a rough heuristic and a coin.

## The pack chooses the room (2026-10-30)

`M_EnemyPack` carries `layouts`, and `ChooseLayout` takes one of them (at random
among those whose player cells match, logging "Pack id=... does not have any
associated layouts" when there are none). `LayoutsByPack.json` gives every
shipped pack exactly one, so **the pack fixes the battlefield**. Each
`EnemyPacks` row also names a `Zone`, `e1` to `e3`, and every zone a Default
pack's rows name exists in that pack's layout. The 11 packs with no layout are
ids 1001 to 1011, which no Default level draws.

The sim used to fight every room in any of the 77 layouts at random, 35 of
which no pack uses at all, with every enemy in `e1`. Now a fight is fought in
its pack's layout with each enemy in its row's zone; `RunState.pack_layouts`
keeps the old draw for measuring. `SpecialRoomLayouts.csv` (`sci-quest` and
the like) belongs to the quest rooms and waits on them.

`tools/ablate.py pack_layouts --seeds 240 --fast`, the heuristic on seeds
30,000 to 30,239:

    pack layouts   3.300  (sd 1.137)
    random layout  3.562  (sd 1.146)

    paired -0.263 [-0.439, -0.086], better on 63 of 240, worse on 94

The packs' own rooms are harder for this squad than a random room was; why has
not been measured. `tools/ablate.py` is the general form of `tools/hp_cost.py`, for any of the
`RunState` switches that keep an old behaviour for measuring (`carry_hp`,
`pack_layouts`, `dialogs_on`).

## DejaVu, the fourteenth consumable (2026-10-30)

`C_DejaVuConsumable` is never used by hand. `Take` enables it, and `Enable`
saves the run (`PrepareData`), subscribes to `C_Session.OnLoss`, and saves again
whenever a level starts while it is carried. `OnLoss` catches the loss and opens
the `DejaVu` dialog ("Now let's backstep a bit and imagine you NOT losing all
your puny humans"), whose one choice fires the `DejaVu` event, and `Use` loads
the save with `Loader.LoadGameFromData`. So it is a second life: a wipe while
carrying one goes back to the last save.

The sim keeps the save in `RunState.dejavu_save` and rewinds in
`dejavu_rewind` on a wipe (not on its own stranded rule, which the game does
not have). Two choices, both in `sim/assumptions.py`: the DejaVu is spent by the
rewind, and the random stream keeps running rather than being rewound with the
rest. The second matters: with the stream rewound a deterministic policy replays
the level move for move and dies in the same place, which a reloaded game does
not do (`notes/rust-core.md`, a 1e-4 nudge flips one fight in twelve). On the
heuristic over 240 seeds, with the stream rewound the DejaVu fired in 91 runs
and the mean moved only from 3.300 to 3.317 (unpaired, against the run before
DejaVu was modelled); with it kept, 96 rewinds and

    DejaVu on against off, paired  +0.071 [+0.034, +0.107]

The heuristic takes it before anything else on a shelf. All fourteen consumables
but `DamageAllEnemies`, which no Default level lists, now do what they do in
the game. With them all, the shop is worth +0.692 [+0.569, +0.814] to the
heuristic (`tools/consumable_value.py`, dialogs and pack layouts on).

## Quests, all eleven implemented (2026-10-30)

### Where the dialogs were

The quest dialogs are in neither `dialogs.json` nor any table under the
encrypted groups, and only four of their localization keys appear as string
literals in the code. They are NodeCanvas `DialogueTree` graphs serialized into
the quest prefabs, each a `DialogueTreeController` whose `_serializedGraph` is
a JSON string inside a MonoBehaviour of `sharedassets2.assets` or
`sharedassets3.assets`. The MonoBehaviours have no type trees (IL2CPP strips
them), so `tools/dump_dialogue_trees.py` reads the string by hand after the
fixed header and names each graph `<prefab root>.<GameObject>`. 34 come out;
32 belong to the eleven quests (the Spider quest's keys say `Monster`), two to
the despot's own dialogues. The same `CheckQuestStatus : DialogConditionTask`
class in `dump.cs` is what pointed at NodeCanvas.

A graph is walked as NodeCanvas walks it: an action node runs its action list
and goes to its child, a condition node to its first child when the condition
holds and its second when not, a statement to its child, and a
multiple-choice node waits for the player. Every graph has node 0 as its only
root, every condition node exactly two children, and no node more than three
choices, so the existing `choose_0..3` actions cover them. Of the 31 action
types only `SetStatus`, `ApplyOutcome`, `CallFunction` (and its
`CallFunctionAndWaitForEvent` and generic forms) and `CheckEndGame` touch the
game; the rest are camera, sound, animation and input. `ApplyOutcome(i)`
applies `outcomes[i - 1]` through `C_Dialogs.ApplyEvents`, the event engine
the level dialogs already use. `CheckQuestStatus` is an equality test whose
unset parameter is 0 (Uninitialized), and `DeselectNovices` starts a UI
coroutine and returns true.

### Which quest, and when its dialog runs

`DataLoaders.LevelData.Load` splits each `Levels.Quest` shortlist on commas into
a `DynamicWeightedList<string>` (factor 2), keeps it in a static dictionary
keyed by the shortlist's text, and draws one name per level into
`M_LevelData.randomizedQuestNames`; `C_Levels.CreateQuestModel` writes that
quest's row into the level data. The sim draws uniformly. The list's bias works
across the runs of one launch, which independent runs cannot carry; within one
run each shortlist is drawn from once, so the bias never acts inside a run.

`V_Rooms.MoveCamera` calls `V_Quest.Launch` on every arrival in a room that has
a `V_Quest`, and `Launch` runs the dialog `GetDialogIndex(room)` names, none at
-1. Only the quest room gets the prefab: `TryGetPrefabName` answers for
`model.room` alone. An extra room's dialog is started by the controller after
its fight.

### The map

`C_Rooms.Generate` copies the quest's row into `GenerationParams`
(`questCount` 1, `extraQuestRoomCount = ExtraRooms`, `secretRoomCount` = the
row's flag plus `SecretRooms`, the allowed and guaranteed doors, and the
`Quest` enum), and `PositionRooms` places the quest between the start and the
shops:

  * `ChooseQuestRoom`: an untyped room whose doors all lie in `Doors` and which
    has every one of `GuaranteedDoors`, drawn uniformly, except that Flight's
    list is sorted by `pos.x` and its leftmost room is taken. The room gets
    `Quest` (512).
  * `ChooseQuestExtraRoom`, once per extra room: untyped, not an articulation
    point, a dead end or two neighbours, more than one room from the quest
    room; uniform, except Flight takes the rightmost candidate and only when it
    lies left of the quest room and at most three rows off. `QuestExtra`
    (1024).
  * `CountValidPaths(start, quest, QuestRoomCondition)` must be positive: some
    simple path from the start reaches the quest room without stepping on the
    boss or an extra room (`test type, 0x402`).

Any of the three failing fails the generation, which is retried as before. All
eleven quests place on every one of 40 seeds; Flight needs 5.7 attempts on
average, the rest about one. `Direction` maps to the sim's (row, col) with Up a
row less, since `RoomNode.GetName` prints `topBorder - pos.y`.

### The rooms

`C_GameMode.GenerateRoom` gives a quest room its `RoomParams` layout from
`SpecialRoomLayouts.csv` and, only when the block has `Packs`, exactly that
pack: the quest room never fights; the extra rooms of Sci (five level-3
Zombies), Cube (six Daleks), Rat (four Mancracks) and SecretRoom (four
FireHeads) do; Flight's and Bot's do not. `C_Rooms.CalculatePower` leaves both
quest kinds out of its weight total, gives them an `expectedPower` of 0, pays a
quest room nothing and an extra room the plain room's gold share (nothing for
Flight and Bot).

The fog had the rule half done: the walk-past reveal already skipped
`QuestExtra`. The other half is `RoomTransition.CheckDirection`, which refuses a
move into a room still at `Unknown`, so an extra room cannot be entered until
the quest puts it on the map (`OnStatus(Accepted)` sets it to `Unexplored`).
The sim had let a move go through any door, and the baseline walked into the
Sci lab before meeting the scientist in 20 of 120 runs. The environment's
neighbour block now shows a door into an unrevealed room as a door and nothing
more.

### The four of level 2

  * **Pit** (`GetDialogIndex` 0 while Uninitialized or Accepted). Five Novices
    (`classValue == 1`, a human holding nothing) jump for mutation 306; with
    nobody else in the team there is no choice, all five die and
    `CheckEndGame` ends the run. With company the player may let them jump or
    stop them; "throw a couple of tokens" pays outcome 2, 30 gold, on every
    branch but the jump.
  * **Cube**: accepting reveals the extra room; winning it, or arriving in it
    already won, while Accepted plays the extra dialog, which sets Acquired and
    adds `cube-part`. Back in the quest room: 25 gold or mutation 328.
  * **Sci**: the beginning offers two lines that both accept. In the lab, a win
    while Accepted runs `Quest-two-units` (`ReplaceUnit`: `FindWeakestUnit`,
    Novices before the armed, then by level and health or item quality and
    cost, becomes a level-1 Zombie through `C_Player.DownGradeUnit`) or, with a
    team of one, `Quest-one-unit` (`CreateZombie`). Back with the Zombie alive:
    mutation 218 or 25 gold, and the Zombie leaves; dead, nothing. Unlike
    Cube's, Sci's controller has no branch for a lab won before acceptance;
    with `CheckDirection` that cannot happen by walking.
  * **Rat**: accepting adds `rat-flute`; winning the extra room adds the Rat
    (class 79, level 1) to the team and sets Acquired. Back with it: mutation
    233 or 25 gold; without it, nothing.

Nothing removes a companion at the level change: the Zombie and the Rat stay in
the team until returned or killed.

Nine of the eleven quest reward mutations are not modelled by the mutation
layer (230 and 668 are, both level-6 rewards), all four of level 2 among them,
so for now a level-2 mutation reward is a mutation that does nothing, and the
baseline takes the gold.

### Levels 6 and 10, later the same day

Six more controllers, read the same way; only SecretRoom is left, because it
needs the secret room itself (below).

  * **Reaper.** The beginning spawns a hater (a level-1 Novice) in the
    Reaper's room and `Run` sends him through a door at once. `ChooseNextRoom`
    draws uniformly among the room's neighbours, never the room the squad came
    from (`OnRoomChange` stores it as `prevRoom` on every move) and never the
    Reaper's room. Each arrival in his room makes him run again while
    `movesMade < maxMoves` (5, the first escape included) and a room
    qualifies; then he stays and offers 50 gold to be let go. Refused,
    `PrepareToFight` turns the room's fight on him (`OnEnemyTeamCreation`),
    the quest is Acquired, and the Reaper pays mutation 230.
  * **Spider.** The monster grabs a unit and throws it back (a knockback, no
    damage). Accepting hands over three leaflets, and every room the squad
    then enters that is not the quest room and has none yet gets one; the third
    sets Acquired. 50 gold or mutation 346.
  * **Bot.** The bot waits in the extra room, which accepting puts on the map.
    Met there it asks for 5 gold: the option shows only with that much in hand
    (`IsOptionEnabled("Quest.Bot.Meet.3.2")`, the one override of the base
    method), and paid it joins the team. Threatened, it runs like the hater
    (`maxMoves` 3); cornered, it fights (a level-1 `Bot`, 1,101 health), and
    beaten it changes sides (`OnBotDeath` recreates it in the player's team).
    Brought back alive: 50 gold or mutation 22.
  * **Nurgle.** While Accepted every enemy that dies adds a `plant-leaf`
    (`OnDeadUnit` on each new enemy team, no condition on the unit); five buy
    35 gold or mutation 302. `AdditionalLevels` 1 keeps the model into the next
    level, whose map places the room again from `Overrides` (`nurgle-quest-2`)
    with its own not-finished, ending and finished dialogs.
  * **Flight.** On foot, accepting reveals the extra room and its dialog plays
    on arrival. By cannon the status is left alone, the squad lands in the
    extra room and `WaitForArrival` sets each unit's health to
    `health * (100 - hpLossPercent) / 100`, 10% off. Either way the helm, then
    35 gold or mutation 237.
  * **Sword.** The sword's "extra" room is the boss room. Taken, a level-1
    `TalkingSword` (4,000 health) joins the team, and arriving in the boss
    room adds a level-1 `DeathSword` to the boss's pack. After the boss falls,
    35 gold or mutation 358, from the sword or its ghost, and Finished takes
    the sword away. Met after the boss is dead, `CheckBossRoomStatus` sets
    Declined and the dialog ends unpaid.

Three generic hooks came with them: kills reach the quest, a quest can add
units to a room's enemy team, and a quest dialog can be followed by another
(the Bot's after-fight dialog) or by an action (the cannon).

### The secret room, and the SecretRoom quest

`LevelGenerator.Generate` makes `secretRoomCount` secret rooms after the shape
and before `PositionRooms`: it puts every position beside a room back on the
open list, draws by `WeighPositions`' weight, and `MaybeMakeSecretRoom` takes
the first cell one step past the draw, in `RoomsDirections` order (Left, Up,
Right, Down: `C_DefaultMode`'s static array, read from its metadata
initialiser), whose four neighbours and four diagonals hold no room. The room
is `Secret | Portal` from birth, plus `MutationShop | Treasure` unless the
level's quest is SecretRoom, so it has no door and is reached by teleport.
`ChoosePortals` counts it among the portals and skips it as a candidate, and
with one on the level (not the quest's) `PositionRooms` asks for at least two
more portals, retrying at shrinking distances, and fails the level without
them. `FromGeneratedData` flags two of the other portals, drawn without
replacement, `activatesSecret`; `C_Rooms.MaybeActivateSecretRoom` counts the
ones the squad stands in and at `roomsToActivateSecret` (2) switches on every
secret room that is not a quest's, which puts it on the map. `CalculatePower`
gives it no Power and no gold, so `GenerateRoom` gives it its layout
(`regular-secret-room`) and no fight. The `directions` order now drives every
neighbour walk in the generator.

What the level-7 room holds is not settled. Its mutation shop shows four and
takes one with no rerolls, from a pool named `secret`, with the
`goldinsecretroom` treasure in the first slot; no shipped ruleset gives any
mutation that pool or any level a `Treasures` block, so the sim leaves it
empty (`sim/assumptions.py`).

The SecretRoom quest uses its own secret room. The quest room ("a mysterious
room") sets Initialized. A unit sacrificed there (`OnUnitRemoved`) sends the
squad to the secret room (`PlayAnimationAndTeleport`), where the Doctor gives
the quest; `FinishSecretRoomDialog` brings it back. The extra room's fight
(four FireHeads) yields the book, and a second sacrifice takes it to the
Doctor for 35 gold or mutation 668. So all eleven controllers are ported.

### What it is worth

On the heuristic, paired by seed, with the fast core (`tools/ablate.py
quests_on`, and a third arm in which the baseline is told to follow the
quests), first over 240 seeds from 30000 and then, with every quest in, over
2400 seeds from 40000:

                                          240 from 30000           2400 from 40000
    quests on, passed by, against off     -0.121 [-0.286, +0.044]  -0.072 [-0.124, -0.020]
    quests on, followed, against passed   -0.113 [-0.199, -0.026]  -0.028 [-0.049, -0.007]

The first is the map and the economy; on 240 seeds it did not clear zero, on
2400 it does, at about half the size. Its sign has
a plain source in `CalculatePower`: a quest room takes a plain room out of the
weight total while the level's Power stays `(num - 1) * PowerPerRoom`, so every
remaining fight on a quest level is a little bigger. Deaths on level 2 went
from 66 to 81 of 240. The second is the baseline walking to the lab and back
for 25 or 30 gold and a mutation the sim does not model, and it loses, so the
baseline's default is to answer a quest dialog when it walks into one and to go
no further (`heuristic_action(quests="pass")`). Whether finishing a quest pays
is left to the agent; once the four level-2 mutations are modelled it is worth
measuring again.

With the six controllers of levels 6 and 10 in, the same measurement reads the
same, -0.121 [-0.286, +0.044], and it has to: with quests on, none of the 240
heuristic runs reaches level 6. Those quests are held to the checks in
`tools/validate_sim.py`, not to this number.

**240 seeds is too few at this size.** With the secret room and the
generator's direction order in, seeds 30000 to 30239 read -0.358
[-0.515, -0.202], a jump that sent this looking for a bug. There was none: the
2400 seeds from 40000 read -0.072 [-0.124, -0.020], their ten 240-seed windows
spread from 0.000 to -0.154, and the old direction order on the same 2400
reads -0.056 [-0.109, -0.003], so the order changes nothing measurable and the
30000 window simply sits low on this effect (every quest measurement taken on
it has). Playing the two arms alternately rather than one after the other
gives the same levels, so no state leaks between episodes.
`tools/ablate.py --jobs 8` runs the 2400 in about a minute.

## The generator's last two steps, read (2026-10-30)

`LevelGenerator.Generate` grows the map once, until it holds `num` rooms, then
calls `Adjust`, `CloseLoops`, `CreateRooms` and `InitRooms` once each, makes the
secret rooms, and runs `PositionRooms`. The port had repeated grow, close and
cut until the count came out right, on a reading of the two steps that was
wrong in both directions.

  * **`Adjust`** never shrinks the map. While there are more dead ends than
    `maxDeadEnds`, which really is 0, it marks the farthest one `Never`
    (`GetDeadends` sorts by `RoomNode.distance`, largest first), drops every
    grid position left touching no room, and puts a replacement room on an
    open position drawn uniformly (`RND.Element`, where growth draws by
    weight), retrying through `MaybeMakeRoom` until one is accepted. When none
    is, it undoes the last cut and stops. That last clause is how a 7-room
    level, which cannot be all loops, comes through a target of zero.
  * **`CloseLoops`** never adds a room either. For each grid position it looks
    at the ring of eight cells around the cell diagonally below and right of
    it; when that centre is empty and exactly seven of the eight are rooms, the
    farthest dead end becomes `No` and the eighth cell becomes a room, written
    directly with no shape check. It repeats while a pass moves something, ten
    passes at most.
  * **`GetDeadends`**, called by both, also turns every `No` position back to
    `Maybe` and puts it on the open list, so a refusal by `MaybeMakeRoom` holds
    only until the next call.
  * **`CheckNeighborCount` has no caller**, direct or through a delegate (there
    is no `Method$` entry for it), and `MaybeMakeRoom` checks only the two
    2x2 rations. The port had applied a ceiling of three room-neighbours.
    Without it most maps have a crossroads room (44 of 60 level-1 maps, 59 of
    60 at level 12), which fits the Pit quest's room allowing doors on all
    four sides.

The count is now `num` on every map. Dead ends per level-1 map went from 2.25
to 2.17, and about 2% of level-1 maps are now compact enough that
`ChoosePortals` lands fewer than two portals and the step is undone. The
heuristic reaches 3.237 on the new maps against 3.183 on the old, over the same
240 seeds but unpaired (the maps differ), so nothing beyond noise.

## The anti-repeat weighting, read (2026-10-30)

`DynamicWeightedList<T>` is the only weighted list in the game with a memory,
and it has two users: `M_ItemData.weightedTemplatesByQuality`, one list per
item quality, and `LevelData`'s static `questLists`, one per quest shortlist.

  * **Its weights.** `WeightedList.PopulateEntity` gives each entry
    `GetWeight(element)`, or 1 when no weight function is passed, and
    `ItemData.Load` passes none, so a quality's items start equal.
    `DynamicWeightedList.GetElement(index)` (the virtual override) adds one to
    the entry's `pickCount` and calls `AdjustWeights`, which rewrites every
    weight as `1 / powf(factor, pickCount - minPickCount)` and rebuilds the
    cumulative sums. `factor` is the constructor's 2. The weight function's
    values are gone after the first pick, which matters only for a list built
    with one; neither user is.
  * **The draw.** `GetRandomIndex` takes `rnd.NextDouble() * totalSum` and
    binary-searches the cumulative sums. `C_ItemShop.Roll` draws the quality
    from `weightedQualities`, a plain `WeightedList<int>` (its weights stay the
    shop level's probabilities), then the index from the quality's list, then
    calls the list's own `GetElement(index)`, so every item the shop shows is
    a pick, bought or not. Both draws use the current room's `rnd`.
  * **Its lifetime.** `ItemData.Load` builds new lists each time
    `RootLoader.LoadData` runs, and `Loader._NewGame` calls that, so the item
    history is the run's. The quest lists are static and outlive the run.

So the shop cycles through a quality: an item shown k more times than the
least-shown one in its quality is drawn at `2^-k` of that one's weight. The sim
keeps `RunState.item_picks` and draws with those weights (`item_weights`).
Against a uniform draw inside the quality, on the heuristic over 2400 seeds
from 40000, it is worth -0.005 [-0.035, +0.025]: fidelity with nothing
measurable behind it, as expected of a rule that only reorders which of ten
equally priced items of a quality turns up.

## The model defaults, read (2026-10-30)

A skill or mutation row fills a model, and a field the row leaves out keeps
the value its property was built with. Four of `sim/assumptions.py`'s entries
were guesses at such values, argued from what a shipped row would do under
each reading, because the dump prints `[DefaultValueAttribute]` without its
argument. The argument is in code: each attribute line in `dump.cs` carries
the RVA of the generator that builds it, and the generator passes the value
to `DefaultValueAttribute..ctor` as an immediate (a `movss` constant, a `dl`
for a bool, a boxed enum built from `mov dword [rsp+0x30], n`). The models'
static constructors build the properties with the same constants
(`M_VampirismSkill..cctor`: `new ReactiveProperty<DamageType>(1)`), and
`Model.CreateSettings` sets `DefaultValueHandling.Ignore`, so the JSON never
populates anything and the constructor's value is the one a row gets.

All four guesses stand: evasion's `healthThreshold` is 100, a damage
reaction's `chance` 100, Craggy's `debuff` Stun (1), vampirism's `percentage`
true. They are facts now and live in `sim/model_defaults.py`.

The read found what the guesses had missed: `damageType`. It gates a passive
the same way in every handler read (`C_VampirismSkill.OnDamageCreated`,
`C_BuffAttackSkill.OnDamageCreated`, `C_DamageReactionSkill.OnDamageInternal`):
the passive fires only when `(damage.type & damageType) == damageType` and the
damage is not `Secondary`. A model without the attribute starts at `Empty`,
which passes everything, and that is what the port gave every row that left
the field out. The defaults are not empty:

    Physical   Vampirism, BuffAttack, PassiveStun, PassiveBurn, PassiveKnockback,
               and the damage reactions (Craggy, Untouchable, LoveOfPain)
    Magical    AggressiveDodge, ReinforcedDamage, PassiveBlock, Blurp,
               M_DamageOverTimeStatus

Shipped rows that leave it out and are live in the sim: every Vampirism row
(two skills, four mutations), six BuffAttack rows (one skill, five
mutations), the Untouchable mutation, and the one Blurp skill. So vampirism
now heals off physical hits only (a mage with a vampirism mutation heals
nothing), and never off `Secondary` damage, which the port had not gated
either; BuffAttack and Untouchable fire on physical damage only; Blurp hits as
magical damage, through resistance rather than armour. `with_model_defaults`
writes the default into a row's parameters when it is parsed, so the oracle
and the core read the same thing.

It changes no heuristic run. Over 2400 seeds from 40000, with the defaults and
with an empty table, every seed ends on the same level, although 79 of 300
runs hold a BuffAttack mutation, 18 Untouchable and 13 Vampirism: the squads
the heuristic builds hit physically, the Vampirism mutation does not attach
to Mage units at all, and the reactions are rarely hit by magic on the levels
these runs reach. `tools/diff_core.py` prints the same table before and
after. The gates matter to a policy that fields mages or meets magic users,
which is where an agent would find the difference.

## HoboFury and HungerFury, implemented (2026-10-30)

Two of the 44 mutation names the sim did nothing with, offered once a level
each, and both a stat bonus that follows the run rather than the fight.

  * **HoboFury** (239: Damage +30, flat, `goldAmount` 10).
    `C_HoboFuryMutation.OnGold` floors the team's gold (`roundss` mode 9) and,
    whenever `goldAmount >= floor(gold)` flips, adds or removes a "HoboFury"
    `StatBonus` status on every unit that is not `Dead` or `DeSpawning`
    (traits `0x401`). So the squad hits 30 harder while it holds under 11
    gold.
  * **HungerFury** (227: AttackSpeed, `lowValue` 40, `highValue` 60,
    percentage). The mutation hands its units `C_HungerFurySkill`, whose
    `Food_HungerLevelChanged` sets the status to nothing at hunger level 0,
    `lowValue` at 1 and `highValue` at 2 and up.

`apply_to_specs` takes the run's gold and hunger level and applies both when
the squad's specs are built for a fight, and the spec cache keys on whether
HoboFury is on. Gold and hunger change between fights here, never inside one,
so reading them once per fight loses nothing the sim could show.

On the heuristic over 2400 seeds from 40000, implemented against
unimplemented: -0.006 [-0.012, +0.001], with 293 runs ending holding one of
them. The heuristic takes a shelf's implemented mutation that covers most of
the squad, so these displace other picks about as often as they help.

**Intelligence** (289: +50% for every class; 270: +100% for a random one),
filed under "mana economy" until now, composes `M_Unit._experience` (static
+0x138): `C_IntelligenceMutation.Modify(comingExp, currentExp)` turns a rise
of `g` into `g * (1 + value / 100)` (`percentage`, true by default) or
`g + value`, never below zero, for units of the mutation's class. Bought
experience goes through the same property, so it grows too.
`RunState.experience_gain` applies it per human. The random class is the one
the shelf description draws off the mutation's id, since the sim keeps no
offered class (`mutation_pool_by_id_all_unlocked`). Implemented against not:
-0.003 [-0.007, +0.000] over 2400 seeds from 40000, 102 runs ending with it.

**UnderFeed** ("Living on Scraps", 269: 20% less food, 30% slower in a fight;
the job-offer variant 286: 10% more food). `C_UnderFeedMutation` composes
`M_Food._needed` (static +0x80), the food a move eats, with `Modify(int)`:
`needed * (1 - value / 100)` (or `needed - value` without `percentage`),
through `Math.Round`, which rounds to even, then at least 1, and when that
leaves a number above 1 unchanged it steps once the way `value` points. So
20% less is 1, 1, 2, 3, 4, 8 for 1, 2, 3, 4, 5, 10 humans, and 10% more is
3 for 2. Every unit also gets an "UnderFeeding" `StatBonus` of
`-slowValue` percent on Speed when `slowValue` is set, which is the "slower in
a fight" half of the description. `RunState.feed_cost` runs the compositor and
`apply_to_specs` the slow. Implemented against not: -0.003 [-0.009, +0.003]
over 2400 seeds from 40000, 283 runs ending with it.

## A mutation is offered for one class (2026-10-30)

`Class` is not a flags enum, and `M_Mutation` carries one `_classValue`. What
the rows' `Class` strings mean is `C_Mutations.ExtractClasses`: "all" adds
`Class.Empty`, which every unit matches; "random" adds every class in
`Const.UpgradedClassValues` (Tank to Scientist, ten, built in `Const..cctor`);
a comma list adds its classes; a name adds itself. `OnLevelChange` keeps that
list per mutation the level offers (filtered by the save's unlocks), and
`TryGetTreasureParameters` draws an (id, class) pair from what is not
reserved, through `WeightedList.GetElementUsingHistory` with factor 2 against
`M_Mutations.pickCountByIdAndClass`, and writes the class into the offer.
`C_MutationShop.CreateMutations` excludes the shelf's own pairs, so one
mutation can sit on a shelf twice for two classes. Held and shown pairs are
reserved, the mutation itself is not.

So a row listing seven classes is not a buff to seven classes: it is seven
offers, each for one of them, and seven times as likely to come up as a row
for every unit. A "random" row is ten such offers. The port had drawn whole
rows uniformly by id and applied a listed row to every class in it, and
re-drew a "random" row's class at every fight.

`RunState.mutation_pool` now lists pairs, `draw_mutation` draws one with the
history weights and counts it, shelves and shrines both draw that way, and an
offer is its row with its one class, so everything downstream (the specs, the
agents, the shelf description) reads a single class as it already could. A
dialog's `GiveMutation` takes the same draw over the mutation's own pairs.

At level 5 the 98 mutations on offer are 316 pairs, and a shelf comes out 96%
single-class. On the heuristic over 2400 seeds from 40000, against the id-level
draw: -0.018 [-0.065, +0.028], mutations held 2.61 against 2.66 a run.

**ChanceIncrease** (215: +15%, 284: +5%, every class).
`C_ChanceIncreaseMutation` decorates the `chance` of every held simple
mutation (and of combined mutations' descendants) and of every one inserted
later, and hands its units `C_ChanceIncreaseSkill`, which decorates the
`chance` of every action and passive skill (`IProbable`) the unit has or gains.
`Modify` is `chance * (1 + value / 100)` with `percentage`, `chance + value`
without, held to 0..100. `boost_agent_chances` runs it over everything a
squad unit carries once the fight's agents are built, so both engines read
the raised numbers: crit, evasions, knockbacks, explosions, resurrection and
the hooks' rolls. A chance left at its default 100 stays 100. Whether a
mutation's own skill takes the bonus twice is open
(`chance_increase_applies_once`). Implemented against not: -0.000
[-0.004, +0.003] over 2400 seeds from 40000, 190 runs ending with one.

