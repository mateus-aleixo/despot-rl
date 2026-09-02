# The Rust core

`core/` is a Rust port of the battle loop, built as a `cdylib` with a plain C
ABI and called from Python through `ctypes` (`sim/fast.py`). The Python
implementation stays the oracle; this exists only to run the same fight faster.

## Why C ABI rather than PyO3

PyO3 would be nicer to write, but it links against CPython, and on Windows that
means matching the MSVC-built interpreter's ABI. A `cdylib` exporting plain
`extern "C"` functions has no such coupling: it builds with the GNU Rust
toolchain, needs no Visual Studio Build Tools, and `ctypes` loads it directly.
The cost is manual marshalling, which is cheap here because everything crossing
the boundary is already flat float arrays.

Build with `cargo build --release` in `core/`.

## What is ported

Movement (grid A*, the waypoint follower, ORCA), the default-attack cycle, the
damage formula, projectiles, and `SplashAround`.

Not ported: class skills, the rest of the per-unit skills, mutations, statuses,
and anything RNG-driven (crit, evasion, dodge). `sim/fast.supported()` decides
whether a fight is inside the envelope and `fast_battle` **refuses** the ones
that are not, rather than returning numbers that quietly drift from the oracle.

Two Python quirks are reproduced deliberately, because both are behaviourally
visible:

- Preferred velocities are computed for every agent *before* any of them move,
  but the ORCA pass then updates position and velocity in place, so an agent
  sees earlier agents already moved and later ones not yet.
- The attack gate tries `StopIfRunning` before the cooldown check, so a unit
  arriving in range while still carrying velocity swings that tick regardless of
  cooldown.

## Speed

Six broadswords against six enemies, on one layout:

| | ms per battle | speedup |
|---|---|---|
| Python oracle | 276 | 1x |
| Rust, one call at a time | 1.9 | **142x** |
| Rust, batched across threads | 0.29 | **957x** |

### Which of those numbers training actually gets, and why batching is not worth it

Training takes the **one call at a time** row. `rl.train.rollout` steps its eight
envs in a Python loop, so every fight is its own `fast_battle`, and the batched
entry point the 957x measures (`despot_battle_batch`, bound as `fast_batch`) has
exactly one caller in the repository: `tools/diff_core.py`. Using it in training
would mean letting `env.step` defer a fight and resume once a batch comes back,
which is a change to the env contract rather than a flag, so it wanted a
measurement first.

`tools/profile_train.py` wraps `rollout`, `DespotRunEnv.step`, `_encode`,
`action_mask`, `RunState.apply` and `fast_battle`, then runs the real
`rl.train.main`. 100k steps, 195 update cycles, on an idle machine, twice:

| part | run 1 | run 2 |
|---|---|---|
| rollout | 85.5% | 86.8% |
| &nbsp;&nbsp;policy forward and buffer writes | 24.4% | 24.9% |
| &nbsp;&nbsp;`env.step`, Python run layer | **44.3%** | **45.1%** |
| &nbsp;&nbsp;&nbsp;&nbsp;`RunState.apply`, less the fight | 19.9% | 20.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;`_encode`, building the 195 floats | 16.2% | 16.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;`action_mask`, called twice a step | 7.6% | 7.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;everything else in `step` | 0.6% | 0.6% |
| &nbsp;&nbsp;`env.step`, Rust battle core | **16.7%** | **16.9%** |
| PPO update and bookkeeping | 14.5% | 13.2% |

10,352 fights at a mean of **0.76 ms**, better than the 1.9 ms the 6v6 benchmark
above measures, because a typical fight is smaller than that one. The table
predates the `action_mask` memo below; with it in, that line reads 4.2% and the
run layer 42.9%, and nothing else moves.

**So batching is worth at most 1.16x.** Fights are 16.8% of the clock, and the
benchmark's best case takes them to 0.29 ms, which is 6.5x on 16.8%, or 14% of
the loop. That is a poor trade for making `env.step` re-entrant, and it would put
a deferred-fight state machine underneath the part of the codebase the fidelity
work lives in.

**The Python run layer is 2.7x the core.** That is where the time is, and none of
it needs the env contract touched:

- **`action_mask`, done.** It was called twice per `step`, once against the
  pre-action state to check the action and once for the info dict afterwards.
  The second is what the caller stores and hands back as the next step's mask,
  so the first was recomputing a value `rollout` already held. `DespotRunEnv`
  now keeps the last mask and clears it at the two points the state moves,
  `reset` and the `apply` inside `step`. The line fell **7.7% to 4.2%**, and end
  to end, three interleaved 100k runs each way, 61.79/64.58/61.73s became
  60.13/60.07/60.25s: about **1.6s, or 2.7% of the run and 3.4% of the training
  loop**, which is what halving a 7.7% line looks like once the warm-up baseline
  and the final evaluation are counted too.

  It is provably behaviour-preserving: 200k steps at seed 0 gives **weights
  bit-identical** to the pre-memo agent, which is a usable regression test in
  general now that training is known to reproduce exactly. `action_mask` also
  returns a copy rather than the cached array, so nothing starts aliasing a
  buffer that used to be freshly allocated on every call, and
  `tools/validate_rl.py` checks the memo against a fresh scan in the window that
  could actually go stale: the mask carried from the previous step, after
  `_encode` has run and called `ensure_stock`.
- `RunState.apply` at 20.0% and `_encode` at 16.3% have not been profiled
  internally yet. `_encode` calls `squad_power`, `mutation_effect` and
  `ensure_stock` on every step, and each has a cache that may not be hitting.
  These are now the whole of the Python run layer's cost.

**And for a sweep, aggregate throughput is the number that matters, not
single-run speed.** Six concurrent runs measured 6,206 steps/s against 2,077 for
one, which is 3x for 6x the processes, so something is already saturating before
the cores are. Finding where that ceiling is may beat any of the above for the
wall clock of a twelve-seed arm.

**Contention scales the whole loop uniformly**, which is worth knowing before
trusting any future profile taken on a busy machine. The same measurement while a
twelve-seed 4M sweep held six cores ran at 1,606 steps/s against 2,120 idle, a
third slower in absolute terms and 1.00 ms a fight rather than 0.76, and every
share landed within a percentage point of the table above: core 16.6%, run layer
44.4%, update 13.8%. Every part of the loop is single-threaded CPU work, so they
all slow together and the ratios survive.

## Differential testing, and what it caught

`tools/diff_core.py` runs the same fights through both engines, restricted to
the ported envelope so any difference is a porting bug rather than a missing
feature.

It immediately caught a real one. Three enemy classes matched 0/12 while two
matched 12/12, and the three failures shared a statline. They were the Size 2
classes: **melee reach is measured against both bodies**
(`attacker.radius + target.radius + margin`), so a radius-3 attacker needs 10 to
touch a radius-6 target, not the 7 it needs against its own size. The port had
baked a single per-unit range into the spec array. After passing the melee flag
and computing reach per target pair, agreement went to **60/60**.

Current agreement inside the envelope: **60/60 winners, mean tick error 7.8%,
mean damage error 2.3%.**

## The residual error is chaos, not logic

7.8% sounds like a bug until you perturb one engine against itself. Nudging
every starting position by `1e-4` and re-running the *same* engine:

| fight | engine self-sensitivity (tick change) | winner flips |
|---|---|---|
| vs Dalek, Python | 2.5% | 0/12 |
| vs Dalek, Rust | 5.3% | - |
| vs Mancrack (splash), Python | 11.5% | 1/12 |

The cross-engine error is the same order as a single engine's response to a
`1e-4` nudge, and float32-vs-float64 across a whole battle is a much larger
perturbation than that. So the residual is chaotic amplification, not a logic
difference -- and winner agreement, which is what the RL reward actually reads,
is exact.

This has a consequence worth stating for the RL side: **individual fight
outcomes near a decision boundary are effectively coin flips.** Splash fights
flip winner on a `1e-4` nudge about one time in twelve. Policies have to be
evaluated over many runs, and a single-run comparison means nothing.

## Coverage of real fights

Porting the battle loop is only useful to the RL loop if real fights fall inside
the envelope. Sampled from actual runs:

| | supported |
|---|---|
| before `SplashAround` was ported | 6/22 = 27% |
| after | 15/28 = **54%** |

`SplashAround` alone was the single biggest blocker. What still falls outside,
in order of frequency: `FurySwipe`, then units carrying a second action (class
skills such as the Medic's heal, and per-unit actives like CatBot and
AoeRanger).

So the honest position is that the core is fast and faithful but **currently
usable for about half of real fights**. Getting the full speedup into training
needs the remaining passives (FurySwipe, crit, evasion, vampirism) and the
action system (class skills and per-unit actives) ported too. The passives are
straightforward; crit and evasion additionally need an RNG story, since the
oracle uses Python's Mersenne Twister and the core will not reproduce it -- for
those, statistical rather than exact agreement is the right target.

## Porting the passives, and the RNG

All the passive skills are now in the core: splash, crit, evasion, both dodges
(the Dodger class skill and the unit-level one), vampirism, fury swipe,
reflection, regeneration, knockback on hit, and projectile splash.

Crit and evasion consume random numbers, which would normally mean giving up
exact comparison and settling for statistical agreement. Instead the core runs
**CPython's MT19937**, including its `init_by_array` seeding for integer seeds
and the 53-bit `genrand_res53` double. It matches bit for bit across every seed
tried, single- and multi-word:

    seed 0, 1, 5, 42, 12345, 2^31+7, 987654321987  ->  all exact

The rolls also have to happen in the same *places*, or the streams desynchronise
even with an identical generator. So the port mirrors `Battle.hit`'s structure
exactly, including the fact that splash victims each consume an evasion roll,
and that the crit roll happens once per swing before any damage.

### How well that works, and where it stops

Against a crit-carrying enemy (Tomato, 20% for 3x), by fight size:

| fight | winner match | crit-side damage error | tick error |
|---|---|---|---|
| 1v1 | 16/16 | **0.00%** | 3.3% |
| 2v2 | 16/16 | **0.00%** | 7.9% |
| 6v6 | 16/16 | 3.76% | 11.8% |

Exactly zero damage error at small scale is the proof the generator and the roll
placement are both right: every crit lands on the same swing in both engines.

The drift at 6v6 is the honest limit. Exact RNG parity only holds while the
*sequence* of RNG-consuming events matches, and float32-vs-float64 eventually
shifts the timing of a swing, which shifts a roll, after which the streams are
independent. So the guarantee is: identical generator, identical roll sites,
and identical results right up until chaos separates the two simulations.

Overall inside the envelope: **66/72 winners**, mean tick error 13.0%, mean
damage error 10.4%. The six disagreements are all in splash and fury fights,
which are the ones already shown to flip winner on a `1e-4` nudge about one time
in twelve.

## Coverage now

| | supported |
|---|---|
| initial port | 6/22 = 27% |
| after `SplashAround` | 15/28 = 54% |
| after all passives | 25/35 = **71%** |

What remains outside is a single category: **units with a second action**. That
is the Medic's heal and the Thrower's bomb (class skills), and per-unit actives
like AoeRanger and CatBot. Porting the action system is the last step to full
coverage, and it is a bigger job than the passives were -- it means the action
list, the cooldown/mana gates and the effect dispatch, not just a per-agent
field.

## Porting the actions

A unit is a priority-ordered list of actions, not a single attack, so the core
now carries an `ActionState` per action with its own cooldown, animation timer
and phase. The ABI takes a flat action table alongside the spec table: one row
per action, naming the owning agent, the kind, its range, cooldown, mana cost
and effect parameters.

The loop mirrors the base tree's dynamic Selector. Actions are tried highest
priority first, and the two tree shapes behave differently:

- a **skill** (NC-Skill) gates on mana, then cooldown, then range; any failure
  returns Failure and the next action is tried
- the **default attack** (NC-DefaultAttack) always consumes the tick once
  reached: it swings, waits out its cooldown, or walks toward the target

Effects ported: direct damage (with the magical flag, which is how the Mage
class skill works), AoE, heal, drain and mana burn. Mana itself is ported too,
since it gates casting -- it arrives through the damage pipeline at the same two
rates the oracle uses, target credited before attacker.

Still out: effects that need machinery beyond a damage number -- summons (they
change the agent count mid-fight), timed buffs, statuses and spirit link.

Agreement across nine enemy types including action carriers: **102/108
winners**, mean tick error 12.2%, mean damage error 9.7%. The Medic's heal and
CatBot's HipThrow both land at 0.00% damage error, which is the useful signal:
the action fires on the same tick in both engines.

## Coverage, and the end-to-end win

| | supported |
|---|---|
| initial port | 6/22 = 27% |
| after `SplashAround` | 15/28 = 54% |
| after all passives | 25/35 = 71% |
| after the action system | 42/43 = **98%** |

The single remaining blocker in the sample is the Samurai's `StatBonusCast`,
which is a timed buff.

`RunState.use_fast_core` routes fights to the core when they are inside the
envelope and falls back to the oracle otherwise, so a run mixes both without the
caller noticing.

Two Python-side wastes showed up once the core was fast enough to expose them,
both found by profiling rather than guesswork:

- `parse_room_layouts` was re-parsing the layout CSV **on every fight**, 25% of
  the fast path's total time. Now memoised.
- a behaviour tree was being built for every agent and then thrown away when the
  fight went to the core. `Battle(build_trees=False)` skips it.

End to end, random policy over full runs:

| | per run | env steps/s |
|---|---|---|
| Python oracle | 0.73 s | 190 |
| Rust core | 0.041 s | **3,062** |

That is **16x** on the whole run loop, against 83x on an isolated battle -- the
gap being everything else a run does in Python. Under PPO the gain is smaller
again, because rollouts then share time with the network and the observation
encoding; training quality is unchanged (greedy mean level 3.92, same as the
pure-Python run).

## Summons, buffs and spirit link

The last three systems are in, which takes the core to **100% of sampled run
fights** (70/70).

**Buffs** are a timer list per agent, with armor, speed and attack speed read
through them rather than stored flat. **Summons** append to the agent vector
mid-fight; the caller passes a table of summon templates (interned, one row per
summoned class) because the core cannot build a spec from the game tables
itself. The spawn offset consumes two random draws through `uniform(-r, r)`
exactly as the oracle does, so the RNG streams stay aligned.

**Spirit link** reproduces a quirk rather than fixing it. The oracle stores one
shared link object on every member and decrements it once per member per tick,
so a seven-member link expires seven times faster than its stated duration. The
core does the same, because the oracle is the definition of correct here, not
the designer's likely intent. It is flagged in the code so nobody quietly
"fixes" one side and breaks parity.

Spot checks against the oracle, 10 seeds each:

| setup | winners | damage error |
|---|---|---|
| Tank buff | 10/10 | 0.0% |
| Cultist summon | 10/10 | 5.2% |
| Scientist summon | 10/10 | 0.0% |
| Monk spirit link | 10/10 | 0.0% |
| Samurai StatBonusCast | 10/10 | 0.9% |
| Slower area debuff | 10/10 | 39.7% |

Slower's 39.7% is chaos: nudging positions by `1e-4` inside the *same* engine
moves its damage 26.6%, because an attack-speed debuff is heavily leveraged.

## One real bug, found by refusing to accept "it's chaos"

Samurai disagreed 5/12 on winners with a tick error of only 1.7%. Every other
disagreement so far had been chaos, and the test for that is whether the same
engine flips under a `1e-4` nudge. **Samurai did not flip: 0/12.** So it was a
logic difference, not noise.

It was mana regeneration. Samurai banks 150 mana every 10 s, which is what pays
for its attack-speed cast, and the core still had `regen_stat == 1` (Mana)
wired as a no-op from when mana was not ported at all. A stale comment even said
"inert here". Fixed.

The remaining Samurai disagreements turned out to be a **third** category, worth
separating from the other two: those fights run to the 120 s timeout, and a 2%
tick difference decides whether one engine finishes just inside it and the other
just outside. That is win-vs-draw, not win-vs-loss, so `tools/diff_core.py` now
reports it as its own column instead of burying it in the agreement rate.

The same tool also stopped dividing damage error by one engine's number: a side
that dealt 41 damage in one and 0 in the other was being reported as 3725%
wrong, when the honest reading is that it did nothing in both. It now scales by
the larger of the two.

## Where it lands

    winners agreeing outright     146/156
    differing only by draw-vs-win     4
    genuinely opposite results        6

All six genuine disagreements are in splash and fury fights, which flip on a
`1e-4` nudge about one time in twelve inside a single engine.

| | supported |
|---|---|
| initial port | 27% |
| after `SplashAround` | 54% |
| after all passives | 71% |
| after the action system | 98% |
| after summons, buffs and spirit link | **100%** |

End to end, random policy over full runs:

| | per run | env steps/s |
|---|---|---|
| Python oracle | 0.734 s | 190 |
| Rust core | 0.017 s | **7,238** |

**38x** on the whole run loop, up from 16x when a third of fights still fell
back to Python. The isolated-battle figure is 74x single-call and 784x batched;
the run loop sits between them because everything a run does outside a fight is
still Python.

## What the agent-level passives did to the envelope

`sim/mutations.py` now implements the mutation passives that hang off the game's
four `C_PassiveSkill` hooks (see `notes/reference-sim.md`). **None of it is
ported**, so `fast.supported` gained a blocker: an agent carrying any of
`on_attack`, `on_damaged`, `on_cast`, `on_death_passives`, `standing`,
`damage_bonus`, `cs_link_bonus` or a resurrection roll sends the fight to the
oracle. That is the whole point of the blocker -- the core would otherwise run
the fight happily and silently drop the effect, which is worse than being slow.
The cost is visible: in `tools/mutation_value.py` the half of the paired test
that takes mutations runs about 30x slower than the half that does not.

One thing did get ported, because it had to. **A projectile did not carry its
source unit** -- on either side. `Battle._step_projectiles` and the core's
`step_projectiles` both called `hit` with attacker `-1`, so a ranged unit's
vampirism, fury stacks and knockback had never fired, and every on-attack
passive would have been dead for Shooter and Mage. `Projectile` now carries an
`attacker` in both engines.

That changes what the differential test is measuring, since ranged fights now
have more going on in them. Winner agreement is unchanged and the drift is a
little larger, which is the expected shape: more RNG-consuming events per fight
means chaos separates the two float widths sooner.

| | winners agreeing | opposite | tick err | damage err |
|---|---|---|---|---|
| without the projectile source | 146/156 | 6 | 12.4% | 11.1% |
| with it | 146/156 | 6 | 16.0% | 12.7% |

A second, smaller divergence surfaced the same way: the core's `K_DIRECT` and
`K_AOE` passed the caster index while Python's `effect_magic` and `effect_bomb`
passed nothing, so the two engines had disagreed about the Mage and the Thrower
since the action system landed. Python now matches the core.

## Porting the hooks

The passives are in the core now, and with them the control statuses they need.
That closes the envelope hole the previous section opened: a fight carrying a
mutation passive no longer falls back to the oracle.

### What the ABI grew

Two things. `STRIDE` went 32 to 33 -- the extra float is the unit's **class
id**, which nothing needed until `CSLinkStatBonus` had to ask whether a spirit
link's member is a Cultist. Class names are strings on the Python side and an
integer here; `fast.class_id` hands ids out on first sight, which is enough
because both halves of one fight are packed by the same call.

And a third flat table alongside specs and actions: one row per (agent,
passive), `PASSIVE_STRIDE = 21`.

    0  agent           6  cast_source     12,13,14  stat/value/percentage
    1  kind            7  status          15,16,17  stat/value/percentage
    2  damage mask     8  amount          18,19,20  stat/value/percentage
    3  chance          9  amount2
    4  duration       10  flag
    5  radius         11  target class

The three (stat, value, percentage) triples are the shape `M_StatBonusStatus`
carries, and every passive that applies a timed buff writes them. The scalars
carry what nothing else does: ManaBreak's `amount`, MultiCast's two refund
shares, Compensation's health threshold, ResurrectionChance's health value.
`attach_passives` files each row onto the per-agent list its hook reads, so the
hot path checks one empty `Vec` rather than a match.

**`Buff` had to be generalised to get there.** It carried four named fields --
`armor_pct`, `armor_flat`, `speed_flat`, `attack_speed_pct` -- which covered
every class skill but none of the mutation statuses, which move Damage,
Resistance and Health too. It is now the same (stat, amount, percentage) triples
in a fixed array of four, and `armor()`, `speed()`, `damage_now()`,
`resistance_now()` and `max_hp()` all fold from one pass. `add_buff` builds the
triples from the old fields, so the action system did not change.

### Statuses, and one thing the port found in the oracle

Stun, silence and panic are three floats on the agent. Stun and panic stop a
unit acting and pin it in place; silence fails the range gate for anything that
is not the plain attack. All three mirror where the oracle enforces them.

Which is how the port found that **panic was inert in the oracle**. `IsPanicked`,
`PanicInPlace` and `PanicFollowPath` live in **NC-Fear**, a tree of its own that
the runtime overlays on a panicked unit -- it is not inside NC-BaseUnit, so
`build_unit_tree` never reaches those leaves and a panicked unit went on
fighting. Applying panic to every enemy for 600 ticks changed their damage by
exactly nothing. It is enforced at the loop level now, in both engines, doing
what NC-Fear does with the fleeing left out.

A second one, in the same family: **FearsomeAttack had no chance roll.** Its
model names the property `percent` rather than `chance`, so the shared gate --
which reads `chance` with a default of 100 -- fired it on every hit instead of
one in ten. `M_FearsomeAttackSkill` implements `IProbable` explicitly and its
`IProbable.get_chance` is a bare `jmp` to `get_percent`, so the two are the same
number under two names; `mutations.CHANCE_PARAM` records that.

### Agreement

RNG parity is the whole difficulty: every hook that rolls has to roll in the
same *place*, or the streams separate even with an identical generator. So the
port mirrors the oracle's ordering exactly -- on-attack after mitigation is
computed and before the damage lands, on-damaged at the end of `hit`, on-cast
right after the effect resolves, and the death pass once, after the whole
per-agent loop.

`tools/diff_core.py` now runs each mechanic at three fight sizes. At 1v1,
16 seeds each, the two engines agree on **every winner with 0.00% damage error
for all sixteen mechanics** -- BuffAttack both ways round, PassiveStun,
FearsomeAttack, Craggy, Untouchable, ManaBreak, BuffOnCasted, MultiCast,
ModifyDamage, StickyBlood, BuffOnDeath, ResurrectionChance, Compensation,
CSLinkStatBonus and ClassDiversity. Zero is the proof: every roll landed on the
same swing in both.

At 2v2 each mechanic sits at or below its own no-mutation baseline (0.90% for
the broadsword squad, 6.25% for the Mage one) with one exception, and by 4v6
they are all indistinguishable from it. Fights without mutations are unchanged:
**146/156 winners, 16.02% tick error, 12.73% damage error**, exactly as before
the port.

The exception is worth naming. **Untouchable** drifts first -- 16.5% at 2v2
against a 0.90% baseline -- because it is the only shipped mechanic that changes
a *per-tick rate*: -40% attack speed makes the cooldown decay `dt * 0.6` instead
of `dt * 1.0`, and f32 and f64 disagree about that sum long before they disagree
about anything else. Every other mechanic leaves the rate at exactly 1.0, which
is why they stay exact.

### What it bought

`tools/mutation_value.py`, the paired mutation test, over 2,000 run seeds:

| | before the port | after |
|---|---|---|
| 240 seeds, mutations taken | 190 s | 10 s |
| 2,000 seeds, both halves | not run | 151 s |

That is the difference between a question you cannot afford to ask and one you
can. See `notes/rl.md` for what the answer turned out to be.

### Still outside

The per-unit **on-death skills** -- TransformAfterDeath and the death-damage
family, which summon or deal damage rather than apply a status -- are the last
envelope blocker, along with a fight that starts with a status already running.
`fast.supported` now also packs the passive table, so a mutation name the core
has no kind for is refused rather than silently dropped.

## Porting the on-death skills, and what a summon actually is

The last envelope blocker was `Agent.on_death`: the per-unit skills that fire
when a unit dies. Four CSClasses, thirteen uses. Three of them
(`FireheadDeath`, `DamageOnDeath`, `FirestarterDeath`) deal damage;
`TransformAfterDeath` spawns something.

They ride the same death pass the mutation passives already used, as two more
kinds in the passive table -- `P_DEATH_DAMAGE` and `P_DEATH_SUMMON`. The oracle
runs the unit's own skills first and the mutation passives second, and both
consume randomness, so the core does the same in the same order.

### `random.choice` is not `random()`

The damage skills have a radius branch and a no-radius branch, and the second
one picks its victim with `battle.rng.choice(foes)`. That is **not** a float
draw: `choice` goes through `Random._randbelow_with_getrandbits`, which takes
`n.bit_length()` bits and **rejects until the draw fits**. How many 32-bit words
it eats therefore depends on what it drew, and an implementation that is merely
close desynchronises the stream rather than picking a slightly different victim.

`rng.rs` now has `getrandbits`, `randbelow` and `choice`, including CPython's
"k == 0 returns 0 without drawing". `despot_choice_probe` exposes it and the
check suite pins it against CPython over five seeds and fifteen sequence
lengths, the way `despot_rng_probe` already pinned `random()`.

### A summon is a unit, not a stat block

Porting `TransformAfterDeath` meant fixing something older. The core's `summon`
built an agent from a template spec row and gave it **a default attack and
nothing else** -- while the oracle's `Battle.summon` runs `_init_agent`, which
attaches the class's own `Skills.json` entries. So a summoned **SciTower never
cast SciCast**, a summoned **Specter never used ManaBurn**, and none of them
carried the passives their rows give them. That has been true since the action
system was ported; nothing noticed because no check compared a summon's
behaviour.

A template is now three things -- a spec row, its action rows and its passive
rows -- and building one can pull in another, because a transform chain is a
summon whose on-death skill summons again (Ostrich2 -> OstrichRider2 ->
Specter). `_Templates` reserves the index before recursing so a cycle
terminates. Two more tables cross the ABI (`tmpl_actions`, `tmpl_passives`),
addressed by template index, and `split_by_owner` buckets them so
`attach_actions` and `attach_passives` can be reused verbatim on the one
summoned unit.

The packer was restructured to get there: `pack_actions` and `pack_passives`
became row builders that a template can call on a hypothetical agent, and
`pack_all` drives both against one `_Templates`. The batch path builds **one**
registry across every fight in the batch, because the batch ABI ships a single
template table and interning per battle would misname every summon after the
first.

### A unit that swings from beyond the grave

The death skills exposed a real porting bug, and it was not in the death code.

`living` is taken once at the top of `step`, so an agent can be killed by an
earlier agent in the same tick and still be walked in the same loop. The oracle
stops it -- NC-BaseUnit's first branch is `ShouldDie -> Die`, which sets the
intent to stop and never reaches the action Selector -- and the core did not
check. A unit killed mid-tick got one free swing.

It was invisible until a death effect put a number on it: three broadswords in
contact with a 1 hp FireHead, both engines finishing on the same tick, the
oracle reporting 194 damage from team 1 and the core 299. The difference is
exactly one FireHead swing (111 - 6 armour = 105).

Fixing it moved the headline differential from 146/156 winners to **147/156**,
tick error 16.02% -> 15.96%, damage error 12.73% -> 12.66%.

### Agreement

Measuring a death effect needs the walking taken out: an approach is hundreds of
ORCA steps and the two float widths disagree about those long before they
disagree about anything else. So `diff_core.py` places both sides already in
contact, puts the carrier on 1 hp, and compares only the seeds where the tick
counts match -- a tick match means no swing has moved, so any damage difference
after it is the death effect and nothing else.

| unit | skill | tick-matched | exact |
|---|---|---|---|
| FireHead | FireheadDeath (random victim) | 24/24 | **24/24** |
| Firestarter | FirestarterDeath (radius) | 24/24 | **24/24** |
| Fish | DamageOnDeath (radius) | 24/24 | **24/24** |
| PoringMedium | TransformAfterDeath x3 | 24/24 | **24/24** |
| Ostrich | TransformAfterDeath | 0/24 | -- |
| OstrichRider2 | TransformAfterDeath | 0/24 | -- |

The two that never tick-match are the honest limit rather than a failure. A
summon's spawn offset is two `random.uniform` draws, and the same stream gives
**-1.0570034110010258 in float64 and -1.0570034980773926 in float32** -- about
1e-7 apart. That is enough to move a swing by one tick once the summon has to
walk, and both of those transforms spawn something that walks. The Porings spawn
in contact and stay exact, which is the control.

### The envelope now

**102 of the 111 enemy classes** are inside it, and over eighty heuristic runs
**every fight resolves in the core** -- 459 of 459. The last thing the run layer
was still falling back on was the timed `OraStatBonus` mutation (the Monk
attack-speed cast), which becomes an Action named `Mutation:OraStatBonus` and
packs like any other `buff_self`; that is wired now.

What is left outside is one category, nine classes: **skills that cast a status
or a blink**. Stun, Silence, Panic and WatcherDisable as *actions*, and
BlinkAway / MushroomBlink. The statuses themselves are ported -- the core has
stun, silence and panic -- so what is missing is only an effect kind that
applies one, plus displacement for the blinks.

## The last two effect kinds, and the leaf that was quietly failing

`K_STATUS` and `K_BLINK` close the envelope. The statuses themselves were
already in the core, so a status cast is only the effect that applies one:
`make_status`'s radius branch picks every enemy **around the caster** rather
than around the target, the no-radius branch is the target alone, and damage --
which only `WatcherDisable` carries -- lands on the same victims. `STATUS_OF`
decides which status: Fear is a panic and WatcherDisable is a silence, so the
skill's name is not the status's name. `ACTION_STRIDE` went 18 to 19 to carry
the status index.

A blink is a teleport `radius` directly away from the current target, and the
oracle clears the follower's waypoints without resetting its index, which leaves
`done()` true and forces a repath next tick. That is reproduced rather than
tidied. Both engines also agree on the degenerate case: with the target exactly
on top of the caster there is no direction to move along -- `hypot(dx, dy) or
1.0` times a zero offset -- and the blink does nothing.

### `EmptyAction` was returning FAILURE

Wiring `Panic` up turned a Rat into a permanent lockdown, and the two engines
then disagreed about it: the oracle had the Rat landing 34 hits over a fight and
the core none. Tracing which action the Rat ran each tick showed the oracle
alternating -- Panic completes at tick 60 and the **default attack starts in the
same tick**, runs to tick 119, then Panic starts again at 120.

The cause is one line in `NOT_MODELLED`. NC-Skill's cycle ends on
`BinarySelector<IsRunningOut> -> EmptyAction`, and `EmptyAction` was listed as
unmodelled, which returns FAILURE. So the whole skill subtree failed on the tick
it finished, and the base tree's Selector fell through to the next action --
every skill in the sim yielded a tick to the default attack the moment it
completed. NC-DefaultAttack has no such tail, which is why only skills did it.

`EmptyAction` is NodeCanvas's own no-op and `EmptyAction.OnUpdate` sets the
node's status to Success (`mov word ptr [rbx + 0x60], 1` after testing it
against Running). So it succeeds now, NC-Skill ends its cycle the way
NC-DefaultAttack does, and a skill whose cooldown is shorter than its animation
cycle simply repeats -- which is what a Rat with a 0-cooldown Panic and 1 damage
should do.

This was worth more than the two kinds it was blocking. It had been shifting
every skill-then-attack sequence by a tick since the trees were wired:

| | winners agreeing | opposite | tick err | damage err |
|---|---|---|---|---|
| before | 147/156 | 6 | 15.96% | 12.66% |
| after | 145/156 | 6 | **13.82%** | **10.92%** |

The two winners that moved both went to the draw boundary rather than to the
other side -- genuinely opposite results stayed at 6 -- while AoeRanger's tick
error halved, 37.6% to 18.2%.

### Agreement

Both sides in contact, and the casters handed their mana, since mana arrives
through the damage pipeline and a Silencer in a two-body fight never banks the
150 its cast costs.

| unit | skill | tick-matched | exact | damage err |
|---|---|---|---|---|
| Silencer | Silence | 24/24 | 24/24 | 0.00% |
| Orb | Stun | 24/24 | 24/24 | 0.00% |
| Watcher | WatcherDisable | 24/24 | 24/24 | 0.00% |
| Rat | Panic | 24/24 | 24/24 | 0.00% |
| PoringSmall2 | Panic | 24/24 | 24/24 | 0.00% |
| Mushroom, Mushroom2 | MushroomBlink | 24/24 | 24/24 | 0.00% |
| RangeBlinker, RangeBlinker2 | BlinkAway | 0/24 | -- | **0.00%** |

The two BlinkAways never tick-match and it does not matter: their **damage is
identical to the last point** in every seed, and only the length of the fight
differs -- 722 ticks against 729. A blinker teleports 25 units and walks back,
over and over, so the two float widths end up a few ticks apart on when the last
body drops without ever disagreeing about a swing.

### The envelope is closed

**111 of 111 enemy classes**, and over 120 heuristic runs **every fight resolves
in the core** -- 694 of 694, in 2.8 s.

What is left is not a skill: a fight that starts with a status already running
has no way to be handed across the ABI, and nothing in the run layer produces
one, since every fight starts clean.

One fidelity note came out of it and is now in `assumptions.py`. Panic is
modelled as "does not act, stays put", and both shipped `Panic` rows carry
`Duration 99999` and `speedBonus 50`. A speed bonus is a number only a fleeing
unit would use, so the game almost certainly sends the target running and lets
it come back, where this removes it from the fight for good. Rat and
PoringSmall2 are stronger here than they are in the game.


## Evasion and vampirism moved off the spec row

Both used to be one float on the agent's spec: `evasion_chance` at column 15 and
`vampirism_pct` at 18. That is the wrong shape, because the game runs one
`C_PassiveSkill` controller per source, so two evasion mutations are two
independent rolls and two vampirism sources heal twice.

They are passive rows now, `P_EVASION` and `P_VAMPIRISM`, filed onto per-agent
`Vec`s by `attach_passives` exactly as the mutation hooks are. Evasion uses
`chance` (column 3), `amount` for the health threshold and `flag` for jumpBack;
vampirism uses `amount` for the value and `flag` for the percentage bit.

**`STRIDE` went 33 to 31.** The two freed columns were removed rather than left
as dead ones, so everything after them shifts down: `dodge_cooldown` is 15 now,
`max_health` 27, `class_id` 30. `sim/fast.py` and `build_agent` are the two
places that name the indices, and a mismatch between them is not subtle -- the
first differential run reports units with someone else's speed.

The core's `evaded` mirrors `Battle._evaded` roll for roll, which matters more
than usual here: it consumes one random draw per *evaluated* source, so a
threshold row that is inactive at full health consumes nothing and the streams
stay aligned. The jump-back reuses `push` with the game's own literals, speed
150 and acceleration -280.

Agreement afterwards, `tools/diff_core.py`:

| | winners | tick err | damage err |
|---|---|---|---|
| before | 146/156 outright, 4 draw-edge, 6 opposite | 13.82% | 10.92% |
| after | 145/156 outright, 5 draw-edge, 6 opposite | 13.82% | 10.92% |

The one fight that moved crossed the draw line, which is the timeout category
rather than a disagreement about a swing. The new mechanics themselves are
exact where exactness is meaningful: Vampirism, Vampirism (fixed), Evasion and
Evasion (threshold) all report **16/16 winners and 0.00% damage error at 1v1 and
2v2**, and at 4v6 they sit on the same 9-12 of 16 as the no-mutation control.
`tools/validate_sim.py` runs the 1v1 pair as a check, so a future port that
drops one of them fails the suite rather than quietly healing nobody.

### Explode and knockback follow them off the spec row

Same treatment, same reason: `P_EXPLODE` carries (damage, radius, chance,
damage mask) and `P_KNOCKBACK` carries (chance, radius, speed, acceleration,
mask), one row per source, and `attach_passives` files them onto per-agent
`Vec`s. `STRIDE` went 31 to 29 as the two knockback columns left the spec row,
so `proj_splash` is 22/23 now, `max_health` 25 and `class_id` 28.

The core grew `addon_gate`, which is `passive_gate` with the extra requirement
that the damage be Physical, and `explode`, which mirrors the oracle's
`_explode` including the fact that the explosion is Magical and therefore
cannot re-enter itself. Both take their randomness in the same place as the
oracle, and a chance of 100 draws nothing in either.

`tools/diff_core.py` gained four cases. At 1v1 and 2v2 ExplodeProjectile,
ExplodeProjectile (chance), EventualKnockback and EventualKnockback (chance)
all report **0.00% damage error**, and the whole-suite figures are unchanged:
145/156 outright, 5 draw-edge, 6 opposite, 13.82% tick error.

### A knockback big enough to cross the room separates the two engines

Worth writing down, because it looked like a porting bug and is not. A squad
given a 200-speed knockback (the shipped `EventualKnockback` value, which
travels 133 world units) agrees 16/16 across the engines when the attacker is
melee, and 12/16 when it is a Shooter:

| push speed | travel | melee | ranged |
|---|---|---|---|
| 0 | 0 | 16/16 | 16/16 |
| 20 | 0.7 | 16/16 | 16/16 |
| 50 | 4.2 | 16/16 | 16/16 |
| 100 | 16.7 | 16/16 | 16/16 |
| 200 | 66.7 | 16/16 | **12/16** |

The mechanic itself is not what differs. Against an immobile enemy, where the
only thing the knockback changes is how far the target is thrown and how long
the shooter takes to close, the two engines land shots at exactly the same
times with the knockback on (0.75, 2.5, 4.5, 6.5, 8.5 s) -- and drift by one
0.25 s bucket *without* it, which is the ordinary tick drift the suite already
reports. Giving the shooter unlimited reach so it never walks leaves the
disagreement at 13/16, so it is not the path either.

What a long push buys is fight length: the pursuit runs for seconds instead of
ticks, and the existing per-tick drift has that much longer to move a swing
across a cooldown boundary. It is not sensitive to a one-off nudge -- neither
engine flips when starting positions move by 1e-4, or when the push speed moves
by 0.1% -- which is what made it look like logic rather than accumulation.
