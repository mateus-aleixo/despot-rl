# The RL environment

`rl/` is the learning side. `sim/` stays a pure simulator with no RL concepts in
it; everything here is a wrapper.

    rl/env.py         Gymnasium env over RunState (the high level)
    rl/placement.py   placement policies (the low level)
    rl/heuristic.py   the hand-written run-level baseline
    rl/train.py       masked PPO

`tools/validate_rl.py` is the check suite, `tools/placement_compare.py` measures
whether placement matters, and `tools/run_policies.py` compares run policies.

## The high level: `DespotRunEnv`

One run per episode. The policy chooses where to move, what to buy and which
mutation to take; battles resolve inside `RunState.apply`. There is no `feed`:
moving feeds the team, so food is the price of a room rather than a decision.

> This section describes the **current** interface, 195 observations by 33
> actions. Both numbers have moved eight times, and the sections below are in
> the order the changes happened, so a table further down may be quoting a
> shape that no longer exists. The history is at the end of this list.

**Observation** is 195 floats, scaled into roughly unit range, with gold, food
and Power through `log1p` because they grow without bound:

| floats | what |
|---|---|
| 15 | scalars: level, gold, food, hunger level, moves left, squad size, mutations held, damage penalty, step fraction, mean and minimum unit level, mean progress toward the next unit level, shop level, squad Power, and squad Power against the Power of the room being stood in |
| 4 | the level as a whole: distance to the boss, fraction of rooms cleared, room count, fraction of the food shops still holding something |
| 11 | squad composition by player class |
| 9 | the kind of room being stood in, one-hot |
| 14 | the item shelf: quality and affordability, per slot, over seven slots |
| 10 | the food shop: stock and affordability, per pack, over five packs |
| 72 | the mutation shelf: seven per slot over ten slots (present, implemented, fraction of the squad touched, relative Damage and Health deltas, the largest other stat delta, and whether it attaches a passive), plus the shop's takes left and the shrine's stock |
| 60 | what lies one step away, per direction: whether there is a room, whether it is cleared, whether it is nearer the boss, and its kind one-hot |

**Actions are fixed and masked**, not variable-length: five moves (north, south,
west, east, portal) then 28 non-move actions, being seven item slots, five food
packs, ten mutation slots, and then reroll, upgrade shop, buy experience, buy
human, take mutation and sacrifice.

**A move is a direction, not a room.** It was a room id until levels started
being generated per level, at which point ids stopped being stable and a
per-room index stopped meaning anything across episodes. Everything else is
still one index per thing rather than one parameterised action, for the reason
that survived: a variable-length action list would relearn the meaning of index
3 every step, which is also why the shop is one index per slot rather than one
`buy_item`.

Both shapes have moved with almost every change to the environment, and **an
agent saved before a change does not load after it**:

    observations  108 -> 128 -> 138 -> 111 -> 134 -> 184 -> 194 -> 195
    actions        28 ->  37 ->  36 ->  40 ->  23 ->  33 (stable since)

progression and the food fix, the food shop, per-level maps, treasure rooms,
the described mutation shelf, the passive flag, and the shrine's stock bit, in
that order. Each has its own section below.

`action_mask()` marks what is currently legal. Stepping an illegal action is a
**no-op with a -0.05 penalty rather than an exception**, so a policy that has
not yet learned the mask still trains instead of crashing the rollout.

**Reward** is dominated by progress: `+10` per level gained, `±1` per fight
won or lost, `-0.5` per human lost, `-0.02` per step to discourage dithering,
`-2` on a wipe and `+50` for finishing the run.

## The low level: placement

The player zone is a 7x7 block of `p` cells, so placement is one cell per unit
inside that block. A placement policy is any callable
`(grid, layout, specs, rng) -> [(row, col), ...]`, installed on `RunState` and
called by `fight`. Two references ship: the random spread, and `frontline`,
which puts melee at the enemy-facing column and ranged at the back.

Placement measurably matters, which is the case for having a low level at all.
Same squads, same enemies, only the starting cells differ:

| squad | placement | wins | survivors | damage taken |
|---|---|---|---|---|
| 3 melee + 3 ranged | random | 3/24 | 0.25 | 1008 |
| 3 melee + 3 ranged | frontline | 3/24 | 0.58 | 983 |
| 2 melee + 4 ranged | random | 0/24 | 0.00 | 980 |
| 2 melee + 4 ranged | frontline | **5/24** | 0.58 | 942 |
| all ranged | random | 0/24 | 0.00 | 540 |
| all ranged | frontline | 0/24 | 0.00 | 540 |

For a mixed squad, positioning alone converts 0 wins into 5. The all-ranged row
is identical under both policies, which is the right sanity check: with no melee
there is no front line to form, so `frontline` degenerates to the same choice.

## Training

`rl/train.py` is masked PPO with GAE, an MLP actor-critic, and the mask applied
to the logits so illegal actions cannot be sampled and entropy is computed over
the legal set only.

It runs on CPU by design. The GPU (an RTX 3060, cu126, confirmed working) does
not help yet: rollouts are the bottleneck at roughly 60-100 environment steps
per second, because every move can trigger a full battle. The network is tiny by
comparison. GPU time only starts to matter once the Rust core makes rollouts
cheap.

## The greedy collapse, and what it was

The first 24k-step run trained cleanly by every metric that was being watched:

    update  5   steps  2560   mean level 2.38   mean return  2.97
    update 30   steps 15360   mean level 3.73   mean return 24.25
    update 47   steps 24064   mean level 3.90   mean return 25.76

and then evaluated at **mean level 1.00, mean return -8.00** -- worse than the
random baseline's 2.10, and worse than a 2048-step model that scored 3.25.

The return gave it away: `-8.00` is exactly `400 x -0.02`, the step cost over a
full episode with nothing else happening. The greedy policy picked `feed` on
every single step, spent the larder down to zero in three steps, and then looped
on an action that no longer did anything.

`feed` was unconditionally legal. Sampling during training stepped past it often
enough that the average return kept climbing; `argmax` locked onto it. So the
training curve and the evaluation disagreed because they exercised different
things, and only the evaluation was telling the truth.

`feed` is now legal only when `hunger_level > 0` and there is enough food, and
`tools/validate_rl.py` asserts that **every legal action changes state**, by
deep-copying the run and diffing a snapshot after each one.

This is the second time the same shape of bug has appeared: unlimited shop stock
let the heuristic policy buy forever, and an always-legal `feed` let the greedy
policy idle forever. The rule worth keeping: any action a policy can repeat must
either change the world or be illegal. An agent will find the one that does
neither, and it will not look like a bug in training -- it looks like a plateau.

## Where the first agent lands

After the no-op fix, greedy evaluation agrees with training instead of
contradicting it -- which is the evidence that the no-op was the whole problem:

| | before the fix | after |
|---|---|---|
| training mean level | 3.90 | 3.90 |
| greedy eval mean level | **1.00** | **3.92** |

All three agents on identical seeds and the same `frontline` placement, 30 runs:

| agent | mean level | median | max | mean return | mean steps |
|---|---|---|---|---|---|
| random | 2.37 | 2.0 | 4 | -1.95 | 116.8 |
| hand-written heuristic | 3.37 | 4.0 | 4 | 19.95 | 25.9 |
| PPO, 24k steps | **3.80** | 4.0 | **5** | **24.78** | **12.7** |

24k environment steps is a very small budget, and the agent already beats a
heuristic that encodes what a person would tell you to do. The step count is the
more interesting column: PPO reaches a deeper level in **half the actions**, so
it is not just surviving more, it is wasting fewer moves -- which matters here
because every move costs food.

An earlier heuristic measurement of 4.60 in `tools/run_policies.py` is not
comparable to the 3.37 above: it predates the `feed` fix and used the default
random placement and different seeds. `tools/compare_agents.py` is the
apples-to-apples one.

Nothing yet finishes 12 levels, so there is a lot of headroom. The obvious next
moves are a learned placement policy at the low level (the hand-written
`frontline` is a weak baseline that only knows "melee forward"), a longer
training budget, and the Rust core -- rollout speed decays from ~120 to ~17
steps/s as episodes lengthen, and that is the binding constraint, not the
network.

## The learned low level

`frontline` was always a placeholder: it knows "melee forward, ranged back" and
nothing else. Replacing it with a learned policy is what makes the hierarchy a
hierarchy rather than a run-level agent with a hand-written subroutine.

Three things had to change before it could be trained at all.

**The policy is shown the enemies.** `RunState.fight` used to place the player
side first and deploy the enemies afterwards. The game shows you the enemy line
before a fight starts, so the order is now reversed and the placement policy
takes a fifth argument: the already-deployed opposing agents. The two reference
policies ignore it, so their behaviour is unchanged, but the rng draws now
happen in a different order and fight outcomes on a given seed are not
comparable with the tables earlier in this file.

**Fights come from real runs.** `rl/place_env.py` plays actual runs with the
heuristic and captures every fight it walks into, keeping the squad, the
mutations, the hunger penalties and the enemy pack. Inventing a distribution of
fights would train a policy for fights the game never deals.

**The action space is spatial.** An episode is one fight and one action per
unit: pick a cell, then the next unit, with taken cells masked out. The network
is a small convolutional stack over the room canvas ending in a 1x1 convolution,
so it scores *cells*, and the layout's own cell list selects the legal ones. A
flat head over a fixed action list would have to relearn what "one row up" means
for every one of the 77 layouts, which are not even all the same size.

## Two failures before it learned anything

The first version was PPO with a learned critic over independent fights, and it
did not move the policy at all: 20k episodes, entropy holding it near uniform,
greedy evaluation flat at the heuristic's level. The diagnosis is that the critic
can predict how hard a *fight* is but not how good a *placement* was inside it,
so nearly all of the advantage was fight-draw noise.

Cloning the heuristic first fixed the starting point but not the gradient. Twenty
epochs of cross-entropy on `frontline`'s own choices reaches about half the cells
exactly and scores level with it, which at least proves the encoding and the head
can represent a sensible placement. But a single PPO update then knocked it back
below the heuristic, and Adam's cloning moments carrying into PPO made the first
few updates worse still.

What worked is a within-fight baseline. Each update now rolls the **same fight
with the same battle seed eight times** and uses a leave-one-out mean as the
baseline, so the difficulty of the fight, the layout and the RNG stream all
cancel and the advantage is purely "was this placement better than the other
seven". The critic is gone from the objective. With that, plus the cloned start,
a low learning rate and a KL cap per batch, the policy trains.

## Where placement can and cannot matter

Before reading any of the training numbers it is worth knowing the ceiling.
`tools/placement_eval.py` measures every policy on the same fights and the same
seeds, and adds a bound that is not a policy at all: eight random placements
scored *after* the fact, best one kept.

On 120 fights taken straight out of real runs, 6 seeds each:

| policy | win rate | mean score | vs frontline (95% CI) |
|---|---|---|---|
| random | 0.719 | 0.795 | -0.035 [-0.089, +0.019] |
| frontline | 0.725 | 0.831 | +0.000 |
| learned | 0.725 | 0.837 | +0.007 [-0.007, +0.021] |
| best-of-8 (hindsight) | 0.747 | 0.968 | +0.137 [+0.071, +0.203] |

Every one of those 120 fights was either won on all six seeds or lost on all six,
whatever the placement. The heuristic is not measurably better than a random
spread, and hindsight selection over eight placements buys 2.2 win-rate points.
**On the fights a run currently produces, placement is close to irrelevant** --
which is also why the first PPO attempt had nothing to learn from.

That is a fact about the fight distribution, not about placement. `sample_close_scenarios`
rescales the enemy side until the outcome actually depends on where the units
stand, bisecting on the win rate *across placements* rather than across seeds --
an important detail, because given a squad and a multiplier the fight is nearly
deterministic, so seed noise finds almost nothing while placement variance finds
plenty.

On 200 held-out fights rescaled that way, 4 seeds each:

| policy | win rate | mean score | vs frontline (95% CI) |
|---|---|---|---|
| random | 0.517 | 0.097 | -0.257 [-0.458, -0.055] |
| frontline | 0.610 | 0.354 | +0.000 |
| **learned** | **0.679** | **0.533** | **+0.178 [+0.017, +0.340]** |
| best-of-8 (hindsight) | 0.989 | 1.445 | +1.091 [+0.907, +1.276] |

So the learned policy beats the heuristic on the fights where placement decides
anything, and covers about a sixth of the distance to a bound that cheats.

**Training set size was what made that significant.** The same recipe on 150
fights scores +0.107 [-0.109, +0.323] on this held-out set -- the same sign and
not distinguishable from zero. Measured instead on its own 60-fight held-out
set, that model looked much stronger (+0.485 [+0.128, +0.843]), which is a
reminder that 60 fights of this variance is not enough to rank two policies.
Four times the fights and twice the episodes is what turned it into a result.

Where the wins come from, for the 600-fight model:

| bucket | fights | random | frontline | learned | best-of-8 |
|---|---|---|---|---|---|
| frontline always loses | 77 | 0.500 | 0.000 | 0.351 | 0.994 |
| close (0 < p < 1) | 3 | 0.417 | 0.667 | 0.917 | 1.000 |
| frontline always wins | 120 | 0.531 | 1.000 | 0.883 | 0.985 |

Random winning half of the fights the heuristic never wins is partly by
construction: the scaling targets a 50% win rate *across placements*, so an
average placement wins half of everything by design. What is not construction is
that `frontline` scores zero in that bucket, which puts it well below an average
placement there. Lining ranged units up along the back row is a bad answer to
some packs. The learned policy recovers a third of those while keeping 88% of the
fights the heuristic always wins, and the net is positive.

## A longer budget at the high level

The Rust core made a 12x bigger training run cheap, so the high level got 300k
environment steps instead of 24k: `runs/ppo_long.pt`, 16 envs, horizon 128,
`--fast-core`.

    update   5   steps  10240   mean level 2.77   2,587 steps/s
    update  20   steps  40960   mean level 3.60   2,143 steps/s
    update  75   steps 153600   mean level 3.67     600 steps/s
    update 145   steps 296960   mean level 3.87     444 steps/s

    trained (greedy): mean level 4.00

Two things to read off it. Throughput falls by a factor of six across the run,
because a policy that survives longer fights deeper levels with bigger squads,
and those battles cost more -- the steps/s figure is a property of how well the
agent is playing, not of the machine. And the level curve is flat from about
40k steps: **twelve times the data bought roughly one tenth of a level.**

The greedy run is also worth looking at next to the 24k agent, because the
headline number improves while the return collapses:

| | 24k steps | 300k steps |
|---|---|---|
| mean level | 3.71 | **4.11** |
| mean return | 24.01 | 10.01 |
| mean steps per run | 12.1 | 100.7 |

The long agent reaches level 4 more reliably and takes eight times as many
actions to do it. Every step costs -0.02 and every lost human -0.5, so a return
that falls while the level rises means it is grinding: fighting more, losing more
humans, and buying depth with attrition. That is a reward-shaping result, not a
skill result -- the step cost is too cheap relative to a level to stop it.

## The hierarchy end to end

`tools/hierarchy_eval.py` crosses each run-level agent with each placement policy
over identical run seeds. 100 runs per cell, with the 300k run agent and the
600-fight placement policy:

| run agent | placement | mean level | mean return | mean steps | vs frontline, per seed |
|---|---|---|---|---|---|
| heuristic | random | 3.46 | 20.96 | 27.2 | -0.01 [-0.12, +0.10] |
| heuristic | frontline | 3.47 | 21.12 | 26.4 | +0.00 |
| heuristic | learned | 3.55 | 22.01 | 27.0 | +0.08 [-0.01, +0.17] |
| PPO | random | 4.04 | 10.05 | 104.7 | -0.07 [-0.26, +0.12] |
| PPO | frontline | 4.11 | 10.01 | 100.7 | +0.00 |
| PPO | learned | 4.13 | 9.96 | 97.8 | +0.02 [-0.10, +0.14] |

Both learned-placement rows are positive and neither is significant. That is
exactly what the fight-level measurement predicts: a run agent that stalls around
level 4 never reaches fights whose outcome placement can change, so a better low
level has nothing to act on.

## Where this leaves the hierarchy

Three measurements point the same way, and it is worth stating the conclusion
plainly because it decides what to do next.

1. On fights a real run produces, placement changes nothing -- not for the
   heuristic, not for a learned policy, and barely for hindsight selection.
2. On fights rescaled until placement matters, the learned policy beats the
   heuristic by a clear margin. The low level works; it just has no work to do.
3. More training at the high level does not get past level 4, and what it does
   buy, it buys by grinding.

So the binding constraint is neither rollout speed (the core fixed that) nor the
low level (it is trained and it wins where it can). It is that **the run agent
plateaus at a third of the game**, and the reward as written pays it to plateau
slowly rather than to push. The next thing worth trying is the reward and the
observation at the high level -- what a level costs, what hunger and attrition
should cost against it -- not more steps, and not more placement.

## What is where

    rl/heuristic.py          the hand-written run-level baseline, in one place
    rl/place_env.py          fights sampled from runs, the placement MDP, the
                             close-fight rescaler, the scenario cache
    rl/place_policy.py       PlacementNet and the wrapper a RunState can use
    rl/train_placement.py    cloning plus PPO with leave-one-out advantages
    tools/placement_eval.py  policies and the hindsight bound, paired, with CIs
    tools/hierarchy_eval.py  both levels crossed, over whole runs
    tools/show_placement.py  what a policy actually does, drawn on the room

    tools/run_autopsy.py     how runs end and where the actions go
    tools/food_probe.py      what a `buy_food` action share hides: offers taken,
                             shop visits, and food bought against food eaten
    tools/ceiling_probe.py   how much room is above the baseline: per-seed
                             agreement, a dice bound, and gold spent per Power
    tools/shop_eval.py       whether upgrading the item shop pays, paired
    tools/power_curve.py     squad Power against room Power, level by level

    runs/placement_final.pt              the trained low level (600 fights, 40k episodes)
    runs/food_shop.pt                    the current high level: 600k steps, seed 0,
                                         unshaped, with .150000/.300000/.450000/.600000
    runs/food_shop_b.pt                  the same budget and seed with `--shaping power`
    runs/food_shop_s1.pt                 both again on seed 1, same checkpoints
    runs/food_shop_b_s1.pt
    runs/mapgen{,_b}{,_s1}.pt            the same four on the first generated maps
    runs/levelgen{,_b}{,_s1}.pt          and on the ported LevelGenerator shape
    runs/treasure{,_b}{,_s1}.pt          and with the mutation shops and the
                                         shop-weighted gold
    runs/mutshelf{,_b}{,_s1}.pt          and with the shelf described in the
                                         observation, but before the mutation
                                         passives were implemented
    runs/passives{,_b}{,_s1}.pt          the same four with the passives live and
                                         the hooks in the Rust core, the current
                                         environment
    runs/arm_live_s0..s11.pt             twelve seeds of that same configuration,
                                         the sample the noise figure comes from
    runs/arm_inert_s0..s11.pt            the same twelve trained with the mutation
                                         hooks inert, graded on the live environment
    runs/food_budget.pt                  the previous high level, free food and a stub
                                         food shop, and it no longer loads
    runs/prog_a_baseline.pt              progression, but the old food model
    runs/fed_a_baseline.pt               the high level before progression
    runs/fed_b_power.pt                  the same with potential shaping
    runs/scenarios_close_600_200_0.pkl   the close-fight set, 600 train + 200 held out

`runs/ppo_long.pt` and everything before it predate the three run-layer fixes, so
their numbers are not comparable with anything measured after. The `fed_*`
agents in turn predate progression and no longer load at all: the observation
went 108 -> 128 and the action space 28 -> 37. See the progression section at
the end of this file.

`rl/heuristic.py` exists because four tools and the scenario sampler each
carried their own copy of the baseline. That was harmless while the run layer
had one shop action and stopped being harmless the moment it had four, since
the thing every agent is measured against could then drift per tool.

The scenario caches matter more than they look: tuning one close fight costs
about 130 battles, so the 800-fight set is 12 minutes of sampling that every
later measurement reuses instead of redoing.


## Reward shaping at the high level, and what it ran into

The plateau at level 4 looked like a reward problem: the long run bought depth by
grinding, so the step cost was presumably too cheap and the level bonus too flat.
The first thing that did was force an autopsy of what the agent was *doing*
(`tools/run_autopsy.py`), and that turned up three places where the sim was
cheaper than the game. They are written up in `notes/reference-sim.md`; the short
version is that humans could be bought in any room rather than only at an item
shop, a squad larger than the deployment zone stacked onto the same cells instead
of leaving people out, and feeding cost a flat 10 instead of `M_Food.needed`,
which the binary sets to the squad's unit count. The 300k agent had found the
first one and was spending 73% of its actions on it.

Fixing those is not reward shaping, but nothing measured before them means much.

### The four rewards

All four trained for 150k steps with the Rust core, identical seeds, and were
scored on the **unshaped** objective (a shaped agent graded on its own bonus
proves nothing):

- **A, baseline.** `+10` a level, `±1` a fight, `-0.02` a step, `-0.5` a human
  lost, `-2` on a wipe, `+50` for the run.
- **B, potential shaping.** A adds `gamma * phi(s') - phi(s)` with
  `phi = 3 log1p(power/1000) + min(1, food / 3 feedings)`. Potential-based is
  the form that cannot change which policy is optimal, so it is a claim about
  learning speed, not about strategy. The two terms come from the autopsy: runs
  ended wiped while holding 115 gold and 130 food, so the agent was dying rich.
- **C, shaping plus a grind tax.** B with the step cost raised to `-0.05`.
- **D, a rising level bonus.** Level *n* is worth `10n` rather than 10, because
  rooms get exponentially harder while a flat bonus and a 0.99 discount make deep
  levels worth *less*, not more.

100 runs each, same seeds, `frontline` placement:

| agent | mean level | reached 5+ | mean return | wiped | how it spends actions |
|---|---|---|---|---|---|
| heuristic | 3.47 | 2/100 | 21.12 | 100 | 57% buy_item |
| A baseline | 3.71 | 1/100 | 18.98 | 93 | 46% move, 22% buy_human, 22% sacrifice |
| B shaping | 3.70 | 0/100 | 21.86 | 26 | 77% move, 12% feed |
| C shaping + tax | 3.67 | 0/100 | 22.02 | 50 | 76% move, 12% feed |
| D rising bonus | 3.71 | 1/100 | 24.01 | 100 | 92% move |

**Every reward lands on level 3.7.** Shaping changes behaviour a great deal --
B wipes in 26 runs out of 100 where the baseline wipes in 93, and instead sits
out the step cap holding 2,509 unspent gold -- but it does not go deeper. It buys
survival, not progress.

(An earlier version of this table, run before the feeding fix, had B at mean
level 3.86 with 16 runs past level 5. That was the flat feed cost: B had bought
70 humans and was feeding all of them for the price of ten. The number was real
and the mechanism was a bug, which is the whole reason the fixes came first.)

### Why no reward could have worked

`tools/power_curve.py` measures the two curves that decide whether a run can
continue: the Power a room is filled to, from `Levels.json`, and the Power the
squad has when it arrives there.

| level | squad | squad Power | room Power | ratio | ratio vs level 1 |
|---|---|---|---|---|---|
| 1 | 5.0 | 20,621 | 800 | 25.8 | 1.00 |
| 2 | 4.8 | 20,535 | 2,450 | 8.4 | 0.33 |
| 3 | 4.0 | 18,563 | 3,920 | 4.7 | 0.18 |
| 4 | 3.7 | 19,186 | 7,700 | 2.5 | 0.10 |

**The squad never gets stronger.** Its Power is flat at about 20,000 from the
first room to the last, while the room budget grows 14x over four levels, so the
squad arrives at level 4 with a tenth of the relative strength it had at level 1.
The only thing that changes is that there are fewer of them.

That is a missing mechanic, not a missing incentive. Three things in the run
layer are stubbed:

- `buy_item` overwrites a **random** member's item with a **random** damaging
  item, so shopping is a coin flip rather than an upgrade. The real shop draws
  from per-shop pools with `ItemShopData`'s `QnProb` quality weights and the
  player picks who gets it.
- Item levels never rise. `Human.experience` accumulates and
  `ExperienceForLevels` is loaded, but `unit_level` is never consumed.
- A bought human is a bare Novice, and there is no way to arm one except by the
  same random overwrite.

So gold has nothing to buy. B ending on 2,509 gold is not hoarding, it is an
agent with no conversion from money to power, which is exactly why every reward
function lands in the same place.

**The next step is progression, not reward.** Item quality and levels, item
levels rising with experience, and equipping a chosen human are what would make
level 5 reachable; after that the reward question can be asked again, and B's
shaping is the variant worth re-testing first, because it already buys survival.

## Progression, and why every number above it is stale

The plateau was not a reward problem and not a training-budget problem. It was
that the run layer had no way to convert gold into strength, so all four reward
variants landed on level 3.7 and squad Power sat flat at ~20,000 while the rooms
grew from 800 to 7,700.

The mechanics are written up in `notes/reference-sim.md` under "Progression:
what gold is for" -- the shop is one model for the whole run with a level, a
quality-weighted stock and per-item prices; experience comes from dead enemies
and from the shop at 400 for 1 gold; a unit level rescales both the class row
and the item, and the item's per-level term compounds rather than adding.

**Three of the four fixes were sim bugs, not missing features, and each made the
run cheaper than the game.** They are why nothing above this section is
comparable with anything below it:

1. **The squad started fully armed.** `RunState.new` read the keys of
   `Game.Team.Items` as a draw pool. `C_Session.LoadPlayerTeam` reads
   `Game.Team.Packs`, which is five bare `Novice` rows with no item. The run
   opened at ~20,000 Power against an 800-Power first room. The Power curve was
   flat because it started at the ceiling, not because nothing raised it.
2. **Gold was paid on every move.** `C_Room.AfterFight` pays `M_Room.goldReward`
   only when it is non-zero and sets it to zero after paying, so a room pays
   once. Paid per move it is an unbounded fountain, and the first agent trained
   on the new shop found it immediately: 57 of 60 runs ran to the 400-step cap,
   22% of its actions were shop rerolls, and it finished holding 1,115 gold.
3. **An item cost the shop's upgrade price.** Every item cost the same, so
   quality bought nothing.

That is the fourth instance of the same pattern, and the count is the point: an
agent finds whatever is cheaper in the sim than in the game, and it looks like a
strategy rather than a bug. `tools/validate_sim.py` now checks that pacing two
cleared rooms earns nothing, alongside the existing check that every legal
action changes state.

### What changed in the environment

The observation is 128 floats, up from 108: the six new scalars are mean and
minimum unit level, mean progress toward the next level, the shop level, log
squad Power, and squad Power against the Power of the room being stood in. The
last one is the readiness ratio the power curve plots, which the policy
previously had no way to see. Fourteen more floats describe the shelf, two per
slot: quality and whether it is affordable.

The action space is 37, up from 28. The shop is **one action per slot**, seven
of them, so slot 3 is always slot 3 and "the expensive one" is learnable; a
single `buy_item` could only ever mean "an item". Reroll, upgrade and buy
experience are three more. Who receives a bought item is not in the action
space: that is a second choice on top of a choice, the same shape as placement,
so `RunState.best_item_target` hands it to whoever the game's own Power
statistic says gains most.

Every saved agent predates both shapes. `tools/hierarchy_eval.make_ppo` now
compares the checkpoint against the environment and raises `StaleCheckpoint`
rather than dying inside a matrix multiply, and the tools skip a stale agent
instead of crashing on import.

### What it bought

150k steps, 16 envs, horizon 128, Rust core, unshaped baseline reward --
the same protocol as the four reward variants above. 60 runs on identical seeds,
`frontline` placement:

| agent | mean level | reached 5+ | mean return | how runs end |
|---|---|---|---|---|
| random | 1.18 | 0/60 | -3.75 | wiped |
| heuristic | 2.75 | 1/60 | 13.26 | wiped 60/60 |
| **PPO 150k** | **3.60** | **5/60** | **18.50** | out of steps 42, no legal action 10, wiped 8 |

The headline level looks like the old 3.71, and that comparison is meaningless:
the run it is playing now starts at 750 Power instead of 20,000. The measurement
that means something is the readiness curve, squad Power over the Power the room
is filled to:

| level | squad | squad Power | room Power | ratio | ratio vs level 1 |
|---|---|---|---|---|---|
| 1 | 5.0 | 750 | 800 | 0.9 | 1.00 |
| 2 | 5.0 | 8,171 | 2,450 | 3.3 | **3.56** |
| 3 | 5.7 | 12,203 | 3,920 | 3.1 | 3.32 |
| 4 | 6.1 | 15,903 | 7,700 | 2.1 | 2.20 |
| 5 | 6.8 | 15,940 | 11,200 | 1.4 | 1.52 |

Before progression the same column read 1.00, 0.33, 0.18, 0.10: the squad
arrived at level 4 with a tenth of the relative strength it had at level 1. It
now arrives at level 5 with half again as much, having grown 21x in absolute
Power over four levels. The agent also keeps everyone armed -- 5.7 humans
holding items at its peak, against the heuristic's 2.6 -- and loses 0.1 fights
per run to the heuristic's 1.0.

The heuristic is the one that got *worse* (3.47 to 2.75), which is the right
sign: it is the same dozen lines playing a game that no longer hands it a free
squad.

### What it did not fix

- **The step cap is still where most runs end**, 42 of 60 at 400 steps, and 75%
  of the agent's actions are moves. The grind the reward-shaping section
  diagnosed is still there; progression gave the agent something to spend on,
  not a reason to hurry.
- **The agent never upgrades the shop**, peak shop level 1.00 across all 60
  runs, so it buys quality-1 and quality-2 items all run and never opens the
  quality-5 pool where an item is worth ten times as much. That looked like a
  gap in the agent. It is not -- see the next section.
- **Ten runs in sixty end with no legal action**, starved in a corner with no
  move left. That state was rare when food was cheap relative to a five-human
  squad and is now the second most common ending -- and at a longer budget it
  becomes the *only* ending, because the reward does not charge for it. See the
  last section.

With the squad able to grow, the reward question from the previous section can
be asked again, and variant B's shaping is still the one to re-test first.

### The shop upgrade does not pay, and the agent was right not to buy it

A trained agent that never takes an available action is either missing something
or reading the mechanic correctly, and those look identical from the outside.
`tools/shop_eval.py` separates them by taking the agent out of it: the same
hand-written baseline, with the shop upgrade moved in its priority order, over
200 identical run seeds.

| upgrade | mean level | 5+ | mean return | peak shop | peak Power | vs never |
|---|---|---|---|---|---|---|
| never | 2.83 | 1 | 14.17 | 1.00 | 7,176 | +0.00 |
| only when the shelf is dull | 2.90 | 2 | 14.96 | 3.17 | 8,532 | +0.07 [-0.02, +0.17] |
| whenever affordable | 1.85 | 1 | 3.63 | 3.96 | 4,575 | **-0.97 [-1.19, -0.76]** |

(Measured again after the food fix in the last section, which moves every level
number up about a quarter of a level and leaves both differences where they
were: +0.07 [-0.03, +0.18] and -0.92 [-1.11, -0.72] before it. The conclusion
does not depend on the food model.)

Buying it eagerly is **clearly worse**, and buying it conservatively is worth
nothing measurable *to this baseline*. (That last qualifier turned out to carry
weight: see the final section. A 600k agent playing a corrected food model does
upgrade, and is the best agent measured. This table is about a policy that dies
on level 3 and never banks the purse quality 3 to 5 needs.)

The mechanism is in the last two columns. A full upgrade path costs
3 + 6 + 12 + 12 = 33 gold, and the run opens with 10. An eager upgrader spends
its first three shops on the shop instead of on weapons and fields an unarmed
squad: peak Power 4,575 against 7,176 for the policy that never upgrades at all.
What a higher shop level buys is access to quality 3 to 5, which cost 18 to 43
gold each -- and a run that ends on level 2 or 3 never banks that much at 11 to
13 gold a room.

That is the same conclusion the placement work reached from the other end. The
low level was fine and had no work to do because the run never reached fights
where placement mattered; the shop upgrade is fine and has no work to do because
the run never reaches a purse where quality is reachable. **Both point at
survival on levels 2 to 4 as the binding constraint**, not at the mechanic being
measured.

### A longer budget, and the ending the reward does not price

`--seed` and `--checkpoint-every` were added to `rl/train.py` for this: one
seeded 600k run saving every 150k gives a budget curve with no network-init or
sampling difference between its points, which two separate runs do not. 60 runs
per checkpoint, identical seeds, `frontline`:

| steps | mean level | 5+ | mean return | peak shop | peak unit level | peak Power | how runs end |
|---|---|---|---|---|---|---|---|
| 150k | 3.53 | 5/60 | 17.91 | **1.50** | 4.25 | 16,495 | 35 capped, 16 wiped, 9 stranded |
| 300k | 3.52 | 0/60 | 22.10 | 1.00 | 4.33 | 14,028 | 37 stranded, 15 capped, 8 wiped |
| 450k | 3.63 | 0/60 | 23.58 | 1.00 | 4.57 | 14,559 | 35 capped, 15 stranded, 10 wiped |
| 600k | 3.02 | 0/60 | 15.48 | 1.00 | 5.00 | 13,695 | **56 stranded**, 4 capped |

**On the shop, this is the clean answer.** At 150k the agent's peak shop level
is 1.50, so it does buy the upgrade; by 300k it is 1.00 and stays there. The
action was explored and then unlearned, which is what a policy does with an
action that costs more than it returns -- and `tools/shop_eval.py` above says
that is exactly what this one does. Not a gap in exploration.

**On the budget, 150k to 450k is flat** (3.53, 3.52, 3.63) and 600k is worse.
The training curve inside the run says the same: it reaches mean level ~3.7 by
about 200k and then oscillates between 3.0 and 3.7 for the remaining 400k.

The last column is the interesting one, and it is a bug in the reward rather
than a fact about the game. A run can end three ways, and the reward prices only
two of them:

    wiped      squad empty        -> terminated, -2
    capped     400 steps          -> truncated, no bootstrap
    stranded   no legal action    -> `if not mask.any(): return ..., 0.0, True`

**Stranding is free.** The squad is alive, so `terminated = st.finished or not
st.squad` is False and the `-2` never lands; the episode ends on the *next*
call, through the early return, with a reward of exactly zero. Dying costs 2,
being soft-locked costs nothing, and the difference is a strategy: stop moving,
never lose a fight, keep every level bonus already banked.

That is what the 600k agent converged to. It ends 56 runs out of 60 holding
**0.0 gold, 0.0 food, hunger 1.90 and no legal move**, having won 5.1 fights and
lost 0.0. Buying experience is 16% of its actions, up from 4%, and its peak unit
level is a clean 5.00: it converts the entire purse into levels, stops, and sits
there. The extra 150k steps did not make it worse at the game, they made it
better at the exploit.

`M_Food.canMove` is `canFeed || movesLeft > 0`, so the game does have this
state and does treat it as the end of the run. Pricing it as a loss looked like
the whole fix. Checking that assumption against `C_Food` first is what turned up
the next section, and the reward was the smaller half of the problem.

### The food model was wrong, and that is what made stranding reachable

Before pricing the stranded ending as a loss, the thing worth checking was
whether the game has that ending at all. It does -- and reading `C_Food` to find
that out showed the sim had the whole food model wrong. The correct one is
written up in `notes/reference-sim.md`; three things were different:

1. **Moving feeds you.** `C_Food.Move` calls `Feed()` and returns when the
   larder covers `needed`, so a room costs one food per human and spends no
   move. The sim charged nothing for a move and offered feeding as an action.
2. **`C_Food.Feed` has no UI call site.** Its three callers are `Move`, a
   mutation, and the sacrifice handler. **`feed` was never a player action**, so
   the run layer had one action too many and the action space is now 36, not 37.
3. **`moves` is a threshold list, not an allowance.** `maxMoves = moves[0]`,
   `movesLeft` is a reserve that drains only while broke, and `hungerLevel` is
   derived by comparing the reserve against `moves` rather than incremented --
   so eating clears hunger in a single step instead of a stage at a time.

The consequence for the agent is the point. With feeding optional, food was
nearly free: you paid only when you chose to, which is why runs ended holding
130 food and why the reward-shaping section had to add a food term by hand to
get the agent to care. With feeding automatic, food is the standing price of
walking around, and a large squad is expensive to keep moving in exactly the way
`M_Food.needed` says it should be.

Stranding is still a real state -- `canMove = canFeed || movesLeft > 0` -- so
`RunState.apply` now ends a run that has no legal action as `finished, won =
True, False`. That is what makes the environment charge it the same `-2` as a
wipe, and `tools/validate_sim.py` asserts the invariant over random play: a run
with no legal action must already be finished.

That makes five sim-fidelity bugs found the same way, by looking at what an
agent did rather than at what it scored. This one is the largest of them, and it
was found by checking a premise before acting on it: the reward fix I was about
to make would have been correct and would have left the real bug in place.

### What the food fix bought, and the shop answer changing sign

Same protocol again: one seeded 600k run, `--checkpoint-every 150000`, 60 runs
per checkpoint on identical seeds. The only difference from the table two
sections up is the food model and the stranded ending.

| steps | mean level | 5+ | mean return | peak shop | peak Power | how runs end |
|---|---|---|---|---|---|---|
| heuristic | 3.08 | 1/60 | 16.90 | 3.40 | 8,739 | 60 wiped |
| 150k | 3.63 | 0/60 | 24.43 | 1.00 | 14,742 | 52 wiped, 8 stranded |
| 300k | 3.82 | 0/60 | 26.89 | 1.00 | 13,649 | 49 stranded, 11 wiped |
| 450k | 3.82 | 0/60 | 28.08 | 1.00 | 14,215 | 50 stranded, 10 wiped |
| 600k | **3.90** | **5/60** | **29.53** | **3.67** | **17,991** | 32 wiped, 28 stranded |

Four things moved, and the first two are the fix working:

**The 400-step cap is gone.** Not reduced -- gone. No run out of 240 reaches it,
where 35 to 57 of every 60 did before. Hitting the cap was never a fact about
the agent; it was what free food let it do.

**Stranding is now a loss, and reads as one.** The ending still happens (28 to 50
runs a checkpoint, since a broke squad is genuinely stuck) but it is
`finished, lost` rather than a silent zero-reward stop, so it costs the same -2
as a wipe. The agent no longer aims for it: at 600k it is 28 runs against 32
wipes, where the old 600k agent had converged on 56 of 60.

**More budget now buys something.** 3.63, 3.82, 3.82, 3.90, with mean return
climbing 24.4 to 29.5 and the training curve still rising at 600k. Under the
wrong food model the same protocol was flat by 200k and worse at 600k. The
budget was not the limit; the exploit was.

**And the shop answer changes sign.** 600k is the first checkpoint that upgrades
the shop -- peak level 3.67, `upgrade_shop` 3% of actions -- and it is also the
only one that reaches level 5, with peak squad Power 17,991 against about
14,000 everywhere else. Under the old model no checkpoint ever upgraded at any
budget.

That is one seed, and it is not proof that the upgrade *causes* the rest: at
600k the agent is better at everything at once. The paired A/B two sections up
still stands on its own terms and still says +0.07 [-0.02, +0.17] for upgrading
conservatively -- but it measures the *heuristic*, which reaches level 3.08 and
never banks the purse that makes quality 3 to 5 reachable. An agent that reaches
level 4 with 64 gold in hand is answering a different question, and the honest
reading is that the upgrade is worthless to a policy that dies on level 3 and
starts to pay to one that does not.

The heuristic is now the weakest thing in the file by a wide margin (3.08
against 3.90) and it ignores food shops entirely, which was free when food was
free. That is the next thing to fix in it, after the food shop stops being a
stub -- see `notes/reference-sim.md`.

### The food shop, and the heuristic learning to buy

The stub is gone: the shop sells the game's five packs ([7, 2] to [250, 55],
food and gold), each room stocks itself once by `DetermineQuantities` and never
restocks, and arriving on a level no longer hands over free food -- the level
row's `TotalFood` turned out to be the shops' gold budget, not a food allowance.
The mechanics and the evidence are in `notes/reference-sim.md`; what matters
here is that the run's food now has to be bought.

That moves the interfaces again. `buy_food` became one action per pack, the same
reason `buy_item` is one per slot -- pack 4 is always the 250-food one -- so the
action space is **40**, and the observation carries the shelf's stock and
affordability per pack, so it is **138**. Every agent saved before this fails to
load, `runs/food_budget.pt` included: observations have gone 108 -> 128 -> 138
and actions 28 -> 37 -> 36 -> 40.

`rl/heuristic.py` now buys. At a food shop it tops the larder up to `moves[0]`
(six) feedings' worth, taking the cheapest pack that closes the gap and the
largest it can afford when none does. The threshold is a guess; ignoring the
shops is not an option, since nothing else fills the larder.

The baseline holds where the economy got harder:

| heuristic | mean level | mean return | mean steps | how runs end |
|---|---|---|---|---|
| free food, stub shop | 3.08 | 16.90 | -- | 60 wiped |
| bought food, real shop | 3.05 | 16.51 | 26.9 | 60 wiped |

60 runs, seeds 30,000+, frontline placement, `tools/compare_agents.py` and
`tools/run_autopsy.py`. Losing every free grant and paying real prices costs the
baseline 0.03 levels, because it spends 5% of its actions on food and reaches
the end of a run still holding 19 food and 26.7 gold. Nothing starves: all 60
runs end wiped, so the food model is a tax on the purse rather than a second way
to die -- for a policy that buys. The agent numbers above are not comparable any
more and the 600k run has to be repeated against this environment.

### Shaping steadies the run; it does not deepen it

Two seeds x two rewards, 600k steps each, `--checkpoint-every 150000`, scored on
the **unshaped** objective over 60 runs on seeds 30,000+ with `frontline`
placement. A is the plain reward; B adds the potential
`phi = 3 log1p(power/1000) + min(1, food / 3 feedings)`, which is the re-test the
reward-shaping section asked for.

| checkpoint | A seed 0 | A seed 1 | B seed 0 | B seed 1 |
|---|---|---|---|---|
| 150k | 3.65 | 3.82 | 3.17 | 3.10 |
| 300k | 3.68 | 3.78 | 3.73 | 3.80 |
| 450k | 3.45 | 3.12 | 3.70 | 3.73 |
| 600k | **2.97** | **3.83** | **3.63** | **3.75** |

The heuristic is 3.05 on the same 60 runs.

**Seed 0 alone reads as a shaping result, and it is not one.** There, A falls
3.68 -> 2.97 while B holds 3.63, which looks like the food term carrying the run.
Seed 1 says otherwise: A's 600k is 3.83, its best checkpoint, and B's is 3.75.
What survives both seeds is not the mean but the spread. Over the six
checkpoints past 150k, A ranges 2.97 to 3.83 and B ranges 3.63 to 3.80 -- means
3.47 and 3.72, spreads 0.86 and 0.17. Shaping is buying variance reduction, and
on two seeds that is worth stating as a direction, not a number.

**What does hold across all four runs** is that the run's food balance goes
negative, and the share `tools/run_autopsy.py` prints for `buy_food` -- 15 to 17%
of actions at 150k, 3 to 4% at 600k -- is not what it looks like. An action share
has the episode length in its denominator, and episodes triple over the same
budget. `tools/food_probe.py` counts the things a share cannot:

| per run, 60 runs | heuristic | s1 150k | s1 600k | s0 600k | s0 600k B |
|---|---|---|---|---|---|
| takes the offer | 30% | 100% | 79% | 78% | 97% |
| priced out, ever | 0 | 0 | 0 | 0 | 0 |
| steps in a food shop | 4.3 | 9.6 | 6.6 | 3.3 | 6.5 |
| food bought | 10.4 | 64.1 | 38.7 | 19.7 | 42.5 |
| food eaten | 31.4 | 57.4 | 86.3 | 68.1 | 56.4 |
| moves | 9.2 | 14.4 | 44.0 | 42.8 | 18.5 |
| moves on the reserve | 0% | 13% | 47% | 50% | 28% |
| food left on the shelves | 24.5 | 0.1 | 0.0 | 0.0 | 0.2 |

**No agent ever stops buying.** Not one of them is priced out on a single step of
3,600, every trained checkpoint takes 78 to 100% of the offers it gets, and all
of them leave the level's shelves empty where the heuristic walks away from 24.5
food. What changes is that the 150k agent banks a surplus (64 bought against 57
eaten) and the 600k agents run a deficit (39 against 86, 20 against 68), covered
by the starting larder and then by the six-move reserve until it runs out.

**The deficit is bought with moves, not with gold.** Same depth, 3.82 against
3.83, and three times the walking: 14.4 moves a run at 150k, 44.0 at 600k, with
half of them made hungry. Food-shop visits fall while total moves triple, so as a
share of where the agent stands, the shops go from two thirds of its moves to a
seventh. It is not avoiding the shop because it cannot pay; it is walking
somewhere else.

Why walking is worth it is in the run economy. A first visit to a room pays
`GoldPerRoom`, 10 to 13 gold, and the exp of whatever is in it; the move costs
one food per human, six food, which is 1.3 gold at the shelf's best rate. Walking
returns about eight times its food cost -- while there is stock. Once a level's
shelves are empty no amount of gold buys another move, and the level's whole
supply is finite (42 food at level 1, 414 by level 5, against a squad that eats
six a room). The step cost is -0.02 and stranding is charged once at -2, so the
reward barely argues with any of this.

So the ending to explain is not a starving agent that forgot to shop. It is an
agent that clears the shelf, keeps farming rooms because rooms pay, and runs out
of a supply that gold cannot extend.

Throughput went the other way, 444 to about 2,000 steps/s, because episodes are
now 27 to 60 steps rather than pinned at the 400-step cap. Food is what paces a
run. The old 3.90 at 600k is not comparable to any row here: that agent had free
food every level and a stub shop.

## One map per level, and what it costs the action space

Every level used to reuse the shipped 22-room `Rooms.json` map. `Levels.json`
asks for 7 rooms and one food shop at level 1, and 10 to 15 rooms with two or
three later, so every level was two to three times its size with three to six
times the shops -- under a food budget written for the row. `sim/mapgen.py` now
generates the level the row asks for; `tools/validate_sim.py` checks the room
count against `MinRooms`..`MaxRooms` and each shop count against its column,
over 40 seeds x 12 levels.

**The move action had to change with it.** A move used to be one action per room
id, which only meant anything because every level was the same map. Room ids are
not stable across generated levels, so a move is now a **direction** -- north,
south, east, west, portal -- decoded against the live state. The observation
lost its four per-room vectors (here, cleared, adjacent, kind, 22 wide each) and
gained a local view instead: the kind of the room you stand in, then per
direction whether there is a room, whether it is cleared, whether it is nearer
the boss and what kind it is, plus four level-wide scalars (distance to the boss,
fraction cleared, rooms this level, food shops still stocked).

That is **observations 138 -> 111 and actions 40 -> 23**, so every agent trained
before this fails to load. Running totals: observations 108 -> 128 -> 138 -> 111,
actions 28 -> 37 -> 36 -> 40 -> 23.

The baseline moved, and not by a little:

| heuristic, 60 runs | fixed 22-room map | generated per level |
|---|---|---|
| mean level | 3.05 | 2.62 |
| reached 5+ | 0/60 | 5/60 |
| mean return | 16.51 | 14.98 |
| fights per run | 3.0 | 5.9 |
| mean steps | 26.9 | 28.4 |
| gold at the end | 26.7 | 42.7 |
| mutations per run | 0.0 | 0.6 |

A right-sized level is a harder level and a swingier one. There are fewer rooms
to farm before the boss, so the squad arrives thinner -- but the same run now
fights nearly twice as often per run, because a level that is 7 rooms instead of
22 is mostly the rooms that matter. Level 5 is reached in 5 runs out of 60 where
the fixed map never got there once, and the run also ends holding 42.7 gold
against 26.7: with fewer shops there is less to spend it on.

Mutations appear at all for the first time -- 0.6 a run. The fixed map has three
mutation shrines on every level; the rows ask for one at level 1 and none
afterwards (levels 2 onwards carry `RerollShrines`, which the sim does not
model). So the old map was handing out roughly thirty-six shrine visits a run
where the data allows one.

### Four runs on right-sized maps, and where the agent's margin went

Same protocol again: 600k steps, `--checkpoint-every 150000`, two seeds x
unshaped (A) and `--shaping power` (B), scored unshaped over 60 runs on seeds
30,000+ with `frontline` placement. `runs/mapgen{,_b}{,_s1}.pt`.

| checkpoint | A seed 0 | A seed 1 | B seed 0 | B seed 1 |
|---|---|---|---|---|
| 150k | 2.58 | 2.82 | 2.77 | 2.62 |
| 300k | 2.78 | 2.07 | 2.67 | 2.63 |
| 450k | 2.65 | 2.83 | 3.07 | 2.72 |
| 600k | 2.68 | 2.92 | 3.07 | 2.77 |

The heuristic is 2.62 on the same runs.

**The shaping result replicates.** Past 150k, A means 2.66 over a 2.07-2.92
range and B means 2.82 over 2.63-3.07: B is a fifth of a level higher and its
spread is half as wide (0.44 against 0.85). That is the same shape as the two
seeds on the fixed map, on a different map regime, which is about as much as two
seeds a side can say.

**The agent's margin over the baseline is mostly gone.** On the 22-room map the
agents reached 3.5 to 3.9 against a 3.05 heuristic; here they reach 2.66 to 2.82
against 2.62. What the oversized map was paying for is visible in the autopsy:
the agent used to make 44 moves a run against the heuristic's 9, farming a map
with three times the rooms the level called for. A level of 7 to 15 rooms has
almost nothing to revisit, so the strategy that made the difference is not
available and both policies play roughly the same game.

**Stranding was a map artifact.** On the fixed map, runs ending broke and stuck
were 42 to 57 of every 60 at the late checkpoints. Here they are 0 to 16, and
mostly under 10. `tools/food_probe.py` says why: the agents now buy more food
than they eat (92 against 61, 117 against 75, 98 against 64), leave 42 to 76 food
unsold on the level's shelves, and spend 2 to 11% of their moves on the hunger
reserve rather than 47 to 50%. The food economy stops binding once the map is
the size the food budget was written for -- and the new constraint is gold: for
the first time the agents are *priced out*, standing in a stocked shop that they
cannot afford, on 10 to 285 steps a checkpoint where the old map never managed it
once.

So the open question from the last section -- an agent that walks itself to death
-- was largely a question about the map, and it closed when the map was made the
size its own row asks for.

### The generated shape is the game's now

`sim/mapgen.py` was a stand-in: the counts were the game's, the graph was a
compact blob. It is now `LevelGenerator`'s own growth -- the weight function, the
2x2-block rations that bound it, `Adjust`, `CloseLoops`, and the
`CouldBeFinishRoom` / `ChooseStartRoom` / `ChooseDiversePositions` placement
predicates. See `notes/reference-sim.md` for what was read and
`sim/assumptions.py` for the three parameters that could not be
(`maxDeadEnds`, `minFinishDistance`, and whether a room may cover several
squares).

Levels stopped being blobs and became webs of corridors: mean degree 2.17 with
about 2.8 dead ends a map, no room with four neighbours, and the boss never on an
articulation point. The observation and action spaces did not move -- a move is
still a direction -- so the `runs/mapgen*` agents still load, but they were
trained on the blob.

| heuristic, 60 runs | fixed 22-room map | blob of the right size | LevelGenerator shape |
|---|---|---|---|
| mean level | 3.05 | 2.62 | 2.50 |
| mean return | 16.51 | 14.98 | 14.61 |
| fights per run | 3.0 | 5.9 | 6.7 |
| gold at the end | 26.7 | 42.7 | 53.7 |

Each step towards the real map is a step down for the baseline, and for the same
reason both times: the room the squad wants to skip is harder to skip. A blob of
7 to 15 rooms already had few detours; a corridor web has fewer, so the run
fights 6.7 times where the fixed map's 22-room sprawl let it fight 3.0. The purse
at the end grows with it -- 53.7 gold unspent -- because the runs end sooner
rather than because the agent is richer.

### The shaping result does not survive the third map regime

Four more 600k runs, same protocol, on the ported `LevelGenerator` shape:
`runs/levelgen{,_b}{,_s1}.pt`, 60 eval runs on seeds 30,000+.

| checkpoint | A seed 0 | A seed 1 | B seed 0 | B seed 1 |
|---|---|---|---|---|
| 150k | 2.43 | 2.53 | 2.45 | 2.45 |
| 300k | 2.52 | 2.75 | 2.40 | 2.37 |
| 450k | 2.57 | 2.70 | 2.55 | 2.53 |
| 600k | 2.77 | 2.72 | 2.67 | 2.58 |

The heuristic is 2.50.

Past 150k, **A** means 2.67 over 2.52-2.77 and **B** means 2.52 over 2.37-2.67.
The unshaped reward is now both higher and no wider, which is the opposite of
what the last two regimes said. Across all three:

| map | A mean | B mean | B - A |
|---|---|---|---|
| fixed 22-room | 3.47 | 3.72 | +0.25 |
| blob of the right size | 2.66 | 2.82 | +0.16 |
| LevelGenerator shape | 2.67 | 2.52 | **-0.15** |

So the honest reading is that **the shaping advantage was a property of those
environments, not of the reward**: two regimes for it, one against, all at two
seeds a side and all differences inside the range a single seed moves. The
earlier section stands as a description of what happened on the oversized maps
and should not be read as a result about `phi` in general. Potential-based
shaping cannot change which policy is optimal, and on the map the game actually
generates it does not measurably change which policy is *found* either.

The other thing that shrank is the agent's margin. Against a 2.50 heuristic, A
is +0.17 and B is +0.02. On the 22-room map the same training budget bought +0.5
to +0.9 levels. Almost all of that margin was the freedom to farm a map three
times the size the level called for.

`tools/food_probe.py` at 600k shows the shops are not the constraint any more:
every agent buys more food than it eats (130 against 104, 96 against 65, 94
against 84, 81 against 57), leaves food unsold, and is priced out on 26 to 270
steps -- it is gold, and the fights themselves, that end these runs.

### Why the margin over the baseline is small now

`tools/ceiling_probe.py`, 30 seeds, against `runs/levelgen.600000.pt`:

| | level | gold earned | gold spent | peak Power |
|---|---|---|---|---|
| heuristic | 2.67 | 146.0 | 99.0 | 9,196 |
| agent | 2.97 | 162.6 | 127.2 | 7,682 |

Three things, and none of them is the agent being bad.

**The dice own more of the run than the policy does.** Best of twelve noisy
heuristic rollouts on the *same seed* reaches 3.23 where the mean rollout is
2.31: luck is worth +0.93 levels. The whole agent-vs-heuristic gap is +0.30 on
these seeds and +0.17 over 60. The two land on the same level on 18 of 30 seeds
and correlate at 0.56, so most of what a run scores was decided by the map it
drew and the fights it rolled.

**The economy has a ceiling and both hit it.** `tools/power_curve.py` on the
heuristic: readiness against the room it stands in goes 0.9, 2.4, 2.3, 1.7, 1.0
over levels 1 to 5, while the room budget grows 800 -> 11,200. A level is 10 or
11 rooms paying `GoldPerRoom` once, so a run earns about 150 gold in total and
the shelf it can afford stops keeping up around level 4. Ratio 1.0 is a coin
flip, and that is where both policies stop.

**They are not even buying the same thing.** The heuristic spends two thirds of
its gold on items at 127 Power per gold and 17% on shop upgrades; the agent
spends 29% on food, buys its items at 79 Power per gold, and never upgrades. So
the agent goes deeper with a *weaker* squad -- 7,682 peak Power against 9,196 --
by surviving more rooms rather than by winning harder fights. Two different
trades, worth about the same, which is what a small margin looks like when the
binding constraint is neither policy's to move.

The room left above the baseline is therefore mostly the room the *sim* has:
consumables, treasure rooms, quests and mutations past level 1 are all
unmodelled income, and each of them is gold or Power the agents cannot currently
spend a decision on.

## Treasure rooms in, consumables ruled out

Two findings and one new decision for the policy.

**Consumables are unreachable in Default play**, so the "unmodelled income" the
section above blamed for the small margin is not income at all: `C_Consumables.Add`
has one caller (`C_ConsumableShop.Buy`), that shop is built only in a `TalentShop`
room, and `StatShops` is 0 on all twelve level rows. Nothing to wire.

**Treasure rooms are the shrine family**, placed at random by
`ChooseTreasureRooms`: a `Shrine` on level 1, a `MutationShop` on every level
after it, and the `TalentShop` never. The mutation shop shows ten mutations and
hands over two for free (`m1`'s shipped config; the generic entry is null), and
it cannot be rerolled. So a run now collects about five mutations instead of
under one, and *which* two of ten it takes is a real decision -- the first one
the run layer has that is about the squad's composition rather than its purse.
`buy_mutation` is one action per slot, so the action space is **33** and the
observation **134**; the `runs/levelgen*` agents no longer load.

**Gold is a level total split by room type** (`C_Rooms.CalculatePower`), not a
flat payment per room: a level pays `(rooms - 1) x GoldPerRoom`, weighted
`ItemsMult` / `FoodMult` / `TreasureMult` / `BossMult` / `DefaultMult`, and the
start room pays nothing. At level 1 that is 10.5 gold for an item shop against
4.4 for a corridor.

| heuristic, 60 runs | flat gold, no mutation shop | level-total gold, shops wired |
|---|---|---|
| mean level | 2.50 | 2.32 |
| mean return | 14.61 | 11.24 |
| mutations per run | 0.6 | 2.1 |
| gold at the end | 53.7 | 50.6 |
| fights per run | 6.7 | 5.1 |

The baseline drops because it is the policy least suited to the new rule: it
walks to the boss and takes what is on the way, and the gold now sits in the
shops it is walking past. That is the first change in a while that should
*favour* a learned policy -- there is a route decision worth making again, and
two mutations a level to choose.

### Four runs with the shops wired, and the free mutations nobody takes

600k each, `--checkpoint-every 150000`, two seeds x unshaped (A) and
`--shaping power` (B), scored unshaped over 60 runs on seeds 30,000+.
`runs/treasure{,_b}{,_s1}.pt`.

| checkpoint | A seed 0 | A seed 1 | B seed 0 | B seed 1 |
|---|---|---|---|---|
| 150k | 2.17 | 2.05 | 2.05 | 2.28 |
| 300k | 2.57 | 2.43 | 2.18 | 2.42 |
| 450k | 2.48 | 2.67 | 2.37 | 2.52 |
| 600k | 2.37 | 2.60 | 2.83 | 2.50 |

The heuristic is 2.32. Past 150k, A means 2.52 (2.37-2.67) and B means 2.47
(2.18-2.83), so **shaping is now 2-2 across the four map regimes** and the
earlier "steadies the run" reading has nothing left to stand on.

The route decision I expected the shop-weighted gold to hand back to a learned
policy did show up, but small: the margin over the baseline is +0.20 unshaped
and +0.15 shaped, against +0.17 and +0.02 before the change.

**What is new is a mistake the agents make and the baseline does not.** A
mutation shop shows ten mutations and gives two away for nothing. The heuristic
takes them -- 2.1 a run. Every trained checkpoint spends 0 to 4% of its actions
on `buy_mutation`, falling to 0-1% by 600k, and ends runs holding **0.4 to 0.6**
mutations. They are leaving a free effect on the shelf, twice a level, at every
level after the first.

The reason is visible in the interfaces rather than in the policy. Taking a
mutation costs a step (`-0.02`) and pays nothing the network can see: the
observation carries `len(mutations) / 20` and nothing about what a mutation
*does*, and the effect itself only shows up inside a later fight, mixed in with
everything else. So the gradient says a free action is a small loss. The
heuristic takes them because it was told to.

That makes the next piece of work a representation question rather than a reward
one: a mutation's effect on this squad is computable -- `sim/mutations.py` can
apply one to the current specs and hand back the Power difference, which is
exactly what `best_item_target` already does for items -- and until the
observation carries it, the shelf is ten indistinguishable slots.

### The mutation shelf, described

The last section left the shelf as ten indistinguishable slots and blamed the
step cost for the agents walking past free mutations. Wiring the effect in says
the step cost was the smaller half of it.

**There is no Power delta to wire.** `UnitSpec.power` is the game's own strength
statistic and it is a *stored* number -- `Units.Power` plus the item's -- so a
mutation that raises the squad's Damage by 25% moves no Power at all. A Power
delta would have read zero for every mutation in the table.
`sim.mutations.effect_on` reports what a mutation actually changes instead: the
handler's kind, whether this sim implements it, the share of the squad it
touches, and the relative stat deltas where there are any.

**Three quarters of the shelf does nothing here.** Over the 1,094 offers across
twelve levels, 259 are implemented and 823 are `unimplemented` -- and of the
implemented ones, most are aimed at classes a squad may not field, in which case
they touch nobody. That is the fact the observation was missing, and it is a
better explanation of the agents' behaviour than the -0.02 step cost: a free
action whose expected effect is nothing most of the time is not obviously worth
a step, and the policy had no way to tell the exceptions apart.

Per slot the observation now carries: present, implemented, the fraction of the
squad touched, the relative Damage and Health deltas, and the largest of the
other stat deltas. Six floats a slot, ten slots, plus the takes left --
**observations 134 -> 184**, action space unchanged at 33. The heuristic picks
the implemented slot that touches most of the squad, the way it already picks
the item with the biggest Power gain, and moves 2.32 -> 2.35 by it.

Only `StatBonus` and `OraStatBonus` produce stat deltas at all: 32 offers of the
1,094. Everything else that is implemented is an agent-level passive whose value
this description cannot express, which is the next thing that would have to
change if the mutation choice turns out to matter.

### Describing the shelf changed nothing, and the paired test says why

Four more 600k runs with the described shelf (`runs/mutshelf{,_b}{,_s1}.pt`),
same protocol:

| checkpoint | A seed 0 | A seed 1 | B seed 0 | B seed 1 |
|---|---|---|---|---|
| 300k | 2.43 | 2.47 | 2.40 | 2.27 |
| 600k | 2.17 | 2.52 | 2.67 | 2.55 |

The heuristic is 2.35, and the agents sit where they sat before the observation
grew by fifty numbers. **`buy_mutation` is still 0 to 4% of their actions**,
falling with budget, and they end runs holding 0.1 to 1.6 mutations against the
heuristic's 2.1. Telling the policy what is on the shelf did not make it shop.

The paired test says the policy is right. Same 60 seeds, the heuristic against
itself with `buy_mutation` and `take_mutation` struck out of its legal actions:

| | mean level | mean return | mutations a run |
|---|---|---|---|
| takes them | 2.35 | 11.61 | 2.1 |
| ignores them | 2.28 | 10.85 | 0.0 |

**+0.07 levels, 95% CI [-0.09, +0.23]**, and the same level reached on 54 of the
60 seeds. Every free mutation a run can collect is worth, as far as this sim can
measure, nothing distinguishable from zero.

That is a statement about the sim, not about the game. 823 of the 1,094 offers
are `unimplemented` here; of the 259 that are implemented, only 32 are the
`StatBonus` rows that reach a unit's stats, and the rest are agent-level
passives whose effects are the long tail `notes/reference-sim.md` lists as not
modelled -- Vampirism, Evasion, CriticalStrike, the class-skill variants. The
observation now describes a shelf whose contents mostly do not exist yet.

So the honest next step for mutations is not another reward or another feature:
it is implementing the passives, and the measurement to repeat afterwards is
this paired test. Until it moves off zero, the mutation shop is scenery.


### The passives went in, and the paired test says the shelf was never the problem

The last section ended by saying the honest next step was implementing the
passives, and that the measurement to repeat afterwards was the paired test.
Both are done. `notes/reference-sim.md` has the mechanics; the numbers here are
what they were for.

**Coverage roughly doubled.** 344 of the 1,094 offers moved from
`unimplemented` to implemented -- BuffAttack alone is 76 of them, ClassDiversity
48 -- so the shelf goes from **271 of 1,094 (25%)** to **603 of 1,094 (55%)**.
Mutation definitions went 50/155 to 85/155.

**The paired test still cannot see much, and at 60 seeds it is the sample size
that says so.** `tools/mutation_value.py` is the test written down: the
heuristic plays the same run seeds twice, once with `buy_mutation` and
`take_mutation` struck out of its mask, everything else identical.

| 240 seeds | mean level | mean return | mutations a run |
|---|---|---|---|
| takes them | 2.37 | 11.85 | 2.3 |
| ignores them | 2.25 | 10.42 | 0.0 |

**+0.12 levels, 95% CI [+0.02, +0.21]**, and the same level reached on 189 of
the 240 seeds. So the shelf is worth something now rather than nothing -- but
the interesting part is the control, which the tool runs with
`--without-passives` (the same offers, every hook put back to `unimplemented`):

| 240 seeds, passives off | mean level | mean return | mutations a run |
|---|---|---|---|
| takes them | 2.35 | 11.68 | 2.2 |
| ignores them | 2.25 | 10.42 | 0.0 |

**+0.10 levels, 95% CI [+0.01, +0.18].** Each estimate sits comfortably inside
the other's interval. **Implementing the passives did not measurably change what
the mutation shop is worth.** What the earlier "worth nothing, +0.07, CI
[-0.09, +0.23]" reading actually reflected was 60 seeds, not 823 unimplemented
offers: the effect was always about a tenth of a level, and a tenth of a level
needs a few hundred paired runs to separate from zero.

That is worth being blunt about, because it retires a hypothesis rather than
confirming one. Two things follow.

**The agents' behaviour is still not obviously wrong.** A free mutation is worth
roughly +0.05 levels each at the heuristic's rate of 2.3 a run. Against a step
cost of -0.02 and a level worth 10, that is positive but small, and it is
delivered several fights later mixed in with everything else. A policy spending
0 to 4% of its actions on `buy_mutation` is leaving something on the table, but
not much of one, and the gradient it would have to follow is genuinely faint.

**The measurement, not the mechanism, is what was limiting.** Every earlier
mutation conclusion in this file was drawn at 60 seeds. `tools/mutation_value.py`
defaults to 60 for speed but the number that settled this was 240; anything
claiming a mutation effect below about a fifth of a level should be run there.

One practical note: a fight carrying an agent-level passive leaves the Rust
core's envelope and falls back to the Python oracle, which is why the
`takes them` half of the test runs ~30x slower than `ignores them`. Porting the
hooks to the core is what would make a several-thousand-seed version cheap.

### Correction: at 2,000 seeds the passives did move the shelf

The section above concluded, from 240 paired seeds, that implementing the
passives "did not measurably change what the mutation shop is worth" -- +0.12
with them against +0.10 without, each inside the other's interval. Porting the
hooks to the Rust core (`notes/rust-core.md`) cut the cost of that test by
roughly nineteen times, from 190 s for 240 runs to 10 s, and at a sample size
that could not be afforded before the answer is different.

`tools/mutation_value.py --runs 2000 --compare` scores the same 2,000 seeds
twice over, once with the hooks live and once with them inert, and reports the
difference of the two paired differences -- which is the number with a CI on it,
rather than two intervals eyeballed against each other:

| | mean level, takes them | mean level, ignores them | paired difference |
|---|---|---|---|
| passives live | 2.36 | 2.23 | **+0.13, CI [+0.10, +0.17]** |
| passives inert | 2.29 | 2.23 | **+0.07, CI [+0.04, +0.10]** |

**What the passives added: +0.06 levels a run, 95% CI [+0.05, +0.08].** The
mutation shop was worth about +0.07 levels with three quarters of the shelf
doing nothing, and is worth about +0.13 now -- roughly doubled, and both
estimates are tight enough that they no longer overlap.

Two oracle fixes the port surfaced are folded into that number and are worth
separating out, because they are not "the passives" so much as the passives
finally being exercised: **panic was inert** (NC-Fear is a tree of its own and
is not inside NC-BaseUnit, so a panicked unit went on fighting), and
**FearsomeAttack had no chance roll** (its model names the property `percent`,
and the shared gate reads `chance`). Both are described in `notes/rust-core.md`.

So the earlier reading stands corrected in one direction and confirmed in the
other. The 60-seed "worth nothing" result *was* a sample-size artefact -- the
shelf was always worth something. But the passives did not leave that number
where they found it: they doubled it, and it took 2,000 seeds to see, which is
what the port paid for.

The agents' behaviour still looks defensible rather than wrong. At 2.3 free
mutations a run, +0.13 levels is about +0.06 a mutation, against a step cost of
-0.02 and a level worth 10 -- positive, small, and delivered several fights
later. Whether a policy can learn to shop for that is now a cheap experiment
rather than an expensive one.


## Retraining on the fast core, with the passives live

The `mutshelf` agents were trained before `sim/mutations.py` implemented the
four `C_PassiveSkill` hooks, and before the two oracle fixes that came out of
porting them (panic was inert, FearsomeAttack had no chance roll). Porting the
hooks to the Rust core closed the envelope hole they opened, so the whole
environment is back inside the core and a 600k run is cheap again. Four of them,
same protocol as every regime before: 16 envs, horizon 128, `frontline`,
`--fast-core`, `--checkpoint-every 150000`, seeds 0 and 1 crossed with unshaped
and `--shaping power`, saved as `runs/passives{,_b}{,_s1}.pt`.

Throughput, four runs in parallel on 16 cores:

| | steps/s per run | 600k steps |
|---|---|---|
| Python oracle (measured earlier, one run) | 190 | ~53 min |
| Rust core, four concurrent | 1,540 to 1,560 | 6.5 min |

All four finished in the same 6.5 minutes of wall clock. Unlike `ppo_long`,
throughput did not decay across the run (1,536 at update 5 against 1,560 at
600k), because these agents do not get deep enough for fight cost to dominate.

Mean level over 60 runs on seeds 30,000+, `tools/run_autopsy.py`:

| checkpoint | A seed 0 | B seed 0 shaped | C seed 1 | D seed 1 shaped |
|---|---|---|---|---|
| 150k | 1.85 | 1.57 | 2.13 | 1.22 |
| 300k | 2.13 | 2.13 | 2.28 | 2.02 |
| 450k | 2.47 | 2.20 | 1.90 | 2.15 |
| 600k | 2.38 | 2.45 | 2.52 | 2.30 |

Heuristic on the same seeds: 2.32. Past 150k the unshaped mean is 2.28 and the
shaped one 2.21, so **shaping is 2 to 3 across five regimes now**, which is the
same non-result the treasure runs gave.

### Retraining did not pay, and the paired test is how that is known

The interesting comparison is not against the heuristic, it is against the
`mutshelf` agents, which still load (184 x 33 is unchanged) and can therefore be
scored on the environment they were not trained on. 240 paired seeds, identical
placement, greedy:

| family | new | old | new minus old, per seed | same level |
|---|---|---|---|---|
| A seed 0 unshaped | 2.47 | 2.31 | **+0.16 [+0.00, +0.31]** | 116/240 |
| B seed 0 shaped | 2.50 | 2.67 | **-0.17 [-0.30, -0.03]** | 144/240 |
| C seed 1 unshaped | 2.51 | 2.66 | **-0.15 [-0.26, -0.04]** | 160/240 |
| D seed 1 shaped | 2.42 | 2.61 | **-0.19 [-0.32, -0.06]** | 144/240 |

Pooled, and the four families are not independent: -0.09 [-0.15, -0.02].

Against the heuristic on the same 240 seeds, the new agents sit at +0.09 to
+0.18 and the old ones at -0.02 to +0.34, both straddling it.

So training on the corrected environment produced agents that are, if anything,
slightly worse than the ones trained on the environment without the passives.
That is not evidence the passives hurt. It is what it looks like when the change
is smaller than the noise it is being read against: the mutation passives are
worth about +0.06 levels a run to a policy that collects mutations, PPO run to
run spread at this budget is about half a level (600k checkpoints here span 2.30
to 2.52, and `mutshelf` spanned 2.17 to 2.67), and four runs cannot separate the
first number from the second. The one honest conclusion is that **the agents did
not need retraining for this change**, which is worth knowing before paying for
it after the next one.

Two practical notes fell out of it. Sample size again: `mutshelf.pt` scores 2.23
over 60 seeds and 2.31 over 240, and `passives.pt` 2.38 over 60 and 2.47 over
240, so 60 seeds moves a tenth of a level on its own. And the 150k column is
much noisier than the rest (1.22 to 2.13), so a budget comparison that starts
there reads mostly as init noise.

The agents still do not shop for mutations: `buy_mutation` is 1 to 4% of their
actions at 600k and they end runs holding 0.5 to 0.8 against the heuristic's
2.1, unchanged by the passives being live. Given +0.13 levels for the whole
shelf, that remains defensible rather than wrong.

### Twelve seeds an arm, and the retraining question closes

Four runs could not separate a +0.06 environment change from PPO run to run
spread, so the same question was asked at a sample size that can. Two arms of
twelve 600k runs, unshaped, differing only in whether the mutation passives were
live *during training*: `runs/arm_live_s0..s11.pt` against
`runs/arm_inert_s0..s11.pt`, where the inert arm trains with every hook put back
to `unimplemented` (the patch `tools/mutation_value.py --without-passives`
applies, held for the whole run). Init seed is shared across the arms, so the
twelve differences are paired. **Every agent is then scored on the live
environment**, 240 runs on seeds 30,000+: the arms differ in what they were
trained against, never in what they are graded on.

24 runs of 600k, eight concurrent, **18 minutes of wall clock** for 14.4M
environment steps, about 1,200 steps/s per run and 13,300 aggregate. On the
Python oracle the same sweep is a little over 21 hours.

| init seed | live | inert | live minus inert |
|---|---|---|---|
| 0 | 2.47 | 2.57 | -0.10 |
| 1 | 2.51 | 2.38 | +0.13 |
| 2 | 2.44 | 2.32 | +0.12 |
| 3 | 2.64 | 2.47 | +0.17 |
| 4 | 2.15 | 2.28 | -0.13 |
| 5 | 2.23 | 2.38 | -0.14 |
| 6 | 2.14 | 2.47 | -0.33 |
| 7 | 2.47 | 2.54 | -0.07 |
| 8 | 2.28 | 2.40 | -0.11 |
| 9 | 2.53 | 2.49 | +0.04 |
| 10 | 2.52 | 2.35 | +0.17 |
| 11 | 2.29 | 2.25 | +0.03 |

    live arm   2.389        inert arm  2.407        heuristic  2.325
    paired difference, per training seed   -0.02 [-0.10, +0.07]
    per eval seed (n=2,880, not independent) -0.02 [-0.06, +0.02]

**Training against the corrected environment is worth nothing measurable, and
the interval now rules out more than +0.07 levels.** The mutation passives
change what the shelf is worth (+0.06 levels to a policy that collects
mutations, from the 2,000-seed paired test) but they do not change what a policy
should learn, which is consistent with agents that spend 1 to 4% of their
actions on `buy_mutation` either way.

The number this sweep really produces is the one the earlier sections were
missing: **the standard deviation across 24 identically configured agents is
0.132 levels**, spanning 2.14 to 2.64. Two runs per cell therefore carry a
standard error of about 0.09, which is the whole of the -0.09 "retraining is
worse" reading in the section above. That reading is retired: it was noise, and
now it has a number saying how much noise.

Two things follow for anything measured this way later. A comparison of two
training configurations needs about a dozen runs an arm to see a tenth of a
level, not two. And the best single agent here (2.64, `arm_live_s3.pt`) is
+0.32 over the heuristic while the arm it came from is +0.06: picking the top of
24 is worth a quarter of a level of pure selection, so a headline agent has to
be quoted with the arm it was drawn from.


### Vampirism and evasion landed, and the baseline moved under the agents

`notes/reference-sim.md` has the mechanics and `notes/rust-core.md` the ABI
change. What matters here is that the environment moved without the observation
or the action space moving, so **every saved agent still loads and every number
measured before this is a tenth of a level out**.

The heuristic over 240 seeds went **2.325 to 2.283**. The direction is the
interesting part: implementing two passives the *squad* can buy made the runs
slightly worse, because 111 enemy classes carry them too and evasion is worth
more to a pack that outnumbers you. Two agents scored across the same change
barely moved (`arm_live_s3` 2.64 to 2.65, `mutshelf_b` 2.67 to 2.67), so the
agents lost nothing; the baseline they are measured against is what fell.

The mutation shelf, `tools/mutation_value.py --runs 2000 --compare`:

| | mean level, takes them | ignores them | paired difference |
|---|---|---|---|
| passives live | 2.36 | 2.24 | **+0.13, CI [+0.09, +0.16]** |
| passives inert | 2.29 | 2.24 | **+0.05, CI [+0.04, +0.07]** |

**What the agent-level passives are worth: +0.08 levels a run, CI [+0.06,
+0.09]**, against +0.06 measured before. Half of that move is the control
widening rather than the new code: "passives inert" now also strips evasion,
vampirism, crit, fury swipe and the projectile shapes, which resolve through the
unit-skill registry and were never in the old list. The shelf's own worth is
unchanged at +0.13, which is the honest headline: 84 of the 1,094 offers are
these two mechanics, and they did not move what a mutation is worth.

They do change what a *fight* looks like, which is where the effect goes. The
agents' `buy_mutation` share is still 1 to 4%, so nothing here argues for
retraining -- and the twelve-seed sweep above says a change this size is below
what a 600k run resolves anyway.

#### The two follow-up passives changed the fights and not the runs

`ExplodeProjectile` and `EventualKnockback` were the same reader bug and are
fixed (`notes/reference-sim.md`). For the RL side the result is a one-liner:
**nothing moved.** The heuristic is 2.283 over the same 240 seeds, the shelf is
+0.13 levels at 2,000 paired seeds and the agent-level passives account for
+0.08 [+0.06, +0.10], all within noise of the numbers before the fix.

That is what 31 of 1,094 offers buys, on classes a run rarely fields, in a run
that mostly ends on level 2 or 3. The fights those rows appear in are different
now -- an explosion is flat magical damage in a radius rather than a share of
one hit, and a knockback throws the victim's whole team when it has a radius --
but a run reaches too few of them for that to show. No agent needs retraining
for this; the twelve-seed sweep already said a change this size is invisible at
600k steps.

### The sweep re-run on the fixed sim, with a control that was not a control

The twelve-seed sweep was repeated on the environment carrying the four fixed
passives (vampirism, evasion, `ExplodeProjectile`, `EventualKnockback`). The
reason to repeat it was not the fixes. It was that **the inert arm had never
been inert.**

`disable_passives` rewrote `MUTATION_REGISTRY[name]` only where the name was
already a key, and `mutations.handler_for` reads that dict *before* it falls
back to the unit-skill registry. The seven passives that live only in the skill
registry (`Evasion`, `StatusEvasion`, `Vampirism`, `CriticalStrike`,
`FurySwipe`, `ExplodeProjectile`, `EventualKnockback`) were therefore skipped by
the guard and stayed live in the arm that was supposed to be the control. The
first sweep's two arms differed in six hooks, not thirteen, and four of the
seven it missed are the ones this section exists to test.

The fix inserts rather than rewrites, so a skill-registry name is shadowed:
`sim.mutations.disable_agent_passives`, now the single source of truth for both
`tools/mutation_value.py` and `rl/train.py --without-passives`. Class skills are
untouched either way, because units resolve theirs through
`unit_skills.handler_for`, which never reads `MUTATION_REGISTRY`. So this
strips what a *mutation* grants and nothing else, which is what the control
always claimed to be.

Same protocol otherwise: 24 runs of 600k, unshaped, init seed shared across the
arms, every agent scored on the live environment over 240 seeds from 30,000.
`runs/arm2_{live,inert}_s0..s11.pt`, kept apart from the first sweep's
`runs/arm_*.pt` rather than overwriting them.

| init seed | live | inert | live minus inert |
|---|---|---|---|
| 0 | 2.60 | 2.36 | +0.23 |
| 1 | 2.27 | 2.47 | -0.20 |
| 2 | 2.48 | 1.81 | +0.68 |
| 3 | 2.66 | 2.58 | +0.08 |
| 4 | 2.69 | 2.63 | +0.06 |
| 5 | 2.50 | 2.52 | -0.02 |
| 6 | 2.48 | 2.50 | -0.02 |
| 7 | 2.48 | 2.70 | -0.22 |
| 8 | 2.61 | 2.46 | +0.15 |
| 9 | 2.55 | 2.33 | +0.23 |
| 10 | 2.67 | 2.60 | +0.07 |
| 11 | 2.52 | 2.55 | -0.03 |

    live arm   2.543        inert arm  2.457        heuristic  2.283
    paired difference, per training seed   +0.09 [-0.05, +0.22]
    per eval seed (n=2,880, not independent) +0.09 [+0.04, +0.13]

**The answer does not change: still nothing measurable at twelve seeds.** The
sign flipped against the first sweep's -0.02 and the point estimate grew, but
the paired interval crosses zero and the supporting statistics are weak in the
same direction. Seven of the twelve differences are positive, a median of
+0.07, and the mean is carried by one agent: `arm2_inert_s2` scores 1.81 where
nothing else in either arm is below 2.27, and dropping that seed takes the
result from +0.09 [-0.05, +0.22] to **+0.03 [-0.06, +0.12]**. An effect that
halves when one of twelve runs is removed is a run, not an effect.

What did change is the spread. The standard deviation across 24 identically
configured agents is **0.183 levels** (1.81 to 2.70) against 0.132 before, and
it is not shared: the live arm sits at 0.115 and the inert arm at 0.230. A real
control is a wider control, which is the honest cost of the fix. Twelve runs an
arm resolved a tenth of a level under the old number and does not quite under
this one, so the "about a dozen runs an arm" rule of thumb should be read as a
floor.

Both arms also clear the heuristic by more than the first sweep's arms did
(+0.26 and +0.17 against +0.06 and +0.08), on a baseline that is unchanged at
2.283. That is the sim fixes moving the agents rather than the arms separating.

One operational note, since the first sweep's cost is quoted elsewhere: 24 runs
of 600k did **not** come in at 18 minutes this time. Each run held about 450
steps per second with eight concurrent, roughly 23 minutes apiece, so two full
batches is about three quarters of an hour. The machine was serving an unrelated
MCP process that was spawning python subprocesses in bursts throughout, which is
the most likely place the throughput went. Treat the 18 minute figure as a
best case on an idle machine.

#### What this invalidates

Two numbers were measured through the broken control and are not what they say:
the "passives inert" row of the mutation shelf table (2.29 mean level, +0.05
paired) and **the agent-level passives are worth +0.08 levels a run, CI [+0.06,
+0.09]**. Both compared the live shelf against a control that still had
evasion, vampirism, explode and knockback running, so both understate the
passives. The shelf's own worth (+0.13) has the passives live on both sides and
is unaffected. `tools/mutation_value.py --runs 2000 --compare` was re-run
against the fixed control, below.


### The whole worth of a mutation is the agent-level passives

`tools/mutation_value.py --runs 2000 --compare`, on the fixed control. The
`--compare` path now also prints the two mean levels of the inert half, which
the paired differences alone never carried.

| | mean level, takes them | ignores them | paired difference |
|---|---|---|---|
| passives live | 2.36 | 2.24 | **+0.13, CI [+0.09, +0.16]** |
| passives inert | 2.24 | 2.24 | **+0.00** |

    shelf worth with the passives : +0.13 levels
    shelf worth without them      : +0.00 levels
    what the passives added       : +0.13, 95% CI [+0.11, +0.15]

**With a control that is actually a control, a mutation shelf with no
agent-level passives is worth exactly nothing**, and the agent-level passives
account for the entire +0.13 rather than +0.08 of it. The old reading had the
shelf worth +0.05 on its own because the control it was measured against still
ran evasion, vampirism, explode and knockback: a chunk of the passives' value
was sitting in the baseline, which both shrank the measured contribution and
left a residue that looked like the shelf being worth something without them.

That retires the line this file carried before, that 84 of the 1,094 offers are
these two mechanics and "they did not move what a mutation is worth." They are
what a mutation is worth. Of 1,094 offers only 259 are implemented and 32
produce stat deltas, and those 32 `StatBonus` rows now measure at zero: the
squad's stat mutations are not what carries a run, the handful of rows that
attach a passive to a unit are.

One consistency check worth recording: "ignores them" is 2.24 in both halves, to
the same two decimals over 2,000 paired seeds. A run that takes no mutation is
unaffected by whether the mutation hooks are live, which is what the control is
supposed to guarantee and is the evidence that it now touches only what a
mutation grants. It also says the enemy packs are not drawing from this shelf,
or the two halves would separate there too.

The RL side is unmoved by this. The agents spend 1 to 4% of their actions on
`buy_mutation` and end runs holding 0.5 to 0.8 against the heuristic's 2.1, so a
shelf worth +0.13 rather than +0.13-with-different-attribution changes none of
their incentives. It does raise the ceiling on what a mutation-shopping policy
could be worth, which is the open question the `arm2` sweep could not resolve at
twelve seeds.


### Describing the passives changed the architecture, not the policy

The shelf measurement above says the agent-level passives are the whole +0.13
and the stat deltas are worth nothing. The observation had that backwards: it
carried three stat-delta features per slot, which fire only for `StatBonus` and
`OraStatBonus`, and described a passive-attaching offer only as `implemented`
plus the share of the squad it touches. So `effect_on` gained a `passive` flag
(`sim.mutations.PASSIVE_ATTACHING`, the same union `disable_agent_passives`
strips) and `rl/env.py` gained a bit per slot, `per_m` 6 to 7. Observations
184 to 194, actions unchanged at 33, so every earlier agent including the whole
`arm2` sweep is stale.

The flag is not a restatement of `implemented`. Of the 1,094 offers, 507 attach
a passive, 491 are unimplemented or noop, and **96 are implemented and attach
nothing** (44 stat bonus, 52 other). Those 96 are exactly the ones the shelf
test measures at zero, and `implemented` alone cannot separate them.

Twelve seeds, 600k, unshaped, scored over the same 240 seeds from 30,000:

    mutpassive arm  2.630  (sd 0.089)
    arm2_live arm   2.543  (sd 0.115)
    heuristic       2.283

    unpaired difference +0.087, SE 0.042, Welch df 20.7
    95% CI [+0.000, +0.175], t = 2.08, 72% of the 144 cross-arm pairs favour it

Unpaired is the right test and paired is not available: the observation gained
ten dimensions, so the first layer is wider and one `--seed` no longer names the
same init draw on both sides. A tenth of a level with a lower bound sitting on
zero would be a weak result on its own. It is worse than weak, because the
mechanism is absent.

**The agents still do not buy mutations.** Across all twelve, `buy_mutation` is
0 to 3% of actions against the heuristic's 7%, and they end runs holding 0.42
mutations on average (0.1 to 1.6) against the heuristic's 2.2. A feature that
only describes mutation shop slots cannot be earning a tenth of a level through
slots that are never taken.

The ablation settles it. `sim.mutations.PASSIVE_ATTACHING` is read in exactly
one place, `effect_on`, so emptying it at evaluation time forces the new bit to
zero and changes nothing else. The same twelve agents, same 240 seeds:

    arm with the bit live   2.630
    arm with it zeroed      2.631

**+0.001.** No seed moves by more than 0.025, and those are a handful of runs
diverging rather than a policy losing an input it was using. The policies do not
read the feature at all, so the +0.087 is not information. It is ten more input
dimensions changing the parameter count, the init draw and the optimization
path.

The lesson generalises past this feature, and is the reason to keep this section
rather than revert the change. **Every observation change in this project
confounds architecture with information**, because the first layer is sized from
`obs_dim`. Six of the retrains recorded above are quoted as "the observation
moved and the agents did or did not improve", and none of them ran this
ablation, which costs one evaluation pass and no training. Any future
"describing X helped" claim needs it before it is believed.

The bit stays in. It costs nothing, it is the honest description of where the
shelf's value is, and a policy that ever learns to shop for mutations will need
it. What it does not do is make one, which remains where the mutation work has
been stuck since the shelf was first described: the agents skip a +0.13 resource
because taking it costs a step now and pays later, and no amount of describing
the shelf has changed that. The next thing to try is the step cost or the
horizon, not another feature.


### The step cost is not what deters the mutation shop

`step_cost` is subtracted once per action at `rl/env.py:408`, uniformly, so it
taxes `buy_mutation` and `take_mutation` exactly as it taxes a move. That makes
"taking one costs a step now and pays later" testable through the existing
`--step-cost` flag. Four seeds each at 0.005 and at 0.0, 600k, unshaped, against
the twelve-seed 0.02 arm, all scored over the same 240 seeds from 30,000:

| step cost | mean level | mutations held | `buy_mutation` | `reroll` |
|---|---|---|---|---|
| 0.02 (n=12) | 2.630 | 0.42 | 0.7% | 3.2% |
| 0.005 (n=4) | 2.647 | 0.55 | 1.2% | 10.8% |
| 0.0 (n=4) | 2.562 | 0.28 | 0.5% | 15.0% |
| heuristic | 2.283 | 2.20 | 7.0% | 0.0% |

**Making a step free does not make them shop.** Mutations held go 0.42, 0.55,
0.28 as the tax falls to zero, which is not a trend and puts the extreme value
below the baseline. `buy_mutation` does the same thing, 0.7% to 1.2% to 0.5%,
against the heuristic's 7%. Four seeds is thin for a tenth of a level but this
is not a level measurement: the action share is averaged over thousands of
actions across 240 runs an agent, and it is flat.

The one thing that does respond is monotone and large. **`reroll` goes 3.2% to
10.8% to 15.0%**, reaching 36% and 24% on two of the free-step agents, and mean
steps a run roughly doubles on those two, to 59 and 57. Given actions that cost
nothing the agents spend them spinning the item shop, not visiting the mutation
shelf. Mean level does not reward it: 2.562 at zero cost is the worst of the
three arms, so the extra rerolling is not even paying for itself.

That is a straight answer to the question that was asked and a redirection of
the one behind it. The deterrent is not the price of the action. Offered free
actions the policy has a clear preference and mutations are not it, which is
consistent with a policy that has correctly learned items dominate a shelf worth
+0.13 over a run that mostly ends on level 2 or 3.

Two things this does not settle. The global tax is not the same experiment as a
per-action one: `buy_mutation` and `take_mutation` could be exempted
specifically, which isolates the cost of the mutation step from the cost of
every other step and is the only version of the original hypothesis still
standing. And the discount is untouched here; a shelf that pays across the rest
of a run is worth less at gamma 0.99 over 40 steps than the arithmetic suggests,
so the horizon remains the other candidate. Neither is worth a budget until
something explains why 15% of actions go to rerolls that do not pay.


### The rerolls: the step cost was the only brake on the item shop

The zero-step-cost agents put 15 to 36% of their actions into `reroll`, and the
arm scored worst of the three. The mechanic is not at fault. `reroll` charges
`RollCost` and replaces the whole shelf (`C_ItemShop.Roll` with both defaults),
`RollCost` is a flat 2 at every shop level while `Quantity` grows 3 to 7, and a
direct test rolls six times for 12 gold and gets six different shelves. The
shelf is in the observation too, quality over 5 and affordability per slot, so
the policy can see what it rolled.

What it does with that is the problem. Per run, over 120 seeds:

| | rerolls | items bought | gold into rolls | gold into items | rolls followed by a purchase |
|---|---|---|---|---|---|
| `sc000_s0` (cost 0.0) | 22.2 | 4.9 | 44.4 | 44.7 | 0.2 |
| `sc000_s1` (cost 0.0) | 15.5 | 6.2 | 31.0 | 56.9 | 0.5 |
| `mutpassive_s0` (0.02) | 3.8 | 9.6 | 7.5 | 88.0 | 2.7 |
| `mutpassive_s4` (0.02) | 5.4 | 4.6 | 10.8 | 42.8 | 0.0 |

The free-step agent burns **half its gold on rolling** and buys half as many
items. Its longest unbroken roll streak, 22.0, is its entire per-run reroll
count: the rolling happens in one sitting, not spread across shops.

What ends a streak says the rest. Over the same 120 runs:

    sc000_s0        79% of streaks end because it cannot afford another roll
                    21% end in a purchase
                    median gold when the streak ends: 1.2, against a roll cost of 2

    mutpassive_s0   84% of streaks end in a purchase
                    16% end broke
                    median gold when the streak ends: 32.7

**The free-step agent rolls until the action mask takes the option away.** It is
not searching for something and stopping when it finds it; it stops at a gold
balance below the price of one more roll. The best quality it saw during a
streak averages 1.82 of 5 against the taxed agent's 1.42, so it rolls past
better shelves than the disciplined agent ever sees, and buys nothing 79% of
the time.

This is an ordinary failure mode rather than a discovery about the game: an
action whose only cost is a small delayed gold loss, with nothing immediate
against it, and a value function that does not attribute the loss. The step cost
is what makes a shop visit terminate. **It is load-bearing, not a mild nudge**,
and that is the real reason the zero-cost arm scored 2.562: not that free steps
let the agent wander, but that they removed the brake on an unbounded action and
the gold that would have bought items went into rolling.

Two consequences. The step cost stays at 0.02 and the earlier reading of that
sweep should be stated as "removing the tax breaks shop discipline", not "free
steps do not help". And it settles the form of the mutation experiment: the
global tax cannot be lowered to make mutations cheaper without also unbracing
the item shop, so the only clean version is **exempting `buy_mutation` and
`take_mutation` specifically** and leaving every other action taxed. That is one
flag in `rl/env.py`, and it is now the obvious next thing to run.


### Free mutation steps changed nothing, and the reason is an unobservable shrine

`--free-mutation-steps` exempts `buy_mutation_*` and `take_mutation` from the
per-step tax and leaves every other action at 0.02 (`rl/env.py`,
`_is_mutation_action`). Verified on a driven run: `take_mutation` goes from
-0.020 to +0.000 while `move` stays at -0.020. Observations and actions are
unchanged at 194 x 33, so `mutpassive` is the matched baseline. Eight seeds,
600k, 240 eval seeds:

| | mutations held | `buy_mutation` | `reroll` | mean level |
|---|---|---|---|---|
| `freemut` (n=8) | 0.28 | 0.6% | 0 to 4% | 2.599 |
| `mutpassive`, cost 0.02 (n=12) | 0.42 | 0.7% | 3.2% | 2.630 |
| heuristic | 2.20 | 7.0% | 0.0% | 2.283 |

**Nothing moved, and the control held**: `reroll` stayed low, so the exemption
isolated the mutation step without unbracing the item shop the way the global
cut did. Since a mutation shop also gives two away for nothing, a free mutation
now costs zero gold and zero step penalty, and the agents still decline it. The
step cost is not the deterrent in any form and that hypothesis is closed.

Availability is not the constraint either. Over 120 runs a mutation action is
legal on 4.4 to 5.2% of steps and in 85 to 88 of the 120 runs, against 78 for
the heuristic. What differs is uptake, and it splits by *which* action:

    heuristic       buy/take offered 280, taken 280 (100%)   free take 62/62 (100%)
    mutpassive_s0   offered 352, taken 212 (60%)             free take 1/82 (1%)
    freemut_s5      offered 300, taken  59 (20%)             free take 0/74 (0%)
    freemut_s0      offered 362, taken   6 ( 2%)             free take 0/131 (0%)

The agents do use `buy_mutation`, up to 60% of the chances they get. What they
never use is `take_mutation`, the free one at a shrine, 0 to 1% against the
heuristic's 100%. Both append to the same list and both are free, so the
economics cannot separate them.

**The observation can.** The shelf block is gated on
`kind == "mutation_shop"`, so a `mutation` room fills none of it, and the check
that matters is this: a shrine holding a mutation and a shrine holding nothing
produce an **identical 194-dim observation**. Only the action mask differs. The
room kind is one-hot encoded, so the policy knows it is standing on a shrine; it
has no feature anywhere saying whether there is anything to take. The logit for
`take_mutation` is therefore computed from evidence that is the same either way,
and the association can never be learned. The mutation shop is fully described,
which is exactly where uptake reaches 60%.

That is the whole asymmetry, and it is the same failure as the passive flag one
section up: a resource the policy is asked to choose is described along axes
that do not carry the decision. It also means every "the agents refuse free
mutations" line in this file, going back to the first mutation-shelf section,
was reading a representation bug as a preference.

The fix is one number in `_encode`: the shrine's `stock` where the shop's
`takes_left` already goes. That is a one-line change and an observation move,
so it costs another twelve-seed arm to measure, and on this file's record an
observation change also has to be run past the zeroing ablation before any of
its gain is believed.


### The shrine is observable now, and the agents still will not touch it

The gap was real and it is closed. `_encode` writes the shrine's stock where the
shop's `takes_left` already went (`rl/env.py`, the `elif room_now.kind ==
"mutation"` branch), so a stocked shrine and a spent one now differ in exactly
one observation entry, index 134, where before they were byte-identical.
Observations 194 to 195, actions 33, so the `mutpassive` arm no longer loads.
Twelve seeds, 600k, unshaped, 240 eval seeds from 30,000:

    shrine arm      2.561  (sd 0.078)
    mutpassive arm  2.630  (sd 0.089)
    heuristic       2.283
    unpaired difference -0.069, SE 0.034, 95% CI [-0.139, +0.002]

**`take_mutation` uptake is still 0%.** Zero of 67, 68, 72 and 75 chances on the
four agents sampled, exactly as before the fix. So the prediction in the section
above was wrong: the observation could not distinguish a stocked shrine from a
spent one, that was worth fixing on its own terms, and fixing it changed nothing.

It is not an argmax artifact, which was the obvious way the result could have
been a measurement problem rather than a behaviour. Reading the distribution
instead of the greedy pick, when `take_mutation` is legal:

| | prob on `take_mutation` | its rank among ~4.7 legal | best `buy_mutation` slot |
|---|---|---|---|
| `shrine_s0` | 0.0019 | 3.8, never better than 3 | 0.417 |
| `shrine_s5` | 0.1234 | 2.5, never better than 2 | 0.257 |
| `shrine_s8` | 0.0003 | 3.8 | 0.211 |
| `shrine_s11` | 0.0010 | 3.9 | 0.163 |

Three of the four put a thousandth of their probability mass on it. One,
`shrine_s5`, reaches 0.12 and second place and still never wins a state. The
policies have an opinion and the opinion is no.

The ablation says the same thing from the other side. Zeroing index 134 at
evaluation time moves the arm from 2.562 to **2.563**, +0.001, which is the
third feature in this file to survive its own removal. The -0.069 against
`mutpassive` is therefore not the feature costing anything either; it is the
same architecture-versus-information confound as before, now running in the
unlucky direction.

**What this closes.** Across four experiments the agents have declined mutations
under every explanation offered for it: the shelf was undescribed (described it,
nothing), the passives were the value and invisible (flagged them, nothing), the
step cost was the deterrent (made it free, nothing), and the shrine was
unobservable (made it observable, nothing). The remaining explanation is the one
that needs no mechanism: **at 600k steps, on runs that end on level 2 or 3, a
random mutation is worth less than what else the step buys**, and the agents are
right. The whole shelf is +0.13 levels to a heuristic that takes every one of
them, and these agents beat that heuristic by 0.28 to 0.35 while holding almost
none.

The shrine bit stays in for the same reason the passive flag did: it costs
nothing and the observation is now honest about what is in the room. Neither is
load-bearing. If mutations are worth another attempt it should not be another
feature, and not the reward either. It should be a budget: every result here is
at 600k, where the standard deviation across identical agents is about a tenth
of a level and a +0.13 resource is under the noise floor by construction.


### 2M steps: the agents were never refusing mutations, they were undertrained

Twelve seeds at 2M, unshaped, `--checkpoint-every 600000`, so 600k / 1.2M / 1.8M
/ 2M come off **one run per seed** and the budget comparison carries no init,
no architecture and no sampling difference. That is the confound that has
muddied every comparison in the sections above, and here it is absent by
construction. `runs/shrine2m_s0..s11.pt` plus their checkpoints, 240 eval seeds:

| budget | mean level | sd | mutations held | `take_mutation` | any mutation |
|---|---|---|---|---|---|
| 600k | 2.561 | 0.078 | 0.34 | 0.2% | 11.8% |
| 1.2M | 2.810 | 0.095 | 1.30 | 22.9% | 42.0% |
| 1.8M | 2.886 | 0.110 | 1.97 | 28.4% | 64.7% |
| 2M | 2.932 | 0.084 | 2.14 | 29.2% | 68.9% |
| heuristic | 2.283 | | 2.20 | 100% | 100% |

    2M minus 600k, paired within seed: +0.370 [+0.310, +0.431]
    positive on 12 of 12, per seed +0.26 to +0.55

**Everything this file has said about the agents refusing mutations was a
statement about 600k steps.** Mutation holdings go 0.34 to 2.14 across the
budget and land essentially on the heuristic's 2.20. `take_mutation`, the free
shrine mutation that four separate interventions could not move off 0%, is
taken 29% of the time at 2M. The behaviour was not absent, it had not been
learned yet.

The 600k row here is 2.561, which is the `shrine` arm's 2.561 to three decimals
off twelve independently trained runs. The curve is anchored to the same number
the separate arm produced, so the checkpoint at 600k is not a different animal
from a run that stopped there.

What this retracts. "The agents skip a +0.13 resource because taking it costs a
step now and pays later" was a story about a behaviour that simply had not
converged. "The remaining explanation needs no mechanism: a random mutation is
worth less than what else the step buys, and the agents are right" was wrong,
and wrong in the confident direction. The four null results stand as facts about
600k and none of them licenses a claim about the mechanism: describing the
shelf, flagging the passives, exempting the step cost and exposing the shrine
were all measured below the budget at which the target behaviour exists.

The shrine bit is still not load-bearing, now tested where it should matter
most. Zeroing index 134 on the 2M agents moves the arm from 2.932 to **2.928**
and `take_mutation` uptake from 29.2% to 28.3%. So the agents learn to take the
shrine mutation without the feature that says one is there, presumably off the
room-kind one-hot and the mask. That is four features in a row surviving their
own removal, and the standing rule holds: in this project an observation change
buys architecture far more often than information, and a budget change buys the
behaviour.

The methodological point is the one worth carrying. Every negative result in the
preceding five sections was collected at a budget where the standard deviation
across identical agents is about a tenth of a level and the behaviour under
study occurs 0.2% of the time. **Before concluding that an agent will not do
something, check that it has trained long enough to do it at all**, and prefer
`--checkpoint-every` off one run to a second arm, because it removes the only
confound this file has repeatedly failed to control.


### Both experiments repeated at 2M, where the behaviour actually exists

The step cost and the shrine bit were both judged at 600k, below the budget at
which the agents take mutations at all. Repeated at 2M, with
`--checkpoint-every 600000` so each arm carries its own curve.

#### The step cost, again

`cost0.02` is the twelve-seed `shrine2m` arm; the other two are four seeds each.

| step cost | budget | level | mutations held | `take_mutation` | `reroll` |
|---|---|---|---|---|---|
| 0.02 (n=12) | 600k | 2.561 | 0.34 | 0.2% | 7.0% |
| 0.02 | 2M | **2.932** | 2.14 | **29.2%** | 0.4% |
| 0.005 (n=4) | 2M | 2.768 | 0.99 | 8.7% | 2.2% |
| 0.0 (n=4) | 2M | 2.924 | 1.64 | 10.0% | 1.7% |

The original hypothesis stays dead and now fails in the opposite direction: the
**standard** 0.02 has the highest mutation uptake at 2M, 29.2% against 8.7% and
10.0% for the cheapened arms. Making the step cheaper has never once moved
mutations in the direction the hypothesis predicted, at either budget.

**A retraction of the reroll finding.** This file says above that the step cost
is "load-bearing, the only brake on the item shop", off four agents at
`step_cost` 0 that put 15% of their actions into `reroll`. That does not
survive. In the 195-dimension setting `reroll` at 600k is 7.0%, 5.9% and 6.1%
across cost 0.02, 0.005 and 0.0, which is no step-cost effect at all, and by 2M
every arm has fallen to 0.4 to 2.2%. The original 15% was two seeds of four at
36% and 24% against two at 0%, so it was a per-seed pathology read as a
mechanism. Budget dissolves it whatever the step cost is, and the section above
overstates its case.

#### The shrine bit, with a control that isolates information

`--blind-shrine` holds the bit at zero and leaves it in the vector, so both arms
are 195 dimensions with the same first layer and differ only in whether that
entry carries information. This is the control the rest of this file never had.
Twelve seeds an arm:

| budget | bit live | blind | blind minus live | live uptake | blind uptake |
|---|---|---|---|---|---|
| 600k | 2.561 | 2.538 | -0.023 [-0.106, +0.059] | 0.2% | 1.7% |
| 1.2M | 2.810 | 2.783 | -0.027 [-0.119, +0.065] | 22.9% | 17.9% |
| 1.8M | 2.886 | 2.881 | -0.005 [-0.082, +0.071] | 28.4% | 39.7% |
| 2M | 2.932 | 2.932 | **+0.000 [-0.067, +0.067]** | 29.2% | 35.7% |

**Identical to three decimals at 2M**, and the blind arm takes *more* free
mutations than the informed one, 35.7% against 29.2%, having never been told one
is in the room. The feature is worthless, and that is now established four ways:
zeroed at evaluation on the 600k agents (+0.001), zeroed on the 2M agents
(-0.003), trained against at four budgets (above), and the blind agents learning
the behaviour regardless.

The agents evidently infer a stocked shrine from the room-kind one-hot and the
action mask, which is all they ever needed. The bit stays in as honest
bookkeeping and nothing more.

#### What the two reruns leave standing

Nothing in the mutation story is a mechanism. Four features and two reward
variants were tested against a behaviour that only exists past about 1.2M steps,
and every one of them measured null both above and below that line. The single
variable that has ever moved mutation uptake is training budget: 0.2% at 600k,
22.9% at 1.2M, 29.2% at 2M, against a heuristic at 100%.


### The sweep at 2M: the same answer, for a completely different reason

The twelve-seed sweep at 600k could not have found anything, and the reason only
became visible once the budget question was settled. Repeated at 2M on the
current environment: `shrine2m` is the live arm and `runs/inert2m_s0..s11.pt`
the inert one, trained with `--without-passives` held for the whole run, init
seed shared so the differences are paired, both graded on the **live**
environment over 240 seeds from 30,000. `--checkpoint-every 600000` carries the
curve.

| init seed | live | inert | live minus inert |
|---|---|---|---|
| 0 | 2.99 | 2.87 | +0.12 |
| 1 | 3.04 | 2.76 | +0.28 |
| 2 | 2.81 | 2.96 | -0.15 |
| 3 | 2.85 | 2.98 | -0.13 |
| 4 | 2.99 | 2.90 | +0.09 |
| 5 | 2.99 | 2.77 | +0.22 |
| 6 | 2.98 | 2.90 | +0.08 |
| 7 | 2.77 | 2.75 | +0.02 |
| 8 | 2.94 | 3.15 | -0.20 |
| 9 | 2.92 | 2.84 | +0.07 |
| 10 | 2.90 | 2.88 | +0.01 |
| 11 | 3.01 | 2.89 | +0.12 |

    live arm   2.932        inert arm  2.888        heuristic  2.283
    paired difference, per training seed  +0.044 [-0.039, +0.126], positive on 9 of 12

The headline is unchanged from every previous version of this sweep: **training
against the corrected environment is not worth a resolvable amount**, and the
interval rules out more than about +0.13. What changed is that the number now
means something.

Across the budget, paired per seed:

    600k   live 2.561   inert 2.559   +0.002 [-0.058, +0.063]   4/12 positive
    1.2M   live 2.810   inert 2.751   +0.059 [-0.009, +0.128]   9/12
    1.8M   live 2.886   inert 2.840   +0.047 [-0.045, +0.138]   7/12
    2M     live 2.932   inert 2.888   +0.044 [-0.039, +0.126]   9/12

| budget | arm | mutations held | `take_mutation` | any mutation |
|---|---|---|---|---|
| 600k | live | 0.34 | 0.2% | 11.8% |
| 600k | inert | 0.38 | 2.3% | 11.8% |
| 2M | live | 2.14 | 29.2% | 68.9% |
| 2M | inert | 0.79 | 3.2% | 28.2% |

**At 600k the two arms are the same policy.** They hold 0.34 and 0.38 mutations
and take the free one 0.2% and 2.3% of the time, which is to say neither of them
engages with the only thing the arms differ in. The sweep that opens this file,
and its predecessor, were comparing two agents that both ignored the mutation
system, and a null result there carries no information about whether the
mutation passives matter to a policy that uses them.

At 2M they are genuinely different policies. The inert-trained agents learn what
is true of their training environment, that mutations do nothing, and carry it
into the live environment where they hold 0.79 against 2.14 and take the free
mutation 3.2% against 29.2%. That is a large behavioural gap and it costs
**+0.044 levels**, which is the honest reading: the correction is worth about
what the shelf is worth (+0.13 to a policy that collects everything), scaled by
how much of it a 2M agent actually collects, and that is small enough to sit
inside twelve-seed noise.

Two numbers worth keeping. The standard deviation across 24 identically
configured 2M agents is **0.098 levels** (2.75 to 3.15) against 0.183 at 600k,
so the budget that makes the behaviour exist also tightens the spread by
almost half. And both arms now clear the heuristic by 0.61 and 0.61 against
0.26 at 600k: the gap between a trained agent and the hand-written baseline is
mostly budget, not design.

The methodological point, which is the third time this file has arrived at it
from a different direction: **an A/B on an environment feature is only
meaningful once both arms have trained long enough to use that feature.** Before
that the comparison is two copies of the same policy with different labels.


### The described shelf at 2M, against a control designed in from the start

Every previous test of the mutation-shelf description was either at 600k, where
no agent engaged with the mutation shop at all, or against a control that moved
`obs_dim` and therefore the architecture with it. `tools/sweep_arms.py --sweep
shelf` is the version without either defect: `--blind-shelf` holds the five
description floats and the passive bit at zero for every slot and leaves
`present`, the takes left and the shrine bit standing, so both arms are 195
dimensions with the same first layer and differ only in information. Each arm is
graded on the vector it trained on, because handing a blind agent a description
it has never seen would measure a distribution shift instead of the value of the
information.

Twelve seeds an arm at 2M, `--checkpoint-every 600000`, init seed shared, 240
evaluation seeds from 30,000. Training took 91.2 minutes for 40M steps at six
concurrent.

| init seed | described | blind | described minus blind |
|---|---|---|---|
| 0 | 2.99 | 2.92 | +0.06 |
| 1 | 3.04 | 2.83 | +0.21 |
| 2 | 2.89 | 2.84 | +0.05 |
| 3 | 2.95 | 2.96 | -0.00 |
| 4 | 2.63 | 2.82 | -0.19 |
| 5 | 2.99 | 2.98 | +0.01 |
| 6 | 3.03 | 2.94 | +0.09 |
| 7 | 2.89 | 2.96 | -0.07 |
| 8 | 2.77 | 2.98 | -0.21 |
| 9 | 2.98 | 2.92 | +0.05 |
| 10 | 2.94 | 2.92 | +0.01 |
| 11 | 2.88 | 2.79 | +0.10 |

    described arm  2.914      blind arm  2.906      heuristic  2.283
    paired per training seed  +0.01 [-0.06, +0.08], positive on 7 of 12
    per eval seed (n=2,880, not independent)  +0.01 [-0.03, +0.05]

| arm | mutations held | `take_mutation` | any mutation |
|---|---|---|---|
| described | 1.92 | 26.3% | 61.0% |
| blind | 1.82 | 19.6% | 58.5% |
| heuristic | 2.23 | 100.0% | 61.3% |

**This null is the informative kind.** Both arms use the mutation shop: they
hold 1.92 and 1.82 mutations and take the free one 26.3% and 19.6% of the time,
against 0.34 and 0.2% for the same comparison at 600k. So this is not two copies
of a policy that ignores the thing under test, which is what every earlier null
on this question turned out to be. The blind agents reach the same level and
nearly the same uptake **without ever being told what is on the shelf**, which
says they get what they need from the presence bit, the action mask and the
room-kind one-hot, and that the description was redundant rather than unused.

That makes five observation features in a row that measure null against their
own control: the described shelf, the passive flag, the exempted step cost, the
shrine bit, and now the shelf again under the protocol the shrine bit forced.
The standing reading is unchanged and now has a fifth data point: **this
project's observation changes buy architecture, not information.**

Standard deviation across the 24 identically configured agents is **0.093
levels** (2.63 to 3.04), which independently reproduces the 0.098 measured
across the 24 agents of the 2M passives sweep. Two separate sets of 24 agents
now agree that the spread at 2M is about a tenth of a level, against 0.183 at
600k.

### Why two sweeps of the same command disagreed, and what it was not

`shrine2m_s*` and `shelf2m_described_s*` are the same command with the same
seeds: 2M steps, `--fast-core`, `--shaping none`, the default step cost, no arm
flags. Comparing the saved weights, seeds 0 and 1 are bit-identical at every
checkpoint and seeds 2 to 11 differ, already at 600k, by 0.42 to 0.55, which is
the scale of the weights themselves.

The first reading here was that training does not reproduce from `--seed`, with
a concurrency-dependent nondeterminism as the suspect. **Both halves of that
were wrong**, and the tests are worth recording because the wrong version was
plausible.

**Training is bit-reproducible.** `tools/determinism_probe.py` trains four
copies of one seed one at a time, then four more all at once, with the workers'
`OMP_NUM_THREADS=1`:

| condition | distinct agents | max weight delta |
|---|---|---|
| alone | 1 of 4 | 0 |
| concurrent | 1 of 4 | 0 |
| both together | 1 of 8 | 0 |

Eight identical agents, exactly zero difference, so the seed and the command
fully determine the result and concurrency has nothing to do with it.

**What actually differed was the code.** The `budget4m` sweep runs the same
configuration on today's tree, and its 2M checkpoints settle it:

| seed | vs `shrine2m` | vs `shelf2m_described` |
|---|---|---|
| 0 | differ | differ |
| 1 | differ | differ |
| 2 | differ | **identical** |
| 3 | differ | **identical** |
| 4 | differ | **identical** |
| 5 | differ | **identical** |

Today's code reproduces `shelf2m_described` bit-for-bit at seeds 2 to 5 and
matches `shrine2m` nowhere. Seeds 0 and 1 match neither, and they are exactly
the two runs `shelf2m` did not train: its log opens `resuming: 4 of 24 already
saved`, and the four that already existed were `described_s0`, `described_s1`,
`blind_s1` and `blind_s2`, written between 02:06 and 02:09 while the twenty
fresh ones start at 02:43. So the environment changed in that half-hour, the
resumed four predate the change, and `shrine2m` from the night before predates
it too.

**What the change was cannot be recovered.** It moved neither `obs_dim` nor
`n_actions`, since every one of these agents still loads. The repository had no
version control at the time, so there is no diff to read. This is the entire
argument for the git history the repo now has: the question would have been a
`git log` rather than three experiments.

**What it costs the shelf sweep.** The four resumed runs are split across the
arms, so seed 0 grades an old-code `described` against a new-code `blind`, and
seed 2 does the reverse. Seed 1 is old against old, and seeds 3 to 11 are new
against new. Two of the twelve pairs are therefore cross-version, and both
happen to sit on the positive side (+0.06 and +0.05). Dropping them moves the
paired difference from **+0.01 to 0.00**, and dropping seed 1 as well takes it
to **-0.02**. The conclusion does not move: the shelf description is worth
nothing, and it is worth slightly less than first reported.

**How to apply.** `--resume` silently mixes code versions across a sweep, which
is worse than the wasted compute it saves, because the mixing is invisible in
the output. Either finish a sweep on one tree or re-run the resumed arms, and
now that the tree is under git, record the commit a sweep was trained on.


### The budget curve past 2M: it flattens

Every result before this one was measured at 600k or 2M, and the single thing
that had ever moved the agents was the budget: 600k to 2M was +0.370 levels,
positive on 12 of 12, and it was the difference between an agent that ignored
the mutation shop and one that used it. The obvious worry was that another
behaviour switches on later and quietly invalidates everything again, the way
the 600k results were invalidated.

`runs/budget4m_long_s0..s11.pt`, twelve seeds at 4M with
`--checkpoint-every 1000000`, plain live environment, unshaped, no arm flags.
The curve comes off checkpoints of **one run per seed**, so the comparison
carries no init, architecture or sampling difference: it is the same twelve
trajectories, graded at four points, over 240 evaluation seeds from 30,000.
Training took 103.9 minutes for 48M steps at six concurrent.

| budget | mean level | sd across seeds | mutations held | `take_mutation` |
|---|---|---|---|---|
| 1M | 2.725 | 0.119 | 0.91 | 27.2% |
| 2M | 2.915 | 0.116 | 1.86 | 25.7% |
| 3M | 2.945 | 0.065 | 2.07 | 36.3% |
| 4M | 2.985 | 0.107 | 2.13 | 12.0% |
| heuristic | 2.283 | | 2.23 | 100.0% |

Paired within seed, which here is paired within *run*:

    1M -> 2M   +0.189 [+0.067, +0.311]   10 of 12 positive
    2M -> 3M   +0.030 [-0.046, +0.106]    8 of 12
    3M -> 4M   +0.040 [-0.030, +0.110]    8 of 12
    2M -> 4M   +0.070 [-0.027, +0.168]    6 of 12
    1M -> 4M   +0.260 [+0.176, +0.343]   11 of 12

| seed | 1M | 2M | 3M | 4M | 4M minus 2M |
|---|---|---|---|---|---|
| 0 | 2.83 | 3.02 | 2.91 | 2.94 | -0.08 |
| 1 | 2.63 | 3.01 | 3.02 | 3.09 | +0.08 |
| 2 | 2.73 | 2.89 | 2.97 | 3.04 | +0.15 |
| 3 | 2.65 | 2.95 | 2.92 | 2.90 | -0.05 |
| 4 | 2.99 | 2.63 | 3.00 | 3.05 | +0.41 |
| 5 | 2.64 | 2.99 | 2.98 | 2.95 | -0.04 |
| 6 | 2.62 | 3.03 | 2.83 | 2.95 | -0.08 |
| 7 | 2.75 | 2.89 | 2.98 | 2.72 | -0.17 |
| 8 | 2.77 | 2.77 | 2.83 | 3.05 | +0.28 |
| 9 | 2.60 | 2.98 | 3.01 | 2.97 | -0.00 |
| 10 | 2.85 | 2.94 | 3.00 | 3.10 | +0.16 |
| 11 | 2.64 | 2.88 | 2.90 | 3.07 | +0.18 |

**Doubling 2M to 4M buys nothing resolvable**: +0.070 [-0.027, +0.168], and only
six of twelve seeds improve, which is what a coin flip looks like. The budget
lever that carried this project from 2.28 to 2.9 is spent by about 2M, and the
last 24M steps of training bought a tenth of what the first 24M did.

**Mutation holdings saturate**, which is the mechanism behind the flattening:
0.91, 1.86, 2.07, 2.13, converging on the heuristic's 2.23 and with nowhere left
to go. The behaviour that appeared between 600k and 1.2M is simply finished by
about 3M, and nothing else takes its place.

**Retract the uptake progression.** `take_mutation` was quoted across this file
as 0.2% at 600k, 22.9% at 1.2M, 29.2% at 2M, read as a behaviour switching on.
The arm means here go 27.2%, 25.7%, 36.3%, 12.0%, which is not a progression at
all, and the per-seed distribution says why:

| budget | min | median | max | seeds at exactly 0% |
|---|---|---|---|---|
| 1M | 0.0% | 7.2% | 93.5% | 4 of 12 |
| 2M | 0.0% | 6.4% | 100.0% | 2 of 12 |
| 3M | 0.0% | 17.1% | 100.0% | 4 of 12 |
| 4M | 0.0% | 4.7% | 51.7% | 3 of 12 |

The median agent takes the free shrine mutation under a fifth of the time at
every budget, a quarter of them never take it at all, and one or two take it
almost always. **The arm mean is a mean of a heavily skewed distribution and
moves on which seeds happen to be extreme**, so it should not be read as a
behavioural trend. Mutation *holdings* are well behaved and monotone; report
those instead. The 600k figure of 0.2% is still real, because there the whole
distribution was at zero.

**How to apply.** 2M is the budget for this environment: 1M is visibly short and
4M is not distinguishable from 2M. Twelve seeds at 2M costs about 50 minutes at
six concurrent and resolves a tenth of a level; going to 4M doubles that for
nothing. If a future change needs more budget than that to show itself, it is
already smaller than the run-to-run spread.


### The starting squad, and what it is worth

`ChipChoice/Squads.json` ships eight startable squads and the sim used none of
them, opening instead on `Game.Team.Packs`: five bare Novices at **750 Power**.
The real squads are **3,783 to 7,176 Power**, five to ten times that, and they
have compositions rather than five copies of one body.

That single correction is what made the room-fight model playable. On the
`room-fights` branch, 60 seeds, `frontline` placement:

| | before, five bare Novices | after, `Squad1` |
|---|---|---|
| heuristic | 1.000 | **4.267** |
| random | 1.000 | 1.817 |
| fights a run | 1.2 | 23.1 |
| fight win rate | 17% | 96% |

So "every room fights" was never the problem. The sim was fielding a squad the
game does not hand anyone, against rooms that now fight back, and dying in the
first one.

### Which squad is best is not written down

120 seeds each under the heuristic, same placement, same environment:

| squad | mean level | max | fights a run | roster |
|---|---|---|---|---|
| Squad2 | **4.492** | 6 | 24.7 | hood-dark, stone-sword, shaman-mask, 2 bare |
| Squad1 | 4.217 | 5 | 22.9 | stone-sword, crossbow, shield, 1 bare |
| Squad5 | 4.050 | 5 | 22.6 | plague-mask, chainsaw |
| Squad4 | 3.967 | 5 | 21.6 | ring-green, lance, shield |
| Squad3 | 3.900 | 5 | 21.1 | football, paddle-ball, gloves |
| Squad8 | 3.825 | 5 | 20.9 | gun, mad-claws, gloves, 1 bare |
| Squad7 | 3.633 | 5 | 19.3 | pretzel, bed, 1 bare |
| Squad6 | 3.467 | 5 | 17.6 | burning-ring, 3 bare |

**The spread is a whole level**, 3.467 to 4.492, against a run-to-run spread of
about a tenth of a level between identically configured 2M agents. Squad choice
is worth more than any environment feature this project has measured, and more
than the entire 600k-to-2M budget gain.

Two things worth noting before this is read as a ranking. It is the heuristic's
ranking, and a trained policy may prefer a different squad, because the squads
are not scaled versions of each other: `Squad5` is two units, which is half the
food per move, and it places third on raw survival despite fielding the fewest
bodies. And `Squad2` is behind a Cultist unlock, so a fresh save cannot choose
it; only `Squad1` is unconditional.

**How to apply.** The choice belongs in the agent, not in a constant. The
cleanest shape is a one-hot of the squad in the observation with the squad
sampled uniformly during training, so one policy learns to play all eight and
the ranking is read off by evaluating it per squad. That conditions the policy on
the roster it actually has, where an outer bandit over eight arms would learn a
ranking for whatever average policy it happened to be paired with. It costs
`obs_dim + 8`, so it needs the blind-control protocol like any other observation
change.


### Where this stands, 2026-09-02

The environment changed twice today and **every number in this file above these
last sections was measured on a sim that is now known to be wrong**. The
methodology survives; the levels do not.

    random     1.429       heuristic  3.946      240 seeds, squads cycled
    obs_dim    203         actions    33

What changed: every room fights and its shop opens only on the win
(`notes/reference-sim.md`), and a run starts from one of eight squads rather
than from five bare Novices. The second is why the first looked broken: with
room fights and the old squad both baselines finished at 1.000 on a 17% win
rate, and with `Squad1` the same code gives 4.267 at 96% over 23 fights a run.

Left running: a twelve-seed 2M sweep, tag `roomfight2m`, `--checkpoint-every
1000000`. Score with

    python tools/sweep_arms.py score --sweep budget --tag roomfight2m --runs 240
    python tools/sweep_arms.py score --sweep budget --tag roomfight2m --at 1000000

and read it against the 3.946 heuristic. Then the questions worth asking:

- **Does a trained agent rank the squads the way the heuristic does?** Pin one
  with `--squad` and score all eight. The heuristic's order is Squad2, Squad1,
  Squad5, Squad4, Squad3, Squad8, Squad7, Squad6, and it plays them all the
  same way, which is exactly the confound a conditioned policy removes.
- **Is the squad one-hot load-bearing?** `--blind-squad` holds it at zero at
  fixed `obs_dim`. Five features in a row here have measured null against that
  control; this is the first one with a mechanism large enough to expect
  otherwise.
- **The 96% fight win rate.** Runs now end at level 4-5 with almost every fight
  won, so something other than losing a fight is ending them. Find out what:
  food, the step cap, or the boss.
- **Quest rooms and the level-entry dialog.** `M_Dialog` carries choices,
  branching outcomes, events and unlocks, and `Quests.json` ships eleven
  quests. None of it is modelled, and a dialog fires on every level change.
- **`C_Fight`'s escalation.** 120s starts a damage bonus over time and 180s a
  dying DoT; `max_fight_seconds = 120` is that number with the wrong meaning.


### The first agent on the corrected environment loses to the heuristic

`runs/roomfight2m_long_s0..s11.pt`, twelve seeds at 2M with
`--checkpoint-every 1000000`, squads cycled, 240 evaluation seeds. Training took
71.5 minutes for 24M steps at six concurrent, about 35 minutes a run against the
50 the old environment took, so the extra fights cost less than expected.

| budget | mean level | sd | against the heuristic's 3.946 | beats it |
|---|---|---|---|---|
| 1M | 3.451 | 0.115 | **-0.495** [-0.560, -0.430] | 0 of 12 |
| 2M | 3.797 | 0.229 | **-0.149** [-0.279, -0.020] | 3 of 12 |

**The agent is behind the hand-written baseline**, and the interval excludes
zero, so the gap is real even if it is narrow. On the superseded environment a
2M agent beat the heuristic by about +0.6; that margin is gone and has changed
sign. The honest reading is that the old margin was partly a fact about a sim in
which two thirds of the rooms were peaceful and every squad was identical: an
environment with more to exploit, and a baseline with less to do.

**Budget has not saturated here, and the earlier guidance does not transfer.**
1M to 2M is **+0.346 [+0.205, +0.486], positive on 12 of 12**, against a
2M-to-4M of +0.070 on the old environment. The note above this one concluding
"2M is the budget for this environment" was about the old one; on the corrected
environment 2M is visibly short, and 4M is the obvious next run rather than a
waste.

| arm | mutations held | `take_mutation` |
|---|---|---|
| trained, 2M | 2.21 | 2.5% |
| heuristic | 4.23 | 100.0% |

The mutation gap is wider than it ever was before: the heuristic holds nearly
twice what the agent does, and the agent takes the free shrine mutation 2.5% of
the time where a 2M agent on the old environment reached 29.2%. Given that
budget is still buying levels here, the most likely explanation is simply that
2M steps no longer covers the same ground: episodes are longer, the observation
is wider, and the policy is now learning eight rosters rather than one. Report
holdings rather than uptake, for the reason recorded earlier in this file.

Also worth keeping: the spread across twelve identically configured agents is
**0.229**, more than double the 0.098 of the old 2M sweep, which is what a
harder environment with a cycled squad does to run-to-run variance. A single
agent means less here than it used to; the best of these twelve scores 4.308 and
the worst 3.458.
