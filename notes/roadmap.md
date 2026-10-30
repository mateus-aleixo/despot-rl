# The road to a complete sim

The goal is every mode and every feature of *Despot's Game*, not a subset chosen
for what an RL agent happens to need. This file is the plan for getting there
and the standard for calling it done.

## Phase 0 — the scope, measured rather than guessed

`tools/coverage.py` enumerates every shipped table against what `sim/` and `rl/`
actually read. Run 2026-09-02:

    91 tables, 1,749 distinct column names, 648 never mentioned (37%)
    28 tables this sim never names at all
    606 C_ and 584 M_ classes in the dump, 26 and 30 mentioned here

**A correction, because the first run of this tool was wrong and the wrong
number was briefly planned against.** It did a plain substring scan and reported
850 misses and `Skills.json` as 23 columns of which the sim reads one. Both were
artifacts: `Param1Name` is reached through `row.get(f"Param{i}Name")`, `Q7Prob`
through `f"Q{q}Prob"`, `DamagePerLevel` through `f"{key}PerLevel"`. The tool now
walks the AST for f-strings, and `Skills.json` reads **21 of 23**.

The remaining numbers want reading carefully too:

- `Meta.json` alone is 553 of the 1,749 columns and 308 of the 648 misses,
  because its keys are class *names*. Excluding it, column coverage is about
  **72%** and the miss list is 340 rather than 648.
- The C_/M_ class counts flatter the gap in the other direction: much of the
  dump is view, tooltip and audio code a headless sim should never have.

**So the shape of the work is structural, not column-level.** The sim reads most
of the columns of the tables it knows about. What is missing is whole
subsystems: quests (13 of 31 columns, table never named), dialogs (table now
located and decoded, nothing implemented), consumables, and the entire
`Arcade/` and `Chips/Hard/` trees.
The batches below are organised around that.

## How much to land at once

The instinct to implement everything together is right about implementation and
wrong about measurement, and the two need separating.

**Implementation batches as large as it can be verified.** The constraint is not
one feature at a time; it is that each feature ships with a check that fails when
that feature is removed. With per-feature checks, batch size does not affect
correctness, and a small batch costs another full re-baseline for nothing. So:
land whole subsystems.

**Measurement cannot batch, because one number cannot attribute a regression.**
The proof is from this project, today. Room fights alone dropped the heuristic
from 3.946 to 1.000, and the cause was found in an hour only because room fights
were the sole change in flight. With five subsystems in flight that search is
combinatorial. Two runs differing in exactly one thing is also how the
`shrine2m`/`shelf2m` divergence was traced to a code change rather than to
nondeterminism.

So: implement in large batches with a check per feature, and take RL numbers
only at batch boundaries, never inside one.

## What "done" can mean

**Not bit-exact behaviour.** This repository measured why: nudging every start
position by `1e-4` and re-running *the same engine* flips the winner in about
one splash fight in twelve, and the cross-engine error is the same order (see
`notes/rust-core.md`, "The residual error is chaos, not logic"). float32 against
float64 over a 120-second fight is a far larger perturbation than that.
Identical inputs will not give identical outcomes in anything that is not the
original binary.

**Done means: every mechanism present and correct, with outcome distributions
matching, and every constant either read from the shipped data or recorded in
`sim/assumptions.py` as a choice.** That is checkable, and `tools/diff_core.py`
already works to exactly this standard for the battle loop.

Some constants will never be recovered: whether a room may cover several
squares, the anti-repeat weighting inside a quality. (`maxDeadEnds` and
`minFinishDistance`, once on this list, were both recovered.) `sim/assumptions.py` is the honest boundary of the work, not a list
that eventually empties.

## The rule that makes completeness real

**Every feature lands with a check that it changes something, not that it
parses.** `vampirism` sat registered as implemented for weeks while reading
`percent` where every shipped row says `value`, healing exactly zero, and it was
counted in the coverage number the whole time. Forty features implemented and
none of them tested is not a complete sim, it is a larger untested surface.

## Measurement policy

1. **Freeze the interface inside a batch.** Land it all, then re-baseline once.
   Never train against an interface that is about to move.
2. **The heuristic is the reference**, not a previous agent: it is the only
   thing that survives an interface change, because it is re-measured rather
   than loaded.
3. **A baseline is random + heuristic over 240 seeds** with the squads cycled,
   recorded in `notes/rl.md` with the date and the batch.
4. **Agents are disposable.** `runs/` holds 675 checkpoints and 332 MB, of which
   only the 96 at 203x33 load at all. They are worth keeping only for forensics
   (comparing two runs' weights is how the `shrine2m`/`shelf2m` divergence was
   traced to a code change), and 332 MB is not a constraint. The durable record
   is `notes/`, never `runs/`.

---

# Batch 1 — the run layer, complete

Everything about what a run *is*. All of it touches `RunState`, `RoomMap` or the
observation, so each item alone would break the interface; landing them together
costs one break instead of a dozen.

**Ordered first because it is the interface-breaking batch.** Doing it before
Batch 2 means Batch 2's measurements survive; the other way round they would be
thrown away.

- ~~**Fog of war.**~~ **Done, 2026-09-02.** `RoomState`, the reveal in
  `C_Rooms.SetCurrent`, `known_to_boss`, an observation built on the revealed
  subgraph, `BossVision`, and a fog-aware `render_run`. It held that fog changes
  *information* and not legality: `legal_actions` is untouched, because a door
  neighbour is always revealed by standing next to it. `obs_dim` went 203 to
  210. Details and the second leak the check caught are in
  `notes/reference-sim.md`.
- **The `to_boss` replacement was picked, not measured, and that is a deviation
  worth naming.** The plan called for three arms. What shipped is the middle
  candidate: distances over the revealed subgraph, empty until the boss is
  found, with a `boss_found` flag and a frontier fraction beside it. The reason
  for not measuring now is the measurement rule itself. Portals, the remaining
  room types, quests, dialogs and consumables are all still to land in Batch 1
  and every one of them moves the observation, so a three-arm sweep taken today
  would be scored against an interface that is about to move, which is the
  single thing this project has wasted the most time on. **The fork moves to the
  Batch 1 boundary**, where it costs one sweep against a stable interface.
  `--lights-on` is the control it needs and it exists now: it reveals the whole
  level at an unchanged `obs_dim`, so the pre-fog observation is an arm rather
  than a code change. The other two candidates are a few lines from here, one
  dropping the two boss features and one dropping the frontier.
- ~~**Portals.**~~ **Done, 2026-10-29**, and with them the rest of
  `PositionRooms`. `portalsCount = num / 7 + 1`, `minPortalDistance = 3` and
  `minFinishDistance = num / 6 + 2` are all written by `C_Rooms.Generate`, so
  three "unrecoverable" parameters were recoverable. Generated levels carry two
  to four portals; a teleport is its own free action (four slots, in reveal
  order); the finish and start are uniform draws; shops spread only against
  their own type; every level after the first opens in an item shop. Portals
  are worth +0.237 [+0.115, +0.360] levels to the heuristic, paired. `obs_dim`
  210 to 258, actions 33 to 36. Details in `notes/reference-sim.md`, "Portals,
  and `PositionRooms` read end to end".
- ~~**The remaining room types.**~~ **Done, 2026-10-30.** `Secret`: made by
  `MaybeMakeSecretRoom` off the door graph, a portal, switched on by two
  `activatesSecret` portals; its level-7 shop's pool has no data in any
  ruleset, so it is empty (`sim/assumptions.py`). `ConsumableShop` and
  `QuestExtra` landed earlier. `TalentShop` (`StatShops`) is 0 on every Default
  level, `FinalBoss`'s item shop is past the end of the run, and
  `PermanentShop` is not made by the Default generator.
- ~~**The shrine split.**~~ **Done, 2026-10-29.** The placement was right
  (`ChooseTreasureRooms` gives `RerollShrines` the `MutationShop` bit), and the
  shop's own parameters were not: `C_Room..ctor` builds a generated mutation
  shop as 3 shown, 1 taken, 2 free rerolls, where the sim used `m1`'s 10 and 2
  and no reroll. Draws exclude what the run holds. The shelf is now worth +0.04
  [-0.02, +0.10] to the heuristic, against +0.13 on the old shelf.
- ~~**Wounds between fights.**~~ **Done, 2026-10-29**, found on the way: in
  Default nothing heals a unit between fights, and the sim healed every unit to
  full before each one. `Human.hp` carries the share. Worth **-1.000 [-1.105,
  -0.895]** levels to the heuristic, the largest single change this project has
  measured. `FoodPackHeal` implemented with it.
- ~~**The boss room and the pipe.**~~ **Done, 2026-10-29.** `C_Levels.WinLevel`
  does not move the squad; `V_Pipes` is the only gameplay caller of
  `ChangeLevel`. The run stays on the level after the boss until it takes the
  pipe (`descend`), and every boss room but the last has a free three-slot
  consumable shop.
- ~~**Quests.**~~ **Done, 2026-10-30**, all eleven. Every quest's rooms are placed
  (`ChooseQuestRoom`, `ChooseQuestExtraRoom`, `CountValidPaths`), typed, paid
  and fought as the game does; the dialogs are the NodeCanvas trees
  `tools/dump_dialogue_trees.py` pulls out of `sharedassets2/3.assets`, run by
  `sim/quests.py`; Pit, Sci, Cube, Rat, Reaper, Spider, Bot, Nurgle (into its
  second level), Flight, Sword and SecretRoom are ported, with their
  companions (Zombie, Rat, Bot, TalkingSword). `RoomTransition.CheckDirection` turned out to
  refuse moves into `Unknown` rooms, which the sim had allowed. Worth -0.072
  [-0.124, -0.020] to the heuristic over 2400 seeds, which passes quests by
  (following them costs it -0.028 [-0.049, -0.007] more; the first 240-seed
  reading, -0.121 and -0.113, sat low). Details in `notes/reference-sim.md`. Still open: the
  level-2 reward mutations (306, 218, 328, 233, Batch 2's mutation work) and
  the placement layer's `guaranteedPositions`.
- ~~**Dialogs.**~~ **Done, 2026-10-30**, the 46 of the Default pool with all 26
  of their event types, plus consumable 13; `DejaVu` still open. Worth +0.196
  [+0.021, +0.370] levels to the heuristic, and choosing well +0.154 over
  random. Details in `notes/reference-sim.md`. The original plan follows.
  `C_Levels.StartDialog` fires one on every level change, drawn
  without replacement, with `M_Dialog`'s `choices`, branching `outcomes` and
  `M_Event[]`. **The agent currently makes zero choices per level transition
  where a player makes one.** The table is **found**: it is
  `EncryptedMainGroup/dialogs.json`, beside `DB/` rather than inside it, with a
  46-entry Default pool and a 26-event vocabulary. `notes/reference-sim.md` has
  the decoded schema and `tools/show_dialog.py` renders one. What remains is
  implementation: the draw without replacement, the weighted outcomes, the 26
  events, the gates that fill `enableds` and `hiddens`, and the choice itself as
  an action in the observation. `unlocks` is unused in the shipped file, so skip
  it. Note the second entry point: consumable ID 10 fires `Dialog.DejaVu`.
- ~~**Consumables.**~~ **Done, 2026-10-29**; Event and DejaVu followed on
  2026-10-30, so thirteen of the fourteen (DamageAllEnemies is on no Default
  list). The first pass, eleven of the fourteen: they are
  live in every Default run, from the boss rooms. Two carried, free to take,
  5 gold to sell; heals, the cleaner, reserves, resurrection, teleport,
  AntiHunger and reroll modelled. DejaVu and Event wait on dialogs;
  DamageAllEnemies is on no Default list. Worth +0.721 [+0.597, +0.844] levels
  to the heuristic. Open: the placement stage, where the game lets a cleaner be
  used after the pack is on screen (`sim/assumptions.py`). The long-run reward
  (a won level 4 makes the next run's start a two-slot consumable shop) is in
  since 2026-10-30: +0.147 [+0.090, +0.205] on a run that has it, off by
  default in the env, "carry" for the save's behaviour.
- ~~**Layouts.**~~ **Done, 2026-10-30**, for the packs: each fights in its one
  `LayoutsByPack` layout with every enemy in its row's zone, where the sim drew
  any of 77 layouts with everyone in `e1`. Worth -0.263 [-0.439, -0.086] to the
  heuristic. `SpecialRoomLayouts.csv` landed with the quests.
- ~~**`InstantTransitions`**~~ **closed, 2026-10-30**: `M_GameData.instantTransitions`
  is false in every shipped ruleset, and the path it would switch on,
  `C_Rooms.InstantMove`, spends food through `C_Food.Move` exactly as a walk
  does. Nothing to model.
- ~~**`maxDeadEnds`**~~ **Done, 2026-10-30.** It really is 0; what the port had
  wrong was `Adjust`, which trades each cut dead end for a room on a uniformly
  drawn open position, keeping the count, and stops when none is accepted
  (undoing the last cut). `CloseLoops` moves dead ends into the gap of a ring
  of seven rooms. Growth, `Adjust`, `CloseLoops` run once each. Still open:
  multi-square rooms.

**Boundary:** re-baseline random + heuristic over 240 seeds, squads cycled, then
one sweep for the `to_boss` question, which is a genuine fork rather than a
feature.

# Batch 2 — the battle layer, complete

Nothing here moves `obs_dim`; all of it changes how fights resolve. Measured
against Batch 1's baseline.

- **The `C_Fight` escalation schedule.** `_MIDDLE_FIGHT_TIME = 120` starting a
  `M_DamageBonusOverTimeStatus`, `_LONG_FIGHT_TIME = 180` starting a
  `M_DyingDotStatus`, `_LONG_TIME_WITHOUT_DAMAGE = 8`, the final-boss variants
  (1200 / 1320 / 12), and `DELAY_SINGLE` / `DELAY_KOH`. Our
  `max_fight_seconds = 120` is that number wearing the wrong meaning: 120s is
  where the game starts pushing a stalemate, not where it gives up. Port to both
  `sim/battle.py` and the Rust core.
- **The 28 mutation names with no handler**, of 101 shipped. 259 of 1,094 offers
  are implemented, which is what caps the shelf at +0.13 and therefore caps
  every mutation result in `notes/rl.md`.
- **An audit of the 79 that do have handlers.** Three read the wrong key and did
  nothing while counting as implemented (`vampirism` reading `percent` where the
  rows say `value`, then `ExplodeProjectile`, then `EventualKnockback`). There
  is no reason to think the other 76 were ever checked.
- **Class and unit skill coverage**, the same way: how much of `Meta.Classes`
  the sim can actually field, and which `ClassSkills` and `Skills` rows resolve
  to a real effect rather than a no-op.

**Boundary:** re-baseline, and re-run `tools/diff_core.py` to confirm the two
engines still agree 60/60 with the escalation in.

# Batch 3 — modes and chips

`load_ruleset(mode, chip)` already layers `Common`, then the mode, then the
chip, so the plumbing exists; the mechanics each turns on do not.

- **Modes:** `Default`, `KingOfTheHill`, `Arcade`, `Tasks`.
- **Default's chips:** `default`, `easy`, `hard`, `hunger`, `mutagens`, `crazy`,
  `shortpvp`, `arcade`.

Some are already only data: `hunger` shrinks the room counts, `shortpvp` is
seven levels of 5-6 rooms, `crazy` is **two** levels, one of 3 rooms and one of
100. Others need mechanics: `KingOfTheHill` has its own directory and its own
fight delay, `shortpvp` implies a PvP resolution path, and `Arcade` ships eight
of its own tables that nothing here reads.

**Boundary:** each mode loads under `strict=True`, has its own check suite, and
gets its own baseline. A mode that loads but has no mechanics is not
implemented, and the coverage report should say so rather than counting it.

# Batch 4 — closing it out

- **`tools/coverage.py` becomes a gate**, not a report: it fails when a table is
  named nowhere, and its numbers go into `notes/` with a date.
- **An `assumptions.py` audit**: every entry either resolved against the binary
  or restated with why it cannot be.
- **A check per feature**, enforced. A feature registered without one is not
  implemented, which is the whole `vampirism` lesson.

## Ordering summary

| batch | what | breaks | boundary |
|---|---|---|---|
| 1 | fog, portals, all room types, quests, dialogs, consumables, layouts | obs + actions | re-baseline, plus a sweep for the `to_boss` fork |
| 2 | fight escalation, 28 mutations, audit the 79, skill coverage | fight outcomes | re-baseline, `diff_core` 60/60 |
| 3 | 4 modes, 8 chips | new rulesets | per-mode suite and baseline |
| 4 | coverage as a gate, assumptions audit, check enforcement | nothing | none |

RL results are worth taking only at a boundary. Training against an interface
that is about to move is the single thing this project has wasted the most time
on.

---

## Where the next session picks up

State at the close of 2026-09-02: nothing running, `main` clean and pushed,
`room-fights` merged. Baselines on the current environment are **random 1.429,
heuristic 3.946** over 240 seeds with the squads cycled; the best trained agent
is `roomfight4m_long_s9` at 4.600 against an arm mean of 3.918, which is a draw
with the heuristic rather than a win.

The `reward3m` sweep (flat against a rising level bonus, 24 runs at 3M) was
**stopped part-way and its checkpoints deleted**. Do not look for its results.
Re-run it after Batch 1 if the question still matters; it will have to be
re-trained against the new interface anyway.

**Start with Batch 1**, and inside it start with the two things that unblock the
rest:

1. ~~Locate the dialog table.~~ **Done, 2026-09-02.** It is
   `EncryptedMainGroup/dialogs.json`, beside `DB/` rather than inside it, which
   is why the `EncryptedMainGroup/DB` sweep missed it. Nothing about the
   level-transition decision is blocked: the schema, the pool and the 26-event
   vocabulary are all decoded in `notes/reference-sim.md`, and
   `tools/show_dialog.py` renders any of them. The dialog work is now ordinary
   implementation, so it can be sequenced with the rest of Batch 1 rather than
   ahead of it.
2. ~~Fog.~~ **Done, 2026-09-02**, along with `BossVision` and a fog-aware
   `render_run`. The `to_boss` fork was **picked rather than measured**, and the
   three-arm sweep it was meant to get moves to the Batch 1 boundary for the
   reason given above; `--lights-on` is the control arm and is in place.

   So Batch 1 now picks up at the rest of the run layer: the remaining room
   types, the 11 quests, the dialog system now that the table is decoded,
   consumables, layouts, and `InstantTransitions`.
3. ~~Portals.~~ **Done, 2026-10-29**, with the rest of `PositionRooms` (see the
   Batch 1 list). The cheapest next step is reading `C_Rooms.FromGeneratedData`,
   because it decides whether the consumables item is a Default feature at all.

Tools that did not exist before this session and are worth knowing about:
`tools/coverage.py` (what the game ships against what the sim reads),
`tools/data_xrefs.py` (rip-relative references to a string literal, which is how
`FightInFirst` was found), `tools/determinism_probe.py`, and
`tools/profile_train.py --deep`.

## Tests and CI

Done 2026-10-04 (commits `2423de8`, `0a51096`). `tests/` holds 160 tests that need no
game data, and GitHub Actions builds the Rust core and runs them on every push:
pathfinding against a reference Dijkstra, ORCA's no-collision guarantee, the
behaviour-tree node semantics, the ruleset layering on a synthetic metadata tree,
level-generation invariants, Rijndael at Nb=4 against the FIPS-197 vectors, and the
core's RNG against CPython's bit for bit. With `data/` present, `pytest` also runs
`tools/validate_sim.py` and `tools/validate_rl.py`. The round-trip test found a real
bug on the way in: `tools/rijndael256.py` could not expand 128- or 192-bit keys at
Nb=8.

Still open from the plan agreed on 2026-09-04:

1. **Characterisation tests over the shipped tables.** Assert what the game's own
   data says, so a change to the extraction pipeline that silently alters a unit
   stat fails loudly instead of quietly retraining the agent against a different
   game. They need `data/`, so they run locally, beside the two validate suites.
2. **Invariants around `sim/assumptions.py`.** Each assumption gets a test that
   states what it implies about the simulation, so an assumption that is later
   corrected fails the tests that depended on the old value rather than being
   forgotten.
3. **Seeded golden runs.** A fixed seed, a fixed squad, a fixed number of
   decisions, and an asserted trajectory digest. This is what catches an
   environment change disguised as a policy change, which is the failure mode the
   `notes/` retractions are all about.
4. **A CPU-only training smoke.** A handful of PPO updates on a tiny budget, to
   prove `rl/train.py` still runs end to end without needing a GPU or minutes.
5. **ruff in CI.** The workflow runs the tests and builds the core; there is no
   lint step yet.
