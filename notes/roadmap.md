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

Some constants will never be recovered: `maxDeadEnds`, `minFinishDistance`,
whether a room may cover several squares, the anti-repeat weighting inside a
quality. `sim/assumptions.py` is the honest boundary of the work, not a list
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
- **Portals.** `portalsCount` and `minPortalDistance` in `mapgen`. `RoomMap`
  already links them and `from_table` reads the shipped three; generated levels
  have none, so the `portal` action has never once been legal.
- **The remaining room types.** `Secret` (with `activatesSecret`,
  `secretIsActive`, `roomsToActivateSecret = 2`, `secretRoomCount`, and
  `SecretRoom: true` on level 7), `TalentShop`, `PermanentShop`,
  `ConsumableShop`, `QuestExtra`, `FinalBoss`.
- **The shrine split.** `Shrines` is 1 at level 1 and 0 after, while
  `RerollShrines` is 1 from level 2 on. The sim conflates them.
- **Quests.** Eleven in `Quests.json`, each with `Doors`, `ExtraRooms`,
  `RoomParams` (a named `Layout`, directional priorities,
  `guaranteedPositions`) and per-quest fields. `C_Rooms.CreateQuest`, the
  `QuestStatus` lifecycle, the per-level shortlist (levels 2, 6 and 10 only),
  and `questCount` / `extraQuestRoomCount` / `questRoomAllowedDoors` /
  `questRoomGuaranteedDoors`. Settles whether a quest room fights.
- **Dialogs.** `C_Levels.StartDialog` fires one on every level change, drawn
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
- **Consumables.** `Consumables.json`, `ConsumablesByLevel.json`, and
  `Rooms.json`'s `Consumables` and `SellCost`. Unreachable in Default (all eight
  chips have `StatShops: 0`) but `TalentShop` builds the shop, and
  `KingOfTheHill`, `Arcade` and `Tasks` are unchecked. Implement the mechanism,
  then record where it fires.
- **Layouts.** `LayoutsByPack.json` and `SpecialRoomLayouts.csv`, neither named
  anywhere in the sim, plus `M_EnemyPack.ChooseLayout`.
- **`InstantTransitions`**, and one more attempt at `maxDeadEnds`,
  `minFinishDistance` and multi-square rooms with `tools/data_xrefs.py`.

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

   So Batch 1 now picks up at the rest of the run layer: **portals**
   (`portalsCount` is 0 in `sim/mapgen.py`, so the portal action has never once
   been legal), the remaining room types, the shrine and reroll split, the 11
   quests, the dialog system now that the table is decoded, consumables,
   layouts, and `InstantTransitions`.

Tools that did not exist before this session and are worth knowing about:
`tools/coverage.py` (what the game ships against what the sim reads),
`tools/data_xrefs.py` (rip-relative references to a string literal, which is how
`FightInFirst` was found), `tools/determinism_probe.py`, and
`tools/profile_train.py --deep`.
