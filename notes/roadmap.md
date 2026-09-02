# The road to a complete sim

The goal is every mode and every feature of *Despot's Game*, not a subset chosen
for what an RL agent happens to need. This file is the plan for getting there
and the standard for calling it done.

## What "done" can mean

**Not bit-exact behaviour.** This repository measured why: nudging every start
position by `1e-4` and re-running *the same engine* flips the winner in about
one splash fight in twelve, and the cross-engine error is the same order (see
`notes/rust-core.md`, "The residual error is chaos, not logic"). float32 against
float64 over a 120-second fight is a far larger perturbation than that. Identical
inputs will not give identical outcomes in anything that is not the original
binary.

**Done means: every mechanism present and correct, with outcome distributions
matching, and every constant either read from the shipped data or recorded in
`sim/assumptions.py` as a choice.** That is checkable. `tools/diff_core.py`
already works to exactly this standard for the battle loop.

Some constants will never be recovered: `maxDeadEnds`, `minFinishDistance`,
whether a room may cover several squares, the anti-repeat weighting inside a
quality. `sim/assumptions.py` is the honest boundary of the work, not a list
that eventually empties.

## The rule that makes completeness real

**Every feature lands with a check that it changes something, not that it
parses.** `vampirism` sat registered as implemented for weeks while reading
`percent` where every shipped row says `value`, healing exactly zero. It was
counted in the coverage number the whole time. Forty features implemented and
none of them tested is not a complete sim, it is a larger untested surface.

Concretely, for each item below: a check in `tools/validate_sim.py` or
`tools/validate_rl.py` that fails if the mechanism is removed.

## Measurement policy while this is going on

Every environment change invalidates every agent and every number, and that has
happened four times in a day. So:

1. **Freeze the interface inside a phase.** Land all of a phase's changes, then
   re-baseline once. Do not train against an interface that is about to move.
2. **The heuristic is the reference**, not a previous agent: it is the only
   thing that survives an interface change, because it is re-measured rather
   than loaded.
3. **Re-baseline is random + heuristic over 240 seeds** with the squads cycled,
   recorded in `notes/rl.md` with the date and the phase.
4. **Agents are disposable.** `runs/` holds 681 checkpoints and 332 MB, of which
   only the 96 files at 203x33 can be loaded at all; everything else is a
   permanently unloadable artifact of a superseded interface. They are worth
   keeping only for forensics (comparing two runs' weights is how the
   `shrine2m`/`shelf2m` divergence was traced to a code change), and 332 MB is
   not a constraint. The durable record is `notes/`, never `runs/`.

---

# Phase 1 — Navigation and knowledge

**Why first:** it is the only gap that makes the task *easier* than the game
rather than cheaper to exploit, so every navigation number measured so far was
measured by a policy playing with the lights on. Portals ride along because they
are a navigation option and fog is what makes navigation a decision; separately
they cost two invalidations for one result.

### 1.1 Room state and the reveal

- `RoomState` is `Unknown = 0`, `Unexplored = 1`, `Explored = 2`, `Current = 4`;
  `Rooms.json` ships `Explored: false`.
- `C_Rooms.SetCurrent` sets the entered room's `state`, then walks
  `M_Room.get_neighbors` reading each neighbour's `get_state`. That is the
  reveal step.
- Add `Room.state` and set it in `RunState.apply`'s move branch and in
  `next_level` / `RunState.new` for the opening room.

**Note on legality:** fog changes *information*, not the action space. Moves are
already restricted to orthogonal neighbours, and a neighbour is always revealed
by standing next to it. So `legal_actions` does not change; only `_encode` does.

**Check:** a fresh level has exactly the start room `Explored` and its
neighbours `Unexplored`, everything else `Unknown`; entering a room moves it to
`Explored` and its neighbours to at least `Unexplored`; the count of non-`Unknown`
rooms is monotone across a run.

### 1.2 The observation, which is a design decision and not a lookup

`RoomMap.to_boss` is a BFS over the whole graph, computed in `__init__` before a
room is entered, and four observation entries read it: distance to the boss,
whether each neighbour is nearer the boss, the room count, and the fraction
cleared. `_move_targets` also picks which portal to expose by `min(to_boss)`.

Under fog there is no whole graph. Three replacements, which are **different
agents rather than different implementations of one agent**:

- **A, blind.** Drop the boss-distance features. Navigate on the local view and
  the room-kind one-hot alone.
- **B, revealed subgraph.** Distance over what is known, with a convention for
  the unreached: optimistic (unknown rooms are free), pessimistic (unreachable),
  or a separate "boss not yet found" flag.
- **C, frontier.** How many revealed rooms are still unentered, and in which
  directions, replacing distance entirely.

**Measure it, do not pick it.** This is a two-arm sweep at minimum, and
`obs_dim` moves either way, so each candidate gets the `--blind-*` control:
hold the replacement entries at zero and keep them in the vector.

### 1.3 Portals

- `RoomType.Portal = 32`; `C_Room.CanBeTeleportedTo` / `CanBeTeleportedFrom`;
  `C_Rooms.TeleportTo`; `V_Room.ActivateTeleportCorners` called from
  `SetCurrent`. `GenerationParams` has `portalsCount` and `minPortalDistance`.
- `RoomMap.__init__` already links every portal to every other, and
  `from_table` reads the shipped `Portals`. `sim/mapgen.py` places **none**, so
  the `portal` action exists and has never once been legal.
- Implement `portalsCount` and `minPortalDistance` in `mapgen`, and recover
  their per-level values (they are not in `Levels.json`; try `data_xrefs.py` on
  the key names, else record as assumptions).

**Check:** a generated level contains the configured number of portals, no two
closer than `minPortalDistance`, and the `portal` move is legal from a portal
room and reaches another portal.

### 1.4 `BossVision`

Mutation ID 188. `C_BossVisionMutation.OnNewLevel` walks every room reading
`get_type` and `get_state`, revealing the boss. It is worth nothing today
because the agent already knows where the boss is; under fog it becomes a real
effect, and it is the first mutation whose value depends on the fog existing.

**Check:** with the mutation held, the boss room is not `Unknown` at level start;
without it, it is.

**Invalidates:** every agent (`obs_dim` moves). Re-baseline after 1.1-1.4 land
together.

---

# Phase 2 — Fight fidelity

**Why second:** it changes how fights resolve, which is the other half of every
number, and it does not move `obs_dim`, so it can be measured against Phase 1's
baseline without a second interface break.

### 2.1 The escalation schedule

`C_Fight` carries constants the sim ignores entirely:

    _MIDDLE_FIGHT_TIME = 120        _MIDDLE_FIGHT_TIME_FINAL_BOSS = 1200
    _LONG_FIGHT_TIME = 180          _LONG_FIGHT_TIME_FINAL_BOSS = 1320
    _LONG_TIME_WITHOUT_DAMAGE = 8   _LONG_TIME_WITHOUT_DAMAGE_FINAL_BOSS = 12
    DELAY_SINGLE = 0.5              DELAY_KOH = 3.5

with `_enhanceStatusController` over `M_DamageBonusOverTimeStatus` and
`_exhaustStatusController` over `M_DyingDotStatus`. So a long fight is escalated,
first by a damage bonus over time and then by a dying damage-over-time, rather
than being cut off. `sim/assumptions.py` caps a fight at
`max_fight_seconds = 120` and calls it a draw: **the same number wearing the
wrong meaning.**

Read `C_Fight._Fight`, `EnhanceUnit`, `ExhaustUnit`, `AreEffectsFinished` and
`AfterOneTeamDied` for the exact application, then port to `sim/battle.py` and
to the Rust core.

**Check:** a fight still running at 121s has the enhance status applied and one
at 181s the exhaust status; a fight that would have drawn at the cap now
resolves; `diff_core.py` still agrees 60/60 between engines.

### 2.2 Mutation coverage

101 distinct `Name` values ship in `Mutations.json`; 79 have handlers; **28 do
not**. Of 1,094 offers across twelve levels only 259 are implemented, which is
what caps the shelf at +0.13 and therefore caps every mutation result in
`notes/rl.md`.

Work each unimplemented `Name` from `dump.cs`'s `M_<Name>Mutation` properties,
**reading the key names the model class actually uses** — that is the
`vampirism` lesson, and `ExplodeProjectile` and `EventualKnockback` were the same
bug found twice more.

**Check:** per mutation, a test that it changes a fight or a run, not that it
parses. Plus a coverage number in `validate_sim.py` that fails if it regresses.

### 2.3 Audit the existing 79 for the same class of bug

Three of them read the wrong key and did nothing while counting as implemented.
There is no reason to think the other 76 were audited. Cross-check every
registered handler's parameter names against the shipped rows.

**Invalidates:** fight outcomes, so every agent and the baselines, but not
`obs_dim`.

---

# Phase 3 — The decision layer that does not exist

**Why third:** it is the largest decision-shaped hole in the sim and a bigger
build than everything above it combined, so it should land on a sim whose
navigation and combat are already right.

### 3.1 The level-entry dialog

`C_Levels.StartDialog` reads the session mode, takes `Services.Dialogs`, picks
one and `RemoveAt`s it (used once per run), and falls back to
`"NoDialogsPlaceholder"`. `C_ResLog.NewLevel(level, dialog, roomCount, quest)`
logs it with a `NEW_LEVEL` event. `OnDialogsClosed` follows.

`M_Dialog` is not a cutscene:

    float weight       string title      string text
    string[] choices   M_Dialog[][] outcomes
    M_Event[] events   IList<M_UnlockEntry> unlocks
    bool[] enableds    bool[] hiddens    float[] weightSums, float[][] cSums

**The agent currently makes zero choices per level transition where a player
makes one.**

First task is finding the table: it is **not** in `EncryptedMainGroup/DB`. Check
`EncryptedLocalizationsGroup`, `EncryptedDLCGroup` and `EncryptedMainTasksGroup`,
and `Game.json`'s top-level `Dialogs`, `WinDialog` and `LossDialog` keys, which
the sim also does not read.

Then: the weighted draw without replacement, the choice, the branching
`outcomes`, and `M_Event` application.

**Check:** a dialog fires on exactly every level change; the pool is drawn
without replacement across a run; a chosen outcome applies its events; the
placeholder appears when the pool empties.

### 3.2 Quests

`Quests.json` ships eleven (`Sci`, `Reaper`, `Nurgle`, `Pit`, `SecretRoom`,
`Cube`, `Spider`, `Rat`, `Flight`, `Sword`, `Bot`), each with `Doors`,
`ExtraRooms`, `RoomParams` (a named `Layout`, directional priorities,
`guaranteedPositions`), and per-quest fields like `class` and `haterName`.
`C_Rooms.CreateQuest` resolves the name through `EnumUtils.ToEnum<Quest>` and
reflection over `"M_" + Name` and `"C_" + Name + "Quest"`, reads an `"outcomes"`
array and sets `M_Room.questType`. `QuestStatus` is
`Uninitialized / Accepted / Declined / Acquired / Finished`.

The level rows carry a per-level shortlist: levels 2, 6 and 10 only
(`Sci,Pit,Cube,Rat`; `Reaper,SecretRoom,Spider`; `Nurgle,Flight,Sword`).
`GenerationParams` has `questCount`, `extraQuestRoomCount`,
`questRoomAllowedDoors`, `questRoomGuaranteedDoors`.

Open question this phase settles: **whether a quest room fights.** There is no
`Quest` row in `EnemyPacks.json`, which suggests not, but that is inference.

**Check:** a level with a `Quest` shortlist generates exactly one quest room of a
listed type with its layout and door constraints; the status lifecycle advances;
each quest's own effect fires.

**Invalidates:** the action space and the observation, so everything.

---

# Phase 4 — The remaining room types and generation

- **Secret rooms.** `RoomType.Secret = 32768`, `M_Room.activatesSecret` /
  `secretIsActive`, `C_Rooms.MaybeActivateSecretRoom`, the constant
  `roomsToActivateSecret = 2`, `secretRoomCount` in `GenerationParams`, and
  `SecretRoom: true` on level 7 only. Also `C_SecretRoomQuest`.
- **`PermanentShop = 16`, `TalentShop = 4160`, `ConsumableShop = 16384`,
  `QuestExtra = 1024`, `FinalBoss = 2048`.** None modelled. `TalentShop` is what
  builds `C_ConsumableShop`, so consumables hang off it.
- **Consumables.** `StatShops` is 0 across all eight Default chips, so they
  cannot fire there, but `KingOfTheHill`, `Arcade` and `Tasks` have not been
  checked. Implement the mechanism, then record where it fires.
- **`RerollShrines` against `Shrines`.** The level rows have `Shrines: 1` at
  level 1 and 0 after, while `RerollShrines: 1` from level 2 on. The sim
  conflates them; the shrine family after level 1 is the **reroll** shrine.
- **The three unrecovered `GenerationParams` values.** Retry with
  `tools/data_xrefs.py` now that data cross-references are possible; if they are
  genuinely absent, keep them in `assumptions.py` with the evidence.

---

# Phase 5 — Modes and chips

`load_ruleset(mode, chip)` already layers `Common` then the mode then the chip,
so the plumbing exists. What is missing is the mechanics each turns on.

- **Modes:** `Default`, `KingOfTheHill`, `Arcade`, `Tasks`.
- **Default's chips:** `default`, `easy`, `hard`, `hunger`, `mutagens`, `crazy`,
  `shortpvp`, `arcade`.

Some are already just data: `hunger` shrinks the room counts (6-7 and 9-11
against 7 and 10-12), `shortpvp` is seven levels of 5-6 rooms, `crazy` is **two**
levels, one of 3 rooms and one of 100. Others will need mechanics:
`KingOfTheHill` has its own directory and its own fight delay constant
(`DELAY_KOH = 3.5`), and `shortpvp` implies a PvP resolution path.

**Check:** every mode and chip loads under `strict=True`, and each one's own
check suite passes. A mode that loads but has no mechanics is a mode that is not
implemented, and the coverage report should say so rather than counting it.

---

# Phase 6 — Closing it out

- **A coverage report**, generated rather than asserted: per subsystem, what the
  shipped data contains against what the sim implements, with the unimplemented
  named. Mutations already have a number (259 of 1,094 offers); rooms, skills,
  quests, dialogs, modes and consumables need the same.
- **An `assumptions.py` audit**: every entry either resolved against the binary
  or restated with why it cannot be.
- **A check per feature**, per the rule at the top. The coverage report should
  fail if a feature is registered without one.

---

## Ordering summary

| phase | what | breaks | re-baseline |
|---|---|---|---|
| 1 | fog, portals, `BossVision` | `obs_dim` | yes |
| 2 | fight escalation, 28 mutations, audit the 79 | fight outcomes | yes |
| 3 | dialogs, 11 quests | actions + obs | yes |
| 4 | secret, remaining shop types, shrine split, generation params | maps | yes |
| 5 | 4 modes, 8 chips | new rulesets | per mode |
| 6 | coverage report, assumptions audit, checks | nothing | no |

RL results are worth taking only after a phase closes and the baseline is
re-measured. Training against an interface that is about to move is the one
thing this project has already wasted the most time on.
