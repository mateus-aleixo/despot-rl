"""Quests: the `C_Quest` controllers and the dialogue trees they run.

Three levels carry a quest shortlist in `Levels.json` (`Quest`): level 2 has
`Sci,Pit,Cube,Rat`, level 6 `Reaper,SecretRoom,Spider,Bot` and level 10
`Nurgle,Flight,Sword`. `DataLoaders.LevelData.Load` turns each shortlist into a
`DynamicWeightedList<string>` (factor 2), keeps it in a static dictionary keyed
by the shortlist's text, and draws one name per level into
`M_LevelData.randomizedQuestNames`. The list is static, so within one launch of
the game it biases against a quest drawn in an earlier run; runs here are
independent, and the draw is uniform, the same reading the item shop's
`DynamicWeightedList` gets (see `sim/run.py`).

## Where a quest lives on the map

`C_Rooms.Generate` copies the quest's row into `GenerationParams`:
`questCount = 1`, `extraQuestRoomCount = ExtraRooms`, `secretRoomCount = the
row's SecretRoom flag + SecretRooms`, `questRoomAllowedDoors = Doors`,
`questRoomGuaranteedDoors = GuaranteedDoors` and `quest` as the `Quest` enum.
`LevelGenerator.PositionRooms` places the rooms right after the start, before
the shops; see `sim/mapgen.py`.

## What the rooms hold

`C_GameMode.GenerateRoom` asks the quest for the room's `RoomParams`
(`EnumerateRooms` names the quest room `room` and each extra room `extra`). A
params block with no `Packs` gives the room its layout and nothing to fight;
one with `Packs` fights exactly that pack in that layout. So the quest room
never fights, the extra rooms of Sci, Cube, Rat and SecretRoom do, and those of
Flight and Bot do not. `C_Rooms.CalculatePower` pays a quest room nothing and
an extra room the plain-room share (nothing for Flight and Bot), and counts
neither in its weight total.

## The dialogs

The quest dialogs are not in `dialogs.json`. They are NodeCanvas
`DialogueTree` graphs bound to the quest prefabs in `sharedassets2.assets` and
`sharedassets3.assets`; `tools/dump_dialogue_trees.py` writes them to
`data/extracted/dialogue_trees/`. `V_Rooms.MoveCamera` launches the room's
`V_Quest` every time the squad arrives in it, and `V_Quest.Launch` runs the
dialog `GetDialogIndex(room)` names, or none when that is -1. Only the quest
room carries the prefab (`TryGetPrefabName` answers for `model.room` only), so
an extra room's dialog is started by the controller, after its fight.

A tree is walked the way NodeCanvas walks one: an action node runs its action
list and goes to its child, a condition node goes to its first child when the
condition holds and its second when not, a statement goes to its child, and a
multiple-choice node waits for the player. A node with no child ends the
dialog. `ApplyOutcome(i)` applies the row's `outcomes[i - 1]` through
`C_Dialogs.ApplyEvents`, the same events the level-entry dialogs use.

## Status

`QuestStatus` is Uninitialized 0, Accepted 1, Declined 2, Acquired 4,
Finished 8, Initialized 16. `CheckQuestStatus` is an equality test, and an
unset `status` parameter is 0.
"""
from __future__ import annotations

import collections
import json
import pathlib
from dataclasses import dataclass, field

TREES = pathlib.Path(__file__).resolve().parent.parent / "data" / "extracted" / "dialogue_trees"

UNINITIALIZED, ACCEPTED, DECLINED, ACQUIRED, FINISHED, INITIALIZED = 0, 1, 2, 4, 8, 16

# The `Quest` enum, which `ChooseQuestRoom` and `ChooseQuestExtraRoom` compare
# against `Flight` (9) and `CalculatePower` against Flight and `Bot` (11).
QUEST_ENUM = {"Sci": 1, "Reaper": 2, "Nurgle": 3, "Pit": 4, "SecretRoom": 5,
              "Cube": 6, "Spider": 7, "Rat": 8, "Flight": 9, "Sword": 10, "Bot": 11}

# A tree can loop (Pit's revisit branch leads back to the choice it came from),
# so a walk that passes this many nodes without a choice stops where it is.
MAX_NODES_PER_STEP = 200

_TREES: dict | None = None


def shortlist(level_row: dict) -> list[str]:
    """The level's quest names, in the row's order."""
    text = level_row.get("Quest") or ""
    return [n.strip() for n in str(text).split(",") if n.strip()]


def draw_quest(level_row: dict, rng) -> str | None:
    """`LevelData.Load`'s draw for one level: uniform over the shortlist."""
    names = shortlist(level_row)
    if not names:
        return None
    return names[0] if len(names) == 1 else rng.choice(names)


def quest_spec(tables: dict, name: str | None) -> dict | None:
    """The quest's row with its `Name`, the shape `C_Rooms.Generate` reads.

    `C_Levels.CreateQuestModel` writes the row into the level data under
    `Quest` and sets `Name`; `Overrides` is the second level of a quest with
    `AdditionalLevels`, which no level-2 quest has.
    """
    if not name:
        return None
    row = (tables.get("Quests") or {}).get(name)
    return dict(row, Name=name) if row else None


def trees() -> dict[str, dict]:
    """Every dumped dialogue tree, by file stem. Empty when none were dumped."""
    global _TREES
    if _TREES is None:
        _TREES = {}
        if TREES.exists():
            for p in sorted(TREES.glob("*.json")):
                if p.name != "index.json":
                    _TREES[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return _TREES


def _short(type_name: str) -> str:
    """`NodeCanvas.DialogueTrees.ActionNode` -> `ActionNode`, and the generic
    `CallFunction<...>` actions down to their own name."""
    head = type_name.split("`")[0].split(",")[0]
    return head.split(".")[-1]


def decision_nodes() -> list[tuple[str, str]]:
    """Every multiple-choice node of every quest tree, as (stem, node id).

    A fixed order, so an index into it names one node across runs: the
    environment one-hot encodes the node a quest dialog is waiting on.
    """
    out = []
    for stem, g in sorted(trees().items()):
        for n in g.get("nodes", []):
            if _short(n["$type"]) == "MultipleChoiceNode":
                out.append((stem, n["$id"]))
    return out


_INDEX: dict[str, tuple[dict, dict]] = {}


def tree_index(stem: str) -> tuple[dict, dict]:
    """A tree's nodes by id and each node's children in connection order."""
    hit = _INDEX.get(stem)
    if hit is None:
        g = trees()[stem]
        nodes = {n["$id"]: n for n in g["nodes"]}
        children: dict[str, list[str]] = collections.defaultdict(list)
        for c in g.get("connections", []):
            children[c["_sourceNode"]["$ref"]].append(c["_targetNode"]["$ref"])
        hit = _INDEX[stem] = (nodes, dict(children))
    return hit


class TreeRun:
    """One quest dialog in progress: which tree, and which node it waits on.

    It holds no part of the tree itself, so a copy of the run state (a DejaVu
    save) copies two strings rather than the graph.
    """

    def __init__(self, quest: "Quest", stem: str):
        self.quest = quest
        self.stem = stem
        self.at: str | None = "0"
        self.closed = False

    @property
    def nodes(self) -> dict:
        return tree_index(self.stem)[0]

    @property
    def children(self) -> dict:
        return tree_index(self.stem)[1]

    @property
    def node(self) -> dict | None:
        return self.nodes.get(self.at) if self.at is not None else None

    def options(self, st) -> list[int]:
        """The choices the waiting node shows, by index."""
        n = self.node
        if n is None or _short(n["$type"]) != "MultipleChoiceNode":
            return []
        out = []
        for i, choice in enumerate(n.get("availableChoices") or []):
            cond = choice.get("condition")
            text = ((choice.get("statement") or {}).get("_text")) or ""
            if ((cond is None or self.quest.check(st, cond))
                    and self.quest.option_enabled(st, text)):
                out.append(i)
        return out

    def _next(self, k: int = 0) -> None:
        kids = self.children.get(self.at, [])
        self.at = kids[k] if k < len(kids) else None
        if self.at is None:
            self.closed = True

    def advance(self, st) -> None:
        """Walk to the next choice, or to the end."""
        for _ in range(MAX_NODES_PER_STEP):
            n = self.node
            if n is None:
                self.closed = True
                return
            kind = _short(n["$type"])
            if kind == "MultipleChoiceNode":
                if self.options(st):
                    return
                self.closed = True
                return
            if kind == "ActionNode":
                act = n.get("_action") or {}
                for a in act.get("actions", [act]):
                    if a and not a.get("_isDisabled"):
                        self.quest.act(st, a)
                    if st.finished:
                        self.closed = True
                        return
                self._next()
            elif kind == "ConditionNode":
                self._next(0 if self.quest.check(st, n.get("_condition") or {}) else 1)
            else:
                self._next()
            if self.closed:
                return
        self.closed = True

    def choose(self, st, index: int) -> None:
        if index not in self.options(st):
            raise ValueError(f"choice {index} is not offered at {self.stem}:{self.at}")
        self._next(index)
        self.advance(st)

    def preview(self, st, index: int) -> list[tuple[str, object]]:
        """What choosing `index` would do, up to the next choice or the end,
        without doing it: ("outcome", i), ("call", name) and ("status", s) in
        the order the walk meets them. Conditions are read on the state as it
        stands, which is exact up to the first action that changes the squad."""
        kids = self.children.get(self.at, [])
        at = kids[index] if index < len(kids) else None
        out: list[tuple[str, object]] = []
        for _ in range(MAX_NODES_PER_STEP):
            n = self.nodes.get(at) if at is not None else None
            if n is None:
                break
            kind = _short(n["$type"])
            if kind == "MultipleChoiceNode":
                break
            nxt = 0
            if kind == "ActionNode":
                act = n.get("_action") or {}
                for a in act.get("actions", [act]):
                    if not a or a.get("_isDisabled"):
                        continue
                    k = _short(a.get("$type", ""))
                    if k == "ApplyOutcome":
                        out.append(("outcome", int(_value(a.get("outcome"), 0))))
                    elif k == "SetStatus":
                        out.append(("status", int(_value(a.get("status"), 0))))
                    elif "function" in a:
                        out.append(("call", str(_value(a.get("function"), ""))))
            elif kind == "ConditionNode":
                nxt = 0 if self.quest.check(st, n.get("_condition") or {}) else 1
            kids = self.children.get(at, [])
            at = kids[nxt] if nxt < len(kids) else None
        return out


@dataclass
class Quest:
    """`C_Quest<T>`: the status, the rooms, and the hooks the run calls."""
    name: str
    spec: dict
    room: str | None = None
    extras: list = field(default_factory=list)
    status: int = UNINITIALIZED
    # Quest props in the inventory (`cube-part`, `rat-flute`): the sim has no
    # inventory, and they do nothing but mark the quest's progress.
    items: set = field(default_factory=set)
    # Levels the quest still runs into after its own (`AdditionalLevels`).
    levels_left: int = 0

    # -- the run layer's hooks --------------------------------------------
    def arrive(self, st, rid: str, fought: bool, won: bool) -> str | None:
        """The squad has arrived in `rid` (after its fight, when it had one).
        Returns the tree to run, if any."""
        if rid == self.room:
            return self.room_dialog(st)
        return None

    def room_dialog(self, st) -> str | None:
        """`GetDialogIndex(room)` for the quest room, as a tree stem."""
        return None

    def killed(self, st, n: int) -> None:
        """`n` enemies died in a fight the squad just fought."""

    def extra_enemies(self, st, room) -> list:
        """Units the quest adds to `room`'s enemy team, as (class, level)
        pairs: `OnEnemyTeamCreation`."""
        return []

    def dialog_ended(self, st) -> None:
        """A quest dialog has just closed; what the controller does after."""

    def set_status(self, st, status: int) -> None:
        self.status = int(status)
        self.on_status(st, self.status)

    def on_status(self, st, status: int) -> None:
        """`C_Quest<T>.OnStatus` and the quest's own override."""

    def marked(self, st) -> list:
        """The rooms the quest points the squad at, as far as the map shows
        them: its extra rooms by default, the runaway's room for the Reaper and
        the Bot. Read by the environment and by the baseline's follow mode."""
        from .run import UNKNOWN
        return [r for r in self.extras if st.rooms.rooms[r].state != UNKNOWN]

    def reveal_extras(self, st) -> None:
        """`M_Room.state = Unexplored` on the extra rooms, which the walk-past
        reveal never touches (`NEVER_REVEALED`): "I've marked it on your map"."""
        from .run import UNEXPLORED, UNKNOWN
        for rid in self.extras:
            room = st.rooms.rooms.get(rid)
            if room is not None and room.state == UNKNOWN:
                room.state = UNEXPLORED
        st.rooms._refresh_known()

    # -- the tree's actions and conditions ---------------------------------
    def act(self, st, action: dict) -> None:
        kind = _short(action.get("$type", ""))
        if kind == "SetStatus":
            self.set_status(st, _value(action.get("status"), 0))
        elif kind == "ApplyOutcome":
            self.apply_outcome(st, int(_value(action.get("outcome"), 0)))
        elif "function" in action:
            self.call(st, str(_value(action.get("function"), "")),
                      _value(action.get("param"), None))
        elif kind == "CheckEndGame":
            if not st.squad:
                st.finished, st.won = True, False
        # Everything else is presentation: camera, sound, animation, input.

    def check(self, st, cond: dict) -> bool:
        kind = _short(cond.get("$type", ""))
        if kind == "CheckQuestStatus":
            ok = self.status == int(_value(cond.get("status"), 0))
        elif kind == "CheckCondition":
            ok = self.condition(st, str(_value(cond.get("function"), "")))
        elif kind == "MoreThanOneUnit":
            ok = len(st.squad) > 1
        elif kind == "ChipIs":
            ok = st.chip == str(_value(cond.get("value"), ""))
        elif kind == "DeselectNovices":
            # `OnCheck` starts a coroutine that clears the selection on screen
            # and returns true.
            ok = True
        else:
            ok = self.condition(st, kind)
        return (not ok) if cond.get("_invert") else ok

    def condition(self, st, name: str) -> bool:
        """`CheckCondition(name)` and the quest's own condition tasks."""
        raise NotImplementedError(f"{self.name}: condition {name!r}")

    def option_enabled(self, st, text: str) -> bool:
        """`C_Quest<T>.IsOptionEnabled(text)`: true unless the quest overrides
        it, which only Bot does."""
        return True

    def call(self, st, fn: str, param) -> None:
        """`CallFunction(name[, param])`. Functions that only move actors,
        cameras and animations are presentation and do nothing here."""

    def apply_outcome(self, st, index: int) -> None:
        """`ApplyOutcome(i)`: the row's `outcomes[i - 1]` through the dialog
        events."""
        outs = self.spec.get("outcomes") or []
        if not 1 <= index <= len(outs):
            return
        from .dialogs import Dialog, Engine
        engine, d = Engine(st), Dialog(name=f"quest:{self.name}", path="#")
        for e in outs[index - 1]:
            engine._apply(d, e)


def _value(param, default):
    """A serialized `BBParameter`: `{"_value": x}`, or `{}` for the default."""
    if isinstance(param, dict):
        return param.get("_value", default)
    return default if param is None else param


def is_novice(h) -> bool:
    """`classValue == Novice`: a human holding nothing."""
    return h.item is None and getattr(h, "cls", None) is None


@dataclass
class PitQuest(Quest):
    """`C_PitQuest`: the Pit wants five Novices, or a few tokens.

    `GetDialogIndex` is 0 while the status is Uninitialized or Accepted and -1
    after. `HasFiveNovices` counts units of class Novice and asks for five;
    `HasSomeoneExceptFiveNovices` holds when there are more than five Novices
    or anyone who is not one. `KillNovices` kills the units standing on the
    first five of the room's `guaranteedPositions`, then `CheckEndGame`.
    Outcome 1 is mutation 306, outcome 2 is 30 gold, reached by throwing
    tokens, which every branch but the jump offers.
    """

    TREE = "Room-pit-quest.Dialog_Beginning"

    def room_dialog(self, st):
        return self.TREE if self.status <= ACCEPTED else None

    def condition(self, st, name):
        novices = sum(1 for h in st.squad if is_novice(h))
        if name == "HasFiveNovices":
            return novices >= 5
        if name == "HasSomeoneExceptFiveNovices":
            return novices > 5 or novices != len(st.squad)
        return super().condition(st, name)

    def call(self, st, fn, param):
        if fn == "KillNovices":
            # Which five stand on the death cells is the dungeon placement's
            # business, which the sim does not model; the first five Novices in
            # squad order go. See `sim/assumptions.py`.
            doomed = [h for h in st.squad if is_novice(h)][:5]
            st.squad = [h for h in st.squad if all(h is not d for d in doomed)]
            if not st.squad:
                st.finished, st.won = True, False


@dataclass
class CubeQuest(Quest):
    """`C_CubeQuest`: fetch the cube part from the extra room.

    `GetDialogIndex` is 0 for Uninitialized and Acquired, and in Accepted 0
    only in the extra room, whose dialog the controller starts. `OnCurrentRoom`
    is subscribed from construction and acts only in the extra room while the
    quest is Accepted: after a fight won there, or at once when the room was
    already won (`fightResult == 1`), the extra dialog sets Acquired.
    `OnStatus(Accepted)` reveals the extra room, `OnStatus(Acquired)` adds the
    `cube-part` item and `OnStatus(Finished)` removes it. Outcome 1 is 25 gold,
    outcome 2 mutation 328.
    """

    MAIN, EXTRA = "Cube-quest-main.Dialog", "Cube-quest-extra.Dialog"

    def room_dialog(self, st):
        return self.MAIN if self.status in (UNINITIALIZED, ACQUIRED) else None

    def arrive(self, st, rid, fought, won):
        if rid in self.extras and self.status == ACCEPTED:
            room = st.rooms.rooms[rid]
            if room.cleared:
                return self.EXTRA
            return None
        return super().arrive(st, rid, fought, won)

    def on_status(self, st, status):
        if status == ACCEPTED:
            self.reveal_extras(st)
        elif status == ACQUIRED:
            self.items.add("cube-part")
        elif status == FINISHED:
            self.items.discard("cube-part")


@dataclass
class SciQuest(Quest):
    """`C_SciQuest`: bring the scientist a zombie.

    `GetDialogIndex` in the quest room: 0 (the beginning) while Uninitialized,
    and in Acquired 1 when the zombie is alive and 2 when it is dead or gone;
    -1 otherwise. In Accepted it is 0 only in the extra room. Unlike Cube's,
    `OnCurrentRoom` has no branch for an extra room won before the quest was
    accepted: it waits for `AfterFight` in that room, so a lab cleared first
    never hands over its zombie.

    After the lab fight `ChooseActors_AfterFight` picks the dialog by team
    size: with more than one unit, `Quest-two-units` runs `ReplaceUnit`, which
    turns `FindWeakestUnit` into a level-1 `Zombie` (`C_Player.DownGradeUnit`,
    class 17, level 1); with one, `Quest-one-unit` runs `CreateZombie`, a new
    level-1 Zombie. Both set Acquired. The ending with the zombie removes it
    (`RemoveZombie`) and pays outcome 1 (mutation 218) or 2 (25 gold).
    """

    BEGIN = "Room-sci-quest.Dialog_Beginning"
    WITH = "Room-sci-quest.Dialog_EndingWithZombie"
    WITHOUT = "Room-sci-quest.Dialog_EndingWithoutZombie"
    ONE, TWO = "Quest-one-unit.Dialog", "Quest-two-units.Dialog"

    zombie: object = None

    def room_dialog(self, st):
        if self.status == UNINITIALIZED:
            return self.BEGIN
        if self.status == ACQUIRED:
            alive = self.zombie is not None and any(h is self.zombie for h in st.squad)
            return self.WITH if alive else self.WITHOUT
        return None

    def arrive(self, st, rid, fought, won):
        if rid in self.extras and self.status == ACCEPTED and fought and won:
            return self.TWO if len(st.squad) > 1 else self.ONE
        return super().arrive(st, rid, fought, won)

    def on_status(self, st, status):
        if status == ACCEPTED:
            self.reveal_extras(st)

    def call(self, st, fn, param):
        if fn == "ReplaceUnit":
            weakest = find_weakest_unit(st)
            if weakest is not None:
                i = next(k for k, h in enumerate(st.squad) if h is weakest)
                self.zombie = st.make_companion("Zombie", 1)
                st.squad[i] = self.zombie
        elif fn == "CreateZombie":
            self.zombie = st.make_companion("Zombie", 1)
            st.squad.append(self.zombie)
        elif fn == "RemoveZombie":
            st.squad = [h for h in st.squad if h is not self.zombie]


def find_weakest_unit(st):
    """`C_SciQuest.FindWeakestUnit`: a Novice before anyone armed; Novices by
    level then health, the armed by item quality then cost. Companions are not
    candidates (the game's list of eligible classes holds the human ones)."""
    from .data import items_by_name
    rows = items_by_name(st.tables)
    humans = [h for h in st.squad if getattr(h, "cls", None) is None]
    if not humans:
        return None

    def key(h):
        if h.item is None:
            return (0, h.level, h.hp)
        row = rows.get(h.item) or {}
        return (1, int(row.get("Quality") or 1), float(row.get("Cost") or 0.0))

    return min(humans, key=key)


@dataclass
class RatQuest(Quest):
    """`C_RatQuest`: take the flute, win the rat back, return it alive.

    `GetDialogIndex` is 0 for Uninitialized, Accepted and Acquired.
    `OnStatus(Accepted)` reveals the extra room, adds `rat-flute` and
    subscribes to that room's fights: a win there, or arriving in it already
    won (`MaybeCleanRoom`), adds the room's rat to the team (`AddToTeam`, class
    `Rat` 79 at its own level, then status Acquired). Back with the trumpeter,
    `RemoveFlute`, then `HasRat` (the rat returned, or a Rat in the team):
    `RemoveRat` and outcome 1 (mutation 233) or 2 (25 gold); without the rat,
    no reward. Either way Finished.
    """

    MAIN, EXTRA = "Rat-quest-main.Dialog", "Rat-quest-extra.Dialog"

    rat: object = None
    returned: bool = False

    def room_dialog(self, st):
        return self.MAIN if self.status in (UNINITIALIZED, ACCEPTED, ACQUIRED) else None

    def arrive(self, st, rid, fought, won):
        if rid in self.extras and self.status == ACCEPTED and st.rooms.rooms[rid].cleared:
            self.rat = st.make_companion("Rat", 1)
            st.squad.append(self.rat)
            self.set_status(st, ACQUIRED)
            return self.EXTRA
        return super().arrive(st, rid, fought, won)

    def on_status(self, st, status):
        if status == ACCEPTED:
            self.reveal_extras(st)
            self.items.add("rat-flute")

    def condition(self, st, name):
        if name == "HasRat":
            return self.returned or any(getattr(h, "cls", None) == "Rat" for h in st.squad)
        return super().condition(st, name)

    def call(self, st, fn, param):
        if fn == "RemoveFlute":
            self.items.discard("rat-flute")
        elif fn == "RemoveRat":
            before = len(st.squad)
            st.squad = [h for h in st.squad if getattr(h, "cls", None) != "Rat"]
            self.returned = self.returned or len(st.squad) < before


def _seen(st, rid: str | None) -> list:
    """`[rid]` when the room is on the map, else nothing."""
    from .run import UNKNOWN
    return [rid] if rid is not None and st.rooms.rooms[rid].state != UNKNOWN else []


def _flee(st, current: str, prev: str | None, forbidden: str | None) -> str | None:
    """`ChooseNextRoom(current, prev)` of the Reaper's hater and the Bot: a
    uniform draw (`RND.Element`) over the rooms beside `current` other than the
    one the squad came from and the quest room; None when there is none."""
    rooms = [n for n in st.rooms.doors(current) if n != prev and n != forbidden]
    return st.rng.choice(rooms) if rooms else None


@dataclass
class ReaperQuest(Quest):
    """`C_ReaperQuest`: catch the hater the Reaper wants dead.

    `GetDialogIndex` in the quest room is 0 (the beginning) while
    Uninitialized and 1 (the ending) once Acquired. The beginning spawns the
    hater there and `Run` sends him through a door at once (`ChooseNextRoom`,
    `movesMade` to 1). While the quest is Accepted, the squad arriving in the
    hater's room (`OnRoomChange`, which also records the room it came from as
    `prevRoom`) makes him run again, while `movesMade < maxMoves` (5) and a
    room qualifies; otherwise he stays and his dialog opens there. He offers
    50 gold (outcome 1) to be let go, which finishes the quest; refusing sets
    Acquired and starts a fight with him (`PrepareToFight`), and the Reaper
    pays mutation 230 (outcome 2) for it. The hater is a level-1 unit of the
    row's `class`, a Novice.
    """

    BEGIN, END = "Room-reaper-quest.Dialog_Beginning", "Room-reaper-quest.Dialog_Ending"
    HATER = "Quest-hater-dialog.Dialog"

    hater_room: str | None = None
    moves: int = 0

    @property
    def max_moves(self) -> int:
        return int(self.spec.get("maxMoves") or 0)

    def marked(self, st):
        return _seen(st, self.hater_room)

    def room_dialog(self, st):
        if self.status == UNINITIALIZED:
            return self.BEGIN
        if self.status == ACQUIRED:
            return self.END
        return None

    def arrive(self, st, rid, fought, won):
        if self.status == ACCEPTED and rid == self.hater_room and rid != self.room:
            nxt = (_flee(st, rid, st.prev_room, self.room)
                   if self.moves < self.max_moves else None)
            if nxt is not None:
                self.hater_room = nxt
                self.moves += 1
                return None
            return self.HATER
        return super().arrive(st, rid, fought, won)

    def call(self, st, fn, param):
        if fn == "Run":
            nxt = _flee(st, self.room, st.prev_room, self.room)
            if nxt is not None:
                self.hater_room = nxt
                self.moves += 1
        elif fn == "RemoveHater":
            self.hater_room = None
        elif fn == "PrepareToFight":
            st.quest_fight([(str(self.spec.get("class") or "Novice"), 1)])
            self.hater_room = None


@dataclass
class SpiderQuest(Quest):
    """`C_SpiderQuest`: post the monster's leaflets.

    The monster grabs a unit and throws it back out (a knockback, no damage).
    `OnStatus(Accepted)` hands over `leafletCount` (3) leaflets, and
    `OnCurrentRoom` posts one in every room the squad then enters that is not
    the quest room and has none yet; the last sets Acquired. Back in the quest
    room: mutation 346 (outcome 1) or 50 gold (outcome 2).
    """

    TREE = "Spider-quest.Dialog"

    posted: list = field(default_factory=list)

    def room_dialog(self, st):
        return self.TREE if self.status in (UNINITIALIZED, ACCEPTED, ACQUIRED) else None

    def arrive(self, st, rid, fought, won):
        if self.status == ACCEPTED and rid != self.room and rid not in self.posted:
            self.posted.append(rid)
            if len(self.posted) >= int(self.spec.get("leafletCount") or 0):
                self.set_status(st, ACQUIRED)
            return None
        return super().arrive(st, rid, fought, won)



@dataclass
class BotQuest(Quest):
    """`C_BotQuest`: bring the gamer back his bot.

    `GetDialogIndex` in the quest room is 0 while Uninitialized and once
    Acquired. `OnStatus(Accepted)` reveals the extra room, where the bot
    starts (`currentBotRoom`). Arriving in the bot's room while Accepted opens
    its dialog the first time: threaten it and it runs (`Run`, as the Reaper's
    hater does, `maxMoves` 3), pay `goldLoss` (outcome 3, 5 gold; the option is
    shown only when the team has that much, `IsOptionEnabled`) and it joins
    the team (`SubscribeToBot`, Acquired), or say there is no money. Once it
    runs, every arrival in its room sends it on while it can move, then opens
    `bot-before-fight`: the squad fights it (`PrepareToFight`), and beaten it
    changes sides (`OnBotDeath` recreates it in the player's team) and
    `_OnEndFight` sets Acquired. Back with the gamer: `HasBot` (returned, or a
    `Bot` in the team), then `RemoveBot` and mutation 22 (outcome 1) or 50
    gold (outcome 2); without it, nothing.
    """

    MAIN = "Room-bot-quest.Dialog"
    MEET, BEFORE, AFTER = "Bot-dialog.Dialog", "Bot-before-fight.Dialog", "Bot-after-fight.Dialog"

    bot_room: str | None = None
    moves: int = 0
    running: bool = False
    returned: bool = False

    def marked(self, st):
        return _seen(st, self.bot_room)

    @property
    def gold_loss(self) -> float:
        outs = self.spec.get("outcomes") or []
        for e in (outs[2] if len(outs) > 2 else []):
            if e.get("className") == "MinusGold":
                return float((e.get("parameters") or {}).get("amount") or 0.0)
        return 0.0

    def room_dialog(self, st):
        return self.MAIN if self.status in (UNINITIALIZED, ACQUIRED) else None

    def on_status(self, st, status):
        if status == ACCEPTED:
            self.reveal_extras(st)
            if self.bot_room is None and self.extras:
                self.bot_room = self.extras[0]

    def arrive(self, st, rid, fought, won):
        if self.status == ACCEPTED and rid == self.bot_room and rid != self.room:
            if not self.running:
                return self.MEET
            nxt = (_flee(st, rid, st.prev_room, self.room)
                   if self.moves < int(self.spec.get("maxMoves") or 0) else None)
            if nxt is not None:
                self.bot_room = nxt
                self.moves += 1
                return None
            return self.BEFORE
        return super().arrive(st, rid, fought, won)

    def option_enabled(self, st, text):
        if text == "Quest.Bot.Meet.3.2":
            return st.gold >= self.gold_loss
        return True

    def condition(self, st, name):
        if name == "HasBot":
            return self.returned or any(getattr(h, "cls", None) == "Bot" for h in st.squad)
        return super().condition(st, name)

    def call(self, st, fn, param):
        if fn == "Run":
            self.running = True
            nxt = _flee(st, self.bot_room, st.prev_room, self.room)
            if nxt is not None:
                self.bot_room = nxt
                self.moves += 1
        elif fn == "SubscribeToBot":
            self._join(st)
        elif fn == "PrepareToFight":
            res = st.quest_fight([(str(self.spec.get("class") or "Bot"), 1)])
            if res.get("won") and not st.finished:
                self._join(st)
                self.set_status(st, ACQUIRED)
                st.queue_quest_dialog(self.AFTER)
        elif fn == "RemoveBot":
            before = len(st.squad)
            st.squad = [h for h in st.squad if getattr(h, "cls", None) != "Bot"]
            self.returned = self.returned or len(st.squad) < before

    def _join(self, st):
        st.squad.append(st.make_companion(str(self.spec.get("class") or "Bot"), 1))
        self.bot_room = None


@dataclass
class NurgleQuest(Quest):
    """`C_NurgleQuest`: five cabbage leaves, on this level or the next.

    While Accepted (`OnStatus` subscribes to every new enemy team's dead
    units) every enemy that dies adds a `plant-leaf`. `GetDialogIndex` counts
    the leaves against `threshold` (5): on the level the quest started, the
    beginning while Uninitialized, then the ending or "not finished"; with
    `AdditionalLevels` 1 the model carries into the next level, whose map gets
    the room again from `Overrides` (`nurgle-quest-2`), with its own
    not-finished, ending and finished dialogs. The ending pays 35 gold
    (outcome 1) or mutation 302 (outcome 2) and `RemoveItems` takes the
    leaves.
    """

    BEGIN = "Room-nurgle-quest.Dialog_Beginning"
    NOT, END = "Room-nurgle-quest.Dialog_NotFinished", "Room-nurgle-quest.Dialog_Ending"
    NOT2, END2 = "Room-nurgle-quest-2.Dialog_NotFinished", "Room-nurgle-quest-2.Dialog_Ending"
    FIN2 = "Room-nurgle-quest-2.Dialog_Finished"

    leaves: int = 0
    started: int | None = None

    def room_dialog(self, st):
        enough = self.leaves >= int(self.spec.get("threshold") or 0)
        if self.started is None or st.level == self.started:
            if self.status == UNINITIALIZED:
                return self.BEGIN
            if self.status == ACCEPTED:
                return self.END if enough else self.NOT
            return None
        if self.status == ACCEPTED:
            return self.END2 if enough else self.NOT2
        if self.status == FINISHED:
            return self.FIN2
        return None

    def on_status(self, st, status):
        if status == ACCEPTED and self.started is None:
            self.started = st.level

    def killed(self, st, n):
        if self.status == ACCEPTED:
            self.leaves += n

    def call(self, st, fn, param):
        if fn == "RemoveItems":
            self.leaves = 0


@dataclass
class FlightQuest(Quest):
    """`C_FlightQuest`: find Mad Mike, on foot or by cannon.

    `GetDialogIndex` is 0 for Uninitialized, Accepted and Acquired. Walking
    sets Accepted and reveals the extra room, and arriving there plays its
    dialog. Flying (`MoveToCannon`) leaves the status where it is, shoots the
    squad into the extra room and takes `hpLossPercent` (10) percent of each
    unit's health on landing (`WaitForArrival`), and the extra dialog plays at
    once with `ChoseToFly`. Either way the dialog sets Acquired and hands over
    `mik-helmet`; back in the quest room, 35 gold (outcome 1) or mutation 237
    (outcome 2).
    """

    MAIN, EXTRA = "Flight-quest-main.Dialog", "Flight-quest-extra-dialog.Dialog"

    chose_to_fly: bool = False
    flying: bool = False

    def room_dialog(self, st):
        return self.MAIN if self.status in (UNINITIALIZED, ACCEPTED, ACQUIRED) else None

    def arrive(self, st, rid, fought, won):
        if rid in self.extras and self.status == ACCEPTED:
            return self.EXTRA
        return super().arrive(st, rid, fought, won)

    def on_status(self, st, status):
        if status == ACCEPTED:
            self.reveal_extras(st)
        elif status == ACQUIRED:
            self.items.add("mik-helmet")
        elif status == FINISHED:
            self.items.discard("mik-helmet")

    def condition(self, st, name):
        if name == "ChoseToFly":
            return self.chose_to_fly
        return super().condition(st, name)

    def call(self, st, fn, param):
        if fn == "MoveToCannon":
            self.chose_to_fly = self.flying = True

    def dialog_ended(self, st):
        if self.flying and self.extras:
            self.flying = False
            st.quest_flight(self.extras[0], float(self.spec.get("HPLossPercent") or 0.0))
            st.open_quest_dialog(self.EXTRA)


@dataclass
class SwordQuest(Quest):
    """`C_SwordQuest`: the talking sword's revenge on its twin.

    The sword's "extra" room is the boss room. `GetDialogIndex` in the quest
    room is 0 while Uninitialized, Declined or Initialized. The dialog first
    runs `CheckBossRoomStatus`, which sets Declined when the boss is already
    dead, and that branch ends the quest unpaid. Otherwise the sword offers
    itself: refused, Initialized (it can be asked again); taken, Accepted and
    a level-1 `TalkingSword` joins the team (`SpawnSword`). Arriving in the
    boss room while Accepted adds a level-1 `DeathSword` to the boss's team
    (`SpawnEnemySword`, `OnEnemyTeamCreation`), and once the boss fight is
    won the after-fight dialog pays 35 gold (outcome 1) or mutation 358
    (outcome 2), from the sword or, if it died, its ghost. Finished takes the
    sword out of the team.
    """

    MAIN, AFTER = "Sword-quest-main.Dialog", "Sword-quest-after-fight.Dialog"

    def room_dialog(self, st):
        return self.MAIN if self.status in (UNINITIALIZED, DECLINED, INITIALIZED) else None

    def arrive(self, st, rid, fought, won):
        if rid == st.rooms.boss and self.status == ACCEPTED and fought and won:
            return self.AFTER
        return super().arrive(st, rid, fought, won)

    def extra_enemies(self, st, room):
        if room.id == st.rooms.boss and self.status == ACCEPTED:
            return [("DeathSword", 1)]
        return []

    def condition(self, st, name):
        if name == "HasSword":
            return any(getattr(h, "cls", None) == "TalkingSword" for h in st.squad)
        return super().condition(st, name)

    def call(self, st, fn, param):
        if fn == "CheckBossRoomStatus":
            if st.rooms.rooms[st.rooms.boss].cleared:
                self.set_status(st, DECLINED)
        elif fn == "SpawnSword":
            st.squad.append(st.make_companion("TalkingSword", 1))

    def on_status(self, st, status):
        if status == FINISHED:
            st.squad = [h for h in st.squad if getattr(h, "cls", None) != "TalkingSword"]


@dataclass
class InertQuest(Quest):
    """A quest whose controller is not ported yet: its rooms are placed and
    typed, its extra rooms fight their packs, and its dialogs never run."""


QUESTS = {"Pit": PitQuest, "Cube": CubeQuest, "Sci": SciQuest, "Rat": RatQuest,
          "Reaper": ReaperQuest, "Spider": SpiderQuest, "Bot": BotQuest,
          "Nurgle": NurgleQuest, "Flight": FlightQuest, "Sword": SwordQuest}


def make_quest(tables: dict, name: str | None) -> Quest | None:
    spec = quest_spec(tables, name)
    if spec is None:
        return None
    q = QUESTS.get(name, InertQuest)(name=name, spec=spec)
    q.levels_left = int(spec.get("AdditionalLevels") or 0)
    return q


def carry_on(quest: Quest) -> bool:
    """Take `quest` into the next level when it has levels left: its row with
    `Overrides` laid over it (one level deep, `RoomParams` key by key), and
    its rooms to be placed again."""
    if quest.levels_left <= 0:
        return False
    quest.levels_left -= 1
    spec = dict(quest.spec)
    for key, value in (spec.pop("Overrides", None) or {}).items():
        if isinstance(value, dict) and isinstance(spec.get(key), dict):
            spec[key] = {**spec[key], **value}
        else:
            spec[key] = value
    quest.spec = spec
    quest.room, quest.extras = None, []
    return True
