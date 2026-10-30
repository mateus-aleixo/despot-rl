"""The level-entry dialogs: `EncryptedMainGroup/dialogs.json` and `Game.Dialogs`.

`C_Levels.StartDialog` runs on every level change, the first included
(`C_Session.Begin` reaches it through `ChangeLevel`). It draws from the pool at
random, asks `C_Dialogs.StartDialog` to start the draw, and tries another when
that refuses; only a dialog that starts is removed, so each is seen at most once
a run and one whose gates fail can come up later. An empty pool shows
`NoDialogsPlaceholder`, which does nothing.

## The tree

    node     text, choices, events
    choice   button, outcomes, events, link
    outcome  a node, plus a weight when its choice has more than one

A choice with no outcomes ends the dialog. A choice with several outcomes draws
one by weight. A `link` names another node of the same dialog by path, in the
scheme the localization keys use: segments joined by `.`, each a 1-based choice
index, with `-<outcome>` added when that choice has more than one outcome; `#`
is the body. A choice that has a link and no outcomes goes to that node (and a
choice with no button is one of these); one with outcomes of its own keeps
them.

## When things happen

`C_Dialogs.PrepareEvents(M_Dialog)` walks each choice's outcomes and builds
their events with the choice as `num`, folding each event's `GetEnabledState`
into the choice's `enableds` and `hiddens`. So a choice is gated by the events
of the nodes it can lead to, and a body's own gates decide whether the dialog
starts at all. A node's events apply when the node is reached, in order; a
choice's events when it is chosen.

## The events, from `Dialogs.Events.*`

    resources   PlusGold (amount or fromVariable), MinusGold, PlusFood, MinusFood,
                GiveExperience (amount, perUnit)
    roster      GetNames picks units at random into the dialog's `units`, which
                RemoveUnit, CloneUnit and RemoveHealth then act on by 1-based
                `id`; GiveUnit adds units by class and item; RemoveHealth
                multiplies health by (1 - percent / 100), every unit when it has
                no `id`
    inventory   GiveItem, RemoveItem (`C_Player.DownGradeUnit`, preferring a
                class, and the item's cost into `itemCost`)
    mutations   GiveMutation by id
    variables   DefineVariables, AddToVariable, MultiplyVariable,
                RemoveVariables, CompareVariable
    gates       CheckFood, CheckGold, CheckItem, HasClass, HasSlot,
                HasSlotForClones, Disable, and the Minus events (MinusFood is a
                CheckFood, MinusGold a CheckGold)
    fight       Ambush: the current room's `fightResult` is reset and its doors
                closed, so its pack attacks

`HasSlot` sums the team's cell sizes against 49, the 7 by 7 player zone. A
`MinusFood` or `MinusGold` with no amount spends the amount of the dialog's
own `CheckFood` or `CheckGold`, the way `Fitness` defines `foodToRemove` for
both.
"""
from __future__ import annotations

import json
import operator
import pathlib
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "extracted" / "json"
DIALOGS = ROOT / "EncryptedMainGroup" / "dialogs.json"

# `HasSlot`: the team's cells against the 7 by 7 player zone.
TEAM_CELLS = 49

# Choices at one node, the most any pool dialog shows.
MAX_CHOICES = 4

# A dialog can loop for ever (`GuidingStone` links its third road back to the
# body), which the game allows and a run cannot. After this many choices the
# dialog closes where it stands.
MAX_CHOICES_PER_DIALOG = 40

_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt,
        "==": operator.eq, "!=": operator.ne}

_TABLE: dict | None = None


def table() -> dict[str, dict]:
    """Every dialog by name, `Dialog.<Name>.Title` stripped to `<Name>`."""
    global _TABLE
    if _TABLE is None:
        entries = json.loads(DIALOGS.read_text(encoding="utf-8"))
        _TABLE = {e["title"][len("Dialog."):-len(".Title")]: e["body"] for e in entries}
    return _TABLE


def pool(tables: dict) -> list[str]:
    """`Game.Dialogs` for this ruleset, the names that exist in the table."""
    names = (tables.get("Game") or {}).get("Dialogs") or []
    known = table() if DIALOGS.exists() else {}
    return [n for n in names if n in known]


def node_paths(body: dict) -> dict[str, dict]:
    """Every node of a dialog by its path; the body is `#` and ``."""
    out = {"#": body, "": body}

    def walk(node, path):
        for ci, c in enumerate(node.get("choices") or [], 1):
            outs = c.get("outcomes") or []
            for oi, o in enumerate(outs, 1):
                seg = f"{ci}" if len(outs) == 1 else f"{ci}-{oi}"
                p = f"{path}.{seg}" if path else seg
                out[p] = o
                walk(o, p)

    walk(body, "")
    return out


def decision_nodes(names: list[str]) -> list[tuple[str, str]]:
    """(dialog, path) for every node that shows two or more choices, in a
    fixed order. The observation one-hots over these."""
    out = []
    for name in names:
        for path, node in sorted(node_paths(table()[name]).items()):
            if path != "#" and len(node.get("choices") or []) >= 2:
                out.append((name, path))
    return out


def _num(value, variables: dict, default=0.0) -> float:
    """A parameter that may be a number, a numeric string, or a variable name."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(variables.get(str(value), default))


@dataclass
class Dialog:
    """One dialog in progress."""
    name: str
    path: str
    variables: dict = field(default_factory=dict)
    units: list = field(default_factory=list)       # GetNames' picks, in order
    food_check: float = 0.0                          # the dialog's CheckFood amount
    gold_check: float = 0.0
    from_consumable: bool = False
    ambush: bool = False
    steps: int = 0
    closed: bool = False

    @property
    def node(self) -> dict:
        return node_paths(table()[self.name])[self.path]


class Engine:
    """Runs dialogs against a `RunState`. Holds no state of its own."""

    def __init__(self, st):
        self.st = st

    # -- gates ---------------------------------------------------------------
    def _event_gate(self, d: Dialog, e: dict, v: dict) -> tuple[bool, bool]:
        """`C_Event.GetEnabledState` for one event: (enabled, hidden)."""
        st, cls = self.st, e.get("className")
        p = e.get("parameters") or {}
        ok = True
        if cls in ("CheckGold", "MinusGold"):
            amount = _num(p.get("amount"), v, d.gold_check)
            ok = st.gold >= amount
        elif cls in ("CheckFood", "MinusFood"):
            amount = _num(p.get("amount"), v, d.food_check)
            ok = st.food.amount >= amount
        elif cls == "CheckItem":
            ok = any(h.item for h in st.squad)
        elif cls == "HasClass":
            have = any(st.item_class(h.item) == p.get("class") for h in st.squad)
            ok = have != bool(p.get("invert"))
        elif cls == "HasSlot":
            ok = len(st.squad) < TEAM_CELLS
        elif cls == "HasSlotForClones":
            ok = len(st.squad) + max(1, len(d.units)) <= TEAM_CELLS
        elif cls == "CompareVariable":
            fn = _OPS.get(str(p.get("op")), operator.ge)
            ok = fn(float(v.get(str(p.get("name")), 0.0)), _num(p.get("value"), v))
            return ok, (not ok) and bool(p.get("hideIfDisabled"))
        elif cls == "Disable":
            ok = not (p.get("consumable") and d.from_consumable)
        elif cls == "GetNames":
            ok = len(st.squad) >= int(_num(p.get("number"), v, 1.0))
        return ok, False

    def _gates(self, d: Dialog, events: list) -> tuple[bool, bool]:
        """Every gate in `events`, prepared in order the way `PrepareEvents`
        does: a `DefineVariables` (or a `MultiplyVariable` applied on creation)
        holds for the gates after it, so `OneHandBandit` can define
        `robotsGold` and test it in the same node."""
        v = dict(d.variables)
        enabled, hidden = True, False
        for e in events or []:
            cls, p = e.get("className"), e.get("parameters") or {}
            if cls == "DefineVariables":
                for k, val in p.items():
                    v[k] = (float(len(self.st.squad)) if val == "NumberOfUnitsInTeam"
                            else _num(val, v))
                continue
            if cls == "MultiplyVariable" and p.get("applyOnCreation"):
                name = str(p.get("name"))
                v[name] = v.get(name, 0.0) * _num(p.get("value"), v)
                continue
            ok, hide = self._event_gate(d, e, v)
            enabled &= ok
            hidden |= hide
        return enabled, hidden

    def choices(self, d: Dialog) -> list[tuple[int, bool]]:
        """(index, enabled) for every choice not hidden at the current node."""
        out = []
        for i, c in enumerate(d.node.get("choices") or []):
            enabled, hidden = self._gates(d, c.get("events"))
            for o in self._outcomes(d, d.path, c):
                ok, hide = self._gates(d, o.get("events"))
                enabled &= ok
                hidden |= hide
            if not hidden:
                out.append((i, enabled))
        return out

    def legal(self, d: Dialog) -> list[int]:
        return [i for i, ok in self.choices(d) if ok]

    def _resolve(self, d: Dialog, link: str):
        """What a link names: ("node", path) or ("choice", parent path, index).

        A path whose last segment is a choice with several outcomes names that
        choice, which is taken again (`RedStringsBar` and `OneHandBandit` loop
        back to a coin flip this way); otherwise it names the node there.
        """
        if link == "#":
            return ("node", "")
        paths = node_paths(table()[d.name])
        if link in paths:
            return ("node", link)
        parent, _, last = link.rpartition(".")
        index = int(last.split("-")[0]) - 1
        if parent in paths and 0 <= index < len(paths[parent].get("choices") or []):
            return ("choice", parent, index)
        return None

    def _outcomes(self, d: Dialog, path: str, choice: dict, depth: int = 0) -> list[dict]:
        outs = choice.get("outcomes") or []
        if outs or "link" not in choice or depth > 4:
            return outs
        target = self._resolve(d, str(choice["link"]))
        if target is None:
            return []
        if target[0] == "node":
            return [node_paths(table()[d.name])[target[1]]]
        parent = node_paths(table()[d.name])[target[1]]
        return self._outcomes(d, target[1], (parent.get("choices") or [])[target[2]], depth + 1)

    def can_start(self, name: str, from_consumable: bool = False) -> bool:
        d = Dialog(name=name, path="", from_consumable=from_consumable)
        self._prepare(d, table()[name])
        enabled, _ = self._gates(d, table()[name].get("events"))
        return enabled and bool(self.legal(d))

    # -- running -------------------------------------------------------------
    def start(self, name: str, from_consumable: bool = False) -> Dialog:
        d = Dialog(name=name, path="", from_consumable=from_consumable)
        self._enter(d, "")
        return d

    def _prepare(self, d: Dialog, node: dict) -> None:
        """What a node sets up before its gates can be read: the variables it
        defines and the amounts its checks name."""
        for e in node.get("events") or []:
            cls, p = e.get("className"), e.get("parameters") or {}
            if cls == "DefineVariables":
                for k, val in p.items():
                    d.variables[k] = (float(len(self.st.squad))
                                      if val == "NumberOfUnitsInTeam" else _num(val, d.variables))
            elif cls == "MultiplyVariable" and p.get("applyOnCreation"):
                name = str(p.get("name"))
                d.variables[name] = d.variables.get(name, 0.0) * _num(p.get("value"), d.variables)
            elif cls == "CheckFood":
                d.food_check = _num(p.get("amount"), d.variables)
            elif cls == "CheckGold":
                d.gold_check = _num(p.get("amount"), d.variables)

    def _enter(self, d: Dialog, path: str) -> None:
        d.path = path
        node = d.node
        self._prepare(d, node)
        for e in node.get("events") or []:
            if not (e.get("className") == "MultiplyVariable"
                    and (e.get("parameters") or {}).get("applyOnCreation")):
                self._apply(d, e)

    def choose(self, d: Dialog, index: int) -> bool:
        """Take choice `index`. Returns whether the dialog is still open."""
        d.steps += 1
        self._take(d, d.path, (d.node.get("choices") or [])[index], index)
        if d.closed or d.steps >= MAX_CHOICES_PER_DIALOG:
            return False
        return bool(d.node.get("choices"))

    def _take(self, d: Dialog, path: str, choice: dict, index: int, depth: int = 0) -> None:
        for e in choice.get("events") or []:
            self._apply(d, e)
        outs = choice.get("outcomes") or []
        if outs:
            weights = [float(o.get("weight") or 1.0) for o in outs]
            oi = (self.st.rng.choices(range(len(outs)), weights=weights)[0]
                  if len(outs) > 1 else 0)
            seg = f"{index + 1}" if len(outs) == 1 else f"{index + 1}-{oi + 1}"
            self._enter(d, f"{path}.{seg}" if path else seg)
            return
        target = self._resolve(d, str(choice["link"])) if "link" in choice else None
        if target is None or depth > 4:
            d.closed = True
        elif target[0] == "node":
            self._enter(d, target[1])
        else:
            parent = node_paths(table()[d.name])[target[1]]
            self._take(d, target[1], (parent.get("choices") or [])[target[2]],
                       target[2], depth + 1)

    # -- events --------------------------------------------------------------
    def _pick(self, d: Dialog, p: dict) -> list:
        """The units an event names: `id` is 1-based into GetNames' picks,
        `number` the first that many, nothing the first."""
        if p.get("id") is not None:
            i = int(_num(p.get("id"), d.variables)) - 1
            return [d.units[i]] if 0 <= i < len(d.units) else []
        n = int(_num(p.get("number"), d.variables, 1.0))
        return d.units[:n]

    def _apply(self, d: Dialog, e: dict) -> None:
        st, cls = self.st, e.get("className")
        p = e.get("parameters") or {}
        v = d.variables
        if cls == "PlusGold":
            if p.get("fromVariable") is not None:
                st.gold += float(v.get(str(p["fromVariable"]), 0.0))
            else:
                st.gold += _num(p.get("amount"), v)
        elif cls == "MinusGold":
            st.gold = max(0.0, st.gold - _num(p.get("amount"), v, d.gold_check))
        elif cls == "PlusFood":
            st.food.amount += _num(p.get("amount"), v)
        elif cls == "MinusFood":
            st.food.amount = max(0.0, st.food.amount - _num(p.get("amount"), v, d.food_check))
        elif cls == "GiveExperience":
            amount = _num(p.get("amount"), v)
            if p.get("perUnit"):
                amount *= max(1, len(st.squad))
            st.gain_experience(amount)
        elif cls == "GiveMutation":
            st.give_mutation(int(_num(p.get("id"), v)))
        elif cls == "GiveItem":
            st.give_item(str(p.get("item")))
        elif cls == "RemoveItem":
            cost = st.take_item(p.get("prioritizeClass"))
            v["itemCost"] = cost
        elif cls == "GiveUnit":
            classes = list(p.get("classes") or [])
            items = list(p.get("items") or [None] * len(classes))
            levels = list(p.get("levels") or [1] * len(classes))
            # The variable a dialog defines for it, when there is one, is how
            # many to give; `Reproductor` makes it half the team.
            count = (int(v["numberOfUnitsToGive"]) if "numberOfUnitsToGive" in v
                     else len(classes))
            for k in range(count):
                if len(st.squad) >= TEAM_CELLS:
                    break
                j = k % max(1, len(classes))
                st.add_human(items[j] if j < len(items) else None,
                             int(levels[j]) if j < len(levels) and levels[j] else 1)
        elif cls == "GetNames":
            n = int(_num(p.get("number"), v, 1.0))
            n = min(n, len(st.squad))
            d.units = st.rng.sample(list(st.squad), n) if n else []
        elif cls == "RemoveUnit":
            for h in self._pick(d, p):
                if h in st.squad and len(st.squad) > 1:
                    st.squad.remove(h)
        elif cls == "CloneUnit":
            for h in self._pick(d, p):
                if len(st.squad) < TEAM_CELLS:
                    st.add_human(h.item, h.level)
        elif cls == "RemoveHealth":
            mult = 1.0 - _num(p.get("percent"), v) / 100.0
            targets = self._pick(d, p) if p.get("id") is not None else list(st.squad)
            for h in targets:
                h.hp = max(0.01, h.hp * mult)
        elif cls == "DefineVariables":
            pass                            # done in `_prepare`
        elif cls == "AddToVariable":
            name = str(p.get("name"))
            v[name] = v.get(name, 0.0) + _num(p.get("value"), v)
        elif cls == "MultiplyVariable":
            name = str(p.get("name"))
            v[name] = v.get(name, 0.0) * _num(p.get("value"), v)
        elif cls == "RemoveVariables":
            for name in p.get("variables") or []:
                v.pop(str(name), None)
        elif cls == "Ambush":
            d.ambush = True
        # Gates and text (CheckGold, CheckFood, CheckItem, HasClass, HasSlot,
        # HasSlotForClones, CompareVariable, Disable) change nothing on apply.
