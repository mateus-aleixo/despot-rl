"""The mutation grid: the gold sink behind the MUTATIONS button.

`MutationGrid.json` (the ruleset's `Mutations` table) lays 22 mutations out on
a 5 x 5 grid, each with a price per level, and a run buys levels in it with
gold. `C_Mutation.Upgrade` checks `M_Mutation.available`, that `level` is below
`costs.Count`, and `M_Team.gold >= cost`; then it raises the level, takes the
cost from the team's gold and calls `C_Mutations.OnMutationUpgrade`. The grid
is `M_Team.mutations`, part of the run's team, so every run starts it empty: it
is not account progression. `V_Mutation.Upgrade` (a click on a node) goes
straight to it, so the grid can be bought from in any room.

Which nodes can be bought:

* A node's links point right. `C_Mutations.CreateModels` indexes the cells
  `y * width + x` and calls `SetupNeighbor` once per flag: `Up` is the cell
  `index - width + 1` (up and right), `Right` is `index + 1`, `Down` is
  `index + width + 1` (down and right). The linked node gets the linking one as
  a parent and starts unavailable; a node nothing links to starts available.
* `OnMutationUpgrade` adds one to `totalLevels`, then makes a node available
  when `totalLevels >= Progress[x]` (`Progress` is 0, 4, 8, 12, 16, one entry
  per column) and every one of its parents has a level. It never takes
  availability away. "Upgrade available mutations to unlock more" is the
  progress bar's tooltip.

A level's numbers: `C_Mutation.UpgradeInternal(level)` sets `cost` to
`costs[level]` (the next level's price) and each property with a per-level list
to that list's entry `level - 1` (`MutationReflectionUtils.GetAtIndexOrDefault`,
held at the last entry). `C_Mutation.Reflect` pairs a property with its list by
the plural of its name, an `s`, or `es` after an `s`: `bonus` and `Bonuses`,
`rooms` and `roomses`, `rerollShrines` and `rerollShrineses`.

What the nodes do is split between the two layers. The fight-side ones are
mutation rows the battle pipeline already handles (`rows`); the run-side ones
are read by `RunState` (`run_value`): see `RUN_SIDE`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Node names whose effect is outside a fight, read by `RunState`:
#   ItemDiscount       one random item of each shop stock is `percent`% off
#   CheapReroll        the item shop's reroll costs `discount` less
#   RoomCount          `rooms` more rooms and `rerollShrines` more altars a level
#   AdditionalRerolls  `value` more rerolls at every altar of the level
#   Deposit            `percent`% interest on the gold at each new level, at
#                      most `threshold`
#   HealAfterFight     the humans heal `percent`% of their health after a fight
RUN_SIDE = frozenset({"ItemDiscount", "CheapReroll", "RoomCount", "AdditionalRerolls",
                      "Deposit", "HealAfterFight"})

# Synthetic mutation ids for the grid's rows and the always-on rows, clear of
# `Mutations.json` (1 to 377) and the combined mutations (10000 and up).
GRID_ID_BASE = 5000
ALWAYS_ON_ID_BASE = 5100

# The link flags and the index step each one takes, for a grid `width` wide.
LINKS = ("Up", "Right", "Down")


def singular(key: str) -> str:
    """The property a per-level list sets: `Bonuses` -> `bonus`,
    `roomses` -> `rooms`, `Values` -> `value`. Keys are matched without regard
    to case, as Newtonsoft matches them; the result keeps the key's own spelling
    with its first letter lowered, which is how `Mutations.json` names params."""
    low = key[0].lower() + key[1:]
    if low.endswith("ses"):
        return low[:-2]
    if low.endswith("s"):
        return low[:-1]
    return low


def _param_name(key: str) -> str:
    """A scalar parameter's name in the spelling `Mutations.json` rows use."""
    return key[0].lower() + key[1:]


@dataclass
class Node:
    """One cell of the grid that holds a mutation."""

    index: int
    x: int
    y: int
    name: str
    costs: list[float]
    scalars: dict[str, Any]
    per_level: dict[str, list]
    links: tuple[str, ...]
    level: int = 0
    available: bool = True
    parents: list[int] = field(default_factory=list)

    @property
    def max_level(self) -> int:
        return len(self.costs)

    @property
    def cost(self) -> float | None:
        """The next level's price, None once the node is at its last level."""
        return float(self.costs[self.level]) if self.level < len(self.costs) else None

    def params(self, level: int | None = None) -> dict[str, Any]:
        """The node's parameters at `level` (its own by default)."""
        lvl = self.level if level is None else level
        out = dict(self.scalars)
        if lvl >= 1:
            for k, values in self.per_level.items():
                if values:
                    out[k] = values[min(lvl, len(values)) - 1]
        return out

    def row(self) -> dict:
        """The node at its level as a mutation row, the shape `Mutations.json`
        rows have, for the battle pipeline. No `Class`: the grid's mutations
        cover every unit their `kind` allows."""
        r: dict[str, Any] = {"ID": GRID_ID_BASE + self.index, "Name": self.name,
                             "Class": "All", "Grid": True}
        for i, (k, v) in enumerate(self.params().items(), start=1):
            r[f"Param{i}Name"] = k
            r[f"Param{i}Value"] = v
        return r


class Grid:
    """The run's grid: levels bought, availability, and the effects they give."""

    def __init__(self, table: dict):
        self.width = int(table.get("Width") or 5)
        self.height = int(table.get("Height") or 5)
        self.progress = [int(p) for p in table.get("Progress") or [0]]
        self.nodes: dict[int, Node] = {}
        for key, raw in (table.get("Mutations") or {}).items():
            x, y = (int(v) for v in key.strip("[]").split(","))
            index = y * self.width + x
            costs, scalars, per_level, links = [], {}, {}, []
            name = ""
            for k, v in raw.items():
                kl = k.lower()
                if kl == "name":
                    name = str(v)
                elif kl == "costs":
                    costs = [float(c) for c in v]
                elif k in LINKS:
                    if v:
                        links.append(k)
                elif isinstance(v, list):
                    per_level[singular(k)] = list(v)
                else:
                    scalars[_param_name(k)] = v
            self.nodes[index] = Node(index, x, y, name, costs, scalars, per_level,
                                     tuple(links))
        step = {"Up": 1 - self.width, "Right": 1, "Down": self.width + 1}
        for node in self.nodes.values():
            for link in node.links:
                child = node.index + step[link]
                if child not in self.nodes:
                    raise ValueError(f"grid node {node.x},{node.y} links {link} "
                                     f"to an empty cell")
                self.nodes[child].parents.append(node.index)
        for node in self.nodes.values():
            node.available = not node.parents
        self.total_levels = 0

    @classmethod
    def from_tables(cls, tables: dict) -> "Grid":
        return cls(tables["Mutations"])

    # -- buying ------------------------------------------------------------
    def can_buy(self, index: int, gold: float) -> bool:
        node = self.nodes.get(index)
        return (node is not None and node.available and node.cost is not None
                and gold >= node.cost)

    def buyable(self, gold: float) -> list[int]:
        return [k for k in sorted(self.nodes) if self.can_buy(k, gold)]

    def buy(self, index: int) -> float:
        """Raise one node a level; returns what it cost. The caller has checked
        `can_buy` and takes the gold."""
        node = self.nodes[index]
        cost = float(node.cost)
        node.level += 1
        self.total_levels += 1
        for other in self.nodes.values():
            if (self.total_levels >= self._gate(other.x)
                    and all(self.nodes[p].level >= 1 for p in other.parents)):
                other.available = True
        return cost

    def _gate(self, x: int) -> int:
        return self.progress[min(x, len(self.progress) - 1)] if self.progress else 0

    # -- effects -----------------------------------------------------------
    def bought(self) -> list[Node]:
        return [n for k, n in sorted(self.nodes.items()) if n.level >= 1]

    def rows(self) -> list[dict]:
        """The bought fight-side nodes, as mutation rows."""
        return [n.row() for n in self.bought() if n.name not in RUN_SIDE]

    def run_value(self, name: str, key: str) -> float:
        """The sum of `key` over the bought nodes called `name`."""
        total = 0.0
        for n in self.bought():
            if n.name == name:
                v = n.params().get(key)
                if v is not None:
                    total += float(v)
        return total

    def key(self) -> tuple:
        """The levels, for a cache key."""
        return tuple(n.level for _, n in sorted(self.nodes.items()))


def inert(node: Node) -> bool:
    """Whether the sim gives the node no effect yet. The three `kind: Summon`
    StatBonus nodes (Fhtagn, Mglw'nafh, Wgah'nagl) raise the stats of units
    summoned in a fight, and a summon's stats come from a template shared by
    both teams in the Rust core, so nothing applies them; they can still be
    bought, as in the game, and still count toward `totalLevels`."""
    return str(node.scalars.get("kind") or "") == "Summon"


def always_on_rows(tables: dict) -> list[dict]:
    """The grid table's `SimpleMutations`: what `C_Mutations.CreateSimpleMutations`
    gives every run of the ruleset, whatever is bought. In Default that is
    GetItemBack (75%) and AdditionalRerolls (1); Sprint (`shortpvp`) swaps in
    AdditionalBuys and a 30% Intelligence."""
    out = []
    for i, raw in enumerate((tables.get("Mutations") or {}).get("SimpleMutations") or []):
        r: dict[str, Any] = {"ID": ALWAYS_ON_ID_BASE + i, "Class": "All", "AlwaysOn": True}
        n = 1
        for k, v in raw.items():
            if k.lower() == "name":
                r["Name"] = str(v)
            else:
                r[f"Param{n}Name"] = _param_name(k)
                r[f"Param{n}Value"] = v
                n += 1
        out.append(r)
    return out
