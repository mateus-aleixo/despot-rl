"""Load Despot's Game's balance data the way the game itself does.

The game keeps its tables in layers: a `Common` set, then a per-mode set, then a
per-chip set, then an optional `WithoutFood` variant. Each entry in
`metadata.json` is `LogicalName -> path[?mergeStrategy]`, and later layers merge
onto earlier ones. This module reproduces that resolution so a ruleset here is
the same ruleset the game would build.

Unimplemented merge strategies raise rather than silently returning a
half-merged table: a wrong number here would be invisible and would poison every
downstream calibration.
"""
from __future__ import annotations

import csv
import io
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
JSON_ROOT = ROOT / "data" / "extracted" / "json"
MAIN = JSON_ROOT / "EncryptedMainGroup"
METADATA = ROOT / "data" / "extracted" / "gamedata" / "metadata.txt"


class MergeNotImplemented(NotImplementedError):
    """Raised for a merge strategy this loader does not reproduce yet."""


# --------------------------------------------------------------------------
# merge strategies
# --------------------------------------------------------------------------

def _merge_default(base: Any, over: Any) -> Any:
    """Newtonsoft's JObject.Merge: recurse into objects, replace anything else."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge_default(out[k], v) if k in out else v
        return out
    return over


def _merge_replace(base: Any, over: Any) -> Any:
    return over


REMOVE_KEY = "__remove"


def _merge_keyed(base: Any, over: Any, key, label: str) -> Any:
    """Overlay two lists of objects by a key function, honouring `__remove`.

    `__remove` appears 134 times across the shipped override files, so it is a
    general part of the merge pipeline, not a quirk of one table.
    """
    if not (isinstance(base, list) and isinstance(over, list)):
        raise MergeNotImplemented(
            f"{label} needs two lists, got {type(base).__name__}/{type(over).__name__}")
    out: dict = {key(row): dict(row) for row in base}
    for row in over:
        k = key(row)
        if row.get(REMOVE_KEY):
            out.pop(k, None)
            continue
        out[k] = _merge_default(out[k], row) if k in out else dict(row)
    return list(out.values())


def _merge_by_id(base: Any, over: Any) -> Any:
    """Both sides are lists of objects with an `ID`; overlay by that key."""
    return _merge_keyed(base, over, lambda r: r.get("ID"), "mergeByID")


def _merge_by_mutation_and_level(base: Any, over: Any) -> Any:
    """Overlay MutationsByLevel rows by (Mutation, Level).

    Verified: the lambda references exactly the string literals "Mutation" and
    "Level" and builds a dictionary keyed by them.
    """
    return _merge_keyed(base, over, lambda r: (r.get("Mutation"), r.get("Level")),
                        "mergeByMutationAndLevel")


# The grid key that mergeGrid pulls out and merges separately, verified from the
# lambda: it takes the "CombinedMutations" property off both sides, Removes it,
# merges what is left normally, then merges that key on its own.
GRID_SPECIAL_KEY = "CombinedMutations"


def _merge_grid(base: Any, over: Any) -> Any:
    if not (isinstance(base, dict) and isinstance(over, dict)):
        raise MergeNotImplemented("mergeGrid needs two objects")
    b_special = base.get(GRID_SPECIAL_KEY)
    o_special = over.get(GRID_SPECIAL_KEY)
    b_rest = {k: v for k, v in base.items() if k != GRID_SPECIAL_KEY}
    o_rest = {k: v for k, v in over.items() if k != GRID_SPECIAL_KEY}

    out = _merge_default(b_rest, o_rest)
    if o_special is None:
        if b_special is not None:
            out[GRID_SPECIAL_KEY] = b_special
    elif b_special is None:
        out[GRID_SPECIAL_KEY] = o_special
    else:
        # CombinedMutations rows carry an ID (10000+), so they key by it.
        out[GRID_SPECIAL_KEY] = _merge_by_id(b_special, o_special)
    return out


MERGERS = {
    None: _merge_default,
    "replace": _merge_replace,
    "mergeByID": _merge_by_id,
    "mergeByMutationAndLevel": _merge_by_mutation_and_level,
    "mergeGrid": _merge_grid,
}

# Strategies the game has that are not reproduced yet. Named explicitly so the
# error says which one, instead of falling through to the default merge.
UNIMPLEMENTED = {"remover", "cloner"}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _read(path: str, root: pathlib.Path = MAIN) -> Any:
    p = root / path
    if not p.exists():
        raise FileNotFoundError(f"{p} (did tools/split_gamedata.py run?)")
    if p.suffix.lower() == ".csv":
        return p.read_text(encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def _layers(metadata: dict, mode: str, chip: str, without_food: bool) -> list[dict]:
    """The file maps to apply, in the order the game applies them."""
    modes = metadata["Modes"]
    out = [modes["Common"], modes[mode]]
    chips = modes[mode].get("Chips", {})
    if chip in chips:
        out.append(chips[chip])
        if without_food and "WithoutFood" in chips[chip]:
            out.append(chips[chip]["WithoutFood"])
    elif chip is not None:
        raise KeyError(f"chip {chip!r} not in mode {mode!r}; have {sorted(chips)}")
    return out


def load_ruleset(mode: str = "Default", chip: str = "default", without_food: bool = False,
                 strict: bool = True) -> dict[str, Any]:
    """Resolve one ruleset into `logical name -> parsed table`.

    With `strict`, a merge strategy this loader does not implement raises. Turn
    it off only to inspect partial data, never to produce numbers.
    """
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    tables: dict[str, Any] = {}
    for layer in _layers(metadata, mode, chip, without_food):
        for section in ("Files", "CSV"):
            for name, spec in layer.get(section, {}).items():
                path, _, strategy = spec.partition("?")
                strategy = strategy or None
                data = _read(path)
                if name not in tables:
                    tables[name] = data
                    continue
                if strategy in UNIMPLEMENTED:
                    if strict:
                        raise MergeNotImplemented(
                            f"{name}: strategy {strategy!r} for {path} is not implemented")
                    continue
                merger = MERGERS.get(strategy)
                if merger is None:
                    raise MergeNotImplemented(f"{name}: unknown merge strategy {strategy!r}")
                tables[name] = merger(tables[name], data)
    return tables


# --------------------------------------------------------------------------
# room layouts
# --------------------------------------------------------------------------

@dataclass
class RoomLayout:
    """One layout block from RoomLayouts.csv.

    `grid` is row-major, each cell the raw token: '' empty, 'p' player zone,
    'p:N' a numbered player corner, 's' the shop/door marker, 'e1'/'e2' enemy
    zones. `tags` are the trailing header codes (a6, b6, f3, ...).
    """
    id: int
    type: int
    tags: list[str]
    grid: list[list[str]] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return len(self.grid), max((len(r) for r in self.grid), default=0)

    def cells(self, token: str) -> list[tuple[int, int]]:
        return [(r, c) for r, row in enumerate(self.grid)
                for c, v in enumerate(row) if v == token]

    def zone(self, prefix: str) -> list[tuple[int, int]]:
        """Cells whose token is `prefix` or starts with `prefix:`."""
        return [(r, c) for r, row in enumerate(self.grid)
                for c, v in enumerate(row) if v == prefix or v.startswith(prefix + ":")]


_LAYOUT_CACHE: dict[int, list] = {}


def parse_room_layouts(text: str) -> list["RoomLayout"]:
    """Memoised: the CSV is fixed per ruleset and was being re-parsed per fight."""
    key = id(text)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    out = _parse_room_layouts(text)
    _LAYOUT_CACHE[key] = out
    return out


def _parse_room_layouts(text: str) -> list[RoomLayout]:
    """Split the CSV into layout blocks. A block starts at an `id,N,type,T` row."""
    rows = list(csv.reader(io.StringIO(text)))
    layouts: list[RoomLayout] = []
    current: RoomLayout | None = None
    for row in rows:
        if row and row[0] == "id":
            tags = [c.strip() for c in row[4:] if c.strip()]
            current = RoomLayout(id=int(row[1]), type=int(row[3]), tags=tags)
            layouts.append(current)
        elif current is not None:
            if not any(c.strip() for c in row):
                continue
            current.grid.append([c.strip() for c in row])
    return layouts


# --------------------------------------------------------------------------
# convenience views
# --------------------------------------------------------------------------

def units_by_class(tables: dict[str, Any]) -> dict[str, dict[int, dict]]:
    """`Units` keyed as class -> level -> row."""
    out: dict[str, dict[int, dict]] = {}
    for row in tables["Units"]:
        out.setdefault(row["Class"], {})[row["Level"]] = row
    return out


_ITEM_CACHE: dict[int, dict[str, dict]] = {}


def items_by_name(tables: dict[str, Any]) -> dict[str, dict]:
    """Memoised: the run layer asks for an item's Cost once per shop slot per
    step, and this was rebuilding all 56 rows each time."""
    key = id(tables["Items"])
    hit = _ITEM_CACHE.get(key)
    if hit is None:
        hit = _ITEM_CACHE[key] = {row["Name"]: row for row in tables["Items"]}
    return hit


_QUALITY_CACHE: dict[int, dict[int, list[str]]] = {}


def giveable_items(tables: dict[str, Any]) -> list[str]:
    """Items the game is willing to hand out, in `Items.json` order.

    `M_Item.neverGiven` comes from `Meta.Items[name].NeverGiven`, and it is
    true for exactly six rows: plant-leaf, leaflet, cube-part, comic-book,
    rat-flute and mik-helmet. All six carry 0 Damage, 0 Health and 0 Power, so
    they are worse than an empty hand -- they only exist as quest props.
    """
    meta = (tables.get("Meta") or {}).get("Items") or {}
    return [row["Name"] for row in tables["Items"]
            if not (meta.get(row["Name"]) or {}).get("NeverGiven")]


def items_by_quality(tables: dict[str, Any]) -> dict[int, list[str]]:
    """The shop's draw pools: `Items.Quality` -> item names, sorted.

    `C_ItemShop.Roll` picks a quality from `M_ItemData.templatesByQuality`'s
    keys and then an item from that quality's list, so the two draws are
    separate and this is the second one's pool.
    """
    key = id(tables["Items"])
    hit = _QUALITY_CACHE.get(key)
    if hit is not None:
        return hit
    rows = {row["Name"]: row for row in tables["Items"]}
    out: dict[int, list[str]] = {}
    for name in giveable_items(tables):
        out.setdefault(int(rows[name].get("Quality") or 1), []).append(name)
    for names in out.values():
        names.sort()
    _QUALITY_CACHE[key] = out
    return out


def skills_by_id(tables: dict[str, Any]) -> dict[int, dict]:
    return {row["ID"]: row for row in tables["Skills"]}


