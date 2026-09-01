"""Compose a fightable unit from the game's tables.

A player human is the `Novice` class carrying an item; the item supplies the
real class name and adds its stats on top of the base row. That composition is
verified: Novice Speed 80 + broadsword Speed 20 = 100, which is exactly the
`UnitMovement.speed` on the shipped Swordsman prefab.

A unit has **one** level, not two. `M_Unit` carries a single `level`, and
`CS_Units.LevelUp` both re-reads the class row at the new level and re-applies
the item's bonus at it, so the same number scales the Novice base row and the
item together.

Enemies are plain `Units.json` rows with no item.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .assumptions import DEFAULT, TILE

# Stats that an item contributes, with an optional per-level term.
ITEM_STATS = ("Health", "Damage", "Armor", "Resistance", "Speed", "Mana", "AttackSpeed", "Range")

# How the per-level term applies. `CS_Item` is one Apply* function per stat and
# they do not agree, so a single rule would be wrong for most of them. Read off
# the binary:
#
#   Health, Damage, Armor, Mana   unit += item * (1 + perLevel/100) ** (level-1)
#   Range                         unit += item + perLevel * (level-1)
#   Speed                         unit += item          (no per-level term at all)
#   AttackSpeed                   unit *= 1 + (item + perLevel*(level-1)) / 100
#   Resistance                    unit  = u + (1-u)*r,  r = (item + perLevel*(level-1))/100
#
# `CS_Item.GetBonus(value, perLevel, level)` is literally
# `value * powf(perLevel/100 + 1, level - 1)`; only the four compounding stats
# route through it. This was additive here, which understates a heavy item at
# level 5 (opm-costume 300 damage at +9 is 423, not 336) and overstates a light
# one (broadsword 60 at +10 is 88, not 100).
COMPOUNDING_STATS = frozenset({"Health", "Damage", "Armor", "Mana"})

# Stats an item modifies by percentage rather than by addition.
PERCENT_STATS = frozenset({"AttackSpeed"})

# Only `M_Unit.level` 1..5 exists: Units.json carries five rows per class and
# `CS_Units.LevelUp` returns early once the new level would be 5 or more.
MAX_UNIT_LEVEL = 5


@dataclass(frozen=True)
class UnitSpec:
    """A unit's resolved stats, in sim units (world distance, seconds)."""
    name: str
    cls: str
    level: int
    health: float
    damage: float
    attack_speed: float      # Units.json AttackSpeed, meaning per assumptions
    range_world: float       # already converted from tiles
    armor: float
    resistance: float        # fraction 0..1, applied as (1 - resistance)
    speed: float             # world units per second
    mana: float
    power: float
    melee: bool = True           # Range stat was 0, so reach is contact-based
    size: int = 1                # Meta.Classes[cls].Size, in tiles
    item: str | None = None
    skills: tuple[int, ...] = ()
    # `Units.json` ExpReward plus the item's, which is what `C_Unit.Die` hands
    # to the other team when this unit dies.
    exp_reward: float = 0.0

    @property
    def attack_period(self) -> float:
        """Seconds between swings."""
        if self.attack_speed <= 0:
            return float("inf")
        return 1.0 / self.attack_speed if DEFAULT.attack_speed_is_rate else self.attack_speed

    @property
    def is_ranged(self) -> bool:
        return not self.melee

    @property
    def radius(self) -> float:
        return self.size * DEFAULT.agent_radius_per_size


def _num(row: dict, key: str) -> float:
    v = row.get(key)
    return 0.0 if v is None else float(v)


def item_term(item_row: dict, key: str, level: int) -> float:
    """The item's own contribution for one stat at `level`, before it is merged.

    Compounding for the four stats that go through `CS_Item.GetBonus`, linear
    for the rest. See COMPOUNDING_STATS for where this comes from.
    """
    base = _num(item_row, key)
    per_level = _num(item_row, f"{key}PerLevel")
    if key in COMPOUNDING_STATS:
        return base * (1.0 + per_level / 100.0) ** (level - 1)
    return base + per_level * (level - 1)


def build_unit(units_by_cls: dict, cls: str, level: int = 1,
               item_row: dict | None = None,
               name: str | None = None, size: int = 1) -> UnitSpec:
    """Resolve one unit. `item_row` is an Items.json row, or None for enemies.

    `level` is the unit level: it selects the class row *and* scales the item.
    """
    levels = units_by_cls[cls]
    row = levels.get(level) or levels[min(levels)]

    stats = {k: _num(row, k) for k in ITEM_STATS}
    if item_row is not None:
        for k in ITEM_STATS:
            contribution = item_term(item_row, k, level)
            if k in PERCENT_STATS:
                # Item AttackSpeed is a percentage modifier, not an addition:
                # values run -40..+70, and the heaviest weapons carry negative
                # ones (opm-costume -40% at 300 damage). The ASAura class skill
                # uses the same convention explicitly (percentage=true).
                stats[k] *= 1.0 + contribution / 100.0
            elif k == "Resistance":
                # `CS_Item.ApplyResistance` is `u + (1 - u) * r`, not `u + r`:
                # two sources of resistance stack multiplicatively. Both sides
                # are percentages here and are scaled to fractions below.
                # Every item in the Default ruleset has Resistance 0, so this
                # only bites on the Hard chip's item table.
                u = stats[k] / 100.0
                stats[k] = 100.0 * (u + (1.0 - u) * contribution / 100.0)
            else:
                stats[k] += contribution

    # Range is in tiles; a range of 0 means melee, whose real reach this sim
    # has not resolved (see assumptions.melee_margin).
    radius = size * DEFAULT.agent_radius_per_size
    melee_reach = 2.0 * radius + DEFAULT.melee_margin
    range_world = stats["Range"] * TILE if stats["Range"] > 0 else melee_reach

    return UnitSpec(
        name=name or (item_row["Name"] if item_row else cls),
        cls=item_row["Class"] if item_row else cls,
        level=level,
        health=stats["Health"],
        damage=stats["Damage"],
        attack_speed=stats["AttackSpeed"],
        range_world=range_world,
        melee=stats["Range"] <= 0,
        size=size,
        armor=stats["Armor"],
        # Units.json Resistance is a percentage (5..90); ApplyResistance
        # computes (1 - resistance) * amount with it as a fraction, so it
        # has to be scaled here or magical damage would come out negative.
        resistance=stats["Resistance"] / 100.0,
        speed=stats["Speed"],
        mana=stats["Mana"],
        # `Power` has no per-level column on an item, so the level moves this
        # only through the class row: a Novice is 150 at level 1 and 750 at 5.
        power=_num(row, "Power") + (_num(item_row, "Power") if item_row else 0.0),
        item=item_row["Name"] if item_row else None,
        skills=tuple(int(row[f"Skill{i}"]) for i in range(1, 9) if row.get(f"Skill{i}")),
        exp_reward=_num(row, "ExpReward") + (_num(item_row, "ExpReward") if item_row else 0.0),
    )


def build_player_squad(tables: dict[str, Any],
                       loadout: list[tuple[str, int]]) -> list[UnitSpec]:
    """`loadout` is [(item name, unit level), ...]; each entry is one human."""
    from .data import items_by_name, units_by_class
    items, ubc = items_by_name(tables), units_by_class(tables)
    classes = tables["Meta"]["Classes"]
    out = []
    for item, lvl in loadout:
        cls = items[item]["Class"]
        size = int((classes.get(cls) or {}).get("Size") or 1)
        out.append(build_unit(ubc, "Novice", lvl, items[item],
                              name=f"{cls}:{item}", size=size))
    return out


def build_enemy_pack(tables: dict[str, Any], pack: dict) -> list[UnitSpec]:
    """Expand one EnemyPacks.json row into its units (uses `Max` count)."""
    from .data import units_by_class
    ubc = units_by_class(tables)
    n = int(pack.get("Max") or pack.get("Min") or 1)
    lvl = int(pack.get("Level") or 1)
    size = int(((tables["Meta"]["Classes"].get(pack["Class"])) or {}).get("Size") or 1)
    return [build_unit(ubc, pack["Class"], lvl, name=f"{pack['Class']}#{i}", size=size)
            for i in range(n)]
