"""The damage formula and damage-type flags (sim/battle.py)."""
import pytest

from sim.battle import DT_MAGICAL, DT_PHYSICAL, DT_SECONDARY, apply_damage, damage_type_mask


@pytest.mark.parametrize("args, kwargs, want", [
    ((60, 5, 0), {}, 55.0),                         # flat armor subtraction
    ((3, 5, 0), {}, 1.0),                           # armor cannot push a hit below 1
    ((0.5, 5, 0), {}, 0.5),                         # nor raise one that was already below 1
    ((-5, 1, 0), {}, 0.0),                          # nothing negative gets through
    ((60, 0, 0), {}, 60.0),
    ((60, 5, 0), {"armor_mult": 2.0}, 50.0),
    ((60, 5, 0), {"bonus_armor": 5.0}, 50.0),
    ((60, 5, 0), {"bonus_armor": 5.0, "armor_mult": 0.5}, 55.0),
    ((60, 0, 0.25), {"magical": True}, 45.0),       # resistance is multiplicative
    ((60, 100, 0.5), {"magical": True}, 30.0),      # and magic ignores armor
])
def test_mitigation(args, kwargs, want):
    assert apply_damage(*args, **kwargs) == want


def test_damage_type_mask():
    assert damage_type_mask(None) == 0 and damage_type_mask("") == 0
    assert damage_type_mask("Physical") == DT_PHYSICAL
    assert damage_type_mask("Physical, Magical") == DT_PHYSICAL | DT_MAGICAL
    assert damage_type_mask("Secondary,Unknown") == DT_SECONDARY
