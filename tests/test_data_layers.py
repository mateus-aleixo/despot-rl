"""The game's override layering and table views (sim/data.py), on synthetic tables."""
import json

import pytest

from sim import data
from sim.data import (
    MergeNotImplemented,
    _merge_by_id,
    _merge_by_mutation_and_level,
    _merge_default,
    _merge_grid,
    _parse_room_layouts,
    giveable_items,
    items_by_quality,
    load_ruleset,
    units_by_class,
)


def test_default_merge_recurses_into_objects_and_replaces_the_rest():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
    over = {"a": {"y": 20, "z": 30}, "b": [9]}
    assert _merge_default(base, over) == {"a": {"x": 1, "y": 20, "z": 30}, "b": [9], "c": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}, "the base must not be mutated"


def test_merge_by_id_overlays_rows_and_honours_remove():
    base = [{"ID": 1, "v": 1}, {"ID": 2, "v": 2, "keep": True}, {"ID": 3, "v": 3}]
    over = [{"ID": 2, "v": 20}, {"ID": 3, "__remove": True}, {"ID": 4, "v": 4}]
    assert _merge_by_id(base, over) == [
        {"ID": 1, "v": 1}, {"ID": 2, "v": 20, "keep": True}, {"ID": 4, "v": 4}]


def test_keyed_merges_need_two_lists():
    with pytest.raises(MergeNotImplemented):
        _merge_by_id({"ID": 1}, [{"ID": 1}])


def test_mutation_rows_key_on_mutation_and_level_together():
    base = [{"Mutation": "Fire", "Level": 1, "W": 1}, {"Mutation": "Fire", "Level": 2, "W": 1}]
    over = [{"Mutation": "Fire", "Level": 2, "W": 5}, {"Mutation": "Ice", "Level": 1, "W": 2}]
    merged = {(r["Mutation"], r["Level"]): r["W"] for r in _merge_by_mutation_and_level(base, over)}
    assert merged == {("Fire", 1): 1, ("Fire", 2): 5, ("Ice", 1): 2}


def test_grid_merge_handles_combined_mutations_by_id():
    base = {"Width": 5, "Mutations": {"a": 1},
            "CombinedMutations": [{"ID": 1, "v": 1}, {"ID": 2, "v": 2}]}
    over = {"CombinedMutations": [{"ID": 2, "v": 99}, {"ID": 3, "v": 3}]}
    out = _merge_grid(base, over)
    assert out["Width"] == 5 and out["Mutations"] == {"a": 1}
    assert {r["ID"]: r["v"] for r in out["CombinedMutations"]} == {1: 1, 2: 99, 3: 3}
    # Present on one side only, the key is carried over untouched.
    assert _merge_grid(base, {"Width": 6})["CombinedMutations"] == base["CombinedMutations"]
    assert _merge_grid({"Width": 5}, over)["CombinedMutations"] == over["CombinedMutations"]


@pytest.fixture
def fake_game(tmp_path, monkeypatch):
    """A three-layer ruleset on disk, wired in place of data/extracted."""
    main = tmp_path / "main"
    files = {
        "Common/Units.json": [{"ID": 1, "Class": "Novice", "Level": 1, "Damage": 20},
                              {"ID": 2, "Class": "Novice", "Level": 2, "Damage": 25}],
        "Common/Balance.json": {"Gold": 1, "Food": 2},
        "Default/Units.json": [{"ID": 2, "Damage": 30},
                               {"ID": 3, "Class": "Archer", "Level": 1, "Damage": 15}],
        "Chip/Balance.json": {"Gold": 5},
        "NoFood/Units.json": [{"ID": 1, "__remove": True}],
        "Remover/Units.json": [{"ID": 3}],
    }
    for rel, content in files.items():
        p = main / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content), encoding="utf-8")
    (main / "Common/RoomLayouts.csv").write_text("id,1,type,0\n,p,\n", encoding="utf-8")
    metadata = {"Modes": {
        "Common": {"Files": {"Units": "Common/Units.json", "Balance": "Common/Balance.json"},
                   "CSV": {"RoomLayouts": "Common/RoomLayouts.csv"}},
        "Default": {"Files": {"Units": "Default/Units.json?mergeByID"},
                    "Chips": {"default": {"Files": {"Balance": "Chip/Balance.json?replace"},
                                          "WithoutFood": {
                                              "Files": {"Units": "NoFood/Units.json?mergeByID"}}},
                              "odd": {"Files": {"Units": "Remover/Units.json?remover"}},
                              "bogus": {"Files": {"Units": "Remover/Units.json?shuffle"}}}},
    }}
    meta_path = tmp_path / "metadata.txt"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(data, "METADATA", meta_path)
    real_read = data._read
    monkeypatch.setattr(data, "_read", lambda path: real_read(path, main))


def test_layers_apply_in_the_games_order(fake_game):
    tables = load_ruleset()
    units = {r["ID"]: r for r in tables["Units"]}
    assert sorted(units) == [1, 2, 3]
    assert units[2] == {"ID": 2, "Class": "Novice", "Level": 2, "Damage": 30}
    assert tables["Balance"] == {"Gold": 5}, "replace drops the base table entirely"
    assert tables["RoomLayouts"].startswith("id,1")
    assert units_by_class(tables) == {"Novice": {1: units[1], 2: units[2]},
                                      "Archer": {1: units[3]}}


def test_without_food_is_the_last_layer(fake_game):
    assert sorted(r["ID"] for r in load_ruleset(without_food=True)["Units"]) == [2, 3]


def test_an_unimplemented_strategy_raises_unless_inspecting(fake_game):
    with pytest.raises(MergeNotImplemented):
        load_ruleset(chip="odd")
    assert sorted(r["ID"] for r in load_ruleset(chip="odd", strict=False)["Units"]) == [1, 2, 3]


def test_an_unknown_strategy_always_raises(fake_game):
    with pytest.raises(MergeNotImplemented):
        load_ruleset(chip="bogus", strict=False)


def test_an_unknown_chip_is_an_error(fake_game):
    with pytest.raises(KeyError):
        load_ruleset(chip="nope")


def test_room_layouts_split_into_blocks():
    text = ("id,1,type,0,a6,b6\n"
            ",p,,e1\n"
            ",p:1,s,e1\n"
            "\n"
            "id,2,type,1,f3\n"
            "e2,,\n")
    first, second = _parse_room_layouts(text)
    assert (first.id, first.type, first.tags) == (1, 0, ["a6", "b6"])
    assert first.size == (2, 4)
    assert first.cells("s") == [(1, 2)]
    assert first.zone("p") == [(0, 1), (1, 1)]
    assert first.zone("e1") == [(0, 3), (1, 3)]
    assert (second.id, second.type, second.tags, second.size) == (2, 1, ["f3"], (1, 3))


def test_never_given_items_stay_out_of_the_shop():
    tables = {"Items": [{"Name": "sword", "Quality": 2}, {"Name": "axe", "Quality": 2},
                        {"Name": "leaflet", "Quality": 1}, {"Name": "stick"}],
              "Meta": {"Items": {"leaflet": {"NeverGiven": True}}}}
    assert giveable_items(tables) == ["sword", "axe", "stick"]
    assert items_by_quality(tables) == {2: ["axe", "sword"], 1: ["stick"]}
