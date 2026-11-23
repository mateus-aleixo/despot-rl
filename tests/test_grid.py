"""The mutation grid's rules on a small hand-made table, no game data needed.

`tools/validate_sim.py` checks the same rules against the shipped grid.
"""
from sim.grid import Grid, always_on_rows, inert, singular

# Three columns, two rows:
#   (0,0) A -R-> (1,0) B -R-> (2,0) D
#   (0,1) C -U-> (1,0) B        (1,1) E is linked from nothing but column 1
TABLE = {
    "Width": 3, "Height": 2, "Progress": [0, 2, 3],
    "Mutations": {
        "[0, 0]": {"Name": "StatBonus", "Stat": "Health", "Costs": [5, 10],
                   "Bonuses": [50, 100], "Right": True},
        "[0, 1]": {"name": "StatBonus", "Stat": "Armor", "Costs": [5],
                   "Bonuses": [10], "Up": True},
        "[1, 0]": {"Name": "RoomCount", "Costs": [20, 40], "roomses": [1, 2],
                   "rerollShrineses": [1, 2], "Right": True},
        "[1, 1]": {"Name": "StatBonus", "kind": "Summon", "Stat": "Damage",
                   "Costs": [7], "Bonuses": [20]},
        "[2, 0]": {"name": "Deposit", "Costs": [30], "percent": 20, "thresholds": [40]},
    },
    "SimpleMutations": [{"name": "GetItemBack", "percent": 75},
                        {"Name": "AdditionalRerolls", "value": 1}],
}


def cell(g, x, y):
    return y * g.width + x


def test_plurals_name_the_property():
    assert singular("Bonuses") == "bonus"
    assert singular("roomses") == "rooms"
    assert singular("rerollShrineses") == "rerollShrines"
    assert singular("Values") == "value"
    assert singular("thresholds") == "threshold"


def test_links_make_parents_and_close_children():
    g = Grid(TABLE)
    b = g.nodes[cell(g, 1, 0)]
    assert sorted(b.parents) == [cell(g, 0, 0), cell(g, 0, 1)]
    assert not b.available
    # (1,1) has no parent, so it starts open whatever its column's progress.
    assert g.nodes[cell(g, 1, 1)].available
    assert g.buyable(100) == [cell(g, 0, 0), cell(g, 0, 1), cell(g, 1, 1)]


def test_a_node_opens_on_all_parents_and_its_column():
    g = Grid(TABLE)
    g.buy(cell(g, 0, 0))
    assert not g.nodes[cell(g, 1, 0)].available      # (0,1) not bought yet
    g.buy(cell(g, 0, 1))
    assert g.total_levels == 2
    assert g.nodes[cell(g, 1, 0)].available          # both parents, progress 2
    g.buy(cell(g, 1, 0))
    assert g.nodes[cell(g, 2, 0)].available          # progress 3 reached


def test_levels_read_their_entry_and_price_the_next():
    g = Grid(TABLE)
    a = g.nodes[cell(g, 0, 0)]
    assert a.cost == 5.0
    assert g.buy(cell(g, 0, 0)) == 5.0
    assert a.params()["bonus"] == 50 and a.cost == 10.0
    g.buy(cell(g, 0, 0))
    assert a.params()["bonus"] == 100 and a.cost is None
    assert not g.can_buy(cell(g, 0, 0), 1e9)          # last level reached


def test_rows_and_run_values():
    g = Grid(TABLE)
    for xy in ((0, 0), (0, 1), (1, 0), (1, 0)):
        g.buy(cell(g, *xy))
    names = [r["Name"] for r in g.rows()]
    assert names == ["StatBonus", "StatBonus"]        # RoomCount is run-side
    assert g.run_value("RoomCount", "rooms") == 2
    assert g.run_value("RoomCount", "rerollShrines") == 2
    assert inert(g.nodes[cell(g, 1, 1)])
    assert not inert(g.nodes[cell(g, 0, 0)])


def test_always_on_rows():
    rows = always_on_rows({"Mutations": TABLE})
    assert [r["Name"] for r in rows] == ["GetItemBack", "AdditionalRerolls"]
    assert rows[1]["Param1Name"] == "value" and rows[1]["Param1Value"] == 1
