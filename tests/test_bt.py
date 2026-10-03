"""Behaviour tree semantics (sim/bt.py), on small hand-built graphs.

The graphs use the shape of the NodeCanvas JSON the game ships ("$id",
"$type", and connections by "$ref"), and the leaves run against a scripted
context instead of a battle.
"""
import pytest

from sim.bt import Status, UnknownNodeType, build_tree

S, F, R = Status.SUCCESS, Status.FAILURE, Status.RUNNING


class Ctx:
    """Conditions read a dict; each action replays a script, repeating its last status."""

    def __init__(self, conditions=None, actions=None):
        self.conditions = dict(conditions or {})
        self.scripts = {k: list(v) for k, v in (actions or {}).items()}
        self.ticked: list[str] = []
        self.interrupts = 0
        self.guards: dict = {}

    def eval_condition(self, cond, node):
        return self.conditions[cond["name"]]

    def run_action(self, action, node):
        name = action["name"]
        self.ticked.append(name)
        script = self.scripts[name]
        return script.pop(0) if len(script) > 1 else script[0]

    def on_interrupt(self, node):
        self.interrupts += 1

    def switch_index(self, raw):
        return raw["index"]

    def tick(self, tree):
        self.ticked = []
        return tree.tick(self)


def act(name):
    return ("ActionNode", {"_action": {"name": name}})


def cond(name):
    return ("ConditionNode", {"_condition": {"name": name}})


def tree(nodes: dict, edges: list):
    """`nodes` is {id: (kind, fields)} with the root at "0"; `edges` are in child order."""
    return build_tree({
        "nodes": [{"$id": nid, "$type": f"NodeCanvas.BehaviourTrees.{kind}", **fields}
                  for nid, (kind, fields) in nodes.items()],
        "connections": [{"_sourceNode": {"$ref": s}, "_targetNode": {"$ref": t}}
                        for s, t in edges],
    })


def composite(kind, children, **fields):
    """A root of `kind` over leaf children, wired in the order given."""
    nodes = {"0": (kind, fields)}
    nodes.update({str(i + 1): child for i, child in enumerate(children)})
    return tree(nodes, [("0", str(i + 1)) for i in range(len(children))])


def test_children_follow_connection_order():
    root = tree({"0": ("Sequencer", {}), "1": act("a"), "2": act("b")},
                [("0", "2"), ("0", "1")])
    ctx = Ctx(actions={"a": [S], "b": [S]})
    assert ctx.tick(root) is S
    assert ctx.ticked == ["b", "a"]


def test_unknown_node_type_is_refused():
    with pytest.raises(UnknownNodeType):
        tree({"0": ("Mystery", {})}, [])


def test_sequencer_stops_at_the_first_failure():
    root = composite("Sequencer", [act("a"), act("b"), act("c")])
    ctx = Ctx(actions={"a": [S], "b": [F], "c": [S]})
    assert ctx.tick(root) is F
    assert ctx.ticked == ["a", "b"]


def test_sequencer_resumes_at_the_running_child():
    root = composite("Sequencer", [act("a"), act("b")])
    ctx = Ctx(actions={"a": [S], "b": [R, S]})
    assert ctx.tick(root) is R
    assert ctx.tick(root) is S
    assert ctx.ticked == ["b"]


def test_dynamic_sequencer_rechecks_earlier_children():
    root = composite("Sequencer", [cond("target_alive"), act("swing")], dynamic=True)
    ctx = Ctx(conditions={"target_alive": True}, actions={"swing": [R]})
    assert ctx.tick(root) is R
    ctx.conditions["target_alive"] = False
    assert ctx.tick(root) is F
    assert ctx.ticked == []


def test_selector_takes_the_first_success():
    root = composite("Selector", [act("a"), act("b"), act("c")])
    ctx = Ctx(actions={"a": [F], "b": [S], "c": [S]})
    assert ctx.tick(root) is S
    assert ctx.ticked == ["a", "b"]


def test_selector_fails_when_every_child_fails():
    root = composite("Selector", [act("a"), act("b")])
    assert Ctx(actions={"a": [F], "b": [F]}).tick(root) is F


def test_dynamic_selector_lets_a_higher_priority_branch_take_over():
    root = tree({"0": ("Selector", {"dynamic": True}),
                 "1": ("ConditionalEvaluator", {"_condition": {"name": "threat"}}),
                 "2": act("flee"),
                 "3": act("wander")},
                [("0", "1"), ("1", "2"), ("0", "3")])
    ctx = Ctx(conditions={"threat": False}, actions={"flee": [R], "wander": [R]})
    assert ctx.tick(root) is R and ctx.ticked == ["wander"]
    ctx.conditions["threat"] = True
    assert ctx.tick(root) is R and ctx.ticked == ["flee"]


def test_parallel_latches_successes_and_fails_fast():
    root = composite("Parallel", [act("a"), act("b")])
    ctx = Ctx(actions={"a": [S], "b": [R, R, S]})
    assert ctx.tick(root) is R and ctx.ticked == ["a", "b"]
    assert ctx.tick(root) is R and ctx.ticked == ["b"]
    assert ctx.tick(root) is S
    root = composite("Parallel", [act("a"), act("b")])
    ctx = Ctx(actions={"a": [F], "b": [R]})
    assert ctx.tick(root) is F and ctx.ticked == ["a"]


def test_interruptor_aborts_its_running_child():
    root = tree({"0": ("Interruptor", {"_condition": {"name": "invalid"}}), "1": act("swing")},
                [("0", "1")])
    ctx = Ctx(conditions={"invalid": False}, actions={"swing": [R]})
    assert ctx.tick(root) is R
    ctx.conditions["invalid"] = True
    assert ctx.tick(root) is F
    assert ctx.interrupts == 1 and ctx.ticked == []


def test_conditional_evaluator_gates_its_child():
    root = tree({"0": ("ConditionalEvaluator", {"_condition": {"name": "ready"}}),
                 "1": act("cast")}, [("0", "1")])
    ctx = Ctx(conditions={"ready": False}, actions={"cast": [S]})
    assert ctx.tick(root) is F and ctx.ticked == []
    ctx.conditions["ready"] = True
    assert ctx.tick(root) is S and ctx.ticked == ["cast"]


def test_binary_selector_picks_a_branch_by_its_condition():
    root = tree({"0": ("BinarySelector", {"_condition": {"name": "ranged"}}),
                 "1": act("shoot"), "2": act("charge")},
                [("0", "1"), ("0", "2")])
    ctx = Ctx(conditions={"ranged": True}, actions={"shoot": [S], "charge": [S]})
    assert ctx.tick(root) is S and ctx.ticked == ["shoot"]
    ctx.conditions["ranged"] = False
    assert ctx.tick(root) is S and ctx.ticked == ["charge"]


def test_guard_lets_one_branch_hold_a_token():
    first = tree({"0": ("Guard", {"token": {"_value": "attack"}}), "1": act("a")}, [("0", "1")])
    second = tree({"0": ("Guard", {"token": {"_value": "attack"}}), "1": act("b")}, [("0", "1")])
    ctx = Ctx(actions={"a": [R, S], "b": [S]})
    assert ctx.tick(first) is R
    assert ctx.tick(second) is F and ctx.ticked == []
    assert ctx.tick(first) is S
    assert ctx.tick(second) is S and ctx.ticked == ["b"]


def test_optional_succeeds_whatever_its_child_does():
    root = tree({"0": ("Optional", {}), "1": act("a")}, [("0", "1")])
    assert Ctx(actions={"a": [F]}).tick(root) is S
    assert Ctx(actions={"a": [R]}).tick(root) is R


def test_switch_clamps_its_index():
    for index, want in ((5, ["last"]), (-1, ["first"]), (0, ["first"])):
        root = composite("Switch", [act("first"), act("last")], index=index)
        ctx = Ctx(actions={"first": [S], "last": [S]})
        assert ctx.tick(root) is S and ctx.ticked == want
