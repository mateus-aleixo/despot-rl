"""A NodeCanvas behaviour tree executor.

Runs the trees the game ships, parsed out of the `_serializedGraph` field of the
`NC-*.asset` files. Node semantics follow NodeCanvas:

  Sequencer   children in order; first Failure fails, all Success succeeds
  Selector    children in order; first Success succeeds, all Failure fails
  dynamic     re-evaluate earlier children every tick instead of resuming at
              the running one, so a higher-priority branch can take over
  SwitchSequence  like a sequence, but remembers which child was active and
              resumes there on re-entry rather than restarting
  Interruptor decorator that aborts its running child when its condition
              becomes true (the default-attack tree inverts IsStillApplicable,
              so the swing aborts when the target stops being valid)
  ConditionalEvaluator  decorator that fails while its condition is false
  BinarySelector        condition picks child 0 or child 1
  Guard       mutex on a token, so one agent runs one guarded branch at a time
  Optional    runs its child but always succeeds

Each agent gets its own tree instance, because leaves hold per-agent state
(animation timers, which child is mid-swing).
"""
from __future__ import annotations

from enum import Enum


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class Node:
    """Base node. `children` is ordered as the graph's connections were."""

    def __init__(self, kind: str, raw: dict):
        self.kind = kind
        self.raw = raw
        self.children: list[Node] = []

    def tick(self, ctx) -> Status:
        raise NotImplementedError(self.kind)

    def reset(self) -> None:
        for c in self.children:
            c.reset()

    def __repr__(self) -> str:
        return f"<{self.kind} x{len(self.children)}>"


# ---------------------------------------------------------------- composites

class Sequencer(Node):
    def __init__(self, kind, raw):
        super().__init__(kind, raw)
        self.dynamic = bool(raw.get("dynamic"))
        self.index = 0

    def tick(self, ctx) -> Status:
        if self.dynamic:
            # re-run from the top; an earlier child failing aborts the branch
            for i, c in enumerate(self.children):
                st = c.tick(ctx)
                if st is Status.FAILURE:
                    for later in self.children[i + 1:]:
                        later.reset()
                    self.index = 0
                    return Status.FAILURE
                if st is Status.RUNNING:
                    self.index = i
                    return Status.RUNNING
            self.index = 0
            return Status.SUCCESS

        while self.index < len(self.children):
            st = self.children[self.index].tick(ctx)
            if st is Status.RUNNING:
                return Status.RUNNING
            if st is Status.FAILURE:
                self.reset()
                return Status.FAILURE
            self.index += 1
        self.reset()
        return Status.SUCCESS

    def reset(self):
        self.index = 0
        super().reset()


class Selector(Node):
    def __init__(self, kind, raw):
        super().__init__(kind, raw)
        self.dynamic = bool(raw.get("dynamic"))
        self.index = 0

    def tick(self, ctx) -> Status:
        if self.dynamic:
            for i, c in enumerate(self.children):
                st = c.tick(ctx)
                if st is Status.SUCCESS:
                    for later in self.children[i + 1:]:
                        later.reset()
                    self.index = 0
                    return Status.SUCCESS
                if st is Status.RUNNING:
                    self.index = i
                    return Status.RUNNING
            self.index = 0
            return Status.FAILURE

        while self.index < len(self.children):
            st = self.children[self.index].tick(ctx)
            if st is Status.RUNNING:
                return Status.RUNNING
            if st is Status.SUCCESS:
                self.reset()
                return Status.SUCCESS
            self.index += 1
        self.reset()
        return Status.FAILURE

    def reset(self):
        self.index = 0
        super().reset()


class SwitchSequence(Node):
    """Sequence that resumes at the child it left off on."""

    def __init__(self, kind, raw):
        super().__init__(kind, raw)
        self.index = 0

    def tick(self, ctx) -> Status:
        while self.index < len(self.children):
            st = self.children[self.index].tick(ctx)
            if st is Status.RUNNING:
                return Status.RUNNING
            if st is Status.FAILURE:
                self.index = 0
                return Status.FAILURE
            self.children[self.index].reset()
            self.index += 1
        self.index = 0
        return Status.SUCCESS

    def reset(self):
        self.index = 0
        super().reset()


class Parallel(Node):
    """Runs every child each tick.

    None of the shipped Parallel nodes set `_policy`, so they take NodeCanvas's
    default: fail as soon as any child fails, succeed once all have succeeded.
    `dynamic` re-ticks children that already succeeded instead of latching them.
    """

    def __init__(self, kind, raw):
        super().__init__(kind, raw)
        self.dynamic = bool(raw.get("dynamic"))
        self.done: set[int] = set()

    def tick(self, ctx) -> Status:
        all_done = True
        for i, c in enumerate(self.children):
            if not self.dynamic and i in self.done:
                continue
            st = c.tick(ctx)
            if st is Status.FAILURE:
                self.reset()
                return Status.FAILURE
            if st is Status.SUCCESS:
                self.done.add(i)
            else:
                all_done = False
        if all_done:
            self.reset()
            return Status.SUCCESS
        return Status.RUNNING

    def reset(self):
        self.done.clear()
        super().reset()


class Switch(Node):
    """Picks a child by a blackboard value; falls back to the first child."""

    def tick(self, ctx) -> Status:
        if not self.children:
            return Status.FAILURE
        idx = ctx.switch_index(self.raw)
        idx = max(0, min(idx, len(self.children) - 1))
        return self.children[idx].tick(ctx)


# ---------------------------------------------------------------- decorators

class Decorator(Node):
    @property
    def child(self) -> Node | None:
        return self.children[0] if self.children else None


class ConditionalEvaluator(Decorator):
    def tick(self, ctx) -> Status:
        if not ctx.eval_condition(self.raw.get("_condition"), self):
            if self.child:
                self.child.reset()
            return Status.FAILURE
        return self.child.tick(ctx) if self.child else Status.SUCCESS


class Interruptor(Decorator):
    def tick(self, ctx) -> Status:
        if ctx.eval_condition(self.raw.get("_condition"), self):
            if self.child:
                self.child.reset()
            ctx.on_interrupt(self)
            return Status.FAILURE
        return self.child.tick(ctx) if self.child else Status.SUCCESS


class BinarySelector(Decorator):
    def tick(self, ctx) -> Status:
        if not self.children:
            return Status.FAILURE
        take = 0 if ctx.eval_condition(self.raw.get("_condition"), self) else 1
        if take >= len(self.children):
            return Status.FAILURE
        for i, c in enumerate(self.children):
            if i != take:
                c.reset()
        return self.children[take].tick(ctx)


class Guard(Decorator):
    """Token mutex. Only one guarded branch per agent per token runs at a time."""

    def __init__(self, kind, raw):
        super().__init__(kind, raw)
        tok = raw.get("token")
        self.token = tok.get("_value") if isinstance(tok, dict) else str(tok)
        self.holding = False

    def tick(self, ctx) -> Status:
        holder = ctx.guards.get(self.token)
        if holder is not None and holder is not self:
            return Status.FAILURE
        ctx.guards[self.token] = self
        self.holding = True
        st = self.child.tick(ctx) if self.child else Status.SUCCESS
        if st is not Status.RUNNING:
            ctx.guards.pop(self.token, None)
            self.holding = False
        return st

    def reset(self):
        self.holding = False
        super().reset()


class Optional(Decorator):
    def tick(self, ctx) -> Status:
        if self.child:
            st = self.child.tick(ctx)
            if st is Status.RUNNING:
                return Status.RUNNING
        return Status.SUCCESS


# --------------------------------------------------------------------- leaves

class ConditionNode(Node):
    def tick(self, ctx) -> Status:
        ok = ctx.eval_condition(self.raw.get("_condition"), self)
        return Status.SUCCESS if ok else Status.FAILURE


class ActionNode(Node):
    def tick(self, ctx) -> Status:
        return ctx.run_action(self.raw.get("_action"), self)

    def reset(self):
        self.state = None
        super().reset()


class SubTree(Node):
    """A nested graph. The runtime injects the agent's action subtrees here."""

    def __init__(self, kind, raw):
        super().__init__(kind, raw)
        self.subtree: Node | None = None

    def tick(self, ctx) -> Status:
        if self.subtree is None:
            return Status.FAILURE
        return self.subtree.tick(ctx)

    def reset(self):
        if self.subtree is not None:
            self.subtree.reset()
        super().reset()


class ActionScope(Node):
    """Binds one of the unit's actions while its subtree runs.

    The tree leaves are generic -- NC-DefaultAttack, NC-Skill and NC-HealCSkill
    all drive `BT.DefaultAttack.*` -- so which action they act on has to come
    from the surrounding scope, the way the runtime binds an M_Action.
    """

    def __init__(self, action, child: Node):
        super().__init__("ActionScope", {})
        self.action = action
        self.children = [child]

    def tick(self, ctx) -> Status:
        prev = getattr(ctx, "action", None)
        ctx.action = self.action
        try:
            return self.children[0].tick(ctx)
        finally:
            ctx.action = prev

    def reset(self):
        self.action.swinging = False
        super().reset()


KINDS = {
    "Sequencer": Sequencer,
    "Selector": Selector,
    "SwitchSequence": SwitchSequence,
    "Parallel": Parallel,
    "Switch": Switch,
    "ConditionalEvaluator": ConditionalEvaluator,
    "Interruptor": Interruptor,
    "BinarySelector": BinarySelector,
    "Guard": Guard,
    "Optional": Optional,
    "ConditionNode": ConditionNode,
    "ActionNode": ActionNode,
    "SubTree": SubTree,
}


class UnknownNodeType(NotImplementedError):
    pass


def build_tree(graph: dict) -> Node:
    """Turn a parsed NC-*.json graph into a runnable tree rooted at node "0"."""
    raw_nodes = {n["$id"]: n for n in graph["nodes"]}
    order: dict[str, list[str]] = {}
    for c in graph.get("connections", []):
        src = c.get("_sourceNode", {}).get("$ref")
        tgt = c.get("_targetNode", {}).get("$ref")
        order.setdefault(src, []).append(tgt)

    built: dict[str, Node] = {}

    def make(nid: str) -> Node:
        if nid in built:
            return built[nid]
        raw = raw_nodes[nid]
        kind = raw.get("$type", "?").split(".")[-1]
        cls = KINDS.get(kind)
        if cls is None:
            raise UnknownNodeType(f"{kind} (node {nid})")
        node = cls(kind, raw)
        built[nid] = node
        for k in order.get(nid, []):
            node.children.append(make(k))
        return node

    return make("0")
