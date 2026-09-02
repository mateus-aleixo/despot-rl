"""The hand-written run-level baseline, in one place.

Four tools and the scenario sampler each carried their own copy of this, which
was fine while the run layer had one shop action and stopped being fine the
moment it had four. Everything that wants "what a person would tell you to do"
now calls `heuristic_action`, so the baseline the agents are measured against
cannot drift between the tools measuring it.

It works on a `RunState`, not on the environment, because `tools/run_policies.py`
and the scenario sampler drive the sim directly.
"""
from __future__ import annotations

import random

from sim.run import MAX_UNIT_LEVEL, RunState


def heuristic_action(st: RunState, legal=None, rng: random.Random | None = None,
                     upgrade: str = "last"):
    """Take what a room offers, otherwise head for the boss.

    At an item shop the order is an item, then experience, then a shop upgrade.
    That ordering is a guess at what a person would do and not a measured
    optimum, but the other way round is measurably bad: experience costs 1 gold
    a click, so a baseline that buys it first drains the purse one gold at a
    time and never arms anybody -- 58% of its actions went on experience and 0%
    on items, and the squad reached the boss holding sticks.

    An item is bought only when it would actually raise the squad's Power. The
    run layer hands a bought item to whoever loses least by it, so without that
    test the baseline would happily downgrade itself.

    At a food shop it tops the larder up to `moves` feedings' worth, `moves`
    being the reserve size the hunger model itself uses (`Game.Food.moves[0]`,
    six). It buys the cheapest pack that closes the gap, and the largest pack it
    can afford when none of them does, since the big packs are the better rate
    (3.5 food per gold at the bottom of the shelf, 4.55 at the top). The
    threshold is a guess, not a measured optimum -- but ignoring food shops is
    not an option any more: a room costs one food per human, nothing else fills
    the larder, and a baseline that walks past the shops starves.

    `upgrade` moves the shop upgrade in that order: `"last"` (the default, only
    when nothing on the shelf is worth buying), `"first"` (whenever it is
    affordable) or `"never"`. It exists so `tools/shop_eval.py` can ask whether
    upgrading pays at all, separately from whether an agent has learned to do
    it -- a trained agent that never upgrades is only a mistake if upgrading is
    worth something.
    """
    legal = list(st.legal_actions() if legal is None else legal)
    if not legal:
        return None
    kinds: dict[str, list] = {}
    for a in legal:
        kinds.setdefault(a[0], []).append(a)
    here = st.rooms.rooms[st.room]

    if here.kind == "item_shop":
        if upgrade == "first" and kinds.get("upgrade_shop"):
            return kinds["upgrade_shop"][0]
        best, best_gain = None, 0.0
        for action in kinds.get("buy_item", ()):
            slot = action[1][0]
            name = st.offer[slot]
            if name is None:
                continue
            target = st.best_item_target(name)
            if target is None:
                continue
            gain = (st.item_power(name, st.squad[target].level)
                    - st.item_power(st.squad[target].item, st.squad[target].level))
            if gain > best_gain:
                best, best_gain = action, gain
        if best is not None:
            return best
        if kinds.get("buy_exp") and any(h.level < MAX_UNIT_LEVEL for h in st.squad):
            return kinds["buy_exp"][0]
        if upgrade == "last" and kinds.get("upgrade_shop"):
            return kinds["upgrade_shop"][0]

    if here.kind == "food_shop" and kinds.get("buy_food"):
        sizes, _ = st.food_packs
        want = st.food.max_moves * st.feed_cost - st.food.amount
        if want > 0:
            packs = sorted((a[1] for a in kinds["buy_food"]),
                           key=lambda i: sizes[i])
            covers = [i for i in packs if sizes[i] >= want]
            return ("buy_food", covers[0] if covers else packs[-1])

    if here.kind == "mutation" and kinds.get("take_mutation"):
        return kinds["take_mutation"][0]

    # A mutation shop's mutations cost nothing, so a baseline takes them. Which
    # one is a real question: three quarters of the pool is unimplemented in
    # this sim and a mutation aimed at a class the squad does not field does
    # nothing either, so this takes the slot that touches the most of the squad
    # among the ones the sim actually models, the way the item rule takes the
    # biggest Power gain. Ties and empty shelves fall back to the first slot.
    if here.kind == "mutation_shop" and kinds.get("buy_mutation"):
        shelf = st.rooms.rooms[st.room].offer_mutations or []

        def score(action):
            m = shelf[action[1]] if 0 <= action[1] < len(shelf) else None
            if m is None:
                return (0.0, 0.0)
            e = st.mutation_effect(m)
            return (1.0 if e["implemented"] else 0.0, e["fraction"])

        return max(kinds["buy_mutation"], key=score)

    if kinds.get("move"):
        # Hops over the room graph, not Manhattan distance on the grid: the map
        # is generated per level now and a straight line across it can cross
        # cells that hold no room.
        #
        # Over the **revealed** subgraph, not the whole map. The baseline plays
        # under the same fog the policy does, because a heuristic that can see
        # the boss from the start room is not a baseline for a policy that
        # cannot. Until the boss is found `known_to_boss` is empty, every room
        # ties, and the first key carries the decision: walk into the room that
        # has not been cleared.
        known = st.rooms.known_to_boss
        far = max(known.values(), default=1)
        return min(kinds["move"], key=lambda a: (
            st.rooms.rooms[a[1]].cleared, known.get(a[1], far)))

    return rng.choice(legal) if rng is not None else legal[0]


def heuristic(env, obs, mask):
    """`heuristic_action` as an environment policy: returns an action index."""
    legal = [env._decode(i) for i in mask.nonzero()[0]]
    pick = heuristic_action(env.state, legal)
    return env._index(pick) if pick is not None else 0
