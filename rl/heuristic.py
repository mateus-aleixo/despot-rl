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

from sim.run import MAX_UNIT_LEVEL, RunState, consumable_family

# What the baseline takes from a consumable shop, best first, and what it sells
# on sight: the consumables it has no rule for, five gold each.
TAKE_ORDER = ("dejavu", "heal_one", "heal_all", "resurrection", "reserve_humans",
              "reserve_ally", "cleaner", "anti_hunger", "event")
USELESS_TO_BASELINE = ("unmodelled", "teleport", "reroll")


def heuristic_action(st: RunState, legal=None, rng: random.Random | None = None,
                     upgrade: str = "last", quests: str = "pass"):
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

    `quests` is the same kind of switch for the level's quest: `"pass"` (the
    default) answers a quest dialog when the squad walks into one and goes no
    further out of its way, `"follow"` walks to the extra room and back for the
    reward. See `_quest_target` for why the default is the first.
    """
    legal = list(st.legal_actions() if legal is None else legal)
    if not legal:
        return None
    kinds: dict[str, list] = {}
    for a in legal:
        kinds.setdefault(a[0], []).append(a)
    here = st.rooms.rooms[st.room]

    # A dialog is modal: the only legal actions are its choices.
    if kinds.get("choose"):
        value = _quest_choice_value if st.quest_dialog is not None else _choice_value
        return max(kinds["choose"], key=lambda a: (value(st, a[1]), -a[1]))

    # Consumables. Sell what the sim gives no effect, heal a badly wounded
    # squad, bring back the dead lying here, and call reserves at once.
    held = {a[1]: consumable_family(st.consumables[a[1]].row)
            for a in kinds.get("use_consumable", ())}
    for a in kinds.get("sell_consumable", ()):
        if consumable_family(st.consumables[a[1]].row) in USELESS_TO_BASELINE:
            return a
    hp = [h.hp for h in st.squad] or [1.0]
    for k, fam in held.items():
        if ((fam == "heal_one" and min(hp) < 0.5)
                or (fam == "heal_all" and sum(hp) / len(hp) < 0.7)
                or fam in ("resurrection", "reserve_humans", "reserve_ally", "event")):
            return ("use_consumable", k)
    # The boss room once won: take what the shop shows while there is room,
    # most useful first, then go down the pipe.
    if kinds.get("take_consumable"):
        shelf = here.consumable_offer or []
        return min(kinds["take_consumable"], key=lambda a: (
            TAKE_ORDER.index(consumable_family(shelf[a[1]]))
            if consumable_family(shelf[a[1]]) in TAKE_ORDER else len(TAKE_ORDER)))
    if kinds.get("descend") and not kinds.get("buy_item"):
        return kinds["descend"][0]

    if here.has("item_shop"):
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

        best = max(kinds["buy_mutation"], key=score)
        # The shop rerolls for nothing, so a shelf with nothing on it the sim
        # models is rerolled while rerolls last.
        if score(best)[0] == 0.0 and kinds.get("reroll_mutations"):
            return kinds["reroll_mutations"][0]
        return best

    known = st.rooms.known_to_boss
    far = max(known.values(), default=1)
    moves = kinds.get("move", [])
    # `quests="follow"`: an accepted quest sends the squad to its extra room,
    # and an acquired one back to the quest room for the reward. See
    # `_quest_target`. Off by default, because it was measured to lose: on 240
    # seeds following cost -0.113 [-0.199, -0.026] levels against passing the
    # quests by, for 25 or 30 gold and a mutation the sim does not model yet.
    target = _quest_target(st) if quests == "follow" else None
    if target is not None and moves:
        to = st.rooms._distances(target, known_only=True)
        here_d = to.get(st.room)
        # Only on a larder that pays for the walk there: a side trip on an
        # empty one strands a squad too small to sacrifice from.
        affordable = (here_d is not None
                      and st.food.amount >= st.feed_cost * here_d)
        nearer = [a for a in moves if affordable
                  and to.get(a[1], here_d) < here_d]
        if nearer:
            return nearer[0]
    # A teleport costs no food, so on equal terms it would win every time and
    # two cleared portals would trade the squad back and forth for ever. It is
    # taken only when it reaches something: a room nobody has entered, a room
    # strictly nearer the boss as far as the map shows, or, when the larder
    # cannot pay for the next door, a food shop with a pack the purse covers.
    teleports = [a for a in kinds.get("teleport", ())
                 if _worth_teleporting(st, a[1], known, far, hungry=not moves)]
    if moves or teleports:
        # Hops over the room graph, not Manhattan distance on the grid: the map
        # is generated per level now and a straight line across it can cross
        # cells that hold no room.
        #
        # Over the **revealed** subgraph, not the whole map. The baseline plays
        # under the same fog the policy does, because a heuristic that can see
        # the boss from the start room is not a baseline for a policy that
        # cannot. Until the boss is found `known_to_boss` is empty, every room
        # ties, and the first key carries the decision: walk into the room that
        # has not been cleared. A door wins a tie against a teleport.
        step = min(moves + teleports, key=lambda a: (
            st.rooms.rooms[a[1]].cleared, known.get(a[1], far),
            a[0] == "teleport"))
        # About to walk into a fight: a cleaner wins it outright (not on the
        # boss, where it would be wasted), an AntiHunger lifts the penalties.
        target = st.rooms.rooms[step[1]]
        if not target.cleared:
            for k, fam in held.items():
                if ((fam == "cleaner" and target.kind != "boss")
                        or (fam == "anti_hunger" and st.food.hunger_level > 0)):
                    return ("use_consumable", k)
        return step

    # Never a teleport here: whatever is left is a teleport that leads nowhere
    # (see above) or something that changes the state, and only the second can
    # end a run that is going nowhere.
    rest = [a for a in legal if a[0] != "teleport"] or legal
    return rng.choice(rest) if rng is not None else rest[0]


# What a dialog event is worth to the baseline, in gold. Guesses, not
# measurements: a mutation or an item about a shop item's price, a human about
# what the shop charges plus what they carry, food at its shop rate, and a
# wound or an ambush as a loss.
_EVENT_GOLD = {"GiveMutation": 10.0, "GiveItem": 15.0, "GiveUnit": 8.0,
               "CloneUnit": 8.0, "RemoveUnit": -15.0, "Ambush": -10.0}


def _event_value(e: dict) -> float:
    cls, p = e.get("className"), e.get("parameters") or {}

    def amount(default=0.0):
        try:
            return float(p.get("amount", default))
        except (TypeError, ValueError):
            return default

    if cls in ("PlusGold",):
        return amount(10.0)
    if cls == "MinusGold":
        return -amount(5.0)
    if cls == "PlusFood":
        return amount() / 4.0
    if cls == "MinusFood":
        return -amount(5.0) / 4.0
    if cls == "GiveExperience":
        return amount() / 100.0
    if cls == "RemoveHealth":
        return -float(p.get("percent") or 0.0) / 5.0
    if cls == "GiveUnit":
        return _EVENT_GOLD["GiveUnit"] * max(1, len(p.get("classes") or [None]))
    return _EVENT_GOLD.get(cls, 0.0)


def _choice_value(st: RunState, index: int) -> float:
    """A choice's expected worth, one node deep, by outcome weight."""
    d = st.dialog
    choice = (d.node.get("choices") or [])[index]
    if str(choice.get("link")) == "#":
        return -1.0                      # back to the start: a loop
    value = sum(_event_value(e) for e in choice.get("events") or [])
    outs = choice.get("outcomes") or []
    if outs:
        weights = [float(o.get("weight") or 1.0) for o in outs]
        total = sum(weights)
        value += sum(w / total * sum(_event_value(e) for e in o.get("events") or [])
                     for w, o in zip(weights, outs))
    return value


def _quest_choice_value(st: RunState, index: int) -> float:
    """A quest choice's worth to the baseline, from what the branch would do.

    The outcomes it applies, valued as the dialog events are, except that a
    mutation the sim does not model is worth nothing to it; a quest kept open
    (Accepted) a little, for what finishing it may pay; and a Novice the Pit
    takes as much as a human costs. Guesses, not measurements, like
    `_EVENT_GOLD`.
    """
    run = st.quest_dialog
    outcomes = st.quest.spec.get("outcomes") or []
    value = 0.0
    for kind, arg in run.preview(st, index):
        if kind == "outcome" and 1 <= int(arg) <= len(outcomes):
            for e in outcomes[int(arg) - 1]:
                if e.get("className") == "GiveMutation":
                    m = st.mutation_by_id(int((e.get("parameters") or {}).get("id") or 0))
                    if m is not None and st.mutation_effect(m)["implemented"]:
                        value += _EVENT_GOLD["GiveMutation"]
                else:
                    value += _event_value(e)
        elif kind == "status" and arg == 1:
            value += 5.0
        elif kind == "call" and arg == "KillNovices":
            value -= 5 * _EVENT_GOLD["GiveUnit"]
    return value


def _quest_target(st: RunState) -> str | None:
    """Where the level's quest wants the squad, if the baseline follows it.

    An accepted quest wants the room it points at on the map (`Quest.marked`):
    its extra room while that room still has something to give (Cube and Rat
    hand it over whenever the room is visited after the quest was accepted,
    Sci only after a fight there), or the room the Reaper's hater or the Bot
    is in. The baseline goes only when an extra room's pack is no stronger than
    the squad by the game's Power statistic, the same readiness test the item
    rule uses; that threshold is a guess. An acquired quest wants the quest
    room.
    """
    q = st.quest
    if q is None or q.room is None or type(q).__name__ == "InertQuest":
        return None
    if q.status == 4:
        return q.room
    marked = q.marked(st)
    if q.status != 1 or not marked:
        return None
    rid = marked[0]
    room = st.rooms.rooms[rid]
    if q.name == "Sci" and room.cleared:
        return None
    if not room.cleared:
        enemies, _ = st.quest_enemies(room)
        if sum(e.power for e in enemies) > st.squad_power():
            return None
    return rid


def _worth_teleporting(st: RunState, rid: str, known: dict, far: int,
                       hungry: bool) -> bool:
    room = st.rooms.rooms[rid]
    if not room.cleared:
        return True
    if known and known.get(rid, far) < known.get(st.room, far):
        return True
    if hungry and room.kind == "food_shop":
        _, costs = st.food_packs
        return any(held > 0 and st.gold >= costs[i]
                   for i, held in enumerate(room.food_stock or []))
    return False


def heuristic(env, obs, mask):
    """`heuristic_action` as an environment policy: returns an action index."""
    legal = [env._decode(i) for i in mask.nonzero()[0]]
    pick = heuristic_action(env.state, legal)
    return env._index(pick) if pick is not None else 0
