"""Condition and action leaves for the game's behaviour trees.

The leaves are generic: NC-DefaultAttack, NC-Skill and NC-HealCSkill all drive
the same `BT.DefaultAttack.*` types. What they act on is the action currently
bound by the enclosing `ActionScope`, mirroring how the runtime binds an
`M_Action`. So range, cooldown, mana cost and effect are read off
`ctx.action`, not off the unit.

Every leaf the shipped trees use must be declared here. Leaves for systems this
sim does not model yet are declared explicitly as NOT_MODELLED so they fail
their branch, which is the correct behaviour for a unit not in that state. An
undeclared leaf raises, so a tree quietly doing nothing is not mistaken for a
tree behaving.

The cycle these implement, read off NC-DefaultAttack and NC-Skill:

    gates (mana, cooldown, range) -> StartExecution -> attack animation
    -> AfterAttackAnimation (the effect fires) -> recovery animation
    -> AfterRecoveryAnimation
"""
from __future__ import annotations

from .bt import Status

NOT_MODELLED = {
    # statuses this sim still does not have
    "OnFlyKnockback", "AfterKnockbackFlyOutAnimation",
    "AfterKnockbackFlyInAnimation", "IsCatatonic", "SetPanicTarget",
    # shop / out-of-fight flow
    "IsAwaitingItem", "OnAwaitingItem", "ReceiveItem",
    "HasGroundTarget", "WaitUntilVisible", "IsAnimationFinished",
    # unit-specific skills that are not class skills
    "BT.CollectLeeches.NeedsHealing",
}

_PREFIX_NOT_MODELLED = ("BT.Cultist.", "BT.Charge.", "BT.Devour.", "BT.Mage.",
                        "BT.Scientist.", "BT.Swoop.", "BT.ChannelingAttack.",
                        "BT.AlternatingAttack.")


class UnknownLeaf(NotImplementedError):
    pass


class BTContext:
    """Binds a tree to one agent for one tick. `action` is set by ActionScope."""

    def __init__(self, agent, battle):
        self.agent = agent
        self.battle = battle
        self.dt = battle.dt
        self.guards: dict = agent.guards
        self.action = None

    # -- dispatch ----------------------------------------------------------
    @staticmethod
    def _type_of(spec) -> str:
        if not isinstance(spec, dict):
            return "?"
        return spec.get("$type", "?")

    def eval_condition(self, spec, node) -> bool:
        """Evaluate a condition, honouring NodeCanvas's `_invert`."""
        if not isinstance(spec, dict):
            return True
        name = self._type_of(spec)
        if name == "NodeCanvas.Framework.ConditionList":
            subs = spec.get("conditions", [])
            # checkMode 1 is ANY; the default is ALL.
            results = [self.eval_condition(s, node) for s in subs]
            value = any(results) if spec.get("checkMode") == 1 else all(results)
        else:
            value = self._condition(name, node)
        return (not value) if spec.get("_invert") else value

    def run_action(self, spec, node) -> Status:
        return self._action(self._type_of(spec), node)

    def switch_index(self, raw) -> int:
        return 0

    def on_interrupt(self, node) -> None:
        """An Interruptor aborted a running branch: the cast is off."""
        if self.action is not None:
            self.action.swinging = False
            self.action.anim_timer = 0.0

    # -- conditions --------------------------------------------------------
    def _condition(self, name: str, node) -> bool:
        a, b, act = self.agent, self.battle, self.action
        if name in NOT_MODELLED or name.startswith(_PREFIX_NOT_MODELLED):
            return False

        if name == "ShouldDie":
            return a.hp <= 0.0
        if name == "IsStunned":
            return a.has("stun")
        if name == "IsKnockedBack":
            return a.knockback is not None
        if name == "IsPanicked":
            return a.has("panic")
        if name == "IsSpawning":
            return a.spawning
        if name == "IsFight":
            return b.in_fight
        if name == "IsNotFight":
            return not b.in_fight
        if name == "Team2IsDying":
            return not any(o.alive for o in b.agents if o.team == 1)
        if name == "CanFight":
            return a.alive and a.target is not None and a.target.alive
        if name == "IsMovingWithoutActions":
            return a.target is None
        if name in ("ReachedTarget", "BT.DefaultAttack.ReachedTarget",
                    "BT.DefaultAttack.ReadyToAct"):
            # SkillRange^2 > sqrDistance, verified against the binary.
            if act is not None and act.name != "attack" and a.has("silence"):
                return False        # silenced: skills gated, plain attacks are not
            if act is not None and act.is_self_cast:
                return True
            t = a.target
            if t is None or not t.alive:
                return False
            rng = b.effective_range(a, act, t)
            dx, dy = t.x - a.x, t.y - a.y
            return rng * rng > dx * dx + dy * dy
        if name == "BT.DefaultAttack.IsStillApplicable":
            if act is not None and act.is_self_cast:
                return a.alive
            t = a.target
            return t is not None and t.alive
        if name == "BT.DefaultAttack.IsNotOnCooldown":
            return act is None or act.cooldown_left <= 0.0
        if name == "BT.DefaultAttack.IsEnoughMana":
            return act is None or a.mana >= act.mana_cost
        if name == "BT.DefaultAttack.IsRunningOut":
            return False
        raise UnknownLeaf(f"condition {name!r}")

    # -- actions -----------------------------------------------------------
    def _action(self, name: str, node) -> Status:
        a, b, act = self.agent, self.battle, self.action
        if name in NOT_MODELLED or name.startswith(_PREFIX_NOT_MODELLED):
            return Status.FAILURE

        if name == "Stunned":
            a.intent = "stop"
            return Status.RUNNING if a.has("stun") else Status.SUCCESS
        if name == "OnPushKnockback":
            a.intent = "stop"
            return Status.RUNNING if a.knockback is not None else Status.SUCCESS
        if name in ("PanicInPlace", "PanicFollowPath"):
            # Panicked units stop fighting; fleeing is not modelled, so they
            # simply do not act while it lasts.
            a.intent = "stop"
            return Status.RUNNING if a.has("panic") else Status.SUCCESS

        # NodeCanvas's own no-op. It was in NOT_MODELLED, so it returned
        # FAILURE, which quietly changed what a skill does when it finishes:
        # NC-Skill's cycle ends on this node, so a failing one made every skill
        # yield its tick to the default attack. `EmptyAction.OnUpdate` sets the
        # node's status to Success, so it succeeds here too, and NC-Skill now
        # ends its cycle the way NC-DefaultAttack does.
        if name == "EmptyAction":
            return Status.SUCCESS

        if name == "Die":
            a.dying = True
            a.intent = "stop"
            return Status.SUCCESS
        if name == "BecomeCorpse":
            a.corpse = True
            return Status.SUCCESS
        if name == "Spawn":
            a.anim_timer = b.a.spawn_anim_s
            return Status.SUCCESS
        if name == "EndSpawn":
            a.spawning = False
            return Status.SUCCESS

        if name == "WaitForAnimation":
            timer_owner = act if act is not None else a
            if timer_owner.anim_timer > 0.0:
                timer_owner.anim_timer -= self.dt
                return Status.RUNNING
            return Status.SUCCESS

        if name in ("FollowPath", "BT.DefaultAttack.FollowPath", "BT.DefaultAttack.Move"):
            if a.target is None:
                return Status.FAILURE
            a.intent = "follow"
            return Status.RUNNING

        if name == "BT.DefaultAttack.StopIfRunning":
            if a.intent == "follow" or (a.vx or a.vy):
                a.intent = "stop"
                return Status.SUCCESS
            return Status.FAILURE

        if name == "BT.DefaultAttack.IsNotOnCooldownContinuous":
            a.intent = "stop"
            return Status.SUCCESS if (act is None or act.cooldown_left <= 0.0) else Status.RUNNING

        if name == "BT.DefaultAttack.StartExecution":
            a.intent = "stop"
            act.anim_timer = b.a.attack_anim_s
            act.swinging = True
            # M_Action tracks `elapsedCooldown`, time since the cast, so the
            # cooldown runs through both animations rather than after them.
            act.cooldown_left = act.cooldown
            a.mana -= act.mana_cost
            b.casts[act.name] = b.casts.get(act.name, 0) + 1
            return Status.SUCCESS

        if name == "BT.DefaultAttack.AfterAttackAnimation":
            if act.effect is not None:
                act.effect(act, a, b)
            # `C_OnSkillCastedSkill` hangs off the skill cast, not the swing,
            # so the default attack does not fire it.
            if a.on_cast and act.name != "attack":
                from .mutations import fire_on_cast
                fire_on_cast(b, a, act)
            act.swinging = False
            act.anim_timer = b.a.recovery_anim_s
            return Status.SUCCESS

        if name == "BT.DefaultAttack.AfterRecoveryAnimation":
            act.swinging = False
            return Status.SUCCESS

        raise UnknownLeaf(f"action {name!r}")
