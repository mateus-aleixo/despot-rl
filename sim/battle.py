"""The reference battle simulation.

Fixed-timestep loop. Each agent runs the game's own behaviour trees: NC-BaseUnit
with one subtree per action injected into the dynamic Selector the runtime fills
via `C_Unit.TryResolveAction`. Actions are the default attack plus the unit's
class skill; the tree leaves are generic and act on whichever action the
enclosing `ActionScope` has bound.

Verified against the game binary:
  * damage: `CS_Damage.Apply` runs resistance, then armor
      resistance:  amount = (1 - resistance) * amount
      armor:       amount = max(amount - (armor + bonusArmor) * armorMult,
                                min(max(amount, 0), 1.0))
  * mana arrives through the damage pipeline: `CS_Damage.InflictDamage` tests
    `DamageType.Mana` (0x20) and routes that branch to get_mana/set_mana
  * in-range test: `BT.DefaultAttack.ReachedTarget` is
      SkillRange^2 > sqrDistance   (plain centre-to-centre, no radii)
  * SkillRange comes from the class-skill group if there is one, else the
    unit's own Range (`C_Action.get_SkillRange`)
  * an item's base value adds: Novice Speed 80 + broadsword Speed 20 = 100,
    matching `UnitMovement.speed` on the shipped Swordsman prefab. Its
    per-level term does not add -- see `sim/spec.py`
  * class skill levels: `Meta.Classes[cls].Skill` plus ClassSkills.json's
    `HumansRequired`

Everything else that carries a number is declared in `assumptions.py`.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
from dataclasses import dataclass, field

from .actions import Action, build_actions, resolve_class_skills
from .assumptions import DEFAULT, Assumptions
from .bt import ActionScope, Node, Status, build_tree
from .bt_leaves import BTContext
from .nav import Grid, PathFollower, astar
from .orca import new_velocity
from .spec import UnitSpec, build_unit

RVO_TIME_HORIZON = 0.5
RVO_MAX_NEIGHBOURS = 10
PICK_NEXT_WAYPOINT_DIST = 12.0

BT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "extracted" / "bt"
_GRAPH_CACHE: dict[str, dict] = {}


def load_graph(name: str) -> dict:
    if name not in _GRAPH_CACHE:
        _GRAPH_CACHE[name] = json.loads((BT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return _GRAPH_CACHE[name]


def build_unit_tree(actions: list[Action]) -> Node:
    """Base tree with one ActionScope-wrapped subtree per action, by priority."""
    root = build_tree(load_graph("NC-BaseUnit"))
    slot = _find_action_slot(root)
    if slot is None:
        raise RuntimeError("no action slot found in NC-BaseUnit")
    for act in sorted(actions, key=lambda a: -a.priority):
        slot.children.append(ActionScope(act, build_tree(load_graph(act.tree_name))))
    return root


def _find_action_slot(node: Node) -> Node | None:
    """The empty dynamic Selector under BinarySelector<CanFight>."""
    if node.kind == "BinarySelector":
        cond = node.raw.get("_condition") or {}
        if cond.get("$type") == "CanFight" and node.children:
            first = node.children[0]
            if first.kind == "Selector" and not first.children:
                return first
    for c in node.children:
        found = _find_action_slot(c)
        if found is not None:
            return found
    return None


def apply_damage(amount: float, armor: float, resistance: float,
                 magical: bool = False, armor_mult: float = 1.0,
                 bonus_armor: float = 0.0) -> float:
    """Damage after mitigation. Mirrors CS_Damage.Apply's order."""
    if magical:
        amount = (1.0 - resistance) * amount
    else:
        eff_armor = (armor + bonus_armor) * armor_mult
        floor = min(max(amount, 0.0), 1.0)
        amount = max(amount - eff_armor, floor)
    return amount


# `DamageType`, verbatim from the dumped enum. It is a flags set, and the
# agent-level passives gate on it: a skill with `damageType: Physical` fires
# only when the damage it sees carries that flag, and `Secondary` (splash and
# other knock-on damage) never triggers a passive at all.
DT_PHYSICAL = 1
DT_MAGICAL = 2
DT_DIRECT = 4
DT_HEAL = 8
DT_REFLECTED = 16
DT_MANA = 32
DT_SECONDARY = 64
DT_CANT_BE_EVADED = 128
DT_EXCEED = 256
DT_HEAL_CLASS_SKILL = 512

# `C_EvasionSkill.OnTryToEvade` builds its jump-back with these two literals,
# read out of the two rip-relative floats it passes to `M_Knockback.Get`.
JUMP_BACK_SPEED = 150.0
JUMP_BACK_ACCEL = -280.0

DAMAGE_TYPES = {
    "Physical": DT_PHYSICAL, "Magical": DT_MAGICAL, "Direct": DT_DIRECT,
    "Heal": DT_HEAL, "Reflected": DT_REFLECTED, "Mana": DT_MANA,
    "Secondary": DT_SECONDARY, "CantBeEvaded": DT_CANT_BE_EVADED,
    "Exceed": DT_EXCEED, "HealClassSkill": DT_HEAL_CLASS_SKILL,
}


def damage_type_mask(name: str | None) -> int:
    """A `damageType` parameter as its flags value. Absent means "any"."""
    if not name:
        return 0
    return sum(DAMAGE_TYPES.get(part.strip(), 0) for part in str(name).split(","))


# The stat names a timed status can carry, mapped onto what the agent folds
# them into. `M_StatBonusStatus` is one (stat, value, percentage) triple, and
# every buff in this sim -- class skill, unit skill or mutation -- is a list of
# them with a duration.
BUFF_STATS = ("armor", "speed", "attack_speed", "damage", "resistance", "health")

STAT_TO_BUFF = {
    "Armor": "armor", "Speed": "speed", "AttackSpeed": "attack_speed",
    "Damage": "damage", "Resistance": "resistance", "Health": "health",
}


@dataclass
class Buff:
    remaining: float
    # (stat, amount, percentage) triples, the shape M_StatBonusStatus carries.
    stats: tuple = ()


@dataclass
class SpiritLink:
    remaining: float
    share_pct: float
    members: list = field(default_factory=list)


@dataclass
class Agent:
    spec: UnitSpec
    team: int
    x: float
    y: float
    hp: float
    mana: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    target: "Agent | None" = None
    intent: str = "stop"
    anim_timer: float = 0.0
    spawning: bool = False
    dying: bool = False
    corpse: bool = False
    summoned: bool = False
    tree: Node | None = None
    actions: list[Action] = field(default_factory=list)
    guards: dict = field(default_factory=dict)
    follower: PathFollower | None = None
    repath_in: float = 0.0
    # passive class skills
    crit_chance: float = 0.0
    crit_mult: float = 1.0
    attack_speed_pct: float = 0.0
    dodge_cooldown: float = 0.0
    dodge_ready_in: float = 0.0
    buffs: list[Buff] = field(default_factory=list)
    link: SpiritLink | None = None
    # per-unit skills
    splash: tuple[float, float] | None = None      # (percent, radius)
    # One entry per evasion source, because `C_EvasionSkill` is one controller
    # per skill and each subscribes to the damage separately: two sources are
    # two independent rolls, not the better of the two.
    evasions: list = field(default_factory=list)   # (chance, threshold, jump_back)
    # Same reasoning for vampirism: one `C_VampirismSkill` per source, each
    # with its own damage addon, so two sources heal twice.
    vampirisms: list = field(default_factory=list)  # (value, percentage)
    fury_per_stack: float = 0.0
    fury_stacks: int = 0
    regen: tuple[str, float, float] | None = None  # (stat, period, value)
    regen_timer: float = 0.0
    on_death: list = field(default_factory=list)   # (cs_class, params)
    death_handled: bool = False
    # control effects: name -> seconds remaining
    statuses: dict = field(default_factory=dict)
    knockback: tuple[float, float, float] | None = None   # (vx, vy, seconds)
    # KnockbackDamageAddon, one per source: (chance, radius, speed, accel, mask)
    knockbacks: list = field(default_factory=list)
    # ExplodeAddon, one per source: (damage, radius, chance, mask)
    explodes: list = field(default_factory=list)
    reflect_pct: float = 0.0
    unit_dodge_cooldown: float = 0.0
    unit_dodge_ready_in: float = 0.0
    projectile_splash: tuple[float, float] | None = None   # (percent, radius)
    # agent-level mutation passives. Each list holds (Name, params) pairs and
    # is dispatched by `sim.mutations`; the hook each one hangs off is the
    # game's own -- C_PassiveSkill.OnDamageCreated, C_DamageReactionSkill's
    # subscription to C_Unit.OnDamage, C_OnSkillCastedSkill.OnSkillCasted and
    # C_BaseOnDeathSkill.OnDeath.
    on_attack: list = field(default_factory=list)     # this agent dealt damage
    on_damaged: list = field(default_factory=list)    # an enemy damaged this agent
    on_cast: list = field(default_factory=list)       # this agent cast a skill
    on_death_passives: list = field(default_factory=list)
    standing: list = field(default_factory=list)      # recomputed every tick
    standing_buff: Buff | None = None                 # what `standing` resolved to
    # ModifyDamage: (damage-type mask, amount, percentage) added to outgoing damage
    damage_bonus: list = field(default_factory=list)
    # ResurrectionChance: (chance%, health value, percentage)
    resurrect: tuple[float, float, bool] | None = None
    resurrected: bool = False
    # CSLinkStatBonus: params applied to the class skill's targets on cast
    cs_link_bonus: list = field(default_factory=list)

    @property
    def alive(self) -> bool:
        return self.hp > 0.0 and not self.corpse

    def has(self, status: str) -> bool:
        return self.statuses.get(status, 0.0) > 0.0

    def apply_status(self, status: str, seconds: float) -> None:
        if seconds <= 0.0:
            return
        self.statuses[status] = max(self.statuses.get(status, 0.0), seconds)

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    def buff_delta(self, stat: str) -> tuple[float, float]:
        """The (percent, flat) a stat is currently buffed by."""
        pct = flat = 0.0
        for b in self.buffs:
            for name, amount, percentage in b.stats:
                if name != stat:
                    continue
                if percentage:
                    pct += amount
                else:
                    flat += amount
        return pct, flat

    def _buffed(self, stat: str, base: float) -> float:
        pct, flat = self.buff_delta(stat)
        return base * (1.0 + pct / 100.0) + flat

    @property
    def armor(self) -> float:
        return max(0.0, self._buffed("armor", self.spec.armor))

    @property
    def damage(self) -> float:
        return max(0.0, self._buffed("damage", self.spec.damage))

    @property
    def resistance(self) -> float:
        return min(1.0, max(0.0, self._buffed("resistance", self.spec.resistance)))

    @property
    def max_hp(self) -> float:
        return max(1.0, self._buffed("health", self.spec.health))

    @property
    def attack_speed_bonus_pct(self) -> float:
        """Class aura, buffs, and stacking FurySwipe gains."""
        pct, _ = self.buff_delta("attack_speed")
        return self.attack_speed_pct + pct + self.fury_stacks * self.fury_per_stack

    @property
    def speed(self) -> float:
        return max(0.0, self._buffed("speed", self.spec.speed))


@dataclass
class Projectile:
    x: float
    y: float
    target: Agent
    damage: float
    speed: float
    team: int
    crit: bool = False
    splash: tuple[float, float] | None = None
    # The damage's source unit. `M_Damage` carries it whatever delivered the
    # hit, so a ranged attacker still owns vampirism, fury and the on-attack
    # passives; without this a Shooter's mutations would silently do nothing.
    attacker: "Agent | None" = None


@dataclass
class BattleResult:
    winner: int | None
    ticks: int
    seconds: float
    survivors: dict[int, int]
    total_damage: dict[int, float]
    healing: dict[int, float]
    casts: dict[str, int]


class Battle:
    def __init__(self, grid: Grid, agents: list[Agent],
                 assumptions: Assumptions = DEFAULT, seed: int = 0,
                 tables: dict | None = None, build_trees: bool = True):
        self.grid = grid
        self.agents = agents
        self.a = assumptions
        self.dt = 1.0 / assumptions.tick_hz
        self.rng = random.Random(seed)
        self.tables = tables
        self.projectiles: list[Projectile] = []
        self.damage_done: dict[int, float] = {0: 0.0, 1: 0.0}
        self.healing_done: dict[int, float] = {0: 0.0, 1: 0.0}
        self.casts: dict[str, int] = {}
        self.tick_count = 0
        self.in_fight = True
        self.build_trees = build_trees
        for ag in agents:
            self._init_agent(ag)

    def _init_agent(self, ag: Agent) -> None:
        ag.follower = PathFollower(self.grid, PICK_NEXT_WAYPOINT_DIST)
        if not ag.actions:
            ag.actions = build_actions(ag.spec, None)
        if self.tables is not None and ag.spec.skills:
            from .unit_skills import apply_unit_skills
            ag.actions = ag.actions + apply_unit_skills(ag, self.tables)
        # Building a behaviour tree per agent is pure waste when the fight is
        # about to be handed to the Rust core.
        if self.build_trees:
            ag.tree = build_unit_tree(ag.actions)

    # -- targeting ---------------------------------------------------------
    def pick_target(self, ag: Agent) -> Agent | None:
        best, best_d = None, float("inf")
        for other in self.agents:
            if other.team == ag.team or not other.alive:
                continue
            d = (other.x - ag.x) ** 2 + (other.y - ag.y) ** 2
            if d < best_d:
                best, best_d = other, d
        return best

    # -- one tick ----------------------------------------------------------
    def step(self) -> None:
        dt = self.dt
        self.tick_count += 1
        living = [a for a in self.agents if a.alive]

        for ag in living:
            if ag.target is None or not ag.target.alive:
                ag.target = self.pick_target(ag)
                ag.repath_in = 0.0
            for act in ag.actions:
                if act.cooldown_left > 0.0:
                    rate = 1.0 + (ag.attack_speed_bonus_pct / 100.0 if act.name == "attack" else 0.0)
                    act.cooldown_left = max(0.0, act.cooldown_left - dt * rate)
            ag.dodge_ready_in = max(0.0, ag.dodge_ready_in - dt)
            ag.unit_dodge_ready_in = max(0.0, ag.unit_dodge_ready_in - dt)
            for bf in ag.buffs:
                bf.remaining -= dt
            ag.buffs = [b for b in ag.buffs if b.remaining > 0.0]
            if ag.link is not None:
                ag.link.remaining -= dt
                if ag.link.remaining <= 0.0:
                    ag.link = None
            if ag.statuses:
                for k in list(ag.statuses):
                    ag.statuses[k] -= dt
                    if ag.statuses[k] <= 0.0:
                        del ag.statuses[k]
            if ag.knockback is not None:
                kx, ky, left = ag.knockback
                left -= dt
                ag.x, ag.y = self.grid.clamp_world(ag.x + kx * dt, ag.y + ky * dt)
                ag.knockback = None if left <= 0.0 else (kx, ky, left)
            if ag.regen is not None:
                stat, period, value = ag.regen
                ag.regen_timer += dt
                if period > 0 and ag.regen_timer >= period:
                    ag.regen_timer -= period
                    if stat == "Mana":
                        ag.mana = min(ag.spec.mana, ag.mana + value)
                    elif stat == "Health":
                        self.heal(ag, value)
            if ag.standing:
                from .mutations import refresh_standing
                refresh_standing(self, ag)
            ag.intent = "stop"
            # Panic is enforced here rather than by a leaf. `IsPanicked`,
            # `PanicInPlace` and `PanicFollowPath` live in NC-Fear, which is a
            # tree of its own that the runtime overlays on a panicked unit --
            # it is not inside NC-BaseUnit, so building the base tree never
            # reaches them and the status was inert. What NC-Fear does with the
            # fleeing left out is stop the unit acting, which is what this does,
            # in the place NC-BaseUnit stops a stunned one.
            if ag.has("panic"):
                continue
            ag.tree.tick(BTContext(ag, self))

        self._process_deaths()

        prefs = {id(ag): self._preferred_velocity(ag, dt) for ag in living}

        for ag in living:
            neighbours = [((o.x, o.y), (o.vx, o.vy), o.spec.radius)
                          for o in living if o is not ag]
            vx, vy = new_velocity(
                ag.pos, (ag.vx, ag.vy), ag.spec.radius, ag.speed,
                prefs[id(ag)], neighbours,
                time_horizon=RVO_TIME_HORIZON, dt=dt,
                max_neighbours=RVO_MAX_NEIGHBOURS,
            )
            ag.vx, ag.vy = vx, vy
            ag.x, ag.y = self.grid.clamp_world(ag.x + vx * dt, ag.y + vy * dt)

        self._step_projectiles(dt)

    def _preferred_velocity(self, ag: Agent, dt: float) -> tuple[float, float]:
        if ag.has("stun") or ag.has("panic") or ag.knockback is not None:
            return (0.0, 0.0)
        if ag.intent != "follow" or ag.target is None:
            return (0.0, 0.0)
        tgt = ag.target

        ag.repath_in -= dt
        if ag.follower.done or ag.repath_in <= 0.0:
            path = astar(self.grid, self.grid.to_cell(ag.x, ag.y),
                         self.grid.to_cell(tgt.x, tgt.y))
            if path:
                ag.follower.set_path(path)
            ag.repath_in = 0.25

        ux, uy = ag.follower.desired_direction(ag.x, ag.y)
        if ux == 0.0 and uy == 0.0:
            dx, dy = tgt.x - ag.x, tgt.y - ag.y
            d = math.hypot(dx, dy) or 1.0
            ux, uy = dx / d, dy / d
        return (ux * ag.speed, uy * ag.speed)

    def effective_range(self, ag: Agent, action, tgt: Agent) -> float:
        """Reach for an in-range test against a specific target.

        A melee unit's Range stat is 0, so its reach is an assumption anyway
        (see assumptions.melee_margin). It has to be measured against BOTH
        bodies: two units cannot be closer than the sum of their radii, so a
        reach built from the attacker's radius alone leaves melee unable to
        touch anything larger than itself. Half the roster is Size 2 or 3.
        """
        if action is not None and action.name != "attack":
            return action.range_world
        if not ag.spec.melee:
            return ag.spec.range_world
        return ag.spec.radius + tgt.spec.radius + self.a.melee_margin

    # -- effects -----------------------------------------------------------
    def land_attack(self, ag: Agent, action: Action) -> None:
        tgt = ag.target
        if tgt is None or not tgt.alive:
            return
        crit = ag.crit_chance > 0.0 and self.rng.random() * 100.0 < ag.crit_chance
        dmg = ag.damage * (ag.crit_mult if crit else 1.0)
        if ag.spec.is_ranged:
            self.projectiles.append(Projectile(
                ag.x, ag.y, tgt, dmg, self.a.projectile_speed, ag.team, crit,
                splash=ag.projectile_splash, attacker=ag))
        else:
            self.hit(ag.team, tgt, dmg, attacker=ag)
            self._splash(ag, tgt, dmg)

    def _splash(self, ag: Agent, tgt: Agent, dmg: float) -> None:
        """SplashAround: percent% of the hit to others within radius."""
        if not ag.splash:
            return
        percent, radius = ag.splash
        if radius <= 0:
            return
        r2 = radius * radius
        for o in self.agents:
            if o is tgt or o.team == ag.team or not o.alive:
                continue
            if (o.x - tgt.x) ** 2 + (o.y - tgt.y) ** 2 <= r2:
                self.hit(ag.team, o, dmg * percent / 100.0, attacker=ag,
                         dtype=DT_PHYSICAL | DT_SECONDARY)

    def hit(self, team: int, tgt: Agent, raw: float, magical: bool = False,
            attacker: 'Agent | None' = None, dtype: int = 0) -> None:
        if not dtype:
            dtype = DT_MAGICAL if magical else DT_PHYSICAL
        if attacker is not None and attacker.damage_bonus:
            raw = self._damage_bonus(attacker, raw, dtype)
        if tgt.dodge_cooldown > 0.0 and tgt.dodge_ready_in <= 0.0:
            tgt.dodge_ready_in = tgt.dodge_cooldown      # Dodger class skill
            return
        if tgt.evasions and self._evaded(tgt, attacker, dtype):
            return                                        # Evasion skill
        if tgt.unit_dodge_cooldown > 0.0 and tgt.unit_dodge_ready_in <= 0.0:
            tgt.unit_dodge_ready_in = tgt.unit_dodge_cooldown   # unit Dodge skill
            return
        dealt = apply_damage(raw, tgt.armor, tgt.resistance, magical=magical)

        # The on-attack passives hang off `C_PassiveSkill.OnDamageCreated`,
        # which fires while the damage is being built -- so before it lands,
        # and while the target is still standing.
        if attacker is not None and attacker.on_attack and tgt.alive:
            from .mutations import fire_on_attack
            fire_on_attack(self, attacker, tgt, dtype)

        if tgt.link is not None and tgt.link.share_pct > 0.0:
            members = [m for m in tgt.link.members if m is not tgt and m.alive]
            if members:
                shared = dealt * tgt.link.share_pct / 100.0
                dealt -= shared
                each = shared / len(members)
                for m in members:
                    m.hp -= each
                    self._gain_mana(m, each)

        victim_hp = tgt.hp
        tgt.hp -= dealt
        self.damage_done[team] += dealt
        if tgt.reflect_pct and attacker is not None and attacker.alive:
            back = dealt * tgt.reflect_pct / 100.0
            attacker.hp -= back                     # Reflection: raw, not re-mitigated
            self.damage_done[tgt.team] += back
        if attacker is not None:
            for value, percentage in attacker.vampirisms:
                # `C_VampirismSkill.Addon.OnApply`: a percentage share of the
                # damage, capped at what the victim actually had left, or a
                # flat `value` when `percentage` is false (the FixedVampirism
                # variant). The game then splits the heal between the damage's
                # healer list and the attacker; nothing fills that list here,
                # so it all lands on the attacker (`vampirism_shares_with_healers`).
                if percentage:
                    heal = min(dealt * value / 100.0, max(0.0, victim_hp))
                else:
                    heal = value
                healed = min(heal, attacker.max_hp - attacker.hp)
                attacker.hp += healed
                self.healing_done[team] += healed
            if attacker.fury_per_stack:
                attacker.fury_stacks += 1
            for chance, radius, speed, accel, mask in attacker.knockbacks:
                if not self._addon_gate(dtype, mask, chance):
                    continue
                # accel is negative, so the push lasts until it stops: v/|a|
                secs = speed / abs(accel) if accel else 0.2
                if radius > 0.0:
                    # `KnockbackDamageAddon.OnApply` walks the victim's team
                    # when Radius is set, so one hit scatters the group.
                    r2 = radius * radius
                    for o in self.agents:
                        if o.team != tgt.team or not o.alive:
                            continue
                        if (o.x - tgt.x) ** 2 + (o.y - tgt.y) ** 2 <= r2:
                            self.push(o, attacker.x, attacker.y, speed, secs)
                elif tgt.alive:
                    self.push(tgt, attacker.x, attacker.y, speed, secs)
        self._gain_mana(tgt, dealt * self.a.mana_per_damage_taken)
        if attacker is not None:
            self._gain_mana(attacker, dealt * self.a.mana_per_damage_dealt)

        # `C_DamageReactionSkill` subscribes to the victim's `C_Unit.OnDamage`,
        # so its passives read the damage after it has been applied.
        if tgt.on_damaged and attacker is not None and attacker.team != tgt.team:
            from .mutations import fire_on_damaged
            fire_on_damaged(self, tgt, attacker, dtype)

        if attacker is not None and attacker.explodes:
            self._explode(attacker, tgt, dtype)

    def _addon_gate(self, dtype: int, mask: int, chance: float) -> bool:
        """`OnDamageCreated` for a damage addon: the damage has to be Physical
        and not Secondary before the chance is even rolled. It is the same gate
        `sim.mutations._gate` implements, because it is the same code in the
        game -- and, as there, a chance of 100 consumes no randomness."""
        if mask and (dtype & mask) != mask:
            return False
        if not (dtype & DT_PHYSICAL) or dtype & DT_SECONDARY:
            return False
        return chance >= 100.0 or self.rng.random() * 100.0 < chance

    def _explode(self, ag: Agent, tgt: Agent, dtype: int) -> None:
        """ExplodeProjectile: `ExplodeAddon.OnApply` walks the attacker's enemy
        team and deals a **flat** `damage` to everyone within `radius` of the
        hit, the victim included. It is built as Magical (`M_Damage.Get` is
        called with type 2), so it cannot be evaded and cannot chain into
        another explosion, but it does feed the on-attack passives."""
        for damage, radius, chance, mask in ag.explodes:
            if not self._addon_gate(dtype, mask, chance):
                continue
            if radius <= 0.0 or damage <= 0.0:
                continue
            r2 = radius * radius
            for o in list(self.agents):
                if o.team == ag.team or not o.alive:
                    continue
                if (o.x - tgt.x) ** 2 + (o.y - tgt.y) ** 2 <= r2:
                    self.hit(ag.team, o, damage, magical=True, attacker=ag,
                             dtype=DT_MAGICAL)

    def _evaded(self, tgt: Agent, attacker: 'Agent | None', dtype: int) -> bool:
        """`C_EvasionSkill.OnTryToEvade`, once per evasion source.

        The gate is the damage's own flags: only Physical damage is evadable
        (`test al, 1`), and Secondary or CantBeEvaded (`test eax, 0xc0`) never
        is -- so splash cannot be dodged. `_active` comes from
        `OnHealthChanged`, which is `healthThreshold >= health / totalMaxHealth
        * 100`, meaning a threshold row only guards while the unit is hurt.
        The roll is `PseudoRandom.Get(chance)`, a shuffle bag; it is taken here
        as a uniform draw (`pseudo_random_is_uniform`).
        """
        if not (dtype & DT_PHYSICAL) or dtype & (DT_SECONDARY | DT_CANT_BE_EVADED):
            return False
        hp_pct = 100.0 * tgt.hp / tgt.max_hp if tgt.max_hp > 0.0 else 0.0
        for chance, threshold, jump_back in tgt.evasions:
            if threshold < hp_pct:
                continue
            if self.rng.random() * 100.0 >= chance:
                continue
            if jump_back and attacker is not None:
                self.push(tgt, attacker.x, attacker.y, JUMP_BACK_SPEED,
                          JUMP_BACK_SPEED / abs(JUMP_BACK_ACCEL))
            return True
        return False

    def _damage_bonus(self, attacker: Agent, raw: float, dtype: int) -> float:
        """ModifyDamage: a flat or percentage addition to one damage type."""
        for mask, amount, percentage in attacker.damage_bonus:
            if mask and (dtype & mask) != mask:
                continue
            raw = raw * (1.0 + amount / 100.0) if percentage else raw + amount
        return max(0.0, raw)

    def heal(self, ag: Agent, amount: float, team: int | None = None) -> float:
        """Heal a unit up to its ceiling. Returns what was actually restored."""
        healed = min(max(0.0, amount), ag.max_hp - ag.hp)
        ag.hp += healed
        if team is not None:
            self.healing_done[team] += healed
        return healed

    def _gain_mana(self, unit: Agent, amount: float) -> None:
        """Mana arrives through the damage pipeline as DamageType.Mana; how much
        it is worth per point of damage is an assumption."""
        if unit.spec.mana <= 0 or amount <= 0:
            return
        unit.mana = min(unit.spec.mana, unit.mana + amount)

    def push(self, ag: Agent, from_x: float, from_y: float,
             speed: float, seconds: float) -> None:
        """Knock a unit directly away from a point."""
        dx, dy = ag.x - from_x, ag.y - from_y
        d = math.hypot(dx, dy) or 1.0
        ag.knockback = (dx / d * speed, dy / d * speed, seconds)

    def blink(self, ag: Agent, away_from: Agent | None, distance: float) -> None:
        """Teleport a unit `distance` away from a reference point."""
        if away_from is None:
            return
        dx, dy = ag.x - away_from.x, ag.y - away_from.y
        d = math.hypot(dx, dy) or 1.0
        ag.x, ag.y = self.grid.clamp_world(ag.x + dx / d * distance,
                                           ag.y + dy / d * distance)
        ag.follower.waypoints = []

    def add_buff(self, ag: Agent, duration: float, armor_pct: float = 0.0,
                 speed_flat: float = 0.0, armor_flat: float = 0.0,
                 attack_speed_pct: float = 0.0) -> None:
        stats = tuple((n, v, p) for n, v, p in (
            ("armor", armor_pct, True), ("armor", armor_flat, False),
            ("speed", speed_flat, False), ("attack_speed", attack_speed_pct, True),
        ) if v)
        self.add_stat_buff(ag, duration, stats)

    def add_stat_buff(self, ag: Agent, duration: float, stats) -> None:
        """A timed `M_StatBonusStatus`: (stat, amount, percentage) triples.

        A Health bonus raises the ceiling only. The game's status is a stat
        contribution rather than a heal, and nothing found says the current
        value follows it up, so the unit gains room to be healed into rather
        than the hit points themselves.
        """
        stats = tuple(stats)
        if duration <= 0.0 or not stats:
            return
        ag.buffs.append(Buff(remaining=duration, stats=stats))

    def add_spirit_link(self, ag: Agent, duration: float, share_pct: float,
                        extra_allies: int) -> None:
        allies = sorted((o for o in self.agents if o.team == ag.team and o.alive and o is not ag),
                        key=lambda o: (o.x - ag.x) ** 2 + (o.y - ag.y) ** 2)[:extra_allies]
        members = [ag] + allies
        link = SpiritLink(remaining=duration, share_pct=share_pct, members=members)
        for m in members:
            m.link = link
        # `C_CSLinkStatBonusSkill.OnCasted` walks the cast's targets, keeps the
        # ones whose class matches `targetClass`, and buffs them for the class
        # skill's own duration.
        if ag.cs_link_bonus:
            from .mutations import status_stats
            from .unit_skills import str_param
            for p in ag.cs_link_bonus:
                want = str_param(p, "targetClass", "TargetClass")
                stats = status_stats(p)
                for m in members:
                    if m.spec.cls == want:
                        self.add_stat_buff(m, duration, stats)

    def summon(self, ag: Agent, cls: str, level: int) -> None:
        if self.tables is None:
            return
        from .data import units_by_class
        ubc = units_by_class(self.tables)
        if cls not in ubc:
            return
        spec = build_unit(ubc, cls, level, name=f"{cls}(summon)")
        new = Agent(spec=spec, team=ag.team,
                    x=ag.x + self.rng.uniform(-spec.radius, spec.radius),
                    y=ag.y + self.rng.uniform(-spec.radius, spec.radius),
                    hp=spec.health, summoned=True)
        new.actions = build_actions(spec, None)
        self._init_agent(new)
        self.agents.append(new)

    def _process_deaths(self) -> None:
        """Fire on-death skills once per unit, then the on-death passives.

        Resurrection runs first: a unit that comes back never counts as dead,
        so nothing that keys off its death should have fired.
        """
        from .mutations import run_death_passives, try_resurrect
        from .unit_skills import run_death_effects
        for ag in list(self.agents):
            if ag.death_handled or ag.hp > 0.0:
                continue
            if ag.resurrect is not None and try_resurrect(self, ag):
                continue
            ag.death_handled = True
            if ag.on_death:
                run_death_effects(ag, self)
            if ag.on_death_passives:
                run_death_passives(ag, self)

    def _step_projectiles(self, dt: float) -> None:
        still = []
        for p in self.projectiles:
            if not p.target.alive:
                continue
            dx, dy = p.target.x - p.x, p.target.y - p.y
            d = math.hypot(dx, dy)
            travel = p.speed * dt
            if d <= travel:
                self.hit(p.team, p.target, p.damage, attacker=p.attacker)
                if p.splash is not None:
                    percent, radius = p.splash
                    r2 = radius * radius
                    for o in self.agents:
                        if o is p.target or o.team == p.team or not o.alive:
                            continue
                        if (o.x - p.target.x) ** 2 + (o.y - p.target.y) ** 2 <= r2:
                            self.hit(p.team, o, p.damage * percent / 100.0,
                                     attacker=p.attacker,
                                     dtype=DT_PHYSICAL | DT_SECONDARY)
                continue
            p.x += dx / d * travel
            p.y += dy / d * travel
            still.append(p)
        self.projectiles = still

    # -- driving -----------------------------------------------------------
    def run(self) -> BattleResult:
        max_ticks = int(self.a.max_fight_seconds * self.a.tick_hz)
        while self.tick_count < max_ticks:
            alive = {t: sum(1 for a in self.agents if a.alive and a.team == t) for t in (0, 1)}
            if alive[0] == 0 or alive[1] == 0:
                winner = None if alive[0] == alive[1] else (0 if alive[0] else 1)
                return self._result(winner, alive)
            self.step()
        alive = {t: sum(1 for a in self.agents if a.alive and a.team == t) for t in (0, 1)}
        return self._result(None, alive)

    def _result(self, winner, alive) -> BattleResult:
        return BattleResult(
            winner=winner, ticks=self.tick_count, seconds=self.tick_count * self.dt,
            survivors=alive, total_damage=dict(self.damage_done),
            healing=dict(self.healing_done), casts=dict(self.casts),
        )


# ------------------------------------------------------------------ squad setup

def apply_class_skills(tables: dict, agents: list[Agent]) -> dict[str, dict]:
    """Resolve and attach class skills for one side, by its composition."""
    counts: dict[str, int] = {}
    for ag in agents:
        counts[ag.spec.cls] = counts.get(ag.spec.cls, 0) + 1
    resolved = resolve_class_skills(tables, counts)

    for ag in agents:
        skill = resolved.get(ag.spec.cls)
        ag.actions = build_actions(ag.spec, skill)
        if skill is None:
            continue
        p, name = skill["params"], skill["name"]
        if name == "CriticalStrike":
            ag.crit_chance = float(p.get("chance") or 0)
            ag.crit_mult = float(p.get("value") or 1)
        elif name == "Dodge":
            ag.dodge_cooldown = float(p.get("cooldown") or 0)
        elif name == "ASAura":
            pass          # applied squad-wide below

    aura = next((s for s in resolved.values() if s["name"] == "ASAura"), None)
    if aura:
        # `isAura` is null and no radius is given, so it is applied to the whole
        # side. percentage=true, so `value` is a percent of attack speed.
        for ag in agents:
            ag.attack_speed_pct = float(aura["params"].get("value") or 0)
    return resolved


def place_at(grid: Grid, specs: list[UnitSpec], cells, team: int = 0) -> list[Agent]:
    """Place units on explicitly chosen cells, one per spec.

    This is what a learned placement policy drives. Cells beyond the list are
    reused cyclically so a short list still deploys everyone.
    """
    cells = list(cells) or [(0, 0)]
    out = []
    for i, spec in enumerate(specs):
        r, c = cells[i % len(cells)]
        x, y = grid.to_world(r, c)
        out.append(Agent(spec=spec, team=team, x=x, y=y, hp=spec.health, mana=0.0))
    return out


def deploy(grid: Grid, layout, specs: list[UnitSpec], team: int,
           zone: str, rng: random.Random) -> list[Agent]:
    """Place units on the layout's spawn cells for a side."""
    cells = layout.zone(zone) if zone == "p" else layout.cells(zone)
    if not cells:
        # Not every layout defines every zone; fall back through the other
        # enemy zone, then to any occupied cell, so deployment cannot fail.
        for alt in ("e1", "e2", "p"):
            cells = layout.zone(alt) if alt == "p" else layout.cells(alt)
            if cells:
                break
    if not cells:
        cells = [(r, c) for r, row in enumerate(layout.grid) for c, _ in enumerate(row)]
    if not cells:
        cells = [(0, 0)]
    chosen = list(cells)
    rng.shuffle(chosen)
    agents = []
    for i, spec in enumerate(specs):
        r, c = chosen[i % len(chosen)]
        x, y = grid.to_world(r, c)
        jitter = 0.0 if i < len(chosen) else spec.radius
        agents.append(Agent(
            spec=spec, team=team,
            x=x + rng.uniform(-jitter, jitter),
            y=y + rng.uniform(-jitter, jitter),
            hp=spec.health, mana=0.0,
        ))
    return agents
