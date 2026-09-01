"""ctypes binding to the Rust battle core.

Deliberately a plain C ABI rather than PyO3: the extension never touches the
CPython ABI, so it builds with the GNU Rust toolchain and needs no MSVC.

The core ports movement (A*, follower, ORCA), the damage formula, projectiles,
every passive skill, and the action system: direct, AoE, heal, drain, mana
burn, self buffs, area debuffs, summons and spirit link. What is left outside
are on-death effects and active statuses (stun, silence, panic).

`supported()` says whether a fight is inside that envelope, and `fast_battle`
refuses the ones that are not rather than silently returning numbers that drift
from the oracle.
"""
from __future__ import annotations

import ctypes
import pathlib
import platform
from typing import NamedTuple

import numpy as np

from .assumptions import DEFAULT, Assumptions
from .battle import PICK_NEXT_WAYPOINT_DIST, RVO_MAX_NEIGHBOURS, RVO_TIME_HORIZON

STRIDE = 29
ACTION_STRIDE = 19
PASSIVE_STRIDE = 21

# Action kinds, matching core/src/lib.rs
(K_ATTACK, K_DIRECT, K_AOE, K_HEAL, K_DRAIN, K_MANABURN,
 K_BUFF_SELF, K_DEBUFF_AROUND, K_SUMMON, K_SPIRIT_LINK,
 K_STATUS, K_BLINK) = range(12)
SCOPE_TEAM, SCOPE_SELF, SCOPE_RADIUS = range(3)

# Control statuses, as indices into the core's `Agent::statuses`.
STATUS_SLOT = {"stun": 0, "silence": 1, "panic": 2, "blinked": 3}

# Passive kinds, matching core/src/lib.rs
(P_BUFF_ATTACK, P_FEARSOME, P_PASSIVE_STUN, P_MANA_BREAK, P_CRAGGY,
 P_UNTOUCHABLE, P_BUFF_ON_CAST, P_MULTICAST, P_BUFF_ON_DEATH, P_STICKY_BLOOD,
 P_RESURRECT, P_COMPENSATION, P_MODIFY_DAMAGE, P_CS_LINK,
 P_DEATH_DAMAGE, P_DEATH_SUMMON, P_EVASION, P_VAMPIRISM,
 P_EXPLODE, P_KNOCKBACK) = range(20)

PASSIVE_KIND = {
    "BuffAttack": P_BUFF_ATTACK, "FearsomeAttack": P_FEARSOME,
    "PassiveStun": P_PASSIVE_STUN, "ManaBreak": P_MANA_BREAK,
    "Craggy": P_CRAGGY, "Untouchable": P_UNTOUCHABLE,
    "BuffOnCasted": P_BUFF_ON_CAST, "MultiCast": P_MULTICAST,
    "BuffOnDeath": P_BUFF_ON_DEATH, "StickyBlood": P_STICKY_BLOOD,
    "Compensation": P_COMPENSATION,
}

# The stat order `Agent.buff_delta` indexes by, shared with the core.
BUFF_STATS = ("armor", "speed", "attack_speed", "damage", "resistance", "health")

# Class names are strings on this side and an integer on the core's, and the
# only thing that reads them is CSLinkStatBonus's `targetClass`. Ids are handed
# out on first sight and live for the process, which is all the packer needs:
# both halves of one fight are packed by the same call.
_CLASS_IDS: dict[str, int] = {}


def class_id(name: str | None) -> int:
    if not name:
        return -1
    if name not in _CLASS_IDS:
        _CLASS_IDS[name] = len(_CLASS_IDS)
    return _CLASS_IDS[name]

_LIB_NAMES = {
    "Windows": "despot_core.dll",
    "Linux": "libdespot_core.so",
    "Darwin": "libdespot_core.dylib",
}
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LIB_PATH = _ROOT / "core" / "target" / "release" / _LIB_NAMES.get(platform.system(), "despot_core.dll")

_lib = None


class CoreUnavailable(RuntimeError):
    pass


class UnsupportedFight(NotImplementedError):
    pass


def load():
    """Load the core, or raise with how to build it."""
    global _lib
    if _lib is not None:
        return _lib
    if not _LIB_PATH.exists():
        raise CoreUnavailable(
            f"{_LIB_PATH} not found; build it with `cargo build --release` in core/")
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.despot_battle.restype = ctypes.c_int32
    lib.despot_battle.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # specs
        ctypes.POINTER(ctypes.c_float),   # positions
        ctypes.c_int32,                   # n_units
        ctypes.c_int32, ctypes.c_int32,   # rows, cols
        ctypes.c_float,                   # tile
        ctypes.POINTER(ctypes.c_uint8),   # walkable
        ctypes.c_float, ctypes.c_float,   # tick_hz, max_seconds
        ctypes.c_float, ctypes.c_float,   # attack_anim, recovery_anim
        ctypes.c_float,                   # projectile_speed
        ctypes.c_float, ctypes.c_int32,   # time_horizon, max_neighbours
        ctypes.c_float,                   # pick_next_dist
        ctypes.c_float,                   # melee_margin
        ctypes.c_float, ctypes.c_float,   # mana per damage dealt / taken
        ctypes.POINTER(ctypes.c_float),   # actions
        ctypes.c_int32,                   # n_actions
        ctypes.POINTER(ctypes.c_float),   # agent-level passives
        ctypes.c_int32,                   # n_passives
        ctypes.POINTER(ctypes.c_float),   # summon templates
        ctypes.c_int32,                   # n_templates
        ctypes.POINTER(ctypes.c_float),   # template action rows
        ctypes.c_int32,                   # n_template_actions
        ctypes.POINTER(ctypes.c_float),   # template passive rows
        ctypes.c_int32,                   # n_template_passives
        ctypes.c_uint64,                  # seed
        ctypes.POINTER(ctypes.c_int32),   # out_ticks
        ctypes.POINTER(ctypes.c_float),   # out_damage
        ctypes.POINTER(ctypes.c_float),   # out_hp
    ]

    lib.despot_battle_batch.restype = ctypes.c_int32
    lib.despot_battle_batch.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.despot_choice_probe.restype = None
    lib.despot_choice_probe.argtypes = [
        ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]

    _lib = lib
    return lib


def choice_probe(seed: int, lens: list[int]) -> list[int]:
    """What the core's `random.choice` picks, for a parity check against CPython."""
    lib = load()
    n = len(lens)
    arr = (ctypes.c_int32 * n)(*lens)
    out = (ctypes.c_int32 * n)()
    lib.despot_choice_probe(seed, arr, n, out)
    return list(out)


def available() -> bool:
    try:
        load()
        return True
    except Exception:
        return False


def supported(agents, tables=None) -> tuple[bool, str]:
    """Is this fight inside what the core ports?

    The passives, the action system, summons, buffs and spirit link are all
    ported, and the core runs a bit-identical MT19937 seeded the way CPython
    seeds it, so crit and evasion consume the same stream. What is left outside
    is on-death effects and active statuses.
    """
    why = pack_all(agents, tables).why
    if why:
        return False, why
    return _non_action_blockers(agents)


def _non_action_blockers(agents) -> tuple[bool, str]:
    """What is still outside the core once the passives are in.

    The agent-level passives and the per-unit on-death skills are both ported,
    so neither sends a fight to the oracle any more -- but an unregistered one
    would be dropped silently, so `pack_passives` raises on any name the core
    has no kind for. What is left is a fight that starts with a status already
    running, which the core has no way to be handed.
    """
    for a in agents:
        if getattr(a, "statuses", None):
            return False, f"{a.spec.name} has an active status"
    return True, ""


def spec_row(a) -> tuple:
    """One agent's spec row. Shared with the summon templates, so a summoned
    unit reaches the core carrying the same passives a deployed one does."""
    s = a.spec
    splash = getattr(a, "splash", None) or (0.0, 0.0)
    psplash = getattr(a, "projectile_splash", None) or (0.0, 0.0)
    regen = getattr(a, "regen", None)
    regen_stat = {"Mana": 1.0, "Health": 2.0}.get(regen[0], 0.0) if regen else 0.0
    period = s.attack_period if s.attack_period != float("inf") else 1e9
    return (
        a.team, a.hp, s.damage, period, s.range_world,
        a.armor, s.resistance, a.speed, s.radius,
        1.0 if s.is_ranged else 0.0, 1.0 if s.melee else 0.0,
        splash[0], splash[1],
        a.crit_chance, a.crit_mult,
        a.dodge_cooldown, a.unit_dodge_cooldown,
        a.fury_per_stack, a.reflect_pct,
        regen_stat, regen[1] if regen else 0.0, regen[2] if regen else 0.0,
        psplash[0], psplash[1],
        a.attack_speed_pct, s.health, a.mana, s.mana,
        class_id(s.cls),
    )


def pack(agents, grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(agents)
    specs = np.zeros((n, STRIDE), dtype=np.float32)
    pos = np.zeros((n, 2), dtype=np.float32)
    for i, a in enumerate(agents):
        specs[i] = spec_row(a)
        pos[i] = (a.x, a.y)
    walk = np.ones((grid.rows, grid.cols), dtype=np.uint8)
    for r in range(grid.rows):
        for c in range(grid.cols):
            walk[r, c] = 1 if grid.walkable[r][c] else 0
    return specs, pos, walk


EFFECT_KIND = {
    "attack": K_ATTACK,
    "Mage": (K_DIRECT, True),      # magical damage
    "Bomb": K_AOE,
    "Heal": (K_HEAL, SCOPE_TEAM),
}


def _num(d, *names, default=0.0):
    """Skills.json is inconsistent about capitalisation and about strings."""
    for n in names:
        v = d.get(n)
        if v is None:
            continue
        if isinstance(v, str):
            if v.lower() in ("true", "false"):
                return 1.0 if v.lower() == "true" else 0.0
            try:
                return float(v)
            except ValueError:
                return default
        return float(v)
    return default


def passive_rows(a, i: int, templates: "_Templates | None" = None) -> list[list[float]]:
    """Flatten every agent's agent-level passives into the core's table.

    One row per (agent, passive). The layout is fixed and shared with
    `attach_passives` in `core/src/lib.rs`:

        0  agent           6  cast_source     12,13,14  stat/value/percentage
        1  kind            7  status          15,16,17  stat/value/percentage
        2  damage mask     8  amount          18,19,20  stat/value/percentage
        3  chance          9  amount2
        4  duration       10  flag
        5  radius         11  target class

    The scalars carry whatever the mechanic needs and nothing else does:
    ManaBreak's `amount`, MultiCast's two refund shares, Compensation's health
    threshold, ResurrectionChance's health value and its percentage flag, and
    for the two on-death skills, the damage or the summon template.
    """
    from .mutations import chance_of, param, status_stats, str_param
    from .mutations import _mask as damage_mask

    stat_id = {name: i for i, name in enumerate(BUFF_STATS)}
    rows: list[list[float]] = []

    def blank(i, kind):
        row = [0.0] * PASSIVE_STRIDE
        row[0], row[1], row[11] = i, kind, -1.0
        row[12] = row[15] = row[18] = -1.0
        return row

    def put_stats(row, stats):
        for k, (stat, amount, pct) in enumerate(stats[:3]):
            row[12 + k * 3] = stat_id.get(stat, -1)
            row[13 + k * 3] = amount
            row[14 + k * 3] = 1.0 if pct else 0.0

    if True:
        hooks = (list(getattr(a, "on_attack", ()))
                 + list(getattr(a, "on_damaged", ()))
                 + list(getattr(a, "on_cast", ()))
                 + list(getattr(a, "on_death_passives", ()))
                 + list(getattr(a, "standing", ())))
        for name, p in hooks:
            kind = PASSIVE_KIND.get(name)
            if kind is None:
                raise UnsupportedFight(f"no core kind for the passive {name!r}")
            row = blank(i, kind)
            row[2] = damage_mask(p)
            row[3] = chance_of(name, p)
            row[4] = param(p, "duration", "Duration")
            row[5] = param(p, "radius", "Radius")
            row[6] = 1.0 if str_param(p, "castTarget", "CastTarget") == "Source" else 0.0
            row[7] = 2.0 if str_param(p, "debuff", "Debuff", default="Stun") == "Silence" else 1.0
            if name == "ManaBreak":
                row[8] = param(p, "amount", "Amount")
            elif name == "MultiCast":
                row[8] = param(p, "saveManaPercent")
                row[9] = param(p, "saveCooldownPercent")
            elif name == "Compensation":
                row[8] = param(p, "healthThreshold", "HealthThreshold")
            put_stats(row, status_stats(p, numbered=(name == "BuffAttack")))
            rows.append(row)

        for mask, amount, pct in getattr(a, "damage_bonus", ()):
            row = blank(i, P_MODIFY_DAMAGE)
            row[2], row[8], row[10] = mask, amount, 1.0 if pct else 0.0
            rows.append(row)

        # Evasion and vampirism are one row per source rather than one number
        # on the spec, because the game attaches one controller per skill and
        # each rolls (or heals) on its own.
        for chance, threshold, jump_back in getattr(a, "evasions", ()):
            row = blank(i, P_EVASION)
            row[3], row[8], row[10] = chance, threshold, 1.0 if jump_back else 0.0
            rows.append(row)

        for value, pct in getattr(a, "vampirisms", ()):
            row = blank(i, P_VAMPIRISM)
            row[8], row[10] = value, 1.0 if pct else 0.0
            rows.append(row)

        for damage, radius, chance, mask in getattr(a, "explodes", ()):
            row = blank(i, P_EXPLODE)
            row[2], row[3], row[5], row[8] = mask, chance, radius, damage
            rows.append(row)

        for chance, radius, speed, accel, mask in getattr(a, "knockbacks", ()):
            row = blank(i, P_KNOCKBACK)
            row[2], row[3], row[5] = mask, chance, radius
            row[8], row[9] = speed, accel
            rows.append(row)

        res = getattr(a, "resurrect", None)
        if res is not None:
            chance, value, pct = res
            row = blank(i, P_RESURRECT)
            row[3], row[8], row[10] = chance, value, 1.0 if pct else 0.0
            rows.append(row)

        for p in getattr(a, "cs_link_bonus", ()):
            row = blank(i, P_CS_LINK)
            row[3] = 100.0
            row[11] = class_id(str_param(p, "targetClass", "TargetClass"))
            put_stats(row, status_stats(p))
            rows.append(row)

        # The per-unit on-death skills. `Skills.json` names them by CSClass
        # rather than by a mutation Name, and they are the one hook that is a
        # skill on the unit and never a mutation, so they are read straight off
        # `Agent.on_death` instead of going through PASSIVE_KIND.
        for cs, p in getattr(a, "on_death", ()):
            if cs == "TransformAfterDeath":
                cls = str_param(p, "Class", "class")
                lvl = int(param(p, "level", "Level", default=1) or 1)
                slot = templates.get(cls, lvl) if templates is not None else -1
                if templates is not None and slot < 0:
                    raise UnsupportedFight(
                        f"{a.spec.name} transforms into {cls}, no template")
                row = blank(i, P_DEATH_SUMMON)
                row[8] = float(int(param(p, "amount", "Amount", default=1) or 1))
                row[11] = slot
                rows.append(row)
            elif cs in ("FireheadDeath", "DamageOnDeath", "FirestarterDeath"):
                row = blank(i, P_DEATH_DAMAGE)
                row[8] = param(p, "value", "Value", "damage", "Damage")
                row[5] = param(p, "radius", "Radius")
                rows.append(row)
            else:
                raise UnsupportedFight(f"no core kind for the on-death skill {cs!r}")

    return rows


def pack_passives(agents, templates: "_Templates | None" = None) -> np.ndarray:
    """Every agent's passive rows, as the core's table.

    `templates` is threaded through so a `TransformAfterDeath` can name the
    blueprint it summons; without one the summon is packed as a no-op, which is
    what `supported()` wants when it is only asking whether a name is known.
    """
    rows: list[list[float]] = []
    for i, a in enumerate(agents):
        rows.extend(passive_rows(a, i, templates))
    if not rows:
        return np.zeros((0, PASSIVE_STRIDE), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


class _Templates:
    """The summon blueprints a fight needs, and everything that hangs off them.

    A summoned unit is a real unit: `Battle.summon` runs `_init_agent`, which
    attaches the class's own `Skills.json` entries, so a SciTower casts SciCast
    and an OstrichRider2 transforms again when it dies. The core used to get a
    bare spec row and give the summon a default attack and nothing else, which
    was a quiet divergence for every summoned class that carries a skill.

    So a template is three things -- a spec row, its action rows and its passive
    rows -- and building one can pull in another, since a transform chain is a
    summon whose on-death skill summons again. The index is reserved before the
    recursion so a cycle terminates instead of unrolling.
    """

    def __init__(self, tables):
        self.tables = tables
        self.index: dict[str, int] = {}
        self.specs: list = []
        self.actions: list = []      # rows, column 0 is the template index
        self.passives: list = []     # rows, column 0 is the template index
        self.failure = ""

    def get(self, cls: str | None, level: int) -> int:
        """The template index for a class at a level, or -1 if there is none."""
        if not cls or self.tables is None:
            return -1
        key = "%s:%s" % (cls, level)
        if key in self.index:
            return self.index[key]
        from .battle import Agent
        from .data import units_by_class
        from .spec import build_unit
        from .actions import build_actions
        from .unit_skills import apply_unit_skills

        ubc = units_by_class(self.tables)
        if cls not in ubc:
            return -1
        spec = build_unit(ubc, cls, level, name=cls + "(summon)")
        agent = Agent(spec=spec, team=0, x=0.0, y=0.0, hp=spec.health, summoned=True)
        agent.actions = build_actions(spec, None)
        if spec.skills:
            agent.actions = agent.actions + apply_unit_skills(agent, self.tables)

        slot = len(self.specs)
        self.index[key] = slot            # reserved before recursing
        self.specs.append(np.zeros(STRIDE, dtype=np.float32))
        self.specs[slot] = np.asarray(spec_row(agent), dtype=np.float32)
        for row in action_rows(agent, 0, self):
            self.actions.append((slot,) + tuple(row[1:]))
        for row in passive_rows(agent, 0, self):
            self.passives.append((slot,) + tuple(row[1:]))
        return slot

    def arrays(self):
        def arr(rows, stride):
            return (np.asarray(rows, dtype=np.float32).reshape(-1, stride)
                    if rows else np.zeros((0, stride), np.float32))
        return (arr(self.specs, STRIDE), arr(self.actions, ACTION_STRIDE),
                arr(self.passives, PASSIVE_STRIDE))


class Unrepresentable(Exception):
    """An action the core has no kind for. Carries the reason, not a row."""


def action_rows(a, i: int, templates: "_Templates") -> list[tuple]:
    """One agent's action rows, priority-descending as the base tree orders them."""
    from .unit_skills import REGISTRY

    out = []
    for act in sorted(getattr(a, "actions", []) or [], key=lambda x: -x.priority):
        kind, magical, amount, scope = None, False, 0.0, SCOPE_TEAM
        duration = armor_pct = armor_flat = speed_flat = as_pct = 0.0
        template, count, status = -1, 0, -1
        damage = float(act.params.get("damage") or act.params.get("Damage") or 0.0)
        radius = float(act.params.get("radius") or act.params.get("Radius") or 0.0)

        if act.name == "attack":
            kind = K_ATTACK
        elif act.name == "Mage":
            kind, magical = K_DIRECT, True
        elif act.name == "Bomb":
            kind = K_AOE
        elif act.name == "Heal":
            kind, amount, scope = K_HEAL, float(act.params.get("heal") or 0.0), SCOPE_TEAM
        elif act.name == "SpiritLink":
            kind = K_SPIRIT_LINK
            duration = _num(act.params, "duration")
            amount = _num(act.params, "share percent")
            count = int(_num(act.params, "additional allied units") or 0)
        elif act.name == "Tank":
            kind = K_BUFF_SELF
            duration = _num(act.params, "duration")
            armor_pct = _num(act.params, "armor bonus (%)")
            speed_flat = _num(act.params, "speed nerf (abs.)")
        elif act.name in ("Cultist", "Scientist"):
            from .actions import SUMMONS
            kind = K_SUMMON
            lvl = int(_num(act.params, "tentacle level", "tower level", default=1) or 1)
            template = templates.get(SUMMONS[act.name], lvl)
            if template < 0:
                raise Unrepresentable(f"{a.spec.name}'s {act.name} has no summon template")
            count = 1
        else:
            h = REGISTRY.get(act.name)
            eff = h.effect if h else ""
            if act.name.startswith("Mutation:"):
                # A `StatBonus` carrying a duration is a timed cast rather than
                # a permanent change, so `apply_to_agents` turns it into an
                # Action. It is built from `EFFECT_BUILDERS["buff_self"]`, so it
                # packs exactly like a StatBonusCast.
                eff = "buff_self"
            magical = str(act.params.get("damageType")
                          or act.params.get("DamageType") or "") == "Magical"
            if eff == "direct":
                kind = K_DIRECT
            elif eff == "aoe":
                kind = K_AOE
            elif eff == "heal":
                kind = K_HEAL
                amount = float(act.params.get("amount") or act.params.get("Amount")
                               or act.params.get("healPerSecond") or 0.0)
                scope = SCOPE_RADIUS if radius else SCOPE_SELF
            elif eff == "drain":
                kind = K_DRAIN
            elif eff == "manaburn":
                kind = K_MANABURN
                amount = float(act.params.get("amount") or act.params.get("Amount") or 0.0)
            elif eff == "buff_self":
                kind = K_BUFF_SELF
                duration = _num(act.params, "duration", "Duration")
                armor_pct = _num(act.params, "armorBonus", "ArmorBuff")
                speed_flat = _num(act.params, "speedBonus")
                stat = act.params.get("stat") or act.params.get("Stat") or ""
                pct = _num(act.params, "percentage", default=1.0) >= 1.0
                if stat == "AttackSpeed" and pct:
                    as_pct = _num(act.params, "value", "Value")
            elif eff == "debuff_around":
                kind = K_DEBUFF_AROUND
                duration = _num(act.params, "duration", "Duration")
                amt = _num(act.params, "amount", "Amount")
                stat = act.params.get("stat") or act.params.get("Stat") or ""
                pct = _num(act.params, "percentage", "Percentage") >= 1.0
                radius = radius or 60.0
                if stat == "Armor":
                    if pct:
                        armor_pct = -amt
                    else:
                        armor_flat = -amt
                elif stat == "AttackSpeed":
                    as_pct = -amt
            elif eff == "status":
                # `STATUS_OF` says which one each CSClass applies: Fear is a
                # panic and WatcherDisable is a silence, so the name of the
                # skill is not the name of the status.
                from .unit_skills import STATUS_OF
                kind = K_STATUS
                duration = _num(act.params, "duration", "Duration")
                status = STATUS_SLOT[STATUS_OF.get(act.name, "stun")]
            elif eff == "blink":
                kind = K_BLINK
                radius = radius or 25.0
                duration = _num(act.params, "duration", "Duration")
            elif eff == "summon":
                kind = K_SUMMON
                cls = (act.params.get("class") or act.params.get("Class")
                       or act.params.get("summonClass"))
                lvl = int(_num(act.params, "level", "Level", default=1) or 1)
                template = templates.get(str(cls) if cls else None, lvl)
                if template < 0:
                    raise Unrepresentable(f"{a.spec.name}'s {act.name} summons "
                                          f"{cls}, no template")
                count = int(_num(act.params, "amount", "Amount", default=1) or 1)
            else:
                raise Unrepresentable(f"{a.spec.name}'s {act.name} "
                                      f"({eff or 'unknown'}) is not ported")
        out.append((i, kind, act.range_world, act.cooldown, act.mana_cost,
                    act.priority, damage, radius, 1.0 if magical else 0.0,
                    amount, scope, duration, armor_pct, armor_flat,
                    speed_flat, as_pct, template, count, status))
    return out


class Packed(NamedTuple):
    """Everything one fight hands the core, plus why it could not be packed."""
    actions: np.ndarray
    passives: np.ndarray
    tmpl_specs: np.ndarray
    tmpl_actions: np.ndarray
    tmpl_passives: np.ndarray
    why: str


def _empty() -> Packed:
    return Packed(np.zeros((0, ACTION_STRIDE), np.float32),
                  np.zeros((0, PASSIVE_STRIDE), np.float32),
                  np.zeros((0, STRIDE), np.float32),
                  np.zeros((0, ACTION_STRIDE), np.float32),
                  np.zeros((0, PASSIVE_STRIDE), np.float32), "")


def pack_all(agents, tables=None) -> Packed:
    """Every table one fight needs, built against one template registry.

    Actions and passives are packed together because an on-death transform is a
    passive that names a summon blueprint, so both have to reach the same
    `_Templates`.
    """
    templates = _Templates(tables)
    acts, pas = [], []
    try:
        for i, a in enumerate(agents):
            acts.extend(action_rows(a, i, templates))
            pas.extend(passive_rows(a, i, templates))
    except (Unrepresentable, UnsupportedFight) as exc:
        return _empty()._replace(why=str(exc))

    def arr(rows, stride):
        return (np.asarray(rows, dtype=np.float32).reshape(-1, stride)
                if len(rows) else np.zeros((0, stride), np.float32))
    tspec, tact, tpas = templates.arrays()
    return Packed(arr(acts, ACTION_STRIDE), arr(pas, PASSIVE_STRIDE),
                  tspec, tact, tpas, "")


def pack_all_batch(battles, tables=None) -> tuple[list[Packed], Packed]:
    """`pack_all` for a batch, against **one** template registry.

    The batch ABI carries a single template table for every fight in it, so the
    indices have to agree across battles; interning per battle and shipping the
    first battle's table would misname every summon in the others.
    """
    templates = _Templates(tables)
    out = []

    def arr(rows, stride):
        return (np.asarray(rows, dtype=np.float32).reshape(-1, stride)
                if len(rows) else np.zeros((0, stride), np.float32))

    for agents in battles:
        acts, pas = [], []
        try:
            for i, a in enumerate(agents):
                acts.extend(action_rows(a, i, templates))
                pas.extend(passive_rows(a, i, templates))
        except (Unrepresentable, UnsupportedFight) as exc:
            return [], _empty()._replace(why=str(exc))
        out.append(_empty()._replace(actions=arr(acts, ACTION_STRIDE),
                                     passives=arr(pas, PASSIVE_STRIDE)))
    tspec, tact, tpas = templates.arrays()
    return out, _empty()._replace(tmpl_specs=tspec, tmpl_actions=tact,
                                  tmpl_passives=tpas)


def fast_battle(grid, agents, assumptions: Assumptions = DEFAULT,
                strict: bool = True, seed: int = 0, tables=None) -> dict:
    """Resolve one fight in the Rust core. Mirrors `Battle.run`'s result."""
    lib = load()
    # pack_actions doubles as the envelope check, so do it once rather than
    # paying for it in supported() and again here.
    packed = pack_all(agents, tables)
    if strict:
        if packed.why:
            raise UnsupportedFight(packed.why)
        ok, why2 = _non_action_blockers(agents)
        if not ok:
            raise UnsupportedFight(why2)
    acts, passives = packed.actions, packed.passives
    tmpl, tmpl_acts, tmpl_pas = (packed.tmpl_specs, packed.tmpl_actions,
                                 packed.tmpl_passives)
    specs, pos, walk = pack(agents, grid)
    n = len(agents)
    out_ticks = ctypes.c_int32(0)
    out_damage = (ctypes.c_float * 2)()
    out_hp = (ctypes.c_float * n)()

    fp = ctypes.POINTER(ctypes.c_float)
    winner = lib.despot_battle(
        specs.ctypes.data_as(fp), pos.ctypes.data_as(fp), n,
        grid.rows, grid.cols, grid.tile,
        walk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        assumptions.tick_hz, assumptions.max_fight_seconds,
        assumptions.attack_anim_s, assumptions.recovery_anim_s,
        assumptions.projectile_speed,
        RVO_TIME_HORIZON, RVO_MAX_NEIGHBOURS, PICK_NEXT_WAYPOINT_DIST,
        assumptions.melee_margin,
        assumptions.mana_per_damage_dealt, assumptions.mana_per_damage_taken,
        acts.ctypes.data_as(fp), len(acts),
        passives.ctypes.data_as(fp), len(passives),
        tmpl.ctypes.data_as(fp), len(tmpl),
        tmpl_acts.ctypes.data_as(fp), len(tmpl_acts),
        tmpl_pas.ctypes.data_as(fp), len(tmpl_pas), seed,
        ctypes.byref(out_ticks), out_damage, out_hp,
    )
    hp = np.ctypeslib.as_array(out_hp, shape=(n,)).copy()
    return {
        "winner": None if winner < 0 else int(winner),
        "ticks": int(out_ticks.value),
        "seconds": out_ticks.value / assumptions.tick_hz,
        "damage": {0: float(out_damage[0]), 1: float(out_damage[1])},
        "hp": hp,
        "survivors": {t: int(sum(1 for i, a in enumerate(agents)
                                 if a.team == t and hp[i] > 0.0)) for t in (0, 1)},
    }


def fast_batch(grid, battles, assumptions: Assumptions = DEFAULT,
               threads: int = 0, seeds=None, tables=None) -> list[dict]:
    """Resolve many fights across OS threads. Every fight needs the same unit count."""
    import os
    lib = load()
    nb = len(battles)
    if nb == 0:
        return []
    upb = len(battles[0])
    specs = np.zeros((nb, upb, STRIDE), dtype=np.float32)
    pos = np.zeros((nb, upb, 2), dtype=np.float32)
    for b, agents in enumerate(battles):
        if len(agents) != upb:
            raise ValueError("every battle in a batch needs the same unit count")
        s, p, _ = pack(agents, grid)
        specs[b], pos[b] = s, p
    walk = np.ones((grid.rows, grid.cols), dtype=np.uint8)
    for r in range(grid.rows):
        for c in range(grid.cols):
            walk[r, c] = 1 if grid.walkable[r][c] else 0

    # every battle needs the same action-row count, so pad to the widest
    packed, shared = pack_all_batch(battles, tables)
    if shared.why:
        raise UnsupportedFight(shared.why)
    per = [x.actions for x in packed]
    tmpl = shared.tmpl_specs
    tmpl_acts, tmpl_pas = shared.tmpl_actions, shared.tmpl_passives
    apb = max((len(x) for x in per), default=0)
    acts_all = np.zeros((nb, apb, ACTION_STRIDE), dtype=np.float32)
    for b, rowset in enumerate(per):
        if len(rowset):
            acts_all[b, :len(rowset)] = rowset
        # pad rows point at agent 0 with kind -1 so they are ignored
        acts_all[b, len(rowset):, 1] = -1.0

    # the passive table is padded the same way, with agent -1 on the pad rows
    per_p = [x.passives for x in packed]
    ppb = max((len(x) for x in per_p), default=0)
    pas_all = np.zeros((nb, max(ppb, 1), PASSIVE_STRIDE), dtype=np.float32)
    pas_all[:, :, 0] = -1.0
    for b, rowset in enumerate(per_p):
        if len(rowset):
            pas_all[b, :len(rowset)] = rowset
    seed_arr = (ctypes.c_uint64 * nb)(*(seeds or [0] * nb))
    out_w = (ctypes.c_int32 * nb)()
    out_t = (ctypes.c_int32 * nb)()
    out_d = (ctypes.c_float * (nb * 2))()
    fp = ctypes.POINTER(ctypes.c_float)
    lib.despot_battle_batch(
        specs.ctypes.data_as(fp), pos.ctypes.data_as(fp), upb, nb,
        grid.rows, grid.cols, grid.tile,
        walk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        assumptions.tick_hz, assumptions.max_fight_seconds,
        assumptions.attack_anim_s, assumptions.recovery_anim_s,
        assumptions.projectile_speed,
        RVO_TIME_HORIZON, RVO_MAX_NEIGHBOURS, PICK_NEXT_WAYPOINT_DIST,
        assumptions.melee_margin,
        assumptions.mana_per_damage_dealt, assumptions.mana_per_damage_taken,
        acts_all.ctypes.data_as(fp), apb,
        pas_all.ctypes.data_as(fp), ppb,
        tmpl.ctypes.data_as(fp), len(tmpl),
        tmpl_acts.ctypes.data_as(fp), len(tmpl_acts),
        tmpl_pas.ctypes.data_as(fp), len(tmpl_pas),
        seed_arr, threads or (os.cpu_count() or 4),
        out_w, out_t, out_d,
    )
    return [{"winner": None if out_w[i] < 0 else int(out_w[i]),
             "ticks": int(out_t[i]),
             "damage": {0: float(out_d[i * 2]), 1: float(out_d[i * 2 + 1])}}
            for i in range(nb)]
