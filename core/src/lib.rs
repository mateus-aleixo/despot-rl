//! Fast battle core, callable over a plain C ABI.
//!
//! A port of the loop in `sim/battle.py`. The Python implementation stays the
//! oracle; this exists only to run the same fight faster. Where the two could
//! differ, this follows Python rather than what would be tidier:
//!
//!   * preferred velocities are computed for every agent BEFORE any of them
//!     move, but the ORCA pass then updates position and velocity in place, so
//!     an agent sees earlier agents already moved and later ones not yet.
//!   * the attack gate reproduces a quirk of NC-DefaultAttack: the inner
//!     Selector tries `StopIfRunning` before the cooldown check, so a unit that
//!     arrives in range while still carrying velocity starts its swing that
//!     tick regardless of cooldown.
//!   * `hit` consumes a random number for evasion in exactly the same places,
//!     including for splash victims, so the RNG streams stay aligned.
//!
//! Ported: movement (A*, follower, ORCA), the damage formula, projectiles, the
//! passive skills, the action system -- a priority-ordered action list per
//! agent with mana and cooldown gates, covering direct, AoE, heal, drain, mana
//! burn, self buffs, area debuffs, summons and spirit link -- control statuses
//! (stun, silence, panic), and the agent-level mutation passives, which hang
//! off the four hooks described in `sim/mutations.py`.
//!
//! The passive hooks consume randomness, so they are placed exactly where the
//! oracle places them: on-attack after mitigation is computed and before the
//! damage lands, on-damaged at the very end of `hit`, on-cast right after the
//! effect resolves, and the death pass after the whole per-agent loop.
//!
//! The per-unit on-death skills are in too, and with them a summon that is a
//! real unit: a template carries its class's own actions and passives, so a
//! summoned SciTower casts SciCast and a transform chain keeps going.
//!
//! Status casts and blinks are in as well, so every enemy class in the roster
//! is inside the envelope.
//!
//! NOT ported: a fight that starts with a status already running, which the ABI
//! has no way to be handed.

mod nav;
mod orca;
mod rng;

use nav::{astar, Follower, Grid};

pub const STRIDE: usize = 29;
/// One row per action: agent, kind, range, cooldown, mana cost, priority,
/// damage, radius, magical, amount, scope, duration, the four buff terms, the
/// summon template and count, and the status a status cast applies.
pub const ACTION_STRIDE: usize = 19;

/// Action kinds, matching `sim/actions.py` and `sim/unit_skills.py`.
const K_ATTACK: i32 = 0;
const K_DIRECT: i32 = 1;
const K_AOE: i32 = 2;
const K_HEAL: i32 = 3;
const K_DRAIN: i32 = 4;
const K_MANABURN: i32 = 5;
const K_BUFF_SELF: i32 = 6;
const K_DEBUFF_AROUND: i32 = 7;
const K_SUMMON: i32 = 8;
const K_SPIRIT_LINK: i32 = 9;
const K_STATUS: i32 = 10;
const K_BLINK: i32 = 11;

/// Heal scope: the class skill heals the whole team, unit heals use a radius.
const SCOPE_TEAM: i32 = 0;
const SCOPE_SELF: i32 = 1;
const SCOPE_RADIUS: i32 = 2;

/// One row per agent-level passive: which agent, which mechanic, its gate and
/// its numbers. Laid out by `fast.pack_passives`.
pub const PASSIVE_STRIDE: usize = 21;

/// Passive kinds, matching the dispatch in `sim/mutations.py`.
const P_BUFF_ATTACK: i32 = 0;
const P_FEARSOME: i32 = 1;
const P_PASSIVE_STUN: i32 = 2;
const P_MANA_BREAK: i32 = 3;
const P_CRAGGY: i32 = 4;
const P_UNTOUCHABLE: i32 = 5;
const P_BUFF_ON_CAST: i32 = 6;
const P_MULTICAST: i32 = 7;
const P_BUFF_ON_DEATH: i32 = 8;
const P_STICKY_BLOOD: i32 = 9;
const P_RESURRECT: i32 = 10;
const P_COMPENSATION: i32 = 11;
const P_MODIFY_DAMAGE: i32 = 12;
const P_CS_LINK: i32 = 13;
/// The per-unit on-death skills, which are `Skills.json` CSClasses rather than
/// mutation Names but hang off the same death pass.
const P_DEATH_DAMAGE: i32 = 14;
const P_DEATH_SUMMON: i32 = 15;
/// Evasion and vampirism arrive as one row per source: the game runs one
/// controller per skill, so two sources are two rolls and two heals.
const P_EVASION: i32 = 16;
const P_VAMPIRISM: i32 = 17;
const P_EXPLODE: i32 = 18;
const P_KNOCKBACK: i32 = 19;

/// `DamageType`, the flags enum from the game. Only the bits the gates read.
const DT_PHYSICAL: i32 = 1;
const DT_MAGICAL: i32 = 2;
const DT_SECONDARY: i32 = 64;
const DT_CANT_BE_EVADED: i32 = 128;

/// The two literals `C_EvasionSkill.OnTryToEvade` passes to `M_Knockback.Get`.
const JUMP_BACK_SPEED: f32 = 150.0;
const JUMP_BACK_ACCEL: f32 = -280.0;

/// Which stat a buff triple names. Matches `BUFF_STATS` in `sim/battle.py`.
const ST_ARMOR: i32 = 0;
const ST_SPEED: i32 = 1;
const ST_ATTACK_SPEED: i32 = 2;
const ST_DAMAGE: i32 = 3;
const ST_RESISTANCE: i32 = 4;
const ST_HEALTH: i32 = 5;

/// Control statuses, as indices into `Agent::statuses`.
const S_STUN: usize = 0;
const S_SILENCE: usize = 1;
const S_PANIC: usize = 2;
/// `MushroomBlink` and `BlinkAway` mark the blinker with this. Nothing reads it
/// in either engine -- the oracle applies it and no leaf tests it -- but the
/// two are kept in the same state so a later reader gets the same answer.
const S_BLINKED: usize = 3;
const N_STATUS: usize = 4;

/// How many (stat, amount, percentage) triples one status may carry. The
/// widest shipped row is BuffAttack's two; `refresh_standing` can aggregate,
/// and anything past this is dropped rather than reallocating per tick.
const MAX_BUFF_STATS: usize = 4;

/// One agent-level passive, unpacked from its row.
#[derive(Clone)]
struct Passive {
    kind: i32,
    damage_mask: i32,
    chance: f32,
    duration: f32,
    radius: f32,
    cast_source: bool,
    status: i32,
    amount: f32,
    amount2: f32,
    flag: bool,
    target_class: i32,
    stats: [(i32, f32, bool); MAX_BUFF_STATS],
    nstats: usize,
}

#[derive(Clone)]
struct ActionState {
    kind: i32,
    range: f32,
    cooldown: f32,
    mana_cost: f32,
    damage: f32,
    radius: f32,
    magical: bool,
    amount: f32,
    scope: i32,
    duration: f32,
    armor_pct: f32,
    armor_flat: f32,
    speed_flat: f32,
    as_pct: f32,
    template: i32,
    count: i32,
    /// For K_STATUS: which status to apply, as an index into `Agent::statuses`.
    status: i32,
    cooldown_left: f32,
    anim: f32,
    phase: Phase,
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Gate,
    AttackAnim,
    RecoveryAnim,
}

struct Agent {
    team: i32,
    hp: f32,
    max_health: f32,
    damage: f32,
    range: f32,
    base_armor: f32,
    resistance: f32,
    base_speed: f32,
    radius: f32,
    is_ranged: bool,
    melee: bool,
    splash_percent: f32,
    splash_radius: f32,
    crit_chance: f32,
    crit_mult: f32,
    evasions: Vec<(f32, f32, bool)>, // (chance, health threshold, jump back)
    dodge_cooldown: f32,
    dodge_ready_in: f32,
    unit_dodge_cooldown: f32,
    unit_dodge_ready_in: f32,
    vampirisms: Vec<(f32, bool)>, // (value, percentage)
    fury_per_stack: f32,
    fury_stacks: i32,
    reflect_pct: f32,
    regen_stat: i32, // 0 none, 1 mana, 2 health
    regen_period: f32,
    regen_value: f32,
    regen_timer: f32,
    // (chance, radius, speed, acceleration, damage mask), one per source
    knockbacks: Vec<(f32, f32, f32, f32, i32)>,
    // (damage, radius, chance, damage mask), one per source
    explodes: Vec<(f32, f32, f32, i32)>,
    proj_splash_percent: f32,
    proj_splash_radius: f32,
    attack_speed_pct: f32,
    mana: f32,
    max_mana: f32,
    actions: Vec<ActionState>,
    buffs: Vec<Buff>,
    link: Option<usize>,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    kb: Option<(f32, f32, f32)>,
    target: i32,
    follower: Follower,
    repath_in: f32,
    intent_follow: bool,
    // -- agent-level passives ------------------------------------------
    class_id: i32,
    /// Seconds left on stun / silence / panic, indexed by S_*.
    statuses: [f32; N_STATUS],
    on_attack: Vec<Passive>,
    on_damaged: Vec<Passive>,
    on_cast: Vec<Passive>,
    on_death: Vec<Passive>,
    standing: Vec<Passive>,
    cs_link: Vec<Passive>,
    /// ModifyDamage: (damage-type mask, amount, percentage).
    damage_bonus: Vec<(i32, f32, bool)>,
    /// ResurrectionChance: (chance%, health value, percentage).
    resurrect: Option<(f32, f32, bool)>,
    resurrected: bool,
    death_handled: bool,
}

/// A timed `M_StatBonusStatus`: a duration and up to `MAX_BUFF_STATS`
/// (stat, amount, percentage) triples, the shape the game's status carries.
#[derive(Clone, Copy)]
struct Buff {
    remaining: f32,
    stats: [(i32, f32, bool); MAX_BUFF_STATS],
    nstats: usize,
    /// Compensation's buff is recomputed every tick and replaced, not stacked.
    standing: bool,
}

impl Buff {
    fn from_stats(remaining: f32, stats: &[(i32, f32, bool)]) -> Buff {
        let mut out = Buff {
            remaining,
            stats: [(-1, 0.0, false); MAX_BUFF_STATS],
            nstats: 0,
            standing: false,
        };
        for &st in stats.iter().take(MAX_BUFF_STATS) {
            out.stats[out.nstats] = st;
            out.nstats += 1;
        }
        out
    }
}

impl Agent {
    /// The (percent, flat) a stat is currently buffed by.
    #[inline]
    fn buff_delta(&self, stat: i32) -> (f32, f32) {
        let (mut pct, mut flat) = (0.0f32, 0.0f32);
        for b in &self.buffs {
            for &(name, amount, percentage) in b.stats[..b.nstats].iter() {
                if name != stat {
                    continue;
                }
                if percentage {
                    pct += amount;
                } else {
                    flat += amount;
                }
            }
        }
        (pct, flat)
    }

    #[inline]
    fn buffed(&self, stat: i32, base: f32) -> f32 {
        let (pct, flat) = self.buff_delta(stat);
        base * (1.0 + pct / 100.0) + flat
    }

    /// Class aura, timed buffs, and stacking FurySwipe gains.
    #[inline]
    fn attack_speed_bonus_pct(&self) -> f32 {
        self.attack_speed_pct
            + self.buff_delta(ST_ATTACK_SPEED).0
            + self.fury_stacks as f32 * self.fury_per_stack
    }

    #[inline]
    fn armor(&self) -> f32 {
        self.buffed(ST_ARMOR, self.base_armor).max(0.0)
    }

    #[inline]
    fn damage_now(&self) -> f32 {
        self.buffed(ST_DAMAGE, self.damage).max(0.0)
    }

    #[inline]
    fn resistance_now(&self) -> f32 {
        self.buffed(ST_RESISTANCE, self.resistance).clamp(0.0, 1.0)
    }

    #[inline]
    fn max_hp(&self) -> f32 {
        self.buffed(ST_HEALTH, self.max_health).max(1.0)
    }

    #[inline]
    fn speed(&self) -> f32 {
        self.buffed(ST_SPEED, self.base_speed).max(0.0)
    }

    #[inline]
    fn has(&self, status: usize) -> bool {
        self.statuses[status] > 0.0
    }

    #[inline]
    fn apply_status(&mut self, status: usize, seconds: f32) {
        if seconds > 0.0 && seconds > self.statuses[status] {
            self.statuses[status] = seconds;
        }
    }
}

/// A spirit link shared by its members. The oracle stores one object on every
/// member and decrements it once per member per tick, so a 7-member link
/// expires seven times faster than its duration suggests; that is reproduced
/// here rather than "fixed", because the oracle is the definition.
struct Link {
    remaining: f32,
    share_pct: f32,
    members: Vec<usize>,
}

struct Projectile {
    x: f32,
    y: f32,
    target: usize,
    damage: f32,
    speed: f32,
    team: i32,
    splash_percent: f32,
    splash_radius: f32,
    // The damage's source unit, or -1. M_Damage carries it whatever delivered
    // the hit, so a ranged attacker still owns vampirism, fury and knockback.
    attacker: i32,
}

#[inline]
fn apply_damage(amount: f32, armor: f32, resistance: f32, magical: bool) -> f32 {
    if magical {
        (1.0 - resistance) * amount
    } else {
        let floor = amount.max(0.0).min(1.0);
        (amount - armor).max(floor)
    }
}

pub struct Sim {
    agents: Vec<Agent>,
    projectiles: Vec<Projectile>,
    grid: Grid,
    dt: f32,
    attack_anim: f32,
    recovery_anim: f32,
    projectile_speed: f32,
    time_horizon: f32,
    max_neighbours: usize,
    melee_margin: f32,
    templates: Vec<Vec<f32>>,
    /// One entry per template: the action and passive rows a summoned unit of
    /// that class gets, so it is a real unit rather than a walking stat block.
    template_actions: Vec<Vec<f32>>,
    template_passives: Vec<Vec<f32>>,
    links: Vec<Link>,
    pick_next_dist: f32,
    mana_per_dealt: f32,
    mana_per_taken: f32,
    rng: rng::PyRandom,
    damage: [f32; 2],
    ticks: i32,
}

impl Sim {
    #[inline]
    fn alive(&self, i: usize) -> bool {
        self.agents[i].hp > 0.0
    }

    fn pick_target(&self, i: usize) -> i32 {
        let (ax, ay, team) = (self.agents[i].x, self.agents[i].y, self.agents[i].team);
        let mut best = -1i32;
        let mut best_d = f32::INFINITY;
        for (j, o) in self.agents.iter().enumerate() {
            if o.team == team || o.hp <= 0.0 {
                continue;
            }
            let d = (o.x - ax).powi(2) + (o.y - ay).powi(2);
            if d < best_d {
                best_d = d;
                best = j as i32;
            }
        }
        best
    }

    /// Teleport a unit `distance` directly away from a reference point.
    ///
    /// The oracle clears the follower's waypoints without resetting its index,
    /// which leaves `done()` true and forces a repath next tick; that is
    /// reproduced rather than tidied, because the oracle is the definition.
    fn blink(&mut self, i: usize, away_from: usize, distance: f32) {
        let (dx, dy) = (
            self.agents[i].x - self.agents[away_from].x,
            self.agents[i].y - self.agents[away_from].y,
        );
        let d = (dx * dx + dy * dy).sqrt();
        let d = if d == 0.0 { 1.0 } else { d };
        let (nx, ny) = (
            self.agents[i].x + dx / d * distance,
            self.agents[i].y + dy / d * distance,
        );
        let (cx, cy) = self.grid.clamp_world(nx, ny);
        self.agents[i].x = cx;
        self.agents[i].y = cy;
        self.agents[i].follower.waypoints.clear();
    }

    fn push(&mut self, tgt: usize, from_x: f32, from_y: f32, speed: f32, seconds: f32) {
        let (dx, dy) = (self.agents[tgt].x - from_x, self.agents[tgt].y - from_y);
        let d = (dx * dx + dy * dy).sqrt().max(1e-6);
        self.agents[tgt].kb = Some((dx / d * speed, dy / d * speed, seconds));
    }

    // ---------------------------------------------------------------------
    // Agent-level passives.
    //
    // Every one of these consumes randomness where the oracle consumes it, so
    // the placement matters as much as the arithmetic: `fire_on_attack` runs
    // after mitigation is computed and before the damage lands, `fire_on_damaged`
    // at the very end of `hit`, `fire_on_cast` right after the effect resolves,
    // and `process_deaths` once, after the whole per-agent loop.

    /// The gate both damage hooks share, plus the chance roll.
    ///
    /// `(damage.type & skill.damageType) == skill.damageType` is a subset test,
    /// so an absent mask matches everything; `Secondary` never triggers a
    /// passive, which is what keeps splash from doubling every on-hit effect.
    #[inline]
    fn passive_gate(&mut self, p: &Passive, dtype: i32) -> bool {
        if p.damage_mask != 0 && (dtype & p.damage_mask) != p.damage_mask {
            return false;
        }
        if dtype & DT_SECONDARY != 0 {
            return false;
        }
        p.chance >= 100.0 || (self.rng.random() as f32) * 100.0 < p.chance
    }

    fn add_stat_buff(&mut self, i: usize, duration: f32, p: &Passive) {
        if duration <= 0.0 || p.nstats == 0 {
            return;
        }
        self.agents[i]
            .buffs
            .push(Buff::from_stats(duration, &p.stats[..p.nstats]));
    }

    /// `C_PassiveSkill.OnDamageCreated`, for the unit that dealt the damage.
    fn fire_on_attack(&mut self, attacker: usize, tgt: usize, dtype: i32) {
        for k in 0..self.agents[attacker].on_attack.len() {
            let p = self.agents[attacker].on_attack[k].clone();
            if !self.passive_gate(&p, dtype) {
                continue;
            }
            match p.kind {
                P_BUFF_ATTACK => {
                    // `castTarget` picks which end of the damage it lands on.
                    let who = if p.cast_source { attacker } else { tgt };
                    self.add_stat_buff(who, p.duration, &p);
                }
                P_FEARSOME => self.agents[tgt].apply_status(S_PANIC, p.duration),
                P_PASSIVE_STUN => self.agents[tgt].apply_status(S_STUN, p.duration),
                P_MANA_BREAK => {
                    self.agents[tgt].mana = (self.agents[tgt].mana - p.amount).max(0.0);
                }
                _ => {}
            }
        }
    }

    /// `C_DamageReactionSkill`: the reaction lands on whoever hit this unit.
    fn fire_on_damaged(&mut self, victim: usize, attacker: usize, dtype: i32) {
        if self.agents[attacker].hp <= 0.0 {
            return;
        }
        for k in 0..self.agents[victim].on_damaged.len() {
            let p = self.agents[victim].on_damaged[k].clone();
            if !self.passive_gate(&p, dtype) {
                continue;
            }
            match p.kind {
                P_CRAGGY => {
                    let st = if p.status == 2 { S_SILENCE } else { S_STUN };
                    self.agents[attacker].apply_status(st, p.duration);
                }
                P_UNTOUCHABLE => self.add_stat_buff(attacker, p.duration, &p),
                _ => {}
            }
        }
    }

    /// `C_OnSkillCastedSkill.OnSkillCasted`, after a skill's effect resolves.
    fn fire_on_cast(&mut self, i: usize, ai: usize) {
        for k in 0..self.agents[i].on_cast.len() {
            let p = self.agents[i].on_cast[k].clone();
            match p.kind {
                P_BUFF_ON_CAST => self.add_stat_buff(i, p.duration, &p),
                P_MULTICAST => {
                    if p.chance <= 0.0 || (self.rng.random() as f32) * 100.0 >= p.chance {
                        continue;
                    }
                    // A repeat cast, refunding the shares the row names.
                    let (cost, cd) = (
                        self.agents[i].actions[ai].mana_cost,
                        self.agents[i].actions[ai].cooldown,
                    );
                    let cap = self.agents[i].max_mana;
                    self.agents[i].mana = (self.agents[i].mana + cost * p.amount / 100.0).min(cap);
                    let left = self.agents[i].actions[ai].cooldown_left;
                    self.agents[i].actions[ai].cooldown_left =
                        (left - cd * p.amount2 / 100.0).max(0.0);
                    self.fire_effect(i, ai);
                }
                _ => {}
            }
        }
    }

    /// `C_CSLinkStatBonusSkill.OnCasted`: the class skill's targets of
    /// `targetClass`, for the skill's own duration.
    fn fire_cs_link(&mut self, i: usize, duration: f32, members: &[usize]) {
        for k in 0..self.agents[i].cs_link.len() {
            let p = self.agents[i].cs_link[k].clone();
            for &m in members {
                if self.agents[m].class_id == p.target_class {
                    self.add_stat_buff(m, duration, &p);
                }
            }
        }
    }

    /// `Compensation`: a bonus recomputed from the squad, every tick.
    ///
    /// `SetBonus` counts the allies under `healthThreshold` percent of their
    /// maximum and multiplies the row's bonus by that count, so it grows as the
    /// squad is worn down. Held as one buff that is replaced, not stacked.
    fn refresh_standing(&mut self, i: usize) {
        let mut stats: Vec<(i32, f32, bool)> = Vec::new();
        for k in 0..self.agents[i].standing.len() {
            let p = self.agents[i].standing[k].clone();
            if p.kind != P_COMPENSATION {
                continue;
            }
            let team = self.agents[i].team;
            let threshold = p.amount / 100.0;
            let hurt = (0..self.agents.len())
                .filter(|&j| {
                    self.agents[j].team == team
                        && self.agents[j].hp > 0.0
                        && self.agents[j].hp < self.agents[j].max_hp() * threshold
                })
                .count();
            if hurt == 0 {
                continue;
            }
            for &(stat, amount, pct) in p.stats[..p.nstats].iter() {
                stats.push((stat, amount * hurt as f32, pct));
            }
        }
        self.agents[i].buffs.retain(|b| !b.standing);
        if !stats.is_empty() {
            let mut b = Buff::from_stats(f32::INFINITY, &stats);
            b.standing = true;
            self.agents[i].buffs.push(b);
        }
    }

    /// The oracle's `_process_deaths`, minus the per-unit on-death skills,
    /// which are still an envelope blocker.
    ///
    /// Resurrection runs first: a unit that comes back never counts as dead, so
    /// nothing keyed off its death should have fired.
    fn process_deaths(&mut self) {
        for i in 0..self.agents.len() {
            if self.agents[i].death_handled || self.agents[i].hp > 0.0 {
                continue;
            }
            if self.agents[i].resurrect.is_some() && self.try_resurrect(i) {
                continue;
            }
            self.agents[i].death_handled = true;
            if !self.agents[i].on_death.is_empty() {
                // the oracle runs the unit's own skills first, then the
                // mutation passives, and both consume randomness
                self.run_unit_death_skills(i);
                self.run_death_passives(i);
            }
        }
    }

    /// `ResurrectionChance`: one roll, once, back at `healthValue`.
    fn try_resurrect(&mut self, i: usize) -> bool {
        if self.agents[i].resurrected {
            return false;
        }
        let (chance, value, percentage) = self.agents[i].resurrect.unwrap();
        self.agents[i].resurrected = true;
        if (self.rng.random() as f32) * 100.0 >= chance {
            return false;
        }
        self.agents[i].hp = if percentage {
            self.agents[i].max_hp() * value / 100.0
        } else {
            value
        };
        self.agents[i].statuses = [0.0; N_STATUS];
        self.agents[i].hp > 0.0
    }

    /// `C_BaseOnDeathSkill.OnDeath`, once, after the unit is confirmed dead.
    fn run_death_passives(&mut self, i: usize) {
        for k in 0..self.agents[i].on_death.len() {
            let p = self.agents[i].on_death[k].clone();
            if p.nstats == 0 {
                continue;
            }
            let (team, ax, ay) = (self.agents[i].team, self.agents[i].x, self.agents[i].y);
            let victims: Vec<usize> = match p.kind {
                // OnDeath walks the dead unit's own team, with no radius term.
                P_BUFF_ON_DEATH => (0..self.agents.len())
                    .filter(|&j| self.agents[j].team == team && self.agents[j].hp > 0.0)
                    .collect(),
                // ApplyStatus takes an `enemy`, inside `radius` of the corpse.
                P_STICKY_BLOOD => {
                    let r2 = p.radius * p.radius;
                    (0..self.agents.len())
                        .filter(|&j| {
                            self.agents[j].team != team
                                && self.agents[j].hp > 0.0
                                && (self.agents[j].x - ax).powi(2)
                                    + (self.agents[j].y - ay).powi(2)
                                    <= r2
                        })
                        .collect()
                }
                _ => Vec::new(),
            };
            for j in victims {
                self.add_stat_buff(j, p.duration, &p);
            }
        }
    }

    /// The per-unit on-death skills, fired from the same death pass.
    ///
    /// The no-radius branch of the damage one picks a living foe with
    /// `random.choice`, which is `_randbelow(len)` -- getrandbits, not a float
    /// -- so the core needs CPython's rejection loop to keep the stream aligned.
    fn run_unit_death_skills(&mut self, i: usize) {
        for k in 0..self.agents[i].on_death.len() {
            let p = self.agents[i].on_death[k].clone();
            match p.kind {
                P_DEATH_DAMAGE => {
                    if p.amount == 0.0 {
                        continue;
                    }
                    let (team, ax, ay) = (self.agents[i].team, self.agents[i].x, self.agents[i].y);
                    if p.radius > 0.0 {
                        let r2 = p.radius * p.radius;
                        let victims: Vec<usize> = (0..self.agents.len())
                            .filter(|&j| {
                                self.agents[j].team != team
                                    && self.agents[j].hp > 0.0
                                    && (self.agents[j].x - ax).powi(2)
                                        + (self.agents[j].y - ay).powi(2)
                                        <= r2
                            })
                            .collect();
                        for j in victims {
                            self.hit(team, j, p.amount, false, -1);
                        }
                    } else {
                        let foes: Vec<usize> = (0..self.agents.len())
                            .filter(|&j| self.agents[j].team != team && self.agents[j].hp > 0.0)
                            .collect();
                        if !foes.is_empty() {
                            let pick = foes[self.rng.choice(foes.len())];
                            self.hit(team, pick, p.amount, false, -1);
                        }
                    }
                }
                P_DEATH_SUMMON => {
                    let count = p.amount as i32;
                    for _ in 0..count.max(0) {
                        self.summon(i, p.target_class);
                    }
                }
                _ => {}
            }
        }
    }

    /// ModifyDamage: a flat or percentage addition to one damage type.
    #[inline]
    fn damage_bonus(&self, attacker: usize, mut raw: f32, dtype: i32) -> f32 {
        for &(mask, amount, percentage) in &self.agents[attacker].damage_bonus {
            if mask != 0 && (dtype & mask) != mask {
                continue;
            }
            raw = if percentage { raw * (1.0 + amount / 100.0) } else { raw + amount };
        }
        raw.max(0.0)
    }

    /// `C_EvasionSkill.OnTryToEvade`, once per evasion source. Only Physical
    /// damage is evadable, and Secondary or CantBeEvaded never is, so splash
    /// cannot be dodged. A row's `healthThreshold` gates it on the unit being
    /// hurt, which is what `OnHealthChanged` computes.
    fn evaded(&mut self, tgt: usize, attacker: i32, dtype: i32) -> bool {
        if dtype & DT_PHYSICAL == 0 || dtype & (DT_SECONDARY | DT_CANT_BE_EVADED) != 0 {
            return false;
        }
        let max_hp = self.agents[tgt].max_hp();
        let hp_pct = if max_hp > 0.0 {
            100.0 * self.agents[tgt].hp / max_hp
        } else {
            0.0
        };
        for k in 0..self.agents[tgt].evasions.len() {
            let (chance, threshold, jump_back) = self.agents[tgt].evasions[k];
            if threshold < hp_pct {
                continue;
            }
            if (self.rng.random() as f32) * 100.0 >= chance {
                continue;
            }
            if jump_back && attacker >= 0 {
                let a = attacker as usize;
                let (ax, ay) = (self.agents[a].x, self.agents[a].y);
                let secs = JUMP_BACK_SPEED / JUMP_BACK_ACCEL.abs();
                self.push(tgt, ax, ay, JUMP_BACK_SPEED, secs);
            }
            return true;
        }
        false
    }

    /// Mirrors `Battle.hit`, including where it consumes randomness.
    fn hit(&mut self, team: i32, tgt: usize, raw: f32, magical: bool, attacker: i32) {
        let dtype = if magical { DT_MAGICAL } else { DT_PHYSICAL };
        self.hit_typed(team, tgt, raw, magical, attacker, dtype)
    }

    fn hit_typed(
        &mut self,
        team: i32,
        tgt: usize,
        raw: f32,
        magical: bool,
        attacker: i32,
        dtype: i32,
    ) {
        let mut raw = raw;
        if attacker >= 0 && !self.agents[attacker as usize].damage_bonus.is_empty() {
            raw = self.damage_bonus(attacker as usize, raw, dtype);
        }
        if self.agents[tgt].dodge_cooldown > 0.0 && self.agents[tgt].dodge_ready_in <= 0.0 {
            self.agents[tgt].dodge_ready_in = self.agents[tgt].dodge_cooldown;
            return;
        }
        if !self.agents[tgt].evasions.is_empty() && self.evaded(tgt, attacker, dtype) {
            return;
        }
        if self.agents[tgt].unit_dodge_cooldown > 0.0
            && self.agents[tgt].unit_dodge_ready_in <= 0.0
        {
            self.agents[tgt].unit_dodge_ready_in = self.agents[tgt].unit_dodge_cooldown;
            return;
        }

        let mut dealt = apply_damage(
            raw,
            self.agents[tgt].armor(),
            self.agents[tgt].resistance_now(),
            magical,
        );

        // The on-attack passives hang off `C_PassiveSkill.OnDamageCreated`,
        // which fires while the damage is being built -- so before it lands,
        // and while the target is still standing.
        if attacker >= 0
            && !self.agents[attacker as usize].on_attack.is_empty()
            && self.agents[tgt].hp > 0.0
        {
            self.fire_on_attack(attacker as usize, tgt, dtype);
        }

        if let Some(li) = self.agents[tgt].link {
            if self.links[li].share_pct > 0.0 {
                let members: Vec<usize> = self.links[li]
                    .members
                    .iter()
                    .copied()
                    .filter(|&m| m != tgt && self.agents[m].hp > 0.0)
                    .collect();
                if !members.is_empty() {
                    let shared = dealt * self.links[li].share_pct / 100.0;
                    dealt -= shared;
                    let each = shared / members.len() as f32;
                    for m in members {
                        self.agents[m].hp -= each;
                        // the oracle credits the raw share here, unscaled
                        self.gain_mana(m, each);
                    }
                }
            }
        }

        let victim_hp = self.agents[tgt].hp;
        self.agents[tgt].hp -= dealt;
        self.damage[team as usize] += dealt;

        if self.agents[tgt].reflect_pct != 0.0 && attacker >= 0 {
            let a = attacker as usize;
            if self.agents[a].hp > 0.0 {
                let back = dealt * self.agents[tgt].reflect_pct / 100.0;
                self.agents[a].hp -= back;
                let tteam = self.agents[tgt].team as usize;
                self.damage[tteam] += back;
            }
        }
        if attacker >= 0 {
            let a = attacker as usize;
            // `C_VampirismSkill.Addon.OnApply`: a share of the damage capped
            // at what the victim had left, or a flat value for FixedVampirism.
            for k in 0..self.agents[a].vampirisms.len() {
                let (value, percentage) = self.agents[a].vampirisms[k];
                let heal = if percentage {
                    (dealt * value / 100.0).min(victim_hp.max(0.0))
                } else {
                    value
                };
                let cap = self.agents[a].max_hp() - self.agents[a].hp;
                self.agents[a].hp += heal.min(cap);
            }
            if self.agents[a].fury_per_stack != 0.0 {
                self.agents[a].fury_stacks += 1;
            }
            for k in 0..self.agents[a].knockbacks.len() {
                let (chance, radius, speed, accel, mask) = self.agents[a].knockbacks[k];
                if !self.addon_gate(dtype, mask, chance) {
                    continue;
                }
                let secs = if accel != 0.0 { speed / accel.abs() } else { 0.2 };
                let (ax, ay) = (self.agents[a].x, self.agents[a].y);
                if radius > 0.0 {
                    // `KnockbackDamageAddon.OnApply` walks the victim's team
                    // when Radius is set.
                    let (tx, ty, tteam) =
                        (self.agents[tgt].x, self.agents[tgt].y, self.agents[tgt].team);
                    let r2 = radius * radius;
                    let hits: Vec<usize> = (0..self.agents.len())
                        .filter(|&o| {
                            self.agents[o].team == tteam
                                && self.agents[o].hp > 0.0
                                && (self.agents[o].x - tx).powi(2)
                                    + (self.agents[o].y - ty).powi(2)
                                    <= r2
                        })
                        .collect();
                    for o in hits {
                        self.push(o, ax, ay, speed, secs);
                    }
                } else if self.agents[tgt].hp > 0.0 {
                    self.push(tgt, ax, ay, speed, secs);
                }
            }
        }
        // mana arrives through the damage pipeline, target first then attacker
        self.gain_mana(tgt, dealt * self.mana_per_taken);
        if attacker >= 0 {
            self.gain_mana(attacker as usize, dealt * self.mana_per_dealt);
        }

        // `C_DamageReactionSkill` subscribes to the victim's `C_Unit.OnDamage`,
        // so its passives read the damage after it has been applied.
        if !self.agents[tgt].on_damaged.is_empty() && attacker >= 0 {
            let a = attacker as usize;
            if self.agents[a].team != self.agents[tgt].team {
                self.fire_on_damaged(tgt, a, dtype);
            }
        }

        if attacker >= 0 && !self.agents[attacker as usize].explodes.is_empty() {
            self.explode(attacker as usize, tgt, dtype);
        }
    }

    /// The gate a damage addon runs before it attaches: Physical, not
    /// Secondary, then the chance. Same shape as `passive_gate`, and like it a
    /// chance of 100 consumes no randomness.
    fn addon_gate(&mut self, dtype: i32, mask: i32, chance: f32) -> bool {
        if mask != 0 && (dtype & mask) != mask {
            return false;
        }
        if dtype & DT_PHYSICAL == 0 || dtype & (DT_SECONDARY) != 0 {
            return false;
        }
        chance >= 100.0 || (self.rng.random() as f32) * 100.0 < chance
    }

    /// `ExplodeAddon.OnApply`: a flat magical hit to every enemy within radius
    /// of the victim, the victim included.
    fn explode(&mut self, ag: usize, tgt: usize, dtype: i32) {
        for k in 0..self.agents[ag].explodes.len() {
            let (damage, radius, chance, mask) = self.agents[ag].explodes[k];
            if !self.addon_gate(dtype, mask, chance) {
                continue;
            }
            if radius <= 0.0 || damage <= 0.0 {
                continue;
            }
            let (tx, ty) = (self.agents[tgt].x, self.agents[tgt].y);
            let team = self.agents[ag].team;
            let r2 = radius * radius;
            let hits: Vec<usize> = (0..self.agents.len())
                .filter(|&o| {
                    self.agents[o].team != team
                        && self.agents[o].hp > 0.0
                        && (self.agents[o].x - tx).powi(2) + (self.agents[o].y - ty).powi(2)
                            <= r2
                })
                .collect();
            for o in hits {
                self.hit(team, o, damage, true, ag as i32);
            }
        }
    }

    #[inline]
    fn gain_mana(&mut self, i: usize, amount: f32) {
        if self.agents[i].max_mana <= 0.0 || amount <= 0.0 {
            return;
        }
        let cap = self.agents[i].max_mana;
        self.agents[i].mana = (self.agents[i].mana + amount).min(cap);
    }

    fn heal(&mut self, i: usize, ai: usize) {
        let (scope, radius, amount) = (
            self.agents[i].actions[ai].scope,
            self.agents[i].actions[ai].radius,
            self.agents[i].actions[ai].amount,
        );
        let (team, ax, ay) = (self.agents[i].team, self.agents[i].x, self.agents[i].y);
        let r2 = radius * radius;
        let pool: Vec<usize> = match scope {
            SCOPE_SELF => vec![i],
            SCOPE_RADIUS => (0..self.agents.len())
                .filter(|&j| {
                    self.agents[j].team == team
                        && self.agents[j].hp > 0.0
                        && (self.agents[j].x - ax).powi(2) + (self.agents[j].y - ay).powi(2) <= r2
                })
                .collect(),
            SCOPE_TEAM | _ => (0..self.agents.len())
                .filter(|&j| self.agents[j].team == team && self.agents[j].hp > 0.0)
                .collect(),
        };
        let mut best: Option<usize> = None;
        let mut worst = 0.0f32;
        for j in pool {
            let missing = self.agents[j].max_hp() - self.agents[j].hp;
            if missing > worst {
                worst = missing;
                best = Some(j);
            }
        }
        if let Some(j) = best {
            let cap = self.agents[j].max_hp();
            self.agents[j].hp = (self.agents[j].hp + amount).min(cap);
        }
    }

    fn add_buff(&mut self, i: usize, act: &ActionState) {
        if act.duration <= 0.0 {
            return;
        }
        let mut stats: Vec<(i32, f32, bool)> = Vec::new();
        for (stat, amount, percentage) in [
            (ST_ARMOR, act.armor_pct, true),
            (ST_ARMOR, act.armor_flat, false),
            (ST_SPEED, act.speed_flat, false),
            (ST_ATTACK_SPEED, act.as_pct, true),
        ] {
            if amount != 0.0 {
                stats.push((stat, amount, percentage));
            }
        }
        if stats.is_empty() {
            return;
        }
        self.agents[i]
            .buffs
            .push(Buff::from_stats(act.duration, &stats));
    }

    /// Spawn a summoned unit from a template.
    ///
    /// The offset consumes two random draws through `uniform(-r, r)`, exactly
    /// as the oracle does, so the RNG streams stay aligned.
    fn summon(&mut self, i: usize, template: i32) {
        if template < 0 || template as usize >= self.templates.len() {
            return;
        }
        let row = self.templates[template as usize].clone();
        let radius = row[8];
        let ox = -radius + 2.0 * radius * self.rng.random() as f32;
        let oy = -radius + 2.0 * radius * self.rng.random() as f32;
        let (x, y) = (self.agents[i].x + ox, self.agents[i].y + oy);
        let mut new = build_agent(&row, x, y, self.pick_next_dist);
        new.team = self.agents[i].team;
        // `Battle.summon` runs `_init_agent`, which gives the summon its class's
        // own Skills.json entries on top of the default attack -- so a SciTower
        // casts SciCast and an OstrichRider2 transforms again when it dies. The
        // packer resolved both into these two tables.
        let t = template as usize;
        let acts = self.template_actions[t].clone();
        let pas = self.template_passives[t].clone();
        let mut one = [new];
        attach_actions(&mut one, &acts);
        attach_passives(&mut one, &pas);
        let [new] = one;
        self.agents.push(new);
    }

    fn fire_effect(&mut self, i: usize, ai: usize) {
        let act = self.agents[i].actions[ai].clone();
        let t = self.agents[i].target;
        let team = self.agents[i].team;
        match act.kind {
            K_ATTACK => self.land_attack(i),
            K_DIRECT => {
                if t >= 0 && self.alive(t as usize) && act.damage != 0.0 {
                    self.hit(team, t as usize, act.damage, act.magical, i as i32);
                }
            }
            K_AOE => {
                if t >= 0 && self.alive(t as usize) && act.damage != 0.0 {
                    let (tx, ty) = (self.agents[t as usize].x, self.agents[t as usize].y);
                    let r2 = act.radius * act.radius;
                    let victims: Vec<usize> = (0..self.agents.len())
                        .filter(|&j| {
                            self.agents[j].team != team
                                && self.agents[j].hp > 0.0
                                && (self.agents[j].x - tx).powi(2)
                                    + (self.agents[j].y - ty).powi(2)
                                    <= r2
                        })
                        .collect();
                    for j in victims {
                        self.hit(team, j, act.damage, act.magical, i as i32);
                    }
                }
            }
            K_HEAL => self.heal(i, ai),
            K_DRAIN => {
                if t >= 0 && self.alive(t as usize) {
                    let tgt = t as usize;
                    let before = self.agents[tgt].hp;
                    self.hit(team, tgt, act.damage, act.magical, i as i32);
                    let drained = (before - self.agents[tgt].hp).max(0.0);
                    let cap = self.agents[i].max_hp() - self.agents[i].hp;
                    self.agents[i].hp += drained.min(cap);
                }
            }
            K_BUFF_SELF => {
                self.add_buff(i, &act);
            }
            K_DEBUFF_AROUND => {
                let (ax, ay) = (self.agents[i].x, self.agents[i].y);
                let r2 = act.radius * act.radius;
                let victims: Vec<usize> = (0..self.agents.len())
                    .filter(|&j| {
                        self.agents[j].team != team
                            && self.agents[j].hp > 0.0
                            && (self.agents[j].x - ax).powi(2)
                                + (self.agents[j].y - ay).powi(2)
                                <= r2
                    })
                    .collect();
                for j in victims {
                    self.add_buff(j, &act);
                }
            }
            K_STATUS => {
                // `make_status`: a radius picks every enemy around the CASTER,
                // not around the target; without one it is the target alone.
                // Damage, when the row carries any, lands on the same victims.
                let (ax, ay) = (self.agents[i].x, self.agents[i].y);
                let victims: Vec<usize> = if act.radius > 0.0 {
                    let r2 = act.radius * act.radius;
                    (0..self.agents.len())
                        .filter(|&j| {
                            self.agents[j].team != team
                                && self.agents[j].hp > 0.0
                                && (self.agents[j].x - ax).powi(2)
                                    + (self.agents[j].y - ay).powi(2)
                                    <= r2
                        })
                        .collect()
                } else if t >= 0 && self.alive(t as usize) {
                    vec![t as usize]
                } else {
                    Vec::new()
                };
                let slot = act.status.max(0) as usize;
                for j in victims {
                    if slot < N_STATUS {
                        self.agents[j].apply_status(slot, act.duration);
                    }
                    if act.damage != 0.0 {
                        self.hit(team, j, act.damage, act.magical, i as i32);
                    }
                }
            }
            K_BLINK => {
                if t >= 0 {
                    self.blink(i, t as usize, act.radius);
                }
                if act.duration > 0.0 {
                    self.agents[i].apply_status(S_BLINKED, act.duration);
                }
            }
            K_SPIRIT_LINK => {
                let (ax, ay) = (self.agents[i].x, self.agents[i].y);
                let mut allies: Vec<usize> = (0..self.agents.len())
                    .filter(|&j| {
                        j != i && self.agents[j].team == team && self.agents[j].hp > 0.0
                    })
                    .collect();
                allies.sort_by(|&a, &b| {
                    let da = (self.agents[a].x - ax).powi(2) + (self.agents[a].y - ay).powi(2);
                    let db = (self.agents[b].x - ax).powi(2) + (self.agents[b].y - ay).powi(2);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
                allies.truncate(act.count.max(0) as usize);
                let mut members = vec![i];
                members.extend(allies);
                let li = self.links.len();
                self.links.push(Link {
                    remaining: act.duration,
                    share_pct: act.amount,
                    members: members.clone(),
                });
                for &m in &members {
                    self.agents[m].link = Some(li);
                }
                // `C_CSLinkStatBonusSkill.OnCasted` rides on this cast.
                if !self.agents[i].cs_link.is_empty() {
                    self.fire_cs_link(i, act.duration, &members);
                }
            }
            K_SUMMON => {
                for _ in 0..act.count.max(0) {
                    self.summon(i, act.template);
                }
            }
            K_MANABURN => {
                let (ax, ay) = (self.agents[i].x, self.agents[i].y);
                let r2 = act.radius * act.radius;
                let victims: Vec<usize> = (0..self.agents.len())
                    .filter(|&j| {
                        self.agents[j].team != team
                            && self.agents[j].hp > 0.0
                            && (self.agents[j].x - ax).powi(2)
                                + (self.agents[j].y - ay).powi(2)
                                <= r2
                    })
                    .collect();
                for j in victims {
                    self.agents[j].mana = (self.agents[j].mana - act.amount).max(0.0);
                }
            }
            _ => {}
        }
    }

    /// SplashAround: percent% of the hit to other enemies within radius.
    fn splash(&mut self, i: usize, tgt: usize, dmg: f32) {
        let (pct, radius, team) = (
            self.agents[i].splash_percent,
            self.agents[i].splash_radius,
            self.agents[i].team,
        );
        if radius <= 0.0 || pct <= 0.0 {
            return;
        }
        let r2 = radius * radius;
        let (tx, ty) = (self.agents[tgt].x, self.agents[tgt].y);
        let victims: Vec<usize> = (0..self.agents.len())
            .filter(|&j| {
                j != tgt
                    && self.agents[j].team != team
                    && self.agents[j].hp > 0.0
                    && (self.agents[j].x - tx).powi(2) + (self.agents[j].y - ty).powi(2) <= r2
            })
            .collect();
        for j in victims {
            self.hit_typed(
                team,
                j,
                dmg * pct / 100.0,
                false,
                i as i32,
                DT_PHYSICAL | DT_SECONDARY,
            );
        }
    }

    fn land_attack(&mut self, i: usize) {
        let t = self.agents[i].target;
        if t < 0 || !self.alive(t as usize) {
            return;
        }
        let tgt = t as usize;
        // The crit roll happens for every swing of a unit that can crit, before
        // any damage, so the RNG stream stays aligned with the oracle's.
        let crit = self.agents[i].crit_chance > 0.0
            && (self.rng.random() as f32) * 100.0 < self.agents[i].crit_chance;
        let dmg = self.agents[i].damage_now() * if crit { self.agents[i].crit_mult } else { 1.0 };
        let (team, ranged) = (self.agents[i].team, self.agents[i].is_ranged);
        if ranged {
            self.projectiles.push(Projectile {
                x: self.agents[i].x,
                y: self.agents[i].y,
                target: tgt,
                damage: dmg,
                speed: self.projectile_speed,
                team,
                splash_percent: self.agents[i].proj_splash_percent,
                splash_radius: self.agents[i].proj_splash_radius,
                attacker: i as i32,
            });
        } else {
            self.hit(team, tgt, dmg, false, i as i32);
            self.splash(i, tgt, dmg);
        }
    }

    /// One tick of the agent's action list.
    ///
    /// The base tree puts every action in a dynamic Selector ordered by
    /// priority, so they are tried highest first. A skill (NC-Skill) whose
    /// gates fail returns Failure and the next action is tried; the default
    /// attack (NC-DefaultAttack) always consumes the tick once reached, either
    /// swinging, waiting out its cooldown, or walking toward the target.
    fn tick_actions(&mut self, i: usize) {
        let t = self.agents[i].target;
        if t < 0 || !self.alive(t as usize) {
            for a in self.agents[i].actions.iter_mut() {
                a.phase = Phase::Gate;
                a.anim = 0.0;
            }
            return;
        }
        let tgt = t as usize;

        for ai in 0..self.agents[i].actions.len() {
            match self.agents[i].actions[ai].phase {
                Phase::AttackAnim => {
                    self.agents[i].actions[ai].anim -= self.dt;
                    if self.agents[i].actions[ai].anim <= 0.0 {
                        self.fire_effect(i, ai);
                        // `C_OnSkillCastedSkill` hangs off the skill cast, not
                        // the swing, so the default attack does not fire it.
                        if !self.agents[i].on_cast.is_empty()
                            && self.agents[i].actions[ai].kind != K_ATTACK
                        {
                            self.fire_on_cast(i, ai);
                        }
                        self.agents[i].actions[ai].phase = Phase::RecoveryAnim;
                        self.agents[i].actions[ai].anim = self.recovery_anim;
                    }
                    return;
                }
                Phase::RecoveryAnim => {
                    self.agents[i].actions[ai].anim -= self.dt;
                    if self.agents[i].actions[ai].anim <= 0.0 {
                        self.agents[i].actions[ai].phase = Phase::Gate;
                    }
                    return;
                }
                Phase::Gate => {
                    let act = self.agents[i].actions[ai].clone();
                    let is_attack = act.kind == K_ATTACK;

                    // Melee reach is measured against BOTH bodies.
                    let reach = if is_attack && self.agents[i].melee {
                        self.agents[i].radius + self.agents[tgt].radius + self.melee_margin
                    } else if is_attack {
                        self.agents[i].range
                    } else {
                        act.range
                    };
                    let (dx, dy) = (
                        self.agents[tgt].x - self.agents[i].x,
                        self.agents[tgt].y - self.agents[i].y,
                    );
                    // A self-cast skill needs no approach; range 0 marks one.
                    // Silence has no leaf of its own, so it is enforced in the
                    // range gate: a silenced unit fails ReadyToAct for anything
                    // that is not the plain attack.
                    let in_range = !(!is_attack && self.agents[i].has(S_SILENCE))
                        && ((!is_attack && act.range <= 0.0)
                            || reach * reach > dx * dx + dy * dy);

                    if is_attack {
                        if !in_range {
                            self.agents[i].intent_follow = true;
                            return;
                        }
                        let moving = self.agents[i].vx != 0.0 || self.agents[i].vy != 0.0;
                        if moving || act.cooldown_left <= 0.0 {
                            self.start(i, ai);
                        }
                        return;
                    }

                    // NC-Skill gates: mana, then cooldown, then range. Any
                    // failure falls through to the next action.
                    if self.agents[i].mana >= act.mana_cost
                        && act.cooldown_left <= 0.0
                        && in_range
                    {
                        self.start(i, ai);
                        return;
                    }
                }
            }
        }
    }

    fn start(&mut self, i: usize, ai: usize) {
        self.agents[i].actions[ai].phase = Phase::AttackAnim;
        self.agents[i].actions[ai].anim = self.attack_anim;
        // M_Action tracks elapsedCooldown, so it runs during the animations.
        let cd = self.agents[i].actions[ai].cooldown;
        self.agents[i].actions[ai].cooldown_left = cd;
        let cost = self.agents[i].actions[ai].mana_cost;
        self.agents[i].mana -= cost;
    }

    fn preferred_velocity(&mut self, i: usize) -> (f32, f32) {
        if self.agents[i].kb.is_some()
            || self.agents[i].has(S_STUN)
            || self.agents[i].has(S_PANIC)
        {
            return (0.0, 0.0);
        }
        if !self.agents[i].intent_follow || self.agents[i].target < 0 {
            return (0.0, 0.0);
        }
        let tgt = self.agents[i].target as usize;
        let (ax, ay) = (self.agents[i].x, self.agents[i].y);
        let (tx, ty) = (self.agents[tgt].x, self.agents[tgt].y);

        self.agents[i].repath_in -= self.dt;
        if self.agents[i].follower.done() || self.agents[i].repath_in <= 0.0 {
            let start = self.grid.to_cell(ax, ay);
            let goal = self.grid.to_cell(tx, ty);
            let path = astar(&self.grid, start, goal);
            if !path.is_empty() {
                let g = &self.grid;
                self.agents[i].follower.set_path(g, &path);
            }
            self.agents[i].repath_in = 0.25;
        }

        let (mut ux, mut uy) = self.agents[i].follower.desired_direction(ax, ay);
        if ux == 0.0 && uy == 0.0 {
            let (dx, dy) = (tx - ax, ty - ay);
            let d = (dx * dx + dy * dy).sqrt().max(1e-6);
            ux = dx / d;
            uy = dy / d;
        }
        let sp = self.agents[i].speed();
        (ux * sp, uy * sp)
    }

    fn step(&mut self) {
        self.ticks += 1;
        let n = self.agents.len();
        let living: Vec<usize> = (0..n).filter(|&i| self.alive(i)).collect();

        for &i in &living {
            let t = self.agents[i].target;
            if t < 0 || !self.alive(t as usize) {
                self.agents[i].target = self.pick_target(i);
                self.agents[i].repath_in = 0.0;
            }
            // fury swipe and auras speed the attack cooldown up; skills tick
            // at 1x, matching `rate` in Battle.step
            let rate = 1.0 + self.agents[i].attack_speed_bonus_pct() / 100.0;
            for a in self.agents[i].actions.iter_mut() {
                if a.cooldown_left > 0.0 {
                    let r = if a.kind == K_ATTACK { rate } else { 1.0 };
                    a.cooldown_left = (a.cooldown_left - self.dt * r).max(0.0);
                }
            }
            self.agents[i].dodge_ready_in = (self.agents[i].dodge_ready_in - self.dt).max(0.0);
            self.agents[i].unit_dodge_ready_in =
                (self.agents[i].unit_dodge_ready_in - self.dt).max(0.0);
            if !self.agents[i].buffs.is_empty() {
                let dt = self.dt;
                for b in self.agents[i].buffs.iter_mut() {
                    b.remaining -= dt;
                }
                self.agents[i].buffs.retain(|b| b.remaining > 0.0);
            }
            if let Some(li) = self.agents[i].link {
                self.links[li].remaining -= self.dt;
                if self.links[li].remaining <= 0.0 {
                    self.agents[i].link = None;
                }
            }
            let dt = self.dt;
            for st in self.agents[i].statuses.iter_mut() {
                if *st > 0.0 {
                    *st -= dt;
                    if *st < 0.0 {
                        *st = 0.0;
                    }
                }
            }

            if let Some((kx, ky, left)) = self.agents[i].kb {
                let left = left - self.dt;
                let (nx, ny) =
                    (self.agents[i].x + kx * self.dt, self.agents[i].y + ky * self.dt);
                let (cx, cy) = self.grid.clamp_world(nx, ny);
                self.agents[i].x = cx;
                self.agents[i].y = cy;
                self.agents[i].kb = if left <= 0.0 { None } else { Some((kx, ky, left)) };
            }

            // Mana regeneration was inert while mana was unported; it is not
            // any more -- the Samurai banks 150 mana every 10 s, which is what
            // pays for its attack-speed cast.
            if self.agents[i].regen_stat != 0 && self.agents[i].regen_period > 0.0 {
                self.agents[i].regen_timer += self.dt;
                if self.agents[i].regen_timer >= self.agents[i].regen_period {
                    self.agents[i].regen_timer -= self.agents[i].regen_period;
                    let v = self.agents[i].regen_value;
                    if self.agents[i].regen_stat == 1 {
                        let cap = self.agents[i].max_mana;
                        self.agents[i].mana = (self.agents[i].mana + v).min(cap);
                    } else {
                        let cap = self.agents[i].max_hp();
                        self.agents[i].hp = (self.agents[i].hp + v).min(cap);
                    }
                }
            }

            if !self.agents[i].standing.is_empty() {
                self.refresh_standing(i);
            }
            self.agents[i].intent_follow = false;
            // `living` is taken before the loop, so an agent can be killed by an
            // earlier agent in the same tick and still be walked here. The
            // oracle's tree stops it: NC-BaseUnit's first branch is
            // `ShouldDie -> Die`, which sets the intent to stop and never
            // reaches the action Selector. Without this the core gave a unit a
            // free swing from beyond the grave, which is exactly the swing a
            // death effect makes visible.
            if self.agents[i].hp <= 0.0 {
                continue;
            }
            // NC-BaseUnit stops a stunned unit before it reaches its actions,
            // and NC-Fear does the same for a panicked one; the oracle enforces
            // panic at the loop level for the reason recorded there.
            if self.agents[i].has(S_STUN) || self.agents[i].has(S_PANIC) {
                continue;
            }
            self.tick_actions(i);
        }

        self.process_deaths();

        let mut prefs = vec![(0.0f32, 0.0f32); n];
        for &i in &living {
            prefs[i] = self.preferred_velocity(i);
        }

        for &i in &living {
            let mut neighbours: Vec<((f32, f32), (f32, f32), f32)> = living
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| {
                    (
                        (self.agents[j].x, self.agents[j].y),
                        (self.agents[j].vx, self.agents[j].vy),
                        self.agents[j].radius,
                    )
                })
                .collect();
            let a = &self.agents[i];
            let (vx, vy) = orca::new_velocity(
                (a.x, a.y),
                (a.vx, a.vy),
                a.radius,
                a.speed(),
                prefs[i],
                &mut neighbours,
                self.time_horizon,
                self.dt,
                self.max_neighbours,
                0.5,
            );
            let a = &mut self.agents[i];
            a.vx = vx;
            a.vy = vy;
            let (nx, ny) = (a.x + vx * self.dt, a.y + vy * self.dt);
            let (cx, cy) = self.grid.clamp_world(nx, ny);
            self.agents[i].x = cx;
            self.agents[i].y = cy;
        }

        self.step_projectiles();
    }

    fn step_projectiles(&mut self) {
        let mut keep: Vec<Projectile> = Vec::with_capacity(self.projectiles.len());
        let taken = std::mem::take(&mut self.projectiles);
        for mut p in taken {
            if self.agents[p.target].hp <= 0.0 {
                continue;
            }
            let (dx, dy) = (self.agents[p.target].x - p.x, self.agents[p.target].y - p.y);
            let d = (dx * dx + dy * dy).sqrt();
            let travel = p.speed * self.dt;
            if d <= travel {
                self.hit(p.team, p.target, p.damage, false, p.attacker);
                if p.splash_radius > 0.0 {
                    let r2 = p.splash_radius * p.splash_radius;
                    let (tx, ty) = (self.agents[p.target].x, self.agents[p.target].y);
                    let victims: Vec<usize> = (0..self.agents.len())
                        .filter(|&j| {
                            j != p.target
                                && self.agents[j].team != p.team
                                && self.agents[j].hp > 0.0
                                && (self.agents[j].x - tx).powi(2)
                                    + (self.agents[j].y - ty).powi(2)
                                    <= r2
                        })
                        .collect();
                    for j in victims {
                        self.hit_typed(
                            p.team,
                            j,
                            p.damage * p.splash_percent / 100.0,
                            false,
                            p.attacker,
                            DT_PHYSICAL | DT_SECONDARY,
                        );
                    }
                }
                continue;
            }
            p.x += dx / d * travel;
            p.y += dy / d * travel;
            keep.push(p);
        }
        self.projectiles = keep;
    }

    fn run(&mut self, max_ticks: i32) -> i32 {
        loop {
            let mut alive = [0i32, 0i32];
            for a in &self.agents {
                if a.hp > 0.0 {
                    alive[a.team as usize] += 1;
                }
            }
            if alive[0] == 0 || alive[1] == 0 {
                return if alive[0] == alive[1] {
                    -1
                } else if alive[0] > 0 {
                    0
                } else {
                    1
                };
            }
            if self.ticks >= max_ticks {
                return -1;
            }
            self.step();
        }
    }
}

fn build_agent(s: &[f32], x: f32, y: f32, pick_next_dist: f32) -> Agent {
    Agent {
        team: s[0] as i32,
        hp: s[1],
        max_health: s[25],
        damage: s[2],
        range: s[4],
        base_armor: s[5],
        resistance: s[6],
        base_speed: s[7],
        radius: s[8],
        is_ranged: s[9] != 0.0,
        melee: s[10] != 0.0,
        splash_percent: s[11],
        splash_radius: s[12],
        crit_chance: s[13],
        crit_mult: s[14],
        evasions: Vec::new(),
        dodge_cooldown: s[15],
        dodge_ready_in: 0.0,
        unit_dodge_cooldown: s[16],
        unit_dodge_ready_in: 0.0,
        vampirisms: Vec::new(),
        fury_per_stack: s[17],
        fury_stacks: 0,
        reflect_pct: s[18],
        regen_stat: s[19] as i32,
        regen_period: s[20],
        regen_value: s[21],
        regen_timer: 0.0,
        knockbacks: Vec::new(),
        explodes: Vec::new(),
        proj_splash_percent: s[22],
        proj_splash_radius: s[23],
        attack_speed_pct: s[24],
        mana: s[26],
        max_mana: s[27],
        actions: Vec::new(),
        buffs: Vec::new(),
        link: None,
        x,
        y,
        vx: 0.0,
        vy: 0.0,
        kb: None,
        target: -1,
        follower: Follower::new(pick_next_dist),
        repath_in: 0.0,
        intent_follow: false,
        class_id: s[28] as i32,
        statuses: [0.0; N_STATUS],
        on_attack: Vec::new(),
        on_damaged: Vec::new(),
        on_cast: Vec::new(),
        on_death: Vec::new(),
        standing: Vec::new(),
        cs_link: Vec::new(),
        damage_bonus: Vec::new(),
        resurrect: None,
        resurrected: false,
        death_handled: false,
    }
}

/// Split a flat table into one bucket per owner, keyed by column 0.
///
/// The template action and passive tables are addressed by template index
/// rather than by agent, and the rows are rewritten to point at agent 0 so
/// `attach_actions` and `attach_passives` can be reused verbatim on the single
/// summoned unit.
unsafe fn split_by_owner(
    ptr: *const f32,
    n_rows: i32,
    stride: usize,
    owners: usize,
) -> Vec<Vec<f32>> {
    let mut out = vec![Vec::new(); owners];
    if n_rows <= 0 || owners == 0 {
        return out;
    }
    for row in std::slice::from_raw_parts(ptr, (n_rows as usize) * stride).chunks_exact(stride) {
        let who = row[0] as i32;
        if who < 0 || who as usize >= owners {
            continue;
        }
        let mut r = row.to_vec();
        r[0] = 0.0;
        out[who as usize].extend_from_slice(&r);
    }
    out
}

/// Attach passive rows to their agents, filing each onto the list its hook
/// reads. A padding row (agent < 0) is skipped, as with the action table.
fn attach_passives(agents: &mut [Agent], rows: &[f32]) {
    for row in rows.chunks_exact(PASSIVE_STRIDE) {
        let who = row[0] as i32;
        if who < 0 || who as usize >= agents.len() {
            continue;
        }
        let mut p = Passive {
            kind: row[1] as i32,
            damage_mask: row[2] as i32,
            chance: row[3],
            duration: row[4],
            radius: row[5],
            cast_source: row[6] != 0.0,
            status: row[7] as i32,
            amount: row[8],
            amount2: row[9],
            flag: row[10] != 0.0,
            target_class: row[11] as i32,
            stats: [(-1, 0.0, false); MAX_BUFF_STATS],
            nstats: 0,
        };
        for k in 0..3 {
            let stat = row[12 + k * 3] as i32;
            if stat < 0 {
                continue;
            }
            p.stats[p.nstats] = (stat, row[13 + k * 3], row[14 + k * 3] != 0.0);
            p.nstats += 1;
        }
        let a = &mut agents[who as usize];
        match p.kind {
            P_BUFF_ATTACK | P_FEARSOME | P_PASSIVE_STUN | P_MANA_BREAK => {
                a.on_attack.push(p)
            }
            P_CRAGGY | P_UNTOUCHABLE => a.on_damaged.push(p),
            P_BUFF_ON_CAST | P_MULTICAST => a.on_cast.push(p),
            P_BUFF_ON_DEATH | P_STICKY_BLOOD | P_DEATH_DAMAGE | P_DEATH_SUMMON => {
                a.on_death.push(p)
            }
            P_COMPENSATION => a.standing.push(p),
            P_CS_LINK => a.cs_link.push(p),
            P_MODIFY_DAMAGE => a.damage_bonus.push((p.damage_mask, p.amount, p.flag)),
            P_RESURRECT => a.resurrect = Some((p.chance, p.amount, p.flag)),
            P_EVASION => a.evasions.push((p.chance, p.amount, p.flag)),
            P_VAMPIRISM => a.vampirisms.push((p.amount, p.flag)),
            P_EXPLODE => a.explodes.push((p.amount, p.radius, p.chance, p.damage_mask)),
            P_KNOCKBACK => {
                a.knockbacks
                    .push((p.chance, p.radius, p.amount, p.amount2, p.damage_mask))
            }
            _ => {}
        }
    }
}

/// Attach action rows to their agents, keeping the caller's order (which is
/// priority-descending, as the base tree's Selector expects).
fn attach_actions(agents: &mut [Agent], rows: &[f32]) {
    for row in rows.chunks_exact(ACTION_STRIDE) {
        // Batched calls pad every battle to the widest action count; padding
        // rows carry kind -1 and must not become phantom actions that consume
        // an agent's tick.
        if row[1] < 0.0 {
            continue;
        }
        let idx = row[0] as usize;
        if idx >= agents.len() {
            continue;
        }
        agents[idx].actions.push(ActionState {
            kind: row[1] as i32,
            range: row[2],
            cooldown: row[3],
            mana_cost: row[4],
            damage: row[6],
            radius: row[7],
            magical: row[8] != 0.0,
            amount: row[9],
            scope: row[10] as i32,
            duration: row[11],
            armor_pct: row[12],
            armor_flat: row[13],
            speed_flat: row[14],
            as_pct: row[15],
            template: row[16] as i32,
            count: row[17] as i32,
            status: row[18] as i32,
            cooldown_left: 0.0,
            anim: 0.0,
            phase: Phase::Gate,
        });
    }
}

/// Emit `n` doubles from a Python-seeded MT19937, for verifying RNG parity.
///
/// # Safety
/// `out` must have room for `n` doubles.
/// `random.choice` over a sequence of `len[k]` items, drawn from one stream.
///
/// `choice` goes through `_randbelow`, which is getrandbits with a rejection
/// loop rather than a float, so it consumes a different number of words than
/// `random()` does -- and how many depends on the values drawn. The on-death
/// damage skills pick their victim that way, so this is worth pinning.
#[no_mangle]
pub unsafe extern "C" fn despot_choice_probe(
    seed: u64,
    lens: *const i32,
    n: i32,
    out: *mut i32,
) {
    let mut r = rng::PyRandom::seed_u64(seed);
    let lens = std::slice::from_raw_parts(lens, n as usize);
    let out = std::slice::from_raw_parts_mut(out, n as usize);
    for k in 0..n as usize {
        out[k] = r.choice(lens[k].max(1) as usize) as i32;
    }
}

#[no_mangle]
pub unsafe extern "C" fn despot_rng_probe(seed: u64, n: i32, out: *mut f64) {
    let mut r = rng::PyRandom::seed_u64(seed);
    let slice = std::slice::from_raw_parts_mut(out, n as usize);
    for v in slice.iter_mut() {
        *v = r.random();
    }
}

/// Run one battle. `specs` is `n_units * STRIDE` floats; `sim/fast.py` owns the
/// field order. Returns the winner (0, 1, or -1 for a draw).
///
/// # Safety
/// All pointers must be valid for the lengths implied by `n_units`, `rows`
/// and `cols`. `out_hp` must have room for `n_units` floats.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn despot_battle(
    specs: *const f32,
    positions: *const f32,
    n_units: i32,
    rows: i32,
    cols: i32,
    tile: f32,
    walkable: *const u8,
    tick_hz: f32,
    max_seconds: f32,
    attack_anim: f32,
    recovery_anim: f32,
    projectile_speed: f32,
    time_horizon: f32,
    max_neighbours: i32,
    pick_next_dist: f32,
    melee_margin: f32,
    mana_per_dealt: f32,
    mana_per_taken: f32,
    actions: *const f32,
    n_actions: i32,
    passives: *const f32,
    n_passives: i32,
    templates: *const f32,
    n_templates: i32,
    tmpl_actions: *const f32,
    n_tmpl_actions: i32,
    tmpl_passives: *const f32,
    n_tmpl_passives: i32,
    seed: u64,
    out_ticks: *mut i32,
    out_damage: *mut f32,
    out_hp: *mut f32,
) -> i32 {
    let n = n_units as usize;
    let specs = std::slice::from_raw_parts(specs, n * STRIDE);
    let positions = std::slice::from_raw_parts(positions, n * 2);
    let walk = std::slice::from_raw_parts(walkable, (rows * cols) as usize);

    let grid = Grid::new(rows, cols, tile, walk.iter().map(|&b| b != 0).collect());
    let agents: Vec<Agent> = (0..n)
        .map(|i| {
            build_agent(
                &specs[i * STRIDE..(i + 1) * STRIDE],
                positions[i * 2],
                positions[i * 2 + 1],
                pick_next_dist,
            )
        })
        .collect();

    let template_rows: Vec<Vec<f32>> = if n_templates > 0 {
        std::slice::from_raw_parts(templates, (n_templates as usize) * STRIDE)
            .chunks_exact(STRIDE)
            .map(|c| c.to_vec())
            .collect()
    } else {
        Vec::new()
    };
    let tmpl_act_rows = split_by_owner(
        tmpl_actions, n_tmpl_actions, ACTION_STRIDE, template_rows.len());
    let tmpl_pas_rows = split_by_owner(
        tmpl_passives, n_tmpl_passives, PASSIVE_STRIDE, template_rows.len());
    let mut agents = agents;
    attach_actions(
        &mut agents,
        std::slice::from_raw_parts(actions, (n_actions as usize) * ACTION_STRIDE),
    );
    if n_passives > 0 {
        attach_passives(
            &mut agents,
            std::slice::from_raw_parts(passives, (n_passives as usize) * PASSIVE_STRIDE),
        );
    }

    let mut sim = Sim {
        agents,
        projectiles: Vec::new(),
        grid,
        dt: 1.0 / tick_hz,
        attack_anim,
        recovery_anim,
        projectile_speed,
        time_horizon,
        max_neighbours: max_neighbours as usize,
        melee_margin,
        templates: template_rows.clone(),
        template_actions: tmpl_act_rows,
        template_passives: tmpl_pas_rows,
        links: Vec::new(),
        pick_next_dist,
        mana_per_dealt,
        mana_per_taken,
        rng: rng::PyRandom::seed_u64(seed),
        damage: [0.0, 0.0],
        ticks: 0,
    };

    let winner = sim.run((max_seconds * tick_hz) as i32);

    *out_ticks = sim.ticks;
    let dmg = std::slice::from_raw_parts_mut(out_damage, 2);
    dmg[0] = sim.damage[0];
    dmg[1] = sim.damage[1];
    // Summons append to the agent vector, so only the units the caller passed
    // in have a slot in out_hp.
    let hp = std::slice::from_raw_parts_mut(out_hp, n);
    for (i, a) in sim.agents.iter().take(n).enumerate() {
        hp[i] = a.hp;
    }
    winner
}

/// Run `n_battles` independent battles across `threads` OS threads.
///
/// # Safety
/// As `despot_battle`, with each array holding `n_battles` consecutive blocks.
/// `seeds` must have `n_battles` entries.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn despot_battle_batch(
    specs: *const f32,
    positions: *const f32,
    units_per_battle: i32,
    n_battles: i32,
    rows: i32,
    cols: i32,
    tile: f32,
    walkable: *const u8,
    tick_hz: f32,
    max_seconds: f32,
    attack_anim: f32,
    recovery_anim: f32,
    projectile_speed: f32,
    time_horizon: f32,
    max_neighbours: i32,
    pick_next_dist: f32,
    melee_margin: f32,
    mana_per_dealt: f32,
    mana_per_taken: f32,
    actions: *const f32,
    actions_per_battle: i32,
    passives: *const f32,
    passives_per_battle: i32,
    templates: *const f32,
    n_templates: i32,
    tmpl_actions: *const f32,
    n_tmpl_actions: i32,
    tmpl_passives: *const f32,
    n_tmpl_passives: i32,
    seeds: *const u64,
    threads: i32,
    out_winner: *mut i32,
    out_ticks: *mut i32,
    out_damage: *mut f32,
) -> i32 {
    let nb = n_battles as usize;
    let upb = units_per_battle as usize;
    let specs = std::slice::from_raw_parts(specs, nb * upb * STRIDE);
    let positions = std::slice::from_raw_parts(positions, nb * upb * 2);
    let seeds = std::slice::from_raw_parts(seeds, nb);
    let walk_raw = std::slice::from_raw_parts(walkable, (rows * cols) as usize);
    let walk: Vec<bool> = walk_raw.iter().map(|&b| b != 0).collect();
    let template_rows: Vec<Vec<f32>> = if n_templates > 0 {
        std::slice::from_raw_parts(templates, (n_templates as usize) * STRIDE)
            .chunks_exact(STRIDE)
            .map(|c| c.to_vec())
            .collect()
    } else {
        Vec::new()
    };
    let apb = actions_per_battle.max(0) as usize;
    let action_rows: &[f32] = if apb > 0 {
        std::slice::from_raw_parts(actions, nb * apb * ACTION_STRIDE)
    } else {
        &[]
    };
    let tmpl_act_rows = split_by_owner(
        tmpl_actions, n_tmpl_actions, ACTION_STRIDE, template_rows.len());
    let tmpl_pas_rows = split_by_owner(
        tmpl_passives, n_tmpl_passives, PASSIVE_STRIDE, template_rows.len());
    let ppb = passives_per_battle.max(0) as usize;
    let passive_rows: &[f32] = if ppb > 0 {
        std::slice::from_raw_parts(passives, nb * ppb * PASSIVE_STRIDE)
    } else {
        &[]
    };

    let winners_ptr = out_winner as usize;
    let ticks_ptr = out_ticks as usize;
    let damage_ptr = out_damage as usize;

    let nthreads = (threads.max(1) as usize).min(nb.max(1));
    let chunk = nb.div_ceil(nthreads);

    std::thread::scope(|scope| {
        for t in 0..nthreads {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(nb);
            if lo >= hi {
                continue;
            }
            let walk = &walk;
            let action_rows = action_rows;
            let passive_rows = passive_rows;
            let template_rows = &template_rows;
            let tmpl_act_rows = &tmpl_act_rows;
            let tmpl_pas_rows = &tmpl_pas_rows;
            scope.spawn(move || {
                for b in lo..hi {
                    let grid = Grid::new(rows, cols, tile, walk.clone());
                    let agents: Vec<Agent> = (0..upb)
                        .map(|i| {
                            let base = (b * upb + i) * STRIDE;
                            let p = (b * upb + i) * 2;
                            build_agent(
                                &specs[base..base + STRIDE],
                                positions[p],
                                positions[p + 1],
                                pick_next_dist,
                            )
                        })
                        .filter(|a| a.hp > 0.0)
                        .collect();
                    let mut agents = agents;
                    if apb > 0 {
                        let lo = b * apb * ACTION_STRIDE;
                        attach_actions(&mut agents, &action_rows[lo..lo + apb * ACTION_STRIDE]);
                    }
                    if ppb > 0 {
                        let lo = b * ppb * PASSIVE_STRIDE;
                        attach_passives(&mut agents, &passive_rows[lo..lo + ppb * PASSIVE_STRIDE]);
                    }
                    let mut sim = Sim {
                        agents,
                        projectiles: Vec::new(),
                        grid,
                        dt: 1.0 / tick_hz,
                        attack_anim,
                        recovery_anim,
                        projectile_speed,
                        time_horizon,
                        max_neighbours: max_neighbours as usize,
                        melee_margin,
                        templates: template_rows.clone(),
                        template_actions: tmpl_act_rows.clone(),
                        template_passives: tmpl_pas_rows.clone(),
                        links: Vec::new(),
                        pick_next_dist,
                        mana_per_dealt,
                        mana_per_taken,
                        rng: rng::PyRandom::seed_u64(seeds[b]),
                        damage: [0.0, 0.0],
                        ticks: 0,
                    };
                    let w = sim.run((max_seconds * tick_hz) as i32);
                    unsafe {
                        *(winners_ptr as *mut i32).add(b) = w;
                        *(ticks_ptr as *mut i32).add(b) = sim.ticks;
                        *(damage_ptr as *mut f32).add(b * 2) = sim.damage[0];
                        *(damage_ptr as *mut f32).add(b * 2 + 1) = sim.damage[1];
                    }
                }
            });
        }
    });
    0
}
