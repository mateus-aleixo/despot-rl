"""Render one run to video: the map the agent walks, and every fight it takes.

The sim is headless, so this is the only thing in the repository that draws
anything. It records rather than renders inline: `RecordingBattle` snapshots
agent positions every few ticks while the fight resolves normally, the run loop
snapshots the level map and the HUD around each decision, and the drawing
happens afterwards over plain data. Nothing here can change a fight's outcome.

Fights have to resolve in the Python oracle for there to be per-tick state to
record, so this forces `fast_core=False` and is correspondingly slow: seconds
per fight rather than milliseconds. It is a demo, not a measurement tool.

    python tools/render_run.py --agent runs/shelf2m_described_s1.pt --seed 30001
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw, ImageFont

import sim.run as sim_run
from rl.env import NON_MOVE_ACTIONS, DespotRunEnv
from sim.run import UNKNOWN
from rl.heuristic import heuristic as policy_heuristic
from rl.placement import POLICIES
from sim.battle import Battle
from sim.data import load_ruleset

W, H = 1280, 720
FPS = 30


# ------------------------------------------------------------------ recording

@dataclass
class BattleTick:
    t: float
    agents: list          # (x, y, team, hp_frac, radius, melee, alive, letter, level)
    shots: list           # (x, y, team)


@dataclass
class Fight:
    rows: int
    cols: int
    tile: float
    ticks: list = field(default_factory=list)
    won: bool | None = None
    seconds: float = 0.0


@dataclass
class Scene:
    """One decision, and the fight it caused if it caused one."""
    step: int
    action: str
    hud: dict
    rooms: list           # (id, row, col, kind, cleared, current, state, portal)
    edges: list           # ((row, col), (row, col))
    context: list         # the decision as lines of text (the end card uses them)
    fight: Fight | None = None
    result: str = ""
    panel: dict = field(default_factory=dict)   # what the room shows: a shop, a fight


_PENDING: list[Fight] = []


class RecordingBattle(Battle):
    """`Battle`, plus a snapshot every `EVERY` ticks. Overrides nothing that
    resolves anything: `step` calls up first and only then reads state."""

    EVERY = 2

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rec = Fight(rows=self.grid.rows, cols=self.grid.cols,
                         tile=self.grid.tile)
        _PENDING.append(self.rec)

    def _snap(self) -> BattleTick:
        ags = []
        for ag in self.agents:
            # The game draws neither a unit inside a MegaNovice (Unrendered)
            # nor a MegaNovice that split back (unsummoned, not a corpse).
            if ag.absorbed or ag.unsummoned:
                continue
            if ag.corpse and ag.hp <= 0 and not ag.dying:
                frac = 0.0
            else:
                frac = max(0.0, min(1.0, ag.hp / max(1.0, ag.spec.health)))
            # Two letters: Monk, MegaNovice and Mancrack, or Cultist and the
            # CatBot a Monk summons, would otherwise share one.
            ags.append((ag.x, ag.y, ag.team, frac, ag.spec.radius,
                        ag.spec.melee, ag.alive, (ag.spec.cls or "?")[:2],
                        ag.spec.level))
        shots = [(p.x, p.y, p.team) for p in self.projectiles]
        return BattleTick(t=self.tick_count * self.dt, agents=ags, shots=shots)

    def step(self) -> None:
        super().step()
        if self.tick_count % self.EVERY == 0:
            self.rec.ticks.append(self._snap())

    def run(self):
        self.rec.ticks.append(self._snap())
        res = super().run()
        self.rec.ticks.append(self._snap())
        self.rec.won = res.winner == 0
        self.rec.seconds = res.seconds
        return res


# ------------------------------------------------------------------ the run

def squad_line(st) -> list[str]:
    out = []
    for h in st.squad[:12]:
        out.append(f"L{h.level}  {h.item or 'unarmed'}")
    if len(st.squad) > 12:
        out.append(f"... and {len(st.squad) - 12} more")
    return out


def context_lines(st, room, action: str) -> list[str]:
    """What the agent was looking at when it chose, as text."""
    kind = room.kind
    if kind == "item_shop":
        lines = ["item shop, level %d" % st.shop_level]
        for i, name in enumerate(st.offer):
            if name is None:
                lines.append(f"  {i}  -- sold --")
            else:
                lines.append(f"  {i}  {name}  {st.item_cost(name):.0f}g")
        lines.append(f"  reroll {st.roll_cost:.0f}g   "
                     f"upgrade {st.upgrade_cost:.0f}g")
        return lines
    if kind == "food_shop":
        sizes, costs = st.food_packs
        stock = room.food_stock or [0] * len(sizes)
        lines = ["food shop"]
        for i, (s, c) in enumerate(zip(sizes, costs)):
            lines.append(f"  {i}  {s:.0f} food  {c:.0f}g   x{stock[i]} left")
        return lines
    if kind in ("mutation_shop", "mutation"):
        lines = [f"{kind.replace('_', ' ')}, {room.takes_left} take(s) left"
                 if kind == "mutation_shop" else
                 f"shrine, {room.stock or 0} left"]
        for i, m in enumerate((room.offer_mutations or [])[:8]):
            if m is None:
                lines.append(f"  {i}  -- taken --")
            else:
                lines.append(f"  {i}  {m.get('Name') or m.get('ID')}")
        return lines
    if kind in ("fight", "boss"):
        return [f"{kind} room", f"  squad power {st.squad_power():.0f}",
                f"  room power  {st.room_power(kind):.0f}"]
    return [f"{kind.replace('_', ' ')} room"]


def room_panel(st, room) -> dict:
    """What the room shows the player, for drawing it the way the game does:
    an item shop's shelf with prices, the food shop's packs, a mutation shelf,
    or the two powers of a fight."""
    kind = room.kind
    if kind == "item_shop":
        return {"type": "item_shop", "level": st.shop_level,
                "items": [None if n is None else (n, st.item_cost(n)) for n in st.offer],
                "reroll": st.roll_cost, "upgrade": st.upgrade_cost}
    if kind == "food_shop":
        sizes, costs = st.food_packs
        stock = room.food_stock or [0] * len(sizes)
        return {"type": "food_shop", "items": list(zip(sizes, costs, stock))}
    if kind in ("mutation_shop", "mutation"):
        return {"type": "mutations", "shop": kind == "mutation_shop",
                "left": room.takes_left if kind == "mutation_shop" else (room.stock or 0),
                "items": [None if m is None else str(m.get("Name") or m.get("ID"))
                          for m in (room.offer_mutations or [])[:8]]}
    if kind in ("fight", "boss"):
        return {"type": "fight", "boss": kind == "boss",
                "squad": st.squad_power(), "room": st.room_power(kind)}
    return {"type": kind}


def squad_units(st) -> list[tuple]:
    """(level, health share, class) per human, for drawing the squad in the room."""
    return [(h.level, h.hp, h.cls or st.item_class(h.item)) for h in st.squad]


def class_counts(st) -> list[tuple[str, int]]:
    """How many of each class the squad fields, like the game's strip of class
    icons along the top bar."""
    counts: dict[str, int] = {}
    for h in st.squad:
        c = h.cls or st.item_class(h.item)
        counts[c] = counts.get(c, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def record(tables, policy, seed: int, placement, max_steps: int | None,
           stop_level: int | None) -> list[Scene]:
    sim_run.Battle = RecordingBattle          # fights record themselves
    env = DespotRunEnv(tables=tables, seed=seed, placement_policy=placement,
                       fast_core=False)
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    scenes: list[Scene] = []
    n = 0
    cap = max_steps or env.max_steps

    while mask.any() and n < cap:
        st = env.state
        room = st.rooms.rooms[st.room]
        action_i = policy(env, obs, mask)
        # The index reads better than the decoded action: a move decodes to the
        # room it lands on, and a room id says nothing on screen.
        if action_i < env.n_moves:
            name = env.move_names[action_i]
            label = (name.replace("_", " ") if name.startswith("teleport")
                     else "move " + name)
        else:
            label = env.non_move_actions[action_i - env.n_moves].replace("_", " ")

        # The room state rides along so the map can be drawn as the agent sees
        # it. Drawing the whole level was fine while the sim handed the whole
        # level over; with fog it would show the viewer a map the policy is not
        # playing with.
        rooms = [(r.id, r.row, r.col, r.kind, r.cleared, r.id == st.room, r.state,
                  st.rooms.is_portal(r.id))
                 for r in st.rooms.rooms.values()]
        edges = []
        for rid, r in st.rooms.rooms.items():
            # Doors only: a portal link drawn as a line would cut across the map.
            for other in st.rooms.doors(rid):
                o = st.rooms.rooms[other]
                if (r.row, r.col) < (o.row, o.col) and UNKNOWN not in (r.state, o.state):
                    edges.append(((r.row, r.col), (o.row, o.col)))

        hud = {"level": st.level, "room": st.room, "kind": room.kind,
               "gold": st.gold, "food": st.food.amount,
               "moves": st.food.moves_left, "hunger": st.food.hunger_level,
               "squad": len(st.squad), "muts": len(st.mutations),
               "power": st.squad_power(), "step": n + 1,
               # Which of the eight `ChipChoice/Squads.json` rosters this run
               # drew. The env cycles them by episode seed, so a run is not
               # playing "the" squad any more.
               "squad_name": env.squad_name or "?",
               "squad_lines": squad_line(st),
               "max_moves": st.food.max_moves, "classes": class_counts(st),
               "units": squad_units(st)}
        ctx = context_lines(st, room, label)
        panel = room_panel(st, room)

        _PENDING.clear()
        obs, r, term, trunc, info = env.step(action_i)
        mask = info["action_mask"]
        n += 1

        fight = _PENDING[0] if _PENDING else None
        res = info.get("result") or {}
        note = ""
        if "won" in res:
            note = ("won, %d left" % res.get("survivors", 0)) if res["won"] \
                else "wiped"
        elif res.get("bought"):
            note = str(res["bought"])

        scenes.append(Scene(step=n, action=label, hud=hud, rooms=rooms,
                            edges=edges, context=ctx, fight=fight, result=note,
                            panel=panel))

        if term or trunc:
            break
        if stop_level and env.state.level > stop_level:
            break

    st = env.state
    fights = sum(1 for s in scenes if s.fight)
    won = sum(1 for s in scenes if s.fight and s.fight.won)
    end = [f"reached level {st.level}" if st.finished
           else f"step cap at level {st.level}",
           f"  {n} decisions, {fights} fights, {won} won",
           f"  {len(st.mutations)} mutations held",
           f"  {st.gold:.0f} gold and {st.food.amount:.0f} food left",
           f"  {len(st.squad)} of the squad still standing"]
    hud = dict(scenes[-1].hud)
    hud.update(level=st.level, gold=st.gold, food=st.food.amount,
               moves=st.food.moves_left, squad=len(st.squad),
               muts=len(st.mutations), power=st.squad_power(), step=n,
               squad_lines=squad_line(st), classes=class_counts(st),
               units=squad_units(st))
    scenes.append(Scene(step=n, action="__end__", hud=hud,
                        rooms=scenes[-1].rooms, edges=scenes[-1].edges,
                        context=end))
    return scenes


# ------------------------------------------------------------------ drawing
#
# Laid out after the game's own screen (a recorded run, 2026-11-22): the
# resources along the top left -- gold, humans, food with its moves bar -- and a
# strip of the squad's classes beside them; the minimap under them, headed
# "LEVEL: n", the current room in cyan brackets; the room as the stage, a stone
# wall over a purple floor, with each shop drawn on the floor where the game
# draws it; pink humans with green health bars and their level, teal enemies
# with red ones; and a unit panel along the bottom, with pause, fast-forward and
# FIGHT at the bottom right. Shapes stand in for the game's sprites: the
# repository ships no game art.

C_BG = (24, 18, 31)
C_PANEL = (38, 31, 52)
C_PANEL_EDGE = (62, 52, 86)
C_WALL = (38, 33, 58)
C_WALL_TRIM = (74, 92, 136)
C_WALL_DARK = (27, 23, 42)
C_FLOOR = (58, 50, 82)
C_FLOOR_EDGE = (46, 39, 66)
C_STRIPE = (217, 79, 120)
C_TEAL = (92, 210, 202)
C_ORANGE = (240, 163, 91)
C_PINK = (255, 112, 168)
C_PINK_DK = (122, 44, 84)
C_FOE = (86, 204, 172)
C_FOE_DK = (30, 84, 76)
C_HP = (96, 212, 140)
C_HP_BACK = (28, 46, 40)
C_HP_FOE = (226, 68, 110)
C_TEXT = (214, 208, 234)
C_DIM = (124, 116, 154)
C_DIGIT_DIM = (66, 58, 90)
C_GOLD = (160, 192, 206)
C_CORPSE = (44, 38, 58)
C_SLOT = (48, 41, 68)

ROOM_ICON = {  # minimap tile: glyph and colour
    "start": ("◎", C_TEAL), "fight": ("☠", (200, 196, 214)), "boss": ("☠", C_HP_FOE),
    "item_shop": ("$", (232, 184, 75)), "food_shop": ("≡", C_ORANGE),
    "mutation": ("✶", (176, 130, 230)), "mutation_shop": ("✶", (176, 130, 230)),
    "talent_shop": ("T", C_TEAL), "consumable_shop": ("✚", C_ORANGE),
    "quest": ("?!", (236, 214, 118)), "quest_extra": ("!", (190, 170, 100)),
    "secret": ("?", (150, 150, 170)), "empty": ("", C_DIM),
}
ROOM_TITLE = {"start": "START", "quest": "QUEST", "quest_extra": "QUEST", "secret": "SECRET",
              "talent_shop": "TALENTS", "consumable_shop": "CONSUMABLES", "empty": ""}


def _font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{n}", size)
        except OSError:
            continue
    return ImageFont.load_default()


F_DIGIT = _font(["consolab.ttf", "consola.ttf"], 26)
F_TITLE = _font(["consolab.ttf", "consola.ttf"], 24)
F_LABEL = _font(["consolab.ttf", "consola.ttf"], 15)
F_SMALL = _font(["consola.ttf"], 13)
F_TINY = _font(["consolab.ttf", "consola.ttf"], 11)
F_BIG = _font(["consolab.ttf", "consola.ttf"], 36)
F_ICON = _font(["seguisym.ttf", "segoeui.ttf"], 17)
F_ICON_L = _font(["seguisym.ttf", "segoeui.ttf"], 30)
F_ICON_S = _font(["seguisym.ttf", "segoeui.ttf"], 12)

TOP = (10, 6, 470, 46)
MINI = (10, 52, 204, 300)
LEFT = (10, 308, 204, 610)
ROOM = (212, 52, 1270, 610)
WALL_H = 70
BOTTOM = (212, 618, 1270, 712)


def stripes(d, x0, y0, x1, y1, step=9):
    """The pink-and-dark diagonal stripes of the game's door thresholds."""
    d.rectangle((x0, y0, x1, y1), fill=C_STRIPE)
    h = y1 - y0
    k = x0 - h
    while k < x1:
        d.polygon([(max(x0, k), y1), (min(x1, k + h), y0), (min(x1, k + h + step / 2), y0),
                   (max(x0, k + step / 2), y1)], fill=(92, 34, 58))
        k += step


def counter(d, x, y, icon, value, digits, col):
    """A top-bar counter: icon, then zero-padded digits with the leading
    zeros dimmed, as the game prints its gold, humans and food."""
    d.text((x, y + 3), icon, font=F_ICON, fill=col)
    s = f"{int(max(0, value)):0{digits}d}"[-digits:]
    lead = len(s) - len(s.lstrip("0")) if s.strip("0") else digits - 1
    cx = x + 22
    for i, ch in enumerate(s):
        d.text((cx, y), ch, font=F_DIGIT, fill=C_DIGIT_DIM if i < lead else col)
        cx += 15
    return cx


def draw_topbar(d, sc, sub):
    h = sc.hud
    d.rounded_rectangle(TOP, 6, fill=C_PANEL, outline=C_PANEL_EDGE)
    x = counter(d, 20, 9, "◎", h["gold"], 6, C_GOLD)
    x = counter(d, x + 14, 9, "♟", h["squad"], 2, C_PINK)
    fx = x + 14
    x = counter(d, fx, 9, "≡", h["food"], 4, C_ORANGE)
    # The bar under the food is the moves left before hunger sets in.
    frac = max(0.0, min(1.0, h["moves"] / max(1, h.get("max_moves") or 1)))
    bx0, bx1 = fx + 22, x - 3
    d.rectangle((bx0, 39, bx1, 42), fill=C_HP_BACK)
    d.rectangle((bx0, 39, bx0 + (bx1 - bx0) * frac, 42), fill=C_HP if frac > 0.34 else C_HP_FOE)
    cs = (h.get("classes") or [])[:8]
    if cs:
        cx = TOP[2] + 8
        d.rounded_rectangle((cx, 6, cx + 16 + 46 * len(cs), 46), 6, fill=C_PANEL,
                            outline=C_PANEL_EDGE)
        for i, (c, n) in enumerate(cs):
            px = cx + 8 + i * 46 + 23
            d.text((px, 11), c[:2].upper(), font=F_LABEL, fill=C_ORANGE, anchor="ma")
            d.text((px, 29), str(n), font=F_TINY, fill=C_DIM, anchor="ma")
    d.text((1262, 10), "despot-rl", font=F_LABEL, fill=C_TEXT, anchor="ra")
    d.text((1262, 29), f"{sub}  ·  step {h['step']}", font=F_SMALL, fill=C_DIM, anchor="ra")


def brackets(d, box, col, arm=6, width=2):
    x0, y0, x1, y1 = box
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(cx, cy), (cx + sx * arm, cy)], fill=col, width=width)
        d.line([(cx, cy), (cx, cy + sy * arm)], fill=col, width=width)


def draw_minimap(d, sc):
    x0, y0, x1, y1 = MINI
    d.rounded_rectangle(MINI, 6, fill=C_PANEL, outline=C_PANEL_EDGE)
    d.text(((x0 + x1) / 2, y0 + 10), f"LEVEL: {sc.hud['level']}", font=F_LABEL,
           fill=C_ORANGE, anchor="ma")
    rows = [r[1] for r in sc.rooms]
    cols = [r[2] for r in sc.rooms]
    if not rows:
        return
    span_r = max(rows) - min(rows) + 1
    span_c = max(cols) - min(cols) + 1
    ax0, ay0, ax1, ay1 = x0 + 10, y0 + 34, x1 - 10, y1 - 10
    cell = min((ax1 - ax0) / span_c, (ay1 - ay0) / span_r, 36)
    size = cell - 4
    ox = ax0 + ((ax1 - ax0) - cell * span_c) / 2
    oy = ay0 + ((ay1 - ay0) - cell * span_r) / 2
    for rid, r, c, kind, cleared, cur, state, portal in sc.rooms:
        # `V_MinimapRoom` draws no icon at all for an Unknown room; the bounds
        # above still count it, so the map does not jump as the level opens.
        if state == UNKNOWN:
            continue
        px = ox + (c - min(cols)) * cell + 2
        py = oy + (r - min(rows)) * cell + 2
        box = (px, py, px + size, py + size)
        fill = (52, 110, 132) if cur else ((42, 36, 60) if cleared else (60, 52, 88))
        d.rectangle(box, fill=fill)
        if cur:
            brackets(d, (px - 2, py - 2, px + size + 2, py + size + 2), C_TEAL)
        glyph, col = ROOM_ICON.get(kind, ("", C_DIM))
        if cleared and not cur:
            col = tuple(int(v * 0.5) for v in col)
        if glyph:
            d.text((px + size / 2, py + size / 2), glyph,
                   font=F_ICON if size >= 22 else F_ICON_S, fill=col, anchor="mm")
        if portal:
            pr = max(3.0, size * 0.14)
            qx, qy = px + size - pr - 2, py + pr + 2
            d.ellipse((qx - pr, qy - pr, qx + pr, qy + pr), outline=C_TEAL, width=2)


def draw_squad_list(d, sc):
    x0, y0, x1, y1 = LEFT
    d.rounded_rectangle(LEFT, 6, fill=C_PANEL, outline=C_PANEL_EDGE)
    d.text((x0 + 12, y0 + 9), f"SQUAD {sc.hud.get('squad_name', '')}".strip(),
           font=F_LABEL, fill=C_ORANGE)
    y = y0 + 34
    for line in sc.hud["squad_lines"]:
        d.text((x0 + 12, y), line, font=F_SMALL, fill=C_TEXT)
        y += 18
        if y > y1 - 18:
            break


def draw_room(d) -> tuple:
    """The stage: a stone wall over a purple floor, a door arch at the top and
    striped thresholds. Returns the floor's box."""
    x0, y0, x1, y1 = ROOM
    d.rectangle(ROOM, fill=C_FLOOR_EDGE)
    d.rectangle((x0, y0, x1, y0 + WALL_H), fill=C_WALL)
    d.rectangle((x0, y0, x1, y0 + 6), fill=C_WALL_TRIM)
    for k in range(x0 + 10, x1 - 70, 76):
        d.rectangle((k, y0 + 16, k + 62, y0 + WALL_H - 10), fill=C_WALL_DARK,
                    outline=(52, 46, 74))
    cx = (x0 + x1) / 2
    d.rounded_rectangle((cx - 64, y0 + 4, cx + 64, y0 + WALL_H + 2), 22, fill=C_WALL_DARK,
                        outline=C_WALL_TRIM, width=2)
    stripes(d, cx - 52, y0 + WALL_H - 8, cx + 52, y0 + WALL_H)
    fl = (x0 + 16, y0 + WALL_H + 8, x1 - 16, y1 - 16)
    d.rectangle(fl, fill=C_FLOOR)
    # floor tiles, faint, as the game's slabs
    for gx in range(int(fl[0]), int(fl[2]), 72):
        d.line([(gx, fl[1]), (gx, fl[3])], fill=(64, 56, 90))
    for gy in range(int(fl[1]), int(fl[3]), 72):
        d.line([(fl[0], gy), (fl[2], gy)], fill=(64, 56, 90))
    sy = (fl[1] + fl[3]) / 2
    stripes(d, x1 - 12, sy - 60, x1 - 4, sy + 60, step=7)
    return fl


def draw_unit(d, px, py, rad, team, frac, alive, letter, level):
    if not alive:
        d.ellipse((px - rad, py - rad * 0.55, px + rad, py + rad * 0.55), fill=C_CORPSE)
        return
    fill, edge = (C_PINK_DK, C_PINK) if team == 0 else (C_FOE_DK, C_FOE)
    d.ellipse((px - rad, py - rad, px + rad, py + rad), fill=fill, outline=edge, width=2)
    if rad >= 9:
        d.text((px, py), letter, font=F_TINY, fill=edge, anchor="mm")
    bw = max(18.0, rad * 2)
    by = py - rad - 8
    d.rectangle((px - bw / 2, by, px + bw / 2, by + 3), fill=C_HP_BACK)
    d.rectangle((px - bw / 2, by, px - bw / 2 + bw * max(0.0, min(1.0, frac)), by + 3),
                fill=C_HP if team == 0 else C_HP_FOE)
    if team == 0 and level:
        d.text((px - bw / 2 - 3, by + 2), str(level), font=F_TINY, fill=C_TEXT, anchor="rm")


def draw_idle_squad(d, sc, fl):
    """Between fights the squad stands in the room, on its left, the way the
    game shows it while the player shops."""
    units = sc.hud.get("units") or []
    per_row = 5
    gap = 40
    x0 = fl[0] + 90
    y0 = (fl[1] + fl[3]) / 2 - gap * ((len(units) - 1) // per_row) / 2
    for i, (level, hp, cls) in enumerate(units[:25]):
        px = x0 + (i % per_row) * gap
        py = y0 + (i // per_row) * gap
        draw_unit(d, px, py, 14, 0, hp, True, (cls or "?")[:2], level)


def wrap_name(name: str, width: int = 10) -> list[str]:
    """An item name on at most two lines, broken at a hyphen where one fits."""
    if len(name) <= width:
        return [name]
    cut = name.rfind("-", 0, width + 1)
    if cut <= 0:
        return [name[:width], name[width:2 * width]]
    return [name[:cut + 1], name[cut + 1:cut + 1 + width]]


def button(d, box, label, lit=False, col=C_TEXT):
    d.rounded_rectangle(box, 4, fill=(56, 48, 78) if lit else C_PANEL,
                        outline=C_TEAL if lit else C_PANEL_EDGE, width=2)
    d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font=F_LABEL,
           fill=col if lit else C_DIM, anchor="mm")


def draw_panel(d, sc, fl):
    """What the room shows between fights, where the game shows it."""
    p = sc.panel or {}
    kind = p.get("type")
    rx = fl[0] + (fl[2] - fl[0]) * 0.52
    top = fl[1] + 34
    if kind == "item_shop":
        d.text((rx + 190, top), f"SHOP LVL {p['level']}", font=F_TITLE, fill=C_ORANGE, anchor="ma")
        items = p["items"]
        for i, it in enumerate(items[:8]):
            cx = rx + 40 + (i % 4) * 100
            cy = top + 60 + (i // 4) * 120
            d.rounded_rectangle((cx, cy, cx + 76, cy + 60), 6,
                                fill=C_SLOT, outline=C_PANEL_EDGE)
            if it is None:
                d.text((cx + 38, cy + 30), "sold", font=F_SMALL, fill=C_DIM, anchor="mm")
                continue
            name, cost = it
            for k, part in enumerate(wrap_name(name)):
                d.text((cx + 38, cy + 22 + 18 * k), part, font=F_TINY, fill=C_TEXT, anchor="mm")
            d.text((cx + 38, cy + 74), f"{cost:.0f}", font=F_LABEL, fill=C_GOLD, anchor="mm")
        by = top + 60 + 2 * 120 + 10
        button(d, (rx + 60, by, rx + 200, by + 32), f"REROLL - {p['reroll']:.0f}", lit=True)
        button(d, (rx + 220, by, rx + 380, by + 32), f"UPGRADE - {p['upgrade']:.0f}", lit=True)
    elif kind == "food_shop":
        d.text((rx + 190, top), "FOOD SHOP", font=F_TITLE, fill=C_ORANGE, anchor="ma")
        for i, (size, cost, stock) in enumerate(p["items"][:4]):
            cx = rx + 40 + i * 110
            cy = top + 70
            d.rounded_rectangle((cx, cy, cx + 86, cy + 64), 6, fill=C_SLOT,
                                outline=C_PANEL_EDGE)
            d.text((cx + 43, cy + 24), f"{size:.0f} food", font=F_SMALL, fill=C_TEXT, anchor="mm")
            d.text((cx + 43, cy + 44), f"x{stock}", font=F_SMALL, fill=C_DIM, anchor="mm")
            d.text((cx + 43, cy + 80), f"{cost:.0f}", font=F_LABEL, fill=C_GOLD, anchor="mm")
    elif kind == "mutations":
        title = "MUTATIONS" if p.get("shop") else "SHRINE"
        d.text((rx + 190, top), title, font=F_TITLE, fill=C_ORANGE, anchor="ma")
        for i, name in enumerate(p["items"][:8]):
            cx = rx + 30 + (i % 2) * 200
            cy = top + 56 + (i // 2) * 52
            d.rounded_rectangle((cx, cy, cx + 186, cy + 40), 6, fill=C_SLOT,
                                outline=C_PANEL_EDGE)
            d.text((cx + 93, cy + 20), name or "taken", font=F_SMALL,
                   fill=C_TEXT if name else C_DIM, anchor="mm")
        d.text((rx + 190, top + 56 + 4 * 52 + 8), f"{p['left']} left", font=F_SMALL,
               fill=C_DIM, anchor="ma")
    elif kind == "fight":
        # The game marks where the enemies stand with a red skull before the fight.
        sx, sy = fl[2] - 170, (fl[1] + fl[3]) / 2
        r = 34 if p.get("boss") else 26
        d.ellipse((sx - r, sy - r, sx + r, sy + r), outline=C_HP_FOE, width=3)
        d.text((sx, sy), "☠", font=F_ICON_L, fill=C_HP_FOE, anchor="mm")
        d.text((sx, sy + r + 14), f"POWER {p['room']:.0f}", font=F_SMALL, fill=C_DIM, anchor="ma")
        d.text((fl[0] + 90, fl[3] - 40), f"SQUAD POWER {p['squad']:.0f}", font=F_SMALL, fill=C_DIM)
    else:
        title = ROOM_TITLE.get(kind or "", (kind or "").replace("_", " ").upper())
        if title:
            d.text((rx + 190, top), title, font=F_TITLE, fill=C_ORANGE, anchor="ma")


def draw_end(d, sc, fl):
    cx = (fl[0] + fl[2]) / 2
    box = (cx - 230, fl[1] + 60, cx + 230, fl[1] + 300)
    d.rounded_rectangle(box, 8, fill=C_PANEL, outline=C_TEAL, width=2)
    d.text((cx, box[1] + 18), "RUN OVER", font=F_TITLE, fill=C_ORANGE, anchor="ma")
    y = box[1] + 64
    for line in sc.context:
        d.text((box[0] + 30, y), line.strip(), font=F_LABEL, fill=C_TEXT)
        y += 30


def draw_bottom(d, sc, tick=None, alive=None):
    x0, y0, x1, y1 = BOTTOM
    h = sc.hud
    tab = "RUN OVER" if sc.action == "__end__" else sc.action.upper()
    tw = d.textlength(tab, font=F_LABEL) + 26
    d.rounded_rectangle((x0 + 80, y0, x0 + 80 + tw, y0 + 26), 4, fill=C_PANEL,
                        outline=C_PANEL_EDGE)
    d.text((x0 + 93, y0 + 5), tab, font=F_LABEL, fill=C_ORANGE)
    box = (x0 + 80, y0 + 30, x0 + 780, y1 - 4)
    d.rounded_rectangle(box, 6, fill=C_PANEL, outline=C_PANEL_EDGE)
    stats = [("POWER", f"{h['power']:.0f}"), ("MUTATIONS", f"{h['muts']}"),
             ("MOVES", f"{h['moves']}/{h.get('max_moves') or 0}"), ("HUNGER", f"{h['hunger']}")]
    for i, (k, v) in enumerate(stats):
        sx = box[0] + 22 + i * 170
        d.text((sx, box[1] + 9), k, font=F_TINY, fill=C_DIM)
        d.text((sx, box[1] + 25), v, font=F_TITLE, fill=C_TEXT)
    # pause, fast-forward and FIGHT, lit while a fight plays
    fighting = tick is not None
    button(d, (x1 - 186, y0 + 22, x1 - 154, y0 + 50), "II", lit=fighting)
    button(d, (x1 - 148, y0 + 22, x1 - 116, y0 + 50), "»", lit=fighting, col=C_ORANGE)
    button(d, (x1 - 108, y0 + 54, x1 - 4, y1 - 4), "FIGHT", lit=fighting, col=C_TEAL)
    if fighting:
        d.text((x1 - 196, y0 + 60), f"{tick.t:4.1f}s", font=F_LABEL, fill=C_TEXT, anchor="ra")
        if alive is not None:
            d.text((x1 - 196, y0 + 80), f"{alive[0]} v {alive[1]}", font=F_LABEL,
                   fill=C_TEXT, anchor="ra")


def draw_arena(d, sc, tick, fl, show_result=False):
    f = sc.fight
    ww, wh = f.cols * f.tile, f.rows * f.tile
    pad = 20
    scale = min((fl[2] - fl[0] - 2 * pad) / ww, (fl[3] - fl[1] - 2 * pad) / wh)
    ox = fl[0] + ((fl[2] - fl[0]) - ww * scale) / 2
    oy = fl[1] + ((fl[3] - fl[1]) - wh * scale) / 2
    alive = [0, 0]
    # corpses first, so the living stand on top of them
    order = sorted(tick.agents, key=lambda a: a[6])
    for (wx, wy, team, frac, radius, melee, is_alive, letter, level) in order:
        px, py = ox + wx * scale, oy + wy * scale
        # Drawn at three quarters of the body radius: the game's sprites are
        # slimmer than the circles the sim moves around.
        rad = max(6.0, radius * scale * 0.75)
        if is_alive:
            alive[team] += 1
        draw_unit(d, px, py, rad, team, frac, is_alive, letter, level)
    for (wx, wy, team) in tick.shots:
        px, py = ox + wx * scale, oy + wy * scale
        col = (127, 227, 255) if team == 0 else C_PINK
        d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=col)
    if show_result and sc.result:
        # A banner over the wall, clear of the units on the floor.
        cx, cy = (fl[0] + fl[2]) / 2, ROOM[1] + WALL_H / 2 + 2
        col = C_TEAL if f.won else C_HP_FOE
        tw = d.textlength(sc.result.upper(), font=F_BIG) + 40
        d.rounded_rectangle((cx - tw / 2, cy - 26, cx + tw / 2, cy + 26), 8, fill=C_PANEL,
                            outline=col, width=2)
        d.text((cx, cy), sc.result.upper(), font=F_BIG, fill=col, anchor="mm")
    return alive


def draw_frame(sc: Scene, tick: BattleTick | None, sub: str,
               show_result: bool = False) -> Image.Image:
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    draw_topbar(d, sc, sub)
    draw_minimap(d, sc)
    draw_squad_list(d, sc)
    fl = draw_room(d)
    alive = None
    if tick is not None:
        alive = draw_arena(d, sc, tick, fl, show_result)
    elif sc.action == "__end__":
        draw_idle_squad(d, sc, fl)
        draw_end(d, sc, fl)
    else:
        draw_idle_squad(d, sc, fl)
        draw_panel(d, sc, fl)
    draw_bottom(d, sc, tick, alive)
    return img


# ------------------------------------------------------------------ assembly

def frames(scenes: list[Scene], hold: int, fight_budget: int, sub: str):
    for sc in scenes:
        if sc.action == "__end__":
            for _ in range(hold * 6):
                yield draw_frame(sc, None, sub)
            continue
        for _ in range(hold):
            yield draw_frame(sc, None, sub)
        if sc.fight and sc.fight.ticks:
            ticks = sc.fight.ticks
            stride = max(1, len(ticks) // fight_budget)
            for t in ticks[::stride]:
                yield draw_frame(sc, t, sub)
            end = draw_frame(sc, ticks[-1], sub, show_result=True)
            for _ in range(14):
                yield end


def encode(imgs, out: str, fps: int) -> int:
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = 0
    for im in imgs:
        p.stdin.write(im.tobytes())
        n += 1
        if n % 60 == 0:
            print(f"  {n} frames", end="\r", flush=True)
    p.stdin.close()
    p.wait()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="runs/fog2m_noboss_s1.pt",
                    help="checkpoint, or 'heuristic'")
    ap.add_argument("--seed", type=int, default=30_185)
    ap.add_argument("--out", default="docs/run.mp4")
    ap.add_argument("--hold", type=int, default=9,
                    help="frames a decision is held for")
    ap.add_argument("--fight-frames", type=int, default=110,
                    help="frame budget per fight, subsampled to fit")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-level", type=int, default=None)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    tables = load_ruleset(strict=True)
    placement = POLICIES["frontline"]
    if args.agent == "heuristic":
        policy, label = policy_heuristic, "heuristic baseline"
    else:
        from tools.hierarchy_eval import make_ppo
        env = DespotRunEnv(tables=tables, fast_core=True)
        policy = make_ppo(args.agent, env)
        label = args.agent.replace("\\", "/").split("/")[-1]

    print(f"recording {label}, seed {args.seed} ...")
    scenes = record(tables, policy, args.seed, placement,
                    args.max_steps, args.stop_level)
    fights = sum(1 for s in scenes if s.fight)
    print(f"  {len(scenes) - 1} decisions, {fights} fights, "
          f"reached level {scenes[-1].hud['level']}")

    sub = f"PPO agent  ·  {label}  ·  seed {args.seed}"
    n = encode(frames(scenes, args.hold, args.fight_frames, sub),
               args.out, args.fps)
    print(f"  {n} frames -> {args.out} ({n / args.fps:.0f}s at {args.fps}fps)")


if __name__ == "__main__":
    main()
