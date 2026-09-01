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
from rl.env import MOVES, NON_MOVE_ACTIONS, DespotRunEnv
from rl.heuristic import heuristic as policy_heuristic
from rl.placement import POLICIES
from sim.battle import Battle
from sim.data import load_ruleset

W, H = 1280, 720
FPS = 30

BG = (16, 19, 25)
PANEL = (23, 27, 35)
LINE = (38, 44, 56)
TEXT = (200, 208, 221)
DIM = (107, 118, 134)
PLAYER = (77, 163, 255)
PLAYER_DK = (31, 77, 122)
ENEMY = (255, 107, 90)
ENEMY_DK = (122, 47, 38)
GOLD = (232, 184, 75)
FOOD = (107, 208, 138)
MUT = (169, 122, 222)

KIND_COLOR = {
    "start": (122, 132, 148),
    "fight": (150, 66, 60),
    "boss": (217, 74, 61),
    "item_shop": GOLD,
    "food_shop": FOOD,
    "mutation": MUT,
    "mutation_shop": MUT,
    "talent_shop": (87, 182, 196),
    "empty": (42, 48, 60),
}
KIND_GLYPH = {"start": "S", "fight": "", "boss": "B", "item_shop": "$",
              "food_shop": "F", "mutation": "*", "mutation_shop": "*",
              "talent_shop": "T", "empty": ""}


def font(size: int, bold: bool = False, mono: bool = False):
    names = (["consolab.ttf", "consola.ttf"] if mono else
             ["segoeuib.ttf", "seguisb.ttf", "arialbd.ttf"] if bold else
             ["segoeui.ttf", "arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{n}", size)
        except OSError:
            continue
    return ImageFont.load_default()


# ------------------------------------------------------------------ recording

@dataclass
class BattleTick:
    t: float
    agents: list          # (x, y, team, hp_frac, radius, melee, alive, letter)
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
    rooms: list           # (id, row, col, kind, cleared, current)
    edges: list           # ((row, col), (row, col))
    context: list         # pre-rendered lines for the decision card
    fight: Fight | None = None
    result: str = ""


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
            if ag.corpse and ag.hp <= 0 and not ag.dying:
                frac = 0.0
            else:
                frac = max(0.0, min(1.0, ag.hp / max(1.0, ag.spec.health)))
            ags.append((ag.x, ag.y, ag.team, frac, ag.spec.radius,
                        ag.spec.melee, ag.alive, (ag.spec.cls or "?")[:1]))
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
            label = "move " + MOVES[action_i][0]
        else:
            label = NON_MOVE_ACTIONS[action_i - env.n_moves].replace("_", " ")

        rooms = [(r.id, r.row, r.col, r.kind, r.cleared, r.id == st.room)
                 for r in st.rooms.rooms.values()]
        edges = []
        for rid, r in st.rooms.rooms.items():
            for other in st.rooms.neighbours(rid):
                o = st.rooms.rooms[other]
                if (r.row, r.col) < (o.row, o.col):
                    edges.append(((r.row, r.col), (o.row, o.col)))

        hud = {"level": st.level, "room": st.room, "kind": room.kind,
               "gold": st.gold, "food": st.food.amount,
               "moves": st.food.moves_left, "hunger": st.food.hunger_level,
               "squad": len(st.squad), "muts": len(st.mutations),
               "power": st.squad_power(), "step": n + 1,
               "squad_lines": squad_line(st)}
        ctx = context_lines(st, room, label)

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
                            edges=edges, context=ctx, fight=fight, result=note))

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
               squad_lines=squad_line(st))
    scenes.append(Scene(step=n, action="__end__", hud=hud,
                        rooms=scenes[-1].rooms, edges=scenes[-1].edges,
                        context=end))
    return scenes


# ------------------------------------------------------------------ drawing

F_H1 = font(24, bold=True)
F_H2 = font(17, bold=True)
F_B = font(15)
F_S = font(13)
F_M = font(14, mono=True)
F_MS = font(12, mono=True)
F_BIG = font(30, bold=True)

MAP_BOX = (20, 68, 430, 396)
HUD_BOX = (20, 408, 430, 700)
STAGE = (446, 68, 1260, 700)


def panel(d, box, title=None):
    d.rounded_rectangle(box, 8, fill=PANEL, outline=LINE)
    if title:
        d.text((box[0] + 14, box[1] + 10), title, font=F_H2, fill=DIM)


def draw_map(d, sc: Scene):
    panel(d, MAP_BOX, "LEVEL %d" % sc.hud["level"])
    x0, y0, x1, y1 = MAP_BOX
    x0, y0, x1, y1 = x0 + 18, y0 + 40, x1 - 18, y1 - 16
    rows = [r[1] for r in sc.rooms]
    cols = [r[2] for r in sc.rooms]
    if not rows:
        return
    span_r = max(rows) - min(rows) + 1
    span_c = max(cols) - min(cols) + 1
    cell = min((x1 - x0) / span_c, (y1 - y0) / span_r)
    size = cell * 0.62
    ox = x0 + ((x1 - x0) - cell * span_c) / 2
    oy = y0 + ((y1 - y0) - cell * span_r) / 2

    def centre(r, c):
        return (ox + (c - min(cols) + 0.5) * cell,
                oy + (r - min(rows) + 0.5) * cell)

    for a, b in sc.edges:
        d.line([centre(*a), centre(*b)], fill=LINE, width=3)
    for rid, r, c, kind, cleared, cur in sc.rooms:
        cx, cy = centre(r, c)
        col = KIND_COLOR.get(kind, KIND_COLOR["empty"])
        if cleared and not cur:
            col = tuple(int(v * 0.45) for v in col)
        box = (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
        d.rounded_rectangle(box, 4, fill=col,
                            outline=(255, 255, 255) if cur else None,
                            width=2 if cur else 0)
        g = KIND_GLYPH.get(kind, "")
        if g:
            d.text((cx, cy), g, font=F_S, fill=(20, 22, 28), anchor="mm")
        if cur:
            d.ellipse((cx - 4, cy - size / 2 - 12, cx + 4, cy - size / 2 - 4),
                      fill=(255, 255, 255))


def draw_hud(d, sc: Scene):
    panel(d, HUD_BOX, "SQUAD")
    x, y = HUD_BOX[0] + 16, HUD_BOX[1] + 38
    h = sc.hud
    stats = [("gold", f"{h['gold']:.0f}", GOLD),
             ("food", f"{h['food']:.0f}", FOOD),
             ("moves left", f"{h['moves']}", TEXT),
             ("squad", f"{h['squad']}", PLAYER),
             ("power", f"{h['power']:.0f}", TEXT),
             ("mutations", f"{h['muts']}", MUT)]
    for i, (k, v, col) in enumerate(stats):
        cx = x + (i % 2) * 200
        cy = y + (i // 2) * 46
        d.text((cx, cy), k.upper(), font=F_S, fill=DIM)
        d.text((cx, cy + 16), v, font=F_H2, fill=col)
    y += 3 * 46 + 6
    d.line([(x, y), (HUD_BOX[2] - 16, y)], fill=LINE, width=1)
    y += 10
    for line in h["squad_lines"]:
        d.text((x, y), line, font=F_MS, fill=DIM)
        y += 16
        if y > HUD_BOX[3] - 18:
            break


def draw_card(d, sc: Scene):
    panel(d, STAGE)
    x0, y0, x1, y1 = STAGE
    end = sc.action == "__end__"
    d.text((x0 + 30, y0 + 34), "RUN OVER" if end else sc.action.upper(),
           font=F_BIG, fill=TEXT)
    # A fight's outcome belongs after the fight, not on the card that precedes
    # it, so it is drawn over the arena instead.
    if sc.result and not sc.fight:
        d.text((x0 + 30, y0 + 76), sc.result, font=F_B, fill=DIM)
    y = y0 + 130
    for i, line in enumerate(sc.context):
        d.text((x0 + 30, y), line, font=F_M if i else F_H2,
               fill=TEXT if i == 0 else DIM)
        y += 26


def draw_arena(d, sc: Scene, tick: BattleTick, show_result: bool = False):
    panel(d, STAGE)
    x0, y0, x1, y1 = STAGE
    f = sc.fight
    ww, wh = f.cols * f.tile, f.rows * f.tile
    pad = 46
    scale = min((x1 - x0 - 2 * pad) / ww, (y1 - y0 - 2 * pad - 40) / wh)
    ox = x0 + ((x1 - x0) - ww * scale) / 2
    oy = y0 + 40 + ((y1 - y0 - 40) - wh * scale) / 2

    d.rectangle((ox, oy, ox + ww * scale, oy + wh * scale),
                fill=(20, 24, 31), outline=LINE)
    for c in range(f.cols + 1):
        gx = ox + c * f.tile * scale
        d.line([(gx, oy), (gx, oy + wh * scale)], fill=(28, 33, 42))
    for r in range(f.rows + 1):
        gy = oy + r * f.tile * scale
        d.line([(ox, gy), (ox + ww * scale, gy)], fill=(28, 33, 42))

    alive = [0, 0]
    for (wx, wy, team, frac, radius, melee, is_alive, letter) in tick.agents:
        px, py = ox + wx * scale, oy + wy * scale
        rad = max(5.0, radius * scale)
        base, dark = (PLAYER, PLAYER_DK) if team == 0 else (ENEMY, ENEMY_DK)
        if not is_alive:
            d.ellipse((px - rad, py - rad, px + rad, py + rad),
                      fill=(38, 42, 50))
            continue
        alive[team] += 1
        d.ellipse((px - rad, py - rad, px + rad, py + rad), fill=dark,
                  outline=base, width=2 if melee else 1)
        d.text((px, py), letter, font=F_S, fill=base, anchor="mm")
        bw = rad * 2
        d.rectangle((px - rad, py - rad - 7, px - rad + bw, py - rad - 4),
                    fill=(45, 50, 60))
        d.rectangle((px - rad, py - rad - 7, px - rad + bw * frac,
                     py - rad - 4), fill=base)
    for (wx, wy, team) in tick.shots:
        px, py = ox + wx * scale, oy + wy * scale
        col = PLAYER if team == 0 else ENEMY
        d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=col)

    d.text((x0 + 30, y0 + 22), "FIGHT", font=F_H2, fill=DIM)
    d.text((x0 + 110, y0 + 22), f"{tick.t:4.1f}s", font=F_M, fill=TEXT)
    d.text((x1 - 30, y0 + 22), f"{alive[0]} v {alive[1]}", font=F_H2,
           fill=TEXT, anchor="ra")
    if show_result and sc.result:
        won = sc.fight is not None and sc.fight.won
        col = PLAYER if won else ENEMY
        cx, cy = (x0 + x1) / 2, y0 + 24 + (y1 - y0) * 0.12
        d.text((cx, cy), sc.result.upper(), font=F_BIG, fill=col, anchor="mm")


def draw_frame(sc: Scene, tick: BattleTick | None, sub: str,
               show_result: bool = False) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((22, 18), "despot-rl", font=F_H1, fill=TEXT)
    d.text((150, 26), sub, font=F_B, fill=DIM)
    d.text((W - 22, 26), f"step {sc.hud['step']}", font=F_B, fill=DIM,
           anchor="ra")
    d.line([(20, 58), (W - 20, 58)], fill=LINE)
    draw_map(d, sc)
    draw_hud(d, sc)
    if tick is not None:
        draw_arena(d, sc, tick, show_result)
    else:
        draw_card(d, sc)
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
    ap.add_argument("--agent", default="runs/shelf2m_described_s1.pt",
                    help="checkpoint, or 'heuristic'")
    ap.add_argument("--seed", type=int, default=30_001)
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
