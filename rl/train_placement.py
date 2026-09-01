"""Masked PPO over the placement environment.

Episodes are short (one action per unit) and the reward only arrives at the end,
so there is no bootstrapping: the return of every step in an episode is the
terminal reward, and the advantage is that minus the critic's estimate.

The reward is already a paired difference against `frontline_placement` on the
same battle seed, so zero means "no better than the heuristic" and the sign of
the mean return is the whole result.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from rl.place_env import (PlacementEnv, cached_scenarios,
                          sample_close_scenarios, sample_scenarios)
from rl.place_policy import PlacementNet
from sim.data import load_ruleset


def run_episode(env, net, device, scenario=None, greedy=False):
    """One fight, one env. Used for evaluation; training rolls many at once."""
    obs, info = env.reset(scenario)
    traj = _Traj()
    while True:
        action = _act(net, device, obs[None], info["action_mask"][None],
                      info["cell_index"][None], greedy, traj)[0]
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            return traj.finish(r, info.get("won"), info.get("baseline_won"),
                               info.get("raw", r))


class _Traj:
    """The per-step tensors of one episode, filled in as it plays."""

    def __init__(self):
        self.obs, self.mask, self.cell_index = [], [], []
        self.act, self.logp, self.val = [], [], []

    def record(self, obs, mask, cidx, act, logp, val):
        self.obs.append(obs); self.mask.append(mask); self.cell_index.append(cidx)
        self.act.append(act); self.logp.append(logp); self.val.append(val)

    def finish(self, reward, won, baseline_won, raw=None):
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "mask": np.asarray(self.mask, dtype=bool),
            "cell_index": np.asarray(self.cell_index, dtype=np.int64),
            "act": np.asarray(self.act, dtype=np.int64),
            "logp": np.asarray(self.logp, dtype=np.float32),
            "val": np.asarray(self.val, dtype=np.float32),
            "reward": reward, "won": won, "baseline_won": baseline_won,
            "raw": reward if raw is None else raw,
        }


def _act(net, device, obs, mask, cidx, greedy, *trajs):
    """One batched forward across envs. Records into each env's trajectory."""
    obs_t = torch.as_tensor(obs, device=device)
    mask_t = torch.as_tensor(mask, device=device)
    idx_t = torch.as_tensor(cidx, device=device)
    with torch.no_grad():
        dist, value = net(obs_t, mask_t, idx_t)
        action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
    acts = action.cpu().numpy()
    logps = logp.cpu().numpy()
    vals = value.cpu().numpy()
    for i, traj in enumerate(trajs):
        traj.record(obs[i], mask[i], cidx[i], int(acts[i]), float(logps[i]),
                    float(vals[i]))
    return [int(a) for a in acts]


def collect(envs, net, device, scenarios, group: int, rng):
    """One batch of fights, each rolled `group` times.

    Every env in a group gets the same scenario *and* the same battle seed, so
    the only thing that differs between its episodes is where the policy chose
    to stand. That is what makes a leave-one-out baseline work: the difficulty
    of the fight, the layout and the RNG stream all cancel, and what is left is
    the placement.

    A batch of independent fights with a learned critic was the first design,
    and it did not move the policy at all: the critic can predict roughly how
    hard a fight is, but not how good one placement was inside it, so the
    advantage was almost all noise.
    """
    n_groups = max(1, len(envs) // group)
    picks = [rng.choice(scenarios) for _ in range(n_groups)]
    seeds = [rng.choice(s.seeds) if s.seeds else rng.randrange(1 << 30)
             for s in picks]

    trajs = [_Traj() for _ in envs]
    states, active = [], []
    for i, env in enumerate(envs):
        g = min(i // group, n_groups - 1)
        states.append(env.reset(picks[g], battle_seed=seeds[g]))
        active.append(True)

    done: list[dict | None] = [None] * len(envs)
    while any(active):
        idx = [i for i, a in enumerate(active) if a]
        obs = np.stack([states[i][0] for i in idx])
        mask = np.stack([states[i][1]["action_mask"] for i in idx])
        cidx = np.stack([states[i][1]["cell_index"] for i in idx])
        acts = _act(net, device, obs, mask, cidx, False, *[trajs[i] for i in idx])

        for k, i in enumerate(idx):
            o, r, term, trunc, info = envs[i].step(acts[k])
            if term or trunc:
                done[i] = trajs[i].finish(r, info.get("won"),
                                          info.get("baseline_won"),
                                          info.get("raw", r))
                active[i] = False
            else:
                states[i] = (o, info)

    # leave-one-out baseline inside each group
    out = []
    for g in range(n_groups):
        members = [done[i] for i in range(g * group, min((g + 1) * group, len(envs)))
                   if done[i] is not None]
        raws = [e["raw"] for e in members]
        total = sum(raws)
        for e, raw in zip(members, raws):
            e["adv"] = raw - (total - raw) / max(1, len(members) - 1)
        out += members
    return out


def clone_dataset(env, scenarios, tables):
    """Replay `frontline_placement` through the env encoding, step by step.

    PPO from a uniform policy barely moves here: with 49 to 84 cells and a
    reward that is a difference of two noisy battles, the gradient is mostly
    noise and the entropy term holds the policy near random. Cloning the
    heuristic first puts the network somewhere competent, which also proves the
    encoding and the head can represent a sensible placement at all.
    """
    from rl.placement import frontline_placement
    from sim.battle import Agent
    from sim.nav import Grid

    obs_l, mask_l, idx_l, act_l = [], [], [], []
    for scn in scenarios:
        layout = env.layouts[scn.layout_index]
        grid = Grid.from_layout(layout)
        enemies = [Agent(spec=sp, team=1, x=x, y=y, hp=sp.health, mana=0.0)
                   for sp, (x, y) in zip(scn.enemy_specs, scn.enemy_xy)]
        target = frontline_placement(grid, layout, scn.specs,
                                     __import__("random").Random(scn.uid), enemies)

        obs, info = env.reset(scn)
        for want in target:
            mask, cidx = info["action_mask"], info["cell_index"]
            zone = env.zone
            if want in zone and mask[zone.index(want)]:
                a = zone.index(want)
            else:
                # the heuristic reuses cells when a column runs out; take the
                # nearest free one instead of teaching an illegal action
                free = [i for i in range(len(zone)) if mask[i]]
                a = min(free, key=lambda i: abs(zone[i][0] - want[0])
                        + abs(zone[i][1] - want[1]))
            obs_l.append(obs); mask_l.append(mask); idx_l.append(cidx); act_l.append(a)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
    return (np.asarray(obs_l, dtype=np.float32), np.asarray(mask_l, dtype=bool),
            np.asarray(idx_l, dtype=np.int64), np.asarray(act_l, dtype=np.int64))


def clone(net, opt, data, device, epochs: int, batch: int = 256):
    """Cross-entropy on the heuristic's choices."""
    obs, mask, cidx, act = (torch.as_tensor(x, device=device) for x in data)
    n = obs.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total, hits = 0.0, 0
        for i in range(0, n, batch):
            sel = perm[i:i + batch]
            dist, _ = net(obs[sel], mask[sel], cidx[sel])
            loss = -dist.log_prob(act[sel]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            total += float(loss) * len(sel)
            hits += int((dist.probs.argmax(dim=-1) == act[sel]).sum())
        yield ep, total / n, hits / n


def evaluate(env, net, device, scenarios, greedy=True):
    """Held-out scenarios, fixed battle seeds: policy against the heuristic."""
    rewards, wins, base_wins = [], 0, 0
    for i, scn in enumerate(scenarios):
        env.rng.seed(90_000 + i)            # same battle seed for both sides
        ep = run_episode(env, net, device, scenario=scn, greedy=greedy)
        rewards.append(ep["reward"])
        wins += 1 if ep["won"] else 0
        base_wins += 1 if ep["baseline_won"] else 0
    return statistics.mean(rewards), wins / len(scenarios), base_wins / len(scenarios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--envs", type=int, default=64, help="episodes rolled in lockstep")
    ap.add_argument("--group", type=int, default=8,
                    help="placements sampled per fight, for the leave-one-out baseline")
    ap.add_argument("--scenarios", type=int, default=400)
    ap.add_argument("--eval-scenarios", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--ppo-lr", type=float, default=0.0,
                    help="learning rate for PPO after cloning (default: --lr)")
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="stop a batch's epochs once the policy has moved this far")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clone-epochs", type=int, default=0,
                    help="pretrain by cloning the heuristic before PPO")
    ap.add_argument("--close", action="store_true",
                    help="train on fights rescaled until placement decides them")
    ap.add_argument("--cache", default="",
                    help="pickle the sampled fights here and reuse them")
    ap.add_argument("--out", default="runs/placement.pt")
    args = ap.parse_args()

    device = torch.device(args.device)
    tables = load_ruleset(strict=True)

    t0 = time.perf_counter()
    sampler = sample_close_scenarios if args.close else sample_scenarios
    cache = args.cache or (f"runs/scenarios_{'close' if args.close else 'run'}"
                           f"_{args.scenarios}_{args.eval_scenarios}_{args.seed}.pkl")
    train_scn, eval_scn = cached_scenarios(cache, lambda: (
        sampler(tables, n=args.scenarios, seed=args.seed),
        sampler(tables, n=args.eval_scenarios, seed=500_000)))
    print(f"sampled {len(train_scn)} training and {len(eval_scn)} held-out fights "
          f"in {time.perf_counter() - t0:.1f}s "
          f"(mean squad {statistics.mean(s.n_units for s in train_scn):.1f}, "
          f"mean enemies {statistics.mean(len(s.enemy_specs) for s in train_scn):.1f})")

    cache: dict = {}
    envs = [PlacementEnv(tables, train_scn, seed=args.seed + 7919 * i,
                         baseline_cache=cache)
            for i in range(args.envs)]
    env = envs[0]
    eval_env = PlacementEnv(tables, eval_scn, seed=1234)
    net = PlacementNet(env.rows, env.cols, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    r0, w0, b0 = evaluate(eval_env, net, device, eval_scn)
    print(f"untrained: mean paired reward {r0:+.3f}  "
          f"win rate {w0:.2f} vs heuristic {b0:.2f}")

    if args.clone_epochs:
        t1 = time.perf_counter()
        data = clone_dataset(env, train_scn, tables)
        for ep, loss, acc in clone(net, opt, data, device, args.clone_epochs):
            if ep % 5 == 0 or ep == args.clone_epochs - 1:
                print(f"  clone epoch {ep:3d}  loss {loss:6.3f}  "
                      f"matches heuristic {acc:.2f}")
        rc, wc, bc = evaluate(eval_env, net, device, eval_scn)
        print(f"cloned   : mean paired reward {rc:+.3f}  "
              f"win rate {wc:.2f} vs heuristic {bc:.2f}  "
              f"({time.perf_counter() - t1:.0f}s)")
        # Adam's moments from the cloning phase are about a different objective,
        # and carrying them into PPO throws the first few updates a long way.
        opt = torch.optim.Adam(net.parameters(), lr=args.ppo_lr or args.lr)

    rng = random.Random(args.seed)
    done, update, t0 = 0, 0, time.perf_counter()
    recent: list[float] = []
    while done < args.episodes:
        eps = collect(envs, net, device, train_scn, args.group, rng)
        done += len(eps)
        update += 1
        recent += [e["reward"] for e in eps]

        obs = torch.as_tensor(np.concatenate([e["obs"] for e in eps]), device=device)
        mask = torch.as_tensor(np.concatenate([e["mask"] for e in eps]), device=device)
        cidx = torch.as_tensor(np.concatenate([e["cell_index"] for e in eps]), device=device)
        act = torch.as_tensor(np.concatenate([e["act"] for e in eps]), device=device)
        oldlogp = torch.as_tensor(np.concatenate([e["logp"] for e in eps]), device=device)
        # terminal reward only: every step in an episode carries the same return
        ret = torch.as_tensor(np.concatenate(
            [np.full(len(e["act"]), e["raw"], dtype=np.float32) for e in eps]),
            device=device)
        adv = torch.as_tensor(np.concatenate(
            [np.full(len(e["act"]), e["adv"], dtype=np.float32) for e in eps]),
            device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        kl = 0.0
        for _ in range(args.epochs):
            dist, value = net(obs, mask, cidx)
            logp = dist.log_prob(act)
            ratio = (logp - oldlogp).exp()
            pg = -torch.min(ratio * adv,
                            ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            vloss = ((value - ret) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + 0.5 * vloss - args.entropy * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                kl = float((oldlogp - logp).mean())
            if args.target_kl and kl > args.target_kl:
                break            # the batch has been squeezed enough

        if update % 5 == 0 or done >= args.episodes:
            last = recent[-200:]
            el = time.perf_counter() - t0
            print(f"  update {update:3d}  episodes {done:5d}  "
                  f"mean paired reward {statistics.mean(last):+7.3f}  "
                  f"kl {kl:6.4f}  {done / el:5.1f} eps/s")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "rows": env.rows, "cols": env.cols,
                "hidden": args.hidden}, args.out)

    r1, w1, b1 = evaluate(eval_env, net, device, eval_scn)
    print(f"\ntrained  : mean paired reward {r1:+.3f}  "
          f"win rate {w1:.2f} vs heuristic {b1:.2f}")
    print(f"untrained: mean paired reward {r0:+.3f}  "
          f"win rate {w0:.2f} vs heuristic {b0:.2f}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
