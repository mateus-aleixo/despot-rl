"""Masked PPO over the run environment.

Small on purpose: the point of this first agent is to show the environment is
learnable and the reward is shaped sensibly, not to squeeze the task. It runs on
CPU comfortably; the GPU only helps once the Rust core makes rollouts cheap
enough that the network is the bottleneck.

Action masking is applied to the logits, so illegal actions cannot be sampled
and the entropy term is computed over the legal set only.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, ".")
from rl.env import DespotRunEnv
from rl.placement import POLICIES
from sim.data import load_ruleset
from sim.mutations import disable_agent_passives

NEG_INF = -1e9


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.body(obs)
        logits = self.pi(h)
        logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
        return Categorical(logits=logits), self.v(h).squeeze(-1)


def rollout(envs, states, masks, model, horizon, device):
    """Collect `horizon` steps from each env. Returns tensors plus episode stats."""
    n = len(envs)
    obs_buf = np.zeros((horizon, n, envs[0].obs_dim), dtype=np.float32)
    mask_buf = np.zeros((horizon, n, envs[0].n_actions), dtype=bool)
    act_buf = np.zeros((horizon, n), dtype=np.int64)
    logp_buf = np.zeros((horizon, n), dtype=np.float32)
    rew_buf = np.zeros((horizon, n), dtype=np.float32)
    done_buf = np.zeros((horizon, n), dtype=np.float32)
    val_buf = np.zeros((horizon, n), dtype=np.float32)
    episodes: list[tuple[float, int]] = []

    for t in range(horizon):
        obs_t = torch.as_tensor(states, device=device)
        mask_t = torch.as_tensor(masks, device=device)
        with torch.no_grad():
            dist, value = model(obs_t, mask_t)
            action = dist.sample()
            logp = dist.log_prob(action)

        obs_buf[t], mask_buf[t] = states, masks
        act_buf[t] = action.cpu().numpy()
        logp_buf[t] = logp.cpu().numpy()
        val_buf[t] = value.cpu().numpy()

        for i, env in enumerate(envs):
            o, r, term, trunc, info = env.step(int(act_buf[t, i]))
            rew_buf[t, i] = r
            done = term or trunc
            done_buf[t, i] = float(done)
            if done:
                episodes.append((env._ep_return + r, info.get("level", env.state.level)))
                env._ep_return = 0.0
                o, info = env.reset()
            else:
                env._ep_return += r
            states[i] = o
            masks[i] = info["action_mask"]
            if not masks[i].any():          # nothing legal: end the episode
                o, info = env.reset()
                states[i], masks[i] = o, info["action_mask"]

    return (obs_buf, mask_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf), episodes


def gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
    horizon, n = rew.shape
    adv = np.zeros_like(rew)
    nextgae = np.zeros(n, dtype=np.float32)
    nextval = last_val
    for t in reversed(range(horizon)):
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * nextval * nonterminal - val[t]
        nextgae = delta + gamma * lam * nonterminal * nextgae
        adv[t] = nextgae
        nextval = val[t]
    return adv, adv + val


def evaluate(model, tables, device, episodes=12, placement=None, greedy=True):
    # Deliberately unshaped: a shaped agent has to be graded on the real
    # objective, not on the bonus it was trained with.
    env = DespotRunEnv(tables=tables, seed=10_000, placement_policy=placement)
    levels, returns = [], []
    for _ in range(episodes):
        obs, info = env.reset()
        mask = info["action_mask"]
        total = 0.0
        while True:
            if not mask.any():
                break
            with torch.no_grad():
                dist, _ = model(torch.as_tensor(obs[None], device=device),
                                torch.as_tensor(mask[None], device=device))
                a = int(dist.probs.argmax()) if greedy else int(dist.sample())
            obs, r, term, trunc, info = env.step(a)
            total += r
            mask = info["action_mask"]
            if term or trunc:
                break
        levels.append(env.state.level)
        returns.append(total)
    return statistics.mean(levels), statistics.mean(returns)


def random_baseline(tables, episodes=12, placement=None):
    env = DespotRunEnv(tables=tables, seed=20_000, placement_policy=placement)
    levels = []
    for _ in range(episodes):
        obs, info = env.reset()
        mask = info["action_mask"]
        while mask.any():
            a = int(np.random.choice(np.flatnonzero(mask)))
            obs, r, term, trunc, info = env.step(a)
            mask = info["action_mask"]
            if term or trunc:
                break
        levels.append(env.state.level)
    return statistics.mean(levels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--placement", default="frontline", choices=sorted(POLICIES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fast-core", action="store_true",
                    help="resolve fights in the Rust core where possible")
    ap.add_argument("--shaping", default="none", choices=("none", "power"),
                    help="potential-based shaping on squad power and food")
    ap.add_argument("--step-cost", type=float, default=0.02)
    ap.add_argument("--level-reward", default="flat", choices=("flat", "rising"))
    ap.add_argument("--out", default="runs/ppo.pt")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="also save at each multiple of this many steps, as "
                         "<out>.<steps>.pt -- a budget comparison off one run "
                         "has no init or sampling difference to confound it")
    ap.add_argument("--seed", type=int, default=0,
                    help="network init and action sampling; the environment "
                         "seeds are fixed either way")
    ap.add_argument("--blind-shrine", action="store_true",
                    help="hold the shrine bit at zero while leaving it in the "
                         "observation, so an arm trained without the information "
                         "has the same obs_dim and first layer as one with it")
    ap.add_argument("--blind-squad", action="store_true",
                    help="hold the squad one-hot at zero while leaving it in "
                         "the vector, so an arm that is not told which squad it "
                         "is playing has the same obs_dim and first layer")
    ap.add_argument("--squad", default=None,
                    help="pin the starting squad; the default cycles all eight "
                         "by episode seed")
    ap.add_argument("--lights-on", action="store_true",
                    help="reveal the whole level instead of playing under the "
                         "fog, at the same obs_dim, which is the pre-fog "
                         "observation and the control arm for what the fog costs")
    ap.add_argument("--blind-shelf", action="store_true",
                    help="hold the five description floats and the passive bit "
                         "at zero for every mutation slot, leaving `present`, "
                         "the takes left and the shrine bit -- the information "
                         "the agents had before the shelf was described, at an "
                         "unchanged obs_dim")
    ap.add_argument("--free-mutation-steps", action="store_true",
                    help="exempt buy_mutation and take_mutation from the step "
                         "cost, leaving every other action taxed -- the clean "
                         "form of \"a mutation costs a step now and pays later\"")
    ap.add_argument("--without-passives", action="store_true",
                    help="train against the shelf with every mutation hook put "
                         "back to unimplemented -- the control arm of the "
                         "twelve-seed sweep, held for the whole run")
    args = ap.parse_args()

    # Before the ruleset loads, so the shelf the agent sees described in its
    # observation is inert too and not just the fights it plays.
    if args.without_passives:
        disable_agent_passives()

    # Without this, two runs differ by network init and sampling noise as well
    # as by whatever is being varied, so a budget comparison cannot attribute
    # its difference to the budget. The env seeds were already fixed; this is
    # the half that was not.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    tables = load_ruleset(strict=True)
    placement = POLICIES[args.placement]
    envs = [DespotRunEnv(tables=tables, seed=1000 * i, placement_policy=placement,
                         fast_core=args.fast_core, shaping=args.shaping,
                         step_cost=args.step_cost,
                         level_reward=args.level_reward,
                         free_mutation_steps=args.free_mutation_steps,
                         blind_shrine=args.blind_shrine,
                         blind_shelf=args.blind_shelf,
                         squad=args.squad,
                         blind_squad=args.blind_squad,
                         lights_on=args.lights_on)
            for i in range(args.envs)]

    states = np.zeros((args.envs, envs[0].obs_dim), dtype=np.float32)
    masks = np.zeros((args.envs, envs[0].n_actions), dtype=bool)
    for i, e in enumerate(envs):
        o, info = e.reset()
        states[i], masks[i] = o, info["action_mask"]
        e._ep_return = 0.0

    model = ActorCritic(envs[0].obs_dim, envs[0].n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    base = random_baseline(tables, episodes=10, placement=placement)
    print(f"random baseline: mean level {base:.2f}")
    print(f"training on {device}, placement={args.placement}, "
          f"{args.envs} envs x {args.horizon} steps per update")

    def save(path: str) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "obs_dim": envs[0].obs_dim,
                    "n_actions": envs[0].n_actions}, path)

    collected, update, t0 = 0, 0, time.perf_counter()
    saved = 0
    recent: list[tuple[float, int]] = []
    while collected < args.steps:
        buffers, eps = rollout(envs, states, masks, model, args.horizon, device)
        obs_b, mask_b, act_b, logp_b, rew_b, done_b, val_b = buffers
        recent += eps
        collected += args.horizon * args.envs
        update += 1

        with torch.no_grad():
            _, last_val = model(torch.as_tensor(states, device=device),
                                torch.as_tensor(masks, device=device))
        adv, ret = gae(rew_b, val_b, done_b, last_val.cpu().numpy())

        flat = lambda x, shape=None: torch.as_tensor(
            x.reshape((-1,) + x.shape[2:]), device=device)
        obs_t, mask_t = flat(obs_b), flat(mask_b)
        act_t, oldlogp_t = flat(act_b), flat(logp_b)
        adv_t, ret_t = flat(adv), flat(ret)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(args.epochs):
            dist, value = model(obs_t, mask_t)
            logp = dist.log_prob(act_t)
            ratio = (logp - oldlogp_t).exp()
            pg = -torch.min(ratio * adv_t,
                            ratio.clamp(1 - args.clip, 1 + args.clip) * adv_t).mean()
            vloss = ((value - ret_t) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + 0.5 * vloss - args.entropy * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

        if args.checkpoint_every and collected // args.checkpoint_every > saved:
            saved = collected // args.checkpoint_every
            save(f"{args.out.removesuffix('.pt')}.{saved * args.checkpoint_every}.pt")

        if update % 5 == 0 or collected >= args.steps:
            last = recent[-30:]
            lv = statistics.mean(l for _, l in last) if last else float("nan")
            rt = statistics.mean(r for r, _ in last) if last else float("nan")
            el = time.perf_counter() - t0
            print(f"  update {update:3d}  steps {collected:6d}  "
                  f"episodes {len(recent):4d}  mean level {lv:5.2f}  "
                  f"mean return {rt:7.2f}  {collected/el:5.1f} steps/s")

    save(args.out)

    lvl, ret = evaluate(model, tables, device, episodes=12, placement=placement)
    print(f"\ntrained (greedy): mean level {lvl:.2f}, mean return {ret:.2f}")
    print(f"random baseline : mean level {base:.2f}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
