"""The placement network, and the wrapper that lets a run use it.

The network scores *cells*, not action indices. Placement is a spatial problem
and the layouts are not all the same shape, so a fully connected head over a
fixed action list would have to relearn what "one cell up" means for every
layout. A small convolutional stack over the room canvas ends in a 1x1
convolution, giving one logit per cell; the environment's `cell_index` map then
picks out the cells the current layout actually offers.

Unit features (is it melee, how far does it reach, how much HP) are broadcast
as extra input channels, so the same board can be scored differently depending
on which unit is being placed.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.place_env import N_PLANES, UNIT_FEATURES, encode_state

NEG_INF = -1e9


class PlacementNet(nn.Module):
    def __init__(self, rows: int, cols: int, hidden: int = 48):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.feat = nn.Linear(UNIT_FEATURES, 8)
        self.body = nn.Sequential(
            nn.Conv2d(N_PLANES + 8, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
        )
        self.pi = nn.Conv2d(hidden, 1, 1)
        self.v = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def _split(self, obs: torch.Tensor):
        n = obs.shape[0]
        board = obs[:, :N_PLANES * self.rows * self.cols]
        feats = obs[:, N_PLANES * self.rows * self.cols:]
        return board.view(n, N_PLANES, self.rows, self.cols), feats

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, cell_index: torch.Tensor):
        board, feats = self._split(obs)
        n = board.shape[0]
        f = torch.tanh(self.feat(feats))
        f = f[:, :, None, None].expand(n, f.shape[1], self.rows, self.cols)
        h = self.body(torch.cat([board, f], dim=1))

        cellmap = self.pi(h).view(n, -1)                       # one logit per cell
        idx = cell_index.clamp(min=0)
        logits = cellmap.gather(1, idx)
        logits = torch.where(cell_index >= 0, logits, torch.full_like(logits, NEG_INF))
        logits = torch.where(mask, logits, torch.full_like(logits, NEG_INF))

        value = self.v(h.mean(dim=(2, 3))).squeeze(-1)
        return Categorical(logits=logits), value


class LearnedPlacement:
    """A trained `PlacementNet` in the shape a `RunState` placement policy takes.

    Sampling rather than argmax is the default: the run env resolves one battle
    per fight, and a deterministic placement makes a whole run turn on a single
    chaotic outcome. `greedy=True` is for measurement, where seeds are fixed.
    """

    def __init__(self, net: PlacementNet, greedy: bool = False, device="cpu"):
        self.net = net.eval()
        self.greedy = greedy
        self.device = torch.device(device)

    def __call__(self, grid, layout, specs, rng: random.Random, enemies=None):
        enemies = enemies or []
        zone = sorted(layout.zone("p"))
        if not zone:
            return [(0, 0)] * len(specs)
        rows, cols = self.net.rows, self.net.cols
        enemy_specs = [a.spec for a in enemies]
        enemy_xy = [(a.x, a.y) for a in enemies]

        n_actions = len(zone)
        cell_index = np.asarray([r * cols + c for r, c in zone], dtype=np.int64)
        # A layout taller than the canvas cannot happen with the shipped set,
        # but clamp rather than index out of range if one ever appears.
        cell_index = np.clip(cell_index, 0, rows * cols - 1)

        taken: set[tuple[int, int]] = set()
        cells: list[tuple[int, int]] = []
        for i in range(len(specs)):
            obs = encode_state(rows, cols, zone, specs, cells, i,
                               enemy_specs, enemy_xy, grid.tile)
            mask = np.asarray([rc not in taken for rc in zone], dtype=bool)
            if not mask.any():
                mask[:] = True
            with torch.no_grad():
                dist, _ = self.net(
                    torch.as_tensor(obs[None], device=self.device),
                    torch.as_tensor(mask[None], device=self.device),
                    torch.as_tensor(cell_index[None], device=self.device))
                if self.greedy:
                    a = int(dist.probs.argmax())
                else:
                    a = int(torch.multinomial(dist.probs[0], 1))
            rc = zone[a % n_actions]
            cells.append(rc)
            taken.add(rc)
        return cells


def load_placement(path: str, greedy: bool = False, device="cpu") -> LearnedPlacement:
    ck = torch.load(path, map_location=device, weights_only=True)
    net = PlacementNet(ck["rows"], ck["cols"], hidden=ck.get("hidden", 48))
    net.load_state_dict(ck["model"])
    return LearnedPlacement(net, greedy=greedy, device=device)
