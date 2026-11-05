"""SFN-World v2: eight latent groups of two dims each, mixed into R^64 by a frozen random MLP."""
import math
from typing import NamedTuple

import torch
from torch import nn

N_ACT = 4
OMEGA = (0.6, 1.4, 2.2, 3.0)
SIGMA6 = math.sqrt(math.log(2))
RHO = (0.98, 0.7)
JUMP_P = 0.05
N_GROUPS = 8
GROUPS = [slice(2 * k, 2 * k + 2) for k in range(N_GROUPS)]  # columns of `gt` per group
FAST = slice(6, 14)  # groups 4-7, scaled by w_fast


class Batch(NamedTuple):
    x: torch.Tensor  # (n, 64) observation at t
    a: torch.Tensor  # (n,) action
    xn: torch.Tensor  # (n, 64) observation at t+1
    gt: torch.Tensor  # (n, 16) true group values at t (probes only)
    gtn: torch.Tensor  # (n, 16) at t+1
    jump: torch.Tensor  # (n,) True if group 3 resampled in this transition


def _emb(theta: torch.Tensor) -> torch.Tensor:
    """(n, m) angles -> (n, 2m) interleaved (cos, sin) per angle."""
    return torch.stack([theta.cos(), theta.sin()], -1).flatten(1)


class Env:
    """Samples i.i.d. transitions from the stationary distribution (theta uniform)."""

    def __init__(self, seed: int, difficulty: int = 0, w_fast: float = 1.0, device="cpu"):
        self.diff, self.dev = difficulty, device
        torch.manual_seed(1000 + seed)  # the mixing MLP is fixed per seed index
        layers = [nn.Linear(16, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(),
                  nn.Linear(64, 64)]
        for m in layers[::2]:
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)
        self.g = nn.Sequential(*layers).to(device).requires_grad_(False)
        self.omega = torch.tensor(OMEGA, device=device)
        self.w = torch.ones(16, device=device)
        self.w[FAST] = w_fast
        self.mean, self.std = 0.0, 1.0
        x = self.sample(16384).x
        self.mean, self.std = x.mean(0), x.std(0)

    def obs(self, gt: torch.Tensor) -> torch.Tensor:
        return (self.g(gt * self.w) - self.mean) / self.std

    @torch.no_grad()
    def sample(self, n: int) -> Batch:
        rn = lambda *s: torch.randn(*s, device=self.dev)  # noqa: E731
        a = torch.randint(0, N_ACT, (n,), device=self.dev)
        om = self.omega[a, None]
        y, yn = [], []
        for rho in RHO:  # groups 1-2: OU
            s = rn(n, 2)
            y.append(s)
            yn.append(rho * s + math.sqrt(1 - rho**2) * rn(n, 2))
        s, jump = rn(n, 2), torch.rand(n, 1, device=self.dev) < JUMP_P  # group 3
        y.append(s)
        yn.append(torch.where(jump, rn(n, 2), s))
        phi = 2 * math.pi * torch.rand(n, 1, device=self.dev)  # group 4
        y.append(_emb(phi))
        yn.append(_emb(phi + math.pi / 2))
        th = 2 * math.pi * torch.rand(n, 3, device=self.dev)  # groups 5-7
        inc = om * torch.tensor([1.0, 1.0, 0.3], device=self.dev)
        if self.diff >= 1:
            inc = inc + 0.8 * torch.sin(2 * th)
        nxt = th + inc
        if self.diff == 2:
            nxt[:, :2] = 2 * th[:, :2] + om
        nxt[:, 1] += SIGMA6 * rn(n)
        y.append(_emb(th))
        yn.append(_emb(nxt))
        y.append(rn(n, 2))  # group 8: fresh noise every step
        yn.append(rn(n, 2))
        gt, gtn = torch.cat(y, 1), torch.cat(yn, 1)
        return Batch(self.obs(gt), a, self.obs(gtn), gt, gtn, jump[:, 0])
