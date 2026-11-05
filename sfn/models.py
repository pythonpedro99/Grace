"""Encoder, predictors, M2 ensemble and the SIGReg loss (Epps-Pulley, as in the LeWM code)."""
import copy
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from .env import N_ACT


def mlp(d_in: int, d_out: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, d_out))


def with_action(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.cat([z, F.one_hot(a, N_ACT).to(z.dtype)], -1)


class Encoder(nn.Module):
    def __init__(self, d: int, d_in: int = 64):
        super().__init__()
        self.net, self.bn = mlp(d_in, d), nn.BatchNorm1d(d)

    def forward(self, x):
        return self.bn(self.net(x))


class Predictor(nn.Module):
    """(z, a) -> mean, and a log-variance when `het` (M3)."""

    def __init__(self, d: int, het: bool = False):
        super().__init__()
        self.het, self.net = het, mlp(d + N_ACT, 2 * d if het else d)

    def forward(self, z, a):
        out = self.net(with_action(z, a))
        return out.chunk(2, -1) if self.het else (out, None)


class Ensemble(nn.Module):
    """K heads of (trainable MLP + frozen random prior) with Bernoulli bootstrap masks (M2)."""

    def __init__(self, d: int, k: int = 5, hidden: int = 64, keep: float = 0.8):
        super().__init__()
        self.keep = keep
        self.f = nn.ModuleList(mlp(d + N_ACT, d, hidden) for _ in range(k))
        self.prior = nn.ModuleList(mlp(d + N_ACT, d, hidden).requires_grad_(False)
                                   for _ in range(k))

    def forward(self, z, a):  # (K, n, D)
        h = with_action(z, a)
        return torch.stack([f(h) + p(h) for f, p in zip(self.f, self.prior, strict=True)])

    def loss(self, out, target):
        mask = (torch.rand(out.shape[:2], device=out.device) < self.keep).float()
        return (mask[..., None] * (out - target) ** 2).mean()


@torch.no_grad()
def ema_(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t, s in zip(target.parameters(), source.parameters(), strict=True):
        t.lerp_(s, 1 - tau)


def frozen_copy(m: nn.Module) -> nn.Module:
    return copy.deepcopy(m).requires_grad_(False)


@lru_cache(maxsize=4)
def _knots(device: str, n: int = 17, t_max: float = 3.0):
    """Knots on [0, t_max] with doubled interior weights (symmetry): LeWM's `SIGReg` module."""
    t = torch.linspace(0, t_max, n, device=device)
    phi = torch.exp(-t**2 / 2)
    dt = t_max / (n - 1)
    w = torch.full((n,), 2 * dt, device=device)
    w[[0, -1]] = dt
    return t, phi, w * phi


def sigreg_slices(z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Epps-Pulley statistic per random direction. z: (T, N, D), W: (D, S) unit columns -> (S,)."""
    t, phi, w = _knots(str(z.device))
    xt = (z @ W).unsqueeze(-1) * t  # (T, N, S, K)
    err = (xt.cos().mean(1) - phi) ** 2 + xt.sin().mean(1) ** 2
    return (err @ w).mean(0) * z.shape[1]
