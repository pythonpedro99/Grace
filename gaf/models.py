"""Frame encoder, clip predictor, EMA helpers and the SIGReg loss (Epps-Pulley, as in LeWM).

The encoder has no BatchNorm: GroupNorm is per-sample, so gradients through the target frame
never leak into the context frames via batch statistics. That keeps gain attribution exact.
"""
import copy
from functools import lru_cache

import torch
from torch import nn

from .env import ACT_DIM


def mlp(d_in: int, d_out: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, d_out))


class Encoder(nn.Module):
    """(..., 3, res, res) -> (..., D). Three stride-2 convs, then an MLP head."""

    def __init__(self, D: int, res: int = 32, width: int = 32):
        super().__init__()
        w = width

        def block(i, o):
            return [nn.Conv2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU()]

        self.conv = nn.Sequential(*block(3, w), *block(w, 2 * w), *block(2 * w, 2 * w),
                                  nn.Flatten())
        self.head = nn.Sequential(nn.Linear(2 * w * (res // 8) ** 2, 256), nn.GELU(),
                                  nn.Linear(256, D))

    def forward(self, x):
        lead = x.shape[:-3]
        return self.head(self.conv(x.reshape(-1, *x.shape[-3:]))).view(*lead, -1)


class Decoder(nn.Module):
    """(..., D) -> (..., 3, res, res). Reconstruction head of the shadow encoder."""

    def __init__(self, D: int, res: int = 32, width: int = 32):
        super().__init__()
        self.w, self.r = 2 * width, res // 8
        self.fc = nn.Sequential(nn.Linear(D, 256), nn.GELU(),
                                nn.Linear(256, self.w * self.r**2), nn.GELU())

        def up(i, o):
            return [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU()]

        self.net = nn.Sequential(*up(self.w, self.w), *up(self.w, width),
                                 nn.ConvTranspose2d(width, 3, 4, 2, 1))

    def forward(self, z):
        lead = z.shape[:-1]
        h = self.fc(z.reshape(-1, z.shape[-1])).view(-1, self.w, self.r, self.r)
        return self.net(h).view(*lead, 3, self.r * 8, self.r * 8)


class Predictor(nn.Module):
    """Context latents (n, T-1, D) and actions (n, T-1, act_dim) -> target latent (n, D)."""

    def __init__(self, D: int, T: int = 3, act_dim: int = ACT_DIM, hidden: int = 256):
        super().__init__()
        self.net = mlp((T - 1) * (D + act_dim), D, hidden)

    def forward(self, zc, a):
        return self.net(torch.cat([zc.flatten(1), a.flatten(1)], 1))


@torch.no_grad()
def ema_(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t, s in zip(target.parameters(), source.parameters(), strict=True):
        t.lerp_(s, 1 - tau)


def frozen_copy(m: nn.Module) -> nn.Module:
    return copy.deepcopy(m).requires_grad_(False).eval()


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


def sigreg(z: torch.Tensor, n_slices: int = 256) -> torch.Tensor:
    """Mean SIGReg over fresh unit-norm random slices. z: (T, N, D)."""
    W = torch.nn.functional.normalize(torch.randn(z.shape[-1], n_slices, device=z.device), dim=0)
    return sigreg_slices(z, W).mean()
