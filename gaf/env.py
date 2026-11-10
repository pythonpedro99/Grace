"""Clip world: low-res RGB clips of T frames built from four kinds of content.

    slow   n_slow smooth colour patterns, amplitudes follow an OU process (rho close to 1).
           Gaussian across clips, so they alone can satisfy SIGReg while being trivially
           predictable: the shortcut the encoder can collapse onto.
    fast   a sprite pushed by the action: p' = p + v_act a (mod 1, a torus).
           Deterministic given the action; unpredictable without it.
    hard   a sprite on an action-driven chaotic circle map q' = k q + h_act a (mod 1).
           Deterministic given the action but harder to learn (degree-k map in the latent).
    noise  n_noise patterns of the same kind as slow, resampled every frame. Unpredictable:
           the only content that should be dropped.

Actions a_t in [-1, 1]^2 (uniform random policy) act between frames t and t+1; the predictor
sees all T-1 of them. Frames 0..T-2 are context, frame T-1 is the target. The pattern bases
(slow and noise together) are orthonormal in pixel space, so every pattern factor is linearly
decodable.
"""
import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

GROUPS = ("slow", "fast", "hard", "noise")
ACT_DIM = 2


@dataclass(frozen=True)
class EnvCfg:
    res: int = 32  # image side, must be divisible by 8
    T: int = 3  # frames per clip; the last one is the target
    n_slow: int = 16
    slow_rho: float = 0.98  # per-frame OU coefficient of the slow amplitudes
    n_noise: int = 8
    v_act: float = 0.3  # fast sprite: displacement per unit action, in image widths
    # without the action, R^2 of e(p') from e(p) is capped at (sin(2 pi v) / (2 pi v))^2: 0.25
    hard_k: int = 2  # hard sprite map multiplier (2 = doubling map, higher = harder)
    # hard: the same cap, but a long plateau before it is learned. The plateau grows steeply with
    # h_act (ground-truth predictor, t90 in steps: 0.1 <750, 0.2-0.25 ~1500, 0.3 2250-3000 over
    # 4 seeds, 0.35 unreliable, >= 0.4 not within 6000 steps: unlearnable, not hard to learn)
    h_act: float = 0.3  # hard sprite: displacement per unit action
    sprite_sigma: float = 0.06  # sprite radius in image widths
    a_slow: float = 1.0  # pixel RMS of the slow part
    a_noise: float = 1.0  # pixel RMS of the noise part
    a_fast: float = 4.0  # sprite peak brightness (fast: white, hard: red)
    a_hard: float = 4.0
    pixel_noise: float = 0.05
    max_freq: int = 3  # highest spatial frequency of the patterns
    seed: int = 0  # fixes the patterns and sprite colours
    swap_colors: bool = False  # fast sprite red and hard sprite white, instead of the reverse


class Clip(NamedTuple):
    x: torch.Tensor  # (n, T, 3, res, res)
    a: torch.Tensor  # (n, T-1, 2) actions; a[:, t] acts between frames t and t+1
    f: dict  # group -> (n, T, d_group) ground-truth factors per frame


def _emb(p: torch.Tensor) -> torch.Tensor:
    """(..., 2) positions on the torus -> (..., 4) = cos, sin of each coordinate."""
    a = 2 * math.pi * p
    return torch.stack([a.cos(), a.sin()], -1).flatten(-2)


def _patterns(n: int, res: int, max_freq: int, gen: torch.Generator, modes: int = 6):
    """n smooth random colour fields: sums of low-frequency Fourier modes with random colours."""
    g = torch.arange(res) / res
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    fx = torch.randint(-max_freq, max_freq + 1, (n, modes), generator=gen).float()
    fy = torch.randint(-max_freq, max_freq + 1, (n, modes), generator=gen).float()
    ph = 2 * math.pi * torch.rand(n, modes, generator=gen)
    col = torch.randn(n, modes, 3, generator=gen)
    wave = torch.cos(2 * math.pi * (fx[..., None, None] * xx + fy[..., None, None] * yy)
                     + ph[..., None, None])
    return torch.einsum("nmc,nmhw->nchw", col, wave)


def _bump(p: torch.Tensor, res: int, sigma: float) -> torch.Tensor:
    """Periodic Gaussian sprite. p: (..., 2) = (x, y) in [0, 1) -> (..., res, res)."""
    g = (torch.arange(res, device=p.device) + 0.5) / res
    d = (g - p[..., None] + 0.5) % 1 - 0.5  # (..., 2, res)
    b = torch.exp(-(d**2) / (2 * sigma**2))
    return b[..., 1, :, None] * b[..., 0, None, :]


class ClipEnv:
    """Samples i.i.d. clips on `device`. Uses the global torch RNG (seed it per run)."""

    act_dim = ACT_DIM
    gt_extra = ()  # extra ground-truth inputs for run_gt (none here)

    def __init__(self, cfg: EnvCfg = EnvCfg(), device="cpu"):
        assert cfg.res % 8 == 0 and cfg.T >= 2
        self.cfg, self.dev = cfg, device
        gen = torch.Generator().manual_seed(cfg.seed)
        n = cfg.n_slow + cfg.n_noise
        P = _patterns(n, cfg.res, cfg.max_freq, gen).flatten(1)  # (n, 3 res^2)
        Q = torch.linalg.qr(P.T)[0].T  # orthonormal rows, still smooth
        npix = Q.shape[1]
        # each group gets pixel RMS a: per-factor scale a * sqrt(npix / n_group)
        self.B_slow = (Q[: cfg.n_slow] * cfg.a_slow * math.sqrt(npix / cfg.n_slow)).to(device)
        self.B_noise = (Q[cfg.n_slow:] * cfg.a_noise * math.sqrt(npix / max(cfg.n_noise, 1))
                        ).to(device)
        self.Q = Q.to(device)  # unit-norm bases, used by the oracle encoder in tests
        cols = torch.tensor([[1.0, 1.0, 1.0], [1.0, -0.6, -0.6]])  # fast: white, hard: red
        cols = (cols / cols.norm(dim=1, keepdim=True) * math.sqrt(3)).to(device)
        self.col_fast, self.col_hard = cols.flip(0) if cfg.swap_colors else cols
        self.dims = {"slow": cfg.n_slow, "fast": 4, "hard": 4, "noise": cfg.n_noise}

    @torch.no_grad()
    def sample(self, n: int, render: bool = True) -> Clip:
        """n i.i.d. clips. render=False skips the pixels (x is None): ground-truth checks only."""
        c, dev, T = self.cfg, self.dev, self.cfg.T
        s = [torch.randn(n, c.n_slow, device=dev)]
        for _ in range(T - 1):
            s.append(c.slow_rho * s[-1] + math.sqrt(1 - c.slow_rho**2)
                     * torch.randn(n, c.n_slow, device=dev))
        s = torch.stack(s, 1)  # (n, T, n_slow)
        noise = torch.randn(n, T, c.n_noise, device=dev)
        a = 2 * torch.rand(n, T - 1, ACT_DIM, device=dev) - 1
        p, q = [torch.rand(n, 2, device=dev)], [torch.rand(n, 2, device=dev)]
        for t in range(T - 1):
            p.append((p[-1] + c.v_act * a[:, t]) % 1)
            q.append((c.hard_k * q[-1] + c.h_act * a[:, t]) % 1)
        p, q = torch.stack(p, 1), torch.stack(q, 1)  # (n, T, 2)

        f = {"slow": s, "fast": _emb(p), "hard": _emb(q), "noise": noise}
        if not render:
            return Clip(None, a, f)
        x = s @ self.B_slow + noise @ self.B_noise  # (n, T, 3 res^2)
        x = x.view(n, T, 3, c.res, c.res)
        for amp, col, pos in ((c.a_fast, self.col_fast, p), (c.a_hard, self.col_hard, q)):
            x = x + amp * col[:, None, None] * _bump(pos, c.res, c.sprite_sigma)[:, :, None]
        x = x + c.pixel_noise * torch.randn_like(x)
        return Clip(x, a, f)

    def factors(self, clip: Clip, t: int = -1) -> torch.Tensor:
        """All groups at frame t, concatenated in GROUPS order: (n, sum of dims)."""
        return torch.cat([clip.f[g][:, t] for g in GROUPS], -1)

    def group_cols(self) -> dict:
        out, i = {}, 0
        for g in GROUPS:
            out[g] = slice(i, i + self.dims[g])
            i += self.dims[g]
        return out
