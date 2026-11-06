"""Progress gate with floor check (one-pager sections 3 and 4).

Per latent direction j (columns of the basis Q):
    gap_j = EMA_W[ L_j(P_lag) - L_j(P) ] / (Lbar_j + delta)       relative learning progress
    e_j   = (Lbar_j - L*_j) / (L*_j + delta)                        excess over the floor
    j in S  <=>  gap_j > eps_g  or  e_j > eps_f                    (with hysteresis)

S = directions where the predictor is still learning: the encoder must not simplify them.
L*_j comes from `floor_probe`: a fresh predictor trained on a frozen snapshot of the encoder.
"""
from collections import deque

import torch
from tqdm.auto import tqdm

from .models import Predictor, frozen_copy


class FloorGate:
    def __init__(self, D, device, window=200, eps_g=0.02, eps_f=0.25, delta=0.01,
                 min_dwell=500, floor_keep=3, floor_margin=0.1, soft=False, gap_ref=0.1):
        self.D, self.dev = D, device
        self.a, self.eps_g, self.eps_f, self.delta = 1 / window, eps_g, eps_f, delta
        self.min_dwell, self.margin = min_dwell, floor_margin
        self.soft, self.gap_ref = soft, gap_ref
        self.Q = torch.eye(D, device=device)
        self.gap = torch.zeros(D, device=device)  # EMA of L_lag - L (absolute)
        self.Lbar = None  # EMA of L
        self.hist = deque(maxlen=floor_keep)  # raw floors of recent refreshes, in basis Q
        self.inS = torch.ones(D, dtype=torch.bool, device=device)  # protect everything at first
        self.dwell = torch.zeros(D, device=device)

    # ---- per-direction quantities ---------------------------------------------------------
    def per_dir(self, r: torch.Tensor) -> torch.Tensor:
        """Residuals (n, D) -> batch-mean squared error per direction of Q, (D,)."""
        return (r @ self.Q).pow(2).mean(0)

    @property
    def Lstar(self):
        if not self.hist:
            return None
        return torch.stack(list(self.hist)).min(0).values * (1 - self.margin)

    @property
    def rgap(self):
        return self.gap / (self.Lbar + self.delta)

    @property
    def excess(self):
        Ls = self.Lstar
        if Ls is None:
            return torch.full((self.D,), float("inf"), device=self.dev)
        return (self.Lbar - Ls) / (Ls + self.delta)

    @torch.no_grad()
    def update(self, L: torch.Tensor, L_lag: torch.Tensor) -> None:
        """One step: both losses per direction, same batch and latents."""
        self.Lbar = L.clone() if self.Lbar is None else torch.lerp(self.Lbar, L, self.a)
        self.gap.lerp_(L_lag - L, self.a)
        g, e = self.rgap, self.excess
        enter = (g > self.eps_g) | (e > self.eps_f)
        stay = (g > self.eps_g / 2) | (e > self.eps_f / 2) | (self.dwell < self.min_dwell)
        new = torch.where(self.inS, stay, enter)
        self.dwell = torch.where(new & self.inS, self.dwell + 1, torch.zeros_like(self.dwell))
        self.inS = new

    # ---- weights over directions ------------------------------------------------------------
    def w_S(self) -> torch.Tensor:
        """Weights of the protected (still learning) directions."""
        return self.inS.float()

    def w_open(self) -> torch.Tensor:
        """Weights of the open target path: saturated directions (soft: scaled by 1 - gap)."""
        w = (~self.inS).float()
        if self.soft:
            w = w * (1 - (self.rgap / self.gap_ref).clamp(0, 1))
        return w

    # ---- floor refresh ------------------------------------------------------------------------
    @torch.no_grad()
    def set_floor(self, Lstar_raw: torch.Tensor, Q_new: torch.Tensor | None = None) -> None:
        """Add a probe floor (in basis Q_new if given). Rotates the running state to Q_new."""
        if Q_new is not None:
            R = (self.Q.T @ Q_new) ** 2  # (old, new); columns sum to 1
            rot = lambda v: R.T @ v  # noqa: E731
            self.gap = rot(self.gap)
            self.Lbar = None if self.Lbar is None else rot(self.Lbar)
            self.inS = rot(self.inS.float()) > 0.5
            self.dwell = rot(self.dwell)
            self.hist = deque((rot(h) for h in self.hist), maxlen=self.hist.maxlen)
            self.Q = Q_new
        self.hist.append(Lstar_raw)

    def state(self) -> dict:
        """Per-direction snapshot for logging (cpu tensors)."""
        Ls = self.Lstar
        return {"gap": self.rgap.cpu(), "excess": self.excess.clamp(max=1e3).cpu(),
                "L": self.Lbar.cpu(), "Lstar": (Ls if Ls is not None
                                                else torch.full_like(self.gap, float("nan"))).cpu(),
                "inS": self.inS.cpu()}


@torch.no_grad()
def encode_pool(enc_ctx, enc_tgt, env, n: int, chunk: int = 2048):
    """n fresh clips: context latents (n, T-1, D), actions (n, T-1, A), target latents (n, D)."""
    zc, a, zt = [], [], []
    for i in range(0, n, chunk):
        clip = env.sample(min(chunk, n - i))
        zc.append(enc_ctx(clip.x[:, :-1]))
        a.append(clip.a)
        zt.append(enc_tgt(clip.x[:, -1]))
    return torch.cat(zc), torch.cat(a), torch.cat(zt)


def excess_basis(C_P, C_star, C_z):
    """Orthonormal basis ordered by the predictor's excess over the floor, in white coordinates.

    Solves (C_P - C_star) v = lam C_z v with C_z = Cov(z_t). SIGReg makes z white (C_z ~ I), so
    lam is the share of the variance along v that the predictor still misses beyond the floor:
    large for content it has not learned yet, ~0 for learned content and for noise at its floor.
    Using the measured C_z (not I) keeps this exact when SIGReg holds only approximately.
    The solutions are C_z-orthonormal; QR in order turns them into a Euclidean-orthonormal Q with
    the same nested top-k subspaces, so the split of the gradient over directions stays exact.
    Returns (Q, lam), lam in decreasing order.
    """
    ev, U = torch.linalg.eigh(C_z)
    W = U @ torch.diag(ev.clamp_min(1e-8).rsqrt()) @ U.T  # C_z^(-1/2)
    M = W @ (C_P - C_star) @ W
    lam, Uw = torch.linalg.eigh((M + M.T) / 2)
    lam, Uw = lam.flip(0), Uw.flip(1)
    Q, R = torch.linalg.qr(W @ Uw)
    return Q * torch.sign(torch.diagonal(R)), lam


def floor_probe(enc_ctx, enc_tgt, env, steps=1500, n_pool=16384, batch=512, lr=1e-3, wd=0.0,
                basis="excess", hidden=256, probe=None, pred=None, desc=None):
    """Floor L*_j: a predictor trained on frozen snapshots of the encoder(s).

    Trained with AdamW(lr, wd) like the predictor, so P and the probe can reach the same error.
    Returns (Lstar (D,), Q (D, D), probe), scored on held-out clips. `probe`: warm start from a
    previous probe (its budget accumulates over refreshes); None trains a fresh one.
    Basis Q of the gate:
      "excess" (needs `pred`, the current predictor): `excess_basis` of the predictor's and the
               probe's held-out residual covariances in the metric Cov(z_t): directions ranked
               by the variance share the predictor still misses. Separates not-yet-learned
               from learned content and noise.
      "resid"  eigenvectors of the probe's residual covariance, largest first: separates noise
               (high floor) from predictable content, but not learned from not-yet-learned.
      "dim"    Q = I (latent dims; arbitrary under SIGReg's rotation invariance).
    desc: label of a progress bar over the probe's training steps (None: none).
    """
    enc_ctx, enc_tgt = frozen_copy(enc_ctx), frozen_copy(enc_tgt)
    zc, a, zt = encode_pool(enc_ctx, enc_tgt, env, n_pool)
    n_tr = int(0.75 * len(zt))
    D, T = zt.shape[1], zc.shape[1] + 1
    probe = probe if probe is not None else Predictor(D, T, a.shape[-1], hidden).to(zt.device)
    opt = torch.optim.AdamW(probe.parameters(), lr, weight_decay=wd)
    for _ in tqdm(range(steps), desc=desc, unit="step", disable=desc is None, leave=False):
        i = torch.randint(0, n_tr, (batch,), device=zt.device)
        loss = (probe(zc[i], a[i]) - zt[i]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        r = probe(zc[n_tr:], a[n_tr:]) - zt[n_tr:]
        C_star = r.T @ r / len(r)
        if basis == "excess":
            rp = pred(zc[n_tr:], a[n_tr:]) - zt[n_tr:]
            Q, _ = excess_basis(rp.T @ rp / len(rp), C_star, torch.cov(zt[n_tr:].T))
        elif basis == "resid":
            Q = torch.linalg.eigh(C_star)[1].flip(-1)
        else:
            Q = torch.eye(D, device=zt.device)
    return (r @ Q).pow(2).mean(0), Q, probe
