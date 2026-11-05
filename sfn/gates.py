"""Training-time statistics (eigenbases, shortcut score) and the target-path gate G of every arm."""
import torch

EPS = 1e-6
C_LP = 0.02  # learning-progress scale of the M1 gates
M_PER_D = 4  # M1b/M1c use M = 4D directions


def _eig(C: torch.Tensor) -> torch.Tensor:
    """Eigenvectors as columns, largest eigenvalue first."""
    return torch.linalg.eigh(C)[1].flip(-1)


def _random_basis(d: int, device) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(d, d, device=device))[0]


class Stats:
    """EMA covariances behind U (full-predictor residual), V (weak predictor) and the score s."""

    def __init__(self, d, device, tau=0.99, every=100, tau_w=0.8, temp=0.05, k=4):
        self.d, self.dev, self.tau, self.every, self.k = d, device, tau, every, k
        self.tau_w, self.temp = tau_w, temp
        self.C = {}
        self.U = self.V = self.Q = torch.eye(d, device=device)
        self.s = torch.zeros(d, device=device)
        self.stab = (1.0, 1.0)  # subspace stability of (U, V) at the last refresh

    @torch.no_grad()
    def update(self, step, r, zn, p_w):
        zc, pc, err = zn - zn.mean(0), p_w - p_w.mean(0), zn - p_w
        new = {k: m.T @ m / len(m) for k, m in dict(r=r, p=pc, z=zc, e=err).items()}
        self.C = new if not self.C else {k: torch.lerp(self.C[k], v, 1 - self.tau)
                                         for k, v in new.items()}
        if step > 0 and step % self.every == 0:
            self.refresh()

    @torch.no_grad()
    def refresh(self):
        U, V = _eig(self.C["r"]), _eig(self.C["p"])
        k = self.k
        self.stab = tuple(((a[:, :k].T @ b[:, :k]) ** 2).sum().item() / k
                          for a, b in ((self.U, U), (self.V, V)))
        self.U, self.V, self.Q = U, V, _random_basis(self.d, self.dev)
        var = torch.einsum("ik,ij,jk->k", V, self.C["z"], V)
        err = torch.einsum("ik,ij,jk->k", V, self.C["e"], V)
        self.s = torch.sigmoid((1 - err / var.clamp_min(EPS) - self.tau_w) / self.temp)

    def basis(self, bfac: str):
        return {"B0": None, "B-rand": self.Q}.get(bfac, self.V)


def slice_weights(W: torch.Tensor, basis, s: torch.Tensor, w_min: float = 0.2):
    """SIGReg weight per slice (B1/B2/B-rand); `basis` is V, a random basis, or None (B0)."""
    if basis is None:
        return 1.0
    return (1 - ((W.T @ basis) ** 2 * s).sum(1)).clamp_min(w_min)


def _progress(now, lag):
    return (lag - now) / (lag + EPS)


def _gate(p):
    return 1 - (p / C_LP).clamp(0, 1)


class Gater:
    """Returns G (D x D) so that target = sg(z') + (z' - sg(z')) @ G."""

    def __init__(self, arm, d, device, alpha=0.0, beta=1.0, warmup=2000):
        self.arm, self.d, self.alpha, self.beta, self.warmup = arm, d, alpha, beta, warmup
        self.dev = device
        self.eye, self.zero = torch.eye(d, device=device), torch.zeros(d, d, device=device)
        self.smooth = self.fast = self.slow = None
        if arm == "M1c":
            self.bank = self._bases()

    def _bases(self):
        return torch.cat([_random_basis(self.d, self.dev) for _ in range(M_PER_D)], 1)

    def _lp_gate(self, B, r, r_lag, scale):
        g = _gate(_progress((r @ B).pow(2).mean(0), (r_lag @ B).pow(2).mean(0)))
        return scale * (B * g) @ B.T

    @torch.no_grad()
    def __call__(self, step, r, r_lag, U, e=None, u=None, v=None):
        arm = self.arm
        if arm in ("A0", "C1", "C3"):
            return self.eye
        if arm == "A1":
            return self.alpha * self.eye
        if arm == "A2":
            return self.zero
        if arm == "C2":
            return self.zero if step < self.warmup else self.eye
        if arm == "M2":  # e, u: per-dim epistemic and total error of the ensemble
            return torch.diag(1 - (self.beta * e / (u + EPS)).clamp(0, 1))
        if arm == "M3":  # v: per-dim predicted residual variance
            return torch.diag((v / (r.pow(2).mean(0) + EPS)).clamp(0, 1))
        if arm == "M1a":
            p = _progress(r.pow(2).mean(0), r_lag.pow(2).mean(0))
            self.smooth = p if self.smooth is None else 0.9 * self.smooth + 0.1 * p
            G = torch.diag(_gate(self.smooth))
        elif arm == "M1b":
            G = self._lp_gate(self._bases(), r, r_lag, 1 / M_PER_D)
        elif arm == "M1c":  # naive slope of the running loss on a fixed bank
            now = (r @ self.bank).pow(2).mean(0)
            if self.fast is None:
                self.fast = self.slow = now
            self.fast, self.slow = 0.9 * self.fast + 0.1 * now, 0.99 * self.slow + 0.01 * now
            G = (1 / M_PER_D) * (self.bank * _gate((self.slow - self.fast)
                                                   / (self.slow + EPS))) @ self.bank.T
        elif arm == "M1d":
            G = self._lp_gate(U, r, r_lag, 1.0)
        else:
            raise ValueError(f"unknown arm {arm}")
        return self.zero if step < self.warmup else G
