"""Reference map of reference-gated retention (docs/reference_gated_retention.md).

A shadow encoder (never shaped by the prediction loss) and a shadow predictor on its latents
give a *reference floor*: how much of each direction of the shadow latent is predictable in
principle. This module keeps the statistics behind it and turns them into per-direction
retention weights for the live encoder:

    shadow latent z_s, whitened:  y = W (z_s - mu),   W = Cov(z_s)^(-1/2)
    shadow predictor residual in whitened coordinates:  R = W C_res W,  R v_j = err_j v_j
        err_j  unexplained fraction of direction v_j        rho_j = 1 - err_j  predictable share
        gap_j  relative improvement of the shadow predictor over its lagged copy (still learning)
    live predictor, mapped through the retention head R(.) into shadow coordinates:
        err_live_j  its error on direction v_j

    retention weight   w_j = max( clip(gap_j / eps_g), rho_j * clip((err_live_j - err_j) / eps_f) )

    unsettled direction (shadow predictor still improving): retained, whatever rho says;
    settled and predictable (rho > 0): retained until the live predictor has caught up;
    settled and unpredictable (rho = 0, noise): free.
    Before the first refresh everything is retained.

Optional floor (ac_ref > 0): content that is hard to predict is not noise. A direction whose coordinate is correlated between
consecutive frames, ac_j = v_j' (W C1 W) v_j with C1 the lag-1 cross covariance of the shadow latent, is not i.i.d. noise (the grid
world: twinkle 0.00, hand 0.34, facing 0.20), so
    w_j = max(w_j, w_min * clip(ac_j / ac_ref, 0, 1)),
i.e. it is never released below w_min, however badly the shadow predictor predicts it. Costs live capacity for every
autocorrelated direction (static content too).
floor_all: the unconditional version, w_j = max(w_j, w_min) for every direction: nothing the shadow represents is released
below w_min, whatever its predictability or autocorrelation (noise is then left to the open target path only).
"""
import torch


class RefMap:
    def __init__(self, D, device, tau=0.995, every=50, eps_g=0.02, eps_f=0.1, delta=0.01, ac_ref=0.0, w_min=0.3, floor_all=False):
        self.D, self.dev, self.tau, self.every = D, device, tau, every
        self.ac_ref, self.w_min, self.floor_all = ac_ref, w_min, floor_all
        self.eps_g, self.eps_f, self.delta = eps_g, eps_f, delta
        eye = torch.eye(D, device=device)
        self.mu = torch.zeros(D, device=device)
        self.W, self.V, self.M = eye.clone(), eye.clone(), eye.clone()
        ones, zeros = torch.ones(D, device=device), torch.zeros(D, device=device)
        self.w, self.err, self.err_live = ones.clone(), ones.clone(), ones.clone()
        self.rho, self.gap, self.u, self.ac = zeros.clone(), zeros.clone(), ones.clone(), zeros.clone()
        self.C, self.n = None, 0

    @torch.no_grad()
    def update(self, z, r, r_lag, e_live, pairs=None):
        """z: shadow latents (M, D); r, r_lag: residuals of the shadow predictor and of its
        lagged copy (N, D); e_live: live prediction mapped into shadow coordinates, minus the
        whitened shadow target (N, D). Fresh clips every step, so these are held-out errors.
        pairs: (z_t, z_t+1) of consecutive frames (K, D) each, for the autocorrelation floor (ac_ref > 0)."""
        sec = lambda x: x.T @ x / len(x)  # noqa: E731
        new = {"g": torch.cov(z.T), "r": sec(r), "rl": sec(r_lag), "el": sec(e_live)}
        if self.ac_ref > 0 and pairs is not None:
            za, zb = pairs
            new["c1"] = (za - za.mean(0)).T @ (zb - zb.mean(0)) / len(za)
        self.C = new if self.C is None else {k: torch.lerp(self.C[k], v, 1 - self.tau)
                                             for k, v in new.items()}
        mu = z.mean(0)
        self.mu = mu if self.n == 0 else torch.lerp(self.mu, mu, 1 - self.tau)
        self.n += 1
        if self.n % self.every == 0:
            self.refresh()

    @torch.no_grad()
    def refresh(self):
        C, eye = self.C, torch.eye(self.D, device=self.dev)
        ev, U = torch.linalg.eigh(C["g"] + 1e-5 * eye)
        self.W = (U * ev.clamp_min(1e-5).rsqrt()) @ U.T
        sym = lambda m: (m + m.T) / 2  # noqa: E731
        Rm, Rl = sym(self.W @ C["r"] @ self.W), sym(self.W @ C["rl"] @ self.W)
        lam, V = torch.linalg.eigh(Rm)  # ascending: most predictable direction first
        diag = lambda m: torch.einsum("ij,ik,kj->j", V, m, V)  # noqa: E731  v_j^T m v_j
        self.V, self.err = V, lam.clamp_min(0)
        self.rho = (1 - self.err).clamp(0, 1)
        self.gap = diag(Rl - Rm) / (self.err + self.delta)
        self.err_live = diag(self.C["el"])
        self.u = (self.gap / self.eps_g).clamp(0, 1)
        rel = ((self.err_live - self.err) / self.eps_f).clamp(0, 1)
        self.w = torch.maximum(self.u, self.rho * rel)
        if self.ac_ref > 0 and "c1" in C:  # hard is not noise: autocorrelated directions keep at least w_min
            self.ac = diag(sym(self.W @ C["c1"] @ self.W))
            self.w = torch.maximum(self.w, self.w_min * (self.ac / self.ac_ref).clamp(0, 1))
        if self.floor_all:
            self.w = self.w.clamp_min(self.w_min)
        self.M = (V * self.w) @ V.T

    def basis(self):
        """Matrix whose columns map a shadow latent to the coordinates (v_j) of the weights."""
        return self.W @ self.V

    def summary(self) -> dict:
        return {"ref_w": self.w.mean().item(), "ref_rho": self.rho.mean().item(),
                "ref_unsettled": int((self.u > 0.5).sum()), "ref_free": int((self.w < 0.1).sum()),
                "ref_ac_n": int((self.ac > self.ac_ref).sum()) if self.ac_ref > 0 else 0}
