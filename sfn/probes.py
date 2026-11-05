"""Ridge probes and the metrics of §7. Fit on one fresh sample, score on another."""
import torch
from torch import nn

from .env import GROUPS, N_GROUPS

FLAGGED = (0, 2, 3, 6)  # groups 1, 3, 4, 7 (0-based): pre-registered shortcut groups
FLAG_COLS = torch.tensor([c for g in FLAGGED for c in range(2 * g, 2 * g + 2)])
EPS = 1e-6


def fit_ridge(X, Y, lam=1e-3):
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    A = Xc.T @ Xc + lam * len(X) * torch.eye(X.shape[1], device=X.device)
    W = torch.linalg.solve(A, Xc.T @ (Y - ym))
    return W, ym - xm @ W


def r2(Y, Yh, per_col=False):
    sse, sst = ((Y - Yh) ** 2).sum(0), ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - sse / sst if per_col else (1 - sse.sum() / sst.sum()).item()


def poly2(y):
    a, b = y[:, :1], y[:, 1:]
    return torch.cat([y, a * a, a * b, b * b], 1)


def _explained(z, y):
    """Variance share and participation ratio of the part of z explained by degree-2 features."""
    W, c = fit_ridge(poly2(y), z)
    zh = poly2(y) @ W + c
    S = torch.cov(zh.T)
    return (S.trace() / torch.cov(z.T).trace()).item(), (S.trace() ** 2 / (S**2).sum()).item()


@torch.no_grad()
def evaluate(enc, P, env, G, stats, n=8192):
    """One flat row of metrics. G is the latest target-path gate (D x D)."""
    enc.eval()
    b = env.sample(2 * n)
    z, zn = enc(b.x), enc(b.xn)
    zh = P(z, b.a)[0]
    enc.train()
    tr, te = slice(0, n), slice(n, None)

    def ridge_r2(X, Y, sel=te):
        W, c = fit_ridge(X[tr], Y[tr])
        return r2(Y[sel], X[sel] @ W + c), W

    row, W_lin = {}, {}
    for k, s in enumerate(GROUPS, 1):
        row[f"r2_{k}"], W_lin[k] = ridge_r2(z, b.gt[:, s])
        row[f"vs_{k}"], row[f"ed_{k}"] = _explained(z, b.gt[:, s])
        Q = torch.linalg.qr(W_lin[k])[0]
        row[f"gate_{k}"] = torch.diagonal(Q.T @ G @ Q).mean().item()
        if 4 <= k <= 7:
            row[f"dyn_{k}"], _ = ridge_r2(zh, b.gtn[:, s])
    jump = torch.zeros(2 * n, dtype=torch.bool, device=z.device)
    jump[n:] = b.jump[n:]
    row["r2_3_post"], _ = ridge_r2(zn, b.gtn[:, GROUPS[2]], sel=jump)

    # shortcut flags: s_k against the ground-truth flagged groups
    zv, yF = z @ stats.V, b.gt[:, FLAG_COLS.to(z.device)]
    W, c = fit_ridge(yF[tr], zv[tr])
    a_k = r2(zv[te], yF[te] @ W + c, per_col=True)
    row["flag_precision"] = ((stats.s * a_k).sum() / (stats.s.sum() + EPS)).item()
    zs = zv * stats.s
    row["flag_recall"] = sum(ridge_r2(zs, b.gt[:, GROUPS[g]])[0] / max(row[f"r2_{g + 1}"], EPS)
                             for g in FLAGGED) / len(FLAGGED)
    row["stab_U"], row["stab_V"] = stats.stab
    return row


def mlp_probe(enc, env, n_train=20000, n_test=8192, steps=500):
    """Final held-out 2-layer MLP probe from z_t to all groups (rules out 'nonlinearly present')."""
    enc.eval()
    with torch.no_grad():
        b = env.sample(n_train + n_test)
        z = enc(b.x)
    enc.train()
    net = nn.Sequential(nn.Linear(z.shape[1], 128), nn.GELU(), nn.Linear(128, 16)).to(z.device)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(steps):
        opt.zero_grad()
        ((net(z[:n_train]) - b.gt[:n_train]) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        yh = net(z[n_train:])
    return {f"mlp_r2_{k}": r2(b.gt[n_train:, s], yh[:, s]) for k, s in enumerate(GROUPS, 1)}


@torch.no_grad()
def calibrate_beta(enc, heads, env, n=8192, target=0.9):
    """M2's beta such that the gate equals `target` on the latent direction aligned with group 8."""
    enc.eval()
    b = env.sample(n)
    z, zn = enc(b.x), enc(b.xn)
    enc.train()
    W, _ = fit_ridge(z, b.gt[:, GROUPS[N_GROUPS - 1]])
    q = torch.linalg.svd(W)[0][:, 0]
    out = heads(z, b.a)
    e, u = (out @ q).var(0).mean(), ((out - zn) @ q).pow(2).mean()
    return ((1 - target) * u / e).item()

