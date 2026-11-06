"""Probing suite: what each latent keeps of slow / fast / hard / noise content.

All probes fit on one fresh sample and score on another (held-out clips, eval mode).
"""
import torch
from torch import nn

from .env import GROUPS


def fit_ridge(X, Y, lam=1e-3):
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    A = Xc.T @ Xc + lam * len(X) * torch.eye(X.shape[1], device=X.device)
    W = torch.linalg.solve(A, Xc.T @ (Y - ym))
    return W, ym - xm @ W


def r2(Y, Yh, per_col=False):
    sse, sst = ((Y - Yh) ** 2).sum(0), ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - sse / sst if per_col else (1 - sse.sum() / sst.sum()).item()


def ridge_r2(X, Y, n_tr, per_col=False):
    W, c = fit_ridge(X[:n_tr], Y[:n_tr])
    return r2(Y[n_tr:], X[n_tr:] @ W + c, per_col)


def eff_rank(X) -> float:
    """Participation ratio of the covariance of X: (sum lambda)^2 / sum lambda^2."""
    ev = torch.linalg.eigvalsh(torch.cov(X.T)).clamp_min(0)
    return (ev.sum() ** 2 / (ev**2).sum()).item()


@torch.no_grad()
def encode(enc, x, chunk=4096):
    """enc over the leading (clip) dim in chunks, to bound GPU memory at eval time."""
    return torch.cat([enc(x[i:i + chunk]) for i in range(0, len(x), chunk)])


@torch.no_grad()
def probe_suite(enc, P, env, Q=None, n=8192, enc_tgt=None):
    """Scalar metrics (dict) and per-direction content (D x groups) of the current model.

    r2_<g>      ridge R^2 of group g at the target frame from z_t (the representation)
    pred_r2_<g> ridge R^2 of group g at the target frame from the prediction P(z_c)
    vs_<g>      share of tr Cov(z_t) linearly explained by group g's factors
    wb_ratio    mean within-clip variance / between-clip variance of z (slow features -> 0)
    rank_dz     effective rank of z_t - z_{t-1} (slow-feature drift)
    content     (D, groups): R^2 of each latent direction (basis Q) from each group
    """
    enc_tgt = enc_tgt or enc
    was = enc.training
    enc.eval()
    clip = env.sample(2 * n)
    zc, zt = encode(enc, clip.x[:, :-1]), encode(enc_tgt, clip.x[:, -1])
    z_last = zt if enc_tgt is enc else encode(enc, clip.x[:, -1])
    z_all = torch.cat([zc, z_last[:, None]], 1)  # online encoder on every frame
    p = P(zc, clip.a)
    enc.train(was)
    Q = torch.eye(zt.shape[1], device=zt.device) if Q is None else Q
    row, content = {}, []
    tot = torch.cov(zt.T).trace()
    for g in GROUPS:
        y = clip.f[g][:, -1]
        row[f"r2_{g}"] = ridge_r2(zt, y, n)
        row[f"pred_r2_{g}"] = ridge_r2(p, y, n)
        W, c = fit_ridge(y[:n], zt[:n])
        row[f"vs_{g}"] = (torch.cov((y[n:] @ W + c).T).trace() / tot).item()
        content.append(ridge_r2(y, zt @ Q, n, per_col=True))
    within = z_all.var(1).mean(0)  # per dim, across frames of a clip
    between = z_all.mean(1).var(0)
    row["wb_ratio"] = (within.mean() / between.mean()).item()
    row["rank_dz"] = eff_rank(z_all[:, -1] - z_all[:, -2])
    row["rank_z"] = eff_rank(zt)
    row["pred_mse"] = (p - zt).pow(2).mean().item()
    return row, torch.stack(content, 1).cpu()


def mlp_probe(enc, env, n_train=20000, n_test=8192, steps=1000, hidden=256):
    """Held-out 2-layer MLP probe from z_t to every group (rules out 'nonlinearly present')."""
    was = enc.training
    enc.eval()
    with torch.no_grad():
        clip = env.sample(n_train + n_test)
        z, y = encode(enc, clip.x[:, -1]), env.factors(clip)
    enc.train(was)
    net = nn.Sequential(nn.Linear(z.shape[1], hidden), nn.GELU(), nn.Linear(hidden, hidden),
                        nn.GELU(), nn.Linear(hidden, y.shape[1])).to(z.device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(steps):
        i = torch.randint(0, n_train, (512,), device=z.device)
        loss = (net(z[i]) - y[i]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        yh = net(z[n_train:])
    return {f"mlp_r2_{g}": r2(y[n_train:, s], yh[:, s]) for g, s in env.group_cols().items()}
