"""Mechanism diagnostics (cfg.log_mech > 0): what the loss terms do to the live encoder, what the retention map weights, and what the shadow
holds, measured directly and not only through probes. train.run calls light() at every eval and heavy() every cfg.log_mech steps; every
number goes to the eval row of metrics.csv (scalars) or to dirs.npz (per-direction arrays, rewritten at every eval).

light (cheap)
    spectrum    eigenvalues of Cov(z) of the live encoder and of the shadow (dirs: spec_live, spec_shadow); shadow SIGReg (sh_sigreg)
    gate        S_frac_<g> / S_n_<g>: the share of each group's directions (live basis) that the floor gate protects
    reference   per shadow direction: retention weight, predictable share, gap, live error, autocorrelation, dominant group
                (dirs: refdir_w, refdir_rho, refdir_gap, refdir_err_live, refdir_ac, refdir_dom, refdir_top)
heavy
    forces      gnorm_<term>, gcos_<a>_<b>: gradient norm of each loss term on the live encoder (pred = context path, open = floor-gated
                target path, ret = retention, sigreg, rec for the autoencoder arm) and the cosines between them
    attribution sens_<term>_<part>: first-order change of the part's linear readout along the step -lr * grad(term), in R^2 points
                per 1000 steps (positive: the term builds the part in the live latent, negative: it erodes it)
    retention   ret_fit_<g>: how well the retention head reads the shadow directions of group g from the live latent (1 - error/variance)
    facing      face_acc_<enc>: orientation of the decoded agent tile (template match, chance 0.25), rec_<class>_<enc>: unweighted
                reconstruction MSE per tile class, lat_sens_<what>_<enc>: how far the latent moves when only that thing changes,
                relative to a random pair of frames per unit of pixel distance
    probe size  pc_<enc>_<part>_<lin|s|l>: the part read by a ridge, a small and a large MLP (information present but hard to read)
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.func import functional_call, jvp

from .env import GROUPS
from .env_grid import N_MAT, TILE_CLASSES
from .models import sigreg
from .probes import collect_last, encode, fit_ridge

REC_GROUPS = {"agent": range(6, 10), "mob": [5], "hand": range(20, 24), "inventory": range(10, 20), "twinkle": [4],
              "map": range(0, 4), "compass": range(24, 28)}
THR = 0.15  # a direction belongs to the group it carries most, if that group explains more than this of it


def group_of(content):
    """(D, groups) R^2 of each direction from each group -> (dominant group index, its R^2)."""
    top, dom = content.max(1)
    return dom, top


# ---- light ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def light(c: SimpleNamespace, content, sh_content, dirs: dict, n: int = 2048) -> dict:
    """Cheap diagnostics at an eval. c: enc, shadow (or None), ref (or None), gate, env, cfg. content: live (D, groups) from probe_suite;
    sh_content: the shadow's (D, groups) in the reference basis (or None). Appends arrays to `dirs`, returns scalars."""
    out, env, cfg = {}, c.env, c.cfg
    clip = env.sample(n)
    z = encode(c.enc, clip.x[:, -1])
    dirs["spec_live"].append(torch.linalg.eigvalsh(torch.cov(z.T)).flip(0).cpu())
    if c.shadow is not None:
        zs = encode(c.shadow, clip.x)  # (n, T, D)
        dirs["spec_shadow"].append(torch.linalg.eigvalsh(torch.cov(zs[:, -1].T)).flip(0).cpu())
        out["sh_sigreg"] = sigreg(zs.transpose(0, 1), cfg.n_slices).item()
    dom, top = group_of(content.cpu())
    inS = c.gate.inS.cpu()
    for gi, g in enumerate(GROUPS):
        mg = (dom == gi) & (top > THR)
        out[f"S_n_{g}"] = int(mg.sum())
        out[f"S_frac_{g}"] = float(inS[mg].float().mean()) if mg.any() else float("nan")
    if c.ref is not None and sh_content is not None:
        sd, st = group_of(sh_content.cpu())
        r = c.ref
        for k, v in (("w", r.w), ("rho", r.rho), ("gap", r.gap), ("err_live", r.err_live), ("ac", r.ac)):
            dirs[f"refdir_{k}"].append(v.detach().cpu().float())
        dirs["refdir_dom"].append(sd.float())
        dirs["refdir_top"].append(st.float())
    return out


# ---- heavy: forces and attribution -----------------------------------------------------------------------------------------------
def _terms(c, clip):
    """The loss terms of the live encoder on a batch (grad enabled): name -> scalar; also the retention error (N, T, D) or None."""
    cfg, enc, D = c.cfg, c.enc, c.cfg.D
    xc, xt, a = clip.x[:, :-1], clip.x[:, -1], clip.a
    zc, zt = enc(xc), enc(xt)
    p = c.P(zc, a)
    terms = {"pred": (p - zt.detach()).pow(2).mean(),
             "sigreg": cfg.lam * sigreg(torch.cat([zc.transpose(0, 1), zt[None]]), cfg.n_slices)}
    err = None
    if c.use_open:
        terms["open"] = ((zt - p.detach()) @ c.gate.Q).pow(2).mul(c.gate.w_open()).mean()
    if c.ref is not None:
        with torch.no_grad():
            y = (c.shadow(clip.x) - c.ref.mu) @ c.ref.W
        err = c.Rh(torch.cat([zc, zt[:, None]], 1)) - y
        terms["ret"] = cfg.ret_lam * ((err @ c.ref.M) * err).sum(-1).mean() / D
    if c.ae_rec is not None:
        terms["rec"] = cfg.shadow_rec * c.ae_rec(clip, zc, zt)
    return terms, err


def forces(c, sh_dom, sh_top, n_probe: int = 1024) -> dict:
    """Gradient norms and cosines of the loss terms on the live encoder, their first-order effect on each part, and the retention fit."""
    out, cfg, enc = {}, c.cfg, c.enc
    params = list(enc.parameters())
    names = [k for k, _ in enc.named_parameters()]
    terms, err = _terms(c, c.env.sample(cfg.batch))
    grads = {}
    for k, v in terms.items():
        g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        grads[k] = [torch.zeros_like(p) if x is None else x.detach() for p, x in zip(params, g, strict=True)]
    flat = {k: torch.cat([x.flatten() for x in g]) for k, g in grads.items()}
    ks = list(flat)
    for k in ks:
        out[f"gnorm_{k}"] = flat[k].norm().item()
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            out[f"gcos_{a}_{b}"] = (flat[a] @ flat[b] / (flat[a].norm() * flat[b].norm()).clamp_min(1e-30)).item()
    # attribution: the readout of each part along the step of each term
    with torch.no_grad():
        clip = c.env.sample(n_probe)
        x, y = clip.x[:, -1], c.env.factors(clip)
        z0 = enc(x)
        cols = c.env.group_cols()
        W = {}
        for part, (g, s) in c.env.parts().items():
            sl = slice(cols[g].start + s.start, cols[g].start + s.stop)
            W[part] = (sl, fit_ridge(z0, y[:, sl])[0])
    base = tuple(p.detach() for p in params)
    for k, g in grads.items():
        tangent = tuple(-cfg.lr * x_ for x_ in g)
        _, dz = jvp(lambda *ps: functional_call(enc, dict(zip(names, ps, strict=True)), (x,)), base, tangent)
        for part, (sl, Wp) in W.items():
            yt = y[:, sl] - y[:, sl].mean(0)
            out[f"sens_{k}_{part}"] = 1000 * ((dz @ Wp) * yt).sum().item() / yt.pow(2).sum().clamp_min(1e-12).item()
    if err is not None:  # how well the retention head reads the shadow directions of each group
        with torch.no_grad():
            fit = 1 - (err.detach() @ c.ref.V).pow(2).mean((0, 1))  # whitened shadow coordinates have unit variance
            for gi, g in enumerate(GROUPS):
                mg = (sh_dom == gi) & (sh_top > THR)
                out[f"ret_fit_{g}"] = float(fit[mg.to(fit.device)].mean()) if mg.any() else float("nan")
    return out


# ---- heavy: what the shadow / encoder holds of the facing, hand, mob ---------------------------------------------------------------
@torch.no_grad()
def decoder_view(c, enc, dec, tag: str, n: int = 1024) -> dict:
    """Orientation accuracy of the decoded agent tile and the unweighted reconstruction error per tile class."""
    env, cfg = c.env, c.cfg.env
    clip = env.sample(n)
    xh = dec(encode(enc, clip.x[:, -1]))  # (n, 3, res, res)
    x = clip.x[:, -1]
    t = cfg.tile
    out = {}
    e = ((xh - x) ** 2).view(n, 3, cfg.W, t, cfg.W, t).mean((1, 3, 5)).flatten(1)  # per-tile MSE (n, tiles)
    cls = clip.f["tile_class"][:, -1]
    for name, ids in REC_GROUPS.items():
        m = torch.isin(cls, torch.tensor(list(ids), device=cls.device))
        out[f"rec_{name}_{tag}"] = float(e[m].mean()) if m.any() else float("nan")
    if cfg.agent_shape == "arrow":  # match the decoded agent tile to the four arrow silhouettes
        xy = ((clip.f["slow"][:, -1, -2:] + 1) / 2 * torch.tensor([cfg.W - 1, cfg.H - 1], device=x.device)).round().long()
        lum = xh.mean(1).view(n, cfg.W, t, cfg.W, t)[torch.arange(n), xy[:, 1], :, xy[:, 0]]  # (n, t, t)
        tmpl = env.arrow_mask[:, :t, :t].float()
        lum, tmpl = lum - lum.mean((1, 2), keepdim=True), tmpl - tmpl.mean((1, 2), keepdim=True)
        pred = torch.einsum("nij,kij->nk", lum, tmpl).argmax(1)
        out[f"face_acc_{tag}"] = (pred == clip.f["fast"][:, -1, :4].argmax(1)).float().mean().item()
    return out


@torch.no_grad()
def latent_sensitivity(c, enc, tag: str, n: int = 512) -> dict:
    """How far the latent moves when only one thing changes (facing, hand, position, twinkle), per unit of pixel distance, relative to
    two unrelated frames. About 1: as sensitive as to a random change; far below 1: the latent barely sees it."""
    v = c.env.variants(n)
    z = {k: encode(enc, x) for k, (x, _) in v.items()}
    xb, zb = v["base"][0].flatten(1), z["base"]
    ref = (zb - zb.roll(1, 0)).norm(dim=1) / (xb - xb.roll(1, 0)).norm(dim=1)
    out = {}
    for k in ("facing", "hand", "position", "twinkle"):
        m = v[k][1]
        r = (z[k] - zb).norm(dim=1) / (v[k][0].flatten(1) - xb).norm(dim=1).clamp_min(1e-9)
        out[f"lat_sens_{k}_{tag}"] = float(r[m].mean() / ref.mean()) if m.any() else float("nan")
    return out


def _mlp_fit(z, y, hidden, steps, n_tr):
    net = nn.Sequential(nn.Linear(z.shape[1], hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, y.shape[1])).to(z.device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(steps):
        i = torch.randint(0, n_tr, (512,), device=z.device)
        loss = (net(z[i]) - y[i]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(z[n_tr:])


def probe_size_curve(c, enc, tag: str, parts=("facing", "mob", "hand"), n_tr: int = 6000, n_te: int = 2000) -> dict:
    """The same parts read by a ridge, a small MLP and a large MLP: information that is present but hard to read gains with size."""
    env = c.env
    z, y, _ = collect_last(enc, env, n_tr + n_te)
    mu, sd = z[:n_tr].mean(0), z[:n_tr].std(0).clamp_min(1e-6)
    z = (z - mu) / sd
    cols, P = env.group_cols(), env.parts()
    sls = {p: slice(cols[P[p][0]].start + P[p][1].start, cols[P[p][0]].start + P[p][1].stop) for p in parts}
    Y = torch.cat([y[:, s] for s in sls.values()], 1)
    W = fit_ridge(z[:n_tr], Y[:n_tr])
    preds = {"lin": z[n_tr:] @ W[0] + W[1], "s": _mlp_fit(z, Y, 64, 200, n_tr), "l": _mlp_fit(z, Y, 256, 1000, n_tr)}
    out, i = {}, 0
    for p, s in sls.items():
        d = s.stop - s.start
        yt = Y[n_tr:, i:i + d]
        for k, ph in preds.items():
            out[f"pc_{tag}_{p}_{k}"] = 1 - ((yt - ph[:, i:i + d]) ** 2).sum().item() / ((yt - yt.mean(0)) ** 2).sum().item()
        i += d
    return out


def heavy(c, sh_dom, sh_top) -> dict:
    """Everything that costs a few seconds: forces, attribution, retention fit, the decoder's and the encoders' view of the facing."""
    out = forces(c, sh_dom, sh_top)
    if c.shadow is not None and c.dec is not None:
        out.update(decoder_view(c, c.shadow, c.dec, "sh"))
        out.update(latent_sensitivity(c, c.shadow, "sh"))
        out.update(probe_size_curve(c, c.shadow, "sh"))
    if c.ae_dec is not None:  # autoencoder arm: the live encoder is the autoencoder
        out.update(decoder_view(c, c.enc, c.ae_dec, "live"))
    out.update(latent_sensitivity(c, c.enc, "live"))
    out.update(probe_size_curve(c, c.enc, "live"))
    return out
