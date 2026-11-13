"""Decoder-based readouts of a run's checkpoint (grid world).

Probes under-read compressed latents: every probe read the facing at about 0.1 while the shadow's decoder rendered its orientation 95 % of the
time (docs/findings.md). These readouts decode instead of probing. Each score per tile class is the share of the error that "decode nothing"
(the average image at that tile position) makes, removed by the decoded frame: 0 = nothing decoded, 1 = perfect. The facing is the
orientation of the decoded agent tile matched to the four arrow silhouettes (chance 0.25); "move steps" are the clips whose last action
moved or turned the agent, where copying the last frame is mostly wrong.

    load(run_dir, step)       the checkpoint at `step` (default: the run's last): cfg, env and the networks (every readout takes `step`)
    dynamics(run_dir)         per system (live: encoder + predictor + its decoder; shadow: shadow + Qs + dec), the decoded next frame from
                              the real latent (ceiling), from the predictor (what the model foresees) and from the last frame (floor)
    decoder_strength(run_dir) a fresh class-weighted decoder trained on each frozen latent: what the latent holds, at decoder strength
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from . import mech
from .env_grid import GridCfg
from .models import Decoder, Encoder, make_predictor
from .train import Cfg, class_weights, device, make_env, rec_error

PARTS = ("agent", "hand", "mob", "inventory", "twinkle", "map")


def load(run_dir, step=None) -> SimpleNamespace:
    """cfg, env, device and the state dicts of ckpt_<step>.pt (default: the run's last step) of a grid-world run."""
    d = Path(run_dir)
    cd = json.loads((d / "summary.json").read_text())["cfg"] if (d / "summary.json").exists() else None
    if cd is None:
        raise FileNotFoundError(f"{d} has no summary.json (finalize it first: analysis.finalize)")
    cd["env"] = GridCfg(**cd["env"])
    c = Cfg(**cd)
    dev = device()
    sd = torch.load(d / f"ckpt_{step or c.steps}.pt", map_location=dev)
    return SimpleNamespace(c=c, env=make_env(c.env, dev), sd=sd, dev=dev)


def _net(ck, cls, key, *args):
    m = cls(*args).to(ck.dev)
    m.load_state_dict(ck.sd[key])
    return m.eval()


def systems(ck) -> dict:
    """name -> (encoder, predictor, decoder or None). live: the trained encoder and predictor (with the autoencoder's decoder for the
    arms ae and joint); shadow: the retention arms' shadow, its predictor Qs and its decoder."""
    c, a = ck.c, ck.env.act_dim
    out = {"live": (_net(ck, Encoder, "enc", c.D, c.env.res), _net(ck, lambda: make_predictor(c.pred_arch, c.D, c.env.T, a), "P"),
                    _net(ck, Decoder, "ae_dec", c.D, c.env.res) if "ae_dec" in ck.sd else None)}
    if "shadow" in ck.sd:
        out["shadow"] = (_net(ck, Encoder, "shadow", c.D, c.env.res), _net(ck, lambda: make_predictor(c.pred_arch, c.D, c.env.T, a), "Qs"),
                         _net(ck, Decoder, "dec", c.D, c.env.res))
    return out


def _scorer(ck, ev):
    """score(decoded frames of the clips' last frame) -> per-class explained share and facing accuracy (all clips, move steps)."""
    c, dev = ck.c, ck.dev
    n, t, W = ev.x.shape[0], c.env.tile, c.env.W
    x, cls = ev.x[:, -1], ev.f["tile_class"][:, -1]
    tiles = x.view(n, 3, W, t, W, t).permute(0, 2, 4, 1, 3, 5).reshape(n, W * W, -1)
    base_t = ((tiles - tiles.mean(0, keepdim=True)) ** 2).mean(-1)
    xy = ((ev.f["slow"][:, -1, -2:] + 1) / 2 * torch.tensor([W - 1, c.env.H - 1], device=dev)).round().long()
    face, move = ev.f["fast"][:, -1, :4].argmax(1), ev.a[:, -1].argmax(1) > 0
    tmpl = ck.env.arrow_mask[:, :t, :t].float()
    tmpl = tmpl - tmpl.mean((1, 2), keepdim=True)
    groups = {g: torch.isin(cls, torch.tensor(list(mech.REC_GROUPS.get(g, ())), device=dev)) for g in PARTS}

    @torch.no_grad()
    def score(xh):
        e = ((xh - x) ** 2).view(n, 3, W, t, W, t).mean((1, 3, 5)).flatten(1)
        row = {g: (1 - (e[m].mean() / base_t[m].mean()).item()) if m.any() and base_t[m].mean() > 0 else float("nan")
               for g, m in groups.items()}
        if c.env.agent_shape == "arrow":
            lum = xh.mean(1).view(n, W, t, W, t)[torch.arange(n), xy[:, 1], :, xy[:, 0]]
            hit = torch.einsum("nij,kij->nk", lum - lum.mean((1, 2), keepdim=True), tmpl).argmax(1) == face
            row["facing acc"], row["facing acc, move steps"] = hit.float().mean().item(), hit[move].float().mean().item()
            # position read apart from the orientation: the tile whose luminance matches any arrow silhouette best is the agent's tile
            lt = xh.mean(1).view(n, W, t, W, t).permute(0, 1, 3, 2, 4).reshape(n, W * W, t, t)
            corr = torch.einsum("nsij,kij->nsk", lt - lt.mean((2, 3), keepdim=True), tmpl).max(-1).values
            row["agent position acc"] = (corr.argmax(1) == xy[:, 1] * W + xy[:, 0]).float().mean().item()
        return row
    return score


@torch.no_grad()
def dynamics(run_dir, n_eval: int = 2048, step=None) -> pd.DataFrame:
    """For every system with a decoder of its own: the decoded next frame from the real latent, the predictor's output and the last
    frame, plus the latent prediction error over the latent's variance. Rows (system, version)."""
    ck = load(run_dir, step)
    ev = ck.env.sample(n_eval)
    score, rows = _scorer(ck, ev), {}
    for name, (E, Pr, dec) in systems(ck).items():
        zc, zt = E(ev.x[:, :-1]), E(ev.x[:, -1])
        zh = Pr(zc, ev.a)
        err = lambda z: (((z - zt) ** 2).mean(0) / zt.var(0).clamp_min(1e-9)).mean().item()  # noqa: E731
        for ver, z in (("actual next frame", zt), ("predicted", zh), ("copy last frame", zc[:, -1])):
            rows[(name, ver)] = {**(score(dec(z)) if dec is not None else {}), "latent err / var": err(z)}
    return pd.DataFrame(rows).T


def decoder_strength(run_dir, steps: int = 5000, n_eval: int = 2048, batch: int = 256, step=None, matched: bool = False) -> pd.DataFrame:
    """Freeze each encoder of the checkpoint, train a fresh decoder (class-weighted pixel loss, as the shadow) on its latent and decode
    held-out frames. Note: with the default class weights (cap 30) terrain and mob are down-weighted, so this under-reads those two;
    matched=True trains the decoder with the checkpoint's own weights (its rec_class_power and rec_class_cap), as the model was trained."""
    ck = load(run_dir, step)
    w, _ = class_weights(ck.env, ck.c.rec_class_power, cap=ck.c.rec_class_cap) if matched else class_weights(ck.env, 1.0)
    ev = ck.env.sample(n_eval)
    score, rows = _scorer(ck, ev), {}
    for name, (E, _, _) in systems(ck).items():
        for p in E.parameters():
            p.requires_grad_(False)
        dec = Decoder(ck.c.D, ck.c.env.res).to(ck.dev)
        opt = torch.optim.AdamW(dec.parameters(), 1e-3, weight_decay=0.01)
        for _ in range(steps):
            cl = ck.env.sample(batch)
            with torch.no_grad():
                z = E(cl.x[:, -1])
            loss = rec_error(dec(z), cl.x[:, -1], cl.f["tile_class"][:, -1], w, ck.c.env.tile)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            rows[name] = score(dec(E(ev.x[:, -1])))
    return pd.DataFrame(rows).T


@torch.no_grad()
def variance_shares(run_dir, n: int = 4096, step=None) -> dict:
    """The share of the live latent's total variance that a linear function of each part's true factors explains (a lower bound: nonlinear
    dependence is missed). Small = that content sits in the latent at low amplitude (invisible to the prediction loss and to distances)."""
    from .probes import fit_ridge
    ck = load(run_dir, step)
    env, E = ck.env, systems(ck)["live"][0]
    clip = env.sample(n)
    z, y = E(clip.x[:, -1]), env.factors(clip)
    tot = (z - z.mean(0)).pow(2).sum()
    cols, out = env.group_cols(), {}
    for part, (g, s) in env.parts().items():
        yp = y[:, cols[g].start + s.start: cols[g].start + s.stop]
        W, b = fit_ridge(yp, z)
        zh = yp @ W + b
        out[part] = ((zh - zh.mean(0)).pow(2).sum() / tot).item()
    return out


def probes(run_dir, n_train: int = 8000, n_test: int = 2048, steps: int = 800, step=None) -> dict:
    """MLP probe of every named part from the live latent (the final-probe style; a lower bound for content, see the decoder readouts)."""
    from .probes import parts_probe
    ck = load(run_dir, step)
    return parts_probe(systems(ck)["live"][0], ck.env, n_train=n_train, n_test=n_test, steps=steps)


def action_test(run_dir, n_eval: int = 2048, step=None, seed: int = 0) -> pd.DataFrame:
    """Does the predictor draw the facing from the ACTION or from the context latent? The predictor's output is decoded and the arrow's
    orientation is read at the tile where the arrow is drawn (no training: the model's own decoder). The predictor is fed altered inputs
    and the answer compared with (a) the true facing of the next frame and (b) the facing the FED last action implies (a move turns the
    agent: facing = action - 1). Rows: real inputs; the last action replaced by a random move; the context latents shuffled across clips
    (actions kept). Columns, on move steps: share equal to the true facing, share equal to the fed action's facing; no-op steps (where the
    facing must come from the context): share equal to the true facing."""
    import torch.nn.functional as Fn
    ck = load(run_dir, step)
    E, Pr, dec = systems(ck)["live"]
    env, dev, c = ck.env, ck.dev, ck.c
    t, W = c.env.tile, c.env.W
    tmpl = env.arrow_mask[:, :t, :t].float()
    tmpl = tmpl - tmpl.mean((1, 2), keepdim=True)
    g = torch.Generator(device="cpu").manual_seed(seed)
    ev = env.sample(n_eval)
    with torch.no_grad():
        zc = E(ev.x[:, :-1])
    face, last = ev.f["fast"][:, -1, :4].argmax(1), ev.a[:, -1].argmax(1)
    move = last > 0

    @torch.no_grad()
    def read(z, a):
        xh = dec(Pr(z, a))
        lt = xh.mean(1).view(n_eval, W, t, W, t).permute(0, 1, 3, 2, 4).reshape(n_eval, W * W, t, t)
        corr = torch.einsum("nsij,kij->nsk", lt - lt.mean((2, 3), keepdim=True), tmpl)
        tile = corr.max(-1).values.argmax(1)
        return corr[torch.arange(n_eval), tile].argmax(-1)

    def row(guess, fed_face):
        return {"vs true (move)": (guess == face)[move].float().mean().item(), "vs fed action (move)": (guess == fed_face)[move].float().mean().item(),
                "vs true (no-op)": (guess == face)[~move].float().mean().item()}

    rows = {"real inputs": row(read(zc, ev.a), last - 1)}
    fed = torch.randint(1, 5, (n_eval,), generator=g).to(dev)  # a random move, whatever the true action was
    a2 = ev.a.clone()
    a2[:, -1] = Fn.one_hot(fed, a2.shape[-1]).to(a2.dtype)
    rows["last action replaced"] = row(read(zc, a2), fed - 1)
    rows["context latents shuffled"] = row(read(zc[torch.randperm(n_eval, generator=g).to(dev)], ev.a), last - 1)
    return pd.DataFrame(rows).T
