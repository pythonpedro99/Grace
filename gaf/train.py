"""One training run per arm, plus the ground-truth learnability and SIGReg premise checks.

Encoder gradient of every arm (one-pager section 7), all built from the same pieces:

    g_ctx    d/dtheta L through z_c (z_t stop-gradded)           context path
    g_tgtS   d/dtheta sum_{j in S} L_j through z_t (p detached)  measuring stick, never applied
    g_open   d/dtheta sum_{j not in S} L_j through z_t           open target path
    g_SIG    d/dtheta lambda SIGReg(z)                           never projected

    arm        context path         target path      what it is
    lewm       g_ctx                all directions   LeWM: full gradient through both paths
    sg         g_ctx                none             stop-grad target; the leak is unchecked
    ema        g_ctx                EMA encoder      BYOL/I-JEPA style target
    detached   none                 none             encoder sees only SIGReg: noise floors
    target     g_ctx                g_open           fix on the target path only
    context    g'_ctx (projected)   none             fix on the context path only
    both       projected            g_open           full method (g_open projected too)
    block      masked to open dirs  g_open           loss gradient to the encoder only in the open
                                                     directions, through both paths (no projection)

The predictor sees the context latents and the actions. The encoder steps with SplitAdamW
(grads.py): exactly AdamW, but the projection acts on the context side of the update that is
actually applied (momentum and weight decay included), in Adam's metric.
"""
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, jvp
from tqdm.auto import tqdm

from .env import GROUPS, ClipEnv, EnvCfg
from .env_grid import GridCfg, GridEnv
from .gate import FloorGate, floor_probe
from .grads import SplitAdamW, vjp
from .models import Decoder, Encoder, Predictor, ema_, frozen_copy, sigreg
from .probes import mlp_probe, parts_probe, probe_suite, r2
from .retain import RefMap

ARMS = {  # arm: (context path, target path)
    "lewm": ("raw", "all"),
    "sg": ("raw", "none"),
    "ema": ("raw", "ema"),
    "detached": ("none", "none"),
    "target": ("raw", "open"),
    "context": ("proj", "none"),
    "both": ("proj", "open"),
    "block": ("mask", "open"),
    "retain": ("raw", "none"),  # sg + reference-gated retention
    "retain_open": ("raw", "open"),  # + the floor-gated open target path (noise removal)
    "ae": ("none", "none"),  # an autoencoder: reconstruction + SIGReg, no prediction loss (B's shadow encoder on its own)
}
RETAIN_ARMS = ("retain", "retain_open")
AE_ARMS = ("ae",)


@dataclass(frozen=True)
class Cfg:
    arm: str = "sg"
    seed: int = 0
    env: EnvCfg = field(default_factory=EnvCfg)
    D: int = 32  # latent size (n_slow = 16: room for every group; D = n_slow is the tight regime)
    steps: int = 20_000
    batch: int = 256  # clips per step
    lr: float = 1e-3
    lam: float = 0.1  # SIGReg weight
    n_slices: int = 256
    opt: str = "adamw"  # adamw | sgd (encoder: SplitAdamW, see grads.py)
    wd: float = 0.01  # decoupled weight decay (encoder and predictor)
    alpha: float = 1.0  # projection strength
    proj_on: str = "update"  # update: the applied AdamW step | grad: the raw gradient (sketch)
    proj_open: bool = True  # also project the open target path (it shares weights with S)
    tau_lag: float = 0.99  # lagged predictor P_lag = EMA(P)
    tau_ema: float = 0.996  # target encoder of the ema arm
    # gate (section 4)
    window: int = 200  # W
    eps_g: float = 0.02  # relative-gap threshold
    eps_f: float = 0.25  # excess-over-floor threshold
    delta: float = 0.01  # floor offset: e = (L - L*) / (L* + delta)
    min_dwell: int = 500
    soft: bool = False  # soft open-path weights 1 - clip(gap / gap_ref)
    basis: str = "excess"  # excess | resid | dim, see gate.floor_probe
    # floor (section 3)
    floor_start: int = 500
    floor_every: int = 1000  # N
    floor_steps: int = 1500  # B
    floor_pool: int = 16384
    floor_keep: int = 3  # L* = min over the last floor_keep refreshes ...
    floor_margin: float = 0.1  # ... times (1 - margin)
    floor_warm: bool = True  # keep training the same probe across refreshes (budget accumulates)
    # logging
    eval_every: int = 500
    diag_every: int = 50
    probe_n: int = 8192  # held-out clips per eval (ridge fit on as many)
    mlp_n: int = 20000  # final MLP probe: training clips
    mlp_steps: int = 1000
    tag: str = ""  # free text, e.g. the name of an ablation
    # added later: part of the run id only when set, so earlier run ids stay valid
    head_start: int = 0  # steps in which the encoder gets only SIGReg while the predictor trains
    early_eval: tuple = ()  # extra probe evals at these steps (before the update), in early.csv
    # reference-gated retention (arms retain, retain_open; docs/reference_gated_retention.md)
    shadow_rec: float = 1.0  # reconstruction weight of the shadow encoder (0: SIGReg only)
    ret_lam: float = 1.0  # weight of the retention loss on the live encoder
    ret_every: int = 50  # steps between refreshes of the reference basis and weights
    ret_tau: float = 0.995  # EMA of the reference statistics
    ret_lag: float = 0.999  # lag of the shadow predictor's copy that defines "still learning"
    ret_eps_g: float = 0.02  # relative improvement above which a direction counts as unsettled
    ret_eps_f: float = 0.1  # excess of the live predictor over the floor that keeps retention
    # shadow reconstructs frame differences x_t - x_{t-1} from the pair (z_{t-1}, z_t) instead of
    # the frames: static content (the grid world's map) cancels, so what changes (facing, hand,
    # agent, mob) carries much more of the loss. Differences have ~1/3.6 of the frame MSE in the
    # grid world: raise shadow_rec to compensate. Off: plain frame reconstruction.
    rec_diff: bool = False
    # SIGReg on the latent differences z_t - z_{t-1} (every consecutive pair of frames), weight lam_diff: static content has a zero
    # difference and cannot satisfy it, changing content can. With lam = 0 the encoder sees only this term (a diagnostic).
    lam_diff: float = 0.0
    # class-balanced reconstruction (arm ae and B's shadow; grid world only): each tile's squared error is weighted by its
    # ground-truth tile class's share of the pixel variance to the power -rec_class_power (tempered, capped at 30x, mean 1).
    # 0: off, plain pixel MSE.
    rec_class_power: float = 0.0
    # label-free version: instead of the ground-truth tile classes, k-means clusters (this many) of the tile pixels, fitted once on a
    # sample of frames, and position types (rec_pos_types, by change rate) define the classes (needs rec_class_power > 0).
    # 0: ground-truth classes.
    rec_clusters: int = 0
    rec_pos_types: int = 6  # position types of the cluster classes (1: appearance only)
    # grid world: at every eval, also the standardised MLP probe per named part (live encoder, and the shadow of retention arms),
    # trained on this many clips (0: off); logged as ev_pm_<part>, ev_sh_pm_<part>. metrics.csv is then rewritten at every eval.
    eval_parts: int = 0
    eval_parts_steps: int = 400  # training steps of that probe (the final probe uses 2 * mlp_steps on mlp_n clips: stronger)
    # early stopping once settled (needs eval_parts; `steps` is then only the cap): at an eval, every ev_pm_* / ev_sh_pm_* readout changed
    # by less than settle_tol_change between the eval `settle_window` steps back and now and swung by less than settle_tol_swing inside
    # that window, at settle_patience evals in a row, and at least settle_min_steps are done. 0: off.
    # retention floor for content that is hard to predict but not noise (RefMap): directions whose shadow coordinate has a lag-1
    # autocorrelation of ret_ac (full effect; 0: off) between consecutive frames keep a retention weight of at least ret_w_min
    ret_ac: float = 0.0
    ret_w_min: float = 0.3
    ret_floor_all: bool = False  # the unconditional floor: every direction keeps a retention weight of at least ret_w_min
    settle_window: int = 0
    settle_min_steps: int = 15_000
    settle_patience: int = 2
    settle_tol_change: float = 0.05
    settle_tol_swing: float = 0.1

    @property
    def run_id(self) -> str:
        d = asdict(self)
        for k, default in _LATE_FIELDS.items():
            if tuple(d[k]) == default if isinstance(d[k], (list, tuple)) else d[k] == default:
                del d[k]
        for k, default in _LATE_ENV_FIELDS.items():
            if d["env"].get(k) == default:
                d["env"].pop(k, None)
        h = sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.arm}_s{self.seed}_{h}"


_LATE_FIELDS = {"head_start": 0, "early_eval": (), "shadow_rec": 1.0, "ret_lam": 1.0,
                "ret_every": 50, "ret_tau": 0.995, "ret_lag": 0.999, "ret_eps_g": 0.02,
                "ret_eps_f": 0.1, "rec_diff": False, "lam_diff": 0.0,
                "rec_class_power": 0.0, "rec_clusters": 0, "rec_pos_types": 6, "eval_parts": 0, "eval_parts_steps": 400, "ret_ac": 0.0, "ret_w_min": 0.3, "ret_floor_all": False, "settle_window": 0, "settle_min_steps": 15_000,
                "settle_patience": 2, "settle_tol_change": 0.05, "settle_tol_swing": 0.1}
_LATE_ENV_FIELDS = {"swap_colors": False, "hand_tiles": 1, "compass_tiles": 0, "hand_p": 1.0,
                    "hand_contrast": 1.0, "agent_shape": "marker", "mob_mode": "jump", "n_mobs": 1, "mob_chase": 0.7}


def make_env(env_cfg, dev):
    """The environment of a config: GridEnv for a GridCfg, ClipEnv for an EnvCfg."""
    return GridEnv(env_cfg, dev) if isinstance(env_cfg, GridCfg) else ClipEnv(env_cfg, dev)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_floor_step(c: Cfg, step: int) -> bool:
    return step == c.floor_start or (step > c.floor_start
                                     and (step - c.floor_start) % c.floor_every == 0)


def _tiles(x, c):
    """(n, T, 3, res, res) -> (n, T, tiles, 3*tile*tile): every tile as a row, in row-major tile order."""
    n = x.shape[0]
    return x.view(n, c.T, 3, c.W, c.tile, c.W, c.tile).permute(0, 1, 3, 5, 2, 4, 6).reshape(n, c.T, c.W * c.W, -1)


def _weights_from(cls, dev2, power, cap, k):
    """Weights per class from a tile -> class map `cls` and each tile's variance contribution `dev2` (flat, same length)."""
    mass = torch.zeros(k, device=dev2.device).index_add_(0, cls, dev2)
    count = torch.bincount(cls, minlength=k).float()
    share = mass / mass.sum()
    present = count > 0
    w = torch.ones(k, device=dev2.device)
    w[present] = (share[present] / share[present].mean()).clamp_min(1e-12) ** (-power)
    w = w.clamp(1 / cap, cap)
    w = w / ((w * count).sum() / count.sum())
    return w, share


@torch.no_grad()
def class_weights(env, power: float, n: int = 2048, cap: float = 30.0):
    """Per tile class (f["tile_class"]): its share of the pixel variance to the power -`power`, capped, with a tile-count
    weighted mean of 1. Classes that do not occur keep weight 1. Returns (weights, shares)."""
    clip = env.sample(n)
    t = _tiles(clip.x, env.cfg)
    dev2 = ((t - t.mean((0, 1), keepdim=True)) ** 2).sum(-1).flatten()  # per tile, against the mean tile at its position
    cls = clip.f["tile_class"].flatten()
    return _weights_from(cls, dev2, power, cap, int(cls.max().item()) + 1)


@torch.no_grad()
def kmeans(x, k: int, iters: int = 25, seed: int = 0):
    """Lloyd's k-means with k-means++ seeding on rows of x (n, d). Returns the centroids (k, d)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    C = x[torch.randint(n, (1,), generator=g).item()][None]
    for _ in range(k - 1):  # next centre drawn proportional to the squared distance to the nearest one
        d2 = torch.cdist(x, C).min(1).values ** 2
        C = torch.cat([C, x[torch.multinomial((d2 + 1e-12).cpu(), 1, generator=g).item()][None]])
    for _ in range(iters):
        a = torch.cdist(x, C).argmin(1)
        S = torch.zeros_like(C).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k).float()[:, None]
        C = torch.where(cnt > 0, S / cnt.clamp_min(1), C)
    return C


@torch.no_grad()
def cluster_classes(env, k: int, power: float, n: int = 1024, cap: float = 30.0, n_fit: int = 60000, seed: int = 0,
                    k_pos: int = 6):
    """The label-free version of class_weights. A tile's class is a pair, both fitted on a sample of frames without labels:
    its appearance (one of k k-means clusters of the raw tile pixels) and its position type (one of k_pos k-means clusters of
    the tile positions, by how often the position changes between frames and its pixel spread). Pixels alone cannot find
    content defined by where it is (a hand tile looks like a map tile); the change rate does (hand 47%, map 3%).
    k_pos = 1: appearance only. Returns ((centroids, position type per tile position, k), weights, shares)."""
    clip = env.sample(n)
    t = _tiles(clip.x, env.cfg)
    flat = t.reshape(-1, t.shape[-1])
    sub = flat[torch.randperm(flat.shape[0], device=flat.device)[:n_fit]]
    C = kmeans(sub, k, seed=seed)
    ptype = torch.zeros(t.shape[2], dtype=torch.long, device=t.device)
    if k_pos > 1:
        change = (t[:, 1:] - t[:, :-1]).pow(2).sum(-1).gt(1e-6).float().mean((0, 1))
        feat = torch.stack([change, t.var((0, 1)).sum(-1).sqrt()], 1)
        feat = (feat - feat.mean(0)) / feat.std(0).clamp_min(1e-6)
        ptype = torch.cdist(feat, kmeans(feat, k_pos, seed=seed)).argmin(1)
    cl = (C, ptype, k)
    cls = assign_tiles(clip.x, cl, env.cfg)
    dev2 = ((t - t.mean((0, 1), keepdim=True)) ** 2).sum(-1).flatten()
    w, share = _weights_from(cls.flatten(), dev2, power, cap, k * k_pos)
    return cl, w, share


@torch.no_grad()
def assign_tiles(x, cl, c):
    """Class of every tile of x (n, T, 3, res, res): (n, T, tiles); cl as returned by cluster_classes."""
    C, ptype, k = cl
    t = _tiles(x, c)
    a = torch.cdist(t.reshape(-1, t.shape[-1]), C).argmin(1).view(t.shape[:-1])
    return a + k * ptype


def is_settled(rows, window, min_steps, tol_change=0.05, tol_swing=0.1) -> bool:
    """Whether the per-part probe curves (ev_pm_*, ev_sh_pm_* of the eval rows) have stopped moving: over the last `window` steps every
    readout changed by < tol_change and swung (max - min) by < tol_swing. Same rule as analysis.drift, on the eval-time probes."""
    last = rows[-1]["step"]
    keys = [k for k in rows[-1] if k.startswith(("ev_pm_", "ev_sh_pm_")) and not k.endswith(("rank", "visible"))]
    base = [r for r in rows if r["step"] <= last - window]
    if last < min_steps or not keys or not base:
        return False
    inside = [r for r in rows if r["step"] >= base[-1]["step"]]
    for k in keys:
        v = [r[k] for r in inside]
        if not np.isfinite(v).all() or abs(v[-1] - v[0]) >= tol_change or max(v) - min(v) >= tol_swing:
            return False
    return True


def rec_error(x_hat, x, cls=None, w=None, tile=None):
    """Mean squared reconstruction error per pixel and channel; with class weights w (per tile class), each tile's error is
    weighted by w[its class]. x, x_hat: (..., 3, res, res); cls: (..., tiles) with tiles = (res / tile)^2."""
    e = (x_hat - x) ** 2
    if w is None:
        return e.mean()
    side = x.shape[-1] // tile
    per_tile = e.view(*e.shape[:-3], 3, side, tile, side, tile).sum((-5, -3, -1)).flatten(-2)  # (..., tiles)
    return (per_tile * w[cls]).sum() / (cls.numel() * 3 * tile * tile)


def run(cfg: Cfg, out, verbose: bool = True, on_step=None, log=print) -> dict:
    """Train one arm. Writes <out>/<run_id>/{metrics.csv, dirs.npz, model.pt, summary.json}
    (and early.csv when cfg.early_eval is set).

    Finished runs (summary.json present) are skipped and their summary returned.
    on_step(steps_done, phase) is called every step (phase "train") and when a longer phase
    starts: "floor probe", "eval", "final probes", then "done". log(str) gets the eval lines.
    """
    assert cfg.arm in ARMS, f"unknown arm {cfg.arm}"
    odir = Path(out) / cfg.run_id
    if (odir / "summary.json").exists():
        return json.loads((odir / "summary.json").read_text())
    odir.mkdir(parents=True, exist_ok=True)
    dev = device()
    torch.manual_seed(cfg.seed)
    env = make_env(cfg.env, dev)
    T, D, N = cfg.env.T, cfg.D, cfg.batch
    ctx_mode, tgt_mode = ARMS[cfg.arm]

    enc, P = Encoder(D, cfg.env.res).to(dev), Predictor(D, T, env.act_dim).to(dev)
    P_lag = frozen_copy(P)
    enc_ema = frozen_copy(enc) if tgt_mode == "ema" else None
    tgt_enc = enc_ema or enc
    params, names = list(enc.parameters()), [n for n, _ in enc.named_parameters()]
    p_params = list(P.parameters())
    sgd = cfg.opt == "sgd"
    opt_e = SplitAdamW(params, cfg.lr, wd=cfg.wd, sgd=sgd)
    opt_p = (torch.optim.SGD(p_params, cfg.lr, weight_decay=cfg.wd) if sgd
             else torch.optim.AdamW(p_params, cfg.lr, weight_decay=cfg.wd))
    gate = FloorGate(D, dev, cfg.window, cfg.eps_g, cfg.eps_f, cfg.delta, cfg.min_dwell,
                     cfg.floor_keep, cfg.floor_margin, cfg.soft)
    use_ret = cfg.arm in RETAIN_ARMS
    if use_ret:  # shadow encoder (+ decoder), shadow predictor, retention head, reference map
        shadow, Qs = Encoder(D, cfg.env.res).to(dev), Predictor(D, T, env.act_dim).to(dev)
        Qs_lag = frozen_copy(Qs)
        dec = (Decoder(D * (2 if cfg.rec_diff else 1), cfg.env.res).to(dev)
               if cfg.shadow_rec > 0 else None)
        Rh = torch.nn.Linear(D, D).to(dev)
        s_params = [*shadow.parameters(), *Qs.parameters(), *(dec.parameters() if dec else [])]
        opt_s = torch.optim.AdamW(s_params, cfg.lr, weight_decay=cfg.wd)
        opt_r = torch.optim.AdamW(Rh.parameters(), cfg.lr, weight_decay=cfg.wd)
        ref = RefMap(D, dev, cfg.ret_tau, cfg.ret_every, cfg.ret_eps_g, cfg.ret_eps_f, ac_ref=cfg.ret_ac, w_min=cfg.ret_w_min,
                     floor_all=cfg.ret_floor_all)
    use_ae = cfg.arm in AE_ARMS
    if use_ae:  # the autoencoder's decoder (its encoder is `enc`)
        ae_dec = Decoder(D, cfg.env.res).to(dev)
        opt_d = torch.optim.AdamW(ae_dec.parameters(), cfg.lr, weight_decay=cfg.wd)
    cls_w, centroids = None, None
    if cfg.rec_class_power > 0:
        assert isinstance(cfg.env, GridCfg), "class-balanced reconstruction needs the grid world's frames"
        if cfg.rec_clusters > 0:  # label-free: clusters of the tile pixels stand in for the tile classes
            centroids, cls_w, _ = cluster_classes(env, cfg.rec_clusters, cfg.rec_class_power, seed=cfg.seed,
                                                  k_pos=cfg.rec_pos_types)
        else:
            cls_w, _ = class_weights(env, cfg.rec_class_power)

    def tile_cls(clip):  # the class of every tile of the batch, for rec_error
        return assign_tiles(clip.x, centroids, cfg.env) if centroids is not None else clip.f.get("tile_class")
    acc, rows, dirs, floors, probe = defaultdict(list), [], defaultdict(list), [], None
    early, early_at = [], set(cfg.early_eval)
    calm, stopped_at = 0, None  # consecutive settled evals; the step at which the run stopped itself
    t0 = time.time()

    for step in range(cfg.steps):
        if on_step:
            on_step(step, "train")
        # head start: the encoder sees only SIGReg, the predictor trains on detached latents
        ctx_m, tgt_m = ("none", "none") if step < cfg.head_start else (ctx_mode, tgt_mode)
        has_tgt_path = tgt_m != "ema" and ctx_m != "none"
        if step in early_at:
            if on_step:
                on_step(step, "eval")
            m0, _ = probe_suite(enc, P, env, gate.Q, cfg.probe_n, enc_tgt=enc_ema)
            early.append({"step": step, "n_S": int(gate.inS.sum()), **m0})
        if _is_floor_step(cfg, step):
            if on_step:
                on_step(step, "floor probe")
            Ls, Q, pr = floor_probe(enc, tgt_enc, env, cfg.floor_steps, cfg.floor_pool,
                                    lr=cfg.lr, wd=cfg.wd, basis=cfg.basis, probe=probe,
                                    pred=P)
            probe = pr if cfg.floor_warm else None
            gate.set_floor(Ls, None if cfg.basis == "dim" else Q)
            floors.append((step, Ls.cpu()))
        diag = step % cfg.diag_every == 0
        clip = env.sample(N)
        xc, xt, a = clip.x[:, :-1], clip.x[:, -1], clip.a
        zc, zt = enc(xc), enc(xt)  # (N, T-1, D), (N, D); separate passes keep the paths apart
        if enc_ema is not None:
            with torch.no_grad():
                zt_d = enc_ema(xt)
        else:
            zt_d = zt.detach()
        p = P(zc.detach() if ctx_m == "none" else zc, a)
        L = (p - zt_d).pow(2).mean()  # (1/D) sum_j L_j: LeWM's MSE scale
        with torch.no_grad():
            gate.update(gate.per_dir(p - zt_d), gate.per_dir(P_lag(zc.detach(), a) - zt_d))
        sig_raw = sigreg(torch.cat([zc.transpose(0, 1), zt[None]]), cfg.n_slices)
        sig = cfg.lam * sig_raw
        if cfg.lam_diff > 0:
            z_all = torch.cat([zc, zt[:, None]], 1)  # (N, T, D): every frame, context and target
            sig = sig + cfg.lam_diff * sigreg((z_all[:, 1:] - z_all[:, :-1]).transpose(0, 1), cfg.n_slices)
        enc_aux = sig  # what shapes the encoder besides the prediction paths: SIGReg (+ retention, + reconstruction)
        if use_ae:  # reconstruction of every frame from its latent; the encoder gets it through z_c and z_t
            rec = rec_error(ae_dec(torch.cat([zc, zt[:, None]], 1)), clip.x, tile_cls(clip), cls_w, cfg.env.tile
                            if cls_w is not None else None)
            enc_aux = enc_aux + cfg.shadow_rec * rec
            gD = torch.autograd.grad(cfg.shadow_rec * rec, list(ae_dec.parameters()), retain_graph=True)
            acc["ae_rec_mse"].append(rec.item())
        if use_ret:
            # shadow: SIGReg (+ reconstruction) only, so it keeps content; Qs gives the floor
            zs = shadow(clip.x)  # (N, T, D)
            zs_c, zs_t = zs[:, :-1].detach(), zs[:, -1].detach()
            qp = Qs(zs_c, a)
            with torch.no_grad():
                r, r_lag = qp - zs_t, Qs_lag(zs_c, a) - zs_t
            loss_s = ((qp - zs_t).pow(2).mean()
                      + cfg.lam * sigreg(zs.transpose(0, 1), cfg.n_slices))
            if dec is not None:
                if cfg.rec_diff:  # the change between consecutive frames, from the pair of latents
                    rec = (dec(torch.cat([zs[:, :-1], zs[:, 1:]], -1))
                           - (clip.x[:, 1:] - clip.x[:, :-1])).pow(2).mean()
                else:
                    rec = rec_error(dec(zs), clip.x, tile_cls(clip), cls_w,
                                    cfg.env.tile if cls_w is not None else None)
                loss_s = loss_s + cfg.shadow_rec * rec
                acc["shadow_rec_mse"].append(rec.item())  # the reconstruction term of shadow_loss
            opt_s.zero_grad(set_to_none=True)
            loss_s.backward()
            opt_s.step()
            ema_(Qs_lag, Qs, cfg.ret_lag)
            # retention: the live latent must keep the weighted part of the shadow latent readable
            y = (zs.detach() - ref.mu) @ ref.W  # whitened shadow coordinates, every frame
            e = Rh(torch.cat([zc, zt[:, None]], 1)) - y
            L_ret = ((e @ ref.M) * e).sum(-1).mean() / D
            enc_aux = sig + cfg.ret_lam * L_ret
            gR = torch.autograd.grad(cfg.ret_lam * L_ret, list(Rh.parameters()), retain_graph=True)
            with torch.no_grad():
                ref.update(zs.detach().flatten(0, 1), r, r_lag, Rh(p.detach()) - y[:, -1],
                           pairs=(zs[:, :-1].detach().flatten(0, 1), zs[:, 1:].detach().flatten(0, 1)) if cfg.ret_ac > 0 else None)
            acc["ret_loss"].append(L_ret.item())
            acc["shadow_loss"].append(loss_s.item())

        # derivatives w.r.t. the latents (cheap), then encoder VJPs per path
        ins = p_params + ([zc] if ctx_m != "none" else [])
        gs = torch.autograd.grad(L, ins, retain_graph=True)
        gP, dL_dzc = gs[:len(p_params)], (gs[-1] if ctx_m != "none" else None)
        if ctx_m == "mask":  # the encoder learns from the open directions only; P still from all
            L_enc = ((p - zt_d) @ gate.Q).pow(2).mul(gate.w_open()).mean()
            dL_dzc = torch.autograd.grad(L_enc, zc, retain_graph=True)[0]
        ds_dzc, ds_dzt = torch.autograd.grad(enc_aux, [zc, zt], retain_graph=True)
        Qb = gate.Q
        rt = (zt_d - p.detach()) @ Qb  # target residual in the gate basis

        def d_tgt(w):  # d/dz_t of mean over (n, j) of w_j * ((z_t - p) Q)_j^2
            return 2 * (rt * w) @ Qb.T / (N * D)

        z_open = ds_dzt
        if tgt_m == "all":
            z_open = z_open + d_tgt(torch.ones(D, device=dev))

        g_rest = vjp([zc, zt], [ds_dzc, z_open], params)  # SIGReg (+ target path of lewm)
        g_ctx = vjp(zc, dL_dzc, params) if ctx_m != "none" else torch.zeros_like(g_rest)
        if tgt_m == "open":  # its step moves z_t on S too (shared weights): project it as well
            g_open = vjp(zt, d_tgt(gate.w_open()), params)
            if ctx_m == "proj" and cfg.proj_open:
                g_ctx = g_ctx + g_open
            else:
                g_rest = g_rest + g_open
        need_S = has_tgt_path and (ctx_m == "proj" or diag)
        g_S = vjp(zt, d_tgt(gate.w_S()), params) if need_S else None
        old = [prm.detach().clone() for prm in params] if diag and has_tgt_path else None
        info = opt_e.step(g_ctx, g_rest, g_S, cfg.alpha if ctx_m == "proj" else 0.0,
                          cfg.proj_on)
        for prm, gp in zip(p_params, gP, strict=True):
            prm.grad = gp
        opt_p.step()
        if use_ae:
            for prm, gd in zip(ae_dec.parameters(), gD, strict=True):
                prm.grad = gd
            opt_d.step()
        if use_ret:
            for prm, gr in zip(Rh.parameters(), gR, strict=True):
                prm.grad = gr
            opt_r.step()
        ema_(P_lag, P, cfg.tau_lag)
        if enc_ema is not None:
            ema_(enc_ema, enc, cfg.tau_ema)

        if diag and has_tgt_path:
            u = info["u_ctx"]
            acc["leak_pos"].append(float(info["leak_pre"] > 0))
            acc["leak_cos"].append(info["leak_cos"])
            acc["leak_cos_post"].append(info["leak_post"] / (g_S.norm() * u.norm()
                                                             ).clamp_min(1e-30).item())
            acc["proj_ratio"].append(info["proj_ratio"])
            # per-direction first-order leak of the applied context step, via one JVP at theta
            tangent, tang, i = -cfg.lr * u, [], 0
            for prm in params:
                tang.append(tangent[i:i + prm.numel()].view_as(prm))
                i += prm.numel()
            _, dz = jvp(lambda *ps: functional_call(enc, dict(zip(names, ps, strict=True)), (xt,)),
                        tuple(old), tuple(tang))
            acc["leak_jvp"].append((2 * ((p.detach() - zt_d) @ Qb) * (dz @ Qb)).mean(0).cpu())
            # exact leak by source: move only the context part / only the rest (SIGReg, open
            # target path) of the applied step and measure how the target's error changes with
            # the predictor held fixed; positive = that part makes z_t easier to predict
            with torch.no_grad():
                for part, upd in (("ctx", u), ("rest", info["u_rest"])):
                    moved, off = {}, 0
                    for n_, p0 in zip(names, old, strict=True):
                        moved[n_] = p0 - cfg.lr * upd[off:off + p0.numel()].view_as(p0)
                        off += p0.numel()
                    zt_part = functional_call(enc, moved, (xt,))
                    lk = gate.per_dir(p.detach() - zt_d) - gate.per_dir(p.detach() - zt_part)
                    acc[f"lk_{part}_S"].append((lk * gate.w_S()).sum().item())
                    acc[f"lk_{part}_open"].append((lk * (1 - gate.w_S())).sum().item())

        acc["loss"].append(L.item())
        acc["sigreg"].append(sig_raw.item())  # the SIGReg of the frames, whatever the weights
        if diag:  # exact check: l_real = L(theta', z_t(theta)) - L(theta', z_t(theta'))
            with torch.no_grad():
                p2 = P(enc(xc), a)
                lreal = gate.per_dir(p2 - zt_d) - gate.per_dir(p2 - tgt_enc(xt))
            acc["lreal"].append(lreal.cpu())
            acc["lreal_S"].append((lreal * gate.w_S()).sum().item())
            acc["lreal_open"].append((lreal * (1 - gate.w_S())).sum().item())

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            if on_step:
                on_step(step + 1, "eval")
            m, content = probe_suite(enc, P, env, gate.Q, cfg.probe_n, enc_tgt=enc_ema)
            st = gate.state()
            if use_ret:  # what the shadow knows, and whether rho separates content without labels
                m_sh, c_sh = probe_suite(shadow, Qs, env, ref.basis(), cfg.probe_n)
                m.update({f"sh_{k}": v for k, v in m_sh.items()
                          if k.startswith(("r2_", "pred_r2_"))})
                c, rho, wt = c_sh.numpy(), ref.rho.cpu().numpy(), ref.w.cpu().numpy()
                dom, top = c.argmax(1), c.max(1)  # direction -> the group it carries most
                for gi, g in enumerate(GROUPS):  # mean rho and count of the directions of a group
                    mg = (dom == gi) & (top > 0.15)
                    m[f"ref_rho_{g}"] = float(rho[mg].mean()) if mg.any() else float("nan")
                    m[f"ref_n_{g}"] = int(mg.sum())
                    m[f"ref_w_{g}"] = float(wt[mg].mean()) if mg.any() else float("nan")  # retention weight on that group's directions
                noisy, signal = (dom == 3) & (top > 0.15), (dom != 3) & (top > 0.15)
                m["ref_auc"] = (float((rho[signal][:, None] > rho[noisy][None]).mean()
                                      + 0.5 * (rho[signal][:, None] == rho[noisy][None]).mean())
                                if noisy.any() and signal.any() else float("nan"))
                m.update(ref.summary())
            if cfg.eval_parts and hasattr(env, "parts"):
                kw = dict(n_train=cfg.eval_parts, n_test=2048, steps=cfg.eval_parts_steps)
                m.update(parts_probe(enc, env, prefix="ev_", **kw))
                if use_ret:
                    m.update(parts_probe(shadow, env, prefix="ev_sh_", **kw))
            row = {"step": step + 1, "n_S": int(st["inS"].sum()),
                   "gap_mean": st["gap"].mean().item(), "excess_med": st["excess"].median().item(),
                   **m}
            for k, v in acc.items():
                if k in ("lreal", "leak_jvp"):
                    dirs[k].append(torch.stack(v).mean(0))
                else:
                    row[k] = float(np.mean(v))
            for k in ("lreal", "leak_jvp"):
                if k not in acc:
                    dirs[k].append(torch.full((D,), float("nan")))
            for k, v in st.items():
                dirs[k].append(v)
            dirs["content"].append(content)
            rows.append(row)
            if cfg.eval_parts:  # a run watched while it trains (analysis.live_probes) needs the curve on disk
                pd.DataFrame(rows).to_csv(odir / "metrics.csv", index=False)
            acc.clear()
            if verbose:
                log(f"  [{cfg.arm} s{cfg.seed}] step {step + 1:>6}  loss {row['loss']:.4f}  "
                      f"|S| {row['n_S']:>2}/{D}  " + "  ".join(
                          f"{g} {row[f'r2_{g}']:.2f}" for g in GROUPS)
                      + f"  ({time.time() - t0:.0f}s)")
            if cfg.settle_window:
                calm = calm + 1 if is_settled(rows, cfg.settle_window, cfg.settle_min_steps, cfg.settle_tol_change,
                                              cfg.settle_tol_swing) else 0
                if calm >= cfg.settle_patience and step + 1 < cfg.steps:
                    stopped_at = step + 1
                    log(f"  [{cfg.arm} s{cfg.seed}] settled at step {stopped_at}: stopping (cap {cfg.steps})")
                    break

    if on_step:
        on_step(cfg.steps, "final probes")
    summary = {"run_id": cfg.run_id, "cfg": asdict(cfg), "seconds": time.time() - t0, "steps_run": stopped_at or cfg.steps,
               "final": rows[-1], **mlp_probe(enc, env, cfg.mlp_n, cfg.probe_n, cfg.mlp_steps)}
    if hasattr(env, "parts"):  # standardised MLP per named part, and the effective rank (grid world)
        kw = dict(n_train=cfg.mlp_n, n_test=cfg.probe_n, steps=2 * cfg.mlp_steps)
        summary.update(parts_probe(enc, env, **kw))
        if use_ret:
            summary.update(parts_probe(shadow, env, prefix="sh_", **kw))
    if use_ret:  # what the shadow holds nonlinearly: the ridge readout of the eval rows can miss it
        summary.update({f"sh_{k}": v for k, v in
                        mlp_probe(shadow, env, cfg.mlp_n, cfg.probe_n, cfg.mlp_steps).items()})
    pd.DataFrame(rows).to_csv(odir / "metrics.csv", index=False)
    if early:
        pd.DataFrame(early).to_csv(odir / "early.csv", index=False)
    np.savez(odir / "dirs.npz", step=np.array([r["step"] for r in rows]),
             floor_step=np.array([s for s, _ in floors]),
             floor=np.stack([f.numpy() for _, f in floors]) if floors else np.zeros((0, D)),
             **{k: torch.stack(v).float().numpy() for k, v in dirs.items()})
    torch.save({"enc": enc.state_dict(), "P": P.state_dict(),
                "enc_ema": enc_ema.state_dict() if enc_ema is not None else None,
                **({"shadow": shadow.state_dict(), "dec": dec.state_dict() if dec else None}
                   if use_ret else {})},
               odir / "model.pt")
    (odir / "summary.json").write_text(json.dumps(summary))
    if on_step:
        on_step(cfg.steps, "done")
    return summary


def load_model(cfg: Cfg, out):
    """(enc, P, target encoder) of a finished run, on the current device, eval mode."""
    dev = device()
    sd = torch.load(Path(out) / cfg.run_id / "model.pt", map_location=dev)
    act_dim = make_env(cfg.env, "cpu").act_dim
    enc, P = Encoder(cfg.D, cfg.env.res).to(dev), Predictor(cfg.D, cfg.env.T, act_dim).to(dev)
    enc.load_state_dict(sd["enc"])
    P.load_state_dict(sd["P"])
    tgt = enc
    if sd["enc_ema"] is not None:
        tgt = Encoder(cfg.D, cfg.env.res).to(dev)
        tgt.load_state_dict(sd["enc_ema"])
    return enc.eval(), P.eval(), tgt.eval()


# ---- checks that need no encoder --------------------------------------------------------------
def run_gt(env_cfg: EnvCfg = EnvCfg(), steps=5000, eval_every=100, seed=0,
           actions=True, desc: str | None = None) -> pd.DataFrame:
    """Learnability on ground truth: a predictor from the true context factors (and actions)
    to the true target factors. Held-out R^2 per group over steps (slow fast, hard slower,
    noise ~0). actions=False hides the actions: fast and hard should then become unpredictable,
    which confirms that they are action-conditioned. desc: label of a progress bar (None: none).
    Inputs also include the env's `gt_extra` factors (grid world: the materials around the agent,
    which the hand depends on but no group holds)."""
    dev = device()
    torch.manual_seed(seed)
    env = make_env(env_cfg, dev)
    cols = env.group_cols()

    def xy(clip):
        x = torch.cat([clip.f[g][:, :-1] for g in (*GROUPS, *env.gt_extra)], -1).flatten(1)
        a = clip.a.flatten(1) if actions else torch.zeros_like(clip.a.flatten(1))
        return torch.cat([x, a], 1), env.factors(clip)

    d_in = sum(env.dims.values())
    probe = env.sample(1, render=False)
    d_x = sum(probe.f[g].shape[-1] for g in (*GROUPS, *env.gt_extra))
    net = torch.nn.Sequential(
        torch.nn.Linear((d_x + env.act_dim) * (env_cfg.T - 1), 256), torch.nn.GELU(),
        torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, d_in)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt, yt = xy(env.sample(8192, render=False))
    rows = []
    for step in tqdm(range(steps), desc=desc, unit="step", disable=desc is None):
        x, y = xy(env.sample(512, render=False))
        loss = (net(x) - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % eval_every == 0:
            with torch.no_grad():
                yh = net(xt)
            rows.append({"step": step + 1, **{g: r2(yt[:, s], yh[:, s]) for g, s in cols.items()}})
    return pd.DataFrame(rows).set_index("step")


def capacity(env_cfg, Ds=(16, 24, 32, 48, 64), steps=4000, batch=256, lam=0.1, lr=1e-3,
             seed=0, groups=("slow", "fast", "hard"), eval_every=500, n_eval=4096,
             desc: bool = True) -> pd.DataFrame:
    """What latent size D can hold the content and support its prediction? An upper bound on what
    any training of that size can reach: the real encoder on rendered frames, *supervised* with the
    ground truth, plus SIGReg (weight lam) so the latent is shaped as in the real runs.

    The targets are standardized per column and the supervised errors are summed over columns, so
    the supervision outweighs SIGReg and SIGReg only shapes the latent. (A mean over columns lets
    SIGReg push out content it finds expensive, such as one-hot factors, at every D: that measures
    the SIGReg trade-off, not the capacity.) R^2 is computed in the original units.

    Per D: z = enc(x) for every frame; a linear head reads the current factors of `groups` from z
    (readout, what z holds) and the predictor P(z_c, a) with a linear head predicts the target
    frame's factors (prediction, what z supports). Held-out R^2 per group, every eval_every steps.
    The smallest D at which every group reaches its plateau is the capacity the content needs; the
    real runs also spend dimensions on unlabelled content (the map, noise), so leave room above it.
    """
    dev = device()
    rows = []
    for D in Ds:
        torch.manual_seed(seed)
        env = make_env(env_cfg, dev)
        cols = env.group_cols()
        sl = [cols[g] for g in groups]

        def y_of(clip, t, sl=sl):
            f = env.factors(clip, t)
            return torch.cat([f[:, s] for s in sl], -1)

        d_y = sum(s.stop - s.start for s in sl)
        with torch.no_grad():
            ev = env.sample(n_eval)
            y_ev, y_now = y_of(ev, -1), y_of(ev, -2)
            mu, sd = y_ev.mean(0), y_ev.std(0).clamp_min(1e-6)  # standardization of the targets
        enc, P = Encoder(D, env_cfg.res).to(dev), Predictor(D, env_cfg.T, env.act_dim).to(dev)
        read, pred = torch.nn.Linear(D, d_y).to(dev), torch.nn.Linear(D, d_y).to(dev)
        params = [*enc.parameters(), *P.parameters(), *read.parameters(), *pred.parameters()]
        opt = torch.optim.AdamW(params, lr, weight_decay=0.01)
        T = env_cfg.T
        for step in tqdm(range(1, steps + 1), desc=f"capacity D={D}", unit="step", disable=not desc):
            clip = env.sample(batch)
            z = enc(clip.x)  # (n, T, D)
            y_all = (torch.stack([y_of(clip, t) for t in range(T)], 1) - mu) / sd
            loss = ((read(z) - y_all).pow(2).sum(-1).mean()
                    + (pred(P(z[:, :-1], clip.a)) - y_all[:, -1]).pow(2).sum(-1).mean()
                    + lam * sigreg(z.transpose(0, 1)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % eval_every == 0 or step == steps:
                with torch.no_grad():
                    ze = torch.cat([enc(ev.x[i:i + 1024]) for i in range(0, n_eval, 1024)])
                    yr, yp = read(ze[:, -2]) * sd + mu, pred(P(ze[:, :-1], ev.a)) * sd + mu
                    row = {"D": D, "step": step, "sigreg": sigreg(ze.transpose(0, 1), 1024).item()}
                    i = 0
                    for g, s in zip(groups, sl, strict=True):
                        j = i + s.stop - s.start
                        row[f"read_{g}"] = r2(y_now[:, i:j], yr[:, i:j])
                        row[f"pred_{g}"] = r2(y_ev[:, i:j], yp[:, i:j])
                        i = j
                rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def premise(env_cfg: EnvCfg = EnvCfg(), D: int = 32, n: int = 4096) -> pd.Series:
    """SIGReg of hand-made latents of size D: can slow content alone satisfy SIGReg?

    gaussian           N(0, I) reference (the floor of the statistic at this n)
    slow               the slow factors, zero-padded if n_slow < D (linear content only)
    slow+copies        slow factors plus Gaussianized nonlinear copies (pairwise products) in the
                       remaining dims: what an encoder can fill spare room with using slow content
                       only. Close to gaussian = the shortcut exists even with spare capacity.
    fast+hard          sprite factors, zero-padded
    """
    env = make_env(env_cfg, device())
    clip = env.sample(n)

    def std(z):
        return (z - z.mean(0)) / z.std(0)

    def pad(z):
        z = std(z)
        if z.shape[1] < D:
            z = torch.cat([z, torch.zeros(n, D - z.shape[1], device=z.device)], 1)
        return z[None, :, :D]

    s = clip.f["slow"][:, -1]
    i, j = torch.triu_indices(s.shape[1], s.shape[1], 1, device=s.device)
    prod = s[:, i] * s[:, j]
    rank = prod.argsort(0).argsort(0).float()
    copies = torch.special.ndtri((rank + 0.5) / n)[:, :max(0, D - s.shape[1])]
    torch.manual_seed(0)
    cand = {"gaussian": torch.randn(1, n, D, device=env.dev), "slow": pad(s),
            "slow+copies": pad(torch.cat([s, copies], 1)),
            "fast+hard": pad(torch.cat([clip.f["fast"][:, -1], clip.f["hard"][:, -1]], 1))}
    return pd.Series({k: sigreg(z, 1024).item() for k, z in cand.items()}, name="SIGReg")
