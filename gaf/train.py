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

from .env import ACT_DIM, GROUPS, ClipEnv, EnvCfg
from .gate import FloorGate, floor_probe
from .grads import SplitAdamW, vjp
from .models import Encoder, Predictor, ema_, frozen_copy, sigreg
from .probes import mlp_probe, probe_suite, r2

ARMS = {  # arm: (context path, target path)
    "lewm": ("raw", "all"),
    "sg": ("raw", "none"),
    "ema": ("raw", "ema"),
    "detached": ("none", "none"),
    "target": ("raw", "open"),
    "context": ("proj", "none"),
    "both": ("proj", "open"),
}


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

    @property
    def run_id(self) -> str:
        h = sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.arm}_s{self.seed}_{h}"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_floor_step(c: Cfg, step: int) -> bool:
    return step == c.floor_start or (step > c.floor_start
                                     and (step - c.floor_start) % c.floor_every == 0)


def run(cfg: Cfg, out, verbose: bool = True) -> dict:
    """Train one arm. Writes <out>/<run_id>/{metrics.csv, dirs.npz, model.pt, summary.json}.

    Finished runs (summary.json present) are skipped and their summary returned.
    """
    assert cfg.arm in ARMS, f"unknown arm {cfg.arm}"
    odir = Path(out) / cfg.run_id
    if (odir / "summary.json").exists():
        return json.loads((odir / "summary.json").read_text())
    odir.mkdir(parents=True, exist_ok=True)
    dev = device()
    torch.manual_seed(cfg.seed)
    env = ClipEnv(cfg.env, dev)
    T, D, N = cfg.env.T, cfg.D, cfg.batch
    ctx_mode, tgt_mode = ARMS[cfg.arm]

    enc, P = Encoder(D, cfg.env.res).to(dev), Predictor(D, T).to(dev)
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
    acc, rows, dirs, floors, probe = defaultdict(list), [], defaultdict(list), [], None
    has_tgt_path = tgt_mode != "ema" and ctx_mode != "none"
    t0 = time.time()

    for step in range(cfg.steps):
        if _is_floor_step(cfg, step):
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
        p = P(zc.detach() if ctx_mode == "none" else zc, a)
        L = (p - zt_d).pow(2).mean()  # (1/D) sum_j L_j: LeWM's MSE scale
        with torch.no_grad():
            gate.update(gate.per_dir(p - zt_d), gate.per_dir(P_lag(zc.detach(), a) - zt_d))
        sig = cfg.lam * sigreg(torch.cat([zc.transpose(0, 1), zt[None]]), cfg.n_slices)

        # derivatives w.r.t. the latents (cheap), then encoder VJPs per path
        ins = p_params + ([zc] if ctx_mode != "none" else [])
        gs = torch.autograd.grad(L, ins, retain_graph=True)
        gP, dL_dzc = gs[:len(p_params)], (gs[-1] if ctx_mode != "none" else None)
        ds_dzc, ds_dzt = torch.autograd.grad(sig, [zc, zt], retain_graph=True)
        Qb = gate.Q
        rt = (zt_d - p.detach()) @ Qb  # target residual in the gate basis

        def d_tgt(w):  # d/dz_t of mean over (n, j) of w_j * ((z_t - p) Q)_j^2
            return 2 * (rt * w) @ Qb.T / (N * D)

        z_open = ds_dzt
        if tgt_mode == "all":
            z_open = z_open + d_tgt(torch.ones(D, device=dev))

        g_rest = vjp([zc, zt], [ds_dzc, z_open], params)  # SIGReg (+ target path of lewm)
        g_ctx = vjp(zc, dL_dzc, params) if ctx_mode != "none" else torch.zeros_like(g_rest)
        if tgt_mode == "open":  # its step moves z_t on S too (shared weights): project it as well
            g_open = vjp(zt, d_tgt(gate.w_open()), params)
            if ctx_mode == "proj" and cfg.proj_open:
                g_ctx = g_ctx + g_open
            else:
                g_rest = g_rest + g_open
        need_S = has_tgt_path and (ctx_mode == "proj" or diag)
        g_S = vjp(zt, d_tgt(gate.w_S()), params) if need_S else None
        old = [prm.detach().clone() for prm in params] if diag and has_tgt_path else None
        info = opt_e.step(g_ctx, g_rest, g_S, cfg.alpha if ctx_mode == "proj" else 0.0,
                          cfg.proj_on)
        for prm, gp in zip(p_params, gP, strict=True):
            prm.grad = gp
        opt_p.step()
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

        acc["loss"].append(L.item())
        acc["sigreg"].append(sig.item() / cfg.lam)
        if diag:  # exact check: l_real = L(theta', z_t(theta)) - L(theta', z_t(theta'))
            with torch.no_grad():
                p2 = P(enc(xc), a)
                lreal = gate.per_dir(p2 - zt_d) - gate.per_dir(p2 - tgt_enc(xt))
            acc["lreal"].append(lreal.cpu())
            acc["lreal_S"].append((lreal * gate.w_S()).sum().item())
            acc["lreal_open"].append((lreal * (1 - gate.w_S())).sum().item())

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            m, content = probe_suite(enc, P, env, gate.Q, cfg.probe_n, enc_tgt=enc_ema)
            st = gate.state()
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
            acc.clear()
            if verbose:
                print(f"  [{cfg.arm} s{cfg.seed}] step {step + 1:>6}  loss {row['loss']:.4f}  "
                      f"|S| {row['n_S']:>2}/{D}  " + "  ".join(
                          f"{g} {row[f'r2_{g}']:.2f}" for g in GROUPS)
                      + f"  ({time.time() - t0:.0f}s)", flush=True)

    summary = {"run_id": cfg.run_id, "cfg": asdict(cfg), "seconds": time.time() - t0,
               "final": rows[-1], **mlp_probe(enc, env, cfg.mlp_n, cfg.probe_n, cfg.mlp_steps)}
    pd.DataFrame(rows).to_csv(odir / "metrics.csv", index=False)
    np.savez(odir / "dirs.npz", step=np.array([r["step"] for r in rows]),
             floor_step=np.array([s for s, _ in floors]),
             floor=np.stack([f.numpy() for _, f in floors]) if floors else np.zeros((0, D)),
             **{k: torch.stack(v).float().numpy() for k, v in dirs.items()})
    torch.save({"enc": enc.state_dict(), "P": P.state_dict(),
                "enc_ema": enc_ema.state_dict() if enc_ema is not None else None},
               odir / "model.pt")
    (odir / "summary.json").write_text(json.dumps(summary))
    return summary


def load_model(cfg: Cfg, out):
    """(enc, P, target encoder) of a finished run, on the current device, eval mode."""
    dev = device()
    sd = torch.load(Path(out) / cfg.run_id / "model.pt", map_location=dev)
    enc, P = Encoder(cfg.D, cfg.env.res).to(dev), Predictor(cfg.D, cfg.env.T).to(dev)
    enc.load_state_dict(sd["enc"])
    P.load_state_dict(sd["P"])
    tgt = enc
    if sd["enc_ema"] is not None:
        tgt = Encoder(cfg.D, cfg.env.res).to(dev)
        tgt.load_state_dict(sd["enc_ema"])
    return enc.eval(), P.eval(), tgt.eval()


# ---- checks that need no encoder --------------------------------------------------------------
def run_gt(env_cfg: EnvCfg = EnvCfg(), steps=5000, eval_every=100, seed=0,
           actions=True) -> pd.DataFrame:
    """Learnability on ground truth: a predictor from the true context factors (and actions)
    to the true target factors. Held-out R^2 per group over steps (slow fast, hard slower,
    noise ~0). actions=False hides the actions: fast and hard should then become unpredictable,
    which confirms that they are action-conditioned."""
    dev = device()
    torch.manual_seed(seed)
    env = ClipEnv(env_cfg, dev)
    cols = env.group_cols()

    def xy(clip):
        x = torch.cat([clip.f[g][:, :-1] for g in GROUPS], -1).flatten(1)
        a = clip.a.flatten(1) if actions else torch.zeros_like(clip.a.flatten(1))
        return torch.cat([x, a], 1), env.factors(clip)

    d_in = sum(env.dims.values())
    net = torch.nn.Sequential(
        torch.nn.Linear((d_in + ACT_DIM) * (env_cfg.T - 1), 256), torch.nn.GELU(),
        torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, d_in)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt, yt = xy(env.sample(8192))
    rows = []
    for step in range(steps):
        x, y = xy(env.sample(512))
        loss = (net(x) - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % eval_every == 0:
            with torch.no_grad():
                yh = net(xt)
            rows.append({"step": step + 1, **{g: r2(yt[:, s], yh[:, s]) for g, s in cols.items()}})
    return pd.DataFrame(rows).set_index("step")


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
    env = ClipEnv(env_cfg, device())
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
