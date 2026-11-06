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
    both       g'_ctx (projected)   g_open           full method
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

from .env import GROUPS, ClipEnv, EnvCfg
from .gate import FloorGate, floor_probe
from .grads import adam_metric, assign, project, vjp
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
    D: int = 16  # latent size
    steps: int = 20_000
    batch: int = 256  # clips per step
    lr: float = 1e-3
    lam: float = 0.1  # SIGReg weight
    n_slices: int = 256
    opt: str = "adam"  # adam | sgd
    precond: bool = True  # with adam: measure and project the leak in Adam's diagonal metric
    alpha: float = 1.0  # projection strength
    tau_lag: float = 0.99  # lagged predictor P_lag = EMA(P)
    tau_ema: float = 0.996  # target encoder of the ema arm
    # gate (section 4)
    window: int = 200  # W
    eps_g: float = 0.02  # relative-gap threshold
    eps_f: float = 0.25  # excess-over-floor threshold
    delta: float = 0.01  # floor offset: e = (L - L*) / (L* + delta)
    min_dwell: int = 500
    soft: bool = False  # soft open-path weights 1 - clip(gap / gap_ref)
    basis: str = "dim"  # dim | resid (eigenbasis of the floor probe's residual)
    # floor (section 3)
    floor_start: int = 500
    floor_every: int = 1000  # N
    floor_steps: int = 1500  # B
    floor_pool: int = 16384
    floor_keep: int = 3  # L* = min over the last floor_keep refreshes ...
    floor_margin: float = 0.1  # ... times (1 - margin)
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
    Opt = (lambda ps: torch.optim.Adam(ps, cfg.lr)) if cfg.opt == "adam" else \
        (lambda ps: torch.optim.SGD(ps, cfg.lr))
    opt_e, opt_p = Opt(params), Opt(p_params)
    gate = FloorGate(D, dev, cfg.window, cfg.eps_g, cfg.eps_f, cfg.delta, cfg.min_dwell,
                     cfg.floor_keep, cfg.floor_margin, cfg.soft)
    acc, rows, dirs, floors = defaultdict(list), [], defaultdict(list), []
    has_tgt_path = tgt_mode != "ema" and ctx_mode != "none"
    t0 = time.time()

    for step in range(cfg.steps):
        if _is_floor_step(cfg, step):
            Ls, Q = floor_probe(enc, tgt_enc, env, cfg.floor_steps, cfg.floor_pool,
                                basis=cfg.basis)
            gate.set_floor(Ls, None if cfg.basis == "dim" else Q)
            floors.append((step, Ls.cpu()))
        diag = step % cfg.diag_every == 0
        clip = env.sample(N)
        xc, xt = clip.x[:, :-1], clip.x[:, -1]
        zc, zt = enc(xc), enc(xt)  # (N, T-1, D), (N, D); separate passes keep the paths apart
        if enc_ema is not None:
            with torch.no_grad():
                zt_d = enc_ema(xt)
        else:
            zt_d = zt.detach()
        p = P(zc.detach() if ctx_mode == "none" else zc)
        L = (p - zt_d).pow(2).mean()  # (1/D) sum_j L_j: LeWM's MSE scale
        with torch.no_grad():
            gate.update(gate.per_dir(p - zt_d), gate.per_dir(P_lag(zc.detach()) - zt_d))
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
        elif tgt_mode == "open":
            z_open = z_open + d_tgt(gate.w_open())

        need_S = has_tgt_path and (ctx_mode == "proj" or diag)
        g_ctx_applied = None
        if ctx_mode == "none":
            g = vjp([zc, zt], [ds_dzc, z_open], params)
        elif not need_S:
            g = vjp([zc, zt], [dL_dzc + ds_dzc, z_open], params)
        else:
            metric = adam_metric(opt_e, params) if cfg.opt == "adam" and cfg.precond else None
            g_ctx = vjp(zc, dL_dzc, params)
            g_S = vjp(zt, d_tgt(gate.w_S()), params)
            alpha = cfg.alpha if ctx_mode == "proj" else 0.0
            g_ctx_applied, ell = project(g_ctx, g_S, alpha, metric)
            g = g_ctx_applied + vjp([zc, zt], [ds_dzc, z_open], params)
            if diag:
                m = 1.0 if metric is None else metric
                nS, nc = torch.dot(g_S, m * g_S).sqrt(), torch.dot(g_ctx, m * g_ctx).sqrt()
                acc["leak_pos"].append(float(ell > 0))
                acc["leak_cos"].append(ell / (nS * nc).clamp_min(1e-30).item())
                acc["proj_ratio"].append((g_ctx_applied.norm() / g_ctx.norm().clamp_min(1e-30)
                                          ).item())
                # per-direction first-order leak of the applied context step, via one JVP
                tangent = -cfg.lr * (metric if metric is not None else 1.0) * g_ctx_applied
                tang, i = [], 0
                for prm in params:
                    tang.append(tangent[i:i + prm.numel()].view_as(prm))
                    i += prm.numel()
                _, dz = jvp(lambda *ps: functional_call(enc, dict(zip(names, ps, strict=True)),
                                                        (xt,)),
                            tuple(prm.detach() for prm in params), tuple(tang))
                leak = (2 * (p.detach() - zt_d) @ Qb * (dz @ Qb)).mean(0)
                acc["leak_jvp"].append(leak.cpu())

        assign(params, g)
        for prm, gp in zip(p_params, gP, strict=True):
            prm.grad = gp
        opt_e.step()
        opt_p.step()
        ema_(P_lag, P, cfg.tau_lag)
        if enc_ema is not None:
            ema_(enc_ema, enc, cfg.tau_ema)

        acc["loss"].append(L.item())
        acc["sigreg"].append(sig.item() / cfg.lam)
        if diag:  # exact check: l_real = L(theta', z_t(theta)) - L(theta', z_t(theta'))
            with torch.no_grad():
                p2 = P(enc(xc))
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
def run_gt(env_cfg: EnvCfg = EnvCfg(), steps=5000, eval_every=100, seed=0) -> pd.DataFrame:
    """Learnability on ground truth: a predictor from the true context factors to the true
    target factors. Held-out R^2 per group over steps (slow fast, hard slower, noise ~0)."""
    dev = device()
    torch.manual_seed(seed)
    env = ClipEnv(env_cfg, dev)
    cols = env.group_cols()

    def xy(clip):
        x = torch.cat([torch.cat([clip.f[g][:, t] for g in GROUPS], -1)
                       for t in range(env_cfg.T - 1)], -1)
        return x, env.factors(clip)

    d_in = sum(env.dims.values())
    net = torch.nn.Sequential(torch.nn.Linear(d_in * (env_cfg.T - 1), 256), torch.nn.GELU(),
                              torch.nn.Linear(256, 256), torch.nn.GELU(),
                              torch.nn.Linear(256, d_in)).to(dev)
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
def premise(env_cfg: EnvCfg = EnvCfg(), D: int = 16, n: int = 4096) -> pd.Series:
    """SIGReg of hand-made latents: can slow content alone satisfy SIGReg at latent size D?

    Each candidate fills D dims with whitened factors (zero-padded if the group is too small).
    'gaussian' is the N(0, I) reference. Slow-only close to gaussian = the shortcut exists.
    """
    env = ClipEnv(env_cfg, device())
    clip = env.sample(n)

    def lat(*groups):
        z = torch.cat([clip.f[g][:, -1] for g in groups], -1)
        z = (z - z.mean(0)) / z.std(0)
        if z.shape[1] < D:
            z = torch.cat([z, torch.zeros(n, D - z.shape[1], device=z.device)], 1)
        return z[None, :, :D]

    torch.manual_seed(0)
    cand = {"gaussian": torch.randn(1, n, D, device=env.dev), "slow": lat("slow"),
            "fast+hard": lat("fast", "hard"), "noise": lat("noise"),
            "slow+fast+hard": lat("fast", "hard", "slow")}
    return pd.Series({k: sigreg(z, 1024).item() for k, z in cand.items()}, name="SIGReg")
