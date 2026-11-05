"""Training loops: `run` (encoder + predictor, one arm) and `run_gt` (Stage 0b, no encoder)."""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .env import GROUPS, N_ACT, Env
from .gates import Gater, Stats, slice_weights
from .models import Encoder, Ensemble, Predictor, ema_, frozen_copy, sigreg_slices, with_action
from .probes import calibrate_beta, evaluate, mlp_probe, r2

LR, MU = 1e-3, 0.01  # learning rate; B2 shrink weight


@dataclass(frozen=True)
class Cfg:
    arm: str = "A0"  # A0 A1 A2 M1a M1b M1c M1d M2 M3 C1 C2 C3
    bfac: str = "B0"  # B0 B1 B2 B-rand
    D: int = 16
    diff: int = 0
    lam: float = 0.1
    w_fast: float = 1.0
    seed: int = 0
    alpha: float = 0.0  # A1 target-path strength
    tau_ema: float = 0.996  # A2 target-encoder EMA
    beta: float = 1.0  # M2 gate scale
    heads: bool = False  # also train the ensemble heads (M2 always does); calibrates beta
    steps: int = 20_000
    batch: int = 512
    eval_every: int = 500
    warmup: int = 2000  # G = 0 for this many steps (C2 and all M1 arms)

    @property
    def run_id(self) -> str:
        return "_".join(f"{k}-{v}" for k, v in asdict(self).items())


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(cfg: Cfg, out: str | Path) -> None:
    """Train one arm; writes <run_id>.csv (metrics every `eval_every` steps) then <run_id>.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{out / cfg.run_id}"  # not with_suffix: run ids contain dots
    if Path(f"{stem}.json").exists():
        return
    dev, D, arm, n = device(), cfg.D, cfg.arm, cfg.batch
    env = Env(cfg.seed, cfg.diff, cfg.w_fast, dev)
    torch.manual_seed(cfg.seed)
    enc, P = Encoder(D).to(dev), Predictor(D, het=arm == "M3").to(dev)
    P_lag, weak = frozen_copy(P), nn.Linear(D + N_ACT, D).to(dev)
    enc_ema = frozen_copy(enc) if arm == "A2" else None
    heads = Ensemble(D).to(dev) if cfg.heads or arm == "M2" else None
    aux = nn.Linear(D, 14).to(dev) if arm == "C3" else None  # oracle: z -> groups 1-7
    stats, gater = Stats(D, dev), Gater(arm, D, dev, cfg.alpha, cfg.beta, cfg.warmup)

    others = [m for m in (enc, weak, aux, heads) if m is not None]
    opt = torch.optim.AdamW(
        [{"params": [p for m in others for p in m.parameters() if p.requires_grad], "lr": LR},
         {"params": P.parameters(), "lr": LR * (5 if arm == "C1" else 1)}],
        weight_decay=0, fused=dev == "cuda")
    half = cfg.steps // 2
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 if s < half else
                                              0.5 * (1 + math.cos(math.pi * (s - half)
                                                                  / (cfg.steps - half))))
    rows, G, t0 = [], torch.eye(D, device=dev), time.time()
    for step in range(cfg.steps):
        b = env.sample(n)
        z = enc(torch.cat([b.x, b.xn])).view(2, n, D)
        zt, zn = z
        mu, logv = P(zt, b.a)
        pw = weak(with_action(zt.detach(), b.a))
        ens = heads(zt.detach(), b.a) if heads is not None else None
        with torch.no_grad():
            r, r_lag = mu - zn, P_lag(zt, b.a)[0] - zn
            stats.update(step, r, zn, pw.detach())
            e = u = v = None
            if arm == "M2":
                e, u = ens.var(0).mean(0), ((ens - zn) ** 2).mean((0, 1))
            elif arm == "M3":
                v = logv.exp().mean(0)
            G = gater(step, r, r_lag, stats.U, e, u, v)
        if enc_ema is not None:
            with torch.no_grad():
                tgt = enc_ema(b.xn)
        else:
            zs = zn.detach()
            tgt = zs + (zn - zs) @ G  # forward value unchanged, gradient mapped through G

        W = F.normalize(torch.randn(D, 256, device=dev), dim=0)
        omega = slice_weights(W, stats.basis(cfg.bfac), stats.s)
        loss = ((mu - tgt) ** 2).mean() + cfg.lam * (omega * sigreg_slices(z, W)).mean()
        loss = loss + ((pw - zn.detach()) ** 2).mean()
        if cfg.bfac == "B2":
            loss = loss + MU * (stats.s * (z.reshape(-1, D) @ stats.V).var(0)).sum()
        if logv is not None:  # M3: variance head fits sg(mu) vs sg(z')
            sq = (zn.detach() - mu.detach()) ** 2
            loss = loss + 0.5 * (logv + sq * torch.exp(-logv)).mean()
        if heads is not None:
            loss = loss + heads.loss(ens, zn.detach())
        if aux is not None:
            loss = loss + F.mse_loss(aux(zt), b.gt[:, :14])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema_(P_lag, P, 0.99)
        if enc_ema is not None:
            ema_(enc_ema, enc, cfg.tau_ema)
        if (step + 1) % cfg.eval_every == 0:
            metrics = evaluate(enc, P, env, G, stats)
            rows.append({"step": step + 1, "loss": loss.item(), **metrics})

    summary = {"run_id": cfg.run_id, "cfg": asdict(cfg), "seconds": time.time() - t0,
               "final": rows[-1], **mlp_probe(enc, env)}
    if heads is not None:
        summary["beta_cal"] = calibrate_beta(enc, heads, env)
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)
    Path(f"{stem}.json").write_text(json.dumps(summary))


def run_gt(diff: int, seed: int, out: str | Path, steps=10_000, eval_every=50) -> None:
    """Stage 0b: predictor and weak (linear) predictor on the true state of groups 1-7."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gt_diff-{diff}_seed-{seed}.csv"
    if path.exists():
        return
    dev = device()
    env = Env(seed, diff, device=dev)
    torch.manual_seed(seed)
    P, weak = Predictor(14).to(dev), nn.Linear(14 + N_ACT, 14).to(dev)
    opt = torch.optim.AdamW([*P.parameters(), *weak.parameters()], lr=LR, weight_decay=0)
    rows = []
    for step in range(steps):
        b = env.sample(512)
        x, y = b.gt[:, :14], b.gtn[:, :14]
        loss = F.mse_loss(P(x, b.a)[0], y) + F.mse_loss(weak(with_action(x, b.a)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % eval_every == 0:
            with torch.no_grad():
                b = env.sample(8192)
                x, y = b.gt[:, :14], b.gtn[:, :14]
                p, w = P(x, b.a)[0], weak(with_action(x, b.a))
            row = {"step": step + 1}
            for k, s in enumerate(GROUPS[:7], 1):
                row[f"p_{k}"], row[f"w_{k}"] = r2(y[:, s], p[:, s]), r2(y[:, s], w[:, s])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
