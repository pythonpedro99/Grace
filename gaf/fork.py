"""Forks of a finished `joint` run: the fade and the anchor, from its checkpoint (no retraining of the warmup).

The joint run is the warmup: reconstruction (keeps the content available) + prediction through the encoder (shapes it for dynamics) + SIGReg.
A fork restarts from the checkpoint at the run's last step with a new schedule and runs a few thousand steps:

    continue   reconstruction stays at full weight (the control)
    fade       the reconstruction weight falls linearly to `r_floor` over `fade` steps, then stays there
    anchor     the same fade, plus prediction distillation: the live (encoder, predictor) is held to the predictions of a frozen snapshot
               of itself taken at the start of the fork, ||P(E(x_c), a) - P_snap(E_snap(x_c), a)||^2 on fresh clips, weight ramped 0 -> beta
               over the fade. It holds what the snapshot could predict (the learned part) and leaves what it could not (noise, unlearned
               content) free; SIGReg keeps the latent roughly unit-variance per dimension, so no extra whitening is applied
    freeze     encoder and decoder frozen, only the predictor trains (V-JEPA 2's order: dynamics on a frozen representation)

Decoded prediction (the predictor's output decoded against the real next frame, `||D(P(z_ctx, a)) - x_next||`, class-weighted):
    dp=None    the fork keeps the parent's `Cfg.dp_weight` (0 for a plain joint parent), constant
    dp=w       the fork introduces it: weight ramped 0 -> w over the same `fade` steps as the reconstruction falls (a cross-fade), so one plain
               joint parent serves every arm. With fade=1 and r_floor=0 the switch is immediate: reconstruction off, decoded prediction on.

A fork writes <run_dir>/forks/<mode>[-<tag>]/ with metrics.csv, a summary.json (the parent's, plus "fork") and ckpt_<parent steps>.pt, so that
gaf.readout.dynamics / decoder_strength read it like a run. CLI: python -m gaf.fork <run_dir> <mode> [steps fade r_floor beta [tag [dp]]].
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from .env_grid import GridCfg
from .models import Decoder, Encoder, make_predictor, sigreg
from .train import Cfg, class_weights, device, make_env, rec_error

MODES = ("continue", "fade", "anchor", "freeze")


def _load(run_dir):
    d = Path(run_dir)
    s = json.loads((d / "summary.json").read_text())
    cd = dict(s["cfg"])  # a copy: the parent's summary is written back with the fork info
    cd["env"] = GridCfg(**cd["env"])
    c = Cfg(**cd)
    return d, s, c, torch.load(d / f"ckpt_{c.steps}.pt", map_location=device())


def fork_name(mode: str, tag: str = "") -> str:
    return f"{mode}-{tag}" if tag else mode


def joint_fork(run_dir, mode: str, steps: int = 6000, fade: int = 3000, r_floor: float = 0.1, beta: float = 10.0, log_every: int = 500,
               seed: int = 0, verbose: bool = True, tag: str = "", dp: float | None = None) -> Path:
    assert mode in MODES, f"unknown mode {mode}"
    d, parent, c, sd = _load(run_dir)
    dp_w = c.dp_weight if dp is None else float(dp)
    dp_ramped = dp is not None and c.dp_weight == 0  # introduced by the fork: cross-faded in
    assert c.arm == "joint" and "ae_dec" in sd, "forks start from a finished joint run (arm joint) with a checkpoint at its last step"
    dev = device()
    torch.manual_seed(seed + 77)
    env = make_env(c.env, dev)
    enc, dec = Encoder(c.D, c.env.res).to(dev), Decoder(c.D, c.env.res).to(dev)
    P = make_predictor(c.pred_arch, c.D, c.env.T, env.act_dim).to(dev)
    enc.load_state_dict(sd["enc"]); dec.load_state_dict(sd["ae_dec"]); P.load_state_dict(sd["P"])
    snap = None
    if mode == "anchor":
        snap = (copy.deepcopy(enc).eval().requires_grad_(False), copy.deepcopy(P).eval().requires_grad_(False))
    if mode == "freeze":
        for m in (enc, dec):
            m.eval().requires_grad_(False)
        params = list(P.parameters())
    else:
        params = [*enc.parameters(), *dec.parameters(), *P.parameters()]
    opt = torch.optim.AdamW(params, c.lr, weight_decay=c.wd)
    w, _ = class_weights(env, c.rec_class_power, cap=c.rec_class_cap)
    out = d / "forks" / fork_name(mode, tag)
    out.mkdir(parents=True, exist_ok=True)
    rows, acc = [], {}
    for step in range(1, steps + 1):
        ramp = min(1.0, step / fade) if fade > 0 else 1.0
        r = 1.0 if mode in ("continue", "freeze") else 1.0 + (r_floor - 1.0) * ramp
        b = beta * ramp if mode == "anchor" else 0.0
        q = dp_w * ramp if dp_ramped else dp_w
        clip = env.sample(c.batch)
        xc, xt, a = clip.x[:, :-1], clip.x[:, -1], clip.a
        with torch.set_grad_enabled(mode != "freeze"):
            zc, zt = enc(xc), enc(xt)
        p = P(zc, a)
        l_pred = (p - zt.detach()).pow(2).mean()
        loss = l_pred
        terms = {"pred": l_pred}
        if mode != "freeze":
            l_sig = c.lam * sigreg(torch.cat([zc.transpose(0, 1), zt[None]]), c.n_slices)
            l_rec = rec_error(dec(torch.cat([zc, zt[:, None]], 1)), clip.x, clip.f["tile_class"], w, c.env.tile)
            loss = loss + l_sig + r * c.shadow_rec * l_rec
            terms.update(sigreg=l_sig, rec=l_rec)
        if dp_w > 0:  # decoded prediction: the parent's weight, or introduced (cross-faded in) by the fork
            l_dp = rec_error(dec(p), xt, clip.f["tile_class"][:, -1], w, c.env.tile)
            loss = loss + q * l_dp
            terms["dp"] = l_dp
        if snap is not None:
            with torch.no_grad():
                target = snap[1](snap[0](xc), a)
            l_anchor = (p - target).pow(2).mean()
            loss = loss + b * l_anchor
            terms["anchor"] = l_anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for k, v in terms.items():
            acc.setdefault(k, []).append(v.item())
        if step % log_every == 0 or step == steps:
            row = {"step": step, "r": r, "beta": b, "dp_w": q, **{k: sum(v) / len(v) for k, v in acc.items()}}
            rows.append(row)
            acc = {}
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            if verbose:
                print(f"[{mode}] step {step}: " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "step"), flush=True)
    torch.save({"enc": enc.state_dict(), "P": P.state_dict(), "ae_dec": dec.state_dict()}, out / f"ckpt_{c.steps}.pt")
    parent["fork"] = dict(mode=mode, tag=tag, steps=steps, fade=fade, r_floor=r_floor, beta=beta, dp_weight=dp_w, dp_ramped=dp_ramped, from_run=str(d))
    (out / "summary.json").write_text(json.dumps(parent))
    (out / "done").write_text("ok")
    return out


def launch_forks(run_dir, modes=MODES, steps: int = 6000, fade: int = 3000, r_floor: float = 0.1, beta: float = 10.0, tag: str = "",
                 dp: float | None = None) -> dict:
    """Start one background process per mode (python -m gaf.fork), surviving the notebook kernel. Modes that are done or running are skipped.
    `tag` keeps variants side by side (forks/<mode>-<tag>); `dp` as in joint_fork. Returns fork name -> pid."""
    run_dir, pids = Path(run_dir), {}
    for m in modes:
        out = run_dir / "forks" / fork_name(m, tag)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "done").exists() or ((out / "pid").exists() and _alive(out / "pid")):
            continue
        with open(out / "log.txt", "a") as log:
            p = subprocess.Popen([sys.executable, "-m", "gaf.fork", str(run_dir), m, str(steps), str(fade), str(r_floor), str(beta), tag]
                                 + ([str(dp)] if dp is not None else []),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (out / "pid").write_text(str(p.pid))
        pids[fork_name(m, tag)] = p.pid
    return pids


def _alive(pidfile) -> bool:
    import os
    try:
        os.kill(int(Path(pidfile).read_text()), 0)
        return True
    except (ProcessLookupError, ValueError, FileNotFoundError):
        return False


def status(run_dir, modes=MODES) -> pd.DataFrame:
    """modes: fork directory names (mode or mode-tag)."""
    rows = {}
    for m in modes:
        out = Path(run_dir) / "forks" / m
        h = pd.read_csv(out / "metrics.csv") if (out / "metrics.csv").exists() else None
        rows[m] = {"state": "done" if (out / "done").exists() else ("running" if (out / "pid").exists() and _alive(out / "pid") else "not started"),
                   "step": int(h.step.iloc[-1]) if h is not None else 0, **({k: round(float(h[k].iloc[-1]), 4) for k in ("pred", "rec", "dp", "anchor") if h is not None and k in h})}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    run_dir, mode = sys.argv[1], sys.argv[2]
    num = [float(x) for x in sys.argv[3:7]]
    kw = {k: (int(v) if k in ("steps", "fade") else v) for k, v in zip(("steps", "fade", "r_floor", "beta"), num, strict=False)}
    joint_fork(run_dir, mode, tag=sys.argv[7] if len(sys.argv) > 7 else "", dp=float(sys.argv[8]) if len(sys.argv) > 8 else None, **kw)
