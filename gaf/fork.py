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

Target gradient (tgt=w > 0): the prediction error also reaches the encoder through the target, ||sg(P(z_ctx, a)) - z_next||^2, weight ramped
0 -> w over the fade (LeWM-style, ungated). The latent prediction above keeps its stop-gradient, so with w = 1 the encoder sees the full
gradient of ||P(z_ctx, a) - z_next||^2. It actively removes what the predictor cannot produce (an active release, not a drift).

Prediction-gated reconstruction (gate=tau > 0): the real next frame is reconstructed again, D(z_next) against x_next, but each tile is weighted by how well
the predictor foresaw it. With e_i the tile's per-pixel squared error of the decoded prediction D(P(z_ctx, a)) (stop-gradient), the tile weight is
exp(-e_i / tau): near 1 where the prediction is right to within tau (the configured detail level, in pixel error), near 0 where it is not (noise,
not-yet-learned content). It is the hold reward on the REAL latent that the decoded prediction lacks (the predictor can draw action-determined content,
such as the facing, without the encoder holding it). Weight ramped 0 -> 1 over the fade; the all-frame reconstruction (`r_floor`) fades as usual.
On the 20k parent, per-pixel error medians are 0.001-0.002 (inventory, map), 0.005 (hand), 0.1 (agent), 0.15 (twinkle), so tau = 0.01-0.02 opens the learned
content and closes the noise.

Two switches for label-free and ablation runs: weights=False drops the class weights from every reconstruction-like term of the fork (the all-frame
reconstruction, the decoded prediction, the gated reconstruction: plain pixel error), so stage 2 needs no labels; lam overrides the parent's SIGReg weight
(0 switches SIGReg off; its value is still logged).

A fork writes <run_dir>/forks/<mode>[-<tag>]/ with metrics.csv, a summary.json (the parent's, plus "fork") and ckpt_<parent steps>.pt, so that
gaf.readout.dynamics / decoder_strength read it like a run. CLI: python -m gaf.fork <run_dir> <mode> [steps fade r_floor beta [tag [dp|none [tgt [gate [weights [lam|none]]]]]]].
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


def _tile_err(x_hat, x, tile):
    """Squared error per tile, summed over channels and pixels: (B, tiles). x, x_hat: (B, 3, res, res)."""
    e = (x_hat - x) ** 2
    side = x.shape[-1] // tile
    return e.view(e.shape[0], 3, side, tile, side, tile).sum((1, 3, 5)).flatten(1)


def fork_name(mode: str, tag: str = "") -> str:
    return f"{mode}-{tag}" if tag else mode


def joint_fork(run_dir, mode: str, steps: int = 6000, fade: int = 3000, r_floor: float = 0.1, beta: float = 10.0, log_every: int = 500,
               seed: int = 0, verbose: bool = True, tag: str = "", dp: float | None = None,
               tgt: float = 0.0, gate: float = 0.0, weights: bool = True, lam: float | None = None) -> Path:
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
    if not weights:
        w = None  # plain pixel error in every reconstruction-like term
    wc = (lambda cls: w[cls]) if w is not None else (lambda cls: 1.0)
    tile_arg = c.env.tile if w is not None else None
    lam_v = c.lam if lam is None else float(lam)
    out = d / "forks" / fork_name(mode, tag)
    out.mkdir(parents=True, exist_ok=True)
    rows, acc = [], {}
    for step in range(1, steps + 1):
        ramp = min(1.0, step / fade) if fade > 0 else 1.0
        r = 1.0 if mode in ("continue", "freeze") else 1.0 + (r_floor - 1.0) * ramp
        b = beta * ramp if mode == "anchor" else 0.0
        q = dp_w * ramp if dp_ramped else dp_w
        tg = tgt * ramp if mode != "freeze" else 0.0
        gr = ramp if gate > 0 and mode != "freeze" else 0.0
        clip = env.sample(c.batch)
        xc, xt, a = clip.x[:, :-1], clip.x[:, -1], clip.a
        with torch.set_grad_enabled(mode != "freeze"):
            zc, zt = enc(xc), enc(xt)
        p = P(zc, a)
        l_pred = (p - zt.detach()).pow(2).mean()
        loss = l_pred
        terms = {"pred": l_pred}
        if mode != "freeze":
            z_sig = torch.cat([zc.transpose(0, 1), zt[None]])
            if lam_v > 0:
                l_sig = lam_v * sigreg(z_sig, c.n_slices)
                sig_log = l_sig
            else:
                with torch.no_grad():  # SIGReg off: not in the loss, but its value is logged to watch the spectrum
                    sig_log = sigreg(z_sig, c.n_slices) * c.lam
                l_sig = torch.zeros((), device=zt.device)
            l_rec = rec_error(dec(torch.cat([zc, zt[:, None]], 1)), clip.x, clip.f["tile_class"], w, tile_arg)
            loss = loss + l_sig + r * c.shadow_rec * l_rec
            terms.update(sigreg=sig_log, rec=l_rec)
        if tgt > 0 and mode != "freeze":  # the target gradient: the encoder moves z_next towards the (fixed) prediction
            l_tgt = (p.detach() - zt).pow(2).mean()
            loss = loss + tg * l_tgt
            terms["tgt"] = l_tgt
        if gate > 0 and mode != "freeze":  # prediction-gated reconstruction of the real next frame
            cls_t = clip.f["tile_class"][:, -1]
            with torch.no_grad():
                gw = torch.exp(-_tile_err(dec(p), xt, c.env.tile) / (3 * c.env.tile ** 2) / gate)  # per-pixel error of each tile's prediction
            l_gate = (_tile_err(dec(zt), xt, c.env.tile) * wc(cls_t) * gw).sum() / (cls_t.numel() * c.env.tile ** 2)
            loss = loss + gr * l_gate
            terms["gate_rec"], terms["gate_w"] = l_gate, gw.mean()
        if dp_w > 0:  # decoded prediction: the parent's weight, or introduced (cross-faded in) by the fork
            l_dp = rec_error(dec(p), xt, clip.f["tile_class"][:, -1], w, tile_arg)
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
            row = {"step": step, "r": r, "beta": b, "dp_w": q, "tgt_w": tg, "gate_r": gr, **{k: sum(v) / len(v) for k, v in acc.items()}}
            rows.append(row)
            acc = {}
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            if verbose:
                print(f"[{mode}] step {step}: " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "step"), flush=True)
    torch.save({"enc": enc.state_dict(), "P": P.state_dict(), "ae_dec": dec.state_dict()}, out / f"ckpt_{c.steps}.pt")
    parent["fork"] = dict(mode=mode, tag=tag, steps=steps, fade=fade, r_floor=r_floor, beta=beta, dp_weight=dp_w, dp_ramped=dp_ramped, tgt=tgt, gate=gate, weights=weights, lam=lam_v, from_run=str(d))
    (out / "summary.json").write_text(json.dumps(parent))
    (out / "done").write_text("ok")
    return out


def launch_forks(run_dir, modes=MODES, steps: int = 6000, fade: int = 3000, r_floor: float = 0.1, beta: float = 10.0, tag: str = "",
                 dp: float | None = None, tgt: float = 0.0, gate: float = 0.0, weights: bool = True, lam: float | None = None) -> dict:
    """Start one background process per mode (python -m gaf.fork), surviving the notebook kernel. Modes that are done or running are skipped.
    `tag` keeps variants side by side (forks/<mode>-<tag>); `dp`, `tgt`, `gate`, `weights`, `lam` as in joint_fork. Returns fork name -> pid."""
    run_dir, pids = Path(run_dir), {}
    for m in modes:
        out = run_dir / "forks" / fork_name(m, tag)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "done").exists() or ((out / "pid").exists() and _alive(out / "pid")):
            continue
        with open(out / "log.txt", "a") as log:
            p = subprocess.Popen([sys.executable, "-m", "gaf.fork", str(run_dir), m, str(steps), str(fade), str(r_floor), str(beta), tag]
                                 + [str(dp) if dp is not None else "none", str(tgt), str(gate), "1" if weights else "0",
                                    str(lam) if lam is not None else "none"],
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
                   "step": int(h.step.iloc[-1]) if h is not None else 0, **({k: round(float(h[k].iloc[-1]), 4) for k in ("pred", "rec", "dp", "tgt", "gate_rec", "anchor") if h is not None and k in h})}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    run_dir, mode = sys.argv[1], sys.argv[2]
    num = [float(x) for x in sys.argv[3:7]]
    kw = {k: (int(v) if k in ("steps", "fade") else v) for k, v in zip(("steps", "fade", "r_floor", "beta"), num, strict=False)}
    dp = sys.argv[8] if len(sys.argv) > 8 else "none"
    joint_fork(run_dir, mode, tag=sys.argv[7] if len(sys.argv) > 7 else "", dp=None if dp == "none" else float(dp),
               tgt=float(sys.argv[9]) if len(sys.argv) > 9 else 0.0, gate=float(sys.argv[10]) if len(sys.argv) > 10 else 0.0,
               weights=sys.argv[11] != "0" if len(sys.argv) > 11 else True, lam=None if len(sys.argv) <= 12 or sys.argv[12] == "none" else float(sys.argv[12]), **kw)
