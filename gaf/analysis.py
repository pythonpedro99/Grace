"""Run grids, result collection, tables, verdicts and figures for the notebook."""
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .env import GROUPS, ClipEnv
from .gate import floor_probe
from .probes import ridge_r2
from .models import Predictor
from .train import Cfg, device, load_model, run

BASELINES = ("lewm", "sg", "ema")
FIXES = ("target", "context", "both")
FIXES_ALL = (*FIXES, "block")  # block: masked loss on both paths, single seed so far
R2 = [f"r2_{g}" for g in GROUPS]

# Fixed colour per arm (reference categorical palette, light mode): colour follows the arm,
# never its position in a chart. At most four arms share a chart.
COLOR = {"sg": "#2a78d6", "both": "#eb6834", "context": "#1baf7a", "target": "#eda100",
         "lewm": "#e87ba4", "ema": "#008300", "detached": "#6250d6", "block": "#e34948"}
GROUP_COLOR = dict(zip(GROUPS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), strict=True))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 110, "lines.linewidth": 2})


# ---- running ----------------------------------------------------------------------------------
def grid(arms, seeds=(0, 1, 2), **common) -> list[Cfg]:
    return [Cfg(arm=a, seed=s, **common) for a in arms for s in seeds]


def _label(c: Cfg) -> str:
    return f"{c.arm} seed={c.seed} D={c.D}" + (f" [{c.tag}]" if c.tag else "")


def _work(cfg: Cfg, out) -> float:
    """One run in a worker process. Progress goes to <run_id>/progress.json (polled by the
    parent), the eval lines to <run_id>/log.txt."""
    torch.set_num_threads(2)  # several workers share the CPU (data sampling, small ops)
    odir = Path(out) / cfg.run_id
    odir.mkdir(parents=True, exist_ok=True)
    pf, tmp, last = odir / "progress.json", odir / "progress.tmp", [-100, ""]

    def on_step(step, phase):  # throttled: every 25 steps, and at every phase change
        if phase != last[1] or step - last[0] >= 25:
            tmp.write_text(json.dumps({"step": step, "phase": phase}))
            os.replace(tmp, pf)  # atomic: the parent never reads half a file
            last[:] = [step, phase]

    t = time.time()
    with open(odir / "log.txt", "a") as f:
        def log(line):
            f.write(line + "\n")
            f.flush()

        run(cfg, out, on_step=on_step, log=log)
    return time.time() - t


def execute(cfgs, out, workers: int = 1, progress: bool = True) -> None:
    """Run every cfg whose summary.json does not exist yet, with one progress bar.

    The bar counts training steps over all runs still to do (so its ETA covers the whole cell)
    and shows what each running run is doing: "<arm> s<seed> <percent> <phase>", where the phase
    is train / floor probe / eval / final probes.

    workers=1: sequentially in this process; the eval lines are printed above the bar.
    workers>1: that many runs at once in separate processes on the same GPU. The model is small
    and launch-bound, so 3-4 workers give close to a proportional speed-up. The eval lines of
    each run are in <run_id>/log.txt. A failing run does not stop the others; the errors are
    raised at the end. progress=False: no bar (the eval lines are still printed).
    """
    todo = [c for c in cfgs if not (Path(out) / c.run_id / "summary.json").exists()]
    print(f"{len(cfgs) - len(todo)} of {len(cfgs)} runs already done, {len(todo)} to go"
          + (f" on {workers} workers" if workers > 1 and todo else ""), flush=True)
    if not todo:
        return
    counted = {c.run_id: 0 for c in todo}  # steps of each run already added to the bar
    bar = tqdm(total=sum(c.steps for c in todo), desc="training", unit="step", unit_scale=True,
               disable=not progress, smoothing=0.05)

    def advance(c: Cfg, step: int) -> None:
        if step > counted[c.run_id]:
            bar.update(step - counted[c.run_id])
            counted[c.run_id] = step

    try:
        if workers <= 1:
            for i, c in enumerate(todo, 1):
                t = time.time()

                def on_step(step, phase, c=c):
                    advance(c, step)
                    bar.set_postfix_str(f"run {i}/{len(todo)}: {c.arm} s{c.seed} {phase}",
                                        refresh=False)

                run(c, out, on_step=on_step, log=bar.write)
                bar.write(f"[{i}/{len(todo)}] {_label(c)} finished in {time.time() - t:.0f}s")
            return
        errors = _execute_parallel(todo, out, workers, bar, advance)
    finally:
        bar.close()
    if errors:
        names = ", ".join(_label(c) for c, _ in errors)
        raise RuntimeError(f"{len(errors)} run(s) failed: {names}")


def _execute_parallel(todo, out, workers, bar, advance) -> list:
    for c in todo:  # a progress file left by an interrupted attempt must not count
        (Path(out) / c.run_id / "progress.json").unlink(missing_ok=True)
    ctx = mp.get_context("spawn")  # CUDA cannot be shared with forked workers
    errors, finished, t0 = [], 0, time.time()
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        futs = {pool.submit(_work, c, str(out)): c for c in todo}
        pending = set(futs)
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in done:
                c, finished = futs[fut], finished + 1
                advance(c, c.steps)
                try:
                    sec = fut.result()
                    bar.write(f"[{finished}/{len(todo)}] {_label(c)} finished in {sec:.0f}s "
                              f"(wall {time.time() - t0:.0f}s)")
                except Exception as e:  # noqa: BLE001
                    errors.append((c, e))
                    bar.write(f"[{finished}/{len(todo)}] {_label(c)} FAILED: {e!r} "
                              f"(log: {Path(out) / c.run_id / 'log.txt'})")
            status = []
            for fut in pending:
                c = futs[fut]
                try:
                    d = json.loads((Path(out) / c.run_id / "progress.json").read_text())
                except (FileNotFoundError, ValueError):
                    continue  # not started yet
                advance(c, d["step"])
                status.append(f"{c.arm} s{c.seed} {100 * d['step'] // c.steps}% {d['phase']}")
            bar.set_postfix_str(" | ".join(status), refresh=False)
            bar.refresh()
    return errors


# ---- collecting -------------------------------------------------------------------------------
def collect(out) -> pd.DataFrame:
    """One row per finished run: cfg fields (env.* prefixed), final metrics, MLP probe."""
    rows = []
    for p in Path(out).glob("*/summary.json"):
        s = json.loads(p.read_text())
        env = {f"env.{k}": v for k, v in s["cfg"].pop("env").items()}
        rows.append({"run_id": s["run_id"], **s["cfg"], **env, "seconds": s["seconds"],
                     **s["final"], **{k: v for k, v in s.items() if k.startswith("mlp_")}})
    return pd.DataFrame(rows)


def history(out, run_id) -> pd.DataFrame:
    return pd.read_csv(Path(out) / run_id / "metrics.csv")


def early(out, run_id) -> pd.DataFrame:
    """Extra evals of a run with cfg.early_eval (empty frame if it has none)."""
    p = Path(out) / run_id / "early.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["step"])


def curve(df, out, sel: dict) -> pd.DataFrame:
    """Metrics over steps of the run(s) matching `sel` (cfg fields): early evals merged with
    the regular ones, averaged over matching runs. One row per step."""
    ids = select(df, **sel).run_id
    assert len(ids), f"no finished run matches {sel}"
    h = pd.concat(pd.concat([early(out, r), history(out, r)]) for r in ids)
    return h.groupby("step").mean(numeric_only=True)


def dirs(out, run_id) -> dict:
    return dict(np.load(Path(out) / run_id / "dirs.npz"))


def select(df, **where) -> pd.DataFrame:
    for k, v in where.items():
        df = df[df[k] == v]
    return df


def table(df, cols=None, by="arm", **where) -> pd.DataFrame:
    """Seed mean (± sem in a second frame) of `cols` per arm, for runs matching `where`."""
    cols = cols or R2 + [f"mlp_r2_{g}" for g in GROUPS]
    d = select(df, **where).groupby(by)[cols]
    order = [a for a in (*BASELINES, "detached", *FIXES_ALL) if a in d.groups]
    return d.mean().reindex(order or None).round(3)


def diag_table(df, **where) -> pd.DataFrame:
    """Leak diagnostics of the final eval window per arm (seed mean)."""
    cols = [c for c in ("n_S", "leak_pos", "leak_cos", "leak_cos_post", "proj_ratio", "lreal_S",
                        "lreal_open", "lk_ctx_S", "lk_rest_S", "lk_ctx_open", "lk_rest_open",
                        "wb_ratio", "rank_dz", "rank_z", "sigreg") if c in df]
    return table(df, cols, **where)


# ---- verdicts ---------------------------------------------------------------------------------
def baseline_verdict(tab, keep=0.8, drop=0.3) -> pd.DataFrame:
    """Does a baseline show the failure? Keeps slow (R2 >= keep), drops fast/hard (< drop)."""
    return pd.DataFrame({a: {"slow kept": tab.loc[a, "r2_slow"] >= keep,
                             "fast dropped": tab.loc[a, "r2_fast"] < drop,
                             "hard dropped": tab.loc[a, "r2_hard"] < drop,
                             "noise dropped": tab.loc[a, "r2_noise"] < drop}
                         for a in tab.index if a in BASELINES}).T


def fix_verdict(tab, ref="sg", gain=0.3, slack=0.1, slow_slack=0.1) -> pd.DataFrame:
    """Fix vs. its reference: fast and hard gain >= `gain`, noise within `slack` of `ref`,
    slow within `slow_slack` of `ref`. In the tight regime (D = n_slow) slow content alone fills
    the latent and keeping the sprites must cost slow R^2: use a larger `slow_slack` there."""
    r = tab.loc[ref]
    return pd.DataFrame({a: {
        f"fast +{gain}": tab.loc[a, "r2_fast"] - r.r2_fast >= gain,
        f"hard +{gain}": tab.loc[a, "r2_hard"] - r.r2_hard >= gain,
        f"noise <= {ref} + {slack}": tab.loc[a, "r2_noise"] <= r.r2_noise + slack,
        f"slow >= {ref} - {slow_slack}": tab.loc[a, "r2_slow"] >= r.r2_slow - slow_slack,
    } for a in tab.index if a in FIXES_ALL}).T


# ---- floors -----------------------------------------------------------------------------------
@torch.no_grad()
def _content(enc, env, Q, n=8192):
    clip = env.sample(2 * n)
    z = enc(clip.x[:, -1]) @ Q
    return np.stack([ridge_r2(clip.f[g][:, -1], z, n, per_col=True).cpu().numpy()
                     for g in GROUPS], 1)


def floors_of(cfg: Cfg, out, steps=3000, basis="excess", progress=True) -> pd.DataFrame:
    """Floor probe on the final encoder of a finished run, with per-direction content.

    One row per latent direction: floor L*_j, and the R^2 of z_j from each group.
    progress: show a bar over the probe's training steps.
    """
    torch.manual_seed(cfg.seed + 777)
    enc, P, tgt = load_model(cfg, out)
    env = ClipEnv(cfg.env, device())
    Ls, Q, _ = floor_probe(enc, tgt, env, steps=steps, lr=cfg.lr, wd=cfg.wd, basis=basis,
                           pred=P, desc=f"floor probe {cfg.arm} s{cfg.seed}" if progress else None)
    c = _content(enc, env, Q)
    d = pd.DataFrame(c, columns=list(GROUPS))
    d.insert(0, "floor", Ls.cpu().numpy())
    d["arm"], d["seed"] = cfg.arm, cfg.seed
    d.index.name = "dir"
    return d


def latent_learning(cfg: Cfg, out, steps=20_000, eval_every=500, n_eval=8192, batch=512,
                    desc=None) -> pd.DataFrame:
    """How many steps does a fresh predictor need to learn each group from the frozen latents
    of a finished run? Fresh clips every step, held-out evals.

    Columns: R^2 of the target-frame factors of each group from the predictor's output (ridge,
    as `pred_r2_*` in the probe suite), and the mse. `.attrs["availability"]` holds the ridge R^2
    of each group from the latent itself, the ceiling of what a predictor could reach there.
    This is the learn time in latent space; compare it with the head start of a run.
    """
    torch.manual_seed(cfg.seed + 4242)
    dev = device()
    enc, _, tgt = load_model(cfg, out)
    env = ClipEnv(cfg.env, dev)
    cols = env.group_cols()
    P = Predictor(cfg.D, cfg.env.T).to(dev)
    opt = torch.optim.AdamW(P.parameters(), cfg.lr, weight_decay=cfg.wd)
    with torch.no_grad():
        ev = env.sample(n_eval)
        zc_e, zt_e, y_e = enc(ev.x[:, :-1]), tgt(ev.x[:, -1]), env.factors(ev)
    availability = {g: ridge_r2(zt_e, y_e[:, s], n_eval // 2) for g, s in cols.items()}
    rows = []
    for step in tqdm(range(1, steps + 1), desc=desc, unit="step", disable=desc is None):
        clip = env.sample(batch)
        with torch.no_grad():
            zc, zt = enc(clip.x[:, :-1]), tgt(clip.x[:, -1])
        loss = (P(zc, clip.a) - zt).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                p = P(zc_e, ev.a)
            rows.append({"step": step, "mse": (p - zt_e).pow(2).mean().item(),
                         **{g: ridge_r2(p, y_e[:, s], n_eval // 2) for g, s in cols.items()}})
    res = pd.DataFrame(rows).set_index("step")
    res.attrs["availability"] = availability
    return res


# ---- figures ----------------------------------------------------------------------------------
def _runs(df, arm, **where):
    return select(df, arm=arm, **where).run_id


def curves(df, out, arms, cols=R2, **where):
    """Seed-mean metric over training, one panel per column, one line per arm (<= 4 arms)."""
    assert len(arms) <= 4, "at most four arms per chart"
    fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3.2), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, col in zip(axes[0], cols, strict=True):
        for a in arms:
            runs = _runs(df, a, **where)
            if not len(runs):
                continue
            h = pd.concat(history(out, r) for r in runs).groupby("step")[col].mean()
            ax.plot(h.index, h.values, color=COLOR[a], label=a)
        ax.set(title=col.replace("r2_", ""), xlabel="step")
    axes[0, 0].set_ylabel("ridge R² from z_t" if cols == R2 else cols[0])
    axes[0, -1].legend(loc="best")
    return fig


def bars(tab, cols=R2, ideal=None):
    """Final R² per group, grouped by arm. `ideal` draws a dashed reference per group."""
    ideal = ideal or {"r2_slow": 1, "r2_fast": 1, "r2_hard": 1, "r2_noise": 0}
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.4), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, col in zip(axes[0], cols, strict=True):
        ax.bar(tab.index, tab[col], color=[COLOR.get(a, MUTED) for a in tab.index], width=0.7,
               edgecolor="white", linewidth=2)
        if col in ideal:
            ax.axhline(ideal[col], color=MUTED, ls="--", lw=1)
        ax.set(title=col.replace("r2_", "").replace("mlp_", "mlp "),
               ylim=(min(0, tab[col].min()), 1.05))
        ax.tick_params(axis="x", rotation=60)
    axes[0, 0].set_ylabel("final R²")
    return fig


def gate_map(out, run_id):
    """Per-direction state over training: S membership, relative gap, excess over floor, and
    which group each direction carries (argmax content)."""
    d = dirs(out, run_id)
    steps = d["step"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
    ext = [steps[0], steps[-1], d["inS"].shape[1] - 0.5, -0.5]
    kw = dict(aspect="auto", extent=ext, interpolation="nearest")
    for ax in axes:
        ax.grid(False)
    axes[0].imshow(d["inS"].T, cmap="Blues", vmin=0, vmax=1.3, **kw)
    axes[0].set(title="in S (dark = protected)", ylabel="direction j", xlabel="step")
    im = axes[1].imshow(d["gap"].T, cmap="Blues", vmin=0, vmax=0.1, **kw)
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    axes[1].set(title="relative gap", xlabel="step")
    im = axes[2].imshow(np.log10(np.clip(d["excess"].T, 1e-3, 1e3)), cmap="Blues", vmin=-2,
                        vmax=2, **kw)
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="log10 e")
    axes[2].set(title="excess over floor", xlabel="step")
    c = d["content"]  # (evals, D, groups)
    top = c.argmax(2).astype(float)
    top[c.max(2) < 0.2] = np.nan  # carries nothing identifiable
    cmap = plt.matplotlib.colors.ListedColormap([GROUP_COLOR[g] for g in GROUPS])
    axes[3].imshow(top.T, cmap=cmap, vmin=-0.5, vmax=len(GROUPS) - 0.5, **kw)
    axes[3].set(title="dominant group (blank: none > 0.2)", xlabel="step")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUPS]
    axes[3].legend(handles, GROUPS, loc="upper right",
                   fontsize=8, framealpha=0.9, frameon=True)
    return fig


def floor_scatter(fl: pd.DataFrame):
    """Floor L*_j against the noise content of direction j, one dot per direction, per arm."""
    arms = list(dict.fromkeys(fl.arm))
    assert len(arms) <= 4
    fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")
    for a in arms:
        d = fl[fl.arm == a]
        ax.scatter(d["noise"], d["floor"], s=40, color=COLOR[a], edgecolor="white", lw=1,
                   label=a)
    ax.set(xlabel="R² of z_j from the noise factors", ylabel="floor L*_j (probe MSE)")
    ax.legend()
    return fig


def leak_panel(df, out, arms, **where):
    """Leak over training: share of steps with ell > 0, and l_real on S vs. open directions."""
    return curves(df, out, arms, cols=["leak_pos", "lreal_S", "lreal_open", "n_S"], **where)


def drift(df, out, runs: dict, cols=("r2_fast", "r2_hard", "r2_noise", "r2_slow", "ref_unsettled"),
          window=4000) -> pd.DataFrame:
    """How much each run still moved over its last `window` steps.

    d_<g>: net change of the metric over the window. swing_<g>: max minus min of an R^2 inside the
    window, which catches oscillations that end where they started. `settled` is False if any R^2
    changed by 0.05 or more, or swung by 0.1 or more. ref_unsettled (retention arms) is the number
    of directions the shadow predictor was still improving on: its change is reported, not judged.
    """
    rows = {}
    for lab, sel in runs.items():
        c = curve(df, out, sel)
        last = c.index.max()
        before = c.index[c.index <= last - window]
        first = before.max() if len(before) else c.index.min()
        inside = c.loc[first:last]
        r, ok = {}, True
        for col in cols:
            if col not in c:
                continue
            g = col.replace("r2_", "")
            r[f"d_{g}"] = c[col].get(last, np.nan) - c[col].get(first, np.nan)
            if col.startswith("r2_"):
                r[f"swing_{g}"] = inside[col].max() - inside[col].min()
                ok = ok and abs(r[f"d_{g}"]) < 0.05 and r[f"swing_{g}"] < 0.1
        rows[lab] = {"steps": int(last), **r, "settled": bool(ok)}
    return pd.DataFrame(rows).T


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, one colour per line


def trace(df, out, runs: dict, cols=("r2_fast", "r2_hard", "r2_noise", "r2_slow"), xlog=True):
    """One panel per metric, one line per entry of `runs` ({label: cfg selector}, <= 4 lines).

    Early evals are merged in, so the first few hundred steps are resolved. xlog: symlog x axis
    (linear below step 50) to see the start of training.
    """
    assert len(runs) <= len(SERIES), "at most four lines per chart"
    cur = {lab: curve(df, out, sel) for lab, sel in runs.items()}
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.3), layout="constrained",
                             squeeze=False)
    for ax, col in zip(axes[0], cols, strict=True):
        for (lab, c), color in zip(cur.items(), SERIES, strict=False):
            ax.plot(c.index, c[col], color=color, label=lab, marker="o", ms=3, lw=1.6)
        ax.set(title=col.replace("r2_", ""), xlabel="step")
        if xlog:
            ax.set_xscale("symlog", linthresh=50)
    axes[0, 0].set_ylabel("ridge R² from z_t" if cols[0].startswith("r2_") else cols[0])
    axes[0, -1].legend(loc="best", fontsize=8)
    return fig


def snapshot(df, out, runs: dict, steps=(0, 50, 100, 200, 500, 1000, 3000),
             cols=("r2_fast", "r2_hard", "r2_noise", "r2_slow")) -> pd.DataFrame:
    """Values of `cols` at given steps, one row per entry of `runs` (NaN where not evaluated)."""
    rows = {}
    for lab, sel in runs.items():
        c = curve(df, out, sel)
        rows[lab] = {f"{col.replace('r2_', '')}@{s}": c[col].get(s, np.nan)
                     for col in cols for s in steps}
    return pd.DataFrame(rows).T


def show_clips(env_cfg, n=4):
    """A few clips (rows) and their frames (columns), colours rescaled for display."""
    env = ClipEnv(env_cfg, "cpu")
    torch.manual_seed(0)
    x = env.sample(n).x
    lo, hi = x.min(), x.max()
    T = x.shape[1]
    fig, axes = plt.subplots(n, T, figsize=(1.6 * T, 1.6 * n), layout="constrained")
    for i in range(n):
        for t in range(T):
            ax = axes[i, t]
            ax.imshow(((x[i, t] - lo) / (hi - lo)).permute(1, 2, 0).numpy())
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
            if i == 0:
                ax.set_title("target" if t == T - 1 else f"context {t}", fontsize=9)
    return fig
