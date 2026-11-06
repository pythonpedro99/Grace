"""Run grids, result collection, tables, verdicts and figures for the notebook."""
import contextlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .env import GROUPS, ClipEnv
from .gate import floor_probe
from .probes import ridge_r2
from .train import Cfg, device, load_model, run

BASELINES = ("lewm", "sg", "ema")
FIXES = ("target", "context", "both")
R2 = [f"r2_{g}" for g in GROUPS]

# Fixed colour per arm (reference categorical palette, light mode): colour follows the arm,
# never its position in a chart. At most four arms share a chart.
COLOR = {"sg": "#2a78d6", "both": "#eb6834", "context": "#1baf7a", "target": "#eda100",
         "lewm": "#e87ba4", "ema": "#008300", "detached": "#6250d6"}
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
    """One run in a worker process; its progress goes to <run_id>/log.txt."""
    torch.set_num_threads(2)  # several workers share the CPU (data sampling, small ops)
    odir = Path(out) / cfg.run_id
    odir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(odir / "log.txt", "a") as f, contextlib.redirect_stdout(f):
        run(cfg, out, verbose=True)
    return time.time() - t


def execute(cfgs, out, workers: int = 1) -> None:
    """Run every cfg whose summary.json does not exist yet.

    workers=1: sequentially in this process (progress printed here). workers>1: that many runs
    at once in separate processes on the same GPU. The model is small and launch-bound, so
    3-4 workers give close to a proportional speed-up; per-run progress is in <run_id>/log.txt.
    A failing run does not stop the others; the errors are raised at the end.
    """
    todo = [c for c in cfgs if not (Path(out) / c.run_id / "summary.json").exists()]
    print(f"{len(cfgs) - len(todo)} of {len(cfgs)} runs already done, {len(todo)} to go"
          + (f" on {workers} workers" if workers > 1 and todo else ""), flush=True)
    t0, errors = time.time(), []
    if workers <= 1:
        for i, c in enumerate(todo, 1):
            t = time.time()
            print(f"[{i}/{len(todo)}] {_label(c)}", flush=True)
            run(c, out)
            print(f"    finished in {time.time() - t:.0f}s", flush=True)
        return
    ctx = mp.get_context("spawn")  # CUDA cannot be shared with forked workers
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        futs = {pool.submit(_work, c, str(out)): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                sec = fut.result()
                print(f"[{i}/{len(todo)}] {_label(c)} finished in {sec:.0f}s "
                      f"(wall {time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append((c, e))
                print(f"[{i}/{len(todo)}] {_label(c)} FAILED: {e!r} "
                      f"(log: {Path(out) / c.run_id / 'log.txt'})", flush=True)
    if errors:
        names = ", ".join(_label(c) for c, _ in errors)
        raise RuntimeError(f"{len(errors)} run(s) failed: {names}")


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
    order = [a for a in (*BASELINES, "detached", *FIXES) if a in d.groups]
    return d.mean().reindex(order or None).round(3)


def diag_table(df, **where) -> pd.DataFrame:
    """Leak diagnostics of the final eval window per arm (seed mean)."""
    cols = [c for c in ("n_S", "leak_pos", "leak_cos", "leak_cos_post", "proj_ratio", "lreal_S",
                        "lreal_open",
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
    } for a in tab.index if a in FIXES}).T


# ---- floors -----------------------------------------------------------------------------------
@torch.no_grad()
def _content(enc, env, Q, n=8192):
    clip = env.sample(2 * n)
    z = enc(clip.x[:, -1]) @ Q
    return np.stack([ridge_r2(clip.f[g][:, -1], z, n, per_col=True).cpu().numpy()
                     for g in GROUPS], 1)


def floors_of(cfg: Cfg, out, steps=3000, basis="excess") -> pd.DataFrame:
    """Floor probe on the final encoder of a finished run, with per-direction content.

    One row per latent direction: floor L*_j, and the R^2 of z_j from each group.
    """
    torch.manual_seed(cfg.seed + 777)
    enc, P, tgt = load_model(cfg, out)
    env = ClipEnv(cfg.env, device())
    Ls, Q, _ = floor_probe(enc, tgt, env, steps=steps, lr=cfg.lr, wd=cfg.wd, basis=basis,
                           pred=P)
    c = _content(enc, env, Q)
    d = pd.DataFrame(c, columns=list(GROUPS))
    d.insert(0, "floor", Ls.cpu().numpy())
    d["arm"], d["seed"] = cfg.arm, cfg.seed
    d.index.name = "dir"
    return d


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
