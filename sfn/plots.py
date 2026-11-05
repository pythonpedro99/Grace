"""The four key figures of §7. Categorical slots 1-4 of the reference palette, thin marks."""
import matplotlib.pyplot as plt
import pandas as pd

from .stages import R2_COLS

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, never cycled past 4
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 110})


def _where(df, **where):
    for k, v in where.items():
        df = df[df[k] == v]
    return df


def race(df, out, labels=("A0", "A2", "M1d"), **where):
    """Fig 1: R2_4 (easy) and R2_5 (hard to learn) over training, seed-mean per arm."""
    assert len(labels) <= len(COLORS), "at most four arms per chart"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, layout="constrained")
    for ax, k in zip(axes, (4, 5), strict=True):
        for lab, c in zip(labels, COLORS, strict=False):
            runs = _where(df, label=lab, **where).run_id
            m = pd.concat(pd.read_csv(f"{out}/{r}.csv") for r in runs).groupby("step")[f"r2_{k}"]
            ax.plot(m.mean().index, m.mean(), color=c, lw=2, label=lab)
        ax.set(title=f"group {k}", xlabel="step")
    axes[0].set_ylabel("ridge R² from z")
    axes[0].legend(loc="lower right")
    return fig


def calibration(tab):
    """Fig 2: final retention of groups 5, 6, 8 per arm (ideal: R²max, R²max, 0)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, (k, ideal) in zip(axes, ((5, 1.0), (6, 0.5), (8, 0.0)), strict=True):
        ax.bar(tab.index, tab[f"r2_{k}"], color=COLORS[0], width=0.7)
        ax.axhline(ideal, color=MUTED, ls="--", lw=1)
        ax.set(title=f"group {k} (ideal {ideal:g})")
        ax.tick_params(axis="x", rotation=70)
    axes[0].set_ylabel("final R²")
    return fig


def pareto(df, **where):
    """Fig 3: retention of groups 1-7 against leakage of group 8; one point per arm."""
    d = _where(df, **where).assign(keep=lambda x: x[R2_COLS[:7]].mean(axis=1))
    g = d.groupby("label").agg(x=("r2_8", "mean"), xe=("r2_8", "sem"), y=("keep", "mean"),
                               ye=("keep", "sem"))
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    ax.errorbar(g.x, g.y, g.ye, g.xe, fmt="o", color=COLORS[0], ms=7, lw=1, mec="white")
    for lab, r in g.iterrows():
        ax.annotate(lab, (r.x, r.y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="leakage: R² of group 8 (lower is better)",
           ylabel="retention: mean R² of groups 1-7")
    return fig


def budget(df, labels=("M1d", "M1d+B1", "M1d+B-rand"), Ds=(16, 48), **where):
    """Fig 4: variance share and effective dims per group; the allocation effect shows at D=16."""
    fig, axes = plt.subplots(2, len(Ds), figsize=(5.5 * len(Ds), 6), squeeze=False,
                             layout="constrained")
    w = 0.8 / len(labels)
    for j, D in enumerate(Ds):
        for i, (pre, name) in enumerate((("vs", "variance share"), ("ed", "effective dims"))):
            ax = axes[i, j]
            for n, (lab, c) in enumerate(zip(labels, COLORS, strict=False)):
                m = _where(df, label=lab, D=D, **where)[[f"{pre}_{k}" for k in range(1, 9)]].mean()
                ax.bar([k + n * w for k in range(1, 9)], m, width=w * 0.9, color=c, label=lab)
            ax.set(title=f"D = {D}", ylabel=name, xticks=range(1, 9))
            ax.set_xlabel("group") if i else None
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    return fig
