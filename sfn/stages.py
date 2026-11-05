"""Stage grids (§8), execution, result collection and the pass/fail rules of §9."""
import json
import time
from pathlib import Path

import pandas as pd

from .train import Cfg, run, run_gt

SEEDS = (0, 1, 2)
R2MAX = {1: 0.96, 2: 0.49, 3: 0.90, 4: 1.0, 5: 1.0, 6: 0.5, 7: 1.0}
R2_COLS = [f"r2_{k}" for k in range(1, 9)]
M_ARMS = ["M1a", "M1b", "M1c", "M1d", "M2", "M3"]
KEYS = ["arm", "bfac", "alpha", "tau_ema"]  # what identifies an arm inside a stage
TARGET_ARMS = [dict(arm="A0"), dict(arm="A1", alpha=0.0), dict(arm="A1", alpha=0.1),
               dict(arm="A2", tau_ema=0.996), *({"arm": a} for a in
               ["M1a", "M1b", "M1c", "M1d", "M2", "M3", "C1", "C2", "C3"])]


def label(arm, bfac="B0", alpha=0.0, **_) -> str:
    return arm + (f"({alpha:g})" if arm == "A1" else "") + ("" if bfac == "B0" else f"+{bfac}")


def _grid(arms, seeds=SEEDS, **common) -> list[Cfg]:
    return [Cfg(**{**common, **a, "seed": s}) for a in arms for s in seeds]


def execute(cfgs: list[Cfg], out) -> None:
    for i, c in enumerate(cfgs, 1):
        t = time.time()
        run(c, out)
        print(f"[{i}/{len(cfgs)}] {label(**vars(c))} D={c.D} diff={c.diff} seed={c.seed} "
              f"({time.time() - t:.0f}s)")


def execute_gt(out, diffs=(0, 1, 2), seeds=SEEDS) -> None:
    for d in diffs:
        for s in seeds:
            run_gt(d, s, out)


# ---- stage grids ----------------------------------------------------------------------------
def stage0(diff=0) -> list[Cfg]:
    return [Cfg(lam=lam, w_fast=w, seed=s, diff=diff, heads=True)
            for lam in (0.02, 0.1, 0.5) for w in (0.3, 1.0) for s in (0, 1)]


def stage1(**common) -> list[Cfg]:
    return _grid(TARGET_ARMS, **common)


def stage1_budget(**common) -> list[Cfg]:
    arms = [dict(arm="A0", bfac="B1"), *({"arm": "M1d", "bfac": b} for b in ("B1", "B2", "B-rand"))]
    return _grid(arms, **common)


def stage1_capacity(best_m: str, **common) -> list[Cfg]:
    arms = [dict(arm="A0"), dict(arm="A2"), dict(arm=best_m), dict(arm="A0", bfac="B1"),
            dict(arm="M1d", bfac="B1")]
    return _grid(arms, **{**common, "D": 48})


def stage1_middle(**common) -> list[Cfg]:  # only if B shows no effect at D = 16
    arms = [dict(arm="A0"), dict(arm="M1d"), dict(arm="M1d", bfac="B1")]
    return _grid(arms, **{**common, "D": 24})


def stage2(best_a1: dict, best_m: dict, **common) -> list[Cfg]:
    arms = [dict(arm="A0"), dict(arm="A2"), best_a1, best_m, dict(arm="M2")]
    return [c for d in (1, 2) for c in _grid(arms, **{**common, "diff": d})]


def stage2_budget(best: dict, **common) -> list[Cfg]:  # only if B passed its §9 criterion
    return [c for d in (1, 2) for c in _grid([best], **{**common, "diff": d})]


# ---- collection -----------------------------------------------------------------------------
def collect(out) -> pd.DataFrame:
    """One row per finished run: cfg fields, final metrics, MLP probe, seconds, label."""
    rows = [json.loads(p.read_text()) for p in Path(out).glob("arm-*.json")]
    df = pd.json_normalize(rows)
    df.columns = [c.split(".")[-1] for c in df.columns]
    df["label"] = [label(**r) for r in df[["arm", "bfac", "alpha"]].to_dict("records")]
    return df


def table(df, cols=R2_COLS + ["r2_3_post"], **where) -> pd.DataFrame:
    """Seed-mean of `cols` per arm label, for the runs matching cfg fields in `where`."""
    for k, v in where.items():
        df = df[df[k] == v]
    return df.groupby("label")[cols].mean().round(3)


def best_cfg(df, arms, **where) -> dict:
    """Arm kwargs (arm, bfac, alpha, tau_ema) with the highest mean of final R2_5 and R2_6."""
    d = df[df.arm.isin(arms)]
    for k, v in where.items():
        d = d[d[k] == v]
    score = ((d.r2_5 + d.r2_6) / 2).groupby([d[k] for k in KEYS]).mean()
    return dict(zip(KEYS, score.idxmax(), strict=True))


def failing_setting(df) -> dict:
    """Stage 0 go/no-go: A0 must drop group 5 (R2_5 < 0.3) while keeping group 1 (R2_1 > 0.8)."""
    m = df[(df.arm == "A0") & df.heads].groupby(["lam", "w_fast"])[["r2_1", "r2_5"]].mean()
    ok = m[(m.r2_5 < 0.3) & (m.r2_1 > 0.8)]
    if ok.empty:
        raise ValueError(f"no Stage 0 setting fails as required:\n{m.round(2)}\n"
                         "lower w_fast or raise difficulty first")
    lam, w = (ok.r2_1 - ok.r2_5).idxmax()
    beta = df[(df.arm == "A0") & df.heads & (df.lam == lam) & (df.w_fast == w)].beta_cal.mean()
    return dict(lam=lam, w_fast=w, beta=beta)


def premise_check(df, **where):
    """Does slow content fill SIGReg's budget? (table, ok) from the end of the matching A0 runs.

    The failure only means something if slow groups (1-3) carry more of tr Cov(z) than the fast,
    hard groups (5-7); otherwise SIGReg is being satisfied by the fast ones and A0 keeping them
    is not a gating question. Effective dims above 2 for a slow group = redundant copies (§7).
    """
    for k, v in where.items():
        df = df[df[k] == v]
    t = pd.DataFrame({"var_share": [df[f"vs_{k}"].mean() for k in range(1, 9)],
                      "eff_dims": [df[f"ed_{k}"].mean() for k in range(1, 9)]},
                     index=pd.RangeIndex(1, 9, name="group")).round(3)
    return t, bool(t.var_share[[1, 2, 3]].sum() > t.var_share[[5, 6, 7]].sum())


def gt_table(out, diffs=(0, 1, 2)) -> pd.DataFrame:
    """Stage 0b: seed-mean t90 (steps to 0.9 * R2max) and final R2_lin per group and difficulty."""
    rows = []
    for d in diffs:
        df = pd.concat(pd.read_csv(p) for p in Path(out).glob(f"gt_diff-{d}_seed-*.csv"))
        df = df.groupby("step").mean()
        for k, m in R2MAX.items():
            hit = df.index[df[f"p_{k}"] >= 0.9 * m]
            rows.append(dict(diff=d, group=k, t90=hit[0] if len(hit) else None,
                             r2_mlp=df[f"p_{k}"].iloc[-1], r2_lin=df[f"w_{k}"].iloc[-1]))
    return pd.DataFrame(rows).set_index(["diff", "group"])


def default_difficulty(gt: pd.DataFrame) -> int:
    """Lowest difficulty with t90(5) >= 3 t90(4) (§8 check 1)."""
    for d in sorted(gt.index.get_level_values(0).unique()):
        if gt.loc[(d, 5), "t90"] >= 3 * gt.loc[(d, 4), "t90"]:
            return int(d)
    raise ValueError("t90(5) < 3 t90(4) at every difficulty: change the environment first")


# ---- decision rules (§9) --------------------------------------------------------------------
def gate_verdict(tab: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Success criterion for a gate, on seed-mean final R2 (check seed consistency by eye)."""
    a0, base = tab.loc["A0"], tab.loc[[a for a in ("A1(0)", "A1(0.1)", "A2", "C1", "C2")
                                        if a in tab.index]]
    return pd.DataFrame({a: {
        "R2_5,R2_6 >= A0 + 0.3": min(tab.loc[a, "r2_5"] - a0.r2_5, tab.loc[a, "r2_6"] - a0.r2_6)
        >= 0.3,
        "R2_5,R2_6 >= best baseline + 0.1": (tab.loc[a, ["r2_5", "r2_6"]]
                                             - base[["r2_5", "r2_6"]].max()).min() >= 0.1,
        "R2_8 <= A0 + 0.1": tab.loc[a, "r2_8"] <= a0.r2_8 + 0.1,
        "groups 1-4 lose <= 0.1": (a0[R2_COLS[:4]] - tab.loc[a, R2_COLS[:4]]).max() <= 0.1,
    } for a in arms}).T


def budget_verdict(tab: pd.DataFrame, g: str = "M1d", b: str = "B1", rand_tab=None) -> dict:
    """Success criterion for factor B at fixed gate g (§9).

    Stage 1c (D = 24) has no B-rand arm: pass the D = 16 table as `rand_tab` for that check.
    """
    gain = lambda t, lab: max(t.loc[lab, "r2_5"] - t.loc[g, "r2_5"],  # noqa: E731
                              t.loc[lab, "r2_6"] - t.loc[g, "r2_6"])
    t, ref = tab.loc[f"{g}+{b}"], tab.loc[g]
    rand_tab = tab if rand_tab is None else rand_tab
    return {"B beats G by >= 0.1 on R2_5/R2_6": gain(tab, f"{g}+{b}") >= 0.1,
            "B-rand does not": gain(rand_tab, f"{g}+B-rand") < 0.1,
            "groups 1-4 lose <= 0.1": (ref[R2_COLS[:4]] - t[R2_COLS[:4]]).max() <= 0.1,
            "R2_8 <= G + 0.1": t.r2_8 <= ref.r2_8 + 0.1}
