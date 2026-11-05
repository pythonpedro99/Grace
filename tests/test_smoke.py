"""CPU sanity checks: env statistics, gate identities and one tiny run per arm."""
import json

import pytest
import torch

from sfn.env import GROUPS, Env
from sfn.gates import Gater, Stats, slice_weights
from sfn.models import sigreg_slices
from sfn.train import Cfg, run, run_gt

ARMS = ["A0", "A1", "A2", "M1a", "M1b", "M1c", "M1d", "M2", "M3", "C1", "C2", "C3"]


def test_env_linear_r2_matches_table():
    b = Env(0).sample(200_000)
    X = torch.cat([b.gt[:, :14], torch.nn.functional.one_hot(b.a, 4).float(),
                   torch.ones(len(b.a), 1)], 1).double()
    res = b.gtn[:, :14].double() - X @ torch.linalg.lstsq(X, b.gtn[:, :14].double()).solution
    y = b.gtn[:, :14].double()
    r2 = [1 - (res[:, s] ** 2).sum() / ((y[:, s] - y[:, s].mean(0)) ** 2).sum() for s in GROUPS]
    expected = [0.96, 0.49, 0.90, 1.0, 0.41, 0.21, 0.93]
    assert all(abs(a - e) < 0.03 for a, e in zip(r2, expected, strict=False))


def test_gates_reduce_to_identity():
    d, r = 8, torch.randn(64, 8)
    lag = r * 2  # lagged residual much larger: strong progress, so all g = 0
    g = Gater("M1b", d, "cpu", warmup=0)
    assert torch.allclose(g(1, r, r, torch.eye(d)), torch.eye(d), atol=1e-5)  # no progress: G = I
    assert g(1, r, lag, torch.eye(d)).abs().max() < 1e-5
    assert torch.allclose(Gater("A0", d, "cpu")(0, r, r, None), torch.eye(d))
    assert slice_weights(torch.randn(d, 5), None, torch.zeros(d)) == 1.0


def test_stats_flags_predictable_direction():
    st = Stats(4, "cpu", every=10, tau=0.0)
    z = torch.randn(512, 4)
    pw = z.clone()
    pw[:, 1:] = 0  # weak predictor nails direction 0 only
    st.update(0, torch.randn(512, 4), z, pw)
    st.refresh()
    assert st.s.max() > 0.9 and st.s.sum() < 1.5


@pytest.mark.parametrize("arm", ARMS)
def test_tiny_run(arm, tmp_path):
    run(Cfg(arm=arm, D=8, steps=200, eval_every=100, warmup=50, heads=arm == "A0"), tmp_path)
    j = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert j["final"]["r2_1"] > -1 and "beta_cal" in j or arm != "A0"


@pytest.mark.parametrize("bfac", ["B1", "B2", "B-rand"])
def test_tiny_budget_levels(bfac, tmp_path):
    run(Cfg(arm="M1d", bfac=bfac, D=8, steps=250, eval_every=250, warmup=50), tmp_path)
    assert list(tmp_path.glob("*.json"))


def test_run_gt(tmp_path):
    run_gt(1, 0, tmp_path, steps=100, eval_every=50)
    assert len(list(tmp_path.glob("gt_*.csv"))) == 1


def test_sigreg_matches_lewm_reference():
    """Verbatim statistic of lucas-maes/le-wm module.py (SIGReg.forward), same projections."""
    z, A = torch.randn(3, 128, 8) ** 3, torch.nn.functional.normalize(torch.randn(8, 64), dim=0)
    t = torch.linspace(0, 3, 17)
    weights = torch.full((17,), 2 * 3 / 16)
    weights[[0, -1]] = 3 / 16
    phi = torch.exp(-t.square() / 2.0)
    weights = weights * phi
    x_t = (z @ A).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    ref = ((err @ weights) * z.size(-2)).mean()
    assert torch.allclose(sigreg_slices(z, A).mean(), ref, rtol=1e-5)
    assert sigreg_slices(torch.randn(2, 4096, 8), A).mean() < sigreg_slices(z, A).mean() / 10


def test_premise_check():
    import pandas as pd

    from sfn.stages import premise_check

    shares = [0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1]
    row = {f"vs_{k}": v for k, v in zip(range(1, 9), shares, strict=True)}
    row |= {f"ed_{k}": 2.0 for k in range(1, 9)}
    t, ok = premise_check(pd.DataFrame([row]))
    assert ok and t.loc[1, "var_share"] == 0.2
    row["vs_5"] = 0.9
    assert not premise_check(pd.DataFrame([row]))[1]
