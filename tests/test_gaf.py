"""CPU checks: env dynamics, SIGReg reference, projection identity, gate logic, floors, arms."""
import json
import math

import pytest
import torch

from gaf.env import GROUPS, ClipEnv, EnvCfg
from gaf.gate import FloorGate, floor_probe
from gaf.grads import project
from gaf.models import sigreg_slices
from gaf.train import ARMS, Cfg, premise, run

SMALL = EnvCfg(res=16)


def _angles(f):  # (n, T, 4) cos/sin embedding -> (n, T, 2) positions in [0, 1)
    return (torch.atan2(f[..., 1::2], f[..., 0::2]) / (2 * math.pi)) % 1


def _circ(a, b):  # circular distance on [0, 1)
    return ((a - b + 0.5) % 1 - 0.5).abs()


def test_env_dynamics():
    env = ClipEnv(EnvCfg(), "cpu")
    torch.manual_seed(0)
    c = env.sample(2000)
    assert c.x.shape == (2000, 3, 3, 32, 32)
    p, q = _angles(c.f["fast"]), _angles(c.f["hard"])
    assert _circ(p[:, 2], 2 * p[:, 1] - p[:, 0]).max() < 1e-3  # constant velocity
    k = env.cfg.hard_k
    assert _circ(q[:, 2], (k + 1) * q[:, 1] - k * q[:, 0]).max() < 1e-3  # chaotic map
    s, n = c.f["slow"], c.f["noise"]
    rho = torch.corrcoef(torch.stack([s[:, 0].flatten(), s[:, 1].flatten()]))[0, 1]
    assert abs(rho - env.cfg.slow_rho) < 0.02
    assert torch.corrcoef(torch.stack([n[:, 0].flatten(), n[:, 1].flatten()]))[0, 1].abs() < 0.03
    assert set(c.f) == set(GROUPS)


def test_patterns_linearly_decodable():
    """Orthonormal bases: slow and noise factors are read out exactly by x @ Q.T."""
    env = ClipEnv(EnvCfg(a_fast=0, a_hard=0, pixel_noise=0), "cpu")
    c = env.sample(64)
    z = c.x.flatten(2) @ env.Q.T  # (n, T, n_slow + n_noise), scaled factors
    s = z[..., :env.cfg.n_slow] / z[..., :env.cfg.n_slow].std()
    assert torch.corrcoef(torch.stack([s.flatten(), c.f["slow"].flatten()]))[0, 1] > 0.999


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


def test_premise_slow_satisfies_sigreg():
    s = premise(EnvCfg(n_slow=16), D=16)
    assert s["slow"] < 3 * s["gaussian"] + 1 and s["fast+hard"] > 5 * s["slow"]


@pytest.mark.parametrize("metric", [None, "rand"])
@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_projection_removes_leak(metric, alpha):
    torch.manual_seed(0)
    g_t = torch.randn(100)
    g_c = 0.5 * g_t + torch.randn(100)
    m = torch.rand(100) + 0.1 if metric else None
    w = 1.0 if m is None else m
    ell0 = torch.dot(g_t, w * g_c)
    g_new, ell = project(g_c, g_t, alpha, m)
    assert ell == pytest.approx(ell0.item(), rel=1e-5) and ell > 0
    assert torch.dot(g_t, w * g_new).item() == pytest.approx((1 - alpha) * ell, abs=1e-4)
    g_same, _ = project(-g_c, g_t, alpha, m)  # negative ell: nothing projected
    assert torch.equal(g_same, -g_c)


def test_gate_hysteresis_and_floor():
    g = FloorGate(2, "cpu", window=1, eps_g=0.1, eps_f=1.0, delta=0.0, min_dwell=3)
    L = torch.tensor([1.0, 1.0])
    for _ in range(5):  # no floor yet: everything protected
        g.update(L, L)
    assert g.inS.all()
    g.set_floor(torch.tensor([1.0, 0.1]) / 0.9)  # margin 0.1 -> L* = (1.0, 0.1)
    g.update(L, L)  # dir 0 at its floor and no progress; dir 1 far above it
    assert g.inS.tolist() == [False, True]
    g.update(L, 1.5 * L)  # progress 0.5 > eps_g: dir 0 re-enters S
    assert g.inS.tolist() == [True, True]
    g.update(L, L)  # dwell not reached: stays
    assert g.inS[0]
    for _ in range(3):
        g.update(L, L)
    assert not g.inS[0]


def test_gate_rotation_keeps_state():
    g = FloorGate(2, "cpu")
    g.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    g.set_floor(torch.tensor([0.5, 0.5]))
    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    g.set_floor(torch.tensor([0.4, 0.6]), swap)
    assert torch.allclose(g.Lbar, torch.tensor([2.0, 1.0]))
    assert torch.allclose(g.Lstar, torch.tensor([0.4, 0.5]) * 0.9)


class _Oracle(torch.nn.Module):
    """Reads out slow and noise factors exactly: z = x @ Q.T (standardised)."""

    def __init__(self, env):
        super().__init__()
        self.Q, self.scale = env.Q, env.B_slow.norm(dim=1).mean()

    def forward(self, x):
        return x.flatten(-3) @ self.Q.T / self.scale


def test_floor_probe_finds_noise_floor():
    cfg = EnvCfg(res=16, n_slow=4, n_noise=4, a_noise=1.0, a_fast=0, a_hard=0, pixel_noise=0)
    env = ClipEnv(cfg, "cpu")
    torch.manual_seed(0)
    Ls, _ = floor_probe(_Oracle(env), _Oracle(env), env, steps=300, n_pool=2048)
    slow, noise = Ls[:4], Ls[4:]
    assert slow.max() < 0.15 and noise.min() > 0.5 * noise.max() > 0.3


@pytest.mark.parametrize("arm", list(ARMS))
def test_tiny_run(arm, tmp_path):
    cfg = Cfg(arm=arm, env=SMALL, D=8, steps=40, batch=32, floor_start=10, floor_every=10,
              floor_steps=10, floor_pool=256, eval_every=20, diag_every=5, min_dwell=2,
              window=5, basis="resid" if arm == "both" else "dim", probe_n=256, mlp_n=512,
              mlp_steps=20)
    s = run(cfg, tmp_path, verbose=False)
    f = s["final"]
    assert all(math.isfinite(f[f"r2_{g}"]) for g in GROUPS)
    if arm in ("context", "both"):
        assert 0 <= f["leak_pos"] <= 1 and f["proj_ratio"] <= 1 + 1e-5
    assert json.loads((tmp_path / cfg.run_id / "summary.json").read_text())["run_id"] == cfg.run_id
