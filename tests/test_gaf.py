"""CPU checks: env dynamics, SIGReg reference, projection identity, gate logic, floors, arms."""
import json
import math

import pytest
import torch

from gaf.env import GROUPS, ClipEnv, EnvCfg
from gaf.gate import FloorGate, excess_basis, floor_probe
from gaf.grads import SplitAdamW, project
from gaf.models import sigreg_slices
from gaf.train import ARMS, Cfg, premise, run, run_gt

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
    assert c.a.shape == (2000, 2, 2) and c.a.abs().max() <= 1
    p, q, e = _angles(c.f["fast"]), _angles(c.f["hard"]), env.cfg
    for t in range(2):  # action-driven dynamics
        assert _circ(p[:, t + 1], p[:, t] + e.v_act * c.a[:, t]).max() < 1e-3
        assert _circ(q[:, t + 1], e.hard_k * q[:, t] + e.h_act * c.a[:, t]).max() < 1e-3
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
    tight = premise(EnvCfg(n_slow=16), D=16)  # slow alone is an isotropic Gaussian
    assert tight["slow"] < 3 * tight["gaussian"] + 1 and tight["fast+hard"] > 5 * tight["slow"]
    roomy = premise(EnvCfg(n_slow=16), D=32)  # linear slow content leaves room ...
    assert roomy["slow"] > 30 * roomy["gaussian"]
    assert roomy["slow+copies"] < 10 * roomy["gaussian"]  # ... that nonlinear copies can fill


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


def _params(seed=0):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(5, 3)), torch.nn.Parameter(torch.randn(4))]


def test_split_adamw_is_adamw_without_projection():
    ps, ref = _params(), _params()
    opt, opt_ref = SplitAdamW(ps, 1e-2, wd=0.1), torch.optim.AdamW(ref, 1e-2, weight_decay=0.1)
    for k in range(5):
        g = torch.randn(19, generator=torch.Generator().manual_seed(k))
        opt.step(0.3 * g, 0.7 * g)  # split arbitrarily: momentum is linear
        ref[0].grad, ref[1].grad = g[:15].view(5, 3).clone(), g[15:].clone()
        opt_ref.step()
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ps, ref, strict=True))


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_split_adamw_projection_controls_actual_step(alpha):
    """The leak of the applied context update (momentum and weight decay included) is
    (1 - alpha) times the leak before projection."""
    ps = _params()
    opt = SplitAdamW(ps, 1e-2, wd=0.1)
    gen = torch.Generator().manual_seed(1)
    for _ in range(4):
        g_S = torch.randn(19, generator=gen)
        g_ctx = g_S + torch.randn(19, generator=gen)  # correlated: leaks
        info = opt.step(g_ctx, torch.randn(19, generator=gen), g_S, alpha)
        if info["leak_pre"] > 0:
            tol = 1e-5 * info["leak_pre"]
            assert info["leak_post"] == pytest.approx((1 - alpha) * info["leak_pre"], abs=tol)
            assert torch.dot(g_S, info["u_ctx"]).item() == pytest.approx(info["leak_post"],
                                                                        abs=tol)


def test_dynamics_need_actions():
    """Without the actions, fast and hard stay below their cap of 0.25; slow does not."""
    r = run_gt(SMALL, steps=300, eval_every=300, actions=False).iloc[-1]
    assert r["fast"] < 0.35 and r["hard"] < 0.35 and r["slow"] > 0.8


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_excess_basis_finds_unlearned_direction(scale):
    """Noise (high floor, P at the floor) and learned content rank low; the direction where the
    predictor is far above its floor ranks first, although the floor alone cannot tell it apart.
    lam is the missed share of Cov(z): a latent with twice the variance halves it."""
    R = torch.linalg.qr(torch.randn(4, 4, generator=torch.Generator().manual_seed(0)))[0]
    C_star = R @ torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0])) @ R.T  # dir 0: noise floor
    C_P = R @ torch.diag(torch.tensor([1.0, 0.5, 0.0, 0.0])) @ R.T  # dir 1: not learned yet
    Q, lam = excess_basis(C_P, C_star, scale * torch.eye(4))
    assert torch.allclose(Q.T @ Q, torch.eye(4), atol=1e-5)
    assert abs(torch.dot(Q[:, 0], R[:, 1])) > 0.999
    assert lam[0] == pytest.approx(0.5 / scale, rel=1e-4) and lam[1:].abs().max() < 1e-5


def test_excess_basis_uses_the_latent_metric():
    """With a non-white latent the solutions are Cov(z)-orthonormal before QR; the top direction
    is the one with the largest missed share of its own variance, not of the Euclidean norm."""
    C_z = torch.diag(torch.tensor([4.0, 1.0]))
    C_P = torch.diag(torch.tensor([1.0, 0.5]))  # misses 1/4 of dim 0, 1/2 of dim 1
    Q, lam = excess_basis(C_P, torch.zeros(2, 2), C_z)
    assert abs(Q[1, 0]) > 0.999 and lam.tolist() == pytest.approx([0.5, 0.25], rel=1e-5)


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
    Ls, _, _ = floor_probe(_Oracle(env), _Oracle(env), env, steps=300, n_pool=2048,
                           basis="dim")
    slow, noise = Ls[:4], Ls[4:]
    assert slow.max() < 0.15 and noise.min() > 0.5 * noise.max() > 0.3


@pytest.mark.parametrize("arm", list(ARMS))
def test_tiny_run(arm, tmp_path):
    cfg = Cfg(arm=arm, env=SMALL, D=8, steps=40, batch=32, floor_start=10, floor_every=10,
              floor_steps=10, floor_pool=256, eval_every=20, diag_every=5, min_dwell=2,
              window=5, basis={"both": "resid", "context": "dim"}.get(arm, "excess"),
              probe_n=256, mlp_n=512,
              mlp_steps=20)
    s = run(cfg, tmp_path, verbose=False)
    f = s["final"]
    assert all(math.isfinite(f[f"r2_{g}"]) for g in GROUPS)
    if arm in ("context", "both"):  # projection is M-orthogonal: the Euclidean norm may grow
        assert 0 <= f["leak_pos"] <= 1 and math.isfinite(f["proj_ratio"])
    assert json.loads((tmp_path / cfg.run_id / "summary.json").read_text())["run_id"] == cfg.run_id


TINY = dict(env=SMALL, D=8, steps=10, batch=16, floor_start=5, floor_every=5, floor_steps=5,
            floor_pool=128, eval_every=10, diag_every=5, min_dwell=2, window=5, probe_n=128,
            mlp_n=256, mlp_steps=5)


def test_on_step_reports_every_phase(tmp_path):
    calls, lines = [], []
    run(Cfg(arm="sg", **TINY), tmp_path, on_step=lambda s, ph: calls.append((s, ph)),
        log=lines.append)
    steps = [s for s, _ in calls]
    assert steps == sorted(steps) and steps[-1] == 10
    assert {ph for _, ph in calls} == {"train", "floor probe", "eval", "final probes", "done"}
    assert [s for s, ph in calls if ph == "train"] == list(range(10))
    assert len(lines) == 1 and lines[0].lstrip().startswith("[sg s0]")  # eval line via log


def test_execute_sequential_shows_a_bar(tmp_path, capsys):
    from gaf.analysis import collect, execute
    execute([Cfg(arm="sg", seed=0, **TINY), Cfg(arm="detached", seed=0, **TINY)], tmp_path)
    err = capsys.readouterr().err
    assert "training" in err and "100%" in err and len(collect(tmp_path)) == 2


def test_parallel_execute(tmp_path):
    import json

    from gaf.analysis import collect, execute
    cfgs = [Cfg(arm="sg", seed=0, **TINY), Cfg(arm="both", seed=0, **TINY)]
    execute(cfgs, tmp_path, workers=2)
    assert len(collect(tmp_path)) == 2
    assert all((tmp_path / c.run_id / "log.txt").read_text() for c in cfgs)
    for c in cfgs:  # the worker reported its last phase to the parent through progress.json
        assert json.loads((tmp_path / c.run_id / "progress.json").read_text()) == {
            "step": 10, "phase": "done"}
    execute(cfgs, tmp_path, workers=2)  # everything done: nothing runs again


def test_parallel_execute_reports_failures(tmp_path):
    from gaf.analysis import execute
    good, bad = Cfg(arm="sg", seed=0, **TINY), Cfg(arm="no-such-arm", seed=1, **TINY)
    with pytest.raises(RuntimeError, match="1 run"):
        execute([good, bad], tmp_path, workers=2, progress=False)
    assert (tmp_path / good.run_id / "summary.json").exists()  # the other run still finished


def test_new_fields_keep_old_run_ids():
    """head_start / early_eval join the hash only when set: finished runs stay valid."""
    assert Cfg(arm="sg").run_id == "sg_s0_f6d8b336"  # ids recorded before the fields existed
    assert Cfg(arm="both", seed=2, D=16, tag="x").run_id == "both_s2_a39c1fae"
    assert Cfg(arm="sg", head_start=100).run_id != Cfg(arm="sg").run_id
    assert Cfg(arm="sg", early_eval=(0, 5)).run_id != Cfg(arm="sg").run_id


def test_head_start_switches_the_encoder_off(tmp_path):
    """During the head start the encoder gets only SIGReg: no target path, no leak diagnostics."""
    off = run(Cfg(arm="sg", head_start=10, **TINY), tmp_path, verbose=False)["final"]
    on = run(Cfg(arm="sg", **TINY), tmp_path, verbose=False)["final"]
    assert "leak_cos" not in off and "leak_cos" in on


def test_early_eval_and_leak_decomposition(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from gaf.analysis import collect, snapshot, trace
    cfg = Cfg(arm="both", early_eval=(0, 5), **TINY)
    f = run(cfg, tmp_path, verbose=False)["final"]
    keys = ("lk_ctx_S", "lk_rest_S", "lk_ctx_open", "lk_rest_open")
    assert all(math.isfinite(f[k]) for k in keys)
    early = __import__("pandas").read_csv(tmp_path / cfg.run_id / "early.csv")
    assert early.step.tolist() == [0, 5] and "r2_fast" in early
    df = collect(tmp_path)
    runs = {"both": dict(arm="both")}
    snap = snapshot(df, tmp_path, runs, steps=(0, 5, 10), cols=("r2_fast",))
    assert snap.loc["both", ["fast@0", "fast@5", "fast@10"]].notna().all()  # early + regular evals
    assert trace(df, tmp_path, runs) is not None


def test_block_is_detached_while_everything_is_protected(tmp_path):
    """No floor yet: every direction is in S, so the masked loss sends the encoder nothing and
    only SIGReg shapes it, exactly as in `detached` (the predictor trains the same way)."""
    kw = {**TINY, "floor_start": 1000}
    ids = {}
    for arm in ("block", "detached"):
        cfg = Cfg(arm=arm, **kw)
        run(cfg, tmp_path, verbose=False)
        ids[arm] = torch.load(tmp_path / cfg.run_id / "model.pt")
    for k, v in ids["block"]["enc"].items():
        assert torch.allclose(v, ids["detached"]["enc"][k], atol=1e-6), k
    for k, v in ids["block"]["P"].items():
        assert torch.allclose(v, ids["detached"]["P"][k], atol=1e-6), k


def test_swap_colors_swaps_the_sprites():
    a, b = ClipEnv(EnvCfg(), "cpu"), ClipEnv(EnvCfg(swap_colors=True), "cpu")
    assert torch.equal(a.col_fast, b.col_hard) and torch.equal(a.col_hard, b.col_fast)
    assert torch.allclose(b.col_fast.norm(), b.col_hard.norm())  # same energy, only colour differs
    assert Cfg(arm="sg", env=EnvCfg(swap_colors=True)).run_id != Cfg(arm="sg").run_id


def test_latent_learning(tmp_path):
    from gaf.analysis import latent_learning
    cfg = Cfg(arm="detached", **TINY)
    run(cfg, tmp_path, verbose=False)
    res = latent_learning(cfg, tmp_path, steps=20, eval_every=10, n_eval=256, batch=32)
    assert res.index.tolist() == [10, 20]
    assert {"slow", "fast", "hard", "noise", "mse"} <= set(res.columns)
    assert res.notna().all().all() and set(res.attrs["availability"]) == set(GROUPS)
