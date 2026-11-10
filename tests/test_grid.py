"""CPU checks of the grid world: dynamics, rendering, caps without actions, runs of every core arm."""
import math

import pytest
import torch

from gaf.env import GROUPS
from gaf.env_grid import DIRS, EMPTY, MOVES, N_MAT, STONE, GridCfg, GridEnv
from gaf.train import Cfg, capacity, run, run_gt

SMALL = GridCfg(W=8, H=6, n_slow=8)  # 32 x 32 frame: fast enough for tiny runs


def _xy(f, c):  # slow factors -> integer agent position (n, T, 2)
    p = (f[..., -2:] + 1) / 2
    return torch.stack([p[..., 0] * (c.W - 1), p[..., 1] * (c.H - 1)], -1).round().long()


def _material(m, p, c):  # material at xy p of maps m; off the map is stone
    out = torch.full(p.shape[:-1], STONE)
    on = (p[..., 0] >= 0) & (p[..., 0] < c.W) & (p[..., 1] >= 0) & (p[..., 1] < c.H)
    i = on.nonzero(as_tuple=True)[0]
    out[i] = m[i, p[i, 1], p[i, 0]]
    return out


@pytest.mark.parametrize("random_map", [True, False])
def test_grid_dynamics(random_map):
    c = GridCfg(random_map=random_map)
    env = GridEnv(c, "cpu")
    torch.manual_seed(0)
    clip = env.sample(3000)
    assert clip.x.shape == (3000, c.T, 3, c.res, c.res) and clip.x.abs().max() <= 1
    assert clip.a.shape == (3000, c.T - 1, 5) and set(clip.f) == {*GROUPS, "local", "map"}
    assert clip.f["map"].shape == (3000, c.T, c.H * c.W * N_MAT)
    assert {g: clip.f[g].shape[-1] for g in GROUPS} == env.dims
    a, face = clip.a.argmax(-1), clip.f["fast"].argmax(-1)
    p, hand = _xy(clip.f["slow"], c), clip.f["hard"].argmax(-1)
    # the map is not returned; rebuild the material of each visited cell from the local factor
    local = clip.f["local"].view(3000, c.T, 8, N_MAT).argmax(-1)  # k=1 dirs 0..3, then k=2
    for t in range(c.T - 1):
        mv = a[:, t] > 0
        assert (face[mv, t + 1] == a[mv, t] - 1).all()  # a move turns the agent ...
        assert (face[~mv, t + 1] == face[~mv, t]).all()  # ... a noop keeps the facing
        step = p[:, t + 1] - p[:, t]
        assert (step.abs().sum(-1) <= 1).all()
        target = local[:, t].gather(1, (a[:, t] - 1).clamp(min=0)[:, None])[:, 0]  # cell ahead
        moved = step.abs().sum(-1) == 1
        assert (moved == (mv & (target == EMPTY))).all()  # moves iff the cell ahead is floor
        assert (step[moved] == MOVES[a[moved, t]]).all()
    for t in range(c.T):  # the hand is the material ahead: local[k=1, facing]
        assert (hand[:, t] == local[:, t].gather(1, face[:, t:t + 1])[:, 0]).all()
    hud = clip.f["slow"][..., :c.n_slow]
    assert ((hud[:, 1:] - hud[:, :-1]).abs() > 0).float().mean() < 2 * c.hud_p
    if not random_map:
        assert (_material(env.fixed_map.expand(3000, -1, -1), p[:, 0] + DIRS[face[:, 0]], c)
                == hand[:, 0]).all()


def test_grid_rendering_shows_agent_and_hand():
    c = GridCfg()
    env = GridEnv(c, "cpu")
    torch.manual_seed(1)
    clip = env.sample(64)
    img = (clip.x + 1) / 2  # (n, T, 3, res, res)
    p, face = _xy(clip.f["slow"], c), clip.f["fast"].argmax(-1)
    for i in range(8):
        tile = img[i, 0, :, p[i, 0, 1] * c.tile:(p[i, 0, 1] + 1) * c.tile,
                   p[i, 0, 0] * c.tile:(p[i, 0, 0] + 1) * c.tile]
        yellow = tile[2] < 0.1  # blue channel: 0 on the facing half, 1 on the white half
        assert yellow.sum() == c.tile * c.tile // 2
        half = env.face_mask[face[i, 0], :c.tile, :c.tile]
        assert torch.equal(yellow, half)
    k = c.n_slow  # the hand slot follows the inventory slots
    hy, hx = c.H + k // c.W, k % c.W
    hand_px = img[:, :, :, hy * c.tile, hx * c.tile]  # (n, T, 3)
    mat_col = env.mat_col[clip.f["hard"].argmax(-1)]
    assert torch.allclose(hand_px, mat_col, atol=1e-5)


def test_grid_needs_actions():
    """Facing and hand are action-driven (low R^2 without actions); position and HUD are slow."""
    no = run_gt(SMALL, steps=600, eval_every=600, actions=False).iloc[-1]
    assert no["fast"] < 0.15 and no["hard"] < 0.4 and no["slow"] > 0.9 and no["noise"] < 0.05


@pytest.mark.parametrize("arm", ["lewm", "sg", "ema", "detached", "target", "retain_open"])
def test_grid_tiny_run(arm, tmp_path):
    cfg = Cfg(arm=arm, env=SMALL, D=8, steps=12, batch=16, floor_start=5, floor_every=5,
              floor_steps=5, floor_pool=128, eval_every=6, diag_every=3, min_dwell=2, window=5,
              probe_n=128, mlp_n=256, mlp_steps=5, ret_every=4,
              head_start=4 if arm == "target" else 0)
    f = run(cfg, tmp_path, verbose=False)["final"]
    assert all(math.isfinite(f[f"r2_{g}"]) for g in GROUPS)
    assert Cfg(arm=arm, env=SMALL).run_id != Cfg(arm=arm).run_id


def test_parts_r2(tmp_path):
    from gaf.analysis import parts_r2
    cfg = Cfg(arm="sg", env=SMALL, D=8, steps=6, batch=16, floor_start=3, floor_every=3,
              floor_steps=3, floor_pool=64, eval_every=6, probe_n=64, mlp_n=128, mlp_steps=3)
    run(cfg, tmp_path, verbose=False)
    t = parts_r2({"sg": cfg}, tmp_path, n=128)
    assert list(t.columns) == ["inventory", "position", "facing", "hand", "mob", "twinkle"]


def test_capacity_runs():
    df = capacity(SMALL, Ds=(4, 8), steps=6, eval_every=3, n_eval=64, batch=16, desc=False)
    assert list(df.D.unique()) == [4, 8] and df.filter(like="pred_").notna().all().all()


def test_latent_determined_by(tmp_path):
    from gaf.analysis import latent_determined_by
    cfg = Cfg(arm="sg", env=SMALL, D=8, steps=6, batch=16, floor_start=3, floor_every=3,
              floor_steps=3, floor_pool=64, eval_every=6, probe_n=64, mlp_n=128, mlp_steps=3)
    run(cfg, tmp_path, verbose=False)
    t = latent_determined_by({"sg": cfg}, tmp_path, n=128, steps=5)
    assert list(t.columns) == ["map", "inventory+position", "facing+hand", "mob+twinkle", "all"]


def test_shadow_reconstruction_mse_is_logged(tmp_path):
    cfg = Cfg(arm="retain_open", env=SMALL, D=8, steps=12, batch=16, floor_start=5, floor_every=5,
              floor_steps=5, floor_pool=128, eval_every=6, diag_every=3, min_dwell=2, window=5,
              probe_n=128, mlp_n=256, mlp_steps=5, ret_every=4)
    f = run(cfg, tmp_path, verbose=False)["final"]
    assert math.isfinite(f["shadow_rec_mse"]) and f["shadow_rec_mse"] > 0
