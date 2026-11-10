"""CPU checks of the grid world: dynamics, rendering, caps without actions, runs of every core arm."""
import json
import math

import pytest
import torch

from gaf.env import GROUPS
from gaf.env_grid import DIRS, EMPTY, MOVES, N_MAT, STONE, TC, GridCfg, GridEnv, component_masses
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
    assert clip.a.shape == (3000, c.T - 1, 5) and set(clip.f) == {*GROUPS, "local", "map", "hand_visible", "tile_class"}
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


def test_frame_difference_reconstruction(tmp_path):
    base = Cfg(arm="retain_open", env=SMALL, D=8)
    assert Cfg(arm="retain_open", env=SMALL, D=8, rec_diff=False).run_id == base.run_id  # old ids stay valid
    assert Cfg(arm="retain_open", env=SMALL, D=8, rec_diff=True).run_id != base.run_id
    cfg = Cfg(arm="retain_open", env=SMALL, D=8, steps=12, batch=16, floor_start=5, floor_every=5,
              floor_steps=5, floor_pool=128, eval_every=6, diag_every=3, min_dwell=2, window=5,
              probe_n=128, mlp_n=256, mlp_steps=5, ret_every=4, rec_diff=True, shadow_rec=4.0)
    f = run(cfg, tmp_path, verbose=False)["final"]
    assert math.isfinite(f["shadow_rec_mse"]) and f["shadow_rec_mse"] > 0
    assert all(math.isfinite(f[f"sh_r2_{g}"]) for g in GROUPS)
    s = json.loads((tmp_path / cfg.run_id / "summary.json").read_text())
    assert all(math.isfinite(s[f"sh_mlp_r2_{g}"]) for g in GROUPS)  # the shadow's nonlinear probe
    assert "shadow" in torch.load(tmp_path / cfg.run_id / "model.pt")


MASS = GridCfg(W=12, H=8, n_slow=16, hand_tiles=4, compass_tiles=3)  # the frame of grid_mass_law.ipynb (48 x 48)


def _tile(img, c, row, col):  # (..., 3, res, res) in [0, 1] -> the colour of tile (row, col) at its top-left pixel
    return img[..., :, row * c.tile, col * c.tile]


def test_hand_and_compass_tiles_show_the_hand_and_the_facing():
    env = GridEnv(MASS, "cpu")
    torch.manual_seed(2)
    clip = env.sample(64)
    img = (clip.x + 1) / 2
    hand, face = clip.f["hard"].argmax(-1), clip.f["fast"].argmax(-1)
    c = MASS
    for j in range(c.n_slow, c.n_slow + c.hand_tiles):
        row, col = c.H + j // c.W, j % c.W
        assert torch.allclose(_tile(img, c, row, col), env.mat_col[hand], atol=1e-5)
        assert (clip.f["tile_class"][:, :, row * c.W + col] == TC["hand:floor"] + hand).all()
    for j in range(c.n_slow + c.hand_tiles, c.n_slow + c.hand_tiles + c.compass_tiles):
        row, col = c.H + j // c.W, j % c.W
        assert torch.allclose(_tile(img, c, row, col), env.compass_col[face], atol=1e-5)
        assert (clip.f["tile_class"][:, :, row * c.W + col] == TC["compass:up"] + face).all()
    with pytest.raises(AssertionError):
        GridEnv(GridCfg(n_slow=16, hand_tiles=4, compass_tiles=1), "cpu")  # 21 tiles > the 20 of a 10 x 10 frame


def test_hand_presence_hides_the_hand_tiles_per_clip():
    env = GridEnv(GridCfg(W=12, H=8, hand_tiles=2, hand_p=0.3), "cpu")
    torch.manual_seed(3)
    clip = env.sample(2000)
    vis = clip.f["hand_visible"][:, 0, 0] > 0.5
    assert 0.2 < vis.float().mean() < 0.4
    assert (clip.f["hand_visible"] == clip.f["hand_visible"][:, :1]).all()  # per clip, not per frame
    row, col = 8 + 16 // 12, 16 % 12
    img = (clip.x + 1) / 2
    assert torch.allclose(_tile(img, env.cfg, row, col)[~vis], env.hand_mean.expand_as(_tile(img, env.cfg, row, col)[~vis]), atol=1e-5)
    assert (clip.f["tile_class"][~vis][:, :, row * 12 + col] == TC["hud:empty"]).all()


def test_tile_classes_follow_what_is_drawn():
    env = GridEnv(GridCfg(), "cpu")
    torch.manual_seed(4)
    clip = env.sample(500)
    c, cls = env.cfg, clip.f["tile_class"]
    pos = _xy(clip.f["slow"], c)
    face = clip.f["fast"].argmax(-1)
    agent_idx = pos[..., 1] * c.W + pos[..., 0]
    assert (cls.gather(-1, agent_idx[..., None])[..., 0] == TC["agent:up"] + face).all()
    assert cls.max() < len(TC) and cls.min() >= 0
    assert ((cls >= TC["inventory:0"]) & (cls <= TC["inventory:9"])).sum() == 500 * c.T * c.n_slow


def test_hand_contrast_scales_the_mass_with_its_square():
    base = GridCfg(W=12, H=8, hand_tiles=2)
    m1 = component_masses(base, n=512)["hand"]
    m5 = component_masses(GridCfg(W=12, H=8, hand_tiles=2, hand_contrast=0.5), n=512)["hand"]
    assert 0.2 < m5 / m1 < 0.3
    env = GridEnv(GridCfg(W=12, H=8, hand_contrast=0.0), "cpu")  # no contrast: every clip shows the average colour
    torch.manual_seed(1)
    img = (env.sample(32).x + 1) / 2
    c = env.cfg
    assert torch.allclose(_tile(img, c, c.H + c.n_slow // c.W, c.n_slow % c.W), env.hand_mean.expand(32, c.T, 3), atol=1e-5)


def test_component_masses_grow_with_the_tiles():
    m1 = component_masses(GridCfg(W=12, H=8, hand_tiles=1), n=256)
    m4 = component_masses(GridCfg(W=12, H=8, hand_tiles=4, compass_tiles=2), n=256)
    assert set(m4) >= {"hand", "compass", "facing marker", "agent (with marker)", "mob", "twinkle", "inventory (one slot)"}
    assert all(0 < v < 1 for v in m4.values())
    assert 3 < m4["hand"] / m1["hand"] < 5  # four tiles carry about four times the mass of one


@pytest.mark.parametrize("kw", [dict(arm="ae"), dict(arm="ae", rec_class_power=0.5), dict(arm="retain_open", rec_class_power=1.0),
                                dict(arm="ae", rec_class_power=1.0, rec_clusters=8, rec_pos_types=3),
                                dict(arm="retain_open", rec_class_power=1.0, rec_clusters=8, rec_pos_types=3)])
def test_autoencoder_and_class_balanced_reconstruction_run(kw, tmp_path):
    cfg = Cfg(env=SMALL, D=8, steps=10, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128,
              mlp_n=256, mlp_steps=5, ret_every=4, lam=0.01, **kw)
    s = run(cfg, tmp_path, verbose=False)
    assert all(math.isfinite(s["final"][f"r2_{g}"]) for g in GROUPS)
    assert {"pm_hand", "pm_facing", "pm_mob", "pm_rank"} <= set(s)
    if kw["arm"] == "ae":
        assert math.isfinite(s["final"]["ae_rec_mse"])
    else:
        assert "sh_pm_hand" in s


def test_class_weights_lift_the_rare_classes():
    from gaf.train import class_weights
    torch.manual_seed(0)
    w, share = class_weights(GridEnv(GridCfg(), "cpu"), 1.0, n=256)
    assert w[TC["agent:left"]] > w[TC["map:floor"]]  # an agent tile is rare, the floor is everywhere
    assert w.max() <= 30 + 1e-4 and abs(share.sum().item() - 1) < 1e-4


def test_cluster_classes_find_the_hand_without_labels():
    """Label-free classes: the hand is a tile that looks like a map tile, so appearance alone cannot isolate it; the position
    type (by change rate) does, and the hand then gets a far larger weight than the map terrain."""
    from gaf.train import assign_tiles, class_weights, cluster_classes
    torch.manual_seed(0)
    env = GridEnv(GridCfg(W=12, H=8, hand_tiles=1), "cpu")
    cl, w, share = cluster_classes(env, 16, 1.0, n=256, n_fit=10000, k_pos=6)
    clip = env.sample(128)
    cls, gt = assign_tiles(clip.x, cl, env.cfg), clip.f["tile_class"]
    assert cls.shape == gt.shape and cls.max() < 16 * 6
    hand, terrain = torch.isin(gt, torch.tensor([TC["hand:floor"], TC["hand:wood"], TC["hand:stone"], TC["hand:water"]])), gt < 4
    assert w[cls[hand]].mean() > 10 * w[cls[terrain]].mean()
    _, w1, _ = cluster_classes(env, 16, 1.0, n=256, n_fit=10000, k_pos=1)  # appearance only: the hand sits in the terrain clusters
    cls1 = assign_tiles(clip.x, (cl[0], torch.zeros_like(cl[1]), 16), env.cfg)
    assert w1[cls1[hand]].mean() < 3 * w1[cls1[terrain]].mean()
    wg, _ = class_weights(env, 1.0, n=256)
    assert abs(share.sum().item() - 1) < 1e-4 and wg.max() <= 30 + 1e-4


def test_visible_hand_probe():
    from gaf.probes import parts_probe
    env = GridEnv(GridCfg(W=12, H=8, hand_tiles=2, hand_p=0.5), "cpu")
    enc = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 48 * 48, 8))
    out = parts_probe(enc, env, n_train=256, n_test=128, steps=5)
    assert "pm_hand_visible" in out and "pm_rank" in out
