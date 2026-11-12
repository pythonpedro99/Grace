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


MASS = GridCfg(W=12, H=8, n_slow=16, hand_tiles=4, compass_tiles=3)  # the frame of notebooks/04_mass_law.ipynb (48 x 48)


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
                                dict(arm="retain_open", rec_class_power=1.0, rec_clusters=8, rec_pos_types=3),
                                dict(arm="retain_open", rec_class_power=1.0, ret_ac=0.1, ret_w_min=0.3),
                                dict(arm="retain_open", ret_floor_all=True, ret_w_min=0.3)])
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
        assert "ref_ac_n" in s["final"] and "ref_w_slow" in s["final"]


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


def test_arrow_agent_encodes_the_facing_in_its_silhouette():
    env = GridEnv(GridCfg(W=12, H=8, agent_shape="arrow"), "cpu")
    clip = env.sample(64)
    cls = clip.f["tile_class"][:, -1]
    for d in range(4):  # the four silhouettes differ, and each is a quarter-to-three-quarters of the tile
        m = env.arrow_mask[d, :4, :4]
        assert 6 <= int(m.sum()) <= 12
        assert all((m != env.arrow_mask[e, :4, :4]).any() for e in range(4) if e != d)
    assert ((cls >= TC["agent:up"]) & (cls <= TC["agent:right"])).flatten(1).any(1).all()  # the agent is drawn in every frame
    m = component_masses(GridCfg(W=12, H=8, agent_shape="arrow"), n=128)
    assert m["facing marker"] > 0.02  # the silhouette carries a few percent of the pixel variance


def test_chase_mobs_persist_move_by_rule_and_are_predictable():
    c = GridCfg(W=12, H=8, mob_mode="chase", n_mobs=2, mob_chase=0.8, T=6)
    env = GridEnv(c, "cpu")
    torch.manual_seed(0)
    clip = env.sample(512)
    assert env.dims["fast"] == 4 + 4 and env.dims["noise"] == c.n_twinkle
    assert clip.f["fast"].shape[-1] == env.dims["fast"] and clip.f["noise"].shape[-1] == env.dims["noise"]
    g, s = env.parts()["mob"]
    assert g == "fast" and s == slice(4, 8)
    mob = clip.f["fast"][..., 4:]  # (n, T, 4) xy in [-1, 1] of both mobs
    xy = ((mob.view(*mob.shape[:2], 2, 2) + 1) / 2 * torch.tensor([c.W - 1, c.H - 1])).round().long()
    pos = ((clip.f["slow"][..., -2:] + 1) / 2 * torch.tensor([c.W - 1, c.H - 1])).round().long()
    assert (xy[..., 1] < c.H).all() and (xy[..., 0] < c.W).all()
    assert not (xy == pos[:, :, None]).all(-1).any()  # a mob never shares the agent's cell
    step = (xy[:, 1:] - xy[:, :-1]).abs().sum(-1)
    assert step.max() <= 1  # one cell per frame
    d0 = (pos[:, 1:, None] - xy[:, :-1]).abs().sum(-1)
    d1 = (pos[:, 1:, None] - xy[:, 1:]).abs().sum(-1)
    moved = step > 0
    assert ((d1 < d0) & moved).sum() / moved.sum() > 0.6  # the moves are mostly toward the agent: predictable
    assert (clip.f["tile_class"][:, -1] == TC["mob"]).sum(-1).float().mean() > 1.5  # both drawn


def test_default_grid_env_is_unchanged_by_the_new_knobs():
    env = GridEnv(GridCfg(), "cpu")
    assert env.dims == {"slow": 18, "fast": 4, "hard": 4, "noise": 6}
    assert env.parts()["mob"] == ("noise", slice(0, 2))


def test_eval_time_part_probes_are_logged_and_curve_is_written_while_running(tmp_path):
    import pandas as pd
    cfg = Cfg(arm="retain_open", env=SMALL, D=8, steps=10, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128,
              mlp_n=256, mlp_steps=5, ret_every=4, lam=0.01, eval_parts=256)
    seen = []

    def on_step(step, phase):
        f = tmp_path / cfg.run_id / "metrics.csv"
        if phase == "eval" and step > 5 and f.exists():
            seen.append(len(pd.read_csv(f)))  # the first eval's row is on disk while the run goes on

    s = run(cfg, tmp_path, verbose=False, on_step=on_step)
    h = pd.read_csv(tmp_path / cfg.run_id / "metrics.csv")
    assert {"ev_pm_hand", "ev_pm_mob", "ev_sh_pm_hand", "ev_sh_pm_position"} <= set(h.columns) and len(h) == 2
    assert seen and seen[0] >= 1 and "pm_hand" in s


def test_is_settled_follows_change_and_swing_of_the_part_probes():
    from gaf.train import is_settled
    flat = [{"step": 1000 * i, "ev_pm_hand": 0.9 + 0.01 * (i % 2), "ev_sh_pm_hand": 0.8, "ev_pm_rank": 5 * i} for i in range(1, 12)]
    assert is_settled(flat, 4000, 8000)  # moves by 0.01: settled
    assert not is_settled(flat, 4000, 20_000)  # not enough steps yet
    assert not is_settled(flat[:3], 4000, 1000)  # no eval a window back
    rising = [{**r, "ev_pm_hand": 0.05 * i} for i, r in enumerate(flat)]
    assert not is_settled(rising, 4000, 8000)  # +0.2 over the window
    swing = [{**r, "ev_sh_pm_hand": 0.8 + (0.15 if i % 2 else 0)} for i, r in enumerate(flat)]
    assert not is_settled(swing, 4000, 8000)  # up and down by 0.15: ends where it started but swings


def test_run_stops_itself_once_settled(tmp_path):
    import pandas as pd
    base = dict(arm="retain_open", env=SMALL, D=8, steps=30, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128,
                mlp_n=256, mlp_steps=5, ret_every=4, lam=0.01, eval_parts=128)
    s = run(Cfg(**base, settle_window=10, settle_min_steps=10, settle_patience=2, settle_tol_change=9, settle_tol_swing=9),
            tmp_path / "a", verbose=False)
    assert s["steps_run"] == 20 and len(pd.read_csv(tmp_path / "a" / s["run_id"] / "metrics.csv")) == 4  # evals at 5, 10, 15, 20
    assert math.isfinite(s["pm_hand"]) and math.isfinite(s["sh_pm_hand"])  # the final probes still ran
    s = run(Cfg(**base, settle_window=10, settle_min_steps=10, settle_tol_change=-1), tmp_path / "b", verbose=False)  # never settled
    assert s["steps_run"] == 30  # settle_window 0 (off by default) behaves the same


@pytest.mark.parametrize("arm", ["retain_open", "ae"])
def test_mechanism_diagnostics_are_logged(arm, tmp_path):
    import numpy as np
    import pandas as pd
    env = GridCfg(W=8, H=6, n_slow=8, agent_shape="arrow", mob_mode="chase")
    cfg = Cfg(arm=arm, env=env, D=8, steps=10, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128, mlp_n=256,
              mlp_steps=5, ret_every=4, lam=0.01, rec_class_power=1.0, log_mech=10, ckpt_every=5)
    s = run(cfg, tmp_path, verbose=False)
    h = pd.read_csv(tmp_path / cfg.run_id / "metrics.csv")
    assert (tmp_path / cfg.run_id / "ckpt_5.pt").exists() and (tmp_path / cfg.run_id / "ckpt_10.pt").exists()
    assert {"S_frac_slow", "S_n_hard"} <= set(h.columns)  # light: every eval
    assert h["gnorm_pred"].isna().iloc[0] and h["gnorm_pred"].notna().iloc[1]  # heavy: every log_mech steps (here the last eval)
    last = h.iloc[-1]
    assert {"gnorm_pred", "gnorm_sigreg", "gcos_pred_sigreg", "sens_pred_facing", "sens_sigreg_hand", "pc_live_facing_lin",
            "pc_live_hand_l", "lat_sens_facing_live", "lat_sens_position_live"} <= set(h.columns)
    assert np.isfinite(last[["gnorm_pred", "gnorm_sigreg", "pc_live_facing_l"]].astype(float)).all()
    d = np.load(tmp_path / cfg.run_id / "dirs.npz")
    assert d["spec_live"].shape == (2, 8) and (d["spec_live"] >= -1e-6).all()
    if arm == "retain_open":
        assert {"gnorm_open", "gnorm_ret", "gcos_pred_ret", "ret_fit_slow", "sh_sigreg", "face_acc_sh", "rec_agent_sh",
                "lat_sens_hand_sh", "pc_sh_mob_s", "sens_ret_hand"} <= set(h.columns)
        assert 0 <= last["face_acc_sh"] <= 1 and np.isfinite(last["rec_agent_sh"])
        assert {"refdir_w", "refdir_rho", "refdir_ac", "refdir_dom", "spec_shadow"} <= set(d.files)
    else:
        assert "gnorm_rec" in h.columns and 0 <= last["face_acc_live"] <= 1


def test_variants_change_one_thing_each():
    env = GridEnv(GridCfg(W=8, H=6, n_slow=8, agent_shape="arrow", mob_mode="chase"), "cpu")
    v = env.variants(64)
    base = v["base"][0]
    assert set(v) == {"base", "facing", "hand", "position", "twinkle"} and all(x.shape == base.shape for x, _ in v.values())
    assert all((x != base).flatten(1).any(1)[m].float().mean() > 0.95 for k, (x, m) in v.items() if k != "base")  # each valid variant differs
    assert v["position"][1].any() and not v["position"][1].all()  # some one-cell moves are blocked


def test_finalize_rebuilds_the_summary_of_a_run_that_died_in_the_final_probes(tmp_path):
    import json

    from gaf import analysis as A
    cfg = Cfg(arm="retain_open", env=SMALL, D=8, steps=10, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128, mlp_n=256,
              mlp_steps=5, ret_every=4, lam=0.01, ckpt_every=5, eval_parts=128)
    full = run(cfg, tmp_path, verbose=False)
    for f in ("summary.json", "model.pt"):  # what a crash in the final probes leaves out
        (tmp_path / cfg.run_id / f).unlink()
    rebuilt = A.finalize(cfg, tmp_path)
    assert set(rebuilt) == set(full) and {"pm_hand", "sh_pm_hand", "mlp_r2_slow", "sh_mlp_r2_slow"} <= set(rebuilt)
    assert json.loads((tmp_path / cfg.run_id / "summary.json").read_text())["run_id"] == cfg.run_id
    assert (tmp_path / cfg.run_id / "model.pt").exists()
    assert A.finalize(cfg, tmp_path) == json.loads((tmp_path / cfg.run_id / "summary.json").read_text())  # second call: reads the file
    assert len(A.collect(tmp_path)) == 1  # collect() accepts it


def test_probes_sample_in_chunks_with_the_same_result():
    from gaf.probes import collect_last, sample_chunked
    env = GridEnv(GridCfg(W=8, H=6, n_slow=8, hand_p=0.5, hand_tiles=2), "cpu")
    enc = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 4))
    z, y, vis = collect_last(enc, env, 300, chunk=128)
    assert z.shape == (300, 4) and y.shape[0] == 300 and vis.shape == (300,)
    c = sample_chunked(env, 300, chunk=128)
    assert c.x.shape[0] == 300 and c.a.shape[0] == 300 and all(v.shape[0] == 300 for v in c.f.values())


@pytest.mark.parametrize("kw", [dict(arm="joint"), dict(arm="joint", pred_arch="film"), dict(arm="retain_open", pred_arch="film")])
def test_joint_arm_film_predictor_and_decoder_readouts(kw, tmp_path):
    from gaf import readout as R
    env = GridCfg(W=8, H=6, n_slow=8, agent_shape="arrow", mob_mode="chase")
    cfg = Cfg(env=env, D=8, steps=10, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128, mlp_n=256, mlp_steps=5,
              ret_every=4, lam=0.01, rec_class_power=1.0, ckpt_every=10, **kw)
    s = run(cfg, tmp_path, verbose=False)
    assert math.isfinite(s["pm_hand"])
    d = tmp_path / cfg.run_id
    dyn = R.dynamics(d, n_eval=64)
    systems = {"live", "shadow"} if kw["arm"] == "retain_open" else {"live"}
    assert set(dyn.index.get_level_values(0)) == systems
    if kw["arm"] == "joint":  # the joint arm has its own decoder: the decoded predicted frame is scored
        row = dyn.loc[("live", "predicted")]
        assert 0 <= row["facing acc"] <= 1 and math.isfinite(row["agent"]) and math.isfinite(row["latent err / var"])
    ds = R.decoder_strength(d, steps=3, n_eval=64)
    assert set(ds.index) == systems and (ds["facing acc"].between(0, 1)).all()


def test_joint_forks_fade_anchor_freeze_and_readout(tmp_path):
    import pandas as pd

    from gaf import fork as F
    from gaf import readout as R
    env = GridCfg(W=8, H=6, n_slow=8, agent_shape="arrow", mob_mode="chase")
    cfg = Cfg(arm="joint", env=env, D=8, steps=10, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128, mlp_n=256,
              mlp_steps=5, lam=0.01, rec_class_power=1.0, rec_class_cap=4.0, ckpt_every=10)
    run(cfg, tmp_path, verbose=False)
    d = tmp_path / cfg.run_id
    snapshots = {}
    for mode in F.MODES:
        out = F.joint_fork(d, mode, steps=20, fade=10, r_floor=0.1, beta=5.0, log_every=10, verbose=False)
        m = pd.read_csv(out / "metrics.csv")
        snapshots[mode] = m
        assert (out / "done").exists() and (out / f"ckpt_{cfg.steps}.pt").exists() and len(m) == 2
    assert snapshots["continue"]["r"].eq(1.0).all() and snapshots["freeze"]["r"].eq(1.0).all()  # no fade there
    assert abs(snapshots["fade"]["r"].iloc[-1] - 0.1) < 1e-6 and abs(snapshots["anchor"]["beta"].iloc[-1] - 5.0) < 1e-6
    assert "anchor" in snapshots["anchor"] and "anchor" not in snapshots["fade"] and "rec" not in snapshots["freeze"]
    st = F.status(d)
    assert (st.state == "done").all()
    dyn = R.dynamics(d / "forks" / "anchor", n_eval=64)  # a fork reads like a run
    assert ("live", "predicted") in dyn.index and 0 <= dyn.loc[("live", "predicted"), "facing acc"] <= 1
    # the cap: a smaller cap keeps terrain's weight from being pushed to the floor
    from gaf.train import class_weights
    torch.manual_seed(0)
    w30, _ = class_weights(GridEnv(env, "cpu"), 1.0, n=256, cap=30.0)
    w4, _ = class_weights(GridEnv(env, "cpu"), 1.0, n=256, cap=4.0)
    assert w4[TC["map:floor"]] > 3 * w30[TC["map:floor"]]


@pytest.mark.parametrize("kw", [dict(dp_weight=1.0), dict(dp_weight=1.0, shadow_rec=0.0)])
def test_decoded_prediction_term_trains_and_forks_keep_it(kw, tmp_path):
    import pandas as pd

    from gaf import fork as F
    from gaf import readout as R
    env = GridCfg(W=8, H=6, n_slow=8, agent_shape="arrow", mob_mode="chase")
    cfg = Cfg(arm="joint", env=env, D=8, steps=10, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128, mlp_n=256,
              mlp_steps=5, lam=0.01, rec_class_power=1.0, rec_class_cap=4.0, pred_arch="film", ckpt_every=5, **kw)
    s = run(cfg, tmp_path, verbose=False)
    h = pd.read_csv(tmp_path / cfg.run_id / "metrics.csv")
    assert "dp" in h.columns and h["dp"].notna().all() and h["dp"].iloc[-1] > 0  # the decoded-prediction term is logged
    d = tmp_path / cfg.run_id
    out = F.joint_fork(d, "fade", steps=10, fade=5, r_floor=0.0, tag="r0", log_every=5, verbose=False)
    assert out.name == "fade-r0" and "dp" in pd.read_csv(out / "metrics.csv").columns  # the fork keeps the decoded prediction, the fade only the rest
    assert (F.status(d, modes=["fade-r0"]).state == "done").all()
    # every readout reads an earlier checkpoint and the latent's variance shares
    assert ("live", "predicted") in R.dynamics(d, n_eval=64, step=5).index
    vs = R.variance_shares(d, n=256, step=5)
    assert set(vs) == {"inventory", "position", "facing", "hand", "mob", "twinkle"} and all(0 <= v <= 1.0001 for v in vs.values())
    assert math.isfinite(R.probes(d, n_train=256, n_test=64, steps=5)["pm_hand"])


def test_fork_introduces_decoded_prediction_on_a_plain_joint_parent(tmp_path):
    """One plain joint parent serves every arm: dp cross-faded in with the fade, or switched on at once (dp only)."""
    import pandas as pd

    from gaf import fork as F
    env = GridCfg(W=8, H=6, n_slow=8, agent_shape="arrow", mob_mode="chase")
    cfg = Cfg(arm="joint", env=env, D=8, steps=10, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=128, mlp_n=256,
              mlp_steps=5, lam=0.01, rec_class_power=1.0, rec_class_cap=4.0, pred_arch="film", ckpt_every=10)
    run(cfg, tmp_path, verbose=False)
    d = tmp_path / cfg.run_id
    m = pd.read_csv(F.joint_fork(d, "fade", steps=10, fade=10, r_floor=0.0, tag="dp", dp=1.0, log_every=5, verbose=False) / "metrics.csv")
    assert abs(m["dp_w"].iloc[0] - 0.5) < 1e-6 and abs(m["dp_w"].iloc[-1] - 1.0) < 1e-6 and abs(m["r"].iloc[-1]) < 1e-6  # cross-fade
    m = pd.read_csv(F.joint_fork(d, "fade", steps=10, fade=1, r_floor=0.0, tag="dponly", dp=1.0, log_every=5, verbose=False) / "metrics.csv")
    assert m["dp_w"].eq(1.0).all() and m["r"].abs().lt(1e-6).all()  # immediate switch
    m = pd.read_csv(F.joint_fork(d, "fade", steps=10, fade=10, log_every=5, verbose=False) / "metrics.csv")
    assert "dp" not in m.columns and m["dp_w"].eq(0.0).all()  # no dp: the plain control
    s = json.loads((d / "forks" / "fade-dp" / "summary.json").read_text())["fork"]
    assert s["dp_weight"] == 1.0 and s["dp_ramped"]


def test_dp_weight_zero_leaves_the_joint_run_unchanged(tmp_path):
    """dp_weight is a late field: default runs keep their ids and have no dp column."""
    import pandas as pd
    env = GridCfg(W=8, H=6, n_slow=8, agent_shape="arrow", mob_mode="chase")
    base = dict(arm="joint", env=env, D=8, steps=5, batch=16, floor_start=10 ** 9, eval_every=5, diag_every=5, probe_n=64, mlp_n=128,
                mlp_steps=3, lam=0.01)
    assert Cfg(**base).run_id == Cfg(**base, dp_weight=0.0).run_id != Cfg(**base, dp_weight=1.0).run_id
    run(Cfg(**base), tmp_path, verbose=False)
    assert "dp" not in pd.read_csv(tmp_path / Cfg(**base).run_id / "metrics.csv").columns


def test_film_predictor_starts_as_copy_last():
    from gaf.models import FiLMPredictor, make_predictor
    q = FiLMPredictor(8, 3, 5)
    zc, a = torch.randn(4, 2, 8), torch.eye(5)[torch.randint(5, (4, 2))]
    assert torch.allclose(q(zc, a), zc[:, -1])
    assert isinstance(make_predictor("film", 8, 3, 5), FiLMPredictor)


def test_visible_hand_probe():
    from gaf.probes import parts_probe
    env = GridEnv(GridCfg(W=12, H=8, hand_tiles=2, hand_p=0.5), "cpu")
    enc = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 48 * 48, 8))
    out = parts_probe(enc, env, n_train=256, n_test=128, steps=5)
    assert "pm_hand_visible" in out and "pm_rank" in out
