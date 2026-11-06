# Gain attribution with floor check

A small, fast test bed for one question. When a predictive world model with a shared encoder and
SIGReg collapses onto slow features, does the gain-attribution fix stop that while still letting the
encoder discard real noise? The fix: the encoder may simplify the target latent only in directions where
the predictor has stopped improving and sits near its achievable error floor.

Spec: `docs/gain_attribution_one_pager.pdf` (section numbers below refer to it). Design decisions
where the one-pager is silent: [plan.md](plan.md).

## The problem in one paragraph

SIGReg checks only batch statistics. Slow features vary strongly between clips, so they satisfy
SIGReg while being trivially predictable, and the encoder can collapse onto them. Stop-grad on the
target does not prevent this. The encoder is shared, so a context-path update also moves the target
`z_t = f(x_t; θ)`. If that makes `z_t` easier to predict, the loss drops through a route that no
gradient accounted for: the **leak**.

## The environment (`gaf/env.py`)

RGB clips, 32×32, `T = 3` frames: two context frames and one target. Four kinds of content:

| Group | What | Dynamics | Should the latent keep it? |
|-------|------|----------|----------------------------|
| **slow** | 16 smooth colour patterns | OU, ρ = 0.98; Gaussian across clips | yes (trivially predictable) |
| **fast** | white sprite, constant velocity per clip, wraps around | `p₂ = 2p₁ − p₀`: deterministic, needs both context frames | yes |
| **hard** | red sprite on a chaotic circle map `q' = k q + c (mod 1)` | `q₂ = (k+1) q₁ − k q₀`: deterministic, slower to learn | yes |
| **noise** | 8 patterns like the slow ones | fresh every frame | **no**: the only content to drop |

Slow and noise patterns share one orthonormal pixel basis. Both are equally easy to *read* from
an image, and only their dynamics differ. With `n_slow ≥ D` the slow factors alone are an isotropic
Gaussian of size D, so SIGReg can be satisfied without any other content. That is the premise of the
failure, and the notebook checks it (`premise()`).

Knobs: `n_slow`, `slow_rho`, `n_noise`, `v_max`, `hard_k` (difficulty), `a_*` (salience),
`pixel_noise`, `res`, `T`.

## Arms (`gaf/train.py`)

Encoder gradient pieces (§2): `g_ctx` (context path), `g_tgt,S` (target path on the still-learning
set S; a measuring stick only), `g_open` (target path on saturated directions), `g_SIG` (never
projected).

| Arm | Context path | Target path | Role |
|-----|--------------|-------------|------|
| `lewm` | `g_ctx` | all directions | baseline: full gradient (LeWM) |
| `sg` | `g_ctx` | none | baseline: stop-grad, the leak is unchecked |
| `ema` | `g_ctx` | EMA encoder | baseline: BYOL / I-JEPA style target |
| `detached` | none | none | encoder sees only SIGReg: measures noise floors |
| `target` | `g_ctx` | `g_open` | fix, target side only |
| `context` | `g′_ctx` (projected) | none | fix, context side only |
| `both` | `g′_ctx` | `g_open` | the full method (§7) |

Every arm runs the same gate and diagnostics, so the leak can be compared across arms.

## The method

- **Floor L\*_j (§3), `gate.floor_probe`:** every `floor_every` steps, freeze a snapshot of the
  encoder, encode 16k fresh clips, and train a fresh predictor (same architecture) for
  `floor_steps`. L\*_j is its held-out per-direction error. To counter probe bias, the floor used is
  the minimum over the last `floor_keep` refreshes, times `(1 − floor_margin)`.
- **Gate (§4), `gate.FloorGate`:** relative gap `EMA_W[L_j(P̄) − L_j(P)] / (L̄_j + δ)`, and
  excess `e_j = (L̄_j − L*_j) / (L*_j + δ)`. A direction enters S when the gap exceeds `eps_g` or the
  excess exceeds `eps_f`, and leaves only when both are below half of their threshold and
  `min_dwell` steps have passed. Before the first floor exists, everything is in S.
- **Projection (§6), `grads.project`:** `ℓ = g_tgt,S · M g_ctx`. If `ℓ > 0`, then
  `g′_ctx = g_ctx − α ℓ / (g_tgt,S · M g_tgt,S) · g_tgt,S`. `M` is Adam's diagonal preconditioner, so
  the leak of the *actual* Adam step is removed (`precond=False` uses M = I, `opt="sgd"` is the exact
  case).
- **Directions:** latent dims (`basis="dim"`), or the eigenbasis of the floor probe's residual
  covariance (`basis="resid"`), which sorts directions from noisy to predictable. The running state is
  rotated into each new basis.

## Probing suite and diagnostics (`gaf/probes.py`, §9)

- **Representation:** ridge R² of every group at the target frame from `z_t` (`r2_*`), from the
  prediction (`pred_r2_*`), and from a final MLP probe (`mlp_r2_*`). Also each group's share of
  `tr Cov(z)` (`vs_*`), and per-direction content (D × groups).
- **Slow-feature drift:** the within/between-clip variance ratio (`wb_ratio`) and the effective rank
  of `z_t − z_{t−1}` (`rank_dz`).
- **Leak:** share of steps with `ℓ > 0` (`leak_pos`), `cos(g_tgt,S, M g_ctx)` (`leak_cos`), and
  `‖g′_ctx‖/‖g_ctx‖` (`proj_ratio`). Per direction, the first-order leak of the applied context step
  comes from one forward-mode JVP (`leak_jvp`). The exact check is
  `ℓ_real = L(θ′, z_t(θ)) − L(θ′, z_t(θ′))`, summed over S and over the open directions
  (`lreal_S`, `lreal_open`).
- **Gate state per direction over training:** `gap`, `excess`, `L`, `Lstar`, `inS`, in `dirs.npz`.

## Run it

**RunPod GPU pod (PyTorch template):**

1. Upload or clone this folder to `/workspace`.
2. Open `gain_attribution.ipynb` and run it top to bottom.

Results go to `/workspace/results/<run_id>/`: `metrics.csv`, `dirs.npz`, `model.pt` and
`summary.json`. A run whose `summary.json` exists is skipped, so re-running a cell resumes.

**Locally** (CPU is enough for the tests):

    pip install -e ".[dev]"
    pytest -q tests

One run: `python -c "from gaf.train import Cfg, run; run(Cfg(arm='both'), 'results')"`.

## Layout

    gaf/env.py       clip world (EnvCfg, ClipEnv)
    gaf/models.py    CNN encoder (GroupNorm, no BatchNorm), predictor, SIGReg (LeWM-verified), EMA
    gaf/gate.py      FloorGate (gap, excess, hysteresis, basis rotation), floor_probe
    gaf/grads.py     flat VJPs, Adam metric, leak projection
    gaf/probes.py    ridge / MLP probes, probing suite
    gaf/train.py     Cfg, ARMS, run(), run_gt() (learnability on ground truth), premise()
    gaf/analysis.py  grids, collect, tables, verdicts, floors_of(), figures
    tests/           CPU checks
    gain_attribution.ipynb

## Cost

A fix arm pays for one extra lagged-predictor forward pass and two extra encoder backward passes
(`g_ctx` and `g_tgt,S` separately). The floor probe is amortised every `floor_every` steps. The
baselines need one encoder backward pass, plus the separate passes every `diag_every` steps for the
diagnostics.
