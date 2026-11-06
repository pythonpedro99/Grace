# Gain attribution with floor check

A small, fast test bed for one question. When a predictive world model with a shared encoder and
SIGReg collapses onto slow features, does the gain-attribution fix stop that while still letting the
encoder discard real noise? The fix: the encoder may simplify the target latent only in directions where
the predictor has stopped improving and sits near its achievable error floor.

Spec: `docs/gain_attribution_one_pager.pdf` (section numbers below refer to it). Design decisions
where the one-pager is silent, and the critical points of the experiment: [plan.md](plan.md).

## The problem in one paragraph

SIGReg checks only batch statistics. Slow features vary strongly between clips, so they satisfy
SIGReg while being trivially predictable, and the encoder can collapse onto them. Stop-grad on the
target does not prevent this. The encoder is shared, so a context-path update also moves the target
`z_t = f(x_t; θ)`. If that makes `z_t` easier to predict, the loss drops through a route that no
gradient accounted for: the **leak**.

## The environment (`gaf/env.py`)

RGB clips, 32×32, `T = 3` frames: two context frames and one target. Actions `a_t ∈ [−1, 1]²`
come from a uniform random policy and act between frames. The predictor sees them; the encoder
does not. Four kinds of content:

| Group | What | Dynamics | Should the latent keep it? |
|-------|------|----------|----------------------------|
| **slow** | 16 smooth colour patterns | OU, ρ = 0.98; Gaussian across clips; action-free | yes (trivially predictable) |
| **fast** | white sprite, wraps around | `p' = p + 0.3·a`: deterministic given the action | yes |
| **hard** | red sprite on a chaotic circle map | `q' = 2q + 0.4·a (mod 1)`: deterministic given the action, slower to learn | yes |
| **noise** | 8 patterns like the slow ones | fresh every frame | **no**: the only content to drop |

The predictable sprite content is action-conditioned. Without the action, R² of the next position is
capped at `(sin 2πv / 2πv)²`: 0.25 for fast and 0.05 for hard. `run_gt(actions=False)` checks this.

Slow and noise patterns share one orthonormal pixel basis. Both are equally easy to *read* from
an image, and only their dynamics differ. With `n_slow ≥ D` the slow factors alone are an isotropic
Gaussian of size D, so SIGReg can be satisfied without any other content. That is the premise of the
failure, and the notebook checks it (`premise()`).

Knobs: `n_slow`, `slow_rho`, `n_noise`, `v_act`, `h_act` (action strength), `hard_k`
(difficulty), `a_*` (salience), `pixel_noise`, `res`, `T`.

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
| `context` | projected | none | fix, context side only |
| `both` | projected | `g_open`, projected too | the full method (§7) |

Every arm runs the same optimizer, gate and diagnostics, so the leak can be compared across arms.

## The method

- **Floor L\*_j (§3), `gate.floor_probe`:**
  - Every `floor_every` steps: freeze a snapshot of the encoder and encode 16k fresh clips. Then
    train a predictor of the same architecture and optimizer (AdamW, same lr and wd, with actions).
    L\*_j is its held-out per-direction error.
  - The probe is warm-started across refreshes (`floor_warm=True`), so its budget accumulates
    while the latents drift slowly.
  - To counter probe bias, the floor used is the minimum over the last `floor_keep` refreshes,
    times `(1 − floor_margin)`.
- **Gate (§4), `gate.FloorGate`:** relative gap `EMA_W[L_j(P̄) − L_j(P)] / (L̄_j + δ)`, and
  excess `e_j = (L̄_j − L*_j) / (L*_j + δ)`. A direction enters S when the gap exceeds `eps_g` or the
  excess exceeds `eps_f`. It leaves only when both are below half of their threshold and `min_dwell`
  steps have passed. Before the first floor exists, everything is in S.
- **Directions:** the default `basis="resid"` is the eigenbasis of the floor probe's held-out
  residual covariance. Under SIGReg the latent is rotation-invariant, so single latent dims are
  arbitrary mixtures of content. The residual eigenbasis separates noisy directions (high floor)
  from predictable ones (floor ≈ 0). The running state is rotated into each new basis.
  `basis="dim"` remains as an ablation.
- **Projection on the applied AdamW step (§6, §10), `grads.SplitAdamW`:**
  - The one-pager derives the projection for SGD. An AdamW step is `−lr (M m̂ + wd θ)`, where
    `M = 1/(√v̂ + ε)` is the preconditioner and the momentum `m̂` is about 90% past gradients.
    Projecting only the current gradient leaves momentum and weight decay free to keep leaking.
  - SplitAdamW is exactly AdamW, tested against `torch.optim.AdamW`. The difference is that it keeps
    one first moment for the projected side and one for the rest.
  - The leak of the step actually taken is `ℓ = g_tgt,S · u_ctx` with `u_ctx = M m̂_ctx + wd θ`. If
    `ℓ > 0`, then `c · g_tgt,S` is removed from that momentum with
    `c = α ℓ / (g_tgt,S · M g_tgt,S)`. That leaves `g_tgt,S · u_ctx = (1 − α) ℓ` exactly, and the
    removed component stays out of the momentum.
  - `proj_on="grad"` is the one-pager sketch (project the raw gradient), kept as an ablation.
    `opt="sgd"` is the case where the derivation holds exactly.
- **The open path is projected too** (`proj_open=True`). `g_open` shares encoder weights with the S
  directions, so its step also moves `z_t` on S. In `both`, it is added to the projected side.

## Probing suite and diagnostics (`gaf/probes.py`, §9)

- **Representation:** ridge R² of every group at the target frame from `z_t` (`r2_*`), from the
  prediction (`pred_r2_*`), and from a final MLP probe (`mlp_r2_*`). Also each group's share of
  `tr Cov(z)` (`vs_*`), and per-direction content (D × groups).
- **Slow-feature drift:** the within/between-clip variance ratio (`wb_ratio`) and the effective rank
  of `z_t − z_{t−1}` (`rank_dz`).
- **Leak:**
  - Share of steps with `ℓ > 0` (`leak_pos`).
  - Cosine of `g_tgt,S` with the applied projected-side update, before and after projection
    (`leak_cos`, `leak_cos_post`; the latter ≈ 0). Also the norm ratio `proj_ratio`.
  - Per direction, the first-order leak of that update comes from one forward-mode JVP
    (`leak_jvp`).
  - The exact check is `ℓ_real = L(θ′, z_t(θ)) − L(θ′, z_t(θ′))`, summed over S and over the open
    directions (`lreal_S`, `lreal_open`).
- **Gate state per direction over training:** `gap`, `excess`, `L`, `Lstar`, `inS`, in `dirs.npz`.

## Run it

**RunPod GPU pod (PyTorch template):**

1. Upload or clone this folder to `/workspace`.
2. Open `gain_attribution.ipynb` and run it top to bottom.

Results go to `/workspace/results/<run_id>/`: `metrics.csv`, `dirs.npz`, `model.pt` and
`summary.json`. A run whose `summary.json` exists is skipped, so re-running a cell resumes.

**Locally** (CPU is enough for the tests, about 40 s):

    pip install -e ".[dev]"
    pytest -q tests

One run: `python -c "from gaf.train import Cfg, run; run(Cfg(arm='both'), 'results')"`.

## Layout

    gaf/env.py       clip world with actions (EnvCfg, ClipEnv)
    gaf/models.py    CNN encoder (GroupNorm, no BatchNorm), action predictor, SIGReg (LeWM-verified)
    gaf/gate.py      FloorGate (gap, excess, hysteresis, basis rotation), floor_probe
    gaf/grads.py     flat VJPs, leak projection, SplitAdamW
    gaf/probes.py    ridge / MLP probes, probing suite
    gaf/train.py     Cfg, ARMS, run(), run_gt() (learnability on ground truth), premise()
    gaf/analysis.py  grids, collect, tables, verdicts, floors_of(), figures
    tests/           CPU checks
    gain_attribution.ipynb

## Cost

Every arm computes `g_ctx` and the SIGReg part separately: two encoder backward passes. That keeps
the optimizer state identical across arms. A fix arm adds one backward pass through the target frame
for `g_tgt,S`, plus one for `g_open` in `target` and `both`. The baselines compute `g_tgt,S` only
every `diag_every` steps. The floor probe is amortised every `floor_every` steps.
