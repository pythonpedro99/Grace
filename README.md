# Gain attribution with floor check

A small, fast test bed for one question. When a predictive world model with a shared encoder and
SIGReg collapses onto slow features, does the gain-attribution fix stop that while still letting the
encoder discard real noise? The fix: the encoder may simplify the target latent only in directions where
the predictor has stopped improving and sits near its achievable error floor.

Spec: `docs/gain_attribution_one_pager.pdf` (section numbers below refer to it). Design decisions
where the one-pager is silent, and the critical points of the experiment: [plan.md](plan.md).

**The two variants for the benchmark** (architecture, losses, hyperparameters, status):
[docs/benchmark_variants.md](docs/benchmark_variants.md): A, `target` + head start, and B, reference-gated retention
with a reconstruction shadow (`retain_open`; details in [docs/reference_gated_retention.md](docs/reference_gated_retention.md)).

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
| **hard** | red sprite on a chaotic circle map | `q' = 2q + 0.3·a (mod 1)`: deterministic given the action, learned after a long plateau | yes |
| **noise** | 8 patterns like the slow ones | fresh every frame | **no**: the only content to drop |

The predictable sprite content is action-conditioned. Without the action, R² of the next position is
capped at `(sin 2πv / 2πv)²`: 0.25 for both (v = 0.3). `run_gt(actions=False)` checks this.

The hard sprite is the not-yet-learned content: on ground truth, a predictor learns fast in about
750 steps and hard in 2250–3000 (4 seeds), after a plateau at R² ≈ 0. The plateau length grows
steeply with `h_act`: below 0.1 it is gone, and from 0.35 on hard is not learned reliably within 6000
steps, which makes it unlearnable instead of hard to learn.

Slow and noise patterns share one orthonormal pixel basis. Both are equally easy to *read* from
an image, and only their dynamics differ.

**Capacity and the premise.** The failure needs slow content that can fill SIGReg's budget on its
own. The default latent, D = 32, is twice n_slow = 16, but the world has 16 slow + 8 noise + 4 fast +
4 hard factor dimensions: 32 in total. **D = 32 is exactly full if noise is encoded too**, so the groups
compete for room, and content that is cheap for SIGReg (slow and noise: linear and Gaussian by
construction) wins it. A SIGReg-only encoder lets a sprite in only as slow and noise content recede (stage 8,
SIGReg-only shadow); a reconstruction objective forces all groups in. Slow content can also fill spare room with
Gaussianized nonlinear copies of itself. `premise()` measures this: at D = 32, slow plus copies reaches SIGReg ≈ 7,
against ≈ 1 for a true Gaussian and ≈ 230 for slow content alone. The tight regime D = n_slow, where
slow content alone is an exact isotropic Gaussian, is its own notebook stage (4b).

Knobs: `n_slow`, `slow_rho`, `n_noise`, `v_act`, `h_act` (action strength), `hard_k`
(difficulty), `a_*` (salience), `swap_colors` (fast red / hard white), `pixel_noise`, `res`, `T`.

## The grid world (`gaf/env_grid.py`, `grid_benchmark.ipynb`)

The stepping stone to Craftax: the same four groups, but discrete, relational and Craftax-shaped. A 10×8 tile map with
wood, stone and water (a new map per clip), two HUD rows below it, 40×40 pixels. Actions are one-hot over noop, up, down,
left, right.

| group | content | R² without the action |
|---|---|---|
| slow | 16 inventory slots (0–9 as tile brightness, ±1 with p = 0.03 per frame) and the agent's position | ≈ 0.99 |
| fast | the agent's facing: a move turns it even when blocked (yellow half of the agent tile) | ≈ 0.03 |
| hard | the hand: the material of the cell the agent faces after its action, shown in a HUD slot | ≈ 0.2 |
| noise | a mob that jumps to a random free cell every frame, and 4 tiles with a random brightness | 0 |

Why facing and hand, not positions: an agent that moves one cell per step is ≈ 95% predictable without the action, so
its position is slow content and nothing would test the method. The hand is a pattern to learn: where the agent ends up,
then what lies in front of it (on ground truth: R² 0.87 after 1000 steps, 0.95 after 2000). The inventory alone can
satisfy SIGReg: `premise()` at D = 32 gives 4.0 for slow plus copies, against 1.2 for a Gaussian.

`make_env` picks the env from the config type (`GridCfg` or `EnvCfg`), so every arm and analysis works on both.
`capacity()` measures which latent size D the content needs, and `A.parts_r2()` reports R² per part (inventory and
position inside slow, mob and tiles inside noise). D = 32 in the notebook is provisional until its stage 2b has run.
Knobs: `W`, `H`, `tile`, `n_slow`, `levels`, `hud_p`, `density`, `random_map`, `n_twinkle`, and for the mass experiments
`hand_tiles` (the hand shown in k HUD tiles), `hand_contrast` (its colour contrast, for masses below one tile), `compass_tiles` (the facing also shown in fixed HUD tiles) and `hand_p` (the hand tiles
visible in a share of the clips; hidden ones show the average hand colour). `component_masses()` gives each component's share of the
pixel variance, and `f["tile_class"]` the ground-truth class of every tile.

`grid_mass_law.ipynb` tests the working rule "a class is held only if its share m of the pixel variance satisfies m · D ≥ κ": the
hand's mass against the latent size D (SIGReg alone and an autoencoder), frequency against size, a fixed-location facing, the recipe
(`sg`, a head start and `retain_open` below and above the threshold), and a class-balanced reconstruction loss (`rec_class_power`).

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
| `block` | masked to the open directions | `g_open` | experimental: the encoder gets the prediction loss only in the open directions, through both paths |
| `retain` | `g_ctx` + retention | none | experimental, unsupervised: reference-gated retention alone (`docs/reference_gated_retention.md`) |
| `retain_open` | `g_ctx` + retention | `g_open` | the same plus the floor-gated open path that removes noise |
| `ae` | none | none | an autoencoder: reconstruction (`shadow_rec`) + SIGReg, no prediction loss; B's shadow encoder on its own |
| `joint` | `g_ctx` | none | one encoder with reconstruction (keeps the content) + prediction (shapes it for dynamics) + SIGReg (`grid_joint.ipynb`) |

Every arm runs the same optimizer, gate and diagnostics, so the leak can be compared across arms.

`head_start=K` (any arm): for the first K steps the encoder gets only SIGReg while the predictor
trains on the detached latents; then the arm switches on. It tests whether the sprites are lost in a
race against the predictor (notebook stage 7).

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
- **Directions, `gate.excess_basis`:**
  - Under SIGReg the latent is rotation-invariant, so single latent dims are arbitrary mixtures of
    content.
  - SIGReg makes Cov(z) ≈ I, so PCA of z is degenerate and gives no basis either. What whitening
    does give: for any covariance A, `vᵀ A v` is a share of the (unit) variance along v.
  - The default `basis="excess"` ranks directions by the gate's own decision quantity. At each
    refresh it solves `(C_P − C*) v = λ Cov(z_t) v`. C_P and C\* are the held-out residual
    covariances of the current predictor and of the floor probe, and all three matrices come from
    the same frozen snapshot and pool. λ is the share of the variance along v that the predictor
    still misses beyond the floor. The measured Cov(z_t) is used, not I, because SIGReg holds only
    approximately.
  - Content the predictor has not learned yet gets large λ (protect). Learned content and noise at
    its floor get λ ≈ 0 (open).
  - The solutions are Cov(z)-orthonormal. QR in order keeps the nested top-k subspaces and gives a
    Euclidean-orthonormal basis, so the split of the gradient over directions stays exact.
  - The running state is rotated into each new basis.
  - Ablations: `basis="resid"` (eigenbasis of C\* alone) separates noise from predictable content,
    but not learned from not-yet-learned, since both sit at floor ≈ 0. `basis="dim"` uses latent
    dims.
  - Other possible bases (slow feature analysis on z_{t+1} − z_t; action controllability from a
    probe without actions; progress C_lag − C_P) are compared in plan.md.
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
- **Leak by source:** `lk_ctx_S`, `lk_rest_S` (and `_open`): at diagnostic steps, only the context
  part or only the rest (SIGReg, open target path) of the applied step is applied to the encoder, and the
  target's error is measured with the predictor held fixed. Positive means that part makes `z_t`
  easier to predict.
- **Early evals:** `early_eval=(0, 25, 50, …)` adds probe evaluations at those steps, written to
  `early.csv` (`A.curve`, `A.trace` and `A.snapshot` merge them with the regular ones).
- **Learn time in latent space:** `A.latent_learning(cfg, out)` freezes the encoder of a finished
  run, trains a fresh predictor on its latents and returns the held-out R² of each group over
  predictor steps, plus how available each group is in the latent (`.attrs["availability"]`).
- **Gate state per direction over training:** `gap`, `excess`, `L`, `Lstar`, `inS`, in `dirs.npz`.

## Run it

**RunPod GPU pod (PyTorch template):**

1. Upload or clone this folder to `/workspace`.
2. Open `gain_attribution.ipynb` and run it top to bottom.

Results go to `/workspace/results/<run_id>/`: `metrics.csv`, `dirs.npz`, `model.pt`,
`summary.json` and `log.txt`. A run whose `summary.json` exists is skipped, so re-running a cell
resumes.

Every long cell shows a progress bar:
- **Training stages (`execute`):** one bar over the training steps of all runs still to do, with
  speed and ETA for the whole cell. Its status line lists each running run as
  `<arm> s<seed> <percent> <phase>`, where the phase is `train`, `floor probe`, `eval` or
  `final probes`. Each finished run prints its time above the bar.
- **Ground-truth check and floor probes of finished runs:** a bar over their training steps.
- With `WORKERS > 1` the workers report to the parent through `<run_id>/progress.json`; their eval
  lines go to `log.txt`. With `WORKERS = 1` the eval lines are printed above the bar.

`WORKERS = 3` in the notebook runs three runs at once, as separate processes on the same GPU. The
model is small and the GPU mostly waits on kernel launches, so this is close to a 3× speed-up. Watch
one run with `tail -f /workspace/results/<run_id>/log.txt`.

Rough budget: 4–7 min per run alone; the full notebook (39 runs) takes about 3–4.5 h sequentially,
roughly 1–2 h with 3 workers. These are estimates; the first stage prints real timings.

**Locally** (CPU is enough for the tests, about 40 s):

    pip install -e ".[dev]"
    pytest -q tests

One run: `python -c "from gaf.train import Cfg, run; run(Cfg(arm='both'), 'results')"`.

## Layout

    gaf/env.py       clip world with actions (EnvCfg, ClipEnv)
    gaf/env_grid.py  grid world, the Craftax-shaped stepping stone (GridCfg, GridEnv)
    gaf/models.py    CNN encoder (GroupNorm, no BatchNorm), action predictor, SIGReg (LeWM-verified)
    gaf/retain.py    RefMap: reference floor and retention weights of reference-gated retention
    gaf/gate.py      FloorGate (gap, excess, hysteresis, basis rotation), floor_probe
    gaf/grads.py     flat VJPs, leak projection, SplitAdamW
    gaf/probes.py    ridge / MLP probes, probing suite
    gaf/train.py     Cfg, ARMS, make_env(), run(), run_gt() (learnability on ground truth), premise(),
                     capacity() (which D the content needs)
    gaf/analysis.py  grids, collect, tables, verdicts, floors_of(), parts_r2(), figures, launch()/live_probes() (background runs), finalize()
    gaf/mech.py      mechanism diagnostics (Cfg.log_mech): per-term gradient forces and attribution, retention weights, spectra
    gaf/readout.py   decoder-based readouts of a checkpoint: dynamics() (decoded predicted frames), decoder_strength()
    gaf/fork.py      forks of a finished joint run: continue / fade / anchor (prediction distillation) / freeze, from its checkpoint
    tests/           CPU checks
    gain_attribution.ipynb   the clip world
    grid_benchmark.ipynb     the grid world (results in /workspace/results_grid)
    sigreg_hypothesis.ipynb  does SIGReg select content by pixel mass? (clip world, and the SIGReg weight in the grid world)
    grid_mass_law.ipynb      the mass law in the grid world (results in /workspace/results_masslaw)
    grid_b_d128.ipynb        B at D = 128 with the arrow agent and chasing mob (results in /workspace/results_b128)
    grid_b_mechanism.ipynb   what removes the hand, holds the noise, collapses the rank (results in /workspace/results_mech)
    grid_joint.ipynb         the joint model: warmup (reconstruction + prediction), fade, prediction-distillation anchor (results in /workspace/results_joint)
    docs/findings.md         everything learned so far, with the numbers, the corrections and the open questions

## Cost

Every arm computes `g_ctx` and the SIGReg part separately: two encoder backward passes. That keeps
the optimizer state identical across arms. A fix arm adds one backward pass through the target frame
for `g_tgt,S`, plus one for `g_open` in `target` and `both`. The baselines compute `g_tgt,S` only
every `diag_every` steps. The floor probe is amortised every `floor_every` steps.
