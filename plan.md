# Design decisions

Where `docs/gain_attribution_one_pager.pdf` is silent or leaves a choice open, this is what the code
does. Shout if you disagree.

## Environment
1. **Clips, not chains.** Every step draws fresh i.i.d. clips: T = 3 frames, 2 context and
   1 target. Fast-sprite velocity and the hard-map offset `c` are per clip, so T = 2 would make both
   partly unpredictable. That is a usable knob, but not the default.
2. **Slow vs. noise are visually identical.** Both are smooth colour patterns from one orthonormal
   basis. Only the dynamics separate them, so the encoder cannot tell them apart by appearance.
3. **The premise is built in.** With `n_slow = D` the slow factors are exactly an isotropic Gaussian
   of size D, so SIGReg is satisfiable by slow content alone. `premise()` checks it numerically.
4. **Sprite salience.** Peak brightness 4 against a pattern RMS of 1, σ = 0.06 image widths, white
   (fast) and red (hard). They are visible by eye. Lower `a_fast`, `a_hard` to make the shortcut more
   attractive.

## Model and losses
5. **No BatchNorm in the encoder** (GroupNorm instead). With BN, a target-path gradient would also
   move the context latents through the batch statistics, and the attribution would no longer be
   clean. Context and target frames go through separate encoder passes.
6. **Loss scale.** The optimised loss is the MSE averaged over directions,
   `L = (1/D) Σ_j L_j` (LeWM's scale). `L_j` is logged per direction without the 1/D.
7. **SIGReg** is the LeWM module (17 knots on [0, 3], statistic × N, 256 fresh slices per step). It is
   applied to the online latents of every frame and averaged over frames.

## Gate
8. **Relative gap.** `gap_j = EMA_W[L_j(P̄) − L_j(P)] / (L̄_j + δ)`, so `eps_g` is dimensionless.
   The one-pager's absolute gap would depend on the latent scale.
9. **Excess uses the smoothed loss**: `e_j = (L̄_j − L*_j) / (L*_j + δ)`. `δ = 0.01` keeps perfectly
   predictable directions (L* ≈ 0) from looking infinitely far from their floor.
10. **Before the first floor, everything is in S.** The first floor probe runs at `floor_start = 500`.
11. **Floor aggregation:** the minimum over the last `floor_keep = 3` refreshes, times
    `(1 − floor_margin)`. This replaces EMA smoothing across refreshes, per the probe-bias caveat
    (an under-trained probe overestimates L*).
12. **The probe** has the predictor's architecture and trains for `floor_steps = 1500` Adam steps on
    12k clips from a frozen snapshot. It is scored on 4k held-out clips. For the `ema` arm, the
    context comes from the online encoder and the target from the EMA encoder.
13. **Basis.** The default is latent dims. `basis="resid"` uses the eigenbasis of the probe residual
    covariance. At each refresh the running state (gap, L̄, S, dwell, floor history) is rotated with
    the squared overlap matrix of old and new bases.
14. **Hysteresis:** enter S above a threshold, leave below half of it, and only after `min_dwell`
    steps in S.

## Gradients
15. **Adam metric.** The leak of an Adam step is `g_tgt,S · M g_ctx` with
    `M = 1/(√v̂ + ε)` from the optimizer state before the step. The projection uses the same `M`, so
    `g_tgt,S · M g′_ctx = (1 − α) ℓ` exactly. Adam's momentum still mixes in past gradients:
    `lreal_S` measures what remains.
16. **The predictor** has its own Adam and a standard step on L. `P̄ = EMA(P)` with `tau_lag = 0.99`.
17. **Diagnostics run for every arm.** Every `diag_every` steps the baselines also compute
    `g_tgt,S` and ℓ, and their leak is compared with the fixes. `ℓ_real` includes everything that
    moves the target in one step (SIGReg, the open path, Adam momentum), not only the context path.

## Verdicts (notebook)
18. **Baseline shows the failure:** slow R² ≥ 0.8, and fast and hard R² < 0.3.
    **Fix works:** fast and hard R² +0.3 over `sg`, while noise and slow stay within 0.1 of `sg`.
    These are seed means of the final ridge R² from z_t. The MLP probe is reported alongside, to rule
    out "nonlinearly present".

## Open
- **Untested on a GPU.** The thresholds `eps_g`, `eps_f`, `delta` and `min_dwell` are first guesses.
  Check `gate_map` of a `both` run before tuning.
- **If `sg` keeps fast and hard content anyway,** the failure needs more pressure. Lower the sprite
  salience, raise `hard_k`, or lower D below `n_slow`.
