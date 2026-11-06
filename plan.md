# Design decisions and critical points

Where `docs/gain_attribution_one_pager.pdf` is silent or leaves a choice open, this is what the code
does. The second half lists what can make the experiment fail or mislead, and what guards against it.

## Decisions

### Environment
1. **Clips, not chains.** Every step draws fresh i.i.d. clips: T = 3 frames, 2 context and
   1 target. Actions `a_t ∈ [−1, 1]²` (uniform random policy) act between frames. The predictor
   sees all T − 1 actions, the encoder none.
2. **The predictable sprite content is action-conditioned.**
   - Fast: `p' = p + 0.3 a`. Hard: `q' = 2q + 0.4 a (mod 1)`.
   - Without the action, R² is capped at `(sin 2πv / 2πv)²`: 0.25 (fast) and 0.05 (hard).
     `test_dynamics_need_actions` checks it.
   - At the first draft's `v = 0.12` the cap was 0.8, and the action would have hardly mattered.
3. **Slow and noise look identical.** Both are smooth colour patterns from one orthonormal basis.
   Only the dynamics separate them.
4. **Capacity and premise.**
   - The default is D = 32 with n_slow = 16: room for every group.
   - The premise holds through nonlinear slow copies: Gaussianized pairwise products of the slow
     factors bring a slow-only latent to SIGReg ≈ 7 at D = 32, against ≈ 1 for a Gaussian and ≈ 230
     for zero padding. `premise()` reports it.
   - The tight regime D = n_slow (slow alone is an exact isotropic Gaussian) is notebook stage 4b.
5. **Sprite salience:** peak brightness 4 against a pattern RMS of 1, σ = 0.06 image widths, white
   (fast) and red (hard). The sprites are visible by eye.

### Model and losses
6. **No BatchNorm in the encoder** (GroupNorm instead). With BN, a target-path gradient would also
   move the context latents through the batch statistics. Context and target frames go through
   separate encoder passes.
7. **Loss scale:** the optimised loss is `L = (1/D) Σ_j L_j` (LeWM's MSE scale). `L_j` is logged
   without the 1/D.
8. **SIGReg** is the LeWM module (17 knots on [0, 3], statistic × N, 256 fresh slices per step), on
   the online latents of every frame, averaged over frames.

### Gate
9. **Relative gap:** `gap_j = EMA_W[L_j(P̄) − L_j(P)] / (L̄_j + δ)`, dimensionless.
10. **Excess uses the smoothed loss:** `e_j = (L̄_j − L*_j) / (L*_j + δ)`. `δ = 0.01` keeps directions
    with L* ≈ 0 from looking infinitely far from their floor.
11. **Before the first floor (step 500), everything is in S.**
12. **Floor:** minimum over the last 3 refreshes, × 0.9. The probe has the predictor's architecture
    and optimizer (AdamW, same lr and wd), sees the actions, and is warm-started across refreshes.
13. **Basis: `excess` by default.**
    - The generalized eigenvectors of `C_P v = λ (C* + δI) v`, from the current predictor and the
      probe on the same held-out pool, orthonormalized in order (QR).
    - `λ − 1` is the relative excess along v: the gate's decision quantity.
    - At each refresh the running state (gap, L̄, S, dwell, floor history) is rotated with the
      squared overlap matrix of the old and new bases. `resid` (eigenbasis of C\* alone) and `dim`
      are ablations.
14. **Hysteresis:** enter above a threshold, leave below half of it after `min_dwell` steps.

### Gradients and optimizer
15. **SplitAdamW** (see README):
    - It is exactly AdamW. The projection acts on the context-side update that is actually applied:
      `u_ctx = M m̂_ctx + wd θ`, momentum and decoupled decay included.
    - The second moment is computed from the raw (unprojected) gradients, so M is known before
      projecting and the identity `g_S · u_ctx = (1 − α) ℓ` holds exactly.
16. **`g_open` is projected along with `g_ctx`** in `both` (`proj_open=True`).
17. **The predictor** uses standard AdamW on L. `P̄ = EMA(P)` with τ = 0.99.
18. **Diagnostics run for every arm.** `ℓ_real` includes everything that moves the target in one
    step: SIGReg, the open path, momentum.

### Verdicts (notebook)
19. **Baseline shows the failure:** slow R² ≥ 0.8, and fast and hard R² < 0.3.
    **Fix works:** fast and hard +0.3 over `sg`, noise and slow within 0.1 of `sg`. In the tight
    regime (stage 4b) slow may drop by 0.3. These are seed means of the final ridge R² from z_t,
    with the MLP probe alongside.

## Critical points

Ordered by how likely they are to break the experiment or its interpretation.

1. **Probe bias on exactly the content we care about.**
   - **Risk:** hard is the slowest content to learn. If the floor probe cannot learn it within its
     budget, L* is overestimated, e looks small, and the gate opens on hard directions. The encoder
     may then drop them: the failure the method is meant to prevent.
   - **Guards:** warm-started probe (the budget accumulates), min over refreshes, a margin, and
     the notebook's check of the learn time on ground truth against `floor_steps`.
   - **Check:** the `gate_map` of a `both` run. Hard directions must stay in S until the predictor
     has learned them.
2. **The basis.**
   - **Risk:** SIGReg makes the latent rotation-invariant, so a latent dim is an arbitrary mix of
     content. Per-dim gating then decides for mixtures. A direction mixing noise with unlearned
     sprite content is either protected (noise survives) or opened (the sprite is lost).
   - **Guard:** the `excess` basis diagonalizes the decision quantity itself. Unlearned content
     (large λ) separates from learned content and noise (λ ≈ 1). Inside the λ ≈ 1 subspace the
     directions are arbitrary, but all of them are open anyway.
   - **Remaining risk 1:** the basis is fixed between refreshes (every `floor_every` steps), while
     the predictor learns. Directions whose content becomes learned stay in S until their excess
     falls; that error is on the protective side.
   - **Remaining risk 2:** if the probe has not learned a sprite (point 1), C\* ≈ C_P on its
     directions. λ ≈ 1, and the basis cannot save it.
3. **The optimizer** (answered by SplitAdamW).
   - **Risk:** with Adam(W), about 90% of the step is momentum, preconditioned by M, plus weight
     decay. Projecting the raw gradient leaves all of that free to leak: the dry run measured
     `leak_cos_post` ≈ 0.30 with `proj_on="grad"` against ≈ 1e-8 with `proj_on="update"`.
   - **Remaining risk:** the projection is first-order and per step; Adam's v also changes the
     metric over time. `lreal_S` is the exact check.
4. **Other routes that move the target on S.**
   - **Covered:** `g_open` is projected. Weight decay is inside `u_ctx`.
   - **Not covered:** SIGReg is never projected (as specified) and can still move `z_t` on S.
     It is visible in `lreal_S`. If `lreal_S` stays clearly positive in `both` while
     `leak_cos_post` ≈ 0, SIGReg (or second-order effects) is the remaining route.
5. **The gap misses slow learners.**
   - **Risk:** `P̄` lags about 100 steps. A direction improving 1% per 100 steps has a relative gap
     of about 0.01, below `eps_g = 0.02`. Polyak averaging can even make the gap negative while P
     is noisy. For slowly learned content (hard), only the floor check protects, which brings back
     point 1.
   - **Knobs:** `tau_lag`, `eps_g`, `window`.
6. **A stale floor.** L* comes from snapshots up to `floor_keep × floor_every` steps old, and the
   latents drift. Taking the minimum errs on the protective side: more stays in S, at worst `both`
   behaves like `context`. `n_S` over training shows it.
7. **Over-projection.**
   - **Risk:** early on everything is in S, and `g_ctx` and `g_tgt,S` are correlated through shared
     weights. Projection can then remove useful context learning, and `context`/`both` learn slowly.
   - **Watch:** `proj_ratio` and the slow R² curve early in training. **Knob:** `alpha`.
8. **Capacity versus premise** (addressed: the default is the roomy regime, D = 32, n_slow = 16).
   - **Gain:** a fix does not have to trade slow content for sprites, and the verdict can ask for
     slow within 0.1 of `sg`.
   - **Cost:** the premise now relies on slow content filling the spare room with nonlinear copies.
     `premise()` shows this is nearly free for SIGReg (≈ 7 against ≈ 1), but it is a weaker
     shortcut than in the tight regime. If `sg` keeps the sprites at D = 32 while dropping them at
     D = 16 (stage 4b), the failure is capacity-driven, and the roomy regime is not a test of the
     fix.
9. **The baseline may not fail.** If `sg` keeps fast and hard content anyway, the stage-2 verdict
   says so, and the fix cannot be evaluated. Add pressure: dimmer sprites (`a_fast`), a harder map
   (`hard_k = 3`), stronger SIGReg (`lam`), or D < n_slow.
10. **First-order leak estimates.** `leak_jvp` and `ℓ` are first-order in lr; `ℓ_real` is exact
    but includes everything that moved. Compare them before trusting either alone.

## Open
- **Not run on a GPU yet.** Only CPU tests and a tiny dry run of the notebook have run.
  `eps_g`, `eps_f`, `delta` and `min_dwell` are first guesses.
