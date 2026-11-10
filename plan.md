# Design decisions and critical points

Where `docs/gain_attribution_one_pager.pdf` is silent or leaves a choice open, this is what the code
does. The second half lists what can make the experiment fail or mislead, and what guards against it.

## Decisions

### Environment
1. **Clips, not chains.** Every step draws fresh i.i.d. clips: T = 3 frames, 2 context and
   1 target. Actions `a_t ∈ [−1, 1]²` (uniform random policy) act between frames. The predictor
   sees all T − 1 actions, the encoder none.
2. **The predictable sprite content is action-conditioned.**
   - Fast: `p' = p + 0.3 a`. Hard: `q' = 2q + 0.3 a (mod 1)`.
   - Without the action, R² is capped at `(sin 2πv / 2πv)²`: 0.25 for both.
     `test_dynamics_need_actions` checks it.
   - At the first draft's `v = 0.12` the cap was 0.8, and the action would have hardly mattered.
   - **`h_act` sets how hard "hard" is, through a plateau.** Ground-truth predictor, MLP on raw
     factors and actions, learn time t90 in steps: `h_act` 0.1 < 750, 0.2–0.25 ≈ 1500, 0.3 ≈ 2250–3000
     (4 seeds, all learned), 0.35 unreliable, 0.4 not learned within 6000 steps, also with
     `hard_k = 3`. The first default, 0.4, made hard unlearnable instead of hard to learn: no
     predictor or probe could ever learn it, so the gate could not tell it from noise. 0.3 is at the
     edge of reliability; fall back to 0.25 if hard is not learned in the real runs.
3. **Slow and noise look identical.** Both are smooth colour patterns from one orthonormal basis.
   Only the dynamics separate them.
4. **Capacity and premise.**
   - The default is D = 32 with n_slow = 16. This is **not** spare room: the factor dimensions are
     16 slow + 8 noise + 4 + 4 sprite = 32, so D is exactly full if noise is encoded too (corrected after
     stage 8: in a SIGReg-only shadow the sprite's readout rises only as slow and noise readouts fall).
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
    - The generalized eigenvectors of `(C_P − C*) v = λ Cov(z_t) v`, from the current predictor,
      the probe and the latents on the same held-out pool, orthonormalized in order (QR).
    - λ is the share of the variance along v that the predictor still misses beyond the floor:
      the gate's excess condition in white coordinates. It does not depend on δ.
    - SIGReg gives no basis itself (Cov(z) ≈ I is fully degenerate), only the metric.
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
     (large λ) separates from learned content and noise (λ ≈ 0). Inside the λ ≈ 0 subspace the
     directions are arbitrary, but all of them are open anyway.
   - **Remaining risk 1:** the basis is fixed between refreshes (every `floor_every` steps), while
     the predictor learns. Directions whose content becomes learned stay in S until their excess
     falls; that error is on the protective side.
   - **Remaining risk 2:** if the probe has not learned a sprite (point 1), C\* ≈ C_P on its
     directions. λ ≈ 0, and the basis cannot save it.
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
8. **Capacity versus premise** (the default D = 32 is exactly the number of factor dimensions, so it is
   tight once noise is encoded; the "roomy regime" wording below was wrong).
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

## Bases considered

All in white coordinates (Cov(z) ≈ I under SIGReg). Values per direction type: slow / fast / hard /
noise.

| Matrix | Meaning of vᵀAv | Values | Separates |
|---|---|---|---|
| I (dims), Cov(z) (PCA) | –, variance | – / 1 / 1 / 1 | nothing |
| Cov(z_{t+1} − z_t) (SFA = AMUSE) | 2 (1 − autocorr.) | 0.04 / 1 / 2 / 2 | slow vs. rest |
| C\* (`resid`) | irreducible error share | 0.04 / 0 / 0 / 1 | noise vs. predictable |
| C_P | current error share | low / high→low / high / 1 | learned vs. unlearned + noise |
| C_lag − C_P | recent progress | 0 / >0 / small / 0 | currently learning |
| C_P − C\* (`excess`, default) | missed share beyond floor | 0 / high→0 / high / 0 | unlearned vs. rest |
| C\*_no-action − C\* | action-explained share | 0 / 0.75 / 0.95 / 0 | action-driven vs. rest |

Candidates for later: controllability as an extra protection rule (it targets the probe-bias risk for
hard), SFA as a diagnostic, and an S defined as a subspace instead of per-direction state.

## Hypothesis: the sprites are lost in a race (notebook stage 7)

Result of `results_v1` (3 seeds): `lewm` keeps only slow content; `sg` and all three fixes also lose fast
and hard; `ema` keeps and learns fast in 2 of 3 seeds (0.99), not hard; `both` drops noise (0.10) with slow
intact. At the first eval (step 500) `sg`, `context`, `target` and `both` are identical: before the first
floor everything is protected, the open path is closed, and the projection changes the update by under
0.2%. The sprites are already gone, while `detached` still has them (fast 0.33).

Hypothesis: the predictor needs about 700 (fast) and 2100 (hard) steps to learn a sprite. Until then, sprite
variance in the context latents is noise for it, and the context-path gradient (the part the projection
leaves alone as legitimate learning) prunes it from the shared encoder. The target loses it too, so nothing
is left to learn. The lagged target of `ema` keeps the sprites available long enough.

One seed, 7 runs, all with early evals:
- **early** (`sg`, `ema`, `detached`, `both`, 3000 steps): when and how fast the sprites vanish.
- **head start** (`sg`, `both` with `head_start=3000`, 8000 steps): does a predictor head start keep them.
- **block** (20k steps): the encoder is trained on the prediction loss restricted to the open directions,
  through both the context and the target path, while the predictor still trains on all directions. This
  applies the one-pager's rule ("in learning directions only the predictor reduces the loss") to the
  context path as well, instead of projecting only the target-side leak. While nothing has a floor yet,
  the whole latent is protected and the encoder gets only SIGReg, so `block` starts with a built-in
  head start of 500 steps (tested: identical to `detached` until the first floor). Risk: the encoder
  step for the open directions can still move the protected ones through the shared weights.
- **leak by source** (`lk_*` columns in all of them): does the remaining exact leak come from SIGReg.
- **hard, seed 1** (stage 7e): seed 0 never builds hard content under SIGReg alone, seed 1 does (detached
  ridge R² 0.27 at step 2000). `sg` and `both` with `head_start=5000` (12k steps) and `block` (20k steps).
- **why hard is late** (stage 7f): hypothesis of two bottlenecks. Availability: hard enters the SIGReg-shaped
  latent late and seed-dependently (perhaps a colour/salience effect: fast white, hard red). Learn time: a
  predictor needs more than 2000 steps behind a plateau, while the encoder prunes content within about 100.
  Tests: `detached` runs with `swap_colors` (3 seeds, 3000 steps), and `A.latent_learning` on the seed-1
  `detached` encoder after 5000 steps.
- **colour or doubling map?** (stage 7g): results so far show hard is available (seed 1: 0.32), the predictor
  learns it (frozen latents: about 2000 steps, plateau 0.18; during the head start 0.26 at step 5000), the gate
  opens (`n_S` = 0 from step 3000), and hard is pruned within 50–400 steps of the switch, while fast at a similar
  predictive R² is reinforced to 0.99. The colour swap showed availability follows colour (white sprite
  appears early). Runs with a head start of 5000: white hard sprite with `k=2` (`both` seeds 0, 1, `sg` seed 1) and
  white hard sprite with `k=1` (`both`, `sg`, seed 1). Hypothesis for the pruning: the doubling map amplifies
  errors in the encoded position, so the prediction gradient prunes hard instead of sharpening it.
- **reference-gated retention** (stage 8, `docs/reference_gated_retention.md`): the proposed unsupervised fix.
  A shadow encoder (SIGReg + reconstruction, never shaped by prediction) and a shadow predictor give a reference
  floor in a space the live encoder cannot damage; the live encoder is asked to keep the shadow latent's
  predictable part readable (retention loss) until its own predictor has caught up, and directions that are settled
  and unpredictable are free (noise). Arms `retain` and `retain_open`. Six runs of 12k steps; success criteria are
  written in the notebook before the results.
- **settledness (flagged by the user after stage 8):** the stage 8 runs stopped at 12k steps and several had not
  settled: white seed 1 hard +0.16 and noise −0.13 over the last 4000 steps, red seed 1 noise +0.10 in the last
  2000, the SIGReg-only shadow still moving, and the number of unsettled shadow directions still falling in all runs
  (14→9, 19→14, 19→11, 18→10). Their numbers are provisional. `A.drift` reports the change over the last steps of any
  run; stage 9 reruns the shadow for 20k steps as a 2 × 2 (SIGReg-only or reconstruction, D=32 or D=64; runs cannot be resumed, they start again). `drift` also reports the swing inside the window (it catches oscillations
  that end where they started). Do not call a result final unless `settled` is True.
- Skipped for now: the lag ablation (`ema` with τ = 0.9, 0.99).

Reading guide: the table at the end of notebook stage 7. If a head start keeps fast, the remedy is a
schedule, not a better gradient rule. Candidates if not: anchoring S to a lagged encoder, or freezing the
encoder's effect on S in the update.

## Grid world (`grid_benchmark.ipynb`)

The stepping stone between the clip world and Craftax: it tests the core models on discrete, relational content before
the first real benchmark. Design decisions:
- **Slow is the HUD, not a continuous field.** 16 inventory slots drawn as tile brightness, i.i.d. per clip. Checked
  before building, on ground truth: an encoder from the slots to D = 32 trained on SIGReg alone reaches SIGReg ≈ 6, like
  the clip world's Gaussian slow factors (≈ 6.5), but only when each slot enters as an intensity. Fed as one-hot glyphs,
  it stays at ≈ 23–34.
- **Facing and hand instead of positions.** One-cell moves leave a position ≈ 95% predictable without the action (cap
  0.95, against 0.25 in the clip world), so positions are slow content. Facing (cap 0.03) is fast. The hand, the material
  in front of the agent after its action (cap ≈ 0.2), is hard: it needs the action and the blocks around the agent.
- **Fixed map in the notebook, random map as the stress run.** With a new map per clip (the first design) the baselines
  collapsed onto the map: SIGReg at its floor, `rank_z` 31, `wb_ratio` ≈ 0, no labelled content decodable, even slow
  (the map has more entropy than the HUD and nothing labels it). Stage 2c on that env (no facing in the detached latent)
  is confounded by it; redo it on the fixed map before dropping variant A. `capacity()` (stage 2b) measures the D the content needs.
- **One seed per stage.** The last cell of the notebook adds seeds 1 and 2 (optional).
- **Arms:** `detached` (availability, stage 2c), `lewm`, `sg`, `ema`, A (`target` + head start 3000 / 5000), the control
  `sg` + head start, and B (`retain_open`). The ablations of B are skipped for now.

Critical points:
- If `sg` keeps facing and hand, the grid world tests nothing (criterion 1 of the notebook). Pressure: smaller D, more
  inventory slots, lower `density`.
- The hand is learned gradually on ground truth (no flat plateau as with the doubling map). If A keeps it with a short
  head start, it is easier than the clip world's hard sprite; `density` is the knob.
- The mob is the key noise test: a sprite like the agent, separable from it only by its dynamics.

## Open
- **Not run on a GPU yet.** Only CPU tests and a tiny dry run of the notebook have run.
  `eps_g`, `eps_f`, `delta` and `min_dwell` are first guesses.
