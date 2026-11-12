# Two variants for the benchmark

Both keep the content that is predictable from the action but not yet learned, and still drop noise.
They share one encoder across all frames (no BatchNorm: per-sample normalisation keeps the target and
context paths separable), a predictor that sees the context latents and the actions, SIGReg, and AdamW
(lr 1e-3, weight decay 0.01, batch 256 clips, SIGReg weight λ = 0.1, 256 slices). Background and evidence:
[plan.md](plan.md) and [reference_gated_retention.md](reference_gated_retention.md).

**The noise mechanism both use: the floor-gated open target path.** Every 1000 steps (from step 500) the
encoder is frozen, 16 384 fresh clips are encoded and a fresh predictor of the same size is trained on them
for 1500 steps (warm-started across refreshes). Its held-out error per direction is the floor L\*. A direction is
*protected* while the live predictor is still improving on it (relative gap against its lagged copy above 0.02) or
sits clearly above the floor (relative excess above 0.25). The target path may simplify only the directions that are
not protected. Directions are the eigenbasis of `(C_P − C*) v = λ Cov(z) v`, so "not yet learned" and "noise" separate.

## Variant A: target + head start

```
steps < K  (head start)                          steps >= K  (main phase)
x -> f -> z_ctx ─┐                               x -> f -> z_ctx ──────────────┐
                 ├-> P(z_ctx, a) -> p            x -> f -> z_tgt ─┬─(open)─────┤-> P -> p
x -> f -> z_tgt ─┘   (detached: P trains only)                    └─ stop-grad ┘
encoder loss: SIGReg only                         encoder loss: prediction through the context path
predictor loss: MSE on detached latents                          + open target path on unprotected directions
                                                                 + SIGReg
```

- **Idea:** the predictor gets a head start on a latent that only SIGReg shapes, so the prediction loss cannot prune
  content it cannot use yet. After step `K` the encoder is trained normally, and the open target path removes noise.
- **Arm:** `Cfg(arm="target", head_start=K)`. `Cfg(arm="both", head_start=K)` additionally projects the context update
  against the target-side leak; it was the tested configuration, but the projection showed no measurable effect and
  `target` is cheaper (one encoder backward pass less per step). `target` + head start has not been run yet.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `head_start` (K) | 0 (we used 3000–5000) | steps with the encoder on SIGReg only. **The one that matters.** |
| `floor_start`, `floor_every` | 500, 1000 | when the floor probe first runs and how often it refreshes |
| `floor_steps`, `floor_pool` | 1500, 16 384 | training steps and clips of the floor probe |
| `floor_keep`, `floor_margin` | 3, 0.1 | floor = minimum over the last 3 refreshes × 0.9 |
| `eps_g`, `eps_f`, `delta` | 0.02, 0.25, 0.01 | gap and excess thresholds; offset in the excess |
| `window`, `tau_lag`, `min_dwell` | 200, 0.99, 500 | gap window, lag of the lagged predictor, minimum dwell in the protected set |

**Choosing K.** It has to cover the predictor's learn time on the action-conditioned content. In our world that was
about 700 steps for a shift-type feature and 2100–3000 for a harder map, measured on ground truth; with the encoder
in the loop it needed 3000–5000. Without ground truth, use a plateau rule: release when the predictor's loss on the
detached latents has stopped improving (not implemented). Cheap check: probe the action-conditioned feature at
steps 25–500 of a baseline. If it decays there, the head start is needed.

**Logs.** `n_S` (protected directions), `gap_mean`, `excess_med`, per-group probe R² if labels exist, and the loss.

**Known limits.** One synthetic world, 1–3 seeds. A feature the SIGReg-only latent never represents (small,
low-contrast) cannot be protected. Doubling-type content survived only where the predictor had reached R² ≈ 0.5
before release. After release the gate may mark a slow learner as saturated.

## Variant B: reconstruction (reference-gated retention)

```
x -> g (shadow encoder) -> z_s ──> decoder -> x̂           loss: SIGReg + reconstruction (no prediction loss)
                            └──> Q (shadow predictor, detached inputs, + lagged copy Q̄)
                                   └─> reference map every 50 steps: rho_j, gap_j per direction v_j
x -> f (live encoder)   -> z   ──> P (live predictor) -> p
                            └──> R (linear head) ~ whitened z_s     retention loss, weight w_j per direction
live encoder loss: prediction (context path) + open target path (floor-gated) + SIGReg + λ_ret · L_ret
```

- **Idea:** decide what may be discarded from predictability measured where the live encoder cannot damage it. The shadow
  keeps the content; its predictor says what is predictable in principle; the live encoder must keep that content readable
  until its own predictor has caught up.
- **Per direction** (whitened shadow coordinates, `v_j` the eigen-directions of the shadow predictor's residual covariance):
  `err_j` is the unexplained fraction, `ρ_j = 1 − err_j`, `gap_j` the relative improvement of `Q` over `Q̄`, `err_live_j` the
  error of the live predictor's output mapped through `R`.
  ```
  w_j   = max( clip(gap_j / ε_g, 0, 1),  ρ_j · clip((err_live_j − err_j) / ε_f, 0, 1) )
  L_ret = (1/D) · E_frames[ (R f(x) − y)ᵀ · V diag(w) Vᵀ · (R f(x) − y) ],   y = whitened shadow latent
  ```
  A direction is retained while the shadow predictor still improves, or while it is predictable and the live predictor is
  behind the floor. It is free when settled and unpredictable (noise), and released when the live predictor has caught up.
  Before the first refresh everything is retained.
- **Arm:** `Cfg(arm="retain_open", shadow_rec=1.0)`. `retain` is retention without the open path (noise stays).

| Hyperparameter | Default | Meaning |
|---|---|---|
| `shadow_rec` | 1.0 | reconstruction weight of the shadow. **0 (SIGReg only) fails on the sprites.** |
| `ret_lam` | 1.0 | weight of the retention loss |
| `ret_every`, `ret_tau` | 50, 0.995 | refresh period and EMA of the reference statistics |
| `ret_lag` | 0.999 | lag of `Q̄`: a learner slower than `ε_g` per lag period counts as settled |
| `ret_eps_g`, `ret_eps_f` | 0.02, 0.1 | improvement that counts as unsettled; excess over the floor that keeps retention |
| floor-gate parameters | as in variant A | for the open target path |

**Needs.** A reconstruction target (pixels, or features of a frozen encoder), a decoder, a second encoder and predictor
(about 1.7× the compute), and a latent roughly the size of the content (D = 32 for 32 factor dimensions worked).

**Logs.** `ref_w` (mean retention weight), `ref_unsettled` (directions the shadow predictor still improves on), `ref_free`,
`ref_rho` and `ref_auc` (they need group labels; without labels use the distribution of ρ), `sh_r2_*` (what the shadow
holds, with labels), `ret_loss`, `shadow_loss`. **If most directions stay unsettled, the release rule is not working.**

**Known limits.** One synthetic world, one seed at 20 000 steps (hard 0.989, fast 0.981, noise 0.149 with a swing of about
0.12). At D = 64 the retention never released (55 of 64 directions unsettled), so noise stayed at 0.76: the release rule
is untested for larger latents. The shadow trains for the whole run.

## Choosing

| | A: target + head start | B: reconstruction |
|---|---|---|
| Cost | about 1× | about 1.7× |
| Needs | a head start length K | a reconstruction target and decoder |
| Hyperparameters that matter | `head_start` | `shadow_rec`, `ret_lam`, `ret_eps_*`, `ret_lag` |
| Evidence | fast kept (2 seeds), hard in 1 of 2 seeds (with `both`) | all four groups at 20 000 steps, one seed |
| Main risk | K is hand-set | the release rule at larger latents |

Control for both: `sg` (stop-gradient target); `lewm` or `ema` if those are your standard baselines.
