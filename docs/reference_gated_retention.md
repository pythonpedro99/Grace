# Reference-gated retention: keeping not-yet-learned content in a shared-encoder world model

**Status:** proposal. Implemented as arms `retain` and `retain_open` (`gaf/retain.py`, `gaf/train.py`) and
tested in notebook stage 8. The first stage 8 runs (12 000 steps, 2 seeds) kept slow, fast and hard content without any
head start or labels, but several runs had **not settled** when they stopped (noise and hard still moved by 0.07–0.16 over
the last 4000 steps, and the retention was still releasing), so their numbers are provisional. Stage 9 repeats the key
shadow for 20 000 steps as a 2 × 2 (SIGReg-only or reconstruction, D = 32 or D = 64), and the notebook reports the change over the last steps
(`A.drift`). Everything called a finding
comes from one synthetic world and 1–3 seeds.

## 1. The problem

A JEPA-style world model has one encoder `f` shared by all frames, a predictor `P` that maps the
context latents and the actions to the next latent, and the loss `MSE(P(z_ctx, a), z_next) + λ·SIGReg(z)`.
SIGReg pushes the latent towards an isotropic Gaussian. It fixes how much variance the latent has, not
what content fills it.

Real worlds contain content that is predictable *given the action* (an object that moves when pushed) but
that the predictor has not learned yet. Until it is learned, that content looks like noise: it adds
unexplained variance to the prediction error. The question is whether training keeps it.

## 2. What we saw

The test world (`gaf/env.py`): 32×32 RGB clips of three frames, 2-D actions, latent size 32, four kinds of
content. **Slow** patterns (predictable without the action), a **fast** sprite (moved by the action: a
shift), a **hard** sprite (the doubling map of its position, plus the action), and **noise** patterns
(fresh every frame, the only thing that should be dropped). Ridge and MLP probes from `z` measure what is kept.

- **Baselines.** A full-gradient target (`lewm`) deletes both sprites and the noise. A stop-gradient target
  (`sg`) keeps the noise and loses the sprites. A lagged (EMA) target keeps the fast sprite only in the seeds
  where the sprite is represented early.
- **The loss happens early and smoothly.** In `sg` the fast sprite starts to decay at about step 50 and halves
  about every 100 steps. An encoder trained by SIGReg alone builds it up in the same period. The predictor needs
  about 700 steps (fast) to 2100 (hard, from clean factors) to learn a sprite.
- **The leak is not the cause.** The gain-attribution idea (project out the target-side leak of the context
  update) works as designed but changes no outcome. The floor-gated open target path is what removes noise
  (0.91 → 0.09 with slow content kept).
- **A predictor head start fixes the shift sprite.** If the encoder sees only SIGReg for the first 3000–5000 steps
  while the predictor trains, the fast sprite is kept and learned (0.99, two seeds), and `both` also removes the
  noise.
- **The doubling sprite is different.** At equal availability and equal predictor skill (R² 0.2) the shift sprite
  is reinforced after release and the doubling sprite pruned within 50 steps. It survived only in the seed where the
  predictor had reached 0.5 before release.
- **Availability varies.** The white sprite appears in the latent early (R² 0.47 at step 500), the red one late
  (0.01–0.15), and in one seed no sprite appears at all.

## 3. Diagnosis

1. **A race with positive feedback.** The prediction loss rewards keeping a feature only after the predictor can
   use it, and rewards deleting it before. Once deleted, nothing is left to learn. Outcomes are bistable and depend
   on the feature (a steeper map needs a better predictor first) and on the seed.
2. **Availability.** SIGReg asks for no particular content; a random encoder takes what is salient first. A feature
   that is not in the latent cannot be protected by any rule that works on the latent.
3. **The gate's evidence is conditional on today's latent.** The floor probe and the lagged-predictor gap measure how
   well the predictor does on the current latent. When the predictor has learned what is there, both read
   "saturated", although the latent carries the feature only partly and would carry it better if kept. The gate cannot
   separate "not yet learned" from "unlearnable".
4. **The head start is hand-set.** Its length came from ground-truth learn times, which a real benchmark does not offer.

## 4. The method

**Principle:** decide what the encoder may discard from predictability measured in a space the live encoder
cannot damage, and keep the content until the live predictor has caught up with it.

Components (all unsupervised; observations and actions only):

1. **Shadow encoder `g`.** Same architecture as `f`, trained by SIGReg plus a reconstruction decoder `D`, never by the
   prediction loss. SIGReg keeps the latent informative and white; the reconstruction rewards representing any
   pixel content (this is the remedy for availability, and an ablation with SIGReg only exists).
2. **Shadow predictor `Q`.** Predicts `g(x_{t+1})` from `g(x_{≤t})` and the actions, on detached latents, so it can
   learn for as long as it needs. Its lagged copy `Q̄` (EMA, lag ~1000 steps) defines "still improving".
3. **Reference map.** In whitened coordinates `y = W(g − μ)`, `W = Cov(g)^(-1/2)`, take the eigen-decomposition of the
   residual covariance of `Q`: `W·C_res·W v_j = err_j v_j`. Then `err_j` is the unexplained fraction of direction `v_j`
   and `ρ_j = 1 − err_j` its predictable share (noise: `ρ ≈ 0`). `gap_j` is the relative improvement of `Q` over
   `Q̄` along `v_j`. All statistics are EMAs over fresh clips, so they are held-out errors.
4. **Retention loss on the live encoder.** A linear head `R` (trained jointly) must read the shadow latent from the live latent:
   `L_ret = (1/D)·E[(R f(x) − y)ᵀ M (R f(x) − y)]`, `M = V·diag(w)·Vᵀ`, over all frames. The weights are
   ```
   w_j = max( clip(gap_j / ε_g, 0, 1),                      # shadow still improving: retain (protect by default)
              ρ_j · clip((err_live_j − err_j) / ε_f, 0, 1) ) # predictable, live predictor still behind the floor: retain
   ```
   `err_live_j` is the error of the live predictor's output, mapped through `R`, against `y` along `v_j`. A direction is
   free when it is settled and unpredictable (noise), or when the live predictor has caught up (release). Before the
   first refresh everything is retained. Using `M` (not the basis `V`) keeps the loss stable when eigenvectors of degenerate
   directions rotate between refreshes.
5. **Noise removal.** Unchanged from the gain-attribution method: the floor-gated open target path of the live
   encoder (`retain_open`). `retain` is retention alone.

The live encoder minimises `L_pred (+ open path) + λ·SIGReg + λ_ret·L_ret`; the shadow minimises
`MSE_Q + λ·SIGReg + λ_rec·MSE_rec`.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `shadow_rec` | 1.0 | reconstruction weight of the shadow (0: SIGReg only) |
| `ret_lam` | 1.0 | retention loss weight |
| `ret_every`, `ret_tau` | 50, 0.995 | refresh period and EMA of the reference statistics |
| `ret_lag` | 0.999 | lag of `Q̄`: how slow a learner may be and still count as improving |
| `ret_eps_g`, `ret_eps_f` | 0.02, 0.1 | relative improvement that counts as unsettled; excess that keeps retention |

## 5. What it assumes, costs, and where it can fail

- **Cost:** a second encoder, a decoder and a predictor (about twice the compute), plus a linear head. Retained
  directions occupy live capacity until released.
- **Assumption:** whatever the shadow keeps and its predictor can predict is worth keeping. Predictable but useless
  content is retained too, until the live predictor catches up.
- **Fails if the shadow lacks the content** (small, low-contrast features the reconstruction ignores), or if sprites compete for
  the shadow's capacity, as they did in the live encoder.
- **Fails if the shadow predictor is misread as settled:** a learner improving less than `ε_g` per lag period. Hence the long lag.
- **Retention may never release** if `R` is imprecise (its error exceeds `ε_f`). This is the conservative direction.

## 6. How it is tested (notebook stage 8)

Six runs of 12 000 steps, no head start: `retain_open` on the default world (seeds 0 and 1) and with the white hard sprite
(seeds 0 and 1); `retain_open` with a SIGReg-only shadow; `retain` alone. Compared with `sg` and with the best earlier fix,
`both` + head start. Success criteria, fixed in advance: fast ≥ 0.9 (default world) and hard ≥ 0.7 (white world) in both
seeds, noise ≤ 0.3, slow ≥ 0.9, and by step 2000 `ref_auc ≥ 0.9` with `ref_rho_fast`, `ref_rho_hard ≥ 0.1` (the reference
separates content from noise, and sees the sprites, without labels). Diagnostics logged: `sh_r2_*` (what the shadow
represents), `ref_rho_<group>` and `ref_n_<group>` (ρ and number of directions of each group), `ref_auc`, `ref_w`,
`ref_unsettled`.

**Pilot, before any retention training.** On three frozen SIGReg-only encoders, with a predictor trained on their latents,
noise-dominated directions all have ρ = 0.000 and slow-dominated ones 0.12–0.32 (AUC of ρ separating content from noise
0.88–0.96). The sprites, however, form no direction of their own (no fast-dominated direction, at most two hard-dominated
ones, ρ ≈ 0.007), so in a SIGReg-only shadow the weights would free them once the shadow predictor settles. Whether the
reconstruction shadow represents them sharply enough to change this is what stage 8 tests; if not, ρ per linear direction
needs a softer or nonlinear version.

## 7. Open questions

Whether the shadow represents sprites that SIGReg alone misses; whether `ρ` is stable early enough; whether retention holds
the doubling sprite in the seed where every earlier fix failed; whether the cost is worth it against a fixed head start plus
a faster predictor; and whether any of this carries over from the synthetic world to a real benchmark.
