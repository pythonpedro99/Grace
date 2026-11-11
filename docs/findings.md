# Findings: availability, retention and dynamics in the grid world

**Status:** working notes, November 2026. Everything below comes from one synthetic world (the Craftax-shaped grid world, `gaf/env_grid.py`),
mostly **one seed**, runs of 5k–100k steps. Numbers are the readouts at the stated step. Treat single-seed results as directions, not results.
Section 7 lists the beliefs that were later revised.

**Goal:** a self-supervised world model that keeps the content Craftax needs (inventory, the faced tile, mobs, the agent) where LeWM-style JEPAs
lose it ("race and prune": the prediction loss deletes content the predictor has not learned yet), first understood in the grid world, then
tested on Craftax (the AAC repo).

---

## 1. Setup in brief

**Grid world.** 48×48 frames: a 12×8 random map (floor, wood, stone, water; a new map per clip) and 4 HUD rows. Clips of 3 frames, 5 actions.

| entity | what it is | kind | pixel mass |
|---|---|---|---|
| hand | the material in front of the agent, in a HUD tile | fixed place, depends on position, facing and map | 0.88 % per tile |
| inventory | 16 HUD slots, 10 levels, slow random walk | fixed place, slow | 1.24 % per slot |
| agent position | where the agent stands | moving, slow | ~10 % (agent tile) |
| facing | marker on the agent (`agent_shape="marker"`) or the agent's arrow silhouette (`"arrow"`) | moving, fast, action-driven | 3.0 % (marker), 4.3 % (arrow) |
| mob | jumps randomly (`mob_mode="jump"`, noise) or chases the agent (`"chase"`, predictable) | moving | 3.7 % |
| twinkle | 4 floor cells with fresh random brightness every frame | noise | 3.2 % |

Pixel mass = the share of the dataset's pixel variance a component explains (`component_masses`).

**Arms** (`gaf/train.py`): `sg`, `lewm`, `ema` (JEPA baselines), `detached` (SIGReg only), `ae` (reconstruction + SIGReg), `retain` /
`retain_open` = **B**: a shadow encoder (reconstruction + SIGReg, never shaped by prediction) whose content a reference map asks the live encoder
to keep (`docs/reference_gated_retention.md`), `retain_open` adding the floor-gated open target path. `joint` (new): one encoder with
reconstruction + prediction + SIGReg.

**Readouts.** Ridge and MLP probes per part (`pm_*`), mechanism diagnostics (`gaf/mech.py`), and decoder-based readouts (`gaf/readout.py`).
**Probes under-read compressed latents** (section 6), so content claims should rest on decoders.

---

## 2. The pixel mass law (`grid_mass_law.ipynb`, D = 32 unless stated)

1. **What an encoder holds follows the content's share of pixel variance.** Hand readout for 1/2/4/8/16 hand tiles (0.88–12.5 %):
   - `ae` (reconstruction): −0.09 / 0.50 / 0.98 / 0.96 / 0.98, crossing 0.5 at about 1.75 % (mass × D ≈ 0.56);
   - `detached` (SIGReg only): never (0.25 at 12.5 %).
2. **D lowers the threshold for reconstruction, not for SIGReg alone.** `ae` crossing: 1.75 % (D 32), 0.40 % (D 64), 0.34 % (D 128), so
   not a clean 1/D. `detached` has no crossing at any D. The threshold κ is model-dependent, not universal.
3. **Frequency trades for size at equal mass.** At 1.75 %, readouts were 0.50 / 0.49 / 0.42 / 0.42.
4. **Location is a barrier.** The facing shown at a fixed place (a compass HUD tile, 3.3 %) is held at 0.96; the moving marker (3.0 %) at
   −0.08. Fixed-place content (hand, inventory) is held, moving content (agent position at 10 %, mob, marker) is not.
5. **Class-balanced reconstruction moves the line.** Each tile's error is weighted by its ground-truth class share of variance to the
   power −1, capped at 30×.
   - It lifts the hand at 0.88 % from −0.09 to 0.92–0.94 and the agent position from −0.1 to 0.72–0.75.
   - It costs inventory (0.78 → 0.30–0.44 at D 32).
   - Facing marker, mob and twinkle stay at about 0. Power 0.5 is worse.
6. **Class weights have a side effect:** terrain is pushed to the lower cap (weight about 0.03) and the mob to about 0.19. The weighted shadow's
   own decoder never reconstructs either (section 6).
7. **Label-free class weights** (`rec_clusters`, `rec_pos_types`): k-means on tile pixels alone fails, because a hand tile looks exactly like a
   map tile (92–97 % of hand tiles land in terrain clusters). Adding a position type from each position's change rate (hand 47 %, map 3 %)
   recovers weights close to the ground truth (hand about 30× the map weight). Twinkle is over-weighted about 5× (no predictability guard).
   Not yet tested in B.

Earlier context (`sigreg_hypothesis.ipynb`): in the clip world a sprite is held by SIGReg alone only at about 7.7 % pixel share. Scaling the
whole frame changes nothing (relative share matters). SIGReg on latent differences does not recover sprites. In the grid world, the JEPA
baselines' latent is about 70–90 % the random map and holds nothing labelled.

## 3. B at D = 128 (`grid_b_d128.ipynb`; arrow agent, chasing mob, hand at 1 tile)

Final strong probe, live encoder:

| run | hand | position | inventory | facing | mob | twinkle | rank |
|---|---|---|---|---|---|---|---|
| plain B, 15k | 0.91 | −0.23 | 0.93 | −0.18 | −0.21 | 0.81 | 90 |
| plain B, 30k | 0.96 | −0.22 | 0.96 | −0.19 | −0.22 | 0.84 | 102 |
| weighted B, 15k | 0.07 | 0.96 | 0.76 | 0.08 | −0.12 | 0.24 | 46 |
| weighted B, 30k | 0.00 | 0.96 | 0.82 | 0.09 | −0.10 | 0.55 | 46 |
| `sg` (no shadow), 15k | −0.10 | −0.10 | −0.09 | −0.11 | −0.12 | −0.11 | 56 |

- **`sg` at D = 128 holds nothing**, so what B holds is B's doing, not D's.
- **The weighted run's live encoder loses the hand while its shadow keeps it** (shadow 0.90–0.93).
- **The plain run also loses the hand, just later.** 100k runs read at 65k with the 12k-clip eval probe:
  - Plain: hand 0.86 at 50k, then 0.25 at 65k, exactly as the mean retention weight falls to 0.09 (93 of 128 directions free) and the
    gate set empties (`n_S` → 0). Twinkle falls 0.85 → 0.71 over the same window.
  - Weighted: settled by 65k at hand 0.00, position 0.97, inventory 0.92 and facing 0.05 (probe). Twinkle peaks at 0.55 (40k) and decays
    to 0.41. Live rank 26 against the shadow's 121.
- **Diagnosis from the logs:** the shadow predictor cannot predict the hand (`sh_pred_r2_hard` 0.02 → 0.09 over 30k steps;
  `ref_rho_hard` 0.03–0.16). The reference map treats it as noise and releases it.
- **The plain run is a natural experiment.** At about 6k its retention weight dips and the live hand dips (0.65 → 0.30). When the weight
  re-engages, the hand recovers (0.73 by 10k).

## 4. The mechanism stage (`grid_b_mechanism.ipynb`; 2×2 at 15k, class-weighted)

Open target path on or off (`retain_open` / `retain`) × retention as is or floor on all (`ret_floor_all`, every direction keeps weight ≥ 0.3).

| | live hand (final probe) | live rank | shadow hand |
|---|---|---|---|
| open on, normal | 0.01 | 39 | 0.93 |
| open on, floor all | **0.82** | 42 | 0.91 |
| open off, normal | 0.03 | 113 | 0.89 |
| open off, floor all | 0.04 | 116 | 0.90 |

1. **The hand's release is abrupt.** In the normal runs the retention weight on the hand's directions falls from 1.0 at 4k to 0.03 at 6k.
   The retention head's fit of the hand falls 0.93 → 0.26 (6k) → −0.08 (12k).
2. **A floor of 0.3 alone does not hold it.** With the open path off, the hand dies between 9k and 12k (head fit 0.60 → 0.00).
   - Prediction and retention gradients conflict there (cosine −0.45 to −0.52).
   - The prediction gradient is about twice as large as with the open path on (0.08–0.11 against 0.04–0.08).
   - Only floor plus open path held it in this seed, and the hand oscillated with the gate state.
3. **The open target path causes the live rank collapse**, not SIGReg's weight and not retention. Rank is 39–42 with the open path on and
   113–116 with it off; the shadow is at 119. Within a run, rank falls to 19–22 when the gate protects only about 2 directions.
   - Live SIGReg reads 2.0–3.4 with the open path on and 1.35 with it off. SIGReg pushes against the open path (gradient cosine −0.28 to −0.84).
   - SIGReg at 0.01 keeps full rank when the open path is off.
4. **Noise is held by retention, not removed by the open path.** Live twinkle (decoder) is 0.41–0.47 with the open path on, 0.03 with it off
   and normal retention, and 0.39 with the floor.
5. **SIGReg sees rank collapse exactly** (Gaussians at D = 128: 1.05 at full rank, 1.17 at participation ratio 100, 2.85 at 26, 6.6 at 10).
   At weight 0.01 the price of a collapse to 26 dimensions is only about 0.02, the size of the prediction loss.

## 5. Rescue and hold (from `ckpt_10000` of "open off, floor all")

Training restarted for 1500 steps with one force changed. Shadow and reference map frozen, fresh optimiser, probe with 6000 clips.

| variant | hand | position | inventory | twinkle | rank |
|---|---|---|---|---|---|
| before | 0.02 | 0.87 | 0.47 | −0.19 | 115 |
| continue as is | −0.08 | 0.88 | 0.47 | −0.17 | 117 |
| retention ×3 | 0.52 | 0.82 | 0.42 | 0.05 | 120 |
| retention ×10 | 0.70 | 0.79 | 0.39 | 0.16 | 121 |
| no prediction | 0.53 | 0.79 | 0.38 | 0.16 | 120 |
| no prediction + retention ×3 | 0.67 | 0.76 | 0.39 | 0.17 | 121 |

- **The hand is outvoted, not irreversibly lost.** Prediction erodes it against a retention weight of 0.3, and capacity is not the limit.
- **Retention ×3 for 4500 steps holds the hand at about 0.5 but erodes slowly**: 0.56, 0.53, 0.47 at +1.5k, +3k, +4.5k. Position, inventory,
  rank and noise stay stable. A force balance is fragile.

## 6. What the latents hold: decoders against probes

**Decoder-strength readout.** A fresh class-weighted decoder on each frozen latent at 15k steps. Each entry is the share of the
"decode nothing" error removed (1 = perfect). Facing is the orientation accuracy of the decoded agent tile (chance 0.25). Values are live / shadow:

| run | hand | agent | inventory | twinkle | facing |
|---|---|---|---|---|---|
| open on, normal | 0.18 / 1.00 | 0.88 / 0.91 | 0.77 / 0.80 | 0.41 / 0.68 | 0.85 / 0.95 |
| open on, floor all | 0.94 / 1.00 | 0.87 / 0.90 | 0.75 / 0.76 | 0.47 / 0.67 | 0.83 / 0.93 |
| open off, normal | 0.23 / 0.99 | 0.74 / 0.91 | 0.72 / 0.76 | 0.03 / 0.66 | 0.45 / 0.93 |
| open off, floor all | 0.28 / 1.00 | 0.82 / 0.90 | 0.72 / 0.77 | 0.39 / 0.68 | 0.71 / 0.93 |

- **The facing is in the shadow, and largely in the live encoder.** Every probe read about 0.1 (ridge 0.02, small MLP 0.05, a classification
  MLP 47 % against the decoder's 95 %; adding the agent's position as input does not help). It appears in the shadow as a sharp transition:
  - decoder accuracy 0.35 (6k) → 0.56 (9k) → 0.93 (12k) → 0.95 (15k);
  - agent-tile reconstruction error 0.47 → 0.06;
  - latent sensitivity to the facing 0.37 → 1.55 (about the same as for position or twinkle).
- **The hand: probes were right.** Released in three runs, kept by the shadow everywhere.
- **Mob 0.07–0.09 and terrain −0.05 everywhere**, but this decoder used the class weights (mob 0.19, terrain 0.03), so it under-reads them. The
  shadow's own decoder never reconstructs either (error flat for 15k steps). So the weighted objective does not ask for them; absence from the
  latent is not proven.
- **Eval-time probes under-read**: the held hand reads 0.46 on the 12k-clip eval probe and 0.82 on the final probe, at the same step.

## 7. Dynamics: holding is not foreseeing

**Shadow + its predictor `Qs` against live + `P`** (from `ckpt_15000`):
- **The live latent is about 5× more predictable.** Prediction error over variance is 0.08–0.14 against 0.42 for the shadow; copying the last
  frame gives 0.21–0.30 against 0.72–0.74. Part of that comes from what the live latent dropped.
- **The shadow holds the action-driven content but does not foresee its change.** Decoded with its own decoder:

| | agent | hand | facing accuracy |
|---|---|---|---|
| actual next frame | 0.97 | 1.00 | 0.95–0.97 |
| predicted | 0.51 | −0.30 | 0.45–0.46 |
| copy last frame | 0.39–0.40 | −0.21 to −0.35 | 0.40–0.42 |

**Predictor forks on the frozen shadow** (3000 steps). Values are agent / hand / facing on move steps (ceiling 0.97 / 1.00 / 0.96; copy-last
0.40 / −0.25 / 0.28):
- `Qs` at the checkpoint: 0.49 / −0.32 / 0.36;
- `Qs` continued, latent loss: 0.51 / −0.26 / 0.39 (training time alone barely helps);
- `Qs` continued, decoded class-weighted pixel loss: 0.71 / −0.58 / 0.59;
- from scratch, plain MLP, decoded loss: 0.07 / −0.05 / 0.27;
- from scratch, residual action-gated predictor (`FiLMPredictor`), decoded loss: 0.45 / −0.23 / 0.38;
- from scratch, `FiLMPredictor`, latent loss: 0.59 / −0.07 / 0.46.

No predictor gets near the ceiling. The facing, the simplest action effect, stays hard in the shadow's latent. **The reconstruction latent is not
shaped for dynamics.** The hand's dynamics need the terrain next to the agent, which the weighted shadow probably lacks.

## 8. Theory: when LeJEPA learns a world model (Klindt, LeCun, Balestriero, arXiv 2605.26379)

- **Linear identifiability needs Gaussian latents.** Non-Gaussian ones are identifiable only up to a monotone map.
- **The rates must be close.** Under isotropic transitions all latents are recovered by the joint objective. With different rates, "the fastest
  latent variable must not be more than twice as fast as the slowest" (`max K < 2 min K`, appendix "The role of isotropic transitions").
  Otherwise the encoder takes higher harmonics of slow latents before the first harmonic of fast ones.
- **Our world violates both.** Lag-1 autocorrelation: inventory 0.998, position 0.984, chasing mob 0.983, hand 0.34, facing 0.20, twinkle
  0.00. The fastest-to-slowest rate ratio is about 4700×. Categorical and bounded latents are not Gaussian. This explains the JEPA path's bias
  against fast content. It says nothing about the reconstruction shadow.
- **VICReg** (second-moment whitening) matches SIGReg for Gaussian latents. The theorem needs only whitening of the embedding. For non-Gaussian
  latents SIGReg is *more* robust than VICReg ("SIGReg's Gaussianization of h is more robust to non-Gaussian latent variable distributions than
  whitening").

## 9. Revised beliefs

| earlier | now |
|---|---|
| the facing is absent from the shadow and creeps in slowly | it is present (decoder 0.93–0.95 by 12k) and the probes under-read it |
| plain B is settled by 15k and keeps the hand | it loses the hand at about 55k, when the retention weights decay |
| the weighted hand loss is a quirk of the class weights | the same release mechanism, just earlier |
| noise gets worse with training | twinkle peaks at about 40k, then decays |
| the live rank collapses because SIGReg at 0.01 is too weak | the open target path collapses it; SIGReg 0.01 keeps full rank without it |
| the open path removes noise | retention release removes noise; the open path does not |
| a decoded prediction loss is the fix for dynamics | it helped a pretrained predictor; from scratch the residual predictor with latent loss did better; nothing got close |
| the live encoder does not earn its cost | its latent is about 5× more predictable, a geometry the dynamics need |
| κ ≈ 1.5 is universal | κ is model-dependent (about 0.56 for `ae` at D 32) |
| crops raise a class's mass | spatial crops do not raise the mass of uniformly spread classes |

## 10. What this means for the architecture

**The flaw in B's design.** The reference map releases a direction when the shadow predictor cannot predict it. That cannot separate "not
learned yet" from "unlearnable", and hard but real content (the hand) is released like noise. Prediction then erodes it. Every tuning only moves
a fragile force balance: lags, floors, weights, the open path.

**Two properties, split across two encoders.** The shadow (reconstruction) **keeps the content**. The live encoder (prediction) **makes the
latent predictable**. B joins them only by a linear readout, so neither latent has both.

**The recipe these results point to:**
1. **One encoder with reconstruction + prediction + SIGReg** (the `joint` arm, `grid_joint.ipynb`). Reconstruction keeps the content and
   prediction shapes it for dynamics.
2. **Balanced reconstruction without suppressing what the dynamics need.** A softer weight cap, so terrain stays (the hand depends on it), and
   label-free weights (clusters with position types) once validated.
3. **A noise criterion based on independence, not on today's predictor skill.** A direction is noise if it is independent of past frames and
   actions, judged by a strong estimator trained long. Lag-1 autocorrelation is the linear special case. Noise that persists over time can't
   be told apart from signal without a task.
4. **A predictor able to learn action effects.** Residual, with the action multiplying the hidden state (FiLM), at least.
5. **Mob handling.** The small, moving, rare entity is the Craftax failure that motivated all this, and **nothing here encodes it**, not even
   the shadow. Candidates: weights from temporal change, patch- or object-level decoders, a much softer cap.
6. **Decoder-based evaluation.** Probes are only a lower bound.

## 11. Open questions, in order

1. **The joint arm** against the shadow alone (content and predicted frames, both decoded), then the fade and the prediction-distillation
   anchor as forks of its checkpoint (`grid_joint.ipynb`, `gaf/fork.py`; ground-truth class weights with a softer cap of 4 so terrain and mob
   are not zeroed). Not run yet.
2. **Terrain and mob in the latent:** an unweighted decoder readout, and a softer class-weight cap.
3. **An independence-based noise criterion** against the current release rule.
4. **Seeds.** Above all the floor-and-open-path interaction and the joint arm.
5. **Why release came at 5k (weighted) but 55k (plain).** Needs the per-group retention logs in a plain run.
6. **Transfer to Craftax.** There, A5 (no prediction loss) lost mobs and ore just like B2/V3, so prediction was not the cause; pixel mass
   (see AAC `reports/pixel_mass`) and availability are.

## 12. Instruments and pitfalls

- **Probes:** eval-time probes (`ev_pm_*`, 12k clips, 1000 steps) under-read the final probe (20k clips, 2000 steps), which under-reads a
  decoder. Use decoders for content (`readout.decoder_strength`) and decoded predicted frames for dynamics (`readout.dynamics`).
- **Facing** is readable by a decoder long before any probe; a position-conditioned code is not the explanation.
- **Mechanism logs** (`log_mech`): per-term gradient norms and cosines, first-order attribution per part, per-group retention weights
  (`ref_w_<group>`), the share of each group the gate protects (`S_frac_<group>`), spectra, retention-head fit. Checkpoints (`ckpt_every`) let
  any question be asked of the weights later (rescue tests, decoders, predictor forks).
- **Several runs on one GPU:** final probes used to render tens of thousands of clips at once and ran out of memory with five workers. Sampling
  is now chunked (`probes.collect_last`), and `analysis.finalize` rebuilds the summary of a run that died there.
- **Background runs:** `analysis.launch` (safe to re-run; skips finished and running runs), `progress`, `live_probes` (stops for dead runs).

## Where things are

| what | where |
|---|---|
| env knobs | `gaf/env_grid.py`: `agent_shape`, `mob_mode`, `n_mobs`, `mob_chase`, `hand_tiles`, `hand_contrast`, `hand_p`, `compass_tiles`, `variants()` |
| class weights | `train.class_weights`, `cluster_classes`, `rec_error` (`rec_class_power`, `rec_clusters`, `rec_pos_types`) |
| retention options | `retain.RefMap` (`ret_floor_all`, `ret_w_min`, `ret_ac`) |
| predictors | `models.Predictor`, `models.FiLMPredictor` (`pred_arch`) |
| notebooks | `grid_mass_law`, `grid_b_d128`, `grid_b_mechanism`, `grid_joint` |
| method description | `docs/reference_gated_retention.md` |
