# Findings: keeping what a world model can learn, releasing the rest

**Status.** Working notes, November 2026. All results come from one synthetic world (the Craftax-shaped grid world, `gaf/env_grid.py`) and, unless
stated, **one seed**. Read every number as a direction. Content is measured with decoders, not probes (§8). Pre-registered rules are in the notebooks.

## Summary

LeWM-style JEPAs lose small, moving or rare content (Craftax: mobs, ores, the faced tile). We asked what an objective must do so that a latent
**holds everything its predictor can learn within its budget and releases the rest**, while staying predictable. Five steps answered most of it:

1. **Availability comes first.** What an encoder holds follows the content's share of pixel variance and whether it sits at a fixed place. Prediction
   plus SIGReg alone holds almost nothing; reconstruction holds what has enough pixel mass, and class-balanced weights extend that to small and moving content.
2. **A separate reconstruction encoder (B) cannot protect content**, because its release rule ("the predictor cannot predict it") does not separate
   *not yet learned* from *unlearnable*. Hard but real content is released like noise.
3. **One encoder with reconstruction and prediction (`joint`) holds the content and shapes it for dynamics**, but it also holds the noise, and fading the
   reconstruction erodes everything.
4. **A decoded-prediction term** (`||D(P(z, a)) − x_next||`) **keeps what the predictor learned, releases the noise and improves the dynamics** under a
   fade. It loses exactly one thing, the agent's facing, because the predictor can draw the facing from the action without the latent holding it.
5. **A reconstruction of the real frame, weighted per tile by how well the predictor foresaw it (`gate`), closes that gap.** With a target gradient on
   top, the model holds all learned content, releases the noise completely and keeps the best dynamics.

Open: the mob (poorly localised by every model, the instruments disagree), seeds, capacity pressure, label-free weights and transfer to Craftax.

---

## 1. Question and setup

**Question.** A JEPA deletes content its predictor has not learned yet ("race and prune"). Can an objective make content **available** first, let
the predictor learn, **judge** what was learned, and then **consolidate** that and release the rest?

**World.** 48×48 frames: a random 12×8 map per clip (floor, wood, stone, water) and 4 HUD rows; clips of 3 frames; 5 actions (no-op, 4 moves).

| content | what it is | kind | pixel mass |
|---|---|---|---|
| inventory | 16 HUD slots, slow random walk | fixed place, slow | 1.2 % per slot |
| hand | the material in front of the agent (HUD tile) | fixed place, depends on position, facing, map | 0.9 % |
| agent | a white arrow; its orientation is the **facing** | moving; facing set by each move action | ~10 % / 4.3 % |
| mob | steps toward the agent with p = 0.7, else a random step | moving, partly predictable | 3.7 % |
| twinkle | 4 floor cells with a new random brightness each frame | **noise** | 3.2 % |

Pixel mass = share of the dataset's pixel variance the content explains. Lag-1 autocorrelation spans 0.998 (inventory) to 0.00 (twinkle).

**Models** (`gaf/train.py`, `gaf/fork.py`). JEPA baselines `sg` / `lewm` / `ema`; `detached` (SIGReg only); `ae` (reconstruction + SIGReg);
**B** (`retain`, `retain_open`: a reconstruction shadow encoder whose content a reference map asks the live encoder to keep); **`joint`** (one encoder:
reconstruction + latent prediction + SIGReg, FiLM predictor). Later variants are forks of a `joint` checkpoint with a changed objective (§6, §7).

**Readouts** (`gaf/readout.py`). A *fresh decoder* trained on the frozen latent (what the latent holds), the model's *own decoder* on the real next
latent and on the *predicted* latent (what the model foresees). Scores are the share of the "decode nothing" error removed (1 = perfect); facing and
positions are accuracies (facing chance 0.25).

---

## 2. Availability: what an encoder holds follows pixel mass and location

(`04_mass_law.ipynb`, `03_sigreg_mass.ipynb`; D = 32 unless stated)

- **Reconstruction holds content above a mass threshold; SIGReg alone holds none.** The hand is held by `ae` from about 1.75 % pixel mass at D = 32
  (0.40 % at D 64, 0.34 % at D 128); `detached` never holds it (0.25 at 12.5 %). The JEPA baselines' latent is 70–90 % a function of the random map and
  holds nothing labelled; `sg` at D = 128 holds nothing either.
- **Location is a barrier.** The same facing information shown at a fixed HUD place is held (0.96 at 3.3 %), on the moving agent it is not (−0.08 at 3.0 %).
- **Class-balanced reconstruction moves the line.** Weighting each tile's error by its class's inverse share of variance lifts the hand at 0.9 % from
  −0.09 to 0.92 and the agent position from −0.1 to 0.72. A high weight cap (30) starves terrain and mob; a cap of 4 lets `ae` hold mob and terrain
  (own decoder 0.98).
- **Label-free weights** are possible in principle: tile-appearance clusters fail (a hand tile looks like a map tile), clusters plus a position type
  from each position's change rate recover weights close to the ground truth, but over-weight the twinkle about 5×. Not yet tested in a model.
- **Craftax agrees.** With the prediction loss switched off (AAC run A5), mobs and ores are lost exactly as in the LeWM runs; content above about 1.4 %
  pixel mass is kept, content below 0.35 % is lost. The Craftax failure is an availability failure, not only the race.

## 3. A separate shadow encoder (B) cannot protect content

(`05_b_d128.ipynb`, `06_b_mechanism.ipynb`; D = 128, class-weighted)

- **B releases the hand although its shadow holds it.** The weighted run's live encoder loses the hand within 6k steps (probe 0.07, shadow 0.90); the
  plain run holds it until about 50k, then loses it (0.86 → 0.25) as the retention weights decay.
- **The cause is the release rule.** The reference map releases a direction when the shadow's predictor cannot predict it, and that predictor never
  learns the hand (R² 0.02 → 0.09 over 30k steps). "Not learned yet" and "unlearnable" look the same.
- **Released content is outvoted, not lost.** Raising the retention weight 3× or 10× from a checkpoint brings the hand back within 1500 steps (0.02 → 0.52 / 0.70),
  so capacity is not the limit; held by a force balance, it erodes again slowly.
- **Two side findings.** The floor-gated open target path collapses the live rank (39–42 against 113–116 without it), and noise is held or released by
  retention, not removed by the open path.

## 4. Holding is not foreseeing; one encoder is needed

(`06_b_mechanism.ipynb`, `07_joint.ipynb`)

- **A reconstruction latent holds the action-driven content but is not shaped for dynamics.** The shadow decodes facing at 0.95, but its predictor's
  predicted facing is 0.45 against 0.40 for copying the last frame; no predictor trained on the frozen shadow came close to the ceiling.
- **`joint` (reconstruction + prediction in one encoder) holds the content and learns dynamics** no earlier model learned: the predicted hand reaches 0.37
  against −0.26 for copy-last (the shadow alone: 0.00). Longer training keeps improving the predictions.
- **But it holds the noise** (twinkle 0.89), and **fading the reconstruction erodes the content** (facing 0.89 → 0.53). A distillation anchor to a frozen
  snapshot holds the content but stalls further learning.
- **Theory fits the JEPA side.** LeJEPA's identifiability result (Klindt et al., arXiv 2605.26379) needs latents of similar rates (fastest < 2× slowest);
  here the ratio is about 4700×, which explains why prediction alone favours slow content.

## 5. The decoded prediction keeps what the predictor learned, except action-determined content

(`08_joint_dp.ipynb`; one `joint` parent at 20k steps, forks of 12k steps whose objective changes over the first 4k; matched fresh decoder)

**The term.** `dp = ||D(P(z_ctx, a)) − x_next||`, the predictor's output decoded and compared with the real next frame. It asks only for what can be
predicted: noise earns nothing, because no prediction can contain it.

| fresh decoder (real latent) | parent | `continue` | `continue + dp` | `fade` | `fade + dp` | `fade-r0 + dp` |
|---|---|---|---|---|---|---|
| reconstruction weight at the end | 1 | 1 | 1 | 0.1 | 0.1 | 0 |
| agent position / facing | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 0.99 | 0.97 / 0.64 | 1.00 / 0.84 | 1.00 / **0.57** |
| hand / inventory / terrain | 0.98 / 0.99 / 0.78 | 0.99 / 0.99 / 0.80 | 0.99 / 0.99 / 0.81 | 0.98 / 0.97 / 0.59 | 0.99 / 0.99 / 0.76 | 0.99 / 0.99 / 0.82 |
| mob | 0.55 | 0.61 | 0.64 | **0.03** | 0.44 | 0.55 |
| twinkle (noise) | 0.89 | 0.92 | 0.87 | 0.79 | 0.23 | **−0.02** |
| predicted facing (move) / hand / mob | 0.67 / 0.34 / 0.22 | 0.83 / 0.43 / 0.29 | **0.95 / 0.61 / 0.55** | 0.64 / 0.41 / −0.03 | 0.94 / 0.63 / 0.60 | 0.97 / 0.62 / 0.62 |

- **The decoded prediction is the largest gain in dynamics** (`continue + dp` against `continue`: facing 0.95 / 0.83, hand 0.61 / 0.43, mob 0.55 / 0.29).
- **Under a fade it stops the erosion and releases the noise.** The plain fade loses the mob (0.55 → 0.03) and terrain; with dp both are held, and with
  no reconstruction at all (`fade-r0 + dp`) the twinkle is fully released.
- **It loses one thing: the facing** (0.99 → 0.57). The agent's position is intact (1.00), so the loss is the arrow's orientation only.
- `dp only` (switch at once instead of over 4k steps) matches `fade-r0 + dp` on every readout; the rank never collapses (98 → 104).
- **Release is binary, not graded by the parent's skill.** The twinkle goes, all learned content stays, including the mob that the parent predicted at
  only 0.22: the predictor keeps learning during the fork (predicted mob 0.22 → 0.62), so what counts is the predictor at the end.

**Why the facing is lost: the action bypass.** The facing after a move is a function of the action, and the predictor receives the last two actions.
The pair (predictor, decoder) can therefore draw the facing without the encoder holding it. A shuffled-action test on the decoded prediction confirms it:

| predicted facing (own decoder) | true inputs (move / no-op) | last action replaced: follows fed / true | context latents shuffled (move / no-op) |
|---|---|---|---|
| parent | 0.59 / 0.93 | 0.59 / 0.25 | 0.36 / 0.24 |
| `fade-r0 + dp` | 0.98 / 0.88 | **0.97 / 0.24** | **0.94 / 0.67** |
| `gate + tgt` (§6) | 0.99 / 0.98 | 0.99 / 0.24 | 0.82 / 0.28 |

In every model the predicted facing follows whatever action is fed. In `fade-r0 + dp` it survives shuffled context latents, even on no-op steps (0.67,
from the earlier action), so the latent never has to hold it, and once the reconstruction is gone it does not.

## 6. A hold on the real latent closes the gap

(`08_joint_dp.ipynb`, same parent; forks from `fade-r0 + dp`)

Two ways to make the real latent carry what the predictor foresees:

- **Target gradient (`tgt`)**: `||sg(P(z_ctx, a)) − E(x_next)||` pulls the real next latent toward the prediction (LeWM-style, ramped in after the warmup).
- **Prediction-gated reconstruction (`gate`)**: the real next frame is reconstructed again, each tile weighted by `exp(−e / τ)`, with `e` the tile's
  per-pixel error of the decoded prediction. Tiles the predictor foresees are held; tiles it cannot foresee (noise) are not. τ is the detail level in
  pixel error.

| fresh decoder (real latent) | `fade-r0 + dp` | `+ tgt` | `gate` (τ 0.02) | `gate + tgt` (τ 0.02) | `gate + tgt` (τ 0.005) |
|---|---|---|---|---|---|
| agent position / facing | 1.00 / 0.57 | 1.00 / 0.95 | 1.00 / **0.99** | 1.00 / **1.00** | 1.00 / **0.99** |
| hand / inventory / terrain | 0.99 / 0.99 / 0.82 | 0.99 / 0.99 / 0.80 | 0.99 / 0.99 / 0.82 | 0.98 / 0.99 / 0.82 | 0.99 / 0.99 / 0.82 |
| mob (fresh / own decoder) | 0.55 / 0.71 | 0.29 / 0.65 | 0.27 / 0.37 | 0.30 / 0.53 | 0.33 / 0.60 |
| twinkle (noise) | −0.02 | −0.02 | 0.20 | **−0.01** | **−0.01** |
| predicted facing (move) / hand / mob | 0.97 / 0.62 / 0.62 | 0.99 / 0.53 / 0.56 | 0.93 / 0.61 / 0.48 | 0.99 / 0.52 / 0.42 | 0.99 / 0.52 / 0.49 |
| latent rank | 104 | 101 | 104 | 101 | 101 |

- **The gate restores the facing on its own** (0.57 → 0.99). It releases the noise only partly (0.20): the soft weight leaves the twinkle tiles about 15 %
  of the pressure.
- **`gate + tgt` holds all learned content and releases the noise completely**, with no rank collapse and dynamics close to the best. All pre-registered
  drops of learned content (agent, inventory, facing) are 0.00 at τ 0.005.
- **The mechanism changes as intended.** With the gate or the target gradient, shuffling the context latents drops the no-op facing to chance (0.24–0.29):
  the predictor now reads the facing from the latent because the latent holds it.
- **The cost is a little dynamics** (predicted hand 0.52 against 0.62 without the hold) and an **unresolved mob** (below).

## 7. The mob: the open item

- **The fresh decoder reads the mob lower in every fork with a hold** (0.27–0.33 against 0.55), the own decoder does not (0.37–0.60 against 0.62).
- **The mob is localised poorly by every model, the parent included.** A position readout (which tile shows the mob) gives 0.45 (own decoder) and 0.20
  (fresh decoder) in the parent, 0.00 for the fresh decoder in the forks. The pixel scores above therefore overstate what is held, and the fresh-decoder
  differences between forks are not trustworthy.
- **What the forks do change:** the predictor learns the chase (predicted position after a step toward the agent: 0.00 in the parent, 0.31 with dp),
  and the real-latent position after a random step falls (own decoder 0.47 → 0.20–0.22) while it stays after predictable steps. That fits a model
  releasing the unpredictable part of the mob. Whether this is acceptable depends on the use: a random step is unpredictable when it happens but becomes
  state for the next prediction.

## 8. Instruments and pitfalls

- **Probes under-read compressed latents.** The facing was read at ~0.1 by every probe while a decoder read 0.95. Content claims rest on decoders.
- **The fresh decoder must use the model's own class weights** (`decoder_strength(matched=True)`). With the default cap of 30 it under-reads mob and terrain;
  the earlier "the mob is never learned" was this artifact.
- **The own decoder can stop reading the real latent** when it is trained only on predictions (reconstruction weight 0). Use it for the predicted
  frame, the fresh decoder for the real latent.
- **Pixel scores mix placement and sharpness.** Position readouts (agent, mob) separate them.
- **Variance shares** of the noise were 0.00 in every run, the parent included, so they carry no information here.
- **Latent prediction error** is dominated by slow content in the joint model (copy-last error / variance 0.01) and is no progress measure.

## 9. What we now believe

| earlier | now |
|---|---|
| JEPAs lose content because prediction prunes it | first, content never becomes available (pixel mass, location); pruning comes second |
| a shadow encoder can protect what the live encoder needs | not with a release rule based on current predictability (§3) |
| the facing is absent from reconstruction latents | it is there; probes missed it |
| fading the reconstruction releases noise | it erodes everything; the decoded prediction makes the fade selective (§5) |
| the decoded prediction holds what the predictor learned | it holds the output; action-determined content needs a hold on the real latent (§6) |
| a target gradient collapses the rank | that was the gated open path; ungated, after the warmup and with the decoded prediction, the rank holds (101) |
| the mob is lost by the fade | the plain fade loses it; dp keeps its pixel score, but no model localises it well (§7) |

**The recipe this points to** (single seed): a joint encoder warmed up with class-balanced reconstruction, latent prediction and SIGReg
(availability); then the content-agnostic reconstruction is switched off and replaced by the decoded prediction, a prediction-gated reconstruction of the
real frame and a target gradient (judgement and consolidation). It holds what the predictor learned, releases what it could not and stays predictable.

## 10. Open questions, in order

1. **The mob.** A reliable instrument (a position readout with a decoder that weights the mob), then whether availability of a small moving entity is
   the limit. This is the Craftax failure that motivated the work.
2. **Seeds** for `gate + tgt` and the key contrasts (`fade`, `fade-r0 + dp`).
3. **Capacity.** At D = 128 the latent has spare room (rank ~100 of 128), so releasing noise has no measured benefit. A small-D comparison would show
   whether release matters.
4. **Label-free weights** in the joint model; the current weights use ground-truth tile classes.
5. **How long availability must last.** Forks switched at 20k; earlier switches or `dp only` from scratch are untested.
6. **Transfer to Craftax.**

## Where things are

| what | where |
|---|---|
| world | `gaf/env_grid.py` (`agent_shape`, `mob_mode`, `mob_chase`, `hand_tiles`, `variants()`) |
| objectives | `gaf/train.py` (arms, `rec_class_power`, `rec_class_cap`, `dp_weight`), `gaf/fork.py` (`dp`, `tgt`, `gate`, `r_floor`) |
| readouts | `gaf/readout.py`: `dynamics`, `decoder_strength(matched=True)`, `action_test`, `variance_shares` |
| notebooks | `notebooks/01`–`08`, numbered in order; the joint model and its forks are `07_joint`, `08_joint_dp` |
| results | `results/<notebook>/`; `results/joint_dp/` holds every fork's checkpoint, `readouts.json`, and the saved tables (`readouts_all.csv`, `gate_readouts.csv`, `action_test.csv`, `mob_position.csv`) |
| earlier method | `docs/reference_gated_retention.md` (B) |
