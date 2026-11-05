# SFN-World: not yet learned vs. unlearnable

A fast synthetic test of how to stop a SIGReg world model (LeWM-style JEPA) from throwing away
features that are **hard to learn but perfectly predictable**, while still discarding real noise.
Spec: `SFN-World experiment draft v0.4 ... .pdf`. Build notes and interpretations: [plan.md](plan.md).

## The problem in one minute

A SIGReg world model has an encoder `z = enc(x)`, a predictor `P(z_t, a_t) ≈ z_{t+1}`, and the loss

    L = ‖P(z_t, a_t) − z_{t+1}‖² + λ · SIGReg(z)

SIGReg pushes `z` toward an isotropic Gaussian. That fixes *how much* variance the latent has, so it
cannot collapse, but not *which* information fills it.

The failure we want to understand:

1. The encoder also gets gradient through the **target** `z_{t+1}`: it is rewarded for making the
   target easier to hit. That route destroys information.
2. A fast, action-dependent feature (say a rotation angle) has a large prediction error early on,
   because the predictor has not learned it yet. The cheapest way to lower the loss is to **delete it
   from `z`**, before the predictor ever gets the chance. A slow or easy feature never has this problem.
3. SIGReg then refills the freed variance with something else, typically redundant nonlinear copies of the
   slow features.

So there is a race between *the predictor learning a feature* and *the encoder deleting it*.

The question: can the loss tell **epistemic** error (learnable, not learned yet: keep) from
**aleatoric** error (irreducible: may drop), and does any intervention keep the first while still
discarding the second?

> **Premise.** The failure only exists if slow features can fill SIGReg's variance budget. If they
> cannot, SIGReg forces the fast features into the latent and the baseline looks healthy, which would be a false
> negative. The notebook checks this after Stage 0 (`premise_check`: slow groups 1-3 must carry more
> variance than fast-hard groups 5-7).

## The test bed (SFN-World v2)

Eight groups of 2 latent dims, mixed into R⁶⁴ by a frozen random MLP. Each group isolates one property:

| # | Group | Timescale | Predictable (R²max) | Learn time | Tests |
|---|-------|-----------|---------------------|------------|-------|
| 1 | slow OU | slow | 0.96 | easy | classic slow-feature sanity |
| 2 | medium OU | medium | 0.49 | easy | same |
| 3 | slow + random jumps | slow | 0.90 | easy | are rare surprises thrown out with the noise? |
| 4 | rotation, no action | fast | 1.0 | easy | control for 5 |
| 5 | rotation driven by action | fast | 1.0 | **hard** | **the key group**: same as 4 but slower to learn |
| 6 | as 5 plus angle noise | fast | 0.5 | hard | half aleatoric: is a gate calibrated? |
| 7 | slower rotation | medium | 1.0 | medium-hard | timescale vs. 5 |
| 8 | i.i.d. noise | none | 0 | never | the only thing that *should* be dropped |

The ideal outcome is to keep groups 1-7 and drop only group 8. **Group 4 vs. group 5** is the most
important comparison: same timescale and predictability, different learning time.

Knobs: latent size `D` (16 tight, 24 middle, 48 roomy), difficulty `D0/D1/D2` (makes groups 5-7 harder
to learn), and fast-feature salience `w_fast`.

## Two knobs: the target path and the SIGReg budget

Every arm is a pair `(G, B)`. The context path (`z_t` made informative) is identical in all arms.

    x_t ──enc──► z_t ─────────────┐
                                  ├─► predictor ─► ẑ_{t+1} ─┐
    x_{t+1} ─enc─► z_{t+1} ─[G]──► target ───────────────────┴─► prediction loss
                   z_t, z_{t+1} ──[SIGReg, slices weighted by B]──► regulariser

`G` is a D×D matrix on the target path's *gradient* (forward value unchanged):
`target = sg(z') + (z' − sg(z')) @ G`. `G = I` is LeWM, `G = 0` is stop-grad.

### Knob G: target-path arms

| Arm | Idea |
|-----|------|
| **A0** | LeWM baseline, `G = I`. Predicted to drop group 5 and keep 4. |
| **A1** | Stop-grad, `G = αI`. Removes the deleting route, and the compression pressure with it. |
| **A2** | EMA target encoder (`G = 0` plus a lagged encoder). Same idea, BYOL-style. |
| **M1a-d** | **Learning-progress gates.** Compare the predictor with a slow EMA copy of itself on the *same* features. If it is improving on a direction, the error is epistemic: protect it (`g → 0`). If it is stuck, let the encoder compress (`g → 1`). They differ only in the *basis* used: |
| M1a | per latent dimension (basis-dependent, since SIGReg makes the latent rotation-invariant) |
| M1b | fresh random projections each step (basis-free) |
| M1c | naive loss slope on a fixed bank (confound check: it can't tell the predictor learning from the encoder shrinking the direction) |
| M1d | eigenbasis of the EMA residual covariance (a data-defined, slowly refreshed basis) |
| **M2** | Ensemble disagreement: heads that disagree relative to their error mean epistemic uncertainty. |
| **M3** | Heteroscedastic head. Pre-registered to **fail**: the variance head absorbs epistemic error too. |
| **C1** | Control: predictor lr ×5. Is a faster predictor enough? |
| **C2** | Control: stop-grad for the first 2k steps only. Is a head start enough? |
| **C3** | Oracle: supervised aux loss on groups 1-7. Upper bound. |

If C1 or C2 matches the gated arms, the machinery is unnecessary. All M1 arms use C2's 2k-step
warm-up, because at a plateau every learning-progress signal reads zero.

### Knob B: SIGReg budget

Idea: when the budget is tight, directions that are *trivially* predictable (slow, linear, easy) hog it.
A deliberately weak **linear** predictor flags those directions with a shortcut score `s`; SIGReg is then
relaxed on them (per-slice weight `max(0.2, 1 − Σ s_k (w·v_k)²)`), freeing room for the hard features.
The flag is never used as a target gate.

| Level | What it does |
|-------|--------------|
| **B0** | LeWM SIGReg. All Stage 1 target arms run here. |
| **B1** | SIGReg relaxed on flagged directions. |
| **B2** | B1 plus a small shrink penalty on the flagged directions' variance. |
| **B-rand** | B1's relaxation on a *random* subspace. Control: if it matches B1, the gain is just weaker SIGReg, not targeting. |

Arms are written `M1d+B1`; no suffix means B0. Arms with the same B differ only in the target path,
and arms with the same G differ only in the budget, so each effect is attributable.

## What counts as success

- **Gate:** R²₅ and R²₆ each ≥ 0.3 above A0 and ≥ 0.1 above the best of A1/A2/C1/C2; R²₈ ≤ A0 + 0.1; no
  group in 1-4 loses more than 0.1; holds over 3 seeds and at D1.
- **Factor B:** M1d+B beats M1d by ≥ 0.1 on R²₅ or R²₆, M1d+B-rand does not, and the same side conditions hold.

`gate_verdict()` and `budget_verdict()` in `sfn/stages.py` compute these on seed means.

## Protocol

| Stage | What | Runs |
|-------|------|------|
| 0b | predictor on ground truth (no encoder): learn time `t90`, `R²_lin`. Needs `t90(5) ≥ 3·t90(4)` | 9 (seconds each) |
| 0 | reproduce the failure with A0 over λ × `w_fast`; calibrate M2's β; **premise check** | 12 |
| 1 | the 13 target arms at B0, D = 16 | 39 |
| 1-B | A0+B1, M1d+B1, M1d+B2, M1d+B-rand | 12 |
| 1b | capacity: D = 48 | 15 |
| 1c | D = 24, only if B showed no effect at D = 16 | 9 (conditional) |
| 2 | difficulty sweep D1, D2 | 30 |
| 2-B | best G+B at D1, D2, only if B passed | 6 (conditional) |

Findings so far: with a clean ground-truth state, group 5 is *not* slower to learn than group 4 at D0,
so the notebook defaults to D1 (the lowest difficulty that passes the check).

## Run it

On a RunPod A100 (PyTorch template): upload this folder, open `sfn_world.ipynb`, run top to bottom.
Results land in `/workspace/results` (one CSV and one JSON per run; finished runs are skipped, so
re-running a cell resumes).

Locally (CPU is enough for the tests):

    pip install -e ".[dev]"
    pytest -q tests          # ~80 s; env stats, SIGReg vs. the LeWM reference, every arm, every B level

## Layout

    sfn/env.py      SFN-World v2          sfn/gates.py   gates G, shortcut stats, slice weights
    sfn/models.py   encoder, predictor,   sfn/probes.py  ridge/MLP probes and all metrics
                    ensemble, SIGReg      sfn/train.py   run() and run_gt() (Stage 0b)
    sfn/stages.py   grids, collect, verdicts, premise check      sfn/plots.py  the four key figures

## Caveats

- Not yet run on a GPU. Timing is unverified (about 11 ms/step on CPU); the first Stage 0 cell prints
  seconds per run and the projected total.
- Where the PDF is silent I chose defaults. They are listed in [plan.md](plan.md).
- The PDF's literature rows for factor B were written from memory and not re-checked.
