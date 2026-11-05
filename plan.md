# Plan: SFN-World v0.4 on a RunPod A100

Source: `SFN-World experiment draft v0.4 · not yet learned vs. unlearnable.pdf` (section numbers below refer to it).

## Decisions (from you)
- **Layout:** small package `sfn/` plus one thin notebook `sfn_world.ipynb`.
- **Execution:** sequential and resumable. One CSV and one JSON per run; a run is skipped if its JSON exists.
- **Scope:** Stages 0, 0b, 1, 1-B, 1b, 1c (conditional), 2, 2-B (conditional). Pixel extension (§9) is out of scope.

## Layout
```
pyproject.toml          deps: torch, numpy, pandas, matplotlib; ruff config
sfn/env.py              SFN-World v2: 8 groups, frozen mixing MLP, D0/D1/D2 difficulty
sfn/models.py           Encoder, Predictor (optionally heteroscedastic), M2 ensemble, SIGReg
sfn/gates.py            Stats (EMA covariances, U, V, shortcut score s) and the gates G (A0..M3, C1..C3)
sfn/probes.py           ridge probes and every metric of §7
sfn/train.py            Cfg, run() (encoder runs), run_gt() (Stage 0b, no encoder)
sfn/stages.py           stage grids, execute(), collect(), best-arm selection
sfn/plots.py            the four key figures
tests/test_smoke.py     env sanity, gate identities, tiny end-to-end run per arm (CPU)
sfn_world.ipynb         setup, GPU check, one cell per stage, plots
```

## Interpretations where the PDF is silent (shout if you disagree)
1. **Data:** i.i.d. transitions, not persistent chains. Each batch draws a fresh state from its stationary distribution (θ uniform) and steps once. This makes R²_lin and R²max exact, and avoids the doubling map collapsing in floating point.
2. **Observation scaling:** x is standardised per feature with statistics from 16k samples at env construction. Salience w is applied before the mixing MLP, so relative salience is unchanged.
3. **Encoder BatchNorm:** z_t and z_{t+1} go through the encoder in one concatenated pass (one BN batch). SIGReg is computed per time step (z_t, z_{t+1}) and averaged.
4. **SIGReg:** identical to the released LeWM `SIGReg` module (checked against its source, tested verbatim): 17 knots on [0, 3] with doubled interior weights, Gaussian window exp(-t²/2), statistic multiplied by the batch size N, mean over time steps and slices, fresh unit-norm Gaussian directions each step. Slices are 256 as in the PDF (LeWM's code uses 1024). The LeJEPA paper's own snippet uses [-5, 5], which differs from its released code.
5. **Stats:** EMA covariances of the full-predictor residual (→ U), weak-predictor output (→ V), z_{t+1}, and weak error. s_k is computed from these in the V basis at each refresh, so column identity changes cannot corrupt it. B-rand reuses the same s_k on a random basis.
6. **M1a smoothing:** EMA of LP with τ = 0.9 (unspecified in the PDF).
7. **M2:** K = 5 heads, each a trainable MLP (64 hidden) plus a frozen prior MLP, on (sg z_t, a) → sg z_{t+1}. They never send gradient to the encoder.
8. **β calibration (M2):** Stage 0 A0 runs also train the ensemble heads (`heads=True`; no effect on the arm). β is set so the gate is 0.9 on the latent direction best aligned with group 8, averaged over those runs.
9. **τ_w, T:** keep the PDF defaults 0.8 / 0.05 unless the Stage 0b check 5 fails.
10. **C3 oracle:** linear head from z_t to the true values of groups 1–7, weight 1, gradient into the encoder.
11. **Metric definitions that were loose:**
    - Effective dims = participation ratio of the covariance of z explained by a group's degree-2 features (latent dims carrying that group).
    - Flag precision = s-weighted R² of the flagged directions on groups 1, 3, 4, 7.
    - Flag recall = mean over groups 1, 3, 4, 7 of R² from the s-scaled z divided by R² from z.
    - Eigensubspace stability = mean cos² of principal angles of the top-4 subspaces between refreshes.
12. **"Best" arm** for Stages 1b and 2 = highest mean of final R²_5 and R²_6 among the candidates.
13. **Probes:** R²_k is ridge from z_t, fit on 8k fresh samples and scored on 8k other fresh samples, with the encoder in eval mode.

## Milestones
- [x] 1. `pyproject.toml`, env, and an env sanity test (ρ, R²max, R²_lin vs the §3 table).
- [x] 2. models: encoder, predictor, ensemble, SIGReg.
- [x] 3. gates and stats (all arms, all B levels).
- [x] 4. probes and metrics.
- [x] 5. `run_gt` (Stage 0b), then check its output against §8 checks 1–6.
- [x] 6. `run` (training loop), then a tiny CPU run of every arm and B level.
- [x] 7. stages, collect, plots.
- [x] 8. notebook, built from the stage helpers.
- [x] 9. ruff clean, tests green, final review.

## Findings during the build
- **Stage 0b on CPU (3 seeds not run; seed 0 only):** weak-predictor R²_lin matches the §3 table (0.96, 0.49, 0.90, 1.0, 0.41, 0.21, 0.93 at D0). §8 check 1 **fails at D0**: groups 4 and 5 are both learned within the first 250 steps. At D1, t90(5) is about 750 vs t90(4) of 250 or less, so it passes (the notebook logs every 50 steps for a finer ratio). `default_difficulty()` picks D1 automatically, per check 6.
- **PDF inconsistencies handled:** Stage 1c has no B-rand arm, so its B-rand check reuses the D = 16 result. Stage 1b has no M1d (B0) or B-rand at D = 48, so budget figure 4 shows blank bars there unless the best M arm is M1d.
- **Speed:** about 11 ms/step on this Mac's CPU, so a 20k-step run is about 4 min here. The A100 is likely launch-bound, so the 1–2 min per run estimate is not guaranteed. The first Stage 0 cell prints seconds per run and the projected total hours.

## Verification (what I can and cannot do from this Mac)
- CPU: unit tests, env checks, Stage 0b, short smoke runs of every arm. Everything must run end to end.
- Not possible here: real A100 timing or the scientific outcomes. The PDF's "1–2 GPU-minutes per run" is unverified; the notebook prints seconds per run in the first Stage 0 cell so you can extrapolate.
