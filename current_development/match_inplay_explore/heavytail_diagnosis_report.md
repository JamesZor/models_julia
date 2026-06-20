# Heavy-Tailed Total (Negative Binomial) — and the OU "Under-Prediction" Re-Diagnosed

*Prompted by the in-play derivatives-pricing thesis (Constant/Local Intensity models): the score
distribution is heavier-tailed than Poisson (the "intensity smile"), which we suspected explained our
Over/Under under-prediction. We implemented the fix and, in doing so, discovered the premise was wrong.*

## TL;DR
The apparent **~5-pt Over under-prediction** reported in `game_state_calibration_report.md` is **NOT a
structural model flaw** — it is **test-split sampling noise**. The model is **mean-unbiased**. A
Negative-Binomial heavy tail therefore does **not** help. Earlier reports blamed a "stoppage-time
exposure undercount"; **that explanation was incorrect** and is corrected here.

## What we tested
The OU markets depend only on the **total** remaining goals. We modelled the total as
`total_rem ~ NegBin(mean = μ_tot, size = r)` (var = μ + μ²/r → heavier tail than Poisson as r shrinks),
fitting one global dispersion `r` by MLE on the model's per-bin mean `μ_tot` (from the l02 intensity GLM).

## Findings

### 1. The conditional overdispersion is mild
Raw `var/mean` of remaining totals ≈ **1.35**, but **conditional on the model mean** it is only
**≈ 1.09** (fitted `r̂ ≈ 15`). The bulk of the raw spread is already explained by `μ_tot` varying across
bins (time, score, team quality). So there is little heavy tail left to add.

### 2. NegBin barely moves OU calibration
On the same single split as r05 (lines 1.5/2.5/3.5):

| total model | mean P(over) | ECE | Brier |
|---|---|---|---|
| Poisson | 0.442 | 0.0622 | 0.1672 |
| Negative Binomial | 0.439 | 0.0619 | 0.1672 |

Negligible — and it moves the *wrong way* on the mean (a heavier tail near these lines doesn't lift the
predicted over-rate toward the actual 0.495).

### 3. The real story: it was a mean issue, and the mean is unbiased
- The model's mean remaining total vs reality: **TRAIN 1.328 vs 1.330 (unbiased)**; the r05 **TEST split
  1.26 vs 1.42** (looks under-predicted).
- **Jensen is not the cause:** full posterior-predictive mean (1.263) ≈ point-estimate mean (1.261).
- **Decisive multi-seed check** (refit the GLM on 15 random 75/25 splits, held-out mean bias = actual −
  predicted): values **[−0.195 … +0.143], mean +0.030, std 0.087, 11/15 positive** → **consistent with
  zero**. The r05 (seed-1) split simply drew a goal-heavy test set.

## Corrected conclusion
- There is **no structural OU under-prediction** to fix. The model is approximately mean-unbiased and
  reasonably calibrated; the residual ECE (~0.06) is dominated by the small sample (63 test matches).
- The **Negative-Binomial / heavy-tail** upgrade is theoretically sound (mild overdispersion is real) but
  **not worth adding** for this data — the effect is within noise.
- **This supersedes** the "stoppage-time exposure undercount + independence" explanation in
  `game_state_calibration_report.md` (that claim was based on the single goal-heavy split).

## The real methodological lesson
With ~253 matches (63 held out), a single split's calibration metrics carry **±0.09 goals / several-pt
sampling error** on the mean. Differences of ~5 pts or ~0.06 ECE between models are **inside the noise**.
**Use k-fold / repeated cross-validation (or more leagues/seasons)** before trusting any calibration
gap — that, not heavier tails, is the highest-leverage next step for the OU work.

## Reproduce
`r07_heavytail_diagnosis.jl`: fits the NegBin dispersion, compares Poisson vs NegBin OU calibration on
the r05 split, and runs the 15-seed held-out mean-bias check.
