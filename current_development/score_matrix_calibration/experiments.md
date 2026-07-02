# Score Matrix Calibration - Experiments Tracking

## Overview
This document tracks the experiments and progress for applying a Layer 2 Exponential Tilt (MaxEnt Shift) to the unified Kelly score matrix (the 144-element grid). 

By extracting the per-market bias from a base model (e.g., `li_smile50`) using a GLM logit fit, we can dynamically suppress or boost specific scorelines (like 1-1, 2-1 for BTTS) to fix systemic biases before submitting the joint probabilities to the Kelly solver. 

## Files
- `l01_score_matrix_calibration.jl`: Loader containing the core math for the exponential tilt (`tilt_score_matrix!`) and fitting functions (`fit_global_bias`, `fit_walk_forward_bias`).
- `r01_eda_calibration.jl`: Runner to extract bias offsets (global vs. walk-forward) and compare the $t$-statistic before and after calibration on the score matrix.

---

## Experiment 1: Bias Extraction & Validation (r01)
**Goal:** Prove the exponential tilt effectively strips the +2.4pp skew from `btts_yes` in `li_smile50` using both a static global offset and an exponentially-decaying walk-forward offset.

**Status:** DONE (li_smile50, Ireland, betfair anchor). Ran via kaimon on server (`ab7808f`).

**Steps to run on server:**
```julia
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"   # LanguageServer/JSONRPC dep is broken; skip env-wide precompile
using BayesianFootball
include("current_development/score_matrix_calibration/r01_eda_calibration.jl")
```

### Key design correction (vs the Gemini draft)
The original runner fit γ against **actual outcomes** (`is_winner`) but validated bias against
the **market** (`prob_fair_close`) — two different targets, so the tilt appeared to make bias
*worse*. `l01` now fits γ (robust bisection MLE, `_fit_shift`) toward **either** target; `r01`
reports both. Both tilts now center on their own target (validation below).

### Result 1 — where the model sits, per line (n in table; Ireland li_smile50)
`mkt_act (t_ka)` = market − actual = **is the market itself miscalibrated vs reality?**

| family | model−mkt | **market−actual (t)** | verdict |
|---|---|---|---|
| **1X2** (home/draw/away) | small | ≈0 (|t|≤1.3) | market ≈ reality → γ_mkt≈γ_real, target doesn't matter |
| **Totals** O/U 1.5–3.5 | model compresses (t up to −6.7 on over_15) | **≈0 (|t|≤0.4)** | market is the honest anchor → calibrate totals to MARKET |
| **BTTS** | +0.024 (t 5.29) | **−0.093 (t −2.68)** | market & reality **DIVERGE**: γ_mkt=−0.097 vs γ_real=+0.286 |

**BTTS is the only line where "which target" is a real bet.** model=0.52, market=0.496,
actual=0.589 — both model and market under-price BTTS vs realized, market more so. Caveat: n=202,
single league, in-sample realized frequency; t=2.68 is suggestive, not bankable.

### Result 2 — tilt validation (btts_yes, n=202): each tilt centers on its own target
| variant | bias vs market (t) | bias vs reality (t) |
|---|---|---|
| raw | +0.024 (5.29) | −0.070 (−2.03) |
| tilt→market (global) | **+0.001 (0.11)** | −0.093 (−2.72) |
| tilt→reality (global) | +0.092 (19.0) | **−0.002 (−0.05)** |
| tilt→market (WF) | +0.005 (1.19) | −0.088 (−2.57) |
| tilt→reality (WF) | +0.074 (12.8) | −0.020 (−0.58) |

Exponential tilt works exactly as intended. WF under-centers (early splits have γ=0 until
`min_history=20` past matches accumulate — expected dilution).

### Open decision (for r02)
Direction of the production calibrator. Philosophy (`calibrate-centre-edge-in-tails`) →
**target = market** everywhere: on 1X2/totals it's a safe re-centering (market≈reality), on BTTS
it deliberately declines to bet the fragile market-vs-reality mean gap, leaving BTTS edge to the
per-match deviations (r13 showed btts glm_coef +7.2, p=0.01 — deviation edge survives separately).

---

## Experiment 2: Kelly Backtest Impact (To be built)
**Goal:** Pass the (market-target) walk-forward calibrated score matrix into the structural Kelly
solver (`unified_staking/l01_structural_kelly.jl`, implementing (P)/(U-MC) of
`docs/bets_multi/unified_kelly_postgrad_notes.md`) to check bias removal prevents catastrophic
losses on the full book. Per notes §8.4: shrinkage fixes *estimation error, not bias* — so the
per-line skew MUST be stripped here, upstream of the allocator.

**Status:** Pending direction decision from Experiment 1.
