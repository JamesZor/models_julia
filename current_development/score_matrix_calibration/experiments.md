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

**Status:** Ready to Run. 

**Steps to run on server:**
```julia
using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
include("current_development/score_matrix_calibration/r01_eda_calibration.jl")
```

**Results (Placeholder):**
* Global BTTS Gamma: `[To be filled]`
* Walk-Forward BTTS Median Gamma: `[To be filled]`
* Tilted BTTS $t$-statistic: `[To be filled]`

---

## Experiment 2: Kelly Backtest Impact (To be built)
**Goal:** Pass the walk-forward calibrated score matrix into the structural Kelly solver to determine if the bias removal prevents catastrophic losses on the full book.

**Status:** Pending results of Experiment 1.
