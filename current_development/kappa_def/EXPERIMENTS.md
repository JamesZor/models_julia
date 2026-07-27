# kappa_def — experiment tracker

> Update this table + the log below after every run. Status: `planned` → `running` →
> `done` / `failed`. Keep verdicts one line; long analysis goes in `NOTES.md` findings log.

| id | file | what | league | status | verdict |
|---|---|---|---|---|---|
| r00 | `r00_persistence_gate.jl` | EDA gate: per-team (goals−xG) attack + (conceded−xG_conceded) defense residual persistence (split-half + season t→t+1) | Ireland + 718 | **planned — run next** | — |
| r01 | `r01_single_split_shakedown.jl` | single split, 3 κ modes in parallel: convergence + per-team κ posterior inspection | Ireland (79) | **done 2026-07-02** | V1/V2 converge superbly (R-hat ≤1.005, ESS ~5k); τ posteriors ≈ prior (τ_def 0.063 [0.006,0.154], P(τ>.05) 0.53 vs prior 0.62) ⇒ one 79-split is UNINFORMATIVE (60-day decay ⇒ ~10–15 eff. matches/team) — question OPEN, effect size if real is min_edge-material; V0 softplus control hit the known base-model metastability (R-hat 1.53) — see NOTES log |
| r02 | `r01_...` with `SEGMENT = IrelandFirstDivision()` | same shakedown on the NB-regime league | 718 | planned | — |
| r03 | `r03_grid_backtest.jl` | full-CV grid: kd_none_src (src control) vs kd_net vs kd_attdef — GLMEdge/LogLoss/LPD vs Betfair + BayesianKelly tearsheet + **pooled-τ across splits** (the power read) | Ireland default, flip to 718 | **built — run after r00/r02 gates** | — |

## Decision rules (agreed up front, so we don't rationalize after)

- **r00 gate:** defensive residual persistence (split-half corr AND/OR season-over-season
  corr) > 0 with permutation p < 0.10 ⇒ κ_def structure is real, proceed. Both ≈ 0 ⇒ the
  attack-only V0 is already right; park the stream (record and stop).
- **r01:** any mode with max R-hat > 1.05 on its κ params (checked on RAW chains, not the
  curated conv.df) is out. τ posteriors pulled to ~0 with team multipliers ≈ 1.00 ⇒ that
  mode learned nothing (σ-hierarchy-null pattern) — note it, don't eval it.
- **r03:** judge vs market (per-line LPD/GLMEdge), not vs goals; the market may already
  price team defense ([[position-aware-ratings-rejected]] precedent).

## Run log

- 2026-07-02 · r01 · Ireland 79, single split 2026, 1000/500×4 depth 10, 3 modes in
  parallel · V2_net max R-hat 1.0045 / V1_attdef 1.0023 / V0 1.53 (stuck chain, whole model
  blown: ha σ_γ mean 4.35, cluster of R-hat ≈1.52 across unrelated params) · spreads:
  att 0.036–0.037, def 0.037, attdef_cor 0.10, κ0_conv ≈ 0.974 both modes → converged-but-
  learned-nothing on 79; decision: control = src engine, next = r00 gate + r02 on 718.
