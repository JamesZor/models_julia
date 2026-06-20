# Causal Momentum as an In-Play Covariate — Findings

*Ireland. Causal SofaScore-momentum AUC added as a global covariate to the Turing in-play intensity
model (`l03` `use_momentum`), fit with vs without on the same 75/25 match split. Feature = decay-weighted
net momentum (own − opp) using ONLY `points[1:t_m]` (no future leak). Code: `l06_momentum_feature.jl`,
`r06_momentum_compare.jl`.*

## Is momentum a good feature? Short answer: yes, modestly — for the goal-count model; a wash for the OU market.

### Coverage & leakage
- **All 253 modelling matches have momentum** (548 Ireland matches in the SofaScore `match_graph` DB),
  so no sample loss.
- Feature is strictly **causal** (only minutes up to `t_m`); the old `l01_momentum.jl` whole-match AUC
  leaks the future and must not be used in-play.

### It carries signal beyond score
- Standardised causal net momentum is only **weakly correlated with current score** (r≈0.085) — it is
  *not* just re-encoding game state — and **modestly predicts realized remaining goals** (r≈0.156).
- In the model (controlling for pregame λ, score, time, home): **β_mom = +0.14, 90% CI [0.09, 0.18]**,
  credibly non-zero. A one-SD increase in accumulated momentum ⇒ **×1.15** on a side's remaining-goal rate.
- **Held-out count elpd improves: −1.0753 → −1.0689** — a bigger lift than any game-state variant gave.

### But it barely moves the Over/Under market calibration
| | ECE | Brier | LogLoss |
|---|---|---|---|
| without momentum | 0.0622 | 0.1672 | 0.494 |
| with momentum | 0.0594 | 0.1681 | 0.4962 |

ECE improves slightly; Brier/LogLoss are flat-to-slightly-worse → **roughly a wash for OU**. The OU
probabilities are dominated by the structural under-prediction of totals (stoppage-time exposure +
independence; see `game_state_calibration_report.md`), which momentum does not address.

## Verdict
- Momentum is a **genuine, credibly non-zero in-play predictor** of remaining goals — a good feature for
  the **goal-count / intensity** model and worth keeping (`use_momentum = true`).
- It does **not** materially improve **Over/Under market calibration** on this sample, because the OU gap
  is structural. Fix the exposure/independence first; momentum is complementary, not a substitute.
- The effect is small (×1.15/SD) and partly overlaps what pregame λ already encodes (momentum ≈ xG).

## Reproduce
`r06_momentum_compare.jl`: `build_momentum_lookup(tournament_ids(Ireland()))` →
`build_intensity_inputs(panel, ds; mom_lookup=…)` → fit `InPlayIntensityConfig(use_momentum=false/true)`
→ compare β_mom, `held_out_elpd`, and OU `build_ou_eval` (ECE/Brier/LogLoss).
