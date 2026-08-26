# Model 00 — Findings (Pure Poisson Baseline)

Append-only. Every gate run gets a dated entry with the config hash. A result that
is not written here does not exist.

---

## 2026-08-26 — Model 00 scaffolded, ready for verification

Status: **Scaffolded and ready for Gates 0–5 walkthrough.**

| File | State | Description |
|---|---|---|
| `MODEL.md` | complete | Pure Poisson, log-intensity formulation |
| `l01_model.jl` | complete | `DynamicPoissonGoalsTimeDecayModel` engine & extractors |
| `l02_equations.jl` | complete | Independent log-Poisson referee equations |
| `l03_gates.jl` | complete | Gates 0–2 (Contract, Config, Features, anti-leakage) |
| `l04_sampling_gates.jl` | complete | Gate 3 (3a Parity, 3b Gradients, 3c Smoke MCMC) |
| `l05_extraction_gates.jl` | complete | Gate 4 (4a Synthetic parity, 4b Real plumbing, 4c Fallbacks) |
| `l06_score_matrix_gates.jl` | complete | Gate 5 (5a Poisson dispatch, 5b Grid parity, 5c Identities) |
| `v01_walkthrough.jl` | complete | Interactive REPL blocks for Gates 0–5 |
| `r01_train.jl` | complete | Full 20-fold grid runner |
| `r02_evaluate.jl` | complete | Gate 6 proper-scoring evaluation vs market |
| `r03_growth_clv.jl` | complete | Gate 7 Betfair CLV and growth backtest |

---
