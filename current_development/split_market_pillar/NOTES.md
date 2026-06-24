# Split-market pillar — level vs supremacy

Research stream for the time-decay **outfield** engines. Lives in its own folder because
`ab_test_dixon_coles/` has been stretched beyond its original purpose.

## Why

The current market pillar (every `*Market*` engine) anchors the two log goal-rates
**independently** with one shared, sampled `σ_market` and one scalar `market_weight`:

```julia
ll_market_h = logpdf.(Normal.(log_λ_h .+ log κ_h, σ_market), market_log_λ_h)
ll_market_a = logpdf.(Normal.(log_λ_a .+ log κ_a, σ_market), market_log_λ_a)
@addlogprob! sum((ll_market_h + ll_market_a) .* w .* mask) * market_weight
```

Anchoring `h` and `a` independently with one σ is — exactly — an **isotropic** Gaussian
penalty on the rotated pair (level, supremacy). So one global knob forces the same trust on
"who wins" and "how many goals". That's the problem: when the model's favourite is wrong it
corrupts correct-score / BTTS / derived 1X2, while on totals the documented edge is *fading
the market's over-dispersion* (see `totals-compression-is-denoising`). We want to trust the
market on supremacy and keep our own view on level.

## What

Rotate the pillar into the level/supremacy basis and make it **anisotropic** with two FIXED
per-axis standard deviations (these replace sampled `market_σ` + `market_weight`, and become
the grid axes):

```
model_sup   = (log_λ_h + log κ_h) − (log_λ_a + log κ_a)      m_sup   = market_log_λ_h − market_log_λ_a
model_level = (log_λ_h + log κ_h) + (log_λ_a + log κ_a)      m_level = market_log_λ_h + market_log_λ_a
ll = N(model_sup; m_sup, σ_supremacy)  +  level_active · N(model_level; m_level, σ_level)
```

- `σ_supremacy` tight → hard "who wins" anchor.
- `σ_level` loose; `σ_level = Inf` → level pillar OFF, model fully owns totals.
- Branch-free in the `@model`: `build_turing_model` converts `σ_level = Inf` into
  `level_active = 0.0` + a dummy finite σ, so the model body stays pure broadcast (AD-safe).
- No new feature extraction — `m_level`/`m_sup` are derived from the already-inverted
  `flat_market_λ_home/away/ρ`.
- Dixon-Coles ρ is anchored at `σ_supremacy` (ρ sharpens correct-score, same goal as supremacy).

## Files

- `l01_split_market.jl` — loader (temporary module). Four engine rungs + `build_turing_model`,
  `required_features`, `extract_parameters`, and prediction overrides.
- `r01_smoke_and_grid.jl` — smoke test then σ-grid backtest + evaluation.

## The ladder (rungs)

| Rung | Struct | Goals likelihood | xG | Notes |
|---|---|---|---|---|
| R1 | `SplitMarketPoissonGoalsModel` | Poisson | no | isolates the split from xG noise |
| R2 | `SplitMarketPoissonXGModel` | Poisson | yes | |
| R3 | `SplitMarketNegBinXGModel` | RobustNegativeBinomial | yes | over-dispersed |
| R4 | `SplitMarketDixonColesXGModel` | DC Poisson | yes | adds ρ anchored at σ_supremacy (= r06 model) |

## Integration gotchas (carried from memory)

- **Prediction dispatch** (`dixoncoles-prediction-dispatch-union`): all player models are
  `<: AbstractNegBinModel`; the Poisson/DC scorers are more-specific Union overrides. Our new
  structs aren't in those Unions, so R1/R2 (Poisson) and R4 (DC) ship explicit
  `extract_params`+`compute_score_matrix` overrides in the loader. R3 (NegBin) routes on the
  `AbstractNegBinModel` default via its `r_h`/`r_a` columns — no override.
- **Betfair training pillar** (`outfield-xg-engine-gotchas`): swap
  `summarize_betfair_market` into `ds.odds` **before** `create_experiment_task` so the
  supremacy anchor is built from Betfair, not SofaScore 1X2. The runner does this.
- **xG=0 → Gamma −Inf**: present xG floored to 1e-3, missing masked (R2–R4). R1 has no xG.
- Data: `Data.Ireland()` (full betfair + xG).

## Grid (per rung)

`days_half_life = 60` fixed. `σ_supremacy ∈ {0.05, 0.1, 0.2}` × `σ_level ∈ {0.5, 1.0, Inf}`.
Judge on **hurdle_G / CLV vs Betfair** per market family (not LogLoss alone). Expect:
supremacy markets (home/draw/away, BTTS) sharpen as σ_sup tightens; totals (over/under)
best where σ_level is loose/Inf if the totals edge is real.

## Findings log

- _(empty — fill in after the smoke test + first grid)_
