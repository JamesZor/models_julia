# staking_layer

Modular staking system: **per-line trust blend → coherent IPF grid tilt → capped unified Kelly**,
with the trust weight produced by a **swappable model** (`AbstractTrustModel`). A dev module
(mirrors `bayesian_layer_2/`): `src/` loaders + top-level `rXX` runners, graduates to `src/` once
validated. Supersedes `staking_sim/` + `staking_real/` + `unified_staking/` (kept until the real
races are re-run on the server — see status below).

## The one seam

```
model targets p ─┐
market probs   q ─┤→ blend_targets → coherent_multiplier (IPF tilt) → solve_P(p,R;cap) → stakes
trust weights  w ─┘
```

Everything downstream of `w` is agnostic to how `w` was produced. Swapping the trust estimator
(EB → distributional → Bayesian → hierarchical) never touches the policy or the runner.

## src/ layout

| file | contents |
|---|---|
| `l01_book_schema.jl` | `StakingMatch`, `SEL_/UNIT_/FAM_` consts, grid geometry, `settle_score` |
| `l02_kelly.jl` | `solve_P` / `proj_cap!` / `G_growth` — the pure (P) solver |
| `l03_coherent_pricing.jl` | `blend_targets`, `coherent_multiplier` (IPF tilt), `normalize_mult` |
| `l04_trust_interface.jl` | `AbstractTrustModel`, `TrustHist` (+team ids), `FlatTrust`/`CuratedTrust` |
| `l05_trust_eb.jl` | `EBTrust` — empirical-Bayes point + grid-posterior draws |
| `l06_trust_bayes.jl` | `BayesianTrust` (Turing) + `HierarchicalTrust` stub |
| `l07_policy.jl` | `FlatPolicy`, `PerBetKellyPolicy`, `UnifiedPolicy` + `stake_for` |
| `l08_sim_source.jl` | `SimSource` — the simulated double-Poisson league |
| `l09_real_source.jl` | `RealSource` (real L1 + Betfair) + the extended multi-market book |
| `l10_runner.jl` | `run_race` / `run_ext_race` + metrics + reporting |

## The system API — adding a trust model

Implement three methods (mirrors `src/Calibration` `fit_calibrator`/`apply_calibration`):

```julia
struct MyTrust <: AbstractTrustModel; ... end
fit_trust(m::MyTrust, h::TrustHist)            = MyFitted(...)          # walk-forward fit
trust_weights(ft::MyFitted, ::StakingMatch)    = w::Vector{Float64}     # 7-vector point
trust_draws(ft::MyFitted, m; D)                = W::Matrix              # 7×D (optional; default = replicate)
```

Then race it with no other change:

```julia
policies = ["MINE" => UnifiedPolicy(trust=MyTrust(), cap=0.2), ...]
rs = run_race(loaded, policies; refit_every=25)
```

`TrustHist` already records `home`/`away` team ids per observation, so a hierarchical model reads
them via `m.home`/`m.away` in `trust_weights(ft, m)`.

## Runners

- `r01_sim_race.jl`   — SimSource race (runs anywhere; validates the harness). ✅ locally validated.
- `r02_real_race.jl`  — real EB parity race (reproduces `staking_real` `e_real_summary_c020.txt`).
- `r03_ext_book.jl`   — extended 7-family book (O/U ladder adds value, CorrectScore drag).
- `r04_trust_models.jl` — **EB vs Bayesian** trust race + per-unit posterior comparison.
- `r05_team_eda.jl`   — Step-0 EDA: does per-unit `w` vary by team? (precondition for hierarchy).

`r02`–`r05` need the real L1 inputs — run `preflight_real.jl` first (needs the
`src_sup40_sw40` experiment payload on the server; caches to `results/_lat_ppd_cache.jls`).

## Roadmap (the reason for the refactor)

0. **EDA** (`r05`) — confirm team-level trust signal exists.
1. **Distributional `w`** — done: `UnifiedPolicy(distributional=true)` averages the Kelly solve over `trust_draws`.
2. **Bayesian trust** (`BayesianTrust`, l06) — done: Turing model, MCMC `w` draws.
3. **Hierarchical per-team `w`** — stub in l06; team-id plumbing already in place.

## Status

- Harness + all trust models + both sources: **loaded & validated locally** on the SimSource path
  (`r01`: CURATED ≻ EB ≻ FLAT ≻ raw-U-ruined; tilt exact to 1e-15) and the BayesianTrust fit
  (samples, sane ordered `w`).
- Real-data races (`r02`–`r05`): **written, parse-clean, awaiting a server run** (no L1 payload
  local this session). Expected to reproduce the `staking_real` committed results.
