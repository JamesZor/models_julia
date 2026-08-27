# Handover — Scottish Lower Poisson feature extensions

**Branch:** `feat/scottish-lower-protocol`  
**Latest implementation commit:** `1a1f5faa20788d9ad4aabad028a1d54bf308af22`  
**Initial scaffold commit:** `57577f10ee418e13bb240f99c2b1ab27d7d561ec`

## Objective

Verify three pure-Poisson extensions under `current_development/scottish_lower/`:

1. `02_poisson_wealth` — starting-XI wealth log ratio.
2. `03_poisson_distance` — stadium Haversine travel-distance fatigue.
3. `04_poisson_wealth_distance` — joint wealth and distance.

The intended stopping point is Gate 5. Do **not** launch the full 20-fold grid/Gate 10 during verification.

## Files created

Each arm has `MODEL.md`, `l01_model.jl`, `l02_equations.jl`, `l03_adapter.jl`, and `v01_walkthrough.jl`. Shared implementation is currently in:

- `02_poisson_wealth/l00_feature_poisson.jl`
- `03_poisson_distance/l00_distance_feature.jl`
- `03_poisson_distance/scottish_stadium_geocodes.csv`

A guarded overnight launcher exists at:

- `current_development/scottish_lower/r01_train_all.jl`

It requires `SL_RUN_GRIDS=true`; without that opt-in it runs only preflight Gates 0–2.

The shared Gate-2 comparator was extended in `_protocol/features.jl` for `:wealth_oos_bridge_by_match_id`. For `(full, truncated)`, every truncated key/value must exist identically in the full map; extra future keys in the full map are ignored.

## Implemented model structure

All arms retain the Model-00 baseline:

```text
η_h = μ + γ + α_home + β_away + feature_shift
η_a = μ     + α_away + β_home - feature_shift
```

The final predictors are clamped to `[-10, 10]`, and the likelihood is evaluated directly in log-intensity space:

```text
yη - exp(η) - loggamma(y + 1)
```

Log-factorials are precomputed outside `@model`; sampled arrays use `view`; no NaN/control-flow guard remains inside the model.

Dedicated engines now avoid prior-only inactive sites:

- TP02: `w_wealth` only, `5 + 2N` parameters.
- TP03: `w_dist` only, `5 + 2N` parameters.
- TP04: both coefficients, `6 + 2N` parameters.

Priors:

```text
w_wealth ~ truncated(Normal(0.10, 0.05), lower=0)
w_dist   ~ truncated(Normal(0.04, 0.03), lower=0)
```

## Feature semantics and findings

### Wealth

The current implementation uses the architecture brief's **log-sum XI** definition, not the production `SquadWealthFeature` geometric-mean definition:

```text
ΔW = log(sum(home starter values)) - log(sum(away starter values))
```

Only non-substitute lineup values with `valuation_timestamp < fixture kickoff` are accepted. Unknown player values use `100_000`; a side must have at least one safely observed valuation or the match receives neutral `0.0` plus fallback flag.

Important limitations:

- It does not currently enforce exactly 11 unique starters.
- It does not model lineup publication time, so this is best described as a kickoff/lineup-close feature, not necessarily an earlier forecast feature.
- It supports only the current `market_value`/`valuation_timestamp` schema, fewer aliases than production `SquadWealthFeature`.

### Distance

Distance uses the versioned 31-ground catalog and catalog-fixed standardization of `log1p(Haversine miles)`. Match-sample moments are not used, so future fixture deletion does not alter past values. Unmapped grounds use the documented 45-mile heuristic fallback and emit a fallback flag.

The road-mile/drive-minute estimates in the copied loader are heuristics, not measured routes. The model consumes `metric=:log_dist_z`.

### OOS extraction bridge

The package extraction API receives OOS rows, a fitted FeatureSet, and a chain, but no DataStore. To prevent all OOS wealth shifts silently becoming zero, the wealth feature stores a point-in-time all-match lookup in `:wealth_oos_bridge_by_match_id`. The training likelihood only consumes the fitted flat vector.

This works with the standard `Experiments.extract_oos_predictions` route when FeatureSets are rebuilt from the same DataStore. However, the bridge is **not persisted as an immutable per-fold snapshot**. If valuation rows drift while split boundaries remain unchanged, re-extraction may rebuild different OOS covariates. The Modeller recommended a later, stronger design: persist exact-ID OOS covariate snapshots with deterministic source/config hashes and attach them before extraction.

Distance does not need a stored bridge because it can be reconstructed from OOS team names and the static catalog.

## Verification completed

Using the cached Scottish Lower DataStore with `max_age_hours=100_000`:

### TP02 wealth

- Gate 0: passed.
- Gate 1: passed.
- Gate 2: 7/7 across all 20 folds.
- Gate 3a: 4/4; maximum log-density parity error `2.274e-13`.
- Gate 3b: 7/7; ReverseDiff/ForwardDiff relative error about `7.67e-16`; median compiled gradient about `0.514 ms`.
- Gate 4 synthetic extraction: passed, maximum parity error `2.220e-16`.
- Gate 4 fallback extraction: passed after fixing global-home-advantage fallback.
- Synthetic Gate 5a/5b/5c: passed, including score-grid and all market identities.

### TP03 distance

- Gates 0–2: passed.
- Gate 3a: passed; parity error `2.274e-13`.
- Gate 3b: 7/7; median compiled gradient about `0.513 ms`.
- Synthetic and fallback Gate 4: passed.
- Synthetic Gate 5 was not yet run in the recorded Builder session.

### TP04 joint

- Gate 2: 7/7.
- Gate 3a: passed; parity error `2.274e-13`.
- Gate 3b: 7/7; median compiled gradient about `0.542 ms`.
- Synthetic and fallback Gate 4: passed.
- Gates 0/1 and synthetic Gate 5 were not explicitly run in the recorded Builder session.

No full-grid sampling was launched.

## Smoke status and compute-node REPL

The user explicitly requested that smoke sampling use the existing compute-node REPL.

Compute node access:

- Outer target: `scottish_runner:1.1`
- It contains a nested tmux session.
- Nested window `2` is the Kaimon Julia REPL (`julia>`/`agent>`).
- Nested window `1` is a bash shell.
- Repository: `/root/BayesianFootball`

The latest local implementation was rsynced to `/root/BayesianFootball` after the final wealth-lookup and global-HA fixes.

One TP02 smoke attempt was made in the REPL, but **sampling never started**. The long-lived REPL already had another `ScottishLowerProtocol` loaded; including it again caused conflicting exports and then ambiguous `sl_contract` in `Main`.

Recommended fix: run the smoke inside a fresh enclosing module (or restart the REPL), for example:

```julia
module TP02SmokeRun
using BayesianFootball, ThreadPinning, LinearAlgebra
include("current_development/scottish_lower/_protocol/ScottishLowerProtocol.jl")
using .ScottishLowerProtocol
include("current_development/scottish_lower/02_poisson_wealth/l01_model.jl")
include("current_development/scottish_lower/02_poisson_wealth/l02_equations.jl")
include("current_development/scottish_lower/02_poisson_wealth/l03_adapter.jl")

ds = BayesianFootball.Data.load_datastore_cached(
    BayesianFootball.Data.ScottishLower(); max_age_hours=100_000)
contract = ScottishLowerProtocol.sl_contract()
adapter = TP02Adapter()
pinthreads(:cores)
BLAS.set_num_threads(1)
results, path = ScottishLowerProtocol.sl_run_experiment(
    ds, adapter, contract; smoke=true)
(results=results, path=path)
end
```

Then run and assert:

```julia
ScottishLowerProtocol.sl_gate_convergence(results, adapter, contract; expected_folds=1)
loaded = ScottishLowerProtocol.sl_load_experiment(path)
g4b, latents = ScottishLowerProtocol.sl_gate_extraction_real(ds, loaded, adapter, contract)
sl_gate_score_dispatch(adapter, first(eachrow(latents.df)); max_goals=contract.max_goals)
sl_gate_score_grid(adapter, latents.df, contract)
sl_gate_market_identities(adapter, latents.df, contract)
```

Repeat in fresh module names for TP03 and TP04.

## Work still required

1. **Run one persisted smoke for each arm in the compute-node REPL.** Check convergence, reload from disk, real Gate 4b, and Gate 5 on real posterior latents.
2. **Update all `v01_walkthrough.jl` files.** They currently stop at non-MCMC/synthetic Gate 5; they need the baseline sequence: smoke Gate 3c → persisted reload → real Gate 4b → posterior Gate 5. They must still stop before full grid.
3. **Run synthetic Gate 5 for TP03 and TP04** if not superseded by real posterior Gate 5.
4. **Add focused tests** for log-sum PIT wealth, unsafe timestamp rejection, neutral fallback, nonzero OOS bridge values, bridge perturbation invariance, and exact match-ID lookup.
5. **Check real OOS feature coverage.** Require at least one nonfallback/nonzero wealth OOS value where data coverage exists; report wealth and distance fallback rates.
6. **Review code readability.** `l00_feature_poisson.jl` and adapters are currently heavily compressed with semicolon-packed one-line functions. The math passes static gates, but the files should be reformatted/documented before long-term use.
7. **Review model documentation.** Ensure each `MODEL.md` explicitly says log-sum XI wealth, current fallback/decision-time semantics, clamp bounds, exact priors, and sampled-site count.
8. **Dry-run `r01_train_all.jl` after refactoring.** Leave `SL_RUN_GRIDS` unset/false. Confirm only preflight executes.
9. **Do not run the full grids** until all three persisted smokes and real Gates 4–5 pass.

## Useful reports

Temporary research reports from the Modeller and Scout are available in the originating environment:

- `/tmp/modeller_poisson_extensions.md`
- `/tmp/scout_poisson_extensions.md`

The Scout report documents archive/data sources and OOS extraction-interface findings. The Modeller report contains equations, AD requirements, bridge recommendations, and an earlier defect audit.
