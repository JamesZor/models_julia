# 06 — Typed Posterior Latents

Dense, typed containers for out-of-sample posterior draws, replacing
`Experiments.LatentStates`' untyped `DataFrame`.

```
julia --project current_development/06_typed_posterior_latents/r01_demo.jl
```

Runs in seconds, needs no database, no cache and no MCMC, and exits non-zero if any
gate fails.

---

## The problem

`extract_oos_predictions` returns a `LatentStates` wrapping a `DataFrame` whose every
cell holds a full posterior sample vector. `_latent_state_dict_to_df`
(`src/experiments/post_processing.jl:205`) builds its columns as `Vector{Any}`, which
gives three distinct costs:

| | |
|---|---|
| **Type instability** | `row.λ_h` is a boxed `Any`. `compute_score_matrix` — the hottest loop in the prediction path — cannot specialise on it. |
| **Fragmentation** | `n_matches × n_parameters` separately-allocated vectors. A 500-fixture fold is ~2,000 heap objects where 4 would do. |
| **Ad-hoc schema** | Consumers rediscover the columns at *prediction* time with `hasproperty` chains (`negativebinomial.jl:9-27`, `recombination.jl:18-40`). A schema mistake surfaces as a plausible wrong price, hours after the run that produced it. |

## The replacement

One `Matrix{Float64}` per parameter, `(n_matches × n_draws)`, and the schema carried in
the **type**:

```julia
CountLatents{Float64, Nothing}      # double-Poisson
CountLatents{Float64, <:NamedTuple} # double-NegBin, carries (; r_h, r_a)
RecombLatents{Float64}              # open play + penalties + own goals + pxG
SmileLatents{Float64, Obs}          # grid + per-strike market smile φ(K)
```

The `hasproperty` cascades become method dispatch, resolved at compile time. A missing
parameter is a `MethodError` at construction rather than a wrong price at settlement.

## Files

| file | contents |
|---|---|
| `l01_latents.jl` | the type hierarchy, construction-time validation, memory accounting |
| `l02_extract.jl` | `extract_latents(model, chain, oos_fixtures, feature_set)`, the family trait, and the two-way legacy `DataFrame` bridge |
| `l03_score_grids.jl` | `compute_score_grid!` and `price_market!` — zero-allocation kernels, one per family |
| `l04_parity.jl` | ULP-exact parity harness, allocation audit, memory/timing comparison, deterministic synthetic posteriors |
| `r01_demo.jl` | the runner: 12 sections, 34 gates |

`r01_demo.jl` is the only file that executes anything. Each loader includes the one
below it, so `include("l04_parity.jl")` loads the whole prototype.

## Using it

```julia
# one container per fold
latents = extract_latents(model, chain, oos_fixtures, feature_set)

# one workspace and one destination grid per WORKER, not per fixture
ws   = GridWorkspace()
S    = alloc_score_grid(latents)
book = alloc_market_book(Market1X2(), n_draws(latents))

for i in 1:n_matches(latents)
    compute_score_grid!(S, ws, latents, i)      # 0 bytes
    price_market!(book, S, Market1X2())         # 0 bytes
    home, draw, away = book
end
```

Allocating forms — `compute_score_grid(latents, i)`, `price_market(S, market)` — exist
for the REPL. `price_market` returns a `Dict{Symbol,Vector{Float64}}` keyed exactly as
`Predictions.compute_market_probs` keys its result, so a consumer can be swapped one
call site at a time.

Migration runs in both directions. `latents_from_legacy_dataframe(model, df)` reads an
already-cached `oos_latents.jls` without refitting — for the count and smile families,
including dispersion and the smile curve φ. `to_legacy_dataframe(latents)` feeds a
container to a consumer that has not moved yet.

The one exception is deliberate and refuses loudly: a `RecombLatents` **cannot** be
rebuilt from a legacy frame. The frame carries the recombined totals but neither
`q_pen` nor `og_rate`, so `λ_total − λ_open = q_pen·λ_pen + og_rate` is one equation in
two unknowns and the penalty and own-goal channels cannot be separated. Guessing would
put own goals in the penalty channel, invisibly. Rebuild it from the chain with
`extract_latents` instead.

## What `r01_demo.jl` establishes

**Parity.** Every price from a typed container is **bit-identical (0 ULP)** to the price
the live `src` kernels produce from the equivalent `latents.df` — across five
containers, six markets, 24 fixtures and every posterior draw. The legacy side always
goes through the real `Predictions.extract_params` → `compute_score_matrix` →
`compute_market_probs`, never a transcription, so the tables track `src` as it changes.

The gate is 0 ULP, not the briefing's `|Δ| < 1e-12`. §8d is a negative control that
perturbs one λ by one ULP and requires exactly that fixture to fail; it is what
established that a tolerance-only gate would have passed a real 16-ULP change.

**Zero allocations.** `compute_score_grid!` and `price_market!` measure 0 bytes under
`@allocated` for all five containers and all six markets, alongside an empty-closure
baseline row that must also read 0.

**Cost, reported honestly.** Heap objects drop from ~50-80 per fold to 2-4. Bytes go
*up* for the NegBin and smile families, because the container materialises per-fixture
dispersion and φ where the engines currently share one object across every row —
`l01_latents.jl` §3 and §5 give the reasoning, and §11 of the runner prints the bill.
The NegBin score grid is ~9.6× faster; Poisson and market pricing are marginally faster.

**Not established:** that any model fits anything. The posteriors are prior draws with a
fixed seed. The claim is only that moving them out of a DataFrame changes no number.

## Defects found (§6 of the runner, reproduced live, none fixed here)

1. **`DynamicPxGRecombModel`'s OOS extractor cannot be called.**
   `recombined_pxg.jl:191,214` call `extract_dynamics(chain, config, n_teams)`; every
   method takes `(chain, config, prefix::String, n_teams)`. Both raise `MethodError`.
   `l02_extract.jl` §4 reimplements the body with the 4-argument call.

2. **Three extractors mis-size their output under multiple chains.**
   `extract_recombination`, `extract_squad_wealth` and `extract_pxg_observation` use
   `size(chain, 1)` where every other extractor uses `size(chain,1) * size(chain,3)`.
   The runner exercises the recombination family with one chain for this reason.

3. **Nine `src` call sites use a `haskey` that MCMCChains 7.7 removed.**
   `haskey(::Chains, ::Symbol)` existed in 7.6 and does not in 7.7;
   `Project.toml` currently allows both. `l02_extract.jl` §0 carries a one-line compat
   shim, flagged as type piracy, so this prototype can call the real `src` extractors
   rather than re-implement four of them. The fix belongs in `src` (move to
   `sym in names(chain)`) or in the `MCMCChains` bound.

## Naming note

The briefing names the smile pricing container `SmileScoreMatrix`. It is
`SmileScoreGrid` here, because `Predictions.SmileScoreMatrix` already exists with a
different shape and `l04_parity.jl` holds both at once to compare them.
