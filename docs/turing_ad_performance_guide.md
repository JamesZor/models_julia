# Turing.jl AD Performance Guide
## Writing Allocation-Free, SIMD-Optimal MCMC Models

> **Audience:** Anyone writing or modifying Turing `@model` blocks in this project.
> **Measured impact:** Following these rules reduced gradient evaluation time from
> multi-millisecond scalar-loop overhead to **0.64ms** on a 500+ match dataset with
> a compiled `ReverseDiff.GradientTape`.

---

## Table of Contents

1. [The Two-Layer Architecture](#1-the-two-layer-architecture)
2. [How ReverseDiff Tapes Work](#2-how-reversediff-tapes-work)
3. [The Cardinal Rules](#3-the-cardinal-rules)
4. [Binary Masking: Eliminating Conditionals](#4-binary-masking-eliminating-conditionals)
5. [View vs Getindex: Parameter Selection](#5-view-vs-getindex-parameter-selection)
6. [The Builder Pattern: Feature Layer Does the Work](#6-the-builder-pattern-feature-layer-does-the-work)
7. [Numerical Safety](#7-numerical-safety)
8. [Benchmarking Your Model](#8-benchmarking-your-model)
9. [Gold Standard Reference](#9-gold-standard-reference)
10. [AD Backend Comparison](#10-ad-backend-comparison)
11. [Checklist](#11-checklist)

---

## 1. The Two-Layer Architecture

Every model in this project follows a strict separation of concerns:

```
┌─────────────────────────────────────┐
│         Feature Builder             │
│  (build_turing_model function)      │
│                                     │
│  • SQL → DataFrame → flat vectors   │
│  • NaN imputation (1.0 / 0.0)       │
│  • Binary mask generation           │
│  • Score-based mask computation      │
│  • Time-decay weight calculation    │
│  • All conditional logic lives HERE │
└──────────────┬──────────────────────┘
               │ Pure Float64 vectors
               ▼
┌─────────────────────────────────────┐
│         Turing @model               │
│  (the engine function)              │
│                                     │
│  • Sample priors (~)                │
│  • Broadcast arithmetic (.+, .*)    │
│  • Broadcast likelihoods (logpdf.)  │
│  • Multiply by masks & weights      │
│  • sum() + @addlogprob!             │
│  • NO loops, NO if/else, NO NaN    │
└─────────────────────────────────────┘
```

> [!IMPORTANT]
> **The model is only for modelling.** Every line inside `@model` must be a pure
> mathematical vector operation. If you find yourself writing `if`, `for`, or
> `isnan` inside `@model`, you are doing it wrong — move that logic to the builder.

---

## 2. How ReverseDiff Tapes Work

ReverseDiff.jl is a **tape-based** reverse-mode AD backend. Understanding the tape
is the key to understanding performance.

### What the tape records

When you call `ReverseDiff.GradientTape(f, θ)`, it executes `f(θ)` once and records
every mathematical operation into a linear instruction list (the "tape"). On each
subsequent `gradient!` call, it replays that tape forwards then backwards to compute
∂f/∂θ.

### TrackedArray vs Array{TrackedReal}

This is the single most important distinction for performance:

| Pattern | Internal Type | Tape Nodes | Performance |
|:--------|:-------------|:-----------|:------------|
| `logpdf.(Poisson.(λ), y)` | `TrackedArray` | **1** array op | 🟢 Fast |
| `[logpdf(Poisson(λ[i]), y[i]) for i in 1:N]` | `Array{TrackedReal}` | **N** scalar ops | 🔴 Slow |

When you use **broadcasting** (the `.` syntax), ReverseDiff wraps the entire
operation in a single `TrackedArray` node. The tape records one instruction that
points to an optimised BLAS/SIMD kernel.

When you use a **scalar loop or comprehension**, each iteration creates individual
`TrackedReal` values. The tape explodes to `O(N)` nodes, each carrying its own
allocation overhead. For N=500 matches with 3 likelihood pillars, that's 1,500+
individual tape nodes instead of ~3.

### compile=true

With `AutoReverseDiff(compile=true)`, Turing records the tape **once** and reuses
it for every NUTS leapfrog step. This is a massive speedup but has a critical
constraint:

> [!WARNING]
> **The compiled tape is static.** If your model contains `if/else` branches that
> depend on sampled parameters, the tape will silently follow whichever branch was
> taken during the initial recording. Gradients will be **silently wrong** —
> no error, no warning, just corrupted MCMC chains.

This is precisely why we use binary masking instead of conditionals.

---

## 3. The Cardinal Rules

### Rule 1: No scalar loops in @model

```julia
# ❌ BAD — creates N TrackedReal nodes
ll = zero(ν)
for i in 1:length(y)
    ll += logpdf(Poisson(λ[i]), y[i])
end
@addlogprob! ll

# ✅ GOOD — creates 1 TrackedArray node
ll = logpdf.(Poisson.(λ), y)
@addlogprob! sum(ll .* match_weights)
```

### Rule 2: No conditionals in @model

```julia
# ❌ BAD — breaks compiled tape, creates branching
ll_xg = if !isnan(home_xg[i])
    logpdf(Gamma(ν, λ[i]/ν), home_xg[i])
else
    0.0
end

# ✅ GOOD — multiply by binary mask (0.0 or 1.0)
ll_xg = logpdf.(Gamma.(ν, λ ./ ν), home_xg)
@addlogprob! sum(ll_xg .* match_weights .* xg_mask)
```

### Rule 3: No array subsetting of intermediate tracked arrays

```julia
# ❌ BAD — getindex on TrackedArray creates Array{TrackedReal}
λ_xg = λ[idx_xg]
ll = logpdf.(Gamma.(ν, λ_xg ./ ν), home_xg[idx_xg])

# ✅ GOOD — compute on ALL matches, zero out missing via mask
ll = logpdf.(Gamma.(ν, λ ./ ν), home_xg)
@addlogprob! sum(ll .* xg_mask)
```

### Rule 4: Use view() for parameter arrays, not getindex

```julia
# ❌ BAD — allocates a new array, may break tracking
ha_match = ha[home_team_indices]

# ✅ GOOD — zero-copy, preserves TrackedArray
ha_match = view(ha, home_team_indices)
```

`view()` on a **parameter** array (sampled via `~`) returns a tracked view that
shares the gradient buffer. `getindex` (`A[indices]`) allocates a new array and
can degrade tracking to `Array{TrackedReal}`.

> [!NOTE]
> The distinction between `view` and `getindex` matters most for **parameter**
> arrays (things you sample). For **data** arrays (things passed as arguments to
> the model function), `getindex` is fine because data isn't tracked.

### Rule 5: No array mutation

```julia
# ❌ BAD — in-place mutation breaks tape tracking
A[i] = x
A .+= 1

# ✅ GOOD — create new arrays via broadcasting
A_new = A .+ 1
```

---

## 4. Binary Masking: Eliminating Conditionals

The masking pattern replaces every conditional branch with a multiplication by
`0.0` or `1.0`. This works because `0.0 * anything = 0.0`, so masked-out
likelihood contributions vanish from the sum without any branching.

### Pattern: Missing data (xG, market odds)

**In the builder** (`build_turing_model`):

```julia
# 1. Extract raw data (may contain missing/NaN)
home_xg_raw = coalesce.(data[:flat_home_xg], NaN)

# 2. Build the mask: 1.0 where data exists, 0.0 where missing
xg_mask = Float64.(.!isnan.(home_xg_raw))

# 3. Impute NaN → harmless constant (1.0 for Gamma, 0.0 for Normal)
#    The value doesn't matter because the mask will zero it out,
#    but it must not cause DomainError in the distribution.
home_xg = [isnan(x) ? 1.0 : Float64(x) for x in home_xg_raw]
```

**In the model** (`@model`):

```julia
ll_xg = logpdf.(Gamma.(ν, λ ./ ν), home_xg)        # computed for ALL matches
@addlogprob! sum(ll_xg .* match_weights .* xg_mask)  # masked-out → 0 contribution
```

### Pattern: Score-based masks (Dixon-Coles τ)

The Dixon-Coles correction factor τ depends on whether the scoreline is (0,0),
(1,0), (0,1), (1,1), or other. Instead of `if home_goals == 0 && away_goals == 0`,
we pre-compute binary masks:

**In the builder:**

```julia
mask_00    = Float64.((home_goals .== 0) .& (away_goals .== 0))
mask_10    = Float64.((home_goals .== 1) .& (away_goals .== 0))
mask_01    = Float64.((home_goals .== 0) .& (away_goals .== 1))
mask_11    = Float64.((home_goals .== 1) .& (away_goals .== 1))
mask_other = 1.0 .- mask_00 .- mask_10 .- mask_01 .- mask_11
```

**In the model:**

```julia
τ = (mask_00 .* τ_00) .+ (mask_10 .* τ_10) .+ (mask_01 .* τ_01) .+
    (mask_11 .* τ_11) .+ mask_other
```

Each match selects exactly one τ value through multiplication. No `if`, no
branching, perfectly static tape.

### Imputation value selection

| Distribution | Impute NaN to | Why |
|:------------|:-------------|:----|
| `Gamma(α, θ)` | `1.0` | Must be > 0 to avoid `DomainError` |
| `Normal(μ, σ)` | `0.0` | Any real number works; 0.0 is neutral |
| `Poisson(λ)` | `0` | Must be non-negative integer |

The imputed value is irrelevant to the posterior because the mask zeros out
its log-probability contribution. It just needs to be in the distribution's
support to avoid a runtime error.

---

## 5. View vs Getindex: Parameter Selection

When you sample a vector parameter like `ha ~ filldist(Normal(0, 1), n_teams)`,
Turing gives you a `TrackedArray`. To select per-match values:

```julia
# ✅ Use view — returns a tracked view, zero-copy
home_adv = view(ha, home_team_indices)

# ❌ Avoid getindex — allocates, may break tracking
home_adv = ha[home_team_indices]
```

For **multi-level indexing** (season intercept + month offset):

```julia
# ✅ Compose views with broadcasting
int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
```

---

## 6. The Builder Pattern: Feature Layer Does the Work

The `build_turing_model` function is where all data wrangling happens. Its job is
to transform messy real-world data into pristine `Vector{Float64}` and
`Vector{Int}` arrays that the model can consume without any branching.

### Builder responsibilities

1. **Extract** raw data from the `FeatureSet` dict
2. **Cast** to concrete types (`Vector{Int}`, `Vector{Float64}`)
3. **Impute** missing values to harmless constants
4. **Build** binary masks for optional data sources
5. **Compute** time-decay weights
6. **Build** score-based masks (e.g., Dixon-Coles τ masks)
7. **Return** the instantiated `@model` function with all data baked in

### Template

```julia
function build_turing_model(config::MyModel, feature_set::FeatureSet)
    data = feature_set.data

    # --- Dimensions ---
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12

    # --- Index vectors (Int) ---
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])

    # --- Observation vectors (Float64) ---
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    # --- Time decay weights ---
    date_deltas   = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.days_half_life)

    # --- Optional data with masking ---
    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    xg_mask     = Float64.(.!isnan.(home_xg_raw))
    home_xg     = [isnan(x) ? 1.0 : Float64(x) for x in home_xg_raw]

    # --- Pass everything as flat vectors ---
    return my_engine(
        home_ids, away_ids, season_ids,
        home_goals, away_goals, match_weights,
        home_xg, xg_mask,
        n_teams, n_seasons, n_months, config
    )
end
```

---

## 7. Numerical Safety

Even with perfect vectorisation, MCMC samplers explore extreme regions of parameter
space. These guards prevent `NaN`/`Inf` from crashing the sampler.

### Clamp log-rates

```julia
log_λ = clamp.(intercept .+ attack .+ defence, -20.0, 20.0)
```

Without this, `exp(50.0) = Inf` and `Poisson(Inf)` throws an error.

### Epsilon floors

```julia
λ = exp.(log_λ) .+ 1e-6
```

Prevents `Poisson(0.0)` which has undefined `logpdf` for non-zero observations.

### AD-Safe rejection

```julia
if any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    Turing.@addlogprob! -Inf
    return
end
```

> [!NOTE]
> This `if` statement is **safe** for `compile=true` because it depends on
> the sampled parameters, not data. If the sampler wanders into NaN territory,
> we reject the proposal. This branch should almost never be taken in practice —
> the `clamp` and `1e-6` guards prevent it. But if it does fire, it cleanly
> rejects rather than corrupting the chain.

### Bounded transformations

For parameters that must stay in a range (like Dixon-Coles ρ):

```julia
ρ = 0.3 .* tanh.(ρ_raw)  # Maps ℝ → (-0.3, 0.3)
```

---

## 8. Benchmarking Your Model

Use this script pattern to measure gradient evaluation time:

```julia
using BayesianFootball
using DynamicPPL, LogDensityProblems, ReverseDiff, BenchmarkTools

# 1. Load data and build model (your standard pipeline)
ds = Data.load_datastore_cached(Data.Ireland())
boundaries = Data.create_id_boundaries(ds, Data.GroupedCVConfig(...))
features = Features.create_features(boundaries, ds, model_config)
model = PreGame.build_turing_model(model_config, features[1])

# 2. Initialise VarInfo and extract parameter vector
vi = DynamicPPL.VarInfo(model)
model(vi)
θ = vi[:]

# 3. Build LogDensityFunction wrapper
lf = DynamicPPL.LogDensityFunction(model)
f = x -> LogDensityProblems.logdensity(lf, x)

# 4. Compile ReverseDiff tape
tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))

# 5. Benchmark gradient evaluation
time_eval = @belapsed ReverseDiff.gradient!($(similar(θ)), $tape, $θ)
println("Gradient eval: $(round(time_eval * 1000, digits=3)) ms")
```

### What to look for

| Metric | Good | Bad |
|:-------|:-----|:----|
| Gradient eval time | < 1ms | > 5ms |
| Tape compilation | < 30s | > 2min |
| Memory per gradient | ~0 allocs | thousands of allocs |

If your gradient eval is slow, check for:
1. Scalar loops inside `@model`
2. `getindex` on intermediate tracked arrays
3. Missing masks (using `if !isempty(idx)` instead)

---

## 9. Gold Standard Reference

The canonical example of a correctly written model is
[outfield_xg_dixon_coles.jl](file:///home/james/bet_project/BayesianFootball/src/models/pregame/engines/player_level/time_decay/outfield_xg_dixon_coles.jl).

It demonstrates every pattern in this guide:
- Binary masking for xG (`xg_mask`), market (`market_mask`), and Dixon-Coles τ (`mask_00`..`mask_other`)
- `view()` for parameter selection
- Pure broadcast arithmetic
- `clamp` + `1e-6` numerical guards
- AD-safe rejection
- Clean builder with full imputation

When writing a new model, start by copying this file and modifying the likelihood
section.

---

## 10. AD Backend Comparison

| Feature | ReverseDiff.jl | Mooncake.jl | ForwardDiff.jl |
|:--------|:--------------|:------------|:---------------|
| **Type** | Tape-based tracing | Source-to-source IR transform | Dual numbers |
| **Scaling** | O(1) in #params (reverse mode) | O(1) in #params | O(N) in #params |
| **Compiled tape** | ✅ `compile=true` | N/A (always compiles) | N/A |
| **Dynamic control flow** | ❌ Silent errors with `compile=true` | ✅ Fully supported | ✅ Fully supported |
| **Array mutation** | ❌ Errors or silent failures | ✅ First-class support | ⚠️ Limited |
| **Sweet spot** | Static, vectorised models (our use case) | Complex Julia code with mutations | Few parameters (< 100) |

> [!TIP]
> We use `AutoReverseDiff(compile=true)` because our models are 100% static
> (no parameter-dependent branching in the likelihood). If you ever need dynamic
> control flow, consider `AutoMooncake()` as an alternative — but you'll lose
> the compiled tape speedup.

---

## 11. Checklist

Before submitting a new model engine, verify:

- [ ] **No `for` loops** inside `@model`
- [ ] **No `if`/`else`** inside `@model` (except the NaN/Inf rejection guard)
- [ ] **No `isnan()`** inside `@model`
- [ ] **No `A[indices]`** on intermediate tracked arrays — use `view(A, indices)` or masks
- [ ] All optional data uses **binary masks** (`xg_mask`, `market_mask`)
- [ ] NaN values **imputed** to distribution-safe constants in the builder
- [ ] **`clamp`** on all log-rates to prevent `exp()` overflow
- [ ] **`1e-6`** epsilon floor on all rate parameters
- [ ] All likelihood terms multiplied by **`match_weights`** before `sum()`
- [ ] `@belapsed` gradient eval is **< 1ms** for typical dataset sizes
- [ ] Model compiles with `ReverseDiff.GradientTape` without errors
