# Turing.jl AD Performance Guide
## Writing Allocation-Free, SIMD-Optimal MCMC Models

> **Audience:** Anyone writing or modifying Turing `@model` blocks in this project.
>
> **Measured impact:** on ScottishLower fold 1 (720 matches, ~50 parameters) a correctly
> written engine evaluates a compiled `ReverseDiff.GradientTape` in **0.035 ms**. The
> `02/03/04_poisson_*` arms, written to the *previous* version of this guide, take
> **0.52–0.61 ms** for a **bit-identical** log-density — a 15x penalty from two lines. The
> `src` NegBin engine measured 1.15 ms on the same fold in
> [T002](tickets/T002-scalar-taped-likelihood.md). Numbers in §5 and §6.
>
> **Revised 2026-08-28.** Rule 4 was **backwards** and every engine followed it. If you have
> read this guide before, read §5 again. See [T002](tickets/T002-scalar-taped-likelihood.md).

---

## Table of Contents

1. [The Two-Layer Architecture](#1-the-two-layer-architecture)
2. [How ReverseDiff Tapes Work](#2-how-reversediff-tapes-work)
3. [The Cardinal Rules](#3-the-cardinal-rules)
4. [Binary Masking: Eliminating Conditionals](#4-binary-masking-eliminating-conditionals)
5. [Parameter Selection: getindex, not view](#5-parameter-selection-getindex-not-view)
6. [Keep Constants Out of Tracked Expressions](#6-keep-constants-out-of-tracked-expressions)
7. [Composing a Variable Number of Terms](#7-composing-a-variable-number-of-terms)
8. [The Builder Pattern: Feature Layer Does the Work](#8-the-builder-pattern-feature-layer-does-the-work)
9. [Numerical Safety](#9-numerical-safety)
10. [Benchmarking and Verifying Your Model](#10-benchmarking-and-verifying-your-model)
11. [Reference Implementations](#11-reference-implementations)
12. [AD Backend Comparison](#12-ad-backend-comparison)
13. [Checklist](#13-checklist)

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
│  • Score-based mask computation     │
│  • Time-decay weight calculation    │
│  • Log-factorials & other constants │
│  • All conditional logic lives HERE │
└──────────────┬──────────────────────┘
               │ Pure Float64 / Int vectors
               ▼
┌─────────────────────────────────────┐
│         Turing @model               │
│  (the engine function)              │
│                                     │
│  • Sample priors (~)                │
│  • Broadcast arithmetic (.+, .*)    │
│  • Indexed parameter selection      │
│  • Broadcast likelihoods (logpdf.)  │
│  • Multiply by masks & weights      │
│  • sum() + @addlogprob!             │
│  • NO loops, NO if/else, NO NaN     │
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

When you use **broadcasting** (the `.` syntax) on a `TrackedArray`, ReverseDiff records
one instruction pointing at a vectorised kernel. When any operand has already degraded to
`Array{TrackedReal}` — a vector whose *elements* are individually tracked — every
downstream broadcast is taped element by element and the tape grows with the row count.

> [!IMPORTANT]
> **The failure mode is contagious.** One operation that degrades a `TrackedArray` to
> `Array{TrackedReal}` puts *everything downstream of it* on the scalar path. This is why
> `view` (§5) is so expensive: it is a single innocuous-looking call at the top of the
> engine that converts the entire likelihood to scalar taping.

### Two numbers, not one

Gradient cost is instruction **count** × instruction **width**. Both matter, and count
alone will mislead you:

| engine | tape instructions | compiled gradient |
|---|---|---|
| composable engine, weight fused into the likelihood | 78 | 0.061 ms |
| composable engine, weight applied separately | **80** | **0.038 ms** |

More instructions, less time. §6 explains why.

### compile=true

With `AutoReverseDiff(compile=true)`, Turing records the tape **once** and reuses
it for every NUTS leapfrog step. This is a large speedup with a hard constraint:

> [!WARNING]
> **The compiled tape is static.** If your model contains branches that depend on sampled
> parameter *values*, the tape follows whichever branch was taken during the initial
> recording. Gradients are **silently wrong** — no error, no warning, just corrupted chains.

This is why we use binary masking instead of conditionals. But note carefully:

> [!CAUTION]
> **`clamp` is a value-dependent branch.** So are `min`, `max`, `abs`, `ifelse` on tracked
> values, and `isinteger`-style dispatch inside a `logpdf`. Writing them without the word
> `if` does not make them static.
>
> This is not hypothetical. Probing both a hand-written `src`-style engine and the
> composable engine with a compiled tape against `ForwardDiff` at 40 points drawn ~0.8 units
> per coordinate from a prior draw, **both disagreed, worst relative error 0.37**, in the
> same places. Near the typical set both are exact (relerr 0.0). The cause was **not
> isolated** — `clamp`, `exp` overflow and the `max` inside `loggamma` are all candidates —
> and is recorded as [T002 §(c)](tickets/T002-scalar-taped-likelihood.md).
>
> Until that is resolved, treat "the tape is exact at a prior draw" as insufficient evidence
> and run the perturbed-point check in §10.3. Warm-up is exactly when the sampler leaves the
> typical set, and exactly when it most needs correct gradients.

---

## 3. The Cardinal Rules

### Rule 1: No scalar loops in @model

```julia
# ❌ AVOID — creates N TrackedReal nodes
ll = zero(ν)
for i in 1:length(y)
    ll += logpdf(Poisson(λ[i]), y[i])
end
@addlogprob! ll

# ✅ DO — creates 1 TrackedArray node
ll = logpdf.(Poisson.(λ), y)
@addlogprob! sum(ll .* match_weights)
```

If the number of *terms* is not known when you write the engine, do not reach for a loop —
see §7.

### Rule 2: No value-dependent branches in @model

```julia
# ❌ AVOID — breaks the compiled tape
ll_xg = if !isnan(home_xg[i])
    logpdf(Gamma(ν, λ[i]/ν), home_xg[i])
else
    0.0
end

# ✅ DO — multiply by a binary mask (0.0 or 1.0) built in the builder
ll_xg = logpdf.(Gamma.(ν, λ ./ ν), home_xg)
@addlogprob! sum(ll_xg .* match_weights .* xg_mask)
```

A branch on a **type** is fine and free — it is resolved at compile time and emits no tape
instruction at all. A branch on a **value** is the problem:

```julia
# ✅ DO — dispatch, resolved before the tape exists
apply_guard(::NoGuard, η)   = η
apply_guard(g::ClampGuard, η) = clamp.(η, g.lo, g.hi)

# ❌ AVOID — a runtime test, even though it looks like configuration
η = config.use_guard ? clamp.(η, -10.0, 10.0) : η
```

### Rule 3: Do not subset observations; mask them

```julia
# ❌ AVOID — a data-dependent index set makes the tape shape depend on which rows
#            happen to have xG, so the same engine tapes differently per fold
idx_xg = findall(!isnan, home_xg)
λ_xg   = λ[idx_xg]
ll     = logpdf.(Gamma.(ν, λ_xg ./ ν), home_xg[idx_xg])

# ✅ DO — compute on ALL matches, zero out the absent ones with a mask
ll = logpdf.(Gamma.(ν, λ ./ ν), home_xg)
@addlogprob! sum(ll .* xg_mask)
```

> [!NOTE]
> **Corrected reason.** Previous versions of this guide justified this rule by claiming
> `getindex` on a `TrackedArray` produces `Array{TrackedReal}`. That is **false** — see §5.
> The rule survives for a different and better reason: a `findall`-style index set is
> conditional logic wearing a costume. It makes the tape a function of the data rather than
> of the model, so it changes between folds, and it hides missingness from every gate that
> checks masks.
>
> A **static** index vector built in the builder — `home_team_indices`, `season_indices` —
> is not subsetting and is entirely fine. That is §5.

### Rule 4: Select parameters with `getindex`, not `view`

**This rule was previously stated the other way round and is wrong in the old direction.**
It has its own section — see §5.

```julia
# ✅ DO
home_adv = ha[home_team_indices]

# ❌ AVOID
home_adv = view(ha, home_team_indices)
```

### Rule 5: No array mutation

```julia
# ❌ AVOID — in-place mutation breaks tape tracking
A[i] = x
A .+= 1

# ✅ DO — create new arrays via broadcasting
A_new = A .+ 1
```

### Rule 6: Constants belong in the builder, and outside the tracked expression

```julia
# ❌ AVOID — loggamma of data, recomputed inside the model
log_fact = loggamma.(Float64.(y) .+ 1.0)

# ✅ DO — computed once in build_turing_model, passed in as a Vector{Float64}
```

And once it is data, keep it out of the tracked kernel — see §6.

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
ll_xg = logpdf.(Gamma.(ν, λ ./ ν), home_xg)          # computed for ALL matches
@addlogprob! sum(ll_xg .* match_weights .* xg_mask)  # masked-out → 0 contribution
```

### Pattern: Score-based masks (Dixon-Coles τ)

The Dixon-Coles correction factor τ depends on whether the scoreline is (0,0),
(1,0), (0,1), (1,1), or other. Instead of `if home_goals == 0 && away_goals == 0`,
pre-compute binary masks:

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

### Pattern: a linear covariate needs no mask at all

For a term that is **linear** in the data — `w * x` — imputing an absent value to `0.0`
*is* the mask, because `w * 0 = 0`. Carrying a second vector buys nothing:

```julia
# ✅ DO — the feature extractor imputes absent covariates to 0.0
q = w .* delta_wealth        # absent fixtures contribute exactly nothing
```

This holds **only** because the term is linear in `x`. A non-linear covariate (a log, a
threshold, anything inside a `logpdf`) needs a real mask. Say which one you are relying on
in a comment, because the next person will copy the pattern into a non-linear term.

### Imputation value selection

| Distribution | Impute NaN to | Why |
|:------------|:-------------|:----|
| `Gamma(α, θ)` | `1.0` | Must be > 0 to avoid `DomainError` |
| `Normal(μ, σ)` | `0.0` | Any real number works; 0.0 is neutral |
| `Poisson(λ)` | `0` | Must be non-negative integer |
| linear term `w·x` | `0.0` | The imputation *is* the mask |

---

## 5. Parameter Selection: `getindex`, not `view`

> [!WARNING]
> **This reverses the previous version of this guide.** Every engine in
> `src/models/pregame/engines/` currently uses `view` because the guide told it to, and each
> one pays roughly 15x for it. See [T002](tickets/T002-scalar-taped-likelihood.md).

When you sample a vector parameter like `ha ~ filldist(Normal(0, 1), n_teams)`, Turing gives
you a `TrackedArray`. To select per-match values:

```julia
# ✅ DO — one vectorised gather node, adjoint is a scatter
home_adv = ha[home_team_indices]

# ❌ AVOID — a SubArray of TrackedReal; every downstream broadcast tapes per element
home_adv = view(ha, home_team_indices)
```

### Minimal reproduction

```julia
idx = rand(1:23, 720); base = randn(23)
ninst(f, x) = length(ReverseDiff.GradientTape(f, x).tape)

ninst(θ -> sum(exp.(view(θ, idx))), base)   # 1439 instructions
ninst(θ -> sum(exp.(θ[idx])),       base)   #    3 instructions
```

`view` on a `TrackedArray` yields a `SubArray` whose elements are individually tracked.
ReverseDiff has no vectorised rule for it, so it falls back to element-wise taping — and
because the degradation is contagious (§2), the *entire* likelihood downstream goes scalar.

### Measured cost

ScottishLower 56+57, fold 1 (720 matches, 50–52 parameters), Julia 1.12.6 /
ReverseDiff 1.17.0. Compiled tape, minimum of 400 reps after 50 warm-up calls. **Identical
log-density in every row.**

**Controlled A/B** — one engine, one line changed, identical log-density:

| | selection | compiled gradient |
|---|---|---|
| isolated micro-model, 720 rows / 23 params | `view(A, idx)` | 0.389 ms |
| isolated micro-model, 720 rows / 23 params | `A[idx]` | **0.076 ms** |

**Whole engines on fold 1**, all four pairs bit-identical in log-density:

| arm | hand-written (`view`) | composable (`A[idx]`) | ratio |
|---|---|---|---|
| 00 baseline | 0.035 ms *(already `getindex`)* | 0.035 ms | 1.00 |
| 02 wealth | 0.557 ms | **0.040 ms** | 0.07 |
| 03 distance | 0.522 ms | **0.040 ms** | 0.08 |
| 04 joint | 0.614 ms | **0.042 ms** | 0.07 |

*(composable column measured with `NoGuard`, matching arm 00; with the default `ClampGuard`
it is 0.041–0.051 ms.)*

Arm 00 is the fastest engine in the repository because it uses `getindex` — by accident,
not by policy, since the guide said otherwise. That is how the defect survived.

### Multi-level indexing

```julia
# ✅ DO
int_m = inter.μ_base[season_indices] .+ inter.δ_month[month_indices]

# ❌ AVOID
int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
```

### When `view` is still correct

`view` remains the right call on **untracked data** arrays — anything passed into the model
as an argument — where it genuinely avoids a copy and there is no tape involved. The rule
above is specifically about arrays produced by `~`.

---

## 6. Keep Constants Out of Tracked Expressions

Multiplying an untracked constant *into* a tracked elementwise expression widens the kernel
ReverseDiff differentiates. Applying it as its own broadcast afterwards is faster, and the
value is identical to the last bit.

```julia
# ❌ AVOID — the decay weight rides inside the tracked kernel
@addlogprob! sum(match_weights .* (y .* η .- exp.(η) .- log_fact))

# ✅ DO — form the log-likelihood, then weight it
ll = y .* η .- exp.(η) .- log_fact
@addlogprob! sum(ll .* match_weights)
```

Measured on fold 1, same log-density, compiled tape:

| form | tape instructions | gradient |
|---|---|---|
| weight fused in | 78 | 0.061 ms |
| weight applied separately | 80 | **0.038 ms** |

Note the direction: the faster version has **more** instructions. Instruction count is a
localisation tool, not the metric (§2).

The pattern `sum(match_weights .* (...))` appears throughout
`src/models/pregame/engines/`. The same applies to masks — apply them in the same separate
broadcast as the weights, not inside the likelihood kernel:

```julia
# ✅ DO
@addlogprob! sum(ll .* match_weights .* xg_mask)
```

---

## 7. Composing a Variable Number of Terms

Sometimes the number of terms is a property of the *configuration*, not of the engine: a
model with wealth, or with wealth and distance, or with neither. The wrong answers are a
`for` loop (Rule 1) and one engine per combination (2^N files).

The right answer is a **typed tuple walked by recursion**, which Julia unrolls at compile
time into straight-line code:

```julia
# ✅ DO — the tuple's LENGTH and element TYPES are in the model's type,
#         so this is unrolled before the tape exists
@model function _cov_block(cs::Tuple, xs::Tuple, n::Int)
    head ~ to_submodel(
        DynamicPPL.prefix(_cov_term(first(cs), first(xs)), Val(covariate_name(first(cs)))),
        false)
    rest ~ to_submodel(_cov_block(Base.tail(cs), Base.tail(xs), n), false)
    return (; h = _cov_acc(head.h, rest.h), a = _cov_acc(head.a, rest.a))
end

@model function _cov_block(cs::Tuple{}, xs::Tuple{}, n::Int)
    return (; h = nothing, a = nothing)     # structural zero
end
```

```julia
# ❌ AVOID — a runtime loop over an abstractly-typed vector inside @model
for c in config.covariates          # Vector{AbstractCovariateConfig}
    η = η .+ sample_weight(c) .* design(c)
end
```

Three points make this work:

1. **The container must be a `Tuple`, not a `Vector`.** A `Vector{AbstractCovariateConfig}`
   puts the length and the element types at runtime, and the loop stays a loop. Freeze the
   vector into a tuple once, outside the model.
2. **`@generated` does not work here.** The `@model` macro rewrites the function body at
   parse time, so a generated body cannot contain `~`. Tuple recursion gets the same
   unrolling through ordinary inference.
3. **Make the empty case a structural zero, not a zero array.**

```julia
# ✅ DO — dispatch the addition away entirely; zero terms cost zero tape nodes
_cov_shift(η, ::Nothing) = η
_cov_shift(η, q)         = η .+ q

# ❌ AVOID — every covariate-free model now pays two broadcasts per gradient, forever
_cov_block(::Tuple{}, ::Tuple{}, n) = (; h = zeros(n), a = zeros(n))
```

The same idiom covers any optional term: a rate guard (`apply_guard(::NoGuard, η) = η`), an
optional pillar, an ablation. **A model that composes nothing should record the same tape as
a model that never had the option.** That is the test — see §10.2.

### Site naming inside submodels

Use `DynamicPPL.prefix(model, Val(name))` with `to_submodel(..., false)` so each term names
itself. Without the `false`, DynamicPPL adds the left-hand-side name too and the recursion
nests prefixes (`head.head.head.w`), making the chain schema depend on how many terms
happened to precede this one.

---

## 8. The Builder Pattern: Feature Layer Does the Work

The `build_turing_model` function is where all data wrangling happens. Its job is
to transform messy real-world data into pristine `Vector{Float64}` and
`Vector{Int}` arrays that the model can consume without any branching.

### Builder responsibilities

1. **Extract** raw data from the `FeatureSet` dict
2. **Cast** to concrete types (`Vector{Int}`, `Vector{Float64}`)
3. **Impute** missing values to harmless constants
4. **Build** binary masks for optional data sources
5. **Compute** time-decay weights
6. **Precompute** every data-only constant — log-factorials especially
7. **Build** score-based masks (e.g. Dixon-Coles τ masks)
8. **Validate** lengths and finiteness, loudly
9. **Return** the instantiated `@model` function with all data baked in

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

    # --- Observation vectors ---
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    # --- Time decay weights ---
    date_deltas   = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.days_half_life)

    # --- Constants: data, so they belong here, not in the tape ---
    log_fact_h = SpecialFunctions.loggamma.(Float64.(home_goals) .+ 1.0)
    log_fact_a = SpecialFunctions.loggamma.(Float64.(away_goals) .+ 1.0)

    # --- Optional data with masking ---
    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    xg_mask     = Float64.(.!isnan.(home_xg_raw))
    home_xg     = [isnan(x) ? 1.0 : Float64(x) for x in home_xg_raw]

    # --- Fail here, not three hours into a grid run ---
    n = length(home_ids)
    all(v -> length(v) == n, (away_ids, season_ids, home_goals, away_goals,
                              match_weights, xg_mask)) ||
        error("design vector length mismatch")
    all(isfinite, match_weights) || error("non-finite match weights")

    return my_engine(
        home_ids, away_ids, season_ids,
        home_goals, away_goals, match_weights,
        log_fact_h, log_fact_a, home_xg, xg_mask,
        n_teams, n_seasons, n_months, config
    )
end
```

> [!TIP]
> The length and finiteness checks are not defensive padding. A mis-aligned covariate column
> is a fold-alignment bug and a non-finite entry becomes a `-Inf` log-density; both are
> cheap to catch here and expensive to catch after an overnight grid.

---

## 9. Numerical Safety

MCMC samplers explore extreme regions of parameter space, and warm-up explores the most
extreme ones. These guards prevent `NaN`/`Inf` from crashing the sampler — but read §2's
caution first, because guards are branches.

### Clamp log-rates

```julia
log_λ = clamp.(intercept .+ attack .+ defence, -10.0, 10.0)
```

Without this, `exp(50.0) = Inf` and `Poisson(Inf)` throws.

> [!CAUTION]
> `clamp` is a value-dependent branch and a suspect in the compiled-tape divergence in §2.
> Keep it — an unguarded `exp` will end a run outright — but:
> - **Make it a component, not a line.** Different engines currently disagree about whether
>   they clamp at all, purely by accident. If it is a config with a name
>   (`ClampGuard` / `NoGuard`), the disagreement becomes visible and testable.
> - **Verify the clamp does not bind** at the draws you care about. If `max |η|` is 4 against
>   a bound of 10, the guard is inactive and cannot be distorting anything; if it is 9.9, you
>   are sampling inside the branch and the compiled tape is suspect.
>   a correctness decision, not a performance one: its cost is ~10-15% of a well-formed
>   gradient (isolated: 0.054 vs 0.049 ms; whole engine: 0.041 vs 0.035 ms on fold 1).

### Epsilon floors

```julia
λ = exp.(log_λ) .+ 1e-6
```

Prevents `Poisson(0.0)`, whose `logpdf` is undefined for non-zero observations.

### Work in log-intensity space where you can

For Poisson, do not build a distribution object at all — evaluate the density directly:

```julia
# ✅ DO — no Poisson object, no loggamma on the tape, one broadcast
ll = y .* η .- exp.(η) .- log_fact      # log_fact precomputed in the builder
@addlogprob! sum(ll .* match_weights)
```

`RobustNegativeBinomial` is ~1.9x the stdlib `NegativeBinomial` and ~6x `Poisson` for the
same job ([T002](tickets/T002-scalar-taped-likelihood.md)); if a Poisson likelihood is
defensible, the direct form is by far the cheapest.

### AD-Safe rejection

```julia
is_bad = any(isnan, λ_h) || any(isinf, λ_h)
λ_h    = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)
```

Prefer this branch-free form over an `if ... return`. The early-return version is
incompatible with Zygote and is only "safe" under ReverseDiff because it becomes dead code
when `compile=true` traces the non-NaN path — which is another way of saying it does not
work when you need it.

### Bounded transformations

For parameters that must stay in a range (like Dixon-Coles ρ), prefer a smooth map over a
clamp — it has no branch at all:

```julia
ρ = 0.3 .* tanh.(ρ_raw)  # Maps ℝ → (-0.3, 0.3), everywhere differentiable
```

---

## 10. Benchmarking and Verifying Your Model

### 10.1 Measure latency properly

```julia
using DynamicPPL, LogDensityProblems, ReverseDiff, ForwardDiff, Statistics, LinearAlgebra

model = PreGame.build_turing_model(model_config, feature_set)
vi = DynamicPPL.VarInfo(model); model(vi); θ = copy(vi[:])
lf = DynamicPPL.LogDensityFunction(model)
f  = x -> LogDensityProblems.logdensity(lf, x)

raw  = ReverseDiff.GradientTape(f, θ)
tape = ReverseDiff.compile(raw)
g    = similar(θ)

for _ in 1:50; ReverseDiff.gradient!(g, tape, θ); end          # WARM UP, discard
t = minimum([@elapsed ReverseDiff.gradient!(g, tape, θ) for _ in 1:400])

println("instructions: ", length(raw.tape))
println("gradient:     ", round(t * 1e3, digits = 4), " ms")
```

Run verbatim against the reference engine on fold 1 this prints `instructions: 80`,
`gradient: 0.0386 ms`. The §10.3 block below continues from these bindings.

> [!TIP]
> **Warm up, then take the minimum.** A cold `median` over 50 reps timed one engine at
> 0.876 ms whose warmed minimum is 0.465 ms — and in the same table reported three
> structurally identical engines at 0.496, 0.589 and 0.876 ms. Those spreads are pure
> measurement noise, and acting on them sent an earlier investigation after the wrong
> suspect twice. The minimum after warm-up is the least-noisy estimator of a deterministic
> cost.
>
> Print `length(raw.tape)` alongside it. Instruction count does not decide the winner (§2),
> but a count that scales with the row count localises the problem immediately: **for a
> correctly vectorised engine the tape size is independent of how many matches you fit.**

| Metric | Good | Bad |
|:-------|:-----|:----|
| Gradient (720 matches, ~50 params) | < 0.1 ms | > 0.5 ms |
| Tape instructions | O(10s), flat in row count | O(N) per observation |
| Tape compilation | < 30 s | > 2 min |

### 10.2 Verify composition is free

If your engine has optional terms (§7), assert that the option costs nothing when unused:

```julia
@assert length(ReverseDiff.GradientTape(f_without_term, θ₀).tape) ==
        length(baseline_instruction_count)
```

### 10.3 Verify the tape, do not trust it

Latency is worthless if the gradient is wrong. Run all four, every time:

```julia
g_tape    = (h = similar(θ); ReverseDiff.gradient!(h, tape, θ); h)
g_fresh   = ReverseDiff.gradient(f, θ)
g_forward = ForwardDiff.gradient(f, θ)

relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)
@assert relerr(g_fresh, g_tape)    <= 1e-8   # compiled == fresh
@assert relerr(g_tape,  g_forward) <= 1e-6   # ReverseDiff == ForwardDiff

# The one that catches a smuggled branch: the SAME compiled tape at OTHER points
for δ in (0.001, -0.002, 0.003)
    θp = θ .+ δ .* sin.(collect(eachindex(θ)))
    gp = similar(θp); ReverseDiff.gradient!(gp, tape, θp)
    @assert relerr(ReverseDiff.gradient(f, θp), gp) <= 1e-8
end
```

> [!WARNING]
> The perturbed-point probe above uses **small** displacements and both a correct and a
> suspect engine pass it. Per §2, at ~0.8 units per coordinate they both fail. Small probes
> confirm the tape is not obviously broken; they do **not** license `compile=true` across
> warm-up. Until [T002 §(c)](tickets/T002-scalar-taped-likelihood.md) is closed, extend the
> probe to the scale your sampler actually explores and look at what changes.

### 10.4 Verify the log-density against a second implementation

The strongest check available, and cheap: write the log-joint a second time as plain Julia,
from the equations rather than from the engine, and compare at several prior draws.

```julia
@assert abs(f(θ) - reference_logjoint(model, params_from(vi), data)) <= 1e-9
```

`current_development/scottish_lower/05_composable_count_builder/l04_equations.jl` is the
pattern. It also gives you a free regression test on the engine: two implementations of one
density that agree to the last bit are very unlikely to be wrong in the same way.

---

## 11. Reference Implementations

**For the hot path — indexing, likelihood shape, optional terms:**
`current_development/scottish_lower/05_composable_count_builder/l03_engine.jl`.
0.035–0.051 ms on fold 1 (guard off / on). Demonstrates `getindex` selection (§5), unfused weights (§6), the
typed-tuple unroll and structural zeros (§7), and a dispatched rate guard (§9).

**For masking patterns:**
[outfield_xg_dixon_coles.jl](file:///home/james/bet_project/BayesianFootball/src/models/pregame/engines/player_level/time_decay/outfield_xg_dixon_coles.jl)
still has the clearest xG / market / Dixon-Coles τ masks in the repository.

> [!WARNING]
> That file — like all 28 engines in `src/models/pregame/engines/` — uses `view` and fused
> weights, because it was written to the previous version of §5 and §6. **Copy its masks,
> not its indexing.** Fixing them is [T002](tickets/T002-scalar-taped-likelihood.md).

---

## 12. AD Backend Comparison

| Feature | ReverseDiff.jl | Zygote.jl | Mooncake.jl | ForwardDiff.jl |
|:--------|:--------------|:----------|:------------|:---------------|
| **Type** | Tape-based tracing | Source-to-source IR transform | Source-to-source IR transform | Dual numbers |
| **Scaling** | O(1) in #params (reverse mode)| O(1) in #params | O(1) in #params | O(N) in #params |
| **Compiled tape** | ✅ `compile=true` | N/A | N/A (prepares caches) | N/A |
| **Dynamic control flow** | ❌ Silent errors with `compile=true` | ❌ Fails on early `return` | ✅ Fully supported | ✅ Fully supported |
| **Array mutation** | ❌ Errors or silent failures | ❌ Strongly unsupported | ✅ First-class support | ⚠️ Limited |
| **Sweet spot** | Static, vectorised models | Purely functional, highly vectorised models | Complex Julia code with mutations and branching | Few parameters (< 100) |

> [!TIP]
> The system hardcodes `AutoReverseDiff(compile=true)` because our models are static.
> ForwardDiff is the **oracle**, not a candidate: it is far too slow for sampling at our
> parameter counts but is the reference every check in §10.3 compares against.
> `AutoMooncake()` is the credible modern alternative if the §2 branch hazard proves real —
> it supports control flow and mutation, at some throughput cost.

---

## 13. Checklist

Before submitting a new model engine:

**Shape**
- [ ] No `for` loops inside `@model` (variable term counts → §7 tuple unroll)
- [ ] No value-dependent branches inside `@model`; type dispatch instead
- [ ] No `isnan()` / `findall()` inside `@model`
- [ ] No array mutation inside `@model`
- [ ] Optional terms are structural zeros (`::Nothing`), not zero arrays

**Indexing and arithmetic**
- [ ] Parameter selection uses **`A[idx]`**, not `view(A, idx)` — §5
- [ ] Weights and masks applied in their **own** broadcast, not fused into the likelihood — §6
- [ ] Log-factorials and every other data-only constant precomputed in the builder
- [ ] Poisson likelihoods evaluated directly in log-intensity space

**Data**
- [ ] All optional data uses binary masks; linear covariates impute to `0.0` (and say so)
- [ ] NaN imputed to distribution-safe constants in the builder
- [ ] Builder validates vector lengths and finiteness and errors loudly

**Numerics**
- [ ] Rate guard present, named, and shown not to bind at the draws you use
- [ ] `1e-6` floor on rate parameters where the density needs it
- [ ] Rejection guards branch-free (`ifelse`), no early `return`

**Verification**
- [ ] Log-density matches an independent re-derivation to ≤ 1e-9 — §10.4
- [ ] Compiled tape == fresh ReverseDiff == ForwardDiff — §10.3
- [ ] Compiled tape still correct at perturbed points, at the scale the sampler explores
- [ ] Tape instruction count is flat in row count
- [ ] Warm-up-then-minimum gradient < 0.1 ms for ~700 rows / ~50 params — §10.1
