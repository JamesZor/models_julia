# Composable Count Model Builder — architectural specification

| | |
|---|---|
| **Status** | prototyped and verified on Scottish Lower; not yet graduated to `src/` |
| **Prototype** | `current_development/scottish_lower/05_composable_count_builder/` |
| **Verified on** | ScottishLower 56+57, fold 1 (720 fitted matches, 23 teams), 64/64 gate rows |
| **Written** | 2026-08-28 |

---

## 1. The problem

`src/models/pregame/engines/` holds 28 engine files. `current_development/scottish_lower/`
holds five more across arms 00-04. Each one costs four things:

1. a struct,
2. a Turing `@model`,
3. a `Features.required_features` method,
4. an `extract_parameters` method.

They are not 33 different models. Arms 02, 03 and 04 are the clearest case:
`02_poisson_wealth/l00_feature_poisson.jl` contains `_engw`, `_engd` and `_engj`, three
`@model` functions differing only in whether the linear predictor carries a wealth term, a
distance term, or both. Each is one line long, and all three lines are the same line.

The cost is combinatorial in the wrong variable. With **N** covariates (wealth, distance,
rest days, red cards, officiating) and **M** observation families (Poisson, negative
binomial, Dixon-Coles, copula), the current architecture needs one engine per **subset**,
so N covariates alone cost 2^N engines. Adding the sixth covariate to a five-covariate
library is 32 new files.

There is a second, quieter cost. Because each engine is written separately, they drift:
arm 00's engine has no `clamp` on the log-rates while arms 02/03/04 do; arm 00 selects team
effects with `getindex` while arms 02/03/04 use `view`. Neither difference was a decision.
The `view` one costs a **15x** slowdown in gradient evaluation (see
[T002](../tickets/T002-scalar-taped-likelihood.md)), and it survived precisely because
nobody compares two engines line by line.

### What is actually shared

Every count engine in the repository computes the same two numbers:

```
η_h = μ_{s(i)} + δ_{m(i)} + γ_{h(i)} + α_{h(i)} + β_{a(i)} + Σ_k q_k,h(i)
η_a = μ_{s(i)} + δ_{m(i)}            + α_{a(i)} + β_{h(i)} + Σ_k q_k,a(i)
```

and then hands them to a count density. Poisson, negative binomial, Dixon-Coles and the
Frank copula differ **only in that last step**. Splitting them into separate files means
every covariate added later must be added four times.

---

## 2. The design in one paragraph

One mutable, order-remembering **builder**; one generic `add!` dispatched on abstract
component type; one strict `build()` that validates and freezes; one Turing `@model` that
walks a **typed tuple** of covariates unrolled at compile time; and `required_features`,
the chain schema, the parameter count, and `extract_parameters` all **derived** from the
assembled components rather than written per model.

```julia
b = CountModelBuilder(:scottish_count)
add!(b, GlobalInterception(), TimeDecayDynamics(), GlobalHomeAdvantage())
add!(b, WealthCovariate(), DistanceCovariate())
model = build(b)
```

---

## 3. Type hierarchy

### 3.1 Two phases, deliberately separate

| | `CountModelBuilder` | `PoissonCountModel` / `NegBinCountModel` |
|---|---|---|
| mutability | mutable | immutable |
| typing | abstract slots, `Vector{AbstractCovariateConfig}` | fully concrete, `covariates::C<:Tuple` |
| validity | invalid by default | valid by construction |
| purpose | accumulate, remember order, explain itself | be sampled from |

`build()` is the **only** type-unstable boundary in the design. `Tuple(b.covariates)` turns
an abstract vector into a concrete tuple type; everything downstream — the engine, the
ReverseDiff tape, extraction — is concrete and static. That is what lets the covariate walk
unroll (§5).

### 3.2 Component families

Existing `src` families are reused unchanged:

| family | abstract type | reused from |
|---|---|---|
| interception | `AbstractInterceptionConfig` | `src/models/pregame/components/interception.jl` |
| dynamics | `AbstractDynamicsConfig` | `.../components/dynamics/` |
| home advantage | `AbstractHomeAdvantageConfig` | `.../components/home_advantage.jl` |
| dispersion | `AbstractDispersionConfig` | `.../components/dispersion.jl` |

Three families are new, and their absence is what forced the engine explosion:

| family | abstract type | why it did not exist |
|---|---|---|
| covariate | `AbstractCovariateConfig` | a `w · x` term had nowhere to live, so it lived in an engine |
| observation | `AbstractObservationConfig` | the count density was implicit in the choice of engine file |
| rate guard | `AbstractRateGuard` | `clamp` was a line, not a decision, so arms disagreed about it silently |

### 3.3 Why two model structs, not one

A single struct parameterised on the observation would be tidier and is **wrong**.
`src/predictions/score_computation/` dispatches the score grid on the model's supertype:
`AbstractPoissonModel` reaches the double-Poisson grid, `AbstractNegBinModel` the
negative-binomial one. A Julia type's supertype cannot depend on its type parameters, so the
observation family must be carried by the struct identity.

This is the real reason a "one model type to rule them all" refactor fails, and the
resolution is the point of the whole design: **one engine, one struct per prediction
family** — four structs, not twenty-eight engines. Getting this wrong does not error; it
silently prices with the wrong grid, which is why `build()` takes the supertype **from the
observation config** rather than letting the caller choose it.

---

## 4. Interface contracts

### 4.1 Covariate — six methods

| method | returns | used by |
|---|---|---|
| `covariate_name(c)` | `Symbol` | chain site prefix (`:wealth` → `wealth.w`) |
| `covariate_role(c)` | `AbstractCovariateRole` | how it reaches the two sides |
| `covariate_prior(c)` | `UnivariateDistribution` | prior on the scalar weight |
| `covariate_features(c)` | `Vector{<:AbstractFeatureConfig}` | `required_features` derivation |
| `covariate_column(c, fs)` | `Vector{Float64}` | fit-time design vector |
| `covariate_oos(c, fs, df)` | `Vector{Float64}` | prediction-time design vector |

There are deliberately **no working fallbacks**: the abstract methods throw with the
missing method's name, and `build()` checks for a specialised method rather than
`hasmethod` (which every covariate trivially satisfies via the throwing fallback).

Two roles, dispatched not branched:

```julia
covariate_sides(::SupremacyRole, q) = (q, -q)   # moves the result, holds the total
covariate_sides(::LevelRole,     q) = (q,  q)   # moves the total, holds the result
```

Wealth and travel distance are both supremacy covariates. A pitch or referee effect would
be a level covariate. Under the current architecture that distinction would be a fifth and
sixth engine.

**Missingness is a zero, not a mask.** For a linear term `w · x`, imputing an absent
covariate to `0.0` *is* a binary mask, because `w · 0 = 0`. The AD guide's masking rule is
satisfied without a second vector. This holds only because the term is linear in `x`; a
non-linear covariate would need a real mask, and that is a contract change, not a config
change.

`covariate_oos` is separate from `covariate_column` because extraction has no `DataStore`
argument. Wealth reaches out-of-sample fixtures through a point-in-time bridge stashed at
feature-build time (no valuation stamped after its own kickoff is ever used); distance is a
static function of the two grounds and is simply recomputed.

### 4.2 Observation — four traits

| trait | purpose |
|---|---|
| `observation_family(o)` | which score grid the built model must route to |
| `observation_wired(o)` | is there an `_observe` method, or is this only declared? |
| `observation_prefixes(o)` | chain-site prefixes it owns, for the collision check |
| `observation_gap(o)` | what still has to be built, quoted back by `build()` |

`observation_wired` is a **value-level** predicate, not just a type-level one:
`NegativeBinomialObservation` is wired for `GlobalDispersion` and `HomeAwayDispersion` (which
return a scalar `(h, a)` pair the engine broadcasts) but not for
`AdvancedVolatilityDispersion`, whose per-team and per-month components need reassembly per
match. The `src` NegBin engine does that reassembly with a `hasproperty` branch **inside
`@model`** — a construct the AD guide forbids and this engine will not inherit. Wiring it
means a dispatching `_reconstruct_dispersion` submodel, not a branch.

### 4.3 What the assembled model derives about itself

None of these are written per model:

```julia
Features.required_features(m)      # structural block + Σ covariate_features
cb_varinfo_sites(m)                # grouped VarInfo site names, in θ order
cb_chain_columns(m, n_teams)       # expanded Chains columns
cb_parameter_count(m, n_teams)     # length(θ)
extract_parameters(m, df, fs, ch)  # walks the same tuple, reads <name>.w
```

`cb_varinfo_sites` and `cb_chain_columns` are kept **separate on purpose**: `VarInfo` groups
vector sites and `Chains` expands them, and comparing either representation against the
other is how a schema error hides. The prototype checks both against what the engine
actually samples.

---

## 5. Compilation strategy

The requirement is stated in `docs/turing_ad_performance_guide.md`: a compiled
`ReverseDiff.GradientTape` with no scalar loops, no conditionals, and no dynamic dispatch
inside `@model`. A composable engine has to satisfy it while iterating over a set of
covariates whose size is not known when the engine is written.

### 5.1 The typed tuple walk

The covariate set lives on the model as a `Tuple`, so its length and element types are in
the model's type. The engine recurses through `Base.tail`, which Julia unrolls at compile
time into straight-line code:

```julia
@model function _cov_block(cs::Tuple, xs::Tuple, n::Int)
    head ~ to_submodel(
        DynamicPPL.prefix(_cov_term(first(cs), first(xs)), Val(covariate_name(first(cs)))),
        false)
    rest ~ to_submodel(_cov_block(Base.tail(cs), Base.tail(xs), n), false)
    return (; h = _cov_acc(head.h, rest.h), a = _cov_acc(head.a, rest.a))
end

@model function _cov_block(cs::Tuple{}, xs::Tuple{}, n::Int)
    return (; h = nothing, a = nothing)
end
```

No runtime loop, no runtime branch, no `Vector{AbstractCovariateConfig}` dispatch. The tape
sees exactly the instruction sequence a hand-written engine emits — which §7.1 verifies as
a bit-identical log-density.

`@generated` was considered and rejected: the `@model` macro rewrites the function body at
parse time, so a generated body cannot contain `~`. Tuple recursion achieves the same
unrolling through ordinary inference.

### 5.2 Site naming

`DynamicPPL.prefix(model, Val(name))` supplies the prefix from the covariate's own
`covariate_name`, and `to_submodel(..., false)` suppresses DynamicPPL's automatic
left-hand-side prefixing. Without the `false`, the recursion would nest names
(`head.head.head.w`) and the chain schema would depend on how many covariates happened to
precede a given one. With it, a covariate's site is `wealth.w` wherever it sits in the
build order.

The alternative — one `w ~ arraydist(priors)` site giving `w[1]`, `w[2]` — was rejected:
positional names make the chain unreadable and make extraction depend on build order in a
way nothing checks.

### 5.3 Structural zeros cost nothing

The empty covariate tuple returns `nothing`, and

```julia
_cov_shift(η, ::Nothing) = η
_cov_shift(η, q)         = η .+ q
```

dispatches the addition away entirely. A model with no covariates records a tape
**identical** to the baseline engine's — composition costs nothing when you compose
nothing. Returning `zeros(n)` instead would have added two broadcast nodes per gradient,
forever, to every model that uses no covariates. The same idiom is used for the rate guard
(`apply_guard(::NoGuard, η) = η`).

### 5.4 Measured cost

Compiled-tape gradient, fold 1, median of 50 reps:

| arm | composable | composable, `NoGuard` | hand-written | ratio |
|---|---|---|---|---|
| 00 baseline | 0.041 ms | 0.034 ms | 0.034 ms | 1.00 at equal guard |
| 02 wealth | 0.044 ms | 0.039 ms | 0.521 ms | **0.08** |
| 03 distance | 0.048 ms | 0.039 ms | 0.583 ms | **0.08** |
| 04 joint | 0.047 ms | 0.042 ms | 0.471 ms | **0.10** |

Against the guide's 0.65 ms target the composable engine has ~13x headroom. It is **~11x
faster** than the three hand-written engines it replaces, and **exactly equal** to arm 00
once the rate guard is matched (arm 00 has none; that difference is the whole residual).

Two departures from the AD guide, both made on measurement, both now corroborated into
[T002](../tickets/T002-scalar-taped-likelihood.md):

- **`A[idx]`, not `view(A, idx)`.** Rule 4 has it backwards for this ReverseDiff version:
  `view` on a `TrackedArray` yields a `SubArray` of `TrackedReal` that the tape walks
  element by element. 5x, same value to the last bit.
- **Apply the decay weight in its own broadcast.** `sum(ll .* w)` rather than
  `sum(w .* (...))` — fusing an untracked constant into a tracked elementwise expression
  widens the kernel ReverseDiff forward-optimises. 1.6x, same value to the last bit.

---

## 6. Validation rules in `build()`

`validate(builder)` returns a table and never throws, so a runner can print every problem at
once; `build()` throws on the first failing row. Every rule below exists because breaking it
produces a model that **runs**. None is a type error.

| # | rule | what it prevents |
|---|---|---|
| R1-R3 | exactly one interception / dynamics / home advantage | a model that silently acquired a default nobody chose |
| R4 | dynamics exposes `days_half_life` | the likelihood decay weights come from it |
| R5 | observation is wired | a declared-but-unbuilt density quietly doing nothing |
| R5b | rate guard declared | arms 00 and 02-04 disagreeing about `clamp` by accident |
| R6 | covariate names unique | two `:wealth` covariates both sampling `wealth.w`; extraction reads one |
| R7 | no reserved site prefixes | a covariate named `:dyn` shadowing the dynamics block |
| R8 | no observation site clash | a covariate colliding with `disp.*` |
| R9 | covariate contract complete (6/6) | a missing method surfacing three hours into a grid run |
| R10 | covariate priors univariate | a weight that is not a scalar |
| R11 | covariates declare features | an all-zero design column reporting its prior as a posterior |

Two further invariants are enforced at `add!` rather than `build()`:

- **Structural slots refuse silent overwrite.** A second dynamics config is a mistake far
  more often than a revision, and a builder that keeps the last one turns the mistake into a
  result. `replace!` exists to mean it, and records the replacement in the provenance.
- **Unknown components are refused loudly**, with the list of accepted families and the
  instruction that adding one is a single `add!` method.

All eleven rules and all five refusals are exercised in `r01_demo.jl` §4.

---

## 7. Verification

`r01_demo.jl` reports **64/64** on fold 1, deterministically, in seconds.

### 7.1 The density proof (§7 of the runner)

For each of the four arms, in increasing order of strength:

| check | result |
|---|---|
| derived VarInfo sites == what the engine samples | exact, all four |
| derived parameter count == `length(θ)` | 50 / 51 / 51 / 52 |
| engine == independent re-derivation (`l04_equations.jl`) | max \|Δ\| ≤ 2.3e-13 |
| rate guard never binds at the compared draws | max \|η\| = 2.08 / 1.77 / 1.79 / 3.95 against 10.0 |
| θ layout identical to the hand-written arm | same sites, same order |
| **log density == hand-written arm** | **max \|Δ\| = 0.000e+00, bit-identical** |

The last row is the claim. It is obtained by drawing θ from the composable model and
evaluating **both** log-densities at that same θ — which is legitimate only because the row
above it establishes that a θ means the same thing to both. Drawing separately from each and
comparing summaries would show the densities are similar; this shows they are the same
function.

The clamp row matters because arm 00's engine has **no** guard while the composable engine
has one by default. Those are the same function only where the guard does not bind, so the
headroom is measured rather than assumed.

### 7.2 Extraction

`extract_parameters` is fully derived for the composable model and hand-written for the
arm. Both are handed the **same posterior draws** — the composable model's chain with its
covariate sites renamed to the arm's convention — so Monte-Carlo noise is removed entirely
and any difference is arithmetic:

> max relative difference in posterior-mean λ = **0.000e+00** over 20 OOS fixtures × 2 sides,
> all four arms.

### 7.3 What is deliberately not claimed

MCMC posterior comparison is implemented (`cb_nuts`, `cb_posterior_parity`,
`cb_calibrated_row`, `cb_convergence_row`) and **off by default**. Two models with a
bit-identical log-density *are* the same posterior; sampling cannot strengthen that, and a
Monte-Carlo comparison can only ever conclude "within noise". The machinery is kept because
it demonstrates the claim end to end through a real sampler, and because it surfaced
something worth knowing (below), but it costs 96 NUTS chains and belongs on the server.

**An observation from that work, recorded rather than acted on.** On the bit-identical
density, the composable engine's chains converged cleanly (R-hat ≤ 1.004, min ESS ≥ 1981)
while `_engw`'s did not (R-hat 1.12-1.32, min ESS 30-103 on `dyn.raw_a[13]`), reproducibly
across three warm-up budgets and two seeds. Since the target is provably identical, this is
a property of the two implementations' *trajectories*, not their posteriors — the leading
suspect is the compiled-tape divergence recorded in
[T002 §(c)](../tickets/T002-scalar-taped-likelihood.md). It is not diagnosed here and should
not be assumed benign.

---

## 8. Scope and migration

### 8.1 What exists

`current_development/scottish_lower/05_composable_count_builder/`, ~1,100 lines:

| file | contents |
|---|---|
| `l01_components.jl` | covariate contract + roles, wealth and distance covariates, rate guards, observation layer |
| `l02_builder.jl` | `CountModelBuilder`, `add!`, `add`, `replace!`, `validate`, `build`, the two model structs, derived schema |
| `l03_engine.jl` | the single `@model`, the unrolled covariate block, the observation block, `build_turing_model`, `extract_parameters` |
| `l04_equations.jl` | independent re-derivation of the log-joint, generic over covariates |
| `l05_parity.jl` | the parity harness |
| `r01_demo.jl` | the proof — 64/64 |

It reproduces five hand-written engines (`_engw`, `_engd`, `_engj`, arm 00's, and the `src`
NegBin path) and spans configurations none of them reach, at a `build()` call each: level-role
covariates, `StaticZeroDynamics` ablations, NegBin with covariates, hierarchical monthly
interception with team home advantage.

### 8.2 Effort estimate for graduation

| phase | scope | estimate |
|---|---|---|
| 1 | Move the three new component families into `src/models/pregame/components/` (`covariates.jl`, `observations.jl`, `guards.jl`); move builder + engine into `src/models/pregame/builder/`. Covariates stay Scottish-specific until a second segment needs them. | 1-2 days |
| 2 | Wire `DixonColesCorrelation` and `FrankCopulaCorrelation` (§8.4), plus `AdvancedVolatilityDispersion` via a dispatching reconstruction submodel. | 2-3 days |
| 3 | Port existing engines arm by arm, each with the `r01_demo.jl` parity harness as the acceptance test: build the composable equivalent, assert bit-identical log-density, then retire the file. | 0.5-1 day per engine family |
| 4 | Fix [T002](../tickets/T002-scalar-taped-likelihood.md) in the shared engine once, rather than 28 times. | folded into phase 1 |

Phase 3 is the long pole and is **incremental by construction**: each engine is retired only
after its replacement is proven bit-identical to it, so the port can stop at any point with
the repository in a working state.

### 8.3 Backward compatibility

The composable models subtype the same `TypesInterfaces` abstract types and extend the same
generic functions (`Features.required_features`, `PreGame.build_turing_model`,
`PreGame.extract_parameters`), so `Experiments`, `Training`, `Samplers`, `Predictions`,
`Signals` and `BackTesting` need **no changes** — the prototype already runs through
`Features.create_features` and the protocol's fold machinery unmodified.

Existing engines keep working throughout. Nothing is deleted until its composable
replacement is proven equal to it at the level of the log-density, and the harness that
proves it is part of this prototype.

The one visible break is **chain site names**: `wealth.w` instead of `w_wealth`. This affects
saved chains and any code reading a weight by name. Options, in order of preference:

1. accept it and treat pre-migration artifacts as pre-migration (the protocol already hashes
   configuration into artifact identity, so they will not be silently mixed);
2. keep an alias map in the extractor for a deprecation window;
3. allow a covariate to override its own site name, which reintroduces exactly the naming
   drift the derivation removes. **Not recommended.**

### 8.4 Declared but not built

`DixonColesCorrelation` and `FrankCopulaCorrelation` are in the type hierarchy, accepted by
`add!`, reported by `validate`, and **refused by `build()`** with the reason. This is the
strict-validation phase doing its job rather than a gap left open.

Wiring Dixon-Coles is one `_observe` method: the τ low-score masks are pre-computable in
`cb_design` exactly as the AD guide's §4 describes, plus a `DixonColesRates` extraction
carrying ρ and a `DixonColesCountModel <: AbstractDixonColesModel` struct. The copula is the
same shape over `src/models/pregame/components/copula_densities.jl`.

Neither was built here because there is no hand-written Scottish arm to check the result
against, and this prototype's entire value is that every claim it makes is checked against
something. An unverified likelihood is worth less than an honest gap.

---

## 9. Recommendations

1. **Adopt the pattern.** The covariate layer alone converts 2^N engine files into N structs,
   and it is verified bit-identical to what it replaces.
2. **Take [T002](../tickets/T002-scalar-taped-likelihood.md) first, or with phase 1.** The
   `view` defect is an 11x cost across 28 engines, and fixing it inside a shared engine is
   one edit instead of 28. Its part (c) — the compiled-tape divergence outside the typical
   set — should be isolated before anyone trusts a warm-up gradient.
3. **Correct `docs/turing_ad_performance_guide.md` Rule 4** when T002 lands. It is currently
   advising the slow path, and the engines follow it faithfully.
4. **Do not merge the observation families into one struct.** The supertype carries the score
   grid; §3.3.
5. **Keep `build()` strict.** Every refusal in §6 is a silent-wrong-answer error, and the
   refusals are cheaper than the gates that would otherwise have to catch them.
