# Julia Coding Context for AI Agents — `BayesianFootball.jl`

> **What this is.** A system-prompt-style context document for any AI agent writing or
> modifying Julia in this repository. Sections are XML-tagged so they can be lifted whole
> into an agent's system prompt, injected selectively, or read top to bottom by a human.
>
> **What this is not.** Not a Julia tutorial, and not a replacement for
> [`docs/turing_ad_performance_guide.md`](../turing_ad_performance_guide.md) (the `@model`
> hot path) or [`docs/prototype_runner_style_guide.md`](../prototype_runner_style_guide.md)
> (how a runner must read). This covers the *language* — the traps that produce wrong
> answers, not just slow ones.
>
> Every rule below is either sourced from the Julia manual (linked in `<references>`) or
> from a defect this repository actually hit, marked **[repo]** with the ticket.
>
> Last verified against the toolchain in `<environment>`: 2026-08-28.

---

<environment>

## Pinned toolchain — check before assuming an API

| | version |
|---|---|
| Julia | 1.12.6 |
| Turing | 0.41.4 |
| DynamicPPL | 0.38.10 |
| AbstractPPL | 0.13.6 |
| ReverseDiff | 1.17.0 |
| MCMCChains | 7.7.0 |
| Distributions | 0.25.127 |

Turing and DynamicPPL move fast and break APIs between minor versions. **Do not write
Turing code from memory.** Verify the signature in the installed package before using it:

```julia
methods(DynamicPPL.to_submodel)          # what arities actually exist here
names(DynamicPPL; all = true)            # what is actually exported
isdefined(MCMCChains, :mcse)             # does this helper exist in THIS version
@doc DynamicPPL.prefix
```

Hosts: laptop at `/home/james/bet_project/BayesianFootball`, compute node `mcmc-beast` at
`/root/BayesianFootball`. See [`AGENTS.md`](../../AGENTS.md) for topology, thread pinning,
and the tmux/REPL protocol.

</environment>

---

<correctness_traps>

## 1. Language traps that produce wrong answers

These are ordered by how much damage they do here, not by how exotic they are.

### 1.1 Assignment does not copy — mutation is visible to the caller

```julia
# ❌ the name promises a new value; the body edits the caller's
function normalise(df)
    df.λ ./= sum(df.λ)         # caller's DataFrame is now changed
    return df
end

# ✅ same behaviour, honestly named — the `!` is the caller's only warning
function normalise!(df)
    df.λ ./= sum(df.λ)
    return df
end

# ✅ or leave the argument alone
normalise(df) = transform(df, :λ => (v -> v ./ sum(v)) => :λ)
```

Julia values are passed by reference; `A = B` aliases. This is the single most common
source of "the feature set changed between folds" bugs. **Any function that mutates its
argument must end in `!`** and must be the only thing in the pipeline that does.

**[repo]** The `DataStore` and `FeatureSet` are passed everywhere. A function that mutates
one is a leakage bug waiting to happen — see
[`docs/guides/feature_validation_methodology.md`](feature_validation_methodology.md).

### 1.2 1-based, inclusive-on-both-ends indexing

`a[2:3]` is two elements. `a[1:end]` is everything. There is no `a[1:]`. Off-by-one against
a Python reference implementation is the classic way to shift an entire time series by one
match week.

### 1.3 `missing` vs `nothing` vs `NaN` — three different things

| value | means | propagates? | use for |
|---|---|---|---|
| `missing` | data exists but is unknown | yes, poisons arithmetic and comparisons | raw DB columns only |
| `nothing` | there is no value | no, it's just a singleton | absent config, `get(d, k, nothing)` |
| `NaN` | a float computation failed or is a sentinel | yes, silently | never carry this deliberately |

```julia
# ❌ `missing == missing` is `missing`, not `true`; this branch is neither taken nor skipped
if x == missing

# ✅
if ismissing(x)

# ❌ a Vector{Union{Missing,Float64}} reaching an @model is an AD-safety violation
# ✅ coalesce in the feature layer, then assert the eltype is concrete
v = Vector{Float64}(coalesce.(raw, 0.0))
```

**Never let `missing` past the feature layer.** The `@model` contract is pure `Float64` /
`Int`. Type purity is a gate in the Scottish protocol precisely because this leaks.

### 1.4 Julia 1.12: redefining a struct silently orphans its methods

New in 1.12 — bindings participate in world age, so `struct` and `const` redefinition is now
*allowed* rather than an error. That makes a new failure mode:

> Redefine a struct and the methods defined on the **old** definition do not apply to the
> **new** one. No error. Old objects still exist, typed by the old definition.

```julia
# ❌ [repo] T006 — a guard that tests a name the file never defines,
#    so the include always fires and the structs are redefined mid-session
if !isdefined(@__MODULE__, :SLFeaturePoissonModel)      # never defined anywhere
    include("../02_poisson_wealth/l00_feature_poisson.jl")
end

# ✅ guard on a name the file genuinely defines — prefer the abstract type,
#    since its absence is the only reliable signal the file did not load
if !isdefined(@__MODULE__, :AbstractSLFPModel)
    include("../02_poisson_wealth/l00_feature_poisson.jl")
end
```

See [`docs/tickets/T006`](../tickets/T006-scottish-lower-arm-include-guards.md). Symptom to
watch for: `MethodError` on a function you can see is defined, or
`TypeA === TypeA` returning `false`.

### 1.5 `using` two packages that export the same name is a landmine

```julia
# ❌ [repo] DataFrames and DynamicPPL both export `subset`.
#    Works in a session with only DataFrames; dies in one that also loads DynamicPPL.
using DynamicPPL
selected = subset(ds.matches, :match_id => ByRow(∈(ids)))
# UndefVarError: `subset` not defined in `Main`

# ✅ in a LOADER that others will include: qualify, or import rather than using
import DynamicPPL                       # everything used is DynamicPPL.x anyway
selected = DataFrames.subset(ds.matches, :match_id => ByRow(∈(ids)))
```

The failure is *conditional on what else is loaded*, so it passes every test that runs the
file alone. Known collisions in this stack: `subset`, `select`, `transform`, `combine`
(DataFrames vs DynamicPPL/Turing), `params` (Distributions vs Turing), `sample`
(StatsBase vs AbstractMCMC).

**Rule:** in `lXX_*.jl` loaders — files other people will `include` into sessions you
cannot see — prefer `import SomePackage` plus qualified calls over `using SomePackage`.
Runners may `using` freely; they are the leaf of the include graph.

### 1.6 Integer division and literal types

`3/2 == 1.5`, not `1`. `/` on two `Int`s always returns `Float64`. Use `div(a, b)` or
`a ÷ b` for truncating integer division, `cld` / `fld` for ceiling / floor.

`2^-1` returns `0.5::Float64` — it does not throw and it does not return an `Int`. So an
integer-typed expression can become `Float64` without any explicit conversion, which is a
type instability (§2.1) rather than an error. **Verified on 1.12.6**; older Julia threw a
`DomainError` here, which is a good illustration of why `<environment>` says not to write
this from memory.

### 1.7 Loop and comprehension scope

Most blocks introduce a new local scope. A variable first assigned inside a `for` body does
not survive the loop. Pre-declare it outside, or restructure — do not reach for `global`.

</correctness_traps>

---

<performance>

## 2. Performance rules

Sourced from the Julia manual's performance tips; the `@model` hot path has its own,
partly *contradictory*, rules — see `<probabilistic_programming>`.

### 2.1 Type stability is the whole game

A function whose return type depends on argument *values* rather than *types* forces runtime
dispatch and blocks every downstream optimisation.

```julia
# ❌ returns Int or Float64 depending on the VALUE of x, and on the eltype of xs
function total(xs)
    acc = 0                      # always Int, whatever xs holds
    for x in xs
        acc += x > 0 ? x : 0.0   # Union{Int,Float64} escapes the branch
    end
    return acc
end

# ✅ the accumulator takes its type from the data; both branches agree
function total(xs)
    acc = zero(eltype(xs))
    for x in xs
        acc += max(x, zero(x))
    end
    return acc
end
```

Diagnose with `@code_warntype total(xs)` and look for red / `Union` / `Any` on the return
type or any intermediate. `@inferred` from `Test` turns it into an assertion you can put in
the test suite.

### 2.2 Concrete types in structs and containers

```julia
# ❌ abstract field: pointer-chasing, no specialisation
struct Config; prior::Distribution; end

# ✅ parametric: one concrete specialisation per instantiation
struct Config{D<:Distribution}; prior::D; end
```

Same for containers: `Vector{Float64}`, never `Vector{Real}` or bare `Array`. A
`Vector{AbstractCovariateConfig}` is a legitimate *builder* type and an illegitimate
*hot-path* type — freeze it to a `Tuple` before it reaches anything that runs per iteration.

### 2.3 No untyped globals in hot code

`const` them, or pass them as arguments. An untyped global can change type at any time, so
the compiler assumes `Any`.

### 2.4 Column-major memory

Julia arrays are stored column by column. Iterate so the **first** index varies fastest;
`A[:, i]` is contiguous, `A[i, :]` is a strided gather.

### 2.5 Function barriers

If a function must do type-uncertain setup, put the loop in a **separate inner function**.
Julia specialises at function boundaries, so the inner loop compiles with concrete types even
when the outer one cannot.

### 2.6 Preallocate and write in place — outside `@model`

```julia
out = Matrix{Float64}(undef, n, 2)
out[:, 1] .= x        # in-place, no allocation
```

**In-place mutation is forbidden inside `@model`** — it breaks tape tracking. This rule
applies to extraction, feature building, and backtesting only.

### 2.7 `@views` for slices — outside `@model`

`a[1:5]` copies; `view(a, 1:5)` / `@views` does not. **Inverted inside `@model`** —
see §3.2. Know which layer you are in.

### 2.8 `@inbounds` / `@simd` are last resorts

`@inbounds` on a wrong index is a segfault or silent corruption, not an error. Do not add
either without a benchmark showing it matters and a bounds argument that is obviously true.

</performance>

---

<probabilistic_programming>

## 3. Turing / DynamicPPL

**Read [`docs/turing_ad_performance_guide.md`](../turing_ad_performance_guide.md) before
writing any `@model`.** This section is the summary and the API facts.

### 3.1 The two-layer contract

`build_turing_model` does *all* conditional logic, casting, imputation, masking, weighting
and precomputation, and hands the `@model` pure `Vector{Float64}` / `Vector{Int}`. Inside
`@model`: sampling statements, broadcast arithmetic, indexed selection, `sum`,
`@addlogprob!`. Nothing else. No `for`, no `if` on values, no `isnan`, no mutation.

### 3.2 Two rules that invert the general Julia advice

| general Julia | inside `@model` with ReverseDiff | why |
|---|---|---|
| `view(A, idx)` avoids a copy | **`A[idx]`** — `view` is ~5x slower | `view` on a `TrackedArray` gives a `SubArray` of `TrackedReal`; the tape goes scalar and the degradation is contagious |
| fuse broadcasts to avoid temporaries | **do not** fuse untracked constants into tracked kernels | `sum(ll .* w)` beats `sum(w .* (...))` by 1.6x |

Both measured; both were wrong in the old guide; both tracked in
[`docs/tickets/T002`](../tickets/T002-scalar-taped-likelihood.md).

### 3.3 Branch on types, never on values

`AutoReverseDiff(compile=true)` records the tape once. A branch on a sampled *value* is
frozen at whatever it did during recording, and the gradient is **silently wrong**.

```julia
# ✅ dispatch — resolved at compile time, emits no tape instruction
apply_guard(::NoGuard, η)     = η
apply_guard(g::ClampGuard, η) = clamp.(η, g.lo, g.hi)

# ❌ a runtime test, even though it reads like configuration
η = cfg.guard_on ? clamp.(η, -10.0, 10.0) : η
```

`clamp`, `min`, `max`, `abs`, `ifelse` on tracked values are all value-dependent branches.
Writing them without the word `if` does not make them static. There is an **unresolved**
compiled-tape divergence associated with this — T002 §(c). Do not assume it is benign.

### 3.4 Submodel API as it exists in DynamicPPL 0.38

```julia
x ~ to_submodel(inner())          # sites become x.a, x.b — prefix inferred from the LHS
x ~ to_submodel(inner(), false)   # no prefix added; sites keep their own names
```

- The LHS receives the submodel's **return value**, not its latent variables. If you need a
  latent downstream, return it explicitly.
- No destructuring on the LHS: `(; a, b) ~ to_submodel(...)` is invalid. Bind, then
  destructure.
- A dummy LHS is required even when the return value is unused.
- To set a prefix that is *not* the LHS name — the only way to name sites from a component
  rather than from where it happens to be called:

```julia
head ~ to_submodel(DynamicPPL.prefix(inner(cfg), Val(name_of(cfg))), false)
```

  The `false` matters: without it DynamicPPL adds the LHS prefix *as well*, and in a
  recursion the names nest (`head.head.head.w`).

### 3.5 Iterating a variable number of terms

Never a runtime loop. Put the terms in a **`Tuple`** so their count and types are in the
model's type, and recurse via `Base.tail` — Julia unrolls it before the tape exists. Make the
empty case a structural `nothing`, dispatched away, so composing nothing costs nothing.
Full pattern and rationale: AD guide §7.

### 3.6 Every engine must implement four things

```julia
Features.required_features(::MyModel)                        # Vector{AbstractFeatureConfig}
PreGame.build_turing_model(::MyModel, ::FeatureSet)          # the two-layer boundary
PreGame.extract_parameters(::MyModel, df, fs, chain::Chains) # Dict{Int, NamedTuple}
```

plus the correct **supertype**, because `src/predictions/score_computation/` dispatches the
score grid on it. `AbstractPoissonModel` → double-Poisson grid, `AbstractNegBinModel` →
NegBin grid. Getting this wrong does not error; it prices with the wrong grid.

**[repo]** Some engines subtype one family but return the rates of another, and are listed
explicitly in the dispatch `Union` — see the annotated
`AbstractDoublePoissonPlayerModels` union at the top of
`src/predictions/score_computation/poisson.jl`. A new engine that belongs there and is not
added fails at prediction time with a misleading "missing `r` column" NegBin error rather
than anything pointing at the real cause. Check that file whenever you add an engine.

</probabilistic_programming>

---

<style_and_api_design>

## 4. Style and API design

From the Julia manual's style guide, filtered to what actually comes up here.

### 4.1 Naming

| kind | convention | example |
|---|---|---|
| modules, types | `UpperCamelCase` | `DataStore`, `ClampGuard` |
| functions, variables | lowercase, `_` only where it aids reading | `build_turing_model`, `haskey` |
| mutating functions | trailing `!` | `add!`, `push!` |
| constants | `UPPER_SNAKE` or `UpperCamel` for types | `EARTH_RADIUS_MILES` |

Avoid abbreviations. `n_teams` not `nt` in any signature or struct field; short locals are
fine inside a five-line function.

### 4.2 Dispatch is the design tool

Prefer one generic function with methods on abstract types over a family of differently-named
functions or a `Symbol` switch.

```julia
# ✅ the component knows what it is; the caller does not repeat it
add!(b::Builder, c::AbstractDynamicsConfig)   = (b.dynamics = c; b)
add!(b::Builder, c::AbstractCovariateConfig)  = (push!(b.covariates, c); b)
add!(b::Builder, c) = error("no family for $(typeof(c)); accepted: ...")

# ❌ a second place to get it wrong, and a switch to maintain
with_component(b, :dynamics, c)
```

Always give the generic function a **loud fallback**. Silent no-ops are the worst outcome.

### 4.3 Avoid type piracy

Do not add methods to `Base`/foreign functions on types you do not own. Extending
`Features.required_features` on *your* model type is fine; extending `Base.show` on
`Vector{SomeoneElsesType}` is piracy and will break code you never see.

### 4.4 Do not over-constrain arguments

```julia
foo(x::Real)                       # ✅
foo(x::T) where {T<:Real}          # ❌ same thing, more noise — only add T if you USE it
foo(x::Float64)                    # ❌ unless you genuinely require exactly Float64
```

Conversion is the caller's job.

### 4.5 Argument order

Follow Base: function first (so `do` blocks work), then IO, then the mutated argument, then
types, then unmutated inputs, then keywords.

### 4.6 Keyword arguments and `Base.@kwdef`

Config structs use `Base.@kwdef` with explicit defaults. Every default is a modelling
decision — state it in the docstring, because a gate will later demand that nothing about
the configuration is hidden.

### 4.7 Errors: fail early, loudly, with the fix in the message

```julia
# ✅
length(x) == n || error("covariate $(name) has length $(length(x)); expected $n — " *
                        "the design column is misaligned with the fold")

# ❌
@assert length(x) == n
```

`@assert` may be compiled out and carries no context. Reserve `try/catch` for genuinely
recoverable I/O; do not use it to paper over a logic error.

</style_and_api_design>

---

<repo_conventions>

## 5. Repository conventions

### 5.1 Prototype before `src/`

New work lives in `current_development/` as a **loader/runner pair**:

- `lXX_*.jl` — types, equations, helpers, infrastructure. Definitions only, no execution.
- `rXX_*.jl` — the readable scientific workflow, in numbered sections.

Read [`docs/prototype_runner_style_guide.md`](../prototype_runner_style_guide.md) before
writing either. A runner is a research notebook, not a framework; technical machinery belongs
in the paired loader. Code graduates to `src/` only after the prototype is validated — and,
in the Scottish stream, only after the gates pass.

### 5.2 Defects found mid-task become tickets, not fixes

If you find a real defect while doing something else, **do not fix it inline**. Write a
self-contained brief in `docs/tickets/` (evidence, root cause with `file:line`, reproduction,
blast radius, proposed fix, acceptance criteria, scope guard), add a row to
`docs/tickets/README.md`, and carry on. Check whether an open ticket already covers it and
extend that instead of raising a duplicate.

### 5.3 Adding things — the one-file answers

| task | where |
|---|---|
| new league/segment | `src/Data/fetchers/segments.jl` only |
| new feature extractor | `add_feature!(F_data, ::Val{:name}, ids, team_map, ds)` in `src/features/extractors/` |
| new model component | `Config` struct + `@model` builder + `extract_*` in `src/models/pregame/components/` |
| new backtest metric | subtype `AbstractWealthMetric` / `AbstractDistributionalMetric` |
| new L2 calibrator | subtype `AbstractCalibrator` / `AbstractGenerativeRateCalibrator` in `src/Calibration/` — in practice a **location law** (`AbstractCalibrationWeightLaw` + `calibration_weight`) or a **dispersion map** (`AbstractDispersionMap` + `residual_map`) |

`AbstractLayerTwoModel` and `BasicLogitShift` (`src/Calibration/shift_models/`) are
**deprecated** and warn once per session. A selection-level logit shift moves each market's
probability with its own offset, so `P(over 2.5) + P(under 2.5) != 1` and the shifted board
is not a scoreline distribution at all. `GenerativeRateCalibrator` shifts the generative
intensity instead, so every derivative price is read off one score tensor and cannot
disagree — see
[`docs/architecture/rfc_layer2_calibration_v2.md`](../architecture/rfc_layer2_calibration_v2.md).
A calibrator returns the **same concrete latent container type** it was given; that contract
is what lets `Evaluation` and `Portfolio` consume a calibrated posterior with no new methods.

### 5.4 Data contract

Every data domain passes **Fetch → Process → QA** (`src/Data/fetchers/interfaces.jl`).
Caches are `.jls` files under `.cache/`. Never write a fetcher that reaches past its segment,
and never let a feature extractor see match IDs it was not given — that is the perturbation
gate, and it is the one that catches future data leaking into a historical fold.

</repo_conventions>

---

<workflow>

## 6. Working practice

### 6.1 Keep the session warm

Julia's cost is compilation, not execution. Starting a fresh `julia` process per edit is the
slowest possible loop.

```julia
using Revise            # BEFORE using BayesianFootball
using Pkg; Pkg.activate(".")
using BayesianFootball
```

`Revise` tracks edits to loaded package files with no restart. `includet` tracks a
standalone file. **What still needs a restart:** changing `Project.toml`, and — despite 1.12
allowing redefinition — anything where orphaned methods (§1.4) have made the session
untrustworthy. When in doubt, restart; a confusing `MethodError` is usually this.

**[repo]** On the server, work through the warm kaimon REPL rather than spawning processes;
`JULIA_PKG_PRECOMPILE_AUTO=0` avoids a known broken-dependency precompile stall.

### 6.2 Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'         # full suite
julia --project -e 'include("test/data_tests.jl")' # one file
julia --project -t 32                              # threads for MCMC
```

### 6.3 Threading

`Threads.@threads` over splits is the repo's pattern. Two hazards: **do not** write to a
shared un-preallocated structure from multiple threads, and set
`LinearAlgebra.BLAS.set_num_threads(1)` during sampling so BLAS does not oversubscribe the
cores the sampler is using.

### 6.4 Where heavy compute runs

Heavy MCMC runs on `mcmc-beast`, not the laptop. Local files reach the server via
`git push` → `git pull`, not by editing there. **[repo]** When the question is whether two
implementations are equivalent, prove it deterministically (§7) instead of sampling — it is
both stronger and free.

</workflow>

---

<verification>

## 7. How to know you are right

Ranked by strength. Prefer the strongest one available; do not present a weak one as if it
were a strong one.

1. **Bit-identical log-density at shared parameters.** Evaluate two implementations at the
   *same* θ and require `Δ == 0.0`. Two models with an identical log-density *are* the same
   posterior. Requires first proving the θ layouts match, or you are evaluating two functions
   at two points.
2. **Independent re-derivation.** Write the maths a second time from the documentation
   rather than from the code, and compare. Never refactor the reference to call the thing it
   checks. If they disagree, that is the finding.
3. **Exact end-to-end comparison with the noise removed.** Push the *same* posterior draws
   through both extraction paths. Any difference is arithmetic, not Monte Carlo.
4. **Gradient agreement.** Compiled tape == fresh ReverseDiff == ForwardDiff, and the
   compiled tape still correct at *perturbed* points — that last one is what catches a
   smuggled branch.
5. **Monte-Carlo comparison.** Can only ever conclude "within noise", needs a measured null
   (the same model against itself at another seed) to mean anything, and is worthless on
   parameters that have not converged. Use it to demonstrate, never to prove.

```julia
# The shape of (1) and (4), as used throughout current_development/
Δ = logdensity_a(θ) - logdensity_b(θ);        @assert Δ == 0.0
@assert relerr(ReverseDiff.gradient(f, θ), ForwardDiff.gradient(f, θ)) <= 1e-6
```

**Report faithfully.** If a check is marginal, say so and say why. If you excluded data from
a comparison, say how much and on what criterion. A green table that hides an exclusion is
worse than a red one.

</verification>

---

<antipatterns>

## 8. Quick reference — avoid / do

| ❌ avoid | ✅ do | why |
|---|---|---|
| `if x == missing` | `ismissing(x)` | `missing == missing` is `missing` |
| `A = B` then mutate `B` | `A = copy(B)` | assignment aliases |
| mutating without `!` | `mutate!(x)` | the name is the only warning the caller gets |
| `using Both; ambiguous_name(...)` in a loader | `import Pkg` + `Pkg.name(...)` | breaks conditionally on load order |
| `isdefined(@__MODULE__, :NameThatNeverExisted)` | guard on a name the file defines | 1.12 redefines silently |
| `struct C; d::Distribution; end` | `struct C{D<:Distribution}; d::D; end` | abstract field kills specialisation |
| `Vector{Real}` | `Vector{Float64}` | boxed, pointer-chasing |
| untyped global in a hot function | `const`, or an argument | assumed `Any` |
| `for` / `if`-on-value inside `@model` | broadcast + masks + type dispatch | scalar tape / silently wrong gradient |
| `view(A, idx)` inside `@model` | `A[idx]` | 5x, tape goes scalar |
| `sum(w .* (y .* η .- ...))` | `ll = y .* η .- ...; sum(ll .* w)` | 1.6x |
| `loggamma.(y .+ 1)` inside `@model` | precompute in the builder | it is data |
| runtime loop over `Vector{AbstractCfg}` in `@model` | `Tuple` + `Base.tail` recursion | unrolls at compile time |
| `zeros(n)` for an absent term | `nothing` + dispatch | costs tape nodes forever |
| `@assert cond` | `cond \|\| error("... and here is the fix")` | may be compiled out; no context |
| fixing a defect you tripped over | write a ticket | derails the task, loses the finding |
| "posteriors look similar" | bit-identical log-density | one is a proof |
| heavy MCMC on the laptop | deterministic proof, or the server | hours vs seconds |

</antipatterns>

---

<escalation>

## 9. When to stop

Stop and ask, or write a ticket, rather than pressing on:

- A gate that used to pass now fails and you did not touch that area — that is a finding, not
  an obstacle. Do not tune the threshold until it passes.
- You are about to weaken a threshold, widen a tolerance, or exclude data to make a check go
  green. Say what you found instead; if exclusion is genuinely correct (unconverged chains,
  say), state the criterion and the count.
- The fix requires editing `src/` when the task was scoped to `current_development/`.
- A `MethodError` on a function you can see exists — suspect §1.4 or §1.5 before rewriting
  anything.
- An API does not behave as documented. Check the installed version first
  (`<environment>`); the docs online may describe a different one.

</escalation>

---

<references>

## Sources

**Julia manual** (verified 2026-08-28)
- [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) — §4
- [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/) — §2
- [Noteworthy Differences from Other Languages](https://docs.julialang.org/en/v1/manual/noteworthy-differences/) — §1.2, §1.6, §1.7
- [Workflow Tips](https://docs.julialang.org/en/v1/manual/workflow-tips/) — §6.1
- [The World Age Mechanism](https://docs.julialang.org/en/v1/manual/worldage/) and
  [Julia 1.12 Highlights](https://julialang.org/blog/2025/10/julia-1.12-highlights/) — §1.4

**Turing / DynamicPPL**
- [Submodels](https://turinglang.org/docs/usage/submodels/) — §3.4
- [DynamicPPL API](https://turinglang.org/DynamicPPL.jl/stable/api/) — §3.4

**This repository**
- [`docs/turing_ad_performance_guide.md`](../turing_ad_performance_guide.md) — the `@model` hot path, in full
- [`docs/prototype_runner_style_guide.md`](../prototype_runner_style_guide.md) — loader/runner structure
- [`docs/architecture/composable_model_builder_specification.md`](../architecture/composable_model_builder_specification.md) — the builder pattern §3.5 refers to
- [`docs/tickets/README.md`](../tickets/README.md) — the ticket workflow; T002 and T006 are cited above
- [`AGENTS.md`](../../AGENTS.md) — infrastructure, tmux, remote execution
- [`CLAUDE.md`](../../CLAUDE.md) — architecture overview and the standard experiment pipeline

</references>
