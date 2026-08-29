# T002 — Engine likelihoods are taped scalar-by-scalar, ~20x more AD work than needed

| | |
|---|---|
| **Status** | open |
| **Severity** | medium — cost only, not correctness. No result is wrong because of this. |
| **Area** | `src/models/pregame/engines/`, `src/MyDistributions/negative_binomial.jl`, `docs/turing_ad_performance_guide.md` |
| **Raised** | 2026-08-26, by model 01 gate 3b in `current_development/scottish_lower/` |
| **Verified on** | `DynamicGoalsTimeDecayModel`, Scottish 56+57, fold 1 (720 matches, 51 parameters) |

## Summary

Gradient evaluation is ~1.15 ms median where the guide targets < 1 ms. The cause is not
expensive maths. The ReverseDiff tape contains **35,421 instructions, 35,387 of them
scalar** — roughly 24.6 tape nodes *per observation* — for a model with 51 parameters.
Almost all gradient time is per-instruction bookkeeping, not arithmetic.

A correctly vectorised formulation produces a tape whose size is independent of row count.

## Evidence

Tape composition for `DynamicGoalsTimeDecayModel` on fold 1 (1440 goal observations):

| op | count | per observation |
|---|---|---|
| `+` | 12,317 | 8.6 |
| `-` | 5,774 | 4.0 |
| `log` | 5,760 | 4.0 |
| `*` | 4,320 | 3.0 |
| `max` | 2,882 | 2.0 |
| `loggamma` | 2,880 | 2.0 |
| `exp` | 1,441 | 1.0 |

These map exactly onto `RobustNegativeBinomial`'s `logpdf` body plus the linear predictor.
Note `log` appears 4x per observation: `log(r + μ)` is computed twice with no CSE.

Self time (idle worker threads filtered out) contains **no maths kernel at all**:

```
 10.6%  getproperty          Base_compiler.jl:54
  9.7%  CallWrapper          FunctionWrappers.jl:65
  7.0%  pull_value!          tracked.jl:180
  6.8%  setproperty!         Base_compiler.jl:58
  5.9%  *                    float.jl:497
  4.6%  pull_deriv!          tracked.jl:188
  3.4%  increment_deriv!     propagation.jl:46
```

`loggamma` does not appear in the top 12. The cost is executing 35k tiny instructions.

## Attribution: engine vs distribution vs inputs

**The inputs are not the problem.** Gate 2 verifies the FeatureSets carry no `missing` or
`NaN` and that goals are `Vector{Int64}`; gate 3a shows the log density matches an
independent reference at max |Δ| = 0.000e+00. The data is correct and the maths is correct.

Measured per observation on the scalar path (720 rows, tape instructions):

| stage | nodes/obs | marginal | attributable to |
|---|---|---|---|
| λ only — `view` + `exp` | 2.0 | — | engine (`view`) |
| + `Poisson` logpdf | 5.0 | +3.0 | — |
| + stdlib `NegativeBinomial` | 11.9 | +9.9 | — |
| + `RobustNegativeBinomial` | **21.0** | **+19.0** | our distribution |

So the cost splits two ways, and both halves are ours:

1. The engine's `view` puts *everything* on the scalar path (see (a) below). This is the
   structural problem — it is what makes the tape scale with row count at all.
2. `RobustNegativeBinomial` is then ~**1.9x the stdlib `NegativeBinomial`** and ~6x
   `Poisson` for the same job, so it pays the worst price for being on that path.

### What is redundant inside `RobustNegativeBinomial`

`src/MyDistributions/negative_binomial.jl:44-63`:

```julia
term1 = loggamma(k + r) - loggamma(k + 1) - loggamma(r)
term2 = r * (log(r) - log(r + μ))
term3 = k * (log(μ) - log(r + μ))
```

- `log(r + μ)` is evaluated **twice** per observation. One is free.
- Under `GlobalDispersion`, `r` is a **single global scalar** shared by every observation,
  yet `loggamma(r)` and `log(r)` are recomputed once per observation — 1,440 identical
  evaluations per gradient. Both should be hoisted out and computed once.
- `loggamma` itself brings a `max` node with it (2,880 `loggamma` produced 2,882 `max`),
  so each avoided `loggamma` saves two tape nodes rather than one.

For integer `k` the whole `loggamma` pair collapses exactly:
`loggamma(k + r) - loggamma(r) = Σ_{j=0}^{k-1} log(r + j)`. Scottish lower-league goal
counts are almost always 0-4, so this is a handful of `log` calls with a trivial
derivative `Σ 1/(r+j)` — and it removes `loggamma` from the tape entirely. This is what the
archived prototype's `_negbin_vector_loglik` ("precomputed gamma recurrences") did, and it
is the most likely explanation for its 0.572 ms.

## Root cause

Two independent problems that compound.

### (a) `view` on a TrackedArray defeats vectorisation

`src/models/pregame/engines/team_level/time_decay/goals.jl:54-57` selects team effects with
`view(dyn.α, home_team_indices)`. On a ReverseDiff `TrackedArray` this yields a `SubArray`
of `TrackedReal`, and every downstream broadcast is then taped one element at a time.

Minimal reproduction (720 rows, 23 parameters):

```julia
idx = rand(1:23, 720); base = randn(23)
ninst(f, x) = length(ReverseDiff.GradientTape(f, x).tape)

ninst(θ -> sum(exp.(view(θ, idx))), base)   # 1439 instructions
ninst(θ -> sum(exp.(θ[idx])),       base)   #    3 instructions
```

**`getindex` is the fast path and `view` is the slow one — the exact opposite of Rule 4 in
`docs/turing_ad_performance_guide.md`.** The guide's advice is wrong for this ReverseDiff
version, and the engines follow the guide. Whoever fixes this must fix the guide too,
otherwise the next engine reintroduces it.

### (b) `RobustNegativeBinomial` crashes on the vectorised path

Switching to `getindex` collapses the tape from 15,120 instructions to **5** for the same
value — and then the gradient throws:

```
InexactError: Int(Int64, Dual{Nothing}(0,1))
  logpdf(d::RobustNegativeBinomial{Dual}, k::Dual{Int64})
  src/MyDistributions/negative_binomial.jl:79
```

`src/MyDistributions/negative_binomial.jl:79`:

```julia
Distributions.logpdf(d::RobustNegativeBinomial, k::Real) = isinteger(k) ? logpdf(d, Int(k)) : -Inf
```

Once operands are true `TrackedArray`s, ReverseDiff differentiates the broadcast via
`tracker_∇broadcast`, which evaluates the kernel under ForwardDiff duals for **every**
broadcast argument — including the integer observation `k`. `Int(::Dual)` throws.

So (a) is currently *masking* (b): the slow scalar path is the only one that runs.

## Blast radius

`view(...)` on tracked parameters appears throughout `src/models/pregame/engines/`.
The `logpdf.(Dist.(params...), data)` broadcast appears in 10 of 28 engines:

```
player_level/time_decay/hierarchical_xg_market.jl
player_level/time_decay/outfield_bigchance_double_poisson.jl
player_level/time_decay/outfield_xg.jl
player_level/time_decay/outfield_xg_double_negbin.jl
player_level/time_decay/outfield_xg_market.jl
player_level/time_decay/xg_market.jl
team_level/time_decay/goals.jl
team_level/time_decay/goals_market.jl
team_level/time_decay/xg.jl
team_level/time_decay/xg_market.jl
```

Severity varies by how heavy the `logpdf` body is. Poisson/Normal bodies are cheap and
mostly constant-folded; `RobustNegativeBinomial` is the worst case because `r` is a
parameter, so `loggamma(k + r)` cannot be folded away.

## Proposed fix

Three options, in increasing order of work.

1. **Fix `negative_binomial.jl:79` and switch `view` → `getindex`.** Make the `Real`
   method dispatch on the *value* rather than converting: guard with
   `ForwardDiff.value`/`ReverseDiff.value`, or add a `logpdf(d, k::Integer)` method and
   stop routing duals through `Int(k)`. Smallest change, largest win.

2. **Hand-write a vectorised NegBin log-likelihood** on plain arrays, as the archived
   prototype did (`_negbin_vector_loglik`, "precomputed gamma recurrences"). For integer
   `k`, `loggamma(k+r) - loggamma(r) = Σ_{j=0}^{k-1} log(r+j)`, which is exact, avoids
   `loggamma` entirely, and differentiates cheaply. That prototype benchmarked at 0.572 ms
   where our stock NegBin sits at 1.15 ms — consistent with this being the difference.

3. Re-benchmark all 10 engines and correct the guide.

**Measure with the same statistic as the archive.** `docs/turing_ad_performance_guide.md`
and `archive/open_play/r04_benchmark_ad_recomb.jl` use `@belapsed`, which reports the
**minimum**. Gate 3b reports the **median**. Comparing the two overstates any regression.

## Reproduction

Deterministic reproducer and full baseline matrix: **`tickets/t002/reproduce.jl`**.
Working notes and open decisions: **`tickets/t002/README.md`**.

```julia
julia --project -t 16
include("tickets/t002/reproduce.jl")
```

Baseline to beat: 35,421 instructions, 1.150 ms median / 1.140 ms min on fold 1.

## Acceptance criteria

- [ ] Tape instruction count is **independent of row count** (compare fold 1 at 720 rows
      against fold 20 at 1060; the count must not scale).
- [ ] Gradient median < 1 ms on fold 20, the largest fold.
- [ ] `ReverseDiff.gradient` runs without `InexactError` on the vectorised path.
- [ ] **Log density is bit-identical to the current implementation.** Gate 3a in
      `current_development/scottish_lower/01_team_poisson/l04_sampling_gates.jl` compares
      against an independent reference and currently passes at max |Δ| = 0.000e+00. It must
      still be exactly 0.
- [ ] `docs/turing_ad_performance_guide.md` Rule 4 corrected, with the measurement above.
- [ ] All 403 package tests pass.

## Scope guard

Do **not** change any model's mathematics, priors, component structure, or default
configuration. This is a pure AD-performance ticket: same posterior, same numbers, fewer
tape nodes. If a change alters the log density by any amount, it is out of scope.

Do not "fix" the score-matrix or extraction paths; they do not run under AD.

---

## Corroboration and one addition — 2026-08-28

Raised independently by `current_development/scottish_lower/05_composable_count_builder`,
which had to choose between the guide and the measurement while writing a new engine.
Same fold (Scottish 56+57, fold 1, 720 matches), Julia 1.12.6 / ReverseDiff 1.17.0 /
DynamicPPL 0.38.10.

### (a) confirmed, with an end-to-end number

Root cause (a) reproduces on the **pure Poisson** arms as well, so it is not specific to
`RobustNegativeBinomial`. Compiled-tape gradient, minimum of 400 reps after 50 warm-up
calls, identical log-density in every case:

| engine | selection | compiled gradient |
|---|---|---|
| composable Poisson engine | `view(A, idx)` | 0.389 ms |
| composable Poisson engine | `A[idx]` | 0.076 ms |
| `_engw` (02_poisson_wealth), unmodified | `view(A, idx)` | 0.521 ms |
| `00_team_poisson` engine, unmodified | `A[idx]` | 0.034 ms |

Arm 00 already uses `getindex` — by accident, not by policy, since the guide says
otherwise — which is why it is the fastest engine in the repository and why nobody
noticed. Arms 02/03/04 follow the guide and pay 15x for it.

### (b) a second, smaller tape-shape cost: fusing constants into the tracked expression

Not covered above and worth fixing at the same time. These two are the same value to the
last bit:

```julia
sum(w .* (y .* η .- exp.(η) .- lf))     # 0.061 ms   weight fused in
ll = y .* η .- exp.(η) .- lf
sum(ll .* w)                            # 0.038 ms   weight applied separately
```

Fusing the (untracked) decay weight into the tracked elementwise expression widens the
kernel ReverseDiff forward-optimises, costing ~1.6x. The pattern
`sum(match_weights .* (...))` appears throughout `src/models/pregame/engines/`.

### (c) unresolved, and the reason this is not purely a cost issue

Both the hand-written and the composable engine were probed with a compiled tape against
`ForwardDiff` at 40 points drawn ~0.8 units per coordinate away from a prior draw. Both
disagreed, worst relative error **0.37**, in the same places.

Near a prior draw and under small perturbations the tape is exact (relerr 0.0). The
divergence appears only well outside the typical set. The leading suspect is
`clamp.(η, -10.0, 10.0)`: it is a value-dependent branch, which is exactly the construct
the guide's own `compile=true` warning describes, and every engine in `src` uses it.

This was NOT isolated — it could equally be `exp` overflow or the `max` inside `loggamma`.
Whoever takes this ticket should isolate it before assuming it is benign, because if it is
`clamp` then every compiled-tape chain is running on wrong gradients whenever warm-up
leaves the typical set, which is precisely when warm-up needs them to be right.

### Scope note

The composable prototype already uses `getindex` and the unfused-weight form, and its
gradient is 0.041-0.051 ms against 0.47-0.63 ms for the arms it reproduces. It is a
worked example of the fix, not a reason to narrow this ticket: the 28 engines under
`src/models/pregame/engines/` are still on the slow path.
