# T002 working notes — scalar-taped engine likelihoods (open)

Canonical brief: [`docs/tickets/T002-scalar-taped-likelihood.md`](../../docs/tickets/T002-scalar-taped-likelihood.md)

## Branch and baseline

- Raised from `feat/scottish-lower-protocol` at `804b715`, by model 01 gate 3b in
  `current_development/scottish_lower/`.
- Suggested branch: `fix/t002-scalar-taped-likelihood`, based on `feat/scottish-lower-protocol`.
- Reproducer and baseline matrix: `tickets/t002/reproduce.jl`.
- Kaimon checkout `/root/BayesianFootball` holds the data cache; the local checkout may not.
  Push local commits, then pull on the server before running anything. It contains unrelated
  untracked research files — do not clean them.

## Baseline measured 2026-08-26

`DynamicGoalsTimeDecayModel`, Scottish 56+57, fold 1 (720 matches, 1440 observations,
51 parameters), on `mcmc-beast` with `-t 16`:

| | |
|---|---|
| tape instructions | 35,421 (35,387 scalar) — **24.6 per observation** |
| gradient | 1.150 ms median, 1.140 ms min |
| primal | 0.097 ms |
| gradient / primal | 11.8x (reverse mode should be ~3-5x) |
| tape compile | 0.10 s |
| allocations | 256 bytes |

Self time contains no maths kernel — it is entirely tape bookkeeping:

```
 10.6%  getproperty        Base_compiler.jl:54
  9.7%  CallWrapper        FunctionWrappers.jl:65
  7.0%  pull_value!        tracked.jl:180
  6.8%  setproperty!       Base_compiler.jl:58
  4.6%  pull_deriv!        tracked.jl:188
  3.4%  increment_deriv!   propagation.jl:46
```

`loggamma` does not appear in the top 12 despite 2,880 calls per gradient. The problem is
the *number* of tape nodes, not the cost of any one of them.

## Confirmed code path

1. `src/models/pregame/engines/team_level/time_decay/goals.jl:54-57` selects team effects
   with `view(dyn.α, home_team_indices)`. On a ReverseDiff `TrackedArray` this produces
   `SubArray{TrackedReal}`, so every downstream broadcast is taped element by element.
2. `goals.jl:78-79` computes `λ_h = exp.(...)` — already 1 scalar node per observation.
3. `goals.jl:82-83` broadcasts `logpdf.(RobustNegativeBinomial.(r, λ), goals)`, adding ~19
   more scalar nodes per observation.
4. `src/MyDistributions/negative_binomial.jl:79` blocks the vectorised alternative:
   `isinteger(k) ? logpdf(d, Int(k)) : -Inf` throws `InexactError` when `k` arrives as a
   ForwardDiff dual, which is what `tracker_∇broadcast` passes on the fast path.

Steps 1 and 4 must be fixed together. Fixing 1 alone converts a slow model into a crashing
one; fixing 4 alone changes nothing, because nothing reaches that path yet.

## Attribution (measured, section 4 of the reproducer)

| stage | nodes/obs | marginal |
|---|---|---|
| λ only (`view` + `exp`) | 2.0 | — |
| + `Poisson` | 5.0 | +3.0 |
| + stdlib `NegativeBinomial` | 11.9 | +9.9 |
| + `RobustNegativeBinomial` | 21.0 | +19.0 |

Isolated `view` vs `getindex`, 720 rows, identical value: **1439 vs 3** instructions.

## Design direction under investigation

Not yet decided — this is the fixing session's call. Three routes, from the canonical brief:

1. **Fix `negative_binomial.jl:79`, then switch `view` → `getindex` in the engines.**
   Smallest change, largest win. The `Real` method should dispatch on the underlying value
   rather than calling `Int(k)` — e.g. add a `logpdf(d, k::Integer)` method and guard the
   `Real` fallback so duals are never converted.
2. **Hand-write a vectorised NegBin log-likelihood** on plain arrays. For integer `k`,
   `loggamma(k+r) - loggamma(r) = Σ_{j=0}^{k-1} log(r+j)` exactly. Scottish goal counts are
   almost always 0-4, so this removes `loggamma` from the tape entirely and has a trivial
   derivative `Σ 1/(r+j)`. This is what the archived prototype's `_negbin_vector_loglik`
   ("precomputed gamma recurrences") did, benchmarking at 0.572 ms.
3. Re-benchmark the other nine affected engines and correct the guide.

Cheap wins available regardless of route, inside `negative_binomial.jl:58-60`:

- `log(r + μ)` is evaluated twice per observation; one is free.
- Under `GlobalDispersion` `r` is a **single global scalar**, yet `loggamma(r)` and `log(r)`
  are recomputed once per observation — 1,440 identical evaluations per gradient. Hoist them.

## Open decisions

- Whether to fix all ten affected engines or only `goals.jl` in this ticket. The
  `logpdf.(Dist.(params), data)` pattern is shared, but severity is dominated by how heavy
  the logpdf body is — Poisson and Normal bodies are much cheaper than NegBin.
- Whether `view` → `getindex` is safe for *every* call site, or only for integer-vector
  indexing. A contiguous `UnitRange` view may behave differently from a fancy-index view;
  the reproducer only tests the fancy-index case, which is what the engines use.
- Whether `docs/turing_ad_performance_guide.md` Rule 4 is simply wrong, or was correct for an
  older ReverseDiff. Worth a version note either way, since the engines were written to it.
- Whether `RobustNegativeBinomial` should keep the `-Inf`-on-non-integer behaviour at all
  once a dedicated `Integer` method exists. It is a silent-wrong-answer risk under AD if a
  dual ever reaches it and `isinteger` returns false.

## Scope guard (from the canonical brief)

Pure AD-performance work: same posterior, same numbers, fewer tape nodes. Do not change any
model's mathematics, priors, component structure, or default configuration. **If the log
density changes by any amount, the change is out of scope.** Section 5 of the reproducer
asserts this and currently passes at max |Δ| = 0.000e+00.

Do not touch the score-matrix or extraction paths; they do not run under AD.
