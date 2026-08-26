# Future-model adapter guide

This guide documents the **planned** interface in `_protocol/interface.jl`. That interface deliberately has no production fallback: unimplemented methods call `_sl_missing`. Models 00 and 01 predate it and are not implementations. Do not claim adapter conformance until a concrete `AbstractSLModelAdapter` implements the hooks and its gates call them.

The adapter makes model-specific predictive mathematics explicit; shared gates own joins, scoring, RQR randomisation, thresholds, and reports. It exists to prevent the archived failure mode: a bespoke prediction path silently used the wrong team map and omitted hierarchical scales.

## Non-negotiable invariants

1. Extend package features, sampling, extraction, and score dispatch where possible. A bespoke engine receives the same independent-referee tests.
2. The independent referee/reference grid must not call `extract_parameters`, `compute_score_matrix`, `compute_market_probs`, the Turing model, or a helper shared with those paths. `Distributions` primitives and l02-style arithmetic are allowed.
3. Preserve complete posterior draws. Never average components first or mix chain flattening order. Synthetic parity uses distinct values in at least two chains.
4. Grids are `[home, away]`, with score `h,a` at `[h+1,a+1]`, and retained support `0:(max_goals-1)`. Report finite-grid mass; do not silently normalise it.
5. Report population fallbacks for every unknown team/player/referee/season feature. A zero-sum team effect may fall back to zero; a global home effect must remain global.
6. Every fitted rating, conversion, or prior is history-only at the fold cutoff and must pass the Gate-2 future perturbation test.

## Exact planned interface

The concrete adapter subtypes `AbstractSLModelAdapter` (defined in `_protocol/types.jl`). The exact declarations currently planned are:

```julia
sl_reference_extract(adapter::AbstractSLModelAdapter, params, fixture, fs)
sl_reference_grid(adapter::AbstractSLModelAdapter, row, draw::Int, max_goals::Int)
sl_marginal_cdf_bounds(adapter::AbstractSLModelAdapter, side::Symbol, row, y::Int)
sl_marginal_logpdf(adapter::AbstractSLModelAdapter, side::Symbol, row, y::Int)
sl_capabilities(adapter::AbstractSLModelAdapter)

sl_referee_eval(adapter::AbstractSLModelAdapter, stage::Symbol, args...) = NamedTuple[]
sl_adapter_check(adapter::AbstractSLModelAdapter, stage::Symbol, args...)
```

`row` is one extracted latent row; `draw` indexes its posterior vectors. A fixture has at least `match_id`, `home_team`, `away_team`, `match_date`, and all model-required prediction-time columns. Use named fields/types in parameter and latent carriers; do not make an untyped `Dict` the public boundary.

## The four mandatory conceptual hooks

### 1. `sl_referee_eval`: adapter-owned independent checks

```julia
sl_referee_eval(adapter::AbstractSLModelAdapter,
                stage::Symbol,
                args...) -> AbstractVector
```

This is the planned model-specific gate hook, not an extraction API. `sl_adapter_check` invokes it and requires a vector of normal gate results (`name`, `pass`, `detail`). Use explicit stages such as `:equation`, `:extraction`, `:grid`, and `:evaluation`; document accepted stages in the model loader.

Use it where a generic gate cannot independently referee a model-specific transformation: convolution, correlation correction, referee component, or unknown-entity semantics. Its arithmetic must rebuild documented quantities from known parameters/fixtures without production extraction. For the team model, that includes

\[
\alpha=(raw_a\odot\sigma_a)-\operatorname{mean}(raw_a\odot\sigma_a),\quad
\beta=(raw_d\odot\sigma_d)-\operatorname{mean}(raw_d\odot\sigma_d),
\]
\[
\lambda_h=\exp(\mu+\delta_m+\gamma+\alpha_h+\beta_a),\qquad
\lambda_a=\exp(\mu+\delta_m+\alpha_a+\beta_h).
\]

Thus `σ`/tau, centering, season/month semantics, global versus indexed home advantage, and fallback are all mandatory. Gate 4 uses the more specific planned `sl_reference_extract(adapter, params, fixture, fs)` for draw-by-draw extraction parity; `sl_referee_eval(..., :extraction, ...)` adds checks that generic parity cannot express.

### 2. `sl_reference_grid`: independent joint score distribution

```julia
sl_reference_grid(adapter::AbstractSLModelAdapter,
                  row,
                  draw::Int,
                  max_goals::Int) -> AbstractMatrix{<:Real}
```

Return `grid[h+1,a+1] = P(Y_h=h,Y_a=a | row, draw)`. This is Gate 5's independent reference, not a wrapper around the production pricer.

For independent Poisson,
\[
S_{h,a}=\operatorname{Pois}(h;\lambda_h)\operatorname{Pois}(a;\lambda_a).
\]
For mean-parameterised NegBin,
\[
S_{h,a}=\operatorname{NB}\!\left(h;r_h,{r_h\over r_h+\lambda_h}\right)
\operatorname{NB}\!\left(a;r_a,{r_a\over r_a+\lambda_a}\right).
\]
A correlated/copula/Dixon–Coles/recombination model must encode its actual documented joint law. It must not claim dependence while returning an independent outer product. Gate 5 compares cells to the production matrix, tests orientation/truncated moments/market partitions, and reports `sum(grid)` as the raw retained mass.

### 3. `sl_marginal_cdf_bounds` and `sl_marginal_logpdf`: predictive evaluator

```julia
sl_marginal_cdf_bounds(adapter::AbstractSLModelAdapter,
                       side::Symbol, row, y::Int) -> (lower, upper)
sl_marginal_logpdf(adapter::AbstractSLModelAdapter,
                   side::Symbol, row, y::Int) -> Real
```

`side` is `:home` or `:away`; unsupported values must throw. Bounds are inclusive posterior-predictive marginal CDF bounds:
\[
(lower,upper)=(F(y-1),F(y)),\quad lower=0\ \text{when}\ y=0.
\]
The current planned comment says `sl_marginal_logpdf` is a **marginal** log density, averaged by its caller across draws. Implement it accordingly; never use the truncated grid merely because it is available. Existing model 01 correctly uses analytic NegBin CDFs for RQR, not its 0–11 grid.

The shared RQR evaluator forms
\[
u\sim U(\bar F(y-1),\bar F(y)),\qquad RQR=\Phi^{-1}(u),
\]
where \(\bar F\) is averaged across posterior draws.

**Documented-vs-planned ambiguity:** Gate 6 also needs *joint scoreline* LPD,
\[
LPD=\log[S^{-1}\sum_s p(y_h,y_a\mid s)].
\]
Existing model 01 obtains it from the joint score grid. The planned interface has no `sl_joint_logpdf`; `sl_marginal_logpdf` cannot faithfully replace it for a dependent model. Until an explicit joint-logpdf hook is added, retain independent joint-LPD code in the model evaluator and report this limitation rather than silently treating a marginal logpdf as joint.

### 4. `sl_capabilities`: declarative semantics

```julia
sl_capabilities(adapter::AbstractSLModelAdapter) -> NamedTuple
```

The exact default fields in `_protocol/interface.jl` are:

```julia
(; uses_home_intensity = true,
   supports_population_fallback = true,
   expected_score_dispatch = nothing,
   expected_params_dispatch = nothing,
   expected_sampled_sites = nothing)
```

Override these with actual dispatch expectations and sampled-site manifest. Additional declarations are encouraged, for example `joint_grid`, `marginal_cdf`, `conditional_independence`, `supports_1x2`, `supports_btts`, `totals_lines`, `score_orientation=:home_away`, `grid_is_normalised=false`, `unknown_team_fallback`, `unknown_referee_fallback`, `has_dispersion`, and `has_dependence`. Capabilities state what code actually prices, never what a design note hopes to price. Gate 1/Gate 5 should reject a contradictory declaration.

## Implementation order

1. Write `MODEL.md` and an independent `l02_equations.jl`: all transformations, target, support, fallback, and joint law.
2. Implement config/features/Turing/extraction/score dispatch through package extension points.
3. Define a model-local concrete adapter and the hooks above. Lift common code only after a second model has identical semantics.
4. Create distinct two-chain synthetic draws. Gate 4 compares production extraction against `sl_reference_extract`, including every latent component and fallback.
5. Gate 5 compares production matrices to `sl_reference_grid`, then tests orientation, truncation price shift, exact truncated moments, and market identities.
6. Gate 6 builds RQR via the marginal hooks. Keep joint LPD independent until the interface gains a joint-logpdf hook.
7. Print capabilities with the config and assert its dispatch/site claims. Unsupported checks must be explicit `N/A` with a reason, never silently absent.

## Minimal Poisson sketch

```julia
struct MyPoissonAdapter <: AbstractSLModelAdapter end

function sl_reference_grid(::MyPoissonAdapter, row, draw::Int, max_goals::Int)
    ph = [pdf(Poisson(row.λ_h[draw]), h) for h in 0:max_goals-1]
    pa = [pdf(Poisson(row.λ_a[draw]), a) for a in 0:max_goals-1]
    ph * pa'
end

function sl_marginal_cdf_bounds(::MyPoissonAdapter, side::Symbol, row, y::Int)
    λ = side === :home ? row.λ_h : side === :away ? row.λ_a : throw(ArgumentError("side"))
    lo = y == 0 ? 0.0 : mean(cdf.(Poisson.(λ), y - 1))
    hi = mean(cdf.(Poisson.(λ), y))
    (lo, hi)
end

function sl_marginal_logpdf(::MyPoissonAdapter, side::Symbol, row, y::Int)
    λ = side === :home ? row.λ_h : side === :away ? row.λ_a : throw(ArgumentError("side"))
    log(mean(pdf.(Poisson.(λ), y)))
end
```

The decisive part remains the independent `sl_reference_extract`/`sl_referee_eval` arithmetic: it must include every scale, mapping, and fallback rather than call the extractor it validates.

## Review checklist

- Is every model-specific referee calculation independent of the production path?
- Do two-chain synthetic draws prove ordering, scale/component parity, and fallbacks?
- Does the reference grid represent actual dependence and `[home,away]` orientation?
- Are CDF bounds full-support/analytic (or explicitly unsupported)?
- Is joint LPD handled honestly despite the current missing joint-logpdf hook?
- Do capabilities agree with code and `MODEL.md`?
- Are truncation, unknown identities, and unsupported markets reported explicitly?
- Are all threshold gates in `GATES.md` retained?
