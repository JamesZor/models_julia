# 07 — Unified Inference Framework

One lifecycle — `split → sample → audit → extract → Fit` — replacing `src/training/`
and `src/experiments/`, extended to in-game models, with the legacy API kept working.

```
julia --project current_development/07_unified_inference_framework/r01_demo.jl
```

Runs in seconds, needs no database, no cache and no MCMC, and exits non-zero if any
gate fails. **89 gates, all passing.**

---

## The problem

`Training.train` is a loop that calls `Samplers.run_sampler`. `Experiments.run_experiment`
is a stopwatch around `Training.train`. Neither does anything the other could not, and
the split costs three things:

| | |
|---|---|
| **Nested configs** | `ExperimentConfig` wraps `TrainingConfig` purely to carry a sampler and a `parallel` flag. Reading the sampler back is `config.training_config.sampler`. |
| **Nested results** | `ExperimentResults` wraps `TrainingResults` wraps `Vector{Tuple}`. One fold's chain is `res.training_results.items[i][1]` — four hops — and `ExperimentResults` still carries `vocabulary::Any`, an NLP leftover every call site sets to `nothing`. |
| **Work done twice** | `extract_oos_predictions` and `Diagnostics.extract_chains` each re-derive the boundaries and rebuild the feature sets from a live `DataStore`, because the run threw them away. `extract_oos_predictions` carries a drift guard (`post_processing.jl:147`) whose only job is to notice that the re-derivation disagreed with the run. |

And the audit is **optional**. A run whose R-hat is 1.4 saves, loads, prices and stakes
exactly like one that converged.

## The replacement

```julia
fit = fit_model(ds, FitConfig(name = "run", model = m, splitter = s, sampler = nuts))

fit[1].chain                  # the fold's chain — one hop
fit.latents.λ_home            # typed OOS posterior, extracted by the run
fit.diagnostics.passed        # audited by the run; a field, not a call
fit.metadata.git_commit       # "806d5cf8-dirty"
save_fit(fit)
```

The audit and the extraction happen **while the feature sets are still in scope**, so
the second derivation, the drift guard, and the class of bug the guard watched for are
all gone. `audit_convergence` and `load_fit` need no `DataStore` at all.

## Files

| file | contents |
|---|---|
| `l01_types.jl` | `FoldFit`, `FitConfig`, `Fit`, `InGameFitConfig`, `InGameFit`, `FitMetadata`, the execution strategies, and the legacy property bridges |
| `l02_convergence.jl` | `audit_convergence` → `ConvergenceSummary`: R-hat, bulk/tail ESS, divergences, tree depth, BFMI, and a `passed` that is a conjunction of stated thresholds |
| `l03_engine.jl` | `fit_model`, the three fold executors, `ReplaySampler`, latent extraction and the fold merge, checkpoints |
| `l04_ingame_bridge.jl` | `NHPPIntensityModel`, `MatchState`, `LiveKernel`, `remaining_intensity!`, `price_live_market!` — the zero-allocation live path |
| `l05_io.jl` | `save_fit` / `load_fit` / `list_fits`, atomic writes, JSON sidecars, the legacy upgrade path |
| `l06_compat_bridge.jl` | `module UnifiedInference` + every legacy name |
| `r01_demo.jl` | the runner: 12 sections, 89 gates |

`l06` includes `l05` includes … `l01`, and `l01` pulls in
[`06_typed_posterior_latents`](../06_typed_posterior_latents/) and
[`05_composable_count_builder`](../scottish_lower/05_composable_count_builder/). One
`include` loads everything.

## Using it

```julia
include("current_development/07_unified_inference_framework/l06_compat_bridge.jl")
using .UnifiedInference

fit = fit_model(ds, FitConfig(
    name     = "scottish_lower_2425",
    model    = model,
    splitter = Data.GroupedCVConfig(target_seasons = ["24/25"], history_seasons = 2),
    sampler  = Samplers.QueuedNUTSConfig(n_samples = 1000, n_chains = 4),
    save_dir = "./data/fits"))

fit.diagnostics.passed || @warn "not converged" fit.diagnostics.failed_gates
save_fit(fit)
```

`fit_model` has a second entry point that takes the folds directly:

```julia
fit_model(config; feature_sets = fss, oos_fixtures = oos)
```

This is not a testing hook. It is the seam `MatchDay` wants — it has its fixtures in
hand and no interest in re-deriving a walk-forward split to price Saturday — and it is
why `r01_demo.jl` can verify the whole lifecycle with no database.

### Re-auditing an old run

```julia
fit = load_fit("./data/experiments/xg_2425_20250104_113000")   # a LEGACY directory
fit.diagnostics          # computed on load; the legacy container had no such field
```

### Scanning

```julia
list_fits("./data/fits")
```

```
IDX  NAME                 MODEL              SAMPLER        FOLDS  TIME    R-HAT   OOS  CONV
[1]  scottish_lower_2425  ComposableCount..  QueuedNUTSCon  38     2h 11m  1.0043  912  PASS
[2]  xg_2425              DynamicXGModel     QueuedNUTSCon  38     1h 46m  1.0891  912  FAIL
```

The convergence columns are the point. The legacy sidecar records name, model, splitter,
sampler, time and whether latents exist — everything except the thing a scan is actually
run to find out.

## In-game models

An in-play intensity model does not learn how good a team is. It learns a multiplier on
a rate the pre-game model already estimated:

```
log λ_side(t) = log λ_side^pre + α + β·z(t) + γ_state·(lead/trail) + γ_red·man_adv + δ_time[bin]
```

`λ^pre` is an offset with a fixed coefficient of 1, so everything the in-play chain
learnt is defined relative to whichever pre-game posterior produced it. `InGameFitConfig`
holds that posterior as a **field**, and `InGameFit` stores the resolved container — a
fit that was never given a baseline cannot be constructed, and one that was can always
say which.

The failure this prevents is the quiet kind: an in-game chain fitted against pre-game
posterior A and priced six weeks later against posterior B is wrong by the ratio of the
two baselines. A few percent. Invisible on a chart, survives every convergence check,
shows up only as a slow bleed.

```julia
K, Λh, Λa, books = live_book(ingame_fit, (Market1X2(), MarketOverUnder(2.5)))
i = match_index(ingame_fit.pregame_latents, match_id)

# then, on every tick, allocating nothing:
state = MatchState(t = 63.0, g_h = 1, g_a = 1, r_a = 1)
remaining_intensity!(Λh, Λa, K, ingame_fit.pregame_latents, i, state)
price_live_market!(books[1], Λh, Λa, state, Market1X2())
```

### Why zero allocations here and not in the pre-game path

The pre-game path prices a fixture once, hours before kickoff. The in-play path prices
every market on every posterior draw every time the score, the man-count or the clock
moves, for every live match at once — at 2,000 draws, six markets and a 20-match card,
one `Vector` per draw is ~1.4 million allocations per repricing tick.

That is a **latency** problem, not a throughput one. Those allocations are what schedules
the garbage collector, and a GC pause between seeing a price and acting on it is the
whole cost.

## Backward compatibility

A legacy call site's **body** is unchanged. Its import line changes:

```julia
# before
using BayesianFootball

# after
import BayesianFootball
using .UnifiedInference.Legacy      # binds `Experiments` and `Training`
```

The second line is needed because `BayesianFootball` *exports* the names `Experiments`
and `Training`, and Julia refuses to rebind an imported name — `const Experiments = …`
in a scope that has done `using BayesianFootball` is an error, not a shadow. Nothing can
make two modules answer to one name; the honest claim is that everything *after* the
import is untouched. `r01_demo.jl` §9 proves it with a `LegacyCallSite` module whose body
is copied from real runners.

Every legacy member access is preserved:

| legacy expression | resolves to |
|---|---|
| `res.training_results.items[i][1]` | the chain |
| `for (c, m) in res.training_results.items` | iterates `(chain, meta)` |
| `length(res.training_results.items)` | fold count |
| `res.config.training_config.sampler` | the sampler (a synthesised view) |
| `res.vocabulary` | `nothing` |
| `latents.df` | the legacy `DataFrame` |
| `nrow(latents)` | fixture count |
| `res isa ExperimentResults` | `res isa Fit` — an alias, not a wrapper |

and every function: `run_experiment`, `train`, `save_experiment`, `load_experiment`,
`list_experiments`, `extract_oos_predictions`, `has_oos_predictions`,
`save_oos_predictions`, `load_oos_predictions`, `create_experiment_task`,
`get_model_name`, `get_model_type`.

**One deliberate behaviour change.** `run_experiment` now audits convergence and extracts
latents, because `fit_model` does. A legacy caller gets a strictly more complete result
and pays the extraction cost up front rather than on its next `extract_oos_predictions`
call, which then returns instantly.

## Where this deviates from the briefing, and why

1. **`AbstractFootballModel` is not redeclared.** Declaring it here would create a second,
   unrelated abstract type; every engine in `src/models/pregame/` subtypes the
   repository's, and none would satisfy the new one. `FitConfig{M<:AbstractFootballModel}`
   would then reject `DynamicXGModel`. The root type is reused and `AbstractInGameModel`
   is carved out beneath it. `AbstractPreGameModel` aliases the root — there is no way to
   retroactively re-parent 40-odd existing engines — and `is_pregame(m)` is the predicate
   that actually discriminates.

2. **`FoldFit`'s chain slot is unconstrained, not `C<:Chains`.** `run_sampler(::MAPConfig)`
   does not return a `Chains`. The tighter bound would make this framework unable to hold
   a run that `Training.train` holds today. Backward compatibility wins; the audit and the
   extractor dispatch on `::Chains` at their own entry points instead.

3. **`FitConfig` gains one field, `execution`.** The briefing drops `TrainingConfig`, which
   carried both the sampler *and* the `Independent(parallel, max_concurrent_tasks)`
   strategy. The strategy has nowhere else to live and `QueuedNUTSConfig`'s whole point is
   the flattened task queue. It defaults to `AutoExecution()`, which reads the strategy off
   the sampler exactly as `train_independent` does (`independent.jl:32`), so every
   construction the briefing writes still compiles.

4. **`ExperimentTask` is a struct, not a `NamedTuple` alias.** A `NamedTuple` type alias
   cannot be called positionally, so `ExperimentTask(ds, config)` — how
   `create_experiment_task` builds one and how every runner writes one — would stop
   compiling. `run_experiment` accepts the briefing's `NamedTuple` shape too.

5. **The prototype is a module.** Forced by the name collision above. The briefing's
   `const Experiments = current_module()` is in `UnifiedInference.Legacy`, which does not
   `using BayesianFootball` and can therefore bind it.

6. **`max_treedepth_rate` is a fifth gate.** Every NUTS run reports tree depth; collecting
   it and then not using it would be worse than not collecting it. It is a *performance*
   gate — the one whose failure does not invalidate the posterior — and it is separated
   from the four correctness gates in the summary.

## What `r01_demo.jl` establishes

**The telemetry works, and works specifically.** A healthy 3-fold run passes all six
gates. Four pathologies each trip their target gate. Two of them — divergences and BFMI —
trip *only* their own, which is asserted rather than eyeballed. The other two do not, and
the runner says so instead of claiming otherwise: R-hat and rank-normalised ESS both read
between-chain variance, so a chain built to break one breaks the other by construction.

**BFMI is the number, not a plausible number.** For an AR(1) energy series, E-BFMI → 2(1−φ).
Measured against that at three coefficients: 0.8%, 0.8%, 5.4% relative error. The check
uses a 20,000-draw series because at 300 draws and φ = 0.97 the ratio holds ~5 effectively
independent points and its expectation is not the ratio of expectations — a property of
the statistic, stated in §5.2 rather than tuned around.

**The fold merge changes no number.** Three per-fold containers concatenated into one:
bit-identical (0 ULP) parameter matrices, score grids and market prices, and the row order
is fold-then-fixture — checked, because a merge that sorted would price fixture *i* with
fixture *j*'s posterior and every downstream number would still look reasonable.

**Zero allocations.** `remaining_intensity!`, `price_live_market!` and the full repricing
loop (18 ticks × 3 markets) all measure 0 bytes under `@allocated`, alongside an
empty-closure baseline that must also read 0. The pre-game kernels still measure 0 through
a container that reached them via `fit_model`.

**The live pricer reduces exactly to the pre-game one.** Under an *identity kernel* — one
bin `[0,1]`, all coefficients zero, so the integral is exactly `1.0` with no rounding —
`Λ_h[k] == λ_home[i,k]` bit for bit, and the live price at an empty kickoff state is
**0 ULP** from the pre-game price across all three markets. Any reassociation in the live
kernel would show up here immediately.

**Persistence is atomic and the sidecar is complete.** `.tmp` → `mv` on every write, a
failed write leaves no partial file, and `meta.json` carries `converged`, `max_rhat`,
`min_ess`, `n_divergent`, `n_folds` and `n_oos_fixtures` — so a 200-directory scan answers
"which of these are usable" without opening one binary.

**A genuine legacy run upgrades.** Not a mock: a real
`BayesianFootball.Experiments.ExperimentResults`, saved the way the legacy runner saves
one, loaded as a `Fit` — folds recovered, `training_config.sampler` flattened, the `time:`
tag parsed back to seconds, the timestamp recovered from the directory name, and
diagnostics computed on load that the legacy container had no field for.

### Not established

That any model fits anything. The posteriors are prior draws with a fixed seed
(`06/l04_parity.jl` §9). No MCMC runs, no database is touched, and the four §5 chains are
*built* to be pathological. The claim is about the lifecycle, not about any price.

The `DataStore` entry points (`fit_model(ds, config)`) are exercised structurally but not
against a live database — they add only the `Data.create_id_boundaries` /
`Features.create_features` / `Data.get_next_matches` calls they forward, and everything
downstream of "here are the folds" goes through the second entry point, which *is* covered.

## Notes carried forward from `06`

The three `src` defects that prototype documented are unchanged and unfixed here:
`DynamicPxGRecombModel`'s OOS extractor cannot be called; three extractors mis-size their
output under multiple chains; nine call sites use a `haskey(::Chains, ::Symbol)` that
MCMCChains 7.7 removed. `l04_ingame_bridge.jl` avoids the third by testing membership
against `names(chain)`. See [`06/README.md`](../06_typed_posterior_latents/README.md).

## Graduating this to `src/`

In dependency order, and each step is independently useful:

1. `l02_convergence.jl` → `src/evaluation/convergence.jl`. It depends on nothing but
   `MCMCChains`, and it can be called on today's `ExperimentResults` immediately.
2. `l01_types.jl` §1-§7 + `l03_engine.jl` → `src/inference/`. `src/training/` and
   `src/experiments/` become the compat bridge, re-exporting from it.
3. `l05_io.jl` → the same module. The atomic-write fix is worth taking on its own even if
   nothing else moves.
4. `l04_ingame_bridge.jl` → `src/models/inplay/`, once `current_development/inplay_scottish`
   settles on a single integrator convention (`:flat` vs `:expo`). Until then the NHPP
   model here reads `inplay_scottish`'s site names so a chain fitted there replays without
   translation.
