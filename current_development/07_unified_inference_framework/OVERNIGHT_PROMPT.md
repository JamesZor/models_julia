# BRIEFING: UNIFIED INFERENCE FRAMEWORK PROTOTYPE (`07_unified_inference_framework`)

> **Objective:** Build, test, and verify the `Unified Inference Framework` in `current_development/07_unified_inference_framework/`. This prototype merges the redundant `src/training/` and `src/experiments/` modules into a cohesive, idiomatic, high-performance architecture (`Fit`, `FitConfig`, `FoldFit`, `InGameFitConfig`, `InGameFit`) supporting both **Pre-Game** and **In-Game (In-Play)** models, directly integrating `05_composable_count_builder` and `06_typed_posterior_latents`, while guaranteeing **100% backward compatibility** for legacy `Experiments` / `Training` callers.

---

## 1. Problem Statement & Motivation

### Why `src/training/` and `src/experiments/` Must Be Unified:
1. **Redundant Module Splitting**: `Training.train` is a pass-through loop calling `Samplers.run_sampler`. `Experiments.run_experiment` wraps `Training.train` with a timer and a save path. This forces awkward double-nested configs (`ExperimentConfig` wrapping `TrainingConfig`) and double-nested results (`ExperimentResults` wrapping `TrainingResults.items`).
2. **Java-Style Verbosity & Dead Fields**: `ExperimentResults` contains dead fields like `vocabulary::Any` (NLP leftover) and requires cumbersome indexing: `exp_results.training_results.items[i][1]` just to read a fold's MCMC chain.
3. **Disconnected In-Game Models**: In-game models (such as the NHPP goal intensity model) depend on pre-game posterior $\lambda$ rates as baseline offsets. Currently, there is no unified contract for a model that consumes a prior pre-game `Fit` / `Latents` to evaluate live match states.
4. **Disconnected Latents & Diagnostics**: MCMC sampling, convergence auditing (R-hat, ESS, divergences), and posterior latent extraction (`06_typed_posterior_latents`) belong together in a single atomic inference lifecycle.

---

## 2. Target Directory & File Structure

Build the following modular files in `current_development/07_unified_inference_framework/`:

```
current_development/07_unified_inference_framework/
├── l01_types.jl            # Clean type hierarchy (Fit, FitConfig, FoldFit, InGameFitConfig, InGameFit, FitMetadata)
├── l02_convergence.jl      # Lightweight 1-file convergence telemetry (R-hat, min ESS, divergences, BFMI)
├── l03_engine.jl           # Unified execution engine: fit_model(task) supporting QueuedNUTS
├── l04_ingame_bridge.jl    # In-game conditional engine: consumes pregame Fit/Latents -> remaining intensity Λ(t) & live pricing
├── l05_io.jl               # Atomic saving (save_fit, load_fit, list_fits) + JSON metadata sidecars + JLD2 upgrade shims
├── l06_compat_bridge.jl    # 100% backward-compat bridge for legacy Experiments / Training / ExperimentResults API
├── r01_demo.jl             # Deterministic verification runner (exercises all gates, benchmarks, and backward compatibility)
└── README.md               # Complete architecture documentation & migration guide
```

---

## 3. Detailed Component Contracts

### 3.1 Type Hierarchy (`l01_types.jl`)

```julia
abstract type AbstractFootballModel end
abstract type AbstractPreGameModel <: AbstractFootballModel end
abstract type AbstractInGameModel  <: AbstractFootballModel end

"""
    FoldFit{C<:Chains, M<:AbstractSplitMetaData}
A single fold's inference outcome: the sampled MCMC chain and split metadata.
"""
struct FoldFit{C<:Chains, M<:AbstractSplitMetaData}
    fold::Int
    chain::C
    meta::M
end

"""
    FitConfig{M<:AbstractPreGameModel, S<:AbstractSplitter, Sam}
Immutable specification for pre-game model inference.
"""
Base.@kwdef struct FitConfig{M<:AbstractFootballModel, S<:AbstractSplitter, Sam}
    name::String
    model::M
    splitter::S
    sampler::Sam
    tags::Vector{String} = String[]
    description::String = ""
    save_dir::String = "./data/fits"
end

"""
    InGameFitConfig{M<:AbstractInGameModel, P<:Union{Fit, AbstractPosteriorLatents}, S<:AbstractSplitter, Sam}
Immutable specification for in-game model inference, explicitly holding its pre-game baseline source.
"""
Base.@kwdef struct InGameFitConfig{M<:AbstractFootballModel, P, S<:AbstractSplitter, Sam}
    name::String
    model::M
    pregame::P                           # Pre-game Fit or CountLatents/RecombLatents
    splitter::S
    sampler::Sam
    tags::Vector{String} = String[]
    description::String = ""
    save_dir::String = "./data/inplay_fits"
end

"""
    FitMetadata
Structured provenance recorded for every run.
"""
struct FitMetadata
    timestamp::DateTime
    elapsed_seconds::Float64
    julia_version::VersionNumber
    n_threads::Int
    git_commit::String
end

"""
    Fit{C<:FitConfig, F<:Vector{<:FoldFit}, L<:Union{Nothing, AbstractPosteriorLatents}, D}
Unified container for pre-game inference results.
Implements IndexLinear so `fit[i]` yields `fit.folds[i]` and `length(fit)` yields number of folds.
"""
struct Fit{C<:FitConfig, F<:Vector{<:FoldFit}, L<:Union{Nothing, AbstractPosteriorLatents}, D}
    config::C
    folds::F
    latents::L
    diagnostics::D                       # ConvergenceSummary
    metadata::FitMetadata
    save_path::String
end

"""
    InGameFit{C<:InGameFitConfig, F<:Vector{<:FoldFit}, P<:AbstractPosteriorLatents, D}
Unified container for in-game inference results.
"""
struct InGameFit{C<:InGameFitConfig, F<:Vector{<:FoldFit}, P<:AbstractPosteriorLatents, D}
    config::C
    folds::F
    pregame_latents::P
    diagnostics::D                       # ConvergenceSummary
    metadata::FitMetadata
    save_path::String
end
```

### 3.2 Convergence Telemetry (`l02_convergence.jl`)
- Collapse the 7-file `src/experiments/diagnostics/` into a single, clean telemetry module.
- `audit_convergence(folds::Vector{<:FoldFit}) -> ConvergenceSummary`
  - Max $\hat{R}$ across all parameters and folds.
  - Min Bulk ESS and Min Tail ESS across all parameters and folds.
  - Divergence count and percentage of total draws.
  - Max tree depth and cap hits.
  - Minimum BFMI across chains.
  - Strict `passed::Bool` status check based on Gate thresholds ($\hat{R} < 1.01$, $\text{ESS} > 400$, $\text{div} < 0.10\%$, $\text{BFMI} > 0.30$).

### 3.3 Execution Engine (`l03_engine.jl`)
- `fit_model(ds::DataStore, config::FitConfig)`:
  - Generates boundaries with metadata via `Data.create_id_boundaries(ds, config.splitter)`.
  - Builds feature sets via `Features.create_features`.
  - Samples chains (supporting both sequential and parallel task flattening for `QueuedNUTSConfig`).
  - Assembles `folds = [FoldFit(i, chains[i], meta[i]) for i in 1:n_splits]`.
  - Automatically audits convergence via `audit_convergence(folds)`.
  - Automatically extracts typed latents (`CountLatents`, `RecombLatents`) via `extract_latents` from `06_typed_posterior_latents`.
  - Returns `Fit(...)`.

### 3.4 In-Game Conditional Bridge (`l04_ingame_bridge.jl`)
- Implements `fit_model(ds::DataStore, config::InGameFitConfig)`:
  - Pulls pre-game baseline $\lambda_h, \lambda_a$ directly from `config.pregame`.
  - Fits or evaluates in-game NHPP goal intensities:
    $$\log \lambda_{\text{side}}(t) = \log \lambda_{\text{side}}^{\text{pre}} + \alpha + \beta \cdot z(t) + \gamma_{\text{state}} \cdot \text{lead/trail} + \gamma_{\text{red}} \cdot \text{man\_adv} + \delta_{\text{time}}[bin]$$
  - Evaluates remaining goal intensity $\Lambda_h(t \to 90), \Lambda_a(t \to 90)$ given match state $(t, g_h, g_a, r_h, r_a)$.
  - Provides zero-allocation live market pricing `price_live_market!(book, Λ_h, Λ_a, state, market)`.

### 3.5 Persistence & Sidecars (`l05_io.jl`)
- `save_fit(fit; path = nothing, quiet = false)`:
  - Atomically saves `results.jld2` (`.tmp` $\to$ `mv`).
  - Writes structured `meta.json` (name, model, sampler, timestamp, elapsed_time, max_rhat, min_ess, divergences, n_folds).
  - Writes `config.json` and serializes `oos_latents.jls`.
- `load_fit(path)`:
  - Deserializes `results.jld2` or transparently upgrades legacy `ExperimentResults` dictionaries.
- `list_fits(dir)`:
  - High-speed directory scanner reading `meta.json`.

### 3.6 Backward Compatibility Bridge (`l06_compat_bridge.jl`)
Guarantees zero downstream breakage for legacy code:
```julia
const Experiments       = current_module()
const Training          = current_module()
const ExperimentResults = Fit
const ExperimentConfig  = FitConfig
const ExperimentTask    = NamedTuple{(:ds, :config), Tuple{Data.DataStore, FitConfig}}

# Property compatibility for legacy callers:
Base.getproperty(f::Fit, s::Symbol) = ...
# - :training_results -> returns a legacy-compatible wrapper where `.items[i]` gives `(f.folds[i].chain, f.folds[i].meta)`
# - :df -> if called on latents, provides legacy DataFrame

# Function shims:
run_experiment(ds::DataStore, cfg::FitConfig) = fit_model(ds, cfg)
run_experiment(task) = fit_model(task.ds, task.config)
save_experiment(res::Fit; kwargs...) = save_fit(res; kwargs...)
load_experiment(path::String) = load_fit(path)
extract_oos_predictions(ds, fit::Fit) = fit.latents
```

### 3.7 Verification Runner (`r01_demo.jl`)
Deterministic, zero-database, zero-MCMC runner verifying:
1. **Pre-Game Pipeline**: `FitConfig` $\to$ `fit_model` $\to$ `Fit` creation.
2. **Ergonomic Access**: `fit[1].chain`, `fit.latents.λ_home`, `fit.diagnostics.max_rhat`.
3. **Convergence Diagnostics**: Correct gate evaluation against synthetic chains.
4. **Typed Latents Integration**: Zero allocations and bit parity for scoring/pricing.
5. **In-Game Pipeline**: `InGameFitConfig` consuming pregame `Fit` $\to$ evaluating remaining intensity $\Lambda(t)$ and pricing live match states with 0 allocations.
6. **Backward Compatibility**: Calling legacy functions (`exp.training_results.items[1][1]`, `extract_oos_predictions`, `save_experiment`, `load_experiment`) works 100% without error.
7. **IO & Metadata**: Atomic save, JSON metadata verification, and round-trip deserialization.

---

## 4. Execution Rules
- **Loader/Runner Architecture**: Code definitions in `l01`–`l06`, execution solely in `r01_demo.jl`.
- **Zero Allocations in Hot Paths**: Ensure in-game remaining intensity and live pricing kernels allocate 0 bytes.
- **Fast & Deterministic**: Run `r01_demo.jl` using synthetic chains/features so it completes in seconds with clean ASCII summary tables and exits 0.
