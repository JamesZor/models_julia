# BayesianFootball.jl

A sophisticated Bayesian hierarchical modeling framework for football (soccer) analytics and betting market evaluation.

## 🚀 Project Overview

`BayesianFootball` is a multi-tier Bayesian predictive and portfolio system built in Julia:
- **Layer 0 (Data):** Strictly typed PostgreSQL extraction, memory-optimized DataStores, and vig-removed market math.
- **Layer 1 (Bayesian Engines):** Fast Turing.jl hierarchical models using ReverseDiff compiled tapes for team ratings, player dynamics, and score distributions.
- **Layer 2 (Unified Inference & Latents):** Automated MCMC convergence audits, typed posterior latents (`CountLatents`), and zero-allocation score tensor kernels.
- **Layer 3 (Unified Evaluation):** Point-in-time prediction pricing, Log-Loss, CRPS, Brier score, RPS, and Expected Calibration Error (ECE) against market closing odds.
- **Layer 4 (Zero-Alloc Portfolio & Staking):** Fast odds indexing (`OddsIndex`), fold-level pre-allocated workspaces (`BookWorkspace`), Baker-McHale parameter shrinkage, and fractional Kelly staking simulation with automated bankroll convergence gating.

---

## 🏗️ Core Architecture (The Unified V2 Stack)

### 1. Composable Count Builder (`src/models/pregame/builder/`)
- **`CountModelBuilder`**: Assemble models modularly with generic `add!` dispatches.
- **`PoissonCountModel` & `NegBinCountModel`**: Concrete, statically-typed prediction engines compiled into $O(1)$ ReverseDiff tapes.
- **Mathematical Lego Blocks**: Interceptions (`GlobalInterception`, `HierarchicalInterception`), Dynamics (`TimeDecayDynamics`, `GRWDynamics`, `MultiScaleGRW`), Home Advantage (`GlobalHomeAdvantage`, `SingleHomeAdvantage`), and Covariates (`WealthCovariate`, `DistanceCovariate`).

### 2. Typed Posterior Latents & Score Grids (`src/models/latents/`, `src/predictions/score_grids/`)
- **`CountLatents`**: Typed container for $\lambda_{\text{home}}, \lambda_{\text{away}}$ posterior draws.
- **`SmileScoreGrid`**: Zero-allocation in-place score grid kernels for Poisson, Negative Binomial, and Dixon-Coles distributions.

### 3. Unified Inference Lifecycle (`src/training/inference/`)
- **`Fit` Container**: Atomic unit of an estimated model, carrying `config`, `folds`, `latents`, `diagnostics`, and `metadata`.
- **`fit_model(FitConfig, ds)`**: End-to-end inference orchestrator with automatic ReverseDiff tape compilation.
- **Automated Convergence Diagnostics (`ConvergenceSummary`)**: Hard audit gates for $\hat{R} < 1.05$, Bulk/Tail ESS, and divergences.
- **Execution Strategies**: `AutoExecution()`, `QueuedExecution()`, `ThreadedExecution()`, `SequentialExecution()`.

### 4. Unified Evaluation Framework (`src/evaluation/`)
- **`OddsView` & `EvaluationWorkspace`**: Zero-copy dense views over odds matrices with strict Point-In-Time (`stamp < kickoff`) assertions.
- **`evaluate_predictions(fit, ds)`**: Prices match probabilities across the posterior score grid and scores Log-Loss, CRPS, Brier score, RPS, and calibration curves vs market closing odds.
- **Convergence Refusal**: Evaluators refuse unconverged fits to prevent misleading benchmarks.

### 5. Unified Portfolio & Staking (`src/Portfolio/`)
- **`OddsIndex`**: O(1) indexed lookups for match markets, eliminating expensive full-frame scans.
- **`BookWorkspace`**: One pre-allocated matrix and probability buffer per fold, enabling zero-allocation Kelly allocation sweeps.
- **`simulate_portfolio` & `run_portfolio_simulation`**: Simulates bankroll trajectories under fractional Kelly staking, slate drawdown caps, and commission modeling.
- **Convergence Gating**: Unconverged models throw a `ConvergenceRefusal` before bankroll capital is risked.

---

## 🛠️ Modern Workflow Example

```julia
using BayesianFootball
using DataFrames, Dates, ThreadPinning

# 1. CPU thread pinning & BLAS isolation
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# 2. Load cached tournament data
ds = Data.load_datastore_cached(Data.ScottishLower())

# 3. Assemble model with Composable Count Builder
model = CountModelBuilder(:poisson_timedecay_2425) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |> # Team-level time decay
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

# 4. Train via Unified Inference Engine
fit_cfg = FitConfig(
    name      = "poisson_2425",
    model     = model,
    splitter  = Data.CVConfig(target_seasons = ["24/25"], window_seasons = 3),
    sampler   = NUTSConfig(n_samples = 1_000, n_chains = 4, target_accept = 0.85),
    execution = AutoExecution()
)
fit = fit_model(fit_cfg, ds)

# 5. Evaluate forecast accuracy vs closing odds
eval_report = evaluate_predictions(fit, ds)
println(eval_report)

# 6. Simulate fractional Kelly portfolio with risk policy
spec = BookSpec(
    markets   = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS()]),
    price     = DeArb(),
    allocator = KellyLogUtility(),
    shrink    = BakerMcHale()
)
policy = PolicySpec(
    trust     = FlatTrust(0.25),      # 25% Quarter Kelly
    risk      = SlateDrawdown(20.0),  # 20% max slate risk
    cap       = FixedCap(0.25)        # 25% max bankroll exposure
)

result, books, rep = run_portfolio_simulation(spec, policy, fit, ds.odds, ds)
display(portfolio_report(result))
```

---

## 🧪 Testing Suite & Test Runners

Tests use Julia's `@testset` framework.

### 1. Fast Concurrent Test Runner (~40s)
Dispatches 4 worker processes concurrently across CPU cores:
```bash
julia --project -t 8 test/run_parallel_tests.jl
```

### 2. Standard Sequential Runner (~3.5 min)
```bash
julia --project -t 8 test/runtests.jl
```

### 3. Targeted Sub-Suite Test (~15s)
```bash
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/unified_portfolio_tests.jl")'
```

---

## 📂 Module Map

- **`Data`**: SQL extraction, ETL, and `Markets` module (vig removal, fair odds, CLM).
- **`Features`**: AD-safe data flattening, team/time indexing, and lineup market valuations.
- **`Models`**: Composable count builder (`CountModelBuilder`), Turing model components, and master engines.
- **`Samplers`**: Sampler configs for NUTS, ADVI, MAP, and `QueuedNUTSConfig`.
- **`Training`**: Unified inference engine (`fit_model`), execution dispatchers, and convergence auditing.
- **`Predictions`**: Typed latents (`CountLatents`) and in-place score grid kernels (`SmileScoreGrid`).
- **`Evaluation`**: Unified evaluation framework (`OddsView`, `evaluate_predictions`, LogLoss, CRPS, RPS, ECE).
- **`Portfolio`**: Fast odds indexing (`OddsIndex`), `BookWorkspace`, Kelly log-utility allocator, and `simulate_portfolio`.
