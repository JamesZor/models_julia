# BayesianFootball.jl

A sophisticated Bayesian hierarchical modeling framework for football (soccer) analytics and betting market evaluation.

## 🚀 Project Overview

`BayesianFootball` is a multi-tier Bayesian predictive and portfolio system built in Julia:
- **Layer 0 (Data):** Strictly typed PostgreSQL extraction, memory-optimized DataStores, and vig-removed market math.
- **Layer 1 (Bayesian Engines):** Fast Turing.jl hierarchical models using ReverseDiff compiled tapes for team ratings, player dynamics, and score distributions.
- **Layer 2 (Unified Inference, Latents & Experiment Truth):** Automated MCMC convergence audits, typed posterior latents (`CountLatents`), zero-allocation score tensor kernels, and PostgreSQL-backed run tracking plus canonical configuration discovery.
- **Layer 3 (Unified Evaluation):** Point-in-time prediction pricing, Log-Loss, CRPS, Brier score, RPS, and Expected Calibration Error (ECE) against market closing odds.
- **Layer 4 (Zero-Alloc Portfolio, Staking & Audit):** Fast odds indexing (`OddsIndex`), fold-level pre-allocated workspaces (`BookWorkspace`), Baker-McHale parameter shrinkage, fractional Kelly staking, and queryable PostgreSQL portfolio/trade persistence.

---

## 🏗️ Core Architecture (The Unified V2 Stack)

### 1. Composable Count Builder (`src/models/pregame/builder/`)
- **`CountModelBuilder`**: Assemble models modularly with generic `add!` dispatches.
- **`PoissonCountModel` & `NegBinCountModel`**: Concrete, statically-typed prediction engines compiled into $O(1)$ ReverseDiff tapes.
- **Mathematical Lego Blocks**: Interceptions (`GlobalInterception`, `HierarchicalInterception`), Dynamics (`TimeDecayDynamics`, `GRWDynamics`, `MultiScaleGRW`), Home Advantage (`GlobalHomeAdvantage`, `SingleHomeAdvantage`), and Covariates (`WealthCovariate`, `DistanceCovariate`).

### 2. Typed Posterior Latents & Score Grids (`src/models/latents/`, `src/predictions/score_grids/`)
- **`CountLatents`**: Typed container for $\lambda_{\text{home}}, \lambda_{\text{away}}$ posterior draws.
- **`SmileScoreGrid`**: Zero-allocation in-place score grid kernels for Poisson, Negative Binomial, and Dixon-Coles distributions.

### 3. Unified Inference Lifecycle & Experiment Truth (`src/training/inference/`)
- **`Fit` Container**: Atomic unit of an estimated model, carrying `config`, `folds`, `latents`, `diagnostics`, and `metadata`.
- **`fit_model(FitConfig, ds)`**: End-to-end inference orchestrator with automatic ReverseDiff tape compilation.
- **Automated Convergence Diagnostics (`ConvergenceSummary`)**: Hard audit gates for $\hat{R} < 1.05$, Bulk/Tail ESS, and divergences.
- **Execution Strategies**: `AutoExecution()`, `QueuedExecution()`, `ThreadedExecution()`, `SequentialExecution()`.
- **PostgreSQL Experiment Tracking**: `PostgresStorage` stores queryable runs, fold diagnostics, match latents, and exact `Fit` artefacts; `DualStorage` also keeps an atomic filesystem copy.
- **Config Truth Engine**: `config_registry`, `save_model`, `save_config`, `search_configs`, and `show_config` provide named, tagged, hash-addressed recipes shared across machines. See the [database guide](docs/guides/experiment_database_and_config_truth_guide.md).

### 4. Unified Evaluation Framework (`src/evaluation/`)
- **`OddsView` & `EvaluationWorkspace`**: Zero-copy dense views over odds matrices with strict Point-In-Time (`stamp < kickoff`) assertions.
- **`evaluate_predictions(fit, ds)`**: Prices match probabilities across the posterior score grid and scores Log-Loss, CRPS, Brier score, RPS, and calibration curves vs market closing odds.
- **Convergence Refusal**: Evaluators refuse unconverged fits to prevent misleading benchmarks.

### 5. Unified Portfolio, Staking & Audit (`src/Portfolio/`)
- **`OddsIndex`**: O(1) indexed lookups for match markets, eliminating expensive full-frame scans.
- **`BookWorkspace`**: One pre-allocated matrix and probability buffer per fold, enabling zero-allocation Kelly allocation sweeps.
- **`simulate_portfolio` & `run_portfolio_simulation`**: Simulates bankroll trajectories under fractional Kelly staking, slate drawdown caps, and commission modeling.
- **PostgreSQL Audit Trail**: `save_portfolio_db` stores headline ROI/risk metrics, individual bets, and an exact `PortfolioResult` artefact linked to the model run UUID.
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

# 4. Define the Unified Inference recipe
fit_cfg = FitConfig(
    name      = "poisson_2425",
    model     = model,
    splitter  = Data.CVConfig(target_seasons = ["24/25"], window_seasons = 3),
    sampler   = NUTSConfig(n_samples = 1_000, n_chains = 4, target_accept = 0.85),
    execution = AutoExecution()
)

# 5. Register the canonical recipe before scheduling compute.
# Credentials resolve from BF_EXPERIMENTS_DB_URL or ~/.pgpass.
db = PostgresStorage("scottish_lower_2426")
ensure_schema!(db)
save_model(db, "poisson_timedecay_2425", model; tags = ["production"])
save_splitter(db, "split_2425", fit_cfg.splitter; tags = ["walkforward"])
save_sampler(db, "nuts_4x1000", fit_cfg.sampler; tags = ["production"])
save_config(db, "poisson_2425", fit_cfg; tags = ["production"])

# 6. Train and persist the queryable experiment record
fit = fit_model(fit_cfg, ds)
run_id = save_fit(fit, db)

# 7. Evaluate forecast accuracy vs closing odds
eval_report = evaluate_predictions(fit, ds)
println(eval_report)

# 8. Simulate fractional Kelly portfolio with risk policy
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
save_book_spec(db, "closing_main", spec; tags = ["production"])
save_policy_spec(db, "quarter_kelly", policy; tags = ["production"])

result, books, rep = run_portfolio_simulation(spec, policy, fit, ds.odds, ds)
portfolio_run_id = save_portfolio_db(
    result,
    run_id,
    db;
    book_spec = spec,
    policy_spec = policy,
)
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
