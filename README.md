# BayesianFootball.jl

A sophisticated Bayesian hierarchical modeling framework for football (soccer) analytics, market evaluation, and portfolio management in Julia.

---

## 🚀 Overview & Key Capabilities

`BayesianFootball.jl` provides an end-to-end Bayesian quantitative workflow:
* **Layer 0 — Memory-Optimized DataStore**: Concurrent SQL extraction via `LibPQ`, strict typed schemas (`InlineStrings`), and vig-removed market math.
* **Layer 1 — Composable Count Builder & Master Engines**: Mathematical Lego blocks assembled modularly into `PoissonCountModel` or `NegBinCountModel` with $O(1)$ ReverseDiff compiled tapes.
* **Layer 2 — Unified Inference, Latents & Experiment Truth (`Fit`)**: Multi-threaded MCMC sampling (NUTS/ADVI), automated convergence auditing ($\hat{R}$, ESS, divergences), atomic disk persistence, and PostgreSQL-backed run tracking plus canonical configuration discovery.
* **Layer 3 — Unified Evaluation Framework**: Zero-copy `OddsView` over match markets with bit-identical Log-Loss, CRPS, Brier score, RPS, and Expected Calibration Error (ECE) against market closing prices.
* **Layer 4 — Zero-Allocation Portfolio, Staking & Audit**: O(1) indexed lookups (`OddsIndex`), fold-level pre-allocated workspaces (`BookWorkspace`), Baker-McHale parameter shrinkage, fractional Kelly staking, and queryable PostgreSQL portfolio/trade persistence.

---

## ⚡ Quick-Start: End-to-End Pipeline

Here is how to train and simulate a team-level time-decay Poisson model for the **24/25** season using the unified v2 stack:

```julia
using BayesianFootball
using DataFrames, Dates, ThreadPinning

# 1. Thread topology & BLAS isolation
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# 2. Load cached tournament data
ds = Data.load_datastore_cached(Data.ScottishLower())

# 3. Assemble model with Composable Count Builder
model = CountModelBuilder(:poisson_timedecay_2425) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |> # Team-level exponential time decay
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

# 4. Define the Unified Inference recipe
fit_cfg = FitConfig(
    name      = "poisson_2425",
    model     = model,
    splitter  = Data.CVConfig(target_seasons = ["24/25"], window_seasons = 3),
    sampler   = NUTSConfig(n_samples = 1_000, n_chains = 4, target_accept = 0.85),
    execution = AutoExecution() # Resolves to QueuedExecution or ThreadedExecution
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
    risk      = SlateDrawdown(20.0),  # 20% max slate risk budget
    cap       = FixedCap(0.25)        # 25% max simultaneous exposure
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

## 🏗️ Architecture Layers

### 🗄️ Layer 0: Data Module (`src/Data/`)
The foundational data layer that handles the extraction, transformation, and validation of raw PostgreSQL data into memory-optimized Julia `DataFrames`.
* **`DataStore`**: Strictly typed container holding domain DataFrames (`matches`, `odds`, `betfair_odds`, `statistics`, `lineups`, `incidents`).
* **`Markets`**: Implied probabilities, vig removal algorithms, fair odds calculations, and closing line movement (CLM).
* **Fetch $\to$ Process $\to$ QA**: 3-step contract ensuring type safety and logical constraints before data reaches Bayesian models.

### 🧠 Layer 1: Composable Count Builder & Models (`src/models/`)
* **`CountModelBuilder`**: Assemble models modularly with generic `add!` dispatches.
* **Master Engines**:
  * `PoissonCountModel` & `NegBinCountModel`: Composable count models with compiled ReverseDiff gradient tapes.
  * `DynamicPxGRecombModel`: Multi-task proxy xG and open-play goals engine co-training with squad market wealth ($\Delta W$).
  * `DynamicCopulaGoalsModel`: Frank Copula joint distribution over Negative Binomial marginals.
* **Component Blocks**:
  * **Interceptions**: `GlobalInterception`, `HierarchicalInterception`, `ConstantInterception`.
  * **Dynamics**: `TimeDecayDynamics` (exponential decay), `GRWDynamics`, `MultiScaleGRW`.
  * **Home Advantage**: `GlobalHomeAdvantage`, `SingleHomeAdvantage`, `HierarchicalHomeAdvantage`.

### 🔄 Layer 2: Unified Inference, Latents & Experiment Truth (`src/training/`, `src/models/latents/`)
* **`Fit`**: The atomic result of a trained model containing configuration, fold results, posterior latents, convergence audit diagnostics, and metadata.
* **`fit_model(FitConfig, ds)`**: End-to-end inference orchestrator supporting `AutoExecution`, `QueuedExecution`, `ThreadedExecution`, and `SequentialExecution`.
* **Automated Convergence Audit (`ConvergenceSummary`)**: Evaluates $\hat{R} < 1.05$, ESS thresholds, and MCMC divergences.
* **Typed Latents (`CountLatents`)**: Structured matrices for $\lambda_{\text{home}}, \lambda_{\text{away}}$ feeding zero-allocation score kernels (`SmileScoreGrid`).
* **PostgreSQL Experiment Tracking**: `PostgresStorage` stores queryable runs, fold diagnostics, match latents, and exact `Fit` artefacts; `DualStorage` also keeps an atomic filesystem copy.
* **Config Truth Engine**: `config_registry`, `save_model`, `save_config`, `search_configs`, and `show_config` provide named, tagged, hash-addressed recipes shared across machines. See the [experiment database guide](docs/guides/experiment_database_and_config_truth_guide.md).

### 📊 Layer 3: Unified Evaluation (`src/evaluation/`)
* **`OddsView`**: Zero-copy dense view over odds matrices with strict Point-In-Time (`stamp < kickoff`) assertion guards.
* **`evaluate_predictions(fit, ds)`**: Prices match probabilities across the posterior score grid and computes:
  * **Log-Loss** (Cross-Entropy vs Market)
  * **Continuous Ranked Probability Score (CRPS)**
  * **Ranked Probability Score (RPS)** & **Brier Score**
  * **Expected Calibration Error (ECE)** & Reliability Diagrams
* **Convergence Refusal**: Automatically prevents evaluating unconverged fits.

### 💰 Layer 4: Zero-Allocation Portfolio, Staking & Audit (`src/Portfolio/`)
* **`OddsIndex`**: O(1) indexed lookups for match markets, eliminating expensive full-frame scans.
* **`BookWorkspace`**: One pre-allocated matrix and probability buffer per fold, enabling zero-allocation Kelly allocation sweeps.
* **`simulate_portfolio` & `run_portfolio_simulation`**: Simulates bankroll trajectories under fractional Kelly staking, slate drawdown caps, and commission modeling.
* **PostgreSQL Audit Trail**: `save_portfolio_db` stores headline ROI/risk metrics, individual bets, and an exact `PortfolioResult` artefact linked to the model run UUID.
* **Convergence Gating**: Unconverged models throw a `ConvergenceRefusal` before bankroll capital is risked.

---

## 🧪 Testing & Test Runners

`BayesianFootball.jl` has a comprehensive test suite (2,460+ passing tests).

### 1. Fast Concurrent Test Runner (~40s)
Dispatches 4 worker processes concurrently across available CPU cores:
```bash
julia --project -t 8 test/run_parallel_tests.jl
```

### 2. Standard Sequential Test Runner (~3.5 min)
```bash
julia --project -t 8 test/runtests.jl
```

### 3. Targeted Single Suite Run (~15s)
```bash
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/unified_portfolio_tests.jl")'
```

---

## 🖥️ Compute & Infrastructure Setup

* **Local Dev Machine (`archpc`)**: 8 Physical Cores / 16 SMT threads, PostgreSQL `betdb` on port 5433.
* **Compute Node (`mcmc-beast`)**: 16 Physical Cores (32 SMT threads), 64GB RAM.
* **Threading Rules**:
  * Always pin Julia to physical cores: `using ThreadPinning; pinthreads(:cores)`.
  * Always isolate BLAS threads during sampling: `LinearAlgebra.BLAS.set_num_threads(1)`.
