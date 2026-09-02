# AGENTS.md — BayesianFootball.jl

> Canonical guide for AI agents (pi, Claude Code, Antigravity) working in this
> repository. `CLAUDE.md` and `GEMINI.md` are pointers to this file — put new
> guidance **here** so the three harnesses cannot drift apart again.

A Bayesian hierarchical modelling framework for football analytics and betting
market evaluation, in Julia.

---

## 1. Quick reference

* **Julia coding context** — [`docs/guides/julia_coding_context_for_agents.md`](docs/guides/julia_coding_context_for_agents.md) — language traps, style, Turing API facts, verification ladder. **Read before writing Julia.**
* **ReverseDiff AD performance & safety** — [`docs/turing_ad_performance_guide.md`](docs/turing_ad_performance_guide.md)
* **Experiment DB & config truth** — [`docs/guides/experiment_database_and_config_truth_guide.md`](docs/guides/experiment_database_and_config_truth_guide.md)
* **Prototype runner style** — [`docs/prototype_runner_style_guide.md`](docs/prototype_runner_style_guide.md)
* **Agent infrastructure & execution context** — [`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](docs/architecture/ai_agent_infrastructure_and_execution_context.md)
* **Tmux subagent & REPL control** — [`docs/setup/agy_tmux_agent_and_repl_control_guide.md`](docs/setup/agy_tmux_agent_and_repl_control_guide.md)
* **Remote execution & tmux protocol** — [`docs/setup/agy_remote_execution_guide.md`](docs/setup/agy_remote_execution_guide.md)

---

## 2. Architecture

A multi-tier Bayesian predictive and portfolio system. The **Unified V2 stack**
is the production standard:

| Layer | Name | Lives in |
|---|---|---|
| L0 | **Data** — typed PostgreSQL extraction, memory-optimised `DataStore`, vig-removed market math | `src/Data/` |
| L1 | **Bayesian engines** — Turing.jl hierarchical models on compiled ReverseDiff tapes | `src/models/pregame/` |
| L2 | **Unified inference, latents & experiment truth** — convergence audits, `CountLatents`, zero-alloc score tensors, Postgres run tracking | `src/training/inference/`, `src/models/latents/` |
| L3 | **Unified evaluation** — point-in-time pricing, LogLoss, CRPS, Brier, RPS, ECE vs closing odds | `src/evaluation/` |
| L4 | **Zero-alloc portfolio, staking & audit** — `OddsIndex`, `BookWorkspace`, Baker-McHale shrinkage, fractional Kelly, Postgres persistence | `src/Portfolio/` |

> **Numbering caveat.** An older four-layer scheme is still referenced in some
> docs and archive material, where "Layer 2" means GLM calibration
> (`src/Calibration/`) and "Layer 3" means the meta-model
> (`current_development/MetaModels/`). Those components still exist and are
> described below; when a doc says "L2/L3", check which scheme it means. The
> table above is canonical for new work.

### L0 — Data (`src/Data/`)

SQL → `DataStore` via `LibPQ`. Every data domain (Matches, Odds, Betfair,
Stats, Lineups, Incidents) passes a strict **Fetch → Process → QA** three-step
contract defined in `src/Data/fetchers/interfaces.jl`. Tournament segments are
singletons (`ScottishLower`, `Ireland`, …) in `src/Data/fetchers/segments.jl`.
The `Markets` submodule does vig removal, fair odds, and closing-line-movement
math. Cached locally as `.jls` files in `.cache/`.

### L1 — Bayesian engines (`src/models/pregame/`)

Component-driven, assembled from mathematical building blocks:

- **Components** (`components/`): `Interception`, `Dispersion`, `HomeAdvantage`,
  `Dynamics`, `Kappa`, `Copula`, `DixonColes`
- **Team-level engines** (`engines/team_level/`): goals, xg, copula-goals,
  market variants — split `standard/` and `time_decay/`
- **Player-level engines** (`engines/player_level/`): outfield xg, hierarchical
  player, Dixon-Coles variants — split `standard/` and `time_decay/`

Notable exports: `DynamicGoalsModel`, `DynamicXGModel`,
`DynamicCopulaGoalsTimeDecayModel`,
`DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` — the local-intensity
per-strike totals "smile" pillar, which prices O/U via its own intensity
`λ_tot·φ(K)` while 1X2/BTTS/CS use the goals grid
(`src/predictions/score_computation/smile_poisson.jl`).

Every model must implement `Features.required_features(model)` returning a
`Vector{Symbol}` declaring the data features it needs.

### Calibration (`src/Calibration/`)

GLM-based bias correction: `build_l2_training_df` → `train_calibrators` →
`apply_calibrators`. Crucially shifts the **entire MCMC posterior
distribution**, not just scalar probabilities, preserving uncertainty for Kelly
staking.

### Meta model (`current_development/MetaModels/`)

In active development. Blends L1 predictions with market-implied probabilities
via a dynamic Gaussian random-walk mixture: `Q_i = θ_t·p_L1_i + (1-θ_t)·m_i`.
Two engine types: `ConvexMixtureMetaModel`, `AffineCalibrationMetaModel`. See
`docs/archive/meta_model_design.md`.

### Module map

| Module | Responsibility |
|---|---|
| `Data` | SQL extraction, ETL, `Markets` (vig removal, fair odds, CLM) |
| `Features` | `DataStore` → `FeatureSet`. AD-safe flattening, team/time indexing, lineup market valuations. Uses `SplitBoundary` (match-ID pointers, not copies) for memory-efficient temporal folds. Extractors in `src/features/extractors/` |
| `Models` | `CountModelBuilder`, Turing components, master engines |
| `Samplers` | NUTS, MAP, MLE, ADVI wrappers. `QueuedNUTSConfig` flattens K-splits × N-chains into one global queue for full CPU utilisation |
| `Training` | `fit_model`, execution dispatchers, convergence auditing. `Independent(max_concurrent_tasks)` uses queued execution |
| `Experiments` | Task creation, execution, persistence, loading |
| `Predictions` | Typed latents (`CountLatents`), in-place score-grid kernels (`SmileScoreGrid`), PPD generation |
| `Evaluation` | `OddsView`, `evaluate_predictions`, LogLoss, CRPS, RPS, ECE |
| `Portfolio` | `OddsIndex`, `BookWorkspace`, Kelly log-utility allocator, `simulate_portfolio` |
| `Signals` | Betting signal generation and Kelly staking |
| `BackTesting` | `AbstractWealthMetric` (Sharpe, Calmar, Sortino, Burke, Sterling) and `AbstractDistributionalMetric` (Hurdle ROI). `run_backtest` → `generate_tearsheet` |
| `MyDistributions` | `RobustNegativeBinomial`, `FrankCopulaNegBin`, `DixonColes`, `BivariatePoissonDist`, … |

---

## 3. Unified V2 pipeline — the production path

```julia
using BayesianFootball
using DataFrames, Dates, ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# 1. Data layer
ds = Data.load_datastore_cached(Data.ScottishLower())

# 2. Composable model builder
model = CountModelBuilder(:poisson_model) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

# 3. Unified inference lifecycle & convergence gating
fit_cfg = FitConfig(
    name      = "poisson_fit",
    model     = model,
    splitter  = Data.CVConfig(target_seasons = ["24/25"], window_seasons = 3),
    sampler   = NUTSConfig(n_samples = 1_000, n_chains = 4),
    execution = AutoExecution(),   # → QueuedExecution or ThreadedExecution
)
fit = fit_model(fit_cfg, ds)

# 4. Unified evaluation (LogLoss, CRPS, Brier, RPS, ECE vs closing odds)
eval_report = evaluate_predictions(fit, ds)

# 5. Zero-alloc portfolio & staking simulation
spec   = BookSpec(markets = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5)]),
                  shrink  = BakerMcHale())
policy = PolicySpec(trust = FlatTrust(0.25), risk = SlateDrawdown(20.0), cap = FixedCap(0.25))
result, books, rep = run_portfolio_simulation(spec, policy, fit, ds.odds, ds)
```

---

## 4. Prototyping workflow (`current_development/`)

New features are prototyped here **before** moving to `src/`. Always create a
pair:

- **`lXX_*.jl` (loader)** — struct definitions, functions, mathematical logic.
  Acts as a temporary module.
- **`rXX_*.jl` (runner)** — execution: load data, call loader functions, run
  experiments, visualise.
- **`XX`** — two-digit iteration counter; increment when starting a fresh
  approach.

Graduate to `src/` only once the prototype is validated in the runner.

Before creating or substantially refactoring a runner, read
`docs/prototype_runner_style_guide.md`. Runners must stay human-readable
research notebooks with numbered package / configuration / data / model /
training / diagnostic / inference sections; technical checkpoint and
persistence machinery belongs in the paired loader.

Active research streams (kept after validated work graduated to `src/`):

| Directory | Topic |
|---|---|
| `MetaModels/` | Meta-model (see Architecture) |
| `match_day_inference/` | Operational match-day fixture inference + runners |
| `match_inplay_explore/` | In-play intensity / NHPP / market-inverse research |
| `basic_hedging/` | Portfolio-Kelly + partial-hedge staking |
| `ab_test_dixon_coles/`, `ab_test_fullposition/` | L1 engine A/B harnesses |
| `betfair_closing_line/`, `order_book/`, `bayesian_layer_2/` | CLV, order-book, Bayesian calibration/staking |

---

## 5. AD-safety — critical for Turing models

- All feature vectors must be `Float64` or `Int`. **No `missing` inside
  `@model`.**
- Use `coalesce.(data, NaN)` for missing xG, then `findall(!isnan, ...)` to
  split the xG vs goals likelihood routes.
- **No `if`/`else` or `for` loops inside `@model`** — use binary masks and
  broadcast arithmetic.
- Use `clamp` and `Turing.@addlogprob! -Inf` to reject numerically unstable
  samples gracefully.
- The feature builder does all conditional logic; the `@model` receives pure
  `Float64` vectors.

Target: compiled `ReverseDiff.GradientTape` at ~0.64 ms per gradient
evaluation. Details in `docs/turing_ad_performance_guide.md`.

---

## 6. Extension recipes

**New league / segment** — edit only `src/Data/fetchers/segments.jl`: define a
struct subtyping `DataTournemantSegment` and implement
`tournament_ids(::MyLeague) = [id1, id2]`.

**New model component** — three steps in `src/models/pregame/components/`:
1. Define a `Config` struct subtyping the relevant abstract (e.g.
   `AbstractDynamicsConfig`).
2. Write the Turing `@model` builder returning the expected NamedTuple.
3. Write `extract_parameters` to pull variables from `MCMCChains.Chains`.

**New feature extractor** — add
`add_feature!(F_data::Dict, ::Val{:feature_name}, ordered_ids, team_map, ds::DataStore)`
in `src/features/extractors/`, then return the symbol from the model's
`required_features`.

**New backtesting metric** — subtype `AbstractWealthMetric` or
`AbstractDistributionalMetric` in `src/backtesting/metrics/`. See
`hurdle_roi.jl` for the distributional pattern, `implentations/` for wealth
metrics.

**New L2 shift model** — subtype `AbstractLayerTwoModel`, implement
`fit_calibrator(model, data, config)` and
`apply_calibration(fitted_model, new_data)` in `src/Calibration/shift_models/`.

---

## 7. Test execution — pick the fastest tier that covers your change

```bash
# 1. Single module (FASTEST, 15-20s)
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/unified_portfolio_tests.jl")'

# 2. Concurrent full suite (4 workers, ~40-45s)
julia --project -t 8 test/run_parallel_tests.jl

# 3. Standard sequential suite (full baseline, ~3.5 min)
julia --project -t 8 test/runtests.jl
```

Suites live in `test/` (`data_tests.jl`, `features_tests.jl`,
`pregame_tests.jl`, …).

---

## 8. Infrastructure & compute rules

**Topology**

| Host | Role |
|---|---|
| Local laptop | Development — `/home/james/bet_project/BayesianFootball` |
| `mcmc-beast` | Compute — 16 physical cores (32 SMT), 64 GB RAM, `/root/BayesianFootball` |
| `archpc:5433` | PostgreSQL `betdb` |

**CPU & threads**

- Launch Julia with `-t 16` on `mcmc-beast`, `-t 8` on `archpc`.
- Always `using ThreadPinning; pinthreads(:cores)` before starting MCMC chains.
- Always `LinearAlgebra.BLAS.set_num_threads(1)` during sampling, to prevent
  CPU oversubscription.
- Keep local inference daemons disabled (`systemctl disable/stop ollama`).

**Code syncing & cache safety**

- `rsync -avz --exclude '.cache/' --exclude 'data/' ...` to push code without
  clobbering remote data caches.
- Point-in-time feature guards must accept match-row values:
  `stamp_ok = (stamp === nothing) || (at === nothing) || (stamp < at)`.

---

## 9. Remote execution protocol

**Never run heavy Turing models locally on the laptop.** The proven loop:

1. Write/edit the model (`lXX_*.jl`) and runner (`rXX_*.jl`) locally.
2. Commit and push from the local branch (only with commit permission).
3. `C-b 2` to `scottish_runner:1.1`, then `git pull origin <branch>` to sync
   code on beast.
4. `C-b 0` (or `C-b 3`) to `scottish_runner:1.1`, then
   `include("path/to/runner.jl")` into the persistent Julia REPL.
5. Monitor with `tmux capture-pane -p -t scottish_runner:1.1 -S -50`.

Full detail: `docs/setup/agy_remote_execution_guide.md`.

### Tmux tooling

```bash
# Send code into the warm REPL (zero TTFX)
tmux send-keys -t scottish_runner:1.1 'include("current_development/scottish_lower/r00_explore_poisson_models.jl")' C-m
tmux capture-pane -t scottish_runner:1.1 -p -S -60

# Drive a subagent pane
tmux send-keys -t features:1.1 'Run r00_explore_poisson_models.jl on Fold 1 and report parameter posteriors.' C-m
tmux capture-pane -t features:1.1 -p -S -50
```

> Session and window indices here are not version-controlled and drift. Verify
> with `tmux ls` before sending keys rather than trusting these literally.

---

## 10. Experiment database & config truth protocol

See `docs/guides/experiment_database_and_config_truth_guide.md`.

1. **Keep credentials out of source and output.** Never commit, paste into
   prompts, or print a raw database password or credential-bearing URL.
   Construct `PostgresStorage(experiment_name)` and let it resolve
   `ENV["BF_EXPERIMENTS_DB_URL"]` or libpq's `~/.pgpass`. Its masked `show` is
   safe; printing `storage.conn_str` is not.
2. **Register canonical recipes before execution.** Save models with
   `save_model`, splitters with `save_splitter`, samplers with `save_sampler`,
   the assembled `FitConfig` with `save_config`; `BookSpec` and `PolicySpec`
   with `save_book_spec` / `save_policy_spec`. Use stable names, descriptions
   and tags in `config_registry` — an untracked REPL object is not the source
   of truth.
3. **Preflight expensive sampling.** Before launching MCMC, query
   `configs.config_hash` for the approved inference-recipe hash. If a completed
   run exists, load it instead of consuming compute. `save_fit` deduplicates by
   hash, but that save-time guard cannot recover time already spent sampling.
   Do not confuse the registry component hash with the run-deduplication hash —
   see the guide's hash-domain section.
4. **Persist immutable run addresses.** `run_id = save_fit(fit, db)` returns the
   model-run UUID. Pass it to
   `save_portfolio_db(result, run_id, db; book_spec, policy_spec)` and retain
   the returned portfolio UUID in reports.
5. **Reconstruct typed objects through the API.** `load_fit(db, run_integer_id)`,
   `load_fit(db, fit_name)` or `load_fit(db, run_uuid)` recovers an exact `Fit`
   including relationally reconstructed `CountLatents`.
   `load_portfolio_db(portfolio_run_uuid, db)` recovers the exact
   `PortfolioResult` from `portfolio_artifacts`. Portfolio loading requires its
   UUID — obtain it from `portfolio_runs` when starting from the sequential
   `id`.
