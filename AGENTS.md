# AGENTS.md

> **AI Agent Orchestration, Infrastructure, and Architecture Guide**  
> Guidance for AI agents (Antigravity CLI, Claude Code, etc.) operating across the `BayesianFootball.jl` distributed mesh.

---

## 1. Quick Reference & Core Guides

* **AI Agent Infrastructure & Context Guide:** [`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](file:///home/james/bet_project/BayesianFootball/docs/architecture/ai_agent_infrastructure_and_execution_context.md)
* **Tmux Subagent & Persistent REPL Control Guide:** [`docs/setup/agy_tmux_agent_and_repl_control_guide.md`](file:///home/james/bet_project/BayesianFootball/docs/setup/agy_tmux_agent_and_repl_control_guide.md)
* **Remote Execution & Tmux Protocol:** [`docs/setup/agy_remote_execution_guide.md`](file:///home/james/bet_project/BayesianFootball/docs/setup/agy_remote_execution_guide.md)
* **ReverseDiff AD Performance & Safety Guide:** [`docs/turing_ad_performance_guide.md`](file:///home/james/bet_project/BayesianFootball/docs/turing_ad_performance_guide.md)
* **Julia Coding Context for AI Agents:** [`docs/guides/julia_coding_context_for_agents.md`](file:///home/james/bet_project/BayesianFootball/docs/guides/julia_coding_context_for_agents.md) — language traps, style, Turing API facts, verification ladder. **Read before writing Julia.**

---

## 2. Unified V2 Pipeline Quick-Start (Production Standard)

The production pipeline is organized into 5 standardized, type-stable stages:

```julia
using BayesianFootball
using DataFrames, Dates, ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# 1. Data Layer
ds = Data.load_datastore_cached(Data.ScottishLower())

# 2. Composable Model Builder
model = CountModelBuilder(:poisson_model) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

# 3. Unified Inference Lifecycle & Convergence Gating
fit_cfg = FitConfig(
    name      = "poisson_fit",
    model     = model,
    splitter  = Data.CVConfig(target_seasons = ["24/25"], window_seasons = 3),
    sampler   = NUTSConfig(n_samples = 1_000, n_chains = 4),
    execution = AutoExecution() # Resolves to QueuedExecution or ThreadedExecution
)
fit = fit_model(fit_cfg, ds)

# 4. Unified Evaluation (LogLoss, CRPS, Brier, RPS, ECE vs Closing Odds)
eval_report = evaluate_predictions(fit, ds)

# 5. Zero-Alloc Portfolio & Staking Simulation
spec   = BookSpec(markets = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5)]), shrink = BakerMcHale())
policy = PolicySpec(trust = FlatTrust(0.25), risk = SlateDrawdown(20.0), cap = FixedCap(0.25))
result, books, rep = run_portfolio_simulation(spec, policy, fit, ds.odds, ds)
```

---

## 3. Fast Test Execution Protocol

Always use the fastest test tier appropriate for your change:

```bash
# 1. Single Module Test (FASTEST: 15-20s)
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/unified_portfolio_tests.jl")'

# 2. Concurrent Full Suite (4 worker processes: ~40-45s)
julia --project -t 8 test/run_parallel_tests.jl

# 3. Standard Sequential Suite (Full baseline: ~3.5 min)
julia --project -t 8 test/runtests.jl
```

---

## 4. Infrastructure & Compute Rules

1. **Topology:**
   - **Local Laptop:** Development workstation (`/home/james/bet_project/BayesianFootball`).
   - **Compute Node (`mcmc-beast`):** 16 Physical Cores (32 SMT threads), 64GB RAM (`/root/BayesianFootball`).
   - **Database Host (`archpc:5433`):** PostgreSQL `betdb`.

2. **CPU & Threads:**
   - Always launch Julia with `-t 16` on `mcmc-beast` (or `-t 8` on `archpc`).
   - Always run `using ThreadPinning; pinthreads(:cores)` before starting MCMC chains.
   - Always set `LinearAlgebra.BLAS.set_num_threads(1)` during sampling to prevent CPU oversubscription.
   - Ensure local inference daemons (like `ollama`) remain disabled (`systemctl disable/stop ollama`).

3. **Code Syncing & Cache Safety:**
   - Use `rsync -avz --exclude '.cache/' --exclude 'data/' ...` to push code without clobbering remote data caches.
   - Ensure Point-In-Time (PIT) feature guards accept match-row values: `stamp_ok = (stamp === nothing) || (at === nothing) || (stamp < at)`.

---

## 5. Agent-to-Agent & REPL Tmux Tooling

### Controlling Claude Subagents
```bash
# Send prompt to Claude Code subagent
tmux send-keys -t features:1.1 'Run r00_explore_poisson_models.jl on Fold 1 and report parameter posteriors.' C-m

# Inspect subagent scrollback / status
tmux capture-pane -t features:1.1 -p -S -50
```

### Controlling Persistent Julia REPL (Zero-TTFX)
```bash
# Send code evaluation into warm REPL
tmux send-keys -t scottish_runner:1.1 'include("current_development/scottish_lower/r00_explore_poisson_models.jl")' C-m

# Inspect REPL output
tmux capture-pane -t scottish_runner:1.1 -p -S -60
```

---

## 6. Experiment Database & Config Truth Engine Protocol

See the production guide:
[`docs/guides/experiment_database_and_config_truth_guide.md`](docs/guides/experiment_database_and_config_truth_guide.md).

1. **Keep credentials out of source and output.** Never commit, paste into prompts, or print a
   raw database password or credential-bearing URL. Construct
   `PostgresStorage(experiment_name)` and let it resolve
   `ENV["BF_EXPERIMENTS_DB_URL"]` or libpq's `~/.pgpass`. Its masked `show` method is safe;
   manually printing `storage.conn_str` is not.
2. **Register canonical recipes before execution.** Save production models with `save_model`,
   splitters with `save_splitter`, samplers with `save_sampler`, and the assembled `FitConfig`
   with `save_config`. Register `BookSpec` and `PolicySpec` with `save_book_spec` and
   `save_policy_spec`. Use stable names, descriptions, and tags in `config_registry`; do not
   treat an untracked REPL object as the source of truth.
3. **Preflight expensive sampling.** Before launching MCMC, query `configs.config_hash` for
   the approved inference-recipe hash. If a completed run already exists, load it instead of
   consuming compute. `save_fit` also deduplicates by hash, but that save-time guard cannot
   recover time already spent sampling. Do not confuse the registry component hash with the
   run-deduplication hash; see the guide's hash-domain section.
4. **Persist immutable run addresses.** `run_id = save_fit(fit, db)` returns the model run
   UUID. Pass that UUID to `save_portfolio_db(result, run_id, db; book_spec, policy_spec)` and
   retain the returned portfolio UUID in reports.
5. **Reconstruct typed objects through the API.** Use `load_fit(db, run_integer_id)`,
   `load_fit(db, fit_name)`, or `load_fit(db, run_uuid)` to recover an exact `Fit`, including
   relationally reconstructed `CountLatents`. Use
   `load_portfolio_db(portfolio_run_uuid, db)` to recover the exact `PortfolioResult` from
   `portfolio_artifacts`. Portfolio loading currently requires its UUID; obtain it from
   `portfolio_runs` when starting from the sequential `id`.
