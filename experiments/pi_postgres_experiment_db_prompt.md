# WORK PACKAGE: POSTGRESQL EXPERIMENT & PORTFOLIO DATABASE ON MCMC-BEAST

<agent_execution_constraints>
STRICT OPERATIONAL DIRECTIVES:
1. YOU MUST EXECUTE ALL WORK DIRECTLY. DO NOT DELEGATE ANY TASK TO A SUBAGENT.
2. Work directly on branch `feat/mcmc-experiments-db` (branched from `feat/pxg-rapm-unified-covariates`).
3. Follow all Julia coding standards, type stability rules, and ReverseDiff safety from `docs/guides/julia_coding_context_for_agents.md`.
4. Run tests after each task and ensure 100% pass rate in `test/runtests.jl`.
5. Once all tests pass 100%, commit, push, checkout `feat/pxg-rapm-unified-covariates`, and merge cleanly.
</agent_execution_constraints>

<infrastructure_context>
- Compute Node: `mcmc-beast` (Docker 29.5.2 installed and running).
- Target Database: PostgreSQL 16 Alpine container named `mcmc_experiments_postgres` on port 5432.
- Database Name: `mcmc_experiments`
- User: `postgres`, Password: `football_mcmc_secure`
- Julia Version: 1.12.6
- Repository: `/home/james/bet_project/BayesianFootball` (Local), `/root/BayesianFootball` (mcmc-beast)
</infrastructure_context>

<work_package_tasks>

## Task 1: Docker PostgreSQL Setup & Management Script
- Create `scripts/setup_experiments_db.sh`:
  * Automates spinning up the persistent Docker PostgreSQL container on `mcmc-beast` (with persistent volume `/root/postgres_experiments_data`).
  * Creates database `mcmc_experiments`.
  * Applies `src/training/inference/db/schema.sql`.

## Task 2: Relational Schema DDL (`src/training/inference/db/schema.sql`)
- Create DDL with the following tables:
  1. `runs`: `run_id UUID PRIMARY KEY`, `experiment_name VARCHAR`, `status VARCHAR`, `git_commit VARCHAR`, `git_branch VARCHAR`, `created_at TIMESTAMP`, `finished_at TIMESTAMP`, `duration_seconds FLOAT`.
  2. `configs`: `config_id UUID PRIMARY KEY REFERENCES runs(run_id)`, `config_hash VARCHAR UNIQUE`, `model_config JSONB`, `split_config JSONB`, `sampler_config JSONB`.
  3. `fold_results`: `fold_id UUID PRIMARY KEY`, `run_id UUID REFERENCES runs(run_id)`, `fold_idx INT`, `r_hat_max FLOAT`, `ess_bulk_min INT`, `ess_tail_min INT`, `divergences INT`, `converged BOOLEAN`, `logloss FLOAT`, `brier FLOAT`, `rps FLOAT`, `runtime_seconds FLOAT`.
  4. `match_latents`: `latent_id BIGSERIAL PRIMARY KEY`, `fold_id UUID REFERENCES fold_results(fold_id)`, `match_id INT`, `mean_lambda_h FLOAT`, `std_lambda_h FLOAT`, `p10_h FLOAT`, `p50_h FLOAT`, `p90_h FLOAT`, `mean_lambda_a FLOAT`, `std_lambda_a FLOAT`, `p10_a FLOAT`, `p50_a FLOAT`, `p90_a FLOAT`, `draws_blob BYTEA`.
  5. `portfolio_runs`: `portfolio_run_id UUID PRIMARY KEY`, `model_run_id UUID REFERENCES runs(run_id)`, `book_spec_hash VARCHAR`, `policy_spec_hash VARCHAR`, `total_return_pct FLOAT`, `flat_roi_pct FLOAT`, `roi_1x2_pct FLOAT`, `max_drawdown_pct FLOAT`, `sharpe_ann FLOAT`, `win_rate FLOAT`, `n_bets INT`, `created_at TIMESTAMP`, `metadata JSONB`.
  6. `portfolio_bets`: `bet_id BIGSERIAL PRIMARY KEY`, `portfolio_run_id UUID REFERENCES portfolio_runs(portfolio_run_id)`, `match_id INT`, `kickoff_date DATE`, `market_family VARCHAR`, `selection VARCHAR`, `odds_close FLOAT`, `stake_fraction FLOAT`, `stake_amount FLOAT`, `pnl FLOAT`.
- Include indexes on `config_hash`, `match_id`, `created_at`, and foreign keys.

## Task 3: Julia Pluggable Storage Backend (`src/training/inference/db_storage.jl`)
- In `src/training/inference/io.jl` and `db_storage.jl`:
  * Implement `AbstractStorageBackend` with:
    - `FileStorage(root_dir::String)` (current behavior, saves results.jld2 + config.json)
    - `PostgresStorage(conn_str::String, experiment_name::String)`
    - `DualStorage(file::FileStorage, db::PostgresStorage)`
  * Implement `save_fit(fit, storage::PostgresStorage)` and `save_fit(fit, storage::DualStorage)`.
  * Compress 3,200 posterior float draws using Zstd into `draws_blob`.
  * Support `load_fit(run_id, storage::PostgresStorage)` reconstructing `Fit` and `CountLatents`.

## Task 4: Portfolio Database Integration (`src/Portfolio/db_storage.jl`)
- In `src/Portfolio/db_storage.jl`:
  * Implement `save_portfolio_db(res::PortfolioResult, run_id::UUID, storage::PostgresStorage)`.
  * Persist high-level summary metrics to `portfolio_runs` and batch insert `bets` rows into `portfolio_bets`.
  * Implement `load_portfolio_db(portfolio_run_id, storage::PostgresStorage)`.

## Task 5: Unit & Migration Test Suite (`test/test_db_storage.jl`)
- Create `test/test_db_storage.jl`:
  * Test `FileStorage`, `PostgresStorage` (mock / SQLite or local PG if available), and `DualStorage`.
  * Verify config hashing deduplication.
  * Verify round-trip compression and decompression of `draws_blob`.
  * Register in `test/runtests.jl` and ensure 100% pass rate across the full test suite.
</work_package_tasks>
