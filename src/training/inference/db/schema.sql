-- PostgreSQL schema for durable inference experiments and portfolio simulations.
-- Idempotent so scripts/setup_experiments_db.sh can also be used as a migration runner.

CREATE TABLE IF NOT EXISTS runs (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL UNIQUE,
    name VARCHAR NOT NULL,
    experiment_name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    git_commit VARCHAR NOT NULL,
    git_branch VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    duration_seconds DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS configs (
    config_id UUID PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    config_hash VARCHAR NOT NULL UNIQUE,
    model_config JSONB NOT NULL,
    split_config JSONB NOT NULL,
    sampler_config JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS fold_results (
    fold_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    fold_idx INT NOT NULL,
    r_hat_max DOUBLE PRECISION,
    ess_bulk_min INT,
    ess_tail_min INT,
    divergences INT NOT NULL DEFAULT 0,
    converged BOOLEAN NOT NULL,
    logloss DOUBLE PRECISION,
    brier DOUBLE PRECISION,
    rps DOUBLE PRECISION,
    runtime_seconds DOUBLE PRECISION,
    n_matches INT,
    first_match_date DATE,
    last_match_date DATE,
    UNIQUE (run_id, fold_idx)
);

CREATE TABLE IF NOT EXISTS match_latents (
    latent_id BIGSERIAL PRIMARY KEY,
    fold_id UUID NOT NULL REFERENCES fold_results(fold_id) ON DELETE CASCADE,
    match_id INT NOT NULL,
    mean_lambda_h DOUBLE PRECISION NOT NULL,
    std_lambda_h DOUBLE PRECISION NOT NULL,
    p10_h DOUBLE PRECISION NOT NULL,
    p50_h DOUBLE PRECISION NOT NULL,
    p90_h DOUBLE PRECISION NOT NULL,
    mean_lambda_a DOUBLE PRECISION NOT NULL,
    std_lambda_a DOUBLE PRECISION NOT NULL,
    p10_a DOUBLE PRECISION NOT NULL,
    p50_a DOUBLE PRECISION NOT NULL,
    p90_a DOUBLE PRECISION NOT NULL,
    draws_blob BYTEA NOT NULL,
    UNIQUE (fold_id, match_id)
);

CREATE TABLE IF NOT EXISTS portfolio_runs (
    id BIGSERIAL PRIMARY KEY,
    portfolio_run_id UUID NOT NULL UNIQUE,
    model_run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    book_spec_hash VARCHAR NOT NULL,
    policy_spec_hash VARCHAR NOT NULL,
    total_return_pct DOUBLE PRECISION NOT NULL,
    flat_roi_pct DOUBLE PRECISION NOT NULL,
    roi_1x2_pct DOUBLE PRECISION,
    max_drawdown_pct DOUBLE PRECISION NOT NULL,
    sharpe_ann DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    n_bets INT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS portfolio_bets (
    bet_id BIGSERIAL PRIMARY KEY,
    portfolio_run_id UUID NOT NULL REFERENCES portfolio_runs(portfolio_run_id) ON DELETE CASCADE,
    match_id INT NOT NULL,
    kickoff_date DATE NOT NULL,
    market_family VARCHAR NOT NULL,
    selection VARCHAR NOT NULL,
    odds_close DOUBLE PRECISION NOT NULL,
    stake_fraction DOUBLE PRECISION NOT NULL,
    stake_amount DOUBLE PRECISION NOT NULL,
    pnl DOUBLE PRECISION NOT NULL
);

-- Named, versioned recipes are the configuration single source of truth. The JSON summary is
-- inspectable from SQL, while config_blob is the exact compressed value used to recreate it.
CREATE TABLE IF NOT EXISTS config_registry (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    experiment_name VARCHAR NOT NULL,
    config_type VARCHAR NOT NULL,
    description VARCHAR NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    config_json JSONB NOT NULL,
    config_blob BYTEA NOT NULL,
    config_hash VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    UNIQUE (experiment_name, name)
);

-- Exact binary artefacts complement, rather than replace, the queryable relational rows.
-- Fit contains chain/config types that cannot be faithfully reconstructed from display JSON.
-- PortfolioResult likewise contains daily states and bootstrap details beyond its headline row.
CREATE TABLE IF NOT EXISTS fit_artifacts (
    run_id UUID PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    fit_blob BYTEA NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_artifacts (
    portfolio_run_id UUID PRIMARY KEY REFERENCES portfolio_runs(portfolio_run_id) ON DELETE CASCADE,
    result_blob BYTEA NOT NULL,
    book_spec_blob BYTEA,
    policy_spec_blob BYTEA
);

-- Add lookup IDs when migrating a database created by the UUID-only v1 schema. Existing UUID
-- keys remain unique stable foreign-key targets. Fresh databases use BIGSERIAL primary keys.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS id BIGSERIAL;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS name VARCHAR;
ALTER TABLE portfolio_runs ADD COLUMN IF NOT EXISTS id BIGSERIAL;
ALTER TABLE config_registry ADD COLUMN IF NOT EXISTS id BIGSERIAL;
ALTER TABLE fold_results ADD COLUMN IF NOT EXISTS n_matches INT;
ALTER TABLE fold_results ADD COLUMN IF NOT EXISTS first_match_date DATE;
ALTER TABLE fold_results ADD COLUMN IF NOT EXISTS last_match_date DATE;
ALTER TABLE portfolio_artifacts ADD COLUMN IF NOT EXISTS book_spec_blob BYTEA;
ALTER TABLE portfolio_artifacts ADD COLUMN IF NOT EXISTS policy_spec_blob BYTEA;
UPDATE runs SET name = experiment_name WHERE name IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_id ON runs(id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_runs_id ON portfolio_runs(id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_config_registry_id ON config_registry(id);
CREATE INDEX IF NOT EXISTS idx_configs_config_hash ON configs(config_hash);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_match_latents_match_id ON match_latents(match_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_runs_created_at ON portfolio_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_portfolio_bets_match_id ON portfolio_bets(match_id);
CREATE INDEX IF NOT EXISTS idx_config_registry_name ON config_registry(name);
CREATE INDEX IF NOT EXISTS idx_config_registry_hash ON config_registry(config_hash);
CREATE INDEX IF NOT EXISTS idx_config_registry_created_at ON config_registry(created_at);
CREATE INDEX IF NOT EXISTS idx_config_registry_type ON config_registry(config_type);
CREATE INDEX IF NOT EXISTS idx_config_registry_tags ON config_registry USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_fold_results_run_id ON fold_results(run_id);
CREATE INDEX IF NOT EXISTS idx_match_latents_fold_id ON match_latents(fold_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_runs_model_run_id ON portfolio_runs(model_run_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_bets_run_id ON portfolio_bets(portfolio_run_id);
