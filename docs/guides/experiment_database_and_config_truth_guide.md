# Experiment Database and Config Truth Engine Guide

> **Scope:** the `mcmc_experiments` database on `mcmc-beast:5432` — PostgreSQL-backed
> experiment tracking, canonical configuration discovery, and lossless reconstruction of
> inference and portfolio results. It is **not** `betdb`, the operational database on
> `archpc:5433` that holds the raw football data and the paper-trading ledgers; section 2
> draws the line between them.
>
> **Implementation:**
> [`src/training/inference/db_storage.jl`](../../src/training/inference/db_storage.jl),
> [`src/training/inference/extension.jl`](../../src/training/inference/extension.jl),
> [`src/Portfolio/db_storage.jl`](../../src/Portfolio/db_storage.jl),
> [`src/Portfolio/extension.jl`](../../src/Portfolio/extension.jl), and
> [`src/training/inference/db/schema.sql`](../../src/training/inference/db/schema.sql).
> The operational counterpart is
> [`src/MatchDay/ledger/schema.jl`](../../src/MatchDay/ledger/schema.jl) and
> [`src/MatchDay/db.jl`](../../src/MatchDay/db.jl).

---

## 1. Executive summary

A local `.jld2` result is useful as a machine-local artefact, but it is not a system of
record for a distributed research mesh. Paths differ between hosts, files can be copied or
renamed without provenance, directory scans cannot efficiently answer cross-run questions,
and two workers can unknowingly repeat the same expensive MCMC recipe. A binary file also
cannot be joined directly to fold diagnostics, match-level posterior summaries, or portfolio
trades.

The experiment database addresses those limitations with a central PostgreSQL service:

- **Queryable run tracking:** run status, Git provenance, duration, convergence, fold metrics,
  match latents, portfolio performance, and bets are ordinary relational rows.
- **Multi-machine discovery:** the laptop and compute workers query one shared source instead
  of reconciling host-specific result directories.
- **Configuration truth:** `config_registry` gives canonical models, splitters, samplers,
  `FitConfig`s, books, and policies stable integer IDs, names, tags, descriptions, and hashes.
- **Duplicate protection:** `configs.config_hash` uniquely identifies an inference recipe;
  `save_fit` returns the existing run UUID when that hash is already present.
- **Lossless reconstruction:** compressed binary artefacts preserve exact Julia `Fit` and
  `PortfolioResult` objects while relational tables remain easy to inspect in SQL.

`FileStorage` remains valuable for offline work and local redundancy. `DualStorage` writes
both a filesystem copy and the central database. PostgreSQL is the shared catalogue and
configuration source of truth, not a claim that every binary value should be flattened into
SQL.

---

## 2. The two databases

This guide is about **one** of the two PostgreSQL services this project uses. Knowing which
is which is the precondition for everything below.

| | **`betdb`** — what happened, and what we did | **`mcmc_experiments`** — what we fitted, and what it scored |
|---|---|---|
| Environment variable | `BF_DB_URL` (**required**; there is no default) | `BF_EXPERIMENTS_DB_URL`, else libpq's `~/.pgpass` |
| Endpoint | `archpc:5433` — LAN `192.168.1.88:5433`, Tailscale `100.124.38.117:5433` | `mcmc-beast:5432` — i.e. `localhost:5432` when you are on that host |
| Default DSN in code | none; `Data.load_datastore_sql` and `MatchDay.paper_connection` raise if unset | `postgresql://postgres@mcmc-beast:5432/mcmc_experiments`, **passwordless**, so libpq resolves the credential |
| Julia entry point | `Data.load_datastore_sql(segment)`, `MatchDay.paper_connection()` | `Training.PostgresStorage(experiment_name)` |
| Organised by | one PostgreSQL **schema per domain** | one flat schema, namespaced by the `experiment_name` column |
| Provisioned by | the collector stack, outside this repository | [`scripts/setup_experiments_db.sh`](../../scripts/setup_experiments_db.sh) — see the provisioning section |
| Written during a match day | yes — the paper ledger, at T−12 on a Saturday | no — a live slate only **reads** a completed run |
| Documented in | this section, and `AGENTS.md` §3 | the rest of this guide |

Neither is a replica or a cache of the other. They are separate servers on separate hosts
with separate credentials and separate failure modes.

### 2.1 `betdb` — the operational database

Raw football data plus the paper-trading ledgers, split by PostgreSQL schema:

| Schema | Holds |
|---|---|
| `sofascore` | `events` / `matches`, `seasons`, `match_player_lineups`, `lineup_provisional` (the pre-match XI scrape, stamped `scraped_at`), `match_statistics`, `match_incidents`, `match_odds` |
| `bbc` | `match_meta`, `match_stats`, `match_lineup`, `live_text` — the commentary stream the proxy-xG (Gamma) arm is built from |
| `betfair` | `match_meta` (the identity crosswalk), `markets`, `odds_history` — the closing-line archive used for CLV and for the de-vigged market baseline |
| `betfair_live` | `market_metadata`, `order_book_1m` — one-minute archived exchange ladders, at most three levels per side, with a running `market_matched` total and **no** traded price series |
| `paper_runbook` | the **live** paper ledger (MatchDay console on port 8085) |
| `paper_replay` | the **replay** paper ledger (replay console on port 8086) |

Each paper schema is the same eight tables, emitted by `MatchDay.paper_ddl(schema)`:
`paper_accounts`, `paper_slates`, `paper_orders`, `paper_fills`, `paper_snapshots`,
`paper_settlements`, `clv_audit`, `account_ledger`. `MatchDay.PAPER_SCHEMA` defaults to
`paper`; the test suite overrides it to `paper_test` and drops it afterwards.

Three unique constraints make the loop re-runnable after a crash without double-staking:
`paper_slates (account_id, slate_window, as_of)`, `paper_orders (slate_id, match_id,
market_group, market_line, selection)`, and `account_ledger (slate_id) WHERE kind =
'RESERVE'`. The last makes double-reserving a slate *unrepresentable* rather than guarded.

### 2.2 `mcmc_experiments` — the experiment database

Nine tables, described column by column in the schema reference below:

| Table | Holds |
|---|---|
| `config_registry` | canonical named components — `model`, `splitter`, `sampler`, `fit`, `book_spec`, `policy_spec`, `portfolio` |
| `configs` | the persisted inference recipe, one-to-one with a run; its `config_hash` is `save_fit`'s deduplication key |
| `runs` | one row per inference run: status, Git provenance, timings |
| `fold_results` | per-fold convergence audit and out-of-sample proper scores |
| `match_latents` | point-in-time posterior predictions per fixture, with compressed draws |
| `fit_artifacts` | the exact serialized `Fit` |
| `portfolio_runs` | headline ROI / drawdown / Sharpe per portfolio simulation |
| `portfolio_bets` | the **backtest** trade ledger — one row per simulated bet |
| `portfolio_artifacts` | the exact serialized `PortfolioResult`, plus the `BookSpec`/`PolicySpec` used |

### 2.3 Why the paper ledger is not in here

`paper_slates.model_run_id` carries `mcmc_experiments.runs.run_id` as an **opaque UUID with
no foreign key** — they are different servers, so a foreign key is not available. A
reconciliation job asserts that it resolves.

The ledger lives in `betdb` deliberately:

1. **Availability.** It is written at T−12 on a Saturday, when `mcmc-beast` may be saturated
   by a training grid. `betdb` is the operational database and is already required to be up
   by the collector, the supervisor and the console.
2. **Locality.** Mark-to-market and CLV both read `betfair_live.order_book_1m`. Putting the
   ledger beside it makes settlement a join rather than a cross-database transfer.
3. **Separation.** `portfolio_bets` here is the *backtest* ledger: one row per simulated bet,
   no lifecycle, no fills, no time. Paper trading has a lifecycle. Overloading one table with
   both would make "what did the backtest say" and "what did we actually do" the same query,
   which is the one distinction the exercise exists to preserve.

---

## 3. Entity-relationship diagram

```text
                                      configuration truth (experiment-scoped)
                         +-------------------------------------------------------+
                         | config_registry                                       |
                         | PK id                                                  |
                         | UQ (experiment_name, name)                             |
                         | name, config_type, config_hash, tags                   |
                         | config_json (queryable) + config_blob (exact object)   |
                         +-------------------------------------------------------+
                                  logical recipe/name/hash association only
                                                (no foreign key)

 +---------------------------+        1 : 1        +---------------------------+
 | runs                      |---------------------| configs                   |
 | PK id                     | run_id = config_id  | PK/FK config_id           |
 | UQ run_id (UUID)          |                     | UQ config_hash            |
 | name, experiment_name     |                     | model/split/sampler JSONB |
 | status, Git, timestamps   |                     +---------------------------+
 +-------------+-------------+
               |
               +-----------------------+-------------------------------+
               | 1 : many              | 1 : 1                         | 1 : many
               v                       v                               v
 +---------------------------+  +---------------------------+  +---------------------------+
 | fold_results              |  | fit_artifacts             |  | portfolio_runs            |
 | PK fold_id (UUID)         |  | PK/FK run_id              |  | PK id                     |
 | FK run_id                 |  | fit_blob BYTEA            |  | UQ portfolio_run_id UUID  |
 | UQ (run_id, fold_idx)     |  +---------------------------+  | FK model_run_id -> runs   |
 | diagnostics + scores      |                                 | ROI/risk metrics + hashes |
 +-------------+-------------+                                 +-------------+-------------+
               | 1 : many                                                  |
               v                                         +-----------------+-----------------+
 +---------------------------+                           | 1 : many                          | 1 : 1
 | match_latents             |                           v                                   v
 | PK latent_id              |             +---------------------------+     +---------------------------+
 | FK fold_id                |             | portfolio_bets            |     | portfolio_artifacts       |
 | UQ (fold_id, match_id)    |             | PK bet_id                 |     | PK/FK portfolio_run_id    |
 | lambda summaries          |             | FK portfolio_run_id      |     | result_blob BYTEA         |
 | compressed draws_blob     |             | market, stake, PnL       |     +---------------------------+
 +---------------------------+             +---------------------------+
```

The UUID columns remain stable cross-machine identifiers. Fresh schemas also expose
`BIGSERIAL` IDs on `runs`, `portfolio_runs`, and `config_registry` for concise interactive
lookups.

---

## 4. Schema reference

PostgreSQL `DOUBLE PRECISION` is the schema's floating-point type. `TIMESTAMP` values are
stored without a PostgreSQL time-zone annotation, so producers and consumers must use a
consistent clock convention.

### 4.1 `runs`

One row per persisted inference run.

| Column | Type | Constraint / meaning |
|---|---|---|
| `id` | `BIGSERIAL` | Primary key; human-friendly lookup ID. |
| `run_id` | `UUID` | Not null, unique; stable run identifier used by foreign keys. |
| `name` | `VARCHAR` | Not null; `FitConfig.name`. |
| `experiment_name` | `VARCHAR` | Not null; namespace supplied by `PostgresStorage`. |
| `status` | `VARCHAR` | Not null; currently saved as `completed`. |
| `git_commit` | `VARCHAR` | Not null; source revision from fit metadata. |
| `git_branch` | `VARCHAR` | Not null; branch detected when saving. |
| `created_at` | `TIMESTAMP` | Not null; inferred run start. |
| `finished_at` | `TIMESTAMP` | Nullable completion time. |
| `duration_seconds` | `DOUBLE PRECISION` | Nullable elapsed time. |

Indexes: unique `idx_runs_id`; `idx_runs_name`; `idx_runs_created_at`; and the unique
constraint index on `run_id`.

### 4.2 `configs`

Queryable inference recipe attached one-to-one to a run.

| Column | Type | Constraint / meaning |
|---|---|---|
| `config_id` | `UUID` | Primary key and foreign key to `runs(run_id)` with `ON DELETE CASCADE`. |
| `config_hash` | `VARCHAR` | Not null, unique SHA-256 recipe identity. |
| `model_config` | `JSONB` | Not null model type, display, and field summary. |
| `split_config` | `JSONB` | Not null splitter summary. |
| `sampler_config` | `JSONB` | Not null sampler, execution, name, tags, and description summary. |

Indexes: unique constraint index plus `idx_configs_config_hash`. The latter is explicit for
migration compatibility even though the unique constraint also supports equality lookup.

### 4.3 `fold_results`

One diagnostics/evaluation row per inference fold.

| Column | Type | Constraint / meaning |
|---|---|---|
| `fold_id` | `UUID` | Primary key. |
| `run_id` | `UUID` | Not null foreign key to `runs(run_id)` with `ON DELETE CASCADE`. |
| `fold_idx` | `INT` | Not null fold number; unique with `run_id`. |
| `r_hat_max` | `DOUBLE PRECISION` | Nullable maximum split $\hat{R}$. |
| `ess_bulk_min` | `INT` | Nullable minimum bulk ESS. |
| `ess_tail_min` | `INT` | Nullable minimum tail ESS. |
| `divergences` | `INT` | Not null, default `0`. |
| `converged` | `BOOLEAN` | Not null fold gate result. |
| `logloss` | `DOUBLE PRECISION` | Nullable evaluation LogLoss. |
| `brier` | `DOUBLE PRECISION` | Nullable Brier score. |
| `rps` | `DOUBLE PRECISION` | Nullable ranked probability score. |
| `runtime_seconds` | `DOUBLE PRECISION` | Nullable fold runtime. |
| `n_matches` | `INT` | Nullable OOS fixture count, populated by incremental extensions. |
| `first_match_date`, `last_match_date` | `DATE` | Nullable OOS date range, populated by incremental extensions. |

Indexes and constraints: `idx_fold_results_run_id` and unique `(run_id, fold_idx)`. The
current `save_fit` path writes convergence diagnostics and runtime, but initially stores
`logloss`, `brier`, and `rps` as `NULL`; those columns report scores only after a downstream
evaluation persistence step populates them.

### 4.4 `match_latents`

Match-level posterior count summaries and exact draws. Current PostgreSQL persistence accepts
`CountLatents`; unsupported latent families should use `FileStorage`.

| Column | Type | Constraint / meaning |
|---|---|---|
| `latent_id` | `BIGSERIAL` | Primary key and insertion-order key. |
| `fold_id` | `UUID` | Not null foreign key to `fold_results(fold_id)` with `ON DELETE CASCADE`. |
| `match_id` | `INT` | Not null match identifier; unique within a fold. |
| `mean_lambda_h`, `std_lambda_h` | `DOUBLE PRECISION` | Not null home-rate mean and standard deviation. |
| `p10_h`, `p50_h`, `p90_h` | `DOUBLE PRECISION` | Not null home-rate quantiles. |
| `mean_lambda_a`, `std_lambda_a` | `DOUBLE PRECISION` | Not null away-rate mean and standard deviation. |
| `p10_a`, `p50_a`, `p90_a` | `DOUBLE PRECISION` | Not null away-rate quantiles. |
| `draws_blob` | `BYTEA` | Not null Zstd-compressed home/away draws and optional NegBin dispersions. |

Indexes and constraints: `idx_match_latents_match_id`, `idx_match_latents_fold_id`, and unique
`(fold_id, match_id)`. The current `Fit` carries one merged run-level latent panel without
source fold IDs, so `save_fit` attaches all persisted latent rows to the run's first fold for
foreign-key ownership. Match IDs and insertion order preserve the exact merged panel; do not
interpret that ownership fold as the fold that generated each individual match.

### 4.5 `portfolio_runs`

Headline output from one portfolio simulation.

| Column | Type | Constraint / meaning |
|---|---|---|
| `id` | `BIGSERIAL` | Primary key; human-friendly lookup ID. |
| `portfolio_run_id` | `UUID` | Not null, unique stable portfolio identifier. |
| `model_run_id` | `UUID` | Not null foreign key to `runs(run_id)` with `ON DELETE CASCADE`. |
| `book_spec_hash` | `VARCHAR` | Not null hash of the pricing/book specification. |
| `policy_spec_hash` | `VARCHAR` | Not null hash of the staking/risk policy. |
| `total_return_pct` | `DOUBLE PRECISION` | Not null compounded return percentage. |
| `flat_roi_pct` | `DOUBLE PRECISION` | Not null flat ROI percentage. |
| `roi_1x2_pct` | `DOUBLE PRECISION` | Nullable 1X2 ROI percentage. |
| `max_drawdown_pct` | `DOUBLE PRECISION` | Not null maximum drawdown percentage. |
| `sharpe_ann` | `DOUBLE PRECISION` | Nullable annualised Sharpe ratio. |
| `win_rate` | `DOUBLE PRECISION` | Nullable bet win rate. |
| `n_bets` | `INT` | Not null bet count. |
| `created_at` | `TIMESTAMP` | Not null persistence time. |
| `metadata` | `JSONB` | Not null, default `{}`; convergence, failed gates, slate count, span, and caller metadata. |

Indexes: unique `idx_portfolio_runs_id`; `idx_portfolio_runs_created_at`;
`idx_portfolio_runs_model_run_id`; and the unique constraint index on `portfolio_run_id`.

### 4.6 `portfolio_bets`

One executed simulated trade per row.

| Column | Type | Constraint / meaning |
|---|---|---|
| `bet_id` | `BIGSERIAL` | Primary key. |
| `portfolio_run_id` | `UUID` | Not null foreign key to `portfolio_runs(portfolio_run_id)` with `ON DELETE CASCADE`. |
| `match_id` | `INT` | Not null match identifier. |
| `kickoff_date` | `DATE` | Not null slate date. |
| `market_family` | `VARCHAR` | Not null market family. |
| `selection` | `VARCHAR` | Not null backed selection. |
| `odds_close` | `DOUBLE PRECISION` | Not null closing decimal odds. |
| `stake_fraction` | `DOUBLE PRECISION` | Not null fraction of opening bankroll. |
| `stake_amount` | `DOUBLE PRECISION` | Not null currency stake reconstructed from daily opening bankroll. |
| `pnl` | `DOUBLE PRECISION` | Not null currency profit or loss. |

Indexes: `idx_portfolio_bets_match_id` and `idx_portfolio_bets_run_id`.

### 4.7 `config_registry`

Canonical, named configuration recipes. Names are unique within an experiment; saving the
same name updates its payload and metadata without creating another row.

| Column | Type | Constraint / meaning |
|---|---|---|
| `id` | `BIGSERIAL` | Primary key and public component lookup ID. |
| `name` | `VARCHAR` | Not null canonical name. |
| `experiment_name` | `VARCHAR` | Not null namespace. |
| `config_type` | `VARCHAR` | Not null classifier: `model`, `splitter`, `sampler`, `fit`, `book_spec`, `policy_spec`, or `portfolio`. |
| `description` | `VARCHAR` | Not null, default empty string. |
| `tags` | `JSONB` | Not null JSON string array, default `[]`. |
| `config_json` | `JSONB` | Not null searchable structural summary. |
| `config_blob` | `BYTEA` | Not null Zstd-compressed Julia serialization for exact reload. |
| `config_hash` | `VARCHAR` | Not null SHA-256 identity of the typed recipe. |
| `created_at` | `TIMESTAMP` | Not null first registration time. |
| `updated_at` | `TIMESTAMP` | Not null latest registration time. |

Indexes and constraints: unique `(experiment_name, name)`; unique `idx_config_registry_id`;
`idx_config_registry_name`; `idx_config_registry_hash`; `idx_config_registry_created_at`;
`idx_config_registry_type`; and GIN `idx_config_registry_tags`.

### 4.8 `fit_artifacts`

| Column | Type | Constraint / meaning |
|---|---|---|
| `run_id` | `UUID` | Primary key and foreign key to `runs(run_id)` with `ON DELETE CASCADE`. |
| `fit_blob` | `BYTEA` | Not null Zstd-compressed serialized `Fit`. |

This exact artefact preserves chains, typed configuration, diagnostics, and metadata.
`load_fit` replaces its latent panel with the copy reconstructed from `match_latents` when
relational latent rows exist.

### 4.9 `portfolio_artifacts`

| Column | Type | Constraint / meaning |
|---|---|---|
| `portfolio_run_id` | `UUID` | Primary key and foreign key to `portfolio_runs(portfolio_run_id)` with `ON DELETE CASCADE`. |
| `result_blob` | `BYTEA` | Not null Zstd-compressed serialized `PortfolioResult`. |
| `book_spec_blob` | `BYTEA` | Nullable exact `BookSpec`, used for lossless roll-forward. |
| `policy_spec_blob` | `BYTEA` | Nullable exact `PolicySpec`, used for lossless roll-forward. |

The artefact preserves daily states, bootstrap output, custom metrics, attribution, and all
other values not represented by the headline tables.

> **Trust boundary:** Julia `Serialization` is not a safe format for untrusted input. Connect
> only to a trusted experiment database, and treat artefact reloads like executing trusted
> project data. Exact deserialization also requires compatible project code and Julia types.

---

## 5. Password-safe connection resolution

The normal constructor takes only an experiment namespace:

```julia
using BayesianFootball

db = PostgresStorage("scottish_lower_2426")
ensure_schema!(db)
```

Resolution order is:

1. If `ENV["BF_EXPERIMENTS_DB_URL"]` is non-empty, use that libpq URI or connection string.
2. Otherwise use `postgresql://postgres@mcmc-beast:5432/mcmc_experiments` **without a
   password** and let libpq resolve the matching `~/.pgpass` entry.

A password-safe `~/.pgpass` setup is:

```text
mcmc-beast:5432:mcmc_experiments:postgres:<password-from-secret-manager>
```

```bash
chmod 600 ~/.pgpass
```

Alternatively, inject `BF_EXPERIMENTS_DB_URL` through the shell, CI secret store, or service
manager. Never put a credential-bearing URL in source, Markdown, logs, command history, or a
committed environment file.

The same rule governs `BF_DB_URL`, the operational database's variable (section 2). It has no
default — `Data.load_datastore_sql` and `MatchDay.paper_connection` raise a message telling
you to export it rather than falling back to a guessed endpoint — and it is read from a
git-ignored `.env` at module init. Redact it before pasting any shell output that contains it.

`Base.show` intentionally renders endpoint metadata but never `conn_str`:

```julia-repl
julia> db
PostgresStorage(host="mcmc-beast", port=5432, db="mcmc_experiments", experiment="scottish_lower_2426")
```

This protects normal REPL display even when the environment URL contains credentials. The
`conn_str` field still carries the connection material internally, so do not introspect or
log fields of `PostgresStorage` manually.

For non-default endpoints, the keyword constructor also omits the password and delegates it
to libpq:

```julia
db = PostgresStorage(
    host = "mcmc-beast",
    port = 5432,
    dbname = "mcmc_experiments",
    user = "postgres",
    experiment_name = "scottish_lower_2426",
)
```

---

## 6. Saving and loading through multiple dispatch

All registry lookups are scoped to `db.experiment_name`. Component loaders accept an integer
registry ID, a canonical name, a full config hash, or a `Symbol` name. Type-specific loaders
refuse rows of the wrong `config_type`.

### 6.1 Register canonical Lego components

```julia
model_id = save_model(
    db,
    "m00_joint_baseline",
    model;
    description = "Production joint count baseline",
    tags = ["production", "baseline"],
)

splitter_id = save_splitter(db, "split_2426", fit_cfg.splitter; tags = ["walkforward"])
sampler_id = save_sampler(db, "nuts_4x1000", fit_cfg.sampler; tags = ["production"])
fit_hash = save_config(
    db,
    "fit_joint_2426",
    fit_cfg;
    description = "Canonical 24/26 inference recipe",
    tags = ["production", "poisson"],
)

book_id = save_book_spec(db, "closing_main", spec; tags = ["production"])
policy_id = save_policy_spec(db, "quarter_kelly", policy; tags = ["production"])
```

`save_model`, `save_splitter`, `save_sampler`, `save_book_spec`, and `save_policy_spec` return
the integer `config_registry.id`. The legacy generic `save_config` returns the 64-character
configuration hash. A `(BookSpec, PolicySpec)` tuple can also be registered with
`save_config` and recovered with `load_portfolio_spec`.

### 6.2 Load by integer ID

```julia
model = load_model(db, 1)
splitter = load_splitter(db, 2)
sampler = load_sampler(db, 1)
fit_cfg = load_fit_config(db, 3)
fit = load_fit(db, 12)
book = load_book_spec(db, 1)
policy = load_policy_spec(db, 1)
```

The component numbers address `config_registry.id`; the fit number addresses `runs.id`.
They are separate ID sequences and need not be contiguous inside one experiment namespace.

### 6.3 Load by name

```julia
model = load_model(db, "m00_joint_baseline")
splitter = load_splitter(db, "split_2426")
sampler = load_sampler(db, "nuts_4x1000")
fit_cfg = load_fit_config(db, "fit_joint_2426")
fit = load_fit(db, "poisson_2426")       # latest run with this FitConfig.name
book = load_book_spec(db, "closing_main")
policy = load_policy_spec(db, "quarter_kelly")
```

Config names resolve to one canonical row because `(experiment_name, name)` is unique. A fit
name resolves to the latest matching run ID. `load_fit(db, run_uuid)` is preferable when a
workflow must identify one immutable run exactly.

### 6.4 Persist and reconstruct results

```julia
run_id = save_fit(fit, db)  # UUID; identical configs return the existing run UUID
fit_again = load_fit(db, run_id)

portfolio_run_id = save_portfolio_db(
    result,
    run_id,
    db;
    book_spec = spec,
    policy_spec = policy,
)
result_again = load_portfolio_db(portfolio_run_id, db)
```

`save_fit` writes `runs`, `configs`, `fold_results`, `match_latents`, and `fit_artifacts` in
one database transaction. `save_portfolio_db` writes `portfolio_runs`, `portfolio_bets`, and
`portfolio_artifacts` in one transaction.

`DualStorage` retains a local atomic filesystem copy as well:

```julia
storage = DualStorage(FileStorage("results"), db)
addresses = save_fit(fit, storage; quiet = true)
# addresses.path   -> filesystem directory
# addresses.run_id -> PostgreSQL UUID
```

### 6.5 Incrementally extend live runs

```julia
plan = preview_extension(db, run_uuid, latest_ds)
fit = extend_fit(db, run_uuid, latest_ds)
result = extend_portfolio(db, portfolio_uuid, fit, latest_ds.odds, latest_ds)
```

`preview_extension` derives current splitter boundaries and reports only positions absent from
`fold_results`. `extend_fit` samples those positions and updates fold diagnostics, OOS scores,
compressed match latents, the exact Fit artefact, and run telemetry in one transaction. Pass
`splitter = updated_splitter` when opening a new target season. `extend_portfolio` prices fixtures
absent from the existing bet ledger, continues from the closing bankroll, and atomically refreshes
the bet ledger, headline summary, and exact result artefact. Book and policy specs are recovered
from `portfolio_artifacts` when they were supplied to `save_portfolio_db`; otherwise pass them
explicitly or register matching canonical specs.

---

## 7. REPL discovery and explorer

### 7.1 Summarise experiments

```julia
experiments = explore_experiments(db)
```

This prints one row per experiment across the database with run count, distinct stored model-
type count, best (minimum) populated LogLoss, best populated Brier score, and last activity.
It also returns a `DataFrame` for further filtering. Score cells remain `—` while the
corresponding `fold_results` columns are `NULL`.

### 7.2 Search canonical configurations

```julia
all_configs = search_configs(db)
baselines = search_configs(db, "baseline")
production = search_configs(db, "tag=\"production\"")
models = search_configs(db, "config_type=:model")
```

The terminal table is `[ID | Type | Name | Tags | Description]`; the return value is the
matching `DataFrame`. The query string is positional and defaults to `""`. Free text searches
name, type, description, tags, and JSON summary. Structured `tag=...` and
`config_type=...` terms can be combined with free text. Equivalent keyword filters are
available through `search_configs(db, ""; tag = "production", config_type = "model")` or
`list_configs`.

### 7.3 Inspect the architecture

```julia
model = show_config(db, 1)
fit_cfg = show_config(db, "fit_joint_2426")
```

`show_config` prints ID, name, type, experiment, hash, tags, description, update time, and a
recursive architectural tree of the exact deserialized object. It returns that object, so the
same call supports both visual inspection and REPL reuse.

---

## 8. Useful SQL recipes

Use parameter placeholders in application code. Literal values below are readable examples,
not an invitation to interpolate untrusted input.

### 8.1 Inspect fold diagnostics and match posterior predictions

```sql
SELECT
    r.id AS run_id,
    r.name AS fit_name,
    fr.fold_idx,
    fr.converged,
    fr.r_hat_max,
    fr.ess_bulk_min,
    fr.divergences,
    fr.logloss,
    ml.match_id,
    ml.mean_lambda_h,
    ml.p10_h,
    ml.p50_h,
    ml.p90_h,
    ml.mean_lambda_a,
    ml.p10_a,
    ml.p50_a,
    ml.p90_a
FROM runs AS r
JOIN fold_results AS fr ON fr.run_id = r.run_id
JOIN match_latents AS ml ON ml.fold_id = fr.fold_id
WHERE r.experiment_name = 'scottish_lower_2426'
  AND r.id = 12
ORDER BY fr.fold_idx, ml.match_id;
```

This reads the queryable summaries. With the current merged `Fit.latents` representation, all
match rows join through the first fold used for foreign-key ownership; other fold diagnostics
still appear only when they have associated latent rows. Use `load_fit(db, 12)` when the full
posterior draw panel or chain is required.

### 8.2 Join portfolio PnL and trades to model metadata

```sql
SELECT
    pr.id AS portfolio_id,
    r.id AS model_run_id,
    r.name AS fit_name,
    c.model_config ->> 'type' AS model_type,
    c.model_config ->> 'display' AS model_display,
    pr.total_return_pct,
    pr.flat_roi_pct,
    pr.roi_1x2_pct,
    pr.max_drawdown_pct,
    pr.n_bets,
    SUM(pb.pnl) OVER (PARTITION BY pr.portfolio_run_id) AS total_trade_pnl,
    pb.match_id,
    pb.kickoff_date,
    pb.market_family,
    pb.selection,
    pb.odds_close,
    pb.stake_amount,
    pb.pnl
FROM portfolio_runs AS pr
JOIN runs AS r ON r.run_id = pr.model_run_id
JOIN configs AS c ON c.config_id = r.run_id
LEFT JOIN portfolio_bets AS pb
       ON pb.portfolio_run_id = pr.portfolio_run_id
WHERE r.experiment_name = 'scottish_lower_2426'
ORDER BY pr.id, pb.kickoff_date, pb.bet_id;
```

### 8.3 Filter runs by a registered model name

`config_registry` is deliberately not a foreign-key parent of historical runs: registry names
can be updated, while a run's JSON recipe remains immutable. For an exact current-summary
match, join the registered model's `config_json.value` to `configs.model_config`:

```sql
WITH registered_model AS (
    SELECT experiment_name, config_json -> 'value' AS model_json
    FROM config_registry
    WHERE experiment_name = 'scottish_lower_2426'
      AND config_type = 'model'
      AND name = 'm00_joint_baseline'
)
SELECT
    r.id,
    r.run_id,
    r.name,
    r.status,
    r.git_commit,
    r.created_at,
    c.config_hash
FROM registered_model AS rm
JOIN runs AS r
  ON r.experiment_name = rm.experiment_name
JOIN configs AS c
  ON c.config_id = r.run_id
 AND c.model_config = rm.model_json
ORDER BY r.created_at DESC;
```

For immutable audit records, retain the run UUID and `configs.config_hash`; do not rely only
on a mutable canonical name.

---

## 9. Avoiding redundant MCMC

Before scheduling expensive sampling, search the run catalogue for the approved recipe's
known `configs.config_hash`:

```sql
SELECT r.id, r.run_id, r.name, r.status, r.finished_at
FROM configs AS c
JOIN runs AS r ON r.run_id = c.config_id
WHERE c.config_hash = $1;
```

If a completed row exists, reconstruct it with `load_fit(db, id)` instead of sampling again.
`save_fit` also enforces hash deduplication, but that save-time guard does not recover compute
already spent; the preflight query is the important operational check.

The two hash domains have different purposes:

- `config_registry.config_hash` identifies one registered typed component or recipe.
- `configs.config_hash` identifies a persisted inference recipe in an experiment namespace
  and is the deduplication key used by `save_fit`.

Do not assume the values are interchangeable. Preserve the previously approved run hash in
an experiment manifest, report, or scheduler record and use it for the preflight lookup.

---

## 10. Infrastructure and provisioning

[`scripts/setup_experiments_db.sh`](../../scripts/setup_experiments_db.sh) provisions the
persistent PostgreSQL 16 service. By default it:

1. connects to `root@mcmc-beast` over SSH (or executes directly when already on that host),
2. creates the persistent data directory `/root/postgres_experiments_data`,
3. creates or starts `mcmc_experiments_postgres` from `postgres:16-alpine`,
4. publishes PostgreSQL on `mcmc-beast:5432`,
5. creates the `mcmc_experiments` database when absent, and
6. reapplies `src/training/inference/db/schema.sql` with `ON_ERROR_STOP=1`.

Run it only from an authorised administration shell. Supply the password from a secret store
rather than writing it into the repository or command text:

```bash
read -sr MCMC_DB_PASSWORD
export MCMC_DB_PASSWORD
./scripts/setup_experiments_db.sh
unset MCMC_DB_PASSWORD
```

Useful overrides are `MCMC_DB_HOST`, `MCMC_DB_CONTAINER`, `MCMC_DB_NAME`, `MCMC_DB_USER`,
`MCMC_DB_PASSWORD`, `MCMC_DB_DATA_DIR`, and `MCMC_DB_PORT`. `MCMC_DB_HOST=local` provisions
on the current machine; this is suitable for a disposable local integration database when
Docker is available.

### Idempotent schema application and migrations

The canonical schema uses `CREATE TABLE IF NOT EXISTS`, `ALTER TABLE ... ADD COLUMN IF NOT
EXISTS`, and `CREATE INDEX IF NOT EXISTS`. The setup script can therefore be rerun to start a
stopped container and apply additive migrations without deleting data. Julia callers can run
the same canonical schema with:

```julia
ensure_schema!(db)
```

`ensure_schema!` also handles the legacy `config_registry.registry_id` default used by the v1
schema. There is currently no separate migration-version table: `schema.sql` is both the
bootstrap DDL and the additive migration runner. Consequently:

- make forward-compatible, idempotent schema changes in `schema.sql`;
- back up the persistent volume before non-additive DDL;
- never edit applied production history by hand; and
- verify migrations against a disposable local PostgreSQL instance before provisioning the
  shared service.

---

## 11. Operational checklist

1. Confirm which database the task needs. `PostgresStorage` is `mcmc_experiments`; anything
   reading fixtures, the order book, lineups or a paper ledger is `betdb` (section 2).
2. Construct `PostgresStorage(experiment_name)`; never embed or print a raw password.
3. Run `ensure_schema!(db)` at deployment/setup boundaries.
4. Register canonical components with names, descriptions, and useful tags.
5. Check `configs.config_hash` before scheduling MCMC.
6. Persist the fit and retain its returned UUID.
7. Persist portfolio output against that immutable model run UUID.
8. Use SQL for discovery and summaries; use loaders for exact Julia reconstruction.
9. Keep trusted code and schema versions aligned with serialized artefacts.
10. When a live or replay slate is involved, remember the direction of travel: it **reads** a
    converged run out of `mcmc_experiments` and **writes** only into `betdb.<paper_schema>`.
