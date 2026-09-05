# AGENTS.md — BayesianFootball.jl

> Canonical guide for AI agents (pi, Claude Code, Antigravity) working in this
> repository. `CLAUDE.md` and `GEMINI.md` are pointers to this file — put new
> guidance **here** so the three harnesses cannot drift apart again.

A Bayesian hierarchical modelling framework for football analytics, market
evaluation, portfolio construction and match-day execution, in Julia.

---

## 1. Quick reference

* **Julia coding context** — [`docs/guides/julia_coding_context_for_agents.md`](docs/guides/julia_coding_context_for_agents.md) — language traps, style, Turing API facts, verification ladder. **Read before writing Julia.**
* **ReverseDiff AD performance & safety** — [`docs/turing_ad_performance_guide.md`](docs/turing_ad_performance_guide.md)
* **Experiment DB & config truth** — [`docs/guides/experiment_database_and_config_truth_guide.md`](docs/guides/experiment_database_and_config_truth_guide.md)
* **Prototype runner style** — [`docs/prototype_runner_style_guide.md`](docs/prototype_runner_style_guide.md)
* **Agent infrastructure & execution context** — [`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](docs/architecture/ai_agent_infrastructure_and_execution_context.md)
* **MatchDay live loop (operator)** — [`current_development/match_day_inference/QUICKSTART_LIVE.md`](current_development/match_day_inference/QUICKSTART_LIVE.md)
* **MatchDay live + replay consoles** — [`current_development/match_day_inference/README.md`](current_development/match_day_inference/README.md)
* **Tmux subagent & REPL control** — [`docs/setup/agy_tmux_agent_and_repl_control_guide.md`](docs/setup/agy_tmux_agent_and_repl_control_guide.md)
* **Remote execution & tmux protocol** — [`docs/setup/agy_remote_execution_guide.md`](docs/setup/agy_remote_execution_guide.md)
* **Portfolio trust & market capacity findings** — [`eda/README.md`](eda/README.md) — Knapsack shadow price, Jensen tail distortion, directional Under 2.5 alpha, and multi-tier conviction ratio law.

---

## 2. Architecture

A multi-tier Bayesian predictive, portfolio and execution system. The
**Unified V2 stack** is the production standard:

| Layer | Name | Lives in |
|---|---|---|
| L0 | **Data** — typed PostgreSQL extraction, memory-optimised `DataStore`, vig-removed market math | `src/Data/` |
| L1 | **Bayesian engines** — Turing.jl hierarchical models on compiled ReverseDiff tapes | `src/models/pregame/` |
| L2 | **Unified inference, latents & experiment truth** — convergence audits, `CountLatents`, zero-alloc score tensors, Postgres run tracking | `src/training/inference/`, `src/models/latents/` |
| L3 | **Unified evaluation** — point-in-time pricing, LogLoss, CRPS, Brier, RPS, ECE vs closing odds | `src/evaluation/` |
| L4 | **Zero-alloc portfolio, staking & audit** — `OddsIndex`, `BookWorkspace`, Baker-McHale shrinkage, fractional Kelly, Postgres persistence | `src/Portfolio/` |
| L5 | **MatchDay operational execution** — point-in-time slate pricing, paper ledger, live and replay consoles | `src/MatchDay/`, `current_development/match_day_inference/` |

> **Numbering caveat.** An older four-layer scheme is still referenced in some
> docs and archive material, where "Layer 2" means GLM calibration
> (`src/Calibration/`) and "Layer 3" means the meta-model
> (`current_development/MetaModels/`). Those components still exist and are
> described below; when a doc says "L2/L3", check which scheme it means. The
> table above is canonical for new work.

### L0 — Data (`src/Data/`)

SQL → `DataStore` via `LibPQ`. Every data domain (Matches, Odds, Betfair,
Stats, Lineups, Incidents, BBC commentary) passes a strict
**Fetch → Process → QA** three-step contract defined in
`src/Data/fetchers/interfaces.jl`. Tournament segments are singletons
(`ScottishLower`, `Ireland`, …) in `src/Data/fetchers/segments.jl`. The
`Markets` submodule does vig removal, fair odds, and closing-line-movement
math. Cached locally as `.jls` files in `.cache/`.

### L1 — Bayesian engines (`src/models/pregame/`)

Component-driven, assembled from mathematical building blocks:

- **Components** (`components/`): `Interception`, `Dispersion`, `HomeAdvantage`,
  `Dynamics`, `Kappa`, `Copula`, `DixonColes`, covariates
  (`ProductionWealthCovariate`, `DistanceCovariate`, `BenchDepthCovariate`),
  and the `PlayerLineupPillar` that composes RAPM teamsheet ratings beside team
  attack/defence.
- **Observations**: `PoissonObservation`, `NegBinObservation`,
  `JointGammaPoissonObservation` (the two-arm proxy-xG + goals likelihood).
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

### L5 — MatchDay (`src/MatchDay/`)

The operational layer. It prices a whole simultaneous fixture slate at a stated
instant, records the planned stake vector in a paper ledger, and serves it to an
operator. **The slate is the execution atom**: `Portfolio` solves one joint
problem for every fixture that settles together, so the stake vector is only
valid *as a vector*, and reservation is one transaction for the whole of it.

```
fixtures → identity → lineups → book → features → inference → gate → stake_sheet
```

Every stage is a seam with an abstract type (`AbstractFixtureSource`,
`AbstractIdentityResolver`, `AbstractLineupSource`, `AbstractBookSource`,
`AbstractGate`, …), which is what lets the replay console swap only the sources
that read a clock or a network while keeping the gates, the instrument rule, the
stake rounding, the market set and the portfolio policy identical to the live
path. Posteriors are never sampled here: `MD.canonical_fit` loads a completed run
out of `mcmc_experiments`. See §7.

### Calibration (`src/Calibration/`) — the L2 calibrator tier

**Generative rate calibration.** The tradeable book is inverted back to
`(lambda_mkt_h, lambda_mkt_a)` by Nelder-Mead on
`Features.DoublePoissonMarketFeature`, every posterior log-rate draw is pooled
with it, and the calibrated container is priced through the **same** score-grid
kernels, evaluator and portfolio the raw one goes through:

```julia
book, refusals = point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))
cal = GenerativeRateCalibrator(name = "scot_lower_t25_inv",
                               law  = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
                               book_as_of_minutes = -25.0)
cf  = calibrate_fit(cal, fit, book)                        # -> CalibratedFit
result, books, rep = run_portfolio_simulation(spec, policy, cf, book, ds)
```

1X2, every totals line and BTTS are then three partitions of **one** 12×12 score
tensor, so derivative coherence is structural rather than audited.
`cf.fit` is a real `Training.Fit` carrying the calibrated latents, which is what
lets L3 and L4 consume it with **no change to `src/Portfolio/`**.

Three location laws (`InverseGaussianLaw`, `StandardGaussianLaw`,
`StaticGeometricLaw`) and four dispersion maps (`PoolDispersion` — the default and
the validated production transform — `PreservedDispersion`, `ConjugateDispersion`,
`SupremacyDispersion`). **Which law wins depends on the sharpness of the book being
pooled with, not on the league**: the standard form wins at the Betfair close and
the inverse form at T−25, and a spec transferred between instants gives up
0.0015–0.0020 LogLoss. A calibrator therefore records `book_as_of_minutes` and
`calibrate_fit` refuses a book from a different instant.

Persistence is `config_registry` (`config_type = 'calibrator'`) plus
`calibration_runs` / `calibration_artifacts` in `mcmc_experiments`; a portfolio run
is linked through `portfolio_runs.metadata`, not a foreign key.

Design record: [`docs/architecture/rfc_layer2_calibration_v2.md`](docs/architecture/rfc_layer2_calibration_v2.md).
Evidence, including two published conclusions the stream's own later phases
retracted: [`current_development/calibration_generative_eda/README.md`](current_development/calibration_generative_eda/README.md).

> **`BasicLogitShift` is DEPRECATED** (`build_l2_training_df` → `train_calibrators`
> → `apply_calibrators`). It fits one GLM offset per selection and applies it
> independently, so `P(over 2.5) + P(under 2.5) != 1` and the shifted board is not
> a scoreline distribution at all. It still runs and warns once per session.

### Meta model (`current_development/MetaModels/`)

In active development. Blends L1 predictions with market-implied probabilities
via a dynamic Gaussian random-walk mixture: `Q_i = θ_t·p_L1_i + (1-θ_t)·m_i`.
Two engine types: `ConvexMixtureMetaModel`, `AffineCalibrationMetaModel`. See
`docs/archive/meta_model_design.md`.

### Module map

| Module | Responsibility |
|---|---|
| `Data` | SQL extraction, ETL, `Markets` (vig removal, fair odds, CLM) |
| `Features` | `DataStore` → `FeatureSet`. AD-safe flattening, team/time indexing, lineup market valuations, player RAPM ratings. Uses `SplitBoundary` (match-ID pointers, not copies) for memory-efficient temporal folds. Extractors in `src/features/extractors/` |
| `Models` | `CountModelBuilder`, Turing components, master engines |
| `Samplers` | NUTS, MAP, MLE, ADVI wrappers. `QueuedNUTSConfig` flattens K-splits × N-chains into one global queue for full CPU utilisation |
| `Training` | `fit_model`, execution dispatchers, convergence auditing, `PostgresStorage`/`FileStorage`/`DualStorage`. `Independent(max_concurrent_tasks)` uses queued execution |
| `Experiments` | Task creation, execution, persistence, loading |
| `Predictions` | Typed latents (`CountLatents`), in-place score-grid kernels (`SmileScoreGrid`), PPD generation |
| `Evaluation` | `OddsView`, `evaluate_predictions`, LogLoss, CRPS, RPS, ECE |
| `Portfolio` | `OddsIndex`, `BookWorkspace`, Kelly log-utility allocator, `simulate_portfolio` |
| `MatchDay` | Point-in-time slate pricing, gates, instruments, paper ledger (`betdb`), operator console |
| `Signals` | Betting signal generation and Kelly staking |
| `BackTesting` | `AbstractWealthMetric` (Sharpe, Calmar, Sortino, Burke, Sterling) and `AbstractDistributionalMetric` (Hurdle ROI). `run_backtest` → `generate_tearsheet` |
| `MyDistributions` | `RobustNegativeBinomial`, `FrankCopulaNegBin`, `DixonColes`, `BivariatePoissonDist`, … |

---

## 3. The two databases

There are **two** PostgreSQL services and they answer two different questions.
Confusing them is the most common orientation error in this repository.

| | **`betdb`** — what happened, and what we did | **`mcmc_experiments`** — what we fitted, and what it scored |
|---|---|---|
| Env var | `BF_DB_URL` (**required**, no default) | `BF_EXPERIMENTS_DB_URL`, else `~/.pgpass` |
| Host | `archpc:5433` (LAN `192.168.1.88:5433`, Tailscale `100.124.38.117:5433`) | `mcmc-beast:5432` — i.e. `localhost:5432` when you are on the beast |
| Provisioned by | the collector stack, outside this repo | [`scripts/setup_experiments_db.sh`](scripts/setup_experiments_db.sh) (`postgres:16-alpine`, `/root/postgres_experiments_data`) |
| Reached from Julia by | `Data.load_datastore_sql`, `MatchDay.paper_connection` | `Training.PostgresStorage(experiment_name)` |
| Schema-per-domain | yes | no — one flat schema |
| Failure mode if down | no data, no live pricing, no ledger | no canonical fits; a live slate cannot be priced but the collector keeps running |

### 3.1 `betdb` — the operational database (`BF_DB_URL`)

Raw football data plus the paper-trading ledgers, split by PostgreSQL schema:

| Schema | Holds |
|---|---|
| `sofascore` | `events` / `matches`, `seasons`, `match_player_lineups`, `lineup_provisional` (the pre-match XI scrape, stamped `scraped_at`), `match_statistics`, `match_incidents`, `match_odds` |
| `bbc` | `match_meta`, `match_stats`, `match_lineup`, `live_text` — the commentary stream the **proxy xG** arm is built from |
| `betfair` | `match_meta` (the identity crosswalk), `markets`, `odds_history` — the **closing-line archive** used for CLV and for the de-vigged market baseline |
| `betfair_live` | `market_metadata`, `order_book_1m` — one-minute archived exchange ladders, at most **three levels** per side, both sides, with a running `market_matched` total (no traded price series) |
| `paper_runbook` | the **live** paper ledger (console on 8085) |
| `paper_replay` | the **replay** paper ledger (console on 8086) |

Each paper schema is the same eight tables, built by `MatchDay.paper_ddl(schema)`:
`paper_accounts`, `paper_slates`, `paper_orders`, `paper_fills`,
`paper_snapshots`, `paper_settlements`, `clv_audit`, `account_ledger`.
`PAPER_SCHEMA` defaults to `paper`; the suite overrides it to `paper_test`.

Three unique constraints make the loop re-runnable after a crash without
double-staking — `paper_slates (account_id, slate_window, as_of)`,
`paper_orders (slate_id, match_id, market_group, market_line, selection)`, and
`account_ledger (slate_id) WHERE kind = 'RESERVE'`. The last one makes
double-reserving a slate *unrepresentable* rather than merely guarded.

### 3.2 `mcmc_experiments` — the experiment database (`BF_EXPERIMENTS_DB_URL`)

Constructed as `PostgresStorage(experiment_name)`; every read and write is
scoped to that experiment namespace. Tables:

| Table | Holds |
|---|---|
| `config_registry` | canonical named components — `model`, `splitter`, `sampler`, `fit`, `book_spec`, `policy_spec`, `portfolio` — with tags, a queryable `config_json` and an exact `config_blob` |
| `configs` | the persisted inference recipe, one-to-one with a run; `config_hash` is the **deduplication key** for `save_fit` |
| `runs` | one row per inference run: status, Git commit and branch, timings |
| `fold_results` | per-fold convergence audit (R̂, bulk/tail ESS, divergences) and OOS proper scores |
| `match_latents` | point-in-time posterior predictions per fixture: λ summaries plus Zstd-compressed draws |
| `fit_artifacts` | the exact serialized `Fit` |
| `portfolio_runs` | headline ROI / drawdown / Sharpe per portfolio simulation, linked to `runs.run_id` |
| `portfolio_bets` | the simulated trade ledger, one row per backtest bet |
| `portfolio_artifacts` | the exact serialized `PortfolioResult`, plus the `BookSpec`/`PolicySpec` used |

### 3.3 The link between them

`betdb.<paper_schema>.paper_slates.model_run_id` carries
`mcmc_experiments.runs.run_id` as an **opaque UUID with no foreign key** — they
are separate servers. A reconciliation job asserts it resolves.

The ledger deliberately lives in `betdb`, not beside the backtest tables:

1. **Availability.** It is written at T−12 on a Saturday, when `mcmc-beast` may
   be saturated by a training grid; `betdb` is already required to be up.
2. **Locality.** Mark-to-market and CLV both read `betfair_live.order_book_1m`,
   so settlement is a join rather than a cross-database transfer.
3. **Separation.** `mcmc_experiments.portfolio_bets` is the **backtest** ledger —
   one row per simulated bet, no lifecycle, no fills. Paper trading has a
   lifecycle. Pooling them would make "what did the backtest say" and "what did
   we actually do" the same query, which is the one distinction the exercise
   exists to preserve.

### 3.4 Credentials

**Never** commit, paste into a prompt, or print a raw password or a
credential-bearing URL. `BF_DB_URL` is read from the environment (`.env` is
loaded at module init and is git-ignored). `PostgresStorage` resolves
`BF_EXPERIMENTS_DB_URL` or lets libpq read `~/.pgpass`; its masked `show` is
safe, printing `storage.conn_str` is not. Full protocol in §12.

---

## 4. Unified V2 pipeline — the production path

```julia
using BayesianFootball
using DataFrames, Dates, ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# 1. Data layer
ds = Data.load_datastore_cached(Data.ScottishLower())

# 2. Composable model builder — here the current production shape:
#    two-arm joint observation + team time decay + lineup RAPM + squad wealth.
model = CountModelBuilder(:m12_joint_hybrid_synergy) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PlayerLineupPillar(rating = :shots_rapm,
                           aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10),
                           fit_on = :history)) |>
    add(ProductionWealthCovariate(role = SupremacyRole())) |>
    add(JointGammaPoissonObservation()) |>
    build

# 3. Unified inference lifecycle & convergence gating
fit_cfg = FitConfig(
    name      = "m12_joint_hybrid_synergy",
    model     = model,
    splitter  = Data.CVConfig(target_seasons = ["24/25", "25/26"], window_seasons = 3),
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

The exact component names for a candidate are in that experiment's `lXX_loader.jl`;
treat the block above as the shape of the call, not as a copy-paste recipe.

---

## 5. Prototyping and experiment layout

Two directories, two purposes.

### 5.1 `current_development/` — prototypes

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
| `match_day_inference/` | **Live and replay execution consoles** — the active operational suite (§7) |
| `player_lineup_dynamics/` | Teamsheet RAPM formulation search, bench weighting, nested-history grids |
| `plus_minus_ratings/` | Ridge plus-minus / RAPM estimation from BBC per-shot commentary |
| `bbc_xg_proxy/` | Proxy xG from BBC live text — the source of the Gamma arm |
| `scottish_proxy_xg/`, `scottish_lower/`, `scottish_upper/` | Segment-specific model exploration |
| `inplay_scottish/`, `match_inplay_explore/` | In-play intensity / NHPP / market-inverse research |
| `orderbook_layer2/`, `betfair_closing_line/` | Exchange microstructure, CLV, Bayesian calibration/staking |
| `manager_wealth/`, `manager_pace_scalar/`, `team_wealth/`, `smile_negbin/` | Covariate and likelihood side-studies |
| `MetaModels/` | Meta-model (see Architecture) |
| `archived/` | Superseded prototypes, retained for the reasoning. **Not** the active path |

### 5.2 `experiments/` — completed benchmark suites

`experiments/<segment>/NN_<topic>/` holds the finished, reproducible research
grids: a loader, a smoke runner, a production runner, a comparison runner, a
portfolio runner, and a `README.md` carrying the *measured* results and the run
UUIDs. This is where a claim about model performance should be sourced from.

Each numbered directory also states its verification gates. A production grid is
not launched until the smoke runner has passed all of them for every candidate —
gradient tape, sampling, the six-part convergence audit, latent extraction,
score-grid construction, `save_fit`/`load_fit` round-trip, and portfolio
persistence with an identical reloaded bet ledger.

Top-level `experiments/*.md` files are work-package prompts handed to agents.
They are inputs, not records; do not cite one as a result.

---

## 6. Model generations — Scottish Lower (tournaments 56 / 57)

Four paradigms, each a full 40-fold walk-forward grid over seasons 24/25 + 25/26
(710 held-out matches, 2,899 scored market observations). Numbers below are the
recorded outcomes in each suite's README, not targets.

| Gen | Suite | Paradigm | Headline |
|---|---|---|---|
| **1** | [`01_poisson_2426_grid/`](experiments/scottish_lower/01_poisson_2426_grid/README.md) | Poisson likelihood; baseline, squad wealth, travel distance, joint, age-adjusted production wealth | `m05_production_wealth` LogLoss **0.6597**; Betfair backtest +125% to +140% bankroll |
| **2** | [`02_negbin_2426_grid/`](experiments/scottish_lower/02_negbin_2426_grid/README.md) | Negative Binomial; empirical overdispersion | `r̂ ≈ 26.0–26.5` (mild overdispersion); LogLoss **0.6598**, no material gain over Poisson |
| **3** | [`03_joint_gamma_poisson/`](experiments/scottish_lower/03_joint_gamma_poisson/README.md) | **Two-arm joint**: shared latent `μ`, Gamma arm on BBC commentary proxy xG, Poisson arm on goals | LogLoss **0.6571** vs Betfair close 0.6568 — the second likelihood is worth ~5× the best covariate |
| **4** | [`05_.../`](experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md) + [`06_.../`](experiments/scottish_lower/06_joint_player_lineup_fusion/README.md) | **Joint + player-lineup hybrid**: `PlayerLineupPillar` (shots-RAPM / pxG-RAPM, starters + bench at fixed `w_bench = 0.10`) composed beside team time decay | `m12` ECE **0.0100** vs Betfair close **0.0139**; +136.6% bankroll, 1.416 annual Sharpe |

### 6.1 Generation 3 — the two-arm joint observation

One log-intensity `η` per side is read by two densities at once:

```
arm 1 (proxy xG)   pxg_s ~ Gamma(ν, μ_s / ν)     evaluated where the mask is 1
arm 2 (goals)      y_s   ~ Poisson(κ · μ_s)      evaluated everywhere
```

`Gamma(shape = ν, scale = μ/ν)` has mean `μ`, so `ν` is a pure precision and the
proxy measurement is unbiased for the latent by construction; `κ` is the
finishing factor. The proxy arm sharpens `μ` on the seasons that carry BBC live
text; the goals arm carries that sharpened `μ` back across the whole history.
`MatchProxyXGFeature(fallback = :none)` emits an explicit availability mask, so a
match without commentary contributes a finite term multiplied by an exact zero
rather than a fabricated observation.

Identified, not assumed: `κ ≈ 1.13` (~13% more goals converted than the BBC
shot-xG cell table predicts, ~76% prior shrinkage) and `ν ≈ 3.9` with posterior
sd ~0.28 against a prior sd of 1.45.

### 6.2 Generation 4 — the player-lineup hybrid

```
η_home,i = μ_int + HA + α_home + β_away + L_home,i + Σ_c w_c · x_c,i
η_away,i = μ_int      + α_away + β_home + L_away,i − Σ_c w_c · x_c,i

L_home,i = w_att · R_home,i − w_def · R_away,i
```

`R_s,i` is the aggregated RAPM rating of side `s`'s named teamsheet. RAPM is a
ridge fit — **never sampled** — over each fold's frozen history block
(`fit_on = :history`), so a target fixture never contributes to the ratings that
price it. Covariates enter in the `SupremacyRole()`
(`covariate_sides(SupremacyRole(), q) = (q, −q)`), so a covariate moves the
*result* and holds the *total*.

Experiment 06 (`scottish_lower_joint_player_2426`) measured, over 2,899 scored
observations and 1,455–1,468 bets:

| Model | LogLoss | ECE | Bankroll | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|
| `m13_joint_composite` (+ distance) | 0.64324 | **0.0088** | **+140.2%** | 1.453 | −21.1% |
| `m12_joint_hybrid_synergy` | 0.64337 | 0.0100 | +136.6% | 1.416 | −20.2% |
| `m05_joint_production_wealth` (control) | **0.64299** | 0.0149 | +131.2% | **1.481** | **−19.1%** |
| `m10_joint_player_shots_bench` | 0.64440 | 0.0090 | +112.2% | 1.217 | −20.0% |
| *Betfair closing line* | *0.64182* | *0.0139* | — | — | — |

**Read this honestly.** The lineup arms do **not** win on LogLoss — the
team-state control is still the sharpest. What they buy is **calibration**: every
lineup arm halves the control's ECE and beats the Betfair closing line's, and
that is what converts into Kelly bankroll growth. The prior EDA (`r59`,
`EDA_FINDINGS.md`) refused H1 (lineups alone improve team state) and H5
(travel), supported H4 (wealth is complementary to RAPM) strongly, and found H2
(bench depth) negligible. The hypotheses were written so they could fail
visibly, and some did.

`m12` is the model the MatchDay consoles load as the hybrid pillar; `m00` and
`m05` are the team-level controls that make a lineup move attributable.

---

## 7. MatchDay — the live and replay consoles

Two long-running processes, side by side, and **neither can reach the other's
rows**.

| | runner | port | tmux | schema | clock | for |
|---|---|---|---|---|---|---|
| **live** | `r07_serve_console.jl` | **8085** | `matchday_console` | `betdb.paper_runbook` | `now()` | committing a slate on a Saturday |
| **replay** | `r08_replay_console.jl` | **8086** | `replay_run` | `betdb.paper_replay` | a scrubber | backtesting a past Saturday |

Isolation is **structural, not conventional**: `assert_replay_schema` refuses
`paper_runbook` at every ledger call site and `serve_replay` refuses to bind
8085. Both refusals are asserted directly in the suite (R1, R2, R18), the last of
them by counting `paper_runbook` rows either side of a full execute-and-settle.

Verify a console is up before assuming it is:

```bash
tmux ls                                  # session names drift; check, do not trust
curl -s localhost:8086/api/health        # {"ok":true,...,"port":8086,"schema":"paper_replay",...}
```

### 7.1 The replay console (8086)

It answers: *"what would this model have said, at this minute, against the book
that actually existed then — and what would it have won?"* It drives the **same**
pipeline as the live console and replaces only the sources that read a clock or a
network:

| replacement | closes |
|---|---|
| `PreloadedBook` | one query for the whole day's ladders, read with `searchsortedlast(stamps, as_of)` — a tick from after the replayed instant is *unreachable*, not merely unqueried |
| `PreloadedLineups` | `scraped_at <= as_of`, with **no historical fallback behind it** — before the scrape lands a player model prices with no lineup and contributes exactly zero |
| `PointInTimeLineupRatings` | `:player_lineup_ratings_map` is emitted over *every* match in the store, so for a finished fixture it already holds the teamsheet that took the field; this materialiser overwrites it each tick from the visible XI |
| `FrozenIdentity` | resolved once at load |

A replay that relaxed a gate, the instrument rule, the stake rounding, the market
set or the portfolio policy would prove nothing about a Saturday, so none of them
are relaxed.

The clock is **minutes relative to kick-off**, T−60 to T+105; the absolute
`as_of` is derived from it and never stored separately. Latents are memoised on a
hash of the point-in-time lineups, so `Features.create_features` runs once per
*model* (~10 s team-level, ~80 s hybrid) rather than once per tick; a tick then
costs ~0.5 s, and 60× means one simulated minute per wall second.

`2026-08-08` is the default and the only replayable day carrying both an archived
book and a provisional-XI scrape (nine fixtures, published T−13 to T−40). On that
day, fixture 16362410 (1X2 away), `m12`'s `p_model` steps 0.3934 → 0.3637 across
the XI drop while `m00` and `m05` are bit-identical across the same transition.
That control is what makes the move attributable to the lineup rather than to the
book.

### 7.2 The Gödel-terminal workspace

The replay page is a modular quant workspace: six draggable, resizable,
stackable windows — **Staking Ticket, Slate Radar, Multi-Ladder Desk, Trajectory
Chart, Team Form, Model Scorecard** — with top-dock toggles, a bottom dock for
minimised panels, tile/cascade, and a layout persisted per browser.

* **Multi-Ladder Desk** — a Bet Angel exchange screen: three bid and three ask
  levels per runner, spread in currency **and in Betfair ticks**, three-level
  weight of money (WOM), the de-vigged market probability beside `p_model`, fair
  odds, EV, and the simulated order marked on the runner it would actually touch
  with the £ consumed per level.
* **Trajectory Chart** — market best back/lay against stepped model fair odds,
  the T−25…T−12 execution band, the XI drop, a needle synced to the replay clock,
  and the matched-volume S-curve beneath.
* **Team Form & Lineup Delta** (`fixture_stats`) — last five results with BBC
  shots/SoT, plus the announced XI against the regular one. Form reads matches
  strictly *before* the replayed day and the XI through the pipeline's own
  point-in-time source, so the panel cannot see a teamsheet the model could not.
* **Model Scorecard** (`model_scorecard`) — three sources kept **apart** because
  they are three different claims: `fold_results` (what the run scored),
  `match_latents` scored against the de-vigged close on one match set (the only
  figure that earns "vs market", CRPS via `Evaluation.compute_crps`), and
  `paper_replay.clv_audit` (what this account's bets did). A run that wrote no
  proper scores reports `nothing` **with the reason**, never an average of the
  folds that did.

### 7.3 The dynamic slate re-solver

A human places bets one at a time, and the vector the account ends up holding is
not the vector `Portfolio` solved. `StakingOverride` records the three facts the
console cannot derive — which legs filled, at what price, and which were skipped —
and `resolve_slate_with_overrides` re-optimises around them:

```
max k   s.t.  Σ_t log Σ_i p_t,i (1 + [R_t a_frozen]_i + k [R_t a_free]_i)^(-λ) ≤ 0
              Σ committed_frac + k · Σ free_frac ≤ exposure_cap
```

Frozen legs enter the wealth relative as **constants**, which is what a placed bet
is; one factor scales what is left. This is a constrained form of the same
`SlateDrawdown` solve and **not a rescale**, so `Portfolio._bisect_k`'s `[0,1]`
search becomes `[0, k_cap]` — a skipped leg can genuinely entitle the survivors to
more than the full-slate solution gave them (measured: skipping one leg of a
25-leg card re-solved the other 24 at `k = 1.0167`).

A placed leg is never reduced (the money is at the venue), its payoff column is
repriced at the price it actually got, and a commitment that fills the cap sends
the uncommitted legs to zero rather than treating a negative residual as room.
`execute!` commits `active_slate`, which is the re-solved vector when one is valid
for the priced minute; a reprice retires the re-solve and keeps the overrides.

Test R36 pins the counter-intuitive finding: under a **joint** log-utility budget
the surviving legs do not automatically grow when one is skipped. At the optimum
the per-fixture penalty terms sum to zero with some negative, so removing a
subsidising leg *tightens* the constraint. Growth is unambiguous only when the
exposure cap is what binds; both regimes are asserted.

### 7.4 What the consoles refuse to pretend

* **No traded VWAP.** `betfair_live.order_book_1m` archives resting depth and a
  running matched total, never a traded price series. The desk shows a **book
  VWAP** and labels it as such.
* **No levels beyond the third.** The archive carries at most three, verified over
  635,765 rows; every depth and WOM figure says `(3 lvls)`.
* **No model opinion on a gated fixture.** A refused fixture shows its book and an
  empty model column, never a number derived from inputs the pipeline declined to
  use. A fold that cannot represent a fixture refuses it **by name** in the
  `NOT COVERED BY …` panel rather than pricing it at the league mean.
* **No xG in the form panel.** `sofascore.match_statistics` holds zero rows for
  tournaments 56/57, so the panel carries shots and says why.
* **`LadderSweep` is the optimistic fill model.** It crosses up to three archived
  levels instantly; the live system rests at the touch. A replay P&L built on it
  is an **upper bound**. `fill_model` is recorded per fill row so a
  `ladder_sweep_v1` track and a `touch_only` one are never pooled by accident.
* **After kick-off the posterior is pre-game and the book is in-play.** Post-T−0
  "edges" measure that gap, not a signal. Execute is disabled and the API refuses
  unless `{"allow_in_play": true}` is passed deliberately.

### 7.5 API surface

The live console (8085) serves `/api/snapshot`, `/api/health`, `/api/execute`,
`/api/kill`. The replay console (8086) adds the VCR, the desk, the ticket and the
intelligence widgets:

```
GET  /api/snapshot | /api/health | /api/replay/matchdays
GET  /api/replay/ladder ?match_id=&market=MATCH_ODDS|OVER_UNDER_25|BOTH_TEAMS_TO_SCORE
GET  /api/replay/history ?match_id=&symbol=&market=[&from=&to=]
GET  /api/replay/stats | /api/replay/model_scorecard
POST /api/replay/play | pause | speed | step | jump | seek
POST /api/replay/set_model {"model":"m00|m05|m12"} | set_matchday {"day":"2026-08-08"}
POST /api/replay/stake/override | stake/resolve | stake/reset
POST /api/replay/execute [{"allow_in_play":true}] | settle | reset
```

Every control also accepts a query string (`POST /api/replay/seek?t=-15`), so the
whole console is drivable from `curl`. Full operator detail, keyboard map and the
replayable-day table are in
[`current_development/match_day_inference/README.md`](current_development/match_day_inference/README.md).

---

## 8. AD-safety — critical for Turing models

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
evaluation; the Scottish joint/hybrid arms compile to 0.025–0.028 ms, and tape
length grows with **model structure**, not fixture count. Details in
`docs/turing_ad_performance_guide.md`.

---

## 9. Extension recipes

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

**New calibration weight law** — subtype `AbstractCalibrationWeightLaw` in
`src/Calibration/types.jl` and implement `calibration_weight(law, delta)`,
`is_identity_law(law)` and `law_label(law)`. **New dispersion map** — subtype
`AbstractDispersionMap` and implement `residual_map(map, w_h, w_a)` returning the
2×2 map row-major. Either is "add a struct + one method"; no existing file changes.

**New L2 shift model (deprecated path)** — subtype `AbstractLayerTwoModel`,
implement `fit_calibrator(model, data, config)` and
`apply_calibration(fitted_model, new_data)` in `src/Calibration/shift_models/`.
Prefer a weight law or a dispersion map: a selection-level shift cannot be coherent
across derivative markets.

**New MatchDay source or gate** — subtype the relevant seam in
`src/MatchDay/interfaces.jl` and add the implementation under
`src/MatchDay/implementations/`. A source that reads a clock or a network needs a
point-in-time twin in `replay_state.jl` before it can be replayed honestly.

---

## 10. Test execution — pick the fastest tier that covers your change

```bash
# 1. Single module (FASTEST, 15-20s)
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/unified_portfolio_tests.jl")'

# 2. Concurrent full suite (4 workers, ~40-45s)
julia --project -t 8 test/run_parallel_tests.jl

# 3. Standard sequential suite (full baseline, ~3.5 min)
julia --project -t 8 test/runtests.jl

# 4. MatchDay replay console (1,015 assertions; NOT in the parallel runner)
julia --project -t 8 test/test_matchday_replay.jl

# 5. Layer-2 calibration v2 alone (in runtests.jl; T10 needs mcmc_experiments)
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/test_calibration_v2.jl")'
```

Suites live in `test/` (`data_tests.jl`, `features_tests.jl`,
`pregame_tests.jl`, …). `run_parallel_tests.jl` dispatches the seventeen
module-level suites across four worker processes; the MatchDay live-pipeline and
replay suites are run on their own because their upper tiers need `betdb`, the
experiment database, and a warm DataStore cache.

Verified 2026-09-03: `runtests.jl` **3,195 / 3,195** in 5m42s;
`test_matchday_replay.jl` **1,015 / 1,015** in 3m03s with no tier skipped.
`run_parallel_tests.jl` reports **16 / 17** — `features_tests.jl` fails in
isolation with `UndefVarError: SplitClockProbe`, because that probe type is
defined in `splitting_tests.jl` and the two share a `Main` only in the sequential
runner. This is the known open [T007](docs/tickets/T007-parallel-feature-test-hidden-dependency.md),
not a regression; confirm against `runtests.jl` before chasing it.

The replay suite runs in four tiers — pure (clock and filtration contract, no
database), the ladder desk, the ledger (`paper_replay` execution and settlement
plus the `paper_runbook` isolation assertion), and models (a real Saturday, real
canonical fits, hot-swapping, the lineup shock). The ledger and model tiers skip
**with a message** when the database or cache is out of reach, never silently. A
"passed" line from a tier that skipped is not evidence.

---

## 11. Infrastructure & compute rules

**Topology**

| Host | Role |
|---|---|
| `archpc` (local laptop) | Development — `/home/james/bet_project/BayesianFootball`; 8 physical cores / 16 SMT; also hosts PostgreSQL `betdb` on **5433** and runs both MatchDay consoles (8085, 8086) |
| `mcmc-beast` | Compute — 16 physical cores (32 SMT), 64 GB RAM, `/root/BayesianFootball`; hosts PostgreSQL `mcmc_experiments` on **5432** |

The three-node Tailscale mesh diagram, per-host specs and the standard agent
prompting block are in
[`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](docs/architecture/ai_agent_infrastructure_and_execution_context.md).

**CPU & threads**

- Launch Julia with `-t 16` on `mcmc-beast`, `-t 8` on `archpc`.
- Always `using ThreadPinning; pinthreads(:cores)` before starting MCMC chains.
- Always `LinearAlgebra.BLAS.set_num_threads(1)` during sampling, to prevent
  CPU oversubscription.
- Keep local inference daemons disabled (`systemctl disable/stop ollama`).
- Do not launch a production grid on `mcmc-beast` while another is sampling.

**Code syncing & cache safety**

- `rsync -avz --exclude '.cache/' --exclude 'data/' ...` to push code without
  clobbering remote data caches.
- `.cache/datastore_<Segment>.jls` holds pre-extracted DataFrames. When a new SQL
  column lands, regenerate the cache or sync it from the host with the freshest
  fetch. `load_datastore_cached(segment; max_age_hours = N)` controls staleness.
- Point-in-time feature guards must accept match-row values:
  `stamp_ok = (stamp === nothing) || (at === nothing) || (stamp < at)`.

---

## 12. Remote execution protocol

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

## 13. Experiment database & config truth protocol

See `docs/guides/experiment_database_and_config_truth_guide.md`; §3 above says
which database this is and how it differs from `betdb`.

1. **Keep credentials out of source and output.** Never commit, paste into
   prompts, or print a raw database password or credential-bearing URL.
   Construct `PostgresStorage(experiment_name)` and let it resolve
   `ENV["BF_EXPERIMENTS_DB_URL"]` or libpq's `~/.pgpass`. Its masked `show` is
   safe; printing `storage.conn_str` is not. The same rule applies to
   `BF_DB_URL`.
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
6. **Extend rather than refit when a season rolls forward.**
   `preview_extension(db, run_uuid, ds)` reports only the fold positions absent
   from `fold_results`; `extend_fit` samples those and updates diagnostics,
   latents, the artefact and telemetry in one transaction; `extend_portfolio`
   prices only the fixtures absent from the bet ledger and continues from the
   closing bankroll.
7. **A live slate reads this database, it does not write it.**
   `MD.canonical_fit(PostgresStorage(experiment), run_name)` loads a completed,
   converged run; everything the operator then does is written to
   `betdb.<paper_schema>`. If you find yourself writing paper-trading rows into
   `mcmc_experiments`, re-read §3.3.
