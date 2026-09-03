# BayesianFootball.jl

A Bayesian hierarchical modelling framework for football (soccer) analytics, market
evaluation, portfolio management and **live match-day execution**, in Julia.

Agents and contributors: **[`AGENTS.md`](AGENTS.md)** is the canonical working guide.
This file is the orientation map.

---

## 🚀 Overview & key capabilities

An end-to-end Bayesian quantitative workflow, from raw scrape to a stake at an exchange:

* **Layer 0 — Memory-optimised DataStore**: concurrent SQL extraction via `LibPQ`, strict typed schemas (`InlineStrings`), vig-removed market math.
* **Layer 1 — Composable count builder & master engines**: mathematical Lego blocks assembled into `PoissonCountModel` or `NegBinCountModel` with $O(1)$ compiled ReverseDiff tapes.
* **Layer 2 — Unified inference, latents & experiment truth (`Fit`)**: multi-threaded NUTS/ADVI, automated convergence auditing ($\hat R$, ESS, divergences, BFMI, tree depth), atomic disk persistence, and PostgreSQL run tracking with canonical configuration discovery.
* **Layer 3 — Unified evaluation**: zero-copy `OddsView` over match markets with bit-identical LogLoss, CRPS, Brier, RPS and Expected Calibration Error against market closing prices.
* **Layer 4 — Zero-allocation portfolio, staking & audit**: $O(1)$ indexed lookups (`OddsIndex`), fold-level pre-allocated workspaces (`BookWorkspace`), Baker-McHale shrinkage, fractional Kelly under a joint slate budget, and queryable PostgreSQL portfolio/trade persistence.
* **Layer 5 — MatchDay execution**: point-in-time slate pricing, a transactional paper ledger, and two interactive browser consoles — **live** and **replay** — driven by the same pipeline.

---

## 🗄️ Two databases, two questions

The single most important orientation fact about this repository: there are **two**
PostgreSQL services and they answer different questions.

| | **`betdb`** — what happened, and what we did | **`mcmc_experiments`** — what we fitted, and what it scored |
|---|---|---|
| Env var | `BF_DB_URL` (required, no default) | `BF_EXPERIMENTS_DB_URL`, else `~/.pgpass` |
| Host | `archpc:5433` | `mcmc-beast:5432` |
| Entry point | `Data.load_datastore_sql`, `MatchDay.paper_connection` | `Training.PostgresStorage(experiment_name)` |
| Organised by | one PostgreSQL **schema per domain** | one flat schema, namespaced by `experiment_name` |

**`betdb` — raw football data and the paper ledgers**

| Schema | Holds |
|---|---|
| `sofascore` | fixtures/`events`, `seasons`, `match_player_lineups`, `lineup_provisional` (pre-match XI scrape, stamped `scraped_at`), `match_statistics`, `match_incidents`, `match_odds` |
| `bbc` | `match_meta`, `match_stats`, `match_lineup`, `live_text` — the commentary stream the **proxy xG** arm is built from |
| `betfair` | `match_meta` (identity crosswalk), `markets`, `odds_history` — the **closing-line archive** for CLV and the de-vigged market baseline |
| `betfair_live` | `market_metadata`, `order_book_1m` — one-minute archived exchange ladders, at most three levels per side |
| `paper_runbook` | the **live** paper ledger (console on 8085) |
| `paper_replay` | the **replay** paper ledger (console on 8086) |

**`mcmc_experiments` — Bayesian experiment and portfolio tracking**

`config_registry` (canonical named components) · `configs` (the hash-addressed inference
recipe) · `runs` (status, Git provenance, timings) · `fold_results` (convergence audit and
OOS proper scores) · `match_latents` (point-in-time posterior predictions with compressed
draws) · `fit_artifacts` (the exact serialized `Fit`) · `portfolio_runs` /
`portfolio_bets` / `portfolio_artifacts` (simulation headline, trade ledger, exact
`PortfolioResult`).

The two are linked by `paper_slates.model_run_id → runs.run_id`, carried as an opaque UUID
with no foreign key — they are separate servers. Rationale and the full schema reference:
[`docs/guides/experiment_database_and_config_truth_guide.md`](docs/guides/experiment_database_and_config_truth_guide.md).

> **Credentials never appear in this repository.** `BF_DB_URL` comes from the environment
> (`.env`, git-ignored); `PostgresStorage` resolves `BF_EXPERIMENTS_DB_URL` or lets libpq
> read `~/.pgpass`, and its `show` method masks the connection string.

---

## ⚡ Quick start: end-to-end pipeline

Train and simulate a two-arm joint model with a player-lineup pillar — the current
production shape — on the unified v2 stack:

```julia
using BayesianFootball
using DataFrames, Dates, ThreadPinning

# 1. Thread topology & BLAS isolation
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# 2. Load cached tournament data (betdb → .cache/datastore_ScottishLower.jls)
ds = Data.load_datastore_cached(Data.ScottishLower())

# 3. Assemble the model with the composable count builder
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

# 4. Define the unified inference recipe
fit_cfg = FitConfig(
    name      = "m12_joint_hybrid_synergy",
    model     = model,
    splitter  = Data.CVConfig(target_seasons = ["24/25", "25/26"], window_seasons = 3),
    sampler   = NUTSConfig(n_samples = 1_000, n_chains = 4, target_accept = 0.85),
    execution = AutoExecution()  # resolves to QueuedExecution or ThreadedExecution
)

# 5. Register the canonical recipe BEFORE scheduling compute.
#    Credentials resolve from BF_EXPERIMENTS_DB_URL or ~/.pgpass.
db = PostgresStorage("scottish_lower_joint_player_2426")
ensure_schema!(db)
save_model(db, "m12_joint_hybrid_synergy", model; tags = ["production"])
save_splitter(db, "split_2426", fit_cfg.splitter; tags = ["walkforward"])
save_sampler(db, "nuts_4x1000", fit_cfg.sampler; tags = ["production"])
save_config(db, "fit_m12_2426", fit_cfg; tags = ["production"])

# 6. Train and persist the queryable experiment record
fit    = fit_model(fit_cfg, ds)
run_id = save_fit(fit, db)          # UUID; an identical recipe returns the existing run

# 7. Evaluate forecast accuracy against closing odds
eval_report = evaluate_predictions(fit, ds)
println(eval_report)

# 8. Simulate a fractional-Kelly portfolio under a risk policy
spec = BookSpec(
    markets   = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS()]),
    price     = DeArb(),
    allocator = KellyLogUtility(),
    shrink    = BakerMcHale()
)
policy = PolicySpec(
    trust = FlatTrust(0.25),      # quarter Kelly
    risk  = SlateDrawdown(20.0),  # 20% joint slate risk budget
    cap   = FixedCap(0.25)        # 25% max simultaneous exposure
)
save_book_spec(db, "closing_main", spec; tags = ["production"])
save_policy_spec(db, "quarter_kelly", policy; tags = ["production"])

result, books, rep = run_portfolio_simulation(spec, policy, fit, ds.odds, ds)
portfolio_run_id = save_portfolio_db(result, run_id, db; book_spec = spec, policy_spec = policy)
display(portfolio_report(result))
```

---

## 🏗️ Architecture layers

### 🗄️ Layer 0: Data (`src/Data/`)
Extraction, transformation and validation of raw PostgreSQL data into memory-optimised Julia
`DataFrame`s.
* **`DataStore`**: strictly typed container holding domain frames (`matches`, `odds`, `betfair_odds`, `statistics`, `lineups`, `incidents`, `bbc`, `bbc_events`).
* **`Markets`**: implied probabilities, vig removal, fair odds, closing line movement.
* **Fetch → Process → QA**: 3-step contract enforcing type safety and logical constraints before data reaches a model.

### 🧠 Layer 1: Composable count builder & models (`src/models/`)
* **`CountModelBuilder`**: assemble models modularly through generic `add` dispatches.
* **Observations**: `PoissonObservation`, `NegBinObservation`, and `JointGammaPoissonObservation` — the two-arm likelihood that reads one shared log-intensity with a Gamma density on proxy xG and a Poisson density on goals.
* **Master engines**: `PoissonCountModel`, `NegBinCountModel`, `DynamicPxGRecombModel` (multi-task proxy xG + open-play goals with squad market wealth), `DynamicCopulaGoalsModel` (Frank copula over NegBin marginals).
* **Component blocks**:
  * *Interceptions*: `GlobalInterception`, `HierarchicalInterception`, `ConstantInterception`.
  * *Dynamics*: `TimeDecayDynamics`, `GRWDynamics`, `MultiScaleGRW`.
  * *Home advantage*: `GlobalHomeAdvantage`, `SingleHomeAdvantage`, `HierarchicalHomeAdvantage`.
  * *Covariates*: `ProductionWealthCovariate` (Richards-sigmoid age-adjusted squad value), `DistanceCovariate` (Haversine travel), `BenchDepthCovariate`.
  * *Player pillar*: `PlayerLineupPillar` with `OutfieldPlayerAggregation` or `BenchWeightedPlayerAggregation(w_bench = 0.10)`, over ridge-fitted shots-RAPM or pxG-RAPM ratings.

### 🔄 Layer 2: Unified inference, latents & experiment truth (`src/training/`, `src/models/latents/`)
* **`Fit`**: the atomic result of a trained model — configuration, fold results, posterior latents, convergence diagnostics, metadata.
* **`fit_model(FitConfig, ds)`**: orchestrator supporting `AutoExecution`, `QueuedExecution`, `ThreadedExecution`, `SequentialExecution`.
* **Automated convergence audit (`ConvergenceSummary`)**: split $\hat R \le 1.05$, bulk/tail ESS, divergences, BFMI, tree-depth saturation.
* **Typed latents (`CountLatents`)**: structured $\lambda_{\text{home}}, \lambda_{\text{away}}$ matrices feeding zero-allocation score kernels (`SmileScoreGrid`).
* **PostgreSQL experiment tracking**: `PostgresStorage` stores queryable runs, fold diagnostics, match latents and exact `Fit` artefacts; `DualStorage` also keeps an atomic filesystem copy.
* **Config truth engine**: `config_registry`, `save_model`, `save_config`, `search_configs`, `show_config` give named, tagged, hash-addressed recipes shared across machines.

### 📊 Layer 3: Unified evaluation (`src/evaluation/`)
* **`OddsView`**: zero-copy dense view over odds matrices with strict point-in-time (`stamp < kickoff`) assertion guards.
* **`evaluate_predictions(fit, ds)`**: prices match probabilities across the posterior score grid and computes LogLoss, CRPS, RPS, Brier, ECE/MCE and reliability diagrams — for the model *and* the de-vigged Betfair closing line, on the same fixture set.
* **Convergence refusal**: an unconverged fit cannot be evaluated.

### 💰 Layer 4: Zero-allocation portfolio, staking & audit (`src/Portfolio/`)
* **`OddsIndex`**: $O(1)$ indexed market lookups instead of full-frame scans.
* **`BookWorkspace`**: one pre-allocated matrix and probability buffer per fold.
* **`simulate_portfolio` / `run_portfolio_simulation`**: bankroll trajectories under fractional Kelly, a **joint** slate drawdown budget (one $k$ for every fixture that settles together), exposure caps and commission modelling.
* **PostgreSQL audit trail**: `save_portfolio_db` stores headline ROI/risk metrics, individual bets, and an exact `PortfolioResult` linked to the model-run UUID.
* **Convergence gating**: an unconverged model throws `ConvergenceRefusal` before capital is risked.

### 🎛️ Layer 5: MatchDay execution (`src/MatchDay/`)
Point-in-time slate pricing through named seams:

```
fixtures → identity → lineups → book → features → inference → gate → stake_sheet
```

**The slate is the execution atom.** `Portfolio` solves one joint problem for every fixture
that settles together, so the stake vector is only valid *as a vector* — reservation is one
transaction for the whole of it, and `account_ledger (slate_id) WHERE kind = 'RESERVE'` is
UNIQUE, which makes double-reserving a slate unrepresentable rather than merely guarded.
Nothing here samples: `MD.canonical_fit` loads a completed run out of `mcmc_experiments`.

---

## 🖥️ The MatchDay consoles

Two long-running browser consoles, side by side, neither able to reach the other's rows.

| | runner | port | tmux | schema | clock |
|---|---|---|---|---|---|
| **Live** | `r07_serve_console.jl` | **8085** | `matchday_console` | `betdb.paper_runbook` | `now()` |
| **Replay** | `r08_replay_console.jl` | **8086** | `replay_run` | `betdb.paper_replay` | a scrubber |

```bash
julia --project -t 8 current_development/match_day_inference/r07_serve_console.jl   # → :8085
julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl  # → :8086

R08_DAY=2026-08-15 R08_MODEL=m12 julia --project -t 8 \
  current_development/match_day_inference/r08_replay_console.jl
```

Isolation is **structural, not conventional**: `assert_replay_schema` refuses `paper_runbook`
at every ledger call site and `serve_replay` refuses to bind 8085. Both are asserted in
`test/test_matchday_replay.jl` (R1, R2, R18) rather than left to convention.

### The replay console — an interactive backtest

*"What would this model have said, at this minute, against the book that actually existed
then — and what would it have won?"*

It drives the **same** pipeline as the live console, with the same gates, instrument rule,
stake rounding, market set and portfolio policy, and replaces only the sources that read a
clock or a network. Three leaks are closed structurally: `PreloadedBook` reads the archived
ladder with `searchsortedlast(stamps, as_of)` so a future tick is unreachable;
`PreloadedLineups` filters `scraped_at <= as_of` with **no** historical fallback behind it;
and `PointInTimeLineupRatings` rebuilds the player ratings map from the *visible* XI each
tick, rather than from the teamsheet that finally took the field.

The clock is minutes relative to kick-off (T−60 to T+105). Latents are memoised on a hash of
the point-in-time lineups, so features are built once per model (~10 s team-level, ~80 s
hybrid) and a tick then costs ~0.5 s — 60× playback is one simulated minute per wall second.

**The Gödel-terminal workspace.** Six draggable, resizable, stackable windows with top-dock
toggles, a bottom dock for minimised panels, tile/cascade, and a layout persisted per browser:

| Window | Shows |
|---|---|
| **Slate Radar** | the live card grid, plus a colour-coded WOM pill and three-level depth per leg |
| **Multi-Ladder Desk** | a Bet Angel exchange screen — three bid and three ask levels per runner, spread in currency **and Betfair ticks**, weight of money, de-vigged market probability beside `p_model`, fair odds, EV, and the simulated order marked on the runner it would actually touch |
| **Trajectory Chart** | market best back/lay against stepped model fair odds, the T−25…T−12 execution band, the XI drop, a clock needle, and the matched-volume S-curve |
| **Team Form & Lineup Delta** | last five results with BBC shots/SoT, and the announced XI against the regular one — read strictly *before* the replayed day, through the pipeline's own point-in-time source |
| **Model Scorecard** | proper scores from three sources kept deliberately apart: `fold_results` (what the run scored), `match_latents` vs the de-vigged close (the only figure that earns "vs market", CRPS included), and `paper_replay.clv_audit` (what this account's bets did) |
| **Staking Ticket** | manual execution with a constrained dynamic re-solve (below) |

**The dynamic slate re-solver.** A human places bets one at a time, so the vector the account
ends up holding is not the vector `Portfolio` solved. `StakingOverride` records which legs
filled, at what price, and which were skipped, and `resolve_slate_with_overrides` re-optimises
around them:

$$\max_k \quad \text{s.t.} \quad \sum_t \log \sum_i p_{t,i}\bigl(1 + [R_t a_{\text{frozen}}]_i + k\,[R_t a_{\text{free}}]_i\bigr)^{-\lambda} \le 0, \qquad \sum \text{committed} + k \sum \text{free} \le \text{cap}$$

Frozen legs enter the wealth relative as **constants** — which is what a placed bet is — and a
single factor scales what is left. This is a constrained form of the same `SlateDrawdown`
solve, **not a rescale**, so the bisection searches $[0, k_{\text{cap}}]$ rather than $[0,1]$:
a skipped leg can genuinely entitle the survivors to more than the full-slate solution gave
them. A placed leg is never reduced, its payoff column is repriced at the price it actually
got, and a commitment that fills the cap sends the uncommitted legs to zero.

**What it refuses to pretend.** No traded VWAP (the archive holds resting depth and a running
matched total, never a traded price series — a *book* VWAP is shown and named); no levels
beyond the third (verified over 635,765 rows); no model opinion on a gated fixture (a fold
that cannot represent a fixture refuses it **by name**); no xG in the form panel
(`sofascore.match_statistics` holds zero rows for tournaments 56/57). `LadderSweep` is the
optimistic fill model and a replay P&L built on it is an **upper bound**. After kick-off the
posterior is pre-game and the book is in-play, so Execute is disabled and the API refuses
unless `{"allow_in_play": true}` is passed deliberately.

**Routes** (every control also accepts a query string, so the console is drivable from `curl`):

```
GET  /api/snapshot | /api/health | /api/replay/matchdays
GET  /api/replay/ladder | /api/replay/history | /api/replay/stats | /api/replay/model_scorecard
POST /api/replay/play | pause | speed | step | jump | seek
POST /api/replay/set_model | set_matchday
POST /api/replay/stake/override | stake/resolve | stake/reset
POST /api/replay/execute | settle | reset
```

Full operator detail — keyboard map, the replayable-day table, the suggested pass —
in [`current_development/match_day_inference/README.md`](current_development/match_day_inference/README.md).

---

## 📈 Model generations and measured results

Four paradigms, each a 40-fold walk-forward grid over Scottish League One and Two
(tournaments 56/57), seasons 24/25 + 25/26 — 710 held-out matches, 2,899 scored market
observations. Every number below is recorded in the linked suite's README beside its run
UUIDs.

| Gen | Suite | Paradigm | Headline |
|---|---|---|---|
| **1** | [`01_poisson_2426_grid`](experiments/scottish_lower/01_poisson_2426_grid/README.md) | Poisson; baseline, squad wealth, travel distance, age-adjusted production wealth | `m05` LogLoss **0.6597**; Betfair backtest +125% to +140% bankroll |
| **2** | [`02_negbin_2426_grid`](experiments/scottish_lower/02_negbin_2426_grid/README.md) | Negative Binomial; empirical overdispersion | $\hat r \approx 26.0$–$26.5$ (mild); LogLoss **0.6598** — no material gain over Poisson |
| **3** | [`03_joint_gamma_poisson`](experiments/scottish_lower/03_joint_gamma_poisson/README.md) | **Two-arm joint**: shared latent $\mu$, Gamma arm on BBC commentary proxy xG, Poisson arm on goals | LogLoss **0.6571** against a Betfair close of 0.6568 — the second likelihood is worth ~5× the best covariate |
| **4** | [`05_player_lineup_and_pxg_fusion`](experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md), [`06_joint_player_lineup_fusion`](experiments/scottish_lower/06_joint_player_lineup_fusion/README.md) | **Joint + player-lineup hybrid**: `PlayerLineupPillar` (shots-RAPM / pxG-RAPM, starters + bench at fixed $w_{\text{bench}} = 0.10$) composed beside 180-day team time decay | `m12` ECE **0.0100** vs Betfair close **0.0139**; **+136.6%** bankroll at **1.416** annual Sharpe |

### Generation 4 in detail (experiment 06, `scottish_lower_joint_player_2426`)

$$\eta_{\text{home},i} = \mu + \gamma + \alpha_{\text{home}} + \beta_{\text{away}} + L_{\text{home},i} + \textstyle\sum_c w_c x_{c,i}, \qquad L_{\text{home},i} = w_{\text{att}} R_{\text{home},i} - w_{\text{def}} R_{\text{away},i}$$

$R_{s,i}$ is the aggregated RAPM rating of side $s$'s named teamsheet. RAPM is a ridge fit —
**never sampled** — over each fold's frozen history block, so a target fixture never
contributes to the ratings that price it.

| Model | LogLoss | ECE | Bankroll | Sharpe | Max DD | Bets |
|---|---:|---:|---:|---:|---:|---:|
| `m13_joint_composite` (+ distance) | 0.64324 | **0.0088** | **+140.2%** | 1.453 | −21.1% | 1,468 |
| `m12_joint_hybrid_synergy` 🏆 | 0.64337 | 0.0100 | +136.6% | 1.416 | −20.2% | 1,462 |
| `m05_joint_production_wealth` (control) | **0.64299** | 0.0149 | +131.2% | **1.481** | **−19.1%** | 1,455 |
| `m10_joint_player_shots_bench` | 0.64440 | 0.0090 | +112.2% | 1.217 | −20.0% | 1,462 |
| `m09_joint_player_shots_outfield` | 0.64448 | 0.0094 | +111.6% | 1.216 | −20.0% | 1,463 |
| `m11_joint_player_pxg_bench` | 0.64485 | 0.0104 | +120.5% | 1.239 | −20.7% | 1,455 |
| *Betfair closing line* | *0.64182* | *0.0139* | — | — | — | — |

**Read that honestly.** The lineup arms do **not** win on LogLoss — the team-state control is
still the sharpest forecaster. What the teamsheet buys is **calibration**: every lineup arm
roughly halves the control's ECE and beats the Betfair closing line's, and it is calibration,
not sharpness, that converts into Kelly bankroll growth. The pre-MCMC EDA (`r59`) refused the
hypothesis that lineups alone improve team state, and refused travel distance; it supported
"squad wealth is complementary to RAPM, not duplicative" strongly. The hypotheses were written
so they could fail visibly, and some did.

`m12` is the hybrid pillar the MatchDay consoles load; `m00_poisson_control` and
`m05_joint_production_wealth` are the team-level controls that make an observed lineup move
attributable to the teamsheet rather than to the book.

---

## 🗂️ Repository map

| Path | Contents |
|---|---|
| `src/` | The package: `Data`, `Features`, `Models`, `Samplers`, `Training`, `Experiments`, `Predictions`, `Evaluation`, `Portfolio`, `MatchDay`, `Signals`, `BackTesting`, `Calibration`, `MyDistributions` |
| `experiments/` | Completed, reproducible benchmark suites — `<segment>/NN_<topic>/` with loader, smoke runner, production runner, comparison, portfolio backtest and a README carrying measured results and run UUIDs |
| `current_development/` | Active prototypes, `lXX_` loader + `rXX_` runner pairs. `match_day_inference/` is the live operational suite |
| `docs/` | Guides, architecture notes, model theory, tickets, and an `archive/` of work that was measured and *not* adopted |
| `test/` | The suite (see below) |
| `scripts/` | Provisioning, including `setup_experiments_db.sh` |

---

## 🧪 Testing

```bash
# 1. Concurrent full suite — 17 module suites over 4 workers (~40s)
julia --project -t 8 test/run_parallel_tests.jl

# 2. Standard sequential suite (~3.5 min)
julia --project -t 8 test/runtests.jl

# 3. Targeted single suite (~15s)
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/unified_portfolio_tests.jl")'

# 4. MatchDay replay console — 1,015 assertions, NOT in the parallel runner
julia --project -t 8 test/test_matchday_replay.jl
```

The replay suite runs in four tiers: pure (clock and filtration contract, no database), the
ladder desk, the ledger (`paper_replay` execution and settlement, plus a direct assertion that
`paper_runbook` row counts are unchanged either side of it), and models (a real Saturday, real
canonical fits, model hot-swapping, the lineup shock). The ledger and model tiers skip **with a
message** when the database or DataStore cache is out of reach — never silently, so a "passed"
line from a tier that skipped is not evidence.

Last verified 2026-09-03: `runtests.jl` **3,195 / 3,195**, `test_matchday_replay.jl`
**1,015 / 1,015** with no tier skipped. The parallel runner reports 16/17 —
`features_tests.jl` cannot run in isolation because it uses a probe type defined in
`splitting_tests.jl`; that is the known open
[T007](docs/tickets/T007-parallel-feature-test-hidden-dependency.md), not a regression.

---

## 🖥️ Compute & infrastructure

| Host | Role |
|---|---|
| **`archpc`** (local) | Development, 8 physical cores / 16 SMT. Hosts PostgreSQL `betdb` on **5433** and both MatchDay consoles (**8085**, **8086**) |
| **`mcmc-beast`** | Compute, 16 physical cores / 32 SMT, 64 GB RAM, `/root/BayesianFootball`. Hosts PostgreSQL `mcmc_experiments` on **5432** |

Threading rules, in every runner before any work:

```julia
using ThreadPinning; pinthreads(:cores)        # pin 1:1 to physical cores
LinearAlgebra.BLAS.set_num_threads(1)          # no BLAS oversubscription during sampling
```

Launch Julia with `-t 16` on `mcmc-beast` and `-t 8` on `archpc`; sync code with
`rsync -avz --exclude '.cache/' --exclude 'data/'` so remote data caches are never clobbered.
Full topology, remote-execution protocol and the standard agent prompting block:
[`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](docs/architecture/ai_agent_infrastructure_and_execution_context.md).
