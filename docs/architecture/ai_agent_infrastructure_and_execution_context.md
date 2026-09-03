# AI Agent Infrastructure & Remote Execution Context Guide

> **BayesianFootball.jl Infrastructure & AI Agent Context Reference**  
> **Status:** Active Standard | **Target Systems:** `archpc` (development laptop, `betdb` on 5433, MatchDay consoles on 8085/8086) and `mcmc-beast` (compute node, `mcmc_experiments` on 5432)

---

## 1. Network Topology & Machine Hierarchy

The `BayesianFootball.jl` platform operates across a three-node Tailscale mesh network. All AI agents and developers must respect the separation of concerns across these hosts:

```
+-----------------------------------------------------------------------------------------+
|                                    TAILSCALE MESH                                       |
|                                                                                         |
|   +--------------------------+       Git Push / rsync      +------------------------+   |
|   |       LOCAL HOST         | ==========================> |       MCMC-BEAST       |   |
|   |  (AGY / Agent CLI Host)  |                             |  (High-Perf Compute)   |   |
|   |  /home/james/bet_project/| <========================== |  /root/BayesianFootball|   |
|   |      BayesianFootball    |                             |  AMD 16-Core (32 thr)  |   |
|   |  betdb  :5433 (Postgres) |                             |  mcmc_experiments :5432|   |
|   |  console:8085 (live)     |                             |  (Postgres)            |   |
|   |  console:8086 (replay)   |                             +------------------------+   |
|   +--------------------------+                                         ^                |
|                 |                                                      |                |
|                 | tmux / ssh controller                                | PostgresStorage |
|                 v                                                      |                |
|   +--------------------------+                                         |                |
|   |  Tmux Session Controller | ----------------------------------------+                |
|   |  (e.g. features:1.1)     |    canonical fits, runs, latents, portfolio artefacts    |
|   |  (SSH to mcmc-beast)     |                                                          |
|   +--------------------------+                                                          |
+-----------------------------------------------------------------------------------------+
```

> **`archpc` is both the development laptop and the operational database host.** Older
> diagrams drew it as a separate dedicated server; it is not. It runs the repository
> checkout, PostgreSQL `betdb` on 5433, and both MatchDay consoles (8085 live, 8086 replay).
> `mcmc-beast` is the compute node **and** the host of PostgreSQL `mcmc_experiments` on 5432.

### Machine Specifications & Endpoints

| Node Name | Network Role | Hardware / Specs | Primary Path | Connection / Endpoint |
| :--- | :--- | :--- | :--- | :--- |
| **`archpc`** (local laptop) | Development, agent orchestration, operational database, MatchDay consoles | 8 physical cores / 16 SMT | `/home/james/bet_project/BayesianFootball` | Local terminal / Antigravity CLI / tmux; PostgreSQL `betdb` on `5433`; consoles on `8085` and `8086` |
| **`mcmc-beast`** | Dedicated MCMC sampling & grids, experiment database | AMD Ryzen (16 Physical Cores, 32 SMT threads, 64 GB RAM) | `/root/BayesianFootball` | `ssh root@mcmc-beast` (via Tailscale / LAN); PostgreSQL `mcmc_experiments` on `5432` |

### The Two Databases

| | **`betdb`** | **`mcmc_experiments`** |
| :--- | :--- | :--- |
| Question it answers | what happened, and what we did | what we fitted, and what it scored |
| Host / port | `archpc:5433` (LAN `192.168.1.88`, Tailscale `100.124.38.117`) | `mcmc-beast:5432` |
| Environment variable | `BF_DB_URL` — **required, no default** | `BF_EXPERIMENTS_DB_URL`, else `~/.pgpass` |
| Julia entry point | `Data.load_datastore_sql`, `MatchDay.paper_connection` | `Training.PostgresStorage(experiment_name)` |
| Contents | schemas `sofascore`, `bbc`, `betfair`, `betfair_live`, plus the paper ledgers `paper_runbook` (live console) and `paper_replay` (replay console) | `config_registry`, `configs`, `runs`, `fold_results`, `match_latents`, `fit_artifacts`, `portfolio_runs`, `portfolio_bets`, `portfolio_artifacts` |

They are separate servers. The link across is
`betdb.<paper_schema>.paper_slates.model_run_id → mcmc_experiments.runs.run_id`, carried as an
opaque UUID with no foreign key; a reconciliation job asserts it resolves. Full treatment in
[`../guides/experiment_database_and_config_truth_guide.md`](../guides/experiment_database_and_config_truth_guide.md) §2.

**Credentials never appear in a document, a log, or a prompt.** `BF_DB_URL` is read from the
git-ignored `.env`; `PostgresStorage` resolves its URL from the environment or lets libpq read
`~/.pgpass`, and masks the connection string in its `show` method.

---

## 2. Hardware Architecture, Threads & Process Rules

### 16 Physical Cores & Thread Pinning (`ThreadPinning.jl`)
* `mcmc-beast` has **16 physical cores** (32 SMT virtual hyperthreads).
* **Launch Flag:** Always launch Julia with `-t 16` (not `-t 32` or `-t auto`):
  ```bash
  julia --project=/root/BayesianFootball -t 16
  ```
* **Thread Pinning:** Always pin threads 1-to-1 to physical cores `0..15` to avoid hyperthread resource contention (preventing sibling thread collisions such as `c22`):
  ```julia
  using ThreadPinning
  pinthreads(:cores)
  ```
* **BLAS Oversubscription Guard:** Set BLAS threads to 1 during MCMC to prevent CPU thread starvation:
  ```julia
  using LinearAlgebra
  BLAS.set_num_threads(1)
  ```

### Background Daemon Management (Ollama / Local LLMs)
* CPU-heavy local inference daemons (such as `ollama` / `llama-server`) must remain **disabled** on `mcmc-beast` to prevent background CPU stealing during sampling:
  ```bash
  systemctl disable ollama
  systemctl stop ollama
  ```

### Remote Process Detachment & SIGHUP Prevention
* When launching background scripts on `mcmc-beast` over SSH, use `setsid` or `nohup` with detached standard input (`< /dev/null`) to avoid process termination on SSH session disconnect:
  ```bash
  setsid nohup env SL_RUN_GRIDS=true julia --project=/root/BayesianFootball -t 16 --startup-file=no script.jl > /root/run.log 2>&1 < /dev/null &
  ```
* **Monitoring:**
  * View streaming log: `ssh root@mcmc-beast "tail -f /root/run.log"`
  * CPU activity: Check `btop` in the dedicated tmux window (`mbtop:0`).

---

## 3. Code Synchronization, Data Stores & Patch Integrity

### 1. Rsync Syncing Protocol
When syncing files between the local laptop and `mcmc-beast`, **NEVER overwrite data directories or stale caches**:
```bash
rsync -avz --exclude '.cache/' --exclude 'data/' /home/james/bet_project/BayesianFootball/ root@mcmc-beast:/root/BayesianFootball/
```

### 2. DataStore Cache Synchronization (`.cache/*.jls`)
* DataStore files (`.cache/datastore_<Segment>.jls`) contain pre-extracted match DataFrames.
* When new SQL columns are added (such as `proposed_market_value` in `ds.lineups`), the cache must be regenerated or synced from the machine with the freshest fetch.
* **Point-In-Time (PIT) Guards:** In feature extractors, if a table lacks a distinct `valuation_timestamp` column, the valuation is attached directly to the match row. Guards should evaluate:
  ```julia
  stamp_ok = (stamp === nothing) || (at === nothing) || (stamp < at)
  ```

### 3. DistributionsAD / Distributions Version Pin (Julia 1.12 / ReverseDiff)
`Manifest.toml` pins **`Distributions` at 0.25.126** against `DistributionsAD 0.6.58`. This is
deliberate: `Distributions 0.25.127` changed the `Gamma` argument-check signature, which breaks
`DistributionsADReverseDiffExt` and makes `ReverseDiff` AD tapes for Gamma distributions fail
precompilation — the Gamma arm of the joint observation is exactly what that hits.

Do not resolve past the pin casually. If a bump to `0.25.127+` becomes necessary, the extension
needs the matching one-line signature fix in
`~/.julia/packages/DistributionsAD/.../DistributionsADReverseDiffExt.jl`:

```julia
@check_args(Gamma, (α, α > zero(α)), (θ, θ > zero(θ)))
```

Patching a file inside `~/.julia/packages` is not reproducible across hosts; prefer holding the
pin.

---

## 4. Model Equations & Feature Architecture

### Pure Poisson Model Framework (Generation 1, Scottish Lower Benchmark)

In the log-intensity Poisson regression framework ($\lambda = \exp(\eta)$):

$$\eta_{\text{home}} = \mu + \gamma_{\text{home}} + \alpha_{\text{home}} + \beta_{\text{away}} + w_{\text{wealth}} \cdot \Delta z_{\text{wealth}} + w_{\text{dist}} \cdot z_{\text{dist}}$$

$$\eta_{\text{away}} = \mu + \alpha_{\text{away}} + \beta_{\text{home}} - w_{\text{wealth}} \cdot \Delta z_{\text{wealth}} - w_{\text{dist}} \cdot z_{\text{dist}}$$

#### 1. Squad Wealth Feature ($\Delta z_{\text{wealth}}$)
* **Starting-XI Sum:** $W_h = \sum_{p \in \text{XI}_h} \text{market\_value}_p$, $W_a = \sum_{p \in \text{XI}_a} \text{market\_value}_p$.
* **Log-Ratio:** $\Delta \log W = \log(W_h) - \log(W_a) = \log\left(\frac{W_h}{W_a}\right)$.
* **Standardization:** $\Delta z_{\text{wealth}} = \frac{\Delta \log W - \mu}{\sigma}$ (anchored historically to prevent lookahead).
* **Prior:** $w_{\text{wealth}} \sim \text{truncated}(\text{Normal}(0.10, 0.05), \text{lower}=0.0)$.

#### 2. Travel Distance Feature ($z_{\text{dist}}$)
* **Haversine Distance:** Distance in km / drive minutes calculated from stadium geocodes (`scottish_stadium_geocodes.csv`).
* **Standardization:** $z_{\text{dist}} = \frac{\text{dist} - \mu_{\text{catalog}}}{\sigma_{\text{catalog}}}$.
* **Prior:** $w_{\text{dist}} \sim \text{truncated}(\text{Normal}(0.04, 0.03), \text{lower}=0.0)$.

### Two-Arm Joint Observation (Generations 3-4)

The current production shape reads **one** log-intensity with **two** densities:

$$\text{pxG}_s \sim \text{Gamma}(\nu,\ \mu_s/\nu) \quad\text{(masked)}, \qquad y_s \sim \text{Poisson}(\kappa\,\mu_s) \quad\text{(everywhere)}$$

`Gamma(shape = ν, scale = μ/ν)` has mean $\mu$, so $\nu$ is a pure precision and the proxy
measurement is unbiased for the latent by construction; $\kappa$ is the finishing factor. The
proxy arm sharpens $\mu$ on the seasons carrying BBC live text; the goals arm carries that
sharpened $\mu$ across the whole history. `MatchProxyXGFeature(fallback = :none)` emits an
availability mask, so a match without commentary contributes a finite term multiplied by an
exact zero rather than a fabricated observation.

Identified, not assumed: $\kappa \approx 1.13$ (~76% prior shrinkage) and $\nu \approx 3.9$
with posterior sd ~0.28 against a prior sd of 1.45.

### Player-Lineup Pillar (Generation 4)

$$L_{\text{home},i} = w_{\text{att}} R_{\text{home},i} - w_{\text{def}} R_{\text{away},i}$$

$R_{s,i}$ is the aggregated RAPM rating of side $s$'s named teamsheet, either the starting
outfield XI or starters plus named substitutes at a fixed $w_{\text{bench}} = 0.10$. RAPM is a
ridge fit — **never sampled** — over each fold's frozen history block (`fit_on = :history`), so
a target fixture never contributes to the ratings that price it. Priors on
$w_{\text{att}}, w_{\text{def}}$ are `Normal(0, 0.3)`, symmetric about zero: the sign of a RAPM
loading is an empirical result, not an assumption.

Measured over 40 folds and 2,899 scored observations, the lineup arms do **not** beat the
team-state control on LogLoss. What they buy is calibration — ECE 0.0088-0.0104 against the
control's 0.0149 and the Betfair closing line's 0.0139 — and it is calibration, not sharpness,
that converts into Kelly bankroll growth. See
[`experiments/scottish_lower/06_joint_player_lineup_fusion/README.md`](../../experiments/scottish_lower/06_joint_player_lineup_fusion/README.md).

---

## 5. Key Experimental Findings (Scottish Lower Fold 1)

Across 720 fitted historical matches and 20 held-out season-opener fixtures:

```
================================================================================================
 FOLD 1 LEADERBOARD (20 held-out fixtures, 24/25 Season Opener)
================================================================================================
  Model                            |   Time | R-hat |    ESS |  div |       γ | w_wealth |  w_dist |  LL 1X2 | LL O2.5
  --------------------------------------------------------------------------------------------------------------------
  00  Pure Poisson control         |  32.6s | 1.010 |    562 |    0 |  +0.146 |      —  |      —  |  1.0809 |  0.7054
  02  + Squad Wealth (Δz)          |  26.7s | 1.008 |    481 |    0 |  +0.142 |  +0.065 |      —  |  1.0639 |  0.7033
  03  + Travel Distance (z)        |  27.5s | 1.012 |    588 |    0 |  +0.135 |      —  |  +0.062 |  1.0985 |  0.7033
  04  + Joint Wealth & Distance    |  28.5s | 1.010 |    610 |    0 |  +0.130 |  +0.066 |  +0.064 |  1.0778 |  0.7022
  --------------------------------------------------------------------------------------------------------------------
```

### Critical Scientific Insights:
1. **Feature Orthogonality:** Solo $w_{\text{wealth}} = +0.065$ and solo $w_{\text{dist}} = +0.062$ match joint estimates ($+0.066, +0.064$) exactly. The covariates do not compete for variance.
2. **Attack Potency Absorption:** Squad wealth absorbs attack-side volatility, dropping team attack variance $\sigma_a$ from $0.175 \to 0.156$.
3. **Geographic Home Advantage Decomposition:** Travel distance absorbs physical travel fatigue, decomposing baseline home advantage $\gamma$ from $+0.146 \to +0.130$.

---

## 6. Unified V2 Pipeline (`05` -> `10` Production Standard)

The production codebase is organized into six graduated pipeline stages:

1. **`05` Composable Count Builder (`src/models/pregame/builder/`)**:
   - Assemble models via generic `add!` calls on `CountModelBuilder`.
   - Compiles into concrete, type-stable `PoissonCountModel` or `NegBinCountModel` with $O(1)$ ReverseDiff tapes.
2. **`06` Typed Posterior Latents (`src/models/latents/`)**:
   - Standardized `CountLatents` container with zero-allocation `SmileScoreGrid` scoring kernels.
3. **`07` Unified Inference Lifecycle (`src/training/inference/`)**:
   - `fit_model(FitConfig, ds)` produces an atomic `Fit` container.
   - Automated MCMC convergence diagnostics (`ConvergenceSummary`) gating bankroll risk.
   - Execution strategies: `AutoExecution()`, `QueuedExecution()`, `ThreadedExecution()`, `SequentialExecution()`.
4. **`08` Unified Evaluation Framework (`src/evaluation/`)**:
   - Zero-copy `OddsView` against match markets.
   - `evaluate_predictions(fit, ds)` scores LogLoss, CRPS, Brier score, RPS, and ECE against closing prices.
5. **`09` Zero-Allocation Portfolio & Staking (`src/Portfolio/`)**:
   - O(1) indexed market lookups via `OddsIndex`.
   - `BookWorkspace` pre-allocates matrix and probability buffers once per fold.
   - `simulate_portfolio` simulates fractional Kelly bankroll growth with automated convergence gating.
6. **`10` MatchDay Operational Execution (`src/MatchDay/`)**:
   - Point-in-time slate pricing through named seams:
     `fixtures → identity → lineups → book → features → inference → gate → stake_sheet`.
   - **The slate is the execution atom.** `Portfolio` solves one joint problem for every fixture
     that settles together, so reservation is one transaction for the whole stake vector.
   - Nothing samples here: `MD.canonical_fit` loads a completed run out of `mcmc_experiments`;
     everything the operator does is written into `betdb.<paper_schema>`.
   - Two consoles: **live** on `:8085` writing `paper_runbook`, **replay** on `:8086` writing
     `paper_replay`, isolated structurally (`assert_replay_schema`, `serve_replay`) rather than
     by convention. See
     [`current_development/match_day_inference/README.md`](../../current_development/match_day_inference/README.md).

---

## 7. Fast Test Execution Protocol

* **Single Suite (Fastest for Dev, ~15-20s):**
  ```bash
  julia --project -t 8 -e 'using Test, BayesianFootball; include("test/unified_portfolio_tests.jl")'
  ```
* **Concurrent Full Suite (4 worker processes, ~40-45s):**
  ```bash
  julia --project -t 8 test/run_parallel_tests.jl
  ```
* **Standard Sequential Suite (Full regression, ~3.5 min):**
  ```bash
  julia --project -t 8 test/runtests.jl
  ```
* **MatchDay Replay Console (1,015 assertions; NOT in the parallel runner):**
  ```bash
  julia --project -t 8 test/test_matchday_replay.jl
  ```
  Four tiers — pure (clock and filtration contract, no database), the ladder desk, the ledger
  (`paper_replay` execution and settlement plus a direct `paper_runbook` isolation assertion),
  and models (a real Saturday, real canonical fits). The ledger and model tiers skip **with a
  message** when `betdb`, `mcmc_experiments` or the DataStore cache is out of reach — never
  silently, so a "passed" line from a tier that skipped is not evidence.

---

## 8. Standard AI Agent Prompting Block

When prompting a new AI agent in this workspace, provide this context block:

```text
You are working on BayesianFootball.jl across a 3-node topology:
1. Local Laptop / archpc: Development workstation (/home/james/bet_project/BayesianFootball). 8 physical cores.
2. Remote Compute (mcmc-beast): 16 physical cores (32 SMT). Always launch Julia with -t 16 and run `using ThreadPinning; pinthreads(:cores)`. BLAS threads = 1.
3. Databases — TWO of them, do not conflate:
   - betdb (archpc:5433, BF_DB_URL): raw football data (sofascore, bbc, betfair, betfair_live)
     plus the paper-trading ledgers paper_runbook (live console :8085) and paper_replay
     (replay console :8086).
   - mcmc_experiments (mcmc-beast:5432, BF_EXPERIMENTS_DB_URL or ~/.pgpass): runs, fold
     results, match latents, fit and portfolio artefacts, config_registry. Reached only as
     Training.PostgresStorage(experiment_name).
   Never print or paste a credential-bearing URL.

Rules for Execution:
- Use the Unified V2 Pipeline: CountModelBuilder, fit_model, evaluate_predictions, run_portfolio_simulation.
- Run tests via fast parallel runner: julia --project -t 8 test/run_parallel_tests.jl (~40s).
- Sync code to mcmc-beast using: rsync -avz --exclude '.cache/' --exclude 'data/' ...
- Do not run colliding jobs on mcmc-beast when an agent or MCMC chain is running.
- Ensure all features operate in log-intensity space (eta = log lambda) with Float64/Int AD-safe vectors.
- Monitor long runs via /root/<script>.log or btop in tmux window mbtop:0.
```
