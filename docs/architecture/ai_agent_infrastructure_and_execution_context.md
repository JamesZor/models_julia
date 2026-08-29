# AI Agent Infrastructure & Remote Execution Context Guide

> **BayesianFootball.jl Infrastructure & AI Agent Context Reference**  
> **Status:** Active Standard | **Target Systems:** Laptop (Development Host), `mcmc-beast` (Compute Node), `archpc` (PostgreSQL Database Host)

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
|   +--------------------------+                             +------------------------+   |
|                 |                                                      |                |
|                 | tmux / ssh controller                                | Database SQL   |
|                 v                                                      v                |
|   +--------------------------+                             +------------------------+   |
|   |  Tmux Session Controller |                             |         ARCHPC         |   |
|   |  (e.g. features:1.1)     |                             |   (Postgres Server)    |   |
|   |  (SSH to mcmc-beast)     |                             |   betdb on port 5433   |   |
|   +--------------------------+                             +------------------------+   |
+-----------------------------------------------------------------------------------------+
```

### Machine Specifications & Endpoints

| Node Name | Network Role | Hardware / Specs | Primary Path | Connection / Endpoint |
| :--- | :--- | :--- | :--- | :--- |
| **Local Laptop** | Development & Agent Orchestration | Multi-core Dev Laptop | `/home/james/bet_project/BayesianFootball` | Local interactive terminal / Antigravity CLI / Tmux |
| **`mcmc-beast`** | Dedicated MCMC Sampling & Grids | AMD Ryzen (16 Physical Cores, 32 SMT threads, 64 GB RAM) | `/root/BayesianFootball` | `ssh root@mcmc-beast` (via Tailscale / LAN) |
| **`archpc`** | PostgreSQL Database Server | Dedicated Database Server | `/var/lib/postgresql` | `postgresql://admin:...@archpc:5433/betdb` |

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

### 3. DistributionsAD Compatibility Patch (Julia 1.12 / ReverseDiff)
On Julia 1.12 with `DistributionsAD 0.6.58`, a 1-line compatibility patch is required in `~/.julia/packages/DistributionsAD/.../DistributionsADReverseDiffExt.jl` at line 127:
```julia
# Patch signature to match Distributions 0.25.127:
@check_args(Gamma, (α, α > zero(α)), (θ, θ > zero(θ)))
```
Without this patch, `ReverseDiff` AD tapes for Gamma distributions will fail precompilation.

---

## 4. Model Equations & Feature Architecture

### Pure Poisson Model Framework (Scottish Lower Benchmark)

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

## 6. Standard AI Agent Prompting Block

When prompting a new AI agent in this workspace, provide this context block:

```text
You are working on BayesianFootball.jl across a 3-node topology:
1. Local Laptop: Development workstation (/home/james/bet_project/BayesianFootball).
2. Remote Compute (mcmc-beast): 16 physical cores (32 SMT). Always launch Julia with -t 16 and run `using ThreadPinning; pinthreads(:cores)`. BLAS threads = 1.
3. Database (archpc:5433): PostgreSQL betdb.

Rules for Remote Execution:
- Sync code to mcmc-beast using: rsync -avz --exclude '.cache/' --exclude 'data/' ...
- Do not run colliding jobs on mcmc-beast when an agent or MCMC chain is running.
- Ensure all features operate in log-intensity space (eta = log lambda) with Float64/Int AD-safe vectors.
- Monitor long runs via /root/<script>.log or btop in tmux window mbtop:0.
```
