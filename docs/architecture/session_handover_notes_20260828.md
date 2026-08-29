# BayesianFootball — Session Handover & Architecture Briefing

> **Date:** August 28, 2026  
> **Conversation ID:** `eb5e0c83-114b-4cba-8a50-f115128644ad`  
> **Hosts:** Local Laptop (`laptop1`), Homelab Server (`archpc`), Compute Node (`mcmc-beast`)

---

## 1. System Topology & Infrastructure Map

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               MESH TOPOLOGY                                     │
├───────────────────┬───────────────────────────────┬─────────────────────────────┤
│ Host              │ Network / IP                  │ Role & Active Processes     │
├───────────────────┼───────────────────────────────┼─────────────────────────────┤
│ Local Laptop      │ Development Station           │ Workspace synced to git     │
│                   │                               │                             │
│ archpc            │ LAN: 192.168.1.88             │ • Database: Postgres 5433   │
│                   │ Tailscale: 100.124.38.117     │ • Tmux: overnight_builder   │
│                   │                               │   (Claude Opus 5 Auto)      │
│                   │                               │                             │
│ mcmc-beast        │ Tailscale: 100.78.134.44      │ • 16 Physical Cores (32 SMT)│
│                   │ (65.109.70.100)               │ • Tmux: sl_2seasons:runner  │
│                   │                               │   (40-Fold Queued NUTS Grid)│
└───────────────────┴───────────────────────────────┴─────────────────────────────┘
```

---

## 2. Key Findings & Empirical Results

### 2.1 Scottish Lower 20-Fold Walk-Forward Grid (Gate 6 & Gate 7 Scorecard)
Evaluated across **360 out-of-sample fixtures (50 chronological matchday slates)** using Bet365 and Betfair Exchange closing prices:

```
================================================================================================================================
 MODEL COMPARISON: FULL BOOK KELLY PORTFOLIO (50 Slates / 20 Folds)
================================================================================================================================
  Model Arm                      | Bets | Final Wealth | Growth / Slate | ROI % [95% Bootstrap CI]  | Max Drawdown (MDD)
  ------------------------------------------------------------------------------------------------------------------------------
  02 + Squad Wealth 🥇           |  838 |    2.068x    |   +0.01453     | +23.74% [+3.8%, +44.4%]   |      -11.5%        
  00 Pure Poisson (Control)      |  843 |    2.001x    |   +0.01387     | +22.22% [+2.1%, +42.4%]   |      -11.9%        
  04 Joint Wealth & Distance     |  842 |    1.905x    |   +0.01289     | +20.91% [+1.3%, +41.2%]   |      -11.9%        
  03 + Travel Distance           |  847 |    1.881x    |   +0.01263     | +19.94% [+0.7%, +39.5%]   |      -11.9%        
================================================================================================================================

================================================================================================================================
 MODEL COMPARISON: 1X2 OUTCOME ONLY KELLY PORTFOLIO (50 Slates / 20 Folds)
================================================================================================================================
  Model Arm                      | Bets | Final Wealth | Growth / Slate | ROI % [95% Bootstrap CI]  | Max Drawdown (MDD)
  ------------------------------------------------------------------------------------------------------------------------------
  02 + Squad Wealth 🥇           |  507 |    2.043x    |   +0.01429     | +29.98% [+4.5%, +56.5%]   |      -13.3%        
  00 Pure Poisson (Control)      |  516 |    1.944x    |   +0.01330     | +27.22% [+2.8%, +52.4%]   |      -13.7%        
  04 Joint Wealth & Distance     |  515 |    1.848x    |   +0.01228     | +25.51% [+1.4%, +50.4%]   |      -13.7%        
  03 + Travel Distance           |  527 |    1.813x    |   +0.01190     | +23.96% [+0.4%, +48.5%]   |      -13.8%        
================================================================================================================================
```

* **Outcome**: `02_poisson_wealth` is the clear empirical champion: lowest 1X2 Log-Loss (`0.6182`), highest bankroll compounding (**`2.068x`**), highest 1X2 ROI (**`+29.98%`**), and lowest max drawdown (**`-11.5%`**).

---

## 3. Major Architectural Breakthroughs

### 3.1 Composable Count Model Builder (`current_development/scottish_lower/05_composable_count_builder/`)
* **Problem Solved**: Replaces the combinatorial explosion of $2^N$ separate Turing engine files with **one composable builder**.
* **Type Hierarchy**: `CountModelBuilder` collects components via `add!(b, component)` and freezes them via `build(b)` into a static, compile-time `Tuple`.
* **ReverseDiff AD Performance Discovery**:
  * `view(A, idx)` on a `TrackedArray` created a `SubArray{TrackedReal}` taped element-by-element; direct indexing `A[idx]` produces a single vector BLAS node on the tape (**5× throughput increase**).
  * Fusing the time-decay weight directly into the broadcasted sum yielded another **1.6× throughput boost**.
  * **Gradient Latency**: Dropped from $0.50\text{ms}$ down to **`0.044 ms`** (>10× faster than legacy prototype engines).
* **Parity**: 64/64 architectural gates passed; bit-identical log-density evaluation (`0.000e+00`) vs legacy hand-written engines.
* **Specification Document**: [`docs/architecture/composable_model_builder_specification.md`](file:///home/james/bet_project/BayesianFootball/docs/architecture/composable_model_builder_specification.md).

### 3.2 Typed Posterior Latents (`current_development/06_typed_posterior_latents/`)
* **Problem Solved**: Replaces `latents.df` (where 3,200 posterior samples per match were stuffed into DataFrame cells causing boxing and slow lookups) with strongly-typed dense matrix containers.
* **Hierarchy (`AbstractPosteriorLatents`)**:
  * `CountLatents`: $\lambda_h, \lambda_a$ (and $r_h, r_a$ for NegBin).
  * `RecombLatents`: $\lambda_{\text{open}}, \lambda_{\text{pen}}, \lambda_{\text{og}}, \text{pxG}$ matrices.
  * `SmileLatents`: $\lambda_h, \lambda_a, \lambda_{\text{tot}}, \phi(K)$ 3D strike tensor.
* **Score Grid & Pricing**: Vectorized SIMD calculation of $(12 \times 12)$ score grids with **zero heap allocations**.

---

## 4. Active Background Tasks

1. **`archpc` (Overnight Builder)**:
   * Session: `overnight_builder:agent`
   * Agent: **Claude Opus 5** building out `06_typed_posterior_latents/` (`l01_latents.jl` $\to$ `r01_demo.jl`).
2. **`mcmc-beast` (2-Season Walk-Forward Grid)**:
   * Session: `sl_2seasons:runner`
   * Task: Sampling 40 folds across both seasons `24/25` and `25/26` with 16 pinned threads.

---

## 5. Antigravity CLI (`agy`) Control on `archpc`

To launch and interact with Antigravity on `archpc`:

```bash
# 1. SSH into archpc
ssh archpc

# 2. Start a persistent tmux session for AGY
tmux new-session -s agy_session -n main

# 3. Inside the tmux window, launch AGY in the repository
cd ~/bet_project/BayesianFootball
agy

# 4. To attach to the overnight Claude Opus builder session:
tmux attach-session -t overnight_builder
```

---

## 6. Conversation Transcript Sync
The full, untruncated conversation history has been copied to:
* **Laptop**: `/home/james/.gemini/antigravity-cli/brain/eb5e0c83-114b-4cba-8a50-f115128644ad/`
* **archpc**: `/home/james/.gemini/antigravity-cli/brain/eb5e0c83-114b-4cba-8a50-f115128644ad/`
