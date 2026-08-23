# Recombination & Multi-Task Modeling Architecture

Welcome to the documentation suite for the **Recombination, Proxy xG (pxG), and Squad Wealth** modeling framework in `BayesianFootball.jl`.

This directory contains modular, targeted documentation designed for AI agents, quantitative researchers, and software engineers.

---

## 📑 Documentation Index

| Module / Document | Topic & Focus | Target Audience |
| :--- | :--- | :--- |
| **[1. Theory & Goal Decomposition](01_theory_and_formulation.md)** | Mathematical formulation decomposing gross match goals into Open Play, Officiating Penalties, and Own Goals. | Modelers & AI Agents |
| **[2. Multi-Task Proxy xG Co-Training](02_proxy_xg_cotraining.md)** | Gamma continuous likelihood for Proxy xG (`pxG`), team finishing efficiency factors ($\kappa_i$), and shot-count noise filtering. | Modelers & Statisticians |
| **[3. Starting-XI Squad Wealth Submodel](03_squad_wealth_submodel.md)** | Standardized squad market valuation differential ($\Delta W_m$), linear latent rate shifts, and lineup timing. | Quantitative Analysts |
| **[4. Exact Discrete Convolution Kernel](04_discrete_convolution_kernel.md)** | Summing independent Poisson processes, Dixon-Coles adjustments, and the $\sum M = 1.000000$ ScoreMatrix invariant. | Systems Engineers |
| **[5. AD Safety & Zero-Allocation Masking](05_ad_performance_and_masking.md)** | Static computational graphs, ReverseDiff tape stability, binary observation masking, and thread pinning. | Core Developers |
| **[6. Scottish Lower 40-Fold Benchmarks](06_scottish_lower_benchmarks.md)** | Comprehensive walk-forward evaluation (CRPS, LogLoss, Kelly Sharpe, Betfair Exchange financial simulations). | Trading & Research |

---

## 🚀 Quick Start Example

```julia
using BayesianFootball
using BayesianFootball.Models.PreGame

# Production Champion: Multi-Task pxG + Squad Wealth + Officiating
model = DynamicPxGRecombModel(
    dynamics_config      = TimeDecayDynamics(days_half_life = 365.0),
    homeadvantage_config = GlobalHomeAdvantage(),
    wealth_config        = LinearSquadWealthConfig(),
    pxg_config           = GammaPxGObservationConfig(),
    recomb_config        = HierarchicalOfficiatingConfig(),
    name                 = "recomb_pxg_wealth_production"
)

# Load dataset and train
ds = Data.load_datastore_cached(Data.ScottishLower())
task = Experiments.create_experiment_task(ds, model, "pxg_prod_run", "./data/production")
results = Experiments.run_experiment(task)
```
