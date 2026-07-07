# staking_layer — experiments log

Modular refactor of the staking system (`staking_sim` + `staking_real` + `unified_staking`) into a
swappable-trust-model architecture. See `README.md` for the API. Prior validated findings live in
the source folders' `experiments.md` and are the parity targets here.

## 2026-07-07 — Refactor + local validation (this session)

**Built** the module (`src/l01`–`l10` + `loader.jl`) and the runners (`r01`–`r05` + `preflight_real`).
`SimMatch → StakingMatch` (carries real data too — removes the confusion), `+home/away` team ids;
`unified_staking/l01` folded into `l02_kelly`. The trust model is now an `AbstractTrustModel`
(`fit_trust`/`trust_weights`/`trust_draws`) — EB (`l05`), Bayesian (`l06`, Turing), Flat/Curated
(`l04`), Hierarchical (`l06` stub). Policy/runner are trust-agnostic.

**Validated locally (no server / no L1 payload needed):**
- **Whole module loads** (l01–l10, incl. Turing l06 + Signals l07 + Data.Markets l09).
- **`r01` SimSource race** (sup-blind world, n=660): reproduces the sim E4 ordering —
  CURATED05 0.806 ≻ FLAT 0.597 ≻ TRUST_EB 0.316 ≻ TRUST05 0.110 ≻ **U_raw & PB_BK RUINED**.
  EB alarm pulls home w 0.50→~0.20, away→~0.15, holds totals/BTTS ~0.33–0.40. `max_tilt_err = 1.3e-15`
  (the w=1 grid tilt reproduces the smile over-probs exactly).
- **BayesianTrust fit** (n=300 synthetic history): NUTS samples cleanly, `trust_draws` shape 7×D,
  posterior means preserve the EB rank order (over_35 top, away/home bottom) but **pool ~0.2 lower**
  (stronger hierarchical shrinkage than EB's per-unit grid mean). Characterising that EB-vs-Bayes
  gap on real data is exactly what `r04` is for — the flat half-Normal(0,1) τ prior is the knob.
- **Distributional staking** (`UnifiedPolicy(distributional=true)`): runs, averages the Kelly solve
  over `trust_draws` (EB grid-posterior samples / Bayes chain draws).

**Pending (server run — kaimon disconnected + no local `src_sup40_sw40` payload this session):**
- `r02` real EB parity → must reproduce `staking_real/results/e_real_summary_c020.txt`
  (CURATED ≈ 26.9×, TRUST_EB ≈ 3.0×, home w→0.18, b21 signs 11/11, `max_tilt_err < 1e-6`).
- `r03` extended book → `e_ext_summary_c020.txt` (CS-excluded CURATED ≈ 34×; CorrectScore drag).
- `r04` EB-vs-Bayes race + posterior table; `r05` team-w EDA (step-0 precondition for hierarchy).
- On parity: retire `staking_sim/` + `staking_real/` + `unified_staking/` (kept until then).

## How to run

```julia
# anywhere:
include("current_development/staking_layer/r01_sim_race.jl")
# server (payload present):
include("current_development/staking_layer/preflight_real.jl")   # builds/caches lat,ppd,odds_bf,ds1
include("current_development/staking_layer/r02_real_race.jl")     # + r03 / r04 / r05
```

## Backlog

- Implement `HierarchicalTrust` (l06 stub) once `r05` confirms team-w spread.
- `PerBetKellyPolicy` currently the only b21 cross-check consumer; generalise attribution if needed.
- Optionally add a `Bet365` q_mkt variant of `RealSource` (memory: anchor to de-vigged Bet365 on
  thin minor-league exchanges).
