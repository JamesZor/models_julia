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

## 2026-07-07 (cont.) — HierarchicalTrust (step 3) built + validated on sim

Motivated by r05 (team spread concentrated in home/away 1X2, 4× totals) + r04 (learned trust
bleeds on 1X2 because it doesn't abstain hard enough → CURATED 17.5× ≫ EB 1.6×; distributional
staking helps, EB_dist 2.17 > EB 1.62).

- **`HierarchicalTrust`** (l06): `w_{u,t} = logistic(w0_u + σ_u·z_{u,t})`, grouped by home team,
  non-centred, `y ~ product_distribution(Bernoulli.(p̃))`. Fitted object carries per-(unit,team)
  means + draws + a pooled fallback for unseen teams.
- **`OverrideTrust`** (l04): composition wrapper hard-overriding units → the 1X2-abstain / totals-EB
  **HYBRID** is `OverrideTrust(EBTrust(), Dict(1=>0,2=>0,3=>0))`.
- **`r06_hier_trust.jl`**: σ_u + shrunk per-team-spread diagnostic + race (CURATED/EB/HYBRID/HIER/HIER_dist).
- **Validated on sim (a true-σ=0 world → the null test):** hierarchical fit samples cleanly; the
  realized per-team home-w spread shrinks to **0.010** (correctly "no team variation"). CAVEAT
  discovered: **σ_u is weakly identified** and floats near its prior when signal is absent (0.6 at
  σ_prior=0.75, 0.31 at 0.4) — so the honest diagnostic is the *shrunk per-team w-spread*, not σ_u.
  Default σ_prior tightened to 0.4. Hybrid override validated (1X2 rows hard-zeroed, EB totals/BTTS).
- **VERDICT (server, real data, n=282):**
  - **(1) NO per-team signal.** Hierarchically-shrunk team-spread: home **0.005**, away 0.007, all
    others 0.002 — every unit BELOW the ~0.02 sim-null floor. Per-team home-w collapses to 0.146
    (Shamrock) … 0.160 (Bohemian), a 0.014 range. **r05's "Bohemian 0.60 vs Shamrock 0.28" was
    28-obs noise** — proper pooling crushes it. σ_u ~0.31 (prior-dominated, as flagged). Kill step 3:
    the model's 1X2 badness is UNIFORM across teams, not team-specific.
  - **(2) HYBRID wins.** term_W: **HYBRID 19.23** (G +0.0105) ≥ CURATED 17.53 (+0.0102) ≫ HIER_dist
    2.39 ≻ TRUST_HIER 2.20 ≻ TRUST_EB 1.78. Hard-abstaining 1X2 is worth ~9×; the whole EB→CURATED
    gap was the soft 1X2 trust bleeding. TRUST_HIER doesn't abstain (pools 1X2 to ~0.15) so it bleeds
    like EB. Distributional still helps within the soft family (HIER_dist > HIER > EB).
  - **OPERATING POINT: HYBRID = `OverrideTrust(EBTrust(), Dict(1=>0,2=>0,3=>0))`** — abstain 1X2,
    EB-learn totals/BTTS. Marginally beats fixed CURATED, same drawdown (0.509). Hierarchy retired.

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
