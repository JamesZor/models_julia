# Veikkausliiga (31) — Stage-A Tournament EDA

**Status:** complete (Stage A — EDA + report only). Results from `r01_veikkausliiga_runner.jl` run in the
kaimon REPL on `mcmc-beast`. 31 loaded with `Data.load_datastore_sql(Data.Veikkausliiga())` (fresh SQL,
n=678 matches).

**TL;DR:** Veikkausliiga (tournament **31**) is a **near-Poisson, only very mildly over-dispersed**
goals regime — far closer to Ireland Premier (79) than to the genuinely over-dispersed First Division
(718). Total goals **V/M ≈ 1.08** (mean 1.395/side, 2.79/match); NB beats Poisson by a trivial **1.6 AIC**,
NB1≈NB2, and **Poisson is not even rejected by χ²** (p=0.26; the COM-Poisson winner fits at p=0.82). The
**Dixon-Coles low-score dependence ρ ≈ 0.012** (independent models win the bivariate ladder) → the τ
correction is unnecessary. The only marginal wrinkle is **mild zero-inflation on home goals** (ZIP edges
Poisson by ~5 AIC on the home side only). **Home advantage is weak/borderline** (+0.142 goals, MWU
**p=0.053 — not significant**), the smallest of the three leagues studied. **Within-team dispersion ≈ 0.99**
(essentially equidispersed once team strength is removed), so what little pooled over-dispersion exists is
cross-team heterogeneity the hierarchical attack/defence structure already absorbs. **Recommendation: a
Poisson or single-knob NB goals engine (NB preferred only to soak up the home zero-excess), DC-free.**

The per-team work cleanly separates the league: **HJK, KuPS and Inter Turku** are the dominant sides
(significantly above league on goals scored, xG-for and squad rating, and best on goals/xG conceded),
while **KTP, IFK Mariehamn and EIF** are weakest at both ends — and the goals, xG and player-rating
rankings **agree**. **Player ratings are fully model-usable from 2023** (~11 rated starters/team, ~100% of
matches; zero before 2023), the same window as xG.

**Stage-B readiness:** 31 carries **full betfair coverage** (every season, 966k rows) → CLV feasible;
**xG and player ratings land from 2023** (~100%); **no `bigChanceCreated` column** (same as 718).

**Files:** `l01_veikkausliiga_logic.jl` (functions — reuses the first_division/ireland fitters, adds the
per-team attack/defence + xG + rating analyses), `r01_veikkausliiga_runner.jl` (execution + captured
output), this report. The only `src/` change is the `Veikkausliiga` segment in
`src/Data/fetchers/segments.jl` (+ export in `src/Data/data-module.jl`).

---

## 1. Motivation & framing

A new top-tier competition is available in the SQL DB: **Veikkausliiga** (Finnish top flight, tournament
id `31`). Before it is used for modelling or betting we characterise it **as a data-generating process**,
chiefly to fix the **goals mean/variance regime** that determines the likelihood family (Poisson /
Negative-Binomial / COM / Weibull) for the L1 Bayesian engines. This mirrors the completed Ireland First
Division (718) study, plus three additions: per-team **attack/defence** goal & xG rankings, a
player-rating **coverage audit**, and per-team **rating distributions**.

Scope is **EDA + report only** (Stage A). This is a **standalone** characterisation (no contrast league);
the only cross-league touch is a validation guard that re-fits Ireland 79 to confirm the fitters behave
before trusting 31's output.

## 2. Data & coverage

Veikkausliiga runs a **spring–autumn calendar** (April–September in the data), ~132 matches/season, ~12
teams per season (16–17 distinct clubs across 2021–2026 with promotion/relegation).

**DataStore field row counts (fresh SQL):**

| Field | rows |
|---|---|
| matches | 678 |
| statistics | 674 |
| odds | 15,133 |
| lineups | 24,734 |
| incidents | 12,729 |
| **betfair_odds** | **966,834** |

**Per-season feature coverage** (fraction of *played* matches whose `ALL`-period stats row is non-missing):

| season | matches | played | stats | xG | bigChance | shots | betfair rows | lineups |
|---|---|---|---|---|---|---|---|---|
| 2021 | 132 | 132 | 0.992 | **0.00** | 0.00 | 0.992 | 177,182 | 4,706 |
| 2022 | 132 | 132 | 0.985 | **0.00** | 0.00 | 0.985 | 182,179 | 4,729 |
| 2023 | 132 | 132 | 1.00 | **1.00** | 0.00 | 1.00 | 175,975 | 4,741 |
| 2024 | 132 | 132 | 1.00 | **1.00** | 0.00 | 1.00 | 220,268 | 4,735 |
| 2025 | 132 | 132 | 0.992 | **0.992** | 0.00 | 0.992 | 191,640 | 5,119 |
| 2026 | 18 | 18 | 1.00 | **1.00** | 0.00 | 1.00 | 19,590 | 704 |

**Coverage headlines:**
- **xG begins in 2023** (~100% from 2023 on; 2021–22 carry stats + shots but no xG).
- **No `bigChanceCreated`** at all → that pillar is unavailable for 31 (same as 718).
- **betfair every season** (966k rows) → Stage-B CLV feasible.

## 3. Marginal moments

`D = Var/Mean`: `D≈1` Poisson, `D>1` over-dispersed, `D<1` under-dispersed.

| vector | n | mean | var | V/M | zeros emp vs Pois | skew | max | regime |
|---|---|---|---|---|---|---|---|---|
| 31 Home | 678 | 1.466 | 1.614 | **1.101** | 0.264 / 0.231 (+0.033) | 0.79 | 8 | mild over |
| 31 Away | 678 | 1.325 | 1.381 | 1.042 | 0.261 / 0.266 (−0.005) | 1.11 | 7 | ≈ equi |
| 31 All | 1356 | 1.395 | 1.501 | **1.076** | 0.263 / 0.248 (+0.015) | 0.94 | 8 | mild over |

Home goals carry a small **zero-excess** (+0.033) and the league's only real over-dispersion; away goals
are essentially Poisson. Home advantage in the mean is **+0.142** (1.466 vs 1.325) — small.

## 4. Candidate-distribution maths & ladders

Families ranked by `AIC = 2k − 2ℓ` / `BIC = k·ln n − 2ℓ`: Poisson (k=1), NB2 `RobustNegativeBinomial(r,μ)`
Var = μ + μ²/r (k=2), NB1 Var = φμ (k=2), Weibull-count, ZIP/ZINB, COM-Poisson, and Dixon-Coles τ on an
independent base.

### 4a. Validation guard (79 total goals)
Ireland 79 total: **Poisson AIC 5787.94 (BIC 5793.55) WINS** AIC+BIC; NegBin 5788.74. Poisson-regime
verdict reproduced (n now 2012 vs 2002 in the 718 study, as the 2026 cache has progressed — hence the
small numeric drift from the published 5760.86). Fitters validated.

### 4b. Univariate + NB1/NB2 ladder (31)

| vector | winner (AIC/BIC) | NB AIC | Poisson AIC | notable | NB r |
|---|---|---|---|---|---|
| 31 Home | **ZIP** (2117.0) | 2121.07 | 2122.42 | ZIP beats Poisson by 5.4 (mild zero-inflation, π=0.074) | 13.48 |
| 31 Away | **Poisson** (1997.73) | 1999.23 | 1997.73 | clean Poisson, NB r=35.9 negligible | 35.9 |
| 31 All | COM/Weibull/NB tied (≈4121.4) | 4121.44 | 4123.02 | NB beats Poisson by only **1.58 AIC** | 18.56 |

NB1 vs NB2 (All): **identical** (ΔAIC 0.000), φ=1.075 — the marginal cannot separate the two variance
functions (decide inside the joint model if needed).

### 4c. Dixon-Coles bivariate ladder (n=678)

| model | AIC | model | AIC |
|---|---|---|---|
| **Indep Weibull** | **4118.87** | DC Weibull | 4120.80 |
| Indep Poisson | 4120.14 | DC Poisson | 4122.08 |
| Indep NB | 4120.31 | DC NB | 4122.24 |

Every **independent** model beats its DC counterpart; **ρ ≈ 0.012** across variants. **No low-score
Dixon-Coles dependence** — the τ correction adds nothing. The six families sit within ~3.4 AIC of each
other, i.e. the bivariate goal field is barely distinguishable from independent Poisson.

## 5. Goodness-of-fit (rootogram + Pearson χ²), total goals

- **COM-Poisson (AIC winner):** χ² = 2.93, df = 6, **p = 0.818** → excellent fit; rootogram max|hang| = 0.53.
- **Poisson (reference):** χ² = 8.88, df = 7, **p = 0.262** → **not rejected**. Unlike 79 (whose Poisson
  failed at p≈0 on a blow-out tail), Veikkausliiga total goals have **no heavy tail** — a plain Poisson is
  already an acceptable marginal.

## 6. League diagnostics

| diagnostic | Veikkausliiga (31) |
|---|---|
| Overdispersion (total goals) | NB marginally justified (AIC 4121.44 < Poisson 4123.02, Δ1.58) |
| Home advantage, mean | +0.142, MWU **p = 0.053 (NOT significant)** |
| Home advantage, variance | ratio 1.169, F **p = 0.042** ✱ |
| Temporal mean drift (Kruskal-Wallis) | p = 0.234 (stable; months 4–9) |
| Temporal variance heteroscedasticity | var ratio 1.58 (present) |
| **Within-team DI** (goals conceded) | **0.985** (16–17 teams) |

The within-team dispersion ≈ 0.99 means that once team strength is removed each team's match-to-match
scoring is **essentially equidispersed**; the small pooled over-dispersion (V/M 1.08) is mostly cross-team
heterogeneity that the hierarchical attack/defence parameters already absorb. Home advantage is the
**weakest of the three leagues** (718 +0.198, 79 +0.308, 31 +0.142) and only borderline-significant.

## 7. Per-team attack & defence

Per team vs the league average, with a quasi-Poisson rate test and a **Gamma–Poisson empirical-Bayes
shrunk rate**; significance is BH-FDR adjusted across the ~12 teams. (`*` = BH p_adj < 0.05.)

**Goals — attack (scored/match), best first** · league μ=1.395, EB prior Gamma(α₀=22.8, β₀=16.3):

| team | n | mean | shrunk | RR | p_adj |
|---|---|---|---|---|---|
| hjk | 113 | 1.867 | 1.808 | 1.34 | 0.001 ✱ |
| inter-turku | 113 | 1.743 | 1.699 | 1.25 | 0.011 ✱ |
| kups | 113 | 1.690 | 1.653 | 1.21 | 0.024 ✱ |
| sjk | 113 | 1.549 | 1.529 | 1.11 | 0.294 |
| … | | | | | |
| fc-lahti | 91 | 1.044 | 1.097 | 0.75 | 0.017 ✱ |
| ifk-mariehamn | 113 | 1.009 | 1.058 | 0.72 | 0.007 ✱ |
| ktp | 66 | 0.939 | 1.030 | 0.67 | 0.011 ✱ |
| hifk | 44 | 0.864 | 1.007 | 0.62 | 0.014 ✱ |

**Goals — defence (conceded/match), best=fewest first** · 7 of 16 teams significant:

| team | n | mean | shrunk | p_adj |
|---|---|---|---|---|
| kups | 113 | 0.814 | 0.864 | 1.0e-5 ✱ |
| hjk | 113 | 0.885 | 0.928 | 9.1e-5 ✱ |
| fc-honka | 66 | 1.015 | 1.067 | 0.028 ✱ |
| … | | | | |
| ac-oulu | 113 | 1.699 | 1.673 | 0.027 ✱ |
| ktp | 66 | 2.015 | 1.930 | 2.3e-4 ✱ |
| eif | 22 | 2.318 | 2.020 | 0.002 ✱ |

The shrinkage does its job: small-n EIF (n=22) is pulled from a raw 0.864 attack toward the league (1.090)
and is **not** flagged on attack, but its 22-match defensive collapse (2.32 conceded) is still significant.

**xG (2023+) corroborates the goals ranking** (Welch team-vs-rest + Normal–Normal shrinkage):
- **xG attack, top:** HJK 1.86✱, SJK 1.79✱, Inter Turku 1.70✱ (SJK ranks *higher* on xG than on goals →
  mild under-finishing). **Bottom:** KTP 1.09✱, EIF/IFK Mariehamn ~1.12✱.
- **xG defence, best:** KuPS 1.04✱ (p=5e-7), HJK 1.15✱, Inter Turku 1.20✱. **Worst:** EIF 2.10✱, KTP 1.94✱.

The goals and xG attack/defence rankings agree on both the top (HJK, KuPS, Inter Turku) and bottom (KTP,
IFK Mariehamn, EIF) of the table.

## 8. Player-rating coverage

| season | played | frac matches w/ rating | mean rated starters / team |
|---|---|---|---|
| 2021 | 132 | 0.00 | 0.0 |
| 2022 | 132 | 0.00 | 0.0 |
| 2023 | 132 | 1.00 | 11.0 |
| 2024 | 132 | 1.00 | 10.99 |
| 2025 | 132 | 0.992 | 10.99 |
| 2026 | 18 | 1.00 | 10.97 |

Ratings are **fully model-usable from 2023**: ~100% of matches, the **entire starting XI** rated per team.
Per-position coverage pooled over all six seasons is G 0.72 / D 0.72 / F 0.72 / M 0.50 — the G/D/F parity
(≈ the 4-of-6 rated-season fraction) shows **no positional bias among known positions**; the lower M figure
is an artefact of (a) pooling the two unrated seasons and (b) `clean_pos` defaulting unknown/blank position
labels into the "M" bucket, which inflates the M denominator. Restricting to 2023+ the XI is fully covered.

## 9. Per-team squad quality (player ratings)

Minute-weighted team match rating vs league (μ=6.968, between-team τ=0.118), best first:

| team | n | mean | shrunk | p_adj |
|---|---|---|---|---|
| kups | 68 | 7.129 | 7.120 | 2.6e-6 ✱ |
| hjk | 68 | 7.124 | 7.113 | 2.5e-5 ✱ |
| fc-honka | 22 | 7.119 | 7.098 | 0.009 ✱ |
| ilves | 69 | 7.053 | 7.047 | 0.012 ✱ |
| inter-turku | 69 | 7.021 | 7.017 | 0.080 |
| … | | | | |
| ac-oulu | 69 | 6.867 | 6.872 | 0.001 ✱ |
| ff-jaro | 25 | 6.850 | 6.866 | 0.031 ✱ |
| ifk-mariehamn | 69 | 6.845 | 6.855 | 0.001 ✱ |
| eif | 22 | 6.702 | 6.751 | 4.2e-4 ✱ |

Squad-quality ranking matches the goals/xG strength ranking: **KuPS and HJK top, EIF and IFK Mariehamn
bottom**. Ratings are tightly clustered (within-team sd ≈ 0.25, between-team τ ≈ 0.12), so absolute gaps
are small but the elite clubs separate clearly.

## 10. Stage-B readiness

- **Trainable seasons:** 2023, 2024, 2025, 2026 carry xG + player ratings (~100%). 2021–2022 are
  goals-/market-only (no xG, no ratings) — usable for goals/market engines but not xG or rating pillars.
- **Betfair:** present every season (966k rows) → CLV / closing-line scoring feasible.
- **bigChance:** unavailable → exclude the bigChanceCreated pillar from any 31 grid.
- **Market anchor:** per `betfair-vs-bet365-market-anchor`, for a thin minor-league exchange consider
  anchoring the market pillar to Bet365 de-vigged while executing on Betfair (revisit at Stage B).
- **Recommended families:** a **Poisson or single-knob Negative-Binomial** goals engine with hierarchical
  team strengths, **Dixon-Coles-free** (ρ≈0). NB is preferred only to absorb the mild home zero-excess via
  its dispersion knob; the league is close enough to Poisson that the choice is second-order. Add **xG as a
  second pillar from 2023** and the **player-rating positional pillar from 2023**. If 31 is ever trained
  jointly with the Irish tiers, its dispersion sits between 79 (Poisson) and 718 (NB r≈11) at **NB r≈18**.
