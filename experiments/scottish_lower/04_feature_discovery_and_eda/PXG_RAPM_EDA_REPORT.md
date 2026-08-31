# pxG & pxG-APM / RAPM — EDA and Validation Findings

> **Branch:** `feat/pxg-rapm-unified-covariates`
> **Store:** `ScottishLower` (tiers 56/57), 2,009 matches, seasons 20/21 → 26/27, snapshot 2026‑08‑30
> **Cross-league validation:** England 1/2/3/84 + Scotland 54/55, 8,622 matches with both BBC live text and official SofaScore xG
> **Scripts:** `r92`, `r93`, `r94`, `r95` with shared loaders `l92_pxg_validation.jl`, `l93_eda_toolkit.jl`

---

## 0. Executive summary

| # | Finding | Consequence |
|---|---|---|
| 1 | The proxy agrees with official SofaScore xG at **r = 0.835, CCC = 0.826, MAE 0.328** over 9,009 team-matches; on the supremacy **difference** it reaches **r = 0.870**. | The proxy is a genuine xG measurement, not a shot counter. |
| 2 | A shot-count control reaches only r = 0.671 / CCC 0.628. **Zonal parsing buys +0.164 r and −27% MAE.** | The parser earns its complexity. |
| 3 | **English League One and League Two have 65–67% of official xG rows zero-filled.** Unfiltered they drag apparent correlation from ~0.85 to ~0.30 and invent a +0.85 xG bias. | A data defect, not a model defect. Any study touching `sofascore.match_statistics` on tiers 3/84 must filter. |
| 4 | The **deployed** cell table (fitted on 56/57 only) transfers with correlation almost intact (r 0.785–0.875 vs leave-one-tier-out 0.787–0.899) but a **+0.43 xG level bias** and slope ≈ 0.75. | Rank transfers; absolute scale does not. Fine for a covariate whose weight is learned; **not** safe as an absolute rate. |
| 5 | `PxGFeature(lookback = 8)` is **under-tuned**. `decay = :exponential, half_life_matches = 8.0` scores r = 0.2845 / AUC 0.6319 against the default's 0.2575 / 0.6175. | A free +0.027 correlation. Recommend changing the default. |
| 6 | RAPM **split-half reliability at the shipped `λ = 200` is 0.247** for `:y_xg`; the curve plateaus near 0.395 at λ ≥ 5,000. `:y_shots` plateaus far higher, at **0.555**. | The shipped λ is well below the reliability plateau. |
| 7 | In-sample, `:y_goals` looked best (r = 0.420, AUC 0.711). **Held-out it collapses to r = 0.004.** It was pure leakage. | The shipped `:y_xg` default is vindicated. `:y_shots` is its only real rival. |
| 8 | `pxg_supremacy` has the **largest incremental R² over squad wealth** of any candidate (+0.0253, t = 3.30) and the best held-out AUC (0.622) of all six covariates. | It carries information wealth does not. |
| 9 | `pxg_supremacy` and `pxg_level` are **orthogonal (r = +0.000)**; `wealth` and `prod_wealth` are **collinear (r = 0.804)**. | The two pxG roles do genuinely different jobs. The two wealth covariates largely do not. |

**Headline recommendation:** ship the pxG supremacy covariate with an exponential decay (`half_life_matches = 8.0`); raise the RAPM `λ` toward the reliability plateau; do not attach `wealth` and `prod_wealth` together.

---

## 1. What was built

| Script | Question |
|---|---|
| `r92_pxg_vs_sofascore_xg_all_leagues.jl` | Does the proxy actually measure xG? Cross-league validation vs official SofaScore xG. |
| `r93_feature_synergy_and_correlations.jl` | How do the six builder covariates overlap, and what survives squad wealth? |
| `r94_pxg_rapm_forensics.jl` | What is the stint ridge fitted on, what does it produce, how hard is it shrunk? |
| `r95_pxg_model_forensics.jl` | What is the proxy made of, and is the form window tuned? |
| `l92_pxg_validation.jl` | SQL pull (cached), shot parsing, fitting regimes. |
| `l93_eda_toolkit.jl` | Shared statistics: description, histograms, correlation, VIF, OLS, deciles, AUC, agreement battery, ASCII scatter, held-out signal. |

Two methodology decisions govern everything below.

**All feature signal is measured held-out.** Two of these features *fit* something — the RAPM ridge and the shot‑xG cell table. Scored on the matches they were fitted on, a goal-differential RAPM target scores r = 0.420; scored held-out it scores r = 0.004. `eda_holdout` builds features from the first 80% of the fixture list and scores only the last 20%.

**All calibration against scorelines is restricted to commentary-sourced matches.** The measurement ladder's bottom rung *is* the scoreline, so "pxG vs goals" on a goals-sourced match is a tautology.

---

## 2. Coverage and the measurement ladder (`r95`)

The 23/24 BBC live-text cutover is exact:

| Season | Matches | Commentary | Shot counts | Goals | Commentary % |
|---|---|---|---|---|---|
| 20/21 | 180 | 0 | 180 | 0 | 0.0% |
| 21/22 | 360 | 0 | 360 | 0 | 0.0% |
| 22/23 | 360 | 0 | 359 | 1 | 0.0% |
| 23/24 | 360 | 360 | 0 | 0 | 100.0% |
| 24/25 | 360 | 360 | 0 | 0 | 100.0% |
| 25/26 | 350 | 350 | 0 | 0 | 100.0% |
| 26/27 | 39 | 39 | 0 | 0 | 100.0% |

Overall 55.2% commentary, 44.7% shot counts, **1 match** on the goals rung, 0 unmeasured. Parse coverage 99.2–99.7%; 20,772 attempts; 109 fitted cells.

**The shot-count rung is load-bearing.** Disabling it (`fallback = :none`) leaves 47.0% of fixtures neutral and drops signal from r = 0.257 to r = 0.171. The goals rung contributes nothing (1 match) and could be dropped without loss.

`ds.statistics` is **empty** on this store, `lineups.rating` is missing on all 74,225 rows, and no match has `has_xg`. That absence is the whole reason a proxy exists, and it is why §3 has to be run on other tiers.

---

## 3. Cross-league validation against official xG (`r92`)

### 3.1 The zero-fill defect — read this before any number below

`sofascore.match_statistics` returns rows for tiers 3 and 84 whose `expectedGoals` is **exactly 0.000 on both sides**. These are placeholders: every match in the comparison frame *has* live-text attempts, and an attempt cannot carry zero xG. Filtered out, those tiers' mean xG (1.44, 1.48) sits right among every other tier (1.41–1.62).

| Tier | Team-obs | Live | Zero-filled | Share |
|---|---|---|---|---|
| ENG Premier League | 1,936 | 1,936 | 0 | 0.0% |
| ENG Championship | 3,311 | 3,311 | 0 | 0.0% |
| **ENG League One** | 3,359 | 1,152 | 2,207 | **65.7%** |
| **ENG League Two** | 3,360 | 1,104 | 2,256 | **67.1%** |
| SCO Premiership | 422 | 422 | 0 | 0.0% |
| SCO Championship | 1,084 | 1,084 | 0 | 0.0% |

Unfiltered, ENG League One scores r = 0.336 with bias +0.833. Filtered, **r = 0.850, bias +0.020**. The entire apparent failure was the reference, not the proxy. `l92_fetch` also de-duplicates with `DISTINCT ON`: the raw table holds 10,324 rows for 8,394 distinct matches.

### 3.2 Agreement — proxy vs shot-count control (9,009 team-matches, pooled table)

| Stratum | n | r | ρ | MAE | RMSE | bias | slope | CCC | R² |
|---|---|---|---|---|---|---|---|---|---|
| **ALL · proxy** | 9,009 | **0.835** | 0.833 | **0.328** | 0.442 | −0.032 | 0.965 | **0.826** | 0.697 |
| ALL · control | 9,009 | 0.671 | 0.678 | 0.451 | 0.594 | −0.038 | 0.966 | 0.628 | 0.450 |

Parsing zone × body part × context buys **+0.164 correlation and −27% MAE** over counting attempts. Bias is −0.032 xG and calibration slope 0.965 — the pooled proxy is essentially unbiased and unit-slope.

### 3.3 By tier

| Tier | n | r | ρ | MAE | bias | slope | CCC |
|---|---|---|---|---|---|---|---|
| ENG Premier League | 1,936 | 0.862 | 0.869 | 0.326 | −0.037 | 1.012 | 0.850 |
| ENG Championship | 3,311 | 0.854 | 0.854 | 0.292 | +0.023 | 0.969 | 0.847 |
| ENG League One | 1,152 | 0.850 | 0.850 | 0.310 | +0.020 | 0.961 | 0.843 |
| ENG League Two | 1,104 | 0.785 | 0.784 | 0.371 | +0.016 | 0.907 | 0.777 |
| SCO Premiership | 422 | 0.900 | 0.891 | 0.298 | +0.033 | 1.031 | 0.891 |
| SCO Championship | 1,084 | 0.812 | 0.786 | 0.428 | −0.319 | 1.130 | 0.702 |

Stable across six tiers and two countries — 0.785 to 0.900.

### 3.4 Transfer — the regime that licenses the production claim

`sco_lower` is the table the deployed feature actually builds (fitted on tiers 56/57 only, which carry no xG at all). `loto` is leave-one-tier-out.

| Tier | Regime | r | MAE | bias | slope | CCC |
|---|---|---|---|---|---|---|
| ENG Premier League | loto | 0.861 | 0.328 | −0.029 | 1.009 | 0.849 |
| ENG Premier League | **sco_lower** | **0.837** | 0.541 | **+0.432** | 0.782 | 0.749 |
| ENG Championship | loto | 0.853 | 0.296 | +0.044 | 0.958 | 0.846 |
| ENG Championship | **sco_lower** | **0.827** | 0.534 | **+0.453** | 0.742 | 0.705 |
| ENG League One | loto | 0.850 | 0.313 | +0.035 | 0.953 | 0.843 |
| ENG League One | **sco_lower** | **0.827** | 0.525 | **+0.437** | 0.749 | 0.717 |
| ENG League Two | loto | 0.787 | 0.372 | +0.031 | 0.901 | 0.780 |
| ENG League Two | **sco_lower** | **0.785** | 0.542 | **+0.429** | 0.752 | 0.686 |
| SCO Premiership | loto | 0.899 | 0.302 | +0.043 | 1.024 | 0.890 |
| SCO Premiership | **sco_lower** | **0.875** | 0.546 | **+0.477** | 0.798 | 0.770 |
| SCO Championship | loto | 0.811 | 0.435 | −0.332 | 1.143 | 0.693 |
| SCO Championship | **sco_lower** | **0.808** | 0.370 | −0.025 | 0.922 | 0.801 |

**Correlation transfers essentially intact** (−0.002 to −0.026 against leave-one-tier-out). **Level does not**: a consistent **+0.43 xG per team-match** over-statement, slope ≈ 0.75.

The mechanism is visible in the fitted tables: the `sco_lower` base rate is **0.1358** against the pooled **0.1046**. Lower tiers convert a higher share of attempts, so a table fitted there assigns more xG per shot. Cell rank agreement between the two tables is ρ = 0.855 over 109 shared cells.

**Consequence.** For `PxGCovariate` — whose scalar weight the engine learns — a constant scale factor is absorbed and this bias is harmless. It would **not** be harmless if pxG were consumed as an absolute rate.

### 3.5 Aggregation level and the decision-relevant quantity

| Level | n | r | MAE | bias | slope | CCC | R² |
|---|---|---|---|---|---|---|---|
| match TOTAL | 4,349 | 0.770 | 0.509 | −0.064 | 0.890 | 0.761 | 0.593 |
| **match DIFFERENCE** | 4,349 | **0.870** | 0.455 | −0.011 | **1.015** | **0.859** | **0.756** |

The difference — exactly what `PxGCovariate`'s supremacy role is built from — tracks *better* than the total, at unit slope. Sign agreement 85.1% over 4,330 decided matches; AUC(proxy diff → official diff > 0) = **0.926**.

Against real results: AUC(proxy diff → home win) = **0.714** against the official reference's own **0.745**. The proxy retains ~96% of official xG's predictive power for actual outcomes.

---

## 4. pxG form-window tuning (`r95`)

Every configuration rebuilt end-to-end; each value is point-in-time, so no match sees itself or a later one.

| Config | r(supremacy) | ρ | AUC(home win) |
|---|---|---|---|
| window = 1 | +0.1380 | +0.1207 | 0.5533 |
| window = 4 | +0.2146 | +0.1980 | 0.5938 |
| **window = 8 (shipped)** | **+0.2575** | +0.2353 | **0.6175** |
| window = 12 | +0.2782 | +0.2574 | 0.6263 |
| window = 19 | +0.2809 | +0.2671 | 0.6326 |
| window = all | +0.2468 | +0.2315 | 0.6112 |
| half-life = 3.0 | +0.2629 | +0.2399 | 0.6172 |
| half-life = 5.0 | +0.2798 | +0.2594 | 0.6277 |
| **half-life = 8.0** | **+0.2845** | **+0.2666** | **0.6319** |
| half-life = 13.0 | +0.2820 | +0.2654 | 0.6317 |

**The shipped 8-match flat window is too short.** Exponential decay with `half_life_matches = 8.0` is the maximum (+0.027 r, +0.014 AUC). Note that `window = all` is *worse* than window = 12–19, so there is a genuine interior optimum: form matters, but ~12–19 matches of it.

`prior_weight` is inert (r 0.2587 → 0.2562 across 0 → 12) and can be left at its default or removed.

### Recommended change

```julia
PxGFeature(decay = :exponential, half_life_matches = 8.0,
           prior_weight = 3.0, min_matches = 3, fallback = :shots)
```

---

## 5. RAPM forensics (`r94`)

### 5.1 Stints

10,655 segments over 1,809 matches (200 matches rejected: `starters_ne_11` 83, `sub_in_unknown` 65, `sub_out_off` 52). 56.6% live-text covered. 1,595 distinct players; median 540 on-pitch minutes.

### 5.2 Targets — per-segment response on covered segments

| Target | n | mean | sd | zeros |
|---|---|---|---|---|
| `y_xg` | 6,031 | 0.0291 | 0.3750 | **26.7%** |
| `y_goals` | 6,031 | 0.0315 | 0.7100 | **70.8%** |
| `y_shots` | 6,031 | 0.1768 | 2.1389 | 35.8% |
| `y_sot` | 6,031 | 0.0720 | 1.3024 | 51.4% |

`y_goals` is zero on 71% of stints — it carries almost nothing about a 20-minute personnel window. `y_xg` is the same event stream weighted by chance quality and is zero on only 27%. `y_xg ~ y_shots` correlate at +0.792.

Note `y_xg` is restricted to live-text-covered segments, so it rates **1,082 players** against `y_goals`'s 1,595. That is a real cost of the xG target.

### 5.3 λ and split-half reliability

Two disjoint halves of the *match* set (904 vs 905 matches), ratings correlated over players with ≥ 40 segments.

| Target | λ = 200 | λ = 1,000 | λ = 5,000 | λ = 20,000 | λ = 50,000 |
|---|---|---|---|---|---|
| `y_xg` | 0.2472 | 0.3601 | 0.3871 | 0.3930 | 0.3951 |
| `y_goals` | 0.1828 | 0.2808 | 0.3277 | 0.3425 | 0.3482 |
| **`y_shots`** | 0.4089 | 0.5200 | **0.5471** | 0.5531 | 0.5546 |

**Read the plateau, not the argmax.** As λ grows the ridge solution converges *in direction* to the unregularised gradient `X'Wy`, so reliability rises and then flattens rather than peaking — an argmax criterion just drives λ to the edge of whatever grid was written. The curves flatten from λ ≈ 2,000–5,000.

Two conclusions:
- **The shipped `λ = 200` sits well below the plateau** (0.247 against 0.395 for `y_xg`).
- **`y_shots` is far more reliable than `y_xg` or `y_goals`** (0.55 vs 0.40 vs 0.35), reproducing the original research's green-lit choice.

### 5.4 In-sample vs held-out — the leak that matters

| Target | In-sample r(sup) | In-sample AUC | **Held-out r(sup)** | **Held-out AUC** |
|---|---|---|---|---|
| `y_xg` (shipped) | +0.2361 | 0.5972 | **+0.1885** | **0.5969** |
| `y_goals` | +0.4204 | 0.7111 | **+0.0040** | 0.5373 |
| `y_shots` | +0.2081 | 0.5824 | **+0.1934** | **0.6020** |
| `y_sot` | +0.2303 | 0.5915 | +0.0803 | 0.5729 |

`y_goals` looked like the clear winner in-sample and is **worth nothing** held-out — a goal-differential ridge fitted on the matches it is scored against simply memorises their goal differences. `y_sot` loses two thirds of its apparent signal the same way.

`:y_xg` and `:y_shots` are stable across the two (0.236→0.189 and 0.208→0.193) and essentially tied held-out. **The shipped `:y_xg` default is vindicated.**

Held-out, λ barely matters (0.1885 at 200 → 0.1682 at 20,000) and `shrink_segments` is inert (0.1879 → 0.1885 across 0 → 60).

### 5.5 Ratings

At the shipped λ = 200: 1,082 rated players, sd 0.0246. By modal position the goalkeeper spread is not meaningfully different from outfield — consistent with the original finding that GK RAPM is worthless, and the reason the player engines' pillar collapses D+M+F and ignores G. The covariate column is well-behaved: mean −0.0024, sd 0.9872, available on 97.5% of matches, 2.5% neutral.

**Caveat carried in the script:** these are ridge coefficients on a stint xG differential with split-half reliability ≈ 0.25 at the shipped λ. Individual player orderings are indicative at best.

---

## 6. Feature synergy (`r93`)

402 played held-out matches, six covariate columns built on one common split.

### 6.1 Correlation (Pearson)

|  | wealth | prod_wealth | distance | pxg_sup | pxg_level | pxg_rapm |
|---|---|---|---|---|---|---|
| wealth | +1.000 | **+0.804** | +0.006 | +0.314 | −0.084 | +0.256 |
| prod_wealth | +0.804 | +1.000 | −0.021 | +0.381 | −0.024 | +0.336 |
| distance | +0.006 | −0.021 | +1.000 | −0.030 | +0.083 | −0.017 |
| pxg_supremacy | +0.314 | +0.381 | −0.030 | +1.000 | **+0.000** | +0.327 |
| pxg_level | −0.084 | −0.024 | +0.083 | +0.000 | +1.000 | +0.022 |
| pxg_rapm | +0.256 | +0.336 | −0.017 | +0.327 | +0.022 | +1.000 |

**`pxg_supremacy` and `pxg_level` are exactly orthogonal (r = +0.000)** — the two roles genuinely separate "who scores" from "how many", which is the design intent. **`wealth` and `prod_wealth` are collinear at +0.804** and should not be attached together.

### 6.2 VIF

| Feature | R² on rest | VIF |
|---|---|---|
| prod_wealth | 0.6771 | 3.097 |
| wealth | 0.6521 | 2.875 |
| pxg_supremacy | 0.1905 | 1.235 |
| pxg_rapm | 0.1607 | 1.191 |
| pxg_level | 0.0208 | 1.021 |
| distance | 0.0101 | 1.010 |

No VIF exceeds 5. Design condition number 9.2.

### 6.3 Held-out univariate signal

| Feature | r(sup) | ρ(sup) | r(total) | AUC(home win) | OLS t |
|---|---|---|---|---|---|
| wealth | +0.2231 | +0.1898 | −0.0313 | 0.5718 | +4.58 |
| prod_wealth | +0.2468 | +0.2127 | +0.0179 | 0.5885 | +5.09 |
| distance | +0.0467 | +0.0431 | +0.0604 | 0.5079 | +0.94 |
| **pxg_supremacy** | +0.2211 | **+0.2271** | **+0.1440** | **0.6221** | +4.53 |
| pxg_level | −0.1098 | −0.0757 | +0.1193 | 0.4855 | −2.21 |
| pxg_rapm | +0.1885 | +0.1926 | +0.0008 | 0.5969 | +3.84 |

`pxg_supremacy` has the **best AUC of all six** (0.622) despite production wealth having a marginally higher Pearson r.

### 6.4 What survives squad wealth — the decisive table

Baseline `goal supremacy ~ wealth`: R² = 0.0498.

| Candidate | r(wealth residual) | joint R² | **ΔR²** | t |
|---|---|---|---|---|
| **pxg_supremacy** | +0.1549 | 0.0751 | **+0.0253** | **+3.30** |
| **pxg_rapm** | +0.1347 | 0.0682 | **+0.0184** | **+2.81** |
| prod_wealth | +0.0690 | 0.0626 | +0.0128 | +2.34 |
| pxg_level | −0.0933 | 0.0581 | +0.0083 | −1.88 |
| distance | +0.0466 | 0.0519 | +0.0021 | +0.93 |

Against the stronger `prod_wealth` baseline (R² = 0.0609):

| Candidate | r(residual) | ΔR² | t |
|---|---|---|---|
| **pxg_supremacy** | +0.1312 | **+0.0189** | **+2.86** |
| **pxg_rapm** | +0.1090 | **+0.0126** | **+2.33** |
| pxg_level | −0.1071 | +0.0108 | −2.15 |
| distance | +0.0535 | +0.0027 | +1.07 |
| wealth | +0.0255 | +0.0017 | +0.86 |

**Both new covariates add significantly beyond either wealth measure.** `pxg_supremacy` adds the most of any candidate. Meanwhile raw `wealth` adds essentially nothing on top of `prod_wealth` (ΔR² +0.0017, t = 0.86).

### 6.5 Joint model

`goal supremacy ~ all six`: R² = 0.1032, adjusted R² = 0.0896, n = 402.

| Feature | beta | t |
|---|---|---|
| wealth | +0.3692 | +0.57 |
| prod_wealth | +0.7159 | +1.52 |
| distance | +0.1380 | +1.33 |
| **pxg_supremacy** | +0.5114 | **+2.43** |
| **pxg_level** | −0.6629 | **−2.29** |
| pxg_rapm | +0.2918 | +1.83 |

Only the two pxG roles survive at |t| ≥ 2. The two wealth covariates cancel each other. On the **total-goals** model (R² = 0.0461) `pxg_level` enters positively (beta +0.5418, t +2.16) alongside `pxg_supremacy` (+0.5776, t +3.17) — the level role does what it was built to do.

---

## 7. Recommendations

1. **Change the `PxGFeature` default** to `decay = :exponential, half_life_matches = 8.0`. Free +0.027 correlation and +0.014 AUC over the shipped 8-match flat window (§4).
2. **Raise `PxGRapmFeature.lambda`** from 200 toward the 2,000–5,000 reliability plateau, and confirm on r40's out-of-sample log loss — split-half reliability is a ceiling, not the decision (§5.3).
3. **Consider `target = :y_shots`** as a second RAPM arm. Its split-half reliability is 0.55 against `y_xg`'s 0.40, and held-out signal is equal or marginally better (§5.3–5.4).
4. **Do not attach `wealth` and `prod_wealth` together** (r = 0.804; neither survives in the joint model). Pick production wealth.
5. **Attach `pxg_supremacy` and `pxg_level` together.** They are orthogonal and both significant, on different response quantities (§6.1, §6.5).
6. **`distance` is not earning its parameter** on this store (t = 0.94 univariate, ΔR² +0.0021 over wealth).
7. **Never consume pxG as an absolute rate.** The deployed table over-states by +0.43 xG per team-match (§3.4). Safe for a learned-weight covariate only.
8. **Filter zero-filled xG** in any future work touching `sofascore.match_statistics` on tiers 3 and 84 (§3.1). Consider fixing this at the scrape.
9. The `fallback = :goals` rung is used on exactly **one** match and could be dropped; `fallback = :shots` carries all the value (§2).

## 8. Limitations

- **Nothing here validates the proxy on tiers 56/57.** Those tiers carry no official xG, no SofaScore statistics and no player ratings. §3 is transfer evidence, and it is the strongest available, but it is indirect.
- **`r93`'s linear regressions are a weak proxy for the count model.** The engine already carries team strength in `dyn.α`/`dyn.β`, so a covariate's real job is to explain what those cannot. A small ΔR² here can still earn its parameter in r40; a large one may simply be re-deriving team strength.
- **All of this is predictive fit, not edge.** Agreement with official xG is a measurement property. Allocation is `r22`'s question.
- Held-out blocks are 402 matches (`r93`, `r94`), so third-decimal differences between covariates are not resolvable.
- Split-half reliability uses one seed (20260830). It was not repeated across seeds.

---

## 9. Reproducing

```bash
source .env                 # BF_DB_URL, needed by r92 only
julia --project -t 8
```
```julia
include("current_development/scottish_lower/r92_pxg_vs_sofascore_xg_all_leagues.jl")  # DB pull, cached
include("current_development/scottish_lower/r93_feature_synergy_and_correlations.jl")
include("current_development/scottish_lower/r94_pxg_rapm_forensics.jl")
include("current_development/scottish_lower/r95_pxg_model_forensics.jl")
```

`r92` caches its pull to `l92_pxg_validation_pull.jls`; set `R92_FORCE_PULL=1` to refresh after a re-scrape. `r93`–`r95` need only the cached `ScottishLower` DataStore.
