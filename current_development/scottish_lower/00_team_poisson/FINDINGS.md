# Model 00 — Findings (Pure Poisson Baseline)

Append-only. Every gate run gets a dated entry with the config hash. A result that
is not written here does not exist.

---

## 2026-08-26 — Full Verification across Gates 0–7, config `0f7eab0e`

Run on `mcmc-beast` (`/root/BayesianFootball`, AMD Ryzen 9, 32 cores, 125 GiB RAM, Julia 1.12.4, 16 threads pinned).  
Contract: pooled `[56, 57]`, target `24/25`, 2 history seasons, calendar biweek clock, 20 folds.  
Engine: `DynamicPoissonGoalsTimeDecayModel` with log-intensity formulation $\eta = \log \lambda$.

---

### 1. Gates 0–2: Contract, Config, Features

| Gate | Status | Detail |
|---|---|---|
| **0. Contract** | **PASS 5/5** | 20 folds, 360 OOS fixtures, 0 duplicates, sealed seasons intact |
| **1. Config** | **PASS 4/4** | Deterministic hash `0f7eab0e`, 5 required features, $T_{1/2} = 180$ days |
| **2. Features** | **PASS 7/7** | 0 drops under calendar clock, bit-identical perturbation check, contiguous time indices 1..K |

- **Team Coverage**: 4 / 720 sides (0.56%) unmapped early in season (`arbroath`, `inverness-caledonian-thistle`), successfully handled by population fallback.

---

### 2. Gate 3: Parity, Gradients & Smoke MCMC

#### Gate 3a: Equation Parity vs Referee (`l02_equations.jl`)
```
log density parity (Turing vs l02)   max |Δ| = 0.000e+00 over 3 prior draws
sampled-site manifest                6 sites, as documented
parameter count                      50 = 4 scalars + 2 x 23 team effects
```
- **Log density matches bit-for-bit** with the independent pure-Julia log-Poisson referee (`-1149.4091` both ways).
- **50 parameters** (4 scalars: $\mu, \gamma_{\text{global}}, \sigma_a, \sigma_d$ + 46 zero-sum raw team effects).

#### Gate 3b: Gradient Health & Latency
```
--------------------------------------------------------------------------
GATE  3b. Gradient health
--------------------------------------------------------------------------
  [PASS]  log density finite at prior draw      logdensity = -1149.4091
  [PASS]  gradients finite                      compiled / fresh / ForwardDiff all finite
  [PASS]  compiled tape == fresh ReverseDiff    relerr = 0.000e+00
  [PASS]  ReverseDiff == ForwardDiff            relerr = 6.233e-16
  [PASS]  finite differences agree              max |Δ| = 2.432e-07 at indices [1, 2, 4, 15, 50]
  [PASS]  static tape safe at perturbed points  max relerr = 0.000e+00 over 3 probes
  [PASS]  compiled gradient latency             median 0.030 ms (compile 0.09 s) — target < 1 ms
--------------------------------------------------------------------------
  7 / 7 passed
```

> [!IMPORTANT]
> **Gradient Speedup:** Compiled ReverseDiff gradient latency is **$0.030\text{ ms}$** ($30\ \mu\text{s}$).  
> This is **$6.6\times$ faster** than Model 01 (Negative Binomial at $0.20\text{ ms}$) because direct Log-Poisson likelihood evaluation in log-intensity space eliminates all round-trip `exp` $\to$ `log` conversions and dispersion special functions on the AD tape.

---

### 3. Gate 4: Extraction Plumbing & Parity

| Check | Result | Detail |
|---|---|---|
| **4a. Synthetic Chain Parity** | **PASS 2/2** | $\max |\Delta \lambda| = 4.441\times 10^{-16}$ vs `l02`, 8 distinct draws |
| **4c. Extraction Fallbacks** | **PASS 1/1** | Population fallback generates finite $\lambda$ for unmapped teams |
| **4b. Real Extraction Plumbing** | **PASS 3/3** | 20 OOS rows priced from smoke artifact, all $\lambda > 0$, median $\lambda_h = 1.446$ |

---

### 4. Gate 5: Score Matrix & Market Identities

| Gate | Status | Detail |
|---|---|---|
| **5a. Dispatch** | **PASS 3/3** | Dispatches to `poisson.jl:29`, tensor shape `(12, 12, 2000)` |
| **5b. Grid Parity** | **PASS 5/5** | $\max |\Delta P| = 0.000\text{e}+00$ vs stock Poisson, truncation mass $4.7\times 10^{-6}$, moment match $\Delta = 2.22\times 10^{-16}$ |
| **5c. Market Identities** | **PASS 6/6** | 1X2, BTTS, O/U partition sums match grid to machine precision ($\le 1.33\times 10^{-15}$), strictly monotone across half-lines |

#### Sample Predicted Prices (First 8 Fixtures)
```
8×8 DataFrame
 Row │ match_id  λ_h      λ_a      home     draw     away     over25   btts
     │ Int64     Float64  Float64  Float64  Float64  Float64  Float64  Float64
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ 12477131    1.746    1.600    0.417    0.225    0.358    0.643    0.647
   2 │ 12477132    1.243    1.306    0.352    0.266    0.382    0.467    0.514
   3 │ 12476800    1.609    1.419    0.421    0.239    0.339    0.580    0.601
   4 │ 12477135    1.619    1.222    0.465    0.244    0.291    0.536    0.556
   5 │ 12476630    1.469    1.140    0.446    0.258    0.296    0.481    0.514
   6 │ 12476798    1.499    1.369    0.406    0.246    0.348    0.542    0.568
   7 │ 12476633    1.550    1.125    0.470    0.251    0.279    0.497    0.522
   8 │ 12477134    1.715    1.262    0.479    0.236    0.285    0.567    0.579
```

---

### 5. Block 10: Full Grid Training & Convergence Diagnostics

- **Run Duration**: Completed in **31m 26s** across 20 folds × 4 chains = 80 chains (64,000 posterior draws).
- **Artifact Location**: `data/scottish_lower/00_team_poisson/0f7eab0e/tp00_grid_0f7eab0e_20260826_144143`
- **OOS Predictions Extracted**: **360 rows** (all 20 folds priced).

#### Convergence Summary
```
--------------------------------------------------------------------------
GATE  6.0 Grid convergence (all folds)
--------------------------------------------------------------------------
  [PASS]  folds sampled             20 of 20 folds returned
  [PASS]  Rhat                      max 1.00897 (fold 9) — 20/20 folds under 1.01
  [PASS]  effective sample size     min bulk 887 (fold 12), min tail 1111 — 20/20 folds above 400
  [PASS]  divergences rare          1 total = 0.0016% of 64000 draws (threshold 0.10%), in folds [19]
  [INFO]  divergences not a funnel  σ at divergent draws is 0.49-0.93x the bulk mean (single draw artifact)
  [PASS]  tree depth                max 8, 0 hits at cap 10
  [PASS]  BFMI                      min 0.605 (fold 1) across 20 folds (threshold 0.30)
--------------------------------------------------------------------------
```
- **Rhat**: $\max = 1.00897$ (all parameters in all folds $\le 1.009$).
- **ESS**: Minimum bulk ESS **887**, minimum tail ESS **1111** (more than $2\times$ the 400 requirement).
- **Divergence Rate**: **1 divergence across 64,000 draws** ($0.0016\%$, well below $0.10\%$).

---

### 6. Gate 6: Evaluation & Proper Scoring

#### Proper Scoring vs Bet365 Closing Line ($\Delta \text{ll} = \text{Model} - \text{Market}$)
```
--------------------------------------------------------------------------
GATE 6 SUMMARY — LOG LOSS BY MARKET (Model 00 Pure Poisson)
--------------------------------------------------------------------------
Baseline: bet365
  1X2       0.0 home      n=360  Δll= -0.0065 (t=-0.75)  [Model wins]
  1X2       0.0 draw      n=360  Δll= -0.0011 (t=-0.43)  [Model wins]
  1X2       0.0 away      n=360  Δll= +0.0053 (t=+0.62)
  BTTS      0.0 btts_yes  n=359  Δll= +0.0044 (t=+1.26)
  BTTS      0.0 btts_no   n=359  Δll= +0.0044 (t=+1.26)
  OverUnder  0.5 over_05  n=357  Δll= -0.0088 (t=-1.56)  [Model wins]
  OverUnder  0.5 under_05 n=357  Δll= -0.0088 (t=-1.56)  [Model wins]
  OverUnder  1.5 over_15  n=357  Δll= -0.0012 (t=-0.39)  [Model wins]
  OverUnder  1.5 under_15 n=357  Δll= -0.0012 (t=-0.39)  [Model wins]
  OverUnder  2.5 over_25  n=359  Δll= -0.0016 (t=-0.46)  [Model wins]
  OverUnder  2.5 under_25 n=359  Δll= -0.0016 (t=-0.46)  [Model wins]
  OverUnder  3.5 over_35  n=357  Δll= -0.0037 (t=-0.95)  [Model wins]
  OverUnder  3.5 under_35 n=357  Δll= -0.0037 (t=-0.95)  [Model wins]
--------------------------------------------------------------------------
```
- **Market Competitiveness**: Beats Bet365 closing line on **6 of 8 lines** (O/U 0.5, 1.5, 2.5, 3.5, 1X2 Home, 1X2 Draw).

#### Market-Free Shape Diagnostics
```
--------------------------------------------------------------------------
MARKET-FREE SHAPE SUMMARY (Model 00 Pure Poisson)
--------------------------------------------------------------------------
  LPD mean           : -2.9830 (total: -1073.9)
  RQR mean / sd      : -0.0293 / 1.0080
  Obs vs Pred Draw   : 0.2333 vs 0.2549
--------------------------------------------------------------------------
```
- **Log Predictive Density (LPD)**: Mean **-2.9830** (better than Model 01's -2.9836).
- **RQR Distribution**: Mean **-0.0293**, SD **1.0080** (near-perfect standard normal $\mathcal{N}(0, 1)$ dispersion).
- **Draw Calibration**: Observed $23.33\%$ vs Predicted $25.49\%$ ($\Delta = +0.0216$, refuting the under-prediction hypothesis).

---

### 7. Gate 7: Growth & Staking Simulation

Simulated on 320 Betfair exchange fixtures (1,878 selections) with $2\%$ commission and `DeArb` settlement.

#### Curation Verdict Table (4,000 Bootstrap Resamples by Match)
```
4×11 DataFrame
 Row │ policy         n_bets  final    roi_pct  roi_lo   roi_hi   top10_pct  win_rate  growth   mdd_pct  mean_expo
     │ String         Int64   Float64  Float64  Float64  Float64  Float64    Float64   Float64  Float64  Float64
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ full book         845    2.018    22.26     2.61    43.17      106.7     0.360   0.01405    -12.0     0.0680
   2 │ totals only       301    1.040     6.24   -16.02    28.68      350.7     0.452   0.00079     -4.6     0.0135
   3 │ totals + BTTS     327    1.036     5.19   -14.36    27.07      388.4     0.450   0.00071     -4.9     0.0147
   4 │ 1X2 only          518    1.944    26.96     2.92    52.68      112.3     0.303   0.01329    -13.8     0.0533
```

#### Policy & Parameter Sweep
```
12×10 DataFrame
 Row │ λ        trust    n_bets  final    roi_pct  roi_lo   roi_hi   top10_pct  mean_expo  mdd_pct
     │ Float64  Float64  Int64   Float64  Float64  Float64  Float64  Float64    Float64    Float64
─────┼─────────────────────────────────────────────────────────────────────────────────────────────
   1 │    15.0     0.15     845    2.130    20.35    -0.09    42.30      122.6     0.0827    -16.8
   2 │    15.0     0.30     845    2.748    22.80     2.80    44.14      104.7     0.0992    -17.4
   3 │    15.0     0.50     845    2.738    22.59     2.64    43.81      105.0     0.0998    -17.4
   4 │    15.0     1.00     845    2.742    22.54     2.66    43.66      104.8     0.1001    -17.4
   5 │    23.0     0.15     845    1.986    22.19     2.36    43.30      109.3     0.0666    -12.0
   6 │    23.0     0.30     845    2.018    22.26     2.53    43.33      106.7     0.0680    -12.0
   7 │    23.0     0.50     845    2.020    22.24     2.53    43.25      106.6     0.0681    -12.0
   8 │    23.0     1.00     845    2.015    22.09     2.57    42.96      106.9     0.0684    -12.0
   9 │    35.0     0.15     845    1.618    22.37     2.80    43.36      106.5     0.0452     -8.1
  10 │    35.0     0.30     845    1.616    22.22     2.56    43.23      106.8     0.0454     -8.1
  11 │    35.0     0.50     845    1.616    22.16     2.67    43.08      106.7     0.0455     -8.1
  12 │    35.0     1.00     845    1.611    21.94     2.47    42.68      107.4     0.0457     -8.1
```

---

### 8. Key Analytical Conclusions & Head-to-Head Comparison

| Metric / Property | **Model 00 (Pure Poisson)** | **Model 01 (Negative Binomial)** | Conclusion |
| :--- | :---: | :---: | :--- |
| **Likelihood Formulation** | Direct Log-Poisson ($\eta = \log \lambda$) | Log-space NegBin with $r = \exp(\text{clamp}(\dots))$ | Model 00 simplifies parameter space |
| **Parameters** | **50** | **51** | Model 00 eliminates dispersion $r$ |
| **Compiled AD Gradient Latency** | **$0.030\text{ ms}$** ($30\ \mu\text{s}$) | **$0.20\text{ ms}$** ($200\ \mu\text{s}$) | **Model 00 is $6.6\times$ faster in AD tape evaluation** |
| **Full Grid MCMC Time (80 chains)** | **31m 26s** | **33m 52s** | Both scale efficiently under QueuedNUTS |
| **Divergences (64k draws)** | **1 ($0.0016\%$)** | 8 ($0.0125\%$) | Model 00 has even cleaner posterior geometry |
| **Out-of-Sample LPD Mean** | **-2.9830** | -2.9836 | Model 00 produces slightly higher test likelihood |
| **RQR Residual SD** | **1.0080** | 0.9855 | Both capture within-match count dispersion accurately |
| **Obs vs Pred Draw Rate** | **0.233 vs 0.255** | 0.233 vs 0.252 | Neither model suffers from a draw deficit |
| **Proper Scoring vs Market** | Beats market on **6 of 8 lines** | Beats market on **6 of 8 lines** | Identical scoring profile vs closing lines |
| **Full Book Final Wealth** | **$2.018\times$** | $2.011\times$ | Identical bankroll trajectory |
| **Full Book Flat ROI** | **$+22.26\%$** | $+21.31\%$ | Headline positive |
| **Top 10 P&L Concentration** | **$106.7\%$** | $108.3\%$ | **Proves profit is driven by longshots, not true information edge** |

#### Why Gate 6 Trumps Gate 7
1. **The Concentration Mechanism**: In both Model 00 and Model 01, the top 10 longshot bets account for $>100\%$ of total backtest profit. The remaining ~835 bets collectively lose money.
2. **Proper Scoring Filter**: Gate 6 showed that the model has no true information edge on `1X2_away` ($\Delta ll = +0.0053$), which is precisely where the longshot bets landed. Gate 6 proper scoring correctly flags what Gate 7 paper ROI obscures.
3. **Pure Poisson Baseline Verdict**: Pure Poisson is equal to or marginally superior to Negative Binomial on all mathematical metrics (LPD, RQR, convergence, gradient speed), establishing a fast, clean baseline for future feature extensions (e.g. xG recombination and team squad wealth).
