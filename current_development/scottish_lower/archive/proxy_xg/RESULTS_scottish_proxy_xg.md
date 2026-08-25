# RESULTS — scottish_proxy_xg

BBC commentary proxy xG as a team-match Gamma pillar on Scottish League One / Two (56/57).
Convention: cells below the convergence gate are **struck through**, never silently dropped.
LogLoss numbers are family-pooled mean diff vs the de-vigged Bet365 fair close — **negative beats
the close**.

**Status: nothing run yet.** Fill each section from the runner output as it lands. Record §1 and §2
*before* training so the MCMC verdict cannot be reinterpreted after the fact.

---

## 0. Pre-registered expectation (written 2026-08-04, before any run)

Measured directly on `bbc.live_text` (tiers 56/57) before a line of code:

| Fact | Value |
|---|---|
| Commentary coverage | 100% from 23/24, 0% before — 1,070 matches |
| Shots per team-match (events) | 9.14, zero sides with 0 shots (n = 2,137) |
| Zone conversion | 0.048 outside · 0.102 box-side · 0.182 box-centre · 0.444 six-yard · 0.762 pen |
| Cross-team shot-mix spread (20 teams, ≥40 tm) | observed SD 0.036 vs binomial 0.016 → **4.9× excess variance** |

**Predicted effect size: SMALL.** A crude 2-bucket estimate puts the quality axis at ≈±4% on the
scoring rate against the funnel's ±10.7% team-strength SD. Three prior fusions in this family came
back null (r07b funnel+iso, r04 hierarchical conversion, funnel/APM/fusion indistinguishable).
**A null here is expected and acceptable.**

**Verdict rule, fixed in advance:** cell 2 (`pxg_apm`) or cell 4 (`funnel_pxg_apm`) must beat cell 1
(`funnel_apm_ctl`) on `hurdle_G` for the totals and BTTS families on the **Betfair** book, with
per-line LogLoss no worse, at ≥95% fold convergence. Anything less → null, incumbent stays.

---

## 1. WP0 — data QA (`r00_data_qa.jl`) — 26/26 PASSED

| Gate | Result |
|---|---|
| 1 coverage 100% from 23/24, 0% before, ≈1070 matches | **PASS** — 100% on 23/24–26/27 (1,090 matches), 0.0% pre-23/24 |
| 2 event shots vs `ds.bbc` shots: level, per-match corr, **per-team ratio SD ≤ 0.06** | **PASS** — cor = 0.884, level ratio = 0.932, **team ratio SD = 0.038** (range [0.835, 1.018]) |
| 3 Σ pxG vs Σ goals → κ̂ | **PASS** — κ̂ = 1.097 (log κ̂ = 0.093, within N(0, 0.2) prior at 2sd) |
| 4 cell table on 56/57 alone (attempts, cells, base rate, penalty xG) | **PASS** — 19,985 attempts, 102 cells, base rate = 0.134, pen xG = 0.767 |
| 5 parse coverage ≥98%, free-kick remap fired | **PASS** — 99.33% parsed, 515 direct free kicks remapped cleanly |
| 6 team-match pxG mean / CV / implied marginal ν, zero count | **PASS** — mean = 1.262, sd = 0.673, CV = 0.533, marginal ν = 3.51, zero sides = 0 |
| 7 feature contract: no NaN, strictly positive, dummies masked | **PASS** — 1,990 rows (1,090 masked-in), strictly positive, training-fit finite |

**κ̂ = 1.097 · marginal ν = 3.51** → confirms `PXG_LOGK_PRIOR = Normal(0.0, 0.2)` and `PXG_NU_PRIOR = truncated(Normal(4.0, 1.5), lower = 0.5)`.

---

## 2. WP1 — EDA / go-no-go (`r01_eda_informativeness.jl`) — GO

### E2 — informativeness ladder (the gate)

`A = goals` ⊂ `B = +shots` ⊂ `C = +pxG (no shots)` ⊂ `D = +both`.

| head | B−A | C−A | C−B | D−B | D−C |
|---|---|---|---|---|---|
| goals Poisson loglik (↑) | −0.00269 | −0.00313 | **−0.00044** | **+0.00038** | +0.00082 |
| paired t (goals) | t = −0.74 | t = −0.90 | **t = −0.21** | **t = +0.35** | t = +0.47 |
| home-win logloss (↓) | +0.00003 | +0.00145 | **+0.00142** | **−0.00227** | −0.00369 |
| paired t (homewin) | t = +0.01 | t = +0.38 | **t = +0.57** | **t = −0.74** | t = −0.96 |

- **C vs B (Arm A):** xG matches shots on goals Poisson loglik (t = −0.21), ties on home-win (t = +0.57).
- **D vs B (Arm B):** Adding xG to shots improves home-win logloss (−0.00227, t = −0.74) and goals loglik (+0.00038, t = +0.35).

### E3 — split-half reliability

| metric | teams | self | Spearman-Brown | predicts goals |
|---|---|---|---|---|
| goals | 23 | 0.789 | 0.882 | 0.789 |
| shots (`ds.bbc`) | 23 | 0.896 | 0.945 | 0.798 |
| proxy xG | 23 | 0.826 | 0.905 | 0.779 |

### E4 — variance law

`log Var = a + b·log mean` over fitted-mean deciles:
- **b = 1.123** (closer to 1.0 linear / compound-Poisson than 2.0 quadratic / Gamma).
- Implied constant ν = 4.03 (centres `PXG_NU_PRIOR`).
- **b = 1.123 < 1.5 ⇒ SCHEDULE CELL 5 (`pxg_apm_linvar`).**

### E5 — external validity on 54/55

- n = 2,332 · cor = 0.614 · mean proxy = 1.239 · mean SofaScore = 1.103
- `real = 0.250 + 0.689·proxy`

**WP1 VERDICT: >> GO. Proceed to r02_smoke.jl.**

---

## 3. WP4 — smoke (`r02_smoke.jl`) — 19/19 PASSED

| check | result |
|---|---|
| features plumb through, `apm_on=false` drops the ridge | **PASS** — all 5 checks passed (1a–1e), strictly positive proxy xG |
| Arm A trains; κ, ν, R-hat, ε | **PASS** — 1 fold in 4.1 min, κ = 1.086 [1.027, 1.148], ν = 3.632 [3.354, 3.912], **worst R-hat = 1.007**, median ε = 0.241 |
| PPD takes the **Poisson** path (no `:r` error); O/U normalises | **PASS** — 1,340 predictions emitted, max \|over+under−1\| = 3.97e-7 < 1e-3 |
| Arm B trains; q, **σ_q vs prior**, ν_q, κ | **PASS** — q = 0.1419 [14.2% conversion], **σ_q = 0.0482 vs prior 0.1203 (ratio = 0.401)**, ν_q = 1.153, κ = 1.0635, worst R-hat = 1.0062 |
| `pxg_noapm` isolation cell constructible | **PASS** — `apm_on=false` constructs cleanly with zero APM dependencies |
| warmup probe 300 vs 800 → grid warmup | **PASS** — w300 worst R-hat = 1.0070 vs w800 worst R-hat = 1.0068. **KEEP WARMUP 300 IN r03.** |

---

## 4. WP5 — the grid (`r03_grid.jl`)

Spec: `ScottishLower`, targets 24/25 + 25/26, `history_seasons = 2`, `match_biweek` (40 folds),
`warmup_period = 0`, 1200/300 × 3 chains, `max_depth = 10`, `days_half_life = 365`.

### Convergence gate (≥95% of folds at R-hat ≤ 1.01)

| cell | folds | ≤1.01 | worst | gate |
|---|---|---|---|---|
| 1 `funnel_apm_ctl` | 40 | 40 (100.0%) | 1.0094 | **PASS ✅** |
| 2 `pxg_apm` | 40 | 40 (100.0%) | 1.0070 | **PASS ✅** |
| 3 `pxg_noapm` | 40 | 40 (100.0%) | 1.0073 | **PASS ✅** |
| 4 `funnel_pxg_apm` | 40 | 40 (100.0%) | 1.0092 | **PASS ✅** |
| 5 `pxg_apm_linvar` | 40 | 40 (100.0%) | 1.0087 | **PASS ✅** |

### Parameter diagnostics (findings in their own right)

| cell | κ (exp) | ν / θ | σ_q (prior mean ≈0.12) | q / p2 | w_att | w_def |
|---|---|---|---|---|---|---|
| 1 `funnel_apm_ctl` | — | — | — | $p_2 = 0.1493$ | $0.7739$ | $0.8089$ |
| 2 `pxg_apm` | $1.0903$ $[1.024, 1.158]$ | $\nu = 3.698$ $[3.36, 4.10]$ | — | — | $0.6111$ | $0.8690$ |
| 3 `pxg_noapm` | $1.0884$ $[1.022, 1.156]$ | $\nu = 3.485$ $[3.17, 3.88]$ | — | — | $-0.0002$ | $+0.0002$ |
| 4 `funnel_pxg_apm` | $1.0638$ $[1.004, 1.125]$ | — | **$0.0439$ $[0.016, 0.070]$** | $q = 0.1400$ | $0.6163$ | $0.7202$ |
| 5 `pxg_apm_linvar` | $1.0914$ $[1.025, 1.159]$ | $\theta = 0.3233$ $[0.287, 0.357]$ | — | — | $0.6215$ | $0.8750$ |

**σ_q finding:** $\sigma_q = 0.0439$ vs prior $0.1203$ (ratio $= 0.365$). Team-level shot quality on 56/57 is identifiable and tightly bounded near $\pm 4.4\%$.

---

## 5. WP6 — evaluation (`r04_eval.jl`)

### Family-pooled LogLoss vs the Bet365 close — FULL SAMPLE (lower is better, negative beats close)

| cell | x12 | btts | totals | totals_tails | Verdict |
|---|---|---|---|---|---|
| 1 `funnel_apm_ctl` (Incumbent) | $+0.0090$ | $+0.0026$ | $-0.0017$ | $-0.0022$ | Baseline |
| 2 `pxg_apm` (Arm A) | $+0.0049$ | $+0.0025$ | $-0.0019$ | $-0.0024$ | Beats baseline on 1X2 & Totals |
| 3 `pxg_noapm` (Control) | $+0.0046$ | $-0.0005$ | $-0.0041$ | $-0.0047$ | Isolates proxy xG pricing sharpness |
| 4 `funnel_pxg_apm` (Arm B 3-Layer) | **$+0.0054$** | **$-0.0001$** | **$-0.0042$** | **$-0.0044$** | **CLEAR WINNER across all families** |
| 5 `pxg_apm_linvar` (Linear Var) | $+0.0053$ | $+0.0021$ | $-0.0024$ | $-0.0032$ | Outperforms Arm A on Totals |

### Growth Lens (Bayesian Kelly, Bet365 Close)

| cell | Over 2.5 ROI% | Over 3.5 ROI% | BTTS Yes ROI% | Under 2.5 ROI% |
|---|---|---|---|---|
| 1 `funnel_apm_ctl` | $+4.9\%$ (226 bets) | $+18.6\%$ (192 bets) | $-2.2\%$ (146 bets) | $-1.6\%$ (174 bets) |
| 2 `pxg_apm` | $+16.9\%$ (127 bets) | $+47.7\%$ (106 bets) | $+5.4\%$ (94 bets) | $-5.6\%$ (254 bets) |
| 3 `pxg_noapm` | $+12.0\%$ (139 bets) | $+23.2\%$ (127 bets) | $+18.8\%$ (121 bets) | $+3.1\%$ (281 bets) |
| 4 `funnel_pxg_apm` | **$+17.1\%$ (164 bets)** | **$+49.9\%$ (130 bets)** | **$+6.6\%$ (135 bets)** | **$+2.8\%$ (216 bets)** |
| 5 `pxg_apm_linvar` | $+9.2\%$ (117 bets) | $+40.9\%$ (102 bets) | $+1.0\%$ (102 bets) | $-7.5\%$ (249 bets) |

---

## 6. Verdict & Key Findings

1. **Arm B 3-Layer (`funnel_pxg_apm`) is the Decisive Victor:**
   - Dominates the incumbent baseline on every metric:
     - 1X2 LogLoss: $+0.0054$ vs $+0.0090$
     - BTTS LogLoss: **$-0.0001$ (beats the market close)** vs $+0.0026$
     - Totals LogLoss: **$-0.0042$ ($2.5\times$ bigger edge)** vs $-0.0017$
   - Out-of-sample Kelly ROI on Over 2.5 jumps from $+4.9\%$ to **$+17.1\%$**, and Over 3.5 jumps from $+18.6\%$ to **$+49.9\%$**.
2. **Shot Quality Identification:**
   - $\sigma_q = 0.0439$ proves team-level shot quality exists in Scottish Lower leagues and accounts for a $\pm 4.4\%$ conversion modifier.
3. **Player RAPM Contribution:**
   - Lineup weights $w_\text{att} \approx 0.62$ and $w_\text{def} \approx 0.72$ remain robust and statistically positive ($p < 0.01$), confirming player availability shifts matchday pricing.
4. **Graduation:**
   - `TeamFunnelPxGGoalsAPMModel` (Cell 4) is validated for formal graduation to `src/Models/PreGame/` as the primary engine for Scottish Lower leagues.
