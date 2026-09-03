# Multi-Tier Categorical Trust System Across Football Market Lines

## Executive summary

Tilting the asymmetric-core trust table into two or three conviction tiers **raises both return and Sharpe on both models, monotonically, over the entire predeclared policy set** — but the gain is not from a new discovery about which markets have edge. It is a direct, mechanical consequence of moving capital from the asymmetric core's weakest positive-ROI leg (Away, tier 2) into its strongest (Under 2.5 and Home, tier 1), which the joint Kelly solve was already leaving under-weighted relative to its price under flat trust.

Headline full-period results (100 slates, both models):

| Policy | Tier 1 : Tier 2 (: Tier 3) | Mean return | Mean Sharpe | Mean max DD |
|---|---|---:|---:|---:|
| `P0_flat_benchmark` (= asymmetric core) | 0.30 : 0.30 | +145.03% | 1.532 | -18.77% |
| `P1_conservative_tilt` | 0.35 : 0.25 | +163.05% | 1.667 | -19.73% |
| `P2_conviction_tilt` | 0.40 : 0.20 | +172.76% | 1.770 | -21.16% |
| `P3_aggressive_tilt` | 0.50 : 0.25 | +172.76% | 1.770 | -21.16% |
| `P4_four_tier_probe` | 0.40 : 0.25 : 0.05 | +168.98% | 1.717 | -20.59% |

Four findings drive the recommendation:

1. **The allocator is scale-invariant in absolute trust; only the tier ratio matters.** `P2_conviction_tilt` (0.40 : 0.20, ratio 2.0) and `P3_aggressive_tilt` (0.50 : 0.25, ratio 2.0) produce **bit-identical** portfolios on both models — final bankroll, every bet, every daily state agree to floating-point precision, despite one policy nominally trusting the model 25% more than the other. The 20% `FixedCap` never once binds in any tested policy (`n_capped = 0` throughout); the mechanism is `SlateDrawdown(23.0)`, which renormalizes the whole slate's risk to a fixed tail-loss budget every day. Doubling both tiers' trust doubles the pre-risk Kelly vector, and the risk overlay simply picks a smaller `k_risk` to land at the same budget — `mean_k_risk` falls from 0.286 to 0.230 (`m12`, P2→P3) while every realized quantity stays fixed. **Conviction should be specified and reasoned about as a ratio between tiers, not as an absolute trust level.**
2. **Conviction tilt is a capital-reallocation mechanism, not a new-edge discovery.** Under `P0`, Tier 2 (Draw + Away) held 55% of core capital at ~7.3% pooled ROI while Tier 1 (Home + Under 2.5) held 45% at ~13% ROI. Tilting the ratio to 2:1 (`P2`/`P3`) moves the split to 59:41 in Tier 1's favour on `m12` (60:40 on `m13`) while **each tier's own realized ROI stays almost exactly fixed** (Tier 1: 12.9%→13.9%, Tier 2: 7.34%→7.33%, `m12`). The extra return is arithmetic: shifting turnover from a ~7% line to a ~13% line raises the blend.
3. **The Pareto frontier has an interior optimum for return, but not (yet) for Sharpe.** Return peaks at a Tier1:Tier2 ratio near **2.0–2.5** on both models (`m12`: 169.39% at ratio 2.25; `m13`: 178.17% at ratio 2.5) and *declines* on both sides of that peak. Sharpe, by contrast, is still rising at the edge of the tested grid (ratio 5.0, i.e. `tau2 = 0.10`) on both models — the requested grid does not contain the Sharpe-optimal ratio.
4. **The four-tier probe does not pay for itself.** Adding Under 1.5 and Over 3.5 at `tau3 = 0.05` draws in 200 (`m12`) / 199 (`m13`) extra bets for a combined P&L of **-$3.9 / -$0.1** and **1.1%** of staked capital — flat-to-negative, exactly reproducing the "all unders" failure from the asymmetric-selection-trust audit, this time at a token trust weight rather than a full one.

**Recommendation:** graduate from flat asymmetric core to a **modest tilt, `tau1 = 0.35, tau2 = 0.25` (`P1_conservative_tilt`, ratio 1.4)**, not the return-maximizing ratio ≈2.0–2.5. `P1` captures roughly 60% of the available Sharpe improvement (1.667 vs. 1.532 baseline vs. 1.770 at the frontier) for about half the added drawdown (+0.96pp vs. +2.39pp average), and it sits close to the ratio that maximizes Calmar (return per unit of max drawdown) on both models. Do not deploy the nominal `P3_aggressive_tilt` table: it is provably identical to `P2_conviction_tilt` under this engine and only obscures that fact. Do not enable the Tier 3 probe.

---

## 1. Experimental contract

### 1.1 Models and provenance

Same immutable canonical 40-fold PostgreSQL fits as the asymmetric-selection-trust audit:

| Role | Model | Run UUID | Convergence |
|---|---|---|---|
| primary | `m12_joint_hybrid_synergy` | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | strict pass; 40/40 folds; max R-hat 1.0104; min ESS-bulk 880.7 |
| sensitivity | `m13_joint_composite` | `5474e824-8c9d-4613-8e39-841426c3f80f` | strict aggregate failure on tail ESS; max R-hat 1.0132; min ESS-bulk 712.8 |

No MCMC was launched and no fit artifact was modified. Deserialization required the artifact-compatible worktree pinned to commit `784c8ea81328760e75498b19d13c2dab762bde8e`, because the current `JointGammaPoissonObservation` type has changed since serialization. Each model produced 632 usable match books from 710 held-out fixtures over exactly 100 daily slate dates (2024-08-03 to 2026-04-25); 78 fixtures lacked a usable controlled book. `m13` outputs remain sensitivity evidence, not a second production convergence pass.

### 1.2 What is held fixed

Every policy uses the same six-market book (1X2, O/U 0.5/1.5/2.5/3.5, BTTS), the same Betfair time-weighted close over the final 20 minutes before kickoff, `DeArb()` + `KellyLogUtility()` + `NoShrinkage()`, 2% commission with a 0.001 minimum selection stake, `SlateDrawdown(23.0)`, `FixedCap(0.20)`, `DailySlate()` grouping, and a 1,000-unit initial bankroll. Only the strict `SelectionTrust` table changes, and every one of the thirteen directions across the six markets is explicitly assigned a value (`strict = true`): a missing key raises rather than silently defaulting to zero.

### 1.3 The tier definition

| Tier | Directions | Role |
|---|---|---|
| Tier 1 | `1X2 Home`, `Under 2.5` | super-alpha (asymmetric-audit pooled ROI +11.67% / +18.67%, IR 1.15 / 1.24) |
| Tier 2 | `1X2 Draw`, `1X2 Away` | moderate diversifiers (pooled ROI +10.28% / +5.63%, IR 0.36 / 0.65) |
| Tier 3 | `Under 1.5`, `Over 3.5` | break-even probe (pooled ROI -0.83% / -0.99%) |
| excluded | remaining 7 directions | toxic/negative in the prior audit; pinned at `tau = 0.00` in every policy tested here |

### 1.4 Policies

| Policy | Tier 1 τ | Tier 2 τ | Tier 3 τ | Status |
|---|---:|---:|---:|---|
| `P0_flat_benchmark` | 0.30 | 0.30 | 0.00 | predeclared; reproduces asymmetric core exactly |
| `P1_conservative_tilt` | 0.35 | 0.25 | 0.00 | predeclared |
| `P2_conviction_tilt` | 0.40 | 0.20 | 0.00 | predeclared |
| `P3_aggressive_tilt` | 0.50 | 0.25 | 0.00 | predeclared |
| `P4_four_tier_probe` | 0.40 | 0.25 | 0.05 | predeclared |
| `P5_grid_sweep` | 0.25–0.50 (step 0.05) | 0.10–0.35 (step 0.05) | 0.00 | 36-cell grid, both models |

`P0_flat_benchmark` was independently re-derived twice per model — once via the named-policy path and once as grid cell `(tau1, tau2) = (0.30, 0.30)` — and the runner asserts the two agree to `1e-6` before writing any CSV.

---

## 2. The knapsack allocation mechanics

### 2.1 Absolute trust is irrelevant; only the tier ratio survives the risk overlay

The grid sweep's central structural result: **every grid cell sharing the same `tau1 / tau2` ratio produces the identical portfolio**, verified across every duplicate ratio in the 36-cell grid on both models (e.g. `(0.30, 0.15)`, `(0.40, 0.20)`, and `(0.50, 0.25)` — all ratio 2.0 — return `168.976%` / `1255` bets / `-21.307%` MDD on `m12`, agreeing to at least 10 significant figures):

| Model | Cell A | Cell B | Ratio | Return A | Return B | Δ |
|---|---|---|---:|---:|---:|---:|
| `m12` | (0.30, 0.15) | (0.50, 0.25) | 2.0 | 168.976% | 168.976% | <1e-9 pp |
| `m12` | (0.25, 0.10) | (0.50, 0.20) | 2.5 | 168.863% | 168.863% | <1e-9 pp |
| `m13` | (0.30, 0.10) | (0.45, 0.15) | 3.0 | 176.61% | 176.61% | <1e-9 pp |

The mechanism is `SlateDrawdown(23.0)`: it solves for one scalar `k_risk ∈ [0, 1]` per slate such that `k_risk` applied to the pre-risk Kelly vector keeps the probability of breaching an 80%-of-bankroll floor under the configured budget. Because `KellyLogUtility` scales each selection's raw fractional stake linearly in that selection's trust, multiplying both tiers' trust by a common factor scales the *entire pre-risk vector* by that factor — and `SlateDrawdown` exactly compensates by shrinking `k_risk` in inverse proportion. Confirming this: `P2` (τ = 0.40/0.20) has `mean_k_risk = 0.2855`/`0.2841` (`m12`/`m13`) while nominally-more-aggressive `P3` (τ = 0.50/0.25) has `mean_k_risk = 0.2304`/`0.2293` — a materially different scaling factor absorbing the entire nominal difference, leaving the realized portfolio untouched. **The explicit 20% `FixedCap` never bound once** (`n_capped = 0` in all ten named-policy runs and all 72 grid cells): the risk overlay is the sole active capacity constraint in this policy family, contrary to the framing that motivated this audit.

The practical implication: a MatchDay trust table specified as raw tau values invites a false sense of "how much conviction" is encoded. Two operators could set nominally different trust levels for the same ratio and get, with certainty, the same bets. **Tier trust should be specified, documented, and reasoned about as a ratio.**

### 2.2 The reallocation is a capital shift between two stable-ROI tiers, not a new edge

Aggregate stake share and realized ROI by tier, `m12` (`m13` in the CSVs; the same pattern holds within 1pp):

| Policy | Tier 1 share | Tier 1 ROI | Tier 2 share | Tier 2 ROI |
|---|---:|---:|---:|---:|
| `P0_flat_benchmark` | 45.0% | 12.89% | 55.0% | 7.34% |
| `P1_conservative_tilt` | 51.8% | 13.41% | 48.2% | 7.39% |
| `P2_conviction_tilt` / `P3_aggressive_tilt` | 59.0% | 13.93% | 41.0% | 7.33% |
| `P4_four_tier_probe` | 53.9% | 13.58% | 45.0% | 7.35% (Tier 3: 1.1% share, -2.21% ROI) |

Each tier's *own* realized ROI is essentially flat across the whole policy sweep (Tier 1: 12.9–13.9%; Tier 2: 7.3–7.4%). The return improvement from `P0` to `P2`/`P3` (+23.9pp average) is not because Draw or Away start winning less on their own bets — it is because the same win rate on those bets now commands a smaller share of turnover, while the same 19%-ROI Under-2.5 leg and 10–12%-ROI Home leg command a larger one. This is a pure Kelly-allocation reallocation, consistent with (and explaining) finding 2 in the executive summary.

Selection-level detail, `m12`, `P0` → `P2_conviction_tilt`:

| Selection | Bets | Stake share, `P0` | Stake share, `P2` | ROI, `P0` | ROI, `P2` |
|---|---:|---:|---:|---:|---:|
| Home | 293 | 32.00% | 40.67% | 10.39% | 11.70% |
| Away | 374 | 37.87% | 28.66% | 5.16% | 3.02% |
| Draw | 361 | 17.16% | 12.36% | 12.16% | 17.32% |
| Under 2.5 | 227 | 12.97% | 18.31% | 19.07% | 18.87% |

Bet counts are identical (selection membership is unchanged; only stake sizing moves), and the correlated joint solve reshuffles per-bet stake weight in a way that is not a uniform per-selection scalar — Away's realized ROI *falls* under tilt (5.16%→3.02%) while Draw's *rises* (12.16%→17.32%), because the size of a bet on one selection changes which slates it is exposed alongside once the tier-1 legs on the same fixtures grow.

---

## 3. Policy comparison — full 100-slate results

| Model | Policy | Final bankroll | Return | Sharpe | Sortino | Calmar | Max DD | Bets | Turnover | Cap-binding slates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m12` | `P0_flat_benchmark` | 2,439.15 | +143.91% | 1.516 | 1.886 | 7.825 | -18.39% | 1,255 | 7.519 | 0 |
| `m12` | `P1_conservative_tilt` | 2,607.61 | +160.76% | 1.645 | 2.088 | 8.056 | -19.95% | 1,255 | 7.469 | 0 |
| `m12` | **`P2_conviction_tilt`** | **2,689.76** | **+168.98%** | **1.742** | **2.183** | 7.930 | -21.31% | 1,255 | 7.192 | 0 |
| `m12` | `P3_aggressive_tilt` | 2,689.76 | +168.98% | 1.742 | 2.183 | 7.930 | -21.31% | 1,255 | 7.192 | 0 |
| `m12` | `P4_four_tier_probe` | 2,657.01 | +165.70% | 1.689 | 2.132 | 8.000 | -20.71% | 1,455 | 7.510 | 0 |
| `m13`† | `P0_flat_benchmark` | 2,461.37 | +146.14% | 1.548 | 1.896 | 7.628 | -19.16% | 1,260 | 7.561 | 0 |
| `m13`† | `P1_conservative_tilt` | 2,653.39 | +165.34% | 1.689 | 2.089 | 8.473 | -19.51% | 1,260 | 7.516 | 0 |
| `m13`† | **`P2_conviction_tilt`** | **2,765.53** | **+176.55%** | **1.798** | **2.293** | 8.406 | -21.00% | 1,260 | 7.244 | 0 |
| `m13`† | `P3_aggressive_tilt` | 2,765.53 | +176.55% | 1.798 | 2.293 | 8.406 | -21.00% | 1,260 | 7.244 | 0 |
| `m13`† | `P4_four_tier_probe` | 2,722.58 | +172.26% | 1.744 | 2.224 | 8.415 | -20.47% | 1,459 | 7.557 | 0 |

† strict aggregate convergence failure on tail ESS.

`P2_conviction_tilt` and `P3_aggressive_tilt` are listed separately for completeness; per §2.1 they are the identical portfolio, and every column above matches to more decimal places than shown.

### 3.1 Increment from tilting (`P2` vs. `P0`)

| Model | Return delta | Sharpe delta | Sortino delta | Max-DD change | Turnover change |
|---|---:|---:|---:|---:|---:|
| `m12` | **+25.06 pp** | **+0.226** | **+0.297** | **-2.92 pp (worse)** | -0.327 |
| `m13`† | **+30.42 pp** | **+0.250** | **+0.397** | **-1.85 pp (worse)** | -0.317 |

Both headline metrics improve while turnover *falls* — the engine is deploying less total capital, more selectively.

---

## 4. Temporal split & overfitting audit

Slates 1–50 (2024-08-03 to 2025-05-03, Season 24/25) vs. slates 51–100 (2025-05-04 to 2026-04-25, Season 25/26). These windows were not used to choose the five predeclared tier assignments.

| Model | Policy | 1st-half return | 1st-half Sharpe | 2nd-half return | 2nd-half Sharpe | 2nd-half max DD |
|---|---|---:|---:|---:|---:|---:|
| `m12` | `P0_flat_benchmark` | +86.89% | 2.456 | +30.52% | 0.922 | -18.39% |
| `m12` | `P1_conservative_tilt` | +95.92% | 2.699 | +33.10% | 0.991 | -19.95% |
| `m12` | **`P2_conviction_tilt`** | **+99.78%** | **2.921** | **+34.64%** | **1.040** | -21.31% |
| `m12` | `P4_four_tier_probe` | +98.82% | 2.781 | +33.64% | 1.013 | -20.71% |
| `m13`† | `P0_flat_benchmark` | +81.81% | 2.397 | +35.38% | 1.051 | -19.16% |
| `m13`† | `P1_conservative_tilt` | +90.52% | 2.655 | +39.27% | 1.142 | -19.51% |
| `m13`† | **`P2_conviction_tilt`** | **+94.65%** | **2.898** | **+42.08%** | **1.210** | -21.00% |
| `m13`† | `P4_four_tier_probe` | +94.10% | 2.762 | +40.27% | 1.169 | -20.47% |

The ranking `P2`/`P3` > `P4` > `P1` > `P0` holds **in both halves independently, on both models**, for both return and Sharpe. Conviction tilt is not an artifact carried by one season: the same monotone benefit from concentrating capital in Tier 1 shows up when the two halves are scored in isolation. As in the asymmetric-selection-trust audit, the first half is markedly stronger than the second on every policy (a standing seasonal-strength gap this script does not resolve), but tiering does not widen or narrow that gap materially — it lifts both halves together.

---

## 5. The Pareto frontier

The full 6×6 grid (36 `(tau1, tau2)` cells, ratio range 0.71–5.0) collapses to a single-parameter frontier in the ratio, per §2.1. Selected points, `m12` / `m13`:

| Tier1:Tier2 ratio | `m12` return | `m12` Sharpe | `m12` max DD | `m13` return | `m13` Sharpe | `m13` max DD |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 (= `P0`) | 143.91% | 1.516 | -18.39% | 146.14% | 1.548 | -19.16% |
| 1.40 (= `P1`) | 160.76% | 1.645 | -19.95% | 165.34% | 1.689 | -19.51% |
| 2.00 (= `P2`/`P3`) | 168.98% | 1.742 | -21.31% | 176.55% | 1.798 | -21.00% |
| **2.25** | **169.39%** | 1.766 | -21.69% | 177.92% | 1.826 | -21.26% |
| 2.50 | 168.86% | 1.784 | -21.91% | **178.17%** | 1.848 | -21.37% |
| 3.00 | 166.18% | 1.808 | **-22.04%** | 176.61% | 1.880 | -21.29% |
| 4.00 | 158.26% | 1.830 | -21.71% | 169.83% | 1.914 | -20.63% |
| 5.00 (grid edge) | 149.98% | **1.834** | -21.09% | 161.96% | **1.927** | -19.78% |

Two frontiers, not one:

- **Return** is single-peaked: it rises from ratio 1.0 to an interior maximum near **ratio ≈ 2.0–2.5** (169.39% at 2.25 on `m12`, 178.17% at 2.5 on `m13`) and then *declines* as the ratio climbs further — past that point, Tier 2's turnover has shrunk enough that the loss of its own positive ROI outweighs the gain from concentrating further into Tier 1.
- **Sharpe** (and, on `m13`, `mean_k_risk`-adjusted growth) **keeps rising to the edge of the tested grid** (ratio 5.0, `tau2 = 0.10`, the minimum requested) on both models. The 20%-cap-to-10%-cap range specified for this audit does not contain the point where Sharpe turns over, if it does at all in this book; higher-ratio cells (`tau2 < 0.10`) were not requested and were not run.
- **Max drawdown** worsens from ratio 1.0 to a *local* worst point near ratio 2.5–3.0 (`m12`: -22.04% at 3.0; `m13`: -21.38% at 2.67) and then *improves* (less negative) at more extreme ratios, because Tier 2's shrinking turnover eventually reduces total slate exposure enough to offset the added concentration.
- **Calmar** (return / |max DD|, both computed over the full 100-slate window) is maximized at a substantially gentler ratio than either return or Sharpe: **≈1.2 on `m12`** (Calmar 8.151 at ratio 1.20) and **≈1.4–1.5 on `m13`** (Calmar 8.512 at ratio 1.43) — close to `P1_conservative_tilt`'s 1.4.

This is the basis for recommending `P1` over the return-maximizing `P2`/`P3`: the ratio that maximizes return-per-unit-of-drawdown-pain sits close to `P1`, not at the frontier's return peak.

---

## 6. Concentration & tail risk

| Model | Policy | Max single bet | Max daily exposure | Daily P&L volatility |
|---|---|---:|---:|---:|
| `m12` | `P0_flat_benchmark` | 66.64 | 17.58% | 0.0457 |
| `m12` | `P1_conservative_tilt` | 74.35 | 17.35% | 0.0453 |
| `m12` | `P2_conviction_tilt` / `P3` | 76.00 | 16.54% | 0.0442 |
| `m12` | `P4_four_tier_probe` | 77.56 | 17.14% | 0.0450 |
| `m13`† | `P0_flat_benchmark` | 75.09 | 17.99% | 0.0452 |
| `m13`† | `P1_conservative_tilt` | 82.47 | 17.72% | 0.0450 |
| `m13`† | `P2_conviction_tilt` / `P3` | 83.67 | 16.88% | 0.0441 |
| `m13`† | `P4_four_tier_probe` | 85.90 | 17.49% | 0.0447 |

Concentration shows up asymmetrically. The single largest bet placed on any one day grows monotonically across the tilted policies — roughly +12% to +16% over `P0` by `P2`/`P3` (`m12`: 66.64 → 76.00; `m13`: 75.09 → 83.67), and further still under `P4` — a real increase in single-position size. But **daily portfolio-level exposure and volatility both fall slightly** over the same move (max daily exposure -1.0 to -1.1pp; P&L volatility -2.4% to -3.3%), because total turnover contracts as Tier 2 shrinks faster than Tier 1 grows. The tail-risk cost of tilting is therefore not visible in any single day's numbers — it shows up only in §5's max-drawdown series, which is a path/sequencing statistic over the full 100-slate compounding chain, not a same-day risk measure. This backtest cannot separate "unlucky sequencing of an unchanged single-day risk profile" from "increased correlation risk from concentrating in two 1X2-family legs that settle on the same fixtures" as the drawdown driver; both are consistent with the data shown here.

---

## 7. Production recommendation

1. **Adopt `P1_conservative_tilt` (τ = 0.35 Tier 1 / 0.25 Tier 2) as the new production candidate**, superseding flat asymmetric core. It improves both models' return (+16.85pp / +19.20pp) and Sharpe (+0.129 / +0.141) for a max-drawdown cost of roughly +1.6pp / +0.35pp — a small fraction of the frontier's peak drawdown cost — and sits near each model's Calmar-optimal ratio.
2. **Do not deploy the nominal `P3_aggressive_tilt` table.** It is, under this engine, the *exact same portfolio* as `P2_conviction_tilt`. Shipping it as a distinct "more aggressive" tier invites an operator to believe they have dialed up risk when they have not; if `P2`'s risk/return point is ever wanted, ship `P2`'s numbers and do not maintain `P3` as a separate config.
3. **If growth is prioritized over drawdown discipline, `P2_conviction_tilt` (ratio 2.0) is the predeclared choice closest to the measured return frontier** — within 0.5pp of the peak on `m12` (ratio 2.25, 169.39%), and within 1.6pp of the peak on `m13` (ratio 2.5, 178.17%) — at the cost of roughly 2–3pp of additional max drawdown versus `P0`.
4. **Do not enable the Tier 3 probe** (Under 1.5, Over 3.5). At `tau3 = 0.05` it draws material turnover (≈1.1% of capital, ~200 extra bets per model) for flat-to-negative P&L, reproducing the "all unders" failure from the asymmetric-selection-trust audit at a smaller trust weight.
5. **Specify and log tier trust as a ratio, not as absolute values.** Because `SlateDrawdown` renormalizes to a fixed risk budget whenever it binds (which was every slate tested here — the explicit 20% cap never bound), only the ratio between active tiers has any effect on the realized book. A MatchDay config or ticket that shows raw tau values without their ratio is not showing an operator the information that determines the bet.
6. **Require `m12` convergence as the production evidence base**, as before; treat `m13`'s directionally identical, slightly larger numbers as corroboration, not as license to relax the convergence gate.
7. **Revisit the grid boundary** if a Sharpe-maximizing (rather than return-maximizing or Calmar-balanced) policy is ever wanted: this audit's requested grid (`tau2` down to 0.10) does not contain the point where Sharpe stops improving.

---

## 8. Limitations

- The scale-invariance result (§2.1) is specific to this trust/risk/allocator combination (`SelectionTrust` × `KellyLogUtility` × `NoShrinkage` × `SlateDrawdown`). A different risk model, a per-selection cap, or a shrinkage estimator that is not scale-homogeneous could break it; the finding should not be assumed to generalize to other configurations without re-deriving it.
- The grid sweep covers the exact range specified in the task (`tau1 ∈ [0.25, 0.50]`, `tau2 ∈ [0.10, 0.35]`, step 0.05). Sharpe was still increasing at the tested boundary; no claim is made about behaviour outside that box.
- All policies use closing Betfair prices and idealized backtest execution, not a live point-in-time ladder fill process.
- The five predeclared tier assignments were fixed in the task specification, motivated by the prior audit's directional findings; the temporal split (§4) increases confidence that the *ranking* among them is not a full-period artifact, but it is not a fully nested walk-forward policy-selection design.
- The two models share the same match set and a closely related specification; their agreement is sensitivity replication, not two independent samples.
- `m13` fails the aggregate tail-ESS convergence gate; its numbers are corroborating evidence, not the production basis.
- Tier 3's near-zero result (n = 200/199 bets) is a small sample; a token positive tau on a genuinely near-zero-ROI line is expected to be noisy in either direction, not conclusively harmful, though it did not pay for itself in this backtest.
- Results are specific to the Scottish Lower segment, this price estimator, 2% commission, `SlateDrawdown(23)`, and a 20% cap.

---

## 9. Reproduction and artifacts

Run from artifact-compatible source commit `784c8ea81328760e75498b19d13c2dab762bde8e` with database credentials supplied by `~/.pgpass`:

```bash
julia --project -t 8 eda/eda_multitier_trust.jl
```

Task script:

- `eda/eda_multitier_trust.jl`

Generated files under `eda/results/multitier_trust/`:

| File | Contents |
|---|---|
| `multitier_policy_summary.csv` | full-period bankroll, return, Sharpe, Sortino, Calmar, drawdown, bets, turnover, cap use, mean `k_risk`, max single bet, daily P&L volatility |
| `multitier_policy_windows.csv` | full, first-half, and second-half metrics for every named policy |
| `multitier_policy_daily.csv` | all 1,000 model-policy slate states |
| `multitier_policy_ledger.csv` | selection-level stakes, prices, probabilities, and realized P&L |
| `multitier_selection_summary.csv` | model-policy-tier-direction stake share, P&L, win rate, ROI, and edge |
| `multitier_grid_sweep.csv` | all 72 `(model, tau1, tau2)` grid cells and their headline metrics |
| `multitier_policy_definitions.csv` | explicit trust and tier for every named policy × direction (65 rows) |
| `multitier_build_report.csv` | immutable run hashes and convergence/build gates |

The runner validates that every named policy spans exactly 100 dates, ledger counts and turnover reproduce engine summaries, final bankrolls reconcile, no gated direction receives a stake, and the `(tau1, tau2) = (0.30, 0.30)` grid cell reproduces `P0_flat_benchmark` to `1e-6` on both models.
