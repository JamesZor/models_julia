# Staking Diagnostics — `min_edge`, Market Selection & Kelly Fraction (all Betfair markets)

**Model:** `DCMH_HalfLife_60` · **League:** Ireland · **Benchmark:** Betfair `odds_close`
(vig-included implied prob, as the live `BayesianKelly` uses) · **Staking:** Baker–McHale
`BayesianKelly`. **Posterior:** raw L1 (no L2 — §8 of the L2 research note showed L2 is not the
staking lever). All metrics are per-bet; `G_emp = exp(mean log(1+stake·roi)) − 1` (the
trustworthy realised geometric growth). Walk-forward = strict, fit threshold on past folds only.

> **Headline.** The staking lever is **not** the calibrator and **not** the Kelly fraction — it
> is (1) **market selection** and (2) a **single fixed `min_edge` ≈ 0.03**. Together they take
> pooled realised growth from **negative** to clearly positive, and a fixed 0.03 roughly
> **doubles** out-of-sample growth vs the current `min_edge=0` default. Per-market tuning
> *overfits* and loses to the fixed default.

---

## 1. Coverage & calibration of all 17 joinable markets

17 selections join L1 PPD ↔ Betfair (correct-scores excluded — not modelled). Calibration check
(`mean_pred` vs realised `win_rate`, on the positive-edge bet subset):

Well calibrated within ~2–3pp on almost everything. **The one material miscalibration is BTTS**:
`btts_yes` under-predicts ~7pp, `btts_no` over-predicts ~7pp (mirror images). Mild draw
under-prediction (~2.8pp). This is the bias L2 correctly detects (research note §8e).

---

## 2. Finding 1 — Market selection is the #1 lever

Per-bet `G_emp` at `min_edge=0`, full Kelly, splits the 17 markets cleanly:

| **Positive-edge (BET) — 10 markets** | **Negative-edge (DROP) — 7 markets** |
|---|---|
| under_15 (+0.0094), btts_yes (+0.0084), over_05 (+0.0081), draw (+0.0074), btts_no (+0.0068), under_25 (+0.0065), under_05 (+0.0025), under_35 (+0.0019), over_25 (+0.0014), under_55 (+0.0002) | **home (−0.022)**, under_45 (−0.011), over_55 (−0.0075), away (−0.0078), over_45 (−0.0069), over_35 (−0.0067), over_15 (−0.0037) |

**Pattern:** the model's edge lives in the **unders / BTTS / low-overs** space; it **loses to the
Betfair close on 1X2 (home/away) and high-overs (over_15/35/45/55)**. This independently
re-confirms the user's original 5-market pick (over_25/under_25/under_35/btts_yes/under_15 all
positive) and extends it to ~10.

**Impact:** pooled realised growth across **all 17** markets is **G_emp = −0.0005** (full Kelly,
e=0) despite +10.5 profit / +10% ROI — i.e. the arithmetic profit hides **negative geometric
growth**. Restricting to the **10 good markets flips it to G_emp = +0.0055.** Dropping the
losers is the single biggest improvement available.

---

## 3. Finding 2 — Full Kelly is correct; fractional Kelly does NOT help here

On the 10 good markets (e=0), scaling the Baker–McHale stake by a fraction f:

| Kelly fraction | ROI% | `G_emp` | profit |
|---|---|---|---|
| 1.00 | 15.9 | **0.00549** | 8.98 |
| 0.50 | 15.9 | 0.00367 | 4.49 |
| 0.25 | 15.9 | 0.00207 | 2.24 |
| 0.10 | 15.9 | 0.00089 | 0.90 |

`G_emp` scales **down** ~linearly with f. The stakes are small enough (avg ≈4% bank) to sit in
the near-linear regime where `log(1+s·r) ≈ s·r`, so the variance penalty that normally justifies
fractional Kelly barely bites. **Full Baker–McHale Kelly is already appropriate** — the earlier
*negative* pooled growth was a market-selection artifact, **not** over-betting. (Reconsider only
if stakes grow or edges are suspected over-estimated.)

---

## 4. Finding 3 — `min_edge`: a fixed 0.03 default; per-market tuning overfits

### In-sample sweep (10 good markets, quarter-Kelly shown; ratio holds for any f)
`G_emp` rises monotonically with `min_edge` while profit declines gently:

| min_edge | bets | ROI% | `G_emp` | profit |
|---|---|---|---|---|
| 0.00 | 966 | 15.9 | 0.00207 | 2.24 |
| 0.02 | 626 | 17.4 | 0.00314 | 2.21 |
| **0.03** | 484 | 18.9 | 0.00390 | 2.11 |
| 0.05 | 283 | 24.3 | 0.00608 | 1.93 |
| 0.07 | 162 | 28.3 | 0.00804 | 1.47 |

Higher `min_edge` trades **total profit** (fewer bets) for **per-bet growth/ROI**. A geometric-
growth maximiser wants a positive threshold; absolute-profit wants ~0.

### Walk-forward (strict OOS) — does per-market tuning beat a fixed default?
Full Kelly, 10 good markets, threshold chosen on past folds only:

| strategy | bets | ROI% | `G_emp` | profit |
|---|---|---|---|---|
| fixed `min_edge=0` (current default) | 626 | 19.3 | 0.00729 | 7.06 |
| **fixed `min_edge=0.03`** | 306 | 23.2 | **0.01430** | 6.70 |
| per-market walk-forward-tuned (mean e\*≈0.031) | 338 | 18.4 | 0.00995 | 5.62 |

**A fixed `min_edge=0.03` ≈ doubles OOS geometric growth** (0.0073 → 0.0143) vs the current
`min_edge=0`, while keeping ~95% of the profit. **Per-market walk-forward tuning is *worse* than
the fixed default** (0.0099 < 0.0143, profit 5.62 < 6.70) — it overfits the thin per-market
history even though its mean choice (≈0.031) lands near the right value. **Keep one global
threshold; do not tune per market.**

---

## 5. Recommendations

1. **Bet a curated market set** (~10 unders/BTTS/low-overs); **exclude 1X2 home/away and the
   high-overs** — the model has negative edge vs the Betfair close there. This is the largest,
   most robust gain (pooled `G_emp` −0.0005 → +0.0055).
2. **Set `BayesianKelly(min_edge ≈ 0.03)` as the default** — a single global value, not
   per-market. ≈2× OOS geometric growth for ~5% less profit. Tune the *single* value
   out-of-sample periodically; do **not** fit it per market.
3. **Keep full Baker–McHale Kelly** (fractional Kelly does not help at current stake sizes).
4. **Use L2 for accuracy/monitoring, not growth** — it correctly flags BTTS as ~7pp
   mis-set, valuable as a bias monitor, but it is not the staking lever (research note §8).

## 6. Caveats
- **Small samples:** 39–138 bets/market, ~966 pooled good-market bets, single league. Treat the
  0.03 figure as "a small positive threshold," not a precise optimum; the per-market `best_e`
  values (15–45 bets) are overfit and must be ignored.
- **Market selection is in-sample** (good/bad split on the same data) — re-derive the bettable
  set on a rolling basis before trusting it live; the *direction* (unders/BTTS win, 1X2/high-
  overs lose) is the robust takeaway.
- **Fill/commission unmodelled** — settles at `odds_close` assuming a back-side fill; discount
  realised ROI for ~half-spread + 2–5% Betfair commission (Betfair study §3).
- **`G_emp` treats bets as independent fractions**, not a compounded bankroll sequence — a
  relative proxy, consistent with the `BernoulliGammaHurdle` metric, not a live equity curve.
