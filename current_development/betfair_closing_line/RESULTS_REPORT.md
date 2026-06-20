# Betfair Price-Path vs Bayesian Posterior — Results & Interpretation

**Model:** `DCMH_HalfLife_60` · **League:** Ireland Premier/Div 1 · **Selections:** `over_25,
under_25, under_35, btts_yes, under_15` · **Sample:** 258 matches, 8,687 panel rows.
**Benchmark:** Betfair free-tier *last-traded* price, vig-removed per market, LOCF-resampled
at each horizon. **Emphasis:** edge / entry timing.

> Headline: the model carries **genuine, statistically significant information beyond the
> Betfair line**, but the edge is **~4–5× smaller against the exchange than against soft
> bookmaker odds**, it is **front-loaded** (largest ~3h–90m out and eroded by kickoff), and the
> L1 posterior is **over-confident** (too narrow), so raw posterior staking would over-bet.

---

## 0. Coverage — the data is a step function, mostly carried forward

LOCF fraction and tick density (per selection × horizon):

| Horizon | LOCF frac (typical) | Mean ticks in window |
|---|---|---|
| −1440 (24h) | 0.86–0.92 | ~0.1 |
| −360 (6h) | 0.49–0.74 | 0.4–1.0 |
| −90 | 0.35–0.65 | 0.5–1.3 |
| −45 | 0.12–0.32 | 1.5–2.9 |
| −5 | 0.01–0.07 | 3.8–6.7 |

**Reading.** This is a thin market. 24h out, ~90% of "prices" are the *last available trade
carried forward* (effectively the opening price); genuine price discovery concentrates in the
final ~45 min. The LOCF design is doing exactly its job — far-horizon rows are real market
state, not fabricated — but interpret early horizons as "the standing price as of τ", not a
fresh two-sided quote. **Caveat:** last-traded ≠ guaranteed fill; we cannot tell a back-side
from a lay-side print, so every price carries an unobserved ± half-spread.

---

## 1. Edge decay (HEADLINE) — model beats the line at every horizon, mildly front-loaded

Pooled `diff_ll = mean(LL_model) − mean(LL_market)` (logarithmic score; **negative = model
better**), with bootstrap 95% CI over matches:

| τ (min) | n | diff_ll | 95% CI | sig? |
|---|---|---|---|---|
| −1440 | 338 | −0.0092 | (−0.023, +0.005) | no |
| −720 | 869 | −0.0064 | (−0.015, +0.002) | no |
| −360 | 1171 | −0.0082 | (−0.015, −0.0007) | **yes** |
| −180 | 1228 | −0.0092 | (−0.017, −0.003) | **yes** |
| −90 | 1254 | −0.0089 | (−0.016, −0.002) | **yes** |
| −45 | 1271 | −0.0089 | (−0.016, −0.002) | **yes** |
| −20 | 1277 | −0.0078 | (−0.015, −0.0007) | **yes** |
| −5 | 1279 | −0.0073 | (−0.014, −0.0004) | **yes (barely)** |

**Reading.**
- The model's log-score beats the Betfair fair line at *every* horizon; the advantage is
  statistically significant from −6h to kickoff. In information terms the edge is a mean
  **log-Bayes-factor of ~0.008 nats per bet** — real but small.
- **Scale check vs the existing metric.** Your `LogLoss` metric reported −0.038 vs `ds.odds`
  (Sofascore bookmaker). Here vs Betfair it is −0.007 to −0.009. This 4–5× gap is **not a
  bug** — it is the central result: the **exchange closing line is far sharper than soft
  bookmaker odds**, and your *honest, tradeable* edge against a sharp counterparty is ~0.008
  nats, not 0.038. The −0.038 number is the edge over the soft book, which you can only
  harvest where you actually bet into soft books.
- The edge is **mildly front-loaded**: most negative ≈ −180 min, eroding ~18% into the close
  (−0.0089 at −45 → −0.0073 at −5). The Betfair line genuinely tightens toward kickoff.

**Per-selection — where the edge actually lives** (`diff_ll`, sig = CI excludes 0):
- **`btts_yes`** — strongest and most consistent: −0.011 to −0.014, significant from −6h and
  at −45/−20/−5. The best edge of the five.
- **`under_15`** — largest magnitude (−0.013 to −0.016 in the −180..−45 band) but noisier
  (significant at −180; thin n at −1440).
- **`over_25` / `under_25`** (complements, mirror each other) — modest (−0.004 to −0.006) and
  **never individually significant** (CIs include 0 at all horizons).
- **`under_35`** — weakest; *positive* (model worse) at −1440/−720 on tiny n, mildly negative
  mid-window, never significant.

> **Important nuance:** the pooled significance is carried by **btts_yes and under_15**. The
> 2.5-goals lines and under_35 are **not distinguishable from the exchange** on their own. The
> backtest growth on those came partly from edge over *bookmakers*, not the Betfair line.

---

## 2. CLV alpha — the model predicts the direction the line moves (early, not late)

OLS `realized_move ~ β·model_signal` per horizon, where `model_signal = p_model − p_fair_τ`
and `realized_move = p_fair_close − p_fair_τ`. **β>0 ⇒ the market drifts toward the model.**

| τ | β | p | dir. hit-rate (binom p) |
|---|---|---|---|
| −1440 | 0.026 | 0.35 | 0.485 (ns) |
| −720 | **0.096** | 2.6e−10 | 0.535 (0.04) |
| −360 | 0.046 | 7.8e−6 | 0.542 (0.004) |
| −180 | 0.032 | 1.9e−4 | 0.545 (0.002) |
| −90 | 0.033 | 3.0e−5 | 0.549 (5e−4) |
| −45 | 0.026 | 2.4e−5 | **0.562 (1.2e−5)** |
| −20 | 0.014 | 1.9e−3 | 0.522 (0.13) |
| −5 | 0.003 | 0.26 | 0.506 (ns) |

**Reading.**
- From **−12h to −20m the model's disagreement with the standing line significantly predicts
  the subsequent move** — i.e., the model holds information the market has not yet priced, and
  the closing line travels toward it. This is the classic "beat-the-close" alpha signature.
- **β decays monotonically to ~0 by kickoff** (0.096 → 0.003): the *predictable* component of
  line movement is largest early and is fully arbitraged away by −5 min. The closing line is
  efficient; the inefficiency is in the hours before.
- Magnitude is small — the market only confirms ~3–10% of the model's signal — so most of the
  model's disagreement is either noise or *persistent* edge the line never closes. The **sign
  and significance**, not the size, are the alpha. The directional hit-rate corroborates,
  peaking 56.2% at −45 (p≈1e−5) and reverting to coin-flip by −5.

---

## 3. Entry-timing P&L — ROI is front-loaded; magnitude is inflated, read with care

Flat-stake filtered bets (enter when `p_model − p_fair_τ > thr`, settle at the snapshot price):

| τ | ROI @0.00 | ROI @0.02 | ROI @0.05 | hit @0.02 | avg_odds |
|---|---|---|---|---|---|
| −1440 | 0.291 | 0.282 | 0.393 | 0.539 | 2.43 |
| −720 | 0.247 | 0.284 | 0.312 | 0.550 | 2.47 |
| −360 | 0.178 | 0.261 | 0.350 | 0.557 | 2.45 |
| **−180** | 0.163 | 0.270 | **0.398** | 0.560 | 2.44 |
| −90 | 0.189 | 0.268 | 0.381 | 0.562 | 2.45 |
| −45 | 0.174 | 0.259 | 0.378 | 0.559 | 2.46 |
| −20 | 0.173 | 0.239 | 0.307 | 0.556 | 2.45 |
| −5 | 0.170 | 0.227 | 0.286 | 0.523 | 2.45 |

**Reading.**
- **Direction is the trustworthy result and it agrees with Stages 1–2: ROI is highest early
  (~−180 to −90) and decays into the close** at every threshold (e.g. @0.05: 0.398 at −180 →
  0.286 at −5). Operating point: **enter ~−180 to −90 min with a moderate edge filter**
  (n≈237–244 bets, ROI≈0.38–0.40 @0.05). Win-rate 0.56–0.59 at avg odds ~2.45 vs breakeven
  1/2.45 = 0.41 — genuinely positive expectancy.
- **The ROI *level* (17–40%) is not real and must be discounted** for three reasons:
  1. **Last-traded fill risk** — we settle at the printed price assuming a back-side fill; a
     fraction of prints are lay-side, so realized returns are over-stated by ~half-spread.
  2. **No commission** — Betfair takes ~2–5% of net winnings (not modelled).
  3. **Posterior over-confidence (§4)** inflates the edge filter, selecting too many/too-large
     bets off the model's own optimistic probabilities.
- **Known code limitation:** `mean_log_growth` (≈ −2.4 to −2.9) is **not interpretable as
  coded** — it assumes 100% bankroll per bet, so any loss drives `log(1+ret)→log(0)≈−6.9`.
  Use ROI for now; a proper fractional-Kelly stake is needed for a real growth number.

---

## 4. Microstructure & calibration — posterior is over-confident; spread unmeasurable

**Roll (1984) effective spread** — degenerate here. 4/5 selections show *positive* lag-1
autocovariance of Δprice (over_25 = +49, outlier-driven), so `2√(−cov)` is undefined; only
under_25 yields a (contaminated) 0.35-in-odds estimate. Roll assumes a driftless random walk
with bid-ask bounce; this sparse, **trending** last-traded series (lineup/news jumps) violates
that, producing positive serial correlation that swamps the bounce. **Conclusion: the
effective spread cannot be reliably recovered from free-tier data** — the ± fill uncertainty
remains the main unquantified risk and partly explains the inflated §3 ROI. (Robustify with a
trimmed/median covariance if pursued, or drop.)

**PIT calibration vs the closing line** — n=1280, **KS D=0.090, p≈0 → reject uniformity.**
Central-interval coverage:

| Nominal | Empirical |
|---|---|
| 50% | 36.6% |
| 80% | 63.6% |
| 95% | 86.5% |

**Reading.** Empirical < nominal at every level ⇒ **the posterior intervals are too narrow —
the L1 model is over-confident / under-disperses.** The closing prob lands in the model's tails
far more often than it should. Two true readings:
1. *Statistical:* the model underestimates its own uncertainty.
2. *Economic:* large, persistent model–market disagreement is partly the **edge** (§1 confirms
   the model wins) and partly **miscalibration**.

The danger is operational: an over-confident posterior **inflates the Stage-3 edge filter and
would make Kelly staking over-bet**. This is precisely what the **L2 calibration layer**
(which widens/shifts the posterior) is for — apply it *before* staking.

---

## Synthesis & recommendations

1. **The model has real alpha over the sharp exchange line** (~0.008 nats/bet, significant
   from −6h), concentrated in **btts_yes and under_15**. The 2.5-goals lines and under_35 are
   **not distinguishable from Betfair** — their backtest value is edge over *bookmakers*.
2. **Enter early.** Stages 1–3 agree: edge, predictable line movement, and ROI are all
   front-loaded and decay into an efficient close. **Operating window ≈ −180 to −90 min.**
3. **Bet into soft books where possible.** Your edge is 4–5× larger vs bookmaker odds than vs
   Betfair; the exchange is the harder benchmark.
4. **Calibrate before staking.** PIT shows over-confidence — run the L2 layer and size with
   fractional Kelly on the *calibrated* posterior, not the raw one. Do **not** trust the §3 ROI
   level; trust its shape.
5. **Respect the data limits.** Last-traded only ⇒ unknown fill side and unmeasurable spread;
   discount realised edge for ~half-spread + ~2–5% commission.

### Follow-ups worth doing
- Replace `mean_log_growth` with a fractional-Kelly bankroll simulation (the real "when to
  enter" growth curve).
- Re-run Stage 1 with the **L2-calibrated** PPD and confirm the edge survives calibration.
- Out-of-sample / walk-forward split to rule out the 257-match in-sample optimism in §3.
- Robust Roll (trimmed covariance) or abandon spread estimation; instead bound fill risk by
  assuming worst-case lay-side fills and re-checking §3 ROI sign.
