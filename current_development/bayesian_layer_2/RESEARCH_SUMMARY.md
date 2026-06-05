# Research Summary — L2 Calibration, Staking & L3 Regime Layer (2026-06)

**Scope:** model `DCMH_HalfLife_60`, Ireland, Betfair `odds_close`, Baker–McHale `BayesianKelly`.
**Question that started it:** can a Layer-2 / Layer-3 layer improve the system's betting returns?
**One-line answer:** **No new model layer helped; disciplined execution did.** The wins were
*market selection*, a fixed `min_edge`, and a *contrarian stake tilt* — not L2 calibration, not a
prediction blend, not a regime gate. The L1 model is genuinely good on a specific market set.

Detailed companions: `docs/l2_bayesian_calibration_research.md`,
`docs/l3_meta_model_research.md` (§9–10), `MIN_EDGE_STAKING_REPORT.md`,
`current_development/betfair_closing_line/RESULTS_REPORT.md`.

---

## 1. The validated staking stack (what to actually do)

```
1. CURATE markets      → bet only where L1 has edge vs the close:
                         unders / BTTS / low-overs (≈10 selections).
                         DROP 1X2 (home/away) and high-overs — negative edge there.
2. min_edge ≈ 0.03     → single GLOBAL value (not per-market). ≈2× OOS geometric growth.
3. Full Baker–McHale   → fractional Kelly does NOT help at these stake sizes (linear regime).
4. ×contrarian tilt    → size UP on recently-cold markets, DOWN on hot (β≈1, clamp [0.3,2.0]).
   (β≈1)                 +26% G_emp, +9% profit, volume preserved.
```

Cumulative OOS effect on the curated set: pooled `G_emp` from **−0.0005 (all 17 markets, no
discipline)** → **+0.0055 (curation)** → **~0.0073 (min_edge floor)** → **~0.0092 (contrarian
tilt)**; ROI ~19% → ~31%. All walk-forward, model out-of-sample.

---

## 2. What we tested, and the verdict on each

| Idea | Tested | Verdict |
|---|---|---|
| **Bayesian L2 calibration** (shift posterior by a distribution) | ✅ built, verified | Maths correct (shift-by-distribution = 1-D **convolution**, variances add). But **doesn't improve staking** — McHale Kelly already integrates the L1 posterior, so widening is redundant. |
| **L2 corrects bias** | ✅ | L1 is already mean-calibrated (±2–3pp) except **BTTS (+5.5pp)**, which L2 *does* correctly fix. Value = a **bias monitor**, not a growth lever. |
| **Fractional shift λ∈[0,1]** | ✅ grid | Forecast log-score has an interior optimum (λ≈0.5 on btts); **staking growth wants λ≈0**. The two objectives disagree. |
| **Market selection** | ✅ | **#1 lever.** Unders/BTTS/low-overs win; 1X2/high-overs lose vs the close. Flips pooled growth negative→positive. |
| **min_edge tuning** | ✅ walk-forward | Fixed **0.03 ≈ doubles** OOS growth. **Per-market tuning overfits** and loses to the fixed default. |
| **Kelly fraction** | ✅ | Full Kelly is right; fractional just scales growth down (stakes too small to need it). |
| **CLV as health signal** | ✅ | **Fails** — corr with realised growth = −0.02. Free-tier open prices too stale, and a close-entry strategy has no later line to beat. |
| **Momentum regime-gate** (bet hot markets) | ✅ walk-forward | **Counterproductive** — performance-chases; more conservative = worse. Loses to static curation. |
| **Contrarian tilt** (bet cold markets bigger) | ✅ walk-forward | **Works.** Per-market performance is mean-reverting (lag-1 autocorr −0.16); cold bets out-grow hot ~4× (G_emp 0.017 vs 0.004). Tilt lifts growth +26%, keeps volume. |

**Net:** more *model* machinery (L2 shifts, prediction blends, regime gates) added little or hurt;
**execution discipline** (which markets, what threshold, what stake, contrarian sizing) added the
returns. Simpler-but-disciplined beat more-layers here.

---

## 3. Is the model good enough?

**Short answer: yes — good enough to trade carefully, on a curated market set, with small stakes,
while you gather more data. Not good enough to scale aggressively or bet indiscriminately yet.**

### The case that it's good
- **Genuinely out-of-sample.** Biweekly walk-forward folds, model re-trained each fold — the
  returns above are not in-sample fit.
- **Beats a sharp benchmark.** L1 log-score beats the **Betfair closing line** by ~0.008 nats/bet
  on the target markets, statistically significant from ~6h out. Beating the *exchange close* (not
  just soft books) is the hard test, and it passes.
- **Well-calibrated.** Mean predictions within ±2–3pp of outcomes on almost all markets.
- **Positive realised growth** on the curated set after disciplined staking (`G_emp` ~0.007–0.009,
  ROI ~19–31% at close prices).

### The case for caution
- **The edge is small.** ~0.008 nats/bet vs the close (the larger −0.038 was vs *soft books*, only
  harvestable where you can bet them). Small edges are fragile.
- **It only works on some markets.** It has *negative* edge vs the close on 1X2 and high-overs —
  betting those loses money. Discipline is mandatory, not optional.
- **One thin league, small samples.** ~40–140 bets/market, single competition (Ireland). The
  headline numbers (e.g. 43.6% contrarian cold-ROI) are high-variance and partly small-sample luck.
- **Execution realism unmodelled.** Settles at `odds_close` assuming a back-side fill; discount
  realised edge for ~½-spread + 2–5% Betfair commission. On a thin market, fills are the real risk.
- **Posterior is over-confident** (PIT: 95%→87% coverage) — benign because McHale Kelly shrinks for
  it, but a reminder the model knows less than it thinks.

### Verdict
The honest read: **the model is the *real* part of this system — the alpha exists and survives a
sharp benchmark.** The binding constraints now are **not modelling** (more layers didn't help) but
**(a) execution realism, (b) breadth of validation, and (c) edge size.** So:

- **Do**: paper-trade / micro-stake the curated stack live, log realised fills vs `odds_close`,
  and watch whether the edge survives commission and slippage.
- **Don't**: scale stakes, or bet the un-curated market menu, on the strength of one league.
- **Can you make money?** Plausibly yes, *net of costs, on the curated markets, at modest stakes* —
  but the margin is thin enough that fills/commission could erase it. That's exactly what
  micro-stake live testing is for. Treat it as a promising edge to *prove out*, not a proven ATM.

---

## 4. Highest-value next steps (validation, not new layers)

1. **More segments.** Re-run the market-selection split + cold/hot decomposition on other leagues
   (and pooled). Confirm "unders/BTTS win, 1X2/high-overs lose" and the −0.16 reversion **generalise**
   — this is the single most important check before trusting any of it live.
2. **More models.** Repeat the staking diagnostics on the other half-life configs (14/30/120) and
   the xG/copula engines — is the curated-market edge a property of *this* model or the *approach*?
3. **Fill realism.** Join the new live order-book capture (once enough data) to estimate true fill
   prices vs `odds_close`; re-deflate every ROI/`G_emp` number here by the measured slippage +
   commission. This is the number that decides whether it's actually profitable.
4. **Lock the production config.** Whitelist the curated markets, set `min_edge≈0.03`, full Kelly,
   optional contrarian tilt (β≈1) — as a single reproducible staking config feeding `run_backtest`.

> The thread's conclusion: **stop adding layers, start validating breadth and execution.** The model
> earns its place; the open questions are whether the edge holds across segments and survives real
> fills — both answerable with data you can now gather.

---

## 5. Prototype code (this folder)
- `src/l01_bayes_calib.jl` — Bayesian Laplace L2 calibrator (global/team shift, convolution, PIT).
- `src/l02_contrarian_tilt.jl` — contrarian stake tilt + cold/hot decomposition (validated).
- `MIN_EDGE_STAKING_REPORT.md` — full market-by-market staking diagnostics.
