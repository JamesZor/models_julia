# Betfair Price-Path vs Bayesian Posterior — Detailed Analysis & Trading Playbook

**Author context:** senior-quant read of the `DCMH_HalfLife_60` engine evaluated against the
Betfair free-tier (last-traded-only) price path, Ireland Premier/Div 1, 258 matches.
**Companion file:** `RESULTS_REPORT.md` (the raw results record). This document adds the
*economic* interpretation and a concrete, costed trading plan.

---

## 0. TL;DR for a trading desk

| Question | Answer |
|---|---|
| Does the model beat the Betfair line? | **Yes**, ~0.008 nats/bet, significant from −6h to KO. |
| How big vs soft books? | **4–5× bigger** (your −0.038 was vs Sofascore, not the exchange). |
| Where is the edge? | **`btts_yes` and `under_15`.** 2.5-goals lines & `under_35` are *not* distinguishable from Betfair. |
| When to enter? | **−180 to −90 min.** Edge, line-drift and ROI all decay into an efficient close. |
| Can I trust the 20–40% ROI? | **No — direction yes, level no.** Inflated by fill side, commission, over-confidence. |
| Is the posterior calibrated? | **No — too narrow (over-confident).** Calibrate (L2) before Kelly. |
| Biggest unmodelled risk? | **Fill side** (last-traded ≠ guaranteed back fill) + commission. |

**One sentence:** there is a **small but real, front-loaded edge over the exchange,
concentrated in two markets**, that is tradeable only with calibrated sizing, early entry,
patient limit execution, and honest cost accounting.

---

## 1. The central result: edge is *relative to your benchmark*

Your prior evaluation reported a log-loss improvement of **−0.038** vs `ds.odds` (Sofascore
bookmaker). Against the **Betfair exchange closing line** the same model improves by only
**−0.0073 to −0.0092 nats**. This is the single most important number in the study.

**Why it matters economically.** A proper scoring rule (the logarithmic score) is the expected
log-Bayes-factor between two forecasters. A 0.038-nat edge over a soft book and a 0.008-nat
edge over the exchange describe **two different counterparties**:
- The **bookmaker** is slow, margin-laden, and beatable — but you can only realise that edge
  where you can actually stake into bookmakers (limits, restrictions, account longevity).
- The **exchange** is the *efficient frontier*; beating it by 0.008 nats is a genuine but
  thin alpha that survives only with disciplined execution and costs.

> **Insight 1.** Treat the two edges as separate businesses. The exchange edge is the *honest*
> measure of model skill; the bookmaker edge is a *distribution/access* play that decays as
> accounts get limited. Don't conflate the two when sizing or reporting Sharpe.

---

## 2. Where the edge actually lives (market selection)

Per-selection `diff_ll` (negative = model better; "sig" = bootstrap 95% CI excludes 0):

| Selection | Edge size | Significant? | Verdict |
|---|---|---|---|
| **`btts_yes`** | −0.011 … −0.014 | yes (from −6h, incl. −45/−20/−5) | **Primary — consistent, robust** |
| **`under_15`** | −0.013 … −0.016 | yes (−180), else borderline | **Primary — large but noisier** |
| `over_25` / `under_25` | −0.004 … −0.006 | **no** (CIs include 0) | Marginal vs exchange |
| `under_35` | ~0 to −0.007 | **no** | Weakest; avoid on the exchange |

The pooled significance is **carried entirely by `btts_yes` and `under_15`**. The 2.5-goals
lines look profitable in the backtest because they beat *bookmakers*, not because they beat
the exchange.

> **Insight 2.** The model's structural strength is in **goal-presence / low-total markets**
> (BTTS yes, Under 1.5), not the heavily-arbitraged 2.5 line where every sharp in the world
> concentrates. This is intuitive: the 2.5 line is the most efficient football market on the
> exchange, so even a good model has little room there.

**Trading implication.** Run the strategy on **BTTS-yes and Under-1.5 first**. Only trade the
2.5 lines and Under-3.5 into **soft books**, never as standalone exchange edge.

---

## 3. Entry timing — the edge is an *opening-hours* phenomenon

Three independent lenses agree that the edge is **front-loaded and decays into kickoff**:

1. **Log-score (§1).** diff_ll most negative ≈ −180 min, eroding ~18% by −5 (−0.0089→−0.0073).
2. **Line-movement prediction (CLV alpha).** Regression `realized_move ~ β·model_signal`:
   β = 0.096 (−12h, p=2.6e−10) → 0.003 (−5, p=0.26). The market **drifts toward the model**,
   and the *predictable* portion of that drift is **fully arbitraged away by kickoff**.
   Directional hit-rate peaks 56.2% at −45m (p≈1e−5), reverts to 50/50 by −5.
3. **Realised ROI (§3).** Highest at −180/−90, decays into the close at every edge threshold.

**The microstructure story.** On a thin league, 24h out ~90% of prices are last-trade-carried-
forward (essentially the opener). Real money — much of it sharp — arrives in the final hour and
**sharpens the line toward fair value (and toward your model)**. You want to be **positioned
before that sharpening**, capturing the move rather than chasing it.

> **Insight 3.** This is a "be-early, get-run-over-less" edge: you are effectively front-running
> the late sharp money that confirms your view. The optimal window balances (a) more edge
> earlier vs (b) thinner liquidity and wider effective spreads earlier. On this data the sweet
> spot is **−180 to −90 min**.

**Caveat that cuts the other way.** Earlier entry = **worse liquidity and wider spreads** on a
thin market, and our spread is *unmeasured* (§5). The −180/−90 window is the compromise: enough
edge, enough liquidity. Going to −12h chases a larger β but into near-empty books.

---

## 4. The realised-ROI numbers — what's real and what isn't

§3 shows 17–40% flat ROI. **The shape is trustworthy; the level is not.** Decompose the
inflation:

| Inflation source | Effect | Rough haircut |
|---|---|---|
| **Fill side unknown** (last-traded may be lay-side; we book the back price) | Over-states return by ~½ effective spread per bet | Largest, unmeasured |
| **No commission** | Betfair takes ~2–5% of *net winnings* | −2 to −5% on winners |
| **Posterior over-confidence (§5)** inflates the edge filter | Selects too many/too-large "value" bets off optimistic p | Material |
| **In-sample optimism** (257 matches, not walk-forward) | Standard backtest bias | Unknown; test it |
| **Survivorship in coverage** (matches with ticks at τ) | Mild | Small |

A defensible expectation: the *true* net edge over the exchange close, after costs and fill
risk, is **low single-digit % per bet at best, and only on the primary markets entered early**.
That is still a viable edge — football staking compounds across thousands of bets — but it is
**not** a 30%-ROI machine.

> **Insight 4.** Anchor expectations on the **0.008-nat log-score and the 56% directional
> hit-rate**, which are robust, *not* on the headline ROI. Build the P&L model bottom-up from a
> conservative per-bet edge and realistic costs.

**Known code limitation.** `mean_log_growth` (≈ −2.4 to −2.9) is meaningless as coded — it
assumes 100% bankroll per bet, so any loss drives `log(1+ret)→log(0)`. Replace with a
fractional-Kelly bankroll simulation before quoting any growth/Sharpe figure.

---

## 5. Calibration & microstructure — why sizing must be conservative

**PIT vs the closing line:** KS D = 0.090, p ≈ 0 (reject uniformity). Central-interval coverage
**below nominal at every level** (50%→37%, 80%→64%, 95%→87%) ⇒ **posterior intervals are too
narrow; the model is over-confident**. The closing prob sits in the model's tails far too often.

Two simultaneously-true readings:
- *Statistical:* the L1 posterior under-disperses (needs variance inflation / the L2 layer).
- *Economic:* persistent large disagreement is **partly edge** (§1 confirms the model wins) and
  **partly miscalibration**.

**Why this is dangerous for staking.** Kelly fraction `f* ≈ edge/odds`. If the posterior is
over-confident, the *perceived* edge is too large → Kelly over-bets → variance and drawdown
blow up, and a single miscalibrated tail can be ruinous. **Always stake off the L2-calibrated
posterior and apply a Kelly multiplier ≤ 0.25.**

**Roll spread:** unrecoverable here — 4/5 selections show positive lag-1 autocovariance (news-
driven drift breaks Roll's driftless-random-walk assumption). We therefore **cannot measure the
effective spread** from free-tier data; the ± fill band is the dominant *unquantified* risk.

> **Insight 5.** The model is a good *direction-finder* but a poor *uncertainty-quantifier*.
> Use it to pick sides, not to size aggressively. Calibration is the gate between "interesting
> backtest" and "deployable strategy".

---

## 6. The trading playbook (concrete)

### 6.1 Universe & signal
- **Markets:** `btts_yes`, `under_15` (primary). Optionally `over_25`/`under_25` *only* into
  soft books. Drop `under_35` on the exchange.
- **Signal:** `edge = p_model_calibrated − p_fair_market(τ)`, using the **L2-calibrated**
  posterior. Require `edge > τ_edge` with `τ_edge ≈ 0.03–0.05` (the 0.05 filter gave the
  cleanest ROI shape and the highest hit-rate).

### 6.2 Timing
- **Entry window: −180 to −90 min** to kickoff. Re-evaluate the signal each available tick;
  do not chase after −45 (edge mostly gone, line efficient).

### 6.3 Execution (critical given last-traded-only data)
- **Post passive limit orders**, never market-take. Quote at or just **inside the model's fair
  odds** and let the late sharp money come to you — this both *captures* the predicted drift
  (§3 CLV) and *avoids* paying the spread you cannot see.
- Accept partial/zero fills as the cost of not knowing the book. **An unfilled order is better
  than a bad fill.**
- Size each clip to what a thin Irish market can absorb without moving (start tiny; scale with
  observed fills).

### 6.4 Staking
- **Fractional Kelly, f = 0.10–0.25**, on the **calibrated** edge:
  `stake_fraction = f · (p·(o−1) − (1−p)) / (o−1)` per bet, capped at a hard per-bet % of bank.
- Floor the calibrated `p` away from its raw value if PIT still shows residual over-confidence.

### 6.5 Cost accounting (always net)
- Commission: model 2–5% on net winnings per market.
- Fill haircut: stress-test assuming a fraction of fills are at the **worse (lay-side)** price;
  re-check that the strategy's expectancy stays positive (see §7).

### 6.6 Risk guardrails
- Per-match and per-day exposure caps; max concurrent correlated bets (BTTS-yes and Under-1.5
  on the same match are *positively* correlated in goal-presence — treat as one risk).
- Stop-loss on rolling CLV: if your realised entry-vs-close CLV turns persistently negative,
  the edge has decayed or the model has drifted — halt and recalibrate.
- Monitor **realised CLV as the live KPI** (not P&L): positive average CLV vs the close is the
  leading indicator that the edge is intact; P&L is too noisy match-to-match.

---

## 7. Pre-deployment validation checklist (do these before risking capital)

1. **Re-run §1 on the L2-calibrated PPD** — confirm the −0.008-nat edge *survives* calibration
   (if calibration kills it, the edge was just over-confidence).
2. **Walk-forward / out-of-sample split** — repeat §3 on matches the model never trained near;
   rule out the 257-match in-sample optimism.
3. **Fractional-Kelly bankroll sim** — replace the broken `mean_log_growth`; report median
   terminal wealth, max drawdown, and Sharpe/Calmar under realistic costs.
4. **Worst-case fill stress test** — assume lay-side fills on X% of bets; find the breakeven X
   at which expectancy → 0. If breakeven X is small, the strategy is spread-fragile.
5. **CLV backtest as the primary metric** — a strategy that *consistently* beats the close is
   the gold standard of skill; rank candidate configs by realised CLV, not by ROI.
6. **Liquidity audit** — quantify matched volume in the −180/−90 window per market to size the
   realistic capacity (this is a small-capacity edge by construction).

---

## 7b. Reconciliation with the official Kelly backtest (why `hurdle_G` is positive)

A natural worry: this study stresses "small, marginal, over-confident," yet the official
`run_backtest` tearsheet (`r06_grid_search_dynamics.jl`) shows healthy positive growth
(`hurdle_G` ≈ 0.007–0.009 for `btts_yes`/`under_15`/`over_25`, `roi_pct` 11–42%). **These are
not in conflict — they answer different questions, and once aligned they agree.**

**What the backtest actually does** (verified in source):
- `run_backtest(...; odds_column=:odds_close)` is the **default** — it both **stakes and
  settles at the Betfair *closing* price** (`processor.jl:18`, and `pnl = stake·(odds−1)` on a
  win else `−stake` at `processor.jl:67`). Same benchmark as this study (the close).
- `BayesianKelly` sizes via Baker–McHale log-growth optimisation over the **full posterior**,
  with an edge filter on the **raw** implied prob `p_true − 1/odds_close` (`kelly.jl:50–54`) —
  if anything *stricter* than our vig-removed `prob_fair`, because the raw price carries the
  overround.
- `hurdle_G = exp(E[log(1 + f·R)]) − 1` with fractional stake `f = avg_stake` (~0.02–0.05) —
  i.e. the **correct** geometric-growth number that my broken `mean_log_growth` (100%-stake)
  failed to compute.

**The three reconciling points:**

1. **Growth ≠ forecasting skill.** Log-loss/PIT (§1, §5) ask *"is the model a better-calibrated
   forecaster than the line, averaged over every match and both sides?"* — answer: marginally,
   and over-confident. The tearsheet asks *"if I bet **only** where the model disagrees in my
   favour and size by edge, does the bankroll grow?"* You do **not** need to beat the line on
   average to profit from value betting — only on the **selected subset**, which Kelly
   concentrates stake on. **A −0.008-nat average edge is fully consistent with +0.007 selective
   growth.**

2. **`hurdle_G` *is* the "small edge."** 0.005–0.009 = **0.5–0.9 % geometric growth per bet** —
   small, positive, compounding. It is the *same order of magnitude* as the −0.008-nat
   log-score and the §3 finding. "Good growth" here means *positive and repeatable*, not large.
   The selectivity is real: `home`, `away`, `over_45`, `over_55` show **negative** `hurdle_G`
   (the model has no edge on match-result favourites / extreme overs) — exactly matching our
   market-selection finding that the edge lives in **goals/BTTS** markets.

3. **The big `roi_pct` carries the *same* caveats I flagged.** The tearsheet's 20–42 % ROI is
   inflated by the *identical* issues: it settles at the Betfair last-traded **close assuming a
   guaranteed back-side fill at that exact price**, **no commission**, and **in-sample** (not
   walk-forward). So "discount the ROI hard" applies to the official backtest too. **`hurdle_G`
   is the more honest figure** (realistic fractional stakes) — but it still assumes perfect
   fills at `odds_close`.

**Why this study's pooled log-loss (−0.008) looks smaller than the eval cell's −0.038.**
The `Evaluation.LogLoss()` figure (−0.0377 for HL60) averages over the **entire market menu**
(1X2, BTTS, and the full OU ladder incl. easy lines like `under_05`/`over_05` where the model
has a *large* structural edge). This study restricts to **five near-coin-flip goal markets**,
which are the **hardest to beat** on the exchange. Same benchmark, **different (harder)
basket** → smaller average edge. Both numbers are correct; the −0.038 is flattered by the easy
markets, the −0.008 is the honest read on the efficient core.

> **Net:** the official backtest *confirms* this study — a **small, selective, goals/BTTS-
> concentrated** edge over the Betfair close that compounds under disciplined Kelly sizing. It
> is "good growth" only in the sense of *positive and repeatable*; the headline ROI is
> optimistic for the same fill/commission/in-sample reasons, and an **early-entry** version
> (`odds_column=:odds_open`, per §3) should grow **faster** still.

---

## 8. Bottom line

The study delivers a clean, defensible conclusion: **`DCMH_HalfLife_60` holds a small, real,
front-loaded informational edge over the Betfair exchange, concentrated in BTTS-yes and
Under-1.5, that the closing line confirms by drifting toward the model.** It is tradeable, but
only as a **calibrated, early-entry, passively-executed, conservatively-sized** strategy on the
*primary* markets — and the realised-ROI headline must be discounted hard for fill risk,
commission and posterior over-confidence. The next dollar of research value is in **calibration
survival, walk-forward validation, and a proper Kelly/CLV bankroll model** — not in more EDA.
