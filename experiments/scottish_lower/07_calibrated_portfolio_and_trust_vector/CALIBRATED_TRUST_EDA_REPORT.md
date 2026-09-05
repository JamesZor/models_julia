# Calibrated Portfolio Forensics, Line Pruning and the Directional Trust Vector

**Scottish Lower (Championship / League One / League Two, 24/25 + 25/26).**
Run on `mcmc-beast`, 16 threads, 2026-09-05, branch `feat/modernize-calibration-layer2`,
commit `e6f61475`. Work package: [`WORK_PACKAGE_PROMPT.md`](WORK_PACKAGE_PROMPT.md).

Every number here is read off `trajectory.bets` or `PortfolioSummary` — the bets the
backtest actually struck, at the price it actually paid. Nothing is re-derived from the
latents, so this report and the portfolio summaries cannot disagree except through a bug.
Gate 1 (§2) asserts that.

---

## 0. Executive summary

Four questions were asked. Three of them have clean answers and one of them has a
clean *negative* answer that contradicts the premise the work package was written on.

1. **Over 2.5 is not rehabilitated.** The published figure the brief cites — `+14.3%`
   Kelly ROI on `m12` under `:pool_mean` anchoring — reproduces here **exactly** (38
   bets, `+14.32%`, `+7.10%` flat). Split at 2025-05-03 it is `+34.79%` on 23
   selection-window bets and **`−25.29%` on 15 evaluation-window bets**. Adding Over
   2.5 to the staked basket wins **24 of 24** paired cells in the selection window and
   **3 of 24** out of sample. The direction is an in-sample artefact and the gate on it
   should stay shut. **(§3, §7.2)**
2. **A different direction is rehabilitated: Over 1.5.** Adding it at tier 2 wins
   **24 of 24 paired cells in the selection window and 24 of 24 out of sample**, on both
   models, all four containers and all three risk ladders, worth `+3.49` points of
   in-sample and `+4.37` points of out-of-sample return and `+0.111` of out-of-sample
   Sharpe. It is the only basket change in the study that survives the split. **(§7.2)**
3. **The trust question is not a trust question.** `SlateDrawdown` makes absolute trust
   inert wherever its bisected `k` is strictly inside `(0,1)` — `t1 = 0.50`, `0.70` and
   `1.00` at ratio 1.4 give **bit-identical** portfolios. The canonical `0.35` sits
   *below* that threshold, where `k` pins at 1 on 20–29% of slates and the risk model is
   doing nothing. Raising trust alone buys ~nothing; lowering λ alone saturates. The two
   are one two-dimensional knob and the production point `(0.35, 23)` is in its dead
   corner. **(§5)**
4. **The risk budget is the whole story, and it only pays on a calibrated container.**
   Moving to `(t1 = 1.00, λ = 8)` on the inverse container takes `m12` from `+151.52%`
   to **`+264.99%`** and `m05` from `+130.15%` to **`+277.95%`** at a drawdown 2.5 points
   deeper than production's, and out of sample from `+31.77 / +30.09%` to
   **`+42.07 / +62.14%`**. Calibration ALONE, at the production budget, *loses* 65–85
   points. The three moves are strongly super-additive because what calibration buys is
   drawdown headroom, and only the risk ladder can spend it. **(§6, §8)**

**Gate 3 fails on every deployable candidate**, and §8.3 argues the threshold is
mis-specified rather than the strategy inadequate.

**Recommendation (§9): change the container, add one direction, and re-point the risk
ladder — all three, or none of them.**

---

## 1. What moved under the old verdicts

| | `MARKET_LINE_EDA_REPORT.md` / `MULTITIER_TRUST_REPORT.md` | this study |
|---|---|---|
| container | raw posterior latents | generative-rate calibrated (`src/Calibration/`) |
| price instant | Betfair close (TWA over `[−20 min, 0]`) | **T−25**, the start of MatchDay's execution band |
| book | 6 markets / 13 directions | 13 for forensics, 11 for staking |
| verdict it produced | prune to `1X2 + O/U 2.5`; trust 0.35 : 0.25 | §9 |

Both premises moved, and neither move is small: the calibration stream measured the same
strategies returning **22–38 points more** at T−25 than at the close
([`calibration_generative_eda/README.md`](../../../current_development/calibration_generative_eda/README.md)
§7.2), and calibration shrinks every edge toward the market by construction. A verdict
fitted under `(raw, close)` is not evidence about `(calibrated, T−25)`.

### 1.1 The book this study can actually trade

Before any ROI table, the coverage. A direction quoted on 76 fixtures and one quoted on
545 produce columns of the same width and only one of them is evidence.

| ladder | rows | fixtures | gate-season fixtures | coverage of the 710 | median staleness | median overround |
|---|---:|---:|---:|---:|---:|---:|
| 1X2 | 4,173 | 1,391 | **545** | **76.8%** | 4.1 min | 1.0014 |
| O/U 2.5 | 1,946 | 973 | 347 | 48.9% | 9.4 | 1.0021 |
| BTTS | 1,020 | 510 | 190 | 26.8% | 14.2 | 1.0015 |
| O/U 3.5 | 1,034 | 517 | 179 | 25.2% | 14.5 | 1.0006 |
| O/U 1.5 | 1,014 | 507 | 166 | 23.4% | 14.4 | 1.0018 |
| **O/U 0.5** | 624 | 312 | **76** | **10.7%** | 17.3 | 1.0005 |

The T−25 book itself reproduces the published one exactly: 10,373 rows, 1,572 fixtures,
median staleness 8 minutes, overround 1.0015, and the market-rate inversion accepting
**580 of 710** gate-season fixtures — 94.9% of the fixtures that had a book at all.

---

## 2. Method, and what would make it wrong

**The forensic lens is `FlatTrust(1.0)` over all 13 directions.** A direction gated to
zero strikes no bets and therefore has no economics to measure; to ask whether a gate
should be lifted, the allocator has to be allowed to stake everything. This is a
*measurement* policy, not a deployable one, and it is the same lens
`MARKET_LINE_EDA_REPORT.md` §1 used — so its table and §3's differ in the container and
the price instant and in nothing else.

**Stakes are fractions of a moving bankroll.** `stake` and `pnl` are fractions of the
bankroll at their own slate, so summing them raw across a compounding backtest adds
different units. Every currency figure is rescaled by the slate's opening bankroll first.

**Removing a line is not subtracting its P&L.** Kelly re-solves over what remains and the
exposure cap binds differently, so the counterfactual is a re-simulation — §7, not §3.

**Selecting and scoring on the same data is bias.** Every pruning verdict and every tuned
parameter is fitted on slates up to 2025-05-03 and reported on the 50 slates after it.
Anything chosen with sight of the evaluation window is labelled `NOT DEPLOYABLE` where it
appears.

### 2.1 Gate 1 — ledger accounting invariants

All eight forensic simulations pass. Non-finite values: **0**. Worst disagreement between
`sum(ledger.stake)` and `trajectory.total_stake`: **2.66e-15**. Worst violation of
`pnl == stake × payoff`, bet by bet: **0.00e+00** — exact.

That third one is load-bearing: if it holds, "Kelly ROI" is realised P&L over realised
turnover and cannot disagree with `PortfolioSummary.roi` except through a stated
weighting choice.

### 2.2 A metric in the work package that does not survive the split

Gate 2 as briefed reads `capital_efficiency >= 0.25`, where capital efficiency is a
line's Kelly ROI over the whole book's **in the same window**. Over the 49 selection-window
slates the book's own ROI is a solid `+5` to `+7%` and the ratio is informative. Over the
50 evaluation slates it collapses toward zero, and the ratio does with it: on the raw
container a `+44.61%` line divided by the book's `+0.70%` reads as efficiency **63.29**,
and on the inverse containers the book's out-of-sample ROI is not positive at all, so the
ratio is undefined for **every** direction and the 0.25 threshold admits or refuses on the
denominator rather than on the line.

This suite therefore reports **both**: the work package's same-window ratio (`eff(sw)` in
the runner's tables, `capital_efficiency` in the CSVs) and an **anchored** version
(`eff*`, `capital_efficiency_anchored`) that divides by the **selection window's** book
ROI — a number known at the split date, so it is stable and still legitimately
out-of-sample. Gate 2 is tested on the anchored version. This is a deviation from the
brief and it is the only one.

---

## 3. Line forensics — 13 directions, raw against calibrated

Pooled over both models, full period, flat trust. `Calib` is empirical win rate minus mean
predicted probability, so positive means the model UNDER-rates the selection.

### 3.1 Ladder level

| ladder | container | bets | Kelly ROI % | flat ROI % | capital % | efficiency |
|---|---|---:|---:|---:|---:|---:|
| 1X2 | raw | 1,854 | +8.13 | +5.02 | 63.75 | 1.44 |
| 1X2 | **inv_anch** | 1,585 | **+10.98** | +4.91 | 58.00 | **1.82** |
| O/U 2.5 | raw | 537 | +22.41 | +11.80 | 14.04 | 3.96 |
| O/U 2.5 | **inv_anch** | 419 | **+29.08** | +12.93 | 15.26 | **4.83** |
| O/U 1.5 | raw | 247 | +2.06 | −1.75 | 4.83 | 0.37 |
| O/U 1.5 | **inv_anch** | 199 | **+5.59** | −2.55 | 5.54 | **0.93** |
| BTTS | raw | 276 | −8.94 | +1.92 | 6.49 | −1.58 |
| BTTS | inv_anch | 225 | **−18.90** | −7.71 | 7.68 | −3.14 |
| O/U 3.5 | raw | 269 | −18.15 | −13.31 | 8.54 | −3.21 |
| O/U 3.5 | inv_anch | 215 | **−25.99** | −18.07 | 9.70 | −4.32 |
| O/U 0.5 | raw | 109 | −27.65 | −71.27 | 2.34 | −4.89 |
| O/U 0.5 | inv_anch | 100 | −29.44 | −63.31 | 3.82 | −4.89 |

Calibration **sharpens the ordering rather than reversing it**: the three ladders that
paid on raw latents pay more, and the three that lost lose more. It is not a repair
mechanism for a broken market; it is a magnifier of a ranking that was already there.

### 3.2 Direction level, and where calibration actually helps

Pooled, full period, `raw → inv_anch` Kelly ROI:

| direction | raw | inv_anch | Δ | reading |
|---|---:|---:|---:|---|
| **1X2 draw** | +7.87 | **+21.83** | **+13.96** | the largest gain in the study |
| 1X2 home | +9.62 | +13.28 | +3.66 | |
| **Under 2.5** | +27.55 | **+33.25** | +5.70 | still the alpha, by a distance |
| Over 1.5 | +11.99 | +16.61 | +4.62 | |
| 1X2 away | +6.99 | +5.85 | −1.15 | |
| Over 2.5 | +5.99 | +4.87 | −1.12 | **not** repaired at 13 directions |
| Over 3.5 | +6.85 | −4.60 | −11.45 | |
| BTTS yes | −18.53 | **−43.11** | −24.58 | worse, decisively |
| Under 0.5 | −100.00 | −100.00 | 0.00 | see §3.3 |

**The Draw is what calibration buys.** On the selection window alone the move is `−2.43%`
→ `+28.79%`, a `+31`-point improvement against `+9` on Home and `+2` on Away. The
mechanism is legible: pooling the model's log-rates with the market's compresses
supremacy, which raises `P(draw)`, and the raw model was systematically under-pricing the
draw (calibration bias `+0.0081` raw, `−0.0043` calibrated).

### 3.3 O/U 0.5 — the exclusion holds, and it is not close

`Under 0.5` struck 74 bets on the raw container and won **zero** of them, at a mean price
of 17.81. The same on `inv` (54 bets) and `inv_anch` (62 bets): zero wins. `std` managed
one win in 68. `l2_tradeable_markets`'s exclusion
of this ladder is correct and this study reproduces the reason rather than inheriting it:
the ladder is quoted on **10.7%** of gate-season fixtures (§1.1), the bets it does produce
are extreme longshots, and no container changes that. Recommendation: leave it out.

### 3.4 The parity check — the published Over 2.5 figure, and what the split does to it

The work package's premise, rebuilt exactly: `m12` alone, `inv_anch`, flat trust, the
**11-direction** tradeable book, whole period.

| window | bets | Kelly ROI % | flat ROI % |
|---|---:|---:|---:|
| **full** | **38** | **+14.32** | **+7.10** |
| *published, README §8.9* | *38* | *+14.32* | *+7.10* |
| selection (to 2025-05-03) | 23 | **+34.79** | −8.53 |
| evaluation (after) | 15 | **−25.29** | +31.06 |

**Reproduces to every printed digit, and does not survive the split.** §8.9 never split
it — it was a mechanism demonstration for a dispersion scheme, and the work package read
it as a portfolio claim. On 38 bets it could not have been anything else.

### 3.5 Cross-model agreement

Gate-2 verdicts, per direction, `m12` against `m05`:

| container | directions agreeing | passing on **both** models |
|---|---:|---|
| raw | 11/13 | none |
| inv | **12/13** | `1X2 draw`, `O/U 1.5 over` |
| std | 10/13 | `O/U 1.5 over` |
| inv_anch | **12/13** | `1X2 draw`, `O/U 1.5 over` |

The calibrated containers agree across models *more* than the raw one does, which is what
you would expect of a transform that removes model-specific location error. The two
directions that pass on both models are exactly the two §7 finds worth acting on.

---

## 4. Two structural identities, checked rather than cited

Both are load-bearing for §5–§6's interpretation, and both are asserted on `m12 raw`
inside the runner.

**Trust and the Kelly fraction are the same knob.** `FractionalKelly(f)` returns a
constant `k_shrink = f` and `stake_slate` applies trust and `k_shrink` as two successive
scalar multiplications. Measured: `FlatTrust(0.50) × FractionalKelly(0.30)` and
`FlatTrust(1.00) × FractionalKelly(0.15)` give `+111.702504%` and `+111.702504%`, 1,592
bets each — **identical to every digit**. README §8.7's `(λ, Kelly)` surface is therefore a
strict subset of this study's `(λ, trust)` grid, reached at higher trust rather than at a
higher Kelly fraction, and there is no separate Kelly axis to sweep.

**Scale invariance is a property of the regime, not of the allocator.**

| pair | returns | mean `k` | identical? |
|---|---|---:|---|
| `t1 = 0.35` vs `t1 = 0.70`, ratio 1.4 | `+151.520460%` vs `+150.983768%` | 0.902 / 0.462 | **no** |
| `t1 = 0.50` vs `t1 = 1.00`, ratio 1.4 | `+150.983768%` vs `+150.983768%` | 0.643 / 0.326 | **yes, exactly** |

`MULTITIER_TRUST_REPORT.md` §2.1's invariance holds wherever `k` is strictly inside
`(0,1)` and can absorb a uniform rescale. It fails **below** the level at which the
constraint starts to bind, because there `k` is pinned at 1 and there is nothing to absorb
with. Where the canonical 0.35 falls relative to that threshold is exactly Hypothesis 2 —
and §5 reads it off.

---

## 5. H2 — the trust regime map

Ratio 1.4, λ = 23, cap 0.25. `k pinned` is the fraction of slates on which
`SlateDrawdown`'s bisected `k` reached 1, i.e. the risk model stopped binding.

`m12`:

| container | `t1` | return % | Sharpe | MDD % | mean `k` | `k` pinned | OOS ret % | OOS Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.30 | +140.42 | 1.625 | −15.30 | 0.980 | **0.64** | +31.29 | 1.007 |
| raw | **0.35** | **+151.52** | 1.592 | −16.15 | 0.902 | **0.20** | +31.77 | 0.941 |
| raw | 0.50 | +150.98 | 1.576 | −16.36 | 0.643 | 0.01 | +31.37 | 0.920 |
| raw | 0.70 | +150.98 | 1.576 | −16.36 | 0.462 | 0.01 | +31.37 | 0.920 |
| raw | 1.00 | +150.98 | 1.576 | −16.36 | 0.326 | 0.01 | +31.37 | 0.920 |
| inv_anch | 0.30 | +62.43 | 1.821 | −6.93 | 0.978 | 0.67 | +15.49 | 1.100 |
| inv_anch | **0.35** | +66.82 | 1.775 | −7.95 | 0.904 | 0.29 | +15.38 | 1.008 |
| inv_anch | 0.50 → 1.00 | +67.78 | 1.778 | −8.02 | 0.651 → 0.340 | 0.03 | +15.82 | 1.022 |

`m05` reproduces the same shape row for row.

**The answer to H2 as posed is no.** The premise — "keeping trust at 0.35 causes excessive
under-betting" — is false at λ = 23. Trust is flat from 0.50 upward (identical to the
last digit) and 0.35 is *marginally the best* of the levels tested, because it sits just
under the binding threshold. There is no trust optimum in `[0.20, 1.00]` to find at the
production risk setting: **the parameter is not live there.**

What is true is the thing H2 was reaching for. The canonical level is in a dead corner —
too high to gain anything from raising it, too low for the risk model to bind cleanly.
Getting anything out of trust requires moving λ at the same time, which is §6.

---

## 6. H3 — the risk ladder, its ceiling, and the frontier

### 6.1 The ceiling, located at the canonical trust level

Max drawdown against λ. A row that is **flat** from some λ downward has hit `k = 1`: the
risk model is inert from there on and the last distinct value is that container's deepest
attainable drawdown at that trust.

`m12`, ratio 1.4, cap 0.25:

| container | `t1` | λ=8 | λ=10 | λ=12 | λ=15 | λ=18 | λ=20 | λ=23 | λ=28 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | **0.35** | **−19.29** | −19.29 | −19.29 | −19.29 | −19.06 | −17.57 | −16.15 | −13.65 |
| raw | 1.00 | −39.53 | −33.91 | −30.33 | −23.97 | −20.42 | −18.58 | −16.36 | −13.65 |
| inv | **0.35** | **−7.85** | −7.85 | −7.85 | −7.85 | −7.85 | −7.85 | −7.76 | −6.50 |
| inv | 1.00 | −20.45 | −16.77 | −14.24 | −11.64 | −9.85 | −8.93 | −7.83 | −6.50 |
| inv_anch | **0.35** | **−8.05** | −8.05 | −8.05 | −8.05 | −8.05 | −8.05 | −7.95 | −6.66 |
| inv_anch | 1.00 | −20.79 | −17.13 | −14.59 | −11.93 | −10.09 | −9.15 | −8.02 | −6.66 |

**At the canonical trust level every container's λ row is flat from λ = 15 down.** The
production policy on a calibrated container cannot spend more than **−8.05%** of drawdown
no matter how λ is set — against a 20% risk budget. This is README §8.8's ceiling, now
located at the level production actually runs. And at `t1 = 1.00` the ceiling disappears:
the row is monotone all the way to λ = 8 and the whole budget becomes reachable.

**λ and trust are not two parameters. They are one two-dimensional knob**, and the two
one-dimensional sweeps that have been run on it before — §8.8's λ-only ladder and
§2.1's trust-only ladder — each found the other one's axis inert, which is exactly what a
two-dimensional knob looks like from either edge.

### 6.2 The frontier, chosen honestly

Best **selection-window** return whose **selection-window** drawdown stayed inside the
budget, then scored on the evaluation window. Budget −18%.

| model | container | trust | λ | cap | IS ret % | IS MDD | **OOS ret %** | OOS Sharpe | OOS MDD |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| m12 | raw | `flat_0.40` | 18 | 0.25 | +133.50 | −17.45 | +6.11 | 0.155 | −28.96 |
| m12 | inv | `tier_1.00_r1.4` | **8** | 0.25 | +153.69 | −15.86 | **+38.01** | 0.855 | −20.45 |
| m12 | inv_anch | `tier_1.00_r1.4` | **8** | 0.25 | +156.98 | −15.90 | **+38.25** | 0.848 | −20.79 |
| m12 | std | `flat_0.80` | 8 | 0.20 | +152.85 | −17.66 | +26.95 | 0.558 | −24.51 |
| m05 | raw | `tier_0.70_r1.4` | 15 | 0.25 | +128.29 | −17.64 | +42.04 | 0.833 | −24.27 |
| m05 | inv | `tier_1.00_r1.4` | **8** | 0.25 | +129.03 | −13.15 | **+51.71** | 1.214 | −18.27 |
| m05 | inv_anch | `tier_1.00_r1.4` | **8** | 0.25 | +132.02 | −13.20 | **+52.83** | 1.212 | −18.43 |
| m05 | std | `flat_0.80` | 10 | 0.25 | +101.81 | −16.59 | +14.19 | 0.348 | −28.12 |

**`(t1 = 1.00, ratio = 1.4, λ = 8, cap = 0.25)` is chosen independently on both models'
selection windows, on both inverse containers.** That is four independent selections
landing on one cell, and H4's strongest form: transferring each model's own pick to the
other model changes nothing at all on the inverse containers, while on `raw` the transfer
moves the out-of-sample return from `+6.11%` to `+44.13%` depending on which model fitted
it. The calibrated containers agree about the risk setting; the raw one does not.

Note also what the raw container's out-of-sample drawdowns do here: `−28.96%`, `−24.27%`.
A raw arm pushed to the same in-sample budget **overshoots badly** out of sample. The
calibrated arms overshoot too (−20.45%, −18.43%) but by a fraction as much.

### 6.3 A selection-window drawdown constraint is not a risk limit

Every arm in §6.2 was chosen subject to an in-sample drawdown inside −18% and **every one
of them breached it out of sample**, by 0.3 to 11 points. This is worth stating plainly
because it is the single most likely way to misread this report: a backtested drawdown
bounds the window it was measured on and nothing else. §9's conservative deployment point
exists for that reason.

---

## 7. The adjudicator — re-simulation, basket by basket

`MARKET_LINE_EDA_REPORT.md` §5.1 is the precedent. Its per-line rule selected a basket
that finished **last** of every configuration tested out of sample, below betting 1X2
alone. The forensic table was right about why each line paid and wrong about which basket
to hold, because Kelly re-solves over whatever remains. So §3's verdicts do not decide
anything; this section does.

Eight baskets × three risk ladders × four containers × two models = 192 simulations.

### 7.1 The basket at the production ladder

`m05`, `inv_anch`, `t1 = 0.35, ratio 1.4, λ 23, cap 0.25`:

| basket | bets | return % | Sharpe | MDD % | Calmar | OOS ret % | OOS Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0_canonical` | 956 | +65.40 | 1.944 | −6.92 | 9.45 | +19.26 | 1.353 |
| **`B2_plus_over15`** | 989 | **+70.47** | **2.055** | −6.92 | **10.18** | **+22.28** | **1.542** |
| `B1_draw_promoted` | 956 | +68.01 | 1.917 | −7.21 | 9.44 | +20.39 | 1.331 |
| `B3_plus_over25` | 998 | +65.96 | 1.932 | −6.94 | 9.50 | +18.25 | 1.280 |
| `B7_x1x2_ou25` | 1,038 | +65.41 | 1.861 | −7.32 | 8.94 | +21.91 | 1.458 |
| `B6_flat11` (unpruned) | 1,330 | +42.48 | 1.209 | −12.53 | 3.39 | +6.36 | 0.413 |
| `B5_is_keep` (fitted rule) | 789 | +37.56 | 1.115 | −12.20 | 3.08 | **−1.06** | −0.072 |
| `B4_minus_away` | 594 | +42.40 | 1.956 | −6.08 | 6.97 | +20.75 | 2.046 |

`B4_minus_away` is marked **NOT DEPLOYABLE** throughout this study: it was named from
Away's sign *reversal* across the split, which reads the evaluation window. On the
selection window Away scores `+25` to `+27%` Kelly ROI with verdict `KEEP`, so no rule
fitted before the split would have dropped it. It is carried as the measurement of what
dropping Away is worth and never as a recommendation. That distinction matters: **it is
the only basket that clears Gate 3** (§8.3).

### 7.2 The pairwise test — one pre-registrable comparison at a time

§8's three in-sample criteria all take an argmax over the whole field and they disagree
with each other, which is the signature of a selection made on too little data: with 49
selection-window slates, the best of forty-eight candidates is mostly the luckiest.

So: **against the production basket, holding the model, the container and the ladder
fixed, does this change win?** 24 paired cells per basket (2 models × 4 containers × 3
ladders), scored on each window separately. The two win columns never share an
observation.

| basket | deployable | IS wins | OOS wins | mean Δ IS % | mean Δ OOS % | mean Δ OOS Sharpe | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `B4_minus_away` | **no** | 0/24 | 18/24 | −58.35 | +5.75 | +0.710 | reverses |
| **`B2_plus_over15`** | yes | **24/24** | **24/24** | **+3.49** | **+4.37** | **+0.111** | **wins in both windows** |
| `B7_x1x2_ou25` | yes | 9/24 | 16/24 | −4.49 | +2.29 | +0.030 | reverses |
| `B1_draw_promoted` | yes | 13/24 | 18/24 | +1.33 | +2.04 | +0.013 | wins in both windows |
| `B3_plus_over25` | yes | **24/24** | **3/24** | +8.29 | **−1.73** | −0.053 | **reverses** |
| `B6_flat11` | yes | 9/24 | 0/24 | −3.18 | −27.52 | −0.758 | loses in both |
| `B5_is_keep` | yes | 22/24 | 0/24 | +2.63 | **−41.72** | −1.194 | **reverses** |

Five things follow, and they are the core of this report.

**1. Adding Over 1.5 at tier 2 is unanimous in both windows.** 48 of 48 paired cells. It
is the only basket change in the study that does not reverse, and its out-of-sample
advantage (`+4.37` points of return, `+0.111` Sharpe) is larger than its in-sample one —
the opposite of the overfitting signature.

**2. Adding Over 2.5 is the textbook overfit.** 24/24 in-sample, 3/24 out-of-sample, mean
out-of-sample effect `−1.73` points. The work package's central hypothesis is refuted on
its own test, and §3.4 shows the published number it rested on splitting `+34.79 / −25.29`.

**3. The fitted pruning rule fails, again, in the same direction as last time.**
`B5_is_keep` — every direction the selection window's rule marked KEEP, tiered at their
median — wins 22 of 24 in-sample cells and **0 of 24** out of sample, at a mean cost of
`−41.72` points. That is `MARKET_LINE_EDA_REPORT` §5.1 reproduced under a different
container, a different price instant and a different rule. **A per-line rule fitted on
this much data does not select baskets.** It explains why lines pay; it does not choose
between them.

**4. Pruning itself is real, even though rules for it are not.** `B6_flat11` — stake all
11 tradeable directions at flat trust — loses in **both** windows, by `−27.52` points out
of sample. The canonical gate is doing genuine work; what fails is re-deriving it from the
data each time.

**5. `MARKET_LINE_EDA_REPORT`'s own recommendation no longer holds up.** `B7_x1x2_ou25` —
prune to `1X2 + O/U 2.5` — loses 15 of 24 selection-window cells against the canonical
tiered basket and wins 16 of 24 out of sample. It reverses, so on this study's evidence it
is neither better nor worse than the canonical tiers; the tiered basket subsumes it.

---

## 8. Attribution, and Gate 3

### 8.1 One factor at a time

Off the deployed policy (`raw`, `B0_canonical`, production ladder). The joint row uses the
matched-risk ladder, so it spends the drawdown budget production already spends and the
comparison is not a comparison of risk appetites.

The container column here is `inv_anch`, which is what the runner's attribution block
uses; on `inv` the same rows land 1–3 points lower and the shape is unchanged.

| model | move | return % | Δ | Sharpe | MDD % | OOS ret % | Δ OOS | OOS Sharpe | OOS MDD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m12 | deployed today | +151.52 | — | 1.592 | −16.15 | +31.77 | — | 0.941 | −16.15 |
| m12 | + calibration only | +66.82 | **−84.70** | 1.775 | −7.95 | +15.38 | −16.39 | 1.008 | −7.95 |
| m12 | + risk ladder only | +169.35 | +17.83 | 1.464 | −19.35 | +31.96 | +0.19 | 0.809 | −19.35 |
| m12 | + Over 1.5 only | +157.19 | +5.67 | 1.638 | −15.61 | +32.48 | +0.70 | 0.969 | −15.61 |
| m12 | **+ all three** | **+268.31** | **+116.79** | 1.696 | −19.18 | **+41.97** | +10.20 | 0.919 | −19.18 |
| m05 | deployed today | +130.15 | — | 1.544 | −16.07 | +30.09 | — | 0.945 | −16.07 |
| m05 | + calibration only | +65.40 | **−64.75** | 1.944 | −6.92 | +19.26 | −10.83 | 1.353 | −6.92 |
| m05 | + risk ladder only | +177.10 | +46.94 | 1.487 | −20.72 | +36.41 | +6.32 | 0.874 | −20.72 |
| m05 | + Over 1.5 only | +140.05 | +9.89 | 1.632 | −16.07 | +33.15 | +3.06 | 1.040 | −16.07 |
| m05 | **+ all three** | **+284.39** | **+154.24** | 1.947 | −18.43 | **+63.44** | +33.35 | 1.403 | −18.43 |

**The parts are worth `−85 + 18 + 6 = −61` and `−65 + 47 + 10 = −8` points; the whole is
worth `+117` and `+154`.** These three changes are not separable and none of them should
be deployed alone — least of all calibration, which at a fixed risk budget is a large
negative on both models.

The mechanism is the one §6.1 measures. What calibration produces is not return; it is
**drawdown headroom** — `−7.95%` where the raw arm runs at `−16.15%`. That headroom is
worthless until the risk ladder is re-pointed to spend it, and the ladder can only be
re-pointed usefully on a container that has headroom to spend. Each half of the pair is
inert without the other, which is precisely why every prior study that moved one of them
alone concluded the move did not pay.

### 8.2 The consolidated deployable surface

Mean over both models, ranked on out-of-sample Sharpe. This table is a **report**, not a
selection rule; nothing in §9 is chosen from it.

| container | basket | ladder | return % | Sharpe | MDD % | OOS ret % | OOS Sharpe | OOS MDD | OOS Calmar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| inv | `B2_plus_over15` | prod | +68.62 | 1.942 | −6.96 | +19.10 | **1.312** | −6.96 | 3.91 |
| **inv_anch** | **`B2_plus_over15`** | **prod** | +69.70 | 1.937 | −7.03 | +19.35 | 1.307 | −7.03 | 3.93 |
| std | `B2_plus_over15` | prod | +72.68 | 1.644 | −10.60 | +24.27 | 1.291 | −8.47 | 4.15 |
| inv_anch | `B7_x1x2_ou25` | prod | +65.90 | 1.787 | −8.19 | +19.73 | 1.285 | −8.19 | 3.51 |
| inv | `B0_canonical` | prod | +64.77 | 1.858 | −7.31 | +17.03 | 1.181 | −7.31 | 3.33 |
| **inv_anch** | **`B2_plus_over15`** | **matched** | **+276.35** | 1.822 | −18.80 | **+52.71** | 1.161 | −18.80 | 4.22 |
| inv | `B2_plus_over15` | matched | +271.47 | 1.830 | −18.61 | +52.11 | 1.170 | −18.61 | 4.21 |
| inv_anch | `B7_x1x2_ou25` | matched | +249.62 | 1.671 | −21.06 | +53.11 | 1.144 | −21.06 | 3.89 |

`B2_plus_over15` heads the ranking at both ladders on both inverse containers. The
`inv` and `inv_anch` rows differ by 1.2–4.9 points of return and ~0.005 of Sharpe, which is
far inside what 99 slates resolve — consistent with README §8.11 item 4, where the anchor
was worth `+1.5` points and was reported as an ordering rather than a result.

### 8.3 Gate 3 — FAIL, and why the threshold is the wrong instrument

Out-of-sample annual Sharpe ≥ 1.65, Calmar ≥ 8.0, max drawdown no worse than −18.0%,
applied to all 192 candidates.

| | count |
|---|---:|
| candidates scored | 192 |
| deployable candidates | 168 |
| pass all three | 10 |
| **pass all three AND deployable** | **0 of 168** |
| pass the Sharpe leg | 12 (best 2.047) |
| pass the Calmar leg | 10 (best 9.79) |
| pass the drawdown leg | 100 (shallowest −3.64%) |

**Every one of the ten passers is `B4_minus_away`** — the basket whose definition read the
evaluation window. Gate 3 is cleared by construction there, not earned. Among candidates
specifiable before the split, **nothing clears it**; the best deployable out-of-sample
Sharpe is `1.542` (`m05`, `inv`, `B2_plus_over15`, production ladder) at Calmar `4.54` —
which clears the Sharpe leg on its own but not the Calmar one.

The binding leg is Calmar, and the reason is arithmetic rather than strategic. Calmar is
an annualised growth rate over a maximum drawdown, and the evaluation window is **50
slates over roughly eleven months**: the drawdown is measured over a window short enough
that one bad fortnight sets it, while the growth rate is not annualised up by much. The
full-period Calmars of the same arms are `13.99` and `15.43` — comfortably above 8.0 —
because they have 99 slates to smooth the denominator. **Gate 3's thresholds were set
from full-period figures and applied to a half-period window.** Reported as measured; the
threshold, not the strategy, is what this study would change.

The Sharpe leg carries a related caveat that is not this study's to fix: the best
out-of-sample Sharpe recorded anywhere in the calibration stream over these same 50 slates
is `1.351`. A 1.65 hurdle on 50 observations is asking the window to resolve something it
cannot.

---

## 9. Recommendation

### 9.1 What to change

**All three, or none.** §8.1 is unambiguous that these do not decompose.

| | today | recommended |
|---|---|---|
| container | raw latents | `GenerativeRateCalibrator`, `InverseGaussianLaw(0.25, 0.35)`, T−25 |
| basket | `CanonicalScottishLowerTrust` (4 directions) | **+ `Over 1.5` at tier 2** (5 directions) |
| ladder | `t1 0.35 : 0.25`, `SlateDrawdown(23.0)`, `FixedCap(0.25)` | see §9.2 — two points, by risk appetite |

`PoolDispersion` (the shipped default, `inv`) and `PreservedDispersion() + :pool_mean`
(`inv_anch`) are indistinguishable here. **Deploy the default.** Nothing in this study
justifies changing it, and §8.2's 1.2–4.9-point gap is inside the noise floor of 99 slates.

### 9.2 Two deployment points

**A — conservative. Container and basket change; risk ladder untouched.**

```julia
book = BookSpec(
    markets   = Data.MarketConfig(Calibration.l2_tradeable_markets()),
    price     = DeArb(),
    allocator = KellyLogUtility(),
    shrink    = Portfolio.FractionalKelly(0.30),
    exec      = ExecutionConfig(commission = PerBetCommission(0.02), budget = 0.99,
                                min_selection_stake = 0.001))

trust = TieredTrust(Dict(
    ("1x2",        0.0, :home)  => 0.35,
    ("over_under", 2.5, :under) => 0.35,
    ("1x2",        0.0, :draw)  => 0.25,
    ("1x2",        0.0, :away)  => 0.25,
    ("over_under", 1.5, :over)  => 0.25,   # <- the one addition
); default = 0.0)

policy = PolicySpec(trust = trust, risk = SlateDrawdown(23.0),
                    cap = FixedCap(0.25), grouping = DailySlate())

cal = GenerativeRateCalibrator(name = "scot_lower_t25_inv",
                               law  = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
                               book_as_of_minutes = -25.0)
```

Measured on `inv` — the shipped `PoolDispersion` default. (`inv_anch` gives
`+68.92 / +70.47` and is within noise; see §9.1.)

| model | bets | return % | Sharpe | MDD % | Calmar | OOS ret % | OOS Sharpe | OOS MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m12 | 1,009 | +68.09 | 1.822 | **−7.06** | 9.65 | +16.36 | 1.081 | −7.06 |
| m05 | 985 | +69.15 | **2.061** | **−6.86** | 10.08 | +21.83 | **1.542** | −6.86 |
| *m12 today* | *1,127* | *+151.52* | *1.592* | *−16.15* | *9.38* | *+31.77* | *0.941* | *−16.15* |
| *m05 today* | *1,113* | *+130.15* | *1.544* | *−16.07* | *8.10* | *+30.09* | *0.945* | *−16.07* |

Less than half the deployed policy's total return, at less than half its drawdown, and the
best risk-adjusted numbers in the study. This is the point to take if the 20% risk budget
is a limit rather than a target.

**B — matched-risk. All three changes, spending the drawdown budget production already
spends.** Same book and calibrator; the trust table scaled to `t1 = 1.00`
(`home`, `under 2.5` at 1.00; `draw`, `away`, `over 1.5` at 0.714) with
`SlateDrawdown(8.0)`, `FixedCap(0.25)`.

| model | bets | return % | Sharpe | MDD % | Calmar | OOS ret % | OOS Sharpe | OOS MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m12 | 1,009 | **+264.99** | 1.703 | −18.94 | 13.99 | **+42.07** | 0.932 | −18.94 |
| m05 | 985 | **+277.95** | 1.956 | −18.27 | 15.21 | **+62.14** | 1.407 | −18.27 |
| *m12 today* | *1,127* | *+151.52* | *1.592* | *−16.15* | *9.38* | *+31.77* | *0.941* | *−16.15* |
| *m05 today* | *1,113* | *+130.15* | *1.544* | *−16.07* | *8.10* | *+30.09* | *0.945* | *−16.07* |

**`+113` and `+148` points of compound return, and `+10` and `+32` points out of sample,
at 2.5 points more drawdown.** (`inv_anch` gives `+268.31 / +284.39` full period and
`+41.97 / +63.44` out of sample — the same statement, 1–2 points apart.)

### 9.3 What must NOT be deployed

* **Over 2.5 un-gated.** 24/24 in-sample, 3/24 out-of-sample (§7.2).
* **Any basket fitted from the per-line rule.** 22/24 in-sample, 0/24 out-of-sample,
  `−41.72` points (§7.2). Twice now.
* **The unpruned 11-direction book.** Loses in both windows (§7.2).
* **O/U 0.5, in any form.** 10.7% quote coverage, zero wins in 136 pooled bets (§3.3).
* **Point B without a live drawdown monitor.** Every arm chosen on an in-sample drawdown
  constraint breached it out of sample (§6.3), and Point B stakes roughly three times
  production's exposure into League Two liquidity.

---

## 10. Boundaries

* **Fill model.** Bets are struck at the archived traded price in whatever size the
  allocator asked for. The live system rests at the touch and `betfair_live.order_book_1m`
  carries at most three levels per side. These returns remain an upper bound, and Point B
  is where that bound bites hardest.
* **Traded price, not the resting ladder.** `betfair.odds_history` archives what someone
  paid; a T−25 traded price is not necessarily what was showing on the side we would have
  taken.
* **The evaluation window is 50 slates.** It settles orderings, not magnitudes. §7.2's
  unanimity across 24 paired cells is the strongest form of evidence available here, and it
  is still 24 correlated cells over one half-season.
* **Out-of-sample metrics are read off the full-period bet path.** Windows are re-based to
  a bankroll of 1.0 at their first slate, but the stakes were sized against the in-sample
  bankroll. Re-simulating from the split would also re-solve every subsequent risk budget
  and answer a different question.
* **`m05`'s canonical fit does not pass strict convergence** (`converged = false`, as in
  experiment 06). It is carried as the sensitivity model, and every conclusion in §7 and
  §9 holds on `m12` alone.
* **One league pair, two seasons, one split date.** A line pruned for weak efficiency is
  not a line that cannot be priced — it is one this book, at this Kelly fraction, under
  this cap, could not pay for.

---

## 11. Reproducing this

```bash
ssh root@mcmc-beast
cd /root/BF_calv2 && git pull origin feat/modernize-calibration-layer2
julia --project -t 16
```
```julia
julia> include("experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/r07_line_forensics_calibrated.jl")
julia> include("experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/r07_trust_and_lambda_sweep.jl")
julia> include("experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/r07_optimal_portfolio_comparison.jl")
```

r07/3 reads r07/1's `market_pruning_audit_calibrated.csv` and r07/2's
`trust_lambda_frontier.csv`, so the order matters. No MCMC is launched; both databases are
read-only and neither paper ledger is opened. Total runtime ≈ 9 minutes.

Artefacts in [`results/`](results/) — see [`README.md`](README.md) §4 for the manifest.
