# 07 — Calibrated portfolio, line pruning and the directional trust vector

Scottish Lower (tournaments 56 / 57), 24/25 + 25/26, 710 held-out fixtures, 99 daily
slates, priced at **T−25** against the archived Betfair book. Two canonical fits, four
calibration containers, 192 portfolio candidates and a 4,096-cell risk grid, all run on
`mcmc-beast` on 2026-09-05 from `feat/modernize-calibration-layer2` @ `e6f61475`.

Full evidence: **[`CALIBRATED_TRUST_EDA_REPORT.md`](CALIBRATED_TRUST_EDA_REPORT.md)**.
Brief: [`WORK_PACKAGE_PROMPT.md`](WORK_PACKAGE_PROMPT.md).

---

## 1. Verdicts

| Question | Verdict |
|---|---|
| Does calibration rehabilitate **Over 2.5**? | **No.** The published `+14.32%` Kelly ROI reproduces exactly and splits `+34.79%` in-sample / **`−25.29%` out of sample** on 15 bets. Adding it to the basket wins 24/24 paired cells in-sample and **3/24** out of sample. Keep the gate shut. |
| Does any pruned direction earn a tier? | **Yes — `Over 1.5`, at tier 2.** The only basket change in the study that wins in **both** windows: 24/24 and 24/24, on both models, all four containers, all three ladders. |
| Does **O/U 0.5** stay broken? | **Yes, decisively.** Quoted on 10.7% of gate-season fixtures; `Under 0.5` won **0 of 74** bets on `raw`, 0 of 54 on `inv`, 0 of 62 on `inv_anch` and 1 of 68 on `std`. `l2_tradeable_markets`'s exclusion stands. |
| Is trust `0.35` under-betting a calibrated container? | **No, and the question is mis-posed.** At λ = 23, `t1 = 0.50 / 0.70 / 1.00` are **bit-identical** portfolios and `0.35` is marginally the best. Trust and λ are one two-dimensional knob; `(0.35, 23)` sits in its dead corner. |
| What is λ worth? | **Everything, but only on a calibrated container.** `(t1 = 1.00, λ = 8)` on `inv` takes `m12` `+151.52% → +264.99%` and `m05` `+130.15% → +277.95%`, out of sample `+31.77/+30.09% → +42.07/+62.14%`. |
| Do the answers hold on both models? | **Yes.** The same risk cell is chosen independently on both models' selection windows; the calibrated containers agree on 12/13 directions against the raw container's 11/13. |
| Gate 3 (OOS Sharpe ≥ 1.65, Calmar ≥ 8.0, MDD ≥ −18%)? | **FAIL — 0 of 168 deployable candidates.** The 10 passers among all 192 are all one basket whose definition read the evaluation window. §8.3 argues the Calmar threshold is mis-specified for a 50-slate window, not that the strategy is inadequate. |

### 1.1 The result that matters most, and it is a negative one

**A per-line pruning rule fitted on this much data does not select baskets.** The
data-driven basket — every direction the selection window's rule marked `KEEP` — wins
**22 of 24** in-sample paired cells and **0 of 24** out of sample, at a mean cost of
`−41.72` points of return. That is `MARKET_LINE_EDA_REPORT.md` §5.1's failure reproduced
under a different container, a different price instant and a different rule.

Line forensics explain *why* a line pays. They do not choose between baskets. The
adjudicator is a paired re-simulation across both windows, and this suite is built so
that it is.

---

## 2. Deployment guide

**All three changes, or none.** They are strongly super-additive: taken alone, calibration
*loses* 65–85 points at a fixed risk budget, the ladder is worth +18 to +47, and the basket
+6 to +10 — while together they are worth **+117 and +154**. What calibration produces is
drawdown headroom (`−7.06%` where the raw arm runs at `−16.15%`); only the risk ladder can
spend it, and only a container with headroom is worth re-pointing the ladder on.

```julia
# 1. The container — the T−25 inverse pool, PoolDispersion (the shipped default)
cal = GenerativeRateCalibrator(name = "scot_lower_t25_inv",
                               law  = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
                               book_as_of_minutes = -25.0)

book, _ = point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0,
                                                                max_staleness_minutes = 90.0))
cf = calibrate_fit(cal, fit, book)

# 2. The book — unchanged, 11 tradeable directions
spec = BookSpec(markets   = Data.MarketConfig(Calibration.l2_tradeable_markets()),
                price     = DeArb(),
                allocator = KellyLogUtility(),
                shrink    = Portfolio.FractionalKelly(0.30),
                exec      = ExecutionConfig(commission = PerBetCommission(0.02),
                                            budget = 0.99, min_selection_stake = 0.001))

# 3. The basket — CanonicalScottishLowerTrust plus ONE direction
trust_A = TieredTrust(Dict(
    ("1x2",        0.0, :home)  => 0.35,
    ("over_under", 2.5, :under) => 0.35,
    ("1x2",        0.0, :draw)  => 0.25,
    ("1x2",        0.0, :away)  => 0.25,
    ("over_under", 1.5, :over)  => 0.25,      # <- the addition
); default = 0.0)

# 4. The ladder — pick ONE
policy_A = PolicySpec(trust = trust_A, risk = SlateDrawdown(23.0),      # conservative
                      cap = FixedCap(0.25), grouping = DailySlate())

trust_B = TieredTrust(Dict(
    ("1x2",        0.0, :home)  => 1.000,
    ("over_under", 2.5, :under) => 1.000,
    ("1x2",        0.0, :draw)  => 0.714,     # 1.00 / 1.4
    ("1x2",        0.0, :away)  => 0.714,
    ("over_under", 1.5, :over)  => 0.714,
); default = 0.0)
policy_B = PolicySpec(trust = trust_B, risk = SlateDrawdown(8.0),       # matched-risk
                      cap = FixedCap(0.25), grouping = DailySlate())
```

| point | model | return % | Sharpe | MDD % | Calmar | OOS ret % | OOS Sharpe | OOS MDD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A** conservative | m12 | +68.09 | 1.822 | **−7.06** | 9.65 | +16.36 | 1.081 | −7.06 |
| **A** conservative | m05 | +69.15 | **2.061** | **−6.86** | 10.08 | +21.83 | **1.542** | −6.86 |
| **B** matched-risk | m12 | **+264.99** | 1.703 | −18.94 | 13.99 | **+42.07** | 0.932 | −18.94 |
| **B** matched-risk | m05 | **+277.95** | 1.956 | −18.27 | 15.21 | **+62.14** | 1.407 | −18.27 |
| *deployed today* | *m12* | *+151.52* | *1.592* | *−16.15* | *9.38* | *+31.77* | *0.941* | *−16.15* |
| *deployed today* | *m05* | *+130.15* | *1.544* | *−16.07* | *8.10* | *+30.09* | *0.945* | *−16.07* |

**A** is less than half today's return at less than half its drawdown, with the best
risk-adjusted numbers measured anywhere in this study. **B** nearly doubles the return at
2.5 points more drawdown. Point A if the 20% budget is a limit; point B if it is a target.

### 2.1 Do not deploy

* **Over 2.5 un-gated** — 24/24 in-sample, 3/24 out of sample.
* **Any basket fitted from the per-line rule** — 22/24 in-sample, 0/24 out of sample.
* **The unpruned 11-direction book** — loses in both windows, `−27.52` points out of sample.
* **O/U 0.5, in any form.**
* **Point B without a live drawdown monitor.** Every arm chosen on an in-sample drawdown
  constraint breached it out of sample, by 0.3 to 11 points, and point B stakes roughly
  three times production's exposure into League Two liquidity. A backtested drawdown is
  not a risk limit.

### 2.2 What is NOT recommended

`PreservedDispersion() + :pool_mean` (`inv_anch`) beats the shipped `PoolDispersion` by
1.2–4.9 points of return and ~0.005 of Sharpe across every arm — consistently, and far
inside what 99 slates resolve. That reproduces the ordering
`calibration_generative_eda/README.md` §8.11 item 4 recorded, and this study likewise
reports it as an ordering rather than a result. **Deploy the default.**

---

## 3. Files

| File | Is |
|---|---|
| [`l07_calibrated_trust_loader.jl`](l07_calibrated_trust_loader.jl) | Loader. Containers, book and policy specs, the geometric conviction ladder, the ledger and its accounting gate, per-line metrics, the pruning rule, window metrics, the sweep. Definitions only. |
| [`r07_line_forensics_calibrated.jl`](r07_line_forensics_calibrated.jl) | Runner 1. 13-direction forensics raw vs calibrated, Gate 1, the IS/OOS pruning audit, the rehabilitation table, and the parity check against README §8.9. |
| [`r07_trust_and_lambda_sweep.jl`](r07_trust_and_lambda_sweep.jl) | Runner 2. Gate 5's two structural identities, the 4,096-cell `(trust, ratio, λ, cap)` grid, the regime map, the λ ceiling, and the honest frontier. |
| [`r07_optimal_portfolio_comparison.jl`](r07_optimal_portfolio_comparison.jl) | Runner 3. Eight baskets × three ladders × four containers × two models, the pairwise test, attribution, Gate 3, and the recommendation with a clustered bootstrap. |
| [`CALIBRATED_TRUST_EDA_REPORT.md`](CALIBRATED_TRUST_EDA_REPORT.md) | The written research report. |
| [`results/`](results/) | Replaceable CSV artefacts; re-running overwrites them. |

Runner 3 reads runner 1's `market_pruning_audit_calibrated.csv` and runner 2's
`trust_lambda_frontier.csv`, so **run them in order**. Total runtime ≈ 9 minutes on
`mcmc-beast` at `-t 16`; no MCMC is launched.

---

## 4. Artefacts

| CSV | Rows | Holds |
|---|---:|---|
| `market_line_breakdown_calibrated.csv` | 684 | Every (container × scope × window × ladder × direction): bets, win rate, calibration bias, edge, flat and Kelly ROI, capital share, both efficiency definitions, standalone drawdown, streaks. |
| `market_pruning_audit_calibrated.csv` | 228 | Per direction: the selection-window verdict, the out-of-sample gate, and whether the sign held across the split. |
| `market_line_rehabilitation.csv` | 39 | Raw against each calibrated container, direction by direction, with a `REHABILITATED` / `LOST` / `both fail` status. |
| `market_line_cross_model.csv` | 52 | Gate-2 agreement between `m12` and `m05`. |
| `over25_parity_with_readme_8_9.csv` | 3 | The published Over 2.5 arm, reproduced and then split. |
| `t25_book_ladder_coverage.csv` | 8 | What the T−25 book actually quotes, per ladder. |
| `forensic_portfolio_summary.csv` | 8 | The eight forensic simulations' headline metrics. |
| `forensic_bet_ledger.csv` | 11,845 | Every struck bet, rescaled into currency, with its window label. |
| `trust_lambda_grid_sweep.csv` | 4,096 | The full grid, with `mean_k_risk` and `frac_k_pinned` on every row. |
| `trust_lambda_frontier.csv` | 48 | Best return inside each drawdown budget, in-sample-optimal and honest. |
| `trust_lambda_cross_model_transfer.csv` | 16 | Each model's chosen cell scored on the other model. |
| `optimal_portfolio_comparison.csv` | 192 | The full head-to-head. |
| `deployable_leaderboard.csv` | 84 | Deployable candidates, mean over both models, ranked out of sample. |
| `basket_pairwise_vs_canonical.csv` | 7 | The paired test — the table §1.1 rests on. |
| `policy_attribution.csv` | 10 | One factor at a time off the deployed policy. |
| `gate3_scoreboard.csv` | 192 | Every candidate against Gate 3's three thresholds. |
| `recommended_policy.csv` | 6 | The three predeclared criteria's picks, with clustered bootstrap ROI intervals. |

---

## 5. Gates

| Gate | Where | Refuses | Result |
|---|---|---|---|
| **1** | r07/1 §5 | Ledger accounting: no non-finite value; the bankroll rescale reproduces `trajectory.total_stake` / `total_pl`; `pnl == stake × payoff` bet by bet. | **PASS** 8/8. Worst identity error `0.00e+00`, worst total error `2.66e-15`. |
| **2** | r07/1 §7 | A direction may earn a tier only with `Kelly ROI > 0` and `capital_efficiency ≥ 0.25` in **both** windows. | Applied. Passing on both models: `1X2 draw` and `O/U 1.5 over` on the inverse containers; nothing on `raw`. |
| **3** | r07/3 §10 | OOS annual Sharpe ≥ 1.65, Calmar ≥ 8.0, MDD ≥ −18.0%. | **FAIL** — 0 of the 168 deployable candidates. The 10 passers among all 192 are every one of them `B4_minus_away`, whose definition read the evaluation window. |
| **5a** | r07/2 §5 | Scale invariance must hold where `SlateDrawdown`'s `k` is interior. | Located rather than assumed: `t1 = 0.50` vs `1.00` identical to the last digit; `0.35` vs `0.70` not. |
| **5b** | r07/2 §5 | `FlatTrust(τ) × FractionalKelly(f)` must equal `FlatTrust(1) × FractionalKelly(τf)`. | **PASS**, exactly. There is no separate Kelly axis to sweep. |
| **parity** | r07/1 §8b | The published Over 2.5 arm must reproduce before it is reinterpreted. | **PASS** to every printed digit: 38 bets, `+14.32%` Kelly, `+7.10%` flat. |

A deviation from the brief is recorded in the report §2.2: Gate 2's `capital_efficiency` is
tested against the **selection window's** book ROI rather than the same window's, because
out of sample the same-window denominator collapses toward zero and the ratio either
explodes or is undefined. Both columns are in the CSVs.

---

## 6. Boundaries

* Reads `mcmc_experiments` (posteriors via `PostgresStorage`) and `betdb` (odds, results).
  **Writes neither.** No run, portfolio, calibration or config registration.
* `betdb.paper_runbook` and `betdb.paper_replay` are never opened. The live console on
  8085 and the replay console on 8086 are not this suite's business.
* Credentials are resolved from the environment by `PostgresStorage` / `Data` and never
  printed.
* Bets are struck at the archived traded price in whatever size the allocator asked for.
  There is no fill model, and point B is where that omission bites hardest.
* `m05`'s canonical fit does not pass strict convergence (as in experiment 06). It is the
  sensitivity model; every conclusion in §1 and §2 holds on `m12` alone.

---

## 7. Provenance

| Role | Model | Run UUID |
|---|---|---|
| primary | `m12_joint_hybrid_synergy` | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` |
| sensitivity | `m05_joint_production_wealth` | `ed541a7c-01e2-447e-a771-783517728d47` |

Experiment namespace `scottish_lower_joint_player_2426`; latents restricted to the 24/25 +
25/26 gate seasons (759 → 710 fixtures) so this suite scores the same fixture set the
published figures were measured on. T−25 book: 10,373 rows, 1,572 fixtures, median
staleness 8 minutes, overround 1.0015; market-rate inversion accepted 580 of 710
gate-season fixtures (94.9% of those with a book).

Related: [`../MARKET_LINE_EDA_REPORT.md`](../MARKET_LINE_EDA_REPORT.md),
[`../../../eda/MULTITIER_TRUST_REPORT.md`](../../../eda/MULTITIER_TRUST_REPORT.md),
[`../../../current_development/calibration_generative_eda/README.md`](../../../current_development/calibration_generative_eda/README.md),
[`../../../docs/architecture/rfc_layer2_calibration_v2.md`](../../../docs/architecture/rfc_layer2_calibration_v2.md).
