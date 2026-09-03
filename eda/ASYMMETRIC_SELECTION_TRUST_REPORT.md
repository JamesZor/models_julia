# Asymmetric Selection-Level Trust Across Football Market Lines

## Executive summary

The central result is clear and stable across both Gen-4 models: **O/U 2.5 should not be treated as one symmetric market arm. Under 2.5 carries the realised value; Over 2.5 destroys it.**

With all other settings fixed, changing the production candidate from symmetric core trust—1X2 plus both O/U 2.5 directions—to asymmetric core trust—1X2 plus Under 2.5 only—improved every requested headline metric:

- `m12`: return **+141.53% → +143.91%**, Sharpe **1.485 → 1.516**, maximum drawdown **-19.80% → -18.39%**, with 59 fewer bets.
- `m13`: return **+144.43% → +146.14%**, Sharpe **1.522 → 1.548**, maximum drawdown **-20.32% → -19.16%**, with 56 fewer bets.
- Averaged across the two models, directional pruning added **+2.05 percentage points** of terminal return and **+0.0286 Sharpe**, improved maximum drawdown by **1.29 pp**, and reduced turnover by 0.214 bankroll units relative to symmetric core.

The mechanism is directly visible in the status-quo ledger:

| Model | Direction | Bets | Stake | P&L | ROI | Mean model p | Mean market p | Realised win rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `m12` | Under 2.5 | 227 | 1,828.17 | +348.76 | **+19.08%** | 0.4937 | 0.4488 | 0.4890 |
| `m12` | Over 2.5 | 59 | 399.22 | -42.79 | **-10.72%** | 0.5152 | 0.4736 | 0.4746 |
| `m13` | Under 2.5 | 229 | 1,839.85 | +336.20 | **+18.27%** | 0.4932 | 0.4486 | 0.4934 |
| `m13` | Over 2.5 | 56 | 398.05 | -48.10 | **-12.08%** | 0.5158 | 0.4729 | 0.4643 |

Both directions displayed approximately 4.2-4.5 percentage points of apparent model-versus-market edge when selected. Only the under edge realised. The over win rate was approximately equal to or below the market probability and 4.1-5.2 points below the model probability. This is not merely weaker value; it is a directional calibration failure in the traded subset.

The predeclared “all unders” expansion failed. Adding Under 1.5 and Under 3.5 to the asymmetric core reduced average return from **+145.03% to +136.68%**, reduced Sharpe from **1.532 to 1.487**, and worsened average maximum drawdown from **-18.77% to -21.32%**. “Under” is not a universal alpha factor. The profitable result is specific to the 2.5 direction.

An exhaustive, descriptive information-ratio search selected **Home, Draw, Under 1.5, and Under 2.5**. Its re-solved portfolio reached Sharpe **1.598** on `m12` and **1.691** on `m13`, but with much lower return (**+73.81%** and **+78.84%**) because it drops Away and deploys about 40% less turnover than asymmetric core. This policy was selected and scored on the same 100 dates. It is an in-sample diagnostic frontier, not a production recommendation.

**Recommendation:** deploy strict `SelectionTrust` at 0.30 for Home, Draw, Away, and Under 2.5; set every other tested direction—including Over 2.5—to zero. This is the strongest predeclared, operationally simple policy supported by converged `m12`, and it also passes the second-half time split on both models.

---

## 1. Experimental contract

### 1.1 Models and provenance

The analysis uses the immutable canonical 40-fold PostgreSQL fits:

| Role | Model | Run UUID | Convergence |
|---|---|---|---|
| primary | `m12_joint_hybrid_synergy` | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | strict pass; 40/40 folds |
| sensitivity | `m13_joint_composite` | `5474e824-8c9d-4613-8e39-841426c3f80f` | strict aggregate failure on tail ESS; 38/40 folds individually converged |

No MCMC was launched and no fit artifact was modified. As in the preceding capacity audit, deserialization required the artifact-compatible source revision `784c8ea81328760e75498b19d13c2dab762bde8e` because the current `JointGammaPoissonObservation` type has changed since serialization.

Each model produced 632 usable match books from 710 held-out fixtures over 100 daily slate dates. Seventy-eight fixtures lacked a usable controlled book. The `m13` outputs are sensitivity evidence rather than a production convergence pass.

### 1.2 What is held fixed

Every policy uses the same:

- six-market book: 1X2, O/U 0.5, 1.5, 2.5, 3.5, and BTTS;
- Betfair time-weighted close over the final 20 minutes before kickoff;
- `DeArb()`, `KellyLogUtility()`, and `NoShrinkage()`;
- 2% commission and 0.001 minimum selection stake;
- `SlateDrawdown(23.0)`, `FixedCap(0.20)`, and `DailySlate()`;
- initial bankroll 1,000;
- exact score-grid outcomes and full joint re-solve after every gate change.

Only the strict `SelectionTrust` table changes. A zero-trust direction is unavailable to the allocator; surviving stakes are not obtained by subtracting or rescaling the old ledger.

### 1.3 Policies

| Policy | Active directions at trust 0.30 | Status |
|---|---|---|
| `P_baseline` | all 13 directions | predeclared |
| `P_symmetric_core` | Home, Draw, Away, Over 2.5, Under 2.5 | predeclared |
| `P_asymmetric_core` | Home, Draw, Away, Under 2.5 | predeclared primary test |
| `P_under_expansion` | Home, Draw, Away, Under 1.5, Under 2.5, Under 3.5 | predeclared |
| `P_pure_alpha` | Home, Draw, Under 1.5, Under 2.5 | exhaustive data-selected diagnostic |

For `P_pure_alpha`, the script creates one zero-filled daily return stream per direction from the baseline ledger, averages the two model streams per date, enumerates all **8,191 non-empty subsets**, and chooses the subset with maximum annualized information ratio. The winning diagnostic subset's baseline-stream IR was 1.810. Because selection and final policy scoring use the same dates, this is explicitly optimistic.

Every underlying match probability is out of sample to its training fold. That does **not** make the policy selection out of sample: the directional rule still observes the settlement history later used to report it.

---

## 2. Policy comparison

### 2.1 Full 100-slate results

| Model | Policy | Final bankroll | Return | Sharpe | Max DD | Bets | Turnover | Cap-binding slates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `m12` | baseline | 2,239.37 | +123.94% | 1.333 | -20.97% | 1,894 | 9.774 | 4 |
| `m12` | symmetric core | 2,415.34 | +141.53% | 1.485 | -19.80% | 1,314 | 7.732 | 0 |
| `m12` | **asymmetric core** | **2,439.15** | **+143.91%** | **1.516** | **-18.39%** | 1,255 | 7.519 | 0 |
| `m12` | under expansion | 2,357.81 | +135.78% | 1.472 | -21.36% | 1,490 | 8.506 | 0 |
| `m12` | pure alpha* | 1,738.07 | +73.81% | 1.598 | -17.21% | 995 | 4.438 | 0 |
| `m13`† | baseline | 2,278.59 | +127.86% | 1.375 | -20.84% | 1,893 | 9.807 | 4 |
| `m13`† | symmetric core | 2,444.27 | +144.43% | 1.522 | -20.32% | 1,316 | 7.775 | 0 |
| `m13`† | **asymmetric core** | **2,461.37** | **+146.14%** | **1.548** | **-19.16%** | 1,260 | 7.561 | 0 |
| `m13`† | under expansion | 2,375.71 | +137.57% | 1.502 | -21.28% | 1,495 | 8.554 | 0 |
| `m13`† | pure alpha* | 1,788.43 | +78.84% | 1.691 | -17.61% | 1,009 | 4.556 | 0 |

\* data-selected on the same dates; diagnostic only.

† strict aggregate convergence failure on tail ESS.

### 2.2 Average ranking

| Policy | Mean return | Mean Sharpe | Mean max DD | Mean bets | Mean turnover |
|---|---:|---:|---:|---:|---:|
| pure alpha* | +76.33% | **1.644** | **-17.41%** | 1,002 | 4.497 |
| **asymmetric core** | **+145.03%** | **1.532** | **-18.77%** | 1,257.5 | 7.540 |
| symmetric core | +142.98% | 1.503 | -20.06% | 1,315 | 7.753 |
| under expansion | +136.68% | 1.487 | -21.32% | 1,492.5 | 8.530 |
| baseline | +125.90% | 1.354 | -20.91% | 1,893.5 | 9.790 |

There are two distinct objectives in this table. Pure alpha maximizes a risk-adjusted statistic by accepting materially lower deployment and growth. Asymmetric core is the best predeclared policy for terminal wealth and still exceeds Sharpe 1.5 on both models. For a production Kelly system whose mandate is bankroll growth under an explicit drawdown budget, asymmetric core is the appropriate comparison winner.

### 2.3 Increment from pruning Over 2.5

| Model | Return delta vs symmetric core | Sharpe delta | Max-DD improvement | Bets removed | Turnover removed |
|---|---:|---:|---:|---:|---:|
| `m12` | **+2.38 pp** | **+0.0305** | **+1.41 pp** | 59 | 0.214 |
| `m13`† | **+1.71 pp** | **+0.0267** | **+1.16 pp** | 56 | 0.214 |

Relative to the six-market baseline, asymmetric core adds **+19.98 pp** return and +0.182 Sharpe for `m12`, and +18.28 pp return and +0.173 Sharpe for `m13`.

---

## 3. The directional anatomy of the baseline book

The table below pools the two model ledgers only for compact diagnosis. Currency stakes and P&L are first reconstructed against each slate's opening bankroll, as in the capacity audit.

| Direction | Bets | Stake | P&L | ROI | Pooled baseline IR |
|---|---:|---:|---:|---:|---:|
| Under 2.5 | 456 | 3,668.02 | +684.95 | **+18.67%** | **1.242** |
| Home | 600 | 9,384.86 | +1,095.12 | **+11.67%** | **1.154** |
| Draw | 720 | 4,861.37 | +499.78 | **+10.28%** | 0.355 |
| Away | 739 | 10,693.95 | +601.99 | **+5.63%** | 0.654 |
| Under 1.5 | 228 | 1,074.50 | -8.94 | -0.83% | 0.082 |
| Over 3.5 | 171 | 904.18 | -8.98 | -0.99% | 0.096 |
| Under 3.5 | 242 | 3,212.49 | -41.14 | -1.28% | -0.124 |
| Over 0.5 | 66 | 1,037.80 | -30.54 | -2.94% | -0.130 |
| Over 1.5 | 94 | 726.14 | -23.76 | -3.27% | -0.205 |
| BTTS yes | 130 | 1,084.71 | -46.74 | -4.31% | -0.144 |
| BTTS no | 100 | 508.99 | -45.81 | -9.00% | -0.397 |
| **Over 2.5** | 115 | 797.27 | **-90.88** | **-11.40%** | -0.163 |
| Under 0.5 | 126 | 216.55 | -67.11 | -30.99% | -0.571 |

The four directional legs in asymmetric core are the only large, clearly positive economic contributors. Under 1.5 and Over 3.5 receive positive full-period standalone IRs because of covariance and the timing of small returns, despite negative pooled currency ROI. Their marginal contribution to a re-solved portfolio must therefore be tested rather than inferred from the sign of one statistic.

---

## 4. Under 2.5 versus Over 2.5

### 4.1 The model finds a similar apparent edge in both directions

For `m12` selections actually bet:

- Under 2.5: mean model 0.4937, mean de-vigged market 0.4488, apparent edge +0.0449.
- Over 2.5: mean model 0.5152, mean de-vigged market 0.4736, apparent edge +0.0417.

For `m13`:

- Under 2.5: edge +0.0446.
- Over 2.5: edge +0.0430.

If model-minus-market edge alone were sufficient, these arms should look comparable. Settlements reject that symmetry.

### 4.2 Only the under disagreement survives reality

For `m12`, Under 2.5 realised at 0.4890—4.02 percentage points above the selected market probability and only 0.47 points below the model. Over 2.5 realised at 0.4746—only 0.10 points above the market but 4.07 points below the model.

For `m13`, Under 2.5 realised at 0.4934—4.48 points above market and effectively equal to the model. Over 2.5 realised at 0.4643—0.86 points below market and 5.15 points below model.

Thus the under trade is not simply “the model predicts fewer goals.” It is a conditional statement: among prices the model elects to trade, the exchange underprices Under 2.5 relative to realised frequency, while the model overstates Over 2.5 probability when it disagrees in that direction.

### 4.3 Public-bias interpretation, with the proper caveat

A plausible market mechanism is recreational preference for positive-skew, action-aligned outcomes: goals, favourites, and visually exciting “overs.” If over demand shortens over prices, the complementary under can carry a systematic premium. The ledger is consistent with that account: selected unders beat the market by about four points, while selected overs do not.

However, this backtest contains closing prices and outcomes—not bettor identities, order flow, traded volume by participant class, or causal demand shocks. It therefore **does not identify public bias directly**. Other mechanisms could produce the same pattern:

- model misspecification conditional on high-total latent states;
- selection effects from the edge threshold;
- stale or heterogeneous quotes in the time-weighted close;
- league-specific low-scoring structure not fully reflected in the exchange;
- small-sample settlement variation (115 Over 2.5 bets pooled).

The production decision does not require choosing among those explanations: the directional gate is justified by the stable re-solve. The public-bias story should be treated as a hypothesis to test with order-flow or cross-league evidence, not as an established cause.

---

## 5. Time stability

The 100 slates split at 2025-05-03. These windows were not used to fit the four predeclared policies; they expose whether the full-period result is carried only by the first half.

### 5.1 Second-half results

| Model | Policy | Second-half return | Second-half Sharpe | Second-half max DD |
|---|---|---:|---:|---:|
| `m12` | baseline | +14.66% | 0.474 | -20.97% |
| `m12` | symmetric core | +27.91% | 0.840 | -19.80% |
| `m12` | **asymmetric core** | **+30.52%** | **0.922** | **-18.39%** |
| `m12` | under expansion | +19.07% | 0.627 | -21.36% |
| `m13`† | baseline | +18.75% | 0.594 | -20.84% |
| `m13`† | symmetric core | +32.35% | 0.958 | -20.32% |
| `m13`† | **asymmetric core** | **+35.38%** | **1.051** | **-19.16%** |
| `m13`† | under expansion | +23.21% | 0.749 | -21.28% |

Asymmetric core beats symmetric core in the second half for both models. It also has the best second-half return and Sharpe among the four predeclared policies. The evidence is therefore stronger than a full-period ledger anecdote, although this remains one league pair and two seasons.

The data-selected pure-alpha policy also had positive second-half results, but it was chosen using the full period, including those dates, so they are not an honest out-of-sample validation for that policy.

---

## 6. Why “all unders” fails

The under-expansion policy assumes a common directional factor across strikes. Its active totals are Under 1.5, Under 2.5, and Under 3.5. The ledger rejects that pooling:

- pooled Under 2.5 ROI: **+18.67%**;
- pooled Under 1.5 ROI: **-0.83%**;
- pooled Under 3.5 ROI: **-1.28%**.

In the re-solved under-expansion policy, Under 1.5 remained negative and Under 3.5 became more negative. Those arms also consumed stake and changed allocations to profitable 1X2/Under 2.5 positions. Consequently, the all-under policy lost 8.35 pp of average return and 0.045 Sharpe relative to asymmetric core while worsening drawdown by 2.54 pp.

Different total strikes are not interchangeable exposures. They probe different regions of the score distribution, carry different odds/payoff convexity, and can have different market microstructure. The correct control key is `(market, line, selection)`, exactly the granularity provided by `SelectionTrust`.

---

## 7. Pure-alpha frontier: what it does and does not prove

The exhaustive subset search chose:

- Home;
- Draw;
- Under 1.5;
- Under 2.5.

It drops Away despite positive ROI because the objective is portfolio information ratio, not total P&L. It includes Under 1.5 despite slightly negative standalone ROI because its dated return stream improves the in-sample mean/volatility ratio of the subset.

After a full Kelly re-solve, this policy delivered the highest Sharpe and shallowest drawdown but much less growth:

| Model | Return | Sharpe | Max DD | Turnover |
|---|---:|---:|---:|---:|
| `m12` | +73.81% | 1.598 | -17.21% | 4.438 |
| `m13`† | +78.84% | 1.691 | -17.61% | 4.556 |

This is a useful efficient-frontier observation: dropping positive-return Away exposure can raise risk-adjusted performance while cutting total wealth. But the search evaluated 8,191 subsets on the same 100 dates, so winner's curse is unavoidable. The subset should be frozen and tested prospectively or selected inside an outer walk-forward loop before any production use.

Calling its input “out-of-sample information ratio” means the **model predictions** are fold-held-out. It does not mean the **policy choice** is out of sample. The CSVs preserve this naming and the report preserves the distinction.

---

## 8. MatchDay recommendation

Use this strict directional trust table for the current production candidate:

```julia
Dict(
    ("1X2", 0.0, :home)       => 0.30,
    ("1X2", 0.0, :draw)       => 0.30,
    ("1X2", 0.0, :away)       => 0.30,
    ("OverUnder", 2.5, :under_25) => 0.30,
    ("OverUnder", 2.5, :over_25)  => 0.00,
    # Every O/U 0.5, 1.5, 3.5 and BTTS direction => 0.00
)
```

In implementation, populate all 13 keys and construct `SelectionTrust(table; strict=true)`. Explicit zeroes and strict lookup prevent a newly introduced line or renamed selection from silently inheriting capital.

Operational recommendations:

1. **Adopt asymmetric core as the default candidate.** It is predeclared, improves both models, survives the time split, and exceeds Sharpe 1.5 on converged `m12`.
2. **Do not enable every under.** Under 1.5 and Under 3.5 dilute the joint solve.
3. **Do not deploy pure alpha yet.** Treat it as a lower-turnover research portfolio pending nested-history or forward validation.
4. **Display the gate direction on the MatchDay ticket.** “O/U 2.5 enabled” is insufficient; the operator needs “Under 2.5 enabled / Over 2.5 disabled.”
5. **Monitor directional calibration and CLV separately.** Pooling Over and Under into one O/U metric hides the failure documented here.
6. **Preserve the joint slate optimizer.** Directional deletion changes all surviving stakes. Do not subtract losing Over 2.5 P&L from a symmetric ledger and call it the counterfactual.
7. **Require `m12` convergence as the production evidence base.** Do not use `m13`'s slightly stronger sensitivity result to relax convergence standards.

---

## 9. Limitations

- The policies use closing Betfair prices and idealised backtest execution rather than a live point-in-time ladder fill process.
- The predeclared directional hypotheses were motivated by inspection of historical ledgers. The second-half comparison improves credibility but is not a fully nested policy-selection design.
- Pure alpha is explicitly selected and scored on the same period after searching 8,191 subsets.
- The public-bias explanation is plausible but not causally identified by these data.
- Selection-level counts are modest for some directions; pooled Over 2.5 has 115 bets.
- The two models are closely related and use the same match set, so their agreement is sensitivity replication, not two independent samples.
- `m13` fails the aggregate tail-ESS gate.
- Results are specific to the Scottish Lower segment, this price estimator, 2% commission, 0.30 trust, `SlateDrawdown(23)`, and a 20% cap.

---

## 10. Reproduction and artifacts

Run from artifact-compatible source commit `784c8ea81328760e75498b19d13c2dab762bde8e` with database credentials supplied by `~/.pgpass`:

```bash
julia --project -t 8 eda/eda_asymmetric_selection_trust.jl
```

Task script:

- `eda/eda_asymmetric_selection_trust.jl`

Generated files under `eda/results/asymmetric_trust/`:

| File | Contents |
|---|---|
| `asymmetric_policy_summary.csv` | full-period bankroll, return, Sharpe, drawdown, bets, turnover, cap use |
| `asymmetric_policy_windows.csv` | full, first-half, and second-half policy metrics |
| `asymmetric_policy_daily.csv` | all 1,000 model-policy slate states |
| `asymmetric_policy_ledger.csv` | selection-level stakes, prices, probabilities, and realised P&L |
| `asymmetric_selection_summary.csv` | model-policy-direction P&L, win rate, ROI, and edge |
| `asymmetric_pure_alpha_scores.csv` | standalone directional information-ratio inputs by model and pooled |
| `asymmetric_subset_search.csv` | all 8,191 candidate subsets and their diagnostic information ratios |
| `asymmetric_policy_definitions.csv` | explicit trust for every policy × direction |
| `asymmetric_build_report.csv` | immutable run hashes and convergence/build gates |

The runner validates that every policy spans exactly 100 dates, ledger counts and turnover reproduce engine summaries, final bankrolls reconcile, and no gated direction receives a stake.
