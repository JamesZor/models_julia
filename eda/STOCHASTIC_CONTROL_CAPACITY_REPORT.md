# Stochastic Optimal Control and Market-Capacity Cannibalization Audit

## Executive summary

This audit tests the proposition that a weak market line can damage a joint Kelly portfolio twice: first through its own realised P&L, and second by occupying scarce slate risk capacity that could have supported stronger selections.

The result is affirmative for the six-market Scottish Lower simulation used here.

- In the status-quo six-market policy, only **16/100** (`m12`) and **18/100** (`m13`) slates crossed the pre-declared constrained threshold of 80% of the 20% cap. On those slates, fringe markets consumed **29.17%** and **28.52%** of stake respectively.
- That constrained-slate fringe capital returned **-13.19%** (`m12`) and **-13.05%** (`m13`) per unit staked, while core 1X2/O-U 2.5 capital returned **+12.62%** and **+11.27%**.
- A genuine core-only re-solve, rather than subtraction of fringe P&L, raised terminal return from **+123.94% to +141.53%** for `m12` and from **+127.86% to +144.43%** for `m13`.
- Replacing the status-quo solution with the core-only solution *only on constrained slates* would have raised terminal return by **+26.46 percentage points** for `m12` and **+28.81 pp** for `m13`. The realised incremental P&L was about **13.1% per unit of fringe stake removed**.
- Hard pruning and a 0.05 tail trust were almost tied. Averaged over `m12` and `m13`, hard pruning produced **+142.98%**, Sharpe **1.503**, and 1,315 bets; damped tails produced **+143.43%**, Sharpe **1.489**, and 1,893.5 bets. The tiny return difference changes sign by model, while hard pruning has the better average Sharpe, lower turnover, and simpler operational contract.
- Uniform drawdown-adaptive trust was effectively a null control: average terminal return was **+124.89%**, below status quo's **+125.90%**. This is consistent with the near-homogeneity of the existing `SlateDrawdown` solve: scaling every arm together usually does not change relative allocation and often does not change final stakes.

**Production recommendation:** use `SelectionTrust` with 0.30 trust on 1X2 and O/U 2.5 and zero trust on O/U 0.5, 1.5, 3.5 and BTTS as the conservative default. If retaining exploration is strategically useful, cap fringe trust at 0.05, log it as an explicit exploration budget, and evaluate it prospectively. Do not deploy the tested uniform drawdown scaler as a meaningful controller.

Two qualifications are important. First, the canonical persisted portfolio ledgers do not contain all requested deep-total lines; the six-market capacity result is therefore a controlled re-simulation from immutable saved fits, not a claim that those lines were present in the old ledgers. Second, `m13` failed the aggregate strict convergence gate on tail ESS (38/40 folds individually converged), so its figures are sensitivity evidence; the converged `m12` result is the primary production result.

---

## 1. Question and stochastic-control framing

Let a daily slate be state \(s_t\), containing bankroll, opening drawdown, available quotes, posterior score distributions, and any already committed positions. An action \(a_t=(f_{t1},\ldots,f_{tn})\) is the vector of bankroll fractions assigned to selections. The portfolio chooses the vector jointly, subject to a slate capacity constraint

\[
\sum_i f_{ti} \le C_t, \qquad C_t=0.20,
\]

and the `SlateDrawdown(23)` risk constraint. With state-contingent net return vector \(R_t\), the one-step log-growth objective has the generic form

\[
\max_{f_t\ge 0}\; \mathbb E_t\left[\log\left(1+R_t^\top f_t\right)\right].
\]

The cap Lagrangian is

\[
\mathcal L(f_t,\lambda_t)=
\mathbb E_t[\log(1+R_t^\top f_t)]
-\lambda_t\left(\sum_i f_{ti}-C_t\right),
\]

with Karush-Kuhn-Tucker condition, for an interior active selection,

\[
\frac{\partial}{\partial f_{ti}}
\mathbb E_t[\log(1+R_t^\top f_t)] = \lambda_t.
\]

When the cap binds, \(\lambda_t>0\) is the shadow value of one additional unit of capacity. A low-efficiency arm can therefore be costly even if its standalone expectation is near zero: its stake must clear the same marginal-value threshold, and its covariance with other bets changes the optimal vector. This is the Bandits-with-Knapsacks interpretation: markets are arms, but pulls consume a shared, perishable resource and their rewards are correlated through the match score and the common bankroll.

This also explains the counterfactual used in this report. The opportunity cost is **not** `minus fringe P&L`. Removing an arm changes the feasible set and requires the Kelly problem to be solved again. All reported policy counterfactuals are full portfolio re-solves.

---

## 2. Data, immutable addresses, and experimental contract

### 2.1 Canonical model runs

The scripts resolve runs by immutable UUID and verify each persisted `config_hash` before use.

| Model | Experiment | Run UUID | Folds converged | Interpretation |
|---|---|---|---:|---|
| `m00_joint_baseline` | `scottish_lower_joint_2426` | `2c6e859c-29e7-4ae7-aa0a-e88343ba7672` | 40/40 | calibration control |
| `m05_joint_production_wealth` | `scottish_lower_joint_2426` | `5eff755c-3591-48d1-a2cc-5fc2744ddf88` | 40/40 | team-state control |
| `m12_joint_hybrid_synergy` | `scottish_lower_joint_player_2426` | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | 40/40 | primary policy model |
| `m13_joint_composite` | `scottish_lower_joint_player_2426` | `5474e824-8c9d-4613-8e39-841426c3f80f` | 38/40 | sensitivity only; aggregate tail-ESS failure |

No model was refitted.

### 2.2 Prices, fixtures, book, and policy constants

The controlled simulations hold fixed:

- Scottish Lower held-out fixtures from the canonical 40-fold fits;
- Betfair exchange close estimated by time-weighted average over the final 20 minutes before kickoff;
- markets: 1X2, O/U 0.5, 1.5, 2.5, 3.5, and BTTS;
- `DeArb()`, `KellyLogUtility()`, `NoShrinkage()`;
- 2% per-bet commission, 0.001 minimum selection stake;
- `SlateDrawdown(23.0)`, `FixedCap(0.20)`, `DailySlate()`;
- initial bankroll 1,000;
- no bootstrap uncertainty in the headline path.

The simulation built 632 match books over 100 slate dates for each Gen-4 model. Seventy-eight of 710 fitted fixtures lacked a usable controlled book. `m12` passed all build/convergence gates. `m13` was run with `require_converged=false` to expose its sensitivity result rather than silently discarding it.

### 2.3 Database-ledger limitation

All four persisted canonical portfolios contain only:

- 1X2;
- O/U 2.5;
- BTTS.

They contain no O/U 0.5, 1.5, or 3.5 positions. Consequently, querying `portfolio_bets` is useful for provenance and for auditing the old three-market allocation, but it cannot answer the full six-market cannibalization question. The scripts write the complete query result to `canonical_database_bets.csv` and the old capacity summary to `canonical_portfolio_capacity.csv`; they then answer the requested six-market question by rebuilding books from the exact saved fits.

This distinction prevents an invalid inference from absent data.

### 2.4 Artifact-compatible runtime

The saved fits were serialized before `JointGammaPoissonObservation` acquired its current type parameter. Direct deserialization at the branch's current source revision therefore fails for a schema-compatibility reason. The EDA was executed against detached repository commit `784c8ea81328760e75498b19d13c2dab762bde8e`, which is compatible with those immutable artifacts, using the current project dependency environment. No fit bytes were edited and no compatibility shim was committed. The analysis scripts remain on the requested branch.

---

## 3. Capacity cannibalization

A slate is classified as constrained when realised total exposure is at least

\[
0.80\times C_t = 0.80\times0.20=0.16.
\]

“Core” means 1X2 and O/U 2.5. “Fringe” means O/U 0.5, 1.5, 3.5, and BTTS.

### 3.1 Stake and efficiency by regime

| Model | Regime | Slates | Bucket | Bets | Stake share | Win rate | P&L / stake |
|---|---|---:|---|---:|---:|---:|---:|
| `m12` | constrained | 16 | core | 306 | 70.83% | 33.33% | **+12.62%** |
| `m12` | constrained | 16 | fringe | 183 | **29.17%** | 39.89% | **-13.19%** |
| `m12` | unconstrained | 84 | core | 1,008 | 79.91% | 33.23% | +8.14% |
| `m12` | unconstrained | 84 | fringe | 397 | 20.09% | 43.58% | +3.56% |
| `m13` | constrained | 18 | core | 349 | 71.48% | 33.81% | **+11.27%** |
| `m13` | constrained | 18 | fringe | 196 | **28.52%** | 39.29% | **-13.05%** |
| `m13` | unconstrained | 82 | core | 967 | 80.15% | 33.92% | +8.65% |
| `m13` | unconstrained | 82 | fringe | 381 | 19.85% | 44.36% | +4.95% |

The interaction with state is the central finding. Fringe positions were mildly profitable in unconstrained states but strongly destructive in constrained states. Their higher nominal win rate is not contradictory: odds and payoff asymmetry, not win rate alone, determine return. The same family of arms that looked tolerable when capacity was slack consumed almost 30% of scarce stake on tight slates and lost approximately 13 cents per unit allocated.

### 3.2 True opportunity cost from a re-solve

| Model | Constrained slates | Status-quo full return | Core-only full return | Full-return delta | Status return on constrained dates | Core return on constrained dates | Binding-only hybrid return | Hybrid delta | Realised shadow value / fringe stake removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m12` | 16/100 | +123.94% | +141.53% | **+17.60 pp** | +17.47% | +31.35% | +150.40% | **+26.46 pp** | **+13.12%** |
| `m13` | 18/100 | +127.86% | +144.43% | **+16.57 pp** | +15.97% | +30.64% | +156.67% | **+28.81 pp** | **+13.23%** |

The “binding-only hybrid” uses the core-only policy on status-quo constrained dates and the status-quo policy everywhere else. It is an ex-post diagnostic, not a deployable oracle, because the classification uses the status-quo slate solution. It answers the causal accounting question: where did most of the damage occur? The answer is the small set of capacity-tight states.

Removing fringe stake did **not** redirect all released nominal capacity into more core stake. Across constrained dates the status-quo fringe stakes summed to 0.860 and 0.904 bankroll fractions, while incremental core stake was only 0.083 for each model—about **9.64%** and **9.14%** of removed stake. This is expected under a joint drawdown-constrained log-utility solve. The gain came from removing poor correlated exposures and changing the feasible risk geometry, not from mechanically transferring every pound to a surviving leg.

The reported shadow-value proxy is realised incremental constrained-date P&L divided by removed fringe stake. It is useful economic accounting, but it is not the optimizer's ex-ante KKT multiplier and should not be represented as one.

---

## 4. Information geometry and calibration across market lines

### 4.1 Alignment and sample sizes

For each of the four models, the evaluation context audited:

- 14,617 available close-odds rows;
- 710 priced fixtures;
- 4,409 aligned selection rows;
- 4,390 scored rows;
- 19 rows without an outcome;
- zero duplicate keys and zero mismatched IDs.

Every alignment audit passed. Selection-row counts vary by market because historical Betfair availability differs by line.

### 4.2 Brier and ECE for the two policy models

The table compares model probabilities with the de-vigged Betfair close on exactly the same scored rows. “Δ Brier” is model minus market, so negative is better.

| Model | Market | Scored rows | Model Brier | Market Brier | Δ Brier | Model ECE | Market ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| `m12` | 1X2 | 1,785 | 0.21311 | 0.21156 | +0.00155 | **0.0155** | 0.0186 |
| `m12` | O/U 0.5 | 533 | **0.05144** | 0.05434 | -0.00290 | **0.0143** | 0.0364 |
| `m12` | O/U 1.5 | 430 | 0.17424 | **0.17266** | +0.00158 | 0.0195 | **0.0103** |
| `m12` | O/U 2.5 | 758 | **0.24711** | 0.24829 | -0.00119 | **0.0100** | 0.0183 |
| `m12` | O/U 3.5 | 528 | 0.21191 | **0.20987** | +0.00205 | **0.0116** | 0.0148 |
| `m12` | BTTS | 356 | 0.24605 | **0.24516** | +0.00089 | **0.0087** | 0.0300 |
| `m13` | 1X2 | 1,785 | 0.21299 | 0.21156 | +0.00143 | **0.0125** | 0.0186 |
| `m13` | O/U 0.5 | 533 | **0.05143** | 0.05434 | -0.00290 | **0.0143** | 0.0364 |
| `m13` | O/U 1.5 | 430 | 0.17413 | **0.17266** | +0.00147 | 0.0192 | **0.0103** |
| `m13` | O/U 2.5 | 758 | **0.24718** | 0.24829 | -0.00111 | **0.0035** | 0.0183 |
| `m13` | O/U 3.5 | 528 | 0.21178 | **0.20987** | +0.00191 | **0.0116** | 0.0148 |
| `m13` | BTTS | 356 | 0.24603 | **0.24516** | +0.00087 | **0.0147** | 0.0300 |

The two control fits were also scored, not omitted:

| Model | Market | Model Brier | Market Brier | Model ECE | Market ECE |
|---|---|---:|---:|---:|---:|
| `m00` | 1X2 | 0.21362 | 0.21156 | 0.0206 | 0.0186 |
| `m00` | O/U 0.5 | 0.05146 | 0.05434 | 0.0146 | 0.0364 |
| `m00` | O/U 1.5 | 0.17292 | 0.17266 | 0.0235 | 0.0103 |
| `m00` | O/U 2.5 | 0.24700 | 0.24829 | 0.0081 | 0.0183 |
| `m00` | O/U 3.5 | 0.21053 | 0.20987 | 0.0320 | 0.0148 |
| `m00` | BTTS | 0.24535 | 0.24516 | 0.0041 | 0.0300 |
| `m05` | 1X2 | 0.21298 | 0.21156 | 0.0220 | 0.0186 |
| `m05` | O/U 0.5 | 0.05145 | 0.05434 | 0.0143 | 0.0364 |
| `m05` | O/U 1.5 | 0.17308 | 0.17266 | 0.0182 | 0.0103 |
| `m05` | O/U 2.5 | 0.24696 | 0.24829 | 0.0100 | 0.0183 |
| `m05` | O/U 3.5 | 0.21063 | 0.20987 | 0.0311 | 0.0148 |
| `m05` | BTTS | 0.24551 | 0.24516 | 0.0040 | 0.0300 |

Across all four models, mean model ECE was 0.0177 for 1X2, 0.0144 for O/U 0.5, 0.0201 for O/U 1.5, 0.0079 for O/U 2.5, 0.0216 for O/U 3.5, and 0.0079 for BTTS. These pooled figures must not be mistaken for investability. Brier and ECE score all quoted outcomes; the portfolio selects conditional tails of the edge distribution, where small local probability errors are amplified by odds and by selection.

### 4.3 Reliability details that matter for deep totals

For `m12` (`m13` is nearly identical):

| Selection | n | Mean model probability | Realised rate | Bias (realised - predicted) | ECE |
|---|---:|---:|---:|---:|---:|
| O/U 0.5 over | 384 | 0.9317 | 0.9375 | +0.0058 | 0.0058 |
| O/U 0.5 under | 149 | 0.0697 | 0.0336 | **-0.0361** | **0.0361** |
| O/U 1.5 over | 215 | 0.7526 | 0.7721 | +0.0195 | 0.0195 |
| O/U 1.5 under | 215 | 0.2474 | 0.2279 | -0.0195 | 0.0195 |
| O/U 2.5 over | 379 | 0.5097 | 0.5013 | -0.0084 | 0.0100 |
| O/U 2.5 under | 379 | 0.4903 | 0.4987 | +0.0084 | 0.0100 |
| O/U 3.5 over | 264 | 0.2952 | 0.3068 | +0.0116 | 0.0116 |
| O/U 3.5 under | 264 | 0.7048 | 0.6932 | -0.0116 | 0.0116 |
| BTTS yes | 178 | 0.5461 | 0.5506 | +0.0045 | 0.0087 |
| BTTS no | 178 | 0.4539 | 0.4494 | -0.0045 | 0.0087 |

The clearest tail failure is under 0.5: the model assigns about 6.97% on average where only 3.36% realise. Yet the all-selection O/U 0.5 Brier is good because the complementary, common over outcome dominates the row count and is easy to predict. A low aggregate Brier therefore coexists with a badly distorted rare side—the side whose long odds can create apparently attractive Kelly edges.

Full ten-bin curves for every model, market, side, and both model/market sources are in `line_reliability_curves.csv`. Sparse edge bins should not be over-read; several apparent maximum gaps come from one to three observations. The weighted ECE and side-level bias are the safer summaries.

### 4.4 Why deep totals amplify latent-intensity error

Let total goals \(N\mid\Lambda\sim\operatorname{Poisson}(\Lambda)\). An under \(K+0.5\) probability is

\[
F_K(\Lambda)=P(N\le K\mid\Lambda)
=e^{-\Lambda}\sum_{j=0}^{K}\frac{\Lambda^j}{j!}.
\]

Its derivative is

\[
F_K'(\Lambda)=-P(N=K\mid\Lambda)
=-e^{-\Lambda}\frac{\Lambda^K}{K!}.
\]

Thus an intensity error \(\delta\) causes first-order probability error

\[
\Delta F_K\approx -P(N=K\mid\Lambda)\,\delta.
\]

For the under-0.5 tail, \(F_0(\Lambda)=e^{-\Lambda}\), so

\[
\frac{\Delta F_0}{F_0}\approx -\delta.
\]

A seemingly modest additive miss in total intensity becomes a comparable *relative* miss in a rare probability. Price and Kelly sizing then magnify it. At decimal odds \(o\), the binary Kelly fraction is

\[
f^*(p,o)=\frac{op-1}{o-1},
\qquad
\frac{\partial f^*}{\partial p}=\frac{o}{o-1}.
\]

More importantly, the sign of the bet is governed by \(op-1\). Near the market break-even probability \(1/o\), a probability bias of only a few percentage points can manufacture an edge and flip a selection from zero stake to positive stake. Under 0.5 exhibits exactly this pattern: mean model probability 0.0697 versus realised 0.0336.

There is a second, specifically Bayesian distortion. The posterior predictive tail is a mixture

\[
P(N\le K)=\mathbb E[F_K(\Lambda)].
\]

For \(K=0\),

\[
P(N=0)=\mathbb E[e^{-\Lambda}]\ge e^{-\mathbb E[\Lambda]}
\]

by Jensen's inequality because \(e^{-\Lambda}\) is convex. Posterior uncertainty therefore increases predicted zero-goal mass relative to a plug-in mean. This is mathematically coherent if the mixing distribution is calibrated, but any excess dispersion or unmodelled structural heterogeneity inflates the rare tail. For larger \(K\), the curvature

\[
F_K''(\Lambda)=P(N=K\mid\Lambda)\left(1-\frac{K}{\Lambda}\right)
\]

changes sign around \(\Lambda=K\), making the effect line- and intensity-dependent. One global latent score distribution can consequently look acceptable near the central 2.5 line while distorting 0.5, 1.5, or 3.5 tails.

Finally, totals at different strikes and BTTS are deterministic transformations of the same score grid. Treating them as independent arms overstates information breadth: adding lines adds correlated expressions of almost the same latent risk. The joint optimizer accounts for payoff covariance, but probability bias shared across these transformations can still put several apparently distinct fringe bets on the same wrong latent-intensity view. This is why line-specific trust is a defensible control layer.

---

## 5. Policy A/B test

### 5.1 Policies

| Policy | Core trust | Fringe trust | State dependence |
|---|---:|---:|---|
| P1 status quo | 0.30 | 0.30 | none |
| P2 hard pruning | 0.30 | 0.00 | none |
| P3 damped tails | 0.30 | 0.05 | none |
| P4 drawdown adaptive | 0.30 base | 0.30 base | multiply all trust by 1.00/0.75/0.50/0.25 at opening drawdowns of 0/5/10/15% |

### 5.2 Results

| Model | Policy | Final bankroll | Return | Sharpe | Max DD | Bets | Turnover | Cap-binding slates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `m12` | P1 status quo | 2,239.37 | +123.94% | 1.333 | -20.97% | 1,894 | 9.774 | 4 |
| `m12` | P2 hard pruning | 2,415.34 | **+141.53%** | **1.485** | -19.80% | 1,314 | 7.732 | 0 |
| `m12` | P3 damped tails | 2,413.60 | +141.36% | 1.467 | **-19.78%** | 1,894 | 8.208 | 0 |
| `m12` | P4 adaptive | 2,231.79 | +123.18% | 1.338 | -20.79% | 1,894 | 9.709 | 4 |
| `m13`* | P1 status quo | 2,278.59 | +127.86% | 1.375 | -20.84% | 1,893 | 9.807 | 4 |
| `m13`* | P2 hard pruning | 2,444.27 | +144.43% | **1.522** | **-20.32%** | 1,316 | 7.775 | 0 |
| `m13`* | P3 damped tails | 2,454.93 | **+145.49%** | 1.512 | -20.54% | 1,893 | 8.246 | 0 |
| `m13`* | P4 adaptive | 2,265.92 | +126.59% | 1.376 | -20.68% | 1,893 | 9.748 | 4 |

\* `m13` failed aggregate strict convergence on tail ESS and is sensitivity evidence.

Across the two models, P2 improved average return by **17.08 pp**, improved average Sharpe from **1.354 to 1.503**, reduced average maximum drawdown by about **0.85 pp**, cut turnover by **20.8%**, and placed about **31% fewer bets** than P1. P3 improved average return by **17.53 pp** with somewhat more turnover and a slightly lower average Sharpe than P2.

The ranking does not justify claiming that 0.05 is the uniquely optimal tail trust. P2 beats P3 by 0.17 pp on `m12`; P3 beats P2 by 1.07 pp on non-converged `m13`. Those differences are small relative to policy uncertainty and were measured on the same history used to motivate the policies. The robust claim is coarser: **0.30 uniform fringe trust is too high; zero to 0.05 is materially better.**

### 5.3 Why the drawdown controller was nearly neutral

P4's multiplier was below one on 35 `m12` slates and 32 `m13` slates, but final exposure/P&L changed on only ten slates for each model. The existing drawdown solver frequently scales a candidate vector to the risk boundary. If all preliminary stakes are multiplied by a common factor \(c\), the solver can offset that with approximately \(k/c\), leaving the final action \(kcf\) unchanged. Uniform scaling therefore provides little control authority in the binding regime.

A useful state-dependent controller must alter something that is not cancelled by this homogeneity—for example the cap, the drawdown risk parameter, market-specific trust ratios, an explicit exploration budget, or a turnover/liquidity constraint. P4 as specified should not be promoted merely because it is “adaptive.”

---

## 6. Production MatchDay recommendations

1. **Default to hard pruning for the current instrument set.** Implement strict `SelectionTrust` with 0.30 on 1X2 and O/U 2.5 and 0.00 on O/U 0.5, 1.5, 3.5 and BTTS. Keep `strict=true` so a newly introduced selection cannot silently inherit trust.
2. **Treat 0.05 fringe trust as exploration, not ordinary production risk.** If retained, label and cap it separately. P3's small terminal-return advantage averaged across the two models is not enough to outweigh P2's higher Sharpe, lower turnover, simpler ticket, and convergence-clean support from `m12`.
3. **Preserve the joint slate solve.** Do not rank bets independently or redirect deleted stake pro rata. Only about 9% of removed constrained-slate fringe stake reappeared as additional core stake; the benefit came from joint risk geometry.
4. **Expose capacity diagnostics in MatchDay.** For each priced slate, log total exposure, fraction of cap used, stake by market line, and whether cap or drawdown risk binds. A useful operator field is the counterfactual core-only stake vector beside the full vector.
5. **Do not call the realised 13.1% ratio a live shadow price.** If an ex-ante shadow price is required, instrument the allocator to return the KKT multiplier or estimate a finite difference by re-solving at \(C_t\pm\epsilon\).
6. **Do not deploy uniform trust drawdown scaling.** Adapt market ratios or actual constraints instead. Any adaptive rule must use opening state only and be replay-tested to preserve causality.
7. **Add line-specific calibration before reconsidering deep totals.** O/U 0.5 under needs explicit rare-event calibration; totals strikes should be calibrated jointly or with monotonicity constraints so probabilities remain coherent across lines.
8. **Require convergence for production.** `m13` should not become the operational basis of a policy claim until its two failed folds / aggregate tail-ESS gate are resolved. `m12` provides the clean evidence here.
9. **Run a prospective or nested-history policy validation.** These A/B policies were motivated by prior analysis of the same historical period. A locked forward period, or fold-local line-policy estimation, is required before assigning a causal expected-return uplift to production.
10. **Keep database truth explicit.** The old canonical ledgers are three-market results. Store any new six-market portfolio with its own `BookSpec`, `PolicySpec`, run UUID, portfolio UUID, and fill assumptions rather than overwriting or conflating histories.

---

## 7. Limitations

- The controlled six-market A/B uses closing Betfair prices and idealised backtest execution, not a point-in-time live order-book fill process.
- The same historical sample informed the market-line hypothesis and evaluates these policies. The large P1-to-P2/P3 difference is compelling diagnostic evidence, not an unbiased estimate of future uplift.
- Only 16-18 slates meet the constrained threshold, so capacity-state estimates have limited effective sample size.
- Market-line quotes are not available for every fixture; line sample sizes differ.
- Brier and ECE are marginal proper-score diagnostics. They do not directly estimate conditional edge quality among selected bets.
- Reliability deciles are sparse at extreme probabilities. Weighted summaries are more stable than maximum bin gaps.
- The “binding-only hybrid” uses an ex-post status-quo classification and is not itself a causal online policy.
- `m13` does not pass the strict convergence gate.
- Serialized-fit compatibility required execution at artifact-compatible source commit `784c8ea8`; this was a source-schema issue, not a refit.

---

## 8. Reproduction and artifact map

Run in the artifact-compatible source worktree with database credentials supplied by `~/.pgpass`:

```bash
julia --project -t 8 eda/eda_policy_ab_test.jl
julia --project -t 8 eda/eda_capacity_cannibalization.jl
```

Primary scripts:

- `eda/stochastic_control_common.jl`
- `eda/eda_policy_ab_test.jl`
- `eda/eda_capacity_cannibalization.jl`

Generated evidence in `eda/results/stochastic_control_capacity/`:

| File | Purpose |
|---|---|
| `canonical_run_inventory.csv` | immutable run UUIDs, hashes, folds, convergence |
| `canonical_portfolio_inventory.csv` | exact persisted portfolio recipes and headline results |
| `canonical_database_bets.csv` | queried canonical `portfolio_bets` rows |
| `canonical_portfolio_capacity.csv` | capacity audit of the old three-market portfolios |
| `policy_definitions.csv` | exact P1-P4 definitions |
| `policy_ab_build_report.csv` | book-build and convergence gates |
| `policy_ab_summary.csv` | headline A/B metrics |
| `policy_ab_daily.csv` | slate states and causal drawdown multipliers |
| `policy_ab_ledger.csv` | controlled selection-level stake and P&L ledger |
| `capacity_segment_summary.csv` | core/fringe metrics by constrained regime |
| `capacity_opportunity_cost.csv` | full re-solve and binding-only opportunity costs |
| `calibration_alignment_audit.csv` | matching/scoring integrity checks |
| `line_calibration_summary.csv` | Brier, ECE, mean calibration by model/line/side/source |
| `line_reliability_curves.csv` | ten-bin reliability curves |

The CSVs are retained so every headline number can be reduced independently without querying credentials or relying on prose.
