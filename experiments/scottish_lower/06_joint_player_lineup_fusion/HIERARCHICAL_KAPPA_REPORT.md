# Hierarchical Team Kappa — 40-Fold Out-of-Sample Evaluation and Portfolio Backtest

**Verdict: DO NOT ADOPT.** `HierarchicalKappa` is correctly implemented, samples better than the
component it replaces, and finds nothing. Across **957 team-fold pairs** — 40 walk-forward folds ×
22–25 teams × two candidates — **not one team's 90% HPDI on its finishing delta excludes zero**.
The out-of-sample proper scores are unchanged to the fifth decimal, the paired LogLoss contrast is
`p = 0.94` and `p = 0.98`, the model-market edge coefficient does not increase, and the fractional-Kelly
backtest **loses 4.5–6.0 points of terminal bankroll in all four model × configuration pairings**. The
component costs `n_teams + 1` extra parameters and **2.1–2.3× the sampling wallclock** to deliver that.

The one statistically clear effect it has is mechanical: partial pooling moves the model's prices
*towards* the closing line (`p < 0.001` in every split), which leaves fewer selections clearing the
Kelly threshold and less capital deployed at an essentially unchanged per-unit edge.

Keep the code. Do not put it in the production recipe.

---

## 1. What was tested, and against what

Two hierarchical candidates, each paired with a shared-κ control that differs from it in **exactly
one component**. Everything else — the two-arm `JointGammaPoissonObservation`, the proxy-xG feature,
`TimeDecayDynamics(180)`, the shots-RAPM pillar, the production-wealth covariate, the 40-boundary
splitter, the book and the policy — is the same object, taken from `l60_loader.jl` rather than
restated, so a difference in a measured number cannot come from a recipe that quietly drifted.

| Arm | Shared-κ control | Hierarchical candidate | What changes |
|---|---|---|---|
| `m05` | `m05_joint_production_wealth` | `m05_hierarchical_kappa` | team state + wealth; κ becomes per-team |
| `m12` | `m12_joint_hybrid_synergy` | `m12_hierarchical_kappa` | + shots-RAPM starters and bench 0.10; κ becomes per-team |

### The component

```
shared        log κ                                            ~ Normal(0, 0.20)
hierarchical  log κ_t = log κ + σ_κ · (raw_t − mean(raw)),  σ_κ ~ truncated(Normal(0, 0.10), 0, ∞)

arm 1 (proxy xG)   pxg_s ~ Gamma(ν, μ_s / ν)          — UNCHANGED, no κ
arm 2 (goals)      y_s   ~ Poisson(κ_{s(i)} · μ_s)    — the home side's goals are converted by the
                                                        HOME side's factor; κ is a property of who
                                                        is shooting, not of the fixture
```

κ stays out of the Gamma arm deliberately. `μ` is what the proxy measures, and the identification
argument for κ is that the proxy is unbiased for `μ`; letting κ into the Gamma arm would make it a
rescale of the latent and σ_κ would then be fitting the pxG measurement scale rather than finishing.

`δ_κ` is a contrast set by construction and sums to exactly zero in **every draw**, so `log κ` and
the deltas are not trading against each other — the smoke test asserts this to `1e-10`.

### The universe

| | |
|---|---|
| Segment | `ScottishLower` — Scottish League One and League Two, tournaments 56 and 57 |
| Target seasons | 24/25 and 25/26 |
| Folds | 40 walk-forward match-biweek boundaries (`GroupedCVConfig`, 2 history seasons) |
| Held-out fixtures | 710 |
| Scored market observations | 2,899 over **627 distinct fixtures** |
| Benchmark | Betfair TWA closing line, (−20 m, 0 m], proportionally de-vigged |
| Sampler | `QueuedNUTSConfig`, 4 chains × 800 warmup × 800 retained; target acceptance **0.90** hierarchical, **0.65** shared |
| Draws per fold | 3,200 · **128,000 per model over the grid** |

### Run addresses

| Model | `runs.run_id` |
|---|---|
| `m05_joint_production_wealth` | `ed541a7c-01e2-447e-a771-783517728d47` |
| `m05_hierarchical_kappa` | `b3e19ad4-f755-4b89-addd-ff7592787deb` |
| `m12_joint_hybrid_synergy` | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` |
| `m12_hierarchical_kappa` | `a0847873-de69-4e25-824f-c03e4a4fd8c4` |

Runners: `r66_compare_hierarchical_kappa.jl` (scores, GLM edge, finishing factor),
`r67_portfolio_hierarchical_kappa.jl` (portfolio and persistence), sharing
`l66_hierarchical_kappa_eval_loader.jl`. Artefacts under `results/hierarchical_kappa/`.

---

## 2. Convergence — the one place the component wins

| Model | Folds | max R̂ | min bulk ESS | min tail ESS | Divergences | Non-converged folds | Wallclock |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m05_joint_production_wealth` | 40 | 1.00839 | 647 | **252** | **1** | **1** | 2.31 h |
| `m05_hierarchical_kappa` | 40 | 1.00974 | 722 | **500** | **0** | 0 | 5.26 h |
| `m12_joint_hybrid_synergy` | 40 | 1.01045 | 881 | 634 | **3** | 0 | 3.18 h |
| `m12_hierarchical_kappa` | 40 | 1.00912 | 910 | 379 | **0** | 0 | 6.80 h |

Both hierarchical grids are **clean: zero divergent transitions in 128,000 draws each**, against 1
and 3 for their controls. `m05`'s control also has one fold that fails the stored convergence audit
on tail ESS (252 against a 300 floor); the hierarchical version reaches 500 on its worst fold.

**Read this carefully — it is confounded and not a finding about the component.** The hierarchical
sampler runs at target acceptance **0.90** and the controls at **0.65**. The raised target was set by
the r64 smoke, which found one divergence in the weakly identified σ_κ geometry at 0.65. Smaller
steps buy cleaner trajectories in any geometry. What this table establishes is that a funnel-prone
`n_teams + 1` block **can** be sampled to a strict zero-divergence gate at a sane budget — not that
adding it improves sampling. The honest cost is the last column: **2.28× and 2.14× the wallclock**
for a posterior that prices identically.

---

## 3. Proper scoring rules — the scores do not move

All 710 held-out fixtures, 2,899 scored observations, against the Betfair TWA closing line.

### 3.1 Full sample

| Model | LogLoss | Δ vs close | Brier | RPS | CRPS | ECE | MCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m05_hierarchical_kappa` | **0.642979** | +0.001164 | **0.225863** | 0.224178 | **0.626881** | 0.013341 | 0.18378 |
| `m05_joint_production_wealth` | 0.642986 | +0.001171 | 0.225864 | **0.224153** | 0.627046 | 0.014930 | **0.06489** |
| `m12_hierarchical_kappa` | 0.643309 | +0.001494 | 0.226015 | 0.224468 | 0.628030 | **0.009046** | 0.38987 |
| `m12_joint_hybrid_synergy` | 0.643370 | +0.001554 | 0.226046 | 0.224472 | 0.628319 | 0.009956 | 0.19640 |
| *Betfair closing line* | *0.641816* | — | *0.225293* | *0.211104* | — | *0.013907* | *0.32902* |

The control numbers reproduce `results/r62_proper_scores.csv` **bit-identically** — every one of
LogLoss, Brier, RPS, CRPS, ECE, MCE and the Betfair benchmark matches to the last stored digit of a
`Float64`. That is the check that says this evaluation and the recorded Experiment 06 leaderboard are
scoring the same thing, and it is also what validates the artefact compatibility shim in §8.

**The differences are in the fifth decimal place.** `m05` gains 0.0000068 LogLoss; `m12` gains
0.0000605. For scale, the gap from the shared-κ control to the Betfair closing line is 0.00117 —
**seventeen to a hundred and seventy times larger** than what the hierarchy moves.

### 3.2 Is it bigger than the noise? No.

A mean over 2,899 observations is not 2,899 independent numbers: the home, draw and away prices on
one fixture are three views of one scoreline. The paired LogLoss difference is therefore tested with
the **fixture** as the unit of evidence (627 clusters).

| Arm | Split | N obs | Clusters | Control | Hierarchical | Δ LogLoss | SE | t | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m05` | all | 2,899 | 627 | 0.642986 | 0.642979 | **−0.0000042** | 0.000205 | −0.02 | **0.9835** |
| `m05` | 24/25 | 1,372 | 318 | 0.640442 | 0.640216 | −0.0001770 | 0.000303 | −0.58 | 0.5600 |
| `m05` | 25/26 | 1,527 | 309 | 0.645272 | 0.645462 | +0.0001735 | 0.000274 | +0.63 | 0.5270 |
| `m12` | all | 2,899 | 627 | 0.643370 | 0.643309 | **+0.0000163** | 0.000228 | +0.07 | **0.9431** |
| `m12` | 24/25 | 1,372 | 318 | 0.640100 | 0.639883 | −0.0001226 | 0.000366 | −0.34 | 0.7376 |
| `m12` | 25/26 | 1,527 | 309 | 0.646308 | 0.646388 | +0.0001592 | 0.000270 | +0.59 | 0.5562 |

Negative favours the hierarchical model. Every p-value is above 0.52, and the two full-sample ones —
0.94 and 0.98 — are about as close to "no difference at all" as a test can report. The sign is not
even stable: both arms are marginally better in 24/25 and marginally worse in 25/26.

### 3.3 How similar are the prices, actually?

| Arm | Correlation of model probabilities | Mean \|Δp\| | Max \|Δp\| |
|---|---:|---:|---:|
| `m05` | 0.99944 | 0.0027 | 0.0150 |
| `m12` | 0.99925 | 0.0030 | 0.0160 |

The hierarchical model is, in price terms, the same model. On a 1.85 shot at 0.54 implied, a 0.003
probability shift is worth roughly one tick.

### 3.4 The calibration numbers, and the trap in MCE

ECE improves in both arms — `m05` 0.01493 → 0.01334, `m12` 0.00996 → 0.00905 — and both hierarchical
fits beat the closing line's 0.01391. **This is the only score that moves in the hierarchy's favour,
and it is not worth much**: the shift is a tenth of the ECE spread the *lineup* pillar already buys
(`m05` 0.0149 → `m12` 0.0100), and it is not accompanied by a LogLoss or Brier gain.

**MCE appears to collapse (0.065 → 0.184 for `m05`, 0.196 → 0.390 for `m12`). Ignore it.** Every one
of those maxima comes from a reliability bin holding **three or four observations**:

| Model | Binding bin | n | Predicted | Observed | ⇒ MCE |
|---|---|---:|---:|---:|---:|
| `m05_joint_production_wealth` | [0.60, 0.70) | 37 | 0.6108 | 0.6757 | 0.0649 |
| `m05_hierarchical_kappa` | [0.10, 0.20) | **3** | 0.1838 | 0.0000 | 0.1838 |
| `m12_joint_hybrid_synergy` | [0.10, 0.20) | **3** | 0.1964 | 0.0000 | 0.1964 |
| `m12_hierarchical_kappa` | [0.60, 0.70) | **3** | 0.6101 | 1.0000 | 0.3899 |

Three coin flips landing the same way is not a calibration failure, and which model draws the short
straw is an accident of where four observations fell relative to a bin edge. The 780–920 observation
bins agree between control and candidate to within 0.005 everywhere. MCE is not evidence in either
direction on this sample and is reported only so that its movement is not mistaken for one.

### 3.5 Out-of-sample temporal split

| Split | N | Betfair LogLoss | `m05` ctl | `m05` hier | `m12` ctl | `m12` hier |
|---|---:|---:|---:|---:|---:|---:|
| **24/25** | 1,372 | 0.642710 | 0.640442 | 0.640216 | 0.640100 | **0.639883** |
| **25/26** | 1,527 | **0.641012** | 0.645272 | 0.645462 | 0.646308 | 0.646388 |
| all | 2,899 | 0.641816 | 0.642986 | 0.642979 | 0.643370 | 0.643309 |

The seasons tell opposite stories, and it is not the hierarchy that separates them. **In 24/25 every
model beats the Betfair closing line** by 0.0023–0.0028 LogLoss. **In 25/26 every model loses to it**
by 0.0043–0.0054. That reversal — not the κ component — is the dominant temporal signal in this
study, and it applies equally to control and candidate. Whatever the market learned or the league
changed between the two seasons, per-team finishing does not address it: the hierarchical models lose
to the line in 25/26 by *slightly more* than their controls.

---

## 4. GLM edge — does the model know something the closing line does not?

```
logit P(win) = β₀ + β_mkt · p_fair_close + β_edge · (p_model − p_fair_close)
```

`β_edge` is the whole question. Under a market that already prices everything the model knows, the
model's disagreement is noise and β_edge = 0. Standard errors are **CR1 cluster-robust on
`match_id`**; the naive ones are shown once to make the size of the dependence correction visible.

### 4.1 Full sample, all scored selections

| Model | β_mkt | β_edge | SE naive | SE clustered | z (cl) | p (cl) | pseudo-R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m05_joint_production_wealth` | 4.501 | **+2.031** | 0.742 | 0.829 | +2.45 | **0.0143** | 0.0475 |
| `m05_hierarchical_kappa` | 4.483 | **+2.002** | 0.753 | 0.837 | +2.39 | **0.0168** | 0.0474 |
| `m12_joint_hybrid_synergy` | 4.552 | **+2.045** | 0.718 | 0.797 | +2.57 | **0.0103** | 0.0477 |
| `m12_hierarchical_kappa` | 4.539 | **+2.036** | 0.731 | 0.804 | +2.53 | **0.0114** | 0.0476 |

Two results, and they point in different directions.

**The edge is real.** All four models carry information the de-vigged closing line does not:
β_edge ≈ +2.0, clustered `p < 0.02`. Clustering matters and was worth doing — it inflates the edge
standard error by 11–12%, which is the difference between `p = 0.0143` and a naive `p = 0.0062`.

**The hierarchy does not increase it — it decreases it, in both arms.** `m05` 2.031 → 2.002,
`m12` 2.045 → 2.036. The movements are far inside their own standard errors and mean nothing
individually, but the *direction* is consistent with §4.3 below and with the point estimate never
once improving in any cut of the data.

### 4.2 By market family and season

| Model | Split | Subset | N | β_edge | SE (cl) | p (cl) |
|---|---|---|---:|---:|---:|---:|
| `m05` ctl | all | 1X2 | 1,785 | +1.921 | 1.375 | 0.162 |
| `m05` hier | all | 1X2 | 1,785 | +1.805 | 1.388 | 0.193 |
| `m12` ctl | all | 1X2 | 1,785 | +2.064 | 1.232 | 0.094 |
| `m12` hier | all | 1X2 | 1,785 | +1.985 | 1.255 | 0.114 |
| `m05` ctl | all | OU 2.5 | 758 | +3.421 | 2.481 | 0.168 |
| `m05` hier | all | OU 2.5 | 758 | +3.667 | 2.489 | 0.141 |
| `m12` ctl | all | OU 2.5 | 758 | +4.616 | 3.266 | 0.158 |
| `m12` hier | all | OU 2.5 | 758 | +5.101 | 3.251 | 0.117 |
| `m05` ctl | 24/25 | all | 1,372 | +3.006 | 1.384 | **0.030** |
| `m05` hier | 24/25 | all | 1,372 | +3.133 | 1.397 | **0.025** |
| `m12` ctl | 24/25 | all | 1,372 | +3.240 | 1.300 | **0.013** |
| `m12` hier | 24/25 | all | 1,372 | +3.371 | 1.324 | **0.011** |
| `m05` ctl | 25/26 | all | 1,527 | +1.485 | 1.049 | 0.157 |
| `m05` hier | 25/26 | all | 1,527 | +1.359 | 1.060 | 0.200 |
| `m12` ctl | 25/26 | all | 1,527 | +1.294 | 1.013 | 0.202 |
| `m12` hier | 25/26 | all | 1,527 | +1.214 | 1.018 | 0.233 |

The full `all × all × BTTS` grid is in `results/hierarchical_kappa/r66_glm_edge.csv`.

* **The edge is a 24/25 phenomenon.** β_edge ≈ +3.0 to +3.4 and significant in 24/25; ≈ +1.2 to +1.5
  and insignificant in 25/26. This is the same season reversal §3.5 found in LogLoss, arriving
  through a completely different estimator, which is what makes it a property of the data rather
  than of one metric.
* **On 1X2 alone the edge does not clear significance on the full sample** for any of the four
  models. The pooled result leans on Over/Under 2.5 and BTTS, where the point estimates are larger
  and the standard errors much wider.
* **Only the Over/Under cut favours the hierarchy** (`m05` 3.42 → 3.67, `m12` 4.62 → 5.10), and that
  is the one place a finishing-conversion parameter *should* show up, since κ scales total goals. But
  `p` never drops below 0.117, `n = 758` across 379 fixtures, and the same cut in 24/25 alone is
  `p = 0.32` and `p = 0.17`. This is a direction, not a result. It is the only thread in the study
  worth pulling if the component is ever revisited.

### 4.3 Why the edge shrinks: the hierarchy moves prices *towards* the market

| Arm | Split | Δ mean \|p_model − p_fair\| | p (clustered) |
|---|---|---:|---:|
| `m05` | all | **−0.000705** | **<0.001** |
| `m05` | 24/25 | −0.000509 | 0.001 |
| `m05` | 25/26 | −0.000906 | <0.001 |
| `m12` | all | **−0.001006** | **<0.001** |
| `m12` | 24/25 | −0.001026 | <0.001 |
| `m12` | 25/26 | −0.000985 | <0.001 |

This is the clearest statistically significant effect in the entire study, and it is a *mechanical*
one, not an economic one. Partially pooling κ across teams shrinks the per-fixture rate estimates
toward the league mean, which shrinks the model's disagreement with the closing line — reliably, in
every split, in both arms. Less disagreement with the market at an unchanged edge coefficient means
**strictly less realisable alpha per fixture**, which is exactly what §5 measures in money.

---

## 5. Fractional-Kelly portfolio backtest — the hierarchy costs money in all four pairings

Proper scores say whether a posterior is sharper. They do not say whether a Kelly stake vector built
from it compounds — Experiment 06 already recorded a case where the LogLoss winner was not the
bankroll winner, which is why `r63` exists separately from `r62` and why `r67` exists separately
from `r66`.

All four models are simulated over the **identical 99 daily slates / 628 books** (82 skipped for want
of a complete priced market), 630 calendar days, from the same Betfair TWA closing prices. The
**BookSpec is one object shared by both configurations**, built once per model, so what differs
between A and B is the policy and only the policy.

| | Configuration A — Experiment 06 baseline | Configuration B — canonical production |
|---|---|---|
| Book | `FractionalKelly(0.30)`, 2% commission, DeArb, 1X2 + O/U 2.5 + BTTS | *identical* |
| Trust | `FlatTrust(1.0)` — every priced selection stakeable | `CanonicalScottishLowerTrust()` — Home & Under 2.5 @ 0.35, Draw & Away @ 0.25, **all else 0.00** |
| Risk | `SlateDrawdown(23.0)` | `SlateDrawdown(23.0)` |
| Cap | `FixedCap(0.20)` | `FixedCap(0.25)` |
| Grouping | `DailySlate()` | `DailySlate()` |

Configuration B is `MatchDay.canonical_scottish_lower_policy()` itself — the object the live console
stakes with — not a reconstruction of it. Its ledger confirms the gate bites: 5,009 bets across
Home, Draw, Away and Under 2.5 only; **zero BTTS and zero Over 2.5**, against 5,815 bets spread over
all seven selections under A.

### 5.1 Configuration A — Experiment 06 baseline

| Model | Bets | Total return | Flat ROI | ROI 95% CI (clustered) | P(ROI>0) | 1X2 ROI | Max DD | Sharpe | Sortino | Calmar | Win rate | Mean exp. | Turnover |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m12_joint_hybrid_synergy` | 1,462 | **+136.61%** | +11.48% | [+0.53, +22.40] | 0.981 | +11.95% | −20.23% | 1.416 | 0.221 | 6.753 | 34.47% | 0.0856 | 8.48× |
| `m12_hierarchical_kappa` | 1,449 | +131.46% | +11.41% | [+0.42, +22.71] | 0.982 | +11.73% | −19.92% | 1.423 | 0.219 | 6.601 | 34.99% | 0.0836 | 8.27× |
| `m05_joint_production_wealth` | 1,455 | +131.17% | **+11.64%** | [+0.77, +22.45] | 0.983 | **+12.22%** | **−19.05%** | **1.481** | **0.250** | **6.885** | 34.43% | 0.0811 | 8.03× |
| `m05_hierarchical_kappa` | 1,449 | +125.14% | +11.44% | [+0.78, +22.59] | **0.985** | +11.74% | −19.13% | 1.480 | 0.249 | 6.541 | 34.16% | 0.0797 | 7.89× |

### 5.2 Configuration B — canonical production policy

| Model | Bets | Total return | Flat ROI | ROI 95% CI (clustered) | P(ROI>0) | 1X2 ROI | Max DD | Sharpe | Sortino | Calmar | Win rate | Mean exp. | Turnover |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m12_joint_hybrid_synergy` | 1,267 | **+155.93%** | **+13.58%** | [+2.42, +25.63] | **0.991** | +12.64% | −19.79% | 1.636 | 0.256 | **7.880** | 32.60% | 0.0774 | 7.66× |
| `m12_hierarchical_kappa` | 1,264 | +151.46% | +13.52% | [+1.54, +24.87] | 0.988 | +12.38% | −19.38% | **1.654** | 0.260 | 7.814 | 33.23% | 0.0759 | 7.52× |
| `m05_joint_production_wealth` | 1,242 | +138.94% | +13.49% | [+2.18, +25.29] | **0.991** | **+12.65%** | **−18.12%** | 1.629 | **0.283** | 7.668 | 32.69% | 0.0718 | 7.11× |
| `m05_hierarchical_kappa` | 1,236 | +134.03% | +13.29% | [+1.65, +25.09] | 0.987 | +12.16% | −18.14% | 1.640 | 0.282 | 7.390 | 32.36% | 0.0709 | 7.02× |

Bootstrap intervals are **match-clustered** (`B = 4,000`, seed 1): the resampler draws *fixtures*,
not bets, because eleven selections on one fixture share one scoreline and resampling bets would
divide the standard error by roughly √11 and make every arm look far more significantly profitable
than it is. `Turnover` is `total_stake` in bankroll-fraction units — how many times the bank was
cycled through the book, and equal to `mean_exposure × 99` by construction.

### 5.3 The paired contrast

| Arm | Configuration | Δ Bets | Δ Total return | Δ Flat ROI | Δ Max DD | Δ Sharpe | Δ Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| `m05` | A baseline | −6 | **−6.02 pp** | −0.20 pp | −0.08 pp | −0.001 | −0.14× |
| `m12` | A baseline | −13 | **−5.15 pp** | −0.06 pp | +0.31 pp | +0.007 | −0.21× |
| `m05` | B production | −6 | **−4.91 pp** | −0.20 pp | −0.02 pp | +0.011 | −0.09× |
| `m12` | B production | −3 | **−4.47 pp** | −0.05 pp | +0.40 pp | +0.018 | −0.14× |

**The hierarchical model loses bankroll in all four pairings, and it loses it the same way each
time.** It stakes fewer bets at lower exposure and cycles less turnover, because §4.3's shrinkage
toward the closing line leaves fewer selections clearing the Kelly threshold. Compounded over 99
slates that costs 4.5–6.0 points of terminal bankroll.

**Do not over-read the size of that.** Flat ROI — the per-unit-staked edge — barely moves (−0.05 to
−0.20 pp) and the ROI intervals of every pair overlap almost completely; the bankroll gap is mostly
*less capital deployed at a near-identical edge*, not a worse edge. And the movement is not uniformly
bad: `m12`'s drawdown improves 0.31–0.40 pp and its Sharpe improves 0.007–0.018, which is what
staking slightly less into the same edge is supposed to do. If the hierarchy were free, this would
read as a defensible mild de-risking.

**It is not free.** It costs `n_teams + 1` parameters and 2.1× the sampling wallclock, and §6 shows
it buys no information at all. A component that identifies nothing and reduces deployed capital is
strictly worse than the same policy run at a slightly lower Kelly fraction, which costs nothing and
is a one-line change.

### 5.4 Two findings that are not about κ, and are worth more than the one that is

**The canonical production policy dominates the research baseline for every model.** Configuration B
raises flat ROI from ~11.5% to ~13.5%, raises annual Sharpe from 1.42–1.48 to 1.63–1.65, and *lowers*
maximum drawdown, while staking 13–14% fewer bets. Gating BTTS and Over 2.5 out entirely and tilting
Home/Under against Draw/Away is worth roughly **two points of ROI and 0.2 of Sharpe** — an order of
magnitude more than anything in this study. `CanonicalScottishLowerTrust()` is doing real work and
the numbers here are independent confirmation of the trust audit that selected it.

**Both seasons are profitable, but not equally.** Flat ROI by season, on staked P/L:

| Model | Config | 24/25 bets | 24/25 ROI | 25/26 bets | 25/26 ROI |
|---|---|---:|---:|---:|---:|
| `m05_joint_production_wealth` | A | 681 | +18.42% | 774 | +6.73% |
| `m05_hierarchical_kappa` | A | 679 | **+18.65%** | 770 | +6.19% |
| `m12_joint_hybrid_synergy` | A | 690 | +19.64% | 772 | +5.52% |
| `m12_hierarchical_kappa` | A | 679 | **+20.08%** | 770 | +5.12% |
| `m05_joint_production_wealth` | B | 603 | +20.76% | 639 | +8.06% |
| `m05_hierarchical_kappa` | B | 600 | **+20.92%** | 636 | +7.54% |
| `m12_joint_hybrid_synergy` | B | 618 | +21.76% | 649 | +7.48% |
| `m12_hierarchical_kappa` | B | 613 | **+22.24%** | 651 | +7.05% |

The 24/25 → 25/26 decay is severe and it is the same reversal §3.5 found in LogLoss and §4.2 found in
β_edge — roughly **a third of the edge, in every model and both configurations**. Three independent
estimators agreeing makes it a property of the data, not of a metric.

Note the one place the hierarchy is consistently *ahead*: it earns a higher flat ROI than its control
in 24/25 in all four pairings, and a lower one in 25/26 in all four. The sign flip is perfectly
correlated with the season, which is the signature of a season-specific fluctuation rather than a
component effect — and it is the same sign flip the paired LogLoss contrast in §3.2 shows at
`p ≈ 0.53–0.74`.

### 5.5 Persistence

Every simulation was written to `mcmc_experiments` with `save_portfolio_db` — headline metrics into
`portfolio_runs`, one row per bet into `portfolio_bets`, and the exact serialized `PortfolioResult`
plus the `BookSpec`/`PolicySpec` into `portfolio_artifacts` — and then **read back with
`load_portfolio_db` and checked bet-for-bet before the runner claimed it was stored**. The runner
refuses to continue if a reloaded return or ledger differs.

| Model | Config | `portfolio_runs.portfolio_run_id` |
|---|---|---|
| `m05_joint_production_wealth` | A | `d0c0c253-f98c-4c84-a120-a51e9cc0cc99` |
| `m05_hierarchical_kappa` | A | `3785a35e-e421-42cc-b92a-e9238d9844ce` |
| `m12_joint_hybrid_synergy` | A | `202b593c-dcbd-4384-a3b4-6c7ed4524dcd` |
| `m12_hierarchical_kappa` | A | `07a62a11-4076-44fe-b6c2-ece6d1d6bcbc` |
| `m05_joint_production_wealth` | B | `625c8666-1ccb-48c2-8757-f4fcce037fc2` |
| `m05_hierarchical_kappa` | B | `c7b91c1a-e9ae-4285-9d9d-f01b13f585dc` |
| `m12_joint_hybrid_synergy` | B | `83ce8c26-83ad-4f4f-bf7d-94eb59bdefe0` |
| `m12_hierarchical_kappa` | B | `ff3e4f8c-cb49-461a-beb1-78ec7a774a57` |

The two control runs under Configuration A reproduce `results/r63_portfolio_summary.csv` to five
significant figures — `m05` +131.17% against a recorded +131.16761%, `m12` +136.61% against a
recorded +136.61151% — on the same 1,455 and 1,462 bets. That is the check that this backtest and
the recorded Experiment 06 leaderboard are simulating the same thing.
---

## 6. Posterior finishing factor — the hierarchy shrank to the common mean

This is the section that decides the component. Everything above measures what the hierarchy *did to
predictions*; this measures whether it *found anything to predict with*.

### 6.1 The league finishing factor is untouched

Averaged over the 40 folds' per-fold posterior summaries:

| Model | κ_league mean | 90% HPDI | ν (Gamma precision) |
|---|---:|---|---:|
| `m05_joint_production_wealth` | 1.0998 | [1.0201, 1.1780] | 3.538 |
| `m05_hierarchical_kappa` | 1.0992 | [1.0184, 1.1779] | 3.537 |
| `m12_joint_hybrid_synergy` | 1.0992 | [1.0195, 1.1781] | 3.608 |
| `m12_hierarchical_kappa` | 1.0987 | [1.0187, 1.1773] | 3.608 |

Goals run about **10% above** what the BBC shot-xG cell table predicts, and adding 22–25 team-level
deviations moves that number by **0.0005** and widens its interval by 0.0017. The league factor is
identified by the goals arm against the proxy arm, and the hierarchy does not disturb that
identification. `ν` is unchanged to three decimals — the proxy measurement scale is untouched, which
is exactly what the decision to keep κ out of the Gamma arm was meant to guarantee.

### 6.2 σ_κ — the data argues the spread *down*

| Model | σ_κ mean | 90% HPDI | P(σ_κ > 0.05) | Range of per-fold means |
|---|---:|---|---:|---|
| *prior* `truncated(Normal(0, 0.10), 0, ∞)` | *0.0798* | *[0, 0.1645]* | *0.617* | — |
| `m05_hierarchical_kappa` | **0.0457** | [0.0001, 0.0939] | **0.379** | 0.0350 – 0.0536 |
| `m12_hierarchical_kappa` | **0.0447** | [0.0001, 0.0923] | **0.369** | 0.0348 – 0.0530 |

**The posterior is not merely "near zero" — it is materially below its prior, and it is that
consistently.** The prior half-normal has mean 0.0798 and puts 61.7% of its mass above 0.05; the
posterior means are 0.045 and the mass above 0.05 falls to 0.37–0.38. The 90% HPDI upper bound drops
from the prior's 0.1645 to 0.094, and the lower bound sits on the boundary at 0.0001 in every one of
the 80 fold-posteriors. The per-fold means never leave a 0.035–0.054 band across 40 folds and two
model structures.

This is a real inferential statement and it is a **negative** one. The likelihood is not indifferent
to σ_κ; it actively pulls it down. The data says the team finishing spread is *smaller* than a prior
already deliberately chosen to be tight — a prior calibrated so that Poisson noise on ~40 matches a
team (~0.13 in log space) could not be mistaken for a finding.

The residual 0.045 is what the prior's own mass near zero cannot be argued away from, not a measured
effect. **Its 90% interval touches its lower bound.**

### 6.3 No team separated — 957 chances, zero

| Model | Fold-team pairs examined | 90% HPDIs on δ_κ excluding zero |
|---|---:|---:|
| `m05_hierarchical_kappa` | 957 (40 folds × 22–25 teams) | **0** |
| `m12_hierarchical_kappa` | 957 | **0** |

This is the headline. In **1,914 opportunities** across both candidates, no team's finishing delta
separated from zero at 90% credibility, on any fold, at any point in the walk-forward.

Nor is the effect merely under-powered relative to a large truth. The largest per-fold posterior mean
δ_κ is **+0.035** and the smallest **−0.043** — a ±3.5% conversion tilt at the extreme, against
90% intervals roughly 0.15 wide. The point estimates themselves are an order of magnitude smaller
than the intervals around them.

### 6.4 The named table — final fold, fullest history

Both candidates on the last boundary (23 teams). Deltas are in log space; `p_over` is the posterior
probability the team out-finishes the league.

**`m05_hierarchical_kappa`** — 23 teams, **0 separated**

| Team | δ_κ | 90% HPDI | κ_team | P(δ > 0) |
|---|---:|---|---:|---:|
| east-kilbride | +0.0215 | [−0.0426, +0.0964] | 1.1353 | 0.659 |
| alloa-athletic | +0.0072 | [−0.0581, +0.0774] | 1.1186 | 0.555 |
| the-spartans-fc | +0.0069 | [−0.0641, +0.0692] | 1.1182 | 0.561 |
| queen-of-the-south | +0.0050 | [−0.0619, +0.0716] | 1.1161 | 0.534 |
| inverness-caledonian-thistle | +0.0034 | [−0.0615, +0.0662] | 1.1144 | 0.530 |
| … | | | | |
| forfar-athletic | −0.0063 | [−0.0757, +0.0601] | 1.1036 | 0.455 |
| stranraer | −0.0077 | [−0.0823, +0.0572] | 1.1021 | 0.438 |
| edinburgh-city-fc | −0.0089 | [−0.0781, +0.0589] | 1.1008 | 0.426 |
| east-fife | −0.0113 | [−0.0884, +0.0598] | 1.0984 | 0.423 |
| kelty-hearts-fc | −0.0137 | [−0.0815, +0.0551] | 1.0957 | 0.400 |

**`m12_hierarchical_kappa`** — 23 teams, **0 separated**

| Team | δ_κ | 90% HPDI | κ_team | P(δ > 0) |
|---|---:|---|---:|---:|
| east-kilbride | +0.0235 | [−0.0424, +0.1120] | 1.1381 | 0.656 |
| alloa-athletic | +0.0097 | [−0.0638, +0.0849] | 1.1220 | 0.568 |
| the-spartans-fc | +0.0090 | [−0.0554, +0.0850] | 1.1211 | 0.573 |
| montrose | +0.0042 | [−0.0677, +0.0735] | 1.1158 | 0.523 |
| queen-of-the-south | +0.0039 | [−0.0567, +0.0807] | 1.1154 | 0.536 |
| … | | | | |
| stranraer | −0.0070 | [−0.0782, +0.0612] | 1.1033 | 0.448 |
| forfar-athletic | −0.0087 | [−0.0813, +0.0587] | 1.1015 | 0.434 |
| edinburgh-city-fc | −0.0095 | [−0.0745, +0.0649] | 1.1007 | 0.439 |
| east-fife | −0.0117 | [−0.0764, +0.0628] | 1.0982 | 0.396 |
| kelty-hearts-fc | −0.0130 | [−0.0903, +0.0541] | 1.0968 | 0.399 |

The full 23-team tables are in `results/hierarchical_kappa/r66_team_kappa_final_fold.csv`, and the
per-fold posteriors in `r66_kappa_by_fold.csv`.

Every `p_over` lies in [0.396, 0.659]; the total implied κ range is 1.096–1.138, a 3.8% span against
a league factor of 1.10. The two candidates agree on the ordering — the same five teams at the top,
the same four at the bottom — which is the one encouraging observation available: whatever weak
signal exists is at least reproducible across model structures. It is not, however, reproducible
against zero, which is the comparison that matters.

**Note on the earlier smoke.** The r64 two-fold smoke reported σ_κ ≈ 0.050 with `P(σ_κ > 0.05)` of
0.43–0.44 and said it provided "no strong evidence of persistent team-specific conversion skill."
The full 40-fold grid does not overturn that reading — it hardens it. With 20× the folds the
posterior mass above 0.05 falls further, to 0.37, and the separation count over 957 team-folds is
exactly zero. The smoke's caution was correct and the grid has now paid for the certainty.

### 6.5 Why the effect is not there — and this is a data problem, not a model problem

A Scottish Lower team contributes roughly 40 fixtures to a fold at about 1.3 goals a side. That is
~52 goal events per team per fold. Poisson noise on 52 events is ±14% in the rate, or ~0.13 in log
space. The largest posterior mean δ_κ observed anywhere in the grid is 0.043 — **a third of the noise
floor**. A per-team finishing effect would have to be roughly three times larger than anything the
posterior entertains before this sample could distinguish it from chance.

The component is not mis-specified and the sampler is not failing. There is simply not enough
per-team goal volume in a two-division lower-league fold to identify a per-team conversion factor.
Nothing about the implementation would change that; only more matches per team, or a pooling
structure that borrows across a much wider universe than tournaments 56 and 57, would.

---

## 7. Verdict

### 7.1 Scientific verdict — the hypothesis is refused

**Refused: "Scottish League One and League Two teams have persistent finishing skill that a
partially pooled per-team κ can identify from 40 walk-forward folds."**

The evidence is not ambiguous, and it is not merely an absence:

1. **σ_κ is pushed below its prior, consistently.** Posterior mean 0.045 against a prior mean of
   0.080; `P(σ_κ > 0.05)` falls from 0.617 to 0.37–0.38; the 90% HPDI upper bound falls from 0.165
   to 0.094 and the lower bound sits on the boundary in all 80 fold-posteriors. The likelihood is
   not indifferent to the spread — it argues it down.
2. **Zero separations in 957 team-fold pairs, twice over.** No team, on any fold, in either
   candidate, has a 90% HPDI on δ_κ that excludes zero.
3. **Nothing downstream moves.** Paired LogLoss `p = 0.98` (`m05`) and `p = 0.94` (`m12`); model
   probabilities correlate at 0.9993 with mean absolute shift 0.003; β_edge does not increase in
   any cut.
4. **The mechanism is understood, and it is a data limit.** ~52 goal events per team per fold puts
   the Poisson noise floor at ~0.13 in log space. The largest posterior mean δ_κ anywhere in the
   grid is 0.043 — a third of that floor. This sample cannot identify a per-team conversion factor,
   and no change to the component would make it able to.

The one genuinely significant effect is `Δ mean |p_model − p_fair| < 0` at `p < 0.001` in every
split: partial pooling shrinks the model toward the market. That is a description of what the prior
does, not a discovery about football.

**What the study establishes positively, and should be credited for:** the component is correctly
specified and identified (`δ_κ` sums to zero in every draw), it samples to a strict zero-divergence
gate over 128,000 draws per model at target acceptance 0.90, and it leaves `κ_league` and `ν`
untouched to three decimals — which is what keeps the two-arm identification argument intact and is
the non-obvious part of the design. This is a well-executed negative result, not a failed experiment.

### 7.2 Production verdict — do not adopt into `src/models/pregame/`

**`HierarchicalKappa` must NOT enter the production Scottish Lower recipe.** The canonical fit the
MatchDay consoles load stays `m12_joint_hybrid_synergy` with `SharedKappa()`.

The costs are concrete and the benefits are not:

| | Shared κ | Hierarchical κ |
|---|---|---|
| Parameters per fold | `2·n_teams + 7` | `3·n_teams + 8` — **+23 to +26** |
| Grid wallclock (`m12`) | 3.18 h | **6.80 h (2.14×)** |
| Out-of-sample LogLoss | 0.643370 | 0.643309 (`p = 0.94`) |
| Model-market edge β_edge | +2.045 | +2.036 |
| Terminal bankroll, production policy | +155.93% | **+151.46%** |
| Terminal bankroll, research baseline | +136.61% | **+131.46%** |
| Teams identified as different finishers | — | **0 of 957** |

A component that cannot be given a number by the data should not be carried by the code that prices
Saturday's slate. It adds a funnel to a posterior that has to be sampled inside a fixed pre-slate
window, it doubles the retrain cost, and its only reliable measurable effect is to make the model
agree with the closing line slightly more than it otherwise would — the opposite of what a staking
model is for.

**Explicitly, this is not a verdict on the code.** `src/models/pregame/builder/` keeps
`HierarchicalKappa`, `SharedKappa`, `kappa_mode_width`, `extract_kappa` and `cb_hpdi` exactly as
they are. They are tested, they carry the r64 smoke and the r66 evaluation behind them, and the
component costs nothing while `SharedKappa()` is the default. What must not happen is a recipe
change.

### 7.3 The one thread worth keeping

Over/Under 2.5 is the only cut where the hierarchy's point estimate improves in both arms
(β_edge `m05` 3.42 → 3.67, `m12` 4.62 → 5.10), and it is the cut where a goal-conversion parameter
*should* appear, because κ scales the total. It is not significant — `p` never falls below 0.117 on
758 observations across 379 fixtures — and it does not survive the season split. Treat it as the
place to look if the component is ever reconsidered, not as a finding.

**If it is reconsidered, change the universe rather than the model.** The binding constraint is
goal volume per team per fold, so the productive versions of this experiment are:

* pool κ across a much wider universe than tournaments 56 and 57 — several segments at once, with
  the league as the pooling level and the team below it;
* or pool over a longer history than 2 seasons, accepting that finishing skill would then have to be
  assumed near-static, which is itself a strong claim;
* or move the question to a market where conversion is the whole instrument (totals, correct score)
  and a smaller effect is worth more.

Re-running the same 40-fold design on the same two divisions will produce this report again.

---

## 8. Threats to validity, stated plainly

* **The convergence comparison is confounded.** The hierarchical grids ran at target acceptance
  0.90 and the controls at 0.65. Their cleaner divergence and ESS numbers are partly, and possibly
  entirely, the step size. §2 says so; nothing else in the report leans on it.
* **The controls were fitted at an earlier commit.** `m05_joint_production_wealth` and
  `m12_joint_hybrid_synergy` were sampled on 2026-09-02, before `JointGammaPoissonObservation`
  gained its fourth type parameter, so their serialized artefacts need the read-side compatibility
  shim in `l66_hierarchical_kappa_eval_loader.jl` §0 to deserialize at all. The shim delegates every
  concrete type to the stock deserializer and reinstates `SharedKappa()` only for the three-parameter
  layout, which had exactly one finishing mode. The check that this is right: the control proper
  scores in §3.1 reproduce `results/r62_proper_scores.csv` to every reported digit, and the control
  portfolio returns in §5 reproduce `results/r63_portfolio_summary.csv` to five significant figures.
* **`m05`'s control has one fold that fails its stored convergence audit** (tail ESS 252 against a
  300 floor) and one divergent transition. Its books were therefore built with
  `require_converged = false` and are flagged, not trusted, in the persisted metadata. The effect on
  the comparison is to make the control's numbers *slightly optimistic* if anything, which does not
  help the hierarchical case.
* **627 of 710 held-out fixtures carry scored Betfair markets.** The 83 fixtures without a priced,
  settled market at the close are absent from every score and every GLM in this report. They are
  absent identically for all four models.
* **MCE is uninformative on this sample** and is reported only so its movement is not misread. See
  §3.4: every binding maximum is a three- or four-observation reliability bin.
* **Two seasons is a short walk-forward.** The 24/25 → 25/26 reversal in §3.5 is the dominant
  temporal effect here and it is not explained by anything this study manipulates. Any statement
  about the model beating the closing line should carry the season it was measured in.

---

## 9. Reproduction

```bash
# proper scores, paired contrasts, GLM edge, finishing-factor posterior
julia --project -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r66_compare_hierarchical_kappa.jl

# fractional-Kelly portfolio backtest under both configurations, with PostgreSQL persistence
julia --project -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r67_portfolio_hierarchical_kappa.jl
```

Both read completed fits out of `mcmc_experiments` and sample nothing. Neither needs
`mcmc-beast` for compute; both need it for the artefacts.

| File | Role |
|---|---|
| `l64_hierarchical_kappa_loader.jl` | The two hierarchical candidates, the σ_κ prior, the 0.90-acceptance sampler |
| `l66_hierarchical_kappa_eval_loader.jl` | Run manifest, Betfair closing frame, fit loader, artefact compatibility shim |
| `r66_compare_hierarchical_kappa.jl` | §2–§4, §6 |
| `r67_portfolio_hierarchical_kappa.jl` | §5, and all PostgreSQL portfolio persistence |
| `results/hierarchical_kappa/r66_proper_scores.csv` | §3.1, §3.5 |
| `results/hierarchical_kappa/r66_paired_contrasts.csv` | §3.2, §3.3, §4.3 |
| `results/hierarchical_kappa/r66_calibration_curves.csv` | §3.4 |
| `results/hierarchical_kappa/r66_glm_edge.csv` | §4 (all model × split × subset cells) |
| `results/hierarchical_kappa/r66_kappa_by_fold.csv` | §6.1–§6.3, per fold |
| `results/hierarchical_kappa/r66_kappa_summary.csv` | §6.1, §6.2 |
| `results/hierarchical_kappa/r66_team_kappa_final_fold.csv` | §6.4, all 23 teams |
| `results/hierarchical_kappa/r66_fold_diagnostics.csv` | §2, per fold |
| `results/hierarchical_kappa/r66_convergence_summary.csv` | §2 |
| `results/hierarchical_kappa/r66_scored_observations.csv` | Every scored observation, all four models |
| `results/hierarchical_kappa/r67_portfolio_summary.csv` | §5 |
| `results/hierarchical_kappa/r67_trade_ledger.csv` | Every simulated bet, both configurations |
| `results/hierarchical_kappa/r67_portfolio_by_season.csv` | §5.3 |

Prior work this report continues: `HIERARCHICAL_KAPPA_SMOKE.md` (the r64 gates and the sampler
correction), `README.md` §"40-Fold Grid Results" (the shared-κ leaderboard these controls come from),
and `EDA_FINDINGS.md`.
