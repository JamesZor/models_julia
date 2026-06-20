# Game-State Model Comparison — Over/Under Calibration Report

*Ireland Premier Division. Four Turing variants of the in-play intensity model, fit on a 75% match
split, evaluated on the held-out 25% (1,806 Over/Under events at lines 1.5/2.5/3.5). Odds derived via
the project pipeline: model μ → remaining-goals `ScoreMatrix` (`compute_score_matrix`) →
`compute_market_probs(S, MarketOverUnder(L − current_total))`.*

## The four variants (game-state representation)
- **none** — no game-state term (pregame λ + time + home only)
- **linear** — global `trailing`/`leading` dummies (the original l03)
- **hier_replace** — drop the dummies, use a partial-pooled per-state intercept `δ_state[goal_diff]` (7 states)
- **hier_addon** — keep the dummies AND add `δ_state` deviations

## Headline: the game-state representation barely matters for calibration

| variant | OU ECE ↓ | Brier ↓ | LogLoss ↓ | held-out elpd (count) |
|---|---|---|---|---|
| linear | **0.0622** | 0.1672 | 0.494 | −1.0753 |
| none | 0.0634 | **0.1655** | **0.489** | **−1.0743** |
| hier_replace | 0.0655 | 0.1681 | 0.4956 | −1.0755 |
| hier_addon | 0.0663 | 0.1678 | 0.495 | −1.0760 (R̂=1.05) |

All four are within ~0.004 ECE of each other. **The hierarchical per-state parameter does not improve
calibration** — `linear` has the best ECE, `none` the best Brier/LogLoss. (`hier_addon`'s redundant
parameters even hurt sampling, R̂=1.05.)

**Why** — the learned per-state effect is essentially *linear*, so the simple dummies already capture it:

| state (goal diff) | δ_state mean | 90% CI |
|---|---|---|
| down 3 | +0.18 | [−0.22, 0.61] (thin, shrunk) |
| down 2 | +0.18 | [−0.08, 0.47] |
| **down 1** | **+0.33** | [0.10, 0.58] |
| level | +0.01 | [−0.23, 0.24] |
| up 1 | −0.20 | [−0.44, 0.05] |
| **up 2** | **−0.40** | [−0.72, −0.11] |
| up 3 | −0.06 | [−0.46, 0.32] (thin, shrunk) |

(σ_state = 0.33.) The trailing→push / leading→protect gradient is monotone and roughly linear; the
extreme states (±3, n=15–22) are partial-pooled toward zero. So the flexible model mostly *confirms*
the linear assumption is adequate — exactly what you'd want to learn.

## The real issue: a systematic under-prediction of Overs (shared by all variants)

Mean predicted P(over) ≈ **0.44** vs mean actual ≈ **0.50** — the model under-predicts overs by ~5–6
points (ECE ≈ 0.06). Calibration **by game state** (identical pattern for `linear` and `hier_replace`):

| goal diff | n | pred P(over) | actual | gap |
|---|---|---|---|---|
| −2 | 66 | 0.66 | 0.80 | **−0.14** |
| −1 | 252 | 0.50 | 0.46 | +0.05 |
| **0 (level)** | **990** | **0.34** | **0.42** | **−0.08** |
| +1 | 276 | 0.49 | 0.56 | −0.07 |
| +2 | 207 | 0.71 | 0.70 | +0.01 |
| +3 | 15 | 0.88 | 1.00 | −0.11 |

The miss looks worst at level games and extreme deficits, and the hierarchical game-state model does
**not** fix it (same gaps).

> **⚠️ CORRECTION (see `heavytail_diagnosis_report.md`).** This ~5-pt under-prediction is **NOT
> structural** — it is **test-split sampling noise**. Across 15 random splits the held-out mean bias is
> **+0.030 ± 0.087 (n.s.)**, and the model is mean-unbiased on training data (1.328 vs 1.330). The
> single split used here happened to be goal-heavy. The earlier hypotheses below (stoppage-time exposure;
> independence) were tested and do **not** hold: a Negative-Binomial heavy tail barely moves OU
> calibration, and the offset is unbiased on train. **Lesson:** with 63 test matches, a ~5-pt /
> ~0.06-ECE gap is within sampling error → use cross-validation / more leagues, not a heavier tail.

## Recommendations
- **Use the `linear` (or even `none`) game-state model** — the hierarchical per-state parameter adds
  complexity and sampling cost without calibration benefit on this sample (the effect is linear, and
  ±3 states are too thin to learn anything pooling can't already infer).
- **To improve OU calibration, fix the structural under-prediction**, not the game-state form:
  (a) make the exposure offset use real remaining time incl. stoppage; (b) try a Dixon-Coles /
  correlated marginal; (c) failing those, add a thin post-hoc **calibration layer** on the OU
  probabilities (the project already has the `Calibration` L2 machinery).
- This is calibration analysis; the l04 report already showed no tradeable in-play edge.

## Reproduce
`r05_inplay_calibration_compare.jl`: fit the four `InPlayIntensityConfig(game_state = …)` variants via
`Samplers.run_sampler`/`NUTSConfig`, then `build_ou_eval` (uses `compute_score_matrix` +
`compute_market_probs` with the `L − T` line shift) and the `ece`/`brier`/`logloss` + per-`gd_bucket`
tables. Game-state model code in `l03_inplay_turing.jl` (`config.game_state` ∈ `:none|:linear|:hier_replace|:hier_addon`).
