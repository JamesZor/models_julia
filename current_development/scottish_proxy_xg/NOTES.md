# scottish_proxy_xg — BBC commentary proxy xG as a Gamma pillar (56/57)

Opened 2026-08-04. Extends the `funnel_xg_apm` line
(`src/models/pregame/engines/player_level/time_decay/goals_funnel_plus_minus_league.jl`), the
current best engine on Scottish League One / Two.

---

## 1. The gap this stream closes

The incumbent's own header states its weakness:

> it reads raw shot COUNTS from `ds.bbc` — a 30-yard speculative effort and a six-yard tap-in are
> the same datum to it.

Meanwhile `src/features/plus_minus/shot_parser.jl` **already** turns BBC commentary into per-shot
xG — a saturated `P(goal | zone × body part × context)` cell table with empirical-Bayes shrinkage,
measured at 98.4–99.8% parse coverage, Brier 11.1% better than base rate (the base paper's
*coordinate* model managed 14.7%, so ~76% of the coordinate gain survives with no coordinates), and
team-level correlation 0.817 vs SofaScore xG.

That quality signal currently reaches a model **only** as an RAPM player rating
(`XGPlusMinusFeature`). It has never been a **team-match observation**. This stream takes the one
step the repo never took: sum per-shot xG to a team-match total and give it a Gamma likelihood, the
Ireland outfield-xG structure, on a league that has no SofaScore xG at all.

Nothing in `shot_parser.jl` is reimplemented. `l01` is the aggregation plus the FeatureSet contract.

---

## 2. Measured before writing any code (2026-08-04, `bbc.live_text`, tiers 56/57)

| Fact | Value | Consequence |
|---|---|---|
| Commentary coverage | **100% from 23/24, 0% before** — 1,070 matches | Hard window constraint; the same one `XGPlusMinusFeature` already lives under |
| Shots per team-match (events) | 9.14, **zero sides with 0 shots** (n = 2,137) | Gamma support is safe; the 1e-3 floor is a guard, not a code path |
| Zone conversion | outside 0.048 · box-side 0.102 · box-centre 0.182 · six-yard 0.444 · pen 0.762 | Shot MIX is a ~9× axis per shot |
| Cross-team mix spread (20 teams, ≥40 tm) | observed SD 0.036 vs binomial SD 0.016 → **4.9× excess variance**, range 0.178 | Team shot quality is **real, not sampling noise** |
| Shooter named in commentary | 97–98% of shot events | But name→`player_id` resolves at only 93.2%, which is why the player route stays RAPM |

**Honest expectation, recorded up front.** A crude 2-bucket estimate puts the quality axis at
≈±4% on the scoring rate, against the funnel's ±10.7% team-strength SD. The full cell table (body
part, set-piece context, penalties) will exceed that, but this is a **second-order** axis. Combined
with three prior nulls in this family — r07b funnel+iso soft-negative, r04 hierarchical conversion
collapsed to ¼–1/7 of prior, funnel/APM/fusion mutually indistinguishable — **a null is a likely and
acceptable outcome.** The stream is designed to measure a small effect precisely, not to find a
large one.

---

## 3. The two arms

### Arm A — `TeamPxGGoalsAPMModel` (xG replaces shots)

```
log μ = μ_base[s] + δ_month[t] + δ_league[ℓ] + ha + α_h + β_a + (w_att·R_h − w_def·R_a)
xG    ~ Gamma(ν, μ/ν)         [masked]
G     ~ Poisson(κ · μ)
```

Julia's `Gamma` is shape–**scale**, so `Gamma(ν, μ/ν)` has mean μ and variance μ²/ν: **ν is a
precision** (inverse squared CV), not a rate.

- **No `shot_scale` offset.** Mean proxy xG is ≈1.22, so `log(1.22) ≈ 0.2` and `UniformInit(-2,2)`
  already starts on scale. The funnel's single biggest gotcha
  (`bbc_xg_proxy/NOTES.md:174-204`) does not apply.
- **κ is global**, centred on 1 via `log κ ~ N(0, 0.2)` — the cell table is a *conversion-rate*
  table, so `Σ proxy xG ≈ Σ goals` by construction (r00 gate 3 measures it). Per-team conversion
  was the r04 null.
- **ν prior widened to `truncated(Normal(4.0, 1.5), 0.5)`.** The zone mix gives E[q] ≈ 0.133,
  E[q²] ≈ 0.038 over 9.14 shots ⇒ CV ≈ 0.49 ⇒ ν ≈ 4.2. Ireland's `Normal(3.0, 0.5)` is both
  mis-centred and far too tight for a compound sum.

### Arm B — `TeamFunnelPxGGoalsAPMModel` (xG joins shots)

```
log λ_s = shot_scale + (same linear predictor)          VOLUME
logit q = q_raw + a_i − d_j                             QUALITY (xG per shot)
     μ  = λ_s · q
S      ~ Poisson(λ_s)                  [ds.bbc, all 6 seasons]
xG | S ~ Gamma(ν_q·S, q/ν_q)           [commentary, 23/24+]
G      ~ Poisson(κ · λ_s · q)
```

**The design point: the xG pillar is CONDITIONAL on the observed shot count.** Proxy xG is
literally `Σ_{i=1..S} q_i` over the same shots the volume pillar counts, so a *marginal* Gamma would
count the volume information twice — over-sharpening the posterior and over-weighting the very axis
the funnel already owns. Conditioning on S strips the volume out and leaves the pillar carrying only
quality. Mean `= S·q`, variance `= S·q²/ν_q`, i.e. CV `= 1/√(ν_q·S)`, shrinking with S exactly as a
sum of S i.i.d. contributions must.

**Goals stay marginal.** r06 proved this matters: routing goals through a conditional
(`cascade_weight = 1`) makes them independent of λ_s given the intermediate count and severs the
goals→team-strength gradient, which is what lost totals at r03. Only the xG pillar is conditional;
the pricing path is untouched and the plain Poisson score grid still applies exactly.

**Two shot series, on purpose.** Volume reads `ds.bbc` (match pages, ~9.89/side, six seasons);
conditioning reads the commentary **event** count (~9.14/side, 23/24+). The conditioning count must
be the one the xG was actually summed over. The ~8% level gap is absorbed by the global κ — valid
only if the gap is not systematic by team, which is **r00 gate 2d**.

**Expect σ_q to be small.** Same shape as the r04 hierarchical-conversion null; the measured mix
spread implies σ_q ≈ 0.05 on the logit against a prior mean ≈0.12. Treat the **posterior/prior
ratio** as a first-class result — it answers "is there team-level shot quality on 56/57?" whether or
not the engine wins.

---

## 4. Known limitations, stated before the result

1. **The cell table is fitted globally**, over every shot in the store rather than per fold. This is
   the accepted precedent (`plus_minus.jl:64-71`) and keeps the two xG routes numerically
   consistent; the table carries no team or player identity, so a fold's ~25 shots move a ~19.5k-row
   league-wide conversion table by ~0.1%. `ProxyXGFeature(fit_on = :training)` refits per fold and
   exists so the difference can be measured rather than asserted.
2. **The cell table sees 56/57 only** (~19.5k attempts), not the pooled 54–57 ~45k the research
   validated, because `ds.bbc_events` is segment-filtered. Same behaviour as `pm_prepared`. r00
   gate 4 records it so drift is detectable.
3. **The Gamma variance law is quadratic; the true process is compound Poisson (linear).** r01-E4
   measures the exponent; cell 5 (`variance_law = :linear`, `Gamma(μ/θ, θ)`) is the matched
   alternative and is scheduled only if E4 says linear.
4. **The proxy can only see shot MIX.** Being a conversion-rate table, it cannot know anything the
   league-average conversion does not, except through which cells a team's shots land in. If 56/57
   shot mix were homogeneous the whole idea would be empty — §2 shows it is not (4.9× excess
   variance), but that is the mechanism, and it bounds the size of the effect.
5. **Coverage confound in the grid.** With `history_seasons = 2`, target 24/25 pulls 22/23 history
   which has no commentary. r04 splits LogLoss by target season to separate data from structure at
   zero extra compute.

---

## 5. File map

| File | Role |
|---|---|
| `l01_proxy_xg_feature.jl` | `ProxyXGFeature` + team-match aggregation + `proxy_team_rows` for the EDA |
| `l02_pxg_engines.jl` | Arm A, Arm B, extractors, loader-local Poisson prediction overrides |
| `r00_data_qa.jl` | WP0 — 7 hard gates. Run first. |
| `r01_eda_informativeness.jl` | WP1 — E2 ladder (the go/no-go), E3 reliability, E4 variance law, E5 external validity |
| `r02_smoke.jl` | WP4 — both arms on a fully-covered window + a warmup probe |
| `r03_grid.jl` | WP5 — 4–5 cells, canonical r07-matched spec, ~25h |
| `r04_eval.jl` | WP6 — per-line LogLoss, season split, Bet365 + Betfair growth |

**Include discipline:** `l02` includes `l01`, so a runner includes exactly one loader — `l01` for
r00/r01, `l02` for r02/r03/r04, never both. Any runner that *evaluates* these experiments must
include `l02` too, or `evaluate_experiments` silently NaNs every row.

---

## 6. At graduation (only after r04 says so)

- `ProxyXGFeature` → `src/features/types.jl` + export in `features-module.jl` + extractor into
  `src/features/extractors/bbc_extractors.jl` beside `ShotsFunnelFeature`.
- Add a `shots::DataFrame` field to `PMPrepared` (`src/features/plus_minus/plus_minus.jl:47-52`) —
  it already builds and then discards exactly the table `l01` re-derives.
- Engine → `src/models/pregame/engines/player_level/time_decay/goals_proxy_xg_plus_minus_league.jl`;
  include in `pregame-module.jl` **after** `goals_plus_minus_league.jl` (which defines
  `_pm_outfield`) and add to the export line.
- ⚠ **`src/predictions/score_computation/poisson.jl` — line 4 (import) AND lines 6-20 (Union).**
  Omit this and PPD takes the NegBin path and errors on a missing `r` column.

---

## Log

- **2026-08-04** — Stream opened. Coverage, zone conversion and cross-team mix spread measured
  directly against `bbc.live_text` before any code was written (§2). Design fixed: two arms, the
  Arm-B xG pillar conditional on S, ν prior widened to ≈4.2, no `shot_scale` on Arm A. All seven
  files written; nothing run yet.
