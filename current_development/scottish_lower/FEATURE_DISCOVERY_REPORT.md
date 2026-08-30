# Feature Discovery — Eleven Hypotheses, Held Out

> **Mandate:** free exploration for novel covariates across the data mesh.
> **Stores:** Scottish Lower 56/57 (2,009 matches, the deployment target) and English 1/2/3/84 (8,192 matches, confirmation).
> **Scripts:** `l60_novel_features.jl`, `l61_holdout_gauntlet.jl`, `r60_form_kernel_forensics.jl`, `r61_squad_dynamics_forensics.jl`, `r62_feature_gauntlet.jl`
> **Snapshot:** 30 August 2026

---

## 0. Headline

Eleven hypotheses were formulated, built and tested. **Three improve held-out log loss on the
deployment store; two are actively harmful; six are dead ends.** The single most consequential
result is not a new feature at all:

> **The shipped `PxGRapmCovariate` HURTS out-of-sample log loss once the model carries explicit
> team strength** — +0.0108 nats [+0.0026, +0.0193] on Scottish Lower. RAPM was largely
> re-deriving what `dyn.α`/`dyn.β` already know.

The best genuinely new candidate is **bench depth** (−0.00319 nats [−0.00614, −0.00030]), which is
invisible to every covariate currently in the builder because all of them read the starting XI only.

---

## 1. Method, and one bug worth recording

Two evaluation layers, both history-fitted and held-out scored (80/20 chronological):

1. **Incremental R²** over production wealth + the shipped pxG column (`r60`, `r61`).
2. **Out-of-sample 1X2 log loss under a Poisson ridge carrying full team effects** (`r62`) —

   `log λₕ = μ + γ + αₕ + βₐ + Σ qₖʰ`,  `log λₐ = μ + αₐ + βₕ + Σ qₖᵃ`

   fitted by IRLS with a ridge penalty on the team effects. This is the layer that decides,
   because it asks the only question that matters: does the candidate explain something *team
   strength cannot*?

Every comparison is a **paired bootstrap over held-out matches**, 90% interval.

### The bug

The first version of the ΔR² bootstrap refit OLS on each resample and reported
`joint_R² − base_R²`. In sample that quantity is **mechanically non-negative** — adding any column
to a least-squares fit raises R², noise included. Every one of ten candidates came back
"ADDS SIGNAL" with a CI floor of +0.00001. The fix is to fit both models on history and score with
**frozen coefficients** on the held-out block, where a useless column scores negative. After the
fix, eight of ten candidates became indistinguishable from zero. Recorded because the broken
version produced a table that looked entirely plausible.

A second correctness fix: level-role candidates were initially scored against *supremacy*, which
asks them to do a job they were not built for. They are now scored against total goals.

---

## 2. Results

### Held-out log loss under full team strength (`r62`)

Baseline `logloss` on Scottish Lower: team strength alone **1.11026**; plus wealth and the shipped
pxG column (the incumbent) **1.08014**, a gain of −0.0301 nats. The incumbent covariates are
earning their place; everything below is measured against *them*.

| candidate | role | Scottish Δlogloss [90% CI] | English Δlogloss [90% CI] | verdict |
|---|---|---|---|---|
| **bench_value** | sup | **−0.00319 [−0.00614, −0.00030]** | −0.00021 [−0.00075, +0.00033] | **improves on target** |
| adj_pxg_sup | sup | −0.00251 [−0.00626, +0.00128] | −0.00014 [−0.00037, +0.00009] | no effect |
| **late_share** | sup | **−0.00178 [−0.00314, −0.00042]** | −0.00008 [−0.00037, +0.00021] | **improves on target** |
| slow_sup | sup | −0.00112 [−0.00487, +0.00266] | **−0.00071 [−0.00130, −0.00013]** | improves on England only |
| **wealth_x_rapm** | sup | **−0.00095 [−0.00164, −0.00025]** | −0.00020 [−0.00047, +0.00007] | **improves on target** |
| continuity | sup | −0.00078 [−0.00256, +0.00104] | −0.00007 [−0.00043, +0.00030] | no effect |
| adj_pxg_level | level | +0.00002 [−0.00033, +0.00038] | **−0.00059 [−0.00092, −0.00028]** | improves on England only |
| **rapm_shots** | sup | **+0.00864 [+0.00180, +0.01572]** | +0.00084 [−0.00111, +0.00272] | **HARMFUL** |
| **rapm_xg** | sup | **+0.01080 [+0.00262, +0.01931]** | +0.00033 [−0.00215, +0.00285] | **HARMFUL** |

All nine together: no effect (+0.00923 [−0.00054, +0.01887]) — the harmful RAPM terms cancel the
gains.

---

## 3. Hypothesis by hypothesis

### H1 — Opponent-adjusted pxG · **partially supported**
Fitting attack and defence ratings on the pxG matrix itself (ridge-penalised weighted least squares
in log space, refit once per distinct kickoff on strictly earlier matches) beats the raw rolling
mean on association: r = **+0.2310** against **+0.2196**, AUC 0.6289 against 0.6179. But under full
team effects the supremacy version has no log-loss effect on either store. The **level** version
does improve on England (−0.00059 [−0.00092, −0.00028]) and not on Scotland.
*Reading:* the opponent adjustment recovers what `dyn.α/β` already recover. Its value is in the
*total*, where the engine has less structure.

### H2 — Volume vs quality · **strongly supported, and it reframes the pxG covariate**
Splitting pxG into shot volume and mean xG per shot:

| | r(supremacy) | AUC |
|---|---|---|
| shot volume | **+0.2441** | 0.6257 |
| xG per shot | **−0.0122** | 0.4986 |

Quality is *exactly nothing*. And shot volume correlates **+0.88** with the shipped pxG column.

This is not a contradiction of `r92`, where the zonal parse beat a shot-count control by +0.164
correlation against official xG. Both are true, and together they say something sharp: **shot
quality matters for measuring a single match and washes out entirely when averaged over eight.**
Volume persists; finishing does not. The rolling pxG covariate is, to a first approximation, a
sophisticated shot counter — and the sophistication is not what makes it work.

### H3 — Over-performance mean reversion · **not supported**
`goals − pxG` rolled forward gives r = **+0.0963**, positive where reversion predicts negative. No
evidence that sides out-scoring their chances regress at this horizon. The positive sign is most
likely team quality leaking in (good sides both out-score xG and win), which the team-effects model
then absorbs — its log-loss effect is null.

### H4 — Dual-horizon kernels · **not supported**
Fast (half-life 2 matches) and slow (half-life 20) correlate +0.708 — genuinely distinct. But the
fast kernel adds **ΔR² = +0.0014, t = +0.77** over the slow one alone. There is no fast/slow
decomposition to exploit; there is just a preference for *slower*. The slow kernel alone
(r = +0.2516) beats the shipped 8-match window (r = +0.2196), confirming `r95`'s sweep, and it is
the one candidate that improves English log loss (−0.00071).

### H5 — Squad continuity · **weak**
Share of the XI retained from the previous match: r = +0.1593, ΔR² = +0.00555 [+0.00033, +0.01083].
Real at the R² layer, null at the log-loss layer. Correlates +0.183 with wealth — settled sides are
richer sides.

### H6 — Rest asymmetry · **dead end**
Only **69 of 402** held-out matches have any rest differential at all; league scheduling is almost
perfectly symmetric outside cup weeks. r = +0.0397, no effect. Not wrong, just not present.

### H7 — Minutes load · **dead end**
r = −0.0259 on supremacy (correct sign, no magnitude), nothing on totals. Correlates −0.589 with
rest, so the two are largely one variable.

### H8 — Bench depth · **the best new finding**
Log bench market-value differential: r = +0.1385, **AUC 0.5885**, t = **+3.63**,
ΔR² = +0.01206 [+0.00067, +0.02312], and Δlogloss = **−0.00319 [−0.00614, −0.00030]**.
It survives the team-effects model on the deployment store and is neutral on England.

Every wealth covariate in the builder — `WealthCovariate`, `ProductionWealthCovariate` — reads
`is_substitute == false` and therefore cannot see the bench at all. This is a genuine blind spot,
and it is cheap to close: the same lineup rows, the opposite filter.

### H9 — Late-game drop-off · **supported with the sign reversed**
Historical share of pxG created after the 70th minute: r = **−0.1689**, AUC 0.4373 — a side that
creates *more* of its chances late is *worse*. Δlogloss = −0.00178 [−0.00314, −0.00042].

The hypothesis was that finishing strong indicates depth. The data says the opposite, and the
mechanism is game state: **teams create late chances because they are losing.** The column is a
proxy for "trails often", which is why it predicts negatively and why it works. Useful, but it
should be named for what it measures, not what it was built to measure.

### H10 — Referee dynamics · **dead end, no data**
`ds.matches` on this segment carries no `referee_id` column. `Features.RefereeOfficiatingFeature`
reads that column and would emit a constant index here. Untestable without an upstream schema
change.

### H11 — Wealth × RAPM synergy · **supported on the target store**
The product term: Δlogloss = **−0.00095 [−0.00164, −0.00025]** on Scottish Lower, neutral on
England. Consistent with `r93`'s finding that the two are near-orthogonal (r = +0.336, both
significant beyond the other) — money converts to results differently depending on whether the
players are individually good. The smallest of the three winners, and the one most likely to be a
multiple-testing artefact.

---

## 4. The RAPM result

`rapm_xg` **+0.01080 [+0.00262, +0.01931]** and `rapm_shots` **+0.00864 [+0.00180, +0.01572]** on
Scottish Lower — both intervals clear of zero, both harmful, and by a margin three times larger
than the best candidate's gain.

This is not inconsistent with earlier work; it completes it. `r93` measured RAPM adding
ΔR² = +0.0184 over squad wealth — but that baseline had **no team effects**. Add explicit
`α`/`β` and RAPM becomes redundant: it was carrying team strength, not player skill. The evidence
converges from four directions:

- `r94`: split-half reliability 0.247 at the shipped λ — the player signal is mostly noise.
- `r50`/`r51`: RAPM correlates with market value (+0.107 to +0.196) more strongly than the SofaScore
  rating does — it is reading squad quality.
- `r51`: dropping the goalkeeper — who has *zero* player signal, r = +0.032 — makes the covariate
  **worse**, because his coefficient is absorbing team quality.
- `r62`: with team strength explicit, RAPM is harmful.

**Recommendation:** re-run `r40` with and without `PxGRapmCovariate` before shipping it. The engine
carries `TimeDecayDynamics`, so `r62`'s baseline is the closer analogue, and it says the covariate
costs more than it earns.

---

## 5. Recommendations

1. **Build a bench-depth covariate.** Same lineup rows as `ProductionWealthFeature`, opposite
   substitute filter. Largest and best-attested gain, and it closes a real blind spot.
2. **Re-test `PxGRapmCovariate` on `r40` against a no-RAPM arm.** §4 suggests it is negative-value
   in the presence of team dynamics.
3. **Change the pxG default half-life to ~20 matches, or the window to 12–19.** Two independent
   experiments now agree (`r95` sweep, H4 here).
4. **Add the late-game share as a game-state proxy**, named for what it measures.
5. **Do not pursue** rest, minutes load, over-performance reversion, dual-horizon kernels, or
   referee effects. All measured, all null or unavailable.
6. **Do not add opponent-adjusted pxG supremacy.** Its value is already captured by `dyn.α/β`.
   The *level* version is worth one more look on totals.

## 6. Limitations

- **Multiple testing.** Nine candidates at a 90% level: roughly one false positive expected.
  `wealth_x_rapm` is the most likely, being the smallest effect. Re-confirm on a different split.
- **402 held-out matches** on the deployment store. Effects below ~0.001 nats are not resolvable.
- **`r62` is not the engine** — MAP Poisson, independent sides, no dispersion, no time decay, no
  Dixon-Coles or copula. It ranks candidates; `r40` arbitrates.
- **Only bench_value and late_share replicate directionally on both stores**, and neither reaches
  significance on England. The two stores differ in coverage (76.2% vs 56.6% stint coverage) and in
  competitive structure.
- The H9 game-state interpretation is a hypothesis consistent with the sign, not a measurement.

## 7. Reproducing

```bash
source .env       # BF_DB_URL, needed for the English confirmation pass only
julia --project -t 8
```
```julia
include("current_development/scottish_lower/r60_form_kernel_forensics.jl")
include("current_development/scottish_lower/r61_squad_dynamics_forensics.jl")
include("current_development/scottish_lower/r62_feature_gauntlet.jl")   # R62_ENGLISH=0 to skip England
```
