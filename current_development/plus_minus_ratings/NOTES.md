# plus_minus_ratings — RAPM player ratings for the Scottish lower leagues

Started 2026-07-23. Plan: `~/.claude/plans/home-james-bet-project-docs-modern-port-parallel-pebble.md`.
Source paper: Kharrat, López Peña & McHale (2019), *Plus-Minus Player Ratings For Soccer*
(`~/bet_project/docs/modern_portfolio_theory/Plus–minus ratings for players in 11-a-side association football.md`).

## Goal

The Ireland-family L1 engines feed on **SofaScore player ratings**
(`PlayerRatingsFeature` → 8 vectors `flat_{home,away}_{G,D,M,F}_rating`,
`src/features/extractors/player_extractors.jl`). Scottish League One (**56**) and League Two
(**57**) have **zero** SofaScore ratings, so they are stuck at team level (current winner: the
`scottish_lower_smile` iso-level pillar).

Build a home-grown **regularized adjusted plus-minus (RAPM)** rating for 56/57 out of what those
leagues *do* have — lineups, minutes, incidents, BBC commentary — and validate it on the upper
Scottish tiers (**54** Premiership, **55** Championship) where the SofaScore rating exists as a
yardstick. Ratings are built from **lower-league-available features only**, so whatever passes on
54/55 transfers verbatim to 56/57.

This is "Option 2 — prior-informed Bayesian RAPM", deferred to in
`../position_aware_ratings/NOTES.md` (2026-06-28).

## Scope (locked with user, 2026-07-23)

- **Four PM targets**: goals-PM (paper baseline), shots/SoT-PM (repo-specific, denser),
  xG-PM (BBC-text xG model), xP-PM (in-play expected points).
- **Estimator**: sparse **ridge first** (paper-faithful reference), **then** a Turing
  hierarchical / prior-informed Bayesian arm that must beat it.
- **Decisive gate**: **rating-agreement statistics only** — no L1 engine retraining this stream.
- Opened with a `/deep-research` pass → `RESEARCH_rapm.md`.

## Data facts (betdb, verified 2026-07-23 via the server REPL)

| tier | matches | lineups | ratings | player xG | BBC live_text | incidents (subs) |
|---|---|---|---|---|---|---|
| 54 Prem | 1188 | 1188 | **all 6 seasons** | 25/26 only (198) | 23/24–25/26 (594) | complete |
| 55 Champ | 1030 | 1030 | 23/24–25/26 (530) | 23/24–25/26 (530) | 23/24–25/26 (535) | complete |
| 56 L1 | 985 | 956 | **none** | none | 23/24–25/26 (535) | **holed**: 23/24 71/180, 25/26 16/175 |
| 57 L2 | 985 | 953 | **none** | none | 23/24–25/26 (535) | complete |

1. **56/57 have full lineups with `position` and `minutes_played`** — only `rating` is null.
   RAPM needs exactly what they have.
2. **`sofascore.match_incidents.data` carries player IDs**, not just slugs:
   `data->'playerIn'->>'id'`, `data->'playerOut'->>'id'`, `data->'player'->>'id'`,
   `data->'assist1'->>'id'` → join directly to `match_player_lineups.player_id`.
   The repo fetcher (`src/Data/fetchers/sql/incidents.jl`) extracts **slugs only**, so this
   stream needs its own SQL (pattern: `../bbc_xg_proxy/l03_funnel_cascade.jl`).
   Cards: `incidentClass ∈ {yellow, red, yellowRed}` with player IDs
   (56: 78 red + 58 yellowRed; 57: 102 + 89).
3. **`minutes_played` is dead before 23/24** (identically 0 for 20/21–22/23) and **NULL for most
   of 25/26** (2310/3850 starters in t=57). ⇒ **incidents are the primary segment source**;
   minutes are a cross-check, not the base.
4. **BBC `live_text` is Opta-style commentary**, e.g.
   `"Attempt saved. Kai Kennedy (Queen of the South) right footed shot from outside the box is
   saved in the top right corner by Robbie Mutch (Cove Rangers)."`
   ⇒ parseable body part / zone / set-piece context / shooter / keeper.
   Player-bearing event types: `goal, attempt_missed, attempt_saved, attempt_blocked, post,
   penalty_*, yellow_card, red_card, second_yellow_card, foul, free_kick_won`.
   **`substitution` rows have no `player` column** — names live only in `text`.
5. **Players**: 54:1278, 55:1047, 56:1163, 57:1116; **3,192 distinct pooled** across 54–57, of
   which **576 (18%) appear in both the upper (54/55) and lower (56/57) groups** — these
   cross-tier movers identify the league-strength coefficients (paper §4.3) and give a free
   external-validity test.
6. 54/55 have odds (1184/1029 matches) → an engine-level A/B on the upper tiers is feasible
   later; **out of scope** this stream.

## Central statistical risk

Segments-per-player. At ≈8 segments/match: pooled 54–57 all seasons ≈ 33k segments vs 3,192
players (**≈10:1**); live-text era only (~2,199 matches) ≈ 17.6k vs ~2,500 (**≈7:1**). The paper
had 130k/10,983 ≈ **12:1**, on leagues with *more* rotation than ours. The paper's own caveats
(always-together players get identical ratings; low-minute players collapse to the global mean;
goalkeepers become a comparison against their backup) bite harder here. The dense targets
(SoT, xG) and cross-tier pooling are the two levers; **WP2's ≥5:1 gate** is where we find out
whether they are enough.

## Spec changes adopted from WP0 (see `RESEARCH_rapm.md` §6)

The deep-research pass found Hvattum, Arntzen & Pantuso (2020) — a more developed formulation
than the base paper — and it changes the build in five places:

1. **Segment weight is a product of three factors**, not just time decay:
   `w = w^TIME · w^DURATION · w^GOALS`, where `w^DURATION = (d + ρ₂)/ρ₃` downweights short
   segments and `w^GOALS = ρ₄` when the segment both starts and ends ≥2 goals apart
   (garbage-time discount).
2. **Teammate-similarity shrinkage** replaces the box-score prior as the headline low-minutes
   treatment: penalise a player's rating toward the ratings of the players he has shared the
   most minutes with, dial `w^SIM`. Needs no data we don't already have, and it is the
   published *football* method. Always report `w^SIM = 0` (plain ridge) beside the tuned value.
3. **League adjustment** becomes Hvattum's per-player competition average
   `(1/|C_p|) Σ_{c∈C_p} β_c^COMP` — simpler than the base paper's `m_il` rule and a better fit
   for our four-tier pool.
4. **Unreplaceable injured players** (substitutions exhausted) are modelled as dismissals — free.
5. **Penalties get one constant xG.** Even with coordinates, no model in the base paper beat the
   penalty base rate (Brier 0.1848 flat).

And it **inverts the reading of WP7.1**: Gelade & Hvattum (2020) measure bottom-up event stats
explaining only **22–38%** of plus-minus variance (GK lowest, forwards highest) ⇒ expect
ρ ≈ **0.47–0.62** against SofaScore. Near-zero means we are broken; **near-1 would mean we had
merely rebuilt the box score**. Low-ish correlation is the success criterion here, so the
decisive evidence must come from reliability (WP7.2) and validity (WP7.4). Minutes floor is
**540** for analysis, **900** for top-N tables. The ≥5:1 segments-per-player gate is our own
engineering judgement, **not** a literature standard — the real evidence is the measured
standard errors and effective degrees of freedom.

## Files

| File | Role | Status |
|---|---|---|
| `NOTES.md` | This — goals, gates, findings log | live |
| `RESEARCH_rapm.md` | WP0 deep-research output (verified; 6 bad citations caught) | **done** |
| `l00_pm_data.jl` | Own SQL loaders + `.jls` caches | **done** |
| `r00_data_qa.jl` | WP1 coverage / integrity gate | **done — PASSED** |
| `l01_segments.jl` | Segment builder → sparse `X`, weights, dismissal + league dummies | **done** |
| `r01_segment_qa.jl` | WP2 segment QA + identifiability gate | **done — PASSED** |
| `l02_shot_parser.jl` | BBC commentary → shot descriptors | **done** |
| `r02_shot_xg.jl` | WP3 shot xG model + calibration vs SofaScore player xG | **done — PASSED** |
| `l03_targets.jl` | The five segment targets + in-play hazard/xP table | **done** |
| `r03_targets_qa.jl` | WP4 target sparsity comparison | **done — PASSED** |
| `l04_ridge_apm.jl` | Sparse ridge RAPM + CV over (λ, ζ, w_SIM) | **done** |
| `r04_ridge_fit.jl` | WP5 fits | **done — PASSED (small effect)** |
| `l05_bayes_apm.jl` | Turing hierarchical / prior-informed RAPM | pending |
| `r05_bayes_fit.jl` | WP6 Bayesian vs ridge | pending |
| `r06_vs_sofascore.jl` | **WP7 decisive gate** | pending |

## Decision rule (WP7)

Green-light integration only if, for ≥1 target × estimator cell: split-half reliability is **≥
the SofaScore rating's**, **and** the team-strength retrodiction is **not materially worse** than
the SofaScore-fed model on held-out Brier. A clean negative is a valid outcome.

## Findings log
<!-- YYYY-MM-DD — WP / gate — result. Append newest-first. -->

- 2026-07-23 — **WP5 (r04_ridge_fit.jl): the ratings carry REAL but SMALL signal — and the
  specification check earned its keep by catching two bugs first.**

  **Two bugs, found because the first run produced an impossible result.**
  1. **Shot side attribution (DATA — corrupted WP3 and WP4 too).** The first run returned a
     *negative* home-advantage coefficient for shot targets but positive for goals. That split
     pointed at attribution, not the ridge. Cause: `(lt.team = mm.bbc_home_slug)` returns
     **FALSE, not NULL**, when the slug matches *neither* side, so every unmatched slug was
     silently attributed to **away**. It hit **7,073 of 45,201 shot rows (15.6%)** and reversed
     the measured home shot advantage (our home mean 9.21 vs BBC ground truth 11.16). The cause
     is a slug variant — `dundee-fc` vs `dundee`, `clyde-fc` vs `clyde`. Normalising the trailing
     `-fc` resolves **all 7,073**, none left over, so they are recovered rather than dropped.
     **Effect of the fix on WP3: team-level xG correlation vs SofaScore 0.698 → 0.817**, MAE
     0.454 → 0.372, bias +0.061 → −0.046. WP3/WP4 numbers above have been amended.
  2. **Link calibration (HARNESS).** Every arm scored at or *below* the no-information floor,
     goals by 0.064 Brier — impossible for a model that can at worst ignore its covariate. The
     ordered logit was being fit on TRAINING matches, whose ratings had been fit on them, so the
     strength covariate was far more predictive there than out-of-sample; the logit learned an
     inflated slope and was overconfident on evaluation. I had explicitly reasoned in the
     docstring that this was "small, shared optimism" — **that reasoning was wrong.** Link and
     floor are now both fit on the evaluation season (3 params on ~640 matches); the ratings stay
     strictly out-of-sample.

  **Results after the fixes** (Δ Brier vs a no-information floor fit on the same matches;
  negative = the ratings help; 1,280 evaluation matches over 24/25 + 25/26):

  | target | best λ | half-life | w_SIM=0 | best w_SIM | Δ Brier @ best |
  |---|---|---|---|---|---|
  | `y_shots` | 1000 | 730 | −0.01010 | 0.90 | **−0.01219** |
  | `y_xg` | 200 | 730 | −0.00895 | 0.75 | −0.00982 |
  | `y_sot` | 1000 | 730 | −0.00649 | 0.75 | −0.00756 |
  | `y_goals` | 1000 | 730 | −0.00447 | 0.90 | −0.00771 |
  | `y_goals_cov` | 20 | 365 | −0.00451 | — | — |
  | `y_xp` | 1000 | 730 | −0.00319 | 0.90 | −0.00657 |

  1. **The denser-target hypothesis is CONFIRMED on the fair comparison.** WP4 required goals to
     be refit on the same 52.2% subset the shot targets live on: `y_xg` **−0.00895** vs
     `y_goals_cov` **−0.00451** — xG roughly **doubles** the goals arm's edge on identical data.
     `y_shots` is better still. Denser targets genuinely buy discrimination, exactly as the base
     paper argued.
  2. **Teammate-similarity shrinkage helps EVERY target, monotonically** — validating the WP0
     research find (Hvattum et al. 2020 §3.1) and **overturning the caution I recorded from
     Gelade & Hvattum** that informed priors buy little in football. Gains: goals −0.00447 →
     −0.00771 (+73%), xP −0.00319 → −0.00657 (+106%), shots −0.01010 → −0.01219 (+21%).
  3. **But the effect is SMALL.** Best Brier **0.63322** against a floor of **0.64541** — a
     **1.9% relative** improvement. And λ optimises at **1000**, the grid boundary, with
     `sd_players` collapsing to 0.06: most per-player variation is noise, and what survives is a
     thin consistent signal. w_SIM also optimises at **0.90**, again the boundary. Both grids
     need extending before any of these are treated as tuned values.
  4. **Specification check vs the base paper's Table 4 — signs PASS.** Home advantage
     **+1.5135** (correctly positive; ≈1.5 more shots per 90, sane). Red cards **−10.66** and
     **−11.16** (clearly negative; the magnitude is right for a shot *difference*, since a
     dismissal both suppresses your shots and lifts the opponent's). Two defects: red-card 3 is
     exactly 0 (no team in the sample took a third dismissal with the opponent still at eleven),
     and reds are **not monotone** in severity as the paper's are.
  5. **League coefficients are NOT ordered by tier** — 54 +0.283, 57 −0.027, 55 −0.099,
     56 −0.157. The expected ordering is 54 > 55 > 56 > 57. Identified only through cross-tier
     players, and evidently too noisy to trust. **Do not read these as league-strength estimates.**
  6. **Face validity is strong.** Top 15 at ≥900 minutes are *entirely* Celtic and Rangers
     players (O'Riley, Taylor, Kyogo, Johnston, Carter-Vickers, Hart; Tavernier, Butland,
     Goldson, Lundstram, Dessers). Bottom 10 are lower-tier players. `cor(rating, log minutes)
     = 0.097` — **the rating is not merely measuring playing time.**

  - **⚠ THE TENSION TO CARRY INTO WP7, and it is structural, not incidental.** The tuning
    criterion is match-outcome Brier, so it **rewards a rating for recovering team strength** —
    team strength predicts results. `w_SIM` improves that criterion *precisely by making ratings
    more team-like* (it shrinks each player toward his most frequent teammates), which is also
    why the top 15 are one-club blocks. So WP5's winner selection is partly **at odds with**
    WP7's requirement that the rating not just be team strength in disguise — the exact pitfall
    Gelade & Hvattum name. WP7 must therefore (a) report split-half reliability at several
    `w_SIM` values, not only the Brier-optimal one, and (b) explicitly measure how much of each
    rating's variance is explained by a team fixed effect, and prefer a cell that keeps
    within-team spread rather than the one that maximises outcome Brier.

- 2026-07-23 — **WP4 (r03_targets_qa.jl): ALL GATES PASSED.** Five targets built over the WP2
  segments. **The sparsity ladder works exactly as the base paper argued it would.**

  | target | % of segments = 0 | sd | sd of per-90 rate | cor with goals |
  |---|---|---|---|---|
  | `y_goals` | **72.1** | 0.690 | 7.50 | 1.000 |
  | `y_sot`   | 52.2 | 1.263 | 14.12 | 0.437 |
  | `y_shots` | 34.3 | 2.371 | 25.44 | 0.271 |
  | `y_xg`    | **25.7** | 0.386 | **4.70** | 0.381 |
  | `y_xp`    | **0.4** | 1.008 | 12.67 | 0.881 |

  1. **The two candidates are xG and xP, and they trade off differently — this is the WP5
     decision.**
     - **`y_xg` is the densest genuinely NEW signal** (25.5% zero vs 72.1%) and has by far the
       *lowest* per-90 variance (4.70 vs goals' 7.50) — it is a denoised goal rate, which is the
       whole point. **But it costs 47.8% of the sample**: live_text only starts in 23/24, so the
       shot-based targets exist on **11,886 of 22,785 segments (52.2%), 1,895 matches**.
     - **`y_xp` is dense (0.4% zero) AND available on the FULL 22,785 segments** — it needs only
       incidents, not live_text. Free density. **But its 0.881 correlation with goals means it
       is substantially a game-state-weighted re-expression of the goals target rather than
       independent information.**
     ⇒ WP5 must run both, and the shot-based arms must be compared to a goals arm **refit on the
     same 52.2% subset**, or xG gets credit for merely being the more recent data.
  2. **T5 in-play model: near-exact.** Expected points at kickoff **1.520 home / 1.237 away**
     against an empirical **1.529 / 1.210**. (Base paper's EPL: 1.63/1.11 — a lower league should
     be flatter, and is.) Hazard coefficients all correctly signed: time bins rise monotonically
     to **+0.341** in the last, manpower **+0.484 (z=13.0)**, home **+0.160 (z=8.1)**. Isolating
     manpower properly (HT level 11v11 → 1.435; HT level a man up → 1.509) gives **+0.074 xP**.
  3. **T1 clock alignment PASSED**: BBC and SofaScore agree with **no systematic offset**
     (mean diff −0.28 min, median 0, 90% within ±1 min, 94.5% within ±2, p95 = 3 min) over 5,208
     paired goal timings in 1,771 matches. **23.9%** of shots sit within ±2 min of a segment
     boundary and are therefore exposed to that spread — but the spread is symmetric, so the
     residual effect on the shot-based targets is noise, not bias.
  4. **T2 attribution: 100.00% exact** — every shot lands in exactly one segment, mean shortfall
     0.000.
  - **Caveat inherited from the base paper's design (§4.2), worth stating plainly:** the in-play
    model is deliberately **team-strength-blind**, so goal difference acts as a strength proxy as
    much as a game-state effect — a side three goals up shows a *higher* scoring rate (+0.353)
    partly because it is the better team. That is intended (conditioning on team strength would
    double-count the very thing plus-minus is estimating), but it does mean **xPPM rewards
    players for being ahead in a way that partly reflects their team**. Relevant to the WP7
    pitfall "ratings that merely recover team strength".
  - Two QA bugs found and fixed in this WP, both in the checks rather than the data: `combine`
    flattening the per-match goal-time vectors turned T1's join into a cartesian product (the
    gate "failed" at p95 = 68 min); and the T5 manpower check compared against kickoff, which
    conflates the manpower gain with the shrinking time remaining and made a correct model look
    broken.

- 2026-07-23 — **WP3 (r02_shot_xg.jl): ALL SEVEN GATES PASSED.** A zonal xG model from BBC
  commentary is real and usable. 45,201 shots over 2,199 matches, 13.28% converted.
  1. **X1 parse coverage: 98.4–99.8%** on every tier × season (zone phrase 97.2–98.5%, body part
     99.1–99.7%). The closed vocabulary holds. Gate was ≥95%.
  2. **X2 face validity: clean football ordering.** Zones: six-yard centre **51.0%** → six-yard
     side 21.4% → box centre 15.3% → difficult angle 8.9% → box side 8.2% → outside box 4.3% →
     very long range 3.5%. Contexts: fast break 34.9% > corner 26.8% > set piece 24.9% > direct
     free kick 15.8% > open play 10.0%. Penalties 76.7% (→ the single constant xG).
  3. **X3 ladder** (leave-one-season-out): Brier 0.10952 (base rate) → 0.10032 (zone) → 0.09920
     (+body) → **0.09735** (+context) = **11.1%** improvement. The base paper's *coordinate*-based
     open-play model improved 14.7% on its baseline, so we retain **~76% of the coordinate-model
     gain with no coordinates at all** — squarely in line with the "real but bounded loss"
     RESEARCH_rapm.md §5.1 predicted. Total xG 5,715 vs 6,004 goals (ratio 0.952; own goals are
     not shot events, which accounts for most of the gap).
  4. **X5 team-level vs SofaScore (the primary calibration gate, needs no name matching):**
     pooled **cor 0.817**, MAE 0.372, bias −0.046 over 1,349 team-innings / 728 matches.
     *(AMENDED 2026-07-23 after WP5 found the shot side-attribution bug; the pre-fix figures
     were cor 0.698 / MAE 0.454 / bias +0.061.)* For
     reference `bbc_xg_proxy`'s frozen team GLM reached Spearman 0.715 / R² 0.442 against the
     same target — so this is **comparable to the existing frozen proxy, but per-shot**, which
     is the whole point: an aggregate match-level GLM cannot be assigned to a segment, and this
     can. Caveat: t54 25/26 over-predicts (bias **+0.39**, mean 1.81 vs 1.42) while t55 is within
     ±0.13 — Premiership shot volume is higher and the shared cell rates over-serve it.
  5. **X6 player-level: cor 0.766, MAE 0.113, bias 0.004** over 8,308 player-matches — *better*
     than team level.
  6. **X7 tier transfer: essentially free.** Upper→L1 Brier 0.11256 vs **in-sample L1 0.10981**
     (gap 0.0028); Prem→Champ 0.10291 vs in-sample Champ 0.09908 (gap 0.0038). Upper→L2 0.10481.
     This is exactly the production operation and it costs almost nothing. (Champ→Prem's 0.08569
     is not comparable — different test set, different base rate.)
  - **LEAKAGE FOUND AND FIXED, caught by the X2 face-validity gate:** 141 shots described as
    "from a free kick with a right footed shot" converted at **100.0%** — BBC uses that phrasing
    *only* in goal descriptions, so a cell keyed on it predicts xG ≈ 1.0. The model was reading
    the outcome off the wording rather than the chance quality. It was worth **2.1 Brier points
    of spurious gain** (the ladder read 13.2% improvement before the fix, 11.1% after) and would
    have injected 141 phantom xG=1.0 events straight into the WP4 xGPM target. Now remapped to
    (direct-free-kick context, modal outside-box location), which is what the phrasing genuinely
    tells us. **This is why the face-validity gate exists** — no aggregate metric would have
    flagged it.
  - **Two carried caveats.** (a) **2.44% of shots (1,105) cannot be attributed to a side** — the
    BBC team slug matches neither home nor away — and are dropped; the same rows will be unusable
    for the WP4 segment targets. (b) **Calibration wobbles in the middle**: deciles 1–6, 8 and 10
    are within ±0.01, but decile 7 under-predicts by 0.052 and decile 9 over-predicts by 0.032.
    Tie-driven (a cell model emits few distinct values), tolerable for a segment-sum target, but
    it should be revisited if xGPM ends up the winning target.
  - **DECISION — do NOT wire the tier-56 live_text substitution hole-fill.** X6 measured the
    number WP1 could not: BBC shooter names resolve to a `player_id` only **93.2%** of the time
    (t56 **91.2%**, t57 92.5%, t55 95.2%, t54 93.8%). The hole-fill would reconstruct lineups
    through exactly this mechanism, so ~9% of tier-56 substitutions would attach to the wrong
    player — and a substitution error corrupts *both* players' on-pitch intervals for the rest of
    the match. Recovering 268 matches (≈7% more sample) is not worth silently poisoning the
    segments. The 281 `no_incidents` matches stay excluded.

- 2026-07-23 — **WP2 (r01_segment_qa.jl): GATE PASSED — and the collinearity fear did not
  materialise.** 22,785 segments over 3,711 matches; 416 matches rejected.
  1. **The segment builder validates against the base paper.** **72.2%** of segments have goal
     difference 0 — the paper reports **72%** on 20,868 European top-flight matches. Landing on
     their number from completely different data is the strongest single check available that
     the segment walk is correct.
  2. **S3 ratio**: 2,692 players / 22,785 segments = **8.5 : 1** overall, but **12.2 : 1** among
     the 1,865 players above the 540-minute floor — i.e. **equal to the base paper's ≈12:1**
     once you exclude the players nobody could rate anyway. 30.7% of players are below 540 min.
     32.6% appear in more than one tier, which is what identifies the league columns.
  3. **S4 always-together clusters: a NON-ISSUE.** Only **15 players (0.6%)** sit in exactly-tied
     clusters, largest cluster 3, and every one of them is a 1–270 minute player. The paper's
     centre-back-pairing caveat does not bite here — six seasons of lower-league rotation breaks
     the ties. This was the risk I flagged hardest in the plan; it is empirically dead.
  4. **S5 identifiability** (weights normalised to mean 1): 38 of 2,700 directions rank-deficient.
     edf and prior-dominated share across λ: λ=0.1 → edf 2223 (82%), 4.0% prior-dominated;
     λ=1 → 1686 (62%), 8.6%; λ=10 → 951 (35%), 17.2%; λ=100 → 348 (13%), 39.4%.
     `cor(log posterior variance, log minutes)` is **−0.87 … −0.74** throughout, so precision
     tracks exposure exactly as it should and there is no variance saturation. **A workable λ
     regime clearly exists** — WP5 tunes inside roughly λ ∈ [0.1, 10].
  5. **S2 shape**: ~6.0–6.6 segments/match (I had assumed ~8, so the raw budget is ~25% thinner
     than planned — the ratio still passes). Duration median 8 min, 28% of segments under 5 min;
     garbage-time segments are 23.5% of segments but only 11.0% of minutes; manpower imbalance
     9.8% of segments / 5.0% of minutes.
  6. **S1 rejects (416)**: 281 `no_incidents` (the tier-56 holes — the cost of not yet wiring the
     live_text fallback), 72 `sub_in_unknown` + 62 `sub_out_off` (WP1's incomplete bench scrape),
     1 `starters_ne_11`. Note the WP1 theory refines: those matches have 11 real starters and a
     truncated *bench*, so they fail at the substitution, not the teamsheet-size check.
  - **BUG FOUND AND FIXED mid-gate:** weights were not normalised, so λ and the half-life were
    confounded — `mean(w) = 0.041` at half-life 365d over six seasons silently inflated every λ
    on the grid ×24. Un-normalised, the gate read "52.7% of players prior-dominated at λ=10"
    (alarming) and `cor(log var, log min) = −0.15` (a saturation ceiling). Normalised, the same
    quantities are 17.2% and −0.78. Had this survived, WP5's (λ, half-life) tuning would have
    been partly chasing a units artefact.

- 2026-07-23 — **WP1 (r00_data_qa.jl): GATE PASSED.** The incident route is validated; the
  segment builder can be written against it.
  1. **G3 (decisive) — incident-reconstructed minutes vs `minutes_played`, 23/24+24/25.**
     Starters: t54 MAE **0.07** min / **99.9%** within ±1; t57 **0.07** / **99.9%**;
     t56 **0.80** / **99.0%**; t55 **0.93** / **87.8%** (96.9% within ±3). Three of four tiers
     essentially exact ⇒ **the incident reconstruction is correct**; tier 55's spread is a
     SofaScore-column artefact, not an incident-route failure (its *substitutes* are far worse,
     MAE 3.47, 37% within ±1, which is SofaScore rounding the entry minute).
  2. **G2 — substitution id resolution**: 100.0% (54), 99.7% (55), 96.1% (56), 97.0% (57);
     46 NULL-id rows dropped. The 3–4% lower-tier misses are **diagnosed, not mysterious**:
     unresolved rows sit on teamsheets averaging **9.2 players (min 0)** vs **36.9** for
     resolved ones ⇒ **incomplete bench scrape**, not id drift. Those matches have unusable
     lineups anyway — **drop the match**, don't patch the player.
  3. **G1 — coverage**: as tabulated above. Tier-56 incident holes total **281 matches**
     (22/23: 13, 23/24: 109, 25/26: 159).
  4. **G4 — hole fill**: **all 268** holed matches in 23/24 + 25/26 have live_text, and the BBC
     substitution phrasing parses at **100.0%** (3,790/3,790 rows, tier 56). 22/23's 13 holed
     matches have **no** live_text → drop them. Caveat carried forward: a clean parse yields
     **names, not ids**; name→player_id matching against the teamsheet is a separate,
     unmeasured risk that WP2 must quantify before this fallback is trusted.
  5. **G5 — goal reconciliation vs final score**: 95.2–100% per tier×season (t54 100% across
     all six). Mismatches to be listed and excluded, not tolerated.
  6. **G6 — positions**: unknown (`U`) share is small — t54 0, t55 69, t56 410, t57 360 rows out
     of ~35k each (~1%). Recorded rather than silently folded into midfield, unlike
     `src/features/extractors/player_extractors.jl`.
- 2026-07-23 — **WP0 deep-research done** → `RESEARCH_rapm.md`. Five spec changes adopted (see
  the section above). The fanout produced **six wrong or fabricated citations** (a Hvattum paper
  attributed to the journal *Water*; a 404 GitHub repo; a wrong arXiv id; three bare domains),
  all logged in `RESEARCH_rapm.md` §7 — the *verified* primary sources are what the spec changes
  rest on. Two worker claims were checked and **overturned**: "split-half reliability is
  unsuitable for RAPM" (it is literally Hvattum & Gelade's reliability axis) and "RAPM does not
  correlate with commercial ratings" (it correlates at ρ ≈ 0.47–0.62).
- 2026-07-23 — stream opened; data facts above verified directly against betdb via the server
  REPL (the betdb MCP is unreachable off the home network).
