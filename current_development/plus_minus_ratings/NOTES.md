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
| `r06_vs_sofascore.jl` / `r07_best_players.jl` / `r08_reliability.jl` | **WP7 decisive gate** | **done — PASSED** |
| `r10_src_experiment.jl` | **WP-D** — the src L1 engine sweep vs the two baselines | see log |

## src graduation (branch `feat/apm-player-rating-l1`)

The rating stopped being a table and became an L1 feature + engine. What moved where:

| src path | what it is |
|---|---|
| `src/Data/fetchers/sql/bbc_events.jl` | 9th DataStore domain `ds.bbc_events` — raw BBC shot commentary (`ds.bbc` stays per-match totals for the funnel) |
| `src/Data/fetchers/sql/incidents.jl` | now also extracts the jsonb **player ids** (was slugs only) |
| `src/Data/fetchers/sql/matches.jl` | now also carries `injury_time1/2` (the monotone match clock needs it) |
| `src/features/plus_minus/` | `l01`–`l04` ported: segments, weights, competition sets, design, shot parser + zonal xG, the four targets, ridge |
| `src/features/types.jl` | `AbstractPlusMinusFeature` + one struct per target + `pm_target` + `rating_base` |
| `src/features/extractors/plus_minus_extractors.jl` | one shared `add_feature!` emitting the standard 8-vector rating contract |
| `.../player_level/time_decay/goals_plus_minus_league.jl` | `DynamicGoalsPlusMinusLeagueTimeDecayModel` |

`y_xp` was **not** ported: it was last of five in WP5 (Δ Brier −0.00319), 0.881 correlated with
`y_goals`, and it alone would have dragged an in-play hazard GLM plus a backward-induction table
into `src/`.

## Decision rule (WP7)

Green-light integration only if, for ≥1 target × estimator cell: split-half reliability is **≥
the SofaScore rating's**, **and** the team-strength retrodiction is **not materially worse** than
the SofaScore-fed model on held-out Brier. A clean negative is a valid outcome.

## Findings log
<!-- YYYY-MM-DD — WP / gate — result. Append newest-first. -->

- 2026-07-28 — **THE MONEY LENS IS THE ONE THAT SEPARATES THESE MODELS — AND IT MUST BE PRICED ON
  BETFAIR, NOT THE SOFASCORE/BET365 CLOSE. Log-loss called the top four arms a statistical tie;
  growth does not.** `BayesianKelly`, curated to O/U + BTTS (1X2 and CorrectScore dropped, see
  below), 710 OOS matches.

  | model | bets | turnover | ROI % | **G** | ROI @5% comm | ROI on Bet365 |
  |---|---|---|---|---|---|---|
  | **funnel_apm_xg** | 1126 | 57.1 | **+6.75** | **+0.0396** | +6.41 | −9.51 |
  | funnel_winner | 1133 | 58.8 | +6.45 | +0.0390 | +6.12 | −9.41 |
  | apm_pillar_only | 1105 | 61.0 | +3.74 | +0.0284 | +3.55 | −13.28 |
  | apm_shots | 1085 | 42.7 | +1.66 | +0.0253 | +1.58 | −14.01 |
  | apm_xg | 1091 | 42.1 | +1.07 | +0.0202 | +1.02 | −12.05 |
  | goals_baseline | 1017 | 17.0 | +0.82 | **−0.0062** | +0.78 | −18.76 |

  **① EXECUTION VENUE IS THE WHOLE GAME.** The identical curated book is **−9.5% ROI / G −0.054**
  priced at the SofaScore ("Bet365") close and **+6.8% / G +0.040** priced on Betfair. Nothing
  about the models changed — the sign flip is entirely the overround:

  | market | Betfair | SofaScore/Bet365 |
  |---|---|---|
  | 1X2 | 1.0001 | 1.100 |
  | O/U | 0.9974 | 1.065 |
  | BTTS | 1.0061 | 1.079 |

  Reproduces `staking-real-mvp`'s design (Ireland results were Betfair-close) and
  `betfair-vs-bet365-market-anchor`: **anchor the model to the de-vigged Bet365 line, execute on
  Betfair.** Any economic verdict quoted off `ds.odds` for this segment is measuring the bookmaker's
  margin, not the model.

  **② The APM pillar and the funnel BOTH earn their keep, and `goals_baseline` is the only arm with
  negative growth.** Funnel family G ≈ 0.039 ≫ APM-only 0.020–0.025 > baseline −0.006.

  **③ Totals is where the edge is, on every lens.** log-loss (O/U −0.0019 to −0.0045 vs market),
  GLMEdge (funnel arms z ≈ 4.0 on O/U, APM-only z ≈ 1.3) and growth all agree. Consistent with
  `totals-compression-is-denoising`: the edge is fading market over-dispersion, not out-predicting
  the market's mean.

  **④ `funnel_apm_xg` vs `funnel_winner` is a TIE** — ROI 6.75 vs 6.45, G 0.0396 vs 0.0390. Do not
  read the fusion as a winner on a 1.5% relative G difference; it is not significance-tested.

  - **⚠ 1X2 ON BETFAIR IS A TRAP: +17.8% to +22.5% ROI with G ≈ 0 or NEGATIVE**
    (`funnel_winner` +17.8% ROI, G **−0.0023**). Positive return, negative growth = a handful of
    large-priced winners carrying the sample. Not a real edge; do not bet it. This is the classic
    per-line curation result (`unified-staking-r01-findings`: weight 0 on 1X2) arriving again.
  - **⚠ CorrectScore is pure longshot noise here** — ~180 bets on turnover ≈ 2.0 producing G
    0.08–0.16, the largest of any family. `staking-real-mvp` measured it as a −20% ROI DRAG on
    Ireland; an estimate that flips sign between samples is variance. Curate out.
  - **⚠ THE BIGGEST UNRESOLVED RISK IS FILLABILITY, not the model.** These are 20-minute TWA prices
    on tournaments 56/57, and `inplay-scottish` measured that exchange as thin (median ≈49
    MATCH_ODDS prints per match on t=56, fewer on t=57). A recorded price is not a fillable one at
    size. The 1X2 anomaly in the previous bullet is what unfillable longshot quotes look like.
    **Verify pre-off depth before believing the LEVEL of these returns** — the ORDERING looks
    robust across all three lenses.
  - Commission is modelled crudely as 5% of gross profit; real Betfair commission is on net market
    winnings, so treat the `@5% comm` column as indicative.
  - ⚠ **Rebuild the DataStore with ALL NINE fields when swapping odds.** The idiom in the existing
    runners — `DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents,
    ds.betfair_odds)` — silently drops `bbc` AND `bbc_events` (verified: both → 0 rows). Because
    `run_backtest` → `extract_oos_predictions` → `create_features` REBUILDS features from the store
    you pass, that would degrade the funnel arms to goals-only and zero the ratings on
    `apm_shots`/`apm_xg`/the fusion (only `apm_goals` survives, since its target needs no live
    text) — six models silently collapsing to near-identical goals-only models, with nothing
    raising an error. Harmless for Ireland (no BBC coverage), lethal on ScottishLower.

- 2026-07-28 — **⚠ SIGNIFICANCE TESTING OVERTURNS HALF OF THE ENTRY BELOW. READ THIS FIRST.**
  The WP-D entry that follows ranks arms on point estimates alone. Paired per-observation log-loss
  differences, **clustered by match** (selections within a match are heavily dependent, so the
  naive t roughly doubles), on the 10,579 clean observations / 709 matches:

  | comparison | mean Δ | t (clustered) | % matches better | verdict |
  |---|---|---|---|---|
  | apm_xg vs goals_baseline | −0.00717 | **−4.53** | 68.3% | **real** |
  | apm_shots vs goals_baseline | −0.00713 | **−4.31** | 69.0% | **real** |
  | funnel vs goals_baseline | −0.00864 | **−3.97** | 68.1% | **real** |
  | apm_pillar_only vs goals_baseline | −0.00528 | **−2.62** | 66.1% | **real** |
  | apm_pillar_only vs apm_shots | +0.00185 | **+2.32** | 48.1% | **real (pillar_only WORSE)** |
  | funnel_apm_xg vs funnel | −0.00030 | −0.57 | 49.4% | ns |
  | funnel_apm_xg vs apm_xg | −0.00177 | −1.02 | 52.3% | ns |
  | apm_xg vs funnel (all) | +0.00147 | +0.77 | 47.2% | ns |
  | apm_xg vs funnel (1X2) | −0.00118 | −0.49 | 53.0% | ns |
  | apm_xg vs funnel (O/U) | +0.00248 | +1.12 | 44.8% | ns |

  **What is actually established:**
  1. **Every engine beats the no-APM baseline, decisively** (t −2.62 to −4.53, ~68% of matches).
     Q1 passes on its own terms — the APM pillar carries real information net of α/β.
  2. **`apm_pillar_only` is significantly WORSE than `apm_shots`** (t = +2.32). H2 is **rejected**:
     the rating adjusts team strength, it cannot replace it. This is a genuine finding, not a
     point-estimate impression.
  3. **NOTHING separates funnel / apm_xg / apm_shots / funnel_apm_xg.** Every pairwise comparison
     among the top four is ns. **Q2 is INCONCLUSIVE, not "the funnel wins".**

  **Two claims in the entry below are therefore WITHDRAWN:**
  - *"funnel_winner is the outright winner, not tied"* — it leads on point estimate but is not
    separable from the APM arms (t = 0.77).
  - *"a clean division of labour: APM owns 1X2, funnel owns totals"* — the 1X2 gap is t = −0.49
    and the O/U gap t = +1.12. The point estimates are consistent with the story, but a coherent
    pattern among non-significant differences is a hypothesis, not a result. **The r12 fusion
    experiment was motivated by this pattern and should be read in that light** — its own gain
    over the funnel is likewise ns (t = −0.57, better on 49.4% of matches, i.e. a coin flip).

  The test is not underpowered: it resolves the baseline comparisons at t ≈ −4 on the same data.
  Separating the top arms needs more matches (more leagues/seasons) or a different lens —
  growth/CLV, which is this project's preferred criterion anyway and is still outstanding.

- 2026-07-28 — **WP-D: the APM pillar WORKS. Rankings among the top arms are NOT significant —
  see the correction entry above.** Seven arms, ScottishLower, target seasons 24/25 + 25/26,
  22 walk-forward folds × 4 chains × 1000 samples (300 warmup), 710 OOS matches.

  **⚠ READ THE CLEAN TABLE, NOT THE HEADLINE `evaluate_experiments` OUTPUT.** `LogLoss` over ALL
  selections is CONTAMINATED on this DataStore by a Double Chance defect (see the separate entry
  below). Everything here excludes DC.

  `diff_ll = model − market`, market = **de-vigged closing line** (`prob_fair_close`, i.e. implied
  close ÷ overround, from `sofascore.match_odds` — the feed this repo labels "Bet365 close"; it is
  NOT Betfair). **Negative = the model BEATS the closing line.**

  | model | 1X2 | BTTS | O/U | all clean |
  |---|---|---|---|---|
  | **funnel_winner** | 0.00475 | 0.00060 | **−0.00435** | **−0.00186** |
  | apm_xg | **0.00357** | 0.00100 | −0.00187 | −0.00039 |
  | apm_shots | 0.00381 | 0.00078 | −0.00185 | −0.00036 |
  | apm_sot | 0.00577 | 0.00146 | −0.00178 | 0.00017 |
  | apm_goals | 0.00748 | 0.00136 | −0.00200 | 0.00036 |
  | apm_pillar_only | 0.00716 | 0.00207 | −0.00034 | 0.00149 |
  | goals_baseline | 0.00680 | 0.00142 | 0.00784 | 0.00677 |

  **① Q1 — apm_shots vs its no-APM twin: PASS, and not marginally.** −0.00036 vs +0.00677, a
  Δ of **−0.00713** on the clean aggregate. Every APM variant beats the twin. The pillar carries
  real information *after* free per-team α/β have already absorbed the team-level component, which
  is the conservative form of the test (WP7's central worry was that RAPM is team strength in
  disguise). The smoke-run posterior showed the engine actually using it: `w_att 0.148 ± 0.130`,
  `w_def 0.263 ± 0.112` — the defensive weight ~2.3 posterior sd from zero, net of team strength.

  **② Q2 — apm_shots vs the funnel: the funnel wins overall (−0.00186 vs −0.00036), but LOSES on
  1X2.** This is the result worth carrying forward. The funnel's advantage is *entirely* totals:
  −0.00435 on O/U against the APM arms' ≈ −0.0019. On 1X2 the ordering reverses — apm_xg 0.00357
  and apm_shots 0.00381 beat the funnel's 0.00475. **A clean division of labour: the APM pillar
  carries WHO WINS (lineup quality → 1X2); the funnel carries HOW MANY GOALS (shot volume →
  totals).** That is what motivates the r12 fusion arms.

  **③ Q3 — the target sweep orders xg ≈ shots > sot > goals, NOT WP7's reliability ordering, and
  that is coherent.** WP7 picked `y_shots` on split-half reliability but recorded `y_xg` as the
  *least team-loaded* cell (club R² 0.212 vs 0.389). In an engine that already models team
  strength with α/β, the least team-loaded rating should contribute the most *incremental*
  information — which is what happened. xg and shots are a dead heat overall (−0.00039 vs
  −0.00036); xg is slightly better on 1X2, shots slightly better on BTTS.

  **④ H1 vs H2 — the rating ADJUSTS team strength, it cannot REPLACE it.** `apm_pillar_only`
  (`StaticZeroDynamics`, α ≡ β ≡ 0, Ireland's `outfield_*` form) scores +0.00149: **better than
  the no-APM baseline (+0.00677), far worse than the same engine with α/β (−0.00036).** So the
  rating genuinely carries team-strength information — a team is partly its players — but not
  enough to stand alone. Consistent with what it is: a 730-day-half-life ridge that shrinks
  low-minute players to zero (663 of ~1,440 rated on fold 6), versus the per-match Kalman filter
  the Ireland engines consume. Priors were widened to `Normal(0, 1.5)` for that arm, since without
  α/β the weights must carry the whole between-team spread rather than a residual — at the r10
  prior of 0.3 it would have produced a null for a purely mechanical reason.

  - **CAVEAT ON apm_xg SPECIFICALLY.** Its target is built from the BBC-commentary zonal xG model
    (56/57 has **zero** SofaScore xG — verified: 0 player-xG rows, 0 matches flagged `has_xg`).
    That model is fitted GLOBALLY over all shots in the store rather than per fold — deliberate,
    since it is how the research computed the `y_xg` the WP7 verdict rests on, and the table
    carries no team or player identity. But it is the only arm with any look-ahead at all, and it
    is the arm that leads on 1X2. Do not rank it above `apm_shots` without refitting the cell
    table per fold; on the aggregate the two are tied anyway.
  - **Coverage:** `y_shots`, `y_sot` and `y_xg` exist only on live-text-covered matches (50.5% of
    segments, 23/24+); `fit_ratings` restricts them to `covered`. Only `y_goals` uses the full
    sample — and it is the weakest arm.
  - **Cost:** ≈58 min per APM arm (88 fold×chain tasks at ~42 s), ≈30 min for `apm_pillar_only`
    (49 parameters vs 101), ≈2.5 h for `goals_baseline` (see the efficiency note below).
  - **NEXT:** r12 fuses the pillar into the funnel — `funnel_apm_xg` (shot QUALITY per player,
    invisible to the funnel; predicted to help) and `funnel_apm_shots` (shot VOLUME decomposed to
    players, already exploited by the funnel; predicted redundant). If the division of labour in
    ② is real the fusion should land near −0.0030. `bbc_xg_proxy` r07b's funnel+iso fusion was
    null/soft-negative, so a null here would not be surprising.

- 2026-07-28 — **⚠ DOUBLE CHANCE IS BROKEN IN `ds.odds`, AND IT SILENTLY CORRUPTS ANY LogLoss
  SCORED OVER ALL SELECTIONS. This is a pre-existing src defect, not something this branch
  introduced — it affects every stream that has ever scored DC.**

  Two errors that are self-consistent for the market and punishing for the model:

  1. **`is_winner` marks only ONE DC selection per match when TWO should win.** Measured over
     ScottishLower (`processing.jl:111` does `out_df.is_winner = df.winning`, passing SofaScore's
     flag through unmodified):

     | 1X2 outcome | DC_1X | DC_X2 | DC_12 |
     |---|---|---|---|
     | home | 1.000 ✔ | 0.000 ✔ | **0.000 ✘** (should be 1) |
     | draw | 0.998 ✔ | **0.002 ✘** (should be 1) | 0.000 ✔ |
     | away | 0.000 ✔ | 1.000 ✔ | **0.000 ✘** (should be 1) |

     Average DC winners per match: **0.995**, not 2. `DC_12` is never marked a winner at all.
  2. **`prob_fair_close` halves DC probabilities.** `_enrich_market_data!` normalises every market
     group to sum to 1.0, but DC selections must sum to **2.0**. Measured mean vig by market:
     1X2 0.100, BTTS 0.079, O/U 0.065, **DoubleChance 1.161** (overround 2.16 — correct for DC,
     then wrongly normalised away).

  Because both are ~halved, the market looks *calibrated* on DC (mean fair p 0.333 vs win rate
  0.331) — which is why it survives a naive calibration check. But the MODEL prices `DC_12`
  correctly from its score grid at ≈0.72 and is then scored against a label saying it lost:
  ≈ −log(0.28) = 1.27 nats per row versus the market's ≈ 0.45. DC is ~13% of the 16,712 scored
  selections, so the artefact is worth **≈0.03 nats — larger than the entire model−market gap and
  ~6× every real effect in the WP-D table.**

  **It reversed the headline conclusion.** With DC included, every arm appeared to LOSE to the
  market (`diff_ll` +0.027 to +0.032) and `apm_xg` topped the table. With DC excluded,
  `funnel_winner` beats the closing line outright (−0.00186) and leads. Both the "we lose to the
  market" claim and the arm ordering were artefacts.

  - **Workaround in use:** score per market family and never aggregate over all selections —
    `LogLoss([:home,:draw,:away])`, `LogLoss([:btts_yes,:btts_no])`, `LogLoss(over/under…)`.
  - **Proper fix (NOT applied — it would change every historical comparison in the repo):**
    normalise DC to 2.0 in `_enrich_market_data!`, and derive DC `is_winner` from the 1X2 result
    rather than trusting the feed flag. Flagging rather than changing unilaterally.
  - Minor, same area: a handful of DC rows carry team-name selections (`"Falkirk or Draw"`,
    n = 1 each) instead of `DC_*` codes.

- 2026-07-28 — **Engine efficiency note (not a correctness issue).** `goals_baseline`
  (`DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel(market_on = false)`) took **2.5 h against
  58 min** for the APM engines. Cause: in `goals_smile_league.jl:102-114` `market_active`
  multiplies the pillars *after* the logpdfs are computed, so with the market off the engine still
  evaluates 909 supremacy logpdfs and a 909×5 smile matrix per gradient — ~5,500 taped ReverseDiff
  nodes contributing exactly 0 — and its goals term uses `logpdf.(Poisson.(λ), goals)` rather than
  sufficient statistics. The posterior is unaffected (multiplying by zero removes the contribution
  exactly; the orphaned `σ_sup`/`σ_smile`/`log_φ` just sample their priors), so it remains a valid
  twin. Gating the *computation* rather than the *contribution* would be a cheap future win.
  For reference the APM engine's gradient benchmarks at **0.558 ms** vs the funnel's 0.709 ms,
  both inside the repo's 0.64 ms target — the slow wall-clock was tape compilation, not the model.

- 2026-07-28 — **src graduation WP-A/WP-B/WP-C: the port is BIT-FAITHFUL to the prototype.**
  Verified on the server against `ScottishLower` (56/57 only, so the numbers below are the
  lower-tier slice of the pooled 54–57 figures in the entries further down).

  **① The ridge reproduces `r08_reliability.jl::fit_ratings` EXACTLY.** Same segment subset
  (`covered`, `y_shots`, λ=1000, `w_SIM`=0, half-life 730d), src vs prototype:

  | check | value |
  |---|---|
  | players fitted | 904 vs 904 |
  | Pearson / Spearman | **1.000000 / 1.000000** |
  | max abs difference | **1.2e-5** |
  | rms difference | 2.0e-6 |
  | sd(rapm) | 0.05085 vs 0.05085 |

  The residual 1e-5 is the only intentional divergence: src anchors the time decay on
  `matches.match_date` (a `Date`) where the prototype used a `DateTime` `start_timestamp`. At a
  730-day half-life that is a sub-part-per-million reweighting.

  **② The segment builder reproduces the prototype exactly** — 8,994 segments over 1,523 matches,
  386 rejects (281 `no_incidents` = the known tier-56 incident holes, 56 `sub_in_unknown`, 49
  `sub_out_off`), 50.5% live-text covered, 5.91 segments/match. Identical on both code paths.

  **③ Target sparsity reproduces WP4** (56/57 only vs the pooled figures in brackets):
  `y_goals` 70.9% zero [72.1], `y_shots` 36.8% [34.3], `y_sot` 51.6% [52.2], `y_xg` 27.2% [25.7];
  garbage-time share 23.7% [23.5].

  **④ Specification check on 56/57 alone is CLEANER than the pooled fit was.** Home advantage
  **+0.804** shots/90 (correctly positive; pooled 54–57 gave +1.51, and a lower league having a
  smaller shot advantage is what you would expect). Red cards **−9.12 / −18.3 / 0.0** — correctly
  negative AND **monotone in severity**, which the pooled fit was not; the exact 0 on the third
  dismissal is the same "no team took a third red with the opponent at eleven" as before. League
  offsets **56 +0.035 > 57 −0.035**, i.e. correctly ordered by tier — the pooled four-tier fit was
  *not* ordered and WP5 warned against reading it. With two adjacent tiers and 33% cross-tier
  players it now behaves.

  - **⚠ THE BRIEF'S "HISTORY-ONLY FIT" RULE WAS WRONG, and following it would have manufactured a
    false negative.** The kickoff brief required the ridge to be fit on
    `SplitBoundary.history_match_ids` only, on the assumption that `target_match_ids` is the
    prediction set. It is not: `create_features` builds `ordered_ids = [history; target]` and the
    engine's likelihood runs over **all** of them, while the out-of-sample matches are fetched
    separately at `dynamics_col == time_step + 1` (`Data.get_next_matches`) and never enter the
    fold. Verified directly: 0 of the 33 OOS matches for fold 6 appear in that fold's id set.
    So fitting on history ∪ target is leak-free *by construction* — it is exactly the information
    set the Turing model is trained on — whereas history-only freezes the rating at the start of
    the target season, leaving it up to nine months stale on the last fold. Measured on fold 6:
    `:training` rates **663** players vs `:history` **496**, and the two rating vectors correlate
    only **0.807**. `fit_on` is a config field (`:training` default) so this is measurable rather
    than asserted.
  - **⚠ SECOND TRAP, caught before it silently nulled the experiment.** `extract_parameters` reads
    `:player_ratings_map` for the OOS matches, which are by construction *not* in `ordered_ids`.
    Building that map over the fold only handed every prediction a zero pillar, collapsing the APM
    engine onto its no-APM twin. The map is now built over the whole store, as
    `player_extractors.jl` already does for the SofaScore rating. Only the rating VECTOR is
    leak-controlled; applying it to a future teamsheet is the pre-match rating under test.
  - Two deliberate deviations from the prototype, both documented in the src files: the zonal
    shot-xG cell table is fitted **globally** (it is how the validated `y_xg` was built, it carries
    no team or player identity, and refitting per fold would break reproduction of the WP7
    numbers); and the positional aggregation weights the **starting XI at 1.0** rather than by
    `minutes_played`, which is identically 0 before 23/24 and NULL for much of 25/26 on these
    tiers. The latter matches `l04_ridge_apm.jl::match_strength`, the research's own covariate.
  - Cost after the fold-independent segment/target cache: **0.16 s per fold** to refit the ridge
    and re-emit the eight vectors.

- 2026-07-23 — **WP7 COMPLETE (r08_reliability.jl). THE DECISION RULE PASSES — but on a low bar,
  and with the team confound unresolved.**

  **① RELIABILITY (decision-rule half #1): PASS.** Split-half over alternating matchdays,
  same 631 players, same halves, goalkeepers excluded:

  | rating | our split-half | SofaScore's | verdict |
  |---|---|---|---|
  | `y_xg` w0 | 0.407 | 0.660 | fail |
  | `y_shots` w0 | **0.669** | 0.660 | **pass** |
  | `y_shots` w0.9 | **0.889** | 0.660 | pass — but see caveat |

  `y_shots` at `w_SIM = 0` is the honest pass: more reliable than the SofaScore rating while
  carrying only 0.389 club-R². **Discount the 0.889** — teammate shrinkage makes ratings stable
  partly by making them team-like, and club identity is near-perfectly stable across halves.

  **② VALIDITY (decision-rule half #2): PASS — RAPM BEATS SofaScore on both held-out seasons.**
  Ordered logit on starting-XI strength, ratings strictly out-of-sample:

  | season | n | no-info floor | **RAPM** | SofaScore | de-vigged close |
  |---|---|---|---|---|---|
  | 24/25 | 512 | 0.6389 | **0.6234** | 0.6387 | 0.5951 |
  | 25/26 | 528 | 0.6528 | **0.6449** | 0.6523 | 0.6127 |

  Note what this really says: **the SofaScore-fed model barely beats the floor at all**
  (0.6387 vs 0.6389; 0.6523 vs 0.6528). So "we beat SofaScore" is a **low bar** — a bottom-up
  rating is nearly useless as a team-strength covariate here. Our margin over the floor is real
  but small (2.4% / 1.2%), and the market remains far ahead (0.595 / 0.613). This does reproduce
  Hvattum & Gelade (2021)'s headline that top-down beats bottom-up on both axes.

  **③ SEASON-TO-SEASON — we roughly MATCH the base paper.** Disjoint single-season fits:
  `y_goals` lag-1 across five pairs **0.372 / 0.292 / 0.404 / 0.228 / 0.156** (mean **0.29**);
  lag-2 mean **0.24**; `y_xg` lag-1 **0.438 / 0.236**. **The paper reports 0.35 (t→t+1) and 0.30
  (t→t+2).** So on the paper's *own* headline reliability metric we are in the same range —
  which is the fairest like-for-like comparison available, and materially more reassuring than
  the SofaScore-correlation framing. SofaScore's own season-to-season stability is much higher
  (≈0.61), as expected for a per-player average.

  **④ THE FORWARD ANOMALY IS EXPLAINED — it is not our bug.** Split-half by position
  (`y_shots` w0): ours **D 0.714, M 0.664, F 0.575**; SofaScore's **D 0.606, M 0.731, F 0.588**.
  **Forwards are the noisiest position for BOTH systems.** Correcting r06's raw correlations for
  attenuation (`ρ_true = ρ_obs / √(rel₁·rel₂)`):

  | position | raw ρ | disattenuated |
  |---|---|---|
  | D | 0.486 | **0.74** |
  | M | 0.374 | **0.54** |
  | F | 0.220 | **0.38** |
  | overall | 0.382 | **0.58** |

  Forwards still agree least, but far less dramatically — much of the r06 gap was two noisy
  instruments, not disagreement. **Caveat on the overall 0.58:** Gelade & Hvattum's 0.47–0.62
  band is presumably *raw*, so comparing our disattenuated 0.58 to their raw band flatters us —
  theirs would rise too if corrected. The honest reading is that we remain below them, but the
  shortfall is substantially measurement noise from a much smaller league, not a broken model.

  **⑤ CROSS-TIER TRANSFER: positive but modest.** 169 players with ≥540 min in both groups;
  RAPM built from **56/57 minutes only** vs SofaScore built from **54/55 minutes only** —
  Pearson **0.177**, Spearman **0.226**. Nothing mechanically links these two quantities (no
  shared match, team or season), so a positive value is genuine external validity for the
  lower-league ratings. It is weak, and it is the single number most worth strengthening.

  - **VERDICT: green-light, with the scope narrowed.** The pre-registered rule — split-half ≥
    SofaScore's AND retrodiction not materially worse — is met by **`y_shots`, `w_SIM = 0`**
    (reliability 0.669 vs 0.660; validity better than SofaScore on both seasons). That is the
    cell to carry forward, **not** WP5's Brier-optimal `w_SIM = 0.9`, whose apparent superiority
    on both axes is substantially club identity.
  - **Three honest limits on that verdict.** (a) Beating SofaScore here is a low bar, since the
    SofaScore-fed model scarcely beats a no-information floor. (b) The margin over the floor is
    ~2%, and the closing line is far beyond both. (c) Cross-tier transfer at ρ≈0.18–0.23 is thin
    evidence for the leagues we actually care about.
  - **NEXT: WP6 (Bayesian arm) is now justified** — the ridge reference has cleared the bar it
    had to clear, so a hierarchical / prior-informed version has something to beat, and would
    supply the posterior Kelly staking wants. Prioritise (i) `w_SIM` as a *hierarchical* prior
    rather than a fixed dial, so shrinkage strength is learned rather than tuned into the team
    confound, and (ii) strengthening cross-tier identification.

- 2026-07-23 — **The MARKET-based "change in probability" target: scoped away, and now shown to
  be unbuildable as a validated arm. Recording the decision properly because it was originally
  requested and I silently substituted for it.**

  The opening brief asked for a target based on *"the change in the odds / change in the match
  probability depending on goal"*. When the four targets were locked I mapped that onto the base
  paper's **xPPM**, which uses a *model-derived* in-play W/D/L probability, and did not flag the
  substitution. `y_xp` as built is therefore the paper's expected-points target driven by the
  local Poisson hazard model in `l03_targets.jl` — **not** market prices.

  **Why the market version cannot be validated in this stream (decisive):**

  | tier | matches | betfair odds_history | MATCH_ODDS | SofaScore rating |
  |---|---|---|---|---|
  | 54 Prem | 1188 | **0** | **0** | ✔ all seasons |
  | 55 Champ | 1030 | **0** | **0** | ✔ 23/24+ |
  | 56 L1 | 985 | 891 | 886 | ✘ none |
  | 57 L2 | 985 | 750 | 739 | ✘ none |

  **Betfair covers 56/57 only; SofaScore ratings cover 54/55 only. The overlap is empty.** The
  whole validation design rests on building a rating from lower-league-available features and
  checking it where a rating yardstick exists — a market-priced target breaks that, because it
  can only be computed exactly where no yardstick exists.

  **And it would make the stream's main weakness worse.** The base paper (§4.2) is explicit that
  the in-play model must be **team-strength-blind**, since conditioning on team strength
  double-counts the thing plus-minus is estimating. Market odds are the opposite of blind — they
  price Celtic vs Ross County very differently by construction. So a market-based xPPM would
  *bake in* team strength, and WP7 has already shown team-loading is this rating's central
  problem (21–39% of variance at `w_SIM = 0`, rising to 76% when shrunk).

  **Supporting evidence that it is not worth forcing:** the model-based `y_xp` was the **weakest
  of the five targets** in WP5 (Δ Brier −0.00319 at `w_SIM = 0`, last place) and is **0.881
  correlated with `y_goals`** — largely a game-state reweighting of goals rather than new
  information. A market variant differs from it mainly by adding team strength.

  ⇒ **Recorded as considered-and-rejected, not as an oversight.** If it is ever wanted, the only
  coherent framing is: build it on 56/57 where betfair exists, validate it on match-outcome
  Brier rather than against a player rating, and expect the team-strength confound to be worse
  not better. Note also `../inplay_scottish/` measured in-play liquidity as thin (median ≈49
  MATCH_ODDS prints per match on t=56, far fewer on t=57), so segment-boundary prices would have
  real gaps.

- 2026-07-23 — **WP7b (r07_best_players.jl): the top-N test is the STRONGEST result so far —
  the two systems disagree in the middle but agree hard at the top.** Per-season ratings
  (2-year trailing window, 365-day half-life anchored at season end), ≥900 minutes, outfield
  only, ~325 players per season.

  `y_xg`, `w_SIM = 0` (the least team-loaded cell):

  | season | our top-20 sits at SofaScore pct | SofaScore top-20 sits at our pct | overlap /20 | chance | within-club pct | within-club overlap |
  |---|---|---|---|---|---|---|
  | 23/24 | **88.2** | 77.8 | **9** | 1.2 | 59.8 | 3 |
  | 24/25 | **84.5** | 81.7 | **10** | 1.2 | 60.6 | 3 |
  | 25/26 | 70.0 | 68.3 | 6 | 1.3 | 62.3 | 4 |

  1. **Our top-20 land at the 84th–88th percentile of SofaScore** in the two complete seasons,
     and the two top-20 lists share **9–10 of 20 players against a chance expectation of 1.2** —
     a **7–8× enrichment**. The relationship is symmetric: SofaScore's top-20 sit at the 78th–82nd
     percentile of ours.
  2. **This resolves the ambiguity r06 left open.** An overall Spearman of 0.29–0.45 is
     compatible with either loose agreement everywhere or disagreement in the middle plus strong
     agreement at the extremes. It is decisively the latter — which is the version that actually
     matters, because a top-N list is what a rating is *for*.
  3. **It partially survives the team control.** Ranking players *within their own club*, our
     top-20 still sit at the **~60th percentile** of within-club SofaScore (chance = 50) with
     **3–4 of 20 overlap against 1.2 expected (≈3×)**. Weaker than the raw figure, as expected,
     but real — we are not purely picking the best clubs.
  4. **Face validity is strong.** 23/24 top-15 is Carter-Vickers, O'Riley, Taylor, Goldson,
     Kyogo, Maeda, Tavernier, Johnston, Scales, Lundstram, McGregor, Cantwell — the Celtic and
     Rangers spine in the season Celtic won the league — plus Moult and Sibbald at Dundee United.
  5. **The forward anomaly from r06 shows up in the names.** The two lowest SofaScore percentiles
     in that top-15 are **Kyogo (57th)** and **Maeda (61st)** — both forwards. Every defender and
     midfielder in the list is at the 81st–100th. Consistent with r06's finding that forwards are
     our weakest positional agreement, and it now has a concrete face.
  6. **25/26 is materially weaker across the board** (70.0 vs 88.2/84.5, overlap 6 vs 9–10). It
     is an in-progress season with fewer matches — treat it as underpowered, not as a decline.
  7. The `y_shots`/`w_SIM=0.9` cell is comparable at the top (86.1 / 87.1 / 67.5) and slightly
     better within-club (overlap 6/4/5), but it carries the 0.755 club-R² loading, so `y_xg` at
     `w_SIM=0` remains the honest choice.

  - **CORRECTION to the r06 entry below — `team_id` is NOT a club identifier.**
    `match_player_lineups.team_id` has **626 distinct values across tiers 54–57 for ~44 actual
    clubs**, and a single id (2351, mostly Rangers) carries dozens of different club names.
    Grouping ~880 players by their modal `team_id` produced **451 groups** — average group size
    under two — which inflates any "share of variance explained by team" statistic through
    degrees of freedom alone. Fixed with a name-derived label (`pm_club_map`, 44 clubs).
    **Corrected figures:**

    | cell | raw ρ | club R² (was) | club R² (correct) | within-club ρ (was) | (correct) |
    |---|---|---|---|---|---|
    | `y_xg` w0 | 0.343 | 0.383 | **0.212** | ~0.172 | **0.240** |
    | `y_shots` w0 | 0.382 | 0.540 | **0.389** | ~0.20 | **0.259** |
    | `y_xg` w0.75 | 0.466 | 0.576 | 0.475 | 0.203 | 0.274 |
    | `y_shots` w0.9 | 0.513 | 0.774 | 0.755 | 0.214 | 0.309 |

    The qualitative conclusion **stands** — raising `w_SIM` inflates raw correlation mostly via
    team — but **the rating is materially less team-loaded than first reported** (21% not 38% at
    the cleanest setting) and player-level agreement is **higher** (0.24–0.31, not 0.17–0.22).
  - Also settled by diagnostics this session: **sample size is NOT the binding constraint**
    (goals on 22,785 segments vs 11,886 gives ρ 0.351 vs 0.348 — doubling the data changes
    nothing), and `dur_pow` — a knob l01 flagged as "settle empirically" and WP5 never swept —
    trades correlation against team-loading like every other knob rather than unlocking signal.

- 2026-07-23 — **WP7 first pass (r06_vs_sofascore.jl): agreement with the SofaScore rating is
  BELOW the expected band, and what agreement exists is largely TEAM strength, not player
  contribution.** 877 players ≥540 minutes in tiers 54/55, 23/24–25/26.

  | target | w_SIM | Pearson | Spearman | **team R²** | sd(rapm) | within-team sd |
  |---|---|---|---|---|---|---|
  | `y_shots` | 0.0 | 0.347 | 0.274 | 0.540 | 0.099 | 0.067 |
  | `y_shots` | 0.9 | 0.476 | 0.388 | **0.774** | 0.349 | 0.166 |
  | `y_xg` | 0.0 | 0.320 | 0.271 | **0.383** | 0.035 | 0.027 |
  | `y_xg` | 0.75 | 0.435 | 0.365 | 0.576 | 0.049 | 0.032 |
  | `y_goals` | 0.0 | 0.299 | 0.265 | 0.435 | 0.017 | 0.013 |
  | `y_goals` | 0.9 | 0.485 | 0.429 | 0.709 | 0.058 | 0.031 |

  1. **Correlation 0.30–0.49 against an expected band of 0.47–0.62** (Gelade & Hvattum 2020).
     Only the heavily team-shrunk cells reach the bottom edge. This is *not* the "healthy
     middling correlation" the research predicted — it is below it.
  2. **THE KEY RESULT: agreement and team R² rise in LOCKSTEP with `w_SIM`.** `y_goals` goes
     ρ 0.299 → 0.485 while team R² goes 0.435 → 0.709; `y_shots` 0.347 → 0.476 while team R²
     goes 0.540 → **0.774**. So the apparent improvement in SofaScore agreement is **bought by
     making the rating more team-like**, not by measuring players better. The WP5 tension is now
     quantified, and it is severe.
  3. **Even at `w_SIM = 0`, 38–54% of rating variance is team identity.** Before any teammate
     shrinkage the rating is already substantially a team-strength estimator. `y_xg` at
     `w_SIM = 0` is the cleanest cell on this criterion (team R² 0.383) — and it is *not* the
     cell WP5's Brier criterion selected, exactly as predicted.
  4. **Position ordering is WRONG, in two specific ways** (`y_shots`, `w_SIM = 0`):
     ours **D 0.486 > M 0.374 > F 0.220 > G −0.001**; expected **F 0.62 > D 0.60 ≈ M 0.59 >
     GK 0.47**.
     - **Goalkeepers ≈ 0.** The base paper's §6 caveat, now measured: a keeper plays nearly every
       minute, so his plus-minus is a comparison against his backup and is barely identified. Our
       GK ratings are worthless and should be excluded, not reported.
     - **Forwards are our WEAKEST outfield agreement (0.22) but should be the STRONGEST.**
       Forwards are where an event-based rating is most informative (goals, shots), so this is a
       real mismatch, not a scale artefact. It needs explaining before the ratings are trusted.
     - Defenders being our *best* agreement is suspicious in light of (3): SofaScore defender
       ratings are heavily driven by team defensive performance, which is a team property — so
       that agreement may itself be team-driven.
  5. **The relationship is flat across the bottom 70%.** Mean RAPM by SofaScore decile:
     −0.015, −0.031, −0.017, −0.027, −0.021, −0.013, −0.004, +0.019, +0.027, **+0.106**. It only
     turns upward in deciles 8–10, and **within-decile sd (0.068–0.137) exceeds the entire
     between-decile range (≈0.12)**. The two systems agree mainly that the very best SofaScore
     players are good; below that they largely disagree player-by-player.
  6. Disagreements are structured, not random: the players RAPM likes far more than SofaScore are
     mostly defenders/midfielders on strong sides; the ones SofaScore likes far more are
     attacking midfielders who accumulate events (Josh McPake z_sofa +5.2 vs z_rapm +1.7; Dom
     Thomas, Elliot Watt, Marc Leonard). That is the classic top-down/bottom-up split — but it
     needs the team confound removed before it can be read as evidence of anything.

  - **VERDICT: not currently a viable drop-in for the SofaScore rating.** The decision rule
    (split-half reliability ≥ SofaScore's AND team-strength retrodiction not materially worse)
    is not yet tested, but agreement this team-loaded means the remaining WP7 work must first
    **partial out team strength** — correlate the *within-team* residuals of both ratings — or
    every downstream number will be measuring how good the club is.
  - **NEXT (WP7 proper):** (a) within-team residual correlation, both systems; (b) split-half
    reliability at several `w_SIM`, ours vs SofaScore's on the same players; (c) exclude
    goalkeepers; (d) explain the forward anomaly; (e) only then the team-strength retrodiction.

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
