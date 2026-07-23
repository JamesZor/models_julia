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
| `l01_segments.jl` | Segment builder → sparse `X`, weights, dismissal + league dummies | pending |
| `r01_segment_qa.jl` | WP2 segment QA + identifiability gate | pending |
| `l02_shot_parser.jl` | BBC commentary → shot descriptors | pending |
| `r02_shot_xg.jl` | WP3 shot xG model + calibration vs SofaScore player xG | pending |
| `l03_targets.jl` | The four segment targets | pending |
| `r03_targets_qa.jl` | WP4 target sparsity comparison | pending |
| `l04_ridge_apm.jl` | Sparse ridge RAPM + CV over (λ, ζ) | pending |
| `r04_ridge_fit.jl` | WP5 fits | pending |
| `l05_bayes_apm.jl` | Turing hierarchical / prior-informed RAPM | pending |
| `r05_bayes_fit.jl` | WP6 Bayesian vs ridge | pending |
| `r06_vs_sofascore.jl` | **WP7 decisive gate** | pending |

## Decision rule (WP7)

Green-light integration only if, for ≥1 target × estimator cell: split-half reliability is **≥
the SofaScore rating's**, **and** the team-strength retrodiction is **not materially worse** than
the SofaScore-fed model on held-out Brier. A clean negative is a valid outcome.

## Findings log
<!-- YYYY-MM-DD — WP / gate — result. Append newest-first. -->

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
