# Position-aware player ratings

Research stream: should a player carry a **different rating per position** (G/D/M/F) rather than one
position-agnostic rating? Out-of-position deployment would then change team strength. EDA-first; build a model
only if the data clears the gates below. See `PROMPT.md` for the fresh-session kickoff brief.

## Why
The current rating pipeline (`src/features/extractors/player_extractors.jl`) Kalman-tracks **one** rating per
player, minute-weights it, and **sums** by the position played each match into 8 vectors
`flat_{home,away}_{G,D,M,F}_rating`. So a striker fielded at midfield still contributes his (single, high)
striker rating to the midfield bucket — the model cannot express that he is *worse out of position*. The repo's
existing "full-position" engine (`PositionalPlayerDynamics` / `DynamicDixonColesXGFullPositionPlayerTimeDecayModel`,
A/B-tested in `current_development/ab_test_fullposition/`) only adds position-specific **weights** on that one
rating — it is **not** a rating-per-position. A prior attempt at this failed because ratings were *combined*,
not tracked per position.

## What (two constructions — EDA both, let the data choose)
- **A. Per-(player×position) rating** — track from only that player's appearances *in that position*; fallback to
  the player's overall rating when an off-position is too sparse.
- **B. Base + position offset** — current per-player rating + a learned `δ_position` adjustment when deployed off
  the player's natural/modal position. Robust to sparsity.

## Data reality (must respect)
- Position + rating exist **2023+ only**.
- Raw position strings: `G/Goalkeeper/GK`, `D/Defender/DF`, `M/Midfielder/MF`, `F/Forward/FW/A`. **Missing is
  silently defaulted to "M"** by the current `clean_pos` → separate *true*-M from *defaulted*-M, or every
  downstream stat is poisoned.
- Within-player rating comparisons are confounded by **game-state / opponent / minutes** — control for them
  (the momentum study was wrecked by exactly this confound).

## Files
| File | Role |
|---|---|
| `PROMPT.md` | Fresh-session kickoff brief (verbatim) |
| `NOTES.md` | This — goals, gates, findings log |
| `l00_position_helpers.jl` | clean_pos taxonomy, modal-position, off-position flag, construction A & B builders |
| `r00_position_eda.jl` | EDA across ALL leagues: coverage, multi-positionality, Δ, A-vs-B |
| `l01_position_ratings_feature.jl` | *(conditional)* new extractor → position-conditioned `flat_*` vectors |
| `r01_mvp_double_poisson.jl` | *(conditional)* baseline vs position-aware, per-line eval |

## Decision gates (Phase 1 EDA, all betdb leagues, 2023+ starters)
1. **Coverage** — % player-matches with a real (non-defaulted) position + rating; position mix; defaulted-M share.
2. **Multi-positionality (make-or-break)** — share of starter player-matches off the player's modal position;
   per-player distinct positions / entropy. **~0 everywhere ⇒ stop, the idea is moot.**
3. **Out-of-position Δ** — within-player `rating ~ player_FE + is_off_modal + controls`; need a measurable,
   sign-consistent Δ (|t|≫2).
4. **A vs B** — which predicts the held-out next-match rating better; how often each materially moves a team's
   position-bucket rating.

→ Build the MVP **only if gates 2 AND 3 pass on ≥1 league.** Judge the MVP on **per-line LogLoss + GLMEdge**
(mirror `split_market_pillar/r12`), never grouped backtest P/L. A clean negative result is a valid outcome.

## Findings log
<!-- YYYY-MM-DD — phase / gate — result. Append newest-first. -->
- 2026-06-27 — Phase 1 EDA **built, not yet run**. `l00_position_helpers.jl` + `r00_position_eda.jl`
  cover all 4 gates over every betdb segment (2023+ starters). Key design choices:
  - `canonical_pos` returns `missing` for unmappable strings → *true*-M kept separate from
    *defaulted*-M (Gate 1 reports `pct_defaultM`); all gate stats use real positions only.
  - Gate 2 off-modal share is appearance-weighted over players with ≥5 real apps.
  - Gate 3 uses **within-player FE via group-demeaning** + controls (is_home, minutes, opponent
    starter-rating sum as a strength proxy) to dodge the game-state/opponent confound.
  - Gate 4 = chronological 70/30 holdout, RMSE of pre-match estimate vs realised rating;
    Construction A = per-(player×position) Kalman (fallback to overall when <4 prior apps),
    B = overall + δ(played-pos) estimated on TRAIN off-modal residuals. Reused the live
    `BayesianTracker(6.5,1.0,0.5,0.01)` and `calculate_player_ratings`.
  - Both files parse clean locally; **DB run pending on server** (git push → pull → restart REPL →
    `include("current_development/position_aware_ratings/r00_position_eda.jl")`).
