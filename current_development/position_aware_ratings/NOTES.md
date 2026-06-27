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
| `r00_position_eda.jl` | Phase-1 EDA across ALL leagues: coverage, multi-positionality, Δ, A-vs-B (verdict: STOP) |
| `RESEARCH_role_aware_ideas.md` | deep-research synthesis → the pivot (role-neutral target + market yardstick) |
| `l01_roleneutral_helpers.jl` | **Option-1 EDA**: discover role-neutral targets, role-standardise (z within position), within-player FE |
| `r01_roleneutral_eda.jl` | **Option-1 EDA** runner: per (league × target) off-modal output penalty across all leagues |
| `l01_position_ratings_feature.jl` | *(superseded)* original rating-based extractor — not built (Gate-3 null) |
| `r01_mvp_double_poisson.jl` | *(superseded)* original rating MVP — not built |

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
- 2026-06-27 — **Option-1 EDA RESULT: signal FOUND (positive).** `r01_roleneutral_eda.jl` re-ran the
  out-of-position test on every role-neutral output, role-standardised within position (z), within-player
  FE. Unlike the rating (Gate-3 null), the result splits cleanly:
  - **Attacking THREAT drops off-position** (the hidden penalty): `onTargetScoringAttempt` negative in
    **all 4 leagues that record it** (mean_t −3.68); `expected_goals` strongly negative (Veik t=−5.27,
    First-Div −2.82); `goals` negative (Veik −4.18, Norway −2.38). Magnitude ≈ 0.08–0.17 role-sd.
  - **Volume/involvement RISES off-position** (t=+8…+14 across all 5 leagues): total/accurate passes,
    touches, long balls, clearances. Partly compositional (versatile high-touch M dropped into low-touch
    roles carry their volume), partly playing safe. Dueling/contesting drifts negative (more passive).
  - **Why Gate 3 was null, now mechanistic:** the composite rating nets the lost attacking output against
    the *gained* safe involvement → ≈0. The threat dimension underneath genuinely degrades. ⇒ the pivot
    was correct; the out-of-position signal lives in **attacking output (shots-on-target / xG / goals)**,
    invisible to the rating.
  - Coverage: shot/xG columns ~30–57% per league (n 10k–20k); present in Korea/Norway/Veik/First-Div, not
    Ireland(79)/ScottishLower. ScottishLower still dead (off_modal all-false → NaN t).
  - **NEXT (gate before building):** the per-player effect is real but small/sporadic; the deep-research
    caveat says lineup edges rarely survive the close. So before any Turing build, run a **team-level
    market-relevance EDA** (`r02`): does a team's off-position attacking-deployment index predict match
    total goals / xG **beyond the de-vigged market** (totals lines, GLMEdge-style)? Pass → build the
    Option-1 feature; null → move to Option 2 (prior-informed RAPM).
- 2026-06-27 — **PIVOT after deep-research** (batch `batch-20260627-130505-1a266db9`; full synthesis in
  `RESEARCH_role_aware_ideas.md`). The Gate-3 null is now *explained*: SofaScore/WhoScored/FotMob ratings
  are computed **conditional on the position played** (role-specific event weights/baselines), so an
  out-of-position player accruing normal counting stats gets a normal rating — the penalty is invisible
  **by construction** (verified vs two first-party methodology pages). Two consequences:
  1. To detect a role effect, switch the target from the (role-conditioned) rating to a **role-neutral
     output** we already have per-player (xG, xA, `bigChanceCreated`, `touchesInOppBox`, shots…), and
     judge everything **against the market** (per-line LogLoss + GLMEdge), not against the rating.
  2. New plan: **Option 1 (this EDA)** redo the out-of-position test on role-neutral, role-standardized
     output → `l01_roleneutral_helpers.jl` + `r01_roleneutral_eda.jl`. **Option 2 (later)** prior-informed
     Bayesian RAPM (team xG-diff target, player rating/xG as priors) as a new team-strength signal.
  - Sobering caveat carried into the plan: lineup/position edges **mostly don't survive the closing line**
    (public lineup news is priced by KO); realistic upside is **totals/derivative markets**, not 1X2.
    Gate any build on a GLMEdge the single rating doesn't already open.
- 2026-06-27 — **Phase 1 VERDICT: STOP. Do not build the MVP.** Gate 2 passes, Gate 3 fails on
  every data-rich league. The SofaScore match rating already prices the role played that match, so
  a per-position rating carries no extra information over the single rating.
  - **Gate 1 (coverage): PASS everywhere.** Ireland/First-Div/Korea/Norway/Finland all ~100% real
    position, 99–100% rated, 0% defaulted-M, date range 2023→2026. (ScottishLower is a dead league:
    0% rated, static one-position-per-player — excluded.)
  - **Gate 2 (multi-positionality): PASS, strongly.** Off-modal share of starter appearances:
    Ireland 13.2%, First-Div 10.6%, Korea 14.7%, Norway 11.5%, Finland 12.2%; 48–62% of players
    multi-positional; mean distinct pos 1.53–1.71. Players genuinely move position — the idea is NOT
    moot on data grounds.
  - **Gate 3 (out-of-position Δ): FAIL, decisively.** Within-player FE (controls is_home, minutes,
    opp_strength all sane, |t|≫20) → off-modal coef ≈ 0 and insignificant, not sign-consistent:
    Ireland +0.019 (t=1.40), First-Div −0.012 (t=−0.73), Korea −0.003 (t=−0.31), Norway −0.012
    (t=−1.14), Finland −0.002 (t=−0.11). With Korea se≈0.008 on n≈31k, even a 0.02-pt effect is ruled
    out. Playing off your modal position does **not** measurably change your match rating.
  - **Decision rule:** build only if Gate 2 AND Gate 3 pass on ≥1 league. Gate 3 fails on all 5 →
    **no Phase 2.** Clean negative result (a valid outcome per the brief).
  - Gate 4 hit an ordering bug first run (`estimate_delta_table` read `pre_overall` before it was
    built); fixed in l00. Re-run will add A-vs-B corroboration, but it cannot overturn a null Gate 3
    (if there's no Δ, A and B ≈ overall by construction).
  - **Implication for the live pipeline:** the current single-rating + position *weights*
    (`PositionalPlayerDynamics`) is the right design; per-position rating tracking is not worth the
    complexity. Don't revisit unless a finer position taxonomy (not G/D/M/F) or a non-rating target
    (xG/xA per role) is proposed.
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
