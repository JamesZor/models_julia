# Kickoff Prompt — Position-aware player ratings

> Paste the block below into a **fresh Claude session** opened in the `BayesianFootball` repo.
> It is self-contained. Work happens in `current_development/position_aware_ratings/`.

---

**Goal:** Investigate whether tracking football player ratings **per position** (a player rated differently at
G/D/M/F) improves team-strength estimation over the current single-per-player rating, and build a basic MVP
double-Poisson model **only if the data justifies it**. Work in `current_development/position_aware_ratings/`,
follow the repo's `lXX`-loader / `rXX`-runner convention, and keep a timestamped findings log in `NOTES.md`.

**Background (already established — do not re-derive):**
- Current ratings: `src/features/extractors/player_extractors.jl` Kalman-tracks **one position-agnostic rating
  per player** (`BayesianTracker(6.5,1.0,0.5,0.01)`), minute-weights it, and **sums** by the position played
  that match into 8 vectors `flat_{home,away}_{G,D,M,F}_rating`. Position + rating data exists **2023+ only**.
- The repo already has position-**weighted** infra (`PositionalPlayerDynamics`,
  `DynamicDixonColesXGFullPositionPlayerTimeDecayModel`, `current_development/ab_test_fullposition/`) — that is
  *one* rating with 4 position weights, **NOT** per-position ratings. **Do not rebuild it.** The new thing is a
  different *rating per position*.
- `ds.lineups` columns: `match_id, player_id, team_side, position, rating, minutes_played, is_substitute`.
  Raw position strings: `G/Goalkeeper/GK`, `D/Defender/DF`, `M/Midfielder/MF`, `F/Forward/FW/A`; **missing is
  silently defaulted to "M"** by the current `clean_pos` — separate true-M from defaulted-M or the analysis is
  poisoned.

**Two constructions to compare (EDA both, let the data choose):**
- **A. Per-(player×position) rating** — track skill from only that player's appearances *in that position*, with
  a fallback to the player's overall rating when an off-position has too few games (sparse).
- **B. Base + position offset** — the current per-player rating + a learned `δ_position` adjustment when deployed
  off the player's natural/modal position (robust to sparsity).

**Phase 1 — EDA across ALL betdb leagues** (`r00_position_eda.jl`, helpers in `l00_position_helpers.jl`).
Loop every segment in `src/Data/fetchers/segments.jl` via `Data.load_datastore_cached(seg)`, filter to **2023+
starters**. Report per league:
1. **Coverage** — % player-matches with a *real* (non-defaulted) position + rating; position mix; defaulted-M
   share. → which leagues are usable.
2. **Multi-positionality (make-or-break)** — per player with ≥k appearances: # distinct positions, position
   entropy, and the **share of starter player-matches off the player's modal position**. If ~0 everywhere →
   **STOP and report**; per-position ratings ≡ the single rating and the idea is moot.
3. **Out-of-position rating Δ** — within-player regression `rating ~ player_FE + is_off_modal_pos + controls`
   (controls: opponent strength, home/away, minutes; **beware the game-state confound** — see the momentum
   findings). Is there a measurable, sign-consistent Δ (|t|≫2)? Magnitude in rating points.
4. **A vs B** — for players with enough off-position games: which construction better predicts the *held-out
   next* match rating; how often each *materially differs* (>ε) from the current single rating, and how often it
   would move a team's position-bucket rating. → picks the construction for the MVP.

**Decision gate:** only proceed to a model if gate 2 (real multi-positionality) **AND** gate 3 (real Δ) pass on
at least one league. Otherwise write up the negative result in `NOTES.md` and stop — **that is a valid outcome.**

**Phase 2 — MVP (only if gated in)** (`l01_position_ratings_feature.jl` + `r01_mvp_double_poisson.jl`).
The change is at the **feature layer, not a new engine.** Write a new extractor mirroring
`player_extractors.jl` that builds the same 8 `flat_{side}_{pos}_rating` vectors from the **winning construction**
(A or B) instead of the position-agnostic Kalman value. Then run the existing baseline engine
`DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel` with `PositionalPlayerDynamics` (4 weights, so position
enters the likelihood) **twice** — current single-rating feature vs the position-aware feature — on the league(s)
the EDA flagged. Compare with **per-line LogLoss + GLMEdge** via `Evaluation.evaluate_experiments` (mirror
`current_development/split_market_pillar/r12_per_line_bias_edge.jl`), **NOT** raw backtest P/L (the r05
grouped-hurdle artifact lesson). The position-aware feature earns its complexity only if it improves a proper
score or opens a GLMEdge the single rating didn't.

**Conventions & ops:**
- AD-safe Turing rules from `CLAUDE.md` (no `if`/`for` in `@model`, all feature vectors `Float64`/`Int`, no
  `missing`, sanitize NaN/Inf with `ifelse`/`clamp`).
- Reuse, don't rebuild: `src/features/trackers/bayesian.jl` (`calculate_player_ratings`),
  `src/models/.../player_level/positional.jl`, the standard
  `create_experiment_task → run_experiment → save_experiment → Diagnostics` pipeline (see
  `current_development/split_market_pillar/r11_run_split_compoisson.jl`).
- Run on the server REPL (kaimon): `git push` → `git pull` on the server → **restart the REPL** before running.
- Keep a timestamped findings log in `NOTES.md`.
