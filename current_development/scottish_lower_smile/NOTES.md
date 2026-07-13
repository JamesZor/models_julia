# Scottish Lower — team-level goals-only time-decay sup+smile

Research stream for **team-level, goals-only** versions of the sup+smile market-pillar engines,
targeting `Data.ScottishLower()` (tournaments **56 = League One, 57 = League Two**). These leagues
have **no xG, no stats, no player ratings, no Betfair** — only goals + full Bet365 (SofaScore) odds
(1X2, BTTS, O/U ladder 0.5–7.5, DC/DNB/AH on ~99% of matches; 5 full seasons 21/22→25/26, 20/21
~50% odds coverage). Season kicks off ~early Aug 2026.

Parent stream: `current_development/split_market_pillar/` (player-level, xG; keeper `li_smile50`
graduated to src as `DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel`). The smile machinery
(`MarketSmileFeature`, `_smile_intensity`, `SmileScoreMatrix`, per-line O/U pricing) is
rate-source-agnostic and reused as-is.

## Canonical naming (inherited from split_market_pillar/NOTES.md — single source of truth)

A model is two independent axes; reference name = `<pillar>-<disp>`; cell name =
`<pillar>_<disp>_<knob><val>`.

**Axis A — market pillar:** `none` (structural only) · `iso` (isotropic shared σ + `mw`) ·
`smile` (supremacy + per-strike Λ(K), knobs `sup`/`sw`). (`split` is NOT built here — nulled at
the latent-rate layer in the parent stream.)

**Axis B — dispersion:** `pois` (double Poisson) · `nb` (NegBin, reference row via the existing
src team engines).

Grid-cell suffixes: `hl<days>` (days_half_life), `hs<n>` (history_seasons), `sup<pct>`, `sw<pct>`,
`mw<pct>`. Examples: `none_pois_hl120_hs2`, `smile_pois_sup70_sw50`, `iso_nb_mw100`.

## Fixed design decisions (2026-07-11 planning session)

- **Targeted ladder:** 3 new loader engines `TeamDPGoalsModel` (`none_pois`),
  `TeamIsoDPGoalsModel` (`iso_pois`), `TeamSmileDPGoalsModel` (`smile_pois`); existing
  `DynamicGoalsTimeDecayModel`/`DynamicMarketGoalsTimeDecayModel` = `nb` reference row.
- **Pooled + league intercept:** one model over both divisions (shared team map survives
  promotion/relegation) + zero-sum per-league offset δ_league on log-λ (goal levels differ:
  2.87 vs 2.71 goals/match). New loader-local `LeagueFeature`.
- **Benchmark:** de-vigged Bet365 close (`prob_fair_close`). NO Betfair swap (none exists for
  56/57; Betfair historical may be loaded later as a secondary check only — thin liquidity, so per
  [[betfair-vs-bet365-market-anchor]] Bet365 stays the anchor regardless).
- **Grids:** Stage A `hl{60,120,180,365} × hs{1,2,3}` on `none_pois`; Stage B
  `sup{40,70,100} × sw{0,40,50}` on `smile_pois` at the Stage-A winner. Priors / Kmax=4 /
  sampled-σ at Ireland-validated values. CV target seasons 23/24→25/26, `dynamics_col=:match_biweek`.
- **Eval:** per-line LogLoss diff vs close + GLMEdge + per-bet Kelly hurdle_G per market family
  (r06 lesson: judge per line, not grouped). Convergence gate (R-hat<1.01, ≥95% splits) before
  reading ANY grid table.
- σ's are **SAMPLED, never fixed** (parent-stream convergence saga); **no σ hierarchy**
  ([[hierarchical-smile-sigma-null]]); market inversion stays **Poisson-referenced**
  ([[no-pregame-intensity-smile]]).

## Known gotchas carried in

- `Features.MarketLambdaFeature` is a **phantom** (exported, never defined) — every src `*Market*`
  engine's `required_features` throws. Loader overrides the NB market engine's `required_features`
  with `DoublePoissonMarketFeature()`; the src fix lands at graduation.
- New structs are not in any score-computation Union → loader ships `extract_params` +
  `compute_score_matrix` overrides ([[dixoncoles-prediction-dispatch-union]]).
- Market λ plausibility mask `0.02 < λ < 20` on the pillar (degenerate inversions on thin closes,
  [[outfield-xg-engine-gotchas]]).
- Silent split drop: check `training_results.items` length per cell before eval
  ([[xg-pillar-nan-and-sampler-gotchas]]).

## Files

- `r00_data_qa.jl` — Stage 0 data QA (seasons, odds-ladder coverage, league diff, team churn,
  fold counts, inversion sanity, phantom-feature confirmation). ✅ run 2026-07-11 (`r00_out.txt`).
- `l01_team_dp_league.jl` — loader: `LeagueFeature` + the 3 engines + prediction overrides +
  NB `required_features` fix.
- `r01_smoke.jl` — short-window smoke (3 engines, last season, biweek≥16): convergence,
  δ_league read, PPD end-to-end, smile≠grid O/U. ✅ run 2026-07-12, 29/29 (`r01_out.txt`).
- `r01b_smile_depth_probe.jl` — smile runtime fix probe: max_depth {6, 5} vs the depth-10
  reference (215m). Sets `MAX_DEPTH` for r04. **Run before Grid B** (~2–3 h total).
- `r02_grid_decay_history.jl` — Stage A grid: `none_pois_hl{60,120,180,365}_hs{1,2,3}` + nb refs,
  saves to `data/scottish_decay_grid/`, gate → `r02_convergence.txt`. Overnight (~5–8 h).
- `r03_eval_decay.jl` — Stage A per-line eval (LogLoss diff / GLMEdge / RQR vs Bet365 close).
  → record (hl\*, hs\*) here in NOTES.
- `r04_grid_smile.jl` — Stage B grid: `smile_pois_sup{40,70,100}_sw{0,40,50}` + iso control at
  (hl\*, hs\*). **⚠ EDIT `BEST_HL`/`BEST_HS`/`RERUN_CONTROLS` at the top after r03.**
  Saves to `data/scottish_smile_grid/`, gate → `r04_convergence.txt`. Overnight (~4–7 h).
- `r05_eval_smile.jl` — Stage B per-line eval + family routing table + BayesianKelly tearsheet +
  informational Betfair-25/26 CLV. **⚠ EDIT `_TAG` to match r04.**
- `r06_smoke_src.jl` — Stage 4 verification: src-only build/train/price of the graduated engine.
  **⚠ EDIT `SUP_W`/`SMILE_W`/`HL`/`HS` to the r05 winner first.**
- `RESULTS_scottish_grid.md` — results template; fill as runners complete.

### Run order (server, kaimon REPL; fresh REPL restart after every git pull with struct changes)

```julia
# after: git -C /root/BayesianFootball pull
include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r01_smoke.jl"))   # gate: all ✅
include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r02_grid_decay_history.jl"))  # overnight
include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r03_eval_decay.jl"))
# -> record (hl*, hs*) in NOTES, edit r04 header, push/pull
include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r04_grid_smile.jl"))           # overnight
include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r05_eval_smile.jl"))
```
Kaimon note: long includes trip the 10-min no-activity gate — the eval "fails" but Julia keeps
running; queue a trivial `ex` on the same session to catch completion, or run from a tmux REPL.

Live match_day_inference wiring for these leagues = follow-up session (NOT this stream).

## Findings log

### 2026-07-11 — r00 data QA (server) ✅ Stage 0 CLOSED

- **Season strings**: `"20/21"…"25/26"` (String7, slash format — use these in `target_seasons`).
  180 matches/season/tournament (20/21 COVID-half 90; 25/26 has 175 — few postponements).
- **O/U ladder**: dense u05–u65 every season; u75 patchy. **Kmax=4 comfortably dense**
  (≥300/360 per strike per season). 1X2 + BTTS ≈ full coverage.
- **Dispersion**: V/M 0.935 (56) / 0.953 (57) — sub-Poisson confirmed; DP base correct, NB inert.
- **δ_league scale**: level gap |log(2.817/2.688)| = **0.047** → prior N(0, 0.1) comfortably covers.
- **Team churn is real**: 2–4 teams swap divisions EVERY season + 1–3 new-to-segment teams
  (relegated from Championship / promoted from Lowland) → pooled team map + league offset justified.
- **Fold counts (targets 23/24–25/26, hs=2)**: match_week 104 · **match_biweek 48** (chosen) ·
  match_month 21. History ≈ 720 matches/fold.
  UPDATE 2026-07-12: grids run with **`warmup_period=0`** (user decision) — season-START folds
  included, since week-1 prediction off decayed prior seasons is the operational regime (season
  kicks off in weeks). ~+15 folds (~63/cell), ~+30% runtime. r04 aligned to match.
- **Market inversion**: DP λ plausible on 99.9% of 1969 odds-matches (λ_home med 1.48);
  smile full-ladder on 1832 (93%), median Λ^mkt(K) rises 2.37→2.91 — textbook market smile.
- **Phantom confirmed**: `required_features(::DynamicMarketGoalsTimeDecayModel)` throws
  `UndefVarError: MarketLambdaFeature` — l01 override required.
- **SURPRISE — Betfair EXISTS for 25/26 only**: `ds.betfair_odds` has 108,924 ticks over 315
  matches, all season 25/26 (earlier betdb `match_meta` join said 0 — wrong join path). Not enough
  for grid eval (one season) → **Bet365 close stays the benchmark**; use Betfair 25/26 as a
  secondary CLV check on the final winner only.

### 2026-07-12 — Stage 4 src graduation LANDED (structure; winner defaults pending r05)

The src changes are additive and grid-independent, so they shipped ahead of the grid verdicts:
- **`DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel`** in
  `src/models/pregame/engines/team_level/time_decay/goals_smile_league.jl` (included + exported
  from PreGame). Defaults = Ireland keeper (sup 1.0 / sw 0.5 / hl 180) — **⚠ update to the r05
  winner before production** (marked in-file).
- **`Features.LeagueFeature`** graduated: struct in `src/features/types.jl`, extractor in
  `src/features/extractors/core_extractors.jl`, exported. l01 now aliases it (`const LeagueFeature
  = Features.LeagueFeature`); loader-local copies removed.
- **Smile dispatch widened**: `AbstractSmilePoissonEngines` Union in
  `src/predictions/score_computation/smile_poisson.jl` covers the player + team smile engines
  for `extract_params`/`compute_score_matrix` (O/U keeps pricing through `SmileScoreMatrix`).
- **Phantom `MarketLambdaFeature` FIXED in src**: all 8 `*Market*` engines now request
  `Features.DoublePoissonMarketFeature()`; the dead export is removed. l01's temporary override
  deleted. (Any old code that imported `Features.MarketLambdaFeature` was already broken.)
- Verify with `r06_smoke_src.jl` after the grids; then bake winner defaults + `Pkg.test()`.

### 2026-07-12 — r01 smoke PASSED 29/29 ✅ + ⚠ SMILE RUNTIME FINDING (r01_out.txt)

Stage 1 CLOSED on quality: all 3 engines converge (global max R-hat ≤ 1.016, every new param
≤ 1.007); δ_league reads correctly (δ₅₆−δ₅₇ ≈ +0.02–0.035, expected +0.047; smile CI excludes 0);
smile prices genuinely (smile vs grid O/U max Δ 0.042); σ_smile ≈ 0.052 (market-hugging, same as
Ireland), σ_sup ≈ 0.249 (loose — model disagrees with market on supremacy, echoes Ireland);
iso σ_market ≈ 0.134 healthy.

**⚠ Runtime:** same 5-split window — none 11m / iso 40m / **smile 3h35m** (~20×). Chain
internals: median tree_depth 4/5/7 (max 8), median leapfrogs/iter **15/31/127**. The tight
sampled σ_smile≈0.05 pillar makes NUTS take ~8.5× more gradient evals/iter × ~2.3× per-eval
(the [n×5] smile matrix). Code audited vs `docs/turing_ad_performance_guide.md` — compliant
(broadcast-only, masks, views; it's the graduated player-engine pattern); the cost is leapfrog
COUNT (geometry), not an AD defect. Trees max at 8 ⇒ depth-8 cap is a no-op; binding caps are
6 (~2×) / 5 (~4×). → **`r01b_smile_depth_probe.jl`** tests max_depth {6, 5} (sampled-σ release
valve should keep them mixing). Grid A (r02) is unaffected — none cells are the fast ones; run
it now. Grid B waits for the r01b verdict (set `MAX_DEPTH` in r04; budget rule: cell wall ≈ 6 ×
probe wall).

### 2026-07-13 — r02 Grid A DONE (30.75h server) + r03 eval verdict: **hl365_hs2** ✅ Stage 2 CLOSED

**Convergence gate (r02_convergence.txt):** 12/14 cells ≥95% folds R-hat≤1.01 (60 folds/cell,
warmup_period=0). Excluded: `none_pois_hl60_hs3` (91.7%, marginal — dead corner anyway) and
`iso_nb_mw100_hl180_hs2` (75%, worst 1.021 — NB+iso-pillar mixes poorly; reference row only,
numbers read with caveat).

**r03 verdict — BEST_HL=365, BEST_HS=2** (`none_pois_hl365_hs2`), from `r03_out.txt`:
- Family-pooled LogLoss diff vs Bet365 close: hl365_hs2 best on **all three families** among
  gate-passing structural cells — x12 0.0143 (next: hl365_hs3 .0150, hl180_hs2 .0154),
  btts 0.0014, totals 0.0002 (≈ market-level on totals with NO pillar).
- Monotone hl gradient on x12 (hl60 .0168 → hl120 .0157 → hl180 .0154 → hl365 .0143): long
  memory wins despite promotion/relegation churn — sub-Poisson, stable-strength leagues.
- GLMEdge: short half-lives are actively pathological — significant NEGATIVE away coefs
  (hl60 ≈ −5.2, hl120 ≈ −4.4, hl180_hs1 −3.4 ⇒ over-reactive ratings anti-predict vs market);
  at hl365 every coef n.s. and closest to 0. hs axis flat; hs2 ≥ hs3 everywhere (older seasons
  add nothing), hs1 truncates the 365-day decay.
- RQR all cells well-centred (|mean| ≤ 0.03, std ≈ 1) — no goal-calibration pathology anywhere.
- References behaved: none_nb_hl180_hs2 ≤ none_pois_hl180_hs2 on every family (r inert on
  sub-Poisson data ⇒ **Poisson base confirmed**, no dispersion escalation). iso_nb (below gate)
  still shows the pillar doing its usual work (x12 .0083, totals −.0030) — motivates Grid B.
- Per-line note: over_K and under_K LogLoss diffs are IDENTICAL by construction (binary log
  score counts both sides) — the "wins both sides" check is per-strike, not per-side.
- Eval runtime gotcha: r03 = ~2h wall on the server (14 cells × 60 folds × 1070-match PPDs);
  kaimon 10-min gate kills the *eval*, Julia finishes fine — run with stdout redirected to
  r03_out.txt and read the file (pattern now standard for r05).

**Files updated:** r04 `BEST_HL=365.0/BEST_HS=2` (RERUN_CONTROLS stays false — winner IS a
Grid-A cell, none_pois control reused from scottish_decay_grid), r05 `_TAG="hl365_hs2"`.
**Next:** r01b depth probe (sets r04 MAX_DEPTH) → r04 Grid B → r05.

### 2026-07-13 — r01b depth probe: BOTH caps fail hard gate → Grid B REDESIGNED (fast-rank + confirm)

**Probe result (r01b_out.txt; ref depth-10: 215m, R-hat ≤1.007):**
depth 6 = 84.9m wall, worst new-param R-hat **1.077**; depth 5 = 43.2m, **1.379**. BUT posterior
means are IDENTICAL to depth-10 (σ_smile .0515/.0516, σ_sup .249/.2493, δ_gap .035/.035) — caps
truncate trajectories ⇒ sluggish-but-UNBIASED mixing. Per-fold diagnosis at depth 6: fold 1 the
log_φ block (1.077, ESS≈45 @600 samples — too thin for tails); folds 2–5 only team ratings
1.02–1.04 (ESS 100–265). Depth 5 = genuinely broken; discarded.

**Budget reality (measured task times @16t: d10≈107m, d6≈42m, d5≈21m per chain):** original
Grid B (10 cells × 60 folds × 4ch, depth 10) ≈ 25h/cell ≈ 10 days — infeasible.

**USER DECISION (option A of three): fast-rank + depth-10 confirm.** r04 rewritten:
- Grid: sup{40,70,100} × sw{0,50} + iso ctl + none ctl (8 cells; sw=0.4 column dropped).
- Trimmed: targets 24/25→25/26 (~40 folds), 1200/300 × 3 chains (samples doubled → ESS ×2).
- Per-cell depth: sw>0 cells depth 6 (**RANKING gate ≥95% folds ≤1.05**); sw=0/iso/none depth 10
  (hard gate — loose geometry, no smile pillar). Cheap cells run first.
- none_pois ctl RE-RUN at this spec (Grid-A cell pools 3 seasons — not comparable).
- Budget ≈ 30h ≈ 1.5 nights @16t.
- **NOTHING graduates from a depth-6 cell**: new `r04b_winner_confirm.jl` re-trains the r05
  winner at the Grid-A reference spec (depth 10, 3 seasons, 800/300×4, ~25h) under the HARD
  gate; then r05 re-run with INCLUDE_CONFIRM=true (compare per-line signs/pattern — confirm row
  pools 3 seasons vs Grid-B 2).
- r05: INCLUDE_CONFIRM knob + stdout-redirect run pattern documented in header.

Run order now: r04 (user, overnight) → r05 (redirect) → pick per-family winner → r04b
(overnight) → r05 INCLUDE_CONFIRM=true → Stage-4 winner defaults + r06.
