# Scottish Upper (54/55) — pregame engine bake-off

Research stream for the **Scottish Premiership (54)** and **Championship (55)**, pooled into one
`Data.ScottishUpper()` segment. Season kicks off **early August 2026**; this stream must produce a
production engine pick before then, in 2–3 server-nights.

Parent streams:
- `current_development/scottish_lower_smile/` — 56/57 (League One/Two). CLOSED. Winner
  `TeamIsoDPGoalsModel` (iso market pillar, `mw=0.40`, `hl=365`, `hs=2`), routed **totals only**.
- `current_development/split_market_pillar/` — Ireland (79). CLOSED. Keeper `li_smile50` →
  `PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` (sup 1.0 / smile 0.5 / Kmax 4).

**Neither transfers by assumption.** The 56/57 stream proved explicitly that Ireland's market-pillar
routing did NOT carry to a different Scottish segment (Ireland's interior `mw` optimum was flat on
56/57; smile/supremacy added nothing to a team-level goals engine). 54/55 gets its own bake-off.

## Data reality (verified against betdb 2026-07-29, pre-stream)

| | Premiership (54) | Championship (55) |
|---|---|---|
| Finished matches | 1,188 / 6 seasons | 1,030 / 6 seasons |
| Bet365 (SofaScore) odds — 1X2 / O-U / BTTS | 99.7% | 99.9% |
| **Betfair** | **0 rows** ⟶ *see r00 correction: a backfill started 2026-07-29, 536 matches usable* | **0 rows** ⟶ *ingest pending* |
| **xG** | **25/26 only** (198) | 23/24+ (530) |
| SofaScore shots / SoT | 100% all seasons | 23/24+ only (533) |
| **BBC shots + SoT (`ds.bbc`)** | **100%, all 6 seasons** | **100%, all 6 seasons** |
| SofaScore player ratings | 100% all seasons | 23/24+ |
| `bigChanceCreated` | 96% | **never scraped (0)** |
| Lineups / incidents | 100% | 100% |
| `bbc_events` (live text → RAPM) | 23/24+ | 23/24+ |

Four consequences that shaped the design:

1. **No Betfair at design time** — `core.tournament_config` had `betfair_competition_id = NULL` for
   both tiers and no betfair component. Training anchor AND benchmark are therefore both the
   **de-vigged Bet365 close** (`prob_fair_close`), as on 56/57. That is self-referential for the
   market-anchored cells and weaker than Ireland's setup — say so in the results, and do not compare
   absolute numbers across streams. **⟶ SUPERSEDED IN PART: a Betfair backfill began during the
   planning session; see the r00 correction in the findings log. The uniform Bet365 training anchor
   still stands for this round.**
2. **The Ireland keepers cannot run.** Every one requires `Features.XGFeature`, and 54 has xG for a
   single season. Their *pillar designs* port to the team/league goals engines; the xG likelihood
   does not.
3. **BBC shots are the standout asset** — 100% for both divisions back to 2020, covering the 495
   Championship matches where SofaScore has no shot stats at all. The funnel engine has better data
   here than it had on 56/57 where it won 1X2. Feed it from `ShotsFunnelFeature` (BBC), **never**
   `ShotsFeature` (SofaScore, half-missing on 55).
4. **Tournament 54 stops at round 33 every season.** The 5 post-split Premiership rounds
   (30 matches/season, 180 total) are absent. Seasons end in April; late-season folds are thinner
   than the fixture list implies.

## Fixed design decisions (2026-07-29 planning session)

- **Pooled `[54, 55]`** in one model + zero-sum per-league offset `δ_league` on log-λ. Shared team map
  survives promotion/relegation (2–4 teams swap tiers every season).
- **Ladder** = the 56/57 goals family + funnel, with **SofaScore player ratings replacing the RAPM /
  plus-minus arms** (user decision — the ratings exist here, unlike on 56/57 where RAPM was built
  precisely because SofaScore ratings are absent).
- **Data floor 23/24** (user decision), i.e. the seasons where both tiers carry ratings.
- **CV**: `target_seasons = ["24/25", "25/26"]`, `history_seasons = 2`, `dynamics_col = :match_biweek`,
  `warmup_period = 0` (season-start folds kept — week-1 prediction off decayed prior seasons IS the
  operational regime, and the season starts in days).
- **`hl = 365`** inherited from the 56/57 verdict (monotone gradient favouring long memory), with one
  `hl = 180` control to check the gradient still points that way on a higher-turnover division.
- **Benchmark** de-vigged Bet365 close. No Betfair swap anywhere (none exists).
- σ's are **SAMPLED, never fixed**; `max_depth` is **never capped**; market inversion stays
  **Poisson-referenced**; `Kmax = 4`.

### ⚠ Stated assumption — the 23/24 floor vs `history_seasons = 2`

`hs = 2` means the 24/25 target folds look back into **22/23**, which is before the 23/24 floor.
Goals and BBC shots exist there; SofaScore ratings on tournament 55 do **not**. Rather than shorten
history for every arm to accommodate one, the CV config stays **identical across all cells** (that is
what comparability requires) and the ratings pillar simply sees fewer players in that window. `r00`
quantifies exactly how many. **If r00 shows 22/23 rating coverage is bad enough to distort the
ratings arm, the fallback is `history_seasons = 1` applied uniformly to every cell** — not just to
the ratings arm.

## Known gotchas carried in

- **`create_experiment_task` reads tournament ids from `ds.segment`** (`src/experiments/presets.jl`,
  `tournament_groups = [Data.tournament_ids(ds.segment)]`). There is no `tournament_groups` kwarg, so
  post-hoc filtering (as `scripts/run_dev.jl:207` does for 54/55) does **not** work. Hence the new
  `ScottishUpper` segment.
- **`TeamIsoDPGoalsModel` never graduated to src.** It lives only at
  `scottish_lower_smile/l01_team_dp_league.jl:163`. `l01_upper.jl` includes that file rather than
  copying — it also ships the `extract_parameters` / `Pred.compute_score_matrix` overrides the
  loader-local structs need. Without them `evaluate_experiments` silently drops those rows.
- **New engines must be in the prediction dispatch Union** or PPD generation takes the NegBin path
  and errors on a missing `r` column (`src/predictions/score_computation/poisson.jl`).
- **Silent split drop**: check `length(res.training_results.items)` per cell before reading any eval
  — a cell can look complete while holding no data.
- **9-field `DataStore`**: a 7-arg positional rebuild silently drops `bbc` + `bbc_events`, degrading
  the funnel engine to goals-only. Pass all 9 fields when rebuilding to swap one domain.
- **kaimon 10-min gate**: long `include`s report a timeout while Julia keeps running. Run evals with
  stdout redirected to a `*_out.txt` and read the file; queue a trivial `ex` on the same session to
  detect completion.
- **Local laptop cannot load the package** (2026-07-29): `DistributionsAD`'s ReverseDiff extension
  fails to precompile against the installed `Distributions` (`@check_args` API change). Unrelated to
  this stream — all verification happens on the server.
- Season strings: `sofascore.seasons.name` for these tiers is `"Premiership 23/24"` /
  `"Championship 23/24"`. **r00 must confirm the matches fetcher strips the competition prefix**, or
  `target_seasons = ["24/25", ...]` silently matches nothing.

## Files

- `r00_data_qa.jl` — Stage-0 gate. Season strings, odds ladder density, BBC/rating coverage,
  `δ_league` scale, V/M, fold counts, market-inversion sanity, betfair-empty confirmation.
- `l01_upper.jl` — loader: includes the 56/57 `l01` (for `TeamDPGoalsModel` / `TeamIsoDPGoalsModel`)
  and adds `TeamRatingDPGoalsLeagueModel` (goals + league offset + SofaScore-ratings pillar).
- `r01_smoke.jl` — 5-fold smoke over every candidate cell: convergence, **runtime calibration**, PPD
  end-to-end, `δ_league` read. Hard gate before any overnight run.
- `r02_grid_family.jl` — Night 1: the family bake-off. Gate → `r02_convergence.txt`.
- `r03_eval_family.jl` — per-line eval + per-family routing table.
- `r04_grid_pillar.jl` — Night 2: market-weight sweep on the winning family.
- `r05_eval_pillar.jl` — final eval.
- `RESULTS_scottish_upper.md` — results doc; fill as runners complete.

### Run order (server, kaimon REPL; REPL restart after any pull that changes structs)

```julia
# after: git -C <repo> pull --rebase --autostash origin feat/scottish-upper-bakeoff
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"   # broken LanguageServer dep on the server env
using BayesianFootball
const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_upper/r00_data_qa.jl"))     # gate
include(joinpath(ROOT, "current_development/scottish_upper/r01_smoke.jl"))       # gate + budget
include(joinpath(ROOT, "current_development/scottish_upper/r02_grid_family.jl")) # overnight
include(joinpath(ROOT, "current_development/scottish_upper/r03_eval_family.jl")) # redirect stdout
# -> pick per-family winner, edit r04 header, push/pull
include(joinpath(ROOT, "current_development/scottish_upper/r04_grid_pillar.jl")) # overnight
include(joinpath(ROOT, "current_development/scottish_upper/r05_eval_pillar.jl"))
```

**Gate discipline:** ≥95% of folds at max R-hat ≤ 1.01 before reading ANY grid table.

## Deferred / out of scope

- **Betfair anchor A/B** (Bet365 vs exchange as the training anchor — the open counter-test from the
  Ireland work, which predicted the verdict could flip on a deep top division). Blocked: no Betfair
  data for 54/55. If history lands, add it at r04, and score **both** arms against the same
  benchmark — anchoring to Betfair and then scoring against Betfair mechanically compresses the
  spread, which is what inflated the original Ireland read.
- **xG arm.** Only 25/26 on 54. If xG gets backfilled, the Ireland `outfield_*` family becomes
  available and deserves its own grid.
- Live match-day wiring for these leagues — separate session.

## Findings log

### 2026-07-29 — r00 data QA ✅ STAGE 0 CLOSED (`r00_out.txt`)

DataStore built fresh: matches 2218 · odds 49463 · lineups 85543 · incidents 37632 · bbc 2218 ·
bbc_events 25168 · **betfair 839667 (see the correction below)**.

- **Season strings are `"20/21"…"25/26"`** (plain slash format) and are **SHARED across both
  tournaments** — the matches fetcher does strip the competition prefix, so a pooled
  `target_seasons` works. Gate cleared.
- **Grid window**: targets `["24/25","25/26"]`, hs=2, warmup_period=0 → **`:match_biweek` = 40 folds**
  (week 77 / month 22), history ≈ 756 matches/fold. Some folds have 0 target matches (`min=0`) — a
  handful of wasted folds, same as 56/57.
- **δ_league**: mean goals 54 = 2.694 vs 55 = 2.620 → gap `|log(m1/m2)| = 0.0279`. Comfortably inside
  `league_offset_sd = 0.1`; **no widening needed**. Goal-total sd ratio 1.612/1.690 = **0.954**, i.e.
  the tiers differ in level but *not* meaningfully in spread — which is precisely the condition a
  level-only offset needs. The "wider quality gap" risk flagged in the plan did **not** materialise;
  pooling is sound.
- **⚠ DISPERSION REGIMES DIFFER**: V/M **54 = 0.965** (sub-Poisson) vs **55 = 1.09** (over-dispersed).
  On 56/57 both tiers were sub-Poisson and the NB reference was inert. Here a single pooled
  dispersion is a genuine simplification, so **`none_nb` is a more informative reference than it was
  on 56/57** — read it carefully rather than as a formality.
- **O/U ladder**: u05–u55 dense in both grid seasons (≈376/378); u65 dense from 22/23; u75 thin.
  **Kmax=4 comfortable.** 1X2 and BTTS ≈ full coverage.
- **BBC shots/SoT: 100% on BOTH tiers in ALL SIX seasons** (2218/2218). Confirms the funnel arm has
  the best shot data in the project.
- **Player ratings**: tier 54 100% every season; **tier 55 has ZERO before 23/24** and ~100% from
  23/24. So the 22/23 history block that `hs=2` pulls in for the 24/25 target folds is **0% rated on
  tier 55** — exactly the case the src masking fix was written for. See the decision below.
- **Team churn**: 12 teams in 54, 10 in 55, steady. 2–4 teams change division every season plus 0–2
  new to the segment — the shared team map + pooled design is justified.
- **Market inversion**: DP λ plausible on **2213/2213 (100%)** odds-matches (λ_home median 1.363);
  smile full-ladder on 2113/2213, median Λ^mkt(K) rises 2.297 → 2.759 — a textbook market smile.

#### ⚠ CORRECTION — Betfair is NOT absent; it is being ingested right now

The plan was written on a verified read of "0 Betfair rows for 54/55". That was true at the time.
**A Betfair backfill for these tiers started 2026-07-29 15:26 and is in progress:**

| tier | SUCCESS | PENDING | usable matches in the DataStore |
|---|---|---|---|
| 54 | 598 | 392 | **536** (23/24: 140 · 24/25: 198 · 25/26: 198) |
| 55 | 0 | 895 | **0** |

Consequences:
1. **The pooled grid keeps the de-vigged Bet365 close as the TRAINING anchor.** A pillar built from
   Betfair on tier 54 and Bet365 on tier 55 would confound the anchor with the league — not a
   comparison anyone can read. Uniform anchor, no exceptions.
2. Betfair on **tier 54** is now good enough for a **secondary CLV check** on the r03/r05 winner
   (536 matches over 3 seasons — more than 56/57 ever had).
3. The **full anchor A/B re-opens once tier 55 completes** (and 54's remaining 392 land). That is the
   open Ireland counter-test — a deep top division is exactly where "book beats exchange" was
   predicted to flip. Re-check `betfair.match_meta` status counts before r04.
4. **⚠ CACHE STALENESS.** `.cache/datastore_ScottishUpper.jls` was written mid-ingest.
   `load_datastore_cached` reuses any cache under 24h old, so **every runner until the ingest
   finishes will silently see a partial Betfair domain**. Re-fetch with
   `Data.load_datastore_cached(Data.ScottishUpper(); force = true)` once the backfill completes, and
   before any run whose result depends on Betfair.

#### Decisions taken

- `history_seasons` — **KEEP 2**. The 22/23 block is unrated on tier 55, but the src masking fix maps
  an unrated side to *league average* rather than to −10·base, so those matches still contribute
  their goals/shots signal and simply carry no rating information. Dropping to hs=1 would truncate
  the 365-day decay for every arm to protect one. **r01 CHECK 4 must confirm the mask holds** (no
  values below −30); if it does not, fall back to hs=1 uniformly.
- `league_offset_sd` — **keep 0.1** (gap 0.0279).
- `INCLUDE_SMILE` — decide from r01's runtime table.
- Betfair — **secondary CLV check only** this round; anchor A/B deferred to r04 pending the ingest.
