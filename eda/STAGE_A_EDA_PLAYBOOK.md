# Stage-A League EDA Playbook

**Purpose.** A repeatable recipe for characterising a *new* tournament as a data-generating
process **before** it is used for modelling/betting. Hand this file to an LLM (or work through it
yourself) when a new league is added; it reproduces the analysis done for
`eda/ireland_validation/` (79), `eda/first_division_validation/` (718) and
`eda/veikkausliiga_validation/` (31).

**What Stage A answers**
1. Which **goals likelihood family** the L1 Bayesian engine should use (Poisson vs NB vs COM/Weibull),
   and whether **Dixon-Coles** low-score dependence (ρ, τ) is needed.
2. Which **teams are genuinely good** (attack/defence for goals *and* xG), shrunk for small samples.
3. Whether **player ratings** are usable (coverage), and the **squad-quality** ranking.
4. **Stage-B readiness**: which feature pillars (xG, bigChance, betfair) exist and from which season.

**What Stage A is NOT.** No model training, no backtest. EDA + a report with real captured numbers.

---

## 0. Before you start — verify the data exists

Do **not** assume coverage. Probe `betdb` (`:5433`, schema-split `sofascore.*`/`betfair.*`/`core.*`)
for the tournament id directly, or run the load and inspect. Confirm, per season:
played matches, xG coverage, player-rating coverage, `bigChanceCreated` presence, betfair presence.
See the [[betdb-data-coverage]] memory — coverage **grows over time**, so re-probe per tournament
rather than trusting old totals.

Record the verified signature (matches, season windows, feature start-dates) in the PROMPT file
*before* writing code, the way `eda/veikkausliiga_validation/PROMPT_veikkausliiga_eda.md` does.

---

## 1. Add the data segment (`src/` — the only `src/` change)

`src/Data/fetchers/segments.jl`:
```julia
struct <LeagueName> <: DataTournemantSegment end
tournament_ids(::<LeagueName>) = [<id>]    # one or more tournament ids
```
`src/Data/data-module.jl`: add `<LeagueName>` to the `export` list.

The Fetch→Process→QA pipeline is fully segment-driven; nothing else in `src/` changes.

> **Server gotcha:** a new struct in the Data module is **not** picked up by Revise. After
> `git pull` on the server you must `manage_repl restart` the REPL (see §6). In-memory globals
> are lost on restart — files are untouched.

---

## 2. Create the EDA folder `eda/<league>_validation/`

Follow the loader/runner convention (`lXX_*` definitions, `rXX_*` execution). Three+1 files:

### `l01_<league>_logic.jl` — loader (reuse, don't rewrite the fitters)
`include` the existing fitter libraries; do not duplicate them:
- `../first_division_validation/l01_first_division_logic.jl` — which transitively includes the
  ireland `l00` + `l01` fitters. This gives you:
  - **Count fitters / ladder:** `get_goals`, `summarise_count`, `compare_count_models`,
    `compare_nb1_nb2`, `analyze_goal_models`, `chi_square_gof`, `rootogram_data`.
  - **Bivariate / heavyweight:** `analyze_heavyweight_models` (Dixon-Coles ρ ladder).
  - **Audit:** `datastore_overview`, `feature_coverage_by_season`.
  - **League diagnostics:** `test_overdispersion`, `test_home_advantage_mean`,
    `test_home_advantage_variance`, `test_team_volatility`, `test_temporal_stability`.
- Watch include order to avoid redefinition clashes (first_division already pulls in ireland's files).

**New per-team / rating functions to add** (these are the generic, reusable additions — copy them
verbatim from `eda/veikkausliiga_validation/l01_veikkausliiga_logic.jl`):
- `clean_pos` — normalise SofaScore positions to G/D/M/F (unknown → M). Mirrors
  `current_development/match_day_inference/src/ratings.jl`.
- `bh_adjust(pvals)` — Benjamini–Hochberg FDR adjustment across the ~12 simultaneous team tests.
- `build_team_match_long(ds)` — one row per (match, side): `match_id, season, team, is_home,
  goals_for, goals_against, xg_for, xg_against`. xG from `expectedGoals_home/away`, `period=="ALL"`.
- `fit_team_attack_defence(long; min_matches=15)` — per team, **Gamma–Poisson empirical-Bayes
  shrinkage** of the count rate (prior `Gamma(α₀,β₀)` moment-matched from team rates; posterior mean
  `(α₀+k)/(β₀+n)`) + a **quasi-Poisson log-rate Wald z-test** vs the pooled league rate
  (`se=√(φ/k)`). Returns attack (goals_for) and defence (goals_against) tables, BH-adjusted.
- `fit_team_xg_attack_defence(long; min_matches=15)` — continuous analogue on xg_for/xg_against:
  per-team mean + **Welch t-test** (`UnequalVarianceTTest`) vs the rest, + **Normal–Normal
  hierarchical shrinkage** (`λ = τ²/(τ²+σ²/n)`). 2023+ only (wherever xG begins).
- `rating_coverage_audit(ds)` — per-season: played, frac matches with any rating, avg rated starters
  per team per match.
- `rating_position_coverage(ds)` — coverage by clean_pos (pooled). **Caveat:** pooling over unrated
  seasons + unknown→M default deflates the M bucket; the per-season table is the reliable signal.
- `build_team_rating_long(ds)` — per (match, side) **minute-weighted** team match rating.
- `fit_team_rating_dist(long; min_matches=10)` — per team Normal fit + Welch vs rest +
  Normal–Normal shrinkage; ranked squad-quality table, BH-adjusted.

### `r01_<league>_runner.jl` — sectioned execution with captured `#= RESULT … =#` blocks
Run section by section in the kaimon REPL; paste real numbers back into the `#= =#` blocks.
The 9 sections (mirror `eda/veikkausliiga_validation/r01_veikkausliiga_runner.jl`):

1. **Load:** `ds = D.load_datastore_sql(D.<LeagueName>())` (fresh SQL — no cache for a new segment).
2. **Audit + per-season coverage:** `datastore_overview(ds)`; `feature_coverage_by_season(ds)`.
   → the Stage-B readiness map (xG/bigChance/betfair start seasons).
3. **Marginal moments:** `summarise_count` on home / away / total goals (mean, var, V/M, zeros, skew).
4a. **Validation guard:** re-fit a *known* league (Ireland 79 total goals via
   `load_datastore_cached(Ireland())`) and confirm the fitter reproduces its published Poisson-regime
   verdict before trusting the new league's output. (Exact AIC may drift as the cache gains matches;
   the qualitative verdict must hold.)
4b. **Goals ladder:** `analyze_goal_models`, `compare_count_models` (home/away/total),
   `compare_nb1_nb2`, `analyze_heavyweight_models` (Dixon-Coles ρ).
5. **Goodness-of-fit** for the AIC-winning family: `rootogram_data` + `chi_square_gof` (+ Poisson as
   reference — does plain Poisson survive χ²?).
6. **League diagnostics:** `test_overdispersion`, `test_home_advantage_mean/variance`,
   `test_team_volatility` (within-team dispersion index), `test_temporal_stability`.
7. **Per-team attack/defence:** `build_team_match_long` → `fit_team_attack_defence` (goals) and
   `fit_team_xg_attack_defence` (xG, 2023+). Check goals and xG rankings agree.
8. **Rating coverage:** `rating_coverage_audit` + `rating_position_coverage`.
9. **Per-team ratings:** `build_team_rating_long` → `fit_team_rating_dist`.

### `veikkausliiga_eda.md`-style report — `<league>_eda.md`
TL;DR + sections mirroring `eda/first_division_validation/first_division_eda.md`: motivation,
coverage, marginal moments, candidate-distribution maths, univariate + DC ladders, GoF, league
diagnostics, per-team attack/defence (goals + xG), rating coverage, per-team ratings, Stage-B
readiness. **Fill with actual captured numbers**, not placeholders.

### `PROMPT_<league>_eda.md` — the brief
Task, goals, locked design decisions, and the verified DB signature. Written *first*.

---

## 3. Locked design decisions (defaults — confirm if the user wants different)

- **Team framing:** attack & defence split (for vs against), for **both** goals and xG.
- **Comparison rigor:** per-team fit + **formal test vs the pooled league rate** + **empirical-Bayes
  shrinkage** ranking (Gamma–Poisson for counts, Normal–Normal for xG/ratings), BH-FDR adjusted.
- **Ratings:** coverage audit **and** per-team rating distribution.
- **Scope:** **standalone** characterisation (no contrast league) except the §4a validation guard.
- **Execution:** end-to-end via the kaimon REPL; capture real numbers into the report.

---

## 4. Statistical methods reference (what each test decides)

| Question | Method | Decision rule |
|---|---|---|
| Over/under-dispersed? | Index of dispersion **V/M** = var/mean | ≈1 Poisson; >1 over-dispersed (NB/COM) |
| Which count family? | AIC/BIC over Poisson, NB1 (Var=φμ), NB2 (Var=μ+μ²/r), COM-Poisson, Weibull-count, ZIP, ZINB | lowest AIC wins; ΔAIC<2 ≈ tie → prefer simpler |
| NB justified vs Poisson? | `test_overdispersion` ΔAIC | ΔAIC<~2 → Poisson is fine |
| Does the family actually fit? | rootogram (hanging, √-scale) + Pearson **χ²** GoF | χ² p>0.05 not rejected; check tail in rootogram |
| Low-score dependence? | Dixon-Coles bivariate ladder, **ρ** | ρ≈0 and independents win AIC → no τ correction needed |
| Home advantage? | mean diff + **Mann–Whitney U**; variance **F-test** | MWU p<0.05 → real edge |
| Stable over time? | **Kruskal–Wallis** across months/seasons | p>0.05 → stable |
| Team strength (counts)? | **Gamma–Poisson EB shrinkage** + quasi-Poisson log-rate **Wald z** | shrunk rate ranks; BH-adj p<0.05 = real |
| Team strength (xG/ratings)? | mean + **Welch t** vs rest + **Normal–Normal shrinkage** | shrunk mean ranks; BH-adj p<0.05 = real |
| Multiple-comparison control | **Benjamini–Hochberg** across ~12 teams | use `p_adj`, not raw p |

**Why shrinkage matters:** with ~12 teams and small per-team n (e.g. a 22-match team), raw rates
over-state extremes. EB/Normal–Normal pulls noisy small samples toward the league mean — e.g. in 31,
EIF's raw attack 0.864 shrank to 1.090 and was correctly *not* flagged, while its genuine 22-match
defensive collapse (2.32) still survived.

---

## 5. Interpreting the result → modelling recommendation

- **Near-Poisson** (V/M≈1.0–1.08, NB beats Poisson by <2 AIC, Poisson not χ²-rejected, ρ≈0):
  → **Poisson or single-knob NB, Dixon-Coles-free**. (Ireland 79, Veikkausliiga 31.)
- **Clearly over-dispersed** (V/M≳1.13, NB wins by ~9–12 AIC): → **NB engine**, consider stratified
  dispersion. (First Division 718.)
- **Low-score dependence** only if ρ materially ≠ 0 *and* DC variants beat independents on AIC.
- **Zero-inflation** localised to home/away → NB usually absorbs it; only reach for ZIP if it wins
  cleanly on that side.
- Judge any later market/staking work on **growth G**, not LogLoss (see [[staking-research-conclusions]],
  [[totals-compression-is-denoising]]).

Record the new league's one-line signature as a memory (like [[veikkausliiga-31-signature]],
[[first-division-718-signature]]).

---

## 6. Execution & sync (kaimon, server)

Edits reach the server only via git (see [[server-file-sync-workflow]], [[kaimon-repl-on-server]]):
1. Branch `eda/<league>-validation` off main; commit segment + EDA files; **push**.
2. Server `/root/BayesianFootball`: `git fetch && git checkout eda/<league>-validation && git pull`.
3. **`manage_repl restart`** (new Data struct → Revise won't see it). `start_session` — **never retry
   on timeout**, it spawns a process anyway ([[kaimon-start-session-no-retry]]).
4. In `ex`: end every expression with a returned string value (`q=false`) — **println output is
   stripped**. Long loads (~40s SQL+betfair) become background jobs → poll with `check_eval`.
5. Sanity checks: load returns ≈ expected match count; coverage table matches the verified signature;
   §4a guard reproduces the known Ireland verdict; per-team tables list ~all teams; ratings ≈100%
   from their start season, 0 before.
6. Paste captured numbers into the report; commit; push. Merge to main after review.
7. Update [[betdb-data-coverage]] (betfair/stats coverage for the new tournament) and add the
   league-signature memory.

**Leave the server as you found it:** if `git checkout` is blocked by an untracked file, back it up,
verify it's identical to a committed copy, then restore afterward. Restore the original branch when done.

---

## 7. Reference implementations (copy from these)

- `eda/veikkausliiga_validation/` — most complete template (this playbook's source). l01 has all the
  generic per-team/rating functions; r01 has the 9-section structure with captured results.
- `eda/first_division_validation/` — the 718 NB-regime study (count + DC ladder library lives here).
- `eda/ireland_validation/` — the original 79 Poisson-regime study (base count fitters + league
  diagnostics live here, in `l00_validation_logic.jl` + `l01_bigchance_logic.jl`).
