# Extending the Scottish Lower joint-player models into 2026/27

**Date** 2026-09-04 · **Branch** `feat/extend-scottish-lower-2627` · **Host** `mcmc-beast`
(16 physical cores / 32 SMT, 125 GB, Julia 1.12.4) · **Experiment**
`scottish_lower_joint_player_2426` on `mcmc_experiments`

Runners: [`r68_extend_joint_player_2627.jl`](r68_extend_joint_player_2627.jl) (samples),
[`r69_verify_matchday_2627.jl`](r69_verify_matchday_2627.jl) (audits).

---

## Verdict

| | |
|---|---|
| **Extension** | **CLEAN.** Folds 41–43 sampled, converged, and persisted for both runs. The 40 historical folds were reloaded, not refitted. |
| **Saturday 2026-09-05 pricing** | **NOT READY.** Two blockers, neither caused by the extension. `MatchDay.matchday_latents` cannot serve this model family at all (§5.1), and tomorrow's exchange data does not exist yet (§5.2). |

The two halves are independent, and the report keeps them apart deliberately: the training
side is sound and can be relied on; the serving side has a defect that predates today.

---

## 1. What was asked for, and what the data supported

The brief expected 3 new folds covering 49 August fixtures. The first preview returned **2**
folds and **39** fixtures:

```
| Run ID | Run Name                 | Existing | New | New Matches | New Date Range           |
|     67 | m12_joint_hybrid_synergy |       40 |   2 |          39 | 2026-08-01 to 2026-08-22 |
```

The cause was cache staleness, not a splitter problem. `mcmc-beast`'s
`.cache/datastore_ScottishLower.jls` was written 2026-09-01 17:59 (65.2 h old) and stops at
2026-08-22. `betdb` itself holds the full August programme:

| date | finished fixtures |
|---|---|
| 2026-08-01 | 10 |
| 2026-08-08 | 10 |
| 2026-08-15 | 9 |
| 2026-08-22 | 10 |
| 2026-08-28 | 1 |
| 2026-08-29 | 9 |
| **total** | **49** |

The 2026-08-28/29 round — 10 fixtures, and precisely the round Fold 43 is built on — was
ingested *after* the cache was written. Re-running with `--refresh` gave the expected plan:

```
| Run ID | Run Name                    | Existing | New | New Matches | New Date Range           | Est.    |
|     67 | m12_joint_hybrid_synergy    |       40 |   3 |          49 | 2026-08-01 to 2026-08-29 | 14m 19s |
|     63 | m05_joint_production_wealth |       40 |   3 |          49 | 2026-08-01 to 2026-08-29 | 10m 24s |
```

**Operational note.** `--refresh` is not optional for a same-week extension, and a preview that
silently reports one fold fewer than expected is the only symptom. The DataStore is now
refreshed (2019 matches) and the cache rewritten.

---

## 2. Sampling

`TT.extend_fit` with `QueuedExecution(16)`, `pinthreads(:cores)`, BLAS pinned to 1 thread.
**Zero crashes; zero folds refitted.**

| run | run id | new folds | new latents | wall time | window |
|---|---|---|---|---|---|
| `m12_joint_hybrid_synergy` | 67 | 41, 42, 43 | 49 | **13.1 min** | 11:14:26 → 11:27:2x |
| `m05_joint_production_wealth` | 63 | 41, 42, 43 | 49 | **9.4 min** | 11:27:17 → 11:36:4x |

Total wall clock ≈ 22.5 min, against a 24m 43s estimate.

The 3-field deserialization shim for `JointGammaPoissonObservation` (r68 §"Compatibility Shim",
mirroring `l66` §0) was required and worked: every historical fold deserialized, so the 40
existing chains were loaded from `fit_artifacts` rather than resampled.

---

## 3. Convergence

### 3.1 The new folds — what the extension is answerable for

Both runs are clean on every gate the brief names.

| run | fold | R̂ | ESS bulk | ESS tail | divergences | fixtures | window |
|---|---|---|---|---|---|---|---|
| m12 | 41 | 1.00618 | 1007 | 1344 | 0 | 20 | 2026-08-01 → 08-08 |
| m12 | 42 | 1.00512 | 1375 | 1080 | 0 | 19 | 2026-08-15 → 08-22 |
| m12 | 43 | 1.00496 | 1612 | 1599 | 0 | 10 | 2026-08-28 → 08-29 |
| m05 | 41 | 1.00464 | 1196 | 1254 | 0 | 20 | 2026-08-01 → 08-08 |
| m05 | 42 | 1.00554 | 1076 | 1046 | 0 | 19 | 2026-08-15 → 08-22 |
| m05 | 43 | 1.00474 | 1176 | 1382 | 0 | 10 | 2026-08-28 → 08-29 |

Worst case across folds 41–43: R̂ 1.0062, bulk ESS 1007.5, tail ESS 1046.0, **0 divergences**.

### 3.2 The whole 43-fold run — what MatchDay's gate sees

Two of the brief's six audit items fail, and **both failures live entirely in the historical
folds**. The extension did not introduce either.

| gate | m12 | m05 |
|---|---|---|
| max split R̂ ≤ 1.05 | PASS — 1.0104 (fold 12) | PASS — 1.0084 (fold 17) |
| min bulk ESS ≥ 100 | PASS — 880.7 | PASS — 647.3 |
| min tail ESS ≥ 100 | PASS — 634.2 (fold 12) | PASS — 251.6 (fold 21) |
| divergences == 0 | **FAIL — 3**, at folds 3, 19, 25 | **FAIL — 1**, at fold 26 |
| `Evaluation.convergence_verdict` | PASS | **FAIL — tail ESS** |

Two things follow, and they are different in kind:

* **m12's 3 divergences** are one transition each in folds 3, 19 and 25. Each fold's chain is
  800 draws x 4 chains = 3,200 transitions, so that is a rate of 0.03% against the library's
  `max_divergence_rate = 0.001` (0.1%) — which is why `convergence_verdict` still passes and
  `canonical_fit(require_converged = true)` loads. The brief's "exactly 0" is a stricter gate
  than the framework applies, and it fails on chains sampled long before today.
* **m05 cannot be loaded for pricing at all.** `convergence_verdict` refuses it on tail ESS
  251.6 at **fold 21**, against the library's `min_ess = 400.0` gate. This is why r68's own step [3/3]
  aborted with a stack trace after printing m12's row — the audit called
  `canonical_fit(...; require_converged = true)` on a run that has never been able to satisfy
  it. The sampling had already completed and persisted at that point; nothing was lost.

`fold_results` and the in-memory `Fit` agree on R̂ for all 43 folds in both runs, so the two
records of the same chain are not drifting.

---

## 4. Persistence

Every check passes for both runs. Counted from `mcmc_experiments` directly, independently of
the serialized artifact.

| check | m12 | m05 |
|---|---|---|
| `Fit` carries 43 folds | PASS | PASS |
| OOS fixtures == 759 (710 + 49) | PASS | PASS |
| `fold_results` rows == 43 | PASS | PASS |
| `match_latents` rows == 759 | PASS | PASS |
| `fit_artifacts` blob rewritten | PASS (103.8 MB) | PASS |
| `fold_results` agrees with `Fit` on R̂ | PASS | PASS |

The run rows were updated in place on the same immutable UUIDs — m12's
`runs` row now reads `status = completed`, `finished_at = 2026-09-04T11:27:12`. Fold numbering
appended rather than renumbered, which is the property `select_split` depends on.

---

## 5. MatchDay readiness for 2026-09-05

Card: 10 fixtures, all kicking off 14:00 UTC (15:00 BST), tournaments 56 and 57. `as_of` for
the checks below is T−25 = `2026-09-05T13:35:00`.

**What works:**

| gate | result |
|---|---|
| `canonical_fit(db, m12; require_converged = true)` loads | PASS |
| reports `converged = true`, `folds = 43` | PASS |
| card has 10 fixtures | PASS |
| `select_split` chose **fold 43** of 43 | PASS |
| `select_split` warning | **empty** |
| every team on the card is in fold 43's `team_map` (24 teams) | PASS |
| a lineup is available for every fixture | PASS |

`select_split` identifying fold 43 with no warning is the specific thing the extension was for,
and it holds. So does the `team_map`: `ross-county` and `airdrieonians` moved up for 26/27 and
were absent from the 40-fold team map — the r06 runbook had to drop them from the card. Fold 43
covers them.

### 5.1 BLOCKER — `matchday_latents` cannot serve this model family

```
FieldError: type PoissonCountModel has no field `player_ratings_feature`,
available fields: `interception`, `dynamics`, `home_advantage`, `covariates`, `observation`, `guard`
  at MatchDay.materialise!(::RatingsFromTracker, ::Val{:player_ratings_map}, …)   src/MatchDay/inference.jl:228
```

`RatingsFromTracker` was written for the older player-level engine family, where the tracker
hangs off `model.player_ratings_feature`. m12 is a builder-family `PoissonCountModel` whose
player term is a `PlayerLineupPillar{ShotsPlusMinusFeature, BenchWeightedPlayerAggregation}`
inside `model.covariates`. The materialiser claims `:player_ratings_map`, then dies reading a
field that does not exist.

**The crash is the smaller half of the problem.** Behind it:

* `MatchDay.INJECTABLE_KEYS` is `(:player_ratings_map, :league_lookup)`.
* The builder engine does **not** read `:player_ratings_map` at OOS. It reads
  `:player_lineup_ratings_map` — `get(d, :player_lineup_ratings_map, Dict{Int,PMLineupAggregate}())`
  in `src/models/pregame/builder/engine.jl:531`, consumed by
  `predictor_oos(::PlayerLineupPillar, …)`.
* That key is in fold 43's FeatureSet (2019 entries) but covers **0 of tomorrow's 10 fixtures**,
  because they are not in `ds.matches`. It is not in `INJECTABLE_KEYS`, so MatchDay neither
  materialises it nor checks it, and `check_coverage` only inspects `:player_ratings_map`.

So fixing the `FieldError` alone would produce a **worse** outcome than the crash: every fixture
would fall through to `_pm_empty_lineup_aggregate()`, the lineup pillar would contribute exactly
zero to all 10 prices, and nothing would raise. m12 would quietly price as m05-with-extra-noise
— and m12 over m05 is the entire thesis of this experiment. This is the silent stale-value
failure `inference.jl`'s own docstring says the guard exists to prevent; the guard just does not
cover this key.

**What a fix needs** (not attempted here — it changes what gets priced, and belongs behind
tests rather than inside a verification run):

1. A `LineupAggregateFromRAPM` materialiser keyed on `Val{:player_lineup_ratings_map}`, building
   a `PMLineupAggregate` per fixture from `fs.data[:plus_minus_ratings]` (already exposed,
   `Dict{Int,Float64}`) and the card's XI. For `BenchWeightedPlayerAggregation` only the
   `home_outfield` / `away_outfield` / `home_bench` / `away_bench` fields are read, mirroring the
   `values[1..4]` accumulation in `src/features/extractors/plus_minus_extractors.jl`. The
   minute-weighted fields need a rolling history the extractor does not export, and m12 does not
   use them.
2. `:player_lineup_ratings_map` added to `INJECTABLE_KEYS` **and** to `check_coverage`, so an
   uncovered fixture is refused rather than priced neutral.
3. `RatingsFromTracker` to decline (`return false`) rather than throw on a model with no
   `player_ratings_feature`, so the chain can fall through to the pillar materialiser.

### 5.2 BLOCKER — tomorrow's exchange data does not exist yet

Every card resolved `UNRESOLVED(absent_from_crosswalk)`. Measured against `betdb`:

* `betfair.match_meta` holds **0 rows** for all 10 fixture ids — the crosswalk job has not run
  for them.
* `betfair_live.market_metadata` holds **0 events and 0 markets** with `open_date` inside
  2026-09-05. The latest live market anywhere in the store opens **2026-09-03T18:45**.

This is expected 27 hours out rather than a defect: markets are published closer to kick-off and
the collector has not seen them. It does mean pricing will depend entirely on the `LiveNameMatch`
fallback (the crosswalk being empty for these fixtures), and that **the identity and book gates
must be re-checked at T−25 tomorrow, not assumed from today's result**. `matchday_latents` itself
does not need identity; `price_slate` does.

---

## 6. Two environment findings worth carrying forward

**`.env` is not loaded in a REPL that uses a precompiled `BayesianFootball`.**
`src/BayesianFootball.jl:9-12` calls `DotEnv.load!(ENV, env_path)` in the module body, which runs
at *precompile* time. When the package is already precompiled the body does not re-run, so
`ENV["BF_DB_URL"]` is absent — confirmed here by a `KeyError` after `using BayesianFootball`.
Anything that reads it (`--refresh`, `MatchDay._conn`, `SofaScoreEvents`, `ProvisionalDB`) fails
until the caller does it by hand:

```julia
using DotEnv; DotEnv.load!(ENV, "/root/BayesianFootball/.env")
```

**r68's step [3/3] should not gate on `require_converged = true`.** It calls
`canonical_fit(...; require_converged = true)` for every model in the list, so one run that
cannot clear the library's gate — m05, on a historical fold — aborts the audit for the whole
batch with a stack trace, after the expensive part has already succeeded. Reporting the verdict
per model would say the same thing without the traceback. `r69` does it that way.

---

## 7. Reproducing

```bash
# on mcmc-beast, /root/BayesianFootball, warm REPL with -t 16
julia> using DotEnv; DotEnv.load!(ENV, "/root/BayesianFootball/.env")
julia> include("experiments/scottish_lower/06_joint_player_lineup_fusion/r68_extend_joint_player_2627.jl")
julia> r68_extend_joint_player_2627(["--preview", "--refresh"])   # plan
julia> r68_extend_joint_player_2627()                              # sample
julia> include("experiments/scottish_lower/06_joint_player_lineup_fusion/r69_verify_matchday_2627.jl")
julia> r69_verify_matchday_2627()                                  # audit
```

`r69` is read-only against both databases and re-runnable.
