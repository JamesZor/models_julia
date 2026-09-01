# Experiment 03 — Notes, Database Addresses and Traps

Companion to `README.md`. This file records what was ingested, what the numbers mean, and
what went wrong along the way.

## 1. PostgreSQL addresses

Namespace: **`scottish_lower_joint_2426`** (database `mcmc_experiments` on `mcmc-beast`).
Ingested by `r48_sync_to_postgres.jl`; all eight fits round-tripped at 40 folds via
`load_fit(db, run_id)`.

| Model | `config_registry.id` | `runs` UUID | `portfolio_runs` UUID |
| :--- | :---: | :--- | :--- |
| `m00_joint_baseline` | 85 | `2c6e859c-29e7-4ae7-aa0a-e88343ba7672` | `cd138387-6760-40e0-ae25-c007cba8c044` |
| `m02_joint_squad_wealth` | 86 | `17315a41-665a-44db-81c5-a244316b5b4a` | `5405a63b-3ba7-49eb-ad3c-8c7cd7714913` |
| `m03_joint_distance` | 87 | `c1e9854b-bc48-46a4-b878-d3f7e6b656f8` | `7a9bedf6-5291-4acf-a211-9724d646e935` |
| `m04_joint_wealth_distance` | 88 | `7af622f3-1c17-4e0f-b7ee-ef6adef12251` | `60c0ae3a-f269-4a62-bddf-5463ed2e2bb2` |
| `m05_joint_production_wealth` | 89 | `5eff755c-3591-48d1-a2cc-5fc2744ddf88` | `c5192d31-11ca-4736-873b-1a2055c232be` |
| `m07_joint_bench_depth` | 90 | `9e65ff44-aeed-4704-8e08-eced05f66c86` | `6395ae9e-1822-4559-b050-ce476ea44525` |
| `m08_joint_composite` | 91 | `61fc5d87-1bd6-46d1-bb2b-c2aaad39e348` | `829f609b-9e8e-4c89-ad90-20105f99fa48` |
| `m00_poisson_control` | 92 | `d63f8877-b825-40ae-9ae5-d829e8b8a7f7` | `e9666640-83d5-4b06-a26e-1f8a217860fa` |

Shared components: **splitter 93**, **sampler 94**, **book 95**, **policy 96**.

```julia
db  = PostgresStorage("scottish_lower_joint_2426")
fit = load_fit(db, UUID("5eff755c-3591-48d1-a2cc-5fc2744ddf88"))   # m05
pf  = load_portfolio_db(UUID("c5192d31-11ca-4736-873b-1a2055c232be"), db)
explore_experiments(db)
```

## 2. ⚠ The database portfolios use a DIFFERENT price source from README §4

This is the single most important thing to know before reading `portfolio_runs`.

| | `README.md` §4 (r47) | PostgreSQL `portfolio_runs` (r48) |
| :--- | :--- | :--- |
| Prices | **Betfair exchange closing**, TWA [-20min, 0min] | **Bookmaker** `ds.odds` |
| `m05_joint_production_wealth` | **+121.02%**, Sharpe 1.409 | **−4.95%**, Sharpe −0.150 |
| `m00_poisson_control` | **+120.30%**, Sharpe 1.232 | **−26.31%**, Sharpe −0.740 |

Same posteriors, same `BookSpec`, same `PolicySpec`, opposite conclusions. Bookmaker prices
carry the overround; Betfair at 2% commission does not. `r48` uses `ds.odds` to stay
consistent with `r21_sync_to_postgres.jl` so experiments 01, 02 and 03 are comparable
*within* the database — but a reader who takes a `portfolio_runs` row as "what this strategy
earns" will reach the wrong answer.

### 2.1 The two price sources tell different, compatible stories
Against **bookmaker** prices the joint model's advantage is large: every joint arm loses
5–14% where the Poisson control loses **26.31%** (Sharpe −0.150 to −0.423 against −0.740).
Against the **exchange closing line** the same advantage nearly vanishes (ΔSharpe +0.01 to
+0.27, and four of seven arms return *less* than the control).

That is coherent rather than contradictory. Better probabilities matter most when there is a
large margin to overcome, and compress toward nothing against a line that has already priced
what the model knows — which is exactly what §3's Brier parity with the closing line implies.

**Neither source shows the joint model turning its predictive gain into exchange profit.**

## 3. Trap: config identity via `string()` is incomplete

`r48` originally refused ingestion, reporting that `m00_joint_baseline` used a different
splitter from `m00_poisson_control`. It did not — `==` on `GroupedCVConfig` falls back to
`===`, and its `Vector` fields are distinct objects after each `Fit` is deserialised.

The obvious fix — compare `string(splitter)` — would have been worse than the bug:

```
GroupedCVConfig(Targets=["24/25", "25/26"], Hist=2)
```

`show` omits `dynamics_col`, `warmup_period`, `end_dynamics` and `stop_early`. Two splitters
differing in `end_dynamics` stringify identically, and that is the difference between a
40-fold grid and one with **zero** out-of-sample fixtures (§5). `r48` now compares
field-by-field with `isequal`.

**This reaches the database.** `_truth_config_canonical` (`src/training/inference/db_storage.jl`)
also uses `string(config)`, so `configs.config_hash` inherits the same blind spot: two
genuinely different splitters can share one hash, and `save_fit`'s deduplication would treat
them as the same recipe. Not fixed — it affects every namespace and would rewrite existing
hashes. Flagged for a decision.

## 4. Trap: `end_dynamics = 0` yields an empty evaluation set

`nothing` is the "run to the end" sentinel. Measured on this store, targets 24/25 + 25/26:

| `end_dynamics` | folds | scored | OOS fixtures |
| :---: | :---: | :---: | :---: |
| `nothing` | 40 | 38 | 710 |
| `1` | 4 | 2 | ~20 |
| `0` | 2 | **0** | **0** |

r46 was launched once with `0`. It loaded the store, assembled eight arms, passed both
preflights and began sampling against a split with no held-out fixtures. `r46` now counts
scored folds and refuses below `R46_MIN_SCORED_FOLDS`.

`r40_train_pxg_rapm_models.jl` sets `R40_END_DYNAMICS = R40_SMOKE ? 1 : 0`, so its *full*
run passes `0`. **Not investigated here** — if r40's leaderboard has been read as
walk-forward evidence, verify it before trusting it.

## 5. Trap: `target_match_ids` is fitted, not held out

`src/Data/splitting/methods.jl:221` names the union of history and target `fitted_ids`. The
target block is the walk-forward's expanding *training* window; genuine out-of-sample
fixtures come from `get_next_matches` (~19 per fold, 710 across the grid). Summing
`target_match_ids` over folds gives 6430 and means nothing.

## 6. Environment

`CodecZstd` was added to `Project.toml` without a corresponding `Manifest.toml` entry on
`mcmc-beast` (the manifest is untracked, so `git pull` brings the declaration and not the
dependency). The package failed to load until it was installed with
`Pkg.add(name="CodecZstd", version="0.8"; preserve=Pkg.PRESERVE_ALL)` — `PRESERVE_ALL`
matters, because `Distributions` compat is `"0.25"` rather than a pin and 0.25.127 breaks
DistributionsAD's ReverseDiff extension. Verified 0.25.126 after the install.

## 7. Open items

1. **No paired significance test.** The −0.0028 log-loss gain and the ΔSharpe spread are
   both small relative to what 710 fixtures resolve. A paired bootstrap over per-fixture
   scores would settle it and needs no resampling — the latents are in the database.
2. **Calibration.** The joint model is sharper but less calibrated (ECE 0.0160 vs the
   control's 0.0099). Kelly is calibration-sensitive, and this is the most likely mechanism
   behind the weak exchange performance. A Layer-2 calibrator on the joint posterior is the
   obvious next experiment.
3. **`config_hash` blind spot** (§3) — needs a decision.
