# Experiment 02 — Database Addresses and Traps

Companion to `README.md`. Records what was ingested into PostgreSQL, and what to know before
reading it. Written when `r35`/`r36`/`r37` were added; the sampling itself is r31's.

## 1. PostgreSQL addresses

Namespace: **`scottish_lower_negbin_2426`** (database `mcmc_experiments` on `mcmc-beast:5432`).
Ingested by `r35_sync_to_postgres.jl` from the completed r31 fits at
`/root/BayesianFootball/experiments/scottish_lower_2426_negbin/`; all five round-tripped at 40
folds with dispersion intact, verified by `r36_verify_postgres_sync.jl`.

| Model | `runs.id` | `runs` UUID | `config_registry.id` |
| :--- | :---: | :--- | :---: |
| `m00_negbin_baseline` | 69 | `aad544b3-a614-4e5a-9873-6338a77f4ddf` | 351 |
| `m02_negbin_wealth` | 70 | `432774c7-be04-4aad-8dca-44d06f7adbdc` | 352 |
| `m03_negbin_distance` | 71 | `a69ca827-d406-45d4-bc2b-5ee7a964e796` | 353 |
| `m04_negbin_joint` | 72 | `8241f725-c72a-4ac9-b942-0f5c95c14c59` | 354 |
| `m05_negbin_production_wealth` | 73 | `f2089ef6-8578-4578-afc6-508977971ac1` | 355 |

Shared components: **splitter 356**, **sampler 357**, **book 358**, **policy 359**.
Assembled `FitConfig` recipes are 360–364.

```julia
db  = PostgresStorage("scottish_lower_negbin_2426")
fit = load_fit(db, "m05_negbin_production_wealth")
observation_family(fit.latents)   # :negbin — see §3
explore_experiments(db)
```

## 2. Two portfolios per model, and why

Each run carries **two** `portfolio_runs` rows under the same `BookSpec` and `PolicySpec`,
distinguished by `metadata->>'odds_source'`. Query with the tag or you will get whichever the
`ORDER BY` happened to surface.

| Model | `betfair_twa_minus20_to_close` | `bookmaker_close` |
| :--- | :--- | :--- |
| `m00_negbin_baseline` | `063325c7-7454-4af9-8b9d-5acb27e35d34` | `63934764-1c30-472e-8724-38d40c470833` |
| `m02_negbin_wealth` | `fc81e41b-f049-47fe-85ed-62c96d3cf614` | `ad5d4b76-21da-4eac-957e-4eb77eb84b38` |
| `m03_negbin_distance` | `8a732834-8704-4256-93e1-7246f13fd8a6` | `c2c1a0f8-1f8d-4afd-bcc5-f9277cbc265f` |
| `m04_negbin_joint` | `df868994-42bc-4b5e-83f8-d537d23d3c22` | `5e680ab6-4281-4cde-8b73-9606d1c64d50` |
| `m05_negbin_production_wealth` | `0ef9e6b5-bf58-4f72-a9e5-c71a1134abe5` | `e6cc05e9-0731-4db3-8417-e2ddf22bb091` |

The two disagree completely, which is the point of storing both:

| Model | Betfair close | Bookmaker close |
| :--- | :--- | :--- |
| `m00_negbin_baseline` | **+114.18%**, Sharpe 1.019 | **−36.93%**, Sharpe −0.895 |
| `m05_negbin_production_wealth` | **+158.99%**, Sharpe 1.288 | **−31.52%**, Sharpe −0.754 |

Identical posteriors, identical specs, opposite conclusions. Bookmaker prices carry the
overround; the exchange at 2% commission does not. Experiments 01 and 03 persisted only the
bookmaker figure and experiments 05 and 06 only the exchange one, which is why
`03_joint_gamma_poisson/NOTES.md` §2 warns that a `portfolio_runs` row read as "what this
strategy earns" gives the wrong answer. Storing both, tagged, is the fix here; rows written
before the convention have no `odds_source` and were priced off `ds.odds`.

`r35`'s deduplication keys on `(model_run_id, book_spec_hash, policy_spec_hash, odds_source)`
for the same reason. Keying on the spec hashes alone — as `r21` does, correctly, for its one
price source — would silently discard the second write as a duplicate.

## 3. Trap: a NegBin run that comes back Poisson is invisible

`CountLatents` carries dispersion in `observation_params` as `(; r_h, r_a)`, or `nothing` for
Poisson. If the dispersion is lost anywhere in the round trip, the container still constructs,
`load_fit` still succeeds, `compute_score_grid!` still dispatches — to the Poisson kernel — and
the portfolio still returns a plausible ROI. Nothing raises. The run is simply a Poisson model
wearing a negative-binomial name.

`r35` checks `observation_family(fit.latents) == :negbin` before saving, `r36` re-checks it on
every reload and compares `r_h`/`r_a` element-wise against the source, and `r37` re-checks it
after extension. This is cheap; discovering it downstream is not.

## 4. 26/27 extension

`r37_extend_negbin_2627.jl` widens the splitter by one target season and lets `extend_fit`
sample only the delta. Executed for `m00_negbin_baseline` on 2026-09-03:

| | Before | After |
| :--- | :---: | :---: |
| Folds | 40 | **42** |
| OOS fixtures | 710 | **749** |
| max R̂ | — | 1.0082 |
| min bulk ESS | — | 850.4 |
| Divergences | — | 8 |

New folds 41–42 cover 39 fixtures, 2026-08-01 to 2026-08-22. The other four arms are
**not** extended; `--all` does them for roughly 26 minutes of compute on 16 threads.

Preview before sampling:

```bash
julia --project -t 16 experiments/scottish_lower/02_negbin_2426_grid/r37_extend_negbin_2627.jl --preview --all
```

### 4.1 Trap: the DataStore cache default fails this runner

`load_datastore_cached(ds)` defaults to a 24-hour cache age, so an expired cache sends the
runner down the SQL path, which needs `BF_DB_URL` — absent from a non-interactive SSH
environment on `mcmc-beast`. The failure arrives *after* the package load, minutes in.
`r37` therefore passes the age explicitly and offers `--refresh` (which checks for `BF_DB_URL`
up front) when the new season's fixtures genuinely postdate the cache.

`r23_extend_poisson_2627.jl` has the same unguarded call and will fail the same way.

## 5. Note for cross-experiment readers

`m00_negbin_baseline` now holds 42 folds where every other model on the unified bench holds
40, so its row in `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md` is scored over a
slightly wider window. The report says so; the comparison script detects the mismatch rather
than assuming uniformity.
