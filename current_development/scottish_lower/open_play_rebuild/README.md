# Scottish Lower open-play rebuild

A clean-room rebuild of the Scottish Lower score-component model. Stages 1–6 now include the remotely validated data/features contract, pure equations, Turing/AD layer, and deterministic-chain OOS extraction/recombination. No sampling, experiment persistence, saved posterior chain, or leaderboard is implemented or reused.

The model reconstructs a score from three explicitly observed components:

1. non-penalty, non-own-goal (NP-NOG) goals;
2. penalty awards and their conversion; and
3. own goals.

V1 intentionally keeps only NP-NOG hierarchical: penalties use global Poisson awards plus shared Binomial conversion, and own goals use one global Poisson rate.

`DESIGN.md` is the implementation contract: data provenance and QA, mathematical model, posterior extraction, recombination, AD constraints, and staged validation are specified before code is written. `DATABASE_EVIDENCE.md` records bounded BBC/SofaScore reconciliation evidence without connection credentials.

## Scope and non-goals

- Train only on information available before each match's kickoff; evaluate held-out/OOS matches without leakage.
- Treat raw league IDs `56` and `57` as categorical source identifiers mapped deterministically to model leagues `1` and `2`.
- Preserve a global, stable canonical team-identity crosswalk across both leagues and all history, while each split's posterior columns contain history-seen teams only in canonical order. The stored name/ID-to-posterior-column map is authoritative; a target-only/unseen OOS team uses `α=β=0` population fallback rather than being dropped, remapped, or assigned an unlearned prior draw.
- Reconcile every component record to the official final score, including the side convention for own goals, before it may enter training.
- Keep all legacy `open_play`, `bug_fix`, experiments, chains, and artifacts untouched. This rebuild starts with no legacy leaderboard comparison or reuse.

## Planned notebook layout

Stage 2 is implemented as a read-only audit only. Stage 3's `l02_rebuild_features.jl` fetches the canonical registry only through an explicit read-only `BF_DB_URL` connection, validates and fingerprints it, and builds a pure history-only `FeatureSet` from an already-fetched registry. Stage 5 now supplies the thin `Features.create_features` dispatch specialized on the rebuild model type; generic feature behavior is unchanged. Neither stage contains sampling or default artifact output. `l01_rebuild_data_contract.jl` deduplicates provider incident IDs and derives NP-NOG separately for each own-goal convention as `official − converted penalties − own goals`; ordinary-goal incidents are reconciliation evidence, not the NP-NOG target. Missing-side/rescinded defects remain explicit quarantines. An alternative own-goal convention failing does not quarantine a match when the other validates; reconciliation quarantines occur only when neither validates. The returned report and `r01_audit_component_history.jl` also show informative own-goal evidence (own-goal matches with exactly one valid convention), without silently selecting a convention. The runner chooses a pooled 56/57 temporal boundary and audits **only its history IDs**, writing nothing by default. Remote results in [FINDINGS.md](FINDINGS.md) support the beneficiary convention 39–0 and quarantine two of 720 history matches.

## Notebook layout

| Stage | Loader (definitions only) | Runner (execution only) |
|---:|---|---|
| 1–2 | `l01_rebuild_data_contract.jl` | `r01_audit_component_history.jl` |
| 3 | `l02_rebuild_features.jl` | `r02_validate_maps_and_filtration.jl` |
| 4 | `l03_rebuild_equations.jl` | `r03_validate_equation_parity.jl` |
| 5 | `l04_rebuild_turing_model.jl` | `r04_profile_turing_gradients.jl` |
| 6 | `l05_rebuild_extraction_recombination.jl` | `r05_validate_oos_recombination.jl` |
| 7 | `l06_rebuild_sampling.jl` | `r06_remote_nuts_smoke.jl` (pending remote execution) |
| 8 | evaluation loader (not started) | OOS evaluation runner (not started) |

Stage 6 passed remotely against the cached DataStore/read-only registry: manifest-exact iteration×chain extraction, stored-map OOS identity fallback diagnostics, explicit three-Poisson convolution, adaptive support, and ordinary Predictions inference all passed using a deterministic synthetic multi-chain `Chains` fixture. It performed no writes or MCMC. Stage 5 adds only a model-owned `Features.create_features` adapter (it preserves generic feature behavior), validates the registry fingerprint before building, and passes a concrete array-only `equation_data(fs)` tuple into a branch-free, vectorized Turing likelihood. Its runner requires `BF_DB_URL` for the existing Stage-3 registry path and performs no writes or sampling; all DynamicPPL sampled-site, fresh/compiled ReverseDiff, ForwardDiff, finite-difference, saturation, timing, and allocation gates passed remotely, with a 0.633 ms median compiled gradient. Stage 3's runner requires `BF_DB_URL`, performs no writes by default, and checks leakage, reconciliation quarantine exclusion, stable 56→1/57→2 indexing, concrete vectors, and OOS identity fallback. Stage 4's runner reuses that exact registry/FeatureSet path and performs equation parity, smooth-saturation, thinning, and ForwardDiff checks; it contains no model or sampling. Loaders contain structs, pure builders, model/extraction/recombination helpers, and no sampling. Runners will use numbered, independently sendable REPL blocks and record observed results in a future `FINDINGS.md`.

Stage 7 is implemented but **pending remote execution**. It hard-requires `julia --project -t16`, pins threads, runs four independently prior-initialized queued NUTS chains at concurrency four, atomically checkpoints each chain and the combined manifest under `data/scottish_open_play_rebuild/`, and blocks OOS promotion on hard convergence failures. It does not run locally, create experiments/leaderboards, or rebuild OOS caches.

See [DESIGN.md](DESIGN.md) for the complete auditable design and acceptance gates.
