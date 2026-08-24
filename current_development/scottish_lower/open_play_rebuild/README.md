# Scottish Lower open-play rebuild

A clean-room rebuild of the Scottish Lower score-component model. The mathematical design and Stage-2 history component audit are complete; no Turing model, experiment, saved chain, or leaderboard is implemented or reused yet.

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

Stage 2 is implemented as a read-only audit only. Stage 3 is also implemented (still no model): `l02_rebuild_features.jl` fetches the canonical registry only through an explicit read-only `BF_DB_URL` connection, validates and fingerprints it, and builds a pure history-only `FeatureSet` from an already-fetched registry. It never extends `Features.create_features`: after `l03_open_play_rebuild_engine.jl` defines a model type, a thin adapter is planned rather than a monkey patch. Neither stage contains sampling or default artifact output. `l01_rebuild_data_contract.jl` deduplicates provider incident IDs and derives NP-NOG separately for each own-goal convention as `official − converted penalties − own goals`; ordinary-goal incidents are reconciliation evidence, not the NP-NOG target. Missing-side/rescinded defects remain explicit quarantines. An alternative own-goal convention failing does not quarantine a match when the other validates; reconciliation quarantines occur only when neither validates. The returned report and `r01_audit_component_history.jl` also show informative own-goal evidence (own-goal matches with exactly one valid convention), without silently selecting a convention. The runner chooses a pooled 56/57 temporal boundary and audits **only its history IDs**, writing nothing by default. Remote results in [FINDINGS.md](FINDINGS.md) support the beneficiary convention 39–0 and quarantine two of 720 history matches.

## Notebook layout

| Stage | Loader (definitions only) | Runner (execution only) |
|---:|---|---|
| 1–2 | `l01_rebuild_data_contract.jl` | `r01_audit_component_history.jl` |
| 3 | `l02_rebuild_features.jl` | `r02_validate_maps_and_filtration.jl` |
| 4–5 | `l03_open_play_rebuild_engine.jl` | `r03_smoke_equation_parity.jl` |
| 6 | `l04_rebuild_extraction_recombination.jl` | `r04_validate_oos_recombination.jl` |
| 7 | `l05_rebuild_experiment_config.jl` | `r05_remote_nuts_smoke.jl` |
| 8 | `l06_rebuild_evaluation.jl` | `r06_oos_evaluate_rebuild.jl` |

Stage 3's runner requires `BF_DB_URL`, performs no writes by default, and checks leakage, reconciliation quarantine exclusion, stable 56→1/57→2 indexing, concrete vectors, and OOS identity fallback. Loaders contain structs, pure builders, model/extraction/recombination helpers, and no sampling. Runners will use numbered, independently sendable REPL blocks and record observed results in a future `FINDINGS.md`.

See [DESIGN.md](DESIGN.md) for the complete auditable design and acceptance gates.
