# Scottish Lower open-play rebuild

A **design-only** clean-room specification for a new Scottish Lower score-component model. No Julia model, feature builder, experiment, saved chain, or leaderboard is implemented or reused here.

The model reconstructs a score from three explicitly observed components:

1. non-penalty, non-own-goal (NP-NOG) goals;
2. penalty awards and their conversion; and
3. own goals.

V1 intentionally keeps only NP-NOG hierarchical: penalties use global Poisson awards plus shared Binomial conversion, and own goals use one global Poisson rate.

`DESIGN.md` is the implementation contract: data provenance and QA, mathematical model, posterior extraction, recombination, AD constraints, and staged validation are specified before code is written.

## Scope and non-goals

- Train only on information available before each match's kickoff; evaluate held-out/OOS matches without leakage.
- Treat raw league IDs `56` and `57` as categorical source identifiers mapped deterministically to model leagues `1` and `2`.
- Preserve a global, stable canonical team-identity crosswalk across both leagues and all history, while each split's posterior columns contain history-seen teams only in canonical order. The stored name/ID-to-posterior-column map is authoritative; a target-only/unseen OOS team uses `α=β=0` population fallback rather than being dropped, remapped, or assigned an unlearned prior draw.
- Reconcile every component record to the official final score, including the side convention for own goals, before it may enter training.
- Keep all legacy `open_play`, `bug_fix`, experiments, chains, and artifacts untouched. This rebuild starts with no legacy leaderboard comparison or reuse.

## Planned notebook layout

The implementation is intentionally deferred. Once Stage 1 is approved, files will be added as REPL-sendable pairs:

| Stage | Loader (definitions only) | Runner (execution only) |
|---:|---|---|
| 1–2 | `l01_rebuild_data_contract.jl` | `r01_audit_component_history.jl` |
| 3 | `l02_rebuild_features.jl` | `r02_validate_maps_and_filtration.jl` |
| 4–5 | `l03_open_play_rebuild_engine.jl` | `r03_smoke_equation_parity.jl` |
| 6 | `l04_rebuild_extraction_recombination.jl` | `r04_validate_oos_recombination.jl` |
| 7 | `l05_rebuild_experiment_config.jl` | `r05_remote_nuts_smoke.jl` |
| 8 | `l06_rebuild_evaluation.jl` | `r06_oos_evaluate_rebuild.jl` |

Loaders will contain structs, pure builders, model/extraction/recombination helpers, and no sampling. Runners will use numbered, independently sendable REPL blocks and record observed results in a future `FINDINGS.md`.

See [DESIGN.md](DESIGN.md) for the complete auditable design and acceptance gates.
