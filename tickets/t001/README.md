# T001 working notes — pooled tournament clock

Canonical brief: [`docs/tickets/T001-pooled-tournament-clock.md`](../../docs/tickets/T001-pooled-tournament-clock.md)

## Branch and baseline

- Branch: `fix/t001-pooled-tournament-clock`, based on `feat/scottish-lower-protocol` at
  `45798bb` (the ticket is not yet on `main`).
- Local checkout has no data cache suitable for the blast-radius measurement.
- Kaimon checkout: `/root/BayesianFootball`; it contains unrelated untracked research files,
  which must not be modified or cleaned.
- Reproducer: `tickets/t001/reproduce.jl`.
- Calendar-clock prototype: `tickets/t001/prototype.jl`.
- Kaimon comparison report: `tickets/t001/PROTOTYPE_REPORT.md`.
- End-to-end feature-time research: `tickets/t001/FEATURE_TIME_ALIGNMENT.md`.

## Confirmed code path

1. `process_data(::MatchesData)` creates tournament-local `match_week`, then derives
   `match_biweek` and the four-week `match_month` from it.
2. Both `_process_tournament_group_ids` (current relational API) and the older
   `_process_tournament_group` compare those values after pooling tournament rows.
3. `get_next_matches` repeats the same comparison to select `time_step + 1`.
4. `create_experiment_task` always uses `GroupedCVConfig`, including one-tournament segments.

The defect must therefore be fixed consistently in boundary construction, legacy split-view
construction, and next-match lookup. Changing only preprocessing would violate the scope guard.

## Design direction under investigation

Use an internal, group-scoped calendar index only when a `GroupedCVConfig` group contains more
than one tournament:

- anchor each `(group, season)` at the Sunday-ending week of its earliest fixture;
- calendar week = elapsed whole weeks from that anchor + 1;
- map `:match_week`, `:match_biweek`, and `:match_month` to fixed 1-, 2-, and 4-week bins;
- retain the stored column unchanged for `CVConfig` and one-member grouped configurations;
- select folds by the next *observed* calendar bin, so empty bins are skipped rather than
  renumbered;
- validate with `match_date + match_hour` that all fitted target-season fixtures precede the
  earliest held-out kickoff.

This preserves single-tournament folds exactly while giving pooled groups one fixed-width clock.
Unknown custom `dynamics_col` values should retain their existing behavior unless the final API
makes calendar clock selection explicit.

## Open decisions

- Unsupported custom `dynamics_col` behavior for multi-tournament groups: preserve only behind
  the strict kickoff assertion, or reject clearly.
- Whether the safety invariant should throw during boundary creation, next-match retrieval, or
  both. Boundary creation has enough information to validate dynamic target rows; next-match
  retrieval is the public point where the held-out set is materialized.
- Clarify acceptance criterion 3: calendar bins guarantee equal temporal width, not equal match
  counts (real leagues can legitimately schedule different numbers of fixtures).
- A separate `MultiScaleGRW` sampled-versus-reconstructed target-increment mismatch should be
  verified and ticketed, not folded into T001.

## Measurement log

Kaimon baseline and prototype comparison completed on all five pooled segments. Across the
shared seasons in the current caches, the incumbent had 295 contaminated transitions out of
453; the prototype had 0 out of 503, no empty folds, and every held-out biweek was shorter than
14 elapsed days. All 235 singleton control folds were exactly unchanged. See
`PROTOTYPE_REPORT.md` for the segment table and side-by-side 2024-10-19 fold.
