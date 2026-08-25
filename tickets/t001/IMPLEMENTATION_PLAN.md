# T001 implementation plan

## Goal

Remove train/predict leakage for multi-tournament pooled folds by giving the group one fixed
calendar clock, while preserving singleton behavior and avoiding empty folds or empty model
states.

## Scope

### In scope

- Shared calendar steps for multi-tournament `GroupedCVConfig` using `:match_week`,
  `:match_biweek`, or `:match_month`.
- Relational `SplitBoundary` creation and legacy split-view creation.
- Next-observed-fixture selection in `get_next_matches`.
- Strict fitted-versus-held-out kickoff validation using date plus hour.
- Row-wise, match-ID-aligned feature time indices.
- Consistent splitter configuration in experiment training, OOS reconstruction, diagnostics,
  and MatchDay.
- Synthetic regression tests, real-cache Kaimon measurement, and ticket resolution report.

### Out of scope

- Changing stored per-tournament `match_week`/`match_biweek`/`match_month` columns.
- Redesigning, tuning, or promoting `MultiScaleGRW` dynamics.
- Changing OOS prediction from the latest fitted latent state to a new forecast state.
- Rebuilding leaderboards/backtests or modifying the Scottish local mitigation.
- Fixing the possible unused final `MultiScaleGRW` increment; verify and raise separately.

## Expected production changes

### 1. Central effective-clock helper (`src/Data/splitting/`)

Add one implementation used by every split consumer. Given matches, group IDs, season, and
splitter configuration, it will provide:

- `match_id → raw effective step`;
- sorted observed steps;
- next observed step;
- configured fixed width (7/14/28 days).

For pooled groups, raw steps are calendar-anchored. For `CVConfig` and one-member grouped configs,
the helper returns the existing stored dynamics values exactly.

No match preprocessing or DataStore schema changes are planned.

### 2. Split construction (`src/Data/splitting/methods.jl`)

Update both `_process_tournament_group_ids` and `_process_tournament_group` to use the helper.
Blank calendar periods retain their labels but generate no fold. Boundaries continue to contain
only match IDs; avoid changing `SplitBoundary` so manual constructors remain compatible.

Before emitting a predictive fold, validate:

```text
maximum(fitted kickoff) < minimum(held-out kickoff)
```

Errors will identify group, season, steps, and offending match IDs.

### 3. OOS retrieval (`get_next_matches`)

Replace pooled `meta.time_step + 1` with “next observed effective step.” Singleton paths retain
current exact-step behavior. OOS fixture IDs must agree with those used for the boundary safety
check.

### 4. Feature-time alignment (`src/features/builder.jl`)

Add a splitter-aware feature API, preferably:

```julia
Features.create_features(splits, ds, model, splitter)
```

Use boundary IDs for membership and assign time indices row-by-row:

- history rows map by season;
- target IDs map to the shared effective raw step;
- observed target steps are compressed to consecutive model indices;
- target indices are offset by the number of history states.

Example: raw calendar bins `1, 2, 4` become model target states `1, 2, 3`. This creates no empty
GRW state. This is data correctness and future-proofing, not a GRW redesign.

Keep a compatibility overload for existing manual/single-column callers if needed.

### 5. Call-site consistency

Pass the complete splitter contract from:

- `src/experiments/runner.jl`;
- `src/experiments/post_processing.jl`;
- `src/experiments/diagnostics/extraction.jl`;
- `src/MatchDay/inference.jl`.

This removes MatchDay's current implicit `:match_month` fallback when the trained experiment uses
another dynamics column.

### 6. Tests

Add focused splitter tests, likely `test/splitting_tests.jl`, and include them from
`test/runtests.jl`.

Required cases:

1. Synthetic 56/57 `24/25`, including 2024-10-19 14:00/16:00.
2. Strict kickoff invariant for every generated synthetic pooled fold.
3. Fixed weekly/biweekly/four-week window bounds.
4. A wholly blank pooled calendar bin: no empty fold and next observed bin selected.
5. Same shared-bin matches receive the same model time state.
6. Feature mapping is invariant to tournament-major and shuffled DataFrame row order.
7. Multiple history seasons plus target-state offset and contiguity.
8. Relational boundaries and legacy split views contain equivalent fitted IDs.
9. Exact singleton golden folds, next-match IDs, and feature mappings for IDs 79, 718, and 31.
10. OOS keeps using the latest fitted state and MatchDay selects the same fold as post-processing.
11. Clear errors for unresolved/duplicate/overlapping IDs and unsafe kickoff ordering where
    applicable.

## Implementation sequence

### Phase A — lock behavior with tests

- Add synthetic fixtures and failing regression tests first.
- Snapshot singleton behavior before production changes.
- Explicitly settle `stop_early` and `end_dynamics` semantics for non-contiguous raw steps.

### Phase B — shared clock and split APIs

- Add the central helper.
- Convert ID boundaries, legacy views, and next-match retrieval.
- Add kickoff assertions.
- Run focused Data/splitting tests.

### Phase C — feature integration

- Add splitter-aware feature creation and row-wise mapping.
- Update experiment/OOS/diagnostic/MatchDay callers.
- Run feature, experiment, and MatchDay focused tests.

### Phase D — remote validation

Push local commits and fast-forward the Kaimon checkout. Then:

1. rerun the exact Scottish comparison;
2. rerun all five pooled segments;
3. verify zero contamination and bounded windows;
4. verify the three singleton controls are identical;
5. run the Scottish local gate and confirm its trim drops zero matches;
6. run the full package test suite.

### Phase E — close ticket

- Record production results under `## Resolution` in the canonical ticket.
- Update ticket status/registry only after all acceptance criteria pass.
- Note pooled fold-count/runtime changes and invalidate comparison with old pooled experiments.

## Expected behavioral changes

### Pooled segments

- Fold membership and count will change.
- Held-out blocks become fixed calendar windows.
- Empty calendar bins are skipped.
- No fitted kickoff may overlap the held-out slate.
- Experiment runtime may rise because the prototype found about 11% more occupied pooled biweeks.
- Historical pooled experiment results become incomparable, as already allowed by T001.

### Singleton segments

- Stored clock, boundaries, OOS matches, and feature-time maps remain unchanged.

### Models

- Every model benefits from leak-free boundaries.
- Only models consuming dynamic `time_indices` are directly affected by the row-alignment fix.
- GRW behavior, priors, and OOS forecasting policy do not otherwise change.

## Decision gate before coding

Confirm how unsupported custom pooled `dynamics_col` values should behave. Recommended default:
reject them with a clear message unless they are explicitly declared group-comparable; silently
pooling an unknown per-tournament index would recreate the original defect class.
