# T001 research — feature-time alignment

## Question

Can the splitter's `SplitBoundary` match IDs be used to keep model dynamics aligned with the
new shared calendar folds?

**Short answer:** the IDs should remain the authoritative membership list, but IDs alone do not
say which dynamic state a match belongs to. We also need one authoritative
`match_id → effective pooled step` lookup, shared by splitting, feature construction, and OOS
selection. We do **not** need to add empty model states or necessarily change `SplitBoundary`.

## End-to-end pipeline today

1. `Experiments.run_experiment` calls `Data.create_id_boundaries`.
2. Each `SplitBoundary` contains frozen history IDs and cumulative target-season IDs.
3. `Features.create_features` resolves those IDs back to `ds.matches` and builds all flat model
   vectors in `ordered_ids` order.
4. The feature builder independently creates `time_indices` from `:season` for history and the
   configured stored `dynamics_col` for target rows.
5. Standard dynamic team engines use each row's `time_indices[i]` to select
   `dyn.α[team, time]` and `dyn.β[team, time]` from `MultiScaleGRW`.
6. After training, `Experiments.extract_oos_predictions` reconstructs boundaries/features,
   calls `Data.get_next_matches`, and passes those rows to `extract_parameters`.
7. OOS standard dynamic engines deliberately default to `t_idx = n_rounds`: the next slate is
   priced from the latest fitted latent state, not a newly sampled random-walk state.
8. MatchDay also rebuilds boundaries and uses `get_next_matches` to select the matching trained
   fold, but currently calls `Features.create_features` with its default `:match_month` rather
   than the experiment splitter's explicit dynamics setting.

## Why match IDs are necessary but insufficient

The boundary IDs correctly answer:

> Which matches may this fold fit?

They do not answer:

> Which of those matches share one dynamic state?

For that, each target match needs a shared effective bin. Deriving it again from the stored
per-tournament `match_biweek` recreates T001 inside the model even if boundary membership is safe.
The calendar label can also contain gaps, while `MultiScaleGRW` requires contiguous array indices.

The useful contract is therefore:

```text
boundary IDs                 exact membership
match_id → raw effective bin shared calendar meaning
raw observed bins → 1:T      contiguous model-state indices
```

Example:

```text
observed calendar biweeks: 1, 2, 4
model target states:        1, 2, 3
```

Calendar biweek 3 remains absent and creates neither a fold nor a latent state. Dates/time-decay
features still retain the real elapsed interval.

## Additional defect found: current `time_indices` are positionally misaligned

`src/features/builder.jl` keeps `ordered_df` in DataStore order, but constructs `time_indices` by
iterating sorted groups and appending group-sized runs. It does not map those values back to the
group rows by match ID or row index.

Pooled matches are normally tournament-major, while the generated time vector is step-major.
Consequently, model rows can receive another match's dynamic state even before the proposed
calendar change.

Kaimon measurement on current ScottishLower fold 10, target `24/25`, stored biweek step 9:

| Block | Rows | Incorrect positional time indices |
|---|---:|---:|
| history (22/23 + 23/24) | 720 | 360 (50.0%) |
| target season | 172 | 149 (86.6%) |

Other flat features remain aligned because their extractors all iterate `ordered_ids` directly;
`time_indices` is the exception. This should be corrected as part of T001 integration because a
shared splitter clock without shared, row-wise feature assignment would be incomplete.

## Model impact

### Directly affected

Standard team-level dynamic engines using `MultiScaleGRW`, including goals/xG and market
variants. Their likelihood indexes a team-by-time latent matrix with `time_indices`.

### Indirectly affected

Time-decay and player-level models generally use static team/player coefficients rather than the
builder's dynamic indices, but their fitted match membership, date weights, fitted feature maps,
and OOS fixture selection still depend on safe boundaries.

### Preserve deliberately

OOS prediction currently uses the latest fitted state (`n_rounds`). T001 should not expand into a
new one-step random-walk forecasting policy. Raw calendar labels must never be passed directly as
latent-array indices.

## Recommended narrow architecture

### 1. One clock implementation in `Data`

Add internal helpers that, for `(matches, group IDs, season, config)`, provide:

- `effective_step_by_match_id`;
- sorted observed effective steps;
- next observed step after a fold step;
- fixed window width for supported pooled columns.

Behavior:

- multi-tournament `GroupedCVConfig` + `:match_week`/`:match_biweek`/`:match_month` uses the
  shared calendar clock;
- singleton groups and `CVConfig` retain the stored column exactly;
- unsupported custom pooled columns retain current behavior only if the kickoff assertion passes,
  or fail clearly rather than pretending they are calendar-safe.

The splitter, legacy split views, and `get_next_matches` must call these helpers rather than each
reimplementing clock logic.

### 2. Pass the splitter contract into feature construction

Prefer:

```julia
Features.create_features(splits, ds, model, splitter)
```

over passing only `splitter.dynamics_col`. The builder needs to know grouped-versus-singleton
semantics, and each tuple's metadata supplies tournament IDs and target season.

Update all callers:

- experiment runner;
- OOS post-processing;
- diagnostics;
- MatchDay (which currently falls back to the builder's default column).

A compatibility overload taking a symbol can remain for non-grouped/manual callers.

### 3. Assign time indices row-by-row

Keep boundary IDs as membership. For the resolved `ordered_ids`:

- map each history row's season to its sorted history-state index;
- map each target ID to its effective raw calendar bin;
- compress the observed target bins in that boundary to consecutive indices;
- offset target indices by `n_history_steps`;
- assign by each row's match ID, never by appending group counts.

Useful diagnostic keys can be retained in `FeatureSet`, such as ordered match IDs and raw effective
steps, but the Turing-facing `time_indices` must remain contiguous `1:n_rounds`.

### 4. Keep `SplitBoundary` structurally unchanged initially

Changing the struct would break manual constructors, including the scoped Scottish mitigation.
Because boundaries are reconstructed from the DataStore for training and prediction, the shared
Data helper can deterministically recover the mapping from `(IDs, metadata, config)`.

A richer future `SplitPlan` could carry the map explicitly, but is not required for T001.

### 5. Enforce invariants

For every fold:

- history and target IDs are unique and disjoint;
- all IDs resolve exactly once;
- all target time indices are row-aligned and contiguous;
- `maximum(time_indices) == n_rounds` for non-empty folds;
- matches in the same pooled calendar bin receive the same model state;
- every fitted kickoff is strictly earlier than the earliest held-out kickoff using
  `match_date + match_hour`;
- an absent calendar bin creates no boundary and no dynamic state.

## Tests required

1. Exact 56/57 2024-10-19 boundary and kickoff regression.
2. Same-calendar-bin fixtures from both tournaments receive the same model state.
3. Tournament-major and randomly shuffled DataFrames produce the same
   `match_id → time_index` map.
4. Two history seasons plus several target bins verify the history offset and contiguous target
   states.
5. A missing pooled calendar bin is skipped by boundaries, feature states, and OOS lookup.
6. OOS extraction continues to use `n_rounds`, with no raw-label/out-of-bounds indexing.
7. MatchDay and experiment post-processing choose the same fold/next fixtures.
8. Golden singleton boundaries, next-match IDs, and feature-time maps remain unchanged.
9. Legacy `create_data_splits` and relational boundaries represent the same fitted IDs.

## Related issue not to fold into T001

`MultiScaleGRW` training appears to sample `n_target` target increments while reconstruction uses
`n_target - 1`; the final sampled increment may be unused. This deserves a separate ticket after
verification, rather than enlarging T001.

## Conclusion

We can and should use the boundary match IDs, but as join keys—not as the clock itself. The safest
T001 production design is:

```text
shared Data clock → safe boundary IDs → row-wise ID/clock join → contiguous model states
                  ↘ same next-observed-bin lookup for OOS and MatchDay
```

This fixes fold leakage, prevents feature-time drift, avoids empty dynamics, and preserves the
current latest-state OOS modeling policy.
