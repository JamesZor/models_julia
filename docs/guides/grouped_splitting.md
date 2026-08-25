# Grouped walk-forward splitting

This guide is the durable contract for `GroupedCVConfig`, `SplitBoundary`,
`create_id_boundaries`, `get_next_matches`, and splitter-aware feature construction.

## Why pooled groups need a shared clock

Stored `match_week`, `match_biweek`, and `match_month` values are dense-ranked separately for
each tournament. If one tournament plays an extra week, equal stored values stop representing
equal dates. They must not be used directly as a shared clock.

For a group containing multiple tournaments, the splitter derives calendar bins anchored at the
week containing that group's first fixture in each season:

| `dynamics_col` | Pooled window |
|---|---:|
| `:match_week` | 7 days |
| `:match_biweek` | 14 days |
| `:match_month` | 28 days |

The name `match_month` is retained for compatibility; in pooled splitting it means a fixed
four-week block, not a calendar month.

One-member groups and `CVConfig` continue using stored dynamics values exactly.

## Blank periods are labels, not jobs

Suppose observed pooled biweeks are `1, 2, 4`. Biweek 3 remains absent from the calendar rather
than causing biweek 4 to be renamed. No empty fold or empty model state is created:

```text
fold through 1 → predict 2
fold through 2 → predict 4
```

`get_next_matches` means “next observed effective block,” not always numeric `time_step + 1`.

## Boundary semantics

```julia
struct SplitBoundary
    fold_id::Int
    target_step::Int
    history_match_ids::Vector{Int}
    target_match_ids::Vector{Int}
end
```

Both ID vectors are fitted data:

- `history_match_ids`: frozen prior-season history;
- `target_match_ids`: expanding target-season observations through `target_step`.

The held-out fixtures are separate:

```julia
heldout = Data.get_next_matches(ds, (boundary, meta), config)
```

Never call `target_match_ids` the test set.

## Season-opening history fold

With `history_seasons = 1`, the first boundary is step zero:

```text
fit:     previous season only
predict: first observed target-season block
```

Later folds retain the same history and add observed target-season blocks.

With `history_seasons = 0`, no empty-training baseline is emitted. The first observed block is
fitted and the first prediction is the next observed block.

The history-only baseline is independent of `warmup_period`. Warmup filters target-season
training steps; `end_dynamics` bounds the training step and can still predict the next observed
block.

Pooled boundaries are prediction-bearing only, so terminal states with no next fixtures are not
emitted. `stop_early` is retained for compatibility but adds nothing for multi-tournament groups.
Singleton behavior is unchanged.

## Temporal safety invariant

Before a pooled boundary is emitted, the splitter requires:

```text
maximum(fitted kickoff) < minimum(held-out kickoff)
```

Kickoff combines `match_date` and `match_hour`; date-only checks are insufficient. Failure is an
error identifying the group, season, effective steps, and offending matches. This assertion is
deliberately redundant with the calendar design so exceptional source data cannot silently leak.

## Feature construction

Use the splitter-aware collection API:

```julia
boundaries = Data.create_id_boundaries(ds, config)
features = Features.create_features(boundaries, ds, model, config)
```

The IDs decide membership. Feature time is assigned row-by-row by match ID using the same
effective pooled clock. Raw observed bins are compressed to consecutive model indices:

```text
raw calendar bins: 1, 2, 4
model states:      1, 2, 3
```

This avoids empty latent states and makes feature construction invariant to DataFrame row order.
It does not change the modelling policy of GRW or other dynamics components. Standard OOS
extraction continues using the latest fitted state unless a model explicitly supplies another
policy.

The symbol-only overload remains for manual and singleton code, but grouped production paths
must pass the complete splitter object.

## Complete example

```julia
using BayesianFootball
const Data = BayesianFootball.Data
const Features = BayesianFootball.Features

segment = Data.ScottishLower()
ds = Data.load_datastore_cached(segment)

config = Data.GroupedCVConfig(
    tournament_groups=[Data.tournament_ids(segment)],
    target_seasons=["24/25"],
    history_seasons=2,
    dynamics_col=:match_biweek,
    warmup_period=0,
)

boundaries = Data.create_id_boundaries(ds, config)
boundary, meta = boundaries[1]
fitted_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
heldout = Data.get_next_matches(ds, (boundary, meta), config)
features = Features.create_features(boundaries, ds, model, config)
```

## Supported and unsupported configurations

Multi-tournament pooled groups support only `:match_week`, `:match_biweek`, and `:match_month`.
An unknown pooled `dynamics_col` fails clearly because the package cannot know whether it is
comparable across tournaments. Tournament IDs within a group must be unique.

## Testing and agent workflow

When changing splitting code, future contributors and AI agents must:

1. Read `docs/tickets/T001-pooled-tournament-clock.md` and this guide.
2. Preserve stored per-tournament preprocessing columns.
3. Exercise both relational boundaries and legacy split views.
4. Test `get_next_matches`; boundary IDs alone do not define the OOS card.
5. Pass the complete splitter into feature construction.
6. Assert row-order-invariant `match_id → time_index` mappings.
7. Keep singleton golden snapshots unchanged.
8. Test a blank pooled period and strict same-day kickoff ordering.
9. Run `julia --project -e 'using Pkg; Pkg.test()'`.
10. Measure every real pooled segment through Kaimon after pushing and pulling the branch.

The deterministic regressions live in `test/splitting_tests.jl`. T001 research artifacts and the
old-versus-new real-cache reports live in `tickets/t001/`.
