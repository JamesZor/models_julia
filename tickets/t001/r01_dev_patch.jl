# r01_dev_patch.jl — T001 package-interface monkey-patch validation
#
# WHAT THIS IS
# A no-MCMC integration test of the proposed shared pooled clock through the real package APIs:
# boundaries, get_next_matches, legacy split views, and Features.create_features.
#
# WHAT THIS IS NOT
# It does not edit src/, train a model, claim GRW effectiveness, or close T001.
#
# FILTRATION CONTRACT
# Every fitted kickoff must be strictly earlier than the earliest held-out kickoff. Calendar
# periods with no matches produce neither an empty fold nor an empty model state.
#
# USAGE (Kaimon checkout after git pull)
#   include("tickets/t001/r01_dev_patch.jl")
#   t001_dev_report

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

using BayesianFootball
using DataFrames
using Dates
using Random

const Data = BayesianFootball.Data
const Features = BayesianFootball.Features

include(joinpath(@__DIR__, "l01_dev_patch.jl"))
using .T001DevPatch

struct T001ClockProbe <: BayesianFootball.AbstractFootballModel end
Features.required_features(::T001ClockProbe) = Features.AbstractFeatureConfig[
    Features.TimeIndicesFeature(),
]

# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const T001_DYNAMICS_COL = :match_biweek
const T001_POOLED_SEGMENTS = Any[
    Data.ScottishLower(),
    Data.ScottishUpper(),
    Data.IrelandAll(),
    Data.SouthKorea(),
    Data.Norway(),
]
const T001_SINGLETON_SEGMENTS = Any[
    Data.Ireland(),
    Data.IrelandFirstDivision(),
    Data.Veikkausliiga(),
]

function t001_common_seasons(matches, tournament_ids)
    season_sets = [
        Set(String.(skipmissing(matches.season[matches.tournament_id .== tournament_id])))
        for tournament_id in tournament_ids
    ]
    return sort!(collect(reduce(intersect, season_sets)))
end

function t001_config(ds)
    tournament_ids = Data.tournament_ids(ds.segment)
    return Data.GroupedCVConfig(
        tournament_groups=[tournament_ids],
        target_seasons=t001_common_seasons(ds.matches, tournament_ids),
        history_seasons=0,
        dynamics_col=T001_DYNAMICS_COL,
        warmup_period=0,
        stop_early=false,
    )
end

# %%
# ===================================================================
# 3. Audit helpers
# ===================================================================

function t001_kickoff(row)
    return DateTime(row.match_date) + Hour(row.match_hour)
end

function t001_boundary_snapshot(ds, config)
    return [
        let next_df = DataFrame(Data.get_next_matches(ds, (boundary, meta), config))
            (
                fold_id=boundary.fold_id,
                target_step=boundary.target_step,
                history_ids=copy(boundary.history_match_ids),
                target_ids=copy(boundary.target_match_ids),
                meta_step=meta.time_step,
                next_ids=Int.(next_df.match_id),
            )
        end
        for (boundary, meta) in Data.create_id_boundaries(ds, config)
    ]
end

function t001_audit_segment(ds, config)
    boundaries = Data.create_id_boundaries(ds, config)
    kickoff_by_id = Dict(Int(row.match_id) => t001_kickoff(row) for row in eachrow(ds.matches))
    contaminated = 0
    empty = 0
    evaluated = 0
    maximum_span_hours = 0.0

    for (boundary, meta) in boundaries
        heldout = DataFrame(Data.get_next_matches(ds, (boundary, meta), config))
        if isempty(heldout)
            empty += 1
            continue
        end

        evaluated += 1
        heldout_kickoffs = t001_kickoff.(eachrow(heldout))
        span_hours = Dates.value(maximum(heldout_kickoffs) - minimum(heldout_kickoffs)) / 3_600_000
        maximum_span_hours = max(maximum_span_hours, span_hours)

        fitted_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
        if !isempty(fitted_ids)
            fitted_last = maximum(kickoff_by_id[Int(id)] for id in fitted_ids)
            contaminated += fitted_last >= minimum(heldout_kickoffs)
        end
    end

    return (
        segment=string(nameof(typeof(ds.segment))),
        folds=length(boundaries),
        evaluated=evaluated,
        empty=empty,
        contaminated=contaminated,
        maximum_span_hours=maximum_span_hours,
    )
end

function t001_feature_map(boundary, meta, ds, config)
    collection = Features.create_features(
        [(boundary, meta)], ds, T001ClockProbe(), config.dynamics_col)
    feature_set = collection[1][1]
    ids = feature_set.data[:ordered_match_ids]
    times = feature_set.data[:time_indices]
    return Dict(Int(id) => Int(time) for (id, time) in zip(ids, times)), feature_set
end

# %%
# ===================================================================
# 4. Load the cache snapshot and capture incumbent controls
# ===================================================================

pooled_stores = [Data.load_datastore_cached(segment) for segment in T001_POOLED_SEGMENTS]
singleton_stores = [Data.load_datastore_cached(segment) for segment in T001_SINGLETON_SEGMENTS]

incumbent_pooled = [t001_audit_segment(ds, t001_config(ds)) for ds in pooled_stores]
incumbent_singletons = [t001_boundary_snapshot(ds, t001_config(ds)) for ds in singleton_stores]

# %%
# ===================================================================
# 5. Install the package-interface patch
# ===================================================================

T001DevPatch.apply!()

# %%
# ===================================================================
# 6. Re-run public splitting and OOS APIs
# ===================================================================

patched_pooled = [t001_audit_segment(ds, t001_config(ds)) for ds in pooled_stores]
patched_singletons = [t001_boundary_snapshot(ds, t001_config(ds)) for ds in singleton_stores]

singleton_controls = [
    (
        segment=string(nameof(typeof(ds.segment))),
        folds=length(before),
        exactly_identical=before == after,
    )
    for (ds, before, after) in zip(
        singleton_stores, incumbent_singletons, patched_singletons)
]

# %%
# ===================================================================
# 7. Feature-time alignment and row-order gate
# ===================================================================

scottish_ds = first(pooled_stores)
scottish_config = Data.GroupedCVConfig(
    tournament_groups=[[56, 57]],
    target_seasons=["24/25"],
    history_seasons=2,
    dynamics_col=T001_DYNAMICS_COL,
    warmup_period=0,
    stop_early=false,
)
scottish_boundaries = Data.create_id_boundaries(scottish_ds, scottish_config)

# Choose the first fitted boundary that already contains the 2024-10-19 slate.
function contains_focus_date(boundary)
    ids = Set(boundary.target_match_ids)
    rows = subset(scottish_ds.matches, :match_id => ByRow(id -> Int(id) in ids))
    return Date("2024-10-19") in rows.match_date
end

focus_index = findfirst(pair -> contains_focus_date(first(pair)), scottish_boundaries)
focus_boundary, focus_meta = scottish_boundaries[focus_index]
focus_map, focus_features = t001_feature_map(
    focus_boundary, focus_meta, scottish_ds, scottish_config)
raw_steps = focus_features.data[:effective_target_steps]

states_per_raw_bin = Dict{Int,Set{Int}}()
for (match_id, raw_step) in raw_steps
    push!(get!(states_per_raw_bin, raw_step, Set{Int}()), focus_map[match_id])
end
same_bin_same_state = all(length(states) == 1 for states in values(states_per_raw_bin))
contiguous_states = sort(unique(values(focus_map))) == collect(1:focus_features.data[:n_rounds])

# Reorder the DataStore matches and prove the ID→state result is independent of physical rows.
rng = MersenneTwister(1001)
shuffled_matches = scottish_ds.matches[randperm(rng, nrow(scottish_ds.matches)), :]
shuffled_ds = Data.DataStore(
    scottish_ds.segment,
    shuffled_matches,
    scottish_ds.statistics,
    scottish_ds.odds,
    scottish_ds.lineups,
    scottish_ds.incidents,
    scottish_ds.betfair_odds,
    scottish_ds.bbc,
    scottish_ds.bbc_events,
)
shuffled_boundaries = Data.create_id_boundaries(shuffled_ds, scottish_config)
shuffled_index = findfirst(pair -> last(pair).time_step == focus_meta.time_step,
                           shuffled_boundaries)
shuffled_boundary, shuffled_meta = shuffled_boundaries[shuffled_index]
shuffled_map, _ = t001_feature_map(
    shuffled_boundary, shuffled_meta, shuffled_ds, scottish_config)
row_order_invariant = focus_map == shuffled_map

feature_gate = (
    fold=focus_boundary.fold_id,
    raw_step=focus_meta.time_step,
    matches=length(focus_map),
    same_bin_same_state=same_bin_same_state,
    contiguous_states=contiguous_states,
    row_order_invariant=row_order_invariant,
)

# %%
# ===================================================================
# 8. Final gates and report object
# ===================================================================

pooled_safe = all(row.contaminated == 0 for row in patched_pooled)
pooled_no_empty = all(row.empty == 0 for row in patched_pooled)
pooled_bounded = all(row.maximum_span_hours < 14 * 24 for row in patched_pooled)
singletons_identical = all(row.exactly_identical for row in singleton_controls)
features_aligned = same_bin_same_state && contiguous_states && row_order_invariant

@assert pooled_safe
@assert pooled_no_empty
@assert pooled_bounded
@assert singletons_identical
@assert features_aligned

t001_dev_report = (
    incumbent=incumbent_pooled,
    patched=patched_pooled,
    singleton_controls=singleton_controls,
    feature_gate=feature_gate,
    gates=(;
        pooled_safe,
        pooled_no_empty,
        pooled_bounded,
        singletons_identical,
        features_aligned,
    ),
)
