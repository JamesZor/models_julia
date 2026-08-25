#!/usr/bin/env julia

# T001 pooled-clock reproducer and acceptance probe.
#
# Run from the repository root:
#   julia --project tickets/t001/reproduce.jl
#   julia --project tickets/t001/reproduce.jl --all
#   julia --project tickets/t001/reproduce.jl --expect-bug
#
# The default checks the confirmed ScottishLower 24/25 case. `--all` measures every
# pooled segment over every season in its cache. Before the fix, `--expect-bug` requires
# the known contamination; after the fix, omit that flag and the script requires safety.

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

using BayesianFootball
using DataFrames
using Dates

const Data = BayesianFootball.Data
const EXPECT_BUG = "--expect-bug" in ARGS
const CHECK_ALL = "--all" in ARGS

kickoff(row) = DateTime(row.match_date) + Hour(row.match_hour)

function common_seasons(matches, tids)
    sets = [Set(String.(skipmissing(matches.season[matches.tournament_id .== tid]))) for tid in tids]
    return sort!(collect(reduce(intersect, sets)))
end

function inspect_segment(segment; seasons=nothing)
    ds = Data.load_datastore_cached(segment)
    tids = Data.tournament_ids(segment)
    target_seasons = isnothing(seasons) ? common_seasons(ds.matches, tids) : seasons
    config = Data.GroupedCVConfig(
        tournament_groups=[tids],
        target_seasons=target_seasons,
        history_seasons=0,
        dynamics_col=:match_biweek,
        warmup_period=0,
        stop_early=false,
    )

    boundaries = Data.create_id_boundaries(ds, config)
    by_id = Dict(Int(row.match_id) => kickoff(row) for row in eachrow(ds.matches))
    bad = NamedTuple[]
    spans = Day[]

    for (boundary, meta) in boundaries
        heldout = DataFrame(Data.get_next_matches(ds, (boundary, meta), config))
        isempty(heldout) && continue

        fitted_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
        isempty(fitted_ids) && continue
        first_heldout = minimum(kickoff(row) for row in eachrow(heldout))
        last_fitted = maximum(by_id[Int(id)] for id in fitted_ids)
        heldout_dates = Date.(kickoff.(eachrow(heldout)))
        push!(spans, maximum(heldout_dates) - minimum(heldout_dates))

        if last_fitted >= first_heldout
            push!(bad, (season=meta.target_season, fold=boundary.fold_id,
                        step=meta.time_step, last_fitted, first_heldout,
                        n_heldout=nrow(heldout)))
        end
    end

    result = (
        segment=string(nameof(typeof(segment))),
        tournaments=tids,
        seasons=target_seasons,
        folds=length(boundaries),
        evaluated=length(spans),
        contaminated=length(bad),
        max_heldout_span=isempty(spans) ? missing : maximum(spans),
        examples=first(bad, min(5, length(bad))),
    )
    display(result)
    return result
end

segments = CHECK_ALL ? Any[
    Data.ScottishLower(), Data.ScottishUpper(), Data.IrelandAll(),
    Data.SouthKorea(), Data.Norway(),
] : Any[Data.ScottishLower()]

results = [inspect_segment(segment; seasons=CHECK_ALL ? nothing : ["24/25"])
           for segment in segments]

if EXPECT_BUG
    @assert first(results).contaminated > 0 "expected the known 56/57 contamination"
else
    @assert all(r.contaminated == 0 for r in results) "fit/predict contamination remains"
end
