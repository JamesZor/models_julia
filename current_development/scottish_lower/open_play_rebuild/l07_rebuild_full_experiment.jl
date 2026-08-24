module RebuildFullExperiment

using BayesianFootball, DataFrames, Serialization, UUIDs, SHA, Dates
include(joinpath(@__DIR__, "l06_rebuild_sampling.jl"))
using .RebuildSampling

export boundary_sha256, registry_subset, atomic_replace_serialize, sanitized_error,
       checkpoint_valid, fold_inventory, true_oos_inventory, immutable_manifest,
       boundary_ids, inference_ids

"""Credential-free stable identity for a temporal boundary's training observations."""
function boundary_sha256(boundary)
    payload = join(vcat("H:" .* string.(sort(Int.(boundary.history_match_ids))),
                        "T:" .* string.(sort(Int.(boundary.target_match_ids)))), "\n")
    bytes2hex(sha256(codeunits(payload)))
end
boundary_ids(boundary) = sort!(unique(vcat(Int.(boundary.history_match_ids), Int.(boundary.target_match_ids))))
inference_ids(boundary, oos_ids) = sort!(unique(vcat(boundary_ids(boundary), Int.(oos_ids))))
_ids_sha256(ids) = bytes2hex(sha256(codeunits(join(string.(sort(Int.(ids))), "\n"))))

"""Take an exact registry snapshot; builders reject both missing and extra rows."""
function registry_subset(registry::DataFrame, ids::AbstractVector{<:Integer})
    wanted = sort!(unique(Int.(ids)))
    out = filter(:match_id => x -> Int(x) in Set(wanted), registry)
    nrow(out) == length(wanted) || throw(ArgumentError("registry snapshot lacks requested match IDs"))
    sort!(out, :match_id)
    Int.(out.match_id) == wanted || throw(ArgumentError("registry snapshot is not an exact requested-ID registry"))
    out
end
registry_subset(registry::DataFrame, boundary::BayesianFootball.Data.SplitBoundary) = registry_subset(registry, boundary_ids(boundary))

"""Return and validate genuine next-step OOS metadata for every `(boundary, meta)` fold."""
function true_oos_inventory(ds, boundaries, splitter)
    out = NamedTuple[]
    for (fold, fold_tuple) in enumerate(boundaries)
        boundary, meta = fold_tuple
        rows = DataFrame(BayesianFootball.Data.get_next_matches(ds, fold_tuple, splitter))
        nrow(rows) > 0 || throw(ArgumentError("fold $fold has no true next-step OOS rows"))
        required = (:match_id, :tournament_id, :season, splitter.dynamics_col)
        all(x -> x in propertynames(rows), required) || throw(ArgumentError("fold $fold OOS metadata lacks required columns"))
        ids = Int.(rows.match_id)
        length(unique(ids)) == length(ids) || throw(ArgumentError("fold $fold OOS IDs are duplicated"))
        observed_tournaments = Set(Int.(rows.tournament_id))
        expected_tournaments = Set(Int.(meta.tournament_ids))
        observed_tournaments ⊆ expected_tournaments || throw(ArgumentError("fold $fold OOS tournament misalignment"))
        all(==(meta.target_season), rows.season) || throw(ArgumentError("fold $fold OOS season misalignment"))
        all(==(meta.time_step + 1), rows[!, splitter.dynamics_col]) || throw(ArgumentError("fold $fold OOS prediction-step misalignment"))
        overlap = intersect(Set(ids), Set(boundary_ids(boundary)))
        isempty(overlap) || throw(ArgumentError("fold $fold true OOS overlaps boundary history/target: $(sort!(collect(overlap)))"))
        metadata_rows = select(rows, :match_id, :tournament_id, :home_team, :away_team, :match_date)
        push!(out, (; fold, boundary, meta, rows=metadata_rows, ids=sort!(ids), count=length(ids),
            ids_sha256=_ids_sha256(ids), prediction_step=meta.time_step + 1))
    end
    out
end

"""Replace mutable progress safely.  This is intentionally distinct from immutable atomic_serialize."""
function atomic_replace_serialize(path::AbstractString, value)
    mkpath(dirname(path)); tmp = path * ".tmp-" * string(uuid4())
    try
        serialize(tmp, value)
        mv(tmp, path; force=true)
    finally
        ispath(tmp) && rm(tmp; force=true)
    end
    path
end

function sanitized_error(err)
    msg = sprint(showerror, err)
    url = get(ENV, "BF_DB_URL", "")
    !isempty(url) && (msg = replace(msg, url => "[REDACTED_DB_URL]"))
    replace(msg, r"://[^\s/@]+:[^\s/@]+@" => "://[REDACTED]@")
end

function checkpoint_valid(path, boundary_hash::AbstractString, chain_id::Int, samples::Int, J::Int, validate_chain)
    isfile(path) || return nothing
    x = try deserialize(path) catch; return nothing end
    (x isa NamedTuple && get(x, :chain_id, nothing) == chain_id &&
     get(x, :boundary_sha256, nothing) == boundary_hash && get(x, :samples, nothing) == samples &&
     haskey(x, :chain)) || return nothing
    try
        validate_chain(x.chain, J)
        size(x.chain, 1) == samples || return nothing
        return x
    catch
        return nothing
    end
end

function fold_inventory(oos_folds)
    [(fold=x.fold, history_count=length(x.boundary.history_match_ids), target_count=length(x.boundary.target_match_ids),
      sha256=boundary_sha256(x.boundary), oos_count=x.count, oos_ids_sha256=x.ids_sha256,
      target_season=x.meta.target_season, target_time_step=x.meta.time_step, prediction_step=x.prediction_step,
      tournament_ids=sort(Int.(x.meta.tournament_ids))) for x in oos_folds]
end
immutable_manifest(; kwargs...) = (; schema_version=1, stage=8, created_utc=string(now(UTC)), kwargs...)

end
