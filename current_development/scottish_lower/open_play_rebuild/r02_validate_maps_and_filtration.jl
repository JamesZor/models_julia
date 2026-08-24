# Stage 3 notebook: registry validation and pure feature filtration.  No model, MCMC, or default writes.

# %% BLOCK 1 — imports and representative pooled boundary
using BayesianFootball
using DataFrames
using Dates
const BFData = BayesianFootball.Data
include(joinpath(@__DIR__, "l02_rebuild_features.jl"))
using .RebuildFeatures

ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
splitter = BFData.GroupedCVConfig(tournament_groups=[[56,57]], target_seasons=["24/25","25/26"],
    history_seasons=2, dynamics_col=:match_biweek, warmup_period=0, stop_early=true)
boundaries = BFData.create_id_boundaries(ds, splitter)
by_match = Dict(Int(r.match_id)=>Int(r.tournament_id) for r in eachrow(ds.matches))
fold = findlast(i -> Set(get(by_match, Int(id), -1) for id in first(boundaries[i]).history_match_ids) >= Set([56,57]), eachindex(boundaries))
isnothing(fold) && error("No pooled 56/57 history boundary")
boundary = first(boundaries[fold])
println("Stage 3 boundary $fold: $(length(boundary.history_match_ids)) history / $(length(boundary.target_match_ids)) target")

# %% BLOCK 2 — read-only canonical registry fetch
haskey(ENV, "BF_DB_URL") || error("BF_DB_URL is required: Stage 3 will not invent credentials or query without it.")
requested = vcat(Int.(boundary.history_match_ids), Int.(boundary.target_match_ids))
registry = fetch_canonical_registry(requested)
checked = validate_canonical_registry(registry, requested; ds=ds)
println("Registry SHA256: $(checked.manifest.registry_fingerprint); aliases: $(checked.manifest.alias_count)")
println("Name/slug diagnostics (display-only; no normalization):")
display(checked.name_slug_diagnostics)

# %% BLOCK 3 — pure builder and filtration assertions
fs = build_rebuild_feature_set(boundary, ds, registry; half_life_days=365, own_goal_policy=:beneficiary)
history, target, included, quarantined = Set(Int.(boundary.history_match_ids)), Set(Int.(boundary.target_match_ids)), Set(fs[:history_match_ids]), Set(fs[:quarantined_match_ids])
@assert isempty(intersect(history,target))
@assert included ⊆ history && isempty(intersect(included,target))
@assert quarantined ⊆ history && isempty(intersect(quarantined,included))
@assert issorted(fs[:team_ids]) && sort!(collect(values(fs[:team_map]))) == collect(1:fs[:n_teams])
@assert fs[:league_map] == Dict(56=>1,57=>2) && Set(fs[:league_ids]) == Set([1,2])
@assert all(isfinite, fs[:weights]) && all(w -> 0 < w <= 1, fs[:weights])
@assert all(x -> x isa Vector{Int}, [fs[k] for k in (:home_team,:away_team,:Y_home,:Y_away,:A_home,:A_away,:C_home,:C_away,:O_home,:O_away,:month_ids,:league_ids)])
@assert fs[:weights] isa Vector{Float64} && all(!ismissing, fs[:weights])
@assert all(id -> id in fs[:history_match_ids], fs[:history_match_ids]) # outcomes are history-only by construction
println("Included $(length(included)); quarantined $(length(quarantined)); posterior teams $(fs[:n_teams]).")

# %% BLOCK 4 — OOS identity resolver tests (no target outcomes accessed)
known_id = first(fs[:team_ids]); known = resolve_oos_identity(fs; canonical_id=known_id)
@assert known.status == :history_seen && known.column > 0
target_registry_ids = setdiff(unique(parse.(Int, vcat(String.(registry.home_id), String.(registry.away_id)))), fs[:team_ids])
if !isempty(target_registry_ids)
    fallback = resolve_oos_identity(fs; canonical_id=first(target_registry_ids))
    @assert fallback.status == :target_only_population_fallback && fallback.column == 0
    println("Target-only fallback checked for canonical ID $(fallback.canonical_id).")
else
    println("Representative boundary has no target-only team; checking synthetic canonical target-only fallback.")
end
# Local synthetic identity only: it exercises the stored-map fallback path without data/DB access.
synthetic_data = copy(fs.data)
synthetic_data[:canonical_id_by_alias] = copy(fs[:canonical_id_by_alias])
synthetic_data[:canonical_id_by_alias]["__synthetic_target_only__"] = -999_999
synthetic_fs = BayesianFootball.FeatureSet(synthetic_data)
synthetic_fallback = resolve_oos_identity(synthetic_fs; name="__synthetic_target_only__")
@assert synthetic_fallback.status == :target_only_population_fallback && synthetic_fallback.column == 0
unknown = resolve_oos_identity(fs; name="__synthetic_unknown__")
@assert unknown.status == :unknown_identity && unknown.column == 0
other_id = first(filter(!=(known_id), fs[:team_ids]))
known_alias = first(k for (k,v) in fs[:canonical_id_by_alias] if v == known_id)
try
    resolve_oos_identity(fs; canonical_id=other_id, name=known_alias)
    error("synthetic ID/name conflict was accepted")
catch e
    e isa ArgumentError || rethrow()
end
println("Known, target-only/unknown, and conflict resolver checks passed.")

# %% BLOCK 5 — MANUAL snapshot export (disabled by default; no writes above)
if false
    # Deliberately opt-in: serialize registry + fs[:registry_manifest] only after review.
    # using Serialization; serialize("stage3_registry_snapshot.jls", (registry=registry, manifest=fs[:registry_manifest]))
end
