module RebuildFeatures

using BayesianFootball
using DataFrames
using Dates
using SHA
using LibPQ

include(joinpath(@__DIR__, "l01_rebuild_data_contract.jl"))
using .RebuildDataContract

export fetch_canonical_registry, validate_canonical_registry, registry_fingerprint,
       build_rebuild_feature_set, resolve_oos_identity

const LEAGUE_MAP = Dict(56 => 1, 57 => 2)
const SCHEMA_VERSION = "open_play_rebuild_features_v1"

"""Read the canonical SofaScore identity registry for *only* `match_ids`.

The connection URL is deliberately obtained solely from `ENV[\"BF_DB_URL\"]`.  This is a
read-only, parameterized query; callers own neither a connection nor any credentials.
"""
function fetch_canonical_registry(match_ids::AbstractVector{<:Integer}; timeout_ms::Int=15_000)
    haskey(ENV, "BF_DB_URL") || throw(ArgumentError("BF_DB_URL is required for canonical registry fetch"))
    ids = sort!(unique(Int.(match_ids))); isempty(ids) && return DataFrame()
    conn = BayesianFootball.Data.connect_to_db(BayesianFootball.Data.DBConfig(ENV["BF_DB_URL"]))
    try
        LibPQ.execute(conn, "BEGIN READ ONLY")
        LibPQ.execute(conn, "SET LOCAL statement_timeout = '$(Int(timeout_ms))ms'")
        # events.raw_data is the provider payload; matches supplies the authoritative finished match row.
        sql = """
        SELECT m.match_id, m.tournament_id,
          e.raw_data -> 'homeTeam' ->> 'id' AS home_id,
          e.raw_data -> 'homeTeam' ->> 'slug' AS home_slug,
          e.raw_data -> 'homeTeam' ->> 'name' AS home_name,
          e.raw_data -> 'awayTeam' ->> 'id' AS away_id,
          e.raw_data -> 'awayTeam' ->> 'slug' AS away_slug,
          e.raw_data -> 'awayTeam' ->> 'name' AS away_name
        FROM sofascore.events e JOIN sofascore.matches m USING (match_id)
        WHERE m.match_id = ANY(\$1) AND m.tournament_id = ANY(\$2)
        ORDER BY m.match_id
        """
        df = DataFrame(LibPQ.execute(conn, sql, [ids, [56, 57]]))
        LibPQ.execute(conn, "COMMIT")
        return df
    catch
        try LibPQ.execute(conn, "ROLLBACK") catch; end
        rethrow()
    finally
        LibPQ.close(conn)
    end
end

_text(x) = ismissing(x) ? "" : strip(String(x))
function _int(x)
    ismissing(x) && throw(ArgumentError("missing canonical team ID"))
    x isa Integer && return Int(x)
    try
        return parse(Int, strip(String(x)))
    catch
        throw(ArgumentError("non-numeric canonical team ID: $x"))
    end
end
# Diagnostic only: provider slugs may legitimately have historical spellings, so never use this to rewrite identities.
_slugish(name) = replace(replace(lowercase(strip(name)), r"[^a-z0-9]+" => "-"), r"(^-|-$)" => "")
function _digest(s::AbstractString) bytes2hex(sha256(codeunits(s))) end

"""Deterministic SHA256 over the row-per-match canonical registry."""
function registry_fingerprint(registry::DataFrame)
    rows = sort(collect(eachrow(registry)); by=r -> Int(r.match_id))
    payload = join([join((Int(r.match_id), Int(r.tournament_id), _int(r.home_id), _text(r.home_slug), _text(r.home_name), _int(r.away_id), _text(r.away_slug), _text(r.away_name)), "|") for r in rows], "\n")
    return _digest(payload)
end

"""Validate registry cardinality and aliases without normalizing conflicts away."""
function validate_canonical_registry(registry::DataFrame, requested_ids; ds=nothing)
    required = (:match_id,:tournament_id,:home_id,:home_slug,:home_name,:away_id,:away_slug,:away_name)
    all(c -> c in propertynames(registry), required) || throw(ArgumentError("registry missing required columns"))
    ids = sort!(unique(Int.(requested_ids)))
    counts = combine(groupby(registry, :match_id), nrow => :n)
    bad = Int.(counts.match_id[counts.n .!= 1]); missing_ids = sort!(collect(setdiff(ids, Set(Int.(registry.match_id)))))
    isempty(bad) || throw(ArgumentError("registry has non-unique match IDs: $bad")); isempty(missing_ids) || throw(ArgumentError("registry missing requested match IDs: $missing_ids"))
    nrow(registry) == length(ids) || throw(ArgumentError("registry contains unrequested match IDs"))
    all(Int(r.tournament_id) in keys(LEAGUE_MAP) for r in eachrow(registry)) || throw(ArgumentError("registry has non-56/57 tournament"))
    aliases = Dict{String,Int}(); id_aliases = Dict{Int,Set{String}}()
    for r in eachrow(registry), side in (:home,:away)
        id = _int(r[Symbol(side, "_id")]); slug = _text(r[Symbol(side, "_slug")]); name = _text(r[Symbol(side, "_name")])
        isempty(slug) && throw(ArgumentError("missing provider slug for match $(r.match_id)")); isempty(name) && throw(ArgumentError("missing provider name for match $(r.match_id)"))
        for alias in (slug, name)
            if haskey(aliases, alias) && aliases[alias] != id
                throw(ArgumentError("provider alias conflict '$alias': IDs $(aliases[alias]) and $id"))
            end
            aliases[alias] = id; push!(get!(id_aliases, id, Set{String}()), alias)
        end
    end
    # These are diagnostics, intentionally not a normalization or a validation criterion.
    name_slug_diagnostics = NamedTuple[]
    if ds !== nothing
        m = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
        for r in eachrow(registry), side in (:home,:away)
            haskey(m, Int(r.match_id)) || continue
            observed = _text(m[Int(r.match_id)][Symbol(side, "_team")]); provider = _text(r[Symbol(side, "_name")]); slug = _text(r[Symbol(side, "_slug")])
            name_match = observed == provider
            slug_name_match = _slugish(provider) == slug
            (name_match && slug_name_match) || push!(name_slug_diagnostics, (match_id=Int(r.match_id), side=side, datastore_name=observed, provider_name=provider, provider_slug=slug, datastore_name_matches_provider=name_match, provider_name_matches_slug=slug_name_match))
        end
    end
    manifest = (schema_version=SCHEMA_VERSION, registry_fingerprint=registry_fingerprint(registry), requested_match_ids=ids,
        registry_match_ids=sort!(Int.(registry.match_id)), alias_count=length(aliases), aliases=Dict(k => sort!(collect(v)) for (k,v) in id_aliases))
    return (registry=sort(registry, :match_id), aliases=aliases, canonical_aliases=id_aliases,
            name_slug_diagnostics=DataFrame(name_slug_diagnostics), manifest=manifest)
end

_date(r) = :match_date in propertynames(r) ? Date(r.match_date) : Date(r.start_timestamp)
function _registry_teams(registry)
    d = Dict{Int,Tuple{Int,Int}}()
    for r in eachrow(registry); d[Int(r.match_id)] = (_int(r.home_id), _int(r.away_id)); end
    d
end

"""Pure Stage-3 builder. It receives an already fetched registry and performs no DB I/O."""
function build_rebuild_feature_set(boundary, ds, registry::DataFrame; half_life_days::Real=365, own_goal_policy::Symbol=:beneficiary)
    half_life_days > 0 || throw(ArgumentError("half_life_days must be positive")); own_goal_policy == :beneficiary || throw(ArgumentError("only audited :beneficiary policy is allowed"))
    history_ids, target_ids = Int.(boundary.history_match_ids), Int.(boundary.target_match_ids)
    isempty(intersect(history_ids,target_ids)) || throw(ArgumentError("history/target leakage"))
    checked = validate_canonical_registry(registry, vcat(history_ids,target_ids); ds=ds)
    report = audit_component_history(ds, history_ids; target_match_ids=target_ids)
    matchrows = Dict(Int(r.match_id)=>r for r in eachrow(ds.matches))
    all(haskey(matchrows,id) for id in history_ids) || throw(ArgumentError("history IDs absent from DataStore matches"))
    target_dates = Date[_date(matchrows[id]) for id in target_ids if haskey(matchrows,id)]
    isempty(target_dates) && throw(ArgumentError("target dates required for cutoff")); cutoff = minimum(target_dates)
    teams = _registry_teams(checked.registry)
    included = sort!(Int[r.match_id for r in eachrow(report.ledger) if r.beneficiary_valid && isempty(r.quarantine_reasons)])
    quarantined = sort!(collect(setdiff(history_ids, included)))
    all(id in history_ids for id in included) || throw(ArgumentError("non-history outcome leakage"))
    team_ids = sort!(unique(vcat([collect(teams[id]) for id in included]...)))
    team_map = Dict(id => i for (i,id) in enumerate(team_ids))
    alias_to_column = Dict{String,Int}()
    for (alias,id) in checked.aliases
        haskey(team_map,id) && (alias_to_column[alias] = team_map[id])
    end
    # A provider alias must not point at two posterior columns (the validator catches ID conflicts).
    length(unique(values(alias_to_column))) <= length(team_ids) || error("impossible alias map conflict")
    rows = [report.ledger[findfirst(==(id), report.ledger.match_id), :] for id in included]
    hids = Int[teams[id][1] for id in included]; aids = Int[teams[id][2] for id in included]
    dates = Date[_date(matchrows[id]) for id in included]; days = Float64[Dates.value(cutoff-d) for d in dates]
    all(>=(0), days) || throw(ArgumentError("history match after cutoff")); weights = Float64[2.0^(-d/Float64(half_life_days)) for d in days]
    all(isfinite,weights) && all(w -> 0 < w <= 1,weights) || throw(ArgumentError("invalid weights"))
    league_ids = Int[Int(matchrows[id].tournament_id) for id in included]; all(haskey(LEAGUE_MAP,l) for l in league_ids) || throw(ArgumentError("unknown league"))
    data = Dict{Symbol,Any}(
        :schema_version=>SCHEMA_VERSION, :history_match_ids=>included, :quarantined_match_ids=>quarantined, :target_match_ids=>sort(target_ids), :cutoff_date=>cutoff,
        :team_ids=>team_ids, :n_teams=>length(team_ids), :team_map=>team_map, :alias_to_column=>alias_to_column, :canonical_id_by_alias=>checked.aliases,
        :home_team=>Int[team_map[x] for x in hids], :away_team=>Int[team_map[x] for x in aids], :Y_home=>Int[r.np_nog_Y_beneficiary_h for r in rows], :Y_away=>Int[r.np_nog_Y_beneficiary_a for r in rows],
        :A_home=>Int[r.penalty_A_h for r in rows], :A_away=>Int[r.penalty_A_a for r in rows], :C_home=>Int[r.penalty_C_h for r in rows], :C_away=>Int[r.penalty_C_a for r in rows], :O_home=>Int[r.own_goal_beneficiary_h for r in rows], :O_away=>Int[r.own_goal_beneficiary_a for r in rows],
        :month_ids=>Int[month(d) for d in dates], :league_ids=>Int[LEAGUE_MAP[l] for l in league_ids], :weights=>weights, :league_map=>copy(LEAGUE_MAP), :half_life_days=>Float64(half_life_days),
        :registry_fingerprint=>checked.manifest.registry_fingerprint, :registry_manifest=>checked.manifest, :audit_diagnostics=>report.diagnostics)
    return BayesianFootball.FeatureSet(data)
end

"""Resolve an OOS identity using only stored Stage-3 maps; this never reads outcomes."""
function resolve_oos_identity(fs; canonical_id=nothing, name=nothing, slug=nothing)
    supplied = String[x for x in (name,slug) if x !== nothing]
    alias_ids = unique(Int[fs[:canonical_id_by_alias][x] for x in supplied if haskey(fs[:canonical_id_by_alias],x)])
    id = canonical_id === nothing ? (length(alias_ids)==1 ? only(alias_ids) : nothing) : Int(canonical_id)
    !isempty(alias_ids) && id !== nothing && any(!=(id),alias_ids) && throw(ArgumentError("conflicting supplied canonical ID/name/slug"))
    length(alias_ids) > 1 && throw(ArgumentError("conflicting supplied name/slug"))
    id === nothing && return (column=0, status=:unknown_identity, canonical_id=nothing)
    haskey(fs[:team_map],id) && return (column=fs[:team_map][id], status=:history_seen, canonical_id=id)
    return (column=0, status=(id in values(fs[:canonical_id_by_alias]) ? :target_only_population_fallback : :unknown_identity), canonical_id=id)
end

end # module
