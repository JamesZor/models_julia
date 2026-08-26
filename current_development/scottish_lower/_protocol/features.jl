# ==============================================================================
# Scottish Lower protocol — GATES 1–2
# ==============================================================================
#
# Model-specific construction and assertions are adapter hooks.  Fold population,
# kickoff safety, feature perturbation, and feature invariants are protocol-wide.
#
# ============================================================================

# ==============================================================================
# GATE 1 — Config
# ==============================================================================

"""
    sl_gate_config(adapter, contract)

Nothing about the configuration is hidden, and everything it declares resolves.
"""
function sl_gate_config(adapter::AbstractSLModelAdapter, contract::SLContract)
    results = []
    model = sl_model(adapter)

    required = sl_required_features(adapter)
    push!(results, sl_result(
        "required features declared",
        !isempty(required),
        join(string.(required), ", "),
    ))

    h = sl_hash(model)
    push!(results, sl_result(
        "config hashes deterministically",
        h == sl_hash(model) && length(h) == 8,
        "hash = $h",
    ))

    # A posterior layout and all model-specific configuration assertions belong
    # to the adapter; keeping them here would make a shared protocol a disguised
    # team-Poisson model.
    schema = sl_posterior_schema(adapter)
    grouped_ok = !isempty(schema.varinfo_sites)
    columns_callable = schema.chain_columns isa Function
    push!(results, sl_result(
        "posterior schema declared",
        grouped_ok && columns_callable && schema.parameter_count isa Function,
        "$(length(schema.varinfo_sites)) grouped sites; expanded columns and parameter count declared as n_teams functions",
    ))

    append!(results, sl_adapter_check(adapter, :config, model, contract))
    return results
end


# ==============================================================================
# GATE 2 — Features
# ==============================================================================

"""
    sl_truncate_datastore(ds, keep_ids) -> DataStore

A copy of the store containing only `keep_ids`, for the perturbation test.
Every domain carrying a `match_id` is filtered; all 9 fields are passed through
so the BBC domains are not silently dropped (see Data.DataStore docstring).
"""
function sl_truncate_datastore(ds, keep_ids::AbstractVector{Int})
    keep = Set(keep_ids)
    trim(df) = (nrow(df) > 0 && "match_id" in names(df)) ?
               DataFrame(DataFrames.subset(df, :match_id => ByRow(id -> id in keep))) : df

    return SLData.DataStore(
        ds.segment,
        trim(ds.matches),
        trim(ds.statistics),
        trim(ds.odds),
        trim(ds.lineups),
        trim(ds.incidents),
        trim(ds.betfair_odds),
        trim(ds.bbc),
        trim(ds.bbc_events),
    )
end

"""
    sl_featureset_equal(a, b) -> (Bool, Vector{Symbol})

Structural comparison of two FeatureSets. Returns whether they match and the
keys that differ.

`player_ratings_map` is intentionally compared only for each FeatureSet's
`ordered_match_ids`. RAPM projects the fixed historical ratings onto teamsheets for
future OOS fixtures, so its whole-store map legitimately changes when those fixtures
are removed by the perturbation test. Entries for fitted fold matches must still be
identical.
"""
function sl_featureset_equal(a, b)
    ka = Set(keys(a.data))
    kb = Set(keys(b.data))
    differing = Symbol[]

    for k in union(ka, kb)
        if !(k in ka) || !(k in kb)
            push!(differing, k)
        elseif k === :player_ratings_map
            ids_a = Set(Int.(a.data[:ordered_match_ids]))
            ids_b = Set(Int.(b.data[:ordered_match_ids]))
            map_a = Dict(id => ratings for (id, ratings) in a.data[k] if id in ids_a)
            map_b = Dict(id => ratings for (id, ratings) in b.data[k] if id in ids_b)
            isequal(map_a, map_b) || push!(differing, k)
        elseif !isequal(a.data[k], b.data[k])
            push!(differing, k)
        end
    end
    return (isempty(differing), sort(differing))
end

"""
    sl_gate_features(ds, folds, adapter, contract; perturb_fold = 1)

The anti-leakage gate.

The perturbation test is deliberately run FIRST: feature extractors must depend
only on explicitly listed match IDs. This is trivial for lookup-only models, but
is the check that catches a global fit leaking future data into historical folds.
"""
function sl_gate_features(ds, folds::Vector{SLFold}, adapter::AbstractSLModelAdapter,
                          contract::SLContract; perturb_fold::Int = 1)
    # Do not turn an empty population into a vacuous pass, or let indexing below
    # hide the actual fault.
    isempty(folds) && return ([
        sl_result("kickoff filtration holds", false, "no folds available"),
        sl_result("kickoff filtration drops", false, "no folds available"),
        sl_result("perturbation (future cannot alter past)", false, "no folds available"),
        sl_result("type purity (no missing / NaN)", false, "no folds available"),
        sl_result("team_map keyed by NAME", false, "no folds available"),
        sl_result("contiguous model time states", false, "no folds available"),
        sl_result("OOS team coverage", false, "no folds available"),
    ], Any[])

    results = []
    model = sl_model(adapter)
    splitter = sl_splitter(contract)

    # --- build every fold's FeatureSet ---------------------------------------
    # Splitter-aware collection API: the complete splitter must be passed so feature
    # time is assigned on the same pooled effective clock the boundaries were cut on,
    # and observed calendar bins are compressed to consecutive model states.
    # The symbol-only overload skips that alignment — see docs/guides/grouped_splitting.md.
    sl_splits = [(f.boundary, f.meta) for f in folds]
    sl_collection = BayesianFootball.Features.create_features(sl_splits, ds, model, splitter)
    feature_sets = [fs for (fs, _) in sl_collection]

    # --- (a) kickoff filtration ----------------------------------------------
    kickoff = sl_kickoff_map(ds)
    violations = Int[]
    for f in folds
        if nrow(f.oos_df) == 0
            push!(violations, f.idx)
            continue
        end
        oos_kickoffs = [get(kickoff, Int(id), nothing) for id in f.oos_df.match_id]
        if any(isnothing, oos_kickoffs)
            push!(violations, f.idx)
            continue
        end
        cutoff = minimum(something.(oos_kickoffs))
        any(!haskey(kickoff, id) || kickoff[id] >= cutoff for id in f.fitted_ids) &&
            push!(violations, f.idx)
    end
    push!(results, sl_result(
        "kickoff filtration holds",
        isempty(violations),
        isempty(violations) ?
        "every fitted kickoff strictly precedes its fold's first OOS kickoff" :
        "VIOLATED in folds $violations",
    ))

    dropped_by_fold = [(f.idx, length(f.dropped_ids)) for f in folds if !isempty(f.dropped_ids)]
    n_dropped = sum(length(f.dropped_ids) for f in folds; init = 0)
    push!(results, sl_result(
        "kickoff filtration drops",
        true, # reported, not enforced — a drop is correct behaviour
        n_dropped == 0 ? "nothing dropped" :
        "$n_dropped observations removed as not-yet-played: $dropped_by_fold",
    ))

    # --- (b) perturbation ----------------------------------------------------
    if !(1 <= perturb_fold <= length(folds))
        push!(results, sl_result(
            "perturbation (future cannot alter past)", false,
            "requested fold $perturb_fold; valid folds are 1:$(length(folds))",
        ))
    else
        f = folds[perturb_fold]
        ds_trunc = sl_truncate_datastore(ds, vcat(f.fitted_ids, Int.(f.oos_df.match_id)))
        fc_trunc = BayesianFootball.Features.create_features([(f.boundary, f.meta)], ds_trunc, model, splitter)
        fs_trunc = fc_trunc[1][1]
        same, differing = sl_featureset_equal(feature_sets[perturb_fold], fs_trunc)
        push!(results, sl_result(
            "perturbation (future cannot alter past)", same,
            same ? "fold $(f.idx) FeatureSet bit-identical with future matches removed" :
                   "DIFFERS on keys: $differing",
        ))
    end

    # --- (c) type purity ------------------------------------------------------
    impure = Tuple{Int, Symbol, String}[]
    for (i, fs) in enumerate(feature_sets)
        for (k, v) in fs.data
            v isa AbstractVector || continue
            # `AbstractVector{<:Real}` excludes `Vector{Union{Missing, Float64}}`.
            # Treat that as numeric too, so missingness can never evade this gate.
            numeric = eltype(v) <: Real ||
                      (Missing <: eltype(v) && all(x -> x isa Real, skipmissing(v)))
            numeric || continue
            if Missing <: eltype(v) || any(ismissing, v)
                push!(impure, (i, k, "missing"))
            elseif any(x -> x isa AbstractFloat && isnan(x), v)
                push!(impure, (i, k, "NaN"))
            end
        end
    end
    push!(results, sl_result(
        "type purity (no missing / NaN)",
        isempty(impure),
        isempty(impure) ? "all numeric vectors clean across $(length(feature_sets)) folds" :
                          "first offender: $(impure[1])",
    ))

    # --- (d) team map identity ------------------------------------------------
    bad_maps = Int[]
    for (i, fs) in enumerate(feature_sets)
        tm = get(fs.data, :team_map, nothing)
        (tm isa AbstractDict && keytype(tm) <: AbstractString) || push!(bad_maps, i)
    end
    push!(results, sl_result(
        "team_map keyed by NAME",
        isempty(bad_maps),
        isempty(bad_maps) ?
        "String-keyed in all folds, matching what extract_parameters looks up" :
        "wrong key type in folds $bad_maps",
    ))

    # --- (e) contiguous model time states -------------------------------------
    ragged = Int[]
    for (i, fs) in enumerate(feature_sets)
        haskey(fs.data, :time_indices) || continue
        ti = sort(unique(Vector{Int}(fs.data[:time_indices])))
        ti == collect(1:length(ti)) || push!(ragged, i)
    end
    push!(results, sl_result(
        "contiguous model time states",
        isempty(ragged),
        isempty(ragged) ?
        "time_indices are 1..K with no gaps in all $(length(feature_sets)) folds" :
        "gaps present in folds $ragged",
    ))

    # --- (f) OOS team coverage ------------------------------------------------
    unmapped = Tuple{Int, String}[]
    for (i, fs) in enumerate(feature_sets)
        tm = get(fs.data, :team_map, nothing)
        tm isa AbstractDict || continue
        for row in eachrow(folds[i].oos_df)
            ismissing(row.home_team) || haskey(tm, row.home_team) || push!(unmapped, (i, String(row.home_team)))
            ismissing(row.away_team) || haskey(tm, row.away_team) || push!(unmapped, (i, String(row.away_team)))
        end
    end
    n_sides = sum(2 * nrow(f.oos_df) for f in folds)
    push!(results, sl_result(
        "OOS team coverage",
        true, # reported, not enforced — promoted sides are legitimate
        "$(length(unmapped)) / $n_sides sides unmapped" *
        (isempty(unmapped) ? "" : " → population fallback: $(unique([u[2] for u in unmapped]))"),
    ))

    return (results, feature_sets)
end
