# ==============================================================================
# Model 00 — GATE IMPLEMENTATIONS (stages ⓪–②)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Covers Gates 0 to 2 for Model 00 (Pure Poisson):
#   Gate 0: Contract (data fingerprint, fold inventory, sealed season protection)
#   Gate 1: Config (parameters, required features, deterministic hash)
#   Gate 2: Features (kickoff filtration, perturbation, type purity, map identity)
#
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics

const Data = BayesianFootball.Data


# ==============================================================================
# 1. Fold Inventory
# ==============================================================================

struct TP00Fold
    idx::Int
    step::Int
    season::String
    boundary::Data.SplitBoundary
    meta::Any
    fitted_ids::Vector{Int}
    dropped_ids::Vector{Int}
    oos_df::DataFrame
end

function tp00_kickoff_map(ds)
    m = ds.matches
    return Dict{Int, DateTime}(
        Int(row.match_id) => DateTime(row.match_date) + Hour(row.match_hour)
        for row in eachrow(m)
    )
end

function tp00_build_folds(ds, contract::SLContract)
    sl_assert_not_sealed(contract, contract.dev_seasons)

    splitter   = sl_splitter(contract)
    boundaries = Data.create_id_boundaries(ds, splitter)
    kickoff    = tp00_kickoff_map(ds)

    folds = TP00Fold[]
    for (i, (boundary, meta)) in enumerate(boundaries)
        oos_df = DataFrame(Data.get_next_matches(ds, (boundary, meta), splitter))

        if nrow(oos_df) == 0
            push!(folds, TP00Fold(i, meta.time_step, meta.target_season, boundary, meta,
                                  vcat(boundary.history_match_ids, boundary.target_match_ids),
                                  Int[], oos_df))
            continue
        end

        cutoff = minimum(kickoff[Int(id)] for id in oos_df.match_id)
        keep(id) = haskey(kickoff, Int(id)) && kickoff[Int(id)] < cutoff

        history_keep = [id for id in boundary.history_match_ids if keep(id)]
        target_keep  = [id for id in boundary.target_match_ids  if keep(id)]

        all_nominal = vcat(boundary.history_match_ids, boundary.target_match_ids)
        fitted_ids  = vcat(history_keep, target_keep)
        dropped_ids = setdiff(all_nominal, fitted_ids)

        trimmed = Data.SplitBoundary(
            boundary.fold_id,
            boundary.target_step,
            history_keep,
            target_keep,
        )

        push!(folds, TP00Fold(i, meta.time_step, meta.target_season, trimmed, meta,
                              fitted_ids, dropped_ids, oos_df))
    end
    return folds
end

tp00_kickoffs(ds, ids::AbstractVector{Int}) =
    DataFrames.subset(ds.matches, :match_id => ByRow(id -> id in Set(ids))).match_date

function tp00_last_kickoff(ds, ids::AbstractVector{Int})
    ks = tp00_kickoffs(ds, ids)
    return isempty(ks) ? nothing : maximum(skipmissing(ks))
end

function tp00_first_kickoff(df::AbstractDataFrame)
    return nrow(df) == 0 ? nothing : minimum(skipmissing(df.match_date))
end

function tp00_fold_table(ds, folds::Vector{TP00Fold})
    println()
    println("-" ^ 74)
    println("FOLD INVENTORY   ($(length(folds)) folds)")
    println("-" ^ 74)
    println("  fold  season   step   fitted  dropped    t+1   last fitted   first OOS")
    for f in folds
        fitted_last = tp00_last_kickoff(ds, f.fitted_ids)
        oos_first   = tp00_first_kickoff(f.oos_df)
        println("  ",
            rpad(f.idx, 5), " ",
            rpad(f.season, 8),
            rpad(f.step, 6),
            lpad(length(f.fitted_ids), 6), " ",
            lpad(length(f.dropped_ids), 7), " ",
            lpad(nrow(f.oos_df), 6), "   ",
            rpad(string(fitted_last), 13), " ",
            string(oos_first))
    end
    println("-" ^ 74)
    return nothing
end


# ==============================================================================
# 2. GATE 0 — Contract
# ==============================================================================

function tp00_gate_contract(ds, folds::Vector{TP00Fold}, contract::SLContract)
    results = []

    tourns = sort(unique(ds.matches.tournament_id))
    push!(results, (
        name   = "tournaments present",
        pass   = issubset(Set(contract.tournaments), Set(tourns)),
        detail = "datastore has $tourns; contract wants $(contract.tournaments)",
    ))

    seasons_used = sort(unique([f.season for f in folds]))
    push!(results, (
        name   = "development seasons only",
        pass   = isempty(intersect(Set(seasons_used), Set(contract.sealed_seasons))),
        detail = "folds target $seasons_used; sealed = $(contract.sealed_seasons)",
    ))

    push!(results, (
        name   = "folds exist",
        pass   = !isempty(folds),
        detail = "$(length(folds)) folds",
    ))

    empty_oos = [f.idx for f in folds if nrow(f.oos_df) == 0]
    push!(results, (
        name   = "every fold has t+1 fixtures",
        pass   = isempty(empty_oos),
        detail = isempty(empty_oos) ? "$(sum(nrow(f.oos_df) for f in folds)) OOS fixtures total" :
                                      "empty at folds $empty_oos",
    ))

    all_oos = vcat([f.oos_df.match_id for f in folds]...)
    push!(results, (
        name   = "no duplicate OOS fixtures",
        pass   = length(all_oos) == length(unique(all_oos)),
        detail = "$(length(all_oos)) rows, $(length(unique(all_oos))) unique",
    ))

    return results
end


# ==============================================================================
# 3. GATE 1 — Config
# ==============================================================================

function tp00_gate_config(model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract)
    results = []

    required = Features.required_features(model)
    push!(results, (
        name   = "required features declared",
        pass   = !isempty(required),
        detail = join([string(typeof(f).name.name) for f in required], ", "),
    ))

    h = sl_hash(model)
    push!(results, (
        name   = "config hashes deterministically",
        pass   = h == sl_hash(model) && length(h) == 8,
        detail = "hash = $h",
    ))

    hl = model.dynamics_config.days_half_life
    push!(results, (
        name   = "half-life is set",
        pass   = hl > 0,
        detail = "$(hl) days",
    ))

    push!(results, (
        name   = "l02 covers these components",
        pass   = try tp00_assert_default(model) catch; false end,
        detail = "parity reference is valid for this pure Poisson component set",
    ))

    return results
end


# ==============================================================================
# 4. GATE 2 — Features
# ==============================================================================

function tp00_truncate_datastore(ds, keep_ids::AbstractVector{Int})
    keep = Set(keep_ids)
    trim(df) = (nrow(df) > 0 && "match_id" in names(df)) ?
               DataFrame(DataFrames.subset(df, :match_id => ByRow(id -> id in keep))) : df

    return Data.DataStore(
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

function tp00_featureset_equal(a, b)
    ka = Set(keys(a.data))
    kb = Set(keys(b.data))
    differing = Symbol[]

    for k in union(ka, kb)
        if !(k in ka) || !(k in kb) || !isequal(a.data[k], b.data[k])
            push!(differing, k)
        end
    end
    return (isempty(differing), sort(differing))
end

function tp00_gate_features(ds, folds::Vector{TP00Fold}, model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract;
                            perturb_fold::Int = 1)
    results = []
    splitter = sl_splitter(contract)

    tp_splits    = [(f.boundary, f.meta) for f in folds]
    tp_collection = Features.create_features(tp_splits, ds, model, splitter)
    feature_sets  = [fs for (fs, _) in tp_collection]

    # --- (a) kickoff filtration ----------------------------------------------
    kickoff = tp00_kickoff_map(ds)
    violations = Int[]
    for f in folds
        nrow(f.oos_df) == 0 && continue
        cutoff = minimum(kickoff[Int(id)] for id in f.oos_df.match_id)
        any(kickoff[Int(id)] >= cutoff for id in f.fitted_ids) && push!(violations, f.idx)
    end
    push!(results, (
        name   = "kickoff filtration holds",
        pass   = isempty(violations),
        detail = isempty(violations) ?
                 "every fitted kickoff strictly precedes its fold's first OOS kickoff" :
                 "VIOLATED in folds $violations",
    ))

    n_dropped = sum(length(f.dropped_ids) for f in folds; init = 0)
    dropped_by_fold = [(f.idx, length(f.dropped_ids)) for f in folds if !isempty(f.dropped_ids)]
    push!(results, (
        name   = "kickoff filtration drops",
        pass   = true,
        detail = n_dropped == 0 ? "nothing dropped" :
                 "$n_dropped observations removed as not-yet-played: $dropped_by_fold",
    ))

    # --- (b) perturbation ----------------------------------------------------
    f  = folds[perturb_fold]
    ds_trunc = tp00_truncate_datastore(ds, vcat(f.fitted_ids, f.oos_df.match_id))
    fc_trunc = Features.create_features([(f.boundary, f.meta)], ds_trunc, model, splitter)
    fs_trunc = fc_trunc[1][1]
    same, differing = tp00_featureset_equal(feature_sets[perturb_fold], fs_trunc)
    push!(results, (
        name   = "perturbation (future cannot alter past)",
        pass   = same,
        detail = same ? "fold $(f.idx) FeatureSet bit-identical with future matches removed" :
                        "DIFFERS on keys: $differing",
    ))

    # --- (c) type purity ------------------------------------------------------
    impure = Tuple{Int, Symbol, String}[]
    for (i, fs) in enumerate(feature_sets)
        for (k, v) in fs.data
            v isa AbstractVector{<:Real} || continue
            if any(ismissing, v)
                push!(impure, (i, k, "missing"))
            elseif eltype(v) <: AbstractFloat && any(isnan, v)
                push!(impure, (i, k, "NaN"))
            end
        end
    end
    push!(results, (
        name   = "type purity (no missing / NaN)",
        pass   = isempty(impure),
        detail = isempty(impure) ? "all numeric vectors clean across $(length(feature_sets)) folds" :
                                   "first offender: $(impure[1])",
    ))

    # --- (d) team map identity ------------------------------------------------
    bad_maps = Int[]
    for (i, fs) in enumerate(feature_sets)
        tm = fs.data[:team_map]
        keytype(tm) <: AbstractString || push!(bad_maps, i)
    end
    push!(results, (
        name   = "team_map keyed by NAME",
        pass   = isempty(bad_maps),
        detail = isempty(bad_maps) ?
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
    push!(results, (
        name   = "contiguous model time states",
        pass   = isempty(ragged),
        detail = isempty(ragged) ?
                 "time_indices are 1..K with no gaps in all $(length(feature_sets)) folds" :
                 "gaps present in folds $ragged",
    ))

    # --- (f) OOS team coverage ------------------------------------------------
    unmapped = Tuple{Int, String}[]
    for (i, fs) in enumerate(feature_sets)
        tm = fs.data[:team_map]
        for row in eachrow(folds[i].oos_df)
            haskey(tm, row.home_team) || push!(unmapped, (i, row.home_team))
            haskey(tm, row.away_team) || push!(unmapped, (i, row.away_team))
        end
    end
    n_sides = sum(2 * nrow(f.oos_df) for f in folds)
    push!(results, (
        name   = "OOS team coverage",
        pass   = true,
        detail = "$(length(unmapped)) / $n_sides sides unmapped" *
                 (isempty(unmapped) ? "" : " → population fallback: $(unique([u[2] for u in unmapped]))"),
    ))

    return (results, feature_sets)
end
