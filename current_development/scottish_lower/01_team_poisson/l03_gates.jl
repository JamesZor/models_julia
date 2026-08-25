# ==============================================================================
# Model 01 — GATE IMPLEMENTATIONS (stages ⓪–②)
# ==============================================================================
#
# Loader. Definitions only, no execution. The walkthrough calls these and prints
# the tables; the mechanics live here so the runner stays readable.
#
# These gates are written CONCRETELY for model 01. They are lifted into
# _protocol/ only when model 02 needs them, and on lifting model 01 is re-run
# against the lifted version and must produce identical output.
# See docs/PLAN.md § "Abstraction order".
#
# Every gate function returns a Vector of NamedTuples (; name, pass, detail),
# ready for `sl_gate_table`.
#
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics

const Data = BayesianFootball.Data


# ==============================================================================
# 1. Fold inventory
# ==============================================================================

"""
    TPFold

One walk-forward fold, with the distinction that matters stated explicitly:

    fitted_ids  every observation through step t  (history + target)
    oos_df      the genuinely held-out fixtures at step t+1

`target_match_ids` is NOT the test set. It is fitted. Calling it a test set is
the error that made the archived Stage 7 report a non-OOS "OOS" check.
"""
struct TPFold
    idx::Int
    step::Int
    season::String
    boundary::Data.SplitBoundary
    meta::Any
    fitted_ids::Vector{Int}
    oos_df::DataFrame
end

"""
    tp_build_folds(ds, contract) -> Vector{TPFold}

Cut the development folds and attach each fold's true `t+1` fixtures.
"""
function tp_build_folds(ds, contract::SLContract)
    sl_assert_not_sealed(contract, contract.dev_seasons)

    splitter   = sl_splitter(contract)
    boundaries = Data.create_id_boundaries(ds, splitter)

    folds = TPFold[]
    for (i, (boundary, meta)) in enumerate(boundaries)
        fitted_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
        oos_df     = Data.get_next_matches(ds, (boundary, meta), splitter)

        push!(folds, TPFold(
            i,
            meta.time_step,
            meta.target_season,
            boundary,
            meta,
            fitted_ids,
            DataFrame(oos_df),
        ))
    end
    return folds
end

"""
    tp_fold_table(folds)

Print the fold inventory. Gate 0 is not a computation, it is a statement of what
is being fitted and what is being predicted, made visible.
"""
function tp_fold_table(ds, folds::Vector{TPFold})
    println()
    println("-" ^ 74)
    println("FOLD INVENTORY   ($(length(folds)) folds)")
    println("-" ^ 74)
    println("  fold  season   step   fitted    t+1   last fitted   first OOS")
    for f in folds
        fitted_last = tp_last_kickoff(ds, f.fitted_ids)
        oos_first   = tp_first_kickoff(f.oos_df)
        println("  ",
            rpad(f.idx, 5), " ",
            rpad(f.season, 8),
            rpad(f.step, 6),
            lpad(length(f.fitted_ids), 6), " ",
            lpad(nrow(f.oos_df), 6), "   ",
            rpad(string(fitted_last), 13), " ",
            string(oos_first))
    end
    println("-" ^ 74)
    return nothing
end


# ==============================================================================
# 2. Kickoff helpers
# ==============================================================================

tp_kickoffs(ds, ids::AbstractVector{Int}) =
    subset(ds.matches, :match_id => ByRow(id -> id in Set(ids))).match_date

function tp_last_kickoff(ds, ids::AbstractVector{Int})
    ks = tp_kickoffs(ds, ids)
    return isempty(ks) ? nothing : maximum(skipmissing(ks))
end

function tp_first_kickoff(df::AbstractDataFrame)
    return nrow(df) == 0 ? nothing : minimum(skipmissing(df.match_date))
end


# ==============================================================================
# 3. GATE 0 — Contract
# ==============================================================================

"""
    tp_gate_contract(ds, folds, contract)

The data snapshot and the fold structure are what they claim to be.
"""
function tp_gate_contract(ds, folds::Vector{TPFold}, contract::SLContract)
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
# 4. GATE 1 — Config
# ==============================================================================

"""
    tp_gate_config(model, contract)

Nothing about the configuration is hidden, and everything it declares resolves.
"""
function tp_gate_config(model, contract::SLContract)
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
        detail = "$(hl) days  (PROVISIONAL — see MODEL.md)",
    ))

    push!(results, (
        name   = "l02 covers these components",
        pass   = try tp_assert_default(model) catch; false end,
        detail = "parity reference is valid for this component set",
    ))

    return results
end


# ==============================================================================
# 5. GATE 2 — Features
# ==============================================================================

"""
    tp_truncate_datastore(ds, keep_ids) -> DataStore

A copy of the store containing only `keep_ids`, for the perturbation test.
Every domain carrying a `match_id` is filtered; all 9 fields are passed through
so the BBC domains are not silently dropped (see Data.DataStore docstring).
"""
function tp_truncate_datastore(ds, keep_ids::AbstractVector{Int})
    keep = Set(keep_ids)
    trim(df) = (nrow(df) > 0 && "match_id" in names(df)) ?
               DataFrame(subset(df, :match_id => ByRow(id -> id in keep))) : df

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

"""
    tp_featureset_equal(a, b) -> (Bool, Vector{Symbol})

Structural comparison of two FeatureSets. Returns whether they match and the
keys that differ.
"""
function tp_featureset_equal(a, b)
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

"""
    tp_gate_features(ds, folds, model, contract; perturb_fold = 1)

The anti-leakage gate.

Note on the perturbation test: for model 01 every extractor is a pure lookup on
explicitly listed match IDs, so this test is expected to pass trivially. It is
run anyway, and it is run FIRST, because it is the test that will actually bite
on model 02, where features are FITTED (ridge/RAPM) and a global fit would leak
the future into every historical fold. Establishing it on a model where it must
pass is how we know the test itself works.
"""
function tp_gate_features(ds, folds::Vector{TPFold}, model, contract::SLContract;
                          perturb_fold::Int = 1)
    results = []
    splitter = sl_splitter(contract)

    # --- build every fold's FeatureSet ---------------------------------------
    feature_sets = [
        Features.create_features(f.boundary, ds, model, contract.dynamics_col)
        for f in folds
    ]

    # --- (a) kickoff filtration ----------------------------------------------
    violations = Tuple{Int, Any, Any}[]
    for f in folds
        last_fit  = tp_last_kickoff(ds, f.fitted_ids)
        first_oos = tp_first_kickoff(f.oos_df)
        if last_fit !== nothing && first_oos !== nothing && !(last_fit < first_oos)
            push!(violations, (f.idx, last_fit, first_oos))
        end
    end
    push!(results, (
        name   = "kickoff filtration",
        pass   = isempty(violations),
        detail = isempty(violations) ? "max fitted kickoff < min OOS kickoff in all $(length(folds)) folds" :
                                       "VIOLATED in folds $([v[1] for v in violations])",
    ))

    # --- (b) perturbation ----------------------------------------------------
    f  = folds[perturb_fold]
    ds_trunc = tp_truncate_datastore(ds, vcat(f.fitted_ids, f.oos_df.match_id))
    fs_trunc = Features.create_features(f.boundary, ds_trunc, model, contract.dynamics_col)
    same, differing = tp_featureset_equal(feature_sets[perturb_fold], fs_trunc)
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

    # --- (e) OOS team coverage ------------------------------------------------
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
        pass   = true,   # reported, not enforced — promoted sides are legitimate
        detail = "$(length(unmapped)) / $n_sides sides unmapped" *
                 (isempty(unmapped) ? "" : " → population fallback: $(unique([u[2] for u in unmapped]))"),
    ))

    return (results, feature_sets)
end
