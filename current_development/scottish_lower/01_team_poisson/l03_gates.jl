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
    boundary::Data.SplitBoundary      # TRIMMED — safe to fit on
    meta::Any
    fitted_ids::Vector{Int}           # trimmed: kickoff strictly before first OOS kickoff
    dropped_ids::Vector{Int}          # removed by kickoff filtration; reported, never silent
    oos_df::DataFrame
end

"""
    tp_kickoff_map(ds) -> Dict{Int, DateTime}

Kickoff instant per match, from `match_date` + `match_hour`.

Day resolution is not enough. On 2024-10-19 tournament 56 sits at biweek 5 while
tournament 57 sits at biweek 6, so a pooled fold can fit on a 16:00 match and
predict a 14:00 match on the same date. Only the hour distinguishes them.
"""
function tp_kickoff_map(ds)
    m = ds.matches
    return Dict{Int, DateTime}(
        Int(row.match_id) => DateTime(row.match_date) + Hour(row.match_hour)
        for row in eachrow(m)
    )
end

"""
    tp_build_folds(ds, contract) -> Vector{TPFold}

Cut the development folds, attach each fold's true `t+1` fixtures, and apply
KICKOFF FILTRATION: any nominally-prior observation whose kickoff is not strictly
before the earliest OOS kickoff is removed from the fitted set and recorded in
`dropped_ids`.

Two distinct things make this necessary:

  * pooled 56/57 biweeks are misaligned by one, so a pooled step mixes League One
    biweek `k` with League Two biweek `k+1` on the same calendar day;
  * postponements move individual matches across the cutoff.

Both produce the same failure — fitting on a match that has not been played at
prediction time — and both are handled here rather than trusted to the splitter.
"""
function tp_build_folds(ds, contract::SLContract)
    sl_assert_not_sealed(contract, contract.dev_seasons)

    splitter   = sl_splitter(contract)
    boundaries = Data.create_id_boundaries(ds, splitter)
    kickoff    = tp_kickoff_map(ds)

    folds = TPFold[]
    for (i, (boundary, meta)) in enumerate(boundaries)
        oos_df = DataFrame(Data.get_next_matches(ds, (boundary, meta), splitter))

        if nrow(oos_df) == 0
            push!(folds, TPFold(i, meta.time_step, meta.target_season, boundary, meta,
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

        push!(folds, TPFold(i, meta.time_step, meta.target_season, trimmed, meta,
                            fitted_ids, dropped_ids, oos_df))
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
    println("  fold  season   step   fitted  dropped    t+1   last fitted   first OOS")
    for f in folds
        fitted_last = tp_last_kickoff(ds, f.fitted_ids)
        oos_first   = tp_first_kickoff(f.oos_df)
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
    kickoff = tp_kickoff_map(ds)
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

    dropped_by_fold = [(f.idx, length(f.dropped_ids)) for f in folds if !isempty(f.dropped_ids)]
    n_dropped = sum(length(f.dropped_ids) for f in folds; init = 0)
    push!(results, (
        name   = "kickoff filtration drops",
        pass   = true,   # reported, not enforced — a drop is correct behaviour
        detail = n_dropped == 0 ? "nothing dropped" :
                 "$n_dropped observations removed as not-yet-played: $dropped_by_fold",
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
