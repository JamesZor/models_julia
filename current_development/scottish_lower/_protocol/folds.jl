# ==============================================================================
# Scottish Lower protocol — FOLD INVENTORY
# ==============================================================================
#
# Shared walk-forward fold construction.  This is the model-agnostic lift of
# 01_team_poisson/l03_gates.jl: models supply their own operations through an
# adapter, but the population presented to every subsequent gate is identical.
#
# ============================================================================

"""
    sl_kickoff_map(ds) -> Dict{Int, DateTime}

Kickoff instant per match, from `match_date` + `match_hour`.

Day resolution is not enough. On 2024-10-19 tournament 56 sits at biweek 5 while
tournament 57 sits at biweek 6, so a pooled fold can fit on a 16:00 match and
predict a 14:00 match on the same date. Only the hour distinguishes them.

Rows without a complete kickoff are deliberately absent: they cannot silently be
ordered as a date-only fixture.
"""
function sl_kickoff_map(ds)
    kickoff = Dict{Int, DateTime}()
    for row in eachrow(ds.matches)
        ismissing(row.match_date) && continue
        ismissing(row.match_hour) && continue
        kickoff[Int(row.match_id)] = DateTime(row.match_date) + Hour(row.match_hour)
    end
    return kickoff
end

"""
    sl_build_folds(ds, contract) -> Vector{SLFold}

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
function sl_build_folds(ds, contract::SLContract)
    sl_assert_not_sealed(contract, contract.dev_seasons)

    splitter   = sl_splitter(contract)
    boundaries = SLData.create_id_boundaries(ds, splitter)
    kickoff    = sl_kickoff_map(ds)

    folds = SLFold[]
    for (i, (boundary, meta)) in enumerate(boundaries)
        oos_df = DataFrame(SLData.get_next_matches(ds, (boundary, meta), splitter))

        if nrow(oos_df) == 0
            fitted_ids = Int.(vcat(boundary.history_match_ids, boundary.target_match_ids))
            push!(folds, SLFold(i, meta.time_step, meta.target_season, boundary, meta,
                                fitted_ids, Int[], oos_df))
            continue
        end

        oos_kickoffs = [get(kickoff, Int(id), nothing) for id in oos_df.match_id]
        any(isnothing, oos_kickoffs) && error("fold $i has OOS fixtures without a full DateTime kickoff")
        cutoff = minimum(something.(oos_kickoffs))
        keep(id) = haskey(kickoff, Int(id)) && kickoff[Int(id)] < cutoff

        history_keep = Int[id for id in boundary.history_match_ids if keep(id)]
        target_keep  = Int[id for id in boundary.target_match_ids  if keep(id)]

        all_nominal = Int.(vcat(boundary.history_match_ids, boundary.target_match_ids))
        fitted_ids  = vcat(history_keep, target_keep)
        dropped_ids = setdiff(all_nominal, fitted_ids)

        trimmed = SLData.SplitBoundary(
            boundary.fold_id,
            boundary.target_step,
            history_keep,
            target_keep,
        )

        push!(folds, SLFold(i, meta.time_step, meta.target_season, trimmed, meta,
                            fitted_ids, dropped_ids, oos_df))
    end
    return folds
end

"""
    sl_fold_table(ds, folds)

Print the fold inventory. Gate 0 is not a computation, it is a statement of what
is being fitted and what is being predicted, made visible. Kickoffs are reported
as full `DateTime`s, including same-day ordering.
"""
function sl_fold_table(ds, folds::Vector{SLFold})
    println()
    println("-" ^ 74)
    println("FOLD INVENTORY   ($(length(folds)) folds)")
    println("-" ^ 74)
    println("  fold  season   step   fitted  dropped    t+1   last fitted          first OOS")
    for f in folds
        fitted_last = sl_last_kickoff(ds, f.fitted_ids)
        oos_first   = sl_first_kickoff(f.oos_df)
        println("  ",
            rpad(f.idx, 5), " ",
            rpad(f.season, 8),
            rpad(f.step, 6),
            lpad(length(f.fitted_ids), 6), " ",
            lpad(length(f.dropped_ids), 7), " ",
            lpad(nrow(f.oos_df), 6), "   ",
            rpad(string(fitted_last), 20), " ",
            string(oos_first))
    end
    println("-" ^ 74)
    return nothing
end


# ==============================================================================
# Kickoff helpers
# ==============================================================================

# `subset` is exported by BOTH DataFrames and DynamicPPL. Qualify it here (and
# at sl_truncate_datastore) -- do not "tidy" this away.
function sl_kickoffs(ds, ids::AbstractVector{Int})
    kickoff = sl_kickoff_map(ds)
    return DateTime[kickoff[id] for id in ids if haskey(kickoff, id)]
end

function sl_last_kickoff(ds, ids::AbstractVector{Int})
    ks = sl_kickoffs(ds, ids)
    return isempty(ks) ? nothing : maximum(ks)
end

function sl_first_kickoff(df::AbstractDataFrame)
    nrow(df) == 0 && return nothing
    kickoffs = DateTime[]
    for row in eachrow(df)
        ismissing(row.match_date) && continue
        ismissing(row.match_hour) && continue
        push!(kickoffs, DateTime(row.match_date) + Hour(row.match_hour))
    end
    return isempty(kickoffs) ? nothing : minimum(kickoffs)
end


# ==============================================================================
# GATE 0 — Contract
# ==============================================================================

"""
    sl_gate_contract(ds, folds, contract)

The data snapshot and the fold structure are what they claim to be.
"""
function sl_gate_contract(ds, folds::Vector{SLFold}, contract::SLContract)
    results = []

    tourns = sort(unique(ds.matches.tournament_id))
    push!(results, sl_result(
        "tournaments present",
        issubset(Set(contract.tournaments), Set(tourns)),
        "datastore has $tourns; contract wants $(contract.tournaments)",
    ))

    seasons_used = sort(unique([f.season for f in folds]))
    push!(results, sl_result(
        "development seasons only",
        isempty(intersect(Set(seasons_used), Set(contract.sealed_seasons))),
        "folds target $seasons_used; sealed = $(contract.sealed_seasons)",
    ))

    push!(results, sl_result("folds exist", !isempty(folds), "$(length(folds)) folds"))

    empty_oos = [f.idx for f in folds if nrow(f.oos_df) == 0]
    push!(results, sl_result(
        "every fold has t+1 fixtures",
        isempty(empty_oos) && !isempty(folds),
        isempty(folds) ? "no folds available" :
        isempty(empty_oos) ? "$(sum(nrow(f.oos_df) for f in folds)) OOS fixtures total" :
                             "empty at folds $empty_oos",
    ))

    all_oos = isempty(folds) ? Int[] : vcat([Int.(f.oos_df.match_id) for f in folds]...)
    push!(results, sl_result(
        "no duplicate OOS fixtures",
        !isempty(all_oos) && length(all_oos) == length(unique(all_oos)),
        isempty(all_oos) ? "no OOS fixtures" : "$(length(all_oos)) rows, $(length(unique(all_oos))) unique",
    ))

    return results
end
