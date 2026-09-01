# src/features/extractors/plus_minus_extractors.jl
#
# ONE extractor for the whole `AbstractPlusMinusFeature` family. `pm_target(config)` picks the
# response column and `config.{w_sim, λ, half_life_days}` parameterise the ridge, so every variant
# — shots / shots-on-target / goals / xG — runs through this single method.
#
# It emits EXACTLY the contract `player_extractors.jl` emits, so an APM rating drops into any engine
# that already reads a player rating:
#   * 8 keys `:flat_{home,away}_{G,D,M,F}_rating`, each a `Vector{Float64}` of `length(ordered_ids)`
#   * `:player_ratings_map :: Dict{Int, Dict{Tuple{String,String}, Float64}}` for OOS prediction
#
# Two design points that are load-bearing:
#
# 1. LEAK-SAFE FIT — Gate 2 requires `fit_on = :history` (the default), so the ridge is fit only
#    on the frozen history block. `fit_on = :training` (history ∪ target) is retained solely as a
#    non-Gate-2 research override; it must be requested explicitly. Either way the time decay is
#    anchored at the LAST match in the fit set, so the most recent permitted matches carry the most
#    weight.
#
# 2. NO `minutes_played` WEIGHTING. `player_extractors.jl` weights each rating by
#    `clamp(minutes_played, 0, 90)/90`, but on tiers 56/57 `minutes_played` is IDENTICALLY 0 before
#    23/24 and NULL for much of 25/26 — coalescing it to 0 would zero out real ratings across whole
#    seasons. Instead the aggregation sums the STARTING XI at weight 1.0, which is exactly the
#    covariate the research's own validity test used (`l04_ridge_apm.jl::match_strength` sums the
#    starters' ratings) and is the only version available pre-match anyway.
#
# The composable pure-player dynamics component also needs starter, bench, positional, and
# expected-minute aggregates as contiguous vectors. Those are emitted here from the SAME
# history-fit rating vector; no regression is repeated and no rating is learned from the
# match being represented.

using DataFrames

const PMLineupAggregate = NamedTuple{
    (:home_outfield, :away_outfield, :home_bench, :away_bench,
     :home_D, :home_M, :home_F, :away_D, :away_M, :away_F,
     :home_bench_D, :home_bench_M, :home_bench_F,
     :away_bench_D, :away_bench_M, :away_bench_F,
     :home_minute, :away_minute),
    NTuple{18,Float64},
}

_pm_empty_lineup_aggregate() = PMLineupAggregate(ntuple(_ -> 0.0, 18))

"""
    pm_lineup_aggregates(lineups, matches, rating_of) -> Dict{Int,PMLineupAggregate}

Collapse one fixed, history-fit player-rating vector into the four pre-match lineup
formulations. Expected minutes are each player's mean positive minutes over his previous five
recorded appearances; a player without usable history defaults to 90 minutes when starting and
zero on the bench. Current-match minutes are applied only after that match is aggregated.
"""
function pm_lineup_aggregates(lineups::AbstractDataFrame, matches::AbstractDataFrame,
                              rating_of::Dict{Int,Float64})
    out = Dict{Int,PMLineupAggregate}()
    nrow(lineups) == 0 && return out
    cols = propertynames(lineups)
    required = (:match_id, :player_id, :team_side, :position)
    all(in(cols), required) || return out

    date_of = Dict{Int,Date}(Int(r.match_id) => r.match_date for r in eachrow(matches))
    row_order = sortperm(1:nrow(lineups); by = i -> begin
        mid = Int(lineups.match_id[i])
        (get(date_of, mid, Date(9999, 12, 31)), mid)
    end)
    minute_history = Dict{Int,Vector{Float64}}()

    cursor = 1
    while cursor <= length(row_order)
        first_index = row_order[cursor]
        match_id = Int(lineups.match_id[first_index])
        last_cursor = cursor
        while last_cursor < length(row_order) &&
              Int(lineups.match_id[row_order[last_cursor + 1]]) == match_id
            last_cursor += 1
        end

        values = zeros(Float64, 18)
        minute_updates = Tuple{Int,Float64}[]
        for k in cursor:last_cursor
            row = lineups[row_order[k], :]
            ismissing(row.player_id) && continue
            ismissing(row.team_side) && continue
            player_id = Int(row.player_id)
            rating = get(rating_of, player_id, 0.0)
            isfinite(rating) || continue
            side = lowercase(String(row.team_side))
            side in ("home", "away") || continue
            home = side == "home"
            position = pm_clean_position(row.position)
            position == "G" && continue
            substitute = :is_substitute in cols && coalesce(row.is_substitute, false)

            if substitute
                values[home ? 3 : 4] += rating
                pos_index = position == "D" ? 0 : position == "M" ? 1 : 2
                values[(home ? 11 : 14) + pos_index] += rating
            else
                values[home ? 1 : 2] += rating
                pos_index = position == "D" ? 0 : position == "M" ? 1 : 2
                values[(home ? 5 : 8) + pos_index] += rating
            end

            history = get(minute_history, player_id, Float64[])
            expected_minutes = isempty(history) ? (substitute ? 0.0 : 90.0) :
                               sum(history) / length(history)
            values[home ? 17 : 18] += rating * (expected_minutes / 90.0)

            if :minutes_played in cols && !ismissing(row.minutes_played)
                minutes = Float64(row.minutes_played)
                if isfinite(minutes) && minutes > 0.0
                    push!(minute_updates, (player_id, min(minutes, 120.0)))
                end
            end
        end
        out[match_id] = PMLineupAggregate(Tuple(values))

        # Update only after every row in the current match has been represented, so the
        # expected-minute vector is strictly pre-match even when a player appears once per side.
        for (player_id, minutes) in minute_updates
            history = get!(minute_history, player_id, Float64[])
            push!(history, minutes)
            length(history) > 5 && popfirst!(history)
        end
        cursor = last_cursor + 1
    end
    return out
end

function _emit_pm_lineup_vectors!(F_data::Dict, ordered_ids, aggregates)
    neutral = _pm_empty_lineup_aggregate()
    picked = PMLineupAggregate[get(aggregates, Int(id), neutral) for id in ordered_ids]
    names = fieldnames(PMLineupAggregate)
    keys = (
        :flat_home_outfield_rating, :flat_away_outfield_rating,
        :flat_home_bench_rating, :flat_away_bench_rating,
        :flat_home_D_rating, :flat_home_M_rating, :flat_home_F_rating,
        :flat_away_D_rating, :flat_away_M_rating, :flat_away_F_rating,
        :flat_home_bench_D_rating, :flat_home_bench_M_rating, :flat_home_bench_F_rating,
        :flat_away_bench_D_rating, :flat_away_bench_M_rating, :flat_away_bench_F_rating,
        :flat_home_minute_weighted_rating, :flat_away_minute_weighted_rating,
    )
    for (key, name) in zip(keys, names)
        F_data[key] = Float64[getproperty(value, name) for value in picked]
    end
    F_data[:player_lineup_ratings_map] = aggregates
    return nothing
end

"""
    add_feature!(F_data, config::AbstractPlusMinusFeature, ordered_ids, team_map, ds)

Fit RAPM on the matches this fold is allowed to learn from (see `config.fit_on`) and emit the 8
positional rating vectors aligned to `ordered_ids`, plus a whole-store `:player_ratings_map`.

Degrades to all-zero ratings (never an error) when the segment table is empty — i.e. on any segment
without BBC/incident coverage, which is every non-Scottish league.
"""
function add_feature!(F_data::Dict, config::AbstractPlusMinusFeature, ordered_ids,
                      team_map::Dict, ds::Data.DataStore)
    n = length(ordered_ids)
    positions = ("G", "D", "M", "F")
    sides = ("home", "away")

    function emit_zeros!(fit_ids::Vector{Int})
        for side in sides, pos in positions
            F_data[Symbol("flat_$(side)_$(pos)_rating")] = zeros(Float64, n)
        end
        # A one means that no fitted RAPM coverage was available, hence neutral zeros.
        F_data[:flat_plus_minus_fallback] = ones(Int, n)
        F_data[:plus_minus_fit_match_ids] = fit_ids
        F_data[:player_ratings_map] = Dict{Int, Dict{Tuple{String, String}, Float64}}()
        F_data[:plus_minus_ratings] = Dict{Int, Float64}()
        empty_aggregates = Dict{Int,PMLineupAggregate}()
        _emit_pm_lineup_vectors!(F_data, ordered_ids, empty_aggregates)
        return nothing
    end

    prep = pm_prepared(ds)

    # --- 1. which matches the ridge may learn from -------------------------------------------
    fit_ids = if config.fit_on === :history
        haskey(F_data, :history_match_ids) || error(
            "AbstractPlusMinusFeature(fit_on = :history) needs F_data[:history_match_ids]. " *
            "The builder stashes it (src/features/builder.jl); a hand-rolled F_data must too.")
        # Gate-2 default: source IDs exclusively from the frozen history boundary.
        Set(Int.(F_data[:history_match_ids]))
    elseif config.fit_on === :training
        # Non-Gate-2 research override; never the default.
        Set(Int.(ordered_ids))
    else
        error("Unknown fit_on = $(config.fit_on); expected :training or :history")
    end
    fit_id_vector = sort!(collect(fit_ids))
    nrow(prep.segments) == 0 && return emit_zeros!(fit_id_vector)
    isempty(fit_ids) && return emit_zeros!(fit_id_vector)

    # Anchor the decay at the last match the ridge is allowed to see.
    date_of  = Dict{Int, Date}(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
    fit_dates = [d for (i, d) in date_of if i in fit_ids]
    T_rating = isempty(fit_dates) ? maximum(prep.segments.match_date) : maximum(fit_dates)

    # --- 2. the ridge fit --------------------------------------------------------------------
    segs = prep.segments[in.(Int.(prep.segments.match_id), Ref(fit_ids)), :]
    if pm_target(config) === :y_xg
        # Unlike count targets, y_xg depends on a fitted shot-xG lookup. Refit that lookup from
        # exactly the permitted matches, then rebuild the copied segments' targets. The prepared
        # cache deliberately fits its lookup store-wide for research parity, which is not Gate-2
        # safe for this variant.
        xg_shots = build_shots(ds)
        xg_shots = xg_shots[in.(Int.(xg_shots.match_id), Ref(fit_ids)), :]
        if nrow(xg_shots) > 0
            xg_model = fit_shot_xg(xg_shots)
            xg_shots.xg = predict_xg(xg_model, xg_shots)
        else
            xg_shots.xg = Float64[]
        end
        segs = copy(segs)
        add_targets!(segs, xg_shots, prep.it1_by_match)
    end
    fit = fit_ratings(segs;
                      target        = pm_target(config),
                      λ             = config.λ,
                      w_sim         = config.w_sim,
                      half_life     = config.half_life_days,
                      T_rating      = T_rating,
                      # Competition-history controls must be reconstructed from the same
                      # permitted matches as the ridge, not the whole-store prepared cache.
                      comp_sets     = competition_sets(ds; match_ids = fit_ids))
    fit === nothing && return emit_zeros!(fit_id_vector)

    rating_of = Dict{Int, Float64}(Int(r.player_id) => Float64(r.rapm) for r in eachrow(fit))

    # --- 3. positional aggregation over the STARTING XI --------------------------------------
    # Built over EVERY match in the store, not just this fold's. `extract_parameters` reads this map
    # for the OUT-OF-SAMPLE matches (the next observed effective step), which are by construction
    # not in `ordered_ids` — restricting the map to the fold would hand every prediction a zero pillar
    # and silently collapse the engine onto its no-APM twin. `player_extractors.jl` does the same
    # for the same reason. Only the RATING VECTOR is leak-controlled; applying it to a future
    # teamsheet is precisely the pre-match rating being tested.
    ratings_map = Dict{Int, Dict{Tuple{String, String}, Float64}}()
    lineups = ds.lineups
    if nrow(lineups) > 0
        for r in eachrow(lineups)
            mid = Int(r.match_id)
            coalesce(r.is_substitute, false) && continue      # starters only; see the header
            ismissing(r.player_id) && continue
            rt = get(rating_of, Int(r.player_id), 0.0)
            rt == 0.0 && continue
            key = (String(r.team_side), pm_clean_position(r.position))
            d = get!(ratings_map, mid, Dict{Tuple{String, String}, Float64}())
            d[key] = get(d, key, 0.0) + rt
        end
    end

    empty_map = Dict{Tuple{String, String}, Float64}()
    get_r(mid, side, pos) = get(get(ratings_map, mid, empty_map), (side, pos), 0.0)

    for side in sides, pos in positions
        F_data[Symbol("flat_$(side)_$(pos)_rating")] =
            [get_r(Int(id), side, pos) for id in ordered_ids]
    end

    # Lookup for ALL matches in the fold (supports OOS prediction), plus the raw per-player vector
    # for diagnostics. The richer lineup map feeds the composable pure-player dynamics component.
    F_data[:player_ratings_map] = ratings_map
    aggregates = pm_lineup_aggregates(ds.lineups, ds.matches, rating_of)
    _emit_pm_lineup_vectors!(F_data, ordered_ids, aggregates)
    F_data[:plus_minus_ratings] = rating_of
    # Matches without any starter carrying a non-neutral fitted rating receive neutral zeros.
    F_data[:flat_plus_minus_fallback] = Int[haskey(ratings_map, Int(id)) ? 0 : 1 for id in ordered_ids]
    F_data[:plus_minus_fit_match_ids] = fit_id_vector
    return nothing
end
