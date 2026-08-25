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
# 1. LEAK-SAFE FIT — and note carefully WHICH set that means.
#    A `SplitBoundary` splits the FOLD's own matches into a frozen history block and an expanding
#    target block; BOTH are training data for that fold (`create_features` builds `ordered_ids =
#    [history; target]`, and the engine's likelihood runs over all of them). The out-of-sample
#    matches are fetched separately from the next observed effective step by
#    `Data.get_next_matches` — they are never in `ordered_ids`.
#    So `fit_on = :training` (the default) fits the ridge on history ∪ target, i.e. EXACTLY the
#    information set the Turing model itself is trained on. That is leak-free by construction and
#    it lets the rating keep updating through the target season.
#    `fit_on = :history` fits on the frozen history block only. That is also leak-free but freezes
#    the rating at the start of the target season, so by the last fold it is ~9 months stale. It is
#    kept as a config option so the difference can be measured rather than asserted.
#    Either way the time decay is anchored at the LAST match in the fit set, so the most recent
#    training matches carry the most weight.
#
# 2. NO `minutes_played` WEIGHTING. `player_extractors.jl` weights each rating by
#    `clamp(minutes_played, 0, 90)/90`, but on tiers 56/57 `minutes_played` is IDENTICALLY 0 before
#    23/24 and NULL for much of 25/26 — coalescing it to 0 would zero out real ratings across whole
#    seasons. Instead the aggregation sums the STARTING XI at weight 1.0, which is exactly the
#    covariate the research's own validity test used (`l04_ridge_apm.jl::match_strength` sums the
#    starters' ratings) and is the only version available pre-match anyway.

using DataFrames

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

    function emit_zeros!()
        for side in sides, pos in positions
            F_data[Symbol("flat_$(side)_$(pos)_rating")] = zeros(Float64, n)
        end
        F_data[:player_ratings_map] = Dict{Int, Dict{Tuple{String, String}, Float64}}()
        F_data[:plus_minus_ratings] = Dict{Int, Float64}()
        return nothing
    end

    prep = pm_prepared(ds)
    nrow(prep.segments) == 0 && return emit_zeros!()

    # --- 1. which matches the ridge may learn from -------------------------------------------
    fit_ids = if config.fit_on === :history
        haskey(F_data, :history_match_ids) || error(
            "AbstractPlusMinusFeature(fit_on = :history) needs F_data[:history_match_ids]. " *
            "The builder stashes it (src/features/builder.jl); a hand-rolled F_data must too.")
        F_data[:history_match_ids]::Set{Int}
    elseif config.fit_on === :training
        Set(Int.(ordered_ids))
    else
        error("Unknown fit_on = $(config.fit_on); expected :training or :history")
    end
    isempty(fit_ids) && return emit_zeros!()

    # Anchor the decay at the last match the ridge is allowed to see.
    date_of  = Dict{Int, Date}(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
    fit_dates = [d for (i, d) in date_of if i in fit_ids]
    T_rating = isempty(fit_dates) ? maximum(prep.segments.match_date) : maximum(fit_dates)

    # --- 2. the ridge fit --------------------------------------------------------------------
    segs = prep.segments[in.(Int.(prep.segments.match_id), Ref(fit_ids)), :]
    fit = fit_ratings(segs;
                      target        = pm_target(config),
                      λ             = config.λ,
                      w_sim         = config.w_sim,
                      half_life     = config.half_life_days,
                      T_rating      = T_rating,
                      comp_sets     = prep.comp_sets)
    fit === nothing && return emit_zeros!()

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
    # for diagnostics.
    F_data[:player_ratings_map] = ratings_map
    F_data[:plus_minus_ratings] = rating_of
    return nothing
end
