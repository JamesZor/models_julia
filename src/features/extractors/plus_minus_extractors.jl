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
# 1. LEAK-SAFE FIT. The ridge is fitted on `F_data[:history_match_ids]` ONLY, and the single
#    resulting rating vector is applied to every match in the fold. Target-fold matches therefore
#    carry a genuinely pre-match rating — the forward-chained design the research validated. The
#    time decay is anchored at the first target kickoff, so the newest history matches count most.
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

Fit RAPM on the fold's history matches and emit the 8 positional rating vectors.

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

    # --- 1. the fold split ------------------------------------------------------------------
    haskey(F_data, :history_match_ids) || error(
        "AbstractPlusMinusFeature needs F_data[:history_match_ids] to fit leak-safe. " *
        "The builder stashes it (src/features/builder.jl); a hand-rolled F_data must set it too.")
    hist = F_data[:history_match_ids]::Set{Int}
    isempty(hist) && return emit_zeros!()

    # Anchor the time decay at the first target kickoff, so the ratings are "as of" the moment they
    # are used. With no target matches (a pure-history fold) fall back to the last history date.
    date_of = Dict{Int, Date}(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
    target_dates = [date_of[i] for i in Int.(ordered_ids) if haskey(date_of, i) && !(i in hist)]
    hist_dates   = [date_of[i] for i in Int.(ordered_ids) if haskey(date_of, i) && i in hist]
    T_rating = !isempty(target_dates) ? minimum(target_dates) :
               (!isempty(hist_dates) ? maximum(hist_dates) : maximum(prep.segments.match_date))

    # --- 2. the ridge fit, history only -----------------------------------------------------
    segs = prep.segments[in.(Int.(prep.segments.match_id), Ref(hist)), :]
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
    id_set = Set(Int.(ordered_ids))
    ratings_map = Dict{Int, Dict{Tuple{String, String}, Float64}}()
    lineups = ds.lineups
    if nrow(lineups) > 0
        for r in eachrow(lineups)
            mid = Int(r.match_id)
            mid in id_set || continue
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
