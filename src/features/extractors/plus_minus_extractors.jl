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

    function emit_zeros!(fit_ids::Vector{Int})
        for side in sides, pos in positions
            F_data[Symbol("flat_$(side)_$(pos)_rating")] = zeros(Float64, n)
        end
        # A one means that no fitted RAPM coverage was available, hence neutral zeros.
        F_data[:flat_plus_minus_fallback] = ones(Int, n)
        F_data[:plus_minus_fit_match_ids] = fit_ids
        F_data[:player_ratings_map] = Dict{Int, Dict{Tuple{String, String}, Float64}}()
        F_data[:plus_minus_ratings] = Dict{Int, Float64}()
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
    # for diagnostics.
    F_data[:player_ratings_map] = ratings_map
    F_data[:plus_minus_ratings] = rating_of
    # Matches without any starter carrying a non-neutral fitted rating receive neutral zeros.
    F_data[:flat_plus_minus_fallback] = Int[haskey(ratings_map, Int(id)) ? 0 : 1 for id in ordered_ids]
    F_data[:plus_minus_fit_match_ids] = fit_id_vector
    return nothing
end
