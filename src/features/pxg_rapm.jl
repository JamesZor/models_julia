# src/features/pxg_rapm.jl
#
# POINT-IN-TIME STINT-LEVEL REGULARIZED ADJUSTED PLUS-MINUS, COLLAPSED TO ONE MATCH COVARIATE.
#
# WHAT THIS IS, AND HOW IT DIFFERS FROM `AbstractPlusMinusFeature`.
# The plus-minus family (types.jl / extractors/plus_minus_extractors.jl) emits EIGHT positional
# rating vectors, because the player-level engines it feeds carry a per-position pillar. The
# composable count builder has no such pillar: a covariate is one scalar per match. So this feature
# reuses the SAME validated estimator — segments, targets, ridge — and collapses its output to the
# single quantity the research's own validity test used:
#
#     x = ( SUM_{i in home XI} r_i  -  SUM_{j in away XI} r_j ) / scale
#
# (`l04_ridge_apm.jl::match_strength` in current_development/plus_minus_ratings/ is that sum; the
# base paper aggregates the starting XI and fits an ordered logit on the difference.)
#
# WHY `:y_xg` IS THE DEFAULT HERE AND `:y_shots` IS THE DEFAULT THERE.
# The plus-minus family's green-lit cell is `y_shots`, chosen on split-half reliability. This
# feature is the pxG-APM arm named in PXG_RAPM_SPEC.md, and its job inside a GOALS model is to carry
# chance quality rather than chance volume, so it regresses the zonal-xG stint differential. That is
# also the LEAST team-loaded target the research measured (club R^2 0.212 against 0.389 for shots),
# which matters more here than anywhere else: the engine already models team strength explicitly
# with `dyn.alpha`/`dyn.beta`, and a covariate that re-derives it is a covariate that fights its own
# model. `lambda = 200.0` is that target's tuned cell, matching `XGPlusMinusFeature`.
#
# THREE THINGS THAT ARE LOAD-BEARING FOR LEAK CONTROL
#
# 1. THE RIDGE IS HISTORY-FIT. `fit_on = :history` (the default) restricts the regression to the
#    fold's frozen history block, exactly as the plus-minus extractor does. `:training` exists only
#    as a research override and is not Gate-2 safe. For `:y_xg` the shot-xG cell table is REFITTED
#    from the permitted matches too — the prepared cache deliberately fits it store-wide for research
#    parity, which this variant cannot inherit.
#
# 2. THE RATING VECTOR IS LEAK-CONTROLLED; THE TEAMSHEET IS NOT. Ratings are applied to every match
#    in the store, including out-of-sample fixtures, because applying a history-fit rating to a
#    future starting XI IS the pre-match rating being tested. Restricting the map to the fold would
#    hand every prediction a zero covariate and silently collapse the model onto its no-RAPM twin.
#
# 3. SPARSE PLAYERS ARE SHRUNK, NOT TRUSTED. A ridge coefficient from four segments is mostly noise,
#    so each rating is scaled by `n / (n + shrink_segments)` on that player's segment count. RAPM is
#    zero-centred by construction, so this shrinks toward the neutral player and an unrated player
#    lands on exactly 0.0 — the builder's "missingness is a zero" contract, for free.

using DataFrames
using Dates
using Statistics

# ==========================================
# 1. CONFIG
# ==========================================
"""
    PxGRapmFeature <: AbstractFeatureConfig

Starting-XI RAPM differential, fit on stint segments.

  * `target`             — the segment response: `:y_xg` (pxG-APM, the default), `:y_shots`,
                           `:y_sot` or `:y_goals`.
  * `lambda`             — ridge penalty. `200.0` is the research's tuned cell for `:y_xg`.
  * `w_sim`              — teammate-similarity shrinkage weight. `0.0` (plain ridge) on purpose;
                           see plus_minus/ridge.jl for why the Brier-optimal `0.9` is NOT default.
  * `half_life_days`     — segment time decay, anchored at the last permitted match.
  * `fit_on`             — `:history` (Gate-2 default) or `:training` (research override).
  * `shrink_segments`    — pseudo-segments shrinking a sparse player's rating toward 0.
  * `min_rated_per_side` — a fixture is neutral unless both XIs contain at least this many rated
                           starters; a one-rated-player side is a differential built from nothing.
  * `k`                  — empirical-Bayes pseudo-count of the shot-xG cell table (`:y_xg` only).
  * `scale`              — divisor for the emitted column. `nothing` standardises on the spread of
                           the permitted matches, which keeps the weight prior meaningful whatever
                           the target's units; a `Float64` fixes it.
"""
Base.@kwdef struct PxGRapmFeature <: AbstractFeatureConfig
    target::Symbol = :y_xg
    lambda::Float64 = 200.0
    w_sim::Float64 = 0.0
    half_life_days::Float64 = 730.0
    fit_on::Symbol = :history
    shrink_segments::Float64 = 20.0
    min_rated_per_side::Int = 3
    k::Float64 = 25.0
    scale::Union{Float64, Nothing} = nothing
end

function _pxg_rapm_validate(config::PxGRapmFeature)
    config.target in PM_TARGETS ||
        error("PxGRapmFeature.target must be one of $(PM_TARGETS); got :$(config.target)")
    config.fit_on in (:history, :training) ||
        error("PxGRapmFeature.fit_on must be :history or :training; got :$(config.fit_on)")
    isfinite(config.lambda) && config.lambda >= 0.0 ||
        error("PxGRapmFeature.lambda must be finite and >= 0")
    isfinite(config.w_sim) && config.w_sim >= 0.0 ||
        error("PxGRapmFeature.w_sim must be finite and >= 0")
    isfinite(config.half_life_days) && config.half_life_days > 0.0 ||
        error("PxGRapmFeature.half_life_days must be finite and > 0")
    isfinite(config.shrink_segments) && config.shrink_segments >= 0.0 ||
        error("PxGRapmFeature.shrink_segments must be finite and >= 0")
    config.min_rated_per_side >= 0 ||
        error("PxGRapmFeature.min_rated_per_side must be >= 0")
    isfinite(config.k) && config.k >= 0.0 || error("PxGRapmFeature.k must be finite and >= 0")
    if config.scale !== nothing
        isfinite(config.scale) && config.scale > 0.0 ||
            error("PxGRapmFeature.scale must be finite and > 0 when given")
    end
    return nothing
end

# ==========================================
# 2. STARTING-XI AGGREGATION
# ==========================================
"""
    pxg_rapm_deltas(lineups, rating_of, exposure_of, config) -> Dict{Int, NamedTuple}

`(delta, home_rated, away_rated, available)` per match: the raw (unscaled) starting-XI rating
differential and how many rated starters each side contributed.

Kept separate from the fit so the aggregation can be exercised against a hand-written rating table.
Only STARTERS count — `minutes_played` is identically 0 before 23/24 on tiers 56/57 and NULL for
much of 25/26, so any minutes weighting would zero out real ratings across whole seasons, and the
starting XI is the only version knowable before kickoff anyway.
"""
function pxg_rapm_deltas(lineups::AbstractDataFrame,
                         rating_of::Dict{Int, Float64},
                         exposure_of::Dict{Int, Float64},
                         config::PxGRapmFeature)
    out = Dict{Int, NamedTuple{(:delta, :home_rated, :away_rated, :available),
                               Tuple{Float64, Int, Int, Float64}}}()
    nrow(lineups) == 0 && return out

    cols = propertynames(lineups)
    (:match_id in cols && :player_id in cols && :team_side in cols) || return out

    sums   = Dict{Tuple{Int, Bool}, Float64}()
    counts = Dict{Tuple{Int, Bool}, Int}()

    for r in eachrow(lineups)
        (:is_substitute in cols && coalesce(r.is_substitute, false)) && continue
        ismissing(r.player_id) && continue
        ismissing(r.team_side) && continue
        side = lowercase(String(r.team_side))
        side in ("home", "away") || continue
        is_home = side == "home"

        pid = Int(r.player_id)
        raw = get(rating_of, pid, 0.0)
        raw == 0.0 && continue
        isfinite(raw) || continue

        # Shrink toward the neutral (zero) player on this player's own segment exposure.
        n_seg = get(exposure_of, pid, 0.0)
        shrunk = raw * (n_seg / (n_seg + config.shrink_segments))
        isfinite(shrunk) || continue

        key = (Int(r.match_id), is_home)
        sums[key]   = get(sums, key, 0.0) + shrunk
        counts[key] = get(counts, key, 0) + 1
    end

    for mid in Set(first(key) for key in keys(sums))
        h_sum = get(sums, (mid, true), 0.0)
        a_sum = get(sums, (mid, false), 0.0)
        h_n   = get(counts, (mid, true), 0)
        a_n   = get(counts, (mid, false), 0)
        ok = h_n >= config.min_rated_per_side && a_n >= config.min_rated_per_side
        delta = ok ? h_sum - a_sum : 0.0
        if !isfinite(delta)
            delta = 0.0
            ok = false
        end
        out[mid] = (delta = delta, home_rated = h_n, away_rated = a_n,
                    available = ok ? 1.0 : 0.0)
    end
    return out
end

"""
    _pxg_rapm_scale(deltas, fit_ids, config) -> Float64

The divisor. A fixed `config.scale` wins; otherwise the standard deviation of the differential over
the PERMITTED matches — never the target block, which would let the fold's own spread set the
covariate's units.
"""
function _pxg_rapm_scale(deltas::Dict{Int, <:NamedTuple}, fit_ids::Set{Int},
                         config::PxGRapmFeature)
    config.scale === nothing || return Float64(config.scale)
    sample = Float64[v.delta for (mid, v) in deltas if mid in fit_ids && v.available > 0.0]
    length(sample) >= 10 || return 1.0
    s = std(sample)
    return (isfinite(s) && s > 1e-6) ? s : 1.0
end

# ==========================================
# 3. THE EXTRACTOR
# ==========================================
"""
    add_feature!(F_data, config::PxGRapmFeature, ordered_ids, team_map, ds)

Fit RAPM on the matches this fold may learn from, aggregate over starting XIs, and emit the scalar
match differential plus a whole-store bridge for prediction-time extraction.

Degrades to an all-zero column (never an error) when the segment table is empty or the permitted
subset is too small for `fit_ratings` — i.e. on any segment without lineup/incident coverage, which
is every non-Scottish league.
"""
function add_feature!(F_data::Dict, config::PxGRapmFeature, ordered_ids, team_map::Dict,
                      ds::Data.DataStore)
    _pxg_rapm_validate(config)
    n = length(ordered_ids)

    function emit_zeros!(fit_id_vector::Vector{Int})
        F_data[:flat_pxg_rapm]           = zeros(Float64, n)
        F_data[:flat_pxg_rapm_available] = zeros(Float64, n)
        F_data[:flat_pxg_rapm_fallback]  = ones(Int, n)
        F_data[:pxg_rapm_by_match_id]    = Dict{Int, Float64}()
        F_data[:pxg_rapm_ratings]        = Dict{Int, Float64}()
        F_data[:pxg_rapm_scale]          = 1.0
        F_data[:pxg_rapm_fit_match_ids]  = fit_id_vector
        return nothing
    end

    # --- 1. which matches the ridge may learn from ------------------------------------------
    fit_ids = if config.fit_on === :history
        haskey(F_data, :history_match_ids) || error(
            "PxGRapmFeature(fit_on = :history) needs F_data[:history_match_ids]. " *
            "The builder stashes it (src/features/builder.jl); a hand-rolled F_data must too.")
        Set(Int.(F_data[:history_match_ids]))
    else
        Set(Int.(ordered_ids))
    end
    fit_id_vector = sort!(collect(fit_ids))

    prep = pm_prepared(ds)
    (nrow(prep.segments) == 0 || isempty(fit_ids)) && return emit_zeros!(fit_id_vector)

    # Anchor the time decay at the last match the ridge is allowed to see.
    date_of = Dict{Int, Date}(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
    fit_dates = [d for (i, d) in date_of if i in fit_ids]
    T_rating = isempty(fit_dates) ? maximum(prep.segments.match_date) : maximum(fit_dates)

    # --- 2. the ridge fit --------------------------------------------------------------------
    segs = prep.segments[in.(Int.(prep.segments.match_id), Ref(fit_ids)), :]
    nrow(segs) == 0 && return emit_zeros!(fit_id_vector)

    if config.target === :y_xg
        # The prepared cache fits its shot-xG lookup store-wide for research parity. That is not
        # Gate-2 safe for this variant, so refit it from exactly the permitted matches and rebuild
        # the copied segments' targets against it.
        xg_shots = build_shots(ds)
        xg_shots = xg_shots[in.(Int.(xg_shots.match_id), Ref(fit_ids)), :]
        if nrow(xg_shots) > 0
            xg_model = fit_shot_xg(xg_shots; k = config.k)
            xg_shots.xg = predict_xg(xg_model, xg_shots)
        else
            xg_shots.xg = Float64[]
        end
        segs = copy(segs)
        add_targets!(segs, xg_shots, prep.it1_by_match)
    end

    fit = fit_ratings(segs;
                      target    = config.target,
                      λ         = config.lambda,
                      w_sim     = config.w_sim,
                      half_life = config.half_life_days,
                      T_rating  = T_rating,
                      comp_sets = competition_sets(ds; match_ids = fit_ids))
    fit === nothing && return emit_zeros!(fit_id_vector)

    rating_of = Dict{Int, Float64}(Int(r.player_id) => Float64(r.rapm) for r in eachrow(fit))

    # Exposure comes from the SAME permitted segments the ratings were fit on: a player's precision
    # is a property of how much the ridge actually saw of him, not of his whole career.
    exposure = player_exposure(segs)
    exposure_of = Dict{Int, Float64}(
        Int(r.player_id) => Float64(r.n_segments) for r in eachrow(exposure))

    # --- 3. starting-XI aggregation over EVERY match in the store ---------------------------
    deltas = pxg_rapm_deltas(ds.lineups, rating_of, exposure_of, config)
    scale = _pxg_rapm_scale(deltas, fit_ids, config)

    bridge = Dict{Int, Float64}(mid => v.delta / scale for (mid, v) in deltas)
    neutral = (delta = 0.0, home_rated = 0, away_rated = 0, available = 0.0)
    pick(id) = get(deltas, Int(id), neutral)

    F_data[:flat_pxg_rapm]           = Float64[pick(id).delta / scale for id in ordered_ids]
    F_data[:flat_pxg_rapm_available] = Float64[pick(id).available for id in ordered_ids]
    F_data[:flat_pxg_rapm_fallback]  = Int[pick(id).available > 0.0 ? 0 : 1 for id in ordered_ids]
    F_data[:pxg_rapm_by_match_id]    = bridge
    F_data[:pxg_rapm_ratings]        = rating_of
    F_data[:pxg_rapm_exposure]       = exposure_of
    F_data[:pxg_rapm_scale]          = scale
    F_data[:pxg_rapm_fit_match_ids]  = fit_id_vector
    return nothing
end
