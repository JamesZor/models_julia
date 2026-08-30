# src/features/late_game.jl
#
# Point-in-time rolling share of proxy xG generated from minute 70 onward.

using DataFrames
using Statistics

"""
    LateGameChanceFeature(; minute_threshold=70, half_life_matches=16.0, scale=nothing)

For each team, exponentially smooth the historical share of commentary-derived
proxy xG generated at or after `minute_threshold`. The emitted design is the
home-minus-away rolling share, standardized on the frozen history block. Teams
without measured history and fixtures without both sides emit exactly `0.0`.
"""
Base.@kwdef struct LateGameChanceFeature <: AbstractFeatureConfig
    minute_threshold::Int = 70
    half_life_matches::Float64 = 16.0
    scale::Union{Float64,Nothing} = nothing
end

function _late_validate(config::LateGameChanceFeature)
    0 <= config.minute_threshold <= 130 ||
        error("LateGameChanceFeature.minute_threshold must be between 0 and 130")
    isfinite(config.half_life_matches) && config.half_life_matches > 0.0 ||
        error("LateGameChanceFeature.half_life_matches must be finite and > 0")
    config.scale === nothing ||
        (isfinite(config.scale) && config.scale > 0.0) ||
        error("LateGameChanceFeature.scale must be nothing or finite and > 0")
    return nothing
end

function _late_weighted_mean(values::Vector{Float64}, half_life::Float64)
    isempty(values) && return 0.0
    numerator = 0.0
    denominator = 0.0
    log_two = log(2.0)
    n = length(values)
    for lag in 0:(n - 1)
        weight = exp(-log_two * lag / half_life)
        numerator += weight * values[n - lag]
        denominator += weight
    end
    return denominator > 0.0 ? numerator / denominator : 0.0
end

"""
    _late_match_proportions(ds, config; fit_ids=nothing)

Per-match, per-side proxy-xG shares. The shot-cell model is fitted only on
`fit_ids` when supplied; an empty permitted set uses the model's fixed neutral
cell rates rather than learning from future outcomes.
"""
function _late_match_proportions(ds::Data.DataStore, config::LateGameChanceFeature;
                                 fit_ids::Union{Nothing,Set{Int}}=nothing)
    shots = build_shots(ds)
    out = Dict{Int,NamedTuple{(:home, :away, :home_available, :away_available),
                              Tuple{Float64,Float64,Bool,Bool}}}()
    nrow(shots) == 0 && return out

    fit_rows = fit_ids === nothing ? shots :
               shots[in.(Int.(shots.match_id), Ref(fit_ids)), :]
    model = fit_shot_xg(fit_rows; k=25.0)
    predicted = predict_xg(model, shots)

    totals = Dict{Tuple{Int,Bool},Float64}()
    late_totals = Dict{Tuple{Int,Bool},Float64}()
    for (index, shot) in enumerate(eachrow(shots))
        ismissing(shot.is_home) && continue
        ismissing(shot.time) && continue
        raw_xg = Float64(predicted[index])
        isfinite(raw_xg) && raw_xg >= 0.0 || continue
        # A tiny all-miss fitting block can give the empirical cell table a zero
        # base rate. Preserve the observed shot and a well-defined proportion.
        xg = max(raw_xg, eps(Float64))
        key = (Int(shot.match_id), shot.is_home === true)
        totals[key] = get(totals, key, 0.0) + xg
        if Float64(shot.time) >= config.minute_threshold
            late_totals[key] = get(late_totals, key, 0.0) + xg
        end
    end

    match_ids = Set(first(key) for key in keys(totals))
    for match_id in match_ids
        home_total = get(totals, (match_id, true), 0.0)
        away_total = get(totals, (match_id, false), 0.0)
        home_available = home_total > 0.0
        away_available = away_total > 0.0
        home = home_available ? get(late_totals, (match_id, true), 0.0) / home_total : 0.0
        away = away_available ? get(late_totals, (match_id, false), 0.0) / away_total : 0.0
        out[match_id] = (; home, away, home_available, away_available)
    end
    return out
end

function _late_rolling_lookup(observations::Dict{Int,<:NamedTuple},
                              matches::AbstractDataFrame,
                              config::LateGameChanceFeature)
    out = Dict{Int,NamedTuple{(:delta, :available),Tuple{Float64,Bool}}}()
    nrow(matches) == 0 && return out

    rows = [(id=Int(row.match_id), kickoff=_pxg_kickoff(row),
             home=String(row.home_team), away=String(row.away_team))
            for row in eachrow(matches)]
    sort!(rows, by=row -> (row.kickoff, row.id))
    histories = Dict{String,Vector{Float64}}()

    i = 1
    while i <= length(rows)
        j = i
        kickoff_day = Date(rows[i].kickoff)
        while j <= length(rows) && Date(rows[j].kickoff) == kickoff_day
            j += 1
        end

        # Emit the entire calendar-day group before updating any team history.
        for index in i:(j - 1)
            row = rows[index]
            home_history = get(histories, row.home, Float64[])
            away_history = get(histories, row.away, Float64[])
            available = !isempty(home_history) && !isempty(away_history)
            delta = available ?
                _late_weighted_mean(home_history, config.half_life_matches) -
                _late_weighted_mean(away_history, config.half_life_matches) : 0.0
            out[row.id] = (; delta, available)
        end

        for index in i:(j - 1)
            row = rows[index]
            observation = get(observations, row.id, nothing)
            observation === nothing && continue
            observation.home_available &&
                push!(get!(histories, row.home, Float64[]), observation.home)
            observation.away_available &&
                push!(get!(histories, row.away, Float64[]), observation.away)
        end
        i = j
    end
    return out
end

function _late_scale(lookup::Dict{Int,<:NamedTuple}, fit_ids,
                     config::LateGameChanceFeature)
    config.scale === nothing || return config.scale
    values = Float64[]
    for id in fit_ids
        record = get(lookup, Int(id), nothing)
        (record === nothing || !record.available) && continue
        push!(values, record.delta)
    end
    length(values) >= 2 || return 1.0
    sigma = std(values)
    return isfinite(sigma) && sigma > sqrt(eps(Float64)) ? sigma : 1.0
end

"""Add the rolling late-game chance-share differential to a feature dictionary."""
function add_feature!(F_data::Dict, config::LateGameChanceFeature, ordered_ids,
                      team_map::Dict, ds::Data.DataStore)
    _late_validate(config)
    fit_ids = haskey(F_data, :history_match_ids) ?
              Set(Int.(F_data[:history_match_ids])) : nothing
    observations = _late_match_proportions(ds, config; fit_ids)
    lookup = _late_rolling_lookup(observations, ds.matches, config)
    scale_ids = fit_ids === nothing ? ordered_ids : fit_ids
    scale = _late_scale(lookup, scale_ids, config)

    bridge = Dict{Int,Float64}(
        match_id => record.delta / scale for (match_id, record) in lookup if record.available)
    F_data[:flat_delta_late_game_chance] = Float64[
        get(bridge, Int(match_id), 0.0) for match_id in ordered_ids]
    F_data[:flat_late_game_chance_fallback] = Int[
        haskey(bridge, Int(match_id)) ? 0 : 1 for match_id in ordered_ids]
    F_data[:late_game_chance_by_match_id] = bridge
    F_data[:late_game_chance_scale] = scale
    F_data[:late_game_proportions_by_match_id] = observations
    return nothing
end
