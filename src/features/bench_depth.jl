# src/features/bench_depth.jl
#
# Point-in-time substitute-bench valuation differential. Unlike the starting-XI
# wealth features, this extractor deliberately reads only `is_substitute == true`.

using DataFrames
using Dates
using Statistics

"""
    BenchDepthFeature(; log_transform=true, scale=nothing, min_bench_count=3)

Pre-match substitute-bench valuation differential. Each side's bench is the sum
of positive player market values known before kickoff. With `log_transform=true`,
that total is mapped to `log(1 + value/1000)` before taking home minus away.

When `scale === nothing`, the differential is divided by the standard deviation
estimated only from `F_data[:history_match_ids]`. Sparse or unavailable benches
emit exactly `0.0`.
"""
Base.@kwdef struct BenchDepthFeature <: AbstractFeatureConfig
    log_transform::Bool = true
    scale::Union{Float64,Nothing} = nothing
    min_bench_count::Int = 3
end

function _bench_validate(config::BenchDepthFeature)
    config.min_bench_count > 0 || error("BenchDepthFeature.min_bench_count must be > 0")
    config.scale === nothing ||
        (isfinite(config.scale) && config.scale > 0.0) ||
        error("BenchDepthFeature.scale must be nothing or finite and > 0")
    return nothing
end

function _bench_datetime(value)
    (ismissing(value) || value === nothing) && return nothing
    value isa DateTime && return value
    value isa Date && return DateTime(value)
    if hasproperty(value, :zone)
        return DateTime(value, Dates.UTC)
    end
    return tryparse(DateTime, string(value))
end

function _bench_kickoffs(matches::AbstractDataFrame)
    out = Dict{Int,DateTime}()
    columns = propertynames(matches)
    for row in eachrow(matches)
        match_id = Int(row.match_id)
        if :start_timestamp in columns
            stamp = _bench_datetime(row.start_timestamp)
            if stamp !== nothing
                out[match_id] = stamp
                continue
            end
        end
        if :match_date in columns && !ismissing(row.match_date)
            date = row.match_date isa Date ? row.match_date : Date(string(row.match_date))
            hour = (:match_hour in columns && !ismissing(row.match_hour)) ?
                   clamp(Int(row.match_hour), 0, 23) : 0
            out[match_id] = DateTime(date) + Hour(hour)
        end
    end
    return out
end

function _bench_side(row, columns)
    if :team_side in columns
        ismissing(row.team_side) && return nothing
        side = lowercase(String(row.team_side))
        side == "home" && return true
        side == "away" && return false
    elseif :is_home_team in columns
        ismissing(row.is_home_team) || return Bool(row.is_home_team)
    elseif :is_home in columns
        ismissing(row.is_home) || return Bool(row.is_home)
    end
    return nothing
end

function _bench_raw_lookup(lineups::AbstractDataFrame, matches::AbstractDataFrame,
                           ids, config::BenchDepthFeature)
    wanted = Set(Int.(ids))
    isempty(wanted) && return Dict{Int,Float64}()
    nrow(lineups) == 0 && return Dict{Int,Float64}()

    columns = propertynames(lineups)
    (:match_id in columns && :is_substitute in columns) ||
        return Dict{Int,Float64}()
    value_column = :proposed_market_value in columns ? :proposed_market_value :
                   (:market_value in columns ? :market_value : nothing)
    value_column === nothing && return Dict{Int,Float64}()

    totals = Dict{Tuple{Int,Bool},Float64}()
    counts = Dict{Tuple{Int,Bool},Int}()
    kickoffs = _bench_kickoffs(matches)

    for row in eachrow(lineups)
        match_id = Int(row.match_id)
        match_id in wanted || continue
        coalesce(row.is_substitute, false) || continue
        side = _bench_side(row, columns)
        side === nothing && continue

        raw = getproperty(row, value_column)
        value = if ismissing(raw) || raw === nothing
            nothing
        else
            try
                Float64(raw)
            catch
                nothing
            end
        end
        value === nothing && continue
        isfinite(value) && value > 0.0 || continue

        valuation_stamp = :valuation_timestamp in columns ?
                          _bench_datetime(row.valuation_timestamp) : nothing
        kickoff = get(kickoffs, match_id, nothing)
        stamp_ok = valuation_stamp === nothing || kickoff === nothing ||
                   valuation_stamp < kickoff
        stamp_ok || continue

        key = (match_id, side)
        totals[key] = get(totals, key, 0.0) + value
        counts[key] = get(counts, key, 0) + 1
    end

    out = Dict{Int,Float64}()
    for match_id in wanted
        home_key = (match_id, true)
        away_key = (match_id, false)
        get(counts, home_key, 0) >= config.min_bench_count || continue
        get(counts, away_key, 0) >= config.min_bench_count || continue
        home = totals[home_key]
        away = totals[away_key]
        delta = if config.log_transform
            log1p(home / 1000.0) - log1p(away / 1000.0)
        else
            home - away
        end
        isfinite(delta) && (out[match_id] = delta)
    end
    return out
end

function _bench_scale(raw::Dict{Int,Float64}, fit_ids, config::BenchDepthFeature)
    config.scale === nothing || return config.scale
    values = Float64[get(raw, Int(id), NaN) for id in fit_ids]
    filter!(isfinite, values)
    length(values) >= 2 || return 1.0
    sigma = std(values)
    return isfinite(sigma) && sigma > sqrt(eps(Float64)) ? sigma : 1.0
end

"""Add the point-in-time bench-depth design column to a feature dictionary."""
function add_feature!(F_data::Dict, config::BenchDepthFeature, ordered_ids,
                      team_map::Dict, ds::Data.DataStore)
    _bench_validate(config)
    all_ids = :match_id in propertynames(ds.matches) ?
              Int.(ds.matches.match_id) : Int.(ordered_ids)
    raw = _bench_raw_lookup(ds.lineups, ds.matches, all_ids, config)
    fit_ids = haskey(F_data, :history_match_ids) ? F_data[:history_match_ids] : ordered_ids
    scale = _bench_scale(raw, fit_ids, config)

    bridge = Dict{Int,Float64}(match_id => value / scale for (match_id, value) in raw)
    F_data[:flat_delta_bench_depth] = Float64[
        get(bridge, Int(match_id), 0.0) for match_id in ordered_ids]
    F_data[:flat_bench_depth_fallback] = Int[
        haskey(bridge, Int(match_id)) ? 0 : 1 for match_id in ordered_ids]
    F_data[:bench_depth_by_match_id] = bridge
    F_data[:bench_depth_scale] = scale
    return nothing
end
