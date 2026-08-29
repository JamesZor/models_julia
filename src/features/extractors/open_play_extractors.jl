# src/features/extractors/open_play_extractors.jl
#
# Feature extractors for:
#   1. Open-Play Goals (Non-Penalty, Non-Own-Goal scores + incident decomposition)
#   2. Open-Play Proxy xG (Zonal Empirical Bayes shot xG with zero-allocation binary masks)
#   3. Starting-XI Squad Wealth Differential (ΔW = W_home - W_away)
#   4. Referee Officiating Indexing & Penalty Whistle Tracking

using DataFrames
using Statistics
using Distributions
using InlineStrings

# ==============================================================================
# 1. OPEN-PLAY GOALS & INCIDENT DECOMPOSITION
# ==============================================================================

function add_feature!(F_data::Dict, ::OpenPlayGoalsFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    matches = copy(ds.matches)
    incidents = copy(ds.incidents)

    pen_scored_h = Dict{Int32, Int}()
    pen_scored_a = Dict{Int32, Int}()
    pen_missed_h = Dict{Int32, Int}()
    pen_missed_a = Dict{Int32, Int}()
    og_for_h     = Dict{Int32, Int}()
    og_for_a     = Dict{Int32, Int}()

    for row in eachrow(incidents)
        m_id = Int32(row.match_id)
        i_type = ismissing(row.incident_type) ? "" : String(row.incident_type)
        i_class = ismissing(row.incident_class) ? "" : String(row.incident_class)
        is_home = coalesce(row.is_home, true)

        if i_type == "goal"
            if i_class == "penalty"
                if is_home
                    pen_scored_h[m_id] = get(pen_scored_h, m_id, 0) + 1
                else
                    pen_scored_a[m_id] = get(pen_scored_a, m_id, 0) + 1
                end
            elseif i_class == "ownGoal"
                if is_home
                    og_for_h[m_id] = get(og_for_h, m_id, 0) + 1
                else
                    og_for_a[m_id] = get(og_for_a, m_id, 0) + 1
                end
            end
        elseif i_type == "inGamePenalty"
            if is_home
                pen_missed_h[m_id] = get(pen_missed_h, m_id, 0) + 1
            else
                pen_missed_a[m_id] = get(pen_missed_a, m_id, 0) + 1
            end
        end
    end

    score_map = Dict(Int32(r.match_id) => (coalesce(r.home_score, 0), coalesce(r.away_score, 0)) for r in eachrow(matches))

    flat_y_open_h = Int[]
    flat_y_open_a = Int[]
    flat_pen_awarded_h = Int[]
    flat_pen_awarded_a = Int[]
    flat_pen_scored_h = Int[]
    flat_pen_scored_a = Int[]
    flat_og_h = Int[]
    flat_og_a = Int[]

    for id in ordered_ids
        m_id = Int32(id)
        raw_h, raw_a = get(score_map, m_id, (0, 0))
        ps_h = get(pen_scored_h, m_id, 0)
        ps_a = get(pen_scored_a, m_id, 0)
        pm_h = get(pen_missed_h, m_id, 0)
        pm_a = get(pen_missed_a, m_id, 0)
        og_h = get(og_for_h, m_id, 0)
        og_a = get(og_for_a, m_id, 0)

        push!(flat_y_open_h, max(0, raw_h - ps_h - og_h))
        push!(flat_y_open_a, max(0, raw_a - ps_a - og_a))
        push!(flat_pen_awarded_h, ps_h + pm_h)
        push!(flat_pen_awarded_a, ps_a + pm_a)
        push!(flat_pen_scored_h, ps_h)
        push!(flat_pen_scored_a, ps_a)
        push!(flat_og_h, og_h)
        push!(flat_og_a, og_a)
    end

    F_data[:flat_y_open_h] = flat_y_open_h
    F_data[:flat_y_open_a] = flat_y_open_a
    F_data[:flat_pen_awarded_h] = flat_pen_awarded_h
    F_data[:flat_pen_awarded_a] = flat_pen_awarded_a
    F_data[:flat_pen_scored_h] = flat_pen_scored_h
    F_data[:flat_pen_scored_a] = flat_pen_scored_a
    F_data[:flat_og_h] = flat_og_h
    F_data[:flat_og_a] = flat_og_a
end

# ==============================================================================
# 2. OPEN-PLAY PROXY xG (pxG) WITH BINARY MASKS
# ==============================================================================

function add_feature!(F_data::Dict, config::OpenPlayPxGFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    all_shots = Features.build_shots(ds)
    pxg_h_map = Dict{Int32, Float64}()
    pxg_a_map = Dict{Int32, Float64}()

    if !isempty(all_shots)
        # Filter out penalty kick events
        open_shots = filter(s -> !s.is_penalty, all_shots)
        if !isempty(open_shots)
            model = Features.fit_shot_xg(open_shots; k = config.k)
            open_shots[!, :pred_xg] = Features.predict_xg(model, open_shots)

            for row in eachrow(open_shots)
                m_id = Int32(row.match_id)
                xg = coalesce(row.pred_xg, 0.0)
                is_home = coalesce(row.is_home, true)
                if is_home
                    pxg_h_map[m_id] = get(pxg_h_map, m_id, 0.0) + xg
                else
                    pxg_a_map[m_id] = get(pxg_a_map, m_id, 0.0) + xg
                end
            end
        end
    end

    flat_pxg_h = Float64[]
    flat_pxg_a = Float64[]
    flat_mask_h = Float64[]
    flat_mask_a = Float64[]

    for id in ordered_ids
        m_id = Int32(id)
        if haskey(pxg_h_map, m_id) && isfinite(pxg_h_map[m_id]) && pxg_h_map[m_id] > 0.0
            push!(flat_pxg_h, max(0.01, pxg_h_map[m_id]))
            push!(flat_mask_h, 1.0)
        else
            push!(flat_pxg_h, 1.0)  # Safe imputed dummy for ReverseDiff static graph
            push!(flat_mask_h, 0.0)
        end

        if haskey(pxg_a_map, m_id) && isfinite(pxg_a_map[m_id]) && pxg_a_map[m_id] > 0.0
            push!(flat_pxg_a, max(0.01, pxg_a_map[m_id]))
            push!(flat_mask_a, 1.0)
        else
            push!(flat_pxg_a, 1.0)  # Safe imputed dummy for ReverseDiff static graph
            push!(flat_mask_a, 0.0)
        end
    end

    F_data[:flat_pxg_h] = flat_pxg_h
    F_data[:flat_pxg_a] = flat_pxg_a
    F_data[:flat_mask_pxg_h] = flat_mask_h
    F_data[:flat_mask_pxg_a] = flat_mask_a
end

# ==============================================================================
# 3. STARTING-XI SQUAD WEALTH DIFFERENTIAL
# ==============================================================================

function _wealth_match_time(row)
    if :start_timestamp in propertynames(row) && !ismissing(row.start_timestamp)
        value = row.start_timestamp
        value isa DateTime && return value
        value isa Date && return DateTime(value)
        try
            return DateTime(String(value))
        catch
        end
    end
    hour = :match_hour in propertynames(row) ? Int(coalesce(row.match_hour, 0)) : 0
    return DateTime(row.match_date) + Hour(hour)
end

function _wealth_side(row, side_col::Symbol)
    raw_side = row[side_col]
    ismissing(raw_side) && return nothing
    if side_col == :team_side
        side = lowercase(String(raw_side))
        side == "home" && return true
        side == "away" && return false
        return nothing
    end
    raw_side isa Bool || return nothing
    return raw_side
end

function _wealth_valid_value(value)
    ismissing(value) && return nothing
    parsed = try
        Float64(value)
    catch
        return nothing
    end
    return isfinite(parsed) && parsed > 0.0 ? parsed : nothing
end

"""Build chronological squad-wealth values and their coverage diagnostics."""
function _build_match_wealth_records(
    lineups::AbstractDataFrame,
    matches::AbstractDataFrame,
    ordered_ids,
    history_ids,
    config::SquadWealthFeature,
)
    config.log_scale === nothing || config.log_scale > 0.0 ||
        throw(ArgumentError("log_scale must be positive"))
    config.decay_half_life_days > 0.0 ||
        throw(ArgumentError("decay_half_life_days must be positive"))
    config.min_valid_players_per_side > 0 ||
        throw(ArgumentError("min_valid_players_per_side must be positive"))

    selected_ids = Set(Int32.(ordered_ids))
    history_set = Set(Int32.(history_ids))
    records = Dict{Int32, NamedTuple}()
    isempty(selected_ids) && return records

    lineup_columns = propertynames(lineups)
    value_col = :market_value in lineup_columns ? :market_value :
                (:proposed_market_value in lineup_columns ? :proposed_market_value : nothing)
    substitute_col = :is_substitute in lineup_columns ? :is_substitute :
                     (:substitute in lineup_columns ? :substitute : nothing)
    side_col = :team_side in lineup_columns ? :team_side :
               (:is_home_team in lineup_columns ? :is_home_team :
                (:is_home in lineup_columns ? :is_home : nothing))

    values = Dict{Tuple{Int32, Bool}, Vector{Float64}}()
    # Without starter metadata we cannot safely distinguish the XI from substitutes.
    if value_col !== nothing && side_col !== nothing && substitute_col !== nothing
        player_col = :player_id in lineup_columns ? :player_id : nothing
        seen_players = Set{Tuple{Int32, Bool, Any}}()
        for row in eachrow(lineups)
            match_id = Int32(row.match_id)
            match_id in selected_ids || continue
            coalesce(row[substitute_col], false) && continue
            side = _wealth_side(row, side_col)
            side === nothing && continue
            value = _wealth_valid_value(row[value_col])
            value === nothing && continue

            if player_col !== nothing && !ismissing(row[player_col])
                player_key = (match_id, side, row[player_col])
                player_key in seen_players && continue
                push!(seen_players, player_key)
            end
            push!(get!(values, (match_id, side), Float64[]), value)
        end
    end

    # The population anchor is fitted only on the history side of the split.
    baseline_logs = Float64[]
    for ((match_id, _), side_values) in values
        match_id in history_set || continue
        append!(baseline_logs, log.(side_values))
    end
    baseline = isempty(baseline_logs) ? 11.46 : mean(baseline_logs)

    match_rows = [row for row in eachrow(matches) if Int32(row.match_id) in selected_ids]
    sort!(match_rows; by=_wealth_match_time)
    last_val = Dict{String, Float64}()
    last_date = Dict{String, Date}()

    raw_records = Dict{Int32, NamedTuple{(:raw_delta, :available, :home_count, :away_count), Tuple{Float64, Float64, Int, Int}}}()

    for row in match_rows
        match_id = Int32(row.match_id)
        match_date = Date(_wealth_match_time(row))
        home_team = String(row.home_team)
        away_team = String(row.away_team)
        home_values = get(values, (match_id, true), Float64[])
        away_values = get(values, (match_id, false), Float64[])
        home_count = length(home_values)
        away_count = length(away_values)

        function team_value(team::String, side_values::Vector{Float64})
            if length(side_values) >= config.min_valid_players_per_side
                observed = mean(log, side_values)
                last_val[team] = observed
                last_date[team] = match_date
                return observed, 1.0
            elseif haskey(last_val, team)
                days = max(0.0, Float64(Dates.value(match_date - last_date[team])))
                weight = 0.5 ^ (days / config.decay_half_life_days)
                return weight * last_val[team] + (1.0 - weight) * baseline, 0.5
            else
                return baseline, 0.0
            end
        end

        home_wealth, home_available = team_value(home_team, home_values)
        away_wealth, away_available = team_value(away_team, away_values)
        raw_delta = Float64(home_wealth - away_wealth)
        # A differential is only as reliable as its least-covered side.
        available = min(home_available, away_available)
        raw_records[match_id] = (raw_delta=raw_delta, available=available,
                                 home_count=home_count, away_count=away_count)
    end

    # Determine scaling denominator: fixed config or fitted over history matches
    scale = if config.log_scale !== nothing
        Float64(config.log_scale)
    else
        history_diffs = [
            r.raw_delta for (mid, r) in raw_records
            if mid in history_set && r.available > 0.0
        ]
        s = length(history_diffs) >= 10 ? std(history_diffs) : 0.50
        (isfinite(s) && s > 1e-4) ? Float64(s) : 0.50
    end
    scale > 0 || error("SquadWealthFeature scale must be positive, got $(scale)")

    for (match_id, r) in raw_records
        records[match_id] = (delta = Float64(r.raw_delta / scale),
                             available = r.available,
                             home_count = r.home_count,
                             away_count = r.away_count)
    end
    return records
end

"""Compatibility helper returning only the match differential lookup."""
function _build_match_wealth_lookup(
    lineups::AbstractDataFrame,
    matches::AbstractDataFrame,
    ordered_ids,
    config::SquadWealthFeature,
)
    records = _build_match_wealth_records(
        lineups, matches, ordered_ids, Int[], config)
    return Dict(id => record.delta for (id, record) in records)
end

function add_feature!(F_data::Dict, config::SquadWealthFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    history_ids = get(F_data, :history_match_ids, Set{Int}())
    records = _build_match_wealth_records(
        ds.lineups, ds.matches, ordered_ids, history_ids, config)

    F_data[:flat_delta_wealth] = Float64[
        get(records, Int32(id), (delta=0.0,)).delta for id in ordered_ids]
    F_data[:flat_wealth_available] = Float64[
        get(records, Int32(id), (available=0.0,)).available for id in ordered_ids]
    F_data[:flat_wealth_home_count] = Int[
        get(records, Int32(id), (home_count=0,)).home_count for id in ordered_ids]
    F_data[:flat_wealth_away_count] = Int[
        get(records, Int32(id), (away_count=0,)).away_count for id in ordered_ids]
    F_data[:wealth_by_match_id] = Dict(
        Int32(id) => F_data[:flat_delta_wealth][i] for (i, id) in enumerate(ordered_ids))
end

# ==============================================================================
# 4. REFEREE OFFICIATING INDEXING
# ==============================================================================

function add_feature!(F_data::Dict, ::RefereeOfficiatingFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    ref_map = Dict{Int32, Int}()

    # Check matches for referee info
    if hasproperty(ds.matches, :referee_id)
        raw_refs = unique(filter(x -> !ismissing(x) && x > 0, ds.matches.referee_id))
        ref_dict = Dict(r => idx for (idx, r) in enumerate(raw_refs))
        for r in eachrow(ds.matches)
            ref_id = coalesce(r.referee_id, 0)
            if haskey(ref_dict, ref_id)
                ref_map[Int32(r.match_id)] = ref_dict[ref_id]
            end
        end
        n_refs = max(1, length(raw_refs))
    else
        n_refs = 1
    end

    flat_ref_ids = Int[get(ref_map, Int32(id), 1) for id in ordered_ids]
    F_data[:flat_referee_ids] = flat_ref_ids
    F_data[:n_referees] = n_refs
end
