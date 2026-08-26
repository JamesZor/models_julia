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

function _wealth_kickoff_map(matches::AbstractDataFrame)
    result = Dict{Int32, DateTime}()
    for row in eachrow(matches)
        kickoff = if :start_timestamp in propertynames(row)
            DateTime(row.start_timestamp)
        elseif :match_date in propertynames(row)
            hour = :match_hour in propertynames(row) ? Int(coalesce(row.match_hour, 0)) : 0
            DateTime(row.match_date) + Hour(hour)
        else
            continue
        end
        result[Int32(row.match_id)] = kickoff
    end
    return result
end

function _wealth_as_datetime(value)
    ismissing(value) && return nothing
    value isa DateTime && return value
    value isa Date && return DateTime(value)
    try
        return DateTime(String(value))
    catch
        return nothing
    end
end

function _wealth_is_home(row, side_col::Symbol)
    side_col == :team_side && return lowercase(String(row[side_col])) == "home"
    return Bool(row[side_col])
end

"""Build a fold-local, point-in-time starting-XI wealth lookup."""
function _build_match_wealth_lookup(
    lineups::AbstractDataFrame,
    matches::AbstractDataFrame,
    ordered_ids,
    config::SquadWealthFeature,
)
    config.fallback_default > 0.0 || throw(ArgumentError("fallback_default must be positive"))
    config.log_scale > 0.0 || throw(ArgumentError("log_scale must be positive"))

    selected_ids = Set(Int32.(ordered_ids))
    wealth_map = Dict{Int32, Float64}()
    isempty(selected_ids) && return wealth_map
    isempty(lineups) && return wealth_map

    lineup_columns = propertynames(lineups)
    value_col = :market_value in lineup_columns ? :market_value :
                (:proposed_market_value in lineup_columns ? :proposed_market_value : nothing)
    value_col === nothing && return wealth_map

    timestamp_col = if :valuation_timestamp in lineup_columns
        :valuation_timestamp
    elseif :market_value_timestamp in lineup_columns
        :market_value_timestamp
    elseif :valuation_date in lineup_columns
        :valuation_date
    else
        nothing
    end
    config.require_valuation_timestamp && timestamp_col === nothing && return wealth_map

    substitute_col = :is_substitute in lineup_columns ? :is_substitute :
                     (:substitute in lineup_columns ? :substitute : nothing)
    side_col = :team_side in lineup_columns ? :team_side :
               (:is_home_team in lineup_columns ? :is_home_team :
                (:is_home in lineup_columns ? :is_home : nothing))
    side_col === nothing && return wealth_map

    kickoffs = _wealth_kickoff_map(matches)
    values = Dict{Tuple{Int32, Bool}, Vector{Float64}}()
    known = Dict{Tuple{Int32, Bool}, Int}()

    for row in eachrow(lineups)
        match_id = Int32(row.match_id)
        match_id in selected_ids || continue
        substitute_col !== nothing && coalesce(row[substitute_col], false) && continue

        is_home = _wealth_is_home(row, side_col)
        key = (match_id, is_home)
        side_values = get!(values, key, Float64[])

        raw_value = row[value_col]
        market_value = if ismissing(raw_value)
            NaN
        else
            try
                Float64(raw_value)
            catch
                NaN
            end
        end

        timestamp_safe = if timestamp_col === nothing
            !config.require_valuation_timestamp
        else
            valuation_time = _wealth_as_datetime(row[timestamp_col])
            kickoff = get(kickoffs, match_id, nothing)
            valuation_time !== nothing && kickoff !== nothing && valuation_time < kickoff
        end

        if isfinite(market_value) && market_value > 0.0 && timestamp_safe
            push!(side_values, market_value)
            known[key] = get(known, key, 0) + 1
        else
            push!(side_values, config.fallback_default)
        end
    end

    for match_id in selected_ids
        home_key = (match_id, true)
        away_key = (match_id, false)
        home_values = get(values, home_key, Float64[])
        away_values = get(values, away_key, Float64[])

        # If either club has no safe point-in-time valuation, treat the match as
        # unmapped and return the neutral population fallback.
        isempty(home_values) && continue
        isempty(away_values) && continue
        get(known, home_key, 0) > 0 || continue
        get(known, away_key, 0) > 0 || continue

        delta = (mean(log, home_values) - mean(log, away_values)) / config.log_scale
        isfinite(delta) && (wealth_map[match_id] = Float64(delta))
    end
    return wealth_map
end

function add_feature!(F_data::Dict, config::SquadWealthFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    wealth_map = _build_match_wealth_lookup(ds.lineups, ds.matches, ordered_ids, config)
    flat_delta_w = Float64[get(wealth_map, Int32(id), 0.0) for id in ordered_ids]
    flat_fallback = Int[haskey(wealth_map, Int32(id)) ? 0 : 1 for id in ordered_ids]

    F_data[:flat_delta_wealth] = flat_delta_w
    F_data[:flat_wealth_fallback] = flat_fallback
    F_data[:wealth_by_match_id] = Dict(
        Int32(id) => flat_delta_w[i] for (i, id) in enumerate(ordered_ids))
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
