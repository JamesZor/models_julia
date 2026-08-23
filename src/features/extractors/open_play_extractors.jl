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

function add_feature!(F_data::Dict, config::SquadWealthFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    fallback_default = config.fallback_default
    wealth_map = Dict{Int32, Float64}()

    if hasproperty(ds, :lineups) && !isempty(ds.lineups)
        lineups = copy(ds.lineups)
        lineup_sub_col = :is_substitute in propertynames(lineups) ? :is_substitute : (:substitute in propertynames(lineups) ? :substitute : nothing)
        starters = lineup_sub_col !== nothing ? filter(r -> !coalesce(r[lineup_sub_col], false), lineups) : lineups

        pos_medians = Dict("G" => 80_000.0, "D" => 100_000.0, "M" => 110_000.0, "F" => 120_000.0)
        pos_col = :position in propertynames(starters) ? :position : nothing
        val_col = :market_value in propertynames(starters) ? :market_value : (:proposed_market_value in propertynames(starters) ? :proposed_market_value : nothing)

        starters.val = [
            let v = (val_col !== nothing && !ismissing(r[val_col])) ? Float64(r[val_col]) : NaN
                if !isnan(v) && v > 0.0
                    v
                elseif pos_col !== nothing && !ismissing(r[pos_col])
                    get(pos_medians, String(r[pos_col]), fallback_default)
                else
                    fallback_default
                end
            end
            for r in eachrow(starters)
        ]

        team_side_col = :team_side in propertynames(starters) ? :team_side : (:is_home_team in propertynames(starters) ? :is_home_team : :is_home)
        is_home_expr(r) = (team_side_col == :team_side) ? (r.team_side == "home") : Bool(r[team_side_col])
        starters.is_home_bool = is_home_expr.(eachrow(starters))

        home_starters = filter(r -> r.is_home_bool, starters)
        away_starters = filter(r -> !r.is_home_bool, starters)

        if !isempty(home_starters) && !isempty(away_starters)
            home_agg = combine(groupby(home_starters, :match_id), :val => (v -> mean(log.(v))) => :log_w_h)
            away_agg = combine(groupby(away_starters, :match_id), :val => (v -> mean(log.(v))) => :log_w_a)
            joined = innerjoin(home_agg, away_agg, on = :match_id)

            all_log_w = vcat(joined.log_w_h, joined.log_w_a)
            mu_w  = isempty(all_log_w) ? 0.0 : mean(all_log_w)
            std_w = isempty(all_log_w) ? 1.0 : std(all_log_w)
            std_w = std_w == 0.0 || isnan(std_w) ? 1.0 : std_w

            for r in eachrow(joined)
                w_h_std = (r.log_w_h - mu_w) / std_w
                w_a_std = (r.log_w_a - mu_w) / std_w
                wealth_map[Int32(r.match_id)] = Float64(w_h_std - w_a_std)
            end
        end
    end

    flat_delta_w = Float64[get(wealth_map, Int32(id), 0.0) for id in ordered_ids]
    F_data[:flat_delta_wealth] = flat_delta_w
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
