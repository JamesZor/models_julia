# l02_portfolio_backtest.jl

using BayesianFootball
using DataFrames, Dates, Statistics, Distributions
using Optim, LinearAlgebra

const BF = BayesianFootball
const D  = BF.Data
const F  = BF.Features
const M  = BF.Models.PreGame
const E  = BF.Experiments
const EV = BF.Evaluation

# ===================================================================
# 1. Configuration & Data Structures
# ===================================================================

Base.@kwdef struct BacktestConfig
    commission::Float64 = 0.02
    alphas::Vector{Float64} = [1.0, 0.7, 0.5, 0.3, 0.0]
end

struct SelData
    market::String
    selection::Symbol
    odds::Float64
    p_model::Float64
    p_fair_market::Float64
    p_blended::Float64
    edge_blended::Float64
    is_winner::Bool
end

# ===================================================================
# 2. Market Extraction & Normalization
# ===================================================================

function extract_market_data(odds_df::DataFrame, match_id::Int, markets_config)
    row_odds = subset(odds_df, :match_id => ByRow(==(match_id)))
    odds_map = Dict{String, Dict{Symbol, Float64}}()
    fair_prob_map = Dict{String, Dict{Symbol, Float64}}()
    winner_map = Dict{String, Dict{Symbol, Bool}}()
    
    for m in markets_config.markets
        m_name_str = string(m)
        m_group    = D.market_group(m)
        m_line     = D.market_line(m)
        
        m_df = subset(row_odds, 
            :market_name => ByRow(==(m_group)),
            :market_line => ByRow(l -> isapprox(l, m_line; atol=1e-3))
        )
        
        if !isempty(m_df)
            o_dict = Dict{Symbol, Float64}()
            f_dict = Dict{Symbol, Float64}()
            w_dict = Dict{Symbol, Bool}()
            for r in eachrow(m_df)
                if hasproperty(r, :odds_close) && !ismissing(r.odds_close) && !ismissing(r.prob_fair_close)
                    o_dict[r.selection] = r.odds_close
                    f_dict[r.selection] = r.prob_fair_close
                    if hasproperty(r, :is_winner) && !ismissing(r.is_winner)
                        w_dict[r.selection] = Bool(r.is_winner)
                    end
                end
            end
            if !isempty(o_dict)
                odds_map[m_name_str]      = o_dict
                fair_prob_map[m_name_str] = f_dict
                winner_map[m_name_str]    = w_dict
            end
        end
    end
    
    return odds_map, fair_prob_map, winner_map
end

function normalize_market_group_odds(odds_map::Dict{String, Dict{Symbol, Float64}})
    norm_odds_map = Dict{String, Dict{Symbol, Float64}}()
    fair_prob_map = Dict{String, Dict{Symbol, Float64}}()
    
    for (m_name, outcome_dict) in odds_map
        O = sum(1.0 / odds_val for (sel, odds_val) in outcome_dict)
        
        n_o_dict = Dict{Symbol, Float64}()
        f_p_dict = Dict{Symbol, Float64}()
        
        for (sel, odds_val) in outcome_dict
            p_fair = (1.0 / odds_val) / O
            d_norm = 1.0 / p_fair
            
            n_o_dict[sel] = d_norm
            f_p_dict[sel] = p_fair
        end
        
        norm_odds_map[m_name] = n_o_dict
        fair_prob_map[m_name] = f_p_dict
    end
    
    return norm_odds_map, fair_prob_map
end

# ===================================================================
# 3. Core Portfolio Optimization Engine
# ===================================================================

function optimize_portfolio(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, config::BacktestConfig; alpha::Float64=1.0)
    max_h, max_a, n_samples = size(score_matrix.data)
    N = max_h * max_a # N = 144
    
    P_model_grid = mean(score_matrix.data, dims=3)[:, :, 1]
    p_model_vec  = vec(P_model_grid)
    
    selections = SelData[]
    win_masks  = Matrix{Bool}[]
    
    for (m_name, outcome_dict) in match_model_prob
        if haskey(odds_map, m_name)
            for (sel, dist) in outcome_dict
                if haskey(odds_map[m_name], sel)
                    odds_val   = odds_map[m_name][sel]
                    p_fair_mkt = fair_prob_map[m_name][sel]
                    p_mod      = mean(dist)
                    
                    is_win     = haskey(winner_map, m_name) && haskey(winner_map[m_name], sel) ? winner_map[m_name][sel] : false
                    
                    p_blend    = alpha * p_mod + (1.0 - alpha) * p_fair_mkt
                    edge_blend = (p_blend * odds_val) - 1.0
                    
                    win_mask = zeros(Bool, max_h, max_a)
                    for c in 1:max_a, r in 1:max_h
                        h_g, a_g = r - 1, c - 1
                        if m_name == "Market[1X2]"
                            if sel == :home && h_g > a_g win_mask[r, c] = true end
                            if sel == :draw && h_g == a_g win_mask[r, c] = true end
                            if sel == :away && h_g < a_g win_mask[r, c] = true end
                        elseif m_name == "Market[BTTS]"
                            if sel == :btts_yes && (h_g > 0 && a_g > 0) win_mask[r, c] = true end
                            if sel == :btts_no  && !(h_g > 0 && a_g > 0) win_mask[r, c] = true end
                        elseif startswith(m_name, "Market[O/U")
                            line = parse(Float64, split(m_name)[2][1:end-1])
                            tot = h_g + a_g
                            if startswith(string(sel), "over") && tot > line win_mask[r, c] = true end
                            if startswith(string(sel), "under") && tot < line win_mask[r, c] = true end
                        end
                    end
                    
                    push!(selections, SelData(m_name, sel, odds_val, p_mod, p_fair_mkt, p_blend, edge_blend, is_win))
                    push!(win_masks, win_mask)
                end
            end
        end
    end
    
    if isempty(selections)
        return selections, Float64[]
    end

    n_m = length(selections)
    B = zeros(Float64, n_m, N)
    d = zeros(Float64, n_m)
    for i in 1:n_m
        B[i, :] = vec(win_masks[i])
        d[i]    = selections[i].odds
    end
    
    d_net_comm = 1.0 .+ (1.0 - config.commission) .* (d .- 1.0)
    R = (B' .* d_net_comm') .- 1.0
    
    function obj_f(a)
        if sum(a) >= 0.99 || any(a .< 0.0) return Inf end
        w = 1.0 .+ (R * a)
        if any(w .<= 1e-8) return Inf end
        return -dot(p_model_vec, log.(w))
    end
    
    function grad_g!(g_stor, a)
        w = 1.0 .+ (R * a)
        if any(w .<= 1e-8) g_stor .= 0.0; return end
        g_stor .= -(R' * (p_model_vec ./ w))
    end
    
    lower_b = zeros(n_m)
    upper_b = fill(0.50, n_m)
    init_a  = fill(0.001, n_m)
    
    res = optimize(obj_f, grad_g!, lower_b, upper_b, init_a, Fminbox(LBFGS()))
    a_naive = Optim.minimizer(res)
    a_naive[a_naive .< 1e-4] .= 0.0
    
    # Market Netting Pass
    a_net = copy(a_naive)
    market_groups_dict = Dict{String, Vector{Int}}()
    for (i, sel) in enumerate(selections)
        push!(get!(market_groups_dict, sel.market, Int[]), i)
    end
    
    for (m_name, idxs) in market_groups_dict
        if length(idxs) == 2
            i1, i2 = idxs[1], idxs[2]
            if a_net[i1] > 0 && a_net[i2] > 0
                net_diff = a_net[i1] - a_net[i2]
                a_net[i1] = max(0.0, net_diff)
                a_net[i2] = max(0.0, -net_diff)
            end
        elseif length(idxs) == 3
            min_stake = minimum(a_net[idxs])
            if min_stake > 0
                for idx in idxs
                    a_net[idx] -= min_stake
                end
            end
        end
    end
    
    a_blended_net = a_net .* alpha
    return selections, a_blended_net, R
end

# ===================================================================
# 4. Evaluation and Backtest Orchestration
# ===================================================================

function evaluate_match(latents_row, expr, odds_df, markets_config, config::BacktestConfig)
    m_id = latents_row.match_id
    
    # Pre-allocate return structure
    res_df = DataFrame(
        match_id    = Int[],
        alpha       = Float64[],
        total_stake = Float64[],
        net_pl      = Float64[],
        bets_placed = Int[],
        max_edge    = Float64[],
        status      = String[],
        placed_bets = String[]
    )
    
    # 1. Fast-Fail if no odds
    raw_odds_map, _, winner_map = extract_market_data(odds_df, m_id, markets_config)
    if isempty(raw_odds_map)
        for α in config.alphas
            push!(res_df, (m_id, α, 0.0, 0.0, 0, 0.0, "MISSING_ODDS", ""))
        end
        return res_df
    end
    
    odds_map, fair_prob_map = normalize_market_group_odds(raw_odds_map)
    
    # 2. Compute Score Matrix (Expensive part, only run if odds exist)
    param = Predictions.extract_params(expr.config.model, latents_row)
    
    # Try/catch for solver or generation failures
    score_matrix = nothing
    try
        score_matrix = Predictions.compute_score_matrix(expr.config.model, param)
    catch e
        for α in config.alphas
            push!(res_df, (m_id, α, 0.0, 0.0, 0, 0.0, "SOLVER_FAIL", ""))
        end
        return res_df
    end
    
    match_model_prob = Dict(
        string(m) => Predictions.compute_market_probs(score_matrix, m)
        for m in markets_config.markets
    )
    
    # 3. Optimize for each alpha
    for α in config.alphas
        selections, stakes, R_mat = optimize_portfolio(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, config; alpha=α)
        
        if isempty(selections)
            push!(res_df, (m_id, α, 0.0, 0.0, 0, 0.0, "NO_SELECTIONS", ""))
            continue
        end
        
        tot_st = sum(stakes)
        net_pl = 0.0
        bets_placed = 0
        max_e = -Inf
        placed_bets_arr = String[]
        
        for i in 1:length(selections)
            sel = selections[i]
            st = stakes[i]
            
            # Track max edge (only considering positive edges or all?)
            # We track max edge found in the match regardless of if we bet, to analyze potential
            if sel.edge_blended > max_e
                max_e = sel.edge_blended
            end
            
            if st > 0
                bets_placed += 1
                if sel.is_winner
                    net_pl += st * (1.0 - config.commission) * (sel.odds - 1.0)
                else
                    net_pl -= st
                end
                
                push!(placed_bets_arr, "$(sel.market):$(sel.selection)($(round(st*100, digits=1))%)")
            end
        end
        
        placed_bets_str = join(placed_bets_arr, ", ")
        push!(res_df, (m_id, α, tot_st, net_pl, bets_placed, max_e == -Inf ? 0.0 : max_e, "SUCCESS", placed_bets_str))
    end
    
    return res_df
end

"""
    run_backtest(latents_df, expr, odds_df, markets_config, config::BacktestConfig)

Multithreaded execution over a DataFrame of matches.
Returns a long-format DataFrame with results.
"""
function run_backtest(latents_df::DataFrame, expr, odds_df::DataFrame, markets_config, config::BacktestConfig)
    n_matches = nrow(latents_df)
    results = Vector{DataFrame}(undef, n_matches)
    
    Threads.@threads for i in 1:n_matches
        results[i] = evaluate_match(latents_df[i, :], expr, odds_df, markets_config, config)
    end
    
    return vcat(results...)
end
