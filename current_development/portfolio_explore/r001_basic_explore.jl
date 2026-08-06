using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Distributions
using ThreadPinning
using Optim, LinearAlgebra

pinthreads(:cores)
@info "threads" n = Threads.nthreads() cores = ThreadPinning.ncores()

const BF = BayesianFootball
const D  = BF.Data
const F  = BF.Features
const M  = BF.Models.PreGame
const E  = BF.Experiments
const EV = BF.Evaluation

# ===================================================================
# SECTION 1: Structs and Core Linear Algebra Helper Functions
# ===================================================================
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

"""
Extracts odds, fair probabilities, and is_winner flags for all markets in markets_config for a given match.
"""
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

"""
Normalizes raw Betfair odds for each market group to ensure probabilities sum to exactly 1.0.
"""
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

"""
Compares Betfair Exchange Window Odds vs Bookmaker Synchronous Odds (ds.odds)
"""
function compare_odds_sources(betfair_df::DataFrame, bookmaker_df::DataFrame, match_id::Int, markets_config)
    b_map, b_fair, _   = extract_market_data(betfair_df, match_id, markets_config)
    bm_map, bm_fair, _ = extract_market_data(bookmaker_df, match_id, markets_config)
    
    println("\n", "="^95)
    println("=== ODDS QA & COMPARISON: BETFAIR WINDOW vs BOOKMAKER SYNCHRONOUS ODDS (Match #$(match_id)) ===")
    println("="^95)
    println(rpad("Market", 16), rpad("Selection", 12), rpad("Betfair Odds", 14), rpad("Bookmaker Odds", 16), rpad("Discrepancy", 14))
    println("-"^95)
    
    for (m_name, b_outcomes) in b_map
        if haskey(bm_map, m_name)
            for (sel, b_odds) in b_outcomes
                if haskey(bm_map[m_name], sel)
                    bm_odds = bm_map[m_name][sel]
                    diff_pct = round(((b_odds - bm_odds) / bm_odds) * 100, digits=2)
                    flag = abs(diff_pct) > 15.0 ? "⚠️ WILD DISCREPANCY" : "OK"
                    
                    println(
                        rpad(m_name, 16),
                        rpad(string(sel), 12),
                        rpad(round(b_odds; digits=2), 14),
                        rpad(round(bm_odds; digits=2), 16),
                        rpad("$(diff_pct)% ($(flag))", 14)
                    )
                end
            end
        end
    end
    
    if haskey(b_map, "Market[1X2]") && haskey(bm_map, "Market[1X2]")
        b_1x2  = b_map["Market[1X2]"]
        bm_1x2 = bm_map["Market[1X2]"]
        
        b_ov  = (1.0/b_1x2[:home]) + (1.0/b_1x2[:draw]) + (1.0/b_1x2[:away])
        bm_ov = (1.0/bm_1x2[:home]) + (1.0/bm_1x2[:draw]) + (1.0/bm_1x2[:away])
        
        println("-"^95)
        println("Betfair 1X2 Implied Overround : ", round(b_ov; digits=4), (b_ov < 1.0 ? " 🚨 ARTIFICIAL ARBITRAGE (Window Artifact)" : " OK"))
        println("Bookmaker 1X2 Implied Overround: ", round(bm_ov; digits=4), " (Normal Vig)")
    end
    println("="^95)
end

"""
Section 20 & 6.2 Analytical Newton Portfolio Solver.
Includes 2% Betfair Exchange Commission and Market Netting Pass.
"""
function optimize_portfolio(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map; alpha::Float64=1.0)
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
    
    n_m = length(selections)
    B = zeros(Float64, n_m, N)
    d = zeros(Float64, n_m)
    for i in 1:n_m
        B[i, :] = vec(win_masks[i])
        d[i]    = selections[i].odds
    end
    
    # Betfair 2% Exchange Commission on Net Profit
    commission = 0.02
    d_net_comm = 1.0 .+ (1.0 - commission) .* (d .- 1.0)
    
    # Return Matrix R (144 x n_m) incorporating 2% commission
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
    
    function hess_h!(h_stor, a)
        w = 1.0 .+ (R * a)
        if any(w .<= 1e-8) h_stor .= 0.0; return end
        h_stor .= R' * ((p_model_vec ./ (w .^ 2)) .* R)
    end
    
    lower_b = zeros(n_m)
    upper_b = fill(0.50, n_m)
    init_a  = fill(0.001, n_m)
    
    res = optimize(obj_f, grad_g!, lower_b, upper_b, init_a, Fminbox(LBFGS()))
    a_naive = Optim.minimizer(res)
    
    # Zero out tiny boundary residual noise (< 0.01% stake)
    a_naive[a_naive .< 1e-4] .= 0.0
    
    # -------------------------------------------------------------------
    # Market Netting Pass (Enforce Binary Exclusivity per Market Line)
    # -------------------------------------------------------------------
    a_net = copy(a_naive)
    market_groups_dict = Dict{String, Vector{Int}}()
    for (i, sel) in enumerate(selections)
        push!(get!(market_groups_dict, sel.market, Int[]), i)
    end
    
    for (m_name, idxs) in market_groups_dict
        if length(idxs) == 2 # Binary market (Over/Under or BTTS)
            i1, i2 = idxs[1], idxs[2]
            if a_net[i1] > 0 && a_net[i2] > 0
                net_diff = a_net[i1] - a_net[i2]
                a_net[i1] = max(0.0, net_diff)
                a_net[i2] = max(0.0, -net_diff)
            end
        elseif length(idxs) == 3 # 1X2 market (Home, Draw, Away)
            min_stake = minimum(a_net[idxs])
            if min_stake > 0
                for idx in idxs
                    a_net[idx] -= min_stake
                end
            end
        end
    end
    
    a_blended_net = a_net .* alpha
    return selections, a_blended_net
end

# ===================================================================
# SECTION 2: Fast Match Portfolio Analysis Function
# ===================================================================
"""
    run_portfolio_analysis(latents, ds, odds, expr; row_idx=200, alphas=[1.0, 0.7, 0.5, 0.3, 0.0])

Fast execution function: Uses pre-loaded data/latents to run odds QA, match result validation,
and multi-market portfolio optimization in milliseconds.
"""
function run_portfolio_analysis(latents, ds, odds, expr; row_idx::Int=200, alphas=[1.0, 0.7, 0.5, 0.3, 0.0])
    row1 = latents.df[row_idx, :]
    param = Predictions.extract_params(expr.config.model, row1)
    score_matrix = Predictions.compute_score_matrix(expr.config.model, param)

    # Get match details (Home vs Away & Final Score)
    m_info = subset(ds.matches, :match_id => ByRow(==(row1.match_id)))
    home_t = isempty(m_info) ? "Home" : m_info.home_team[1]
    away_t = isempty(m_info) ? "Away" : m_info.away_team[1]
    h_sc   = isempty(m_info) ? 0 : m_info.home_score[1]
    a_sc   = isempty(m_info) ? 0 : m_info.away_score[1]

    scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
    over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
    markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

    # 1. Run Odds QA Comparison Table
    compare_odds_sources(odds, ds.odds, row1.match_id, markets_config)

    # 2. Extract & Normalize Betfair Exchange Odds
    match_model_prob = Dict(
        string(m) => Predictions.compute_market_probs(score_matrix, m)
        for m in markets_config.markets
    )
    raw_odds_map, _, winner_map = extract_market_data(odds, row1.match_id, markets_config)
    odds_map, fair_prob_map = normalize_market_group_odds(raw_odds_map)

    # 3. Solve Multi-Market Portfolio across alpha grid
    results_map = Dict{Float64, Vector{Float64}}()
    selections_list = SelData[]

    for α in alphas
        selections_list, a_stakes = optimize_portfolio(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map; alpha=α)
        results_map[α] = a_stakes
    end

    # 4. Print Clean Summary Table with Match Result & Realized P/L / ROI
    println("\n", "="^125)
    println("=== BETFAIR NORMALIZED PORTFOLIO ANALYSIS: $(home_t) $(h_sc) - $(a_sc) $(away_t) (Match #$(row1.match_id)) ===")
    println("="^125)
    println(
        rpad("Market", 16),
        rpad("Selection", 12),
        rpad("Result", 10),
        rpad("Odds", 8),
        rpad("Model P", 9),
        rpad("Fair P", 9),
        rpad("α=1.0", 10),
        rpad("α=0.7", 10),
        rpad("α=0.5", 10),
        rpad("α=0.3", 10),
        rpad("α=0.0", 10)
    )
    println("-"^125)

    commission = 0.02
    for i in 1:length(selections_list)
        sel = selections_list[i]
        res_str = sel.is_winner ? "WON (✓)" : "LOST (✗)"
        
        st_10 = round(results_map[1.0][i] * 100, digits=2)
        st_07 = round(results_map[0.7][i] * 100, digits=2)
        st_05 = round(results_map[0.5][i] * 100, digits=2)
        st_03 = round(results_map[0.3][i] * 100, digits=2)
        st_00 = round(results_map[0.0][i] * 100, digits=2)

        println(
            rpad(sel.market, 16),
            rpad(string(sel.selection), 12),
            rpad(res_str, 10),
            rpad(round(sel.odds; digits=2), 8),
            rpad(round(sel.p_model; digits=3), 9),
            rpad(round(sel.p_fair_market; digits=3), 9),
            rpad("$(st_10)%", 10),
            rpad("$(st_07)%", 10),
            rpad("$(st_05)%", 10),
            rpad("$(st_03)%", 10),
            rpad("$(st_00)%", 10)
        )
    end
    println("-"^125)

    # Calculate Realized Profit/Loss and ROI for each alpha
    pl_map  = Dict{Float64, Float64}()
    roi_map = Dict{Float64, Float64}()

    for α in alphas
        stakes = results_map[α]
        net_pl = 0.0
        tot_st = sum(stakes)
        for i in 1:length(selections_list)
            sel = selections_list[i]
            st = stakes[i]
            if st > 0
                if sel.is_winner
                    net_pl += st * (1.0 - commission) * (sel.odds - 1.0)
                else
                    net_pl -= st
                end
            end
        end
        pl_map[α]  = net_pl
        roi_map[α] = tot_st > 0 ? (net_pl / tot_st) * 100.0 : 0.0
    end

    println(
        rpad("Total Risk Stake", 46),
        rpad("$(round(sum(results_map[1.0])*100, digits=2))%", 10),
        rpad("$(round(sum(results_map[0.7])*100, digits=2))%", 10),
        rpad("$(round(sum(results_map[0.5])*100, digits=2))%", 10),
        rpad("$(round(sum(results_map[0.3])*100, digits=2))%", 10),
        rpad("$(round(sum(results_map[0.0])*100, digits=2))%", 10)
    )
    println(
        rpad("Realized Net P/L (% Bankroll)", 46),
        rpad("$(round(pl_map[1.0]*100, digits=2))%", 10),
        rpad("$(round(pl_map[0.7]*100, digits=2))%", 10),
        rpad("$(round(pl_map[0.5]*100, digits=2))%", 10),
        rpad("$(round(pl_map[0.3]*100, digits=2))%", 10),
        rpad("$(round(pl_map[0.0]*100, digits=2))%", 10)
    )
    println(
        rpad("Realized Portfolio ROI (%)", 46),
        rpad("$(round(roi_map[1.0], digits=2))%", 10),
        rpad("$(round(roi_map[0.7], digits=2))%", 10),
        rpad("$(round(roi_map[0.5], digits=2))%", 10),
        rpad("$(round(roi_map[0.3], digits=2))%", 10),
        rpad("$(round(roi_map[0.0], digits=2))%", 10)
    )
    println("="^125)

    return results_map, selections_list
end

# ===================================================================
# SECTION 3: Main Script Global Load (Executes ONCE when file included)
# ===================================================================
@info "Loading datastore and experiment latents once..."
ds = D.load_datastore_cached(D.ScottishLower())
odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

src_dir = "./data/experiments/plus_minus_biweek"
list_of_experiments = Experiments.list_experiments(src_dir, data_dir="")
expr = Experiments.load_experiment(list_of_experiments, 3)

latents = E.extract_oos_predictions(ds, expr)
@info "Loaded successfully. Call run_portfolio_analysis(latents, ds, odds, expr; row_idx=200) to analyze any match."

# Initial run on Match Row #200
run_portfolio_analysis(latents, ds, odds, expr; row_idx=210);
