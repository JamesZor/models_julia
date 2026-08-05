# r12_multi_match.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics

include("l04_vector_alpha.jl") 
include("l05_multi_match.jl")

function run_multi_match_backtest()
    @info "Loading datastore and experiment latents..."
    ds = D.load_datastore_cached(D.ScottishLower())
    odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

    src_dir = "./data/experiments/plus_minus_biweek"
    list_of_experiments = E.list_experiments(src_dir, data_dir="")
    expr = E.load_experiment(list_of_experiments, 3) 

    latents = E.extract_oos_predictions(ds, expr)
    
    # 1. Join with ds.matches to get the actual Date of the match
    match_dates = select(ds.matches, :match_id, :match_date)
    latents_df = innerjoin(latents.df, match_dates, on=:match_id)
    
    # Sort chronologically by date
    sort!(latents_df, :match_date)
    
    optimal_alpha_dict = Dict{String, Float64}(
        "O/U 3.5_over_35" => 0.002, "BTTS_btts_no" => 0.003, "O/U 4.5_over_45" => 0.003,
        "O/U 1.5_over_15" => 0.004, "O/U 1.5_under_15" => 0.027, "O/U 0.5_under_05" => 0.029,
        "O/U 3.5_under_35" => 0.131, "1X2_away" => 0.157, "BTTS_btts_yes" => 0.183,
        "1X2_home" => 0.432, "1X2_draw" => 0.445, "O/U 2.5_over_25" => 0.486,
        "O/U 2.5_under_25" => 0.818, "O/U 0.5_over_05" => 0.846, "O/U 4.5_under_45" => 1.000
    )
    v_config = VectorAlphaConfig(0.02, optimal_alpha_dict)
    
    # Using the strict lambda constraint of 20
    r_config = RiskConfig(20.0)

    scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
    over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
    markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

    # Group by exact Date to form concurrent slates
    daily_groups = groupby(latents_df, :match_date)
    
    bankroll_history = Float64[1.0]
    total_matches_bet = 0
    total_days_bet = 0
    
    @info "Simulating $(length(daily_groups)) chronological days using Global Multi-Match Kelly..."

    for day_df in daily_groups
        date = day_df.match_date[1]
        
        day_probs = Vector{Vector{Float64}}()
        day_returns = Vector{Vector{Float64}}()
        day_selections = []
        day_vec_stakes = []
        
        # Pre-compute all naive stakes for the day
        for row in eachrow(day_df)
            m_id = row.match_id
            
            raw_odds_map, _, winner_map = extract_market_data(odds, m_id, markets_config)
            if isempty(raw_odds_map) continue end
            
            odds_map, fair_prob_map = normalize_market_group_odds(raw_odds_map)
            
            param = Predictions.extract_params(expr.config.model, row)
            local score_matrix
            try
                score_matrix = Predictions.compute_score_matrix(expr.config.model, param)
            catch e
                continue
            end
            
            match_model_prob = Dict(string(m) => Predictions.compute_market_probs(score_matrix, m) for m in markets_config.markets)
            selections, vec_stakes, R_mat = optimize_portfolio_vector(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, v_config)
            
            if isempty(selections) continue end
            
            P_model_grid = mean(score_matrix.data, dims=3)[:, :, 1]
            p_model_vec  = vec(P_model_grid)
            returns_vec  = R_mat * vec_stakes
            
            push!(day_probs, p_model_vec)
            push!(day_returns, returns_vec)
            push!(day_selections, selections)
            push!(day_vec_stakes, vec_stakes)
        end
        
        L = length(day_probs)
        if L == 0
            continue
        end
        
        # -------------------------------------------------------------------
        # THE MULTI-MATCH STOCHASTIC CONSTRAINT
        # -------------------------------------------------------------------
        # Calculate the single global shrinkage factor for the ENTIRE day's slate
        k_shrink = solve_global_drawdown_multiplier(day_probs, day_returns, r_config.lambda)
        
        day_pl_fraction = 0.0
        
        for i in 1:L
            selections = day_selections[i]
            vec_stakes = day_vec_stakes[i]
            
            # The exact same k_shrink applies to every match on this day
            risk_stakes = vec_stakes .* k_shrink
            
            match_wagered = false
            for j in 1:length(selections)
                st = risk_stakes[j]
                sel = selections[j]
                if st > 0
                    match_wagered = true
                    if sel.is_winner
                        day_pl_fraction += st * (1.0 - 0.02) * (sel.odds - 1.0)
                    else
                        day_pl_fraction -= st
                    end
                end
            end
            if match_wagered
                total_matches_bet += 1
            end
        end
        
        # Compound the bankroll by the net result of the entire day
        new_bankroll = bankroll_history[end] * (1.0 + day_pl_fraction)
        push!(bankroll_history, new_bankroll)
        total_days_bet += 1
    end
    
    # -------------------------------------------------------------------
    # Compute Final Performance Metrics
    # -------------------------------------------------------------------
    run_max = accumulate(max, bankroll_history)
    dd = (bankroll_history .- run_max) ./ run_max
    dd_pct = dd .* 100.0

    fb = bankroll_history[end]
    mdd_pct = minimum(dd_pct)
    ulcer_index = sqrt(mean(dd_pct .^ 2))
    
    tot_ret = (fb - 1.0) * 100.0
    martin = ulcer_index > 0.0 ? (tot_ret / ulcer_index) : 0.0
    calmar = mdd_pct < 0.0 ? (tot_ret / abs(mdd_pct)) : 0.0

    println("\n", "="^80)
    println("=== MULTI-MATCH STOCHASTIC PORTFOLIO (GLOBAL SHINKAGE) ===")
    println("Constraint: 20% Max Drawdown Limit (Lambda = $(r_config.lambda))")
    println("="^80)
    println("Valid Betting Days    : ", total_days_bet)
    println("Total Matches Bet     : ", total_matches_bet)
    println("-"^80)
    println("Final Bankroll        : ", round(fb, digits=2), "x")
    println("Max Drawdown          : ", round(mdd_pct, digits=2), "%")
    println("Calmar Ratio          : ", round(calmar, digits=3))
    println("Martin Ratio          : ", round(martin, digits=3))
    println("="^80)
end

run_multi_match_backtest()
