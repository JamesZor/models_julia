# r09_vector_alpha_backtest.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics

include("l04_vector_alpha.jl")

function run_theoretical_maximum_backtest()
    @info "Loading datastore and experiment latents..."
    ds = D.load_datastore_cached(D.ScottishLower())
    odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

    src_dir = "./data/experiments/plus_minus_biweek"
    list_of_experiments = E.list_experiments(src_dir, data_dir="")
    expr = E.load_experiment(list_of_experiments, 3)

    latents = E.extract_oos_predictions(ds, expr)
    n_matches = nrow(latents.df)

    # -------------------------------------------------------------------
    # The optimal alpha vector discovered by the Nelder-Mead Optimizer
    # -------------------------------------------------------------------
    optimal_alpha_dict = Dict{String, Float64}(
        "O/U 3.5_over_35" => 0.002,
        "BTTS_btts_no" => 0.003,
        "O/U 4.5_over_45" => 0.003,
        "O/U 1.5_over_15" => 0.004,
        "O/U 1.5_under_15" => 0.027,
        "O/U 0.5_under_05" => 0.029,
        "O/U 3.5_under_35" => 0.131,
        "1X2_away" => 0.157,
        "BTTS_btts_yes" => 0.183,
        "1X2_home" => 0.432,
        "1X2_draw" => 0.445,
        "O/U 2.5_over_25" => 0.486,
        "O/U 2.5_under_25" => 0.818,
        "O/U 0.5_over_05" => 0.846,
        "O/U 4.5_under_45" => 1.000
    )

    config = VectorAlphaConfig(0.02, optimal_alpha_dict)

    scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
    over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
    markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

    returns = Float64[]
    total_stake = 0.0
    net_pl = 0.0
    valid_match_count = 0

    @info "Running Optimized Vector Backtest..."

    for i in 1:n_matches
        row = latents.df[i, :]
        m_id = row.match_id
        
        raw_odds_map, _, winner_map = extract_market_data(odds, m_id, markets_config)
        if isempty(raw_odds_map)
            continue
        end
        
        odds_map, fair_prob_map = normalize_market_group_odds(raw_odds_map)
        
        param = Predictions.extract_params(expr.config.model, row)
        local score_matrix
        try
            score_matrix = Predictions.compute_score_matrix(expr.config.model, param)
        catch e
            continue
        end
        
        match_model_prob = Dict(
            string(m) => Predictions.compute_market_probs(score_matrix, m)
            for m in markets_config.markets
        )
        
        selections, stakes, _ = optimize_portfolio_vector(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, config)
        
        if isempty(selections)
            continue
        end
        
        match_pl = 0.0
        match_stake = 0.0
        
        for j in 1:length(selections)
            st = stakes[j]
            sel = selections[j]
            if st > 0
                match_stake += st
                if sel.is_winner
                    match_pl += st * (1.0 - 0.02) * (sel.odds - 1.0)
                else
                    match_pl -= st
                end
            end
        end
        
        if match_stake > 0
            valid_match_count += 1
            total_stake += match_stake
            net_pl += match_pl
            push!(returns, match_pl) 
        else
            # Track matches where we placed zero bets due to small alphas
            push!(returns, 0.0)
        end
    end
    
    filter!(x -> x != 0.0, returns)

    # -------------------------------------------------------------------
    # Compute Final Performance Metrics
    # -------------------------------------------------------------------
    bankroll = [1.0; cumprod(1.0 .+ returns)]
    run_max = accumulate(max, bankroll)
    dd = (bankroll .- run_max) ./ run_max
    dd_pct = dd .* 100.0

    fb = bankroll[end]
    mdd_pct = minimum(dd_pct)
    ulcer_index = sqrt(mean(dd_pct .^ 2))
    ulcer_index = max(ulcer_index, 1e-4)

    tot_ret = (fb - 1.0) * 100.0
    martin = (tot_ret / ulcer_index)
    calmar = mdd_pct < 0.0 ? (tot_ret / abs(mdd_pct)) : 0.0

    sharpe = std(returns) > 0 ? (mean(returns) / std(returns)) : 0.0
    roi = (net_pl / total_stake) * 100.0

    println("\n", "="^80)
    println("=== THEORETICAL MAXIMUM VECTOR PORTFOLIO ===")
    println("="^80)
    println("Valid Betting Matches : ", valid_match_count)
    println("Total Stake (Units)   : ", round(total_stake, digits=2))
    println("Net P/L (Units)       : ", round(net_pl, digits=3))
    println("Flat ROI (%)          : ", round(roi, digits=2), "%")
    println("Sharpe (Match-Level)  : ", round(sharpe, digits=3))
    println("-"^80)
    println("Final Bankroll        : ", round(fb, digits=2), "x")
    println("Max Drawdown          : ", round(mdd_pct, digits=2), "%")
    println("Calmar Ratio          : ", round(calmar, digits=3))
    println("Martin Ratio          : ", round(martin, digits=3))
    println("="^80)
end

run_theoretical_maximum_backtest()
