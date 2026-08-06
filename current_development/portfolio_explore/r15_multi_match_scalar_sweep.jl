# r15_multi_match_scalar_sweep.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, CSV

include("l04_vector_alpha.jl") 
include("l05_multi_match.jl")

function run_multi_match_scalar_sweep()
    @info "Loading datastore and experiment latents..."
    ds = D.load_datastore_cached(D.ScottishLower())
    odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

    src_dir = "./data/experiments/plus_minus_biweek"
    list_of_experiments = E.list_experiments(src_dir, data_dir="")
    expr = E.load_experiment(list_of_experiments, 3) # funnel_apm_xg

    latents = E.extract_oos_predictions(ds, expr)
    
    match_dates = select(ds.matches, :match_id, :match_date)
    latents_df = innerjoin(latents.df, match_dates, on=:match_id)
    sort!(latents_df, :match_date)
    
    scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
    over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
    markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

    daily_groups = groupby(latents_df, :match_date)
    
    # We will test 5 different scalar alpha values
    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    lambdas = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    
    sweep_results = DataFrame(
        alpha = Float64[],
        lambda = Float64[],
        Avg_Global_Shrink = Float64[],
        Total_Stake = Float64[],
        Net_PL = Float64[],
        ROI = Float64[],
        Final_Bankroll = Float64[],
        Max_Drawdown = Float64[],
        Calmar = Float64[],
        Martin = Float64[]
    )
    
    for alpha in alphas
        @info "Pre-computing Naive Kelly Vectors for Scalar α = $alpha..."
        
        # Build the naive dict where every market shares the exact same alpha
        default_alpha_dict = Dict{String, Float64}(
            "O/U 3.5_over_35" => alpha, "BTTS_btts_no" => alpha, "O/U 4.5_over_45" => alpha,
            "O/U 1.5_over_15" => alpha, "O/U 1.5_under_15" => alpha, "O/U 0.5_under_05" => alpha,
            "O/U 3.5_under_35" => alpha, "1X2_away" => alpha, "BTTS_btts_yes" => alpha,
            "1X2_home" => alpha, "1X2_draw" => alpha, "O/U 2.5_over_25" => alpha,
            "O/U 2.5_under_25" => alpha, "O/U 0.5_over_05" => alpha, "O/U 4.5_under_45" => alpha
        )
        v_config = VectorAlphaConfig(0.02, default_alpha_dict)
        
        precomputed_slates = []
        
        for day_df in daily_groups
            date = day_df.match_date[1]
            day_probs = Vector{Vector{Float64}}()
            day_returns = Vector{Vector{Float64}}()
            day_selections = []
            day_vec_stakes = []
            
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
            
            if length(day_probs) > 0
                push!(precomputed_slates, (date=date, probs=day_probs, returns=day_returns, selections=day_selections, vec_stakes=day_vec_stakes))
            end
        end
        
        # Now sweep all lambdas for this specific alpha
        for lambda in lambdas
            bankroll_history = Float64[1.0]
            lambda_stake = 0.0
            lambda_net_pl = 0.0
            shrink_sum = 0.0
            
            for slate in precomputed_slates
                L = length(slate.probs)
                k_shrink = solve_global_drawdown_multiplier(slate.probs, slate.returns, lambda)
                shrink_sum += k_shrink
                
                day_pl_fraction = 0.0
                day_stake = 0.0
                
                for i in 1:L
                    selections = slate.selections[i]
                    vec_stakes = slate.vec_stakes[i]
                    risk_stakes = vec_stakes .* k_shrink
                    
                    for j in 1:length(selections)
                        st = risk_stakes[j]
                        sel = selections[j]
                        if st > 0
                            day_stake += st
                            if sel.is_winner
                                day_pl_fraction += st * (1.0 - 0.02) * (sel.odds - 1.0)
                            else
                                day_pl_fraction -= st
                            end
                        end
                    end
                end
                
                new_bankroll = bankroll_history[end] * (1.0 + day_pl_fraction)
                push!(bankroll_history, new_bankroll)
                
                lambda_stake += day_stake
                lambda_net_pl += day_pl_fraction
            end
            
            run_max = accumulate(max, bankroll_history)
            dd = (bankroll_history .- run_max) ./ run_max
            dd_pct = dd .* 100.0

            fb = bankroll_history[end]
            mdd_pct = minimum(dd_pct)
            ulcer_index = sqrt(mean(dd_pct .^ 2))
            ulcer_index = max(ulcer_index, 1e-4)
            
            tot_ret = (fb - 1.0) * 100.0
            martin = ulcer_index > 0.0 ? (tot_ret / ulcer_index) : 0.0
            calmar = mdd_pct < 0.0 ? (tot_ret / abs(mdd_pct)) : 0.0
            roi = lambda_stake > 0 ? (lambda_net_pl / lambda_stake) * 100.0 : 0.0
            
            avg_shrink = shrink_sum / length(precomputed_slates)
            
            push!(sweep_results, (
                alpha,
                lambda, 
                round(avg_shrink, digits=3), 
                round(lambda_stake, digits=2), 
                round(lambda_net_pl, digits=3), 
                round(roi, digits=2), 
                round(fb, digits=2), 
                round(mdd_pct, digits=2), 
                round(calmar, digits=3), 
                round(martin, digits=3)
            ))
        end
    end
    
    println("\n", "="^110)
    println("=== MULTI-MATCH STOCHASTIC PORTFOLIO (2D SCALAR ALPHA & LAMBDA SWEEP) ===")
    println("="^110)
    
    # Sort by Martin Ratio so the best configurations bubble to the top
    sort!(sweep_results, :Martin, rev=true)
    
    # Display the top 20 configurations
    display(first(sweep_results, 20))
end

run_multi_match_scalar_sweep()
