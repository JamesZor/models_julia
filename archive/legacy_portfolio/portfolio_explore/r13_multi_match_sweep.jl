# r13_multi_match_sweep.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, CSV

include("l04_vector_alpha.jl") 
include("l05_multi_match.jl")

function run_multi_match_sweep()
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
    
    optimal_alpha_dict = Dict{String, Float64}(
        "O/U 3.5_over_35" => 0.002, "BTTS_btts_no" => 0.003, "O/U 4.5_over_45" => 0.003,
        "O/U 1.5_over_15" => 0.004, "O/U 1.5_under_15" => 0.027, "O/U 0.5_under_05" => 0.029,
        "O/U 3.5_under_35" => 0.131, "1X2_away" => 0.157, "BTTS_btts_yes" => 0.183,
        "1X2_home" => 0.432, "1X2_draw" => 0.445, "O/U 2.5_over_25" => 0.486,
        "O/U 2.5_under_25" => 0.818, "O/U 0.5_over_05" => 0.846, "O/U 4.5_under_45" => 1.000
    )
    v_config = VectorAlphaConfig(0.02, optimal_alpha_dict)

    scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
    over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
    markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

    daily_groups = groupby(latents_df, :match_date)
    
    # -------------------------------------------------------------------
    # STEP 1: Pre-compute Naive Kelly Bets for all slates (Extremely Fast Sweep)
    # -------------------------------------------------------------------
    @info "Pre-computing Naive Kelly Vectors for $(length(daily_groups)) days..."
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

    # -------------------------------------------------------------------
    # STEP 2: Sweep the Global Risk Manager across the slates
    # -------------------------------------------------------------------
    lambdas = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    
    daily_ledger = DataFrame(
        date = Date[],
        lambda = Float64[],
        num_matches = Int[],
        global_shrink_k = Float64[],
        total_risk_stake = Float64[],
        day_pl_fraction = Float64[]
    )
    
    sweep_results = DataFrame(
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
    
    @info "Sweeping Multi-Match Global Constraint across Lambdas..."

    for lambda in lambdas
        bankroll_history = Float64[1.0]
        lambda_stake = 0.0
        lambda_net_pl = 0.0
        shrink_sum = 0.0
        
        for slate in precomputed_slates
            L = length(slate.probs)
            
            # The Multi-Match Stochastic Formula (computes instantly!)
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
            
            # Bankroll Update (Stochastic Jump)
            new_bankroll = bankroll_history[end] * (1.0 + day_pl_fraction)
            push!(bankroll_history, new_bankroll)
            
            lambda_stake += day_stake
            lambda_net_pl += day_pl_fraction # Just keeping track of absolute units wagered
            
            # Save to daily ledger for EDA
            push!(daily_ledger, (slate.date, lambda, L, k_shrink, day_stake, day_pl_fraction))
        end
        
        run_max = accumulate(max, bankroll_history)
        dd = (bankroll_history .- run_max) ./ run_max
        dd_pct = dd .* 100.0

        fb = bankroll_history[end]
        mdd_pct = minimum(dd_pct)
        ulcer_index = sqrt(mean(dd_pct .^ 2))
        ulcer_index = max(ulcer_index, 1e-4)
        
        tot_ret = (fb - 1.0) * 100.0
        martin = (tot_ret / ulcer_index)
        calmar = mdd_pct < 0.0 ? (tot_ret / abs(mdd_pct)) : 0.0
        roi = lambda_stake > 0 ? (lambda_net_pl / lambda_stake) * 100.0 : 0.0
        
        avg_shrink = shrink_sum / length(precomputed_slates)
        
        push!(sweep_results, (
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
    
    # Save ledger
    out_file = joinpath(@__DIR__, "multi_match_daily_allocations.csv")
    CSV.write(out_file, daily_ledger)
    
    println("\n", "="^100)
    println("=== MULTI-MATCH STOCHASTIC PORTFOLIO SWEEP (GLOBAL SHRINKAGE) ===")
    println("="^100)
    display(sweep_results)
    println("\n✓ Saved detailed daily allocation ledger to $(out_file)")
end

run_multi_match_sweep()




#=
====================================================================================================
=== MULTI-MATCH STOCHASTIC PORTFOLIO SWEEP (GLOBAL SHRINKAGE) ===
====================================================================================================
9×9 DataFrame
 Row │ lambda   Avg_Global_Shrink  Total_Stake  Net_PL   ROI      Final_Bankroll  Max_Drawdown  Calmar   Martin
     │ Float64  Float64            Float64      Float64  Float64  Float64         Float64       Float64  Float64
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     0.0              1.0          42.31    5.152    12.18           34.21        -59.31   55.984  159.362
   2 │     5.0              0.829        33.91    3.868    11.4            17.44        -48.76   33.706   89.578
   3 │    10.0              0.489        19.12    2.17     11.35            6.28        -32.07   16.461   51.662
   4 │    15.0              0.34         13.13    1.493    11.37            3.8         -22.7    12.321   39.551
   5 │    20.0              0.261        10.0     1.138    11.37            2.84        -17.55   10.5     34.073
   6 │    25.0              0.213         8.08    0.919    11.38            2.36        -14.3     9.5     31.024
   7 │    30.0              0.18          6.77    0.771    11.38            2.07        -12.06    8.871   29.096
   8 │    40.0              0.138         5.12    0.583    11.38            1.75         -9.19    8.13    26.807
   9 │    50.0              0.113         4.11    0.468    11.39            1.57         -7.42    7.709   25.498

✓ Saved detailed daily allocation ledger to /root/BayesianFootball/multi_match_daily_allocations.csv
=#

