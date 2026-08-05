# r10_vector_risk_backtest.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics

# We include both the Risk Manager and the Vector Alpha modules
include("l03_risk_manager.jl") 
include("l04_vector_alpha.jl") 

function run_theoretical_maximum_risk_managed()
    @info "Loading datastore and experiment latents..."
    ds = D.load_datastore_cached(D.ScottishLower())
    odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

    src_dir = "./data/experiments/plus_minus_biweek"
    list_of_experiments = E.list_experiments(src_dir, data_dir="")
    expr = E.load_experiment(list_of_experiments, 2)

    latents = E.extract_oos_predictions(ds, expr)
    n_matches = nrow(latents.df)

    # 1. The Vector Alpha Configuration (Shapes the Bets)
    optimal_alpha_dict = Dict{String, Float64}(
        "O/U 3.5_over_35" => 0.002, "BTTS_btts_no" => 0.003, "O/U 4.5_over_45" => 0.003,
        "O/U 1.5_over_15" => 0.004, "O/U 1.5_under_15" => 0.027, "O/U 0.5_under_05" => 0.029,
        "O/U 3.5_under_35" => 0.131, "1X2_away" => 0.157, "BTTS_btts_yes" => 0.183,
        "1X2_home" => 0.432, "1X2_draw" => 0.445, "O/U 2.5_over_25" => 0.486,
        "O/U 2.5_under_25" => 0.818, "O/U 0.5_over_05" => 0.846, "O/U 4.5_under_45" => 1.000
    )
    v_config = VectorAlphaConfig(0.02, optimal_alpha_dict)

    lambdas = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
    over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
    markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

    res_df = DataFrame(
        lambda = Float64[],
        shrink_k = Float64[],
        risk_stake = Float64[],
        risk_pl = Float64[]
    )

    @info "Running Risk-Managed Vector Backtest Sweep..."

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
        
        # -------------------------------------------------------------------
        # Step A: Generate Vectorized Stakes
        # -------------------------------------------------------------------
        selections, vec_stakes, R_mat = optimize_portfolio_vector(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, v_config)
        
        if isempty(selections)
            continue
        end
        
        # -------------------------------------------------------------------
        # Step B: Apply Global Risk Management
        # -------------------------------------------------------------------
        P_model_grid = mean(score_matrix.data, dims=3)[:, :, 1]
        p_model_vec  = vec(P_model_grid)
        returns_vec  = R_mat * vec_stakes # Calculate expected portfolio returns for each outcome
        
        for lambda in lambdas
            # The Risk Manager inspects the vector stakes and calculates a universal shrink factor 'k'
            k_shrink = solve_drawdown_multiplier(p_model_vec, returns_vec, lambda)
            
            # Apply the final risk boundary
            risk_stakes = vec_stakes .* k_shrink
            
            match_pl = 0.0
            match_stake = 0.0
            
            for j in 1:length(selections)
                st = risk_stakes[j]
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
                push!(res_df, (lambda, k_shrink, match_stake, match_pl))
            end
        end
    end
    
    # -------------------------------------------------------------------
    # Compute Final Performance Metrics Per Lambda
    # -------------------------------------------------------------------
    sweep_summary = combine(groupby(res_df, :lambda),
        :shrink_k => mean => :Avg_Shrinkage,
        :risk_stake => sum => :Total_Stake,
        :risk_pl => sum => :Net_PL
    )
    
    final_banks = Float64[]
    max_dds = Float64[]
    sharpes = Float64[]
    calmars = Float64[]
    martins = Float64[]

    for row in eachrow(sweep_summary)
        sub = subset(res_df, :lambda => ByRow(==(row.lambda)))
        returns = sub.risk_pl
        
        if length(returns) > 0
            bankroll = [1.0; cumprod(1.0 .+ returns)]
            run_max = accumulate(max, bankroll)
            dd = (bankroll .- run_max) ./ run_max
            dd_pct = dd .* 100.0
            
            fb = bankroll[end]
            mdd_pct = minimum(dd_pct)
            ulcer_index = sqrt(mean(dd_pct .^ 2))
            ulcer_index = max(ulcer_index, 1e-4)
            
            tot_ret = (fb - 1.0) * 100.0
            calmar = mdd_pct < 0.0 ? (tot_ret / abs(mdd_pct)) : 0.0
            martin = (tot_ret / ulcer_index)
            
            push!(final_banks, fb)
            push!(max_dds, mdd_pct)
            push!(sharpes, std(returns) > 0 ? (mean(returns) / std(returns)) : 0.0)
            push!(calmars, calmar)
            push!(martins, martin)
        else
            push!(final_banks, 1.0)
            push!(max_dds, 0.0)
            push!(sharpes, 0.0)
            push!(calmars, 0.0)
            push!(martins, 0.0)
        end
    end
    
    sweep_summary.Final_Bankroll = final_banks
    sweep_summary.Max_Drawdown = max_dds
    sweep_summary.Sharpe = sharpes
    sweep_summary.Calmar = calmars
    sweep_summary.Martin = martins
    
    sweep_summary.ROI = (sweep_summary.Net_PL ./ sweep_summary.Total_Stake) .* 100.0
    
    # Formatting
    sweep_summary.Avg_Shrinkage = round.(sweep_summary.Avg_Shrinkage, digits=3)
    sweep_summary.Total_Stake   = round.(sweep_summary.Total_Stake, digits=2)
    sweep_summary.Net_PL        = round.(sweep_summary.Net_PL, digits=3)
    sweep_summary.ROI           = round.(sweep_summary.ROI, digits=2)
    sweep_summary.Final_Bankroll= round.(sweep_summary.Final_Bankroll, digits=2)
    sweep_summary.Max_Drawdown  = round.(sweep_summary.Max_Drawdown, digits=2)
    sweep_summary.Sharpe        = round.(sweep_summary.Sharpe, digits=3)
    sweep_summary.Calmar        = round.(sweep_summary.Calmar, digits=3)
    sweep_summary.Martin        = round.(sweep_summary.Martin, digits=3)

    select!(sweep_summary, :lambda, :Avg_Shrinkage, :Total_Stake, :Net_PL, :ROI, :Final_Bankroll, :Max_Drawdown, :Sharpe, :Calmar, :Martin)

    println("\n", "="^80)
    println("=== RISK-MANAGED VECTOR PORTFOLIO SWEEP ===")
    println("="^80)
    display(sweep_summary)
end

run_theoretical_maximum_risk_managed()




#=
Loading: funnel_apm_xg_20260729_044741
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
[ Info: Running Risk-Managed Vector Backtest Sweep...
jk
================================================================================
=== RISK-MANAGED VECTOR PORTFOLIO SWEEP ===
================================================================================
9×10 DataFrame
 Row │ lambda   Avg_Shrinkage  Total_Stake  Net_PL   ROI      Final_Bankroll  Max_Drawdown  Sharpe   Calmar   Martin
     │ Float64  Float64        Float64      Float64  Float64  Float64         Float64       Float64  Float64  Float64
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     0.0          1.0          42.31    5.152    12.18           30.12        -56.97    0.108   51.121  140.841
   2 │     5.0          0.826        32.16    3.717    11.56           14.21        -51.29    0.1     25.759   69.599
   3 │    10.0          0.594        20.4     2.134    10.46            5.56        -37.07    0.092   12.288   33.388
   4 │    15.0          0.431        14.2     1.442    10.16            3.45        -27.06    0.09     9.069   25.303
   5 │    20.0          0.333        10.82    1.098    10.15            2.67        -21.2     0.09     7.856   22.543
   6 │    25.0          0.273         8.74    0.885    10.14            2.24        -17.41    0.09     7.151   20.88
   7 │    30.0          0.232         7.33    0.742    10.12            1.99        -14.77    0.09     6.697   19.785
   8 │    40.0          0.18          5.54    0.56     10.1             1.7         -11.33    0.09     6.151   18.441
   9 │    50.0          0.148         4.45    0.45     10.09            1.54         -9.19    0.09     5.842   17.677
=#



#=
Loading: apm_shots_20260729_093047
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
[ Info: Running Risk-Managed Vector Backtest Sweep...

================================================================================
=== RISK-MANAGED VECTOR PORTFOLIO SWEEP ===
================================================================================
9×10 DataFrame
 Row │ lambda   Avg_Shrinkage  Total_Stake  Net_PL   ROI      Final_Bankroll  Max_Drawdown  Sharpe   Calmar   Martin
     │ Float64  Float64        Float64      Float64  Float64  Float64         Float64       Float64  Float64  Float64
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     0.0          1.0          51.11    3.989     7.8             3.34        -72.49    0.065    3.229    5.886
   2 │     5.0          0.791        36.98    2.716     7.35            3.16        -65.53    0.059    3.296    6.631
   3 │    10.0          0.539        22.34    1.541     6.9             2.61        -48.5     0.056    3.322    8.03
   4 │    15.0          0.387        15.46    1.034     6.69            2.13        -35.79    0.055    3.151    7.968
   5 │    20.0          0.298        11.78    0.787     6.68            1.87        -28.08    0.055    3.086    7.991
   6 │    25.0          0.243         9.51    0.637     6.69            1.7         -23.13    0.055    3.024    7.956
   7 │    30.0          0.206         7.98    0.535     6.71            1.58        -19.65    0.055    2.973    7.915
   8 │    40.0          0.158         6.03    0.407     6.74            1.44        -15.11    0.055    2.903    7.855
   9 │    50.0          0.129         4.85    0.328     6.77            1.35        -12.27    0.055    2.857    7.823
=#



#=
Experiments in: ./data/experiments/plus_minus_biweek
=============================================================================================================================
IDX  | NAME                      | MODEL                | SPLITTER           | SAMPLER         | TIME       | PATH ID
-----------------------------------------------------------------------------------------------------------------------------
[1]  | apm_shots                 | DynamicGoalsPlusMi.. | GroupedCVConfig    | QueuedNUTSCon.. | 1h 45m     | apm_shots_20260729_093047
[2]  | funnel_winner             | DynamicFunnelDoubl.. | GroupedCVConfig    | QueuedNUTSCon.. | 2h 57m     | funnel_winner_20260729_074452
[3]  | funnel_apm_xg             | DynamicFunnelPlusM.. | GroupedCVConfig    | QueuedNUTSCon.. | 3h 20m     | funnel_apm_xg_20260729_044741
=============================================================================================================================

Loading: funnel_winner_20260729_074452
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
[ Info: Running Risk-Managed Vector Backtest Sweep...

================================================================================
=== RISK-MANAGED VECTOR PORTFOLIO SWEEP ===
================================================================================
9×10 DataFrame
 Row │ lambda   Avg_Shrinkage  Total_Stake  Net_PL   ROI      Final_Bankroll  Max_Drawdown  Sharpe   Calmar   Martin
     │ Float64  Float64        Float64      Float64  Float64  Float64         Float64       Float64  Float64  Float64
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     0.0          1.0          41.37    4.385    10.6            15.8         -74.66    0.095   19.828   53.805
   2 │     5.0          0.823        31.42    2.93      9.33            7.03        -70.74    0.082    8.521   23.928
   3 │    10.0          0.582        19.86    1.729     8.7             3.81        -53.61    0.077    5.233   15.217
   4 │    15.0          0.42         13.79    1.171     8.49            2.67        -40.65    0.076    4.111   12.458
   5 │    20.0          0.324        10.51    0.89      8.47            2.18        -32.28    0.076    3.664   11.358
   6 │    25.0          0.265         8.49    0.719     8.48            1.91        -26.72    0.076    3.413   10.729
   7 │    30.0          0.224         7.12    0.604     8.48            1.74        -22.76    0.076    3.247   10.303
   8 │    40.0          0.173         5.38    0.457     8.48            1.53        -17.52    0.076    3.048    9.783
   9 │    50.0          0.142         4.33    0.367     8.49            1.42        -14.21    0.076    2.937    9.487
=#


