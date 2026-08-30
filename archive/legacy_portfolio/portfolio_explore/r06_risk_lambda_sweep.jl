# r06_risk_lambda_sweep.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics
using ThreadPinning
using CSV

include("l03_risk_manager.jl")

# -------------------------------------------------------------------
# 1. Environment & Data Loading
# -------------------------------------------------------------------
pinthreads(:cores)
@info "Loading datastore and experiment latents..."
ds = D.load_datastore_cached(D.ScottishLower())
odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

src_dir = "./data/experiments/plus_minus_biweek"
list_of_experiments = E.list_experiments(src_dir, data_dir="")
expr = E.load_experiment(list_of_experiments, 3)

latents = E.extract_oos_predictions(ds, expr)
n_matches = nrow(latents.df)

# -------------------------------------------------------------------
# 2. Configuration Setup
# -------------------------------------------------------------------
scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

b_config = BacktestConfig(commission=0.02, alphas=[1.0])
OPTIMAL_ALPHA = 0.35

# The sweep grid for the single risk parameter λ
# λ = 0 is equivalent to no constraint (Risk Manager is turned off)
# Higher λ = stronger variance penalty and more aggressive stake shrinkage
lambdas = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]

println("\n", "="^80)
println("=== RUNNING LAMBDA RISK SWEEP ON $(n_matches) MATCHES ===")
println("="^80)

sweep_results = DataFrame(
    Lambda         = Float64[],
    Avg_Shrinkage  = Float64[],
    Total_Stake    = Float64[],
    Net_PL         = Float64[],
    ROI            = Float64[],
    Final_Bankroll = Float64[],
    Max_Drawdown   = Float64[],
    Sharpe         = Float64[]
)

# -------------------------------------------------------------------
# 3. Sweep Execution
# -------------------------------------------------------------------
for λ_val in lambdas
    r_config = RiskConfig(λ_val)
    
    print("Evaluating λ = $(lpad(λ_val, 5))... ")
    
    # Run multithreaded backtest for this lambda
    res_df = run_risk_backtest(latents.df, expr, odds, markets_config, b_config, r_config; optimal_alpha=OPTIMAL_ALPHA)
    
    valid_matches = subset(res_df, :status => ByRow(==("SUCCESS")))
    sort!(valid_matches, :match_id)
    
    # Compute compounding trajectory
    returns = valid_matches.risk_pl
    bankroll = [1.0; cumprod(1.0 .+ returns)]
    run_max = accumulate(max, bankroll)
    dd = (bankroll .- run_max) ./ run_max
    mdd = minimum(dd) * 100.0
    sharpe = std(returns) > 0 ? (mean(returns) / std(returns)) : 0.0
    
    tot_st = sum(valid_matches.risk_stake)
    net_pl = sum(returns)
    roi = tot_st > 0 ? (net_pl / tot_st) * 100.0 : 0.0
    
    push!(sweep_results, (
        λ_val,
        mean(valid_matches.shrink_k),
        tot_st,
        net_pl,
        roi,
        bankroll[end],
        mdd,
        sharpe
    ))
    println("Done (MDD: $(round(mdd, digits=2))%)")
end

# -------------------------------------------------------------------
# 4. Results Display & Export
# -------------------------------------------------------------------
println("\n", "="^80)
println("=== LAMBDA SWEEP SUMMARY (Efficient Frontier) ===")
println("="^80)

# Format for pretty printing
sweep_results.Avg_Shrinkage  = round.(sweep_results.Avg_Shrinkage, digits=3)
sweep_results.Total_Stake    = round.(sweep_results.Total_Stake, digits=1)
sweep_results.Net_PL         = round.(sweep_results.Net_PL, digits=2)
sweep_results.ROI            = round.(sweep_results.ROI, digits=2)
sweep_results.Final_Bankroll = round.(sweep_results.Final_Bankroll, digits=2)
sweep_results.Max_Drawdown   = round.(sweep_results.Max_Drawdown, digits=2)
sweep_results.Sharpe         = round.(sweep_results.Sharpe, digits=3)

display(sweep_results)

out_file = "current_development/portfolio_explore/r06_lambda_sweep_results.csv"
CSV.write(out_file, sweep_results)
println("\n✓ Sweep results exported to $(out_file)")
