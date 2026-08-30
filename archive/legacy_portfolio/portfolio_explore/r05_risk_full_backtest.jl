# r05_risk_full_backtest.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics
using ThreadPinning
using CSV

# Load the new Risk Manager Loader
include("l03_risk_manager.jl")

# -------------------------------------------------------------------
# 1. Environment & Data Loading
# -------------------------------------------------------------------
pinthreads(:cores)
@info "Multithreading configured."

@info "Loading datastore and experiment latents..."
ds = D.load_datastore_cached(D.ScottishLower())
odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

src_dir = "./data/experiments/plus_minus_biweek"
list_of_experiments = E.list_experiments(src_dir, data_dir="")
expr = E.load_experiment(list_of_experiments, 3)

latents = E.extract_oos_predictions(ds, expr)
n_matches = nrow(latents.df)
@info "Data successfully loaded. Total matches to evaluate: $(n_matches)"

# -------------------------------------------------------------------
# 2. Configuration Setup
# -------------------------------------------------------------------
scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

# Base portfolio alpha (locked to empirical optimum 0.35)
OPTIMAL_ALPHA = 0.35
b_config = BacktestConfig(commission=0.02, alphas=[1.0]) # alpha grid ignored

# Strict Risk Configuration: Limit 20% drawdown with 1% prob tolerance
r_config = RiskConfig(0.8, 0.01)

println("\n", "="^80)
println("=== RUNNING FULL RISK-MANAGED BACKTEST ON $(n_matches) MATCHES ===")
println("Base Alpha : $(OPTIMAL_ALPHA)")
println("Drawdown D : $(r_config.D)")
println("Tolerance β: $(r_config.beta)")
println("="^80)

# -------------------------------------------------------------------
# 3. Execute Full Multithreaded Backtest
# -------------------------------------------------------------------
@time results_df = run_risk_backtest(latents.df, expr, odds, markets_config, b_config, r_config; optimal_alpha=OPTIMAL_ALPHA)

# -------------------------------------------------------------------
# 4. Aggregation and Advanced Metrics
# -------------------------------------------------------------------
# Filter for successful matches to calculate compounding trajectories
valid_matches = subset(results_df, :status => ByRow(==("SUCCESS")))
sort!(valid_matches, :match_id)

n_valid = nrow(valid_matches)

# Base Trajectories
base_returns = valid_matches.base_pl
base_bankroll = [1.0; cumprod(1.0 .+ base_returns)]
base_run_max = accumulate(max, base_bankroll)
base_dd = (base_bankroll .- base_run_max) ./ base_run_max
base_mdd = minimum(base_dd) * 100.0
base_sharpe = std(base_returns) > 0 ? (mean(base_returns) / std(base_returns)) : 0.0

# Risk-Managed Trajectories
risk_returns = valid_matches.risk_pl
risk_bankroll = [1.0; cumprod(1.0 .+ risk_returns)]
risk_run_max = accumulate(max, risk_bankroll)
risk_dd = (risk_bankroll .- risk_run_max) ./ risk_run_max
risk_mdd = minimum(risk_dd) * 100.0
risk_sharpe = std(risk_returns) > 0 ? (mean(risk_returns) / std(risk_returns)) : 0.0

println("\n", "="^80)
println("=== FULL BACKTEST COMPARISON SUMMARY ===")
println("="^80)

summary = DataFrame(
    Metric = [
        "Total Stake (Units)", 
        "Net Profit (Units)", 
        "ROI (%)", 
        "Final Bankroll (Multiplier)",
        "Max Drawdown (%)", 
        "Sharpe Ratio (Match-Level)"
    ],
    Base_Portfolio = [
        round(sum(valid_matches.base_stake), digits=2),
        round(sum(base_returns), digits=3),
        round((sum(base_returns) / sum(valid_matches.base_stake)) * 100, digits=2),
        round(base_bankroll[end], digits=2),
        round(base_mdd, digits=2),
        round(base_sharpe, digits=3)
    ],
    Risk_Managed_Portfolio = [
        round(sum(valid_matches.risk_stake), digits=2),
        round(sum(risk_returns), digits=3),
        round((sum(risk_returns) / sum(valid_matches.risk_stake)) * 100, digits=2),
        round(risk_bankroll[end], digits=2),
        round(risk_mdd, digits=2),
        round(risk_sharpe, digits=3)
    ]
)

println("Total Matches Evaluated: ", n_matches)
println("Valid Matches Bet      : ", n_valid)
println("Avg Shrinkage k        : ", round(mean(valid_matches.shrink_k), digits=4))
println("-"^80)
display(summary)

# -------------------------------------------------------------------
# 5. Export to CSV
# -------------------------------------------------------------------
out_dir = "current_development/portfolio_explore"
out_full = joinpath(out_dir, "risk_full_backtest_results.csv")
out_sum  = joinpath(out_dir, "risk_full_backtest_summary.csv")

CSV.write(out_full, results_df)
CSV.write(out_sum, summary)

println("\n✓ Granular Match Results saved to $(out_full)")
println("✓ Comparison Summary saved to $(out_sum)")
