# r03_full_backtest.jl

using Revise
using BayesianFootball
using DataFrames, Dates
using ThreadPinning
using CSV

# Load the loader file
include("l02_portfolio_backtest.jl")

# -------------------------------------------------------------------
# 1. Environment & Threading Setup
# -------------------------------------------------------------------
pinthreads(:cores)
@info "Multithreading configured." n_threads = Threads.nthreads() n_cores = ThreadPinning.ncores()

# -------------------------------------------------------------------
# 2. Data Loading
# -------------------------------------------------------------------
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
# 3. Configuration Setup
# -------------------------------------------------------------------
scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

b_config = BacktestConfig(
    commission = 0.02,
    alphas = [1.0, 0.7, 0.5, 0.3, 0.0]
)

# -------------------------------------------------------------------
# 4. Execute Full Multithreaded Backtest
# -------------------------------------------------------------------
println("\n", "="^80)
println("=== RUNNING FULL BACKTEST ON $(n_matches) MATCHES ===")
println("="^80)

@time results_df = run_backtest(latents.df, expr, odds, markets_config, b_config)

# -------------------------------------------------------------------
# 5. Aggregation and Advanced Metrics
# -------------------------------------------------------------------
println("\n", "="^80)
println("=== FULL BACKTEST SUMMARY ===")
println("="^80)

# Group by alpha
gdf = groupby(results_df, :alpha)

summary = combine(gdf,
    :match_id => length => :total_matches,
    :status => (x -> sum(x .== "SUCCESS")) => :valid_matches,
    :status => (x -> sum(x .== "MISSING_ODDS")) => :missing_odds,
    :total_stake => sum => :total_staked,
    :net_pl => sum => :total_net_pl,
    :bets_placed => sum => :total_bets
)

# Calculate standard ROI
summary.roi = ifelse.(summary.total_staked .> 0, (summary.total_net_pl ./ summary.total_staked) .* 100, 0.0)

# Calculate Sequential Compounding (Bankroll Trajectory) & Geometric Growth Rate
# (Assuming sequential betting, W_t = W_{t-1} * (1 + net_pl))
compounding_returns = Float64[]
geom_growth_rates = Float64[]
sharpe_ratios = Float64[]
max_drawdowns = Float64[]

for row in eachrow(summary)
    α = row.alpha
    alpha_subset = subset(results_df, :alpha => ByRow(==(α)))
    # Sort strictly by match_id (proxy for chronological order here)
    sort!(alpha_subset, :match_id)
    
    # Calculate product of (1 + PL) for valid matches
    valid_subset = subset(alpha_subset, :status => ByRow(==("SUCCESS")))
    
    if nrow(valid_subset) > 0
        returns = valid_subset.net_pl
        total_compounded = prod(1.0 .+ returns)
        avg_growth = exp(mean(log.(1.0 .+ returns))) - 1.0
        
        # Sharpe Ratio (Match-level)
        stdev_ret = std(returns)
        sharpe = stdev_ret > 0 ? (mean(returns) / stdev_ret) : 0.0
        
        # Max Drawdown
        bankroll_curve = [1.0; cumprod(1.0 .+ returns)]
        running_max = accumulate(max, bankroll_curve)
        drawdowns = (bankroll_curve .- running_max) ./ running_max
        mdd = minimum(drawdowns) * 100.0 # As a negative percentage
        
        push!(compounding_returns, total_compounded)
        push!(geom_growth_rates, avg_growth * 100.0) # as percentage
        push!(sharpe_ratios, sharpe)
        push!(max_drawdowns, mdd)
    else
        push!(compounding_returns, 1.0)
        push!(geom_growth_rates, 0.0)
        push!(sharpe_ratios, 0.0)
        push!(max_drawdowns, 0.0)
    end
end

summary.final_bankroll = round.(compounding_returns, digits=2)
summary.geom_growth_pct = round.(geom_growth_rates, digits=4)
summary.sharpe = round.(sharpe_ratios, digits=3)
summary.max_dd_pct = round.(max_drawdowns, digits=2)

# Format for pretty printing
summary.total_staked = round.(summary.total_staked, digits=2)
summary.total_net_pl = round.(summary.total_net_pl, digits=3)
summary.roi = round.(summary.roi, digits=2)

display(sort(summary, :alpha, rev=true))

# -------------------------------------------------------------------
# 6. Export to CSV
# -------------------------------------------------------------------
out_dir = "current_development/portfolio_explore"
out_full = joinpath(out_dir, "full_backtest_results.csv")
out_sum  = joinpath(out_dir, "full_backtest_summary.csv")

CSV.write(out_full, results_df)
CSV.write(out_sum, sort(summary, :alpha, rev=true))

println("\n✓ Results saved to $(out_full)")
println("✓ Summary saved to $(out_sum)")
