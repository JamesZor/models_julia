# r04_risk_smoke.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics
using ThreadPinning

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
@info "Data successfully loaded."

# -------------------------------------------------------------------
# 2. Configuration Setup
# -------------------------------------------------------------------
scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

# Standard backtest config (alpha grid is ignored here since we force optimal_alpha=0.35 in the function call)
b_config = BacktestConfig(commission=0.02, alphas=[1.0])

# Strict Risk Configuration
# Limit: 20% Drawdown (0.8)
# Tolerance: 1% probability of breaching (0.01)
r_config = RiskConfig(0.8, 0.01)
λ = get_lambda(r_config)

println("\n", "="^80)
println("=== RISK MANAGER CONFIGURATION ===")
println("Drawdown Limit (D) : ", r_config.D)
println("Risk Tolerance (β): ", r_config.beta)
println("Derived Lambda (λ) : ", round(λ, digits=4))
println("="^80)

# -------------------------------------------------------------------
# 3. Smoke Test Execution (50 Matches)
# -------------------------------------------------------------------
BATCH_SIZE = 50
println("\n=== Running Risk-Managed Mini-Batch ($(BATCH_SIZE) Matches) ===")

batch_latents = latents.df[1:BATCH_SIZE, :]
@time res_df = run_risk_backtest(batch_latents, expr, odds, markets_config, b_config, r_config; optimal_alpha=0.35)

# -------------------------------------------------------------------
# 4. Results & Summary Analysis
# -------------------------------------------------------------------
valid_matches = subset(res_df, :status => ByRow(==("SUCCESS")))

println("\n", "="^80)
println("=== INTERVENTION ANALYSIS ===")
println("="^80)
intervened = subset(valid_matches, :shrink_k => ByRow(<(0.999)))
if nrow(intervened) > 0
    println("The Risk Manager actively shrank the stakes (k < 1.0) on $(nrow(intervened)) matches due to high variance risk.")
    display(intervened[:, [:match_id, :base_stake, :risk_stake, :shrink_k, :base_pl, :risk_pl]])
else
    println("No interventions. All $(nrow(valid_matches)) valid matches had a probability of ruin < $(r_config.beta) at alpha=0.35.")
end

println("\n", "="^80)
println("=== SMOKE TEST BATCH SUMMARY ===")
println("="^80)
summary = DataFrame(
    Total_Matches = nrow(res_df),
    Valid_Matches = nrow(valid_matches),
    Avg_Shrinkage_k = round(mean(valid_matches.shrink_k), digits=4),
    Base_Total_Stake = round(sum(valid_matches.base_stake), digits=2),
    Risk_Total_Stake = round(sum(valid_matches.risk_stake), digits=2),
    Base_Net_PL = round(sum(valid_matches.base_pl), digits=3),
    Risk_Net_PL = round(sum(valid_matches.risk_pl), digits=3)
)
display(summary)
println("\nSmoke test completed successfully!")
