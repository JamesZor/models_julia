# r02_smoke_test.jl

using Revise
using BayesianFootball
using DataFrames, Dates
using ThreadPinning

# Load the loader file
include("l02_portfolio_backtest.jl")

# -------------------------------------------------------------------
# 1. Environment & Threading Setup
# -------------------------------------------------------------------
pinthreads(:cores)
@info "Multithreading configured." n_threads = Threads.nthreads() n_cores = ThreadPinning.ncores()

# -------------------------------------------------------------------
# 2. Data Loading (Run once)
# -------------------------------------------------------------------
@info "Loading datastore and experiment latents..."
ds = D.load_datastore_cached(D.ScottishLower())
odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

src_dir = "./data/experiments/plus_minus_biweek"
list_of_experiments = E.list_experiments(src_dir, data_dir="")
expr = E.load_experiment(list_of_experiments, 3)

latents = E.extract_oos_predictions(ds, expr)
@info "Data successfully loaded."

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
# 4. Smoke Test 1: Single Match Validation
# -------------------------------------------------------------------
println("\n", "="^80)
println("=== SMOKE TEST 1: Single Match Evaluation (Row 200, Match #12476705) ===")
println("="^80)
row200 = latents.df[200, :]
res200 = evaluate_match(row200, expr, odds, markets_config, b_config)
display(res200)

# Expected Check: At alpha=1.0, we should see net_pl ≈ -0.0372 and total_stake ≈ 0.201

# -------------------------------------------------------------------
# 5. Smoke Test 2: Multithreaded Mini-Batch
# -------------------------------------------------------------------
BATCH_SIZE = 50
println("\n", "="^80)
println("=== SMOKE TEST 2: Running $(BATCH_SIZE)-Match Multithreaded Mini-Batch ===")
println("="^80)

batch_latents = latents.df[1:BATCH_SIZE, :]

# Run the multithreaded backtest
@time results_df = run_backtest(batch_latents, expr, odds, markets_config, b_config)

# -------------------------------------------------------------------
# 6. Aggregation and Summary
# -------------------------------------------------------------------
println("\n", "="^80)
println("=== BATCH SUMMARY RESULTS ===")
println("="^80)

summary = combine(groupby(results_df, :alpha),
    :match_id => length => :total_matches,
    :status => (x -> sum(x .== "SUCCESS")) => :valid_matches,
    :status => (x -> sum(x .== "MISSING_ODDS")) => :missing_odds,
    :total_stake => sum => :total_staked,
    :net_pl => sum => :total_net_pl,
    :bets_placed => sum => :total_bets
)

# Calculate ROI safely (avoid divide by zero)
summary.roi = ifelse.(summary.total_staked .> 0, (summary.total_net_pl ./ summary.total_staked) .* 100, 0.0)

# Format for pretty printing
summary.total_staked = round.(summary.total_staked, digits=2)
summary.total_net_pl = round.(summary.total_net_pl, digits=3)
summary.roi = round.(summary.roi, digits=2)

display(sort(summary, :alpha, rev=true))
println("\nSmoke tests completed successfully!")
