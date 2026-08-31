# ==============================================================================
# Scottish Lower — 5-Model Betfair Exchange Portfolio Backtest (24/25 + 25/26)
# BayesianFootball.jl Unified V2 Stack
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, Printf, Statistics, MCMCChains

println("\n" * "="^115)
println(" SCOTTISH LOWER: 5-MODEL BETFAIR EXCHANGE PORTFOLIO BACKTEST (24/25 + 25/26)")
println("="^115)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

# 2. Summarize Betfair Exchange Closing Odds (TWA in [-20min, 0min] before kickoff)
println("\n[1/3] Summarizing Betfair Exchange closing prices (TWA [-20min, 0min])...")
bf_raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
bf_odds = DataFrame(
    match_id    = Int.(bf_raw.match_id),
    market_name = String.(bf_raw.market_name),
    market_line = Float64.(bf_raw.market_line),
    selection   = Symbol.(bf_raw.selection),
    odds_close  = Float64.(bf_raw.odds)
)
println("  Betfair quotes: $(nrow(bf_odds)) across $(length(unique(bf_odds.match_id))) matches")

# 3. Load Saved 5 Model Fits
models_to_compare = [
    "m00_baseline",
    "m02_wealth",
    "m03_distance",
    "m04_joint",
    "m05_production_wealth"
]

save_root = "./data/scottish_lower_2426_grid"
println("\n[2/3] Scanning for unified model fits in $save_root...")
fits = Dict{String, Fit}()

for m_name in models_to_compare
    dir = joinpath(save_root, m_name)
    if isdir(dir)
        found = list_fits(dir)
        if !isempty(found)
            latest = last(found)
            println(" Loaded $(m_name): $(latest.path)")
            fits[m_name] = load_fit(latest.path)
        end
    end
end

if isempty(fits)
    println("No unified fits found in $save_root. Please run r20_train_5models_2426_unified.jl first.")
    exit(0)
end

# 4. Configure Betfair Portfolio & Book Specifications
# 2% net Betfair commission, DeArb pricing, Baker-McHale parameter shrinkage
book_spec = BookSpec(
    markets   = Data.MarketConfig([
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS()
    ]),
    price     = DeArb(),
    allocator = KellyLogUtility(),
    shrink    = BakerMcHale(),
    exec      = ExecutionConfig(
        commission          = PerBetCommission(0.02), # Betfair 2%
        budget              = 0.99,
        min_selection_stake = 0.001
    )
)

policy_spec = PolicySpec(
    trust    = FlatTrust(0.30),
    risk     = SlateDrawdown(23.0),
    cap      = FixedCap(0.20),
    grouping = DailySlate()
)

# 5. Execute Portfolio Backtests on Betfair Odds
println("\n[3/3] Simulating Betfair Exchange Portfolio Trajectories...")

results = Dict{String, Any}()

for m_name in models_to_compare
    haskey(fits, m_name) || continue
    fit = fits[m_name]
    
    res, books, _ = run_portfolio_simulation(book_spec, policy_spec, fit, bf_odds, ds;
                                             bootstrap = false, require_converged = false)
    results[m_name] = res
    s = res.summary
    
    println("\n--- Betfair Summary: $m_name ---")
    display(portfolio_report(res))
end

# 6. Formatted Comparison Table
println("\n" * "="^125)
println(" BETFAIR EXCHANGE 5-MODEL HEAD-TO-HEAD LEADERBOARD (Seasons 24/25 + 25/26)")
println("="^125)
@printf(" %-22s | %6s | %10s | %10s | %10s | %10s | %10s | %8s\n",
        "Model", "Bets", "Return %", "Flat ROI %", "1X2 ROI %", "Max DD %", "Sharpe (ann)", "Win Rate")
println("-"^125)

for m_name in models_to_compare
    haskey(results, m_name) || continue
    s = results[m_name].summary
    @printf(" %-22s | %6d | %9.2f%% | %9.2f%% | %9.2f%% | %9.2f%% | %10.3f | %7.2f%%\n",
            m_name, s.n_bets, s.total_return_pct, s.roi, s.roi_1x2, s.mdd, s.sharpe_ann, 100 * s.win_rate)
end
println("="^125)
println(" Betfair Exchange portfolio benchmark completed successfully!")
