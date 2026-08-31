# ==============================================================================
# Scottish Lower — Poisson vs NegBin 5-Model Betfair Portfolio Benchmark (24/26)
# BayesianFootball.jl Unified V2 Stack
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, Printf, Statistics, MCMCChains

println("\n" * "="^125)
println(" SCOTTISH LOWER: POISSON VS NEGBIN BETFAIR EXCHANGE PORTFOLIO BENCHMARK (24/25 + 25/26)")
println("="^125)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

# 2. Summarize Betfair Exchange Closing Odds
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

# 3. Directories & Models
poisson_candidates = [
    "/root/BayesianFootball/data/scottish_lower_2426_grid",
    "/root/BayesianFootball/experiments/scottish_lower_2426",
    joinpath(pwd(), "data", "scottish_lower_2426_grid")
]
poisson_dir = something(findfirst(isdir, poisson_candidates), 1) |> i -> poisson_candidates[i]
negbin_dir  = isdir("/root/BayesianFootball/experiments/scottish_lower_2426_negbin") ? 
    "/root/BayesianFootball/experiments/scottish_lower_2426_negbin" :
    joinpath(pwd(), "experiments", "scottish_lower_2426_negbin")

function resolve_fit_dir(path::String)
    isfile(joinpath(path, "results.jld2")) && return path
    if isdir(path)
        subdirs = filter(d -> isfile(joinpath(path, d, "results.jld2")), readdir(path))
        if !isempty(subdirs)
            sort!(subdirs; rev = true)
            return joinpath(path, first(subdirs))
        end
    end
    return path
end

model_pairs = [
    ("Baseline",          "m00_baseline",          "m00_negbin_baseline"),
    ("Squad Wealth",      "m02_wealth",            "m02_negbin_wealth"),
    ("Travel Distance",   "m03_distance",          "m03_negbin_distance"),
    ("Joint Wealth+Dist", "m04_joint",             "m04_negbin_joint"),
    ("Production Wealth", "m05_production_wealth", "m05_negbin_production_wealth"),
]

# 4. Book and Policy Specifications
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
        commission          = PerBetCommission(0.02),
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

# 5. Run Simulations
results_p  = Dict{String, Any}()
results_nb = Dict{String, Any}()

println("\n[2/3] Simulating Portfolio Trajectories on Betfair Odds...")

for (label, p_name, nb_name) in model_pairs
    p_path  = resolve_fit_dir(joinpath(poisson_dir, p_name))
    nb_path = resolve_fit_dir(joinpath(negbin_dir, nb_name))
    
    if isfile(joinpath(p_path, "results.jld2"))
        fit_p = load_fit(p_path)
        res_p, _, _ = run_portfolio_simulation(book_spec, policy_spec, fit_p, bf_odds, ds;
                                              bootstrap = false, require_converged = false)
        results_p[label] = res_p
    end
    
    if isfile(joinpath(nb_path, "results.jld2"))
        fit_nb = load_fit(nb_path)
        res_nb, _, _ = run_portfolio_simulation(book_spec, policy_spec, fit_nb, bf_odds, ds;
                                                bootstrap = false, require_converged = false)
        results_nb[label] = res_nb
    end
end

# 6. Combined Comparison Table
println("\n" * "="^140)
println(" BETFAIR EXCHANGE 2-SEASON (24/25 + 25/26) HEAD-TO-HEAD PORTFOLIO LEADERBOARD")
println("="^140)
@printf(" %-20s | %8s | %10s | %10s | %10s | %10s | %10s | %10s | %8s\n",
        "Model Architecture", "Likelihood", "Bets", "Return %", "Flat ROI %", "1X2 ROI %", "Max DD %", "Sharpe(ann)", "Win Rate")
println("-"^140)

for (label, _, _) in model_pairs
    if haskey(results_p, label)
        sp = results_p[label].summary
        @printf(" %-20s | %10s | %8d | %9.2f%% | %9.2f%% | %9.2f%% | %9.2f%% | %10.3f | %7.2f%%\n",
                label, "Poisson", sp.n_bets, sp.total_return_pct, sp.roi, sp.roi_1x2, sp.mdd, sp.sharpe_ann, 100 * sp.win_rate)
    end
    if haskey(results_nb, label)
        snb = results_nb[label].summary
        @printf(" %-20s | %10s | %8d | %9.2f%% | %9.2f%% | %9.2f%% | %9.2f%% | %10.3f | %7.2f%%\n",
                label, "NegBin", snb.n_bets, snb.total_return_pct, snb.roi, snb.roi_1x2, snb.mdd, snb.sharpe_ann, 100 * snb.win_rate)
    end
    println("-"^140)
end
println("="^140)
