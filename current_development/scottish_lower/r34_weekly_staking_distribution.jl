# ==============================================================================
# Scottish Lower — Weekly Slate Staking Distribution & 4 Statistical Moments
# BayesianFootball.jl Unified V2 Stack
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, Printf, Statistics, StatsBase

println("\n" * "="^135)
println(" WEEKLY SLATE STAKING DISTRIBUTION & 4 STATISTICAL MOMENTS (24/25 + 25/26)")
println("="^135)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

# 2. Summarize Betfair Exchange Closing Odds
bf_raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
bf_odds = DataFrame(
    match_id    = Int.(bf_raw.match_id),
    market_name = String.(bf_raw.market_name),
    market_line = Float64.(bf_raw.market_line),
    selection   = Symbol.(bf_raw.selection),
    odds_close  = Float64.(bf_raw.odds)
)

# 3. Model Paths
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

# 5. Extract Weekly Staking Series
struct StakingMoments
    model_name::String
    likelihood::String
    n_weeks::Int
    mean_pct::Float64
    std_pct::Float64
    skewness::Float64
    kurtosis::Float64
    p25_pct::Float64
    median_pct::Float64
    p75_pct::Float64
    p95_pct::Float64
    max_pct::Float64
    mean_gbp_1k::Float64
    max_gbp_1k::Float64
end

stats_list = StakingMoments[]

for (label, p_name, nb_name) in model_pairs
    for (lik, dir_path, m_file) in [("Poisson", poisson_dir, p_name), ("NegBin", negbin_dir, nb_name)]
        full_path = resolve_fit_dir(joinpath(dir_path, m_file))
        isfile(joinpath(full_path, "results.jld2")) || continue
        
        fit = load_fit(full_path)
        res, books, _ = run_portfolio_simulation(book_spec, policy_spec, fit, bf_odds, ds;
                                                 bootstrap = false, require_converged = false)
        
        # Build ledger DataFrame with dates
        bets_df = res.bets
        
        # Match bet dates to calendar weeks
        # Look up kickoff dates from ds.matches
        match_date_map = Dict(r.match_id => r.kickoff for r in eachrow(ds.matches))
        
        # Calculate stake as % of current bankroll (or initial bankroll)
        # In bets_df: match_id, stake, stake_fraction, etc.
        # Group by Year-Week
        if :kickoff in propertynames(bets_df)
            bets_df.week = [(Dates.year(d), Dates.week(d)) for d in bets_df.kickoff]
        elseif :match_id in propertynames(bets_df)
            dates = [get(match_date_map, mid, Date(2024, 8, 1)) for mid in bets_df.match_id]
            bets_df.week = [(Dates.year(d), Dates.week(d)) for d in dates]
        else
            continue
        end
        
        # Weekly total stake fraction (%)
        weekly_stakes = combine(groupby(bets_df, :week), :stake => sum => :total_weekly_stake)
        w_vals = weekly_stakes.total_weekly_stake .* 100.0 # as percentage of nominal base
        
        if isempty(w_vals)
            continue
        end
        
        m_mean = mean(w_vals)
        m_std  = std(w_vals)
        m_skew = skewness(w_vals)
        m_kurt = kurtosis(w_vals) # excess kurtosis
        
        p25    = quantile(w_vals, 0.25)
        p50    = median(w_vals)
        p75    = quantile(w_vals, 0.75)
        p95    = quantile(w_vals, 0.95)
        p_max  = maximum(w_vals)
        
        push!(stats_list, StakingMoments(
            label, lik, length(w_vals),
            m_mean, m_std, m_skew, m_kurt,
            p25, p50, p75, p95, p_max,
            m_mean * 10.0, # for £1,000 bankroll
            p_max * 10.0
        ))
    end
end

println("\n" * "="^145)
println(" TABLE 1: FOUR STATISTICAL MOMENTS OF WEEKLY STAKING (% of Bankroll)")
println("="^145)
@printf(" %-18s | %8s | %6s | %10s | %10s | %10s | %12s | %14s | %14s\n",
        "Model", "Likelihood", "Weeks", "Mean (μ)", "Std (σ)", "Skewness (S)", "Ex.Kurt (K)", "Avg £/wk (£1k)", "Peak £/wk (£1k)")
println("-"^145)
for s in stats_list
    @printf(" %-18s | %10s | %6d | %9.2f%% | %9.2f%% | %10.3f | %12.3f |     £%7.2f    |     £%7.2f   \n",
            s.model_name, s.likelihood, s.n_weeks, s.mean_pct, s.std_pct, s.skewness, s.kurtosis, s.mean_gbp_1k, s.max_gbp_1k)
end
println("="^145)

println("\n" * "="^145)
println(" TABLE 2: QUANTILES OF WEEKLY STAKING EXPOSURE (% of Bankroll)")
println("="^145)
@printf(" %-18s | %8s | %10s | %10s | %10s | %10s | %10s\n",
        "Model", "Likelihood", "25th Pct", "Median (50th)", "75th Pct", "95th Pct", "Max Peak Stake")
println("-"^145)
for s in stats_list
    @printf(" %-18s | %10s | %9.2f%% | %12.2f%% | %9.2f%% | %9.2f%% | %13.2f%%\n",
            s.model_name, s.likelihood, s.p25_pct, s.median_pct, s.p75_pct, s.p95_pct, s.max_pct)
end
println("="^145)
