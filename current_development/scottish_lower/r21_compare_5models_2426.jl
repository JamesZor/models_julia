# ==============================================================================
# Scottish Lower — Compare 5 Models on 24/25 + 25/26 Seasons
# BayesianFootball.jl
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, Printf, Statistics, MCMCChains

println("\n" * "="^115)
println(" SCOTTISH LOWER: 5-MODEL 24/25 + 25/26 BENCHMARK & COMPARISON")
println("="^115)

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

models_to_compare = [
    "m00_baseline",
    "m02_wealth",
    "m03_distance",
    "m04_joint",
    "m05_production_wealth"
]

save_root = "./data/scottish_lower_2426_grid"

println("\nScanning for unified model fits in $save_root...")
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

# 1. Prediction Evaluation Metrics (Log-Loss, Brier, ECE, RPS)
println("\n" * "="^115)
println(" 1. OUT-OF-SAMPLE PREDICTION ACCURACY (Seasons 24/25 + 25/26)")
println("="^115)
@printf(" %-22s | %10s | %10s | %10s | %10s | %10s | %10s\n",
        "Model", "LogLoss", "Brier", "ECE", "RPS", "Market LL", "Market Brier")
println("-"^115)

for m_name in models_to_compare
    haskey(fits, m_name) || continue
    fit = fits[m_name]
    ev = evaluate_predictions(fit, ds)
    @printf(" %-22s | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f\n",
            m_name, ev.model.logloss, ev.model.brier, ev.model.ece, ev.model.rps,
            ev.market.logloss, ev.market.brier)
end

# 2. Parameter Posteriors Comparison
println("\n" * "="^115)
println(" 2. PARAMETER POSTERIOR ESTIMATES (Fold 1 Mean Posteriors)")
println("="^115)
@printf(" %-22s | %10s | %10s | %10s | %10s | %10s | %10s\n",
        "Model", "γ (HA)", "σ_att", "σ_def", "w_raw", "w_prod", "w_dist")
println("-"^115)

for m_name in models_to_compare
    haskey(fits, m_name) || continue
    fit = fits[m_name]
    ch = fit.folds[1].chain
    ch_vars = Set(Symbol.(names(ch)))
    
    get_stat(sym) = sym in ch_vars ? @sprintf("%.3f", mean(ch[sym])) : "—"
    
    ha_sym   = Symbol("ha.γ") in ch_vars ? Symbol("ha.γ") :
               Symbol("ha.γ_global") in ch_vars ? Symbol("ha.γ_global") : Symbol("ha.γ_raw")
    ha_str   = get_stat(ha_sym)
    sa_str   = get_stat(Symbol("dyn.σ_a"))
    sd_str   = get_stat(Symbol("dyn.σ_d"))
    w_raw    = get_stat(Symbol("wealth.w"))
    w_prod   = get_stat(Symbol("production_wealth.w"))
    w_dist   = get_stat(Symbol("distance.w"))
    
    @printf(" %-22s | %10s | %10s | %10s | %10s | %10s | %10s\n",
            m_name, ha_str, sa_str, sd_str, w_raw, w_prod, w_dist)
end

# 3. Fractional Kelly Portfolio & Staking Simulation
println("\n" * "="^115)
println(" 3. FRACTIONAL KELLY PORTFOLIO SIMULATION (DeArb + BakerMcHale Shrinkage + 20% Slate Cap)")
println("="^115)
@printf(" %-22s | %6s | %10s | %10s | %10s | %10s | %10s\n",
        "Model", "Bets", "Return %", "Flat ROI %", "1X2 ROI %", "Max DD %", "Sharpe (ann)")
println("-"^115)

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

for m_name in models_to_compare
    haskey(fits, m_name) || continue
    fit = fits[m_name]
    res, books, _ = run_portfolio_simulation(book_spec, policy_spec, fit, ds.odds, ds;
                                             bootstrap = false, require_converged = false)
    s = res.summary
    @printf(" %-22s | %6d | %9.2f%% | %9.2f%% | %9.2f%% | %9.2f%% | %10.3f\n",
            m_name, s.n_bets, s.total_return_pct, s.roi, s.roi_1x2, s.mdd, s.sharpe_ann)
end

println("="^115)
println(" 5-Model comparison report generated successfully.")
