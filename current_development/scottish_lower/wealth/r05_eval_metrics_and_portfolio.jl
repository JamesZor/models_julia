# current_development/scottish_wealth/r05_eval_metrics_and_portfolio.jl
#
# RUNNER: Comprehensive Evaluation Suite (LogLoss, GLMEdge, RQR, CRPS)
#         & Betfair Exchange Multi-Market Portfolio Backtest
#
# Comparing 5 Models:
# 1. funnel_apm_ctl_hl365_hs2         (Baseline Goals Funnel Control)
# 2. pxg_apm_hl365_hs2                (Arm A: Proxy xG Control)
# 3. pxg_apm_wealth_hl365_hs2         (Arm A + Starting-XI Squad Wealth)
# 4. funnel_pxg_apm_hl365_hs2         (Arm B: Champion 3-Layer Control)
# 5. funnel_pxg_apm_wealth_hl365_hs2  (Arm B Champion + Starting-XI Squad Wealth)

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Portfolio   = BayesianFootball.Portfolio
const BackTesting = BayesianFootball.BackTesting
const Signals     = BayesianFootball.Signals
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l02_pxg_engines.jl"))
include("l01_wealth_data.jl")
include("l02_wealth_engines.jl")

banner(s) = (println("\n", "="^95); println(s); println("="^95))

banner("1. LOADING DATASTORE & EXPERIMENTS")

CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 720)

pxg_folders    = Experiments.list_experiments("scottish_pxg_grid"; data_dir = joinpath(ROOT, "data"))
wealth_folders = Experiments.list_experiments("scottish_wealth_grid"; data_dir = joinpath(ROOT, "data"))
all_folders    = vcat(pxg_folders, wealth_folders)
all_loaded     = Experiments.load_experiments(all_folders)

target_models = [
    "funnel_apm_ctl_hl365_hs2",
    "pxg_apm_hl365_hs2",
    "pxg_apm_wealth_hl365_hs2",
    "funnel_pxg_apm_hl365_hs2",
    "funnel_pxg_apm_wealth_hl365_hs2"
]

experiments_dict = Dict{String, Any}()
for exp in all_loaded
    for t in target_models
        if startswith(exp.config.name, t)
            experiments_dict[t] = exp
        end
    end
end

experiments = [experiments_dict[t] for t in target_models if haskey(experiments_dict, t)]

println("✓ Loaded DataStore ($(nrow(ds.matches)) matches)")
println("✓ Loaded $(length(experiments)) benchmark experiments:")
for exp in experiments
    println("  - $(exp.config.name) ($(length(exp.training_results.items)) folds)")
end

# ==============================================================================
# 2. EVALUATION SUITE: LOGLOSS, GLMEDGE, RQR, CRPS
# ==============================================================================
banner("2. EXECUTING COMPREHENSIVE EVALUATION SUITE (RQR, GLMEdge, LogLoss, CRPS)")

selections = [:home, :draw, :away, :btts_yes, :btts_no,
              :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
              :over_35, :under_35, :over_45, :under_45]

metrics = Evaluation.AbstractScoringRule[
    Evaluation.RQR(),
    Evaluation.CRPS()
]
append!(metrics, [Evaluation.LogLoss(s) for s in selections])
append!(metrics, [Evaluation.GLMEdge(s) for s in selections])

eval_df = Evaluation.evaluate_experiments(metrics, experiments, ds)

banner("3. EVALUATION METRIC TABLES")

# A. RQR Summary (Randomized Quantile Residuals: Mean ~ 0.0, Std ~ 1.0)
println("\n" * "="^85)
println("📈 RANDOMIZED QUANTILE RESIDUALS (RQR Calibration: Mean ~ 0.0, Std ~ 1.0)")
println("="^85)
if "rqr_all_mean" in names(eval_df)
    rqr_df = DataFrame(
        model       = eval_df.model,
        mean_all    = [round(v, digits = 4) for v in eval_df.rqr_all_mean],
        std_all     = [round(v, digits = 4) for v in eval_df.rqr_all_std],
        mean_home   = [round(v, digits = 4) for v in eval_df.rqr_home_mean],
        std_home    = [round(v, digits = 4) for v in eval_df.rqr_home_std],
        mean_away   = [round(v, digits = 4) for v in eval_df.rqr_away_mean],
        std_away    = [round(v, digits = 4) for v in eval_df.rqr_away_std]
    )
    show(rqr_df; allrows = true, allcols = true, truncate = 0); println()
end

# B. Family-Pooled GLMEdge Summary (Spread Fair Coef: Positive = Model edge over market close)
fam = Dict(
    :x12    => [:home, :draw, :away],
    :btts   => [:btts_yes, :btts_no],
    :totals => [:over_05, :under_05, :over_15, :under_15, :over_25, :under_25, :over_35, :under_35, :over_45, :under_45],
)

_col(df, model, colname) = begin
    colname in names(df) || return NaN
    r = df[df.model .== model, colname]
    (isempty(r) || ismissing(r[1])) ? NaN : round(Float64(r[1]), digits = 4)
end

present_models = unique(eval_df.model)

println("\n" * "="^85)
println("🎯 GLMEDGE: MARKET SPREAD FAIR COEF (Positive = Structural Edge over Close)")
println("="^85)
glm_df = DataFrame(model = present_models)
for (fname, sels) in fam
    glm_df[!, fname] = [round(mean(filter(!isnan,
        [_col(eval_df, mm, "glmedge_$(s)_spread_fair_coef") for s in sels])), digits = 4)
        for mm in present_models]
end
show(glm_df; allrows = true, allcols = true, truncate = 0); println()

println("\n" * "="^85)
println("📊 FAMILY-POOLED LOGLOSS DIFF (Model − Market De-Vigged Close; Negative is Better)")
println("="^85)
f_df = DataFrame(model = present_models)
for (fname, sels) in fam
    f_df[!, fname] = [round(mean(filter(!isnan,
        [_col(eval_df, mm, "logloss_$(s)_overall_diff_ll") for s in sels])), digits = 5)
        for mm in present_models]
end
show(f_df; allrows = true, allcols = true, truncate = 0); println()

# ==============================================================================
# 3. BETFAIR EXCHANGE PORTFOLIO BENCHMARK
# ==============================================================================
banner("4. BETFAIR EXCHANGE MULTI-MARKET PORTFOLIO BENCHMARK")

# 1. Summarize Betfair Exchange Closing Odds
ODDS_CACHE = joinpath(CACHE_DIR, "betfair_summary_odds.jls")
local betfair_odds
if isfile(ODDS_CACHE)
    @info "Restoring Betfair closing summary from cache" ODDS_CACHE
    betfair_odds = deserialize(ODDS_CACHE)
else
    @info "Building Betfair closing summary [-20min, 0min] (cached)..."
    betfair_odds = Data.summarize_betfair_market(ds, open_window = (-100000.0, -10.0), close_window = (-20.0, 0.0))
    serialize(ODDS_CACHE, betfair_odds)
end
@info "Betfair Odds loaded" n_matches=length(unique(betfair_odds.match_id)) n_quotes=nrow(betfair_odds)

# 2. Extract / Restore Out-of-Sample Latents
LATENTS_CACHE = joinpath(CACHE_DIR, "latents_map_scottish_wealth.jls")
local latents_map
if isfile(LATENTS_CACHE)
    @info "Restoring OOS latents from cache" LATENTS_CACHE
    latents_map = deserialize(LATENTS_CACHE)
else
    @info "Extracting OOS predictions for all benchmark models..."
    latents_map = Dict{String, DataFrame}()
    for exp in experiments
        latents_map[exp.config.name] = Experiments.extract_oos_predictions(ds, exp).df
    end
    serialize(LATENTS_CACHE, latents_map)
end

# 3. Betting Markets Universe (1X2, BTTS, O/U 0.5 - 4.5)
MARKETS = Data.MarketConfig(reduce(vcat, (
    Data.AbstractMarket[Data.Market1X2(), Data.MarketBTTS()],
    [Data.MarketOverUnder(i + 0.5) for i in 0:4],
)))

# 4. BookSpec with 2% Betfair Commission and 800 Posterior Draws
spec = Portfolio.BookSpec(
    markets   = MARKETS,
    price     = Portfolio.DeArb(),
    allocator = Portfolio.KellyLogUtility(),
    shrink    = Portfolio.BakerMcHale(n_draws = 800),
    exec      = Portfolio.ExecutionConfig(
                    commission = Portfolio.PerBetCommission(0.02),
                    max_selection_stake = 0.50,
                    budget = 0.99,
                    require_complete_markets = true
                )
)

books_map = Dict{String, Vector{Portfolio.MatchBook}}()
for exp in experiments
    m_name = exp.config.name
    cache_file = joinpath(CACHE_DIR, "books_$(m_name)_betfair_bm800.jls")
    
    if isfile(cache_file) && get(ENV, "REBUILD_BOOKS", "0") != "1"
        @info "Reusing cached Betfair MatchBooks for: $m_name" cache_file
        books_map[m_name] = deserialize(cache_file)
    else
        @info "Building Betfair MatchBooks for: $m_name..."
        m_latents = latents_map[m_name]
        t0 = time()
        b = Portfolio.build_books(spec, m_latents, exp, betfair_odds, ds)
        elapsed = round(time() - t0, digits = 1)
        @info "Completed MatchBooks for $m_name in $(elapsed)s" n_books=length(b)
        serialize(cache_file, b)
        books_map[m_name] = b
    end
end

all_slates = Dict{String, Vector{Portfolio.Slate}}()
for (m_name, b) in books_map
    all_slates[m_name] = Portfolio.group(Portfolio.DailySlate(), b)
end

# Reference Portfolio Policies
policies = [
    ("Conservative (Cap 10%, λ=23)", Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(0.25),
        risk     = Portfolio.SlateDrawdown(23.0),
        cap      = Portfolio.FixedCap(0.10),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    )),
    ("Balanced Growth (Cap 15%, λ=15)", Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(0.25),
        risk     = Portfolio.SlateDrawdown(15.0),
        cap      = Portfolio.FixedCap(0.15),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    )),
    ("Aggressive (Cap 25%, λ=10)", Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(0.50),
        risk     = Portfolio.SlateDrawdown(10.0),
        cap      = Portfolio.FixedCap(0.25),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    ))
]

for (pol_name, pol) in policies
    println("\n", "="^95)
    println("BETFAIR PORTFOLIO SIMULATION: $pol_name (2% Comm, Baker-McHale 800 Draws)")
    println("="^95)
    
    port_df = DataFrame(
        model = String[],
        final_wealth = Float64[],
        growth_slate = Float64[],
        roi_pct = Float64[],
        mean_expo = Float64[],
        mdd_pct = Float64[],
        sharpe = Float64[],
        calmar = Float64[],
        n_bets = Int[]
    )
    
    for exp in experiments
        m_name = exp.config.name
        haskey(all_slates, m_name) || continue
        slates = all_slates[m_name]
        traj = Portfolio.simulate(pol, slates; use_shrink = true)
        m = Portfolio.path_metrics(traj)
        
        ret_series = traj.slate_pl
        sh = length(ret_series) > 1 && std(ret_series) > 1e-6 ? (mean(ret_series) / std(ret_series)) * sqrt(35) : 0.0
        calm = m.mdd > 0.0 ? (m.final - 1.0) / (m.mdd / 100.0) : 0.0
        
        total_bets = sum(sum(b.a_kelly .> 1e-5) for b in books_map[m_name])
        
        push!(port_df, (
            m_name,
            round(m.final, digits = 3),
            round(m.growth_per_slate, digits = 5),
            round(m.roi, digits = 2),
            round(m.mean_exposure * 100, digits = 1),
            round(m.mdd, digits = 2),
            round(sh, digits = 2),
            round(calm, digits = 2),
            total_bets
        ))
    end
    
    show(port_df; allrows = true, allcols = true, truncate = 0); println()
end

println("\n✓ Comprehensive Evaluation and Betfair Portfolio Backtest Complete!")
