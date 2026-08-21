# current_development/scottish_lower/open_play/r07_eval_recomb_benchmark.jl
#
# EVALUATION & BETFAIR BACKTEST: Recombination Head-to-Head Comparison
#
# Compares:
# 1. goals_negbin_ctl_hl365_hs2: Baseline Gross Goals Control (All Goals)
# 2. goals_negbin_open_play_hl365_hs2: Pure Open-Play NegBin (Un-recombined)
# 3. goals_pois_open_play_hl365_hs2: Pure Open-Play Poisson (Un-recombined)
# 4. recomb_pois_integrated_hl365_hs2: Integrated Co-Trained Turing MCMC Recombination

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, LinearAlgebra

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Portfolio   = BayesianFootball.Portfolio
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")
include("l03_recombination_models.jl")

function banner(msg::String)
    println("\n", "="^95)
    println("  " * msg)
    println("="^95)
end

banner("🔍 EVALUATION & BETFAIR BACKTEST: RECOMBINATION VS ALL-GOALS BASELINE")

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded DataStore ($(nrow(ds.matches)) matches)")

# 2. Discover & Load Experiments
ctl_folders = Experiments.list_experiments("scottish_negbin_grid"; data_dir = joinpath(ROOT, "data"))
op_folders  = Experiments.list_experiments("scottish_open_play_grid"; data_dir = joinpath(ROOT, "data"))
all_folders = vcat(ctl_folders, op_folders)
all_loaded  = Experiments.load_experiments(all_folders)

target_models = [
    "goals_negbin_ctl_hl365_hs2",
    "goals_negbin_open_play_hl365_hs2",
    "goals_pois_open_play_hl365_hs2",
    "recomb_pois_integrated_hl365_hs2"
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

println("✓ Loaded $(length(experiments))/$(length(target_models)) target experiments:")
for exp in experiments
    println("  - $(exp.config.name) ($(length(exp.training_results.items)) folds)")
end

# 3. Standard Evaluation Suite
banner("📊 RUNNING COMPREHENSIVE EVALUATION SUITE (RQR, CRPS, LogLoss)")

selections = [:home, :draw, :away, :btts_yes, :over_25, :under_25]

fam = Dict(
    :x12    => [:home, :draw, :away],
    :btts   => [:btts_yes],
    :totals => [:over_25, :under_25],
)

_col(df, model, colname) = begin
    colname in names(df) || return NaN
    r = df[df.model .== model, colname]
    (isempty(r) || ismissing(r[1])) ? NaN : round(Float64(r[1]), digits = 4)
end

metrics = Evaluation.AbstractScoringRule[
    Evaluation.RQR(),
    Evaluation.CRPS()
]
append!(metrics, [Evaluation.LogLoss(s) for s in selections])

eval_df = Evaluation.evaluate_experiments(metrics, experiments, ds)
present_models = unique(eval_df.model)

banner("📈 1. RANDOMIZED QUANTILE RESIDUALS (RQR Calibration: Mean ~ 0.0, Std ~ 1.0)")
if "rqr_all_mean" in names(eval_df)
    rqr_df = DataFrame(
        model       = eval_df.model,
        mean_all    = [round(v, digits = 4) for v in eval_df.rqr_all_mean],
        std_all     = [round(v, digits = 4) for v in eval_df.rqr_all_std],
        mean_home   = [round(v, digits = 4) for v in eval_df.rqr_home_mean],
        std_home    = [round(v, digits = 4) for v in eval_df.rqr_home_std],
        mean_away   = [round(v, digits = 4) for v in eval_df.rqr_away_mean],
        std_away    = [round(v, digits = 4) for v in eval_df.rqr_away_std],
    )
    show(stdout, MIME("text/plain"), rqr_df)
    println()
end

banner("📉 2. CRPS & FAMILY-POOLED LOG LOSS DIFFERENCES (Lower / More Negative vs Market = Better)")
rows = []
for m in present_models
    crps_val = _col(eval_df, m, "crps_all_score")
    
    # 1X2 LogLoss diff vs market close
    h_ll = _col(eval_df, m, "logloss_home_diff")
    d_ll = _col(eval_df, m, "logloss_draw_diff")
    a_ll = _col(eval_df, m, "logloss_away_diff")
    
    # Family pooled
    x12_vals = [_col(eval_df, m, "logloss_$(s)_diff") for s in fam[:x12]]
    btts_vals = [_col(eval_df, m, "logloss_$(s)_diff") for s in fam[:btts]]
    tot_vals = [_col(eval_df, m, "logloss_$(s)_diff") for s in fam[:totals]]
    
    mean_x12  = round(mean(filter(!isnan, x12_vals)), digits = 5)
    mean_btts = round(mean(filter(!isnan, btts_vals)), digits = 5)
    mean_tot  = round(mean(filter(!isnan, tot_vals)), digits = 5)
    
    push!(rows, (
        model     = m,
        crps      = crps_val,
        LL_Home   = h_ll,
        LL_Draw   = d_ll,
        LL_Away   = a_ll,
        LL_1X2    = mean_x12,
        LL_BTTS   = mean_btts,
        LL_Totals = mean_tot
    ))
end
ll_summary_df = DataFrame(rows)
show(stdout, MIME("text/plain"), ll_summary_df)
println()

# 4. Betfair Exchange Portfolio Backtest
banner("💰 3. BETFAIR EXCHANGE MULTI-MARKET KELLY SIMULATION (2% Commission, BM 800 Draws)")

bf_summary = Data.summarize_betfair_market(ds)

MARKETS = Data.MarketConfig(reduce(vcat, (
    Data.AbstractMarket[Data.Market1X2(), Data.MarketBTTS()],
    [Data.MarketOverUnder(i + 0.5) for i in 0:4],
)))

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
    oos_latents = Experiments.extract_oos_predictions(ds, exp)
    b = Portfolio.build_books(spec, oos_latents.df, exp, bf_summary, ds)
    books_map[m_name] = b
end

all_slates = Dict{String, Vector{Portfolio.Slate}}()
for (m_name, b) in books_map
    all_slates[m_name] = Portfolio.group(Portfolio.DailySlate(), b)
end

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
        model        = String[],
        final_wealth = Float64[],
        growth_slate = Float64[],
        roi_pct      = Float64[],
        mean_expo    = Float64[],
        mdd_pct      = Float64[],
        sharpe       = Float64[],
        calmar       = Float64[],
        n_bets       = Int[]
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
        
        push!(port_df, (
            model        = m_name,
            final_wealth = round(m.final, digits=3),
            growth_slate = round(m.growth, digits=5),
            roi_pct      = round(m.roi, digits=2),
            mean_expo    = round(m.mean_exposure * 100.0, digits=1),
            mdd_pct      = round(m.mdd, digits=2),
            sharpe       = round(sh, digits=2),
            calmar       = round(calm, digits=2),
            n_bets       = m.n_bets
        ))
    end
    
    sort!(port_df, :final_wealth, rev=true)
    show(stdout, MIME("text/plain"), port_df)
    println()
end

banner("✓ Recombination Benchmark Evaluation Complete!")
