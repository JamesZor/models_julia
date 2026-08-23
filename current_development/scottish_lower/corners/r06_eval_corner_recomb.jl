# current_development/scottish_lower/corners/r06_eval_corner_recomb.jl
#
# EVALUATION & BETFAIR BACKTEST: 4-Way Goal & Corner Recombination Benchmark
#
# Compares:
# 1. goals_pois_ctl_hl365_hs2: Gross Goals Poisson Control
# 2. goals_pois_open_play_hl365_hs2: Pure Open-Play Poisson
# 3. recomb_pois_integrated_hl365_hs2: 3-Way Recombined Poisson (Open Play + Penalties + Own Goals)
# 4. recomb_corner_integrated_hl365_hs2: 4-Way Goal & Corner Recombination

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, LinearAlgebra

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Portfolio   = BayesianFootball.Portfolio
const Signals     = BayesianFootball.Signals
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_corner_data.jl")
include("l05_corner_recomb_pipeline.jl")

# Also include open play definitions for loading existing baselines
const OP_DIR = joinpath(ROOT, "current_development/scottish_lower/open_play")
if isdir(OP_DIR)
    include(joinpath(OP_DIR, "l01_open_play_feature.jl"))
    include(joinpath(OP_DIR, "l02_open_play_engines.jl"))
    include(joinpath(OP_DIR, "l03_recombination_models.jl"))
    include(joinpath(OP_DIR, "l04_recomb_wealth_models.jl"))
    include(joinpath(OP_DIR, "l05_recomb_pxg_models.jl"))
end

function banner(msg::String)
    println("\n", "="^95)
    println("  " * msg)
    println("="^95)
end

banner("🔍 EVALUATION & BETFAIR BACKTEST: 4-WAY CORNER RECOMBINATION BENCHMARK")

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded Scottish Lower DataStore ($(nrow(ds.matches)) matches)")

# 2. Discover & Load Experiments
grid_folders_corner = Experiments.list_experiments("scottish_corner_grid"; data_dir = joinpath(ROOT, "data"))
grid_folders_op     = Experiments.list_experiments("scottish_open_play_grid"; data_dir = joinpath(ROOT, "data"))
all_folders         = vcat(grid_folders_corner, grid_folders_op)
all_loaded          = Experiments.load_experiments(all_folders)

target_models = [
    "goals_pois_ctl_hl365_hs2",
    "goals_pois_open_play_hl365_hs2",
    "recomb_pois_integrated_hl365_hs2",
    "recomb_corner_integrated_hl365_hs2"
]

experiments_dict = Dict{String, Any}()
for exp in all_loaded
    if length(exp.training_results.items) > 0
        for t in target_models
            if startswith(exp.config.name, t)
                if !haskey(experiments_dict, t) || exp.save_path > experiments_dict[t].save_path
                    experiments_dict[t] = exp
                end
            end
        end
    end
end

experiments = [experiments_dict[t] for t in target_models if haskey(experiments_dict, t)]

println("✓ Loaded $(length(experiments))/$(length(target_models)) target experiments:")
for exp in experiments
    println("  - $(exp.config.name) ($(length(exp.training_results.items)) folds)")
end

# 3. Standard Evaluation Suite
banner("📊 RUNNING COMPREHENSIVE EVALUATION SUITE (RQR, CRPS, LogLoss on 15 Markets)")

selections = [:home, :draw, :away, :btts_yes, :btts_no,
              :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
              :over_35, :under_35, :over_45, :under_45]

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

banner("📉 2. CRPS & LOG LOSS EVALUATION (Lower / More Negative vs Market = Better)")

if "crps_all_mean" in names(eval_df)
    crps_df = DataFrame(
        model      = eval_df.model,
        crps_all   = [round(v, digits = 4) for v in eval_df.crps_all_mean],
        crps_home  = [round(v, digits = 4) for v in eval_df.crps_home_mean],
        crps_away  = [round(v, digits = 4) for v in eval_df.crps_away_mean]
    )
    sort!(crps_df, :crps_all)
    show(stdout, MIME("text/plain"), crps_df)
    println()
end

println("\n" * "="^85)
println("📊 FAMILY-POOLED LOGLOSS DIFF (Model − Market De-Vigged Close; Negative is Better)")
println("="^85)
f_df = DataFrame(model = present_models)
for (fname, sels) in fam
    f_df[!, fname] = [round(mean(filter(!isnan,
        [_col(eval_df, mm, "logloss_$(s)_overall_diff_ll") for s in sels])), digits = 5)
        for mm in present_models]
end
show(stdout, MIME("text/plain"), f_df)
println()

println("\n" * "="^85)
println("📉 1X2 & BTTS LOGLOSS DIFF BY SELECTION")
println("="^85)
x12_df = DataFrame(
    model    = present_models,
    home     = [_col(eval_df, mm, "logloss_home_overall_diff_ll") for mm in present_models],
    draw     = [_col(eval_df, mm, "logloss_draw_overall_diff_ll") for mm in present_models],
    away     = [_col(eval_df, mm, "logloss_away_overall_diff_ll") for mm in present_models],
    btts_yes = [_col(eval_df, mm, "logloss_btts_yes_overall_diff_ll") for mm in present_models],
    btts_no  = [_col(eval_df, mm, "logloss_btts_no_overall_diff_ll") for mm in present_models]
)
show(stdout, MIME("text/plain"), x12_df)
println()

println("\n" * "="^85)
println("📉 TOTALS OVER/UNDER LOGLOSS DIFF BY LINE")
println("="^85)
tot_df = DataFrame(
    model    = present_models,
    over_15  = [_col(eval_df, mm, "logloss_over_15_overall_diff_ll") for mm in present_models],
    under_15 = [_col(eval_df, mm, "logloss_under_15_overall_diff_ll") for mm in present_models],
    over_25  = [_col(eval_df, mm, "logloss_over_25_overall_diff_ll") for mm in present_models],
    under_25 = [_col(eval_df, mm, "logloss_under_25_overall_diff_ll") for mm in present_models],
    over_35  = [_col(eval_df, mm, "logloss_over_35_overall_diff_ll") for mm in present_models],
    under_35 = [_col(eval_df, mm, "logloss_under_35_overall_diff_ll") for mm in present_models]
)
show(stdout, MIME("text/plain"), tot_df)
println()

# ==============================================================================
# 4. FULL-MARKET BOOKMAKER KELLY BACKTEST (All 710 Matches, ds.odds)
# ==============================================================================
banner("📈 4. FULL-MARKET BOOKMAKER BACKTEST (All 710 OOS Matches, Signals.BayesianKelly)")

try
    ledger = BackTesting.run_backtest(
        ds, experiments, [Signals.BayesianKelly()];
        market_config = Data.Markets.DEFAULT_MARKET_CONFIG
    )
    tearsheet = BackTesting.generate_tearsheet(ledger; groupby_cols = [:model_name])
    show(stdout, MIME("text/plain"), tearsheet)
    println()
catch e
    @warn "Full bookmaker backtest skipped: $e"
end

banner("✓ 4-WAY CORNER RECOMBINATION BENCHMARK COMPLETE")
