# current_development/scottish_wealth/r04_eval_wealth.jl
#
# RUNNER: Multi-Model Evaluation & Betfair Growth Benchmark for Scottish Wealth Models
#
# Loads all control models from `data/scottish_pxg_grid/` alongside the newly trained
# wealth models from `data/scottish_wealth_grid/` and evaluates:
# 1. Per-Line LogLoss vs De-Vigged Market Close (1X2, BTTS, Totals)
# 2. Family-Pooled Mean LogLoss Diff
# 3. Betfair Exchange Portfolio Wealth Growth & ROI (Kelly staking)

using Revise
using BayesianFootball
using DataFrames
using Statistics
using Printf

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const BackTesting = BayesianFootball.BackTesting
const Signals     = BayesianFootball.Signals

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l02_pxg_engines.jl"))
include("l02_wealth_engines.jl")

println("="^95)
println("SCOTTISH LOWER WEALTH MODELS: MULTI-MODEL EVALUATION & BENCHMARK")
println("="^95)

# 1. LOAD EXPERIMENTS FROM BOTH GRID DIRECTORIES
pxg_folders    = Experiments.list_experiments("scottish_pxg_grid"; data_dir = joinpath(ROOT, "data"))
wealth_folders = Experiments.list_experiments("scottish_wealth_grid"; data_dir = joinpath(ROOT, "data"))

all_folders = vcat(pxg_folders, wealth_folders)
all_results = Experiments.load_experiments(all_folders)

# Filter to key comparison models
target_models = [
    "funnel_apm_ctl_hl365_hs2",
    "pxg_apm_hl365_hs2",
    "pxg_apm_wealth_hl365_hs2",
    "funnel_pxg_apm_hl365_hs2",
    "funnel_pxg_apm_wealth_hl365_hs2"
]

results = filter(r -> any(m -> startswith(r.config.name, m), target_models), all_results)

println("[INFO] Loaded $(length(results)) benchmark models:")
for r in results
    println("  - $(r.config.name) ($(length(r.training_results.items)) folds)")
end

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 720)

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

# 2. EVALUATE SCORING RULES
println("\n--- EVALUATING OUT-OF-SAMPLE LOGLOSS ---")
metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
append!(metric, [Evaluation.LogLoss(s) for s in selections])

metrics_eval = Evaluation.evaluate_experiments(metric, results, ds)
present_models = unique(metrics_eval.model)

println("\n" * "="^85)
println("📊 FAMILY-POOLED LOGLOSS DIFF (Model − Market De-Vigged Close; Negative is Better)")
println("="^85)

f_df = DataFrame(model = present_models)
for (fname, sels) in fam
    f_df[!, fname] = [round(mean(filter(!isnan,
        [_col(metrics_eval, mm, "logloss_$(s)_overall_diff_ll") for s in sels])), digits = 5)
        for mm in present_models]
end
show(f_df; allrows = true, allcols = true, truncate = 0); println()

println("\n" * "="^85)
println("📉 1X2 LOGLOSS DIFF BY SELECTION (Home / Draw / Away)")
println("="^85)
x12_df = DataFrame(
    model = present_models,
    home  = [_col(metrics_eval, mm, "logloss_home_overall_diff_ll") for mm in present_models],
    draw  = [_col(metrics_eval, mm, "logloss_draw_overall_diff_ll") for mm in present_models],
    away  = [_col(metrics_eval, mm, "logloss_away_overall_diff_ll") for mm in present_models]
)
show(x12_df; allrows = true, allcols = true, truncate = 0); println()

println("\n✓ Evaluation complete. Next: inspect head-to-head gains from wealth integration!")
