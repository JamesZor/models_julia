# current_development/scottish_lower/open_play/r07_smoke_eval.jl
#
# FAST SMOKE TEST: Validates experiment loading, OOS extraction, and inference
# without waiting for the full 15-market batch evaluation.

using Revise
using BayesianFootball
using DataFrames, Statistics, Printf

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Portfolio   = BayesianFootball.Portfolio
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")
include("l03_recombination_models.jl")

println("\n", "="^85)
println("🔍 FAST SMOKE TEST: EXPERIMENTS, OOS EXTRACTION & SCORE MATRICES")
println("="^85)

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

experiments = [first(filter(x -> startswith(x.config.name, t), all_loaded)) for t in target_models if any(x -> startswith(x.config.name, t), all_loaded)]

println("✓ Successfully loaded $(length(experiments))/$(length(target_models)) experiments:")
for exp in experiments
    println("  - $(exp.config.name) ($(length(exp.training_results.items)) folds)")
end

# 3. Test OOS Latents Extraction & Caching for each model
println("\n--- Testing OOS Latent Extraction & Score Matrices ---")
for exp in experiments
    t0 = time()
    latents = Experiments.extract_oos_predictions(ds, exp)
    elapsed = round((time() - t0) * 1000.0, digits = 1)
    
    first_row = latents.df[1, :]
    S = Predictions.compute_score_matrix(exp.config.model, first_row)
    s_sum = round(sum(S.data), digits=4)
    
    println("✓ Model: $(rpad(exp.config.name, 35)) | $(nrow(latents.df)) matches extracted/loaded in $(elapsed)ms | Score Matrix Sum = $s_sum")
end

# 4. Fast Single-Metric Test (RQR Calibration)
println("\n--- Fast Scoring Rule Smoke Test (RQR Calibration) ---")
rqr_metric = Evaluation.RQR()
for exp in experiments
    latents = Experiments.extract_oos_predictions(ds, exp)
    rqr_res = Evaluation.compute_metric(rqr_metric, exp, ds, latents)
    println("✓ $(rpad(exp.config.name, 35)) RQR Mean = $(round(rqr_res.all.mean, digits=4)), Std = $(round(rqr_res.all.std, digits=4))")
end

println("\n", "="^85)
println("✓ ALL EXPERIMENT LOADERS, EXTRACTORS, AND SCORING PIPELINES VERIFIED!")
println("="^85)
