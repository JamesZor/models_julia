# current_development/scottish_lower/distance/r07_smoke_extract_all_models.jl
#
# SMOKE TEST: Fast Verification of Parameter Extraction & Score Matrix Computation
#             across all 8 Scottish Lower Benchmark Models.
#
# Tests:
# 1. Experiments.extract_parameters for each model type
# 2. Pred.extract_params & Pred.compute_score_matrix
# 3. Quick 1-split test match prediction

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Pred        = BayesianFootball.Predictions
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l01_proxy_xg_feature.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l02_pxg_engines.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/wealth/l01_wealth_data.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/neg_bin/l01_negbin_engines.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/neg_bin/l02_negbin_wealth_engines.jl"))
include("l01_distance_features.jl")
include("l02_negbin_distance_engines.jl")
include("l03_negbin_wealth_distance_engines.jl")

println("\n", "="^95)
println("🧪 FAST SMOKE TEST: PARAMETER EXTRACTION FOR ALL 8 BENCHMARK MODELS")
println("="^95)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)

wealth_folders   = Experiments.list_experiments("scottish_negbin_wealth_grid"; data_dir = joinpath(ROOT, "data"))
negbin_folders   = Experiments.list_experiments("scottish_negbin_grid"; data_dir = joinpath(ROOT, "data"))
dist_folders     = Experiments.list_experiments("scottish_distance_negbin_grid"; data_dir = joinpath(ROOT, "data"))
champ_folders    = Experiments.list_experiments("scottish_full_champion_grid"; data_dir = joinpath(ROOT, "data"))
all_folders      = vcat(wealth_folders, negbin_folders, dist_folders, champ_folders)
all_loaded       = Experiments.load_experiments(all_folders)

target_models = [
    "goals_negbin_ctl_hl365_hs2",
    "goals_negbin_dist_hl365_hs2",
    "goals_negbin_wealth_hl365_hs2",
    "goals_negbin_wealth_dist_hl365_hs2",
    "pxg_apm_negbin_hl365_hs2",
    "pxg_apm_negbin_dist_hl365_hs2",
    "pxg_apm_negbin_wealth_hl365_hs2",
    "pxg_apm_negbin_wealth_dist_hl365_hs2"
]

experiments_dict = Dict{String, Any}()
for exp in all_loaded
    for t in target_models
        if startswith(exp.config.name, t)
            experiments_dict[t] = exp
        end
    end
end

println("✓ Found $(length(experiments_dict))/$(length(target_models)) target models on disk")

for t in target_models
    if !haskey(experiments_dict, t)
        println("❌ MISSING: $t")
        continue
    end
    exp = experiments_dict[t]
    model = exp.config.model
    mname = exp.config.name
    
    print("Testing extraction for [$mname] ($(typeof(model).name.name))... ")
    try
        # Test split 1
        result_tuple = exp.training_results.items[1]
        feature_tuple = exp.features.items[1]
        chain = result_tuple[1]
        split_meta = result_tuple[2]
        f_set = feature_tuple[1]
        
        # Test extract_parameters
        test_df = filter(r -> r.match_id in split_meta.test_match_ids, ds.matches)
        
        params_dict = Experiments.extract_parameters(model, test_df, f_set, chain)
        
        # Test score matrix for 1 match
        first_mid = first(keys(params_dict))
        p = params_dict[first_mid]
        s_mat = Pred.compute_score_matrix(model, p; max_goals = 10)
        
        println("✅ OK (extracted $(length(params_dict)) matches, score matrix sum = $(round(sum(s_mat.matrix[:,:,1]), digits=4)))")
    catch e
        println("❌ FAILED!")
        println("  Error: $e")
        Base.show_backtrace(stdout, catch_backtrace())
        println()
    end
end

println("\n", "="^95)
println("✓ EXTRACTION SMOKE TEST COMPLETE!")
println("="^95)
