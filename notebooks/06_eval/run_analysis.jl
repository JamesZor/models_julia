# run_analysis.jl
using JLD2, DataFrames, Printf
# Assuming BayesianFootball and other dependencies are in your environment
using BayesianFootball 

# Include the new analysis logic
include("./analysis_logic.jl")

println("--- Starting Betting Strategy Analysis ---")

# 1. Load Experiment Data & Data Store 
println("Loading experiment data...")
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
result = r1[1]
experiment_config = r1[2]
mapping = result.mapping

println("Loading data store...")
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

# 2. Get Target Season Matches & Predictions
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)
println("Found $(nrow(target_matches)) matches in the target season.")

println("Predicting outcomes for target season...")
matches_predictions_map = predict_target_season(target_matches, result, mapping)

# 3. Perform Quantile Sensitivity Analysis
println("\n--- Running Quantile Sensitivity Analysis ---")
quantiles_to_test = 0.1:0.1:0.9
all_quantile_results = Dict{Float64, Dict{String, Float64}}()

for q in quantiles_to_test
    print("Analyzing for quantile $(@sprintf("%.2f", q))... ")
    analysis_result = run_analysis_for_quantile(
        target_matches, 
        matches_predictions_map, 
        data_store;
        quantile_level=q,
        fractional_kelly=1.0 # Set your desired fractional kelly (safety factor)
    )
    all_quantile_results[q] = analysis_result.rois
    println("Done.")
end

# 4. Generate ROI vs. Quantile Plot
plot_quantile_sensitivity(all_quantile_results)

# 5. Analyze and Plot Cumulative Wealth for an Optimal Quantile
# Let's choose the median (0.5) as our default strategy for the wealth curve
chosen_quantile = 0.5
println("\n--- Generating Cumulative Wealth Curve for Quantile $(chosen_quantile) ---")

final_analysis = run_analysis_for_quantile(
    target_matches, 
    matches_predictions_map, 
    data_store;
    quantile_level=chosen_quantile,
    fractional_kelly=1.0
)

# Print final ROI table for the chosen quantile
println("\nFinal ROI Breakdown for Quantile $(chosen_quantile):")
println("┌────────────────┬───────────┐")
println("│ Market Group   │    ROI %  │")
println("├────────────────┼───────────┤")
for (group, roi) in sort(collect(pairs(final_analysis.rois)))
    println("│ $(rpad(group, 14)) │ $(@sprintf("%9.2f", roi)) │")
end
println("└────────────────┴───────────┘")

plot_wealth_curve(final_analysis.wealth_curve, chosen_quantile)

println("\n--- Analysis Complete ---")
