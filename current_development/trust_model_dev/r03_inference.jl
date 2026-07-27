# current_development/trust_model_dev/r03_inference.jl

# 1. Run the Training Script to get the posterior `chain`
include("r02_training.jl")

# 2. Include the Compositor Logic
include("l02_inference.jl")

println("\n=========================================================")
println("🚀 RUNNING L2 INFERENCE COMPOSITOR")
println("=========================================================")

# 3. Build the final L2 Posterior Predictive Distribution
# We pass in `l1_ppd_df` (the raw L1 mock), `ds` (for the raw market odds), 
# the `chain` (from Turing), and the `model_config` and `team_map`.
l2_ppd_df = compose_l2_ppd(l1_ppd_df, ds, chain, model_config, team_map)

println("✅ Compositor finished successfully!")
println("L2 PPD Shape: $(nrow(l2_ppd_df)) Rows")

println("\n=========================================================")
println("📊 FINAL L2 PREDICTION CHECK")
println("=========================================================")
using Statistics

# PROOF: The output is a full MCMC distribution, not just a scalar!
mc_length = length(l2_ppd_df.distribution[1])
println("Note: The L2 output is a FULL Monte Carlo array of $mc_length samples per match.")
println("Sample slice for row 1: ", round.(l2_ppd_df.distribution[1][1:5], digits=4))
println("Below are just the MEANS of those distributions for readability:\n")

# Compare the means of the raw L1 predictions against the new L2 blended predictions!
for i in 1:min(15, nrow(l2_ppd_df))
    row = l2_ppd_df[i, :]
    raw_l1_row = l1_ppd_df[(l1_ppd_df.match_id .== row.match_id) .& (l1_ppd_df.selection .== row.selection), :][1, :]
    
    l1_mean = mean(raw_l1_row.distribution)
    l2_mean = mean(row.distribution)
    q_market = row.q_market
    
    sel = rpad(String(row.selection), 10)
    
    println("Match $(row.match_id) | $sel | L1: $(round(l1_mean, digits=3)) | Market: $(round(q_market, digits=3)) | L2: $(round(l2_mean, digits=3))")
end
