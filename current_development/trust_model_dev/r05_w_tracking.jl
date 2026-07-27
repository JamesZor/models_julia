# current_development/trust_model_dev/r05_w_tracking.jl

# Run the full pipeline to get the `layered_results` and `team_map` in memory
include("r04_full_pipeline.jl")

using DataFrames
using StatsFuns: logistic
using MCMCChains
using Statistics
using StatsBase

println("\n=========================================================")
println("📈 TRACKING TRUST WEIGHTS ACROSS SPLITS")
println("=========================================================")

# Reverse team_map to easily get string names from the index
reverse_team_map = Dict(v => k for (k, v) in global_team_map)

# Get string names for the parent markets
market_names = [String(first(values(BayesianFootball.Data.Markets.outcomes(m)))) for m in TRUST_MARKETS]

w_tracking_rows = []

# Loop over the splits chronologically
sorted_splits = sort(collect(keys(layered_results.l2_results.chains)))

for s in sorted_splits
    chain = layered_results.l2_results.chains[s]
    chain_df = DataFrame(chain)
    
    sigma_col = chain_df[!, Symbol("σ_team")]
    
    for t in 1:length(global_team_map)
        team_name = reverse_team_map[t]
        team_z_col = chain_df[!, Symbol("team_z[$t]")]
        
        for m in 1:length(TRUST_MARKETS)
            market_name = market_names[m]
            w0_col = chain_df[!, Symbol("w0[$m]")]
            
            # Reconstruct the full MCMC posterior distribution for η
            eta_samples = w0_col .+ (sigma_col .* team_z_col)
            
            # Map the entire distribution back to [0, 1] probability space
            w_samples = logistic.(eta_samples)
            
            push!(w_tracking_rows, (
                Split = s,
                Team = team_name,
                Market = market_name,
                Mean = round(mean(w_samples), digits=4),
                Std = round(std(w_samples), digits=4),
                Skewness = round(skewness(w_samples), digits=4),
                Kurtosis = round(kurtosis(w_samples), digits=4)
            ))
        end
    end
end

w_tracking_df = DataFrame(w_tracking_rows)

println("✅ Successfully extracted $(nrow(w_tracking_df)) rows of Trust tracking data!")

# Pick the very first team dynamically to show the evolution
sample_team = reverse_team_map[1]
sample_market = market_names[3]

println("\n--- Trust Evolution for [$sample_team] on [$sample_market] ---")
sample_df = subset(w_tracking_df, :Team => ByRow(==(sample_team)), :Market => ByRow(==(sample_market)))
sort!(sample_df, :Split)
display(sample_df)

println("\n--- Cross-Section of Split $(sorted_splits[end]) (Latest Data) for $sample_market ---")
latest_split_df = subset(w_tracking_df, :Split => ByRow(==(sorted_splits[end])), :Market => ByRow(==(sample_market)))
sort!(latest_split_df, :Mean, rev=true)
display(latest_split_df)
