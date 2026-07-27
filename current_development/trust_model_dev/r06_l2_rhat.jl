# current_development/trust_model_dev/r06_l2_rhat.jl

include("r04_full_pipeline.jl")

using DataFrames
using MCMCChains
using BayesianFootball.Experiments.Diagnostics: ChainDiagnostic

println("\n=========================================================")
println("🔍 CHECKING R-HAT CONVERGENCE ACROSS L2 SPLITS")
println("=========================================================")

rhat_rows = []

# Extract the R-hat summary from every chain in every split
for (s, chain) in layered_results.l2_results.chains
    # summarystats(chain) returns the parameter summary programmatically
    summary_df = DataFrame(summarystats(chain))
    
    for row in eachrow(summary_df)
        push!(rhat_rows, (
            fold = s,
            raw_symbol = row.parameters,
            rhat = row.rhat
        ))
    end
end

rhat_df = DataFrame(rhat_rows)

# Inject this directly into the existing ChainDiagnostic display logic!
diag = ChainDiagnostic(rhat_df)
display(diag)
