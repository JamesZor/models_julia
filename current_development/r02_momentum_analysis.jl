# current_development/r02_momentum_analysis.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball

# Include the loader/logic script
include(joinpath(@__DIR__, "l02_momentum_analysis.jl"))

println("--- Starting Momentum Statistical Validation Runner ---")

# Define path to the output report at the project root
report_path = joinpath(dirname(@__DIR__), "momentum_statistical_analysis.md")

try
    # Run the validation pipeline
    run_full_validation_pipeline(report_path)
    println("--- Momentum Statistical Validation Completed Successfully ---")
catch e
    @error "An error occurred during momentum analysis execution" exception=(e, catch_backtrace())
    exit(1)
end
