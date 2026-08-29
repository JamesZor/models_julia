# Import the necessary modules that will be used across all tests
using Test
using BayesianFootball
using DataFrames, Dates, InlineStrings # Add any other packages your tests need here

# Start the main test suite for the entire package
@testset "BayesianFootball.jl Tests" begin

    println("running data module tests...")
    include("data_tests.jl")

    println("Running grouped splitting tests...")
    include("splitting_tests.jl")

    println("Running Features Module tests...")
    include("features_tests.jl")

    println("Running Pre Game Module tests...")
    include("pregame_tests.jl")

    println("Running composable count-model builder tests...")
    include("builder_tests.jl")

    println("Running Portfolio Module tests...")
    include("portfolio_tests.jl")

    println("Running MatchDay Module tests...")
    include("matchday_tests.jl")

    println("Running Caching and OOS Predictions tests...")
    include("caching_tests.jl")

    println("Running Recombination and Squad Wealth Engine tests...")
    include("recombination_tests.jl")

    println("Running typed posterior latent tests...")
    include("latents_tests.jl")

    println("Running unified inference and fit lifecycle tests...")
    include("inference_tests.jl")

    println("Running unified evaluation framework tests...")
    include("evaluation_tests.jl")

    println("Running unified portfolio framework tests...")
    include("unified_portfolio_tests.jl")
end

