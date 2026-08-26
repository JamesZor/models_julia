using Test
using BayesianFootball
using ThreadPinning
using DataFrames
using Distributions
using Statistics

pinthreads(:cores)

include(normpath(joinpath(@__DIR__, "../current_development/scottish_lower/_protocol/ScottishLowerProtocol.jl")))
using .ScottishLowerProtocol

include(normpath(joinpath(@__DIR__, "../current_development/scottish_lower/01_team_poisson/l01_model.jl")))
include(normpath(joinpath(@__DIR__, "../current_development/scottish_lower/01_team_poisson/l02_equations.jl")))
include(normpath(joinpath(@__DIR__, "../current_development/scottish_lower/01_team_poisson/l03_adapter.jl")))

@testset "Model 01 (Negative Binomial / Dispersion) Protocol Gates 0 to 5" begin
    # Gate 0: Contract
    contract = sl_contract()
    ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
    folds = sl_build_folds(ds, contract)
    gate0 = sl_gate_contract(ds, folds, contract)
    @test sl_gate_table("0. Contract", gate0)

    # Gate 1: Config
    adapter = TP01Adapter(half_life_days = 180.0)
    gate1 = sl_gate_config(adapter, contract)
    @test sl_gate_table("1. Config", gate1)

    # Gate 2: Features
    gate2, features = sl_gate_features(ds, folds, adapter, contract)
    @test sl_gate_table("2. Features", gate2)

    # Gate 3a: Equation Parity
    gate3a = sl_gate_equation_parity(adapter, features[1])
    @test sl_gate_table("3a. Equation parity", gate3a)

    # Gate 3b: Gradients
    gate3b, grad = sl_gate_gradients(adapter, features[1])
    @test sl_gate_table("3b. Gradient health", gate3b)

    # Gate 4a & 4c: Synthetic Extraction & Fallbacks
    gate4a = sl_gate_extraction_synthetic(adapter, features[1])
    @test sl_gate_table("4a. Synthetic extraction", gate4a)
    gate4c = sl_gate_extraction_fallbacks(adapter, features[1])
    @test sl_gate_table("4c. Extraction fallbacks", gate4c)

    # Gate 5a: Score Matrix Dispatch
    synthetic_row = (λ_h = [1.4], λ_a = [1.1], true_xg_h = [1.4], true_xg_a = [1.1], r_h = [7.389], r_a = [7.389])
    gate5a = sl_gate_score_dispatch(adapter, synthetic_row; max_goals = contract.max_goals)
    @test sl_gate_table("5a. Score-matrix dispatch", gate5a)
end
