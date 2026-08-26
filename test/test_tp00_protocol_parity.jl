using Test
using BayesianFootball
using ThreadPinning
using DataFrames
using Distributions
using Statistics

pinthreads(:cores)

include(normpath(joinpath(@__DIR__, "../current_development/scottish_lower/_protocol/ScottishLowerProtocol.jl")))
using .ScottishLowerProtocol

include(normpath(joinpath(@__DIR__, "../current_development/scottish_lower/00_team_poisson/l01_model.jl")))
include(normpath(joinpath(@__DIR__, "../current_development/scottish_lower/00_team_poisson/l02_equations.jl")))
include(normpath(joinpath(@__DIR__, "../current_development/scottish_lower/00_team_poisson/l03_adapter.jl")))

@testset "Model 00 (Pure Poisson) Protocol Gates 0 to 7" begin
    # Gate 0: Contract
    contract = sl_contract()
    ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
    folds = sl_build_folds(ds, contract)
    gate0 = sl_gate_contract(ds, folds, contract)
    @test sl_gate_table("0. Contract", gate0)

    # Gate 1: Config
    adapter = TP00Adapter(half_life_days = 180.0)
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

    # Load Saved 20-Fold Grid Artifact
    grid_path = "data/scottish_lower/00_team_poisson/0f7eab0e/tp00_grid_0f7eab0e_20260826_144143"
    grid_loaded = sl_load_experiment(grid_path)

    # Gate 3d: Grid Convergence
    gate10 = sl_gate_convergence(grid_loaded, adapter, contract; expected_folds = length(folds))
    @test sl_gate_table("3d. Grid convergence", gate10)

    # Gate 4a & 4c: Synthetic Extraction & Fallbacks
    gate4a = sl_gate_extraction_synthetic(adapter, features[1])
    @test sl_gate_table("4a. Synthetic extraction", gate4a)
    gate4c = sl_gate_extraction_fallbacks(adapter, features[1])
    @test sl_gate_table("4c. Extraction fallbacks", gate4c)

    # Gate 4b: Real Extraction
    gate4b, latents = sl_gate_extraction_real(ds, grid_loaded, adapter, contract)
    @test sl_gate_table("4b. Extraction plumbing", gate4b)

    # Gate 5: Score Matrix & Market Identities
    gate5a = sl_gate_score_dispatch(adapter, first(eachrow(latents.df)); max_goals = contract.max_goals)
    @test sl_gate_table("5a. Score-matrix dispatch", gate5a)
    gate5b = sl_gate_score_grid(adapter, latents.df, contract)
    @test sl_gate_table("5b. Score-matrix grid", gate5b)
    gate5c = sl_gate_market_identities(adapter, latents.df, contract)
    @test sl_gate_table("5c. Market identities", gate5c)

    # Gate 6: Evaluation
    oos_ids = Set(Int.(latents.df.match_id))
    book_b365 = sl_market_book(ds.odds, contract; ids = oos_ids)
    book_bf, _ = sl_drop_incomplete(sl_betfair_book(ds, contract, book_b365; ids = oos_ids))
    @test sl_gate_table("6a. Bet365 book integrity", sl_gate_book_integrity(book_b365, contract))
    @test sl_gate_table("6a. Betfair book integrity", sl_gate_book_integrity(book_bf, contract))

    model_book, fixtures = sl_model_book(adapter, latents, ds, contract)
    joined = sl_join_books(model_book, Dict("bet365" => book_b365, "betfair" => book_bf))
    gate6b = sl_gate_alignment(joined, model_book)
    gate6c = sl_gate_shape(fixtures)
    scores_b365 = sl_score_table(joined["bet365"])
    gate6d = sl_gate_not_broken(scores_b365)
    @test sl_gate_table("6b. Alignment", gate6b)
    @test sl_gate_table("6c. Shape", gate6c)
    @test sl_gate_table("6d. Not broken", gate6d)

    # Gate 7: Portfolio-Kelly Growth & CLV
    bf_odds = sl_betfair_odds_df(ds, contract; ids = oos_ids)
    book_spec = sl_book_spec(contract)
    books_bf = BayesianFootball.Portfolio.build_books(book_spec, latents.df, grid_loaded, bf_odds, ds)
    gate7a = sl_gate_books(books_bf, latents.df, bf_odds)
    @test sl_gate_table("7a. Book construction", gate7a)

    policy = sl_growth_policies(contract)[1].policy
    slates = BayesianFootball.Portfolio.group(policy.grouping, books_bf)
    trajectory = BayesianFootball.Portfolio.simulate(policy, slates)
    gate7b = sl_gate_simulation(trajectory, slates, contract)
    @test sl_gate_table("7b. Simulation integrity", gate7b)

    growth = sl_growth_table(books_bf, contract)
    gate7c = sl_gate_growth(growth)
    @test sl_gate_table("7c. Growth verdict", gate7c)
    println("\nFinal Growth Summary Table:")
    display(growth)
end
