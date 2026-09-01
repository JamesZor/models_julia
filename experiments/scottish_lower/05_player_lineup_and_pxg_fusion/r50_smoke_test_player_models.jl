# ==============================================================================
# r50 — Automated one-scored-fold smoke test for all player-lineup candidates
# Run on archpc: julia --project -t 8 <this file>
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Test
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() in (8, 16) || error(
    "r50 must run with `julia --project -t 8` locally or `-t 16` on mcmc-beast; " *
    "got $(Threads.nthreads()) threads")

include(joinpath(@__DIR__, "l50_loader.jl"))

const R50_PG = BayesianFootball.Models.PreGame
const R50_FEATURES = BayesianFootball.Features
const R50_THRESHOLDS = ConvergenceThresholds(
    max_rhat = 1.05,
    min_ess = 100.0,
    # The audit comparator is strict `<`; eps() expresses an exact-zero count gate.
    max_divergence_rate = eps(Float64),
    min_bfmi = 0.30,
    max_treedepth_rate = 0.05,
)

# CVConfig calls the historical-window field `history_seasons` in the installed API.
# Selecting the first scored boundary below makes this exactly one sampled/OOS fold.
r50_splitter = Data.CVConfig(
    tournament_ids = [56],
    target_seasons = ["24/25"],
    history_seasons = 1,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    end_dynamics = 1,
    stop_early = true,
)

r50_sampler = NUTSConfig(
    n_samples = 200,
    n_warmup = 200,
    n_chains = 4,
    accept_rate = 0.65,
    show_progress = false,
)

function r50_one_fold_inputs(model)
    boundaries = Data.create_id_boundaries(ds, r50_splitter)
    idx = findfirst(pair -> !isempty(first(pair).target_match_ids), boundaries)
    idx === nothing && error("smoke splitter produced no scored boundary")
    selected = [boundaries[idx]]
    feature_sets = R50_FEATURES.create_features(selected, ds, model, r50_splitter)
    oos = [Data.get_next_matches(ds, feature_sets[1], r50_splitter)]
    nrow(oos[1]) > 0 || error("selected smoke boundary has no out-of-sample fixtures")
    return feature_sets, oos
end

function r50_expected_params(name::String, n_teams::Int)
    name == L50_MODEL_NAMES[1] && return 2 * n_teams + 7
    name in L50_MODEL_NAMES[2:4] && return 2 * n_teams + 8
    name == L50_MODEL_NAMES[5] && return 2 * n_teams + 9
    error("no structural parameter contract for $name")
end

function r50_assert_structural_contract(name, model, feature_set, chain, oos)
    n_teams = Int(feature_set.data[:n_teams])
    n_params = cb_parameter_count(model, n_teams)
    @test n_params == r50_expected_params(name, n_teams)

    grouped_sites = Set(cb_varinfo_sites(model))
    @test Set(Symbol.(("dyn.raw_a", "dyn.raw_d", "dyn.σ_a", "dyn.σ_d"))) ⊆ grouped_sites
    if name != L50_MODEL_NAMES[1]
        @test Set(Symbol.(("lineup.w_att", "lineup.w_def"))) ⊆ grouped_sites
    end
    if name in (L50_MODEL_NAMES[1], L50_MODEL_NAMES[5])
        @test Symbol("production_wealth.w") in grouped_sites
    end

    chain_names = String.(names(chain))
    @test "dyn.σ_a" in chain_names
    @test "dyn.σ_d" in chain_names
    @test any(value -> startswith(value, "dyn.raw_a["), chain_names)
    @test any(value -> startswith(value, "dyn.raw_d["), chain_names)
    name == L50_MODEL_NAMES[1] || begin
        @test "lineup.w_att" in chain_names
        @test "lineup.w_def" in chain_names
    end
    if name in (L50_MODEL_NAMES[1], L50_MODEL_NAMES[5])
        @test "production_wealth.w" in chain_names
    end

    # Two home teams, one fixed opponent, and deliberately absent lineup IDs.
    # Any rate difference must come from team α/β, not lineup inputs.
    teams = collect(keys(feature_set.data[:team_map]))
    length(teams) >= 3 || error("team identity contract needs at least three fitted teams")
    fixtures = DataFrame(
        match_id=[-9_000_001, -9_000_002],
        home_team=[teams[1], teams[2]],
        away_team=[teams[3], teams[3]],
        match_date=fill(oos.match_date[1], 2),
        season_idx=fill(
            hasproperty(oos, :season_idx) ? Int(oos.season_idx[1]) :
            Int(feature_set.data[:n_seasons]), 2),
    )
    rates = R50_PG.extract_parameters(model, fixtures, feature_set, chain)
    @test rates[-9_000_001].λ_h != rates[-9_000_002].λ_h
    return nothing
end

function r50_assert_six_gate_audit(diagnostics)
    @test diagnostics.passed
    @test diagnostics.max_rhat <= 1.05
    @test diagnostics.min_ess_bulk >= 100.0
    @test diagnostics.min_ess_tail >= 100.0
    @test diagnostics.n_divergent == 0
    @test diagnostics.min_bfmi >= 0.30
    @test diagnostics.treedepth_rate < 0.05
    return nothing
end

function r50_assert_parameter_extraction(model, fit, feature_sets, oos)
    chain = fit.folds[1].chain
    raw = R50_PG.extract_parameters(model, oos[1], first(feature_sets[1]), chain)
    @test !isempty(raw)
    @test Set(keys(raw)) == Set(Int.(oos[1].match_id))
    for value in values(raw)
        @test haskey(value, :λ_h)
        @test haskey(value, :λ_a)
        @test all(isfinite, value.λ_h)
        @test all(isfinite, value.λ_a)
        @test all(>(0.0), value.λ_h)
        @test all(>(0.0), value.λ_a)
    end
    @test fit.latents isa CountLatents
    @test n_matches(fit.latents) == nrow(oos[1])
    return nothing
end

# These candidates do not estimate a smile. A neutral phi(K)=1 compatibility wrapper
# exercises the SmileScoreGrid generation and pricing plumbing without changing their rates.
function r50_neutral_smile_grid(latents::CountLatents)
    strikes = [0.5, 1.5, 2.5]
    λ_tot = latents.λ_home .+ latents.λ_away
    φ = ones(Float64, n_matches(latents), length(strikes), n_draws(latents))
    smile_latents = SmileLatents(
        latents.match_ids,
        latents.λ_home,
        latents.λ_away,
        latents.observation_params,
        λ_tot,
        φ,
        strikes,
    )
    grid = compute_score_grid(smile_latents, 1)
    @test grid isa SmileScoreGrid
    @test all(isfinite, grid.grid)
    @test all(isfinite, grid.λ_tot)
    @test all(isfinite, grid.φ)
    return grid
end

function r50_assert_round_trip(fit, run_id)
    loaded = load_fit(db, run_id)
    @test loaded isa Fit
    @test length(loaded.folds) == length(fit.folds)
    for i in eachindex(fit.folds)
        @test names(loaded.folds[i].chain) == names(fit.folds[i].chain)
        @test Array(loaded.folds[i].chain) == Array(fit.folds[i].chain)
    end
    @test loaded.latents isa CountLatents
    @test loaded.latents.match_ids == fit.latents.match_ids
    left = latent_matrices(loaded.latents)
    right = latent_matrices(fit.latents)
    @test propertynames(left) == propertynames(right)
    for key in propertynames(right)
        @test getproperty(left, key) == getproperty(right, key)
    end
    return loaded
end

println("\n" * "="^100)
println(" EXPERIMENT 05 · PLAYER LINEUP AND pxG FUSION · ONE-FOLD SMOKE")
println("="^100)
@printf("  matches %d | lineups %d | odds %d | threads %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.odds), Threads.nthreads())

session = first(string(uuid4()), 8)
summary = NamedTuple[]

@testset "Experiment 05 player model integration smoke" begin
    for (name, model) in l50_candidate_models
        @testset "$name" begin
            println("\n--- $name ---")
            feature_sets, oos = r50_one_fold_inputs(model)
            smoke_name = name * "_smoke_" * session
            config = FitConfig(
                name = smoke_name,
                model = model,
                splitter = r50_splitter,
                sampler = r50_sampler,
                execution = ThreadedExecution(),
                tags = [L50_TAGS; "smoke"],
                description = "One-scored-fold archpc smoke for $name.",
                save_dir = joinpath("/tmp", "scottish_lower_player_smoke", smoke_name),
            )

            started = time()
            fit = fit_model(
                config;
                feature_sets = feature_sets,
                oos_fixtures = oos,
                thresholds = R50_THRESHOLDS,
                quiet = false,
            )
            elapsed = time() - started

            @test length(fit.folds) == 1
            r50_assert_six_gate_audit(fit.diagnostics)
            r50_assert_structural_contract(
                name, model, first(feature_sets[1]), fit.folds[1].chain, oos[1])
            r50_assert_parameter_extraction(model, fit, feature_sets, oos)
            r50_neutral_smile_grid(fit.latents)

            run_id = save_fit(fit, db)
            r50_assert_round_trip(fit, run_id)

            result, books, report = run_portfolio_simulation(
                l50_book, l50_policy, fit, ds.odds, ds;
                bootstrap = false,
                require_converged = true,
                quiet = true,
            )
            @test result isa PortfolioResult
            @test report.n_books == length(books)
            portfolio_id = save_portfolio_db(
                result, run_id, db;
                book_spec = l50_book,
                policy_spec = l50_policy,
                metadata = (; smoke = true, candidate = name),
            )
            loaded_result = load_portfolio_db(portfolio_id, db)
            @test loaded_result.summary.total_return_pct == result.summary.total_return_pct
            @test isequal(loaded_result.trajectory.bets, result.trajectory.bets)

            push!(summary, (
                name = name,
                elapsed = elapsed,
                rhat = fit.diagnostics.max_rhat,
                ess = min(fit.diagnostics.min_ess_bulk, fit.diagnostics.min_ess_tail),
                divergences = fit.diagnostics.n_divergent,
                n_latents = n_matches(fit.latents),
                n_books = length(books),
                run_id = run_id,
                portfolio_id = portfolio_id,
            ))
        end
    end
end

println("\n" * "="^128)
@printf(" %-45s | %7s | %7s | %7s | %4s | %7s | %7s\n",
        "Model", "Seconds", "R-hat", "ESS", "Div", "Latents", "Books")
println("-"^128)
for row in summary
    @printf(" %-45s | %7.1f | %7.4f | %7.0f | %4d | %7d | %7d\n",
            row.name, row.elapsed, row.rhat, row.ess, row.divergences,
            row.n_latents, row.n_books)
end
println("="^128)
println("All five smoke models passed sampling, convergence, extraction, score-grid, database, and portfolio gates.")
