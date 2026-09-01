# ==============================================================================
# r60 — Automated one-scored-fold smoke test for all joint + player candidates
# Run on archpc: julia --project -t 8 <this file>
# ==============================================================================
#
# CONTRACT
#   Seven gates per candidate, all asserted through `@testset`:
#     1. the compiled ReverseDiff gradient tape builds and replays under 0.1 ms;
#     2. NUTS sampling completes with no crashed chain or fold;
#     3. the six-part convergence audit passes;
#     4. chain parameter extraction and held-out `CountLatents` succeed;
#     5. a `SmileScoreGrid` can be generated and priced;
#     6. `save_fit`/`load_fit` round-trips through PostgreSQL exactly;
#     7. the portfolio simulation runs and persists.
#
#   This runs on the laptop only. It never launches work on `mcmc-beast`.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and shared experiment state
# ==============================================================================
using BayesianFootball
using DataFrames
using Dates
using DynamicPPL
using LinearAlgebra
using LogDensityProblems
using Printf
using Random
using ReverseDiff
using Test
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() == 8 || error("r60 must run with `julia --project -t 8`; got $(Threads.nthreads()) threads")

include(joinpath(@__DIR__, "l60_loader.jl"))

const R60_PG = BayesianFootball.Models.PreGame
const R60_FEATURES = BayesianFootball.Features

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
const R60_THRESHOLDS = ConvergenceThresholds(
    max_rhat = 1.05,
    min_ess = 100.0,
    # The audit comparator is strict `<`; eps() expresses an exact-zero count gate.
    max_divergence_rate = eps(Float64),
    min_bfmi = 0.30,
    max_treedepth_rate = 0.05,
)

# Gate 1's budget. The compiled tape is replayed many times per NUTS leapfrog step,
# so this is the number that decides whether a 40-fold production grid is affordable.
const R60_GRADIENT_BUDGET_MS = 0.10
const R60_GRADIENT_REPLAYS = 200

# CVConfig calls the historical-window field `history_seasons` in the installed API.
# Selecting the first scored boundary below makes this exactly one sampled/OOS fold.
r60_splitter = Data.CVConfig(
    tournament_ids = [56],
    target_seasons = ["24/25"],
    history_seasons = 1,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    end_dynamics = 1,
    stop_early = true,
)

r60_sampler = NUTSConfig(
    n_samples = 200,
    n_warmup = 200,
    n_chains = 4,
    accept_rate = 0.65,
    show_progress = false,
)

# %%
# ==============================================================================
# 3. Fold construction and gate implementations
# ==============================================================================
function r60_one_fold_inputs(model)
    boundaries = Data.create_id_boundaries(ds, r60_splitter)
    idx = findfirst(pair -> !isempty(first(pair).target_match_ids), boundaries)
    idx === nothing && error("smoke splitter produced no scored boundary")
    selected = [boundaries[idx]]
    feature_sets = R60_FEATURES.create_features(selected, ds, model, r60_splitter)
    oos = [Data.get_next_matches(ds, feature_sets[1], r60_splitter)]
    nrow(oos[1]) > 0 || error("selected smoke boundary has no out-of-sample fixtures")
    return feature_sets, oos
end

"Gate 1: the model's log density must compile to a replayable ReverseDiff tape."
function r60_assert_gradient_tape(model, feature_set; seed::Int = 20260901)
    turing_model = R60_PG.build_turing_model(model, feature_set)
    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    f = x -> LogDensityProblems.logdensity(density, x)

    @test isfinite(f(θ))
    raw = ReverseDiff.GradientTape(f, θ)
    tape = ReverseDiff.compile(raw)
    gradient = similar(θ)
    ReverseDiff.gradient!(gradient, tape, θ)
    @test all(isfinite, gradient)

    # A vectorised typed-tuple walk has a tape length set by the model's structure,
    # not by the observation count; a tape that grows with n is the failure this catches.
    @test length(raw.tape) < 500

    # Warm the replay path, then time it. `minimum` is the honest estimator here:
    # it is the run least polluted by scheduler noise, not an average of noise.
    for _ in 1:20
        ReverseDiff.gradient!(gradient, tape, θ)
    end
    best_ns = Inf
    for _ in 1:R60_GRADIENT_REPLAYS
        started = time_ns()
        ReverseDiff.gradient!(gradient, tape, θ)
        best_ns = min(best_ns, Float64(time_ns() - started))
    end
    latency_ms = best_ns / 1.0e6
    @test latency_ms < R60_GRADIENT_BUDGET_MS
    return (; latency_ms, n_instructions = length(raw.tape), n_params = length(θ))
end

function r60_assert_six_gate_audit(diagnostics)
    @test diagnostics.passed
    @test diagnostics.max_rhat <= 1.05
    @test diagnostics.min_ess_bulk >= 100.0
    @test diagnostics.min_ess_tail >= 100.0
    @test diagnostics.n_divergent == 0
    @test diagnostics.min_bfmi >= 0.30
    @test diagnostics.treedepth_rate < 0.05
    return nothing
end

function r60_assert_parameter_extraction(model, fit, feature_sets, oos)
    chain = fit.folds[1].chain
    raw = R60_PG.extract_parameters(model, oos[1], first(feature_sets[1]), chain)
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
function r60_neutral_smile_grid(latents::CountLatents)
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

function r60_assert_round_trip(fit, run_id)
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

# %%
# ==============================================================================
# 4. Execution
# ==============================================================================
println("\n" * "="^100)
println(" EXPERIMENT 06 · TWO-ARM JOINT + PLAYER LINEUP FUSION · ONE-FOLD SMOKE")
println("="^100)
@printf("  matches %d | lineups %d | odds %d | threads %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.odds), Threads.nthreads())

session = first(string(uuid4()), 8)
summary = NamedTuple[]

@testset "Experiment 06 joint player model integration smoke" begin
    for (name, model) in l60_candidate_models
        @testset "$name" begin
            println("\n--- $name ---")
            feature_sets, oos = r60_one_fold_inputs(model)
            tape = r60_assert_gradient_tape(model, first(feature_sets[1]))
            @printf("  tape: %d instructions | %d parameters | %.4f ms compiled gradient\n",
                    tape.n_instructions, tape.n_params, tape.latency_ms)

            smoke_name = name * "_smoke_" * session
            config = FitConfig(
                name = smoke_name,
                model = model,
                splitter = r60_splitter,
                sampler = r60_sampler,
                execution = ThreadedExecution(),
                tags = [L60_TAGS; "smoke"],
                description = "One-scored-fold archpc smoke for $name.",
                save_dir = joinpath("/tmp", "scottish_lower_joint_player_smoke", smoke_name),
            )

            started = time()
            fit = fit_model(
                config;
                feature_sets = feature_sets,
                oos_fixtures = oos,
                thresholds = R60_THRESHOLDS,
                quiet = false,
            )
            elapsed = time() - started

            @test length(fit.folds) == 1
            r60_assert_six_gate_audit(fit.diagnostics)
            r60_assert_parameter_extraction(model, fit, feature_sets, oos)
            r60_neutral_smile_grid(fit.latents)

            run_id = save_fit(fit, db)
            r60_assert_round_trip(fit, run_id)

            result, books, report = run_portfolio_simulation(
                l60_book, l60_policy, fit, ds.odds, ds;
                bootstrap = false,
                require_converged = true,
                quiet = true,
            )
            @test result isa PortfolioResult
            @test report.n_books == length(books)
            portfolio_id = save_portfolio_db(
                result, run_id, db;
                book_spec = l60_book,
                policy_spec = l60_policy,
                metadata = (; smoke = true, candidate = name),
            )
            loaded_result = load_portfolio_db(portfolio_id, db)
            @test loaded_result.summary.total_return_pct == result.summary.total_return_pct
            @test isequal(loaded_result.trajectory.bets, result.trajectory.bets)

            push!(summary, (
                name = name,
                elapsed = elapsed,
                grad_ms = tape.latency_ms,
                tape_len = tape.n_instructions,
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

# %%
# ==============================================================================
# 5. Report
# ==============================================================================
println("\n" * "="^150)
@printf(" %-40s | %7s | %8s | %5s | %7s | %7s | %4s | %7s | %5s\n",
        "Model", "Seconds", "Grad ms", "Tape", "R-hat", "ESS", "Div", "Latents", "Books")
println("-"^150)
for row in summary
    @printf(" %-40s | %7.1f | %8.4f | %5d | %7.4f | %7.0f | %4d | %7d | %5d\n",
            row.name, row.elapsed, row.grad_ms, row.tape_len, row.rhat, row.ess,
            row.divergences, row.n_latents, row.n_books)
end
println("="^150)
println("All six candidates passed the gradient, sampling, convergence, extraction, " *
        "score-grid, database, and portfolio gates.")
