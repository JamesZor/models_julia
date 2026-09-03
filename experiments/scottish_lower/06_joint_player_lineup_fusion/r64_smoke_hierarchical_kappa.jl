# ==============================================================================
# r64 — Extended two-fold smoke test for the hierarchical team-kappa candidates
# Run on mcmc-beast: julia --project -t 16 <this file>
# ==============================================================================
#
# WHAT THIS IS FOR. `r60` proved the shared-κ candidates on ONE fold at reduced
# sampler settings, because all it had to establish was that the plumbing held.
# The hierarchical component adds `n_teams + 1` parameters to a posterior that
# already funnels, and the two failure modes it introduces — a σ_κ that sticks to
# its lower bound with terrible tail ESS, and a per-team lookup that unrolls the
# ReverseDiff tape over the fold — are both invisible at 200 draws. So this runner
# samples at FULL PRODUCTION SETTINGS (4 chains, 800 warmup, 800 retained) over two
# representative folds, which is the only configuration whose convergence numbers
# say anything about the 40-fold grid.
#
# CONTRACT — eight gates per candidate, all asserted through `@testset`:
#   G1  compiled ReverseDiff tape builds, replays under 0.15 ms, and its LENGTH is
#       independent of the fold size (the per-team `getindex` stayed vectorised);
#   G2  the structural parameter contract holds — 3·n_teams + k, and the chain
#       actually carries `obs.σ_κ` and `obs.κ_team_raw[1:n_teams]`;
#   G3  NUTS completes on every fold with no crashed chain;
#   G4  the six-part convergence audit passes, at the STRICTER production reading:
#       max R̂ < 1.05, min bulk ESS > 400, min tail ESS > 300, divergences == 0;
#   G5  κ extraction is identified — δ_κ sums to exactly zero in every draw — and
#       κ_global, σ_κ and the per-team deltas are reported with 90% HPDIs;
#   G6  held-out `CountLatents` extract, finite and positive;
#   G7  a `SmileScoreGrid` builds and prices off those latents;
#   G8  `save_fit`/`load_fit` round-trips through PostgreSQL bit-identically, and
#       the portfolio simulation persists and reloads with an identical ledger.
#
# A NINTH GATE, on `m05` only: the engine's hand-inlined hierarchical log-density
# is compared against `builder/equations.jl`, which is written from `logpdf(Gamma)`
# and `logpdf(Poisson)` and never calls the engine. `m12` carries a lineup pillar,
# which the reference does not cover, so it is checked without it — an honest gap,
# not a skipped test: the observation arm is identical in both models.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and shared experiment state
# ==============================================================================
using BayesianFootball
using DataFrames
using Dates
using Distributions
using DynamicPPL
using LinearAlgebra
using LogDensityProblems
using Printf
using Random
using ReverseDiff
using Statistics
using Test
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads() >= 8 || error(
    "r64 samples 2 folds x 4 chains at production settings; run with " *
    "`julia --project -t 16` on mcmc-beast. Got $(Threads.nthreads()) threads")

include(joinpath(@__DIR__, "l64_hierarchical_kappa_loader.jl"))

const R64_PG = BayesianFootball.Models.PreGame
const R64_API = R64_PG.Builder
const R64_FEATURES = BayesianFootball.Features

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
#
# `ConvergenceThresholds` carries ONE ESS floor, applied to bulk and tail alike, so
# it is set to the tail requirement (300) and the stricter bulk requirement (400) is
# asserted separately in G4. Stating both is the point: bulk ESS is about the mean,
# tail ESS is about the 90% HPDI this runner prints, and a σ_κ pressed against zero
# fails the second long before the first.
const R64_THRESHOLDS = ConvergenceThresholds(
    max_rhat = 1.05,
    min_ess = 300.0,
    # The audit comparator is strict `<`; eps() expresses an exact-zero count gate.
    max_divergence_rate = eps(Float64),
    min_bfmi = 0.30,
    max_treedepth_rate = 0.05,
)

const R64_MIN_ESS_BULK = 400.0
const R64_MIN_ESS_TAIL = 300.0

# The budget the work package sets. The shared-κ arm measured ~0.03 ms in r60; the
# per-team lookup is one extra vectorised node, so anything near 0.15 ms would mean
# the tape unrolled rather than that the model grew.
const R64_GRADIENT_BUDGET_MS = 0.15
const R64_GRADIENT_REPLAYS = 200

# Two folds off the PRODUCTION splitter, not a bespoke one: an early fold (thin
# history, the hardest case for a per-team parameter) and a late fold (full history).
const R64_N_FOLDS = 2

r64_smoke_sampler = l64_production_sampler   # QueuedNUTSConfig(800 warmup, 800 draws, 4 chains)

# %%
# ==============================================================================
# 3. Fold construction
# ==============================================================================
"""
The first scored boundary and the median scored boundary of the production split.

Sampling both is what makes the ESS numbers transferable to the grid: the early fold
has the fewest matches per team, which is exactly where a hierarchical κ either
shrinks cleanly to the league factor or funnels.
"""
function r64_fold_inputs(model)
    boundaries = Data.create_id_boundaries(ds, l64_production_splitter)
    scored = findall(pair -> !isempty(first(pair).target_match_ids), boundaries)
    isempty(scored) && error("production splitter produced no scored boundary")
    picks = length(scored) >= R64_N_FOLDS ?
            [scored[1], scored[cld(length(scored), 2)]] : scored[1:1]

    selected = [boundaries[i] for i in picks]
    feature_sets = R64_FEATURES.create_features(selected, ds, model, l64_production_splitter)
    oos = [Data.get_next_matches(ds, feature_sets[i], l64_production_splitter)
           for i in eachindex(feature_sets)]
    all(f -> nrow(f) > 0, oos) ||
        error("a selected smoke boundary has no out-of-sample fixtures")
    return feature_sets, oos, picks
end

function r64_density(model, feature_set; seed::Int = 20260903)
    turing_model = R64_PG.build_turing_model(model, feature_set)
    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    return (; turing_model, varinfo, θ, f = x -> LogDensityProblems.logdensity(density, x))
end

# %%
# ==============================================================================
# 4. Gate implementations
# ==============================================================================

"G1: the model's log density must compile to a replayable, size-independent tape."
function r64_assert_gradient_tape(model, feature_sets)
    small = r64_density(model, first(feature_sets[1]))
    @test isfinite(small.f(small.θ))

    raw = ReverseDiff.GradientTape(small.f, small.θ)
    tape = ReverseDiff.compile(raw)
    gradient = similar(small.θ)
    ReverseDiff.gradient!(gradient, tape, small.θ)
    @test all(isfinite, gradient)

    # THE hierarchical-specific check. `log_κ_team[home_idx]` is one vectorised node;
    # had it been written as a `view` or a comprehension the tape would grow with the
    # match count, and the second fold — several times larger — would show it.
    if length(feature_sets) > 1
        large = r64_density(model, first(feature_sets[end]))
        raw_large = ReverseDiff.GradientTape(large.f, large.θ)
        @test length(raw_large.tape) == length(raw.tape)
    end
    @test length(raw.tape) < 500

    for _ in 1:20
        ReverseDiff.gradient!(gradient, tape, small.θ)
    end
    best_ns = Inf
    for _ in 1:R64_GRADIENT_REPLAYS
        started = time_ns()
        ReverseDiff.gradient!(gradient, tape, small.θ)
        best_ns = min(best_ns, Float64(time_ns() - started))
    end
    latency_ms = best_ns / 1.0e6
    @test latency_ms < R64_GRADIENT_BUDGET_MS
    return (; latency_ms, n_instructions = length(raw.tape), n_params = length(small.θ))
end

"G2: the parameter count and the chain schema must both show the hierarchical block."
function r64_assert_structural_contract(name, model, feature_set, chain, n_params)
    n_teams = Int(feature_set.data[:n_teams])
    @test n_params == l64_expected_params(name, n_teams)
    @test cb_parameter_count(model, n_teams) == n_params

    sites = Set(cb_varinfo_sites(model))
    @test Symbol("obs.σ_κ") in sites
    @test Symbol("obs.κ_team_raw") in sites

    chain_names = String.(names(chain))
    @test "obs.ν" in chain_names
    @test "obs.log_κ" in chain_names
    @test "obs.σ_κ" in chain_names
    @test count(c -> startswith(c, "obs.κ_team_raw["), chain_names) == n_teams
    return n_teams
end

"G4: the six-part audit, read at the stricter production thresholds."
function r64_assert_six_gate_audit(diagnostics)
    @test diagnostics.passed
    @test diagnostics.max_rhat < 1.05
    @test diagnostics.min_ess_bulk > R64_MIN_ESS_BULK
    @test diagnostics.min_ess_tail > R64_MIN_ESS_TAIL
    @test diagnostics.n_divergent == 0
    @test diagnostics.min_bfmi >= 0.30
    @test diagnostics.treedepth_rate < 0.05
    return nothing
end

"G5: κ extraction, identification, and the posterior summaries the work package asks for."
function r64_assert_kappa(model, chain, feature_set)
    n_teams = Int(feature_set.data[:n_teams])
    k = R64_PG.extract_kappa(chain, model.observation, n_teams;
                             team_map = feature_set.data[:team_map])

    @test k.mode === :hierarchical
    @test all(isfinite, k.κ_global)
    @test all(>(0.0), k.κ_global)
    @test all(isfinite, k.σ_κ)
    @test all(>=(0.0), k.σ_κ)
    @test size(k.δ_κ, 2) == n_teams
    @test size(k.κ_team, 2) == n_teams

    # IDENTIFICATION. δ_κ is a contrast set by construction, so it sums to zero in
    # EVERY draw — not on average. If this fails, `log κ_global` and the delta mean
    # are trading against each other and neither number means what it is labelled.
    @test maximum(abs, sum(k.δ_κ, dims = 2)) < 1e-10
    @test nrow(k.summary) == n_teams
    @test sum(k.summary.δ_mean) ≈ 0.0 atol = 1e-10
    @test issorted(k.summary.δ_mean, rev = true)
    return k
end

"G6: held-out latents."
function r64_assert_parameter_extraction(model, fit, feature_sets, oos)
    for i in eachindex(fit.folds)
        raw = R64_PG.extract_parameters(model, oos[i], first(feature_sets[i]),
                                        fit.folds[i].chain)
        @test Set(keys(raw)) == Set(Int.(oos[i].match_id))
        for value in values(raw)
            @test all(isfinite, value.λ_h) && all(>(0.0), value.λ_h)
            @test all(isfinite, value.λ_a) && all(>(0.0), value.λ_a)
            # The joint model keeps μ and λ apart; the hierarchical one adds the spread.
            @test haskey(value, :μ_h) && haskey(value, :σ_κ)
        end
    end
    @test fit.latents isa CountLatents
    @test n_matches(fit.latents) == sum(nrow, oos)
    return nothing
end

# G7. These candidates do not estimate a smile; a neutral φ(K) = 1 wrapper exercises
# the grid generation and pricing plumbing without changing their rates.
function r64_neutral_smile_grid(latents::CountLatents)
    strikes = [0.5, 1.5, 2.5]
    λ_tot = latents.λ_home .+ latents.λ_away
    φ = ones(Float64, n_matches(latents), length(strikes), n_draws(latents))
    smile_latents = SmileLatents(
        latents.match_ids, latents.λ_home, latents.λ_away,
        latents.observation_params, λ_tot, φ, strikes)
    grid = compute_score_grid(smile_latents, 1)
    @test grid isa SmileScoreGrid
    @test all(isfinite, grid.grid)
    @test all(isfinite, grid.λ_tot)
    @test all(isfinite, grid.φ)
    return grid
end

"G8: the fit must come back out of PostgreSQL bit-identical."
function r64_assert_round_trip(fit, run_id)
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

"""
G9 (m05 only): the engine's inlined hierarchical density against the independent
reference in `builder/equations.jl`.

`cb_equation_data` covers covariate-only predictor sets, so this runs on the control.
The observation arm is the SAME object in both candidates, so checking it once checks
it for both; what m12 adds is a lineup pillar, which this file does not claim to check.
"""
function r64_assert_reference_parity(model, feature_set)
    data = R64_API.cb_equation_data(model, feature_set)
    d = r64_density(model, feature_set)
    params = R64_API.cb_params_from_varinfo(model, d.varinfo)

    @test params.σ_κ !== nothing
    @test length(params.κ_raw) == Int(feature_set.data[:n_teams])
    @test isfinite(d.f(d.θ))
    @test d.f(d.θ) ≈ R64_API.cb_logjoint(model, params, data) rtol = 1e-9

    for δ in (0.01, -0.02)
        point = d.θ .+ δ .* cos.(collect(eachindex(d.θ)))
        vi = DynamicPPL.unflatten(d.varinfo, point)
        p = R64_API.cb_params_from_varinfo(model, vi)
        @test d.f(point) ≈ R64_API.cb_logjoint(model, p, data) rtol = 1e-8
    end
    return nothing
end

function r64_report_kappa(name, k, n_teams)
    lo, hi = cb_hpdi(k.κ_global, 0.90)
    slo, shi = cb_hpdi(k.σ_κ, 0.90)
    println("\n  κ posterior · $name  ($n_teams teams)")
    @printf("    κ_global : mean %.4f   90%% HPDI [%.4f, %.4f]\n", mean(k.κ_global), lo, hi)
    @printf("    σ_κ      : mean %.4f   90%% HPDI [%.4f, %.4f]   P(σ_κ > 0.05) = %.3f\n",
            mean(k.σ_κ), slo, shi, mean(>(0.05), k.σ_κ))

    top = first(k.summary, 3)
    bottom = last(k.summary, 3)
    println("    over-converting (highest δ_κ):")
    for row in eachrow(top)
        @printf("      %-28s δ_κ %+.4f  [%+.4f, %+.4f]  κ_t %.4f  P(δ>0) %.3f\n",
                row.team, row.δ_mean, row.δ_lo, row.δ_hi, row.κ_mean, row.p_over)
    end
    println("    under-converting (lowest δ_κ):")
    for row in eachrow(bottom)
        @printf("      %-28s δ_κ %+.4f  [%+.4f, %+.4f]  κ_t %.4f  P(δ>0) %.3f\n",
                row.team, row.δ_mean, row.δ_lo, row.δ_hi, row.κ_mean, row.p_over)
    end
    return nothing
end

# %%
# ==============================================================================
# 5. Execution
# ==============================================================================
println("\n" * "="^100)
println(" EXPERIMENT 06 · HIERARCHICAL TEAM KAPPA · EXTENDED TWO-FOLD SMOKE")
println("="^100)
@printf("  matches %d | lineups %d | odds %d | threads %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.odds), Threads.nthreads())
@printf("  sampler   : QueuedNUTSConfig(800 warmup, 800 retained, 4 chains, target %.2f) · QueuedExecution\n",
        r64_smoke_sampler.accept_rate)
println("  thresholds: R̂ < 1.05 | bulk ESS > $(R64_MIN_ESS_BULK) | tail ESS > $(R64_MIN_ESS_TAIL) | divergences == 0")
println("  started   : ", Dates.now())

session = first(string(uuid4()), 8)
summary_rows = NamedTuple[]
kappa_rows = NamedTuple[]

@testset "Experiment 06 hierarchical team kappa smoke" begin
    for (name, model) in l64_candidate_models
        @testset "$name" begin
            println("\n" * "-"^100)
            println(" $name · ", Dates.now())
            println("-"^100)

            feature_sets, oos, picks = r64_fold_inputs(model)
            @printf("  folds: boundaries %s | %d fitted matches | %d held-out fixtures\n",
                    string(picks),
                    sum(length(first(fs).data[:flat_home_ids]) for fs in feature_sets),
                    sum(nrow, oos))

            tape = r64_assert_gradient_tape(model, feature_sets)          # G1
            @printf("  G1 tape: %d instructions | %d parameters | %.4f ms compiled gradient\n",
                    tape.n_instructions, tape.n_params, tape.latency_ms)

            if name == "m05_hierarchical_kappa"
                r64_assert_reference_parity(model, first(feature_sets[1]))  # G9
                println("  G9 parity: engine log-joint == equations.jl reference")
            end

            smoke_name = name * "_smoke_" * session
            config = FitConfig(
                name = smoke_name,
                model = model,
                splitter = l64_production_splitter,
                sampler = r64_smoke_sampler,
                execution = QueuedExecution(),
                tags = [L64_TAGS; "smoke"],
                description = "Two-fold production-settings smoke for $name.",
                save_dir = joinpath("/tmp", "scottish_lower_hierarchical_kappa_smoke", smoke_name),
            )

            started = time()
            fit = fit_model(                                               # G3
                config;
                feature_sets = feature_sets,
                oos_fixtures = oos,
                thresholds = R64_THRESHOLDS,
                quiet = false,
            )
            elapsed = time() - started
            @test length(fit.folds) == length(feature_sets)

            r64_assert_six_gate_audit(fit.diagnostics)                     # G4
            n_teams = r64_assert_structural_contract(                      # G2
                name, model, first(feature_sets[1]), fit.folds[1].chain, tape.n_params)

            # G5 on the LAST fold: the one with the most history behind each team, and
            # therefore the one whose team deltas are worth naming.
            k = r64_assert_kappa(model, fit.folds[end].chain, first(feature_sets[end]))
            r64_report_kappa(name, k, Int(first(feature_sets[end]).data[:n_teams]))

            r64_assert_parameter_extraction(model, fit, feature_sets, oos) # G6
            r64_neutral_smile_grid(fit.latents)                            # G7

            run_id = save_fit(fit, db)                                     # G8
            r64_assert_round_trip(fit, run_id)

            result, books, report = run_portfolio_simulation(
                l60_book, l60_policy, fit, ds.odds, ds;
                bootstrap = false, require_converged = true, quiet = true)
            @test result isa PortfolioResult
            @test report.n_books == length(books)
            portfolio_id = save_portfolio_db(
                result, run_id, db;
                book_spec = l60_book, policy_spec = l60_policy,
                metadata = (; smoke = true, candidate = name, kappa = "hierarchical"))
            loaded_result = load_portfolio_db(portfolio_id, db)
            @test loaded_result.summary.total_return_pct == result.summary.total_return_pct
            @test isequal(loaded_result.trajectory.bets, result.trajectory.bets)

            κ_lo, κ_hi = cb_hpdi(k.κ_global, 0.90)
            σ_lo, σ_hi = cb_hpdi(k.σ_κ, 0.90)
            push!(kappa_rows, (
                name = name, n_teams = size(k.δ_κ, 2),
                κ_mean = mean(k.κ_global), κ_lo = κ_lo, κ_hi = κ_hi,
                σ_mean = mean(k.σ_κ), σ_lo = σ_lo, σ_hi = σ_hi,
                δ_max = maximum(k.summary.δ_mean), δ_min = minimum(k.summary.δ_mean),
                summary = k.summary,
            ))
            push!(summary_rows, (
                name = name, elapsed = elapsed,
                grad_ms = tape.latency_ms, tape_len = tape.n_instructions,
                n_params = tape.n_params, n_teams = n_teams,
                rhat = fit.diagnostics.max_rhat,
                ess_bulk = fit.diagnostics.min_ess_bulk,
                ess_tail = fit.diagnostics.min_ess_tail,
                divergences = fit.diagnostics.n_divergent,
                n_latents = n_matches(fit.latents),
                n_books = length(books),
                run_id = run_id, portfolio_id = portfolio_id,
            ))
        end
    end
end

# %%
# ==============================================================================
# 6. Report
# ==============================================================================
println("\n" * "="^165)
@printf(" %-26s | %7s | %8s | %5s | %6s | %7s | %8s | %8s | %4s | %7s | %36s\n",
        "Model", "Seconds", "Grad ms", "Tape", "Params", "R-hat", "ESS bulk", "ESS tail",
        "Div", "Latents", "Run UUID")
println("-"^165)
for row in summary_rows
    @printf(" %-26s | %7.1f | %8.4f | %5d | %6d | %7.4f | %8.0f | %8.0f | %4d | %7d | %36s\n",
            row.name, row.elapsed, row.grad_ms, row.tape_len, row.n_params, row.rhat,
            row.ess_bulk, row.ess_tail, row.divergences, row.n_latents, string(row.run_id))
end
println("="^165)

println("\n FINISHING FACTOR")
@printf(" %-26s | %6s | %-26s | %-26s | %8s | %8s\n",
        "Model", "Teams", "κ_global (90% HPDI)", "σ_κ (90% HPDI)", "max δ_κ", "min δ_κ")
println("-"^130)
for row in kappa_rows
    @printf(" %-26s | %6d | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] | %+8.4f | %+8.4f\n",
            row.name, row.n_teams, row.κ_mean, row.κ_lo, row.κ_hi,
            row.σ_mean, row.σ_lo, row.σ_hi, row.δ_max, row.δ_min)
end
println("="^130)
println("Finished: ", Dates.now())
println("\nAll gates passed: gradient tape, structural contract, sampling, six-part " *
        "convergence audit, κ identification and extraction, held-out latents, score " *
        "grid, PostgreSQL round-trip, and portfolio persistence.")
