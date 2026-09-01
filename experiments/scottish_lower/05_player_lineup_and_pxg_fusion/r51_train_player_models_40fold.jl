# ==============================================================================
# r51 — Production 40-fold player-lineup grid
# Prepared for mcmc-beast. Do not launch while r46 is sampling.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l50_loader.jl"))

println("\n" * "="^110)
println(" EXPERIMENT 05 · PRODUCTION PLAYER-LINEUP GRID · 24/25 + 25/26")
println("="^110)
println("  sampler   : QueuedNUTSConfig(800 warmup, 800 retained, 4 chains)")
println("  execution : QueuedExecution")
println("  threads   : ", Threads.nthreads())
println("  database  : ", db)
println("  started   : ", Dates.now())

boundaries = Data.create_id_boundaries(ds, l50_production_splitter)
scored = count(pair -> !isempty(first(pair).target_match_ids), boundaries)
target_matches = sum(pair -> length(first(pair).target_match_ids), boundaries; init = 0)
println("  boundaries: $(length(boundaries)) total | $scored scored | $target_matches held-out matches")
length(boundaries) == 40 || error(
    "production splitter produced $(length(boundaries)) boundaries; expected 40. " *
    "Do not sample until the datastore/splitter drift is resolved.")
scored > 0 || error("production splitter has no scored folds")

run_ids = Dict{String,Any}()
rows = NamedTuple[]

for (name, _) in l50_candidate_models
    config = l50_configs[name]
    println("\n" * "-"^100)
    println(" GRID: $name · ", Dates.now())
    println("-"^100)

    checkpoint_dir = joinpath(config.save_dir, "checkpoints")
    started = time()
    fit = fit_model(
        config, ds;
        checkpoint_dir = checkpoint_dir,
        cleanup_checkpoints = true,
        quiet = false,
    )
    elapsed = time() - started

    fit.latents isa CountLatents || error(
        "$name returned $(typeof(fit.latents)); PostgreSQL production persistence requires CountLatents")
    run_id = save_fit(fit, db)
    run_ids[name] = run_id

    push!(rows, (
        name = name,
        elapsed = elapsed,
        passed = fit.diagnostics.passed,
        rhat = fit.diagnostics.max_rhat,
        ess = min(fit.diagnostics.min_ess_bulk, fit.diagnostics.min_ess_tail),
        divergences = fit.diagnostics.n_divergent,
        n_latents = n_matches(fit.latents),
        run_id = run_id,
    ))
    @printf("  persisted %s in %.1f minutes · run %s\n", name, elapsed / 60, string(run_id))
end

println("\n" * "="^145)
@printf(" %-45s | %9s | %7s | %7s | %4s | %8s | %36s\n",
        "Model", "Minutes", "R-hat", "ESS", "Div", "Latents", "Run UUID")
println("-"^145)
for row in rows
    @printf(" %-45s | %9.1f | %7.4f | %7.0f | %4d | %8d | %36s\n",
            row.name, row.elapsed / 60, row.rhat, row.ess, row.divergences,
            row.n_latents, string(row.run_id))
end
println("="^145)
println("Finished: ", Dates.now())
