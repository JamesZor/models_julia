# ==============================================================================
# Scottish Lower — Poisson vs Negative Binomial 24/25 + 25/26 Benchmark
# BayesianFootball.jl Unified V2 Stack
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, ThreadPinning, LinearAlgebra, Printf, Statistics

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

println("\n" * "="^120)
println(" POISSON VS NEGATIVE BINOMIAL 24/25 + 25/26 HEAD-TO-HEAD BENCHMARK")
println("="^120)

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

poisson_candidates = [
    "/root/BayesianFootball/data/scottish_lower_2426_grid",
    "/root/BayesianFootball/experiments/scottish_lower_2426",
    joinpath(pwd(), "data", "scottish_lower_2426_grid")
]
poisson_dir = something(findfirst(isdir, poisson_candidates), 1) |> i -> poisson_candidates[i]
negbin_dir  = isdir("/root/BayesianFootball/experiments/scottish_lower_2426_negbin") ? 
    "/root/BayesianFootball/experiments/scottish_lower_2426_negbin" :
    joinpath(pwd(), "experiments", "scottish_lower_2426_negbin")

models_base = [
    ("m00", "Baseline"),
    ("m02", "Squad Wealth"),
    ("m03", "Travel Distance"),
    ("m04", "Joint Wealth+Dist"),
    ("m05", "Production Wealth"),
]

model_pairs = [
    ("Baseline",          "m00_baseline",          "m00_negbin_baseline"),
    ("Squad Wealth",      "m02_wealth",            "m02_negbin_wealth"),
    ("Travel Distance",   "m03_distance",          "m03_negbin_distance"),
    ("Joint Wealth+Dist", "m04_joint",             "m04_negbin_joint"),
    ("Production Wealth", "m05_production_wealth", "m05_negbin_production_wealth"),
]

comparison_rows = []

for (label, p_name, nb_name) in model_pairs
    p_path  = joinpath(poisson_dir, p_name)
    nb_path = joinpath(negbin_dir, nb_name)
    
    if !isdir(p_path) || !isdir(nb_path)
        continue
    end
    
    fit_p  = load_fit(p_path)
    fit_nb = load_fit(nb_path)
    
    eval_p  = evaluate_predictions(fit_p, ds)
    eval_nb = evaluate_predictions(fit_nb, ds)
    
    push!(comparison_rows, (
        feature     = label,
        p_logloss   = eval_p.model.logloss,
        nb_logloss  = eval_nb.model.logloss,
        p_brier     = eval_p.model.brier,
        nb_brier    = eval_nb.model.brier,
        p_rps       = eval_p.model.rps,
        nb_rps      = eval_nb.model.rps,
        p_rhat      = fit_p.diagnostics.max_rhat,
        nb_rhat     = fit_nb.diagnostics.max_rhat
    ))
end

if isempty(comparison_rows)
    println(" Waiting for both grid runs to complete...")
else
    @printf(" %-20s | %9s | %9s | %8s | %9s | %9s | %8s\n",
            "Model Feature", "P-LogLoss", "NB-LogLoss", "Δ LogLoss", "P-Brier", "NB-Brier", "Δ Brier")
    println("-"^100)
    for r in comparison_rows
        d_ll    = r.nb_logloss - r.p_logloss
        d_brier = r.nb_brier - r.p_brier
        @printf(" %-20s | %9.4f | %9.4f | %+8.4f | %9.4f | %9.4f | %+8.4f\n",
                r.feature, r.p_logloss, r.nb_logloss, d_ll, r.p_brier, r.nb_brier, d_brier)
    end
    println("="^100)
end
