using Revise
using BayesianFootball
using DataFrames

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_corner_data.jl")
include("l05_corner_recomb_pipeline.jl")

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
all_folders = Experiments.list_experiments("scottish_corner_grid"; data_dir = joinpath(ROOT, "data"))
exps = Experiments.load_experiments(all_folders)
exp_corner = [e for e in exps if length(e.training_results.items) > 0][end]

println("Testing extract_oos_predictions for: ", exp_corner.config.name)
try
    latents = Experiments.extract_oos_predictions(ds, exp_corner; force=true)
    println("✓ Latents extracted successfully! Nrow = ", nrow(latents.df))
    println("Columns: ", names(latents.df))

    println("Testing CRPS computation...")
    crps_res = Evaluation.compute_metric(Evaluation.CRPS(), exp_corner, ds, latents)
    println("✓ CRPS: ", crps_res)

    println("Testing LogLoss computation...")
    ll_res = Evaluation.compute_metric(Evaluation.LogLoss(:home), exp_corner, ds, latents)
    println("✓ LogLoss(:home): ", ll_res)

    println("Testing batch evaluation...")
    metrics = Evaluation.AbstractScoringRule[Evaluation.CRPS(), Evaluation.LogLoss(:home)]
    eval_df = Evaluation.evaluate_experiments(metrics, [exp_corner], ds)
    println("✓ Batch evaluation: ")
    show(stdout, MIME("text/plain"), eval_df)
    println()
catch e
    println("\n!!! EXCEPTION CAUGHT !!!")
    println(e)
    showerror(stdout, e, catch_backtrace())
    println("\n")
end
