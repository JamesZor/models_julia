# current_development/team_wealth/r09_eval_fourway.jl
#
# RUNNER: 4-Way Model Evaluation via BayesianFootball.Evaluation
#
# Comparing:
# 1. Baseline Unanchored (`l2_ire79_noanchor`)
# 2. Team Wealth Unanchored (`l2_ire79_wealth`)
# 3. Market Anchored (`l2_ire79_sup40_sw40`)
# 4. Team Wealth + Market Anchored (`l2_ire79_wealth_sup40_sw40`)

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

include(joinpath(@__DIR__, "l01_wealth_data.jl"))
include(joinpath(@__DIR__, "l02_wealth_engine.jl"))
include(joinpath(@__DIR__, "l03_wealth_predict.jl"))

include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l00_corpus.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l01_l2_experiment.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l02_l2_ledger.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l03_l2_metrics.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l04_corpus_replay.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l05_curation.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l06_fullbook.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l07_route2.jl"))

const ENGINE_DIR = "./data/l2_ireland_engines"
const OUT_DIR    = "./data/l2_route2_wealth"
mkpath(OUT_DIR)

banner(s) = (println("\n", "="^95); println(s); println("="^95))

function find_newest_experiment(prefix::String)
    dirs = filter(d -> startswith(basename(d), prefix),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR) if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r09: No experiment matching prefix '$prefix' found in $ENGINE_DIR")
    sorted = sort(dirs, by = mtime, rev = true)
    return Experiments.load_experiment(sorted[1])
end

banner("1. LOADING DATASTORE & EXPERIMENTS")

const PIN_PATH = joinpath(ENGINE_DIR, "ds_ire79.jls")
ds = isfile(PIN_PATH) ? deserialize(PIN_PATH) : Data.load_datastore_cached(Data.IrelandPremier())

exp_noanchor       = find_newest_experiment("l2_ire79_noanchor")
exp_wealth         = find_newest_experiment("l2_ire79_wealth")
exp_anchored       = find_newest_experiment("l2_ire79_sup40_sw40")
exp_wealth_anchor  = find_newest_experiment("l2_ire79_wealth_sup40_sw40")

experiments = [exp_noanchor, exp_wealth, exp_anchored, exp_wealth_anchor]

println("✓ Loaded DataStore ($(nrow(ds.matches)) matches)")
println("✓ Loaded [1] Baseline Unanchored      : $(exp_noanchor.config.name)")
println("✓ Loaded [2] Team Wealth Unanchored   : $(exp_wealth.config.name)")
println("✓ Loaded [3] Market Anchored          : $(exp_anchored.config.name)")
println("✓ Loaded [4] Team Wealth + Anchored   : $(exp_wealth_anchor.config.name)")

banner("2. EXECUTING BATCH EVALUATION SUITE")

metrics = [
    Evaluation.LogLoss(),
    Evaluation.GLMEdge(),
    Evaluation.RQR(),
    Evaluation.CRPS()
]

eval_df = Evaluation.evaluate_experiments(metrics, experiments, ds)

banner("3. STANDARDIZED EVALUATION SUMMARIES")

println("\n" * "="^80)
Evaluation.display_summary_metric(eval_df, :logloss)
Evaluation.display_summary_metric(eval_df, :glmedge)
Evaluation.display_summary_metric(eval_df, :rqr)
Evaluation.display_summary_metric(eval_df, :crps)
println("="^80)

# Save evaluation output artifact
out_path = joinpath(OUT_DIR, "evaluation_suite_fourway_results.jls")
serialize(out_path, eval_df)
println("\n✓ Serialized full evaluation DataFrame to: $out_path")
