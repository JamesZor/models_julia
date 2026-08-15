# current_development/manager_pace_scalar/r06_eval_manager_pace_scalar_anchored.jl
#
# RUNNER: Comprehensive Model Evaluation via BayesianFootball.Evaluation
#
# Comparing:
# 1. Baseline Unanchored (`l2_ire79_noanchor`)
# 2. Market Anchored Baseline (`l2_ire79_sup40_sw40`)
# 3. Team Wealth Unanchored (`l2_ire79_wealth`)
# 4. Team Wealth + Market Anchored (`l2_ire79_wealth_sup40_sw40`)
# 5. Scalar Manager Pace + Wealth Unanchored (`l2_ire79_mgr_pace_scalar`)
# 6. Scalar Manager Pace + Wealth + Market Anchored (`l2_ire79_wealth_pace_sup40_sw40`)

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

include(joinpath(dirname(@__DIR__), "team_wealth", "l01_wealth_data.jl"))
include(joinpath(dirname(@__DIR__), "team_wealth", "l02_wealth_engine.jl"))
include(joinpath(dirname(@__DIR__), "team_wealth", "l03_wealth_predict.jl"))

include(joinpath(@__DIR__, "l01_manager_pace_data.jl"))
include(joinpath(@__DIR__, "l02_manager_pace_engine.jl"))
include(joinpath(@__DIR__, "l03_manager_pace_predict.jl"))

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

function find_newest_experiment(pattern::Regex)
    dirs = filter(d -> occursin(pattern, basename(d)),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR) if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r06: No experiment matching pattern '$pattern' found in $ENGINE_DIR")
    sorted = sort(dirs, by = mtime, rev = true)
    return Experiments.load_experiment(sorted[1])
end

banner("1. LOADING DATASTORE & CANDIDATE EXPERIMENTS")

const PIN_PATH = joinpath(ENGINE_DIR, "ds_ire79.jls")
ds = isfile(PIN_PATH) ? deserialize(PIN_PATH) : Data.load_datastore_cached(Data.IrelandPremier())

exp_noanchor          = find_newest_experiment(r"^l2_ire79_noanchor_\d+")
exp_anchored          = find_newest_experiment(r"^l2_ire79_sup40_sw40_\d+")
exp_wealth            = find_newest_experiment(r"^l2_ire79_wealth_\d+")
exp_wealth_anchor     = find_newest_experiment(r"^l2_ire79_wealth_sup40_sw40_\d+")
exp_mgr_pace_scalar   = find_newest_experiment(r"^l2_ire79_mgr_pace_scalar_\d+")
exp_mgr_pace_anchored = find_newest_experiment(r"^l2_ire79_wealth_pace_sup40_sw40_\d+")

experiments = [
    exp_noanchor,
    exp_anchored,
    exp_wealth,
    exp_wealth_anchor,
    exp_mgr_pace_scalar,
    exp_mgr_pace_anchored
]

println("✓ Loaded DataStore ($(nrow(ds.matches)) matches)")
println("✓ Loaded [1] Baseline Unanchored                 : $(exp_noanchor.config.name)")
println("✓ Loaded [2] Market Anchored Baseline            : $(exp_anchored.config.name)")
println("✓ Loaded [3] Team Wealth Unanchored              : $(exp_wealth.config.name)")
println("✓ Loaded [4] Team Wealth + Anchored              : $(exp_wealth_anchor.config.name)")
println("✓ Loaded [5] Scalar Pace + Wealth Unanchored     : $(exp_mgr_pace_scalar.config.name)")
println("✓ Loaded [6] Scalar Pace + Wealth + Anchored     : $(exp_mgr_pace_anchored.config.name)")

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

# Serialize artifact
out_file = joinpath(OUT_DIR, "evaluation_suite_anchored_pace_scalar_results.jls")
serialize(out_file, eval_df)
println("\n✓ Serialized full evaluation DataFrame to: $out_file")
