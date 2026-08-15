# current_development/manager_pace_scalar/r02_train_manager_pace_scalar.jl
#
# RUNNER: Train the 1-Parameter Scalar Manager Pace + Unanchored Team Wealth Model on Ireland 79
#
# ==============================================================================
# EXPERIMENT SETUP
# ==============================================================================
# - Model: DynamicSmileDoublePoissonXGWealthManagerPacePlayerTimeDecayModel
# - Market Anchor: OFF (market_on = false)
# - Manager Tactical Pace: ManagerPaceFeature (pseudo_matches = 15.0)
# - Priors:
#     w_wealth ~ Truncated(Normal(0.105, 0.05), lower = 0.0)
#     w_pace   ~ Truncated(Normal(0.05, 0.03), lower = 0.0)
# - Data: ds_ire79.jls (pinned DataStore)
# - Target Seasons: ["2025", "2026"] (history_seasons = 2, match_biweek)
# - Sampler: QueuedNUTSConfig (800 samples, 300 warmup, 4 chains, max_depth = 10)
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization
using ThreadPinning

# Pin OS threads to physical cores
pinthreads(:cores)
println("ThreadPinning: $(Threads.nthreads()) threads pinned across physical cores.")

include(joinpath(@__DIR__, "l01_manager_pace_data.jl"))
include(joinpath(@__DIR__, "l02_manager_pace_engine.jl"))
include(joinpath(@__DIR__, "l03_manager_pace_predict.jl"))

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Data        = BayesianFootball.Data

const OUT_DIR = "./data/l2_ireland_engines"
const PIN_PATH = joinpath(OUT_DIR, "ds_ire79.jls")

isfile(PIN_PATH) || error("r02: Missing pinned DataStore at $PIN_PATH")
ds = deserialize(PIN_PATH)
println("✓ Loaded pinned DataStore for Ireland 79 ($(nrow(ds.matches)) matches).")

# ==============================================================================
# 1. Model Configuration
# ==============================================================================

function make_manager_pace_scalar_engine()
    return DynamicSmileDoublePoissonXGWealthManagerPacePlayerTimeDecayModel(
        interception_config    = PreGame.HierarchicalMonthlyInterception(),
        player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
        homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
        kappa_config           = PreGame.HierarchicalTeamKappa(),
        player_ratings_feature = Features.PlayerRatingsFeature(
            Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
        ),
        wealth_feature         = TeamWealthFeature(),
        manager_pace_feature   = ManagerPaceFeature(pseudo_matches = 15.0),
        w_wealth_prior         = truncated(Normal(0.105, 0.05), lower=0.0),
        w_pace_prior           = truncated(Normal(0.05, 0.03), lower=0.0),
        market_feature_config  = Features.DoublePoissonMarketFeature(),
        smile_feature          = Features.MarketSmileFeature(Kmax = 4),
        market_on              = false,        # Unanchored
        supremacy_weight       = 0.4,
        smile_weight           = 0.4,
    )
end

# ==============================================================================
# 2. Training Task
# ==============================================================================

const TARGET_SEASONS = ["2025", "2026"]
const DYNAMICS_COL   = :match_biweek
const SAMPLES, WARMUP, CHAINS, MAX_DEPTH = 800, 300, 4, 10

println("\n", "="^90)
println("STARTING FULL TRAINING: l2_ire79_mgr_pace_scalar (Scalar Manager Pace + Team Wealth)")
println("="^90)

task = Experiments.create_experiment_task(
    ds, make_manager_pace_scalar_engine(), "l2_ire79_mgr_pace_scalar", OUT_DIR;
    target_seasons        = TARGET_SEASONS,
    history_seasons       = 2,
    warmup_period         = 0,
    dynamics_col          = DYNAMICS_COL,
    samples               = SAMPLES,
    warmup                = WARMUP,
    chains                = CHAINS,
    use_queue             = true,
    max_depth             = MAX_DEPTH,
    max_concurrent_splits = 8,
    max_concurrent_tasks  = 16,
)

t0 = time()
res = Experiments.run_experiment(task)
Experiments.save_experiment(res)
elapsed = (time() - t0) / 60
@printf("\n✓ Training complete in %.2f minutes (%d splits saved).\n", elapsed, length(res.training_results.items))

# ==============================================================================
# 3. Convergence Diagnostics Gate
# ==============================================================================

println("\n", "="^90)
println("CONVERGENCE GATE CHECK")
println("="^90)

try
    diag = Diagnostics.check_convergence(Diagnostics.extract_chains(ds, res)).df
    if !isempty(diag)
        maxr  = maximum(diag.rhat)
        worst = string(diag[argmax(diag.rhat), :parameter])
        bad   = filter(r -> r.rhat >= 1.05, diag)
        @printf("Max R-hat across all folds: %.4f (worst parameter: %s)\n", maxr, worst)
        @printf("Number of parameters with R-hat >= 1.05: %d / %d\n", nrow(bad), nrow(diag))
        if !isempty(bad)
            println("\nTop 10 elevated R-hat parameters:")
            sort!(bad, :rhat, rev=true)
            show(stdout, MIME"text/plain"(), first(bad, 10))
            println()
        end
    end
catch e
    @warn "Convergence diagnostic calculation skipped: $e"
end

println("\n", "="^90)
println("🎉 l2_ire79_mgr_pace_scalar TRAINING & PERSISTENCE FINISHED")
println("="^90)
