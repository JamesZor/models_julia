# current_development/team_wealth/r07_train_wealth_anchored.jl
#
# RUNNER: Train 4th Candidate Arm — Team Wealth + Market Anchoring
# (`l2_ire79_wealth_sup40_sw40`)
#
# Market Anchoring Settings:
#   market_on        = true
#   supremacy_weight = 1.0
#   smile_weight     = 0.5

using ThreadPinning
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

# Pin OS threads to physical cores
pinthreads(:cores)
println("ThreadPinning: $(Threads.nthreads()) threads pinned across physical cores.")

include(joinpath(@__DIR__, "l01_wealth_data.jl"))
include(joinpath(@__DIR__, "l02_wealth_engine.jl"))
include(joinpath(@__DIR__, "l03_wealth_predict.jl"))

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Data        = BayesianFootball.Data

const OUT_DIR  = "./data/l2_ireland_engines"
const PIN_PATH = joinpath(OUT_DIR, "ds_ire79.jls")

isfile(PIN_PATH) || error("r07: Missing pinned DataStore at $PIN_PATH")
ds = deserialize(PIN_PATH)
println("✓ Loaded pinned DataStore for Ireland 79 ($(nrow(ds.matches)) matches).")

# ==============================================================================
# 1. Model Configuration
# ==============================================================================

function make_wealth_anchored_engine()
    return DynamicSmileDoublePoissonXGWealthPlayerTimeDecayModel(
        interception_config    = PreGame.HierarchicalMonthlyInterception(),
        player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
        homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
        kappa_config           = PreGame.HierarchicalTeamKappa(),
        player_ratings_feature = Features.PlayerRatingsFeature(
                                     Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
        wealth_feature         = TeamWealthFeature(),
        w_wealth_prior         = truncated(Normal(0.105, 0.05), lower=0.0),
        market_feature_config  = Features.DoublePoissonMarketFeature(),
        smile_feature          = Features.MarketSmileFeature(Kmax = 4),
        market_on              = true,         # Market Anchoring Active
        supremacy_weight       = 1.0,          # User setting: 1.0
        smile_weight           = 0.5,          # User setting: 0.5
    )
end

# ==============================================================================
# 2. Training Task
# ==============================================================================

const TARGET_SEASONS = ["2025", "2026"]
const DYNAMICS_COL   = :match_biweek
const SAMPLES, WARMUP, CHAINS, MAX_DEPTH = 800, 300, 4, 10

println("\n", "="^90)
println("STARTING TRAINING: l2_ire79_wealth_sup40_sw40 (Team Wealth + Market Anchored)")
println("="^90)

task = Experiments.create_experiment_task(
    ds, make_wealth_anchored_engine(), "l2_ire79_wealth_sup40_sw40", OUT_DIR;
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
        bad   = filter(r -> r.rhat >= 1.01, diag)
        @printf("Max R-hat across all folds: %.4f (worst parameter: %s)\n", maxr, worst)
        @printf("Parameters with R-hat >= 1.01: %d / %d\n", nrow(bad), nrow(diag))
        if maxr < 1.02
            println("✓ CONVERGENCE GATE PASSED")
        else
            @warn "CONVERGENCE GATE FAILED (max R-hat >= 1.02)"
        end
    end
catch e
    @warn "Diagnostics extraction skipped: $e"
end
