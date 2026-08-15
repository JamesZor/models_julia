# current_development/manager_pace_scalar/r01_smoke.jl
#
# ==============================================================================
# RUNNER: Smoke Test & AD Compilation for Scalar Manager Pace Engine
# ==============================================================================
#
# PURPOSE:
#   Validates ReverseDiff AD compilation, fast NUTS sampling across 4 chains,
#   parameter extraction for (w_wealth, w_pace), and score matrix calculation.
#
# ==============================================================================

using Revise
using BayesianFootball
using Turing
using ReverseDiff
using DynamicPPL
using Distributions
using DataFrames
using Serialization
using Printf
using ThreadPinning
pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Data        = BayesianFootball.Data
const Pred        = BayesianFootball.Predictions
const Diagnostics = BayesianFootball.Experiments.Diagnostics

# 1. Include Loaders
include("l01_manager_pace_data.jl")
include("l02_manager_pace_engine.jl")
include("l03_manager_pace_predict.jl")

# 2. Load Ireland Premier Dataset
println("\n" * "="^80)
println("1. LOADING DATASTORE")
println("="^80)

const PIN_PATH = "./data/l2_ireland_engines/ds_ire79.jls"
ds = isfile(PIN_PATH) ? deserialize(PIN_PATH) : Data.load_datastore_cached(Data.IrelandPremier())
println("✓ Loaded DataStore with $(nrow(ds.matches)) matches")

# 3. Instantiate Dynamic Scalar Manager Pace + Wealth Model
println("\n" * "="^80)
println("2. CONFIGURING UNANCHORED SCALAR MANAGER PACE + WEALTH MODEL")
println("="^80)

model = DynamicSmileDoublePoissonXGWealthManagerPaceModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    manager_pace_feature   = ManagerPaceFeature(pseudo_matches = 15.0),
    player_ratings_feature = Features.PlayerRatingsFeature(
        Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
    ),
    wealth_feature         = TeamWealthFeature(),
    w_wealth_prior         = truncated(Normal(0.105, 0.05), lower=0.0),
    w_pace_prior           = truncated(Normal(0.05, 0.03), lower=0.0),
    market_on              = false
)

# 4. Build Features
println("\n" * "="^80)
println("3. BUILDING FEATURE SET")
println("="^80)

# Fast slice of matches for rapid smoke testing (first 150 matches)
smoke_matches = first(ds.matches, 150)
smoke_ds = Data.DataStore(
    ds.segment,
    smoke_matches,
    filter(r -> r.match_id in smoke_matches.match_id, ds.statistics),
    filter(r -> r.match_id in smoke_matches.match_id, ds.odds),
    filter(r -> r.match_id in smoke_matches.match_id, ds.lineups),
    filter(r -> r.match_id in smoke_matches.match_id, ds.incidents),
    filter(r -> r.match_id in smoke_matches.match_id, ds.betfair_odds),
    ds.bbc,
    ds.bbc_events
)

boundary = Data.SplitBoundary(
    1, 1,
    Vector{Int}(smoke_matches.match_id[1:100]),
    Vector{Int}(smoke_matches.match_id[101:150])
)
feature_set = Features.create_features(boundary, smoke_ds, model, :season)
println("✓ Feature set built successfully:")
println("  - Matches    : $(length(feature_set.data[:flat_home_ids]))")
println("  - Teams      : $(feature_set.data[:n_teams])")
println("  - Wealth diff: $(length(feature_set.data[:flat_wealth_diff]))")
println("  - Pace sum   : $(length(feature_set.data[:pace_sum])) (mean = $(round(mean(feature_set.data[:pace_sum]), digits=4)))")

# 5. Turing Model Instantiation & AD Tape Compilation
println("\n" * "="^80)
println("4. TESTING TURING MODEL & REVERSEDIFF AD COMPILATION")
println("="^80)

turing_model = PreGame.build_turing_model(model, feature_set)
println("✓ Turing model constructed")

# Evaluate logjoint at prior mode / initialization
println("Evaluating logjoint...")
logp = DynamicPPL.logjoint(turing_model, DynamicPPL.VarInfo(turing_model))
println("✓ Logjoint initial evaluation: $logp (finite = $(isfinite(logp)))")
@assert isfinite(logp) "Initial logjoint is not finite!"

# 6. Fast NUTS MCMC Sampling (4 Chains x 100 Samples)
println("\n" * "="^80)
println("5. FAST NUTS SAMPLING (100 samples, 4 chains)")
println("="^80)

sampler = NUTS(100, 0.65)
chain = sample(
    turing_model,
    sampler,
    MCMCThreads(),
    100,
    4;
    progress = true,
    adtype = AutoReverseDiff(compile = true)
)
println("✓ Sampling complete!")

# 7. Check Sampled Values for Scalar Parameters
println("\n" * "="^80)
println("6. CONVERGENCE & PARAMETER CHECKS")
println("="^80)

describe(chain[[:w_wealth, :w_pace, :σ_γ, :σ_κ, :φ_h, :φ_a]])

# 8. Test Parameter Extraction & Prediction Pipeline
println("\n" * "="^80)
println("7. PARAMETER EXTRACTION & PREDICTION VALIDATION")
println("="^80)

param_dict = PreGame.extract_parameters(model, smoke_matches, feature_set, chain)
println("✓ Extracted parameters for $(length(param_dict)) matches")

first_match = first(smoke_matches)
first_params = param_dict[first_match.match_id]
score_matrix = Pred.compute_score_matrix(model, first_params; max_goals=12)

prob_grid = score_matrix.grid.data # (max_goals+1) x (max_goals+1) x n_samples
prob_sums = vec(sum(prob_grid, dims=(1, 2)))
println("✓ Probabilities sum test: Mean = $(mean(prob_sums)), Min = $(minimum(prob_sums)), Max = $(maximum(prob_sums))")
@assert all(abs.(prob_sums .- 1.0) .< 1e-5) "Score probabilities do not sum to 1.0!"

println("\n" * "="^80)
println("🎉 ALL SMOKE CHECKS PASSED SUCCESSFULLY!")
println("="^80)
