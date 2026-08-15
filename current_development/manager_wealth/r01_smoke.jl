# current_development/manager_wealth/r01_smoke.jl
#
# ==============================================================================
# RUNNER: Smoke Test & AD Compilation for Manager + Wealth Model
# ==============================================================================
#
# PURPOSE:
#   Validates ReverseDiff AD compilation, fast NUTS sampling across 4 chains,
#   parameter extraction for (δ_pace, α_mgr, w_wealth), and score matrix calculation.
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

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Data        = BayesianFootball.Data
const Pred        = BayesianFootball.Predictions
const Diagnostics = BayesianFootball.Experiments.Diagnostics

# 1. Include Loaders
include("l01_manager_wealth_data.jl")
include("l02_manager_wealth_engine.jl")
include("l03_manager_wealth_predict.jl")

# 2. Load Ireland Premier Dataset
println("\n" * "="^80)
println("1. LOADING DATASTORE")
println("="^80)

const PIN_PATH = "./data/l2_ireland_engines/ds_ire79.jls"
ds = isfile(PIN_PATH) ? deserialize(PIN_PATH) : Data.load_datastore_cached(Data.IrelandPremier())
println("✓ Loaded DataStore with $(nrow(ds.matches)) matches")

# 3. Instantiate Dynamic Manager + Wealth Model
println("\n" * "="^80)
println("2. CONFIGURING UNANCHORED MANAGER + WEALTH MODEL")
println("="^80)

model = DynamicSmileDoublePoissonXGWealthManagerModel(
    interception_config = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config = PreGame.HierarchicalTeamKappa(),
    manager_pace_config = HierarchicalManagerPace(
        σ_pace = truncated(Normal(0.0, 0.15), lower=0.0)
    ),
    manager_quality_config = HierarchicalManagerQuality(
        σ_mgr = truncated(Normal(0.0, 0.15), lower=0.0)
    ),
    player_ratings_feature = Features.PlayerRatingsFeature(
        Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
    ),
    wealth_feature = TeamWealthFeature(),
    manager_feature = ManagerFeature(),
    w_wealth_prior = truncated(Normal(0.105, 0.05), lower=0.0),
    market_on = false
)

# 4. Build Features
println("\n" * "="^80)
println("3. BUILDING FEATURE SET")
println("="^80)

# Fast slice of matches for rapid smoke testing (first 150 matches)
smoke_matches = first(ds.matches, 150)
smoke_ds = Data.DataStore(
    smoke_matches,
    filter(r -> r.match_id in smoke_matches.match_id, ds.odds),
    filter(r -> r.match_id in smoke_matches.match_id, ds.betfair_odds),
    filter(r -> r.match_id in smoke_matches.match_id, ds.statistics),
    filter(r -> r.match_id in smoke_matches.match_id, ds.lineups),
    filter(r -> r.match_id in smoke_matches.match_id, ds.incidents)
)

feature_set = Features.build_feature_set(model, smoke_ds)
println("✓ Feature set built successfully:")
println("  - Matches    : $(length(feature_set.data[:flat_home_ids]))")
println("  - Teams      : $(feature_set.data[:n_teams])")
println("  - Managers   : $(feature_set.data[:n_managers])")
println("  - Wealth diff: $(length(feature_set.data[:flat_wealth_diff]))")

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

Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

sampler = NUTS(50, 0.65)
chain = sample(turing_model, sampler, MCMCThreads(), 100, 4; progress=true)
println("✓ Sampling complete!")

# 7. Check Sampled Values for Key Hierarchical Parameters
println("\n" * "="^80)
println("6. CONVERGENCE & PARAMETER CHECKS")
println("="^80)

println(chain[[:w_wealth, Symbol("mgr_pace.σ_pace"), Symbol("mgr_qual.σ_mgr"), Symbol("ha.σ_γ"), Symbol("kap.σ_κ")]])

# 8. Test Parameter Extraction & Prediction Pipeline
println("\n" * "="^80)
println("7. PARAMETER EXTRACTION & PREDICTION VALIDATION")
println("="^80)

param_dict = PreGame.extract_parameters(model, smoke_matches, feature_set, chain)
println("✓ Extracted parameters for $(length(param_dict)) matches")

first_match = first(smoke_matches)
first_params = param_dict[first_match.match_id]
score_matrix = Pred.compute_score_matrix(model, first_params; max_goals=12)

prob_grid = score_matrix.matrix.matrix # 12 x 12 x n_samples
prob_sums = vec(sum(prob_grid, dims=(1, 2)))
println("✓ Probabilities sum test: Mean = $(mean(prob_sums)), Min = $(minimum(prob_sums)), Max = $(maximum(prob_sums))")
@assert all(abs.(prob_sums .- 1.0) .< 1e-5) "Score probabilities do not sum to 1.0!"

println("\n" * "="^80)
println("🎉 ALL SMOKE CHECKS PASSED SUCCESSFULLY!")
println("="^80)
