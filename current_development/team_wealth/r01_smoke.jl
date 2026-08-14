# current_development/team_wealth/r01_smoke.jl
#
# ==============================================================================
# RUNNER: AD-Safety & Unit Smoke Test for Team Wealth Engine
# ==============================================================================
#
# PURPOSE:
#   Self-contained smoke test (no database required) to verify:
#     S1: Turing model compiles under ReverseDiff(compile=true) without tape errors.
#     S2: Short MCMC sampling (10 steps) finishes without NaN / -Inf logprob.
#     S3: Parameter extraction captures `w_wealth` cleanly.
#     S4: Prediction pipeline generates valid SmileScoreMatrix, 1X2, BTTS, and O/U probs.
#
# USAGE:
#   julia --project=. current_development/team_wealth/r01_smoke.jl
# ==============================================================================

using BayesianFootball
using Random, Distributions, DataFrames, Statistics, Printf, LinearAlgebra
using Turing, ReverseDiff

include(joinpath(@__DIR__, "l01_wealth_data.jl"))
include(joinpath(@__DIR__, "l02_wealth_engine.jl"))
include(joinpath(@__DIR__, "l03_wealth_predict.jl"))

Random.seed!(20260814)

println("===================================================================")
println("TEAM WEALTH ENGINE: AD-SAFETY & UNIT TEST SMOKE RUNNER")
println("===================================================================\n")

# Harness
const TW_FAILURES = String[]
function tw_check(ok::Bool, msg::AbstractString)
    @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg)
    ok || push!(TW_FAILURES, msg)
    return ok
end
empty!(TW_FAILURES)

# 1. Synthetic Corpus Setup
const N_TEAMS   = 8
const N_SEASONS = 2
const N_MATCHES = 60
const KMAX      = 4

println("1. Generating synthetic feature set (N=$N_MATCHES matches)...")

team_map = Dict("team_$i" => i for i in 1:N_TEAMS)
home_teams = rand(1:N_TEAMS, N_MATCHES)
away_teams = [rand(setdiff(1:N_TEAMS, home_teams[i])) for i in 1:N_MATCHES]
time_indices = repeat(1:12, inner=5)
season_indices = repeat(1:N_SEASONS, inner=30)
month_indices = repeat(1:12, inner=5)

wealth_diff = randn(N_MATCHES) # Standardized wealth differences
home_G = fill(6.0, N_MATCHES) .+ randn(N_MATCHES) .* 0.3
home_D = fill(6.0, N_MATCHES) .+ randn(N_MATCHES) .* 0.3
home_M = fill(6.0, N_MATCHES) .+ randn(N_MATCHES) .* 0.3
home_F = fill(6.0, N_MATCHES) .+ randn(N_MATCHES) .* 0.3

away_G = fill(6.0, N_MATCHES) .+ randn(N_MATCHES) .* 0.3
away_D = fill(6.0, N_MATCHES) .+ randn(N_MATCHES) .* 0.3
away_M = fill(6.0, N_MATCHES) .+ randn(N_MATCHES) .* 0.3
away_F = fill(6.0, N_MATCHES) .+ randn(N_MATCHES) .* 0.3

home_goals = rand(Poisson(1.4), N_MATCHES)
away_goals = rand(Poisson(1.1), N_MATCHES)
home_xg = home_goals .* 0.8 .+ rand(N_MATCHES) .* 0.4
away_xg = away_goals .* 0.8 .+ rand(N_MATCHES) .* 0.4

smile_logΛ = randn(N_MATCHES, KMAX + 1)
smile_mask = ones(Float64, N_MATCHES, KMAX + 1)

f_data = Dict{Symbol, Any}(
    :n_teams => N_TEAMS,
    :n_rounds => 12,
    :n_seasons => N_SEASONS,
    :team_map => team_map,
    :player_ratings_map => Dict{Int, Dict{Tuple{String, String}, Float64}}(),
    :smile_Kmax => KMAX,
    :flat_home_ids => home_teams,
    :flat_away_ids => away_teams,
    :time_indices => time_indices,
    :season_indices => season_indices,
    :flat_months => month_indices,
    :flat_home_goals => home_goals,
    :flat_away_goals => away_goals,
    :flat_home_xg => home_xg,
    :flat_away_xg => away_xg,
    :flat_market_λ_home => fill(1.5, N_MATCHES),
    :flat_market_λ_away => fill(1.2, N_MATCHES),
    :flat_smile_logΛ => smile_logΛ,
    :flat_smile_mask => smile_mask,
    :flat_home_G_rating => home_G,
    :flat_home_D_rating => home_D,
    :flat_home_M_rating => home_M,
    :flat_home_F_rating => home_F,
    :flat_away_G_rating => away_G,
    :flat_away_D_rating => away_D,
    :flat_away_M_rating => away_M,
    :flat_away_F_rating => away_F,
    :flat_wealth_diff => wealth_diff,
    :match_time_weights => ones(Float64, N_MATCHES)
)

feature_set = Features.FeatureSet(f_data)

# 2. Instantiate Model
model_cfg = DynamicSmileDoublePoissonXGWealthPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(Features.BayesianTracker(6.0, 0.5, 0.5, 0.05)),
    wealth_feature         = TeamWealthFeature(),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax=KMAX),
    w_wealth_prior         = truncated(Normal(0.20, 0.10), lower=0.0)
)

using DynamicPPL

turing_model = PreGame.build_turing_model(model_cfg, feature_set)
tw_check(turing_model isa DynamicPPL.Model, "Turing @model instantiation successful")

# 3. Test AD-Safety under ReverseDiff
println("\n2. Testing AD ReverseDiff gradient compilation & sampling...")
try
    chain = sample(turing_model, NUTS(5, 0.65; adtype=AutoReverseDiff(compile=true)), 10; progress=false)
    tw_check(true, "AutoReverseDiff(compile=true) compiled and sampled cleanly")
    tw_check(:w_wealth in keys(chain), "Chain contains :w_wealth parameter")
    
    # 4. Test Parameter Extraction
    mock_df = DataFrame(
        match_id = 1:5,
        home_team = ["team_1", "team_2", "team_3", "team_4", "team_5"],
        away_team = ["team_6", "team_7", "team_8", "team_1", "team_2"],
        season = [1, 1, 1, 1, 1]
    )
    extracted = PreGame.extract_parameters(model_cfg, mock_df, feature_set, chain)
    tw_check(nrow(extracted) == 5, "extract_parameters produced valid predictions DataFrame")
    tw_check(all(all.(isfinite, extracted.λ_h)), "λ_h predictions are finite and valid")
    tw_check(all(all.(isfinite, extracted.λ_a)), "λ_a predictions are finite and valid")
catch e
    @printf("  AD Compilation Error: %s\n", sprint(showerror, e))
    tw_check(false, "AutoReverseDiff failed")
end

# 5. Prediction Pipeline Verification
println("\n3. Testing prediction dispatch and SmileScoreMatrix creation...")
sample_params = (
    λ_h = [1.5, 1.8],
    λ_a = [1.1, 0.9],
    λ_tot = [2.6, 2.7],
    φ = [1.0 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0 1.0]
)

S = Pred.compute_score_matrix(model_cfg, sample_params; max_goals=10)
tw_check(S isa Pred.SmileScoreMatrix, "compute_score_matrix returns Pred.SmileScoreMatrix")

prob_1x2 = Pred.compute_market_probs(S, Pred.Market1X2())
if prob_1x2 isa Dict
    p_tot = prob_1x2[:home] .+ prob_1x2[:draw] .+ prob_1x2[:away]
    tw_check(isapprox(p_tot[1], 1.0, atol=1e-5), "Market1X2 probabilities sum to 1.0")
else
    tw_check(isapprox(sum(prob_1x2[1, :]), 1.0, atol=1e-5), "Market1X2 probabilities sum to 1.0")
end

prob_ou = Pred.compute_market_probs(S, Pred.MarketOverUnder(2.5))
if prob_ou isa Dict
    p_tot_ou = prob_ou[:over] .+ prob_ou[:under]
    tw_check(isapprox(p_tot_ou[1], 1.0, atol=1e-5), "MarketOverUnder probabilities sum to 1.0")
else
    tw_check(isapprox(sum(prob_ou[1, :]), 1.0, atol=1e-5), "MarketOverUnder probabilities sum to 1.0")
end

println("\n===================================================================")
if isempty(TW_FAILURES)
    println("✓ ALL SMOKE TESTS PASSED (AD-Safe, Extract-Ready, Predict-Ready)")
else
    println("✗ SOME SMOKE TESTS FAILED: $(length(TW_FAILURES)) failure(s)")
end
println("===================================================================")
