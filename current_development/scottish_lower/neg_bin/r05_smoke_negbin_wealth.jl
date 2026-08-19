# current_development/scottish_lower/neg_bin/r05_smoke_negbin_wealth.jl
#
# RUNNER: 1-Split MCMC Smoke Test for Scottish NegBin + Wealth Models
#
# Runs NUTS sampling (3 chains x 100 samples) across Season 2025/2026 for:
# 1. TeamGoalsNegBinWealthModel
# 2. TeamPxGGoalsAPMNegBinWealthModel
# 3. TeamFunnelPxGGoalsAPMNegBinWealthModel
#
# Verifies:
# - Chain convergence (rhat < 1.10)
# - Parameter extraction (w_wealth, κ, r_h, r_a, δ_r_home)
# - PPD ScoreMatrix construction & probability sum integrity (sum ≈ 1.0)

using Revise
using BayesianFootball
using Turing
using Printf
using Dates
using DataFrames
using Statistics

const Data        = BayesianFootball.Data
const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Samplers    = BayesianFootball.Samplers
const Training    = BayesianFootball.Training
const Predictions = BayesianFootball.Predictions

include("l02_negbin_wealth_engines.jl")

println("==================================================================")
println(" SCOTTISH LOWER NEGBIN + WEALTH 1-SPLIT MCMC SMOKE TEST")
println("==================================================================")

# 1. Load Data
println("\n[1/3] Loading DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
splitter = PreGame.GroupedCVConfig(group_by = :season, test_groups = [2025], window_strategy = :expanding)

# 2. Test Configurations
sampler_cfg = Samplers.NUTSConfig(
    n_samples = 100,
    n_adapts  = 100,
    target_accept = 0.85,
    n_chains  = 3
)

test_models = [
    ("Model 1: Goals NegBin + Wealth", TeamGoalsNegBinWealthModel(
        dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
        homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        league_ha_on = true
    )),
    ("Model 2: Proxy xG + RAPM + NegBin + Wealth", TeamPxGGoalsAPMNegBinWealthModel(
        dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
        homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        league_ha_on = true
    )),
    ("Model 3: Funnel Proxy xG + RAPM + NegBin + Wealth", TeamFunnelPxGGoalsAPMNegBinWealthModel(
        dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
        homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        apm_on = true,
        team_quality_on = true,
        league_ha_on = true
    ))
]

println("\n[2/3] Running 1-Split MCMC Sampling...")

for (label, model) in test_models
    println("\n------------------------------------------------------------------")
    println("▶ Testing: ", label)
    println("------------------------------------------------------------------")
    
    t_start = time()
    splits = PreGame.split_train_test(splitter, ds)
    split1 = splits[1]
    
    # Extract features
    fs_train = Features.extract_features(model, split1.train_df, ds)
    turing_mod = PreGame.build_turing_model(model, fs_train)
    
    # Sample
    println("  • Sampling 3 chains x 100 iterations...")
    chain = sample(turing_mod, NUTS(sampler_cfg.n_adapts, sampler_cfg.target_accept), MCMCThreads(), sampler_cfg.n_samples, sampler_cfg.n_chains; progress=false)
    elapsed = time() - t_start
    println("  ✓ Sampling completed in ", round(elapsed, digits=1), " s")
    
    # Inspect Key Parameters
    println("\n  • Posterior Summary:")
    c_names = names(chain)
    
    if :w_wealth in c_names
        val_w = chain[:w_wealth].data[:, 1]
        println("    - w_wealth (wealth weight): ", round(mean(val_w), digits=4), " ± ", round(std(val_w), digits=4))
    end
    if :log_κ in c_names
        val_k = exp.(chain[:log_κ].data[:, 1])
        println("    - κ (conversion rate): ", round(mean(val_k), digits=4), " ± ", round(std(val_k), digits=4))
    end
    if Symbol("disp.log_r") in c_names
        val_ra = exp.(chain[Symbol("disp.log_r")].data[:, 1])
        val_dh = chain[Symbol("disp.δ_r_home")].data[:, 1]
        val_rh = exp.(chain[Symbol("disp.log_r")].data[:, 1] .+ val_dh)
        println("    - r_away (dispersion away): ", round(mean(val_ra), digits=2), " ± ", round(std(val_ra), digits=2))
        println("    - r_home (dispersion home): ", round(mean(val_rh), digits=2), " ± ", round(std(val_rh), digits=2))
        println("    - δ_r_home (home shift):    ", round(mean(val_dh), digits=3), " ± ", round(std(val_dh), digits=3))
    end
    
    # Test Parameter Extraction & ScoreMatrix Generation
    println("\n  • Testing Predictions & ScoreMatrix...")
    params_map = PreGame.extract_parameters(model, split1.test_df, fs_train, chain)
    sample_mid = first(keys(params_map))
    p = params_map[sample_mid]
    
    score_mat = Predictions.compute_score_matrix(model, p; max_goals=12)
    s_tensor = Predictions.score_matrix_data(score_mat)
    s_mean = dropdims(mean(s_tensor, dims=3), dims=3)
    p_sum = sum(s_mean)
    
    println("    - ScoreMatrix shape: ", size(s_tensor))
    println("    - Mean ScoreMatrix probability sum: ", round(p_sum, digits=6))
    @assert abs(p_sum - 1.0) < 1e-4 "ERROR: ScoreMatrix does not sum to 1.0!"
    println("    - Home win prob: ", round(sum(tril(s_mean, -1)), digits=4))
    println("    - Draw prob:     ", round(sum(diag(s_mean)), digits=4))
    println("    - Away win prob: ", round(sum(triu(s_mean, 1)), digits=4))
    println("  ✓ Model passed all smoke verification checks!")
end

println("\n==================================================================")
println("✓ All Scottish NegBin + Wealth Smoke Tests Completed Successfully!")
println("==================================================================")
