# ==============================================================================
# r15_smoke_recomb_pxg.jl
#
# Smoke test & parameter recovery verification for:
# Integrated Open-Play Proxy xG (pxG) Recombination + Squad Wealth Engine
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf
using MCMCChains
using Turing

include("l05_recomb_pxg_models.jl")

println("="^90)
println("🔬 SMOKE TEST: OPEN-PLAY PROXY xG (pxG) RECOMBINATION + SQUAD WEALTH MODEL")
println("="^90)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower())
println("✓ Loaded DataStore ($(nrow(ds.matches)) matches, $(nrow(ds.incidents)) incidents, $(nrow(ds.lineups)) lineups)")

# 2. Instantiate Model
model = TeamPxGRecombWealthIntegratedModel(
    dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
    w_wealth_prior = truncated(Normal(0.10, 0.05), lower = 0.0),
    ν_xg_prior = truncated(Normal(3.5, 0.5), lower = 0.5),
    name = "recomb_pxg_wealth_smoke"
)

# 3. Create Splitter & 1-Fold FeatureSet
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25", "25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true
)

boundaries = Data.create_id_boundaries(ds, splitter)
bound1 = boundaries[end][1] # Most recent fold (Fold 40)
println("✓ Fold 40: $(length(bound1.history_match_ids)) history matches, $(length(bound1.target_match_ids)) test matches")

fset = Features.create_features(bound1, ds, model)
f = fset.data
println("✓ FeatureSet built:")
println("  • n_teams = $(f[:n_teams]), n_refs = $(f[:n_refs])")
println("  • Valid pxG observations (Home): $(count(>(0.0), f[:mask_pxg_h])) / $(length(f[:mask_pxg_h]))")
println("  • Valid pxG observations (Away): $(count(>(0.0), f[:mask_pxg_a])) / $(length(f[:mask_pxg_a]))")
println("  • Mean open-play pxG (Home): $(round(mean(f[:pxg_open_h][f[:mask_pxg_h] .> 0]), digits=3))")
println("  • Mean open-play pxG (Away): $(round(mean(f[:pxg_open_a][f[:mask_pxg_a] .> 0]), digits=3))")
println("  • Non-zero wealth diffs: $(count(!iszero, f[:wealth_diff])) / $(length(f[:wealth_diff]))")
println("  • Mean |ΔW|: $(round(mean(abs.(f[:wealth_diff])), digits=3))")

# 4. Build Turing Model & Run Fast NUTS Sample
turing_mod = PreGame.build_turing_model(model, fset)

sampler_cfg = Samplers.NUTSConfig(
    n_samples   = 400,
    n_warmup    = 150,
    n_chains    = 2,
    accept_rate = 0.65,
    max_depth   = 8,
    show_progress = false
)

println("\n[MCMC] Sampling with Queued/Optimized NUTS (ReverseDiff)...")
t0 = time()
chain = Samplers.run_sampler(turing_mod, sampler_cfg)
elapsed = round(time() - t0, digits=2)
println("✓ Sampled in $(elapsed)s")

# 5. Inspect Posterior Distributions
println("\n" * "="^90)
println("📊 POSTERIOR PARAMETER RECOVERY")
println("="^90)

w_wealth_mean = mean(Array(chain["w_wealth"]))
w_wealth_sd   = std(Array(chain["w_wealth"]))
nu_xg_mean    = mean(Array(chain["ν_xg"]))
nu_xg_sd      = std(Array(chain["ν_xg"]))
base_mu       = mean(Array(chain["base_mu"]))
ha_home       = mean(Array(chain["ha_home"]))
tau_alpha     = mean(Array(chain["tau_alpha"]))
tau_beta      = mean(Array(chain["tau_beta"]))
sigma_ref     = mean(Array(chain["sigma_ref"]))

println("  • base_mu (Open-Play Intercept)       : $(round(base_mu, digits=4))")
println("  • ha_home (Home Advantage)            : $(round(ha_home, digits=4))")
println("  • w_wealth (Squad Wealth Sensitivity) : $(round(w_wealth_mean, digits=4)) ± $(round(w_wealth_sd, digits=4))")
println("  • ν_xg (Open-Play pxG Precision)      : $(round(nu_xg_mean, digits=4)) ± $(round(nu_xg_sd, digits=4))")
println("  • τ_alpha (Attack Spread)             : $(round(tau_alpha, digits=4))")
println("  • τ_beta (Defense Spread)             : $(round(tau_beta, digits=4))")
println("  • σ_ref (Referee Whistle Spread)      : $(round(sigma_ref, digits=4))")

# 6. Test Score Matrix Recombination
println("\n" * "="^90)
println("🎲 SCORE MATRIX TEST")
println("="^90)

test_match = first(bound1.target_match_ids)
m_row = filter(r -> r.match_id == test_match, ds.matches)[1, :]
params = Dict(
    :chain        => chain,
    :team_map     => f[:team_map],
    :ref_map      => f[:ref_map],
    :wealth_map   => f[:wealth_map],
    :home_team_id => m_row.home_team_id,
    :away_team_id => m_row.away_team_id,
    :match_id     => m_row.match_id,
    :referee_id   => 0
)

sm = Predictions.compute_score_matrix(model, params; max_goals=12)
sm_data = Predictions.score_matrix_data(sm)
matrix_sum = sum(mean(sm_data, dims=3))
p_home = sum(tril(mean(sm_data, dims=3), -1))
p_draw = sum(diag(mean(sm_data, dims=3)[:, :, 1]))
p_away = sum(triu(mean(sm_data, dims=3), 1))

println("✓ Match: $(m_row.home_team) vs $(m_row.away_team)")
println("  • Matrix Sum = $(round(matrix_sum, digits=6)) (Expected: 1.000000)")
println("  • Probabilities: Home=$(round(p_home*100, digits=1))% | Draw=$(round(p_draw*100, digits=1))% | Away=$(round(p_away*100, digits=1))%")

println("\n✓ SMOKE TEST COMPLETED SUCCESSFULLY!")
