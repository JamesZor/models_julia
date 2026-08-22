# ==============================================================================
# r12_smoke_recomb_wealth.jl
#
# Smoke test & parameter verification for:
# Integrated Poisson Recombination + Starting-XI Squad Wealth Model (Scottish Lower)
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf
using MCMCChains
using Turing

include("l04_recomb_wealth_models.jl")

println("="^90)
println("🔬 SMOKE TEST: INTEGRATED POISSON RECOMBINATION + SQUAD WEALTH MODEL")
println("="^90)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower())
println("✓ Loaded DataStore ($(nrow(ds.matches)) matches, $(nrow(ds.incidents)) incidents, $(nrow(ds.lineups)) lineups)")

# 2. Instantiate Model
model = TeamGoalsRecombIntegratedPoisWealthModel(
    dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
    w_wealth_prior = truncated(Normal(0.10, 0.05), lower = 0.0),
    name = "recomb_pois_wealth_smoke"
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
bound1 = boundaries[end][1] # Most recent fold (SplitBoundary)
println("✓ Fold 40: $(length(bound1.history_match_ids)) history matches, $(length(bound1.target_match_ids)) test matches")

fset = Features.create_features(bound1, ds, model)
f = fset.data
println("✓ FeatureSet built:")
println("  • n_teams = $(f[:n_teams]), n_refs = $(f[:n_refs])")
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
base_mu       = mean(Array(chain["base_mu"]))
ha_home       = mean(Array(chain["ha_home"]))
tau_alpha     = mean(Array(chain["tau_alpha"]))
tau_beta      = mean(Array(chain["tau_beta"]))
sigma_ref     = mean(Array(chain["sigma_ref"]))

println("  • base_mu (Open-Play Intercept)       : $(round(base_mu, digits=4))")
println("  • ha_home (Home Advantage)            : $(round(ha_home, digits=4))")
println("  • w_wealth (Squad Wealth Sensitivity) : $(round(w_wealth_mean, digits=4)) ± $(round(w_wealth_sd, digits=4))")
println("  • τ_alpha (Attack Spread)             : $(round(tau_alpha, digits=4))")
println("  • τ_beta (Defense Spread)             : $(round(tau_beta, digits=4))")
println("  • σ_ref (Referee Whistle Spread)      : $(round(sigma_ref, digits=4))")

# 6. Test Out-of-Sample Parameter Extraction & Score Matrix
latents = Predictions.extract_params(model, fset, chain)
println("\n✓ Extracted OOS LatentStates for $(nrow(latents.df)) target matches")

first_match = latents.df[1, :]
params = (
    mu_open_h    = first_match.mu_open_h_samples[1],
    mu_open_a    = first_match.mu_open_a_samples[1],
    lambda_pen_h = first_match.lambda_pen_h_samples[1],
    lambda_pen_a = first_match.lambda_pen_a_samples[1],
    q_pen        = first_match.q_pen_samples[1],
    rho          = first_match.rho_samples[1]
)

S = Predictions.compute_score_matrix(model, params)
println("✓ Computed Score Matrix for Match $(first_match.match_id) (Sum = $(round(sum(S), digits=6)))")
println("  • μ_open_h = $(round(params.mu_open_h, digits=3)), μ_open_a = $(round(params.mu_open_a, digits=3))")
println("  • λ_pen_h  = $(round(params.lambda_pen_h, digits=3)), λ_pen_a  = $(round(params.lambda_pen_a, digits=3))")
println("  • P(Home Win) = $(round(sum(tril(S, -1)), digits=4))")
println("  • P(Draw)     = $(round(sum(diag(S)), digits=4))")
println("  • P(Away Win) = $(round(sum(triu(S, 1)), digits=4))")

println("\n" * "="^90)
println("✓ RECOMBINATION + SQUAD WEALTH SMOKE TEST COMPLETED SUCCESSFULLY!")
println("="^90)
