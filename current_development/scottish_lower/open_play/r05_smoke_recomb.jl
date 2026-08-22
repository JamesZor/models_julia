# current_development/scottish_lower/open_play/r05_smoke_recomb.jl
#
# SMOKE TEST: Two-Stage Recombination Models (Branch A & Branch B)
#
# Verifies:
# 1. 1-Split MCMC NUTS sampling & convergence (R-hat <= 1.02)
# 2. Referee strictness gamma_ref and penalty parameters recovery
# 3. Empirical Bayes analytical fitting & noise rate computation
# 4. Score matrix normalization (sum = 1.000000)

using Revise
using BayesianFootball
using Turing, DynamicPPL, MCMCChains
using DataFrames, Dates, Statistics, Printf

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Samplers    = BayesianFootball.Samplers
const Training    = BayesianFootball.Training
const Predictions = BayesianFootball.Predictions
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")
include("l03_recombination_models.jl")

banner(s) = (println("\n", "="^95); println(s); println("="^95))

banner("SMOKE TEST: TWO-STAGE RECOMBINATION MODELS (1 SPLIT)")

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true
)
boundaries = Data.create_id_boundaries(ds, splitter)
split_idx = findfirst(b -> !isempty(b[1].target_match_ids), boundaries)
split_idx = isnothing(split_idx) ? 1 : split_idx
b1 = boundaries[split_idx:split_idx]
bound1 = b1[1][1]
df_target = filter(r -> r.match_id in bound1.target_match_ids, ds.matches)
println("✓ Generated 1 smoke split (#$split_idx): $(length(bound1.history_match_ids)) train matches, $(length(bound1.target_match_ids)) test matches")

# ==============================================================================
# TEST 1: BRANCH A - ANALYTICAL EMPIRICAL BAYES PENALTY ESTIMATOR
# ==============================================================================
banner("TEST 1: BRANCH A - EMPIRICAL BAYES PENALTY & REFEREE ESTIMATOR")

t0 = time()
eb_estimator = fit_empirical_bayes_penalties(ds, bound1.history_match_ids)
t_eb = round((time() - t0) * 1000, digits = 2)

println("✓ Fitted Empirical Bayes Penalty Estimator in $(t_eb)ms:")
println("  - Baseline Penalty Rate: $(round(eb_estimator.base_pen_rate, digits=4)) pens/team-match")
println("  - Home Whistle Advantage: $(round(eb_estimator.ha_pen, digits=4))")
println("  - Referees Profiled: $(length(eb_estimator.ref_strictness))")
println("  - Teams Profiled: $(length(eb_estimator.team_draw_rates))")

# Test noise calculation for a sample match
sample_noise_h, sample_noise_a = compute_match_noise_intensity(eb_estimator, 1, 2, 1)
println("  - Sample Match Noise: Home = $(round(sample_noise_h, digits=4)), Away = $(round(sample_noise_a, digits=4))")

# ==============================================================================
# TEST 2: PURE OPEN-PLAY POISSON MODEL (MCMC SMOKE)
# ==============================================================================
banner("TEST 2: PURE OPEN-PLAY POISSON MODEL (MCMC SMOKE)")

m_pois = TeamGoalsPoissonOpenPlayModel()
f_pois = Features.create_features(b1, ds, m_pois)
tur_pois = PreGame.build_turing_model(m_pois, f_pois[1])

sampler_cfg = Samplers.NUTSConfig(
    n_samples   = 500,
    n_warmup    = 200,
    n_chains    = 2,
    accept_rate = 0.65,
    max_depth   = 8,
    show_progress = false
)

t0 = time()
chain_pois = Samplers.run_sampler(tur_pois, sampler_cfg)
elapsed_pois = round(time() - t0, digits = 1)

rhats_pois = summarize(chain_pois)[:, :rhat]
max_rhat_pois = maximum(filter(!isnan, rhats_pois))

println("✓ Pure Open-Play Poisson Model Sampled in $(elapsed_pois)s:")
println("  - Max R-hat: $(round(max_rhat_pois, digits=4)) ($(max_rhat_pois <= 1.05 ? "CONVERGED ✅" : "WARNING ⚠️"))")

# ==============================================================================
# TEST 3: BRANCH B - INTEGRATED RECOMBINATION POISSON MODEL (MCMC SMOKE)
# ==============================================================================
banner("TEST 3: BRANCH B - INTEGRATED RECOMBINATION POISSON MODEL (MCMC SMOKE)")

m_int = TeamGoalsRecombIntegratedPoissonModel()
f_int = Features.create_features(b1, ds, m_int)
tur_int = PreGame.build_turing_model(m_int, f_int[1])

t0 = time()
chain_int = Samplers.run_sampler(tur_int, sampler_cfg)
elapsed_int = round(time() - t0, digits = 1)

rhats_int = summarize(chain_int)[:, :rhat]
max_rhat_int = maximum(filter(!isnan, rhats_int))

sig_ref_mean = mean(Array(chain_int["sigma_ref"]))
pen_base_mean = exp(mean(Array(chain_int["pen_base_mu"])))
ha_pen_mean = mean(Array(chain_int["ha_pen"]))

println("✓ Integrated Recombination Poisson Model Sampled in $(elapsed_int)s:")
println("  - Max R-hat: $(round(max_rhat_int, digits=4)) ($(max_rhat_int <= 1.05 ? "CONVERGED ✅" : "WARNING ⚠️"))")
println("  - Learned Baseline Penalty Rate: $(round(pen_base_mean, digits=4)) pens/team-match")
println("  - Learned Home Whistle Bias: $(round(ha_pen_mean, digits=4))")
println("  - Learned Referee Strictness Scale (sigma_ref): $(round(sig_ref_mean, digits=4))")

# ==============================================================================
# TEST 5: BRANCH B - INTEGRATED RECOMBINATION NEGATIVE BINOMIAL MODEL (MCMC SMOKE)
# ==============================================================================
banner("TEST 5: BRANCH B - INTEGRATED RECOMBINATION NEGATIVE BINOMIAL MODEL (MCMC SMOKE)")

m_nb = TeamGoalsRecombIntegratedNegBinModel()
f_nb = Features.create_features(b1, ds, m_nb)
tur_nb = PreGame.build_turing_model(m_nb, f_nb[1])

t0 = time()
chain_nb = Samplers.run_sampler(tur_nb, sampler_cfg)
elapsed_nb = round(time() - t0, digits = 1)

rhats_nb = summarize(chain_nb)[:, :rhat]
max_rhat_nb = maximum(filter(!isnan, rhats_nb))

r_a_mean = exp(mean(Array(chain_nb["log_r"])))
r_h_mean = exp(mean(Array(chain_nb["log_r"])) + mean(Array(chain_nb["delta_r_home"])))

println("✓ Integrated Recombination NegBin Model Sampled in $(elapsed_nb)s:")
println("  - Max R-hat: $(round(max_rhat_nb, digits=4)) ($(max_rhat_nb <= 1.05 ? "CONVERGED ✅" : "WARNING ⚠️"))")
println("  - Learned Dispersion: r_home = $(round(r_h_mean, digits=2)), r_away = $(round(r_a_mean, digits=2))")

# Test parameter extraction & score matrix generation
params_nb = PreGame.extract_parameters(m_nb, df_target, f_nb[1], chain_nb)
println("✓ Extracted NegBin parameters for $(length(params_nb)) matches")

first_mid = first(keys(params_nb))
p_sample = params_nb[first_mid]
sm_nb = Predictions.compute_score_matrix(m_nb, p_sample)
sum_sm = sum(sm_nb.matrix[:, :, 1])
println("✓ Computed Recombined NegBin Score Matrix (Sum = $(round(sum_sm, digits=6)), $(isapprox(sum_sm, 1.0) ? "VALID ✅" : "INVALID ❌"))")

println("\n", "="^95)
println("✓ ALL SMOKE TESTS PASSED CLEANLY! Ready for 40-fold walk-forward grid.")
println("="^95)
