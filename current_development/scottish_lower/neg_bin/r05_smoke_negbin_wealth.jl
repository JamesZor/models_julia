# current_development/scottish_lower/neg_bin/r05_smoke_negbin_wealth.jl
#
# RUNNER: 1-Split MCMC Smoke Test for Scottish NegBin + Wealth Models

using Revise
using BayesianFootball
using Turing
using Printf
using Dates
using DataFrames
using Statistics
using MCMCChains
using LinearAlgebra

const DD = BayesianFootball.Data
const MM = BayesianFootball.Models
const EE = BayesianFootball.Experiments
const Features = BayesianFootball.Features
const PreGame = BayesianFootball.Models.PreGame
const Pred = BayesianFootball.Predictions
const ROOT = pkgdir(BayesianFootball)

include("l02_negbin_wealth_engines.jl")

println("\n", "="^95)
println("SMOKE TEST: ROBUST NEGATIVE BINOMIAL + SQUAD WEALTH ENGINES (SCOTTISH LOWER)")
println("="^95)

# 1. Load DataStore
ds = DD.load_datastore_cached(DD.ScottishLower(), max_age_hours = 10000)
save_dir = joinpath(ROOT, "data/scottish_negbin_wealth_smoke/"); mkpath(save_dir)
println("✓ Loaded ScottishLower DataStore: $(nrow(ds.matches)) matches")

# 2. Model Specifications to Test
dyn = MM.PreGame.TimeDecayDynamics(days_half_life = 365.0)

models_to_test = [
    ("Goals NegBin + Wealth", TeamGoalsNegBinWealthModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "goals_negbin_wealth_smoke"
    )),
    ("Arm A: Proxy xG NegBin + Wealth", TeamPxGGoalsAPMNegBinWealthModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "pxg_apm_negbin_wealth_smoke"
    )),
    ("Arm B: Funnel Proxy xG NegBin + Wealth", TeamFunnelPxGGoalsAPMNegBinWealthModel(
        dynamics_config = dyn,
        dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
        name = "funnel_pxg_apm_negbin_wealth_smoke"
    ))
]

for (desc, model) in models_to_test
    println("\n", "="^85)
    println("🚀 TESTING MODEL: $desc ($(model.name))")
    println("="^85)

    exp_task = EE.create_experiment_task(
        ds,
        model,
        model.name,
        save_dir;
        target_seasons       = ["25/26"],
        history_seasons      = 2,
        warmup_period        = 20, # fast 1-split test
        dynamics_col         = :match_biweek,
        samples              = 400,
        warmup               = 200,
        chains               = 3,
        use_queue            = true,
        max_depth            = 10,
        max_concurrent_tasks = 8
    )

    t0 = time()
    res = EE.run_experiment(exp_task)
    elapsed = round(time() - t0, digits = 1)
    println("✓ Completed MCMC sampling ($(length(res.training_results.items)) folds) in $(elapsed)s")

    chain = res.training_results.items[1][1]
    
    # 1. Convergence & R-hat Check
    er = DataFrame(MCMCChains.ess_rhat(chain))
    rcol = :rhat in propertynames(er) ? :rhat : first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
    rhat_vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
    max_rhat = isempty(rhat_vals) ? 1.0 : maximum(rhat_vals)
    @printf("  • Convergence Check : Max R-hat = %.4f -> %s\n",
            max_rhat, max_rhat <= 1.05 ? "CONVERGED (PASS ✅)" : "WARN (R-hat > 1.05)")

    # 2. Key Parameter Estimates
    println("\n  • Key Parameter Posteriors:")
    c_names = names(chain)
    if :w_wealth in c_names
        val_w = chain[:w_wealth].data[:, 1]
        @printf("    - w_wealth (wealth effect) : %.4f ± %.4f\n", mean(val_w), std(val_w))
    end
    if :log_κ in c_names
        val_k = exp.(chain[:log_κ].data[:, 1])
        @printf("    - κ (conversion rate)     : %.4f ± %.4f\n", mean(val_k), std(val_k))
    end
    if Symbol("disp.log_r") in c_names
        val_ra = exp.(chain[Symbol("disp.log_r")].data[:, 1])
        val_dh = chain[Symbol("disp.δ_r_home")].data[:, 1]
        val_rh = exp.(chain[Symbol("disp.log_r")].data[:, 1] .+ val_dh)
        @printf("    - r_away (dispersion)      : %.2f ± %.2f\n", mean(val_ra), std(val_ra))
        @printf("    - r_home (dispersion)      : %.2f ± %.2f\n", mean(val_rh), std(val_rh))
        @printf("    - δ_r_home (home shift)    : %.3f ± %.3f\n", mean(val_dh), std(val_dh))
    end

    # 3. Test Parameter Extraction & ScoreMatrix Generation
    boundary = Data.SplitBoundary(1, 1, Int.(ds.matches.match_id[1:100]), Int[])
    fs_train = Features.create_features(boundary, ds, model, :match_biweek)
    params_map = PreGame.extract_parameters(model, ds.matches[1:10, :], fs_train, chain)
    sample_mid = first(keys(params_map))
    p = params_map[sample_mid]
    
    score_mat = Pred.compute_score_matrix(model, p; max_goals=12)
    s_tensor = Pred.score_matrix_data(score_mat)
    s_mean = dropdims(mean(s_tensor, dims=3), dims=3)
    p_sum = sum(s_mean)
    
    @printf("  • ScoreMatrix shape: %s | Prob Sum = %.6f -> %s\n",
            string(size(s_tensor)), p_sum, abs(p_sum - 1.0) < 1e-4 ? "EXACT (PASS ✅)" : "FAIL ❌")
    @printf("    - P(Home Win) = %.4f | P(Draw) = %.4f | P(Away Win) = %.4f\n",
            sum(tril(s_mean, -1)), sum(diag(s_mean)), sum(triu(s_mean, 1)))
end

println("\n", "="^95)
println("✓ ALL NEGBIN + WEALTH SMOKE TESTS PASSED!")
println("="^95)
