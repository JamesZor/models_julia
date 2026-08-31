# ==============================================================================
# Scottish Lower — Negative Binomial 5-Model Smoke Test (Fold 1)
# BayesianFootball.jl Unified V2 Stack
# ==============================================================================
#
# Tests all 5 Negative Binomial model architectures on Fold 1 of 24/25:
#   1. m00_negbin_baseline          : Pure NegBin + Time Decay + Home Adv + Global Dispersion
#   2. m02_negbin_wealth            : m00 + Raw Squad Wealth LogSum Differential
#   3. m03_negbin_distance          : m00 + Away Ground Travel Distance
#   4. m04_negbin_joint             : m00 + Raw Wealth + Travel Distance
#   5. m05_negbin_production_wealth : m00 + Age-Adjusted Production Wealth (Richards Sigmoid)
#
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, ThreadPinning, LinearAlgebra, Printf, Statistics, Distributions, MCMCChains

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

println("\n" * "="^100)
println(" SCOTTISH LOWER: NEGATIVE BINOMIAL 5-MODEL SMOKE TEST (FOLD 1)")
println("="^100)

# 1. Load Cached DataStore
println("\n[1/4] Loading DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
println("  Matches: $(nrow(ds.matches)) | Lineups: $(nrow(ds.lineups)) | Odds: $(nrow(ds.odds))")

# 2. Assemble 5 Negative Binomial Models via Composable Builder
println("\n[2/4] Assembling 5 Composable Negative Binomial Models...")

m00 = CountModelBuilder(:m00_negbin_baseline) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

m02 = CountModelBuilder(:m02_negbin_wealth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(WealthCovariate(prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

m03 = CountModelBuilder(:m03_negbin_distance) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(DistanceCovariate(prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

m04 = CountModelBuilder(:m04_negbin_joint) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(WealthCovariate(prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
    add(DistanceCovariate(prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

m05 = CountModelBuilder(:m05_negbin_production_wealth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(ProductionWealthCovariate(
        feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
        prior   = truncated(Normal(0.10, 0.05), lower = 0.0)
    )) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

models = [
    ("m00_negbin_baseline",          m00),
    ("m02_negbin_wealth",            m02),
    ("m03_negbin_distance",          m03),
    ("m04_negbin_joint",             m04),
    ("m05_negbin_production_wealth", m05),
]

# 3. Single-Fold Walk-Forward Splitter (Fold 1 of 24/25)
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    end_dynamics      = 1,
    stop_early        = true,
)

# Sampler settings for smoke test: 500 warmup + 500 samples, 4 chains
sampler_cfg = NUTSConfig(
    n_samples   = 500,
    n_warmup    = 500,
    n_chains    = 4,
    accept_rate = 0.65
)

# Single fold execution
save_root = "/tmp/scottish_lower_negbin_smoke"
mkpath(save_root)

# 4. Train and Audit Each Model on Fold 1
println("\n[3/4] Running Fold 1 Smoke Fits...")
fits = Dict{String, Fit}()
summary_rows = []

for (name, m) in models
    println("\n" * "-"^80)
    println(" SMOKE FIT: $name")
    println("-"^80)
    
    # Restrict to Fold 1 for fast verification
    fit_cfg = FitConfig(
        name      = name,
        model     = m,
        splitter  = splitter,
        sampler   = sampler_cfg,
        execution = AutoExecution(),
        save_dir  = joinpath(save_root, name)
    )
    
    t0 = time()
    fit = fit_model(fit_cfg, ds; quiet = false)
    elapsed = round(time() - t0, digits = 1)
    fits[name] = fit
    
    conv = fit.diagnostics
    ch1 = fit.folds[1].chain
    ch_vars = Set(Symbol.(names(ch1)))
    get_param(sym) = sym in ch_vars ? mean(ch1[sym]) : NaN
    
    mean_gamma  = Symbol("ha.γ") in ch_vars ? get_param(Symbol("ha.γ")) :
                  Symbol("ha.γ_global") in ch_vars ? get_param(Symbol("ha.γ_global")) :
                  get_param(Symbol("ha.γ_raw"))
    mean_sa     = get_param(Symbol("dyn.σ_a"))
    mean_sd     = get_param(Symbol("dyn.σ_d"))
    mean_wealth = get_param(Symbol("wealth.w"))
    mean_dist   = get_param(Symbol("distance.w"))
    mean_prod_w = get_param(Symbol("production_wealth.w"))
    mean_log_r  = get_param(Symbol("disp.log_r"))
    mean_r      = !isnan(mean_log_r) ? exp(mean_log_r) : NaN
    
    # Out-of-sample prediction evaluation
    eval_rep = evaluate_predictions(fit, ds)
    ll = eval_rep.model.logloss
    brier = eval_rep.model.brier
    ece = eval_rep.model.ece
    rps = eval_rep.model.rps
    
    println("\n--- Evaluation on Fold 1 OOS: $name ---")
    @printf("  LogLoss: %.4f | Brier: %.4f | ECE: %.4f | RPS: %.4f\n", ll, brier, ece, rps)
    @printf("  Parameters: γ=%.3f | σ_a=%.3f | σ_d=%.3f | log(r)=%.3f (r=%.2f)\n",
            mean_gamma, mean_sa, mean_sd, mean_log_r, mean_r)
    if !isnan(mean_wealth) @printf("  w_raw=%.3f\n", mean_wealth) end
    if !isnan(mean_dist)   @printf("  w_dist=%.3f\n", mean_dist) end
    if !isnan(mean_prod_w) @printf("  w_prod=%.3f\n", mean_prod_w) end
    
    push!(summary_rows, (
        name        = name,
        elapsed     = elapsed,
        passed      = conv.passed,
        max_rhat    = conv.max_rhat,
        min_ess     = conv.min_ess_bulk,
        divs        = conv.n_divergent,
        gamma       = mean_gamma,
        r           = mean_r,
        w_raw       = mean_wealth,
        w_dist      = mean_dist,
        w_prod      = mean_prod_w,
        logloss     = ll,
        brier       = brier
    ))
end

# 5. Leaderboard Summary Table
println("\n" * "="^120)
println(" FOLD 1 NEGATIVE BINOMIAL SMOKE TEST LEADERBOARD")
println("="^120)
@printf(" %-28s | %7s | %5s | %5s | %4s | %7s | %7s | %7s | %7s | %7s | %7s\n",
        "Model", "Time", "R-hat", "ESS", "Div", "γ (HA)", "r (disp)", "w_raw", "w_prod", "w_dist", "LogLoss")
println("-"^120)

for r in summary_rows
    rhat_str   = isnan(r.max_rhat) ? "N/A" : @sprintf("%.3f", r.max_rhat)
    ess_str    = isnan(r.min_ess) ? "N/A" : @sprintf("%d", Int(round(r.min_ess)))
    div_str    = @sprintf("%d", r.divs)
    gamma_str  = isnan(r.gamma) ? "—" : @sprintf("%+.3f", r.gamma)
    r_str      = isnan(r.r) ? "—" : @sprintf("%.2f", r.r)
    wealth_str = isnan(r.w_raw) ? "—" : @sprintf("%+.3f", r.w_raw)
    prod_str   = isnan(r.w_prod) ? "—" : @sprintf("%+.3f", r.w_prod)
    dist_str   = isnan(r.w_dist) ? "—" : @sprintf("%+.3f", r.w_dist)
    ll_str     = isnan(r.logloss) ? "—" : @sprintf("%.4f", r.logloss)

    @printf(" %-28s | %6.1fs | %5s | %5s | %4s | %7s | %7s | %7s | %7s | %7s | %7s\n",
            r.name, r.elapsed, rhat_str, ess_str, div_str, gamma_str, r_str, wealth_str, prod_str, dist_str, ll_str)
end
println("="^120)
println(" Smoke test completed successfully!")
