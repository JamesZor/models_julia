# ==============================================================================
# Scottish Lower — Negative Binomial 5-Model Walk-Forward Grid (Seasons 24/25 + 25/26)
# BayesianFootball.jl Unified V2 Stack
# ==============================================================================
#
# Models:
#   1. m00_negbin_baseline          : Pure NegBin + Time Decay + Home Advantage + Global Dispersion
#   2. m02_negbin_wealth            : m00 + Raw Starting-XI Squad Wealth
#   3. m03_negbin_distance          : m00 + Away Ground Travel Distance
#   4. m04_negbin_joint             : m00 + Raw Wealth + Travel Distance
#   5. m05_negbin_production_wealth : m00 + Age-Adjusted Production Wealth (Richards Sigmoid)
#
# Scope: Scottish League One (56) + League Two (57) pooled across 24/25 and 25/26 (40 folds)
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, ThreadPinning, LinearAlgebra, Printf, Statistics, Distributions, MCMCChains

# 1. Thread Pinning & BLAS Isolation
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

println("\n" * "="^100)
println(" SCOTTISH LOWER NEGATIVE BINOMIAL 5-MODEL GRID (24/25 + 25/26 SEASONS)")
println("="^100)

# 2. Data Loading (Cached DataStore with DOB Timestamps)
println("\n[1/4] Loading Scottish Lower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
println("  Matches: $(nrow(ds.matches)) | Lineups: $(nrow(ds.lineups)) | Odds: $(nrow(ds.odds))")

# 3. Model Definitions via Composable Count Builder
println("\n[2/4] Assembling 5 Composable Negative Binomial Models...")

# Model 00: Baseline Pure NegBin
m00 = CountModelBuilder(:m00_negbin_baseline) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

# Model 02: + Squad Wealth (Raw Transfermarkt LogSum differential)
m02 = CountModelBuilder(:m02_negbin_wealth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(WealthCovariate(prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

# Model 03: + Travel Distance (Away ground fatigue)
m03 = CountModelBuilder(:m03_negbin_distance) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(DistanceCovariate(prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

# Model 04: + Joint Squad Wealth & Travel Distance
m04 = CountModelBuilder(:m04_negbin_joint) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(WealthCovariate(prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
    add(DistanceCovariate(prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
    add(NegativeBinomialObservation(GlobalDispersion())) |>
    build

# Model 05: + Age-Adjusted Production Wealth (Richards Generalized Sigmoid)
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

# 4. Multi-Season Walk-Forward Splitter (24/25 + 25/26)
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25", "25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

# Production Sampler Config: 800 warmup + 800 samples x 4 chains with bounded task queue
sampler_cfg = QueuedNUTSConfig(
    n_samples   = 800,
    n_warmup    = 800,
    n_chains    = 4,
    accept_rate = 0.65
)

save_root = "/root/BayesianFootball/experiments/scottish_lower_2426_negbin"
mkpath(save_root)

# 5. Execute 5-Model Walk-Forward Inference
println("\n[3/4] Running 5-Model Multi-Season Grid...")
fits = Dict{String, Fit}()
train_times = Dict{String, Float64}()

for (name, model) in models
    println("\n" * "="^80)
    println(" TRAINING MODEL: $name")
    println("="^80)

    fit_cfg = FitConfig(
        name      = name,
        model     = model,
        splitter  = splitter,
        sampler   = sampler_cfg,
        execution = QueuedExecution(),
        save_dir  = joinpath(save_root, name)
    )

    t0 = time()
    fit = fit_model(fit_cfg, ds; quiet = false)
    elapsed = round(time() - t0, digits = 1)
    train_times[name] = elapsed
    fits[name] = fit
    
    save_fit(fit, joinpath(save_root, name))
    println(" Saved fit artifacts to $(joinpath(save_root, name)) in $(elapsed)s")
end

# 6. Comprehensive Metrics Evaluation
println("\n[4/4] Evaluating Predictions Across All 5 Negative Binomial Models...")

results_summary = []

for (name, _) in models
    fit = fits[name]
    rep = evaluate_predictions(fit, ds)
    conv = fit.diagnostics
    
    # Aggregate parameter posteriors across all folds
    last_chain = fit.folds[end].chain
    ch_vars = Set(Symbol.(names(last_chain)))
    get_param(sym) = sym in ch_vars ? mean(last_chain[sym]) : NaN

    gamma_val   = Symbol("ha.γ") in ch_vars ? get_param(Symbol("ha.γ")) :
                  Symbol("ha.γ_global") in ch_vars ? get_param(Symbol("ha.γ_global")) :
                  get_param(Symbol("ha.γ_raw"))
    sigma_att   = get_param(Symbol("dyn.σ_a"))
    sigma_def   = get_param(Symbol("dyn.σ_d"))
    wealth_w    = get_param(Symbol("wealth.w"))
    dist_w      = get_param(Symbol("distance.w"))
    prod_w      = get_param(Symbol("production_wealth.w"))
    log_r_val   = get_param(Symbol("disp.log_r"))
    r_disp      = !isnan(log_r_val) ? exp(log_r_val) : NaN

    push!(results_summary, (
        name        = name,
        train_time  = train_times[name],
        passed      = conv.passed,
        max_rhat    = conv.max_rhat,
        min_ess     = conv.min_ess_bulk,
        divs        = conv.n_divergent,
        gamma       = gamma_val,
        sigma_att   = sigma_att,
        sigma_def   = sigma_def,
        r_disp      = r_disp,
        wealth_w    = wealth_w,
        dist_w      = dist_w,
        prod_w      = prod_w,
        logloss     = rep.model.logloss,
        brier       = rep.model.brier,
        ece         = rep.model.ece,
        rps         = rep.model.rps
    ))
end

# 7. Print Comparative Leaderboard
println("\n" * "="^135)
println(" SCOTTISH LOWER 24/25 + 25/26 NEGATIVE BINOMIAL 5-MODEL BENCHMARK LEADERBOARD")
println("="^135)
@printf(" %-28s | %7s | %5s | %5s | %4s | %7s | %7s | %7s | %7s | %7s | %7s | %7s\n",
        "Model", "Time", "R-hat", "ESS", "Div", "γ (HA)", "r (disp)", "w_raw", "w_prod", "w_dist", "LogLoss", "Brier")
println("-"^135)

for r in results_summary
    rhat_str   = isnan(r.max_rhat) ? "N/A" : @sprintf("%.3f", r.max_rhat)
    ess_str    = isnan(r.min_ess) ? "N/A" : @sprintf("%d", Int(round(r.min_ess)))
    div_str    = @sprintf("%d", r.divs)
    gamma_str  = isnan(r.gamma) ? "—" : @sprintf("%+.3f", r.gamma)
    r_str      = isnan(r.r_disp) ? "—" : @sprintf("%.2f", r.r_disp)
    wealth_str = isnan(r.wealth_w) ? "—" : @sprintf("%+.3f", r.wealth_w)
    prod_str   = isnan(r.prod_w) ? "—" : @sprintf("%+.3f", r.prod_w)
    dist_str   = isnan(r.dist_w) ? "—" : @sprintf("%+.3f", r.dist_w)

    @printf(" %-28s | %6.1fs | %5s | %5s | %4s | %7s | %7s | %7s | %7s | %7s | %7.4f | %7.4f\n",
            r.name, r.train_time, rhat_str, ess_str, div_str, gamma_str, r_str, wealth_str, prod_str, dist_str, r.logloss, r.brier)
end
println("="^135)
