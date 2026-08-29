# ==============================================================================
# Scottish Lower — Unified 4-Model Smoke Test (Fold 1: 500 warmup + 500 samples)
# BayesianFootball.jl Graduated Composable Stack (05 -> 09)
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, ThreadPinning, LinearAlgebra, Printf, Statistics, Distributions, MCMCChains

# 1. Thread topology & BLAS isolation
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

println("\n" * "="^80)
println(" SCOTTISH LOWER UNIFIED SMOKE TEST (4 Models on Fold 1)")
println("="^80)

# 2. Data Loading
println("\n[1/4] Loading Scottish Lower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

# 3. Model Definitions via Composable Count Builder (05)
println("\n[2/4] Assembling 4 Composable Count Models...")

# Model 00: Baseline Pure Poisson
m00 = CountModelBuilder(:m00_baseline_poisson) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

# Model 02: + Squad Wealth (LogSum starting-XI differential)
m02 = CountModelBuilder(:m02_poisson_wealth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(WealthCovariate(prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
    add(PoissonObservation()) |>
    build

# Model 03: + Travel Distance (Away ground fatigue)
m03 = CountModelBuilder(:m03_poisson_distance) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(DistanceCovariate(prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
    add(PoissonObservation()) |>
    build

# Model 04: + Joint Squad Wealth & Travel Distance
m04 = CountModelBuilder(:m04_poisson_joint) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(WealthCovariate(prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
    add(DistanceCovariate(prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
    add(PoissonObservation()) |>
    build

models = [
    ("m00_baseline", m00),
    ("m02_wealth",   m02),
    ("m03_distance", m03),
    ("m04_joint",    m04),
]

# 4. Split and Sampling Configuration
# Use Fold 1 of 24/25 season (exact same split as r00_explore_poisson_models.jl)
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

boundaries = Data.create_id_boundaries(ds, splitter)
fold_1_boundary = [boundaries[1]]
println("  Using Fold 1 boundary (Target season $(fold_1_boundary[1][2].target_season), Step $(fold_1_boundary[1][2].time_step))")

# Sampler settings: 500 warmup, 500 samples, 4 chains
sampler_cfg = NUTSConfig(
    n_samples   = 500,
    n_warmup    = 500,
    n_chains    = 4,
    accept_rate = 0.65
)

# Portfolio & Book Specs (09)
book_spec = BookSpec(
    markets   = Data.MarketConfig([
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS()
    ]),
    price     = DeArb(),
    allocator = KellyLogUtility(),
    shrink    = BakerMcHale(),
    exec      = ExecutionConfig(
        commission          = PerBetCommission(0.02),
        budget              = 0.99,
        min_selection_stake = 0.001
    )
)

policy_spec = PolicySpec(
    trust    = FlatTrust(0.25),
    risk     = SlateDrawdown(20.0),
    cap      = FixedCap(0.25),
    grouping = DailySlate()
)

# 5. Execute 4 Models End-to-End
println("\n[3/4] Running 4-Model Inference, Evaluation & Portfolio Simulation on Fold 1...")

results_summary = []

for (name, m) in models
    println("\n" * "-"^60)
    println(" Fitting Model: $name")
    println("-"^60)
    
    t0 = time()
    # Feature extraction for Fold 1
    fs = Features.create_features(fold_1_boundary, ds, m, splitter)
    oos_fx = Any[DataFrame(Data.get_next_matches(ds, fold_1_boundary[1], splitter))]
    
    fit_cfg = FitConfig(
        name      = name,
        model     = m,
        splitter  = splitter,
        sampler   = sampler_cfg,
        execution = SequentialExecution()
    )
    
    # Unified fit
    fit = fit_model(fit_cfg; feature_sets = fs, oos_fixtures = oos_fx, quiet = false)
    elapsed = round(time() - t0, digits = 1)
    
    # Convergence Audit
    conv = fit.diagnostics
    verdict = conv.passed
    ch = fit.folds[1].chain
    
    # Extract parameter posteriors
    ch_vars = Set(Symbol.(names(ch)))
    get_param(sym) = sym in ch_vars ? mean(ch[sym]) : NaN

    mean_mu     = get_param(Symbol("inter.μ_base[1]"))
    mean_gamma  = get_param(Symbol("ha.γ"))
    if isnan(mean_gamma)
        mean_gamma = get_param(Symbol("ha.γ_raw"))
    end
    mean_sa     = get_param(Symbol("dyn.σ_a"))
    mean_sd     = get_param(Symbol("dyn.σ_d"))
    mean_wealth = get_param(Symbol("wealth.w"))
    mean_dist   = get_param(Symbol("distance.w"))
    
    # Unified Evaluation (08)
    eval_rep = evaluate_predictions(fit, ds)
    ll_model = eval_rep.model.logloss
    brier_model = eval_rep.model.brier
    ece_model = eval_rep.model.ece
    rps_model = eval_rep.model.rps
    
    # Portfolio Simulation (09)
    port_res, books, _ = run_portfolio_simulation(book_spec, policy_spec, fit, ds.odds, ds;
                                                  bootstrap = false, require_converged = false)
    p_sum = portfolio_summary(port_res)
    
    push!(results_summary, (
        name        = name,
        elapsed     = elapsed,
        passed      = verdict,
        max_rhat    = conv.max_rhat,
        min_ess     = conv.min_ess_bulk,
        divs        = conv.n_divergent,
        gamma       = mean_gamma,
        w_wealth    = mean_wealth,
        w_dist      = mean_dist,
        logloss     = ll_model,
        brier       = brier_model,
        ece         = ece_model,
        rps         = rps_model,
        n_bets      = p_sum.n_bets,
        pnl         = p_sum.terminal_bankroll - 1.0
    ))
end

# 6. Formatted Comparison Table
println("\n" * "="^120)
println(" FOLD 1 BENCHMARK SUMMARY TABLE (500 warmup / 500 samples / 4 chains)")
println("="^120)
@printf(" %-15s | %6s | %5s | %5s | %5s | %7s | %8s | %8s | %7s | %6s | %6s | %5s | %7s\n",
        "Model", "Time", "R-hat", "ESS", "Div", "γ (HA)", "w_wealth", "w_dist", "LogLoss", "Brier", "ECE", "Bets", "PnL")
println("-"^120)

for r in results_summary
    rhat_str   = isnan(r.max_rhat) ? "N/A" : @sprintf("%.3f", r.max_rhat)
    ess_str    = isnan(r.min_ess) ? "N/A" : @sprintf("%d", Int(round(r.min_ess)))
    gamma_str  = isnan(r.gamma) ? "—" : @sprintf("%+.3f", r.gamma)
    wealth_str = isnan(r.w_wealth) ? "—" : @sprintf("%+.3f", r.w_wealth)
    dist_str   = isnan(r.w_dist) ? "—" : @sprintf("%+.3f", r.w_dist)
    ll_str     = isnan(r.logloss) ? "—" : @sprintf("%.4f", r.logloss)
    brier_str  = isnan(r.brier) ? "—" : @sprintf("%.4f", r.brier)
    ece_str    = isnan(r.ece) ? "—" : @sprintf("%.3f", r.ece)
    pnl_str    = @sprintf("%+.2f%%", 100 * r.pnl)

    @printf(" %-15s | %5.1fs | %5s | %5s | %5d | %7s | %8s | %8s | %7s | %6s | %6s | %5d | %7s\n",
            r.name, r.elapsed, rhat_str, ess_str, r.divs, gamma_str, wealth_str, dist_str,
            ll_str, brier_str, ece_str, r.n_bets, pnl_str)
end
println("="^120)
println(" Smoke test completed successfully!")
