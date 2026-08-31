# ==============================================================================
# Scottish Lower — 5-Model Walk-Forward Grid (Seasons 24/25 + 25/26)
# BayesianFootball.jl Unified V2 Stack
# ==============================================================================
#
# Models:
#   1. m00_baseline          : Pure Poisson + Time Decay + Home Advantage
#   2. m02_wealth            : m00 + Raw Starting-XI Squad Wealth
#   3. m03_distance          : m00 + Away Ground Travel Distance
#   4. m04_joint             : m00 + Raw Wealth + Travel Distance
#   5. m05_production_wealth : m00 + Age-Adjusted Production Wealth (Richards Sigmoid)
#
# Scope: Scottish League One (56) + League Two (57) pooled across 24/25 and 25/26
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, ThreadPinning, LinearAlgebra, Printf, Statistics, Distributions, MCMCChains

# 1. Thread Pinning & BLAS Isolation
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

println("\n" * "="^100)
println(" SCOTTISH LOWER UNIFIED 5-MODEL GRID (24/25 + 25/26 SEASONS)")
println("="^100)

# 2. Data Loading (Fresh Cached DataStore with DOB Timestamps)
println("\n[1/4] Loading Scottish Lower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
println("  Matches: $(nrow(ds.matches)) | Lineups: $(nrow(ds.lineups)) | Odds: $(nrow(ds.odds))")

# 3. Model Definitions via Composable Count Builder (05)
println("\n[2/4] Assembling 5 Composable Count Models...")

# Model 00: Baseline Pure Poisson
m00 = CountModelBuilder(:m00_baseline_poisson) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

# Model 02: + Squad Wealth (Raw Transfermarkt LogSum differential)
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

# Model 05: + Age-Adjusted Production Wealth (Richards Generalized Sigmoid)
m05 = CountModelBuilder(:m05_poisson_production_wealth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(ProductionWealthCovariate(
        feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
        prior   = truncated(Normal(0.10, 0.05), lower = 0.0)
    )) |>
    add(PoissonObservation()) |>
    build

models = [
    ("m00_baseline",          m00),
    ("m02_wealth",            m02),
    ("m03_distance",          m03),
    ("m04_joint",             m04),
    ("m05_production_wealth", m05),
]

# 4. Split and Sampler Configuration (Protocol Contract Alignment)
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25", "25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

boundaries = Data.create_id_boundaries(ds, splitter)
println("  Total walk-forward folds (24/25 + 25/26): $(length(boundaries))")

# Sampler settings: 800 warmup + 800 samples per chain, 4 chains
sampler_cfg = QueuedNUTSConfig(
    n_samples   = 800,
    n_warmup    = 800,
    n_chains    = 4,
    accept_rate = 0.65
)

# Portfolio & Book Specs (Aligned with _protocol/config.jl)
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
    trust    = FlatTrust(0.30),
    risk     = SlateDrawdown(23.0),
    cap      = FixedCap(0.20),
    grouping = DailySlate()
)

save_root = "./data/scottish_lower_2426_grid"
mkpath(save_root)

# 5. Execute Walk-Forward Grid Across All 5 Models
println("\n[3/4] Running 5-Model Walk-Forward Grid on 24/25 + 25/26...")

fits = Dict{String, Fit}()
port_results = Dict{String, Any}()
bench_summary = []

for (name, m) in models
    println("\n" * "="^80)
    println(" TRAINING GRID: $name (Seasons 24/25 + 25/26)")
    println("="^80)
    
    fit_cfg = FitConfig(
        name      = name,
        model     = m,
        splitter  = splitter,
        sampler   = sampler_cfg,
        execution = QueuedExecution(max_concurrent_tasks = Threads.nthreads()),
        save_dir  = joinpath(save_root, name)
    )
    
    t0 = time()
    fit = fit_model(fit_cfg, ds; quiet = false)
    elapsed = round(time() - t0, digits = 1)
    
    fits[name] = fit
    
    # Save fit to disk atomically
    save_path = save_fit(fit)
    println(" Saved Fit to: $save_path")
    
    conv = fit.diagnostics
    verdict = conv.passed
    
    # Parameter posteriors from Fold 1
    ch1 = fit.folds[1].chain
    ch_vars = Set(Symbol.(names(ch1)))
    get_param(sym) = sym in ch_vars ? mean(ch1[sym]) : NaN
    
    mean_mu     = get_param(Symbol("inter.μ_base[1]"))
    mean_gamma  = Symbol("ha.γ") in ch_vars ? get_param(Symbol("ha.γ")) :
                  Symbol("ha.γ_global") in ch_vars ? get_param(Symbol("ha.γ_global")) :
                  get_param(Symbol("ha.γ_raw"))
    mean_sa     = get_param(Symbol("dyn.σ_a"))
    mean_sd     = get_param(Symbol("dyn.σ_d"))
    mean_wealth = get_param(Symbol("wealth.w"))
    mean_dist   = get_param(Symbol("distance.w"))
    mean_prod_w = get_param(Symbol("production_wealth.w"))
    
    # Out-of-Sample Predictions Evaluation
    eval_rep = evaluate_predictions(fit, ds)
    ll_model    = eval_rep.model.logloss
    brier_model = eval_rep.model.brier
    ece_model   = eval_rep.model.ece
    rps_model   = eval_rep.model.rps
    
    println("\n--- Evaluation: $name ---")
    @printf("  LogLoss: %.4f | Brier: %.4f | ECE: %.4f | RPS: %.4f (Market LL: %.4f)\n",
            ll_model, brier_model, ece_model, rps_model, eval_rep.market.logloss)
    
    # Simulate Portfolio & Kelly Staking
    res, books, _ = run_portfolio_simulation(book_spec, policy_spec, fit, ds.odds, ds;
                                             bootstrap = true, require_converged = false)
    port_results[name] = res
    p_sum = res.summary
    
    println("\n--- Portfolio Summary: $name ---")
    display(portfolio_report(res))
    
    push!(bench_summary, (
        name        = name,
        elapsed     = elapsed,
        passed      = verdict,
        max_rhat    = conv.max_rhat,
        min_ess     = conv.min_ess_bulk,
        divs        = conv.n_divergent,
        gamma       = mean_gamma,
        w_wealth    = mean_wealth,
        w_dist      = mean_dist,
        w_prod      = mean_prod_w,
        logloss     = ll_model,
        brier       = brier_model,
        ece         = ece_model,
        rps         = rps_model,
        n_bets      = p_sum.n_bets,
        pnl         = p_sum.final_bankroll - 1.0,
        roi         = p_sum.roi,
        mdd         = p_sum.mdd,
        sharpe      = p_sum.sharpe_ann
    ))
end

# 6. Formatted Final Leaderboard
println("\n" * "="^140)
println(" 24/25 + 25/26 SEASONS FULL BENCHMARK LEADERBOARD (5 Models)")
println("="^140)
@printf(" %-22s | %7s | %5s | %5s | %4s | %7s | %7s | %7s | %7s | %7s | %6s | %5s | %8s | %6s | %6s\n",
        "Model", "Time", "R-hat", "ESS", "Div", "γ (HA)", "w_raw", "w_prod", "w_dist", "LogLoss", "Brier", "Bets", "PnL", "ROI", "MDD")
println("-"^140)

for r in bench_summary
    rhat_str   = isnan(r.max_rhat) ? "N/A" : @sprintf("%.3f", r.max_rhat)
    ess_str    = isnan(r.min_ess) ? "N/A" : @sprintf("%d", Int(round(r.min_ess)))
    div_str    = @sprintf("%d", r.divs)
    gamma_str  = isnan(r.gamma) ? "—" : @sprintf("%+.3f", r.gamma)
    wealth_str = isnan(r.w_wealth) ? "—" : @sprintf("%+.3f", r.w_wealth)
    prod_str   = isnan(r.w_prod) ? "—" : @sprintf("%+.3f", r.w_prod)
    dist_str   = isnan(r.w_dist) ? "—" : @sprintf("%+.3f", r.w_dist)
    ll_str     = isnan(r.logloss) ? "—" : @sprintf("%.4f", r.logloss)
    brier_str  = isnan(r.brier) ? "—" : @sprintf("%.4f", r.brier)
    pnl_str    = @sprintf("%+.2f%%", 100 * r.pnl)
    roi_str    = @sprintf("%+.2f%%", r.roi)
    mdd_str    = @sprintf("%.2f%%", r.mdd)
    bets_str   = @sprintf("%d", r.n_bets)

    @printf(" %-22s | %6.1fs | %5s | %5s | %4s | %7s | %7s | %7s | %7s | %7s | %6s | %5s | %8s | %6s | %6s\n",
            r.name, r.elapsed, rhat_str, ess_str, div_str, gamma_str, wealth_str, prod_str, dist_str,
            ll_str, brier_str, bets_str, pnl_str, roi_str, mdd_str)
end
println("="^140)
println(" All 5 models trained, evaluated, and simulated across 24/25 + 25/26 successfully!")
