# ==============================================================================
# Scottish Lower — Full 24/25 Season Walk-Forward Grid (All 4 Models)
# BayesianFootball.jl Graduated Composable Stack (05 -> 09)
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, ThreadPinning, LinearAlgebra, Printf, Statistics, Distributions, MCMCChains

# 1. Thread topology & BLAS isolation
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

println("\n" * "="^90)
println(" SCOTTISH LOWER UNIFIED 24/25 SEASON GRID (All 4 Models)")
println("="^90)

# 2. Data Loading
println("\n[1/4] Loading Scottish Lower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

# 3. Model Definitions
println("\n[2/4] Assembling 4 Composable Count Models...")

# Model 00: Baseline Pure Poisson
m00 = CountModelBuilder(:m00_baseline_poisson) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

# Model 02: + Squad Wealth
m02 = CountModelBuilder(:m02_poisson_wealth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(WealthCovariate(prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
    add(PoissonObservation()) |>
    build

# Model 03: + Travel Distance
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

# 4. Walk-Forward Splitter (24/25 Season, 2 History Seasons)
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

boundaries = Data.create_id_boundaries(ds, splitter)
println("  Total walk-forward folds in 24/25: $(length(boundaries))")

# Sampler settings: 800 warmup + 800 samples per chain, 4 chains
sampler_cfg = QueuedNUTSConfig(
    n_samples   = 800,
    n_warmup    = 800,
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

save_root = "./data/scottish_lower_unified"
mkpath(save_root)

# 5. Execute Walk-Forward Grid Across All Models
println("\n[3/4] Running Full Walk-Forward Grid Across 4 Models...")

fits = Dict{String, Fit}()
port_results = Dict{String, Any}()
eval_results = Dict{String, Any}()
bench_summary = []

for (name, m) in models
    println("\n" * "="^70)
    println(" TRAINING FULL GRID: $name")
    println("="^70)
    
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
    
    # Save fit to disk
    save_path = save_fit(fit)
    println(" Saved Fit to: $save_path")
    
    # Convergence summary
    conv = fit.diagnostics
    verdict = conv.passed
    
    # Parameter posteriors from Fold 1
    ch1 = fit.folds[1].chain
    ch_vars = Set(Symbol.(names(ch1)))
    get_param(sym) = sym in ch_vars ? mean(ch1[sym]) : NaN
    
    mean_mu     = get_param(Symbol("inter.μ_base[1]"))
    mean_gamma  = get_param(Symbol("ha.γ"))
    if isnan(mean_gamma)
        mean_gamma = get_param(Symbol("ha.γ_global"))
    end
    if isnan(mean_gamma)
        mean_gamma = get_param(Symbol("ha.γ_raw"))
    end
    mean_sa     = get_param(Symbol("dyn.σ_a"))
    mean_sd     = get_param(Symbol("dyn.σ_d"))
    mean_wealth = get_param(Symbol("wealth.w"))
    mean_dist   = get_param(Symbol("distance.w"))
    
    # Evaluate model predictions across the season
    eval_rep = evaluate_predictions(fit, ds)
    eval_results[name] = eval_rep
    ll_model    = eval_rep.model.logloss
    brier_model = eval_rep.model.brier
    ece_model   = eval_rep.model.ece
    rps_model   = eval_rep.model.rps
    
    println("\n--- Evaluation: $name ---")
    @printf("  LogLoss: %.4f | Brier: %.4f | ECE: %.4f | RPS: %.4f\n",
            ll_model, brier_model, ece_model, rps_model)
    
    # Simulate portfolio across the season
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
        logloss     = ll_model,
        brier       = brier_model,
        ece         = ece_model,
        rps         = rps_model,
        n_bets      = p_sum.n_bets,
        pnl         = p_sum.final_bankroll - 1.0,
        roi         = p_sum.roi,
        mdd         = p_sum.mdd,
        sharpe      = p_sum.sharpe
    ))
end

# 6. Formatted Comparison Table
println("\n" * "="^135)
println(" 24/25 SEASON FULL BENCHMARK LEADERBOARD (4 Models, Walk-Forward Grid)")
println("="^135)
@printf(" %-15s | %7s | %5s | %5s | %4s | %7s | %8s | %8s | %7s | %6s | %6s | %5d | %8s | %6s | %6s\n",
        "Model", "Time", "R-hat", "ESS", "Div", "γ (HA)", "w_wealth", "w_dist", "LogLoss", "Brier", "ECE", "Bets", "PnL", "ROI", "MDD")
println("-"^135)

for r in bench_summary
    rhat_str   = isnan(r.max_rhat) ? "N/A" : @sprintf("%.3f", r.max_rhat)
    ess_str    = isnan(r.min_ess) ? "N/A" : @sprintf("%d", Int(round(r.min_ess)))
    gamma_str  = isnan(r.gamma) ? "—" : @sprintf("%+.3f", r.gamma)
    wealth_str = isnan(r.w_wealth) ? "—" : @sprintf("%+.3f", r.w_wealth)
    dist_str   = isnan(r.w_dist) ? "—" : @sprintf("%+.3f", r.w_dist)
    ll_str     = isnan(r.logloss) ? "—" : @sprintf("%.4f", r.logloss)
    brier_str  = isnan(r.brier) ? "—" : @sprintf("%.4f", r.brier)
    ece_str    = isnan(r.ece) ? "—" : @sprintf("%.3f", r.ece)
    pnl_str    = @sprintf("%+.2f%%", 100 * r.pnl)
    roi_str    = @sprintf("%+.2f%%", r.roi)
    mdd_str    = @sprintf("%.2f%%", r.mdd)

    @printf(" %-15s | %6.1fs | %5s | %5s | %4d | %7s | %8s | %8s | %7s | %6s | %6s | %5d | %8s | %6s | %6s\n",
            r.name, r.elapsed, rhat_str, ess_str, r.divs, gamma_str, wealth_str, dist_str,
            ll_str, brier_str, ece_str, r.n_bets, pnl_str, roi_str, mdd_str)
end
println("="^135)
println(" All 4 models trained, evaluated, and simulated successfully!")
