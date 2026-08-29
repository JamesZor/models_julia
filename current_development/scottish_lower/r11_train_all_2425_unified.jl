# ==============================================================================
# Scottish Lower — Full 24/25 Season Walk-Forward Grid (All 4 Models)
# BayesianFootball.jl Graduated Composable Stack (05 -> 09)
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, ThreadPinning, LinearAlgebra, Printf, Statistics, Distributions, MCMCChains

# 1. Thread topology & BLAS isolation
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

println("\n" * "="^80)
println(" SCOTTISH LOWER UNIFIED 24/25 SEASON GRID (All 4 Models)")
println("="^80)

# 2. Data Loading
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

# 3. Model Definitions
m00 = CountModelBuilder(:m00_baseline_poisson) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

m02 = CountModelBuilder(:m02_poisson_wealth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(WealthCovariate(prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
    add(PoissonObservation()) |>
    build

m03 = CountModelBuilder(:m03_poisson_distance) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(GlobalHomeAdvantage()) |>
    add(DistanceCovariate(prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
    add(PoissonObservation()) |>
    build

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

# 4. Walk-Forward Splitter
splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

# 800 warmup + 800 samples per chain, 4 chains
sampler_cfg = QueuedNUTSConfig(
    n_samples     = 800,
    n_warmup      = 800,
    n_chains      = 4,
    target_accept = 0.65
)

# Portfolio & Book Specs
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
fits = Dict{String, Fit}()
port_results = Dict{String, Any}()

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
    elapsed = time() - t0
    
    fits[name] = fit
    
    # Save fit to disk
    save_path = save_fit(fit)
    println(" Saved Fit to: $save_path")
    
    # Evaluate model predictions
    eval_rep = evaluate_predictions(fit, ds)
    println("\n--- Evaluation Report: $name ---")
    println(eval_rep)
    
    # Simulate portfolio
    res, books, _ = run_portfolio_simulation(book_spec, policy_spec, fit, ds.odds, ds; bootstrap = true)
    port_results[name] = res
    println("\n--- Portfolio Summary: $name ---")
    display(portfolio_report(res))
end

println("\n" * "="^80)
println(" ALL 4 MODELS TRAINED AND SIMULATED SUCCESSFULLY!")
println("="^80)
