#=
CONTROL RUNNER for SplitMarketDoublePoissonModel — convergence + kappa comparison.

Motivation: with the supremacy anchor on (level_on=false) the team kappas spread WAY out
(σ_κ 0.064 -> 0.30, κ range 0.69–1.39) vs the standard isotropic model. Is that κ spread real
structure or small-sample noise? Run the three variants on ONE split, confirm each CONVERGES,
and compare how much kappa differentiates across them:

  A. supremacy-only  (market_on=true,  level_on=false)  — the headline split model
  B. market-OFF      (market_on=false, level_on=false)  — no anchor at all (model fully free)
  C. supremacy+level (market_on=true,  level_on=true )   — both axes anchored (≈ isotropic)

Expectation: κ spread should order C (most anchored) < A < B (least anchored). If A sits between
C and B it's structured; if A ≈ B the supremacy anchor isn't constraining kappa at all.

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r03_split_controls.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

include("l02_split_market_poisson.jl")

# ==========================================
# 1. DATA — Betfair market pillar
# ==========================================
ds = Data.load_datastore_cached(Data.Ireland())
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds_market = Data.DataStore(
    ds.segment, ds.matches, ds.statistics, odds,
    ds.lineups, ds.incidents, ds.betfair_odds
)

save_dir = "./data/split_market_dev_area/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIG (matches r00/r02)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

_make_model(; market_on, level_weight, supremacy_weight=1.0) = SplitMarketDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = market_on,
    supremacy_weight       = supremacy_weight,
    level_weight           = level_weight,
)

# ==========================================
# 3. RUN ALL VARIANTS IN PARALLEL, THEN DIAGNOSE
# ==========================================
# Each experiment's queue holds only 1 split x 4 chains = 4 work items, so on its own it uses
# 4 of the 16 pinned cores. Running the 3 variants concurrently fills 12 cores (<= 16, no
# oversubscription). The queued trainer uses per-call locks/semaphore, so concurrent
# run_experiment is safe — only the console output (3 progress meters) interleaves; ignore the
# garbled progress bars. All 3 share identical required_features, so this is just 12 chains.
function _build_task(model, name)
    Experiments.create_experiment_task(
        ds_market, model, name, save_dir;
        target_seasons  = ["2026"],
        history_seasons = 2,
        warmup_period   = 21,
        dynamics_col    = :match_week,
        samples         = 1000,
        warmup          = 500,
        chains          = 4,
        use_queue       = true,
        max_depth       = 10,
    )
end

variants = [
    ("A_supremacy_only", _make_model(market_on=true,  level_weight=0.0)),
    ("B_market_off",     _make_model(market_on=false, level_weight=0.0)),
    ("C_supremacy_level",_make_model(market_on=true,  level_weight=1.0)),
]

# --- Phase 1: run all 3 concurrently (12 chains across 16 pinned cores) ---
println("\n>> Launching $(length(variants)) variants in parallel ($(length(variants))×4 = $(length(variants)*4) chains, $(Threads.nthreads()) threads)...")
raw_results = Dict{String, Any}()
rlock = ReentrantLock()
@sync for (name, model) in variants
    Threads.@spawn begin
        res = Experiments.run_experiment(_build_task(model, name))
        Experiments.save_experiment(res; quiet=true)
        lock(rlock) do
            raw_results[name] = res
        end
    end
end

# --- Phase 2: diagnose each sequentially (clean output; extract_chains rebuilds features) ---
runs = Dict{String, NamedTuple}()
for (name, _) in variants
    println("\n", "="^70, "\n>> VARIANT: $name\n", "="^70)
    chains = Experiments.Diagnostics.extract_chains(ds_market, raw_results[name])
    println(Experiments.Diagnostics.check_convergence(chains))   # ✅/⚠️ banner
    runs[name] = (; conv_df = chains.df)
end

# ==========================================
# 4. COMPARISON — convergence + kappa differentiation
# ==========================================
function _summarise(conv_df)
    max_rhat  = maximum(filter(!isnan, conv_df.rhat))
    κ_teams   = conv_df[conv_df.parameter .== "kappa", :mean]          # per-team κ posterior means
    σκ_row    = conv_df[conv_df.raw_symbol .== Symbol("kap.σ_κ"), :mean]
    σ_sup_row = conv_df[conv_df.raw_symbol .== :σ_sup, :mean]
    σ_lev_row = conv_df[conv_df.raw_symbol .== :σ_lev, :mean]
    return (
        max_rhat   = round(max_rhat, digits=3),
        kappa_min  = round(minimum(κ_teams), digits=3),
        kappa_max  = round(maximum(κ_teams), digits=3),
        kappa_std  = round(std(κ_teams), digits=3),     # spread of team κ means
        sigma_kappa= isempty(σκ_row)    ? NaN : round(σκ_row[1], digits=3),
        sigma_sup  = isempty(σ_sup_row) ? NaN : round(σ_sup_row[1], digits=3),
        sigma_lev  = isempty(σ_lev_row) ? NaN : round(σ_lev_row[1], digits=3),
    )
end

summary = DataFrame(variant=String[], max_rhat=Float64[], kappa_min=Float64[],
                    kappa_max=Float64[], kappa_std=Float64[], sigma_kappa=Float64[],
                    sigma_sup=Float64[], sigma_lev=Float64[])
for (name, _) in variants
    s = _summarise(runs[name].conv_df)
    push!(summary, (name, s.max_rhat, s.kappa_min, s.kappa_max, s.kappa_std,
                    s.sigma_kappa, s.sigma_sup, s.sigma_lev))
end

println("\n", "="^70, "\nCONTROL COMPARISON (single split)\n", "="^70)
println("Convergence OK if max_rhat <= 1.05 for every variant.")
println("kappa_std / sigma_kappa = how much team finishing (goals-vs-xG) differentiates.")
show(summary, allcols=true, allrows=true)
println()
