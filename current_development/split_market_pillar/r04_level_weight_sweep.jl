#=
LEVEL-WEIGHT SWEEP for SplitMarketDoublePoissonModel.

Question (from the κ-artifact finding): the supremacy-only config (level_weight=0) blows the
team-strength kappa spread up to σ_κ≈0.30, while the market-off model shows κ uniform (σ_κ≈0.07)
— so that spread is a CONSTRAINT ARTIFACT (κ absorbing the supremacy-vs-xG tension when the level
is left free), not real finishing signal. Adding level weight gives κ a 2nd constraint and
collapses it back.

This sweeps level_weight ∈ {0.0, 0.25, 0.5, 1.0} (market_on=true, supremacy_weight=1.0 throughout)
to find the MINIMAL level weight that tames κ while still letting the model mostly own totals.
Watch σ_κ / kappa_std fall as level_weight rises; σ_lev should sharpen (model agrees with market
on totals) as the level anchor bites.

4 variants × 4 chains = 16 chains == your 16 pinned cores (one full parallel wave).

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r04_level_weight_sweep.jl")
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
# 2. SHARED COMPONENT CONFIG
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

_make_model(level_weight) = SplitMarketDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = level_weight,
)

_build_task(model, name) = Experiments.create_experiment_task(
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

# ==========================================
# 3. SWEEP — run all level_weights in parallel
# ==========================================
level_weights = [0.0, 0.25, 0.5, 1.0]
variants = [("lw$(replace(string(lw), "." => ""))", lw) for lw in level_weights]

println("\n>> Sweeping level_weight $(level_weights) in parallel ($(length(variants))×4 = $(length(variants)*4) chains, $(Threads.nthreads()) threads)...")
raw_results = Dict{String, Any}()
rlock = ReentrantLock()
@sync for (name, lw) in variants
    Threads.@spawn begin
        res = Experiments.run_experiment(_build_task(_make_model(lw), name))
        Experiments.save_experiment(res; quiet=true)
        lock(rlock) do
            raw_results[name] = res
        end
    end
end

# ==========================================
# 4. DIAGNOSE + COMPARE κ response to level_weight
# ==========================================
function _summarise(conv_df)
    max_rhat  = maximum(filter(!isnan, conv_df.rhat))
    κ_teams   = conv_df[conv_df.parameter .== "kappa", :mean]
    σκ_row    = conv_df[conv_df.raw_symbol .== Symbol("kap.σ_κ"), :mean]
    σ_sup_row = conv_df[conv_df.raw_symbol .== :σ_sup, :mean]
    σ_lev_row = conv_df[conv_df.raw_symbol .== :σ_lev, :mean]
    return (
        max_rhat    = round(max_rhat, digits=3),
        kappa_min   = round(minimum(κ_teams), digits=3),
        kappa_max   = round(maximum(κ_teams), digits=3),
        kappa_std   = round(std(κ_teams), digits=3),
        sigma_kappa = isempty(σκ_row)    ? NaN : round(σκ_row[1], digits=3),
        sigma_sup   = isempty(σ_sup_row) ? NaN : round(σ_sup_row[1], digits=3),
        sigma_lev   = isempty(σ_lev_row) ? NaN : round(σ_lev_row[1], digits=3),
    )
end

summary = DataFrame(level_weight=Float64[], max_rhat=Float64[], kappa_min=Float64[],
                    kappa_max=Float64[], kappa_std=Float64[], sigma_kappa=Float64[],
                    sigma_sup=Float64[], sigma_lev=Float64[])
for (name, lw) in variants
    println("\n", "="^70, "\n>> level_weight = $lw\n", "="^70)
    chains = Experiments.Diagnostics.extract_chains(ds_market, raw_results[name])
    println(Experiments.Diagnostics.check_convergence(chains))
    s = _summarise(chains.df)
    push!(summary, (lw, s.max_rhat, s.kappa_min, s.kappa_max, s.kappa_std,
                    s.sigma_kappa, s.sigma_sup, s.sigma_lev))
end

println("\n", "="^70, "\nLEVEL-WEIGHT SWEEP (single split)\n", "="^70)
println("Expect kappa_std / sigma_kappa to FALL as level_weight rises (artifact tamed),")
println("and sigma_lev to sharpen (model pulled onto the market's totals view).")
show(sort(summary, :level_weight), allcols=true, allrows=true)
println()
