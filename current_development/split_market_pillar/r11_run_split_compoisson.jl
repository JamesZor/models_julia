#=
RUNNER for l06_split_market_compoisson.jl — SplitMarketCMPModel.

Single-split diagnostic. The COM-Poisson sibling of r02/r08/r09: same split (level/supremacy) pillar,
but goals ~ CMP(θ, ν) with the dispersion ν FREE. Unlike NegBin (l04), CMP can go sub-Poisson, which
is what Ireland-79 actually is (V/M 0.94). The CMP mean m(θ,ν) is anchored to the market/xG (not the
raw rate θ), so ν is free to capture dispersion instead of being forced to 1.

Goals:
  1. CMP HELPER SANITY (runs first, no training) — pmf sums to 1, ν>1 ⇒ var<mean (sub-Poisson).
  2. CONVERGES — R-hat ≤ ~1.05 incl. σ_sup, σ_lev, ω.
  3. ν (=ω) > 1 ⇒ the model found sub-Poisson structure; ν≈1 ⇒ ≈ Poisson.

Run after: git push -> git pull (server) -> RESTART REPL, then:
    include("current_development/split_market_pillar/r11_run_split_compoisson.jl")
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

include("current_development/split_market_pillar/l06_split_market_compoisson.jl")

# ==========================================
# 0. CMP HELPER SANITY (no training) — confirm the distribution behaves
# ==========================================
println("\n--- CMP helper sanity (pmf normalisation + dispersion direction) ---")
let
    p = zeros(Float64, 40)
    for (θ, ν) in [(1.4, 0.7), (1.4, 1.0), (1.4, 1.3)]
        _cmp_pmf!(p, θ, ν, 40)
        js = 0:39
        mass = sum(p)
        mean = sum(js .* p) / mass
        var  = sum((js .- mean).^2 .* p) / mass
        tag  = var < mean - 1e-6 ? "under" : var > mean + 1e-6 ? "over" : "poisson"
        println("  θ=$θ ν=$ν :  Σpmf=$(round(mass,digits=5))  mean=$(round(mean,digits=3))  " *
                "var=$(round(var,digits=3))  V/M=$(round(var/mean,digits=3))  ($tag)")
    end
    # mean-route check: _cmp_logZ_mean mean should match the pmf mean
    lz, m = _cmp_logZ_mean([log(1.4)], 1.3)
    println("  _cmp_logZ_mean mean (θ=1.4,ν=1.3) = $(round(m[1],digits=3))  (should match pmf mean above)")
end
println("Expect: Σpmf≈1.0, ν=1.0 V/M≈1.0, ν=1.3 V/M<1 (sub-Poisson). If so, the helper is correct.\n")

# ==========================================
# 1. DATA — Betfair market pillar (Ireland 79; sub-Poisson, V/M 0.94)
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
disp_cfg  = PreGame.HomeAwayDispersion()      # config-compat only (CMP dispersion is ν)
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

# ==========================================
# 3. THE SPLIT-MARKET COM-POISSON MODEL
# ==========================================
model = SplitMarketCMPModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = 1.0,    # both marginals anchored (totals rate + supremacy)
)

task = Experiments.create_experiment_task(
    ds_market, model, "split_cmp_r1", save_dir;
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

results = Experiments.run_experiment(task)
Experiments.save_experiment(results)

# ==========================================
# 4. CONVERGENCE + DISPERSION ν DIAGNOSTICS
# ==========================================
chains_obj = Experiments.Diagnostics.extract_chains(ds_market, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv = Experiments.Diagnostics.check_convergence(chains_obj)
display(conv.df)

chain = results.training_results.items[1][1]
println("\n--- CMP dispersion ν (=ω) ---")
if :ω in keys(chain)
    v = vec(Array(chain[:ω]))
    tag = mean(v) > 1.03 ? "SUB-POISSON (found it)" : mean(v) < 0.97 ? "over-dispersed" : "≈ Poisson"
    println("  ω: mean=$(round(mean(v),digits=4))  std=$(round(std(v),digits=4))  " *
            "q=[$(round(quantile(v,0.05),digits=4)), $(round(quantile(v,0.95),digits=4))]  -> $tag")
    println("  (Ireland-79 is V/M 0.94 ⇒ expect ω just above 1; the BTTS sharpening, if any, lives here)")
else
    println("  (ω not in chain — check raw_symbol names in conv.df)")
end
for p in (:σ_sup, :σ_lev, :ν_xg)
    p in keys(chain) && println("  $p: mean=$(round(mean(vec(Array(chain[p]))),digits=4))")
end
println("\nNext: full-CV + r06_per_line_eval.jl (point include at l06) — read BTTS first (the P(0) lever).")
