#=
RUNNER for l03_local_intensity_poisson.jl — LocalIntensitySmileDoublePoissonModel.

Single-split diagnostic. Goals:
  1. CONVERGENCE — the {goals + xG + supremacy + per-strike SMILE + outfield} model converges
     (R-hat ≤ ~1.05), incl. the new pillar scalars σ_sup, σ_smile.
  2. SMILE RECOVERED — the learned global shape φ(K) RISES with K, tracking the market's implied
     per-strike intensity smile Λ^mkt(K). φ≈1 flat would mean no smile recovered (bug / no signal).

Run after: git push (laptop) -> git pull (server) -> RESTART REPL (Revise can't re-track new
@model macros / structs), then:
    include("current_development/split_market_pillar/r07_local_intensity.jl")
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

include("l03_local_intensity_poisson.jl")

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
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

KMAX = 6

# ==========================================
# 3. THE LOCAL-INTENSITY SMILE MODEL
# ==========================================
model = LocalIntensitySmileDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = MarketSmileFeature(Kmax = KMAX),
    market_on              = true,
    supremacy_weight       = 1.0,
    smile_weight           = 1.0,
)

task = Experiments.create_experiment_task(
    ds_market, model, "local_intensity_r1", save_dir;
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
# 4. CONVERGENCE DIAGNOSTICS
# ==========================================
chains_obj = Experiments.Diagnostics.extract_chains(ds_market, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv = Experiments.Diagnostics.check_convergence(chains_obj)
display(conv.df)
println("\nWatch: σ_sup (~0.05–0.2, who-wins), σ_smile (how tight the model tracks the smile).")

# ==========================================
# 5. SMILE RECOVERY — learned φ(K) vs market Λ^mkt(K)
# ==========================================
chain = results.training_results.items[1][1]   # raw MCMCChains for the single split

# Learned global smile shape φ(K) = exp(posterior-mean log_φ(K))
nK   = KMAX + 1
φ_hat = [exp(mean(vec(Array(chain[Symbol("log_φ[$k]")])))) for k in 1:nK]

# Market-implied per-strike intensity smile, averaged across matches (independent of the model).
# For each strike K, gather de-vigged fair under-K probs and invert via Def 25 (l03 helper).
mkt_logΛ_means = fill(NaN, nK)
mkt_counts     = zeros(Int, nK)
for K in 0:KMAX
    sel = Symbol("under_$(K)5")
    sub = subset(ds_market.odds,
                 :selection => ByRow(s -> Symbol(s) == sel),
                 :prob_fair_close => ByRow(p -> !ismissing(p) && 1e-4 < Float64(p) < 1 - 1e-4))
    isempty(sub) && continue
    Λs = [_smile_intensity(Float64(p), K) for p in sub.prob_fair_close]
    Λs = filter(x -> isfinite(x) && x > 1e-4, Λs)
    isempty(Λs) && continue
    mkt_logΛ_means[K + 1] = mean(log.(Λs))
    mkt_counts[K + 1]     = length(Λs)
end
Λ_mkt = exp.(mkt_logΛ_means)

# Normalise both shapes to the central strike K=2 (most reliable) for a like-for-like comparison.
ref = 3  # index of K=2
smile_tbl = DataFrame(
    K          = 0:KMAX,
    n_matches  = mkt_counts,
    Λ_mkt      = round.(Λ_mkt, digits=3),
    φ_hat      = round.(φ_hat, digits=3),
    Λ_mkt_rel  = round.(Λ_mkt ./ Λ_mkt[ref], digits=3),
    φ_hat_rel  = round.(φ_hat ./ φ_hat[ref], digits=3),
)
println("\n", "="^70)
println("LOCAL-INTENSITY SMILE RECOVERY (single split)")
println("="^70)
println("Λ_mkt(K) = market-implied total intensity at strike K (RISING with K = the smile).")
println("φ_hat(K) = learned global shape (should RISE with K too; *_rel normalised to K=2).")
show(smile_tbl; allrows=true, allcols=true)
println()
for p in (:σ_smile, :σ_sup, :ν_xg)
    if p in keys(chain)
        v = vec(Array(chain[p]))
        println("  $p: mean=$(round(mean(v), digits=4))  std=$(round(std(v), digits=4))  " *
                "q=[$(round(quantile(v,0.05),digits=4)), $(round(quantile(v,0.95),digits=4))]")
    end
end
println("\nSUCCESS = φ_hat rises with K (≈ tracks Λ_mkt_rel) and σ_smile is finite/tight.")
println("Next: full-CV run + r06_per_line_eval.jl (add include of l03) -> per-strike O/U LogLoss.")
