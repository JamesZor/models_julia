#=
RUNNER for l04_split_market_negbin.jl — SplitMarketDoubleNegBinModel.

Single-split diagnostic. The NegBin sibling of r02: same split (level/supremacy) market pillar,
but goals ~ RobustNegativeBinomial(r, λ) so the model carries a structural DISPERSION the market's
independent-Poisson template ignores (this is what moves BTTS / correct-score → the derived-market
edge). Defaults to Ireland FIRST DIVISION (718, V/M≈1.14) where the dispersion actually bites — on
the near-Poisson top flight (79, V/M 0.94) r fits large and this ≈ the double-Poisson.

Goals:
  1. CONVERGES — R-hat ≤ ~1.05 incl. σ_sup, σ_lev and the dispersion r.
  2. DISPERSION is real — fitted r is finite/small (over-dispersion), not r→∞ (=Poisson).

Run after: git push -> git pull (server) -> RESTART REPL, then:
    include("current_development/split_market_pillar/r08_run_split_negbin.jl")
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

include("l04_split_market_negbin.jl")

# ==========================================
# 1. DATA — Betfair market pillar
# ==========================================
# 718 = Ireland First Division (over-dispersed, NB regime). Swap to Data.Ireland() for the
# near-Poisson top flight (there NegBin ≈ double-Poisson).
segment = Data.IrelandFirstDivision()
ds = Data.load_datastore_cached(segment)
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
disp_cfg  = PreGame.HomeAwayDispersion()        # NegBin r (now USED)
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

# ==========================================
# 3. THE SPLIT-MARKET NEGBIN MODEL (both marginals anchored)
# ==========================================
model = SplitMarketDoubleNegBinModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = 1.0,    # anchor BOTH marginals (totals rate + supremacy)
)

task = Experiments.create_experiment_task(
    ds_market, model, "split_negbin_r1", save_dir;
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
# 4. CONVERGENCE + DISPERSION DIAGNOSTICS
# ==========================================
chains_obj = Experiments.Diagnostics.extract_chains(ds_market, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv = Experiments.Diagnostics.check_convergence(chains_obj)
display(conv.df)

println("\n--- Dispersion rows (the NegBin r; small r = over-dispersed, large r ≈ Poisson) ---")
disp_rows = filter(r -> occursin("disp", lowercase(string(r.raw_symbol))) ||
                        occursin("disp", lowercase(string(r.parameter))), conv.df)
isempty(disp_rows) ? println("(no 'disp' rows matched — inspect conv.df raw_symbol column)") :
                     display(disp_rows[:, [:raw_symbol, :mean, :std, :rhat]])

println("""

Read:
 • σ_sup (who-wins disagreement) and σ_lev (totals disagreement) — both sampled, should be finite.
 • Dispersion r: a SMALL finite r (say < ~15) means real over-dispersion the NegBin is capturing;
   r drifting very large means the data is ≈ Poisson and NegBin adds nothing (expected on tnmt 79).
 • Next: full-CV run + r06_per_line_eval.jl (point its include at l04) — the edge to look for is on
   BTTS / correct-score (where dispersion reshapes P(0)/tails), NOT totals.
""")
