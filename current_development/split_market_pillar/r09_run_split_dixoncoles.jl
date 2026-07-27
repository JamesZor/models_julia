#=
RUNNER for l05_split_market_dixoncoles.jl — SplitMarketDixonColesModel.

Single-split diagnostic. The Dixon-Coles sibling of r02/r08: same split (level/supremacy) market
pillar on the RATES, but goals ~ DC-Poisson with a FREE low-score correlation ρ (the τ correction
reshapes 0-0/1-0/0-1/1-1 → BTTS / correct-score). ρ is NOT market-anchored: it's the structural
edge. Defaults to Ireland top flight (79); DC ρ is ≈0 there empirically, so expect a weak τ
correction (this engine ≈ double-Poisson on low-ρ leagues).

Goals: (1) converges R-hat ≤ ~1.05 incl. σ_sup, σ_lev, dc.ρ_base; (2) ρ_base posterior visible.

Run after: git push -> git pull (server) -> RESTART REPL, then:
    include("current_development/split_market_pillar/r09_run_split_dixoncoles.jl")
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

include("current_development/split_market_pillar/l05_split_market_dixoncoles.jl")

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
dc_cfg    = PreGame.GlobalDixonColesConfig()      # single global ρ (robust); swap to Hierarchical for per-team
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

# ==========================================
# 3. THE SPLIT-MARKET DIXON-COLES MODEL
# ==========================================
model = SplitMarketDixonColesModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),   # rates only; ρ free
    dixon_coles_config     = dc_cfg,
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = 1.0,    # both marginals anchored
)

task = Experiments.create_experiment_task(
    ds_market, model, "split_dc_r1", save_dir;
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
# 4. CONVERGENCE + DIXON-COLES ρ DIAGNOSTICS
# ==========================================
chains_obj = Experiments.Diagnostics.extract_chains(ds_market, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv = Experiments.Diagnostics.check_convergence(chains_obj)
display(conv.df)

#=
julia> display(conv.df)
33×11 DataFrame
 Row │ std        mean        ess      train_season  raw_symbol            rhat      target_season  fold   week   parameter             entity               
     │ Float64    Float64     Float64  String        Symbol                Float64   String         Int64  Int64  String                String               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 0.319414    3.24742        NaN  2026          ν_xg                  1.00274   2026               0      0  ν_xg                  global
   2 │ 0.0345272   0.386161       NaN  2026          σ_sup                 1.00087   2026               0      0  σ_sup                 global
   3 │ 0.025297    0.160025       NaN  2026          σ_lev                 1.00263   2026               0      0  σ_lev                 global
   4 │ 0.0509344   0.227079       NaN  2026          ha.γ_team_raw[1]      1.00078   2026               0      0  home_advantage        bohemian
   5 │ 0.0513365   0.212431       NaN  2026          ha.γ_team_raw[2]      1.00008   2026               0      0  home_advantage        cork-city
   6 │ 0.0510091   0.219181       NaN  2026          ha.γ_team_raw[3]      0.999577  2026               0      0  home_advantage        derry-city
   7 │ 0.0519738   0.212407       NaN  2026          ha.γ_team_raw[4]      1.00201   2026               0      0  home_advantage        drogheda-united
   8 │ 0.0561859   0.222079       NaN  2026          ha.γ_team_raw[5]      1.00088   2026               0      0  home_advantage        dundalk-fc
   9 │ 0.0510595   0.23071        NaN  2026          ha.γ_team_raw[6]      1.00039   2026               0      0  home_advantage        galway-united
  10 │ 0.0510321   0.220608       NaN  2026          ha.γ_team_raw[7]      1.00306   2026               0      0  home_advantage        shamrock-rovers
  11 │ 0.0509069   0.22583        NaN  2026          ha.γ_team_raw[8]      1.00133   2026               0      0  home_advantage        shelbourne
  12 │ 0.0521286   0.226242       NaN  2026          ha.γ_team_raw[9]      1.00134   2026               0      0  home_advantage        sligo-rovers
  13 │ 0.0521842   0.225693       NaN  2026          ha.γ_team_raw[10]     1.00175   2026               0      0  home_advantage        st-patricks-athletic
  14 │ 0.0511854   0.229381       NaN  2026          ha.γ_team_raw[11]     1.00198   2026               0      0  home_advantage        waterford-fc
  15 │ 0.0455924   0.223068       NaN  2026          ha.γ_base             0.999616  2026               0      0  ha.γ_base             global
  16 │ 0.0205994   0.0258232      NaN  2026          ha.σ_γ                1.00121   2026               0      0  ha.σ_γ                global
  17 │ 0.063757    1.09199        NaN  2026          kap.κ_team_raw[1]     1.00115   2026               0      0  kappa                 bohemian
  18 │ 0.0593294   1.0475         NaN  2026          kap.κ_team_raw[2]     1.00189   2026               0      0  kappa                 cork-city
  19 │ 0.0687058   1.0228         NaN  2026          kap.κ_team_raw[3]     1.00114   2026               0      0  kappa                 derry-city
  20 │ 0.0636135   1.02859        NaN  2026          kap.κ_team_raw[4]     1.00033   2026               0      0  kappa                 drogheda-united
  21 │ 0.0732847   1.06516        NaN  2026          kap.κ_team_raw[5]     0.999983  2026               0      0  kappa                 dundalk-fc
  22 │ 0.0611967   1.06883        NaN  2026          kap.κ_team_raw[6]     1.00162   2026               0      0  kappa                 galway-united
  23 │ 0.0622674   1.04906        NaN  2026          kap.κ_team_raw[7]     1.00004   2026               0      0  kappa                 shamrock-rovers
  24 │ 0.0610469   1.05533        NaN  2026          kap.κ_team_raw[8]     1.0016    2026               0      0  kappa                 shelbourne
  25 │ 0.0652163   1.09394        NaN  2026          kap.κ_team_raw[9]     1.0003    2026               0      0  kappa                 sligo-rovers
  26 │ 0.0611534   1.08144        NaN  2026          kap.κ_team_raw[10]    1.00255   2026               0      0  kappa                 st-patricks-athletic
  27 │ 0.0657136   1.09663        NaN  2026          kap.κ_team_raw[11]    1.0024    2026               0      0  kappa                 waterford-fc
  28 │ 0.082108    0.642848       NaN  2026          kap.κ_base            1.00041   2026               0      0  kap.κ_base            global
  29 │ 0.0414732   0.0667929      NaN  2026          kap.σ_κ               1.00454   2026               0      0  kap.σ_κ               global
  30 │ 0.0685012  -0.127073       NaN  2026          p_dyn.w_G_att         0.999852  2026               0      0  p_dyn.w_G_att         global
  31 │ 0.0636671   0.170458       NaN  2026          p_dyn.w_G_def         1.0002    2026               0      0  p_dyn.w_G_def         global
  32 │ 0.0157378   0.108924       NaN  2026          p_dyn.w_Outfield_att  1.00097   2026               0      0  p_dyn.w_Outfield_att  global
  33 │ 0.0148175  -0.123085       NaN  2026          p_dyn.w_Outfield_def  1.00138   2026               0      0  p_dyn.w_Outfield_def  global
=#


chain = results.training_results.items[1][1]
println("\n--- Dixon-Coles correlation ρ (after 0.3·tanh squash) ---")
if Symbol("dc.ρ_base") in keys(chain)
    rb = vec(Array(chain[Symbol("dc.ρ_base")]))
    ρ  = 0.3 .* tanh.(rb)
    println("  ρ_base: mean=$(round(mean(rb),digits=4))  ->  ρ: mean=$(round(mean(ρ),digits=4))  " *
            "q=[$(round(quantile(ρ,0.05),digits=4)), $(round(quantile(ρ,0.95),digits=4))]")
    println("  (ρ≈0 = DC correction negligible ⇒ ≈ double-Poisson; |ρ|>~0.05 = real low-score structure)")
else
    println("  (dc.ρ_base not in chain — check raw_symbol names in conv.df)")
end
for p in (:σ_sup, :σ_lev, :ν_xg)
    if p in keys(chain)
        v = vec(Array(chain[p]))
        println("  $p: mean=$(round(mean(v),digits=4))  std=$(round(std(v),digits=4))")
    end
end




#=
(ρ≈0 = DC correction negligible ⇒ ≈ double-Poisson; |ρ|>~0.05 = real low-score structure)
=#

println("\nNext: full-CV + r06_per_line_eval.jl (point include at l05) — look for edge on BTTS /")
println("correct-score (where ρ reshapes the low-score cells), not totals.")
