#=
r01 — SINGLE-SPLIT SHAKEDOWN: the three κ modes, in parallel, Ireland (79).

Goals (in order):
  1. CONVERGENCE — R-hat ≤ ~1.01 / healthy ESS for ALL modes, checked on the RAW chains for
     the new κ params (κ0, τ_net/τ_att/τ_def, δ_net/z_att/z_def). NB check_convergence's
     curated conv.df DROPS params it doesn't know — do NOT rely on it for the new latents
     (the r17/r18 lesson).
  2. WHAT DID κ LEARN — per-team att/def multipliers (mean ± sd), team spread, and for
     :attdef the att-def correlation. τ pulled to ~0 with multipliers ≈ 1.00 = "learned
     nothing" (σ-hierarchy-null pattern).
  3. V0 sanity — :attack_only must reproduce the known HierarchicalTeamKappa behaviour
     (κ multipliers ~0.9–1.15, σ_κ small).

Execution: 3 variants × 4 chains = 12 concurrent chains via @sync/@spawn (r03 pattern;
each experiment's queue holds 4 items ⇒ 12 ≤ 16 pinned cores, no oversubscription).

Run on the server after git push → git pull → RESTART REPL:
    include("current_development/kappa_def/r01_single_split_shakedown.jl")
Flip SEGMENT to Data.IrelandFirstDivision() for the 718 follow-up (r02).
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

include("current_development/kappa_def/l01_kappa_def_models.jl")

# ==========================================
# 1. DATA (no market swap needed — market OFF)
# ==========================================
SEGMENT = Data.Ireland()
seg_tag = lowercase(string(nameof(typeof(SEGMENT))))
println("[INFO] Loading $(seg_tag) DataStore...")
ds = Data.load_datastore_cached(SEGMENT)

save_dir = "./data/kappa_def_dev_$(seg_tag)/"
mkpath(save_dir)

# ==========================================
# 2. SHARED CONFIG (r08/r03 conventions)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()          # carried, unused (Poisson)
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()       # V0 control only
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

_make(mode) = KappaDefDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    kappa_mode             = mode,
)

variants = [
    ("V0_attack_only", _make(:attack_only)),
    ("V2_net",         _make(:net)),
    ("V1_attdef",      _make(:attdef)),
]

function _build_task(model, name)
    Experiments.create_experiment_task(
        ds, model, name, save_dir;
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

# ==========================================
# 3. RUN ALL VARIANTS IN PARALLEL (12 chains ≤ 16 cores)
# ==========================================
println("\n>> Launching $(length(variants)) κ modes in parallel " *
        "($(length(variants))×4 = $(length(variants)*4) chains, $(Threads.nthreads()) threads)...")
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

# ==========================================
# 4. CONVERGENCE — curated banner + RAW κ-param diagnostics
# ==========================================
"raw-chain R-hat/ESS for parameter names matching any of `pats` (curated conv.df drops these)"
function raw_kappa_diag(ch, pats::Vector{String})
    s  = ess_rhat(ch)
    nm = string.(s.nt.parameters)
    keep = [any(occursin(p, n) for p in pats) for n in nm]
    DataFrame(parameter = nm[keep],
              rhat = round.(s.nt.rhat[keep], digits=4),
              ess  = round.(s.nt.ess[keep], digits=1))
end

KAPPA_PATS = ["κ0", "τ_net", "δ_net", "τ_att", "τ_def", "z_att", "z_def", "kap."]

"team index → name, recovered from the curated conv.df home_advantage rows (always present)"
function team_names_from_conv(conv_df, n_teams)
    ha_rows = conv_df[conv_df.parameter .== "home_advantage", [:raw_symbol, :entity]]
    names_ = fill("", n_teams)
    for r in eachrow(ha_rows)
        m = match(r"\[(\d+)\]", string(r.raw_symbol))
        m === nothing && continue
        i = parse(Int, m.captures[1])
        1 <= i <= n_teams && (names_[i] = string(r.entity))
    end
    any(isempty, names_) ? ["team_$i" for i in 1:n_teams] : names_
end

summaries = Dict{String, Any}()
for (name, model) in variants
    println("\n", "="^72, "\n>> MODE: $name\n", "="^72)
    res = raw_results[name]
    ch  = res.training_results.items[1][1]     # single split → one (chains-combined) Chains

    # curated banner (components) — informative but NOT sufficient for the new params
    chains_obj = Experiments.Diagnostics.extract_chains(ds, res)
    conv = Experiments.Diagnostics.check_convergence(chains_obj)
    println(conv)

    # n_teams from the FOLD's ha vector (chain-side truth; full-ds team count can mismatch)
    ha_idx = [match(r"\[(\d+)\]", string(s)) for s in conv.df.raw_symbol[conv.df.parameter .== "home_advantage"]]
    n_teams = maximum(parse(Int, m.captures[1]) for m in ha_idx if m !== nothing)

    # raw κ diagnostics — the real gate
    kd = raw_kappa_diag(ch, KAPPA_PATS)
    println("\n--- RAW κ-param diagnostics (the real convergence gate) ---")
    show(kd; allrows=true, allcols=true); println()
    max_rhat = isempty(kd) ? NaN : maximum(filter(!isnan, kd.rhat))
    println("max κ-param R-hat: $max_rhat  ", max_rhat <= 1.01 ? "✅" : (max_rhat <= 1.05 ? "⚠️ marginal" : "❌"))

    # what κ learned
    tnames = team_names_from_conv(conv.df, n_teams)
    tsum, glob = kappa_team_summary(model, ch, n_teams; team_names=tnames)
    println("\n--- per-team κ multipliers (goals-vs-xG conversion) ---")
    show(sort(tsum, :att_mult, rev=true); allrows=true, allcols=true); println()
    println("globals: κ0_conv=$(glob.κ0_conv)  att_spread=$(glob.att_spread)  " *
            "def_spread=$(glob.def_spread)  attdef_cor=$(glob.attdef_cor)")

    summaries[name] = (; max_rhat, glob)
end

#=
========================================================================                                                                                                                                                                                                                                                    
>> MODE: V0_attack_only                                                                                                                                                                                                                                                                                                     
======================================================================== 
BayesianFootball.Experiments.Diagnostics.ChainDiagnostic(31×11 DataFrame                                                                                                                                                                                                                                                    
 Row │ std         mean        ess      train_season  raw_symbol            rhat     target_season  fold   week   parameter             entity                                                                                                                                                                              
     │ Float64     Float64     Float64  String        Symbol                Float64  String         Int64  Int64  String                String                                                                                                                                                                              
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                               
   1 │  1.21354     2.54531        NaN  2026          ν_xg                  1.52404  2026               0      0  ν_xg                  global                                                                                                                                                                              
   2 │  5.97615    -3.23339        NaN  2026          ha.γ_team_raw[1]      1.1051   2026               0      0  home_advantage        bohemian                                                                                                                                                                            
   3 │  9.38917     5.59598        NaN  2026          ha.γ_team_raw[2]      1.25807  2026               0      0  home_advantage        cork-city                                                                                                                                                                           
   4 │  9.5395     -5.2941         NaN  2026          ha.γ_team_raw[3]      1.20953  2026               0      0  home_advantage        derry-city                                                                                                                                                                          
   5 │ 14.0962     -7.9445         NaN  2026          ha.γ_team_raw[4]      1.30329  2026               0      0  home_advantage        drogheda-united                                                                                                                                                                     
   6 │  6.58487    -3.59689        NaN  2026          ha.γ_team_raw[5]      1.07688  2026               0      0  home_advantage        dundalk-fc                                                                                                                                                                          
   7 │  0.288685    0.0613723      NaN  2026          ha.γ_team_raw[6]      1.52873  2026               0      0  home_advantage        galway-united                                                                                                                                                                       
   8 │  2.823      -1.42681        NaN  2026          ha.γ_team_raw[7]      1.52688  2026               0      0  home_advantage        shamrock-rovers                                                                                                                                                                     
   9 │ 10.0072     -5.56833        NaN  2026          ha.γ_team_raw[8]      1.22844  2026               0      0  home_advantage        shelbourne                                                                                                                                                                          
  10 │ 13.6957     -7.70182        NaN  2026          ha.γ_team_raw[9]      1.34195  2026               0      0  home_advantage        sligo-rovers                                                                                                                                                                        
  11 │ 12.8012     -7.18176        NaN  2026          ha.γ_team_raw[10]     1.31683  2026               0      0  home_advantage        st-patricks-athletic                                                                                                                                                                
  12 │  8.65716     5.19183        NaN  2026          ha.γ_team_raw[11]     1.19365  2026               0      0  home_advantage        waterford-fc                                                                                                                                                                        
  13 │  0.0826077   0.184493       NaN  2026          ha.γ_base             1.08714  2026               0      0  ha.γ_base             global                                                                                                                                                                              
  14 │  7.44533     4.3529         NaN  2026          ha.σ_γ                1.5321   2026               0      0  ha.σ_γ                global                                                                                                                                                                              
  15 │  0.474343    0.839068       NaN  2026          kap.κ_team_raw[1]     1.19823  2026               0      0  kappa                 bohemian                                                                                                                                                                            
  16 │  0.464091    0.827332       NaN  2026          kap.κ_team_raw[2]     1.15164  2026               0      0  kappa                 cork-city                                                                                                                                                                           
  17 │  0.489034    0.83463        NaN  2026          kap.κ_team_raw[3]     1.47286  2026               0      0  kappa                 derry-city                                                                                                                                                                          
  18 │  0.285928    0.938692       NaN  2026          kap.κ_team_raw[4]     1.52697  2026               0      0  kappa                 drogheda-united                                                                                                                                                                     
  19 │  1.38477     1.8995         NaN  2026          kap.κ_team_raw[5]     1.21349  2026               0      0  kappa                 dundalk-fc                                                                                                                                                                          
  20 │  1.07139     1.70733        NaN  2026          kap.κ_team_raw[6]     1.19008  2026               0      0  kappa                 galway-united                                                                                                                                                                       
  21 │  1.58503     2.01536        NaN  2026          kap.κ_team_raw[7]     1.25456  2026               0      0  kappa                 shamrock-rovers                                                                                                                                                                     
  22 │  0.952276    1.65937        NaN  2026          kap.κ_team_raw[8]     1.09429  2026               0      0  kappa                 shelbourne                                                                                                                                                                          
  23 │  1.57466     2.00753        NaN  2026          kap.κ_team_raw[9]     1.24419  2026               0      0  kappa                 sligo-rovers                                                                                                                                                                        
  24 │  0.166752    1.17048        NaN  2026          kap.κ_team_raw[10]    1.47417  2026               0      0  kappa                 st-patricks-athletic                                                                                                                                                                
  25 │  1.0131      1.68225        NaN  2026          kap.κ_team_raw[11]    1.14997  2026               0      0  kappa                 waterford-fc                                                                                                                                                                        
  26 │  0.16592     0.627615       NaN  2026          kap.κ_base            1.43986  2026               0      0  kap.κ_base            global                                                                                                                                                                              
  27 │  1.15634     0.741707       NaN  2026          kap.σ_κ               1.53191  2026               0      0  kap.σ_κ               global                                                                                                                                                                              
  28 │  0.432439    0.114151       NaN  2026          p_dyn.w_G_att         1.52664  2026               0      0  p_dyn.w_G_att         global                                                                                                                                                                              
  29 │  0.38929     0.228771       NaN  2026          p_dyn.w_G_def         1.52208  2026               0      0  p_dyn.w_G_def         global                                                                                                                                                                              
  30 │  1.25521    -0.640327       NaN  2026          p_dyn.w_Outfield_att  1.53014  2026               0      0  p_dyn.w_Outfield_att  global                                                                                                                                                                              
  31 │  1.27042    -0.848204       NaN  2026          p_dyn.w_Outfield_def  1.52988  2026               0      0  p_dyn.w_Outfield_def  global)                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                            
--- RAW κ-param diagnostics (the real convergence gate) ---                                                                                                                                                                                                                                                                 
13×3 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ parameter           rhat     ess                                                                                                                                                                                                                                                                                     
     │ String              Float64  Float64                                                                                                                                                                                                                                                                                 
─────┼──────────────────────────────────────                                                                                                                                                                                                                                                                                
   1 │ kap.κ_base           1.4399     15.7                                                                                                                                                                                                                                                                                 
   2 │ kap.σ_κ              1.5319     14.2                                                                                                                                                                                                                                                                                 
   3 │ kap.κ_team_raw[1]    1.1982     27.4                                                                                                                                                                                                                                                                                 
   4 │ kap.κ_team_raw[2]    1.1516     34.7                                                                                                                                                                                                                                                                                 
   5 │ kap.κ_team_raw[3]    1.4729     15.1                                                                                                                                                                                                                                                                                 
   6 │ kap.κ_team_raw[4]    1.527    4344.7                                                                                                                                                                                                                                                                                 
   7 │ kap.κ_team_raw[5]    1.2135     26.1                                                                                                                                                                                                                                                                                 
   8 │ kap.κ_team_raw[6]    1.1901     28.2                                                                                                                                                                                                                                                                                 
   9 │ kap.κ_team_raw[7]    1.2546     22.8                                                                                                                                                                                                                                                                                 
  10 │ kap.κ_team_raw[8]    1.0943     53.9                                                                                                                                                                                                                                                                                 
  11 │ kap.κ_team_raw[9]    1.2442     23.2                                                                                                                                                                                                                                                                                 
  12 │ kap.κ_team_raw[10]   1.4742    461.9                                                                                                                                                                                                                                                                                 
  13 │ kap.κ_team_raw[11]   1.15       34.9                                                                                                                                                                                                                                                                                 
max κ-param R-hat: 1.5319  ❌                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                            
--- per-team κ multipliers (goals-vs-xG conversion) ---                                                                                                                                                                                                                                                                     
11×5 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ team                  att_mult  att_sd   def_mult  def_sd                                                                                                                                                                                                                                                            
     │ String                Float64   Float64  Float64   Float64                                                                                                                                                                                                                                                           
─────┼────────────────────────────────────────────────────────────                                                                                                                                                                                                                                                          
   1 │ shamrock-rovers         2.0154   1.585        1.0      0.0                                                                                                                                                                                                                                                           
   2 │ sligo-rovers            2.0075   1.5747       1.0      0.0                                                                                                                                                                                                                                                           
   3 │ dundalk-fc              1.8995   1.3848       1.0      0.0                                                                                                                                                                                                                                                           
   4 │ galway-united           1.7073   1.0714       1.0      0.0                                                                                                                                                                                                                                                           
   5 │ waterford-fc            1.6822   1.0131       1.0      0.0                                                                                                                                                                                                                                                           
   6 │ shelbourne              1.6594   0.9523       1.0      0.0                                                                                                                                                                                                                                                           
   7 │ st-patricks-athletic    1.1705   0.1668       1.0      0.0                                                                                                                                                                                                                                                           
   8 │ drogheda-united         0.9387   0.2859       1.0      0.0                                                                                                                                                                                                                                                           
   9 │ bohemian                0.8391   0.4743       1.0      0.0                                                                                                                                                                                                                                                           
  10 │ derry-city              0.8346   0.489        1.0      0.0                                                                                                                                                                                                                                                           
  11 │ cork-city               0.8273   0.4641       1.0      0.0                                                                                                                                                                                                                                                           
globals: κ0_conv=1.0  att_spread=1.1881  def_spread=0.0  attdef_cor=NaN
=#


 

#=
========================================================================
>> MODE: V2_net                                                                
========================================================================
BayesianFootball.Experiments.Diagnostics.ChainDiagnostic(18×11 DataFrame
 Row │ std        mean        ess      train_season  raw_symbol            rhat      target_season  fold   week   parameter             entity               
     │ Float64    Float64     Float64  String        Symbol                Float64   String         Int64  Int64  String                String               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 0.32962     3.22855        NaN  2026          ν_xg                  1.00158   2026               0      0  ν_xg                  global
   2 │ 0.102125    0.223206       NaN  2026          ha.γ_team_raw[1]      1.00069   2026               0      0  home_advantage        bohemian
   3 │ 0.110292    0.185796       NaN  2026          ha.γ_team_raw[2]      1.001     2026               0      0  home_advantage        cork-city
   4 │ 0.102375    0.223202       NaN  2026          ha.γ_team_raw[3]      1.00137   2026               0      0  home_advantage        derry-city
   5 │ 0.104229    0.202414       NaN  2026          ha.γ_team_raw[4]      1.00009   2026               0      0  home_advantage        drogheda-united
   6 │ 0.110326    0.211261       NaN  2026          ha.γ_team_raw[5]      1.00033   2026               0      0  home_advantage        dundalk-fc
   7 │ 0.102163    0.227379       NaN  2026          ha.γ_team_raw[6]      1.00041   2026               0      0  home_advantage        galway-united
   8 │ 0.10164     0.21308        NaN  2026          ha.γ_team_raw[7]      0.999971  2026               0      0  home_advantage        shamrock-rovers
   9 │ 0.1019      0.218354       NaN  2026          ha.γ_team_raw[8]      1.00061   2026               0      0  home_advantage        shelbourne
  10 │ 0.101856    0.210849       NaN  2026          ha.γ_team_raw[9]      1.00028   2026               0      0  home_advantage        sligo-rovers
  11 │ 0.101448    0.216543       NaN  2026          ha.γ_team_raw[10]     1.00067   2026               0      0  home_advantage        st-patricks-athletic
  12 │ 0.103078    0.204626       NaN  2026          ha.γ_team_raw[11]     0.999695  2026               0      0  home_advantage        waterford-fc
  13 │ 0.0845773   0.212459       NaN  2026          ha.γ_base             1.0       2026               0      0  ha.γ_base             global
  14 │ 0.0431468   0.0562354      NaN  2026          ha.σ_γ                1.00103   2026               0      0  ha.σ_γ                global
  15 │ 0.137715   -0.118982       NaN  2026          p_dyn.w_G_att         1.00033   2026               0      0  p_dyn.w_G_att         global
  16 │ 0.132705    0.0178605      NaN  2026          p_dyn.w_G_def         0.999795  2026               0      0  p_dyn.w_G_def         global
  17 │ 0.0293716   0.0864726      NaN  2026          p_dyn.w_Outfield_att  1.00174   2026               0      0  p_dyn.w_Outfield_att  global
  18 │ 0.0279376  -0.113586       NaN  2026          p_dyn.w_Outfield_def  1.00151   2026               0      0  p_dyn.w_Outfield_def  global)

--- RAW κ-param diagnostics (the real convergence gate) ---                    
13×3 DataFrame                                                                 
 Row │ parameter  rhat     ess                                                 
     │ String     Float64  Float64                                             
─────┼─────────────────────────────                                            
   1 │ κ0          1.0017   5339.1                                             
   2 │ τ_net       1.0002   3025.3                                             
   3 │ δ_net[1]    1.0005   5095.8                                             
   4 │ δ_net[2]    1.0037   6488.7                                             
   5 │ δ_net[3]    1.0015   5089.5                                             
   6 │ δ_net[4]    1.0016   5054.3                                             
   7 │ δ_net[5]    1.0003   5446.7                                             
   8 │ δ_net[6]    0.9997   5703.6                                             
   9 │ δ_net[7]    1.0045   4784.3                                             
  10 │ δ_net[8]    1.0009   4519.4                                             
  11 │ δ_net[9]    1.0005   4422.9                                             
  12 │ δ_net[10]   1.0004   5850.5                                             
  13 │ δ_net[11]   1.0019   6115.7                                             
max κ-param R-hat: 1.0045  ✅                                                  

--- per-team κ multipliers (goals-vs-xG conversion) ---                        
11×5 DataFrame                                                                 
 Row │ team                  att_mult  att_sd   def_mult  def_sd               
     │ String                Float64   Float64  Float64   Float64              
─────┼────────────────────────────────────────────────────────────             
   1 │ shelbourne              1.0211   0.0618    1.0211   0.0618              
   2 │ derry-city              1.0209   0.0622    1.0209   0.0622              
   3 │ shamrock-rovers         1.0079   0.0574    1.0079   0.0574              
   4 │ st-patricks-athletic    1.0056   0.0564    1.0056   0.0564              
   5 │ dundalk-fc              1.0013   0.0647    1.0013   0.0647              
   6 │ bohemian                1.0007   0.0524    1.0007   0.0524              
   7 │ sligo-rovers            0.9991   0.0553    0.9991   0.0553              
   8 │ galway-united           0.9936   0.0547    0.9936   0.0547              
   9 │ drogheda-united         0.9935   0.0553    0.9935   0.0553              
  10 │ waterford-fc            0.9911   0.055     0.9911   0.055               
  11 │ cork-city               0.984    0.0586    0.984    0.0586              
globals: κ0_conv=0.9746  att_spread=0.0371  def_spread=0.0371  attdef_cor=NaN
=#




#=
========================================================================
>> MODE: V1_attdef                                                             
========================================================================

BayesianFootball.Experiments.Diagnostics.ChainDiagnostic(18×11 DataFrame
 Row │ std        mean        ess      train_season  raw_symbol            rhat      target_season  fold   week   parameter             entity               
     │ Float64    Float64     Float64  String        Symbol                Float64   String         Int64  Int64  String                String               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 0.316448    3.2336         NaN  2026          ν_xg                  1.00087   2026               0      0  ν_xg                  global
   2 │ 0.106183    0.220073       NaN  2026          ha.γ_team_raw[1]      1.00116   2026               0      0  home_advantage        bohemian
   3 │ 0.109862    0.179811       NaN  2026          ha.γ_team_raw[2]      1.00136   2026               0      0  home_advantage        cork-city
   4 │ 0.107815    0.219248       NaN  2026          ha.γ_team_raw[3]      1.00145   2026               0      0  home_advantage        derry-city
   5 │ 0.104911    0.199024       NaN  2026          ha.γ_team_raw[4]      1.00145   2026               0      0  home_advantage        drogheda-united
   6 │ 0.111188    0.208946       NaN  2026          ha.γ_team_raw[5]      1.002     2026               0      0  home_advantage        dundalk-fc
   7 │ 0.106379    0.225783       NaN  2026          ha.γ_team_raw[6]      1.00186   2026               0      0  home_advantage        galway-united
   8 │ 0.105121    0.208583       NaN  2026          ha.γ_team_raw[7]      1.0009    2026               0      0  home_advantage        shamrock-rovers
   9 │ 0.107855    0.2145         NaN  2026          ha.γ_team_raw[8]      1.00116   2026               0      0  home_advantage        shelbourne
  10 │ 0.105982    0.210336       NaN  2026          ha.γ_team_raw[9]      0.999854  2026               0      0  home_advantage        sligo-rovers
  11 │ 0.108265    0.212104       NaN  2026          ha.γ_team_raw[10]     1.00022   2026               0      0  home_advantage        st-patricks-athletic
  12 │ 0.107239    0.201894       NaN  2026          ha.γ_team_raw[11]     1.00102   2026               0      0  home_advantage        waterford-fc
  13 │ 0.0893474   0.208806       NaN  2026          ha.γ_base             0.99999   2026               0      0  ha.γ_base             global
  14 │ 0.043661    0.056839       NaN  2026          ha.σ_γ                1.00145   2026               0      0  ha.σ_γ                global
  15 │ 0.134281   -0.1182         NaN  2026          p_dyn.w_G_att         1.00156   2026               0      0  p_dyn.w_G_att         global
  16 │ 0.135179    0.0157748      NaN  2026          p_dyn.w_G_def         1.00411   2026               0      0  p_dyn.w_G_def         global
  17 │ 0.028348    0.0861292      NaN  2026          p_dyn.w_Outfield_att  1.00085   2026               0      0  p_dyn.w_Outfield_att  global
  18 │ 0.0279832  -0.1141         NaN  2026          p_dyn.w_Outfield_def  1.00027   2026               0      0  p_dyn.w_Outfield_def  global)

--- RAW κ-param diagnostics (the real convergence gate) ---                    
25×3 DataFrame                                                                 
 Row │ parameter  rhat     ess                                                 
     │ String     Float64  Float64                                             
─────┼─────────────────────────────                                            
   1 │ κ0          1.0019   4582.3                                             
   2 │ τ_att       1.0      2534.3                                             
   3 │ τ_def       1.0023   2591.6                                             
   4 │ z_att[1]    1.0009   4870.0                                             
   5 │ z_att[2]    1.0004   4948.6                                             
   6 │ z_att[3]    1.0011   5231.6                                             
   7 │ z_att[4]    1.001    5876.8                                             
   8 │ z_att[5]    0.9997   5423.3                                             
   9 │ z_att[6]    1.0023   5157.9                                             
  10 │ z_att[7]    1.002    5974.8                                             
  11 │ z_att[8]    0.9998   5582.2                                             
  12 │ z_att[9]    0.9999   7544.3                                             
  13 │ z_att[10]   0.9997   6491.4                                             
  14 │ z_att[11]   1.0012   4344.2                                             
  15 │ z_def[1]    1.0021   6340.0                                             
  16 │ z_def[2]    1.0002   5130.5                                             
  17 │ z_def[3]    1.0009   5095.1                                             
  18 │ z_def[4]    1.0014   5516.9                                             
  19 │ z_def[5]    1.0018   4875.1                                             
  20 │ z_def[6]    1.0005   5284.2                                             
  21 │ z_def[7]    1.0001   5725.2                                             
  22 │ z_def[8]    1.0016   5765.5                                             
  23 │ z_def[9]    0.9998   5618.9                                             
  24 │ z_def[10]   0.9998   5565.2                                             
  25 │ z_def[11]   1.0007   5547.7                                             
max κ-param R-hat: 1.0023  ✅                                                  

--- per-team κ multipliers (goals-vs-xG conversion) ---                        
11×5 DataFrame                                                                 
 Row │ team                  att_mult  att_sd   def_mult  def_sd               
     │ String                Float64   Float64  Float64   Float64              
─────┼────────────────────────────────────────────────────────────             
   1 │ shelbourne              1.0233   0.0801    1.0156   0.0752              
   2 │ derry-city              1.0222   0.0806    1.011    0.0726              
   3 │ bohemian                1.0143   0.0721    0.9878   0.0683              
   4 │ shamrock-rovers         1.0046   0.0725    1.0083   0.0698              
   5 │ waterford-fc            1.0038   0.0709    0.9858   0.0698              
   6 │ dundalk-fc              1.0021   0.0778    1.0034   0.0781              
   7 │ sligo-rovers            1.0001   0.0718    1.0015   0.0698              
   8 │ drogheda-united         0.9935   0.0694    0.9984   0.0681              
   9 │ cork-city               0.99     0.0713    0.9897   0.065               
  10 │ st-patricks-athletic    0.9888   0.0684    1.0224   0.0812              
  11 │ galway-united           0.9871   0.07      1.0047   0.07                
globals: κ0_conv=0.9735  att_spread=0.0362  def_spread=0.0366  attdef_cor=0.1
=#




# ==========================================
# 5. VERDICT TABLE
# ==========================================
println("\n", "█"^72, "\n  SHAKEDOWN SUMMARY ($seg_tag)\n", "█"^72)
for (name, _) in variants
    s = summaries[name]
    println(rpad(name, 16), " max_rhat=", rpad(s.max_rhat, 8),
            " κ0_conv=", rpad(s.glob.κ0_conv, 8),
            " att_spread=", rpad(s.glob.att_spread, 8),
            " def_spread=", rpad(s.glob.def_spread, 8),
            " attdef_cor=", s.glob.attdef_cor)
end
println("""

[READ]
 • Convergence gate: max κ-param R-hat ≤ 1.01 (from the RAW table, not the curated banner).
 • "Learned nothing" pattern: τ near 0, att/def_spread ≲ 0.02 (all multipliers ≈ 1.00) —
   the σ-hierarchy-null outcome; note it in EXPERIMENTS.md and don't bother evaluating.
 • V0 sanity: att_mult range should look like the familiar HierarchicalTeamKappa (~0.9–1.15).
 • :attdef vs :net — if att and def multipliers are strongly correlated across teams,
   the net (V2) parameterization captures it with half the params.
 • Cross-check the r00 persistence gate: def_spread here should only be trusted if the EDA
   said defensive residuals persist. Update EXPERIMENTS.md either way.
 • Next: flip SEGMENT to IrelandFirstDivision() (r02); then full-CV vs V0 judged vs the
   market (r03).
""")
