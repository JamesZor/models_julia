#=
PER-LINE BIAS-vs-EDGE diagnostic on the saved double-Poisson grid (r05 output). No retrain.

PHILOSOPHY (confirmed): in a Bayesian-Kelly frame, single-bet growth is LINEAR in E[p] (unified_kelly
§6), so the edge is the per-match mean deviation E[p]−p_market; the posterior tails only SIZE the stake
(shrinkage). So per line we want the model's CENTRE unbiased vs the market (no systematic skew — the
thing per-line calibration would fix), with the per-MATCH deviations carrying the edge. This measures,
per market line:

  1. SYSTEMATIC BIAS   mean(model_p − market_p), with se = std/√n and t = bias/se.
       |t| ≫ 2  -> a real per-line skew (calibration target).   |t| ≈ 0 -> already centred.
  2. DEVIATION BUDGET  std(model_p − market_p)  -> how much genuine match-level disagreement remains
       AFTER the bias = the size of the edge you bet from. Big std + small bias = lots of clean signal.
  3. DO DEVIATIONS WIN  GLMEdge spread_fair_coef (+p): does (model−market) predict the outcome beyond
       the market? >0 & p<0.1 -> the tails carry real edge on that line; ≈0 -> deviations are noise.

Reads: lines with big |t| AND positive GLMEdge -> per-line calibration is worth building THERE (recentre
the skew, keep the deviations). Lines already centred -> single λ is fine. Bias but GLMEdge≈0 -> noise,
calibration is only a staking-coherence layer.

    include("current_development/split_market_pillar/r12_per_line_bias_edge.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data        = BayesianFootball.Data

# Loaders for the saved experiments' types (split cells need l02; src dp_* need none).
include("current_development/split_market_pillar/l02_split_market_poisson.jl")

# ==========================================
# 1. LOAD saved grid + build Betfair eval ds1
# ==========================================
folders     = Experiments.list_experiments("double_poisson_market_grid"; data_dir="./data")
all_results = Experiments.load_experiments(folders)
by_name     = Dict(r.config.name => r for r in all_results)
println("[INFO] Loaded $(length(all_results)) experiments: ", join(keys(by_name), ", "))

ds   = Data.load_datastore_cached(Data.Ireland())
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)



results = Evaluation.evaluate_experiments([Evaluation.LPD()], all_results, ds1)
Evaluation.display_summary_metric(results, :lpd)

#=
Row │ model           lpd_overall_model_lpd  lpd_overall_market_lpd  lpd_overall_diff_lpd  lpd_overall_elpd  lpd_overall_n_obs
     │ String          Float64                Float64                 Float64               Float64           Int64
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ dp_nomarket                 -0.569126               -0.591852             0.0227262          -1824.05               3205
   2 │ dp_old_mw100                -0.567962               -0.591852             0.0238898          -1820.32               3205
   3 │ dp_old_mw50                 -0.570618               -0.591852             0.021234           -1828.83               3205
   4 │ dp_split_lw0                -0.581333               -0.591852             0.0105188          -1863.17               3205
   5 │ dp_split_lw100              -0.567301               -0.591852             0.0245509          -1818.2                3205
   6 │ dp_split_lw25               -0.579276               -0.591852             0.0125758          -1856.58               3205
   7 │ dp_split_lw50               -0.576312               -0.591852             0.0155404          -1847.08               3205
=#


results = Evaluation.evaluate_experiments([Evaluation.LPD(:over_25)], all_results, ds1)
  Evaluation.display_summary_metric(results, :lpd)

selections = [
    :home, :draw, :away,
    :btts_yes, :btts_no,
    :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
    :over_35, :under_35, :over_45, :under_45,
]
_order = ["dp_nomarket", "dp_old_mw50", "dp_old_mw100",
          "dp_split_lw0", "dp_split_lw25", "dp_split_lw50", "dp_split_lw100"]
_order = filter(m -> haskey(by_name, m), _order)

# ==========================================
# 2. Parts 1–2: raw per-line bias + deviation (own per-match join, posterior-mean p)
# ==========================================
function per_line_bias_dev(exp, ds1, selections)
    latents = Experiments.extract_oos_predictions(ds1, exp)
    ppd = Predictions.model_inference(latents)
    mf  = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
    select!(mf, :match_id, :market_name, :market_line, :selection, :prob_model)
    adf = innerjoin(ds1.odds, mf, on = [:match_id, :market_name, :market_line, :selection])
    dropmissing!(adf, [:prob_fair_close, :is_winner])
    adf.spread = Float64.(adf.prob_model) .- Float64.(adf.prob_fair_close)
    sel_sym = Symbol.(adf.selection)

    out = Dict{Symbol, NamedTuple}()
    for s in selections
        sp = adf.spread[sel_sym .== s]
        n  = length(sp)
        n < 3 && continue
        b  = mean(sp); sd = std(sp); se = sd / sqrt(n)
        out[s] = (n = n, market_p = round(mean(Float64.(adf.prob_fair_close[sel_sym .== s])), digits=3),
                  bias = round(b, digits=4), t = round(b / se, digits=2), dev_std = round(sd, digits=4))
    end
    return out
end

# ==========================================
# 3. Part 3: GLMEdge per line (reuse the tested per-selection metric)
# ==========================================
metric = Evaluation.AbstractScoringRule[Evaluation.GLMEdge(s) for s in selections]
glm_eval = Evaluation.evaluate_experiments(metric, all_results, ds1)

function _glm(model, s)
    cc = "glmedge_$(s)_spread_fair_coef"; pp = "glmedge_$(s)_spread_fair_p_value"
    (cc in names(glm_eval) && pp in names(glm_eval)) || return (NaN, NaN)
    r = glm_eval[glm_eval.model .== model, :]
    isempty(r) && return (NaN, NaN)
    (ismissing(r[1, cc]) ? NaN : round(Float64(r[1, cc]), digits=3),
     ismissing(r[1, pp]) ? NaN : round(Float64(r[1, pp]), digits=3))
end

# ==========================================
# 4. OUTPUT — per model: line × {market_p, bias(t), dev_std, GLMEdge(coef,p)}
# ==========================================
for m in _order
    bd = per_line_bias_dev(by_name[m], ds1, selections)
    tbl = DataFrame(line=Symbol[], n=Int[], market_p=Float64[],
                    bias=Float64[], t=Float64[], dev_std=Float64[],
                    glm_coef=Float64[], glm_p=Float64[])
    for s in selections
        haskey(bd, s) || continue
        b = bd[s]; (gc, gp) = _glm(m, s)
        push!(tbl, (s, b.n, b.market_p, b.bias, b.t, b.dev_std, gc, gp))
    end
    println("\n", "="^78)
    println("MODEL: $m")
    println("  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred")
    println("  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN")
    println("="^78)
    show(tbl; allrows=true, allcols=true, truncate=0)
    println()
end




#=
==============================================================================                                                                                                                                                                                                 
MODEL: dp_nomarket                                                                                                                                                                                                                                                             
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                           
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                      
==============================================================================                                                                                                                                                                                                 
15×8 DataFrame                                                                                                                                                                                                                                                                 
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                   
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                 
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                
   1 │ home        273     0.432   0.0128     1.74   0.1218    -0.62     0.704                                                                                                                                                                                                 
   2 │ draw        273     0.268  -0.0226    -9.66   0.0387    -4.309    0.502                                                                                                                                                                                                 
   3 │ away        273     0.3     0.0098     1.53   0.1057     1.696    0.407                                                                                                                                                                                                 
   4 │ btts_yes    202     0.496   0.0376    10.5    0.0509     8.825    0.039                                                                                                                                                                                                 
   5 │ btts_no     202     0.504  -0.0376   -10.5    0.0509     8.826    0.039                                                                                                                                                                                                 
   6 │ over_05     179     0.923   0.0073     2.98   0.033    -21.914    0.298                                                                                                                                                                                                 
   7 │ under_05    159     0.084  -0.015     -7.75   0.0244   -19.278    0.365                                                                                                                                                                                                 
   8 │ over_15     212     0.713   0.0386     9.91   0.0568     1.393    0.755                                                                                                                                                                                                 
   9 │ under_15    212     0.287  -0.0386    -9.91   0.0568     1.391    0.755                                                                                                                                                                                                 
  10 │ over_25     247     0.462   0.0553    12.01   0.0723     3.152    0.245                                                                                                                                                                                                 
  11 │ under_25    247     0.538  -0.0553   -12.01   0.0723     3.151    0.245                                                                                                                                                                                                 
  12 │ over_35     203     0.246   0.0545    12.46   0.0624     1.701    0.628                                                                                                                                                                                                 
  13 │ under_35    203     0.754  -0.0545   -12.46   0.0624     1.7      0.628                                                                                                                                                                                                 
  14 │ over_45      90     0.116   0.0322     7.33   0.0416    -1.012    0.932                                                                                                                                                                                                 
  15 │ under_45    116     0.907  -0.0556    -9.85   0.0608     0.001    1.0
=#


#=
==============================================================================                                                                                                                                                                                                 
MODEL: dp_old_mw50                                                                                                                                                                                                                                                             
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                           
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                      
==============================================================================                                                                                                                                                                                                 
15×8 DataFrame                                                                                                                                                                                                                                                                 
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                   
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                 
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                
   1 │ home        273     0.432  -0.0045    -0.62   0.1203    -1.283    0.475                                                                                                                                                                                                 
   2 │ draw        273     0.268  -0.0049    -1.75   0.0463    -0.104    0.982                                                                                                                                                                                                 
   3 │ away        273     0.3     0.0065     1.02   0.1049     2.061    0.338                                                                                                                                                                                                 
   4 │ btts_yes    202     0.496  -0.0146    -3.85   0.0537     3.478    0.352                                                                                                                                                                                                 
   5 │ btts_no     202     0.504   0.0108     2.78   0.0554     4.372    0.248                                                                                                                                                                                                 
   6 │ over_05     179     0.923  -0.0173    -4.79   0.0483     0.027    0.997                                                                                                                                                                                                 
   7 │ under_05    159     0.084   0.008      2.39   0.0421     2.01     0.771                                                                                                                                                                                                 
   8 │ over_15     212     0.713  -0.0116    -2.62   0.0643    -0.842    0.817                                                                                                                                                                                                 
   9 │ under_15    212     0.287   0.008      1.8    0.0643    -0.828    0.829                                                                                                                                                                                                 
  10 │ over_25     247     0.462  -0.0069    -1.49   0.073      4.437    0.165                                                                                                                                                                                                 
  11 │ under_25    247     0.538   0.0041     0.85   0.0757     3.519    0.242                                                                                                                                                                                                 
  12 │ over_35     203     0.246   0.0        0.01   0.0641     1.144    0.783                                                                                                                                                                                                 
  13 │ under_35    203     0.754  -0.0029    -0.59   0.0695    -0.662    0.862                                                                                                                                                                                                 
  14 │ over_45      90     0.116  -0.0035    -0.7    0.0476   -13.361    0.45                                                                                                                                                                                                  
  15 │ under_45    116     0.907  -0.0247    -3.61   0.0738   -13.296    0.365
=#

#=
==============================================================================                                                                                                                                                                                                 
MODEL: dp_old_mw100                                                                                                                                                                                                                                                            
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                           
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                      
==============================================================================                                                                                                                                                                                                 
15×8 DataFrame                                                                                                                                                                                                                                                                 
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                   
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                 
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                
   1 │ home        273     0.432  -0.0055    -0.82   0.1101    -0.476    0.8                                                                                                                                                                                                   
   2 │ draw        273     0.268  -0.0009    -0.32   0.0451    -3.018    0.467                                                                                                                                                                                                 
   3 │ away        273     0.3     0.0064     1.1    0.0958     2.698    0.269                                                                                                                                                                                                 
   4 │ btts_yes    202     0.496  -0.0142    -4.02   0.0502     7.587    0.131                                                                                                                                                                                                 
   5 │ btts_no     202     0.504   0.0142     4.02   0.0502     7.588    0.131                                                                                                                                                                                                 
   6 │ over_05     179     0.923  -0.0153    -4.28   0.0479     1.812    0.757                                                                                                                                                                                                 
   7 │ under_05    159     0.084   0.0074     2.39   0.0389     2.834    0.642                                                                                                                                                                                                 
   8 │ over_15     212     0.713  -0.0133    -3.2    0.0608     1.738    0.629                                                                                                                                                                                                 
   9 │ under_15    212     0.287   0.0133     3.2    0.0608     1.738    0.629                                                                                                                                                                                                 
  10 │ over_25     247     0.462  -0.0143    -3.23   0.0696     3.633    0.279                                                                                                                                                                                                 
  11 │ under_25    247     0.538   0.0143     3.23   0.0696     3.633    0.279                                                                                                                                                                                                 
  12 │ over_35     203     0.246  -0.0096    -2.41   0.0571    -3.317    0.522                                                                                                                                                                                                 
  13 │ under_35    203     0.754   0.0096     2.41   0.0571    -3.317    0.522                                                                                                                                                                                                 
  14 │ over_45      90     0.116  -0.0108    -2.77   0.037    -13.122    0.509                                                                                                                                                                                                 
  15 │ under_45    116     0.907  -0.0122    -2.32   0.0565   -12.767    0.478
=#



#=
==============================================================================                                                                                                                                                                                                 
MODEL: dp_split_lw0                                                                                                                                                                                                                                                            
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                           
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                      
==============================================================================                                                                                                                                                                                                 
15×8 DataFrame                                                                                                                                                                                                                                                                 
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                   
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                 
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                
   1 │ home        273     0.432   0.0057     1.31   0.0723     0.078    0.972                                                                                                                                                                                                 
   2 │ draw        273     0.268  -0.0203    -7.66   0.0438    -0.944    0.824                                                                                                                                                                                                 
   3 │ away        273     0.3     0.0145     4.05   0.0593     3.474    0.231                                                                                                                                                                                                 
   4 │ btts_yes    202     0.496   0.0174     2.59   0.0958     0.797    0.671                                                                                                                                                                                                 
   5 │ btts_no     202     0.504  -0.0175    -2.6    0.0957     0.798    0.67                                                                                                                                                                                                  
   6 │ over_05     179     0.923  -0.0005    -0.13   0.0505   -10.757    0.247                                                                                                                                                                                                 
   7 │ under_05    159     0.084  -0.0074    -2.02   0.0459   -12.838    0.185                                                                                                                                                                                                 
   8 │ over_15     212     0.713   0.0262     3.58   0.1065    -1.995    0.328                                                                                                                                                                                                 
   9 │ under_15    212     0.287  -0.0262    -3.59   0.1065    -1.99     0.329                                                                                                                                                                                                 
  10 │ over_25     247     0.462   0.0401     4.8    0.1315     1.593    0.231                                                                                                                                                                                                 
  11 │ under_25    247     0.538  -0.0402    -4.8    0.1314     1.594    0.231                                                                                                                                                                                                 
  12 │ over_35     203     0.246   0.0465     5.91   0.1121     0.155    0.934                                                                                                                                                                                                 
  13 │ under_35    203     0.754  -0.0465    -5.92   0.1121     0.161    0.932                                                                                                                                                                                                 
  14 │ over_45      90     0.116   0.0304     3.48   0.083     -4.334    0.435                                                                                                                                                                                                 
  15 │ under_45    116     0.907  -0.0559    -6.61   0.0911    -4.793    0.359
=#

#=
==============================================================================                                                                                                                                                                                                 
MODEL: dp_split_lw25                                                                                                                                                                                                                                                           
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                           
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                      
==============================================================================                                                                                                                                                                                                 
15×8 DataFrame                                                                                                                                                                                                                                                                 
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                   
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                 
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                
   1 │ home        273     0.432  -0.0025    -0.53   0.0788    -0.894    0.682                                                                                                                                                                                                 
   2 │ draw        273     0.268  -0.005     -1.53   0.0536    -3.711    0.236                                                                                                                                                                                                 
   3 │ away        273     0.3     0.0075     1.86   0.0664     3.253    0.24                                                                                                                                                                                                  
   4 │ btts_yes    202     0.496  -0.0068    -1.09   0.0885     0.621    0.764                                                                                                                                                                                                 
   5 │ btts_no     202     0.504   0.0068     1.09   0.0885     0.621    0.764                                                                                                                                                                                                 
   6 │ over_05     179     0.923  -0.0163    -3.44   0.0635    -2.843    0.61                                                                                                                                                                                                  
   7 │ under_05    159     0.084   0.0088     1.92   0.0581    -3.392    0.566                                                                                                                                                                                                 
   8 │ over_15     212     0.713  -0.0004    -0.06   0.1018    -0.913    0.64                                                                                                                                                                                                  
   9 │ under_15    212     0.287   0.0004     0.06   0.1018    -0.913    0.64                                                                                                                                                                                                  
  10 │ over_25     247     0.462   0.0077     1.02   0.1191     1.419    0.342                                                                                                                                                                                                 
  11 │ under_25    247     0.538  -0.0077    -1.02   0.1191     1.419    0.342                                                                                                                                                                                                 
  12 │ over_35     203     0.246   0.0166     2.4    0.0985    -0.621    0.783                                                                                                                                                                                                 
  13 │ under_35    203     0.754  -0.0166    -2.4    0.0985    -0.621    0.783                                                                                                                                                                                                 
  14 │ over_45      90     0.116   0.0092     1.29   0.0681    -5.415    0.449                                                                                                                                                                                                 
  15 │ under_45    116     0.907  -0.0338    -4.64   0.0785    -4.959    0.458
=#




#=
==============================================================================                                                                                                                                                                                                 
MODEL: dp_split_lw50                                                                                                                                                                                                                                                           
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                           
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                      
==============================================================================                                                                                                                                                                                                 
15×8 DataFrame                                                                                                                                                                                                                                                                 
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                   
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                 
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                
   1 │ home        273     0.432  -0.0093    -1.53   0.1002    -0.08     0.965                                                                                                                                                                                                 
   2 │ draw        273     0.268  -0.0026    -0.86   0.0494    -4.44     0.225                                                                                                                                                                                                 
   3 │ away        273     0.3     0.0084     1.66   0.0839     4.559    0.055                                                                                                                                                                                                 
   4 │ btts_yes    202     0.496  -0.0191    -3.54   0.0766     1.325    0.598                                                                                                                                                                                                 
   5 │ btts_no     202     0.504   0.0156     3.05   0.0729     0.853    0.747                                                                                                                                                                                                 
   6 │ over_05     179     0.923  -0.0234    -5.29   0.0593    -5.065    0.406                                                                                                                                                                                                 
   7 │ under_05    159     0.084   0.0125     3.16   0.0498    -3.86     0.558                                                                                                                                                                                                 
   8 │ over_15     212     0.713  -0.0124    -2.1    0.0861    -1.595    0.516                                                                                                                                                                                                 
   9 │ under_15    212     0.287   0.0101     1.75   0.0844    -2.019    0.433                                                                                                                                                                                                 
  10 │ over_25     247     0.462  -0.01      -1.62   0.097      1.127    0.563                                                                                                                                                                                                 
  11 │ under_25    247     0.538   0.0063     1.04   0.096     -0.153    0.938
  12 │ over_35     203     0.246   0.0        0.0    0.082     -1.958    0.52
  13 │ under_35    203     0.754  -0.0037    -0.63   0.0833    -2.874    0.337
  14 │ over_45      90     0.116   0.0024     0.41   0.0571    -4.328    0.625
  15 │ under_45    116     0.907  -0.03      -4.5    0.0719    -6.511    0.391
=#


#=
==============================================================================
MODEL: dp_split_lw100
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN
==============================================================================
15×8 DataFrame
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p   
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64 
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ home        273     0.432  -0.0063    -0.82   0.1264    -0.964    0.568
   2 │ draw        273     0.268  -0.0033    -1.36   0.0395    -5.894    0.333
   3 │ away        273     0.3     0.0094     1.43   0.109      1.271    0.556
   4 │ btts_yes    202     0.496  -0.0161    -5.58   0.0411    10.406    0.093
   5 │ btts_no     202     0.504   0.0161     5.57   0.041     10.272    0.099
   6 │ over_05     179     0.923  -0.0124    -4.97   0.0333     4.697    0.721
   7 │ under_05    159     0.084   0.0046     2.37   0.0245     8.031    0.524
   8 │ over_15     212     0.713  -0.0148    -4.51   0.0479     4.663    0.365
   9 │ under_15    212     0.287   0.0146     4.4    0.0484     4.712    0.356
  10 │ over_25     247     0.462  -0.0164    -4.43   0.0582     4.119    0.333
  11 │ under_25    247     0.538   0.0162     4.35   0.0586     4.31     0.307
  12 │ over_35     203     0.246  -0.0119    -3.45   0.0493     1.813    0.754
  13 │ under_35    203     0.754   0.0117     3.35   0.0498     2.528    0.653
  14 │ over_45      90     0.116  -0.0142    -4.28   0.0314    21.771    0.39
  15 │ under_45    116     0.907  -0.0102    -1.9    0.058      3.382    0.823
=#


println("""

$("="^78)
READ (per the confirmed philosophy):
 • CALIBRATE where |t|≫2  (systematic per-line skew = miscalibration, not edge).
 • EDGE lives where dev_std is large AND glm_coef>0 with p<0.1 (deviations predict the outcome).
 • The build is justified on lines that have BOTH a systematic bias to remove AND surviving edge in
   the deviations. Lines already centred (|t|≈0) need no calibration; bias+glm≈0 = noise (coherence only).
 • Expect dp_nomarket to show the LARGEST biases (no market anchor); anchored cells (lw*) smaller.
$("="^78)
""")
