#=
r13 — PER-LINE BIAS-vs-EDGE diagnostic across the SMILE grid + the r05 grid together.

Same diagnostic as r12, but instead of only the 7 dp_* cells from double_poisson_market_grid, this
loads BOTH saved grids (smile cells from double_poisson_smile_grid + the r05 cells) so the li_smile* /
li_sup_only models sit in the SAME per-line table as dp_nomarket / dp_old_* / dp_split_lw*. The r10
summary (GLMEdge / LogLoss / Kelly tearsheet) showed the smile cells top BOTH summary tables but the
per-MARKET-FAMILY story (smile helps totals/BTTS, hurts 1X2) was only visible in the backtest P/L.
This decomposes it on PROPER scoring, no retrain.

PHILOSOPHY (confirmed, = r12): single-bet growth is LINEAR in E[p] (unified_kelly §6), so the edge is
the per-match mean deviation E[p]−p_market; posterior tails only SIZE the stake. Per line we want:

  1. SYSTEMATIC BIAS   mean(model_p − market_p), t = bias/se.  |t|≫2 = per-line skew (calibration target).
  2. DEVIATION BUDGET  std(model_p − market_p) = the size of the edge you bet from.
  3. DO DEVIATIONS WIN GLMEdge spread_fair_coef (+p): does (model−market) predict the outcome beyond the
       market? >0 & p<0.1 = the deviations carry real edge on that line; ≈0 = noise.

READ: per market FAMILY (1X2 vs totals vs BTTS), which cell is centred (|t|≈0) AND has surviving edge
(glm_coef>0, p<0.1). Expect the smile cells to win the totals/BTTS rows and the supremacy cell
(li_sup_only / dp_split_lw0) to be the cleanest on home/away.

Server REPL: git pull, restart, then
    include("current_development/split_market_pillar/r13_per_line_bias_edge_smile.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data        = BayesianFootball.Data

# Both saved grids' types must be defined to deserialize: l02 = SplitMarketDoublePoissonModel (r05
# split cells), l03 = LocalIntensitySmileDoublePoissonModel (smile cells). dp_* src models need none.
include("current_development/split_market_pillar/l02_split_market_poisson.jl")
include("current_development/split_market_pillar/l03_local_intensity_poisson.jl")

# ==========================================
# 1. LOAD both grids + build Betfair eval ds1
# ==========================================
smile_dir = "./data/double_poisson_smile_grid/"
r05_dir   = "./data/double_poisson_market_grid/"

_load(dir) = isdir(dir) ? Experiments.load_experiments(Experiments.list_experiments(dir; data_dir="")) : Any[]
all_results = convert(Vector{Experiments.ExperimentResults}, vcat(_load(smile_dir), _load(r05_dir)))
by_name     = Dict(r.config.name => r for r in all_results)
println("[INFO] Loaded $(length(all_results)) experiments: ", join(keys(by_name), ", "))

ds   = Data.load_datastore_cached(Data.Ireland())
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

selections = [
    :home, :draw, :away,
    :btts_yes, :btts_no,
    :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
    :over_35, :under_35, :over_45, :under_45,
]
# grouped by pillar family so the per-family read is easy to scan
_order = ["dp_nomarket", "dp_old_mw50", "dp_old_mw100",
          "dp_split_lw0", "dp_split_lw25", "dp_split_lw50", "dp_split_lw100",
          "li_sup_only", "li_smile50", "li_smile100", "li_smile_only"]
_order = filter(m -> haskey(by_name, m), _order)

# ==========================================
# 2. raw per-line bias + deviation (posterior-mean p, own per-match join)
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
# 3. GLMEdge per line (reuse the tested per-selection metric)
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


#=
==============================================================================                                                                                                                                                                                                                                              
MODEL: li_sup_only                                                                                                                                                                                                                                                                                                          
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                                                                        
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                                                                   
==============================================================================                                                                                                                                                                                                                                              
15×8 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                                                                
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                                                              
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                             
   1 │ home        273     0.432   0.0034     0.77   0.073      0.904    0.678                                                                                                                                                                                                                                              
   2 │ draw        273     0.268  -0.0161    -5.35   0.0499    -1.063    0.747                                                                                                                                                                                                                                              
   3 │ away        273     0.3     0.0128     3.49   0.0604     1.951    0.488                                                                                                                                                                                                                                              
   4 │ btts_yes    202     0.496   0.0147     2.15   0.0975     0.348    0.848                                                                                                                                                                                                                                              
   5 │ btts_no     202     0.504  -0.0147    -2.15   0.0975     0.348    0.848                                                                                                                                                                                                                                              
   6 │ over_05     179     0.923  -0.0348    -7.99   0.0583   -10.476    0.216                                                                                                                                                                                                                                              
   7 │ under_05    159     0.084   0.0266     6.38   0.0527   -12.545    0.168                                                                                                                                                                                                                                              
   8 │ over_15     212     0.713  -0.0057    -0.86   0.0963    -3.101    0.198                                                                                                                                                                                                                                              
   9 │ under_15    212     0.287   0.0057     0.86   0.0963    -3.101    0.198                                                                                                                                                                                                                                              
  10 │ over_25     247     0.462   0.048      6.65   0.1134     2.351    0.16                                                                                                                                                                                                                                               
  11 │ under_25    247     0.538  -0.048     -6.65   0.1134     2.351    0.16                                                                                                                                                                                                                                               
  12 │ over_35     203     0.246   0.1012    14.24   0.1013    -0.023    0.991                                                                                                                                                                                                                                              
  13 │ under_35    203     0.754  -0.1012   -14.24   0.1013    -0.023    0.991                                                                                                                                                                                                                                              
  14 │ over_45      90     0.116   0.1049    11.57   0.086     -4.11     0.413                                                                                                                                                                                                                                              
  15 │ under_45    116     0.907  -0.131    -15.04   0.0938    -3.833    0.414
=#



#=
==============================================================================                                                                                                                                                                                                                                              
MODEL: li_smile50                                                                                                                                                                                                                                                                                                           
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                                                                        
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                                                                   
==============================================================================                                                                                                                                                                                                                                              
15×8 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                                                                
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                                                              
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                             
   1 │ home        273     0.432  -0.0134    -1.49   0.1483    -1.641    0.251                                                                                                                                                                                                                                              
   2 │ draw        273     0.268  -0.0038    -0.95   0.0656    -2.325    0.371                                                                                                                                                                                                                                              
   3 │ away        273     0.3     0.016      2.04   0.1297     1.597    0.387                                                                                                                                                                                                                                              
   4 │ btts_yes    202     0.496   0.024      5.29   0.0644     7.235    0.01                                                                                                                                                                                                                                               
   5 │ btts_no     202     0.504  -0.0242    -5.38   0.0638     7.349    0.01                                                                                                                                                                                                                                               
   6 │ over_05     179     0.923  -0.0435    -7.7    0.0757     1.756    0.558                                                                                                                                                                                                                                              
   7 │ under_05    159     0.084   0.0355     6.73   0.0665     2.339    0.45                                                                                                                                                                                                                                               
   8 │ over_15     212     0.713  -0.031     -6.74   0.0669     3.167    0.181                                                                                                                                                                                                                                              
   9 │ under_15    212     0.287   0.031      6.74   0.0669     3.167    0.181                                                                                                                                                                                                                                              
  10 │ over_25     247     0.462  -0.0144    -3.97   0.0569     2.968    0.329                                                                                                                                                                                                                                              
  11 │ under_25    247     0.538   0.0144     3.97   0.0569     2.968    0.329                                                                                                                                                                                                                                              
  12 │ over_35     203     0.246   0.0009     0.27   0.0482     4.073    0.373                                                                                                                                                                                                                                              
  13 │ under_35    203     0.754  -0.0009    -0.27   0.0482     4.073    0.373                                                                                                                                                                                                                                              
  14 │ over_45      90     0.116   0.007      1.56   0.0423    -2.123    0.848                                                                                                                                                                                                                                              
  15 │ under_45    116     0.907  -0.0273    -5.25   0.0561    -3.584    0.734
=#

#=
==============================================================================                                                                                                                                                                                                                                              
MODEL: li_smile100                                                                                                                                                                                                                                                                                                          
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                                                                        
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                                                                   
==============================================================================                                                                                                                                                                                                                                              
15×8 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                                                                
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                                                              
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                             
   1 │ home        273     0.432  -0.0196    -2.06   0.1571    -1.1      0.469                                                                                                                                                                                                                                              
   2 │ draw        273     0.268  -0.0078    -2.66   0.0485    -1.578    0.658                                                                                                                                                                                                                                              
   3 │ away        273     0.3     0.0274     3.22   0.1403     1.585    0.391                                                                                                                                                                                                                                              
   4 │ btts_yes    202     0.496   0.0328     8.76   0.0532     5.963    0.057                                                                                                                                                                                                                                              
   5 │ btts_no     202     0.504  -0.0328    -8.76   0.0532     5.968    0.056                                                                                                                                                                                                                                              
   6 │ over_05     179     0.923  -0.0331    -9.37   0.0472    -0.501    0.935                                                                                                                                                                                                                                              
   7 │ under_05    159     0.084   0.0249     8.19   0.0384     1.515    0.8                                                                                                                                                                                                                                                
   8 │ over_15     212     0.713  -0.0234    -7.09   0.048      3.649    0.27                                                                                                                                                                                                                                               
   9 │ under_15    212     0.287   0.0234     7.09   0.048      3.649    0.27                                                                                                                                                                                                                                               
  10 │ over_25     247     0.462  -0.0099    -3.3    0.0472     3.525    0.308                                                                                                                                                                                                                                              
  11 │ under_25    247     0.538   0.0099     3.3    0.0472     3.525    0.308                                                                                                                                                                                                                                              
  12 │ over_35     203     0.246   0.0091     3.31   0.0393     1.366    0.794                                                                                                                                                                                                                                              
  13 │ under_35    203     0.754  -0.0091    -3.31   0.0393     1.366    0.794                                                                                                                                                                                                                                              
  14 │ over_45      90     0.116   0.0101     3.55   0.0271    17.769    0.248                                                                                                                                                                                                                                              
  15 │ under_45    116     0.907  -0.0291    -6.88   0.0456    11.777    0.379
=#

#=
==============================================================================                                                                                                                                                                                                                                              
MODEL: li_smile_only                                                                                                                                                                                                                                                                                                        
  bias=mean(model_p−market_p)  |t|≫2 ⇒ systematic skew to calibrate; |t|≈0 ⇒ centred                                                                                                                                                                                                                                        
  dev_std = match-level deviation budget (the edge);  glm_coef>0 & p<0.1 ⇒ deviations WIN                                                                                                                                                                                                                                   
==============================================================================                                                                                                                                                                                                                                              
15×8 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ line      n      market_p  bias     t        dev_std  glm_coef  glm_p                                                                                                                                                                                                                                                
     │ Symbol    Int64  Float64   Float64  Float64  Float64  Float64   Float64                                                                                                                                                                                                                                              
─────┼─────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                                                             
   1 │ home        273     0.432  -0.0353    -3.55   0.1644    -1.096    0.491
   2 │ draw        273     0.268  -0.0063    -2.18   0.0481    -1.29     0.722
   3 │ away        273     0.3     0.0416     4.66   0.1477     0.974    0.607
   4 │ btts_yes    202     0.496   0.035      9.5    0.0524     6.342    0.05
   5 │ btts_no     202     0.504  -0.035     -9.5    0.0524     6.342    0.05
   6 │ over_05     179     0.923  -0.0331    -9.36   0.0472    -0.545    0.929
   7 │ under_05    159     0.084   0.0248     8.19   0.0382     1.564    0.794
   8 │ over_15     212     0.713  -0.0213    -7.1    0.0437     2.448    0.512
   9 │ under_15    212     0.287   0.0213     7.1    0.0437     2.448    0.512
  10 │ over_25     247     0.462  -0.0082    -2.88   0.045      2.461    0.497
  11 │ under_25    247     0.538   0.0082     2.88   0.045      2.461    0.497
  12 │ over_35     203     0.246   0.0071     2.79   0.0362     3.093    0.596
  13 │ under_35    203     0.754  -0.0071    -2.79   0.0362     3.093    0.596
  14 │ over_45      90     0.116   0.0116     4.05   0.0271    13.426    0.365
  15 │ under_45    116     0.907  -0.0301    -7.22   0.0449     9.27     0.48
=#

# ==========================================
# 5. CROSS-MODEL PER-LINE GLMEdge matrix (line × model coef, p) — the per-family read in one grid
# ==========================================
println("\n", "="^78, "\nPER-LINE GLMEdge coef (p) — rows=line, cols=model  [>0 & p<0.1 ⇒ deviations win]\n", "="^78)
coef_tbl = DataFrame(line = selections)
for m in _order
    coef_tbl[!, Symbol(m)] = [(gc = _glm(m, s); isnan(gc[1]) ? missing : "$(gc[1]) ($(gc[2]))") for s in selections]
end
show(coef_tbl; allrows=true, allcols=true, truncate=0); println()

println("""

$("="^78)
READ (per the confirmed philosophy, now per FAMILY):
 • 1X2 (home/draw/away): pick the cell with |t|≈0 AND surviving glm_coef — expect li_sup_only /
   dp_split_lw0 (supremacy pillar) cleanest; smile cells distort this axis.
 • TOTALS (over/under_*) + BTTS: pick the smile cell that stays centred (|t|≈0) AND keeps glm_coef>0,
   p<0.1 — this is where r10 located the lift.
 • Final config = per-market-family routing (supremacy→1X2, smile→totals/BTTS), NOT one global weight.
 • Lines with bias but glm≈0 = noise (coherence-only calibration); already-centred lines need nothing.
$("="^78)
""")
