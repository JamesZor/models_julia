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
