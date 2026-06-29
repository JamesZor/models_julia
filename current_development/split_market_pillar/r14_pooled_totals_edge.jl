#=
r14 — POOLED totals-ladder GLMEdge. Resolves the r13 power problem: per-line GLMEdge at n≈200–250
can't clear significance even if a real (small) totals edge exists, so r13's per-cell p>0.1 was
UNDER-POWERED, not a null (the totals coefs were consistently POSITIVE across the ladder + models, same
direction as the backtest P/L). This stacks the over-ladder into ONE regression per model:

    is_over ~ line_FE + market_logit + spread          (Binomial / Logit)

  - line_FE      : per-line baseline (over_05 ≫ over_35), so we pool the SLOPE not the level.
  - market_logit : the de-vigged market view = the GLMEdge control ("info beyond market").
  - spread       : model_p − market_p (prob scale, = r12/r13 spread). Its coefficient is the pooled
                   edge across the whole ladder. >0 = model's per-match deviation predicts the over
                   beyond the market.

POWER vs HONESTY: pooling 4 lines ×~220 matches ≈ 900 rows lifts power, BUT the 4 ladder rows of one
match are the SAME goal total at different thresholds → correlated. Naive SE treats them as independent
(pseudo-replication) and over-states significance. So we report BOTH:
  - p_naive    : GLM model-based SE (optimistic — ignores within-match correlation).
  - p_cluster  : Huber–White SE CLUSTERED on match_id (CR1 small-sample adj). THIS is the honest test.
A real edge shows spread coef>0 with p_cluster<~0.1; if only p_naive is small, it was pseudo-replication.

Two ladders: CORE = 1.5/2.5/3.5 (liquid, balanced); FULL adds the 0.5 line (extreme p, can destabilise).
Run for OVER and UNDER sides; unders are the algebraic mirror of overs (same coef) — a symmetry check.

Server REPL: git pull, restart, then
    include("current_development/split_market_pillar/r14_pooled_totals_edge.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using LinearAlgebra
using Distributions
using GLM

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data        = BayesianFootball.Data

include("current_development/split_market_pillar/l02_split_market_poisson.jl")
include("current_development/split_market_pillar/l03_local_intensity_poisson.jl")

# ==========================================
# 1. LOAD both grids + Betfair eval ds1 (= r13)
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

_order = ["dp_nomarket", "dp_old_mw50", "dp_old_mw100",
          "dp_split_lw0", "dp_split_lw25", "dp_split_lw50", "dp_split_lw100",
          "li_sup_only", "li_smile50", "li_smile100", "li_smile_only"]
_order = filter(m -> haskey(by_name, m), _order)

# NOTE on unders: under_X is the exact algebraic complement of over_X (outcome flips, spread negates,
# market logit negates), and a logit is invariant under that joint sign-flip → the UNDER ladder returns
# the SAME spread coefficient and p as the OVER ladder (up to tiny coverage differences in the odds
# join). Included as a symmetry CHECK, not as new information; the real power gain is cross-league.
LADDER_SPECS = [
    ("OVER  CORE (over 1.5/2.5/3.5)",   [:over_15, :over_25, :over_35]),
    ("OVER  FULL (+ over_05)",          [:over_05, :over_15, :over_25, :over_35]),
    ("UNDER CORE (under 1.5/2.5/3.5)",  [:under_15, :under_25, :under_35]),
    ("UNDER FULL (+ under_05)",         [:under_05, :under_15, :under_25, :under_35]),
]

# ==========================================
# 2. per-MATCH spread frame for a model (one row per match × over-line)
# ==========================================
function per_match_overs(exp, ds1, lines)
    latents = Experiments.extract_oos_predictions(ds1, exp)
    ppd = Predictions.model_inference(latents)
    mf  = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
    select!(mf, :match_id, :market_name, :market_line, :selection, :prob_model)
    adf = innerjoin(ds1.odds, mf, on = [:match_id, :market_name, :market_line, :selection])
    dropmissing!(adf, [:prob_fair_close, :is_winner])
    sel = Symbol.(adf.selection)
    keep = [s in lines for s in sel]
    d = DataFrame(
        match_id = adf.match_id[keep],
        line     = String.(string.(sel[keep])),
        y        = Float64.(adf.is_winner[keep]),
        mkt_p    = clamp.(Float64.(adf.prob_fair_close[keep]), 1e-6, 1 - 1e-6),
        spread   = Float64.(adf.prob_model[keep]) .- Float64.(adf.prob_fair_close[keep]),
    )
    d.mkt = log.(d.mkt_p ./ (1 .- d.mkt_p))
    return d
end

# ==========================================
# 3. cluster-robust (CR1) SE for a fitted GLM, clustered on `clusters`
# ==========================================
function cluster_robust_se(m, X, y, clusters)
    p = predict(m)
    e = y .- p                       # score residual (y − p̂)
    w = p .* (1 .- p)
    A = Symmetric(X' * (w .* X))      # expected information (bread⁻¹)
    Ainv = inv(A)
    k = size(X, 2)
    B = zeros(k, k)
    for g in unique(clusters)
        idx = clusters .== g
        s = X[idx, :]' * e[idx]       # summed score over the cluster
        B .+= s * s'
    end
    G = length(unique(clusters)); n = length(y)
    adj = (G / (G - 1)) * ((n - 1) / (n - k))   # CR1 small-sample correction
    V = Ainv * B * Ainv .* adj
    return sqrt.(max.(diag(V), 0.0))
end

# ==========================================
# 4. pooled fit for one model × ladder
# ==========================================
function pooled_edge(d::DataFrame)
    nrow(d) < 50 && return nothing
    m  = glm(@formula(y ~ line + mkt + spread), d, Binomial(), LogitLink())
    nm = coefnames(m); r = findfirst(==("spread"), nm)
    r === nothing && return nothing
    β  = coef(m)[r]
    se_n = stderror(m)[r]
    X  = modelmatrix(m)
    se_c = cluster_robust_se(m, X, d.y, d.match_id)[r]
    z_n, z_c = β / se_n, β / se_c
    p_n = 2 * ccdf(Normal(), abs(z_n)); p_c = 2 * ccdf(Normal(), abs(z_c))
    (n = nrow(d), n_match = length(unique(d.match_id)), coef = β,
     z_naive = z_n, p_naive = p_n, z_cluster = z_c, p_cluster = p_c)
end

# ==========================================
# 5. RUN — both ladders, all models
# ==========================================
for (label, lines) in LADDER_SPECS
    side = startswith(label, "UNDER") ? "is_under" : "is_over"
    println("\n", "="^96)
    println("POOLED TOTALS-LADDER GLMEdge — $label")
    println("  $side ~ line_FE + market_logit + spread ;  coef>0 & p_cluster<~0.1 ⇒ real pooled totals edge")
    println("  (p_naive ignores within-match correlation across the ladder — p_cluster on match_id is honest)")
    println("="^96)
    tbl = DataFrame(model=String[], n=Int[], n_match=Int[], spread_coef=Float64[],
                    z_naive=Float64[], p_naive=Float64[], z_cluster=Float64[], p_cluster=Float64[])
    for mname in _order
        d = per_match_overs(by_name[mname], ds1, lines)
        res = pooled_edge(d)
        res === nothing && continue
        push!(tbl, (mname, res.n, res.n_match, round(res.coef, digits=3),
                    round(res.z_naive, digits=2), round(res.p_naive, digits=4),
                    round(res.z_cluster, digits=2), round(res.p_cluster, digits=4)))
    end
    sort!(tbl, :p_cluster)
    show(tbl; allrows=true, allcols=true, truncate=0); println()
end


#=
================================================================================================                                                                                                                                                                                                                            
POOLED TOTALS-LADDER GLMEdge — CORE (over 1.5/2.5/3.5)                                                                                                                                                                                                                                                                      
  is_over ~ line_FE + market_logit + spread ;  coef>0 & p_cluster<~0.1 ⇒ real pooled totals edge                                                                                                                                                                                                                            
  (p_naive ignores within-match correlation across the ladder — p_cluster on match_id is honest)                                                                                                                                                                                                                            
================================================================================================
11×8 DataFrame
 Row │ model           n      n_match  spread_coef  z_naive  p_naive  z_cluster  p_cluster 
     │ String          Int64  Int64    Float64      Float64  Float64  Float64    Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────
   1 │ li_smile50        662      270        3.154     1.83   0.067        1.27     0.2027
   2 │ li_smile100       662      270        3.08      1.43   0.1524       1.02     0.3085
   3 │ dp_split_lw100    662      270        3.379     1.22   0.221        0.89     0.3759
   4 │ dp_nomarket       662      270        2.229     1.18   0.2398       0.85     0.3978
   5 │ li_smile_only     662      270        2.442     1.04   0.2995       0.73     0.4668
   6 │ dp_old_mw50       662      270        1.704     0.85   0.3942       0.64     0.5218
   7 │ dp_old_mw100      662      270        1.451     0.67   0.501        0.48     0.6342
   8 │ dp_split_lw0      662      270        0.381     0.41   0.6852       0.29     0.7745
   9 │ li_sup_only       662      270        0.347     0.31   0.7601       0.22     0.8277
  10 │ dp_split_lw50     662      270       -0.382    -0.28   0.7763      -0.2      0.8392
  11 │ dp_split_lw25     662      270        0.263     0.25   0.7989       0.18     0.8608
=#




#=
================================================================================================
POOLED TOTALS-LADDER GLMEdge — FULL (+ over_05)
  is_over ~ line_FE + market_logit + spread ;  coef>0 & p_cluster<~0.1 ⇒ real pooled totals edge
  (p_naive ignores within-match correlation across the ladder — p_cluster on match_id is honest)
================================================================================================
11×8 DataFrame
 Row │ model           n      n_match  spread_coef  z_naive  p_naive  z_cluster  p_cluster 
     │ String          Int64  Int64    Float64      Float64  Float64  Float64    Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────
   1 │ li_smile50        841      273        2.882     1.97   0.0487       1.47     0.1412
   2 │ li_smile100       841      273        2.782     1.42   0.1569       0.98     0.3258
   3 │ dp_split_lw100    841      273        3.286     1.27   0.2044       0.97     0.3311
   4 │ dp_nomarket       841      273        1.986     1.08   0.2805       0.79     0.4296
   5 │ li_smile_only     841      273        2.227     1.05   0.2954       0.71     0.4787
   6 │ dp_old_mw50       841      273        1.485     0.8    0.4215       0.56     0.5759
   7 │ dp_old_mw100      841      273        1.489     0.76   0.4456       0.53     0.5947
   8 │ dp_split_lw50     841      273       -0.553    -0.46   0.6433      -0.35     0.7298
   9 │ dp_split_lw0      841      273        0.237     0.26   0.7946       0.19     0.8528
  10 │ dp_split_lw25     841      273        0.144     0.15   0.883        0.1      0.92
  11 │ li_sup_only       841      273        0.078     0.07   0.9423       0.05     0.958
=#



#=
================================================================================================                                                                                                                                                                                                                            
POOLED TOTALS-LADDER GLMEdge — UNDER CORE (under 1.5/2.5/3.5)                                                                                                                                                                                                                                                               
  is_under ~ line_FE + market_logit + spread ;  coef>0 & p_cluster<~0.1 ⇒ real pooled totals edge                                                                                                                                                                                                                           
  (p_naive ignores within-match correlation across the ladder — p_cluster on match_id is honest)                                                                                                                                                                                                                            
================================================================================================
11×8 DataFrame
 Row │ model           n      n_match  spread_coef  z_naive  p_naive  z_cluster  p_cluster 
     │ String          Int64  Int64    Float64      Float64  Float64  Float64    Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────
   1 │ li_smile50        662      270        3.154     1.83   0.067        1.27     0.2027
   2 │ li_smile100       662      270        3.08      1.43   0.1524       1.02     0.3085
   3 │ dp_split_lw100    662      270        3.643     1.33   0.1827       0.95     0.3416
   4 │ dp_nomarket       662      270        2.228     1.18   0.2398       0.85     0.3979
   5 │ li_smile_only     662      270        2.442     1.04   0.2995       0.73     0.4668
   6 │ dp_split_lw50     662      270       -1.321    -0.96   0.336       -0.7      0.4821
   7 │ dp_old_mw100      662      270        1.451     0.67   0.501        0.48     0.6342
   8 │ dp_old_mw50       662      270        1.013     0.52   0.6011       0.4      0.689
   9 │ dp_split_lw0      662      270        0.385     0.41   0.6822       0.29     0.7723
  10 │ li_sup_only       662      270        0.347     0.31   0.7601       0.22     0.8277
  11 │ dp_split_lw25     662      270        0.263     0.25   0.799        0.18     0.8608
=#
 

#=
================================================================================================
POOLED TOTALS-LADDER GLMEdge — UNDER FULL (+ under_05)
  is_under ~ line_FE + market_logit + spread ;  coef>0 & p_cluster<~0.1 ⇒ real pooled totals edge
  (p_naive ignores within-match correlation across the ladder — p_cluster on match_id is honest)
================================================================================================
11×8 DataFrame
 Row │ model           n      n_match  spread_coef  z_naive  p_naive  z_cluster  p_cluster 
     │ String          Int64  Int64    Float64      Float64  Float64  Float64    Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────
   1 │ li_smile50        821      273        2.85      1.95   0.0511       1.46     0.1449
   2 │ dp_split_lw100    821      273        3.52      1.37   0.1701       1.04     0.2999
   3 │ li_smile100       821      273        2.759     1.4    0.1602       0.98     0.3295
   4 │ dp_nomarket       821      273        1.979     1.08   0.2822       0.79     0.4314
   5 │ dp_split_lw50     821      273       -1.352    -1.04   0.2981      -0.75     0.4529
   6 │ li_smile_only     821      273        2.2       1.03   0.3012       0.7      0.4841
   7 │ dp_old_mw100      821      273        1.462     0.75   0.4543       0.52     0.602
   8 │ dp_old_mw50       821      273        1.041     0.57   0.5662       0.4      0.6869
   9 │ dp_split_lw0      821      273        0.238     0.26   0.7957       0.18     0.8547
  10 │ dp_split_lw25     821      273        0.133     0.13   0.893        0.09     0.9275
  11 │ li_sup_only       821      273        0.068     0.06   0.9501       0.04     0.9642
=#


println("""

$("="^96)
READ:
 • spread_coef>0 & p_cluster<~0.1  ⇒ the model's per-match totals deviation predicts the over BEYOND the
   market, pooled across the ladder = a REAL (if small) totals edge that single lines were too thin to show.
 • p_naive small but p_cluster large ⇒ the apparent significance was pseudo-replication (4 correlated
   ladder rows per match), NOT edge. Trust p_cluster.
 • Compare li_smile* (smile pillar) vs dp_split_lw* vs li_sup_only: if the smile cells carry the pooled
   totals edge AND we already saw the BTTS edge (r13 li_smile50 p=0.01), the smile earns keep on BOTH
   derivative families, not just BTTS.
 • UNDER tables should match the OVER tables (same spread coef/p) — that's the expected algebraic mirror,
   a sanity check that the join + sign handling are correct, NOT a second independent confirmation.
$("="^96)
""")
