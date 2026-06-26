#=
PER-MARKET-LINE re-eval of the saved double-Poisson grid (r05 output).

WHY: r05's headline used Σ hurdle_G across all selections, which is misleading — a model with a
structural UNDER-bias will cash unders/btts_no over a short 2-season window and inflate the grouped
P/L, masquerading as skill. The proper-scoring lens (LogLoss) and CLV (GLMEdge) CANNOT be gamed that
way IF read PER LINE: a model that only wins unders by shading goals down will score BADLY on the
over lines. So we break every market line out individually instead of grouping.

This LOADS the 7 saved experiments from ./data/double_poisson_market_grid/ (no retrain) and runs
RQR + per-selection GLMEdge + per-selection LogLoss, then pivots into model×selection matrices.

Needs the evaluation-module fix (per-selection metrics now get distinct columns via
to_dataframe_row selection-suffix) — git pull + REPL restart first.

    include("current_development/split_market_pillar/r06_per_line_eval.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

# ==========================================
# 1. LOAD saved grid + build Betfair eval ds1
# ==========================================
save_dir = "double_poisson_market_grid"
folders  = Experiments.list_experiments(save_dir; data_dir="./data")
all_results = Experiments.load_experiments(folders)
println("[INFO] Loaded $(length(all_results)) experiments: ",
        join([r.config.name for r in all_results], ", "))

ds   = Data.load_datastore_cached(Data.Ireland())
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

# ==========================================
# 2. PER-LINE metric vector
# ==========================================
# Every market line evaluated on its OWN — this is the whole point.
selections = [
    :home, :draw, :away,
    :btts_yes, :btts_no,
    :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
    :over_35, :under_35, :over_45, :under_45, :over_55, :under_55,
]

metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
append!(metric, [Evaluation.LogLoss(s) for s in selections])
append!(metric, [Evaluation.GLMEdge(s) for s in selections])

metrics_eval = Evaluation.evaluate_experiments(metric, all_results, ds1)

# ==========================================
# 3. PIVOT into model × selection matrices
# ==========================================
_order = ["dp_nomarket", "dp_old_mw50", "dp_old_mw100",
          "dp_split_lw0", "dp_split_lw25", "dp_split_lw50", "dp_split_lw100"]
_sortmodels(df) = sort(df, :model, by = m -> something(findfirst(==(m), _order), 99))

# value getter: per-model column value (NaN/missing-safe)
function _col(df, model, colname)
    colname in names(df) || return NaN
    r = df[df.model .== model, colname]
    (isempty(r) || ismissing(r[1])) ? NaN : round(Float64(r[1]), digits=4)
end

# LogLoss diff (model - market): NEGATIVE = model beats the closing line on THIS line.
println("\n", "="^70, "\n📉 LogLoss diff (model−market) per line — NEGATIVE = beats market\n", "="^70)
ll_mat = DataFrame(model = _order)
for s in selections
    ll_mat[!, s] = [_col(metrics_eval, m, "logloss_$(s)_overall_diff_ll") for m in _order]
end
show(ll_mat; allrows=true, allcols=true, truncate=0); println()

# GLMEdge spread_fair coef per line: >0 + low p = model edge predicts CLV on this line.
println("\n", "="^70, "\n📈 GLMEdge spread_fair_coef per line  (·=p≥0.10)\n", "="^70)
ge_mat = DataFrame(model = _order)
for s in selections
    coefs = [_col(metrics_eval, m, "glmedge_$(s)_spread_fair_coef") for m in _order]
    ps    = [_col(metrics_eval, m, "glmedge_$(s)_spread_fair_p_value") for m in _order]
    ge_mat[!, s] = [isnan(c) ? "—" : (isnan(p) || p ≥ 0.10 ? "$(c)·" : "$(c)") for (c, p) in zip(coefs, ps)]
end
show(ge_mat; allrows=true, allcols=true, truncate=0); println()

# RQR (goal-count calibration, not per-line) — under-bias shows as +mean (residuals shifted).
println("\n", "="^70, "\n🎯 RQR (all) — mean≈0 well-centred; +mean ⇒ model UNDER-predicts goals\n", "="^70)
Evaluation.display_summary_metric(_sortmodels(metrics_eval), :rqr)

# ==========================================
# 4. READOUT
# ==========================================
println("""

$("="^70)
READ:
 • If dp_nomarket's totals win is a real edge, it beats market on BOTH over_* AND under_* LogLoss.
   If it's just an under-bias, it beats market on under_* / btts_no but LOSES on over_* (positive diff).
 • RQR mean for dp_nomarket: strongly negative-skew / shifted ⇒ confirms the structural goal bias.
 • For the supremacy thesis: compare dp_split_lw0 vs dp_nomarket on home/away/draw LogLoss + GLMEdge
   PER LINE — does anchoring supremacy actually beat market on the 1X2 lines, line by line.
$("="^70)
""")
