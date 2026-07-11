#=
r03 — PER-LINE eval of the saved Grid-A cells (r02 output). NO retraining.

Benchmark = de-vigged Bet365 (SofaScore) close in plain `ds` — NO Betfair swap (56/57 have
Betfair for 25/26 only; secondary check comes later on the winner, not here).

Read (r06 lesson — judge PER LINE, never grouped):
 • LogLoss diff (model−market) per selection: NEGATIVE = beats the close on that line.
   A real totals edge wins on BOTH over_* AND under_*; an under-bias wins unders only.
 • GLMEdge spread_fair_coef per selection: >0 with low p = model−market spread predicts outcomes.
 • RQR: goal-count calibration (+mean ⇒ model under-predicts goals).
Pick (hl*, hs*) on pooled per-line LogLoss + GLMEdge STABILITY across lines. Cells below the
r02_convergence.txt gate (<95% folds R-hat≤1.01) are excluded from the read.

    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r03_eval_decay.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
# REQUIRED: loader-local prediction overrides for the TeamDP structs; without it
# evaluate_experiments silently drops those cells' rows.
include(joinpath(ROOT, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))

# ==========================================
# 1. LOAD saved grid
# ==========================================
folders = Experiments.list_experiments("scottish_decay_grid"; data_dir=joinpath(ROOT, "data"))
all_results = Experiments.load_experiments(folders)
println("[INFO] Loaded $(length(all_results)) experiments: ",
        join([r.config.name for r in all_results], ", "))

ds = Data.load_datastore_cached(Data.ScottishLower())

# ==========================================
# 2. PER-LINE metric vector (Bet365 close benchmark)
# ==========================================
selections = [
    :home, :draw, :away,
    :btts_yes, :btts_no,
    :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
    :over_35, :under_35, :over_45, :under_45, :over_55, :under_55,
]
metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
append!(metric, [Evaluation.LogLoss(s) for s in selections])
append!(metric, [Evaluation.GLMEdge(s) for s in selections])

metrics_eval = Evaluation.evaluate_experiments(metric, all_results, ds)

# ==========================================
# 3. PIVOT into model × selection matrices
# ==========================================
_order = vcat(
    ["none_pois_hl$(hl)_hs$(hs)" for hl in (60, 120, 180, 365) for hs in (1, 2, 3)],
    ["none_nb_hl180_hs2", "iso_nb_mw100_hl180_hs2"])
_present = [m for m in _order if m in metrics_eval.model]

function _col(df, model, colname)
    colname in names(df) || return NaN
    r = df[df.model .== model, colname]
    (isempty(r) || ismissing(r[1])) ? NaN : round(Float64(r[1]), digits=4)
end

println("\n", "="^70, "\n📉 LogLoss diff (model−market) per line — NEGATIVE = beats Bet365 close\n", "="^70)
ll_mat = DataFrame(model = _present)
for s in selections
    ll_mat[!, s] = [_col(metrics_eval, m, "logloss_$(s)_overall_diff_ll") for m in _present]
end
show(ll_mat; allrows=true, allcols=true, truncate=0); println()

# pooled-per-model summary: mean LogLoss diff over 1X2 / totals / BTTS families
fam = Dict(
    :x12    => [:home, :draw, :away],
    :btts   => [:btts_yes, :btts_no],
    :totals => [:over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
                :over_35, :under_35, :over_45, :under_45],
)
println("\n", "="^70, "\n📊 Family-pooled mean LogLoss diff (lower better)\n", "="^70)
fam_mat = DataFrame(model = _present)
for (fname, sels) in fam
    fam_mat[!, fname] = [round(mean(filter(!isnan,
        [_col(metrics_eval, m, "logloss_$(s)_overall_diff_ll") for s in sels])), digits=4)
        for m in _present]
end
show(fam_mat; allrows=true, allcols=true, truncate=0); println()

println("\n", "="^70, "\n📈 GLMEdge spread_fair_coef per line  (·=p≥0.10)\n", "="^70)
ge_mat = DataFrame(model = _present)
for s in selections
    coefs = [_col(metrics_eval, m, "glmedge_$(s)_spread_fair_coef") for m in _present]
    ps    = [_col(metrics_eval, m, "glmedge_$(s)_spread_fair_p_value") for m in _present]
    ge_mat[!, s] = [isnan(c) ? "—" : (isnan(p) || p ≥ 0.10 ? "$(c)·" : "$(c)") for (c, p) in zip(coefs, ps)]
end
show(ge_mat; allrows=true, allcols=true, truncate=0); println()

println("\n", "="^70, "\n🎯 RQR — mean≈0 well-centred; +mean ⇒ model UNDER-predicts goals\n", "="^70)
Evaluation.display_summary_metric(metrics_eval, :rqr)

println("""

$("="^70)
READ (Grid A decision):
 • Exclude any cell < 95% on r02_convergence.txt FIRST.
 • hl axis: expect too-short half-life (60) noisy team ratings; too-long (365) stale after
   promotion/relegation churn. hs axis: more seasons only help if hl is long enough to see them
   (hs=3 with hl=60 is wasted data; hs=1 with hl=365 is truncated decay).
 • Winner = best family-pooled LogLoss with per-line GLMEdge signs stable across BOTH over_* and
   under_* (no bias masquerade). Record (hl*, hs*) in NOTES.md; r04 uses it.
 • nb references: expect none_nb ≈ none_pois at same (hl,hs) (r inert, V/M<1) — if nb clearly
   beats pois, STOP and investigate dispersion before r04.
$("="^70)
""")
