#=
r03 — PER-LINE eval of the saved r02 family cells. NO retraining.

Benchmark = de-vigged Bet365 (SofaScore) close in plain `ds`. There is NO Betfair swap anywhere in
this stream — 54/55 have zero exchange data. That makes the benchmark SELF-REFERENTIAL for the
market-anchored cells (`iso_*`, `smile_*`): they are trained toward the same close they are scored
against. Structural cells (`none_*`, `funnel`, `rating`) do not have that circularity. Say so when
reporting, and weight the routing decision accordingly.

How to read (the r06 lesson — judge PER LINE, never grouped; grouped Σ hurdle_G is gameable by a
directional bias cashing over a short window):
 • LogLoss diff (model−market) per selection: NEGATIVE = beats the close on that line.
   A real totals edge wins on BOTH over_* AND under_*; winning unders only is an under-bias.
   NOTE over_K and under_K diffs are IDENTICAL by construction (binary log score counts both
   sides) — the "wins both sides" check is per-STRIKE, not per-side.
 • GLMEdge spread_fair_coef per selection: >0 with low p = the model−market spread predicts outcomes.
 • RQR: goal-count calibration (+mean ⇒ model UNDER-predicts goals).
 • Then the money lens: per-bet Kelly hurdle_G / ROI per selection.

⚠ Cells below the r02_convergence.txt gate (<95% folds R-hat≤1.01) are EXCLUDED from the read.
⚠ Run with stdout redirected — kaimon's 10-min no-activity gate kills the eval call while Julia
  keeps running:
      open("current_development/scottish_upper/r03_out.txt","w") do io
          redirect_stdout(io) do
              include(joinpath(pkgdir(BayesianFootball),
                      "current_development/scottish_upper/r03_eval_family.jl"))
          end
      end
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const BackTesting = BayesianFootball.BackTesting
const Signals     = BayesianFootball.Signals

const ROOT = pkgdir(BayesianFootball)
# REQUIRED: the loader defines TeamDPGoalsModel / TeamIsoDPGoalsModel and their prediction
# overrides. Without it those cells deserialize but eval-fail to NaN — silently dropped rows.
include(joinpath(ROOT, "current_development/scottish_upper/l01_upper.jl"))

# ==========================================
# 1. LOAD saved grid
# ==========================================
folders = Experiments.list_experiments("scottish_upper_family"; data_dir = joinpath(ROOT, "data"))
all_results = Experiments.load_experiments(folders)
println("[INFO] Loaded $(length(all_results)) experiments: ",
        join([r.config.name for r in all_results], ", "))

ds = Data.load_datastore_cached(Data.ScottishUpper())

# Fold counts per cell — a cell can be saved with zero folds and still look present.
println("\n[INFO] fold counts per cell (silent-drop check):")
for r in all_results
    println("    ", rpad(r.config.name, 36), length(r.training_results.items))
end

# ==========================================
# 2. PER-LINE metric vector
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
_present = sort(unique(String.(metrics_eval.model)))

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

println("\n", "="^70, "\n📈 GLMEdge spread_fair_coef per line  (·= p≥0.10)\n", "="^70)
ge_mat = DataFrame(model = _present)
for s in selections
    coefs = [_col(metrics_eval, m, "glmedge_$(s)_spread_fair_coef") for m in _present]
    ps    = [_col(metrics_eval, m, "glmedge_$(s)_spread_fair_p_value") for m in _present]
    ge_mat[!, s] = [isnan(c) ? "—" : (isnan(p) || p ≥ 0.10 ? "$(c)·" : "$(c)") for (c, p) in zip(coefs, ps)]
end
show(ge_mat; allrows=true, allcols=true, truncate=0); println()

println("\n", "="^70, "\n🎯 RQR — mean≈0 well-centred; +mean ⇒ model UNDER-predicts goals\n", "="^70)
Evaluation.display_summary_metric(metrics_eval, :rqr)

# ==========================================
# 4. MONEY LENS — per-bet Kelly, PER SELECTION
# ==========================================
# Assigned to GLOBALS on purpose: the 56/57 r05 lost its tearsheet to `try` scope and had to re-run
# an hour of backtest to inspect it.
println("\n", "="^70, "\n💰 BayesianKelly tearsheet (per selection)\n", "="^70)
R03_LEDGER = nothing
R03_TEARSHEET = nothing
try
    global R03_LEDGER = BackTesting.run_backtest(ds, all_results, [Signals.BayesianKelly()];
                                                 market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
    global R03_TEARSHEET = BackTesting.generate_tearsheet(R03_LEDGER)
    cols = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover,
            :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_n_bets, :hurdle_avg_stake,
            :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
    have = [c for c in cols if String(c) in names(R03_TEARSHEET)]
    for sel in unique(R03_TEARSHEET.selection)
        println("\n--- selection: $sel ---")
        show(subset(R03_TEARSHEET, :selection => ByRow(isequal(sel)))[!, have];
             allrows=true, allcols=true, truncate=0)
        println()
    end
catch e
    @error "backtest failed" exception=(e, catch_backtrace())
end

println("""

$("="^70)
READ (family decision):
 • Exclude any cell below 95% in r02_convergence.txt FIRST, and any cell whose fold count above is
   short of the others (silent split drop).
 • Produce a PER-FAMILY ROUTING TABLE (1X2 / totals / BTTS), not a single winner. Both parent
   streams ended in per-family routing, and both found NOBODY makes Kelly money on 1X2 — expect to
   abstain there.
 • Circularity caveat: market-anchored cells are scored against their own training anchor. If an
   `iso`/`smile` cell wins a family only narrowly over a structural cell, prefer the structural one.
 • nb reference: expect none_nb ≈ none_pois (r inert on sub-Poisson data). If NB clearly WINS, stop
   and investigate the dispersion regime before drawing any other conclusion.
 • hl control: hl180 vs hl365 should reproduce the 56/57 gradient (long memory better). If it
   REVERSES, the higher turnover of the top two tiers is real and r04 should carry an hl axis.
 • Record the winner per family in NOTES.md, then set WINNER in r04_grid_pillar.jl.
$("="^70)
""")
