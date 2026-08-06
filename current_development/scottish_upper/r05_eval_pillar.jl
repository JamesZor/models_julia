#=
r05 — FINAL eval: the r04 pillar grid, read against the r02 family references. NO retraining.

Same per-line lens as r03 (LogLoss diff / GLMEdge / RQR + per-selection Kelly), applied to the
pillar sweep, with the r02 cells loaded alongside so the anchor axis is read in the context of the
families it came from.

Deliverable (write into RESULTS_scottish_upper.md):
  production engine + weights + (hl, hs) + WHICH MARKET FAMILIES TO ROUTE and which to abstain on.

⚠ Cells below the r04_convergence.txt / r02_convergence.txt gates are EXCLUDED from the read.
⚠ Run with stdout redirected (kaimon 10-min gate):
      open("current_development/scottish_upper/r05_out.txt","w") do io
          redirect_stdout(io) do
              include(joinpath(pkgdir(BayesianFootball),
                      "current_development/scottish_upper/r05_eval_pillar.jl"))
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
include(joinpath(ROOT, "current_development/scottish_upper/l01_upper.jl"))

const INCLUDE_FAMILY_REFS = true   # load the r02 grid alongside for context

_load_dir(tag) = begin
    d = joinpath(ROOT, "data", tag)
    isdir(d) ? Experiments.load_experiments(Experiments.list_experiments(tag; data_dir = joinpath(ROOT, "data"))) : Any[]
end

pillar_results = _load_dir("scottish_upper_pillar")
family_results = INCLUDE_FAMILY_REFS ? _load_dir("scottish_upper_family") : Any[]
all_results = vcat(pillar_results, family_results)
println("[INFO] pillar cells: ", length(pillar_results), "   family refs: ", length(family_results))
println("[INFO] models: ", join([r.config.name for r in all_results], ", "))

ds = Data.load_datastore_cached(Data.ScottishUpper())

println("\n[INFO] fold counts per cell (silent-drop check):")
for r in all_results
    println("    ", rpad(r.config.name, 40), length(r.training_results.items))
end

# ==========================================
# PER-LINE METRICS
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

println("\n", "="^70, "\n🎯 RQR\n", "="^70)
Evaluation.display_summary_metric(metrics_eval, :rqr)

# ==========================================
# MONEY LENS — globals, per selection
# ==========================================
println("\n", "="^70, "\n💰 BayesianKelly tearsheet (per selection)\n", "="^70)
R05_LEDGER = nothing
R05_TEARSHEET = nothing
try
    global R05_LEDGER = BackTesting.run_backtest(ds, all_results, [Signals.BayesianKelly()];
                                                 market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
    global R05_TEARSHEET = BackTesting.generate_tearsheet(R05_LEDGER)
    cols = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover,
            :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_n_bets, :hurdle_avg_stake,
            :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
    have = [c for c in cols if String(c) in names(R05_TEARSHEET)]
    for sel in unique(R05_TEARSHEET.selection)
        println("\n--- selection: $sel ---")
        show(subset(R05_TEARSHEET, :selection => ByRow(isequal(sel)))[!, have];
             allrows=true, allcols=true, truncate=0)
        println()
    end
catch e
    @error "backtest failed" exception=(e, catch_backtrace())
end

println("""

$("="^70)
FINAL READ — how to turn this into a production decision:
 1. Drop every cell below its convergence gate, and every cell short on folds.
 2. Pick PER MARKET FAMILY on LogLoss diff + GLMEdge, then confirm the sign with hurdle_G / ROI.
    Scoring and money must AGREE; if they disagree the sample is too short to act on.
 3. If the mw axis is FLAT on scoring (the 56/57 pattern), do NOT pick on noise — tiebreak on
    sampler health (the sole hard-gate pass), and prefer a mid-range weight over a boundary one.
    This model retrains weekly in-season, so gate robustness has real operational value.
 4. If the axis has an INTERIOR optimum (the Ireland pattern), take it, and record that 54/55
    behaves like Ireland rather than 56/57.
 5. Circularity: market-anchored cells are trained toward the same close they are scored against.
    Where an anchored cell only narrowly beats a structural one, prefer the structural cell.
 6. Expect to ABSTAIN on 1X2 — no engine made Kelly money there in either parent stream.
 7. Write engine + weights + (hl, hs) + routing + the convergence tables into
    RESULTS_scottish_upper.md, and update NOTES.md's findings log.
$("="^70)
""")
