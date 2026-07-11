#=
r05 — PER-LINE eval + backtest of the saved Grid-B smile cells (r04 output). NO retraining.

⚠ EDIT FIRST: set _TAG to match r04's (hl*, hs*), and add Grid-A control folders if reused.

Three lenses, decided PER MARKET FAMILY (verify the Ireland routing — supremacy→1X2,
smile→totals/BTTS — transfers to a ratings-free team base rather than assuming it):
 1. Per-line LogLoss diff vs Bet365 close + family-pooled summary.
 2. Per-line GLMEdge spread_fair_coef(p) — the CLV-style signal lens.
 3. BayesianKelly backtest tearsheet per selection (hurdle_G) — the money lens, read LAST.
Plus a per-strike O/U focus table: does φ pricing beat the grid at the 0.5/1.5 and 3.5/4.5 tails?

Secondary (winner only, informational): Betfair CLV on 25/26 — ds.betfair_odds covers 315
matches of 25/26 only; too thin to decide anything, prints for the record.

Cells below the r04_convergence.txt gate are excluded.

    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r05_eval_smile.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const BackTesting = BayesianFootball.BackTesting
const Signals     = BayesianFootball.Signals
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))

const _TAG = "hl180_hs2"   # ⚠ match r04's BEST_HL/BEST_HS tag

# ==========================================
# 1. LOAD saved Grid-B (+ reused Grid-A controls)
# ==========================================
folders = Experiments.list_experiments("scottish_smile_grid"; data_dir=joinpath(ROOT, "data"))
all_results = Experiments.load_experiments(folders)

# Pull the none_pois control from Grid A when it wasn't re-run in r04 (RERUN_CONTROLS=false).
ctl_name = "none_pois_$(_TAG)"
if !any(r -> occursin("none_pois", r.config.name), all_results)
    try
        fa = Experiments.list_experiments("scottish_decay_grid"; data_dir=joinpath(ROOT, "data"))
        fa_ctl = filter(f -> occursin(ctl_name, f), fa)
        isempty(fa_ctl) || append!(all_results, Experiments.load_experiments(fa_ctl))
    catch e
        @warn "could not load Grid-A control $ctl_name" exception=e
    end
end
println("[INFO] Loaded $(length(all_results)) experiments: ",
        join([r.config.name for r in all_results], ", "))

ds = Data.load_datastore_cached(Data.ScottishLower())

# ==========================================
# 2. PER-LINE metrics (Bet365 close benchmark — plain ds)
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

_present = sort(unique(metrics_eval.model))
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
    :totals_tails => [:over_05, :under_05, :over_15, :under_15,
                      :over_35, :under_35, :over_45, :under_45],
)
println("\n", "="^70, "\n📊 Family-pooled mean LogLoss diff (lower better) — the ROUTING table\n", "="^70)
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

# ==========================================
# 3. BACKTEST — BayesianKelly per selection (the money lens; read LAST)
# ==========================================
try
    println("\n", "="^60, "\n💰 Backtest (BayesianKelly, Bet365 close)\n", "="^60)
    ledger = BackTesting.run_backtest(
        ds, all_results, [Signals.BayesianKelly()];
        market_config = Data.Markets.DEFAULT_MARKET_CONFIG
    )
    tearsheet = BackTesting.generate_tearsheet(ledger)
    cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover,
                    :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_n_bets, :hurdle_avg_stake,
                    :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
    for m_name in unique(tearsheet.selection)
        println("\nSelection: $m_name")
        show(subset(tearsheet, :selection => ByRow(isequal(m_name)))[!, cols_to_show]; truncate=0)
        println()
    end
catch e
    @error "Backtest phase failed (metric tables above are unaffected)" exception=(e, catch_backtrace())
end

# ==========================================
# 4. SECONDARY (informational): Betfair CLV on 25/26 winner-check
# ==========================================
try
    if nrow(ds.betfair_odds) > 0
        odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
        ds_bf = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups,
                               ds.incidents, ds.betfair_odds)
        println("\n", "="^60, "\n📊 SECONDARY: GLMEdge vs Betfair close (25/26 only, ~315 matches — informational)\n", "="^60)
        Evaluation.display_summary_metric(
            Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds_bf), :glmedge)
    end
catch e
    @warn "Betfair secondary check failed (expected if coverage too thin)" exception=e
end

println("""

$("="^70)
READ (Grid B decision — verdict PER MARKET FAMILY):
 • Exclude cells < 95% on r04_convergence.txt first.
 • TOTALS/BTTS: does any sw>0 column beat sw=0 AND none_pois on family-pooled LogLoss, winning
   on BOTH over_* and under_*? Check the tails table — φ's value concentrates at 0.5/1.5, 3.5/4.5.
 • 1X2: does sup weight help home/draw/away vs none_pois, or does the Ireland lesson (market
   1X2 edge not extractable at the latent-rate layer) repeat here?
 • iso_pois_mw100 vs smile cells = old-pillar-vs-smile A/B on the same base.
 • Record the per-family winner(s) in NOTES.md → RESULTS_scottish_grid.md; then Stage 4
   graduation with the winner's weights as struct defaults.
$("="^70)
""")
