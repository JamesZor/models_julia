#=
r08 — eval of the iso mw sweep (r07) against the Grid-B references. NO retraining.

Loads: scottish_iso_grid (mw25/40/70 @3ch + mw100_c4 @4ch) + from scottish_smile_grid:
none_pois ctl, iso_pois_mw100 (old 3ch — sanity: does the 4ch re-run reproduce it?),
sup100_sw0 (the BTTS reference). All same spec (2 seasons, 40 folds) → tables comparable.

Read (decides the production mw):
 • TOTALS is the decision family (the only money family — Grid B §3). Winner = best
   family-pooled LogLoss with ROI/hurdle_G agreement, subject to the HARD gate in
   r07_convergence.txt.
 • Ireland prior: mw 0.25–0.4 optimal, mw→1 backfires (denoising lesson). Does 56/57 agree?
 • mw100_c4 vs old mw100: numbers should reproduce (~identical rows); if not, investigate
   before trusting anything.
 • BTTS: compare winner vs sup100_sw0 row — is a second engine worth it? (margin was 0.001.)

RUNTIME ~30–40m; run with stdout redirected (kaimon gate kills the eval job cosmetically):
    open(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r08_out.txt"), "w") do io
        redirect_stdout(io) do
            include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r08_eval_iso.jl"))
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
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))

# ==========================================
# 1. LOAD r07 cells + Grid-B references
# ==========================================
folders = Experiments.list_experiments("scottish_iso_grid"; data_dir=joinpath(ROOT, "data"))
all_results = Experiments.load_experiments(folders)

ref_names = ["none_pois_hl365_hs2_ctl", "iso_pois_mw100_hl365_hs2", "smile_pois_sup100_sw0_hl365_hs2"]
try
    fb = Experiments.list_experiments("scottish_smile_grid"; data_dir=joinpath(ROOT, "data"))
    fb_refs = filter(f -> any(occursin(r, f) for r in ref_names), fb)
    isempty(fb_refs) || append!(all_results, Experiments.load_experiments(fb_refs))
catch e
    @warn "could not load Grid-B reference cells" exception=e
end
println("[INFO] Loaded $(length(all_results)) experiments: ",
        join([r.config.name for r in all_results], ", "))

ds = Data.load_datastore_cached(Data.ScottishLower())

# ==========================================
# 2. PER-LINE metrics (Bet365 close)
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
println("\n", "="^70, "\n📊 Family-pooled mean LogLoss diff (lower better) — mw DECISION table\n", "="^70)
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

# ==========================================
# 3. BACKTEST — money lens (globals: survive the include, r05 try-scope lesson)
# ==========================================
R08_LEDGER = nothing; R08_TEARSHEET = nothing
try
    global R08_LEDGER = BackTesting.run_backtest(
        ds, all_results, [Signals.BayesianKelly()];
        market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
    global R08_TEARSHEET = BackTesting.generate_tearsheet(R08_LEDGER)
    core = [:home, :draw, :away, :btts_yes, :btts_no,
            :under_15, :over_15, :under_25, :over_25, :under_35, :over_35, :under_45, :over_45]
    for val in (:roi_pct, :hurdle_G, :bets_placed)
        println("\n", "="^70, "\n💰 $val (BayesianKelly, Bet365 close)\n", "="^70)
        piv = DataFrame(model = _present)
        for s in core
            col = Any[]
            for m in _present
                r = subset(R08_TEARSHEET, :model_name => ByRow(isequal(m)), :selection => ByRow(isequal(s)))
                push!(col, nrow(r) == 0 || ismissing(r[1, val]) ? NaN : round(Float64(r[1, val]), digits=3))
            end
            piv[!, s] = col
        end
        show(piv; allrows=true, allcols=true, truncate=0); println()
    end
catch e
    @error "Backtest phase failed (metric tables above are unaffected)" exception=(e, catch_backtrace())
end

println("""

$("="^70)
READ (mw decision):
 • Exclude cells below the HARD gate in r07_convergence.txt.
 • Production mw = best TOTALS family LogLoss with money-lens agreement (ROI/hurdle_G on the
   ladder). Check the Ireland shape: expect an interior optimum (mw 25–40) if the lesson
   transfers; monotone-in-mw would mean 56/57 wants full anchoring.
 • mw100_c4 must ≈ reproduce the Grid-B mw100 row — if not, STOP.
 • Record winner in NOTES.md + RESULTS §3.7 → Stage 4: graduate TeamIsoDPGoalsModel to src
   with the winning mw as default; route totals only (BTTS from sup*_sw0 optional).
$("="^70)
""")
