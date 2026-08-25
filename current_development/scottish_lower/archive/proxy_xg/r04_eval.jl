#=
r04 — WP6 EVAL of the proxy-xG grid. NO retraining. ~30-40 min.

METRIC HIERARCHY (the stream's standing order — do not reorder it after seeing the numbers):
  1. CONVERGENCE GATE first. Never read a table for a cell below it. Check the fold count too — a
     cell can save with zero folds and look complete.
  2. PER-LINE LogLoss vs the de-vigged Bet365 close, family-pooled into x12 / btts / totals /
     totals_tails. NEVER aggregate across all selections ([[double-chance-scoring-defect]]: the DC
     is_winner flag marks 1 of 2 winners AND halves the fair probs, which reversed the APM
     headline). GLMEdge + RQR are pathology checks, not rankings.
  3. GROWTH DECIDES. run_backtest -> generate_tearsheet -> roi_pct and hurdle_G, on the BETFAIR
     close where it exists. [[apm-l1-graduation]]: the LogLoss ranking REVERSED under Betfair growth
     (+6.8% vs −9.5% ROI on the same book). Betfair on 56/57 is 25/26 only (~315 matches), so treat
     it as DIRECTIONAL, never significant.
  4. SEASON SPLIT — the coverage diagnosis. 25/26 folds have fully-covered history for every cell;
     24/25 folds do not (22/23 has no commentary). Winning on 25/26 while losing on 24/25 is a DATA
     story, not a STRUCTURE story.

RUNTIME ~30-40m; run with stdout redirected (kaimon's 10-min gate kills the job cosmetically while
Julia keeps going):
    open(joinpath(pkgdir(BayesianFootball), "current_development/scottish_proxy_xg/r04_out.txt"), "w") do io
        redirect_stdout(io) do
            include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_proxy_xg/r04_eval.jl"))
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
# ⚠ REQUIRED. Without the loader, evaluate_experiments cannot dispatch extract_params /
# compute_score_matrix for the prototype structs and silently NaNs every row.
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l02_pxg_engines.jl"))

# ==========================================
# 1. LOAD
# ==========================================
folders = Experiments.list_experiments("scottish_pxg_grid"; data_dir = joinpath(ROOT, "data"))
all_results = Experiments.load_experiments(folders)
println("[INFO] Loaded $(length(all_results)) experiments: ",
        join([r.config.name for r in all_results], ", "))
for r in all_results
    println("    $(r.config.name): $(length(r.training_results.items)) folds")
end

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 720)

selections = [:home, :draw, :away, :btts_yes, :btts_no,
              :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
              :over_35, :under_35, :over_45, :under_45, :over_55, :under_55]
fam = Dict(
    :x12    => [:home, :draw, :away],
    :btts   => [:btts_yes, :btts_no],
    :totals => [:over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
                :over_35, :under_35, :over_45, :under_45],
    :totals_tails => [:over_05, :under_05, :over_15, :under_15,
                      :over_35, :under_35, :over_45, :under_45],
)

_col(df, model, colname) = begin
    colname in names(df) || return NaN
    r = df[df.model .== model, colname]
    (isempty(r) || ismissing(r[1])) ? NaN : round(Float64(r[1]), digits = 4)
end

function line_tables(me, present; header)
    println("\n", "="^74, "\n📉 $header — LogLoss diff (model−market), NEGATIVE beats the close\n", "="^74)
    m = DataFrame(model = present)
    for s in selections
        m[!, s] = [_col(me, mm, "logloss_$(s)_overall_diff_ll") for mm in present]
    end
    show(m; allrows = true, allcols = true, truncate = 0); println()

    println("\n", "="^74, "\n📊 $header — family-pooled mean LogLoss diff (lower better)\n", "="^74)
    f = DataFrame(model = present)
    for (fname, sels) in fam
        f[!, fname] = [round(mean(filter(!isnan,
            [_col(me, mm, "logloss_$(s)_overall_diff_ll") for s in sels])), digits = 4)
            for mm in present]
    end
    show(f; allrows = true, allcols = true, truncate = 0); println()
    return f
end

# ==========================================
# 2. PER-LINE METRICS (Bet365 close, full sample)
# ==========================================
metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
append!(metric, [Evaluation.LogLoss(s) for s in selections])
append!(metric, [Evaluation.GLMEdge(s) for s in selections])

metrics_eval = Evaluation.evaluate_experiments(metric, all_results, ds)
present = sort(unique(metrics_eval.model))
fam_full = line_tables(metrics_eval, present; header = "FULL SAMPLE")

println("\n", "="^74, "\n📈 GLMEdge spread_fair_coef per line  (·= p ≥ 0.10)\n", "="^74)
ge = DataFrame(model = present)
for s in selections
    cs = [_col(metrics_eval, m, "glmedge_$(s)_spread_fair_coef") for m in present]
    ps = [_col(metrics_eval, m, "glmedge_$(s)_spread_fair_p_value") for m in present]
    ge[!, s] = [isnan(c) ? "—" : (isnan(p) || p ≥ 0.10 ? "$(c)·" : "$(c)") for (c, p) in zip(cs, ps)]
end
show(ge; allrows = true, allcols = true, truncate = 0); println()

# ==========================================
# 3. SEASON SPLIT — the coverage diagnosis (free)
# ==========================================
println("\n", "="^74, "\n🗓  SEASON SPLIT — data story vs structure story\n", "="^74)
println("""
    25/26 folds: history = 23/24 + 24/25, 100% commentary-covered for every cell.
    24/25 folds: history = 22/23 + 23/24, and 22/23 has NO commentary — the proxy cells see goals
                 only there while the funnel control keeps its ds.bbc shots.
    If the new engines win on 25/26 and lose on 24/25, the gap is COVERAGE, not structure.
""")
# ⚠ Restrict ONLY the scoring domains. Filtering `bbc_events` too would change its objectid, which
# is the ProxyXGFeature cache key — the cell table would silently be refitted on one season and the
# split would no longer be comparing like with like.
const SPLIT_DOMAINS = (:matches, :statistics, :odds, :betfair_odds)

for season in ("24/25", "25/26")
    try
        keep  = Set(Int.(ds.matches.match_id[String.(ds.matches.season) .== season]))
        parts = Any[]
        for f in fieldnames(typeof(ds))
            v = getfield(ds, f)
            push!(parts, (f in SPLIT_DOMAINS && v isa DataFrame && "match_id" in names(v)) ?
                         v[in.(Int.(v.match_id), Ref(keep)), :] : v)
        end
        ds_s = typeof(ds)(parts...)   # all 9 fields — a 7-arg rebuild silently drops both BBC domains
        me_s = Evaluation.evaluate_experiments(
            Evaluation.AbstractScoringRule[Evaluation.LogLoss(s) for s in selections],
            all_results, ds_s)
        line_tables(me_s, sort(unique(me_s.model)); header = "TARGET SEASON $season")
    catch e
        @warn "season split failed for $season (diagnostic only; the full-sample tables stand)" exception = e
    end
end

# ==========================================
# 4. MONEY LENS — Bet365 then Betfair
# ==========================================
# Globals on purpose: r05 lost an hour of backtest to `try` scope.
R04_LEDGER = nothing; R04_TEARSHEET = nothing
R04_LEDGER_BF = nothing; R04_TEARSHEET_BF = nothing

core = [:home, :draw, :away, :btts_yes, :btts_no,
        :under_15, :over_15, :under_25, :over_25, :under_35, :over_35, :under_45, :over_45]

function money_tables(tear, present; book)
    for val in (:roi_pct, :hurdle_G, :bets_placed)
        println("\n", "="^74, "\n💰 $val (BayesianKelly, $book close)\n", "="^74)
        piv = DataFrame(model = present)
        for s in core
            colv = Any[]
            for m in present
                r = subset(tear, :model_name => ByRow(isequal(m)), :selection => ByRow(isequal(s)))
                push!(colv, nrow(r) == 0 || ismissing(r[1, val]) ? NaN :
                            round(Float64(r[1, val]), digits = 3))
            end
            piv[!, s] = colv
        end
        show(piv; allrows = true, allcols = true, truncate = 0); println()
    end
end

try
    global R04_LEDGER = BackTesting.run_backtest(
        ds, all_results, [Signals.BayesianKelly()];
        market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
    global R04_TEARSHEET = BackTesting.generate_tearsheet(R04_LEDGER)
    money_tables(R04_TEARSHEET, present; book = "Bet365")
catch e
    @error "Bet365 backtest failed (metric tables above are unaffected)" exception = (e, catch_backtrace())
end

println("\n", "="^74, "\n🎯 BETFAIR CLOSE — the deciding book (25/26 only, ~315 matches)\n", "="^74)
println("""
    [[apm-l1-graduation]]: on the SAME book the funnel family scored +6.8% on Betfair and −9.5% on
    Bet365, and the LogLoss ranking did not survive the switch. Directional only at this n.
""")
try
    # summarize_betfair_market(ds::DataStore) -> a graded DataFrame matching the ds.odds schema
    # (src/Data/betfair_util.jl:178). It returns an EMPTY frame when ds.betfair_odds is empty.
    bf = Data.summarize_betfair_market(ds)
    nrow(bf) == 0 && error("ds.betfair_odds is empty on this segment — nothing to price against")
    println("[INFO] Betfair book: $(nrow(bf)) graded selections over " *
            "$(length(unique(bf.match_id))) matches")
    # Rebuild ALL 9 fields: a 7-arg rebuild drops ds.bbc AND ds.bbc_events, which would collapse
    # every proxy/funnel cell onto a goals-only engine at eval time (src/Data/types.jl:46-48).
    ds_bf = typeof(ds)(Any[f === :odds ? bf : getfield(ds, f) for f in fieldnames(typeof(ds))]...)
    global R04_LEDGER_BF = BackTesting.run_backtest(
        ds_bf, all_results, [Signals.BayesianKelly()];
        market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
    global R04_TEARSHEET_BF = BackTesting.generate_tearsheet(R04_LEDGER_BF)
    money_tables(R04_TEARSHEET_BF, present; book = "Betfair")
catch e
    @error """Betfair backtest failed. Re-check summarize_betfair_market's signature and the
              DataStore rebuild against current_development/scottish_lower_smile/r05_eval_smile.jl §4
              — the Bet365 tables above are unaffected.""" exception = (e, catch_backtrace())
end

println("""

$("="^74)
VERDICT RULE (fixed before the run, so it cannot drift):
  cell 2 (pxg_apm) or cell 4 (funnel_pxg_apm) must beat cell 1 (funnel_apm_ctl) on hurdle_G for the
  TOTALS and BTTS families on the BETFAIR book, with per-line LogLoss no worse, at ≥95% fold
  convergence. Anything less is written up as a null and the incumbent stays.

  Then read, win or lose:
   • cell 2 vs cell 3 — is the gain the xG PILLAR or the RAPM pillar?
   • cell 2 vs cell 4 — REPLACE or ADD? (r07b/r04 both say fusion tends to null here.)
   • sigma_q from r03_convergence.txt — is team-level shot quality identified at all?
   • the season split above — coverage or structure?

  A null is a likely and perfectly acceptable outcome. The measured quality axis is ~±4% against
  the funnel's ±10.7% team-strength SD, and three prior fusions in this stream came back null.
  Record it in RESULTS_scottish_proxy_xg.md either way.
$("="^74)
""")
