# current_development/plus_minus_ratings/r13_biweekly.jl
#
# RUNNER. Re-run the key arms at BIWEEKLY fold granularity.
#
# WHY. r10/r11/r12 all used `dynamics_col = :match_month`, which in this repo is
# `cld(match_week, 4)` — FOUR-WEEK blocks (matches.jl:62 overwrites the calendar month). That gave
# 22 folds / 710 OOS matches. But the funnel winner's own validation (bbc_xg_proxy r06) used
# **60 folds over 3 seasons ≈ biweekly**, so the incumbent was established on a protocol we never
# reproduced. The r10 comparison is internally fair — every arm went through the same splitter —
# but it is not like-for-like against the number the funnel was graduated on.
#
# WHAT IT SHOULD CHANGE. Finer folds retrain more often, so a prediction is made on parameters at
# most 2 weeks stale instead of 4. That should mildly help EVERY arm, and plausibly help the APM
# arms MORE: the RAPM ridge is refit per fold, so a fresher fold means a fresher rating, whereas
# the funnel's shot-volume signal is less time-sensitive. If so, biweekly flatters the APM arms
# relative to what r10 measured — which is exactly the bias worth checking before ranking anything.
#
# ARM ORDER IS DELIBERATE — most informative first, so an overnight run that does not finish still
# answers the primary question:
#   1. funnel_apm_xg    — the fusion (top arm on every point estimate)
#   2. funnel_winner    — the incumbent, on its own native protocol
#   3. apm_shots        — the leak-free APM arm (pure counts, no fitted target)
#   4. goals_baseline   — the anchor for "APM beats its twin", the only SIGNIFICANT model result.
#                         Slowest by far (~2x the others even at monthly) because the smile engine
#                         computes its market pillars then multiplies them by zero.
#   5. apm_xg           — completes the xg-vs-shots contrast
#
# Separate SAVE_DIR: `plus_minus_biweek`. NOT the r10 directory — a `_bw` name suffix would still
# match `startswith(d, "apm_shots_")` in the loader and silently mix protocols.
#
#   JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16 \
#       current_development/plus_minus_ratings/r13_biweekly.jl

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Distributions
using ThreadPinning

pinthreads(:cores)
@info "threads" n = Threads.nthreads() cores = ThreadPinning.ncores()

const BF = BayesianFootball
const D  = BF.Data
const F  = BF.Features
const M  = BF.Models.PreGame
const E  = BF.Experiments
const EV = BF.Evaluation

const SAVE_DIR        = "./data/experiments/plus_minus_biweek"
const TARGET_SEASONS  = ["24/25", "25/26"]
const HISTORY_SEASONS = 2
const DYNAMICS_COL    = :match_biweek        # ← the whole point of this runner
const SAMPLES         = 1000
const WARMUP          = 300
const CHAINS          = 4

ds = D.load_datastore_cached(D.ScottishLower())

ARMS = [
    "funnel_apm_xg"  => M.DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel(
                            player_ratings_feature = F.XGPlusMinusFeature()),
    "funnel_winner"  => M.DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel(),
    "apm_shots"      => M.DynamicGoalsPlusMinusLeagueTimeDecayModel(
                            player_ratings_feature = F.ShotsPlusMinusFeature()),
    "goals_baseline" => M.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel(market_on = false),
    "apm_xg"         => M.DynamicGoalsPlusMinusLeagueTimeDecayModel(
                            player_ratings_feature = F.XGPlusMinusFeature()),
]

# Only the first N arms run. Set to 3 (funnel_apm_xg, funnel_winner, apm_shots) so the sweep
# finishes overnight — that trio answers the primary question, "does finer fold granularity move
# the APM arms relative to the funnel?". `goals_baseline` and `apm_xg` are DEFERRED, not dropped:
# the baseline is the slowest arm by ~2x (the smile engine computes its market pillars and then
# multiplies them by zero) and would eat the whole night on its own. Consequence to remember when
# reading the output: WITHOUT `goals_baseline` at biweekly, this run CANNOT re-test "APM beats its
# twin" — the one comparison that was statistically significant at monthly. It only re-tests the
# rankings that were ties.
const RUN_N = 3

for (name, model) in ARMS[1:min(RUN_N, length(ARMS))]
    @info "=== $name (biweekly) ==="
    try
        task = E.create_experiment_task(
            ds, model, name, SAVE_DIR;
            target_seasons  = TARGET_SEASONS,
            history_seasons = HISTORY_SEASONS,
            dynamics_col    = DYNAMICS_COL,
            samples         = SAMPLES,
            warmup          = WARMUP,
            chains          = CHAINS,
        )
        E.save_experiment(E.run_experiment(task))
        @info "--- $name DONE ---"
    catch err
        @error "$name FAILED" exception = (err, catch_backtrace())
    end
end

# ==========================================
# SCORING — Double Chance EXCLUDED
# ==========================================
# DC labels are broken at source: `processing.jl:111` passes SofaScore's `winning` flag through and
# it marks only ONE of the two winning DC selections (DC_12 never wins), while `prob_fair_close`
# normalises DC to 1.0 instead of 2.0. Worth ~0.03 nats — larger than every real effect here.
function load_latest(dir, names)
    isdir(dir) || return E.ExperimentResults[], String[]
    dirs = readdir(dir); out = E.ExperimentResults[]; got = String[]
    for n in names
        c = filter(d -> startswith(d, n * "_"), dirs); isempty(c) && continue
        try; push!(out, E.load_experiment(joinpath(dir, maximum(c)))); push!(got, n)
        catch err; @warn "could not load $n" exception = err; end
    end
    return out, got
end

exps, got = load_latest(SAVE_DIR, first.(ARMS))   # skips arms that were not run
@info "loaded (biweekly)" got
isempty(exps) && error("nothing trained")

const OU = [:over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
            :over_35,:under_35,:over_45,:under_45]
mets = EV.AbstractScoringRule[
    EV.LogLoss([:home,:draw,:away]), EV.LogLoss([:btts_yes,:btts_no]), EV.LogLoss(OU),
    EV.LogLoss(vcat([:home,:draw,:away,:btts_yes,:btts_no], OU)),
    EV.GLMEdge(vcat([:home,:draw,:away,:btts_yes,:btts_no], OU))]
sc = EV.evaluate_experiments(mets, exps, ds)
dl = filter(c -> occursin("diff_ll", c), names(sc))
T = select(sc, :model, dl...); rename!(T, Dict(zip(dl, ["1X2","BTTS","OU","ALL_CLEAN"])))
for c in ["1X2","BTTS","OU","ALL_CLEAN"]; T[!, c] = round.(T[!, c], digits = 5); end
println("\n===== BIWEEKLY: CLEAN SCORING (negative = beats the de-vigged close) =====")
println(sort(T, "ALL_CLEAN"))

# ==========================================
# ECONOMIC — priced on BETFAIR, which is the venue that matters
# ==========================================
# The identical curated book is -9.5% ROI on the SofaScore/Bet365 close and +6.8% on Betfair;
# overround 1.100/1.065/1.079 vs 1.000/0.997/1.006. Anything quoted off `ds.odds` measures the
# bookmaker's margin, not the model.
#
# ⚠ REBUILD WITH ALL NINE FIELDS. The 7-arg idiom used across current_development drops `bbc` AND
# `bbc_events`, and run_backtest -> extract_oos_predictions -> create_features REBUILDS features
# from the store passed in — so a 7-arg rebuild degrades the funnel arms to goals-only and zeroes
# the APM ratings, with no error raised.
fam(s) = (t = String(s);
    t in ("home","draw","away") ? "1X2" :
    startswith(t, "btts") ? "BTTS" :
    (startswith(t, "over_") || startswith(t, "under_")) ? "O/U" :
    startswith(t, "DC") ? "DC(broken)" : "other")

for (venue, store) in ("BET365 (ds.odds)" => ds,
                       "BETFAIR" => D.DataStore(ds.segment, ds.matches, ds.statistics,
                            D.summarize_betfair_market(ds, open_window=(-100000.0,-10.0),
                                                       close_window=(-20.0,0.0)),
                            ds.lineups, ds.incidents, ds.betfair_odds, ds.bbc, ds.bbc_events))
    try
        led = BF.BackTesting.run_backtest(store, exps, [BF.Signals.BayesianKelly()];
                market_config = D.Markets.DEFAULT_MARKET_CONFIG)   # REQUIRED, defaults to nothing
        ts = BF.BackTesting.generate_tearsheet(led)                # NOT summarize_models
        ts.family = fam.(ts.selection)
        # Curated: O/U + BTTS. 1X2 bleeds (and on Betfair shows +20% ROI with NEGATIVE growth =
        # longshot variance); CorrectScore is noise (~180 bets, sign-flips vs Ireland's -20%).
        cur = filter(r -> r.family in ("O/U","BTTS"), ts)
        agg = combine(groupby(cur, :model_name),
            :bets_placed=>sum=>:bets, :turnover=>sum=>:turnover, :profit=>sum=>:profit,
            [:profit,:turnover]=>((p,t)->100*sum(p)/max(sum(t),1e-9))=>:roi_pct,
            :hurdle_G=>(g->sum(skipmissing(g)))=>:sum_G)
        agg.roi_pct = round.(agg.roi_pct, digits=2); agg.sum_G = round.(agg.sum_G, digits=4)
        println("\n===== BIWEEKLY CURATED (O/U + BTTS) @ $venue =====")
        println(sort(agg, :sum_G, rev=true))
        byfam = combine(groupby(ts, [:family,:model_name]),
            [:profit,:turnover]=>((p,t)->round(100*sum(p)/max(sum(t),1e-9),digits=1))=>:roi,
            :hurdle_G=>(g->round(sum(skipmissing(g)),digits=4))=>:G)
        println("\n-- ROI% by family @ $venue --")
        println(unstack(select(byfam,:family,:model_name,:roi), :model_name,:family,:roi))
        println("\n-- growth G by family @ $venue --")
        println(unstack(select(byfam,:family,:model_name,:G), :model_name,:family,:G))
    catch err
        @error "backtest failed @ $venue" exception = (err, catch_backtrace())
    end
end

println("""

COMPARE AGAINST THE MONTHLY RUN (r10/r12, dynamics_col = :match_month, 22 folds / 710 OOS):
  clean ALL   funnel_apm_xg -0.00216 | funnel_winner -0.00186 | apm_shots -0.00036 | base +0.00677
  Betfair G   funnel_apm_xg  0.0396  | funnel_winner  0.0390  | apm_shots  0.0253  | base -0.0062
KEY QUESTION: do the APM arms gain relative to the funnel at finer granularity, as the
"fresher fold -> fresher ridge" argument predicts? If they do not, the monthly ranking stands.
""")
