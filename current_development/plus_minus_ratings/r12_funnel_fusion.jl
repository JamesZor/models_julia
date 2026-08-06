# current_development/plus_minus_ratings/r12_funnel_fusion.jl
#
# RUNNER. Does the APM pillar and the shots funnel carry the SAME information or DIFFERENT
# information?
#
# THE MOTIVATION IS A DIVISION OF LABOUR VISIBLE IN THE r10/r11 CLEAN SCORING
# (diff_ll = model − market, negative = beats the de-vigged close; Double Chance excluded because
# its labels are broken at source — see NOTES.md):
#
#   model           1X2        BTTS       O/U         all
#   funnel_winner   0.00475    0.00060    -0.00435    -0.00186
#   apm_xg          0.00357    0.00100    -0.00187    -0.00039
#
# The funnel owns TOTALS (shot volume -> how many goals). The APM pillar owns 1X2 (lineup quality
# -> who wins). If those are separable, a fused engine should inherit both and land near -0.0030,
# ahead of either parent.
#
# TWO ARMS, BECAUSE THE MECHANISM MAKES A FALSIFIABLE PREDICTION:
#   funnel_apm_xg     — xG plus-minus is shot QUALITY per player, which the funnel cannot see
#                       (it reads raw shot COUNTS). Predicted to HELP.
#   funnel_apm_shots  — shots plus-minus is shot VOLUME decomposed to players, i.e. the very
#                       signal the funnel already exploits. Predicted to be REDUNDANT.
# If both help equally, the "quality vs volume" story is wrong and the gain is something else.
#
# ⚠ PRIOR EVIDENCE AGAINST: bbc_xg_proxy r07b found funnel + iso-market fusion null / soft-negative
# ("fusion regresses the 1X2 edge to market, keep modular"). That was a MARKET pillar, not a player
# pillar, but a null here would be consistent with it and must be recorded as such.
#
#   JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16 \
#       current_development/plus_minus_ratings/r12_funnel_fusion.jl

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Distributions
using ThreadPinning

pinthreads(:cores)

const BF = BayesianFootball
const D  = BF.Data
const F  = BF.Features
const M  = BF.Models.PreGame
const E  = BF.Experiments
const EV = BF.Evaluation

const SAVE_DIR        = "./data/experiments/plus_minus"
const TARGET_SEASONS  = ["24/25", "25/26"]
const HISTORY_SEASONS = 2
const DYNAMICS_COL    = :match_month
const SAMPLES         = 1000
const WARMUP          = 300
const CHAINS          = 4

ds = D.load_datastore_cached(D.ScottishLower())

fuse(f) = M.DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel(player_ratings_feature = f)

ARMS = [
    "funnel_apm_xg"    => fuse(F.XGPlusMinusFeature()),      # predicted complementary
    "funnel_apm_shots" => fuse(F.ShotsPlusMinusFeature()),   # predicted redundant
]

for (name, model) in ARMS
    @info "=== $name ==="
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
    catch err
        @error "$name FAILED" exception = (err, catch_backtrace())
    end
end

# ==========================================
# SCORE EVERYTHING, CLEANLY
# ==========================================
# Double Chance is EXCLUDED on purpose: `processing.jl` passes SofaScore's `winning` flag straight
# through and that flag marks only ONE DC selection per match (DC_12 never wins), while
# `prob_fair_close` normalises DC to sum to 1.0 instead of 2.0. The model prices DC_12 correctly at
# ~0.72 and is then scored as though it lost — worth ~0.03 nats of pure artefact, larger than every
# real effect being measured here.
function load_latest(save_dir, names)
    dirs = readdir(save_dir); out = E.ExperimentResults[]; got = String[]
    for n in names
        c = filter(d -> startswith(d, n * "_"), dirs); isempty(c) && continue
        try; push!(out, E.load_experiment(joinpath(save_dir, maximum(c)))); push!(got, n)
        catch err; @warn "could not load $n" exception = err; end
    end
    return out, got
end

wanted = ["funnel_apm_xg", "funnel_apm_shots", "funnel_winner",
          "apm_xg", "apm_shots", "apm_sot", "apm_goals", "apm_pillar_only", "goals_baseline"]
exps, got = load_latest(SAVE_DIR, wanted)
@info "loaded arms" got

const OU = [:over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
            :over_35,:under_35,:over_45,:under_45]
mets = EV.AbstractScoringRule[
    EV.LogLoss([:home,:draw,:away]),
    EV.LogLoss([:btts_yes,:btts_no]),
    EV.LogLoss(OU),
    EV.LogLoss(vcat([:home,:draw,:away,:btts_yes,:btts_no], OU)),
    EV.CRPS(),
]
scores = EV.evaluate_experiments(mets, exps, ds)
dl = filter(c -> occursin("diff_ll", c), names(scores))
T = select(scores, :model, dl...)
rename!(T, Dict(zip(dl, ["1X2","BTTS","OU","ALL_CLEAN"])))
for c in ["1X2","BTTS","OU","ALL_CLEAN"]; T[!, c] = round.(T[!, c], digits = 5); end
println("\n===== CLEAN SCORING, DC EXCLUDED (negative = beats the de-vigged close) =====")
println(sort(T, "ALL_CLEAN"))

# ==========================================
# ECONOMIC — growth / ROI per selection
# ==========================================
# NB `generate_tearsheet`, not `summarize_models`: analysis.jl is commented out of
# backtesting-module.jl, so summarize_models does not exist in the loaded module.
try
    ledger = BF.BackTesting.run_backtest(
        ds, exps, [BF.Signals.BayesianKelly()];
        market_config = D.Markets.DEFAULT_MARKET_CONFIG)
    ts = BF.BackTesting.generate_tearsheet(ledger)
    core = [:home, :draw, :away, :btts_yes, :btts_no,
            :under_15, :over_15, :under_25, :over_25, :under_35, :over_35]
    for val in (:roi_pct, :hurdle_G, :bets_placed)
        println("\n", "="^70, "\n $val by selection\n", "="^70)
        println(unstack(filter(r -> Symbol(r.selection) in core, ts), :selection, :model_name, val))
    end
catch err
    @error "backtest failed — the scoring above still stands" exception = (err, catch_backtrace())
end
