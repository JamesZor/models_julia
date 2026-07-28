# current_development/plus_minus_ratings/r10_src_experiment.jl
#
# RUNNER. WP-D of the src graduation — the actual measurement.
#
# Everything below drives `src/` code; nothing is prototyped here. It answers two questions on
# ScottishLower (tournaments 56/57):
#
#   Q1  apm_shots vs goals_baseline   — does the APM pillar add anything to its OWN no-APM twin?
#   Q2  apm_shots vs funnel_winner    — does it beat the current best team-level engine?
#
# plus a variant sweep over the four plus-minus targets (shots / xG / goals / SoT).
#
# Primary gate: fold-level proper scoring (`LogLoss`, `LPD`, `CRPS`) via `evaluate_experiments`,
# reading `diff_ll` (model − market). Secondary: a Kelly backtest for growth/CLV, because the
# stream's own standing conclusion is to judge on growth G rather than LogLoss alone.
#
# A CLEAN NEGATIVE IS A VALID RESULT and must be written into NOTES.md the same way a positive is.
#
# Run on the server (kaimon REPL), where the DB is local:
#   ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
#   include("current_development/plus_minus_ratings/r10_src_experiment.jl")

using BayesianFootball
using DataFrames, Dates, Statistics, Printf

const BF = BayesianFootball
const D  = BF.Data
const F  = BF.Features
const M  = BF.Models.PreGame
const E  = BF.Experiments
const EV = BF.Evaluation

const SAVE_DIR       = "./data/experiments/plus_minus"
const TARGET_SEASONS = ["24/25", "25/26"]
const HISTORY_SEASONS = 2
const DYNAMICS_COL   = :match_month
const SAMPLES        = 1000
const WARMUP         = 300
const CHAINS         = 4

# ==========================================
# 0. DATA
# ==========================================
# The store now carries `bbc_events` (raw BBC shot commentary) and player ids on `incidents`.
# `force=true` once after the schema change; the cache is picked up on later runs.
ds = D.load_datastore_cached(D.ScottishLower())

@printf("matches %d | lineups %d | incidents %d | bbc %d | bbc_events %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.incidents), nrow(ds.bbc), nrow(ds.bbc_events))

# ==========================================
# 1. MODELS
# ==========================================
apm(f) = M.DynamicGoalsPlusMinusLeagueTimeDecayModel(player_ratings_feature = f)

MODELS = [
    # --- the APM variant sweep -----------------------------------------------------------
    "apm_shots"      => apm(F.ShotsPlusMinusFeature()),           # GREEN-LIT cell
    "apm_xg"         => apm(F.XGPlusMinusFeature()),              # least team-loaded
    "apm_goals"      => apm(F.GoalsPlusMinusFeature()),           # base paper's target
    "apm_sot"        => apm(F.ShotsOnTargetPlusMinusFeature()),
    # --- the two baselines ---------------------------------------------------------------
    # The no-APM twin: same goals likelihood, same dynamics, market pillars OFF so the ONLY
    # difference from apm_* is the player pillar.
    "goals_baseline" => M.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel(market_on = false),
    # The current best team-level engine on this segment.
    "funnel_winner"  => M.DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel(),
]

# ==========================================
# 2. RUN
# ==========================================
results = Dict{String, Any}()
for (name, model) in MODELS
    @info "=== $name ==="
    task = E.create_experiment_task(
        ds, model, name, SAVE_DIR;
        target_seasons  = TARGET_SEASONS,
        history_seasons = HISTORY_SEASONS,
        dynamics_col    = DYNAMICS_COL,
        samples         = SAMPLES,
        warmup          = WARMUP,
        chains          = CHAINS,
    )
    r = E.run_experiment(task)
    E.save_experiment(r)
    results[name] = r
end

# ==========================================
# 3. HEAD-TO-HEAD — proper scoring (the primary gate)
# ==========================================
order = first.(MODELS)
exps  = [results[n] for n in order]

scores = EV.evaluate_experiments(
    EV.AbstractScoringRule[EV.LogLoss(), EV.LPD(), EV.CRPS()], exps, ds)
println("\n===== PROPER SCORING (diff_ll = model − market; negative is better) =====")
println(scores)

# ==========================================
# 4. ECONOMIC — Kelly growth / ROI / CLV
# ==========================================
bt = BF.BackTesting.run_backtest(ds, exps, [BF.Signals.KellyCriterion(1.0)])
summary = BF.BackTesting.summarize_models(bt)
println("\n===== KELLY BACKTEST =====")
println(summary)

# ==========================================
# 5. VERDICT SCAFFOLD
# ==========================================
println("""

Write the verdict into current_development/plus_minus_ratings/NOTES.md:
  Q1  apm_shots vs goals_baseline  (its no-APM twin)   -> does the pillar add anything?
  Q2  apm_shots vs funnel_winner   (best team-level)   -> does it beat the incumbent?
  Q3  which PM target wins the sweep, and does the ordering match WP7's reliability ordering?
A clean negative is a valid, publishable outcome.
""")
