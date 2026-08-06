# current_development/plus_minus_ratings/r11_pillar_only.jl
#
# RUNNER. The arm r10 does NOT cover: can the APM rating REPLACE team strength, rather than merely
# adjust it?
#
# r10's `apm_shots` is the team model PLUS a player pillar:
#     log λ_h = μ + δ_month + δ_league + ha_h + dyn.α_h + dyn.β_a + w_att·R_h − w_def·R_a
# so `w_att`/`w_def` can only earn their keep on the LINEUP-VARYING RESIDUAL — the free per-team
# α/β soak up the team-level component. That is the right conservative test of marginal value
# (WP7's central worry was that RAPM is team strength in disguise: club R² 0.389 on `y_shots`),
# and it keeps the difference from the no-APM twin down to a single term.
#
# But it is NOT the form the Ireland engines take. `OutfieldPlayerDynamicsConfig`
# (components/dynamics/player_level/positional.jl) samples FOUR GLOBAL SCALARS and nothing
# per-team, so in `outfield_xg.jl` a team is good purely because its players are good. That is a
# different and more ambitious hypothesis, and 56/57 has never been tested on it.
#
# This runner supplies it by swapping in `StaticZeroDynamics` (α ≡ β ≡ 0, samples nothing), which
# holds EVERYTHING else fixed — same likelihood, interception, home advantage, league offset and
# pillar parameterisation. Any difference is attributable to the team-dynamics term alone.
#
# PRIOR NOTE. With α/β gone, `w_att`/`w_def` must carry the whole between-team spread, not a
# residual, so the r10 prior Normal(0, 0.3) is too tight: the outfield rating sum has sd ~0.2, and
# reproducing a realistic team spread in log λ (~0.35) needs weights of order 1.5-2. The priors are
# widened accordingly — leaving them at 0.3 would hobble this arm and produce a null for a purely
# mechanical reason.
#
# Run AFTER r10 has saved its experiments. Loads them from disk and scores all arms together:
#   JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16 \
#       current_development/plus_minus_ratings/r11_pillar_only.jl

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

# ==========================================
# 1. THE PILLAR-ONLY ARM
# ==========================================
pillar_only = M.DynamicGoalsPlusMinusLeagueTimeDecayModel(
    dynamics_config        = M.StaticZeroDynamics(),          # α ≡ β ≡ 0
    player_ratings_feature = F.ShotsPlusMinusFeature(),       # the green-lit cell
    w_att_prior            = Normal(0.0, 1.5),                # must carry the whole team spread
    w_def_prior            = Normal(0.0, 1.5),
)

@info "=== apm_pillar_only ==="
task = E.create_experiment_task(
    ds, pillar_only, "apm_pillar_only", SAVE_DIR;
    target_seasons  = TARGET_SEASONS,
    history_seasons = HISTORY_SEASONS,
    dynamics_col    = DYNAMICS_COL,
    samples         = SAMPLES,
    warmup          = WARMUP,
    chains          = CHAINS,
)
res = E.run_experiment(task)
E.save_experiment(res)

# ==========================================
# 2. SCORE IT ALONGSIDE THE r10 ARMS
# ==========================================
# r10 saved one timestamped directory per model; take the newest per model name.
function load_latest(save_dir::AbstractString, names::Vector{String})
    isdir(save_dir) || return E.ExperimentResults[], String[]
    dirs = readdir(save_dir)
    out, got = E.ExperimentResults[], String[]
    for n in names
        cands = filter(d -> startswith(d, n * "_"), dirs)
        isempty(cands) && continue
        try
            push!(out, E.load_experiment(joinpath(save_dir, maximum(cands))))
            push!(got, n)
        catch err
            @warn "could not load $n" exception = err
        end
    end
    return out, got
end

wanted = ["apm_shots", "apm_xg", "apm_goals", "apm_sot",
          "apm_pillar_only", "goals_baseline", "funnel_winner"]
exps, got = load_latest(SAVE_DIR, wanted)
@info "loaded arms" got

scores = EV.evaluate_experiments(
    EV.AbstractScoringRule[EV.LogLoss(), EV.LPD(), EV.CRPS()], exps, ds)
println("\n===== PROPER SCORING, ALL ARMS (diff_ll = model − market; negative is better) =====")
println(scores)

# ==========================================
# 3. ECONOMIC — growth / ROI per selection
# ==========================================
# `market_config` is REQUIRED: run_backtest defaults it to `nothing` and model_inference then
# errors ("market_config must be provided"). r10 omitted it and lost its backtest after a 7-hour
# train. Wrapped in try so a backtest failure can never discard the scoring above.
LEDGER = nothing; TEARSHEET = nothing
try
    global LEDGER = BF.BackTesting.run_backtest(
        ds, exps, [BF.Signals.BayesianKelly()];
        market_config = D.Markets.DEFAULT_MARKET_CONFIG)
    println("\n===== KELLY BACKTEST, ALL ARMS (by model) =====")
    println(BF.BackTesting.summarize_models(LEDGER))

    global TEARSHEET = BF.BackTesting.generate_tearsheet(LEDGER)
    core = [:home, :draw, :away, :btts_yes, :btts_no,
            :under_15, :over_15, :under_25, :over_25, :under_35, :over_35]
    for val in (:roi_pct, :hurdle_G, :bets_placed)
        println("\n", "="^70, "\n $val by selection\n", "="^70)
        sub = filter(r -> Symbol(r.selection) in core, TEARSHEET)
        println(unstack(sub, :selection, :model_name, val))
    end
catch err
    @error "backtest failed — the scoring table above still stands" exception = (err, catch_backtrace())
end

println("""

H1 vs H2 — the question this arm exists to answer:
  apm_shots       = team dynamics + player pillar  ("players ADJUST team strength")
  apm_pillar_only = player pillar only             ("players ARE team strength", Ireland's form)
If pillar_only collapses, the RAPM rating cannot stand in for team strength on 56/57 — which
would be consistent with it being a 730-day-half-life ridge that zeroes low-minute players,
rather than a per-match form tracker like the SofaScore rating the Ireland engines consume.
""")
