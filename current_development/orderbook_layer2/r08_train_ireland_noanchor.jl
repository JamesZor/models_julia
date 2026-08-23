# current_development/orderbook_layer2/r08_train_ireland_noanchor.jl
#
# WP10. The `market_on = false` control r21 never ran against an executable de-vigged close.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS IS AND IS NOT
# ---------------------------------------------------------------------------------------------
#
# Identical to `r02_train_ireland.jl` in every respect except one flag. Same pinned DataStores
# (`ds_ire79.jls`, `ds_ire718_bfpillar.jls` — NOT rebuilt), same target seasons, same splitter,
# same sampler settings, same `supremacy_weight`/`smile_weight` (irrelevant once gated to zero,
# kept identical so nothing else about the config differs). The only change is
# `market_on = false`, which zeroes BOTH `Turing.@addlogprob!` market-anchor terms
# (`outfield_xg_smile_double_poisson.jl:142,150`) and leaves everything else — goals, xG,
# player dynamics, kappa, home advantage, interception — untouched.
#
# This is the WP8/RESULTS.md §8.1 question: two of the engine's four likelihood pillars are the
# market (supremacy + smile, both weight 0.4). If the unanchored engine is much worse against the
# executable de-vigged close, the anchoring is carrying the accuracy. If it is comparable, the
# anchoring may be suppressing a signal the xG/goals pillars would otherwise express. r21's own
# grid had a `market_on = false` cell but only ever scored it on held-out log-loss against other
# model cells, never against the closing Betfair/SofaScore price the way Route 2 now can.
#
# ---------------------------------------------------------------------------------------------
# ONE MODELLING CONSEQUENCE WORTH KNOWING BEFORE READING THE OUTPUT
# ---------------------------------------------------------------------------------------------
#
# `log_φ` (the O/U smile shape) enters ONLY through the smile pillar — it never touches the goals
# likelihood (see the model file's header). With `market_on = false` that pillar is zeroed, so
# `log_φ`'s posterior is its prior, `Normal(0, smile_shape_sd)`. In expectation that is `φ(K) ≈ 1`,
# i.e. totals default to a plain-Poisson shape rather than a market-informed smile. So this control
# does not isolate "no supremacy info" in isolation — it also removes the smile pricing mechanism
# for O/U specifically. Worth stating up front rather than discovering it while reading a totals
# result that looks different from the goals/1X2 story.
#
# `σ_sup` and `σ_smile` are likewise unidentified under `market_on = false` (both SAMPLED
# parameters, both only appear inside the now-zeroed addlogprob terms) — they will sample their
# priors. Not a defect; `gate_experiment`'s R-hat check should still pass since sampling a proper
# prior is a well-behaved geometry, but it means those two diagnostics are uninformative here.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server, 32 threads / 16 in the connected kaimon session)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/orderbook_layer2/r08_train_ireland_noanchor.jl")
#
# ~11h combined on the same queue that trained the anchored pair (79: 6h57m, 718: 4h01m).

using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Data        = BayesianFootball.Data

const OUT_DIR = "./data/l2_ireland_engines"
mkpath(OUT_DIR)

# ===================================================================
# 1. Pinned DataStores — load the SAME pins r02 wrote, never rebuild
# ===================================================================

"""
    loaded_pin(tag) -> DataStore

Load a pin `r02_train_ireland.jl` already wrote. Errors rather than building, because building
here would silently create a DataStore that is not bit-identical to the one the anchored engine
trained on (season coverage, xG presence etc. can drift with the source DB) — and the entire
point of this control is that the only thing that differs between the two engines is the flag.
"""
function loaded_pin(tag::String)
    path = joinpath(OUT_DIR, "ds_$(tag).jls")
    isfile(path) || error("r08: no pin at $path — run r02_train_ireland.jl first, " *
                          "this control must share its DataStores exactly")
    return deserialize(path)
end

# ===================================================================
# 2. The engine — r21's winning cell, market pillar OFF
# ===================================================================

src_noanchor() = PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    dispersion_config      = PreGame.HomeAwayDispersion(),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(
                                 Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = 4),
    market_on              = false,        # <-- the ONE difference from src_sup40_sw40
    supremacy_weight       = 0.4,          # inert once market_on = false; kept identical to r02
    smile_weight           = 0.4,          # so nothing else about the config differs
)

# Sampler / splitter, identical to r02/r21 for comparability.
const TARGET_SEASONS = ["2025", "2026"]
const DYNAMICS_COL   = :match_biweek
const SAMPLES, WARMUP, CHAINS, MAX_DEPTH = 800, 300, 4, 10

# ===================================================================
# 3. Train
# ===================================================================

function train_one(ds, name::String)
    println("\n", "#"^90, "\n# TRAIN: $name\n", "#"^90)
    task = Experiments.create_experiment_task(
        ds, src_noanchor(), name, OUT_DIR;
        target_seasons  = TARGET_SEASONS,
        history_seasons = 2,
        warmup_period   = 0,
        dynamics_col    = DYNAMICS_COL,
        samples         = SAMPLES,
        warmup          = WARMUP,
        chains          = CHAINS,
        use_queue       = true,
        max_depth       = MAX_DEPTH,
    )
    t0 = time()
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    @printf("[%s] done in %.1f min, %d splits\n", name, (time() - t0) / 60,
            length(res.training_results.items))
    return res
end

# ===================================================================
# 4. Gates — identical to r02's gate_experiment
# ===================================================================

function gate_experiment(res, ds, label::String)
    println("\n", "="^90, "\nGATES: $label\n", "="^90)

    bounds  = Data.create_id_boundaries(ds, res.config.splitter)
    n_built = length(bounds)
    n_kept  = length(res.training_results.items)
    ga = n_built == n_kept
    @printf("G-A  splits kept       %3d / %3d built            %s\n", n_kept, n_built,
            ga ? "PASS" : "FAIL  <- queued trainer dropped splits SILENTLY")

    maxr, n_par, worst = NaN, 0, ""
    maxr_w, n_par_w, bad_folds = NaN, 0, Any[]
    try
        diag = Diagnostics.check_convergence(Diagnostics.extract_chains(ds, res)).df
        if !isempty(diag)
            maxr  = maximum(diag.rhat)
            n_par = nrow(diag)
            worst = string(diag[argmax(diag.rhat), :parameter])

            bad = filter(r -> r.rhat >= 1.01, diag)
            bad_folds = isempty(bad) ? Any[] :
                sort(unique([(string(r.target_season), r.week) for r in eachrow(bad)]))

            win = filter(r -> string(r.target_season) == "2026" && 8 <= r.week <= 12, diag)
            if !isempty(win)
                maxr_w  = maximum(win.rhat)
                n_par_w = nrow(win)
            end
        end
    catch e
        @warn "convergence extraction failed" exception = e
    end
    gb_all = !isnan(maxr)   && maxr   < 1.01
    gb     = !isnan(maxr_w) && maxr_w < 1.01
    @printf("G-B  max R-hat (all)   %.4f  (n=%d, worst=%s)   %s\n", maxr, n_par, worst,
            gb_all ? "PASS" : "FAIL")
    @printf("     max R-hat (win)   %.4f  (n=%d, 2026 biweeks 8..12)   %s\n", maxr_w, n_par_w,
            gb ? "PASS" : "FAIL  <- the corpus window itself did not converge")
    isempty(bad_folds) || @printf("     non-converged folds (season, biweek): %s\n",
                                  string(bad_folds))

    steps_2026 = sort(unique([md.time_step for (_, md) in bounds
                              if string(md.target_season) == "2026"]))
    ob = filter(r -> !ismissing(r.match_date) &&
                     Date(2026, 5, 29) <= r.match_date <= Date(2026, 8, 9), ds.matches)
    need = isempty(ob) ? Int[] : sort(unique(skipmissing(ob.match_biweek)))
    gc = !isempty(need) && all(n -> n in steps_2026, need)
    @printf("G-C  order-book window  biweeks needed %s\n", string(need))
    @printf("     folds reach 2026 biweeks %s        %s\n",
            string(steps_2026), gc ? "PASS" : "FAIL")

    return (label = label, splits_kept = n_kept, splits_built = n_built,
            max_rhat = maxr, max_rhat_window = maxr_w, bad_folds = bad_folds,
            ga = ga, gb = gb, gb_all = gb_all, gc = gc, pass = ga && gb && gc)
end

# ===================================================================
# 5. Run
# ===================================================================

println("\n", "="^90)
println("WP10  —  market_on = false control on Ireland 79 and 718")
println("="^90)

ds79  = loaded_pin("ire79")
ds718 = loaded_pin("ire718_bfpillar")   # the SAME Betfair-pillar store 718's anchored run used

res79  = train_one(ds79,  "l2_ire79_noanchor")
res718 = train_one(ds718, "l2_ire718_noanchor")

g79  = gate_experiment(res79,  ds79,  "79 Ireland Premier (noanchor)")
g718 = gate_experiment(res718, ds718, "718 Ireland First Division (noanchor)")

println("\n", "="^90)
for g in (g79, g718)
    @printf("%-32s splits %3d/%3d   max R-hat %.4f   %s\n",
            g.label, g.splits_kept, g.splits_built, g.max_rhat, g.pass ? "PASS" : "FAIL")
end
println("="^90)
println("\nSaved to $OUT_DIR — l2_ire79_noanchor / l2_ire718_noanchor, alongside the anchored pair.")

nothing
