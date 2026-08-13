# current_development/smile_negbin/r02_train_ireland.jl
#
# RUNNER. Train the smile-NegBin engine on Ireland 79 and 718, independently.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS IS AND IS NOT
# ---------------------------------------------------------------------------------------------
#
# This is a CONVERGENCE + FIT validation of the new engine, not an allocation study. It answers
# one question: does replacing the goals likelihood with `RobustNegativeBinomial(r, λ)` still
# train cleanly on the real folds, and what does the posterior say `r` actually is?
#
# It deliberately does NOT re-run the WP9/WP10-style Route-2 judging. That is a follow-up, and it
# reuses `orderbook_layer2`'s existing harness (`route2_setup`, `w_star`, `book_skill`) pointed at
# this experiment exactly as `r09` pointed it at the `noanchor` control.
#
# ---------------------------------------------------------------------------------------------
# EVERYTHING EXCEPT THE LIKELIHOOD IS HELD FIXED, ON PURPOSE
# ---------------------------------------------------------------------------------------------
#
# Same pinned DataStores (`ds_ire79.jls`, `ds_ire718_bfpillar.jls` — loaded, never rebuilt), same
# `target_seasons`, same `dynamics_col`, same sampler settings, same `supremacy_weight` /
# `smile_weight` (0.4 / 0.4) as `orderbook_layer2/r02_train_ireland.jl` and
# `r08_train_ireland_noanchor.jl`. So the resulting experiments sit alongside the anchored and
# unanchored Poisson pair on identical folds, and any later comparison is apples-to-apples by
# construction rather than by argument.
#
# The pins matter for a specific reason documented at length in the WP2 runner: the DataStore
# cache has a 48h TTL and rebuilds SILENTLY. A rebuilt store would re-derive boundaries that get
# zipped POSITIONALLY against the old training results, mis-pairing folds without erroring. This
# runner therefore errors if a pin is missing rather than building one.
#
# ---------------------------------------------------------------------------------------------
# THE THING TO ACTUALLY WATCH
# ---------------------------------------------------------------------------------------------
#
# WP2 found the Poisson version's GLOBAL R-hat failing on funnel parameters in some folds (79:
# global 1.616 against a window figure of 1.0097). `HomeAwayDispersion` adds exactly two scalars
# (`disp.log_r`, `disp.δ_r_home`), which is the whole reason it was chosen over the hierarchical
# ladder — but two more parameters is still two more, and G-B is where that shows up. §5's
# dispersion report is the payoff: if the posterior `r` comes back very large on both leagues,
# the extra parameter bought nothing and the Poisson parent was right; if it comes back small
# (r ≲ 20, i.e. Var/E ≈ 1 + λ/r meaningfully above 1), the parent was structurally understating
# variance exactly as WP2's 1X2 dispersion diagnostic suggested.
#
# ---------------------------------------------------------------------------------------------
# PERSISTENCE CAVEAT (prototype stream)
# ---------------------------------------------------------------------------------------------
#
# The model type lives in `Main` (it is a prototype, per CLAUDE.md), so the serialized
# `ExperimentResults` reference a `Main.DynamicSmileDoubleNegBin...` type. Any session that loads
# these results must `include` l01/l02 FIRST or deserialization will fail to resolve the type.
# That constraint disappears if/when the engine graduates to `src/`.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server, 32 threads / 16 in the connected kaimon session)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/smile_negbin/r02_train_ireland.jl")
#
# Budget: the Poisson pair took 6h57m (79) + 4h01m (718) on this queue. NegBin logpdf costs more
# per gradient than Poisson (two extra `loggamma` calls per observation), so expect somewhat more.

using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization

include(joinpath(@__DIR__, "l01_smile_negbin_engine.jl"))
include(joinpath(@__DIR__, "l02_smile_negbin_predict.jl"))

const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics

# Globals are `sn_`-prefixed: these runners share a long-lived REPL with `orderbook_layer2`'s,
# whose `train_one` / `gate_experiment` / `res79` are already bound there.
const SN_OUT_DIR = "./data/l2_ireland_engines"
mkpath(SN_OUT_DIR)

# ===================================================================
# 1. Pinned DataStores — load the SAME pins WP2 wrote, never rebuild
# ===================================================================

function sn_loaded_pin(tag::String)
    path = joinpath(SN_OUT_DIR, "ds_$(tag).jls")
    isfile(path) || error("r02(smile_negbin): no pin at $path — run " *
                          "orderbook_layer2/r02_train_ireland.jl first; this engine must share " *
                          "its DataStores exactly for the comparison to mean anything")
    return deserialize(path)
end

# ===================================================================
# 2. The engine — src_sup40_sw40's cell, NegBin goals likelihood
# ===================================================================

sn_smile_negbin() = DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    dispersion_config      = PreGame.HomeAwayDispersion(),   # the field the parent left inert
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(
                                 Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = 4),
    market_on              = true,
    supremacy_weight       = 0.4,     # r02-exact; the market pillars are untouched by this engine
    smile_weight           = 0.4,
)

# Sampler / splitter, identical to orderbook_layer2's r02/r08 for comparability.
const SN_TARGET_SEASONS = ["2025", "2026"]
const SN_DYNAMICS_COL   = :match_biweek
const SN_SAMPLES, SN_WARMUP, SN_CHAINS, SN_MAX_DEPTH = 800, 300, 4, 10

# ===================================================================
# 3. Train
# ===================================================================

function sn_train_one(ds, name::String)
    println("\n", "#"^90, "\n# TRAIN: $name\n", "#"^90)
    task = Experiments.create_experiment_task(
        ds, sn_smile_negbin(), name, SN_OUT_DIR;
        target_seasons  = SN_TARGET_SEASONS,
        history_seasons = 2,
        warmup_period   = 0,
        dynamics_col    = SN_DYNAMICS_COL,
        samples         = SN_SAMPLES,
        warmup          = SN_WARMUP,
        chains          = SN_CHAINS,
        use_queue       = true,
        max_depth       = SN_MAX_DEPTH,
    )
    t0 = time()
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    @printf("[%s] done in %.1f min, %d splits\n", name, (time() - t0) / 60,
            length(res.training_results.items))
    return res
end

# ===================================================================
# 4. Gates — identical bar to WP2 and WP10
# ===================================================================
#
# G-A splits kept: the queued trainer drops a split that fails NUTS initialisation WITHOUT
#     raising, so a short `training_results` is the only symptom.
# G-B convergence, reported twice: globally, and restricted to the folds the order-book corpus
#     actually consumes (2026, biweeks 8..12). `MatchDay.select_split` picks ONE fold per fixture
#     by date, so a fold outside that window never prices anything.
# G-C coverage: folds must reach the biweeks the corpus lives in (2026-05-29 .. 2026-08-09).

function sn_gate_experiment(res, ds, label::String)
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
# 5. Dispersion report — the reason this engine exists
# ===================================================================
#
# Read `disp.log_r` / `disp.δ_r_home` straight off each fold's chain rather than through
# `extract_parameters`, so the report is available even if a fold has no OOS matches. The
# variance ratio is the interpretable quantity: Var/E = 1 + λ/r for NegBin(r, λ), evaluated at a
# typical league λ.
function sn_dispersion_report(res, label::String; λ_ref::Float64 = 1.35)
    println("\n", "-"^90, "\nDISPERSION: $label  (Var/E = 1 + λ/r at λ = $λ_ref)\n", "-"^90)
    r_h_all, r_a_all = Float64[], Float64[]
    for (chain, _) in res.training_results.items
        try
            log_r = vec(Array(chain[Symbol("disp.log_r")]))
            δ     = vec(Array(chain[Symbol("disp.δ_r_home")]))
            push!(r_a_all, mean(exp.(log_r)))
            push!(r_h_all, mean(exp.(log_r .+ δ)))
        catch e
            @warn "dispersion extraction failed on a fold" exception = e
        end
    end
    if isempty(r_h_all)
        println("  no folds yielded dispersion parameters")
        return (label = label, r_h = NaN, r_a = NaN)
    end
    r_h, r_a = median(r_h_all), median(r_a_all)
    @printf("  folds %d   median r_h %.2f  (Var/E %.3f)   median r_a %.2f  (Var/E %.3f)\n",
            length(r_h_all), r_h, 1 + λ_ref / r_h, r_a, 1 + λ_ref / r_a)
    @printf("  fold range r_h [%.2f, %.2f]   r_a [%.2f, %.2f]\n",
            minimum(r_h_all), maximum(r_h_all), minimum(r_a_all), maximum(r_a_all))
    println("  (very large r on both sides => the Poisson parent lost nothing; " *
            "r ≲ 20 => it was understating variance)")
    return (label = label, r_h = r_h, r_a = r_a)
end

# ===================================================================
# 6. Run
# ===================================================================

println("\n", "="^90)
println("smile_negbin WP1  —  NegBin goals likelihood on Ireland 79 and 718")
println("="^90)

sn_ds79  = sn_loaded_pin("ire79")
sn_ds718 = sn_loaded_pin("ire718_bfpillar")   # the SAME Betfair-pillar store the pair trained on

sn_res79  = sn_train_one(sn_ds79,  "l2_ire79_smilenb")
sn_res718 = sn_train_one(sn_ds718, "l2_ire718_smilenb")

sn_g79  = sn_gate_experiment(sn_res79,  sn_ds79,  "79 Ireland Premier (smile-negbin)")
sn_g718 = sn_gate_experiment(sn_res718, sn_ds718, "718 Ireland First Division (smile-negbin)")

sn_d79  = sn_dispersion_report(sn_res79,  "79 Ireland Premier")
sn_d718 = sn_dispersion_report(sn_res718, "718 Ireland First Division")

println("\n", "="^90)
for g in (sn_g79, sn_g718)
    @printf("%-38s splits %3d/%3d   max R-hat %.4f   %s\n",
            g.label, g.splits_kept, g.splits_built, g.max_rhat, g.pass ? "PASS" : "FAIL")
end
println("="^90)
println("\nSaved to $SN_OUT_DIR — l2_ire79_smilenb / l2_ire718_smilenb, alongside the " *
        "anchored and noanchor Poisson pairs.")

nothing
