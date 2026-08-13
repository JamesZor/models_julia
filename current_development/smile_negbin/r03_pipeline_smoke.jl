# current_development/smile_negbin/r03_pipeline_smoke.jl
#
# RUNNER. The REAL pipeline, on two folds of real data.
#
# `r01_smoke.jl` proved the model compiles and the prediction round trip is valid, but it did so
# on 140 fabricated matches with one chain. That says nothing about whether the engine CONVERGES
# on real folds — which is the question that decides whether a multi-hour full run is worth
# starting, and the one WP2 already found the Poisson parent failing globally (79: global R-hat
# 1.616 on funnel parameters, window 1.0097).
#
# So this runs the shipped pipeline end to end —
#
#     create_id_boundaries -> create_features -> Training.train -> extract_oos_predictions
#                          -> model_inference
#
# — on the pinned Ireland 79 DataStore, restricted to two folds.
#
# ---------------------------------------------------------------------------------------------
# HOW THE FOLDS ARE RESTRICTED, AND WHY THIS WAY
# ---------------------------------------------------------------------------------------------
#
# `GroupedCVConfig` already has the two knobs: `warmup_period` is the FIRST time step kept and
# `end_dynamics` the LAST (splitting/methods.jl:150-152). Setting them to 11 and 12 keeps 2026
# biweeks 11-12 — inside the order-book corpus window (biweeks 8..12), so these are folds the
# study actually consumes rather than arbitrary ones.
#
# The important property: restricting the FOLD LIST does not shrink any fold's TRAINING SET. Each
# fold still trains on two history seasons plus all of 2026 up to its own biweek, so these are the
# two LARGEST, SLOWEST folds of the run. Per-chain cost and posterior geometry are production-
# faithful; only the number of folds is reduced. That is what makes the R-hat here meaningful.
#
# A cheaper alternative — slicing the boundary vector after the fact — was rejected: it would
# leave `extract_oos_predictions` rebuilding the FULL boundary list and zipping it positionally
# against a short `training_results`, mis-pairing folds without erroring. Restricting the splitter
# instead keeps every downstream stage consistent by construction. (Same trap WP2's runner
# documents for the DataStore cache.)
#
# ---------------------------------------------------------------------------------------------
# WHAT IS AND IS NOT PRODUCTION-EXACT
# ---------------------------------------------------------------------------------------------
#
#   production-exact:  DataStore (pinned), history_seasons, dynamics_col, chains (4),
#                      max_depth (10), accept_rate, UniformInit, queued execution
#   reduced:           samples 300 (vs 800), warmup 250 (vs 300) — ~50% of the draws
#
# The reduction only makes the convergence test HARDER, never easier: fewer draws means noisier
# R-hat and lower ESS. A pass here is therefore informative; a marginal fail would need re-testing
# at full length before being believed.
#
# ---------------------------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/smile_negbin/r03_pipeline_smoke.jl")
#
# Expect ~30-45 min: 2 folds × 4 chains = 8 chain-tasks, one wave on a 16-thread session.
# The sampler is silent for that whole stretch — a background-job watchdog reporting "failed" on
# no-output is a false alarm; check `data/l2_ireland_engines/smoke_ire79_smilenb_*` on disk.

using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization

include(joinpath(@__DIR__, "l01_smile_negbin_engine.jl"))
include(joinpath(@__DIR__, "l02_smile_negbin_predict.jl"))

using Turing: MCMCChains

const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Samplers    = BayesianFootball.Samplers
const Training    = BayesianFootball.Training

const SN3_OUT_DIR = "./data/l2_ireland_engines"
const SN3_LOG     = joinpath(@__DIR__, "r03_out.txt")

# Fold window: 2026 biweeks 11-12, inside the order-book corpus window (8..12).
const SN3_FIRST_STEP = 11
const SN3_LAST_STEP  = 12

# Sampler: production-exact except for draw counts.
const SN3_SAMPLES, SN3_WARMUP, SN3_CHAINS, SN3_MAX_DEPTH = 300, 250, 4, 10

const SN3_FAILURES = String[]
function sn3_check(ok::Bool, msg::AbstractString)
    @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg)
    ok || push!(SN3_FAILURES, msg)
    return ok
end
empty!(SN3_FAILURES)

# ===================================================================
# 1. Pinned DataStore — the same one WP2 trained on
# ===================================================================

sn3_pin_path = joinpath(SN3_OUT_DIR, "ds_ire79.jls")
isfile(sn3_pin_path) || error("r03: no pin at $sn3_pin_path — run " *
                              "orderbook_layer2/r02_train_ireland.jl first")
sn3_ds = deserialize(sn3_pin_path)
@printf("\nDataStore: %d matches, segment %s\n", nrow(sn3_ds.matches), string(sn3_ds.segment))

# ===================================================================
# 2. Config — the engine, the restricted splitter, the queued sampler
# ===================================================================

sn3_model = DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    dispersion_config      = PreGame.HomeAwayDispersion(),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(
                                 Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = 4),
    market_on              = true,
    supremacy_weight       = 0.4,
    smile_weight           = 0.4,
)

sn3_splitter = Data.GroupedCVConfig(
    tournament_groups = [Data.tournament_ids(sn3_ds.segment)],
    target_seasons    = ["2026"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = SN3_FIRST_STEP,
    end_dynamics      = SN3_LAST_STEP,
    stop_early        = false,
)

sn3_sampler = Samplers.QueuedNUTSConfig(
    n_samples      = SN3_SAMPLES,
    n_chains       = SN3_CHAINS,
    n_warmup       = SN3_WARMUP,
    accept_rate    = 0.65,
    max_depth      = SN3_MAX_DEPTH,
    initialisation = Samplers.UniformInit(-2.0, 2.0),
    show_progress  = false,
)

sn3_config = Experiments.ExperimentConfig(
    name            = "smoke_ire79_smilenb",
    model           = sn3_model,
    splitter        = sn3_splitter,
    training_config = Training.TrainingConfig(
                          sn3_sampler,
                          Training.Independent(parallel = true,
                                               max_concurrent_tasks = Threads.nthreads()),
                          nothing, false),
    save_dir        = SN3_OUT_DIR,
)

# Show the folds BEFORE spending on them — a mis-set window is cheap to catch here and expensive
# to discover 40 minutes later.
sn3_bounds = Data.create_id_boundaries(sn3_ds, sn3_splitter)
println("\nFolds to train (", length(sn3_bounds), "):")
for (ids, md) in sn3_bounds
    @printf("  season %s  biweek %2d   train ids %d\n",
            string(md.target_season), md.time_step, length(ids))
end
isempty(sn3_bounds) && error("r03: splitter produced no folds — check warmup_period/end_dynamics")

# ===================================================================
# 3. Train
# ===================================================================

println("\n", "="^90)
@printf("TRAIN  %d folds × %d chains, %d+%d draws, max_depth %d, %d threads\n",
        length(sn3_bounds), SN3_CHAINS, SN3_WARMUP, SN3_SAMPLES, SN3_MAX_DEPTH, Threads.nthreads())
println("="^90)

sn3_t0  = time()
sn3_res = Experiments.run_experiment(sn3_ds, sn3_config)
sn3_mins = (time() - sn3_t0) / 60
Experiments.save_experiment(sn3_res)
@printf("\ntrained in %.1f min\n", sn3_mins)

sn3_check(length(sn3_res.training_results.items) == length(sn3_bounds),
          @sprintf("G-A splits kept %d / %d built (the queued trainer drops failed inits SILENTLY)",
                   length(sn3_res.training_results.items), length(sn3_bounds)))

# ===================================================================
# 4. Convergence — every parameter, not just the ones diagnostics knows about
# ===================================================================
#
# `Diagnostics.extract_chains` walks named component families (interception, kappa, home
# advantage, dynamics, dispersion). It has no branch for `log_φ`, `σ_smile`, `σ_sup` or `ν_xg`,
# so those would silently never be checked. The raw sweep below summarizes the chain itself, so
# EVERY sampled parameter is covered — including the four the component walker cannot see.

println("\n", "="^90)
println("CONVERGENCE — raw sweep over every sampled parameter")
println("="^90)

function sn3_family(p::AbstractString)
    for pre in ("inter.", "disp.", "ha.", "kap.", "p_dyn.", "log_φ")
        startswith(p, pre) && return rstrip(pre, '.')
    end
    return p in ("lp", "n_steps", "acceptance_rate", "tree_depth", "numerical_error",
                 "step_size", "nom_step_size", "is_accept", "hamiltonian_energy",
                 "hamiltonian_energy_error", "max_hamiltonian_energy_error") ? "«sampler»" : p
end

# MCMCChains renamed `ess` to `ess_bulk` at some point; accept either rather than silently
# reporting NaN for the whole ESS column.
sn3_getf(r, syms) = (for s in syms; hasproperty(r, s) && return Float64(getproperty(r, s)); end; NaN)

sn3_rows = DataFrame(fold = Int[], parameter = String[], family = String[],
                     rhat = Float64[], ess = Float64[], mean = Float64[])
for (chain, meta) in sn3_res.training_results.items
    summ = DataFrame(MCMCChains.summarize(chain))
    for r in eachrow(summ)
        p = string(r.parameters)
        push!(sn3_rows, (meta.time_step, p, sn3_family(p),
                         sn3_getf(r, (:rhat,)),
                         sn3_getf(r, (:ess, :ess_bulk, :ess_tail)),
                         sn3_getf(r, (:mean,))))
    end
end

sn3_par = filter(r -> r.family != "«sampler»" && !isnan(r.rhat), sn3_rows)
@printf("  %d sampled parameters summarized across %d folds\n",
        nrow(sn3_par), length(sn3_res.training_results.items))

println("\n  per family (max R-hat / min ESS over folds):")
sn3_fam = sort(combine(groupby(sn3_par, :family),
                       :rhat => maximum => :max_rhat,
                       :ess  => minimum => :min_ess,
                       nrow  => :n), :max_rhat, rev = true)
show(stdout, MIME"text/plain"(), sn3_fam)
println()

sn3_max_rhat = maximum(sn3_par.rhat)
sn3_worst    = sn3_par[argmax(sn3_par.rhat), :]
sn3_bad      = filter(r -> r.rhat >= 1.01, sn3_par)
sn3_ess_ok   = filter(!isnan, sn3_par.ess)
sn3_min_ess  = isempty(sn3_ess_ok) ? NaN : minimum(sn3_ess_ok)

@printf("\n  max R-hat %.4f   (%s, fold biweek %d)\n",
        sn3_max_rhat, sn3_worst.parameter, sn3_worst.fold)
@printf("  parameters with R-hat >= 1.01: %d / %d\n", nrow(sn3_bad), nrow(sn3_par))
@printf("  min ESS %.0f  (of %d post-warmup draws per fold)\n",
        sn3_min_ess, SN3_SAMPLES * SN3_CHAINS)

if !isempty(sn3_bad)
    println("\n  worst 10 non-converged:")
    show(stdout, MIME"text/plain"(), first(sort(sn3_bad, :rhat, rev = true), 10))
    println()
end

sn3_check(sn3_max_rhat < 1.01, @sprintf("G-B max R-hat < 1.01 (got %.4f)", sn3_max_rhat))
isnan(sn3_min_ess) ? println("  [SKIP] ESS not reported by this MCMCChains version") :
    sn3_check(sn3_min_ess > 100, @sprintf("min ESS > 100 (got %.0f)", sn3_min_ess))

# The two parameters this engine adds. If THESE fail while the rest pass, the NegBin edit is the
# cause; if the failures are in the funnel families (ha./kap./p_dyn.) they are the parent's,
# already documented in WP2.
for p in ("disp.log_r", "disp.δ_r_home")
    sub = filter(r -> r.parameter == p, sn3_par)
    if isempty(sub)
        sn3_check(false, "$p missing from the chain summary")
    else
        @printf("  %-16s R-hat %.4f   ESS %6.0f   mean %+.3f\n",
                p, maximum(sub.rhat), minimum(sub.ess), mean(sub.mean))
        sn3_check(maximum(sub.rhat) < 1.01, @sprintf("%s converged (R-hat %.4f)", p, maximum(sub.rhat)))
    end
end

# Divergences, per fold.
println()
for (chain, meta) in sn3_res.training_results.items
    if :numerical_error in Symbol.(names(chain))
        @printf("  fold biweek %2d  divergences %.1f%%\n",
                meta.time_step, 100 * mean(vec(Array(chain[:numerical_error]))))
    end
end

# The repo's own diagnostic view, for continuity with WP2/WP10's reported numbers.
try
    sn3_diag = Diagnostics.check_convergence(Diagnostics.extract_chains(sn3_ds, sn3_res)).df
    if !isempty(sn3_diag)
        @printf("\n  Diagnostics.check_convergence: max R-hat %.4f over %d component rows (worst %s)\n",
                maximum(sn3_diag.rhat), nrow(sn3_diag),
                string(sn3_diag[argmax(sn3_diag.rhat), :parameter]))
    end
catch e
    @warn "Diagnostics.extract_chains failed" exception = e
end

# ===================================================================
# 5. Dispersion — the quantity the engine exists to estimate
# ===================================================================

println("\n", "="^90)
println("DISPERSION  (Var/E = 1 + λ/r for NegBin(r, λ))")
println("="^90)

for (chain, meta) in sn3_res.training_results.items
    log_r = vec(Array(chain[Symbol("disp.log_r")]))
    δ     = vec(Array(chain[Symbol("disp.δ_r_home")]))
    r_a   = exp.(log_r); r_h = exp.(log_r .+ δ)
    @printf("  biweek %2d   r_h %6.2f [%5.2f, %6.2f]   r_a %6.2f [%5.2f, %6.2f]\n",
            meta.time_step,
            median(r_h), quantile(r_h, 0.05), quantile(r_h, 0.95),
            median(r_a), quantile(r_a, 0.05), quantile(r_a, 0.95))
end
println("  (r large on both sides => the Poisson parent lost nothing; r ≲ 20 => it was " *
        "understating variance)")

# ===================================================================
# 6. OOS latents and market probabilities, through the shipped path
# ===================================================================

println("\n", "="^90)
println("OOS LATENTS + INFERENCE")
println("="^90)

sn3_latents = Experiments.extract_oos_predictions(sn3_ds, sn3_res)
@printf("  %d OOS matches across %d folds\n", nrow(sn3_latents), length(sn3_bounds))
sn3_check(nrow(sn3_latents) > 0, "extract_oos_predictions returned matches")

sn3_ldf = sn3_latents.df
for c in (:λ_h, :λ_a, :λ_tot, :φ, :r_h, :r_a)
    sn3_check(c in propertynames(sn3_ldf), "latent frame carries :$c")
end
sn3_check(all(v -> all(isfinite, v) && all(>(0), v), sn3_ldf.λ_h), "every λ_h finite and positive")
sn3_check(all(v -> all(isfinite, v) && all(>(0), v), sn3_ldf.r_h), "every r_h finite and positive")
@printf("  mean λ_h %.3f   mean λ_a %.3f   mean r_h %.2f   mean r_a %.2f\n",
        mean(mean.(sn3_ldf.λ_h)), mean(mean.(sn3_ldf.λ_a)),
        mean(mean.(sn3_ldf.r_h)), mean(mean.(sn3_ldf.r_a)))

sn3_ppd = BayesianFootball.Predictions.model_inference(sn3_latents)
sn3_pdf = sn3_ppd.df
@printf("  PPD rows %d across %d markets\n", nrow(sn3_pdf), length(unique(sn3_pdf.market_name)))
sn3_check(nrow(sn3_pdf) > 0, "model_inference produced a PPD")
sn3_check(all(d -> all(isfinite, d) && all(p -> -1e-9 <= p <= 1 + 1e-9, d), sn3_pdf.distribution),
          "every PPD probability finite and in [0,1]")

# Per-market outcome mass, summed over selections within (match, market, line).
sn3_mass = combine(groupby(sn3_pdf, [:match_id, :market_name, :market_line]),
                   :distribution => (v -> mean(reduce(.+, v))) => :mass)
println("\n  mean outcome mass per market:")
show(stdout, MIME"text/plain"(),
     sort(combine(groupby(sn3_mass, :market_name),
                  :mass => minimum => :min, :mass => maximum => :max, nrow => :n), :market_name))
println()

sn3_1x2 = filter(r -> r.market_name == "1X2", sn3_mass)
sn3_check(isempty(sn3_1x2) || maximum(abs.(sn3_1x2.mass .- 1.0)) < 2e-3,
          "1X2 outcomes sum to 1 on every OOS match")
sn3_ou = filter(r -> r.market_name == "OverUnder", sn3_mass)
sn3_check(isempty(sn3_ou) || maximum(abs.(sn3_ou.mass .- 1.0)) < 2e-3,
          "O/U outcomes sum to 1 on every OOS match and line")

# ===================================================================
# 7. Verdict
# ===================================================================

sn3_verdict = isempty(SN3_FAILURES) ?
    "PIPELINE SMOKE TEST PASSED — all checks green." :
    "PIPELINE SMOKE TEST FAILED — " * string(length(SN3_FAILURES)) * " check(s): " *
        join(SN3_FAILURES, " | ")

println("\n", "="^90)
println(sn3_verdict)
println("="^90)

open(SN3_LOG, "w") do io
    println(io, "smile_negbin r03 — pipeline smoke, Ireland 79, 2026 biweeks ",
                SN3_FIRST_STEP, "-", SN3_LAST_STEP)
    println(io, "run at ", now())
    @printf(io, "folds %d, chains %d, draws %d+%d, max_depth %d, %.1f min\n",
            length(sn3_bounds), SN3_CHAINS, SN3_WARMUP, SN3_SAMPLES, SN3_MAX_DEPTH, sn3_mins)
    @printf(io, "max R-hat %.4f (%s)   params >= 1.01: %d/%d   min ESS %.0f\n",
            sn3_max_rhat, sn3_worst.parameter, nrow(sn3_bad), nrow(sn3_par), sn3_min_ess)
    println(io, "\nper family:")
    show(io, MIME"text/plain"(), sn3_fam)
    println(io, "\n\nOOS matches: ", nrow(sn3_latents), "   PPD rows: ", nrow(sn3_pdf))
    println(io, "\n", sn3_verdict)
end
println("\nwrote $SN3_LOG")

nothing
