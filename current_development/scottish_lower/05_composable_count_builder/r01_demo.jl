# ==============================================================================
# 05 — COMPOSABLE COUNT MODEL BUILDER : THE PROOF
# ==============================================================================
#
# WHAT THIS IS
#   One builder, one Turing engine, one reference, one extractor — assembled into
#   the four Scottish Lower Poisson arms that currently cost five hand-written
#   engines, and shown to be THE SAME MODELS.
#
#       00_team_poisson            baseline                 → build(b)
#       02_poisson_wealth          + squad wealth            → add!(b, WealthCovariate())
#       03_poisson_distance        + travel distance         → add!(b, DistanceCovariate())
#       04_poisson_wealth_distance + both                    → add!(b, both)
#
# WHAT THIS IS NOT
#   Not a betting experiment, not a model-selection run, not a gate 6/7 evaluation.
#   Nothing here decides whether wealth or distance is worth having; arms 02-04
#   already own that question. This decides only whether the composable builder
#   reproduces them exactly, so that future covariates cost one struct instead of
#   one engine.
#
# FILTRATION / COMPARABILITY CONTRACT
#   Every comparison is made on ONE FeatureSet — the fold-1 features built from the
#   HAND-WRITTEN arm's `required_features`. Both models are then fitted to
#   literally the same numbers, so any difference is the engine's and nothing
#   else's. Fold 1 comes from the shared protocol (`sl_build_folds`), kickoff
#   filtration included; this runner does not cut its own folds.
#
# WHAT WOULD FALSIFY THE CLAIM
#   Any non-zero row in §7 or §9. "Close enough" is not the standard here: two
#   implementations of the same density evaluated at the same θ must agree to the
#   last bit, and a sampler given the same seed and the same θ layout must produce
#   the same chain. Section 6 states the one place the engines genuinely differ
#   (arm 00 has no `clamp`) and measures the headroom that makes it irrelevant.
#
# USAGE
#   julia --project -t 8
#   include("current_development/scottish_lower/05_composable_count_builder/r01_demo.jl")
#
#   Sections 1-8 and 10 are deterministic and take seconds. Section 9 fits 96 NUTS
#   chains and is OFF by default — see the note at CB_RUN_POSTERIOR in §2.
#
# ==============================================================================


# %% ===========================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using DataFrames
using Dates
using Distributions
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "_protocol", "ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol

# The thing under test. Loading it also pulls in, unmodified, the two feature
# loaders the hand-written arms use —
#   02_poisson_wealth/l00_feature_poisson.jl   (the wealth feature AND the three
#                                               hand-written arm model structs)
#   03_poisson_distance/l00_distance_feature.jl
# — so both sides of every comparison below are wired to literally the same
# extractors.
include(joinpath(@__DIR__, "l05_parity.jl"))

# Arm 00 lives in its own file and can be included directly.
include(joinpath(@__DIR__, "..", "00_team_poisson", "l01_model.jl"))          # tp00_model

# Arms 02/03/04 are NOT included from their `l01_model.jl`. Those three files
# re-`include` 02_poisson_wealth/l00_feature_poisson.jl — arm 02 unconditionally,
# arms 03 and 04 behind a guard on `SLFeaturePoissonModel`, a name that file never
# defines — so including them here would REDEFINE the very structs under test,
# midway through this runner. Harmless when each arm's walkthrough is run alone;
# not harmless here. Their configurations are reconstructed in §3 instead, from
# the same component objects §2 gives the builder, which is a stricter comparison
# than trusting two sets of constructor defaults to agree.
# Filed as docs/tickets/T006-scottish-lower-arm-include-guards.md.


# %% ===========================================================================
# 2. Configuration
# ==============================================================================
# Every number here is arm 00/02/03/04's own default. The builder is not allowed a
# more favourable configuration than the engines it claims to reproduce.

const CB_HALF_LIFE     = 180.0
const CB_INTERCEPTION  = CB_PG.GlobalInterception(μ = Normal(0.2, 0.1))
const CB_HOME_ADV      = CB_PG.GlobalHomeAdvantage(γ_global = Normal(0.2, 0.2))
const CB_DYNAMICS      = CB_PG.TimeDecayDynamics(days_half_life = CB_HALF_LIFE,
                                                 σ_att = Gamma(2.0, 0.15),
                                                 σ_def = Gamma(2.0, 0.15))
const CB_WEALTH_PRIOR   = truncated(Normal(0.10, 0.05), lower = 0.0)
const CB_DISTANCE_PRIOR = truncated(Normal(0.04, 0.03), lower = 0.0)

const CB_PARITY_SEEDS = [20260825, 7, 991, 20240101]
const CB_FOLD         = 1

# §9 IS OFF BY DEFAULT, AND NOT BECAUSE IT FAILS.
#
# Sections 1-8 already settle the question deterministically and in seconds:
# bit-identical log-densities at every compared draw, identical θ layout, identical
# feature derivation, compiled-tape gradients agreeing with ForwardDiff, and an
# `extract_parameters` comparison that is exact rather than statistical. Two models
# with the same log-density ARE the same posterior; no amount of sampling can add to
# that, and a Monte-Carlo comparison can only ever say "within noise".
#
# §9 exists to demonstrate it end to end through a real sampler. It costs 96 NUTS
# chains, which belongs on the server, not on a laptop. Turn it on there.
const CB_RUN_POSTERIOR = false
const CB_NUTS_SEED     = 20260825
const CB_NUTS_GROUPS   = 3      # independent chain groups per model; the within-model pairs are the null
const CB_NUTS_CHAINS   = 4      # per group; MCSE from one chain is too noisy to test with
const CB_NUTS_WARMUP   = 500
const CB_NUTS_SAMPLES  = 1000

CB_CONTRACT = sl_contract()


# %% ===========================================================================
# 3. Engine construction — the entire point of the exercise
# ==============================================================================
# Four models. One builder. Zero new `@model` functions.
#
# Note what changes between the four calls and what does not: the STRUCTURAL block
# is written once and reused verbatim, and each arm differs by exactly the
# covariates it adds. That sentence is the design.

"A fresh builder carrying the shared linear predictor, before any covariate."
function cb_base_builder()
    b = CountModelBuilder(:scottish_count)
    add!(b, CB_INTERCEPTION, CB_DYNAMICS, CB_HOME_ADV)   # variadic, dispatched on abstract type
    return b
end

CB_MODEL_00 = build(cb_base_builder())

CB_MODEL_02 = build(add!(cb_base_builder(), WealthCovariate(prior = CB_WEALTH_PRIOR)))

CB_MODEL_03 = build(add!(cb_base_builder(), DistanceCovariate(prior = CB_DISTANCE_PRIOR)))

# The pipeline form, to show `add` composes; identical to `add!(b, w, d)`.
CB_MODEL_04 = cb_base_builder() |>
              add(WealthCovariate(prior = CB_WEALTH_PRIOR)) |>
              add(DistanceCovariate(prior = CB_DISTANCE_PRIOR)) |>
              build

# The hand-written counterparts, built from the SAME component objects — so the
# comparison cannot be passed or failed by a prior differing between two sets of
# defaults. Field-for-field these are `tp00_model` / `tp02_model` / `tp03_model` /
# `tp04_model` at their published defaults.
CB_ARM_00 = tp00_model(half_life_days = CB_HALF_LIFE,
                       interception = CB_INTERCEPTION, home_advantage = CB_HOME_ADV,
                       sigma_att = CB_DYNAMICS.σ_att, sigma_def = CB_DYNAMICS.σ_def)

CB_ARM_02 = DynamicPoissonWealthGoalsTimeDecayModel(
    interception_config = CB_INTERCEPTION, dynamics_config = CB_DYNAMICS,
    homeadvantage_config = CB_HOME_ADV,
    wealth_feature = SLFPLogSumWealthFeature(), w_wealth_prior = CB_WEALTH_PRIOR)

CB_ARM_03 = DynamicPoissonDistanceGoalsTimeDecayModel(
    interception_config = CB_INTERCEPTION, dynamics_config = CB_DYNAMICS,
    homeadvantage_config = CB_HOME_ADV,
    distance_feature = ScottishDistanceFeature(metric = :log_dist_z),
    w_dist_prior = CB_DISTANCE_PRIOR)

CB_ARM_04 = DynamicPoissonWealthDistanceGoalsTimeDecayModel(
    interception_config = CB_INTERCEPTION, dynamics_config = CB_DYNAMICS,
    homeadvantage_config = CB_HOME_ADV,
    wealth_feature = SLFPLogSumWealthFeature(),
    distance_feature = ScottishDistanceFeature(metric = :log_dist_z),
    w_wealth_prior = CB_WEALTH_PRIOR, w_dist_prior = CB_DISTANCE_PRIOR)

CB_ARMS = [
    ("00 baseline",  CB_MODEL_00, CB_ARM_00),
    ("02 wealth",    CB_MODEL_02, CB_ARM_02),
    ("03 distance",  CB_MODEL_03, CB_ARM_03),
    ("04 joint",     CB_MODEL_04, CB_ARM_04),
]

println("\n", "=" ^ 78)
println("ASSEMBLED MODELS")
println("=" ^ 78)
for (label, built, arm) in CB_ARMS
    @printf("  %-12s  %s\n", label, built)
    @printf("  %-12s  hand-written counterpart: %s\n", "", nameof(typeof(arm)))
end
println("\n  provenance of the joint model:")
for line in (cb_base_builder() |> add(WealthCovariate()) |> add(DistanceCovariate())).provenance
    println("      ", line)
end


# %% ===========================================================================
# 4. Validation — the referee, including the cases it must refuse
# ==============================================================================
# A builder is only useful if `build()` is a real gate. These four are structural
# errors that produce a model that RUNS: none is a type error, and every one of
# them silently corrupts a posterior.

@assert sl_gate_table("4a. Validation — the joint model",
    validate(cb_base_builder() |> add(WealthCovariate()) |> add(DistanceCovariate())))

println("-" ^ 78)
println("4b. REFUSALS — each of these must fail to build")
println("-" ^ 78)

# `do`-block form, so `f` comes first.
"Show that a malformed builder is refused, and with what reason."
function cb_expect_refusal(f, label::AbstractString)
    try
        f()
        @printf("  [MISS]  %-34s built anyway — the referee has a hole\n", label)
        return false
    catch err
        msg = first(split(sprint(showerror, err), "\n  "))
        @printf("  [OK]    %-34s %s\n", label, first(split(msg, "\n")))
        return true
    end
end

CB_REFUSALS = [
    cb_expect_refusal("no dynamics component") do
        b = CountModelBuilder(); add!(b, CB_INTERCEPTION, CB_HOME_ADV); build(b)
    end,
    cb_expect_refusal("two dynamics components") do
        b = cb_base_builder(); add!(b, CB_PG.StaticZeroDynamics()); build(b)
    end,
    cb_expect_refusal("duplicate covariate name") do
        build(cb_base_builder() |> add(WealthCovariate()) |> add(WealthCovariate()))
    end,
    cb_expect_refusal("declared-but-unwired observation") do
        build(add!(cb_base_builder(), CB_PG.GlobalDixonColesConfig()))
    end,
    cb_expect_refusal("component from no known family") do
        add!(cb_base_builder(), "half_life = 180")
    end,
]
@assert all(CB_REFUSALS)
println()


# %% ===========================================================================
# 5. Data snapshot and temporal splits
# ==============================================================================
# The shared protocol's folds — this runner does not cut its own.

CB_DS    = BayesianFootball.Data.load_datastore_cached(
               BayesianFootball.Data.ScottishLower(); max_age_hours = 100_000)
CB_FOLDS = sl_build_folds(CB_DS, CB_CONTRACT)

println("=" ^ 78)
println("DATA — ScottishLower [56, 57], dev seasons $(CB_CONTRACT.dev_seasons)")
println("=" ^ 78)
sl_fold_table(CB_DS, CB_FOLDS[1:min(3, length(CB_FOLDS))])
println("  ... $(length(CB_FOLDS)) folds; this runner uses fold $(CB_FOLD) only\n")


# %% ===========================================================================
# 6. Feature construction — derived vs hand-written
# ==============================================================================
# One FeatureSet per arm, built from the HAND-WRITTEN model's `required_features`,
# then handed to both models. The builder's own `required_features` is checked
# against it rather than used, so a derivation bug cannot hide by also changing
# the data.

CB_SPLITTER = sl_splitter(CB_CONTRACT)
CB_FEATURES = Dict{String, Any}()

for (label, built, arm) in CB_ARMS
    fold = CB_FOLDS[CB_FOLD]
    fc = BayesianFootball.Features.create_features(
             [(fold.boundary, fold.meta)], CB_DS, arm, CB_SPLITTER)
    CB_FEATURES[label] = fc[1][1]
end

@assert sl_gate_table("6. Feature derivation",
    [cb_feature_parity(built, arm) for (label, built, arm) in CB_ARMS])

let fs = CB_FEATURES["04 joint"]
    @printf("  fold %d: %d fitted matches, %d teams, %d season(s)\n\n",
            CB_FOLD, length(fs.data[:flat_home_ids]),
            Int(fs.data[:n_teams]), Int(fs.data[:n_seasons]))
end


# %% ===========================================================================
# 7. The density proof
# ==============================================================================
# For each arm, in order of increasing strength:
#
#   (a) the derived chain schema is what the engine actually samples
#   (b) the engine agrees with an INDEPENDENT re-derivation of the maths
#   (c) the [-10, 10] clamp is nowhere near binding, so (d) is not an artefact
#   (d) the composable model and the hand-written arm place parameters identically
#   (e) at the SAME θ, the two log-densities are bit-identical

CB_DENSITY = Dict{String, Bool}()

for (label, built, arm) in CB_ARMS
    fs      = CB_FEATURES[label]
    n_teams = Int(fs.data[:n_teams])
    draw    = cb_prior_draw(built, fs; seed = first(CB_PARITY_SEEDS))

    rows = Any[
        cb_result("derived VarInfo sites == sampled",
                  Set(String.(cb_varinfo_sites(built))) == Set(draw.sites),
                  "$(length(draw.sites)) grouped sites: $(join(draw.sites, ", "))"),
        cb_result("derived parameter count == θ length",
                  cb_parameter_count(built, n_teams) == length(draw.θ),
                  "$(length(draw.θ)) parameters for $(n_teams) teams"),
        cb_reference_parity(built, fs; seeds = CB_PARITY_SEEDS),
        cb_clamp_headroom(built, fs; seeds = CB_PARITY_SEEDS),
        cb_layout_parity(built, arm, fs; seed = first(CB_PARITY_SEEDS)),
        cb_density_parity(built, arm, fs; seeds = CB_PARITY_SEEDS),
    ]
    CB_DENSITY[label] = sl_gate_table("7. Density parity — arm $label", rows)
end
@assert all(values(CB_DENSITY))


# %% ===========================================================================
# 8. AD compliance — docs/turing_ad_performance_guide.md
# ==============================================================================
# The composable engine must not cost anything. Both models are measured on the
# same fold with the same compiled-tape protocol; the builder's number has to sit
# on top of the hand-written one, not merely "near" it.

CB_GRADIENT = DataFrame(arm = String[], builder_ms = Float64[], noguard_ms = Float64[],
                        arm_ms = Float64[], ratio = Float64[])

"The same model with the rate guard removed — arm 00's setting, one `add!` away."
cb_unguarded(m) = _assemble(m.observation, m.interception, m.dynamics,
                            m.home_advantage, m.covariates, NoGuard())

for (label, built, arm) in CB_ARMS
    fs = CB_FEATURES[label]
    rows_b, ms_b = cb_gradient_report(built, fs)
    _,      ms_n = cb_gradient_report(cb_unguarded(built), fs)
    _,      ms_a = cb_gradient_report(arm,   fs)
    @assert sl_gate_table("8. Gradient health — arm $label (composable engine)", rows_b)
    push!(CB_GRADIENT, (label, ms_b, ms_n, ms_a, ms_b / ms_a))
end

println("-" ^ 78)
println("8z. COMPILED GRADIENT LATENCY — composable vs hand-written")
println("-" ^ 78)
println("  Arms 02/03/04 clamp; arm 00 does not. The `no guard` column is the")
println("  composable model with `NoGuard()`, i.e. arm 00's own setting, so the")
println("  00 row is finally comparing the same two functions.")
println()
@printf("  %-12s %12s %12s %12s %10s\n", "arm", "builder ms", "no guard ms", "arm ms", "ratio")
for r in eachrow(CB_GRADIENT)
    @printf("  %-12s %12.3f %12.3f %12.3f %10.3f\n",
            r.arm, r.builder_ms, r.noguard_ms, r.arm_ms, r.ratio)
end
println()


# %% ===========================================================================
# 9. Posterior parity on fold 1
# ==============================================================================
# Same fold, same seed, same NUTS settings, same compiled ReverseDiff backend.
# What has to hold is that the two chains are draws from the SAME posterior — checked against an autocorrelation-aware Monte-Carlo standard
# error, not against `std/sqrt(n)`, which is optimistic by roughly the square root
# of the autocorrelation time and would fail a model compared against itself.
#
# THREE statistics, each calibrated against the same-model seed-to-seed floor:
#
#   identified parameters, max |z|   the level, home advantage, the two rating
#                                    spreads, and every covariate weight
#   R-hat per model                  reported first: a disagreement between two
#                                    unconverged chains says nothing about two
#                                    models
#   converged parameters, mean |z|   an aggregate over every parameter BOTH models
#                                    converged on, the fifty non-centred team
#                                    z-scores included; a maximum over those is
#                                    dominated by whichever mixed worst, a mean is
#                                    not
#   OOS posterior-mean rates         the fold's real t+1 fixtures priced through
#                                    `extract_parameters` — the quantity the score
#                                    grid, the book and the stake actually consume
#
# plus one EXACT check that needs no calibration at all: the same posterior draws
# pushed through both extractors, which must give identical rates.
#
# §7 is the PROOF that the two models are the same: bit-identical log-densities.
# §9 is the DEMONSTRATION that nothing outside the density — chain schema, sampler
# wiring, parameter layout — differs once a real sampler is put through them. It
# cannot be stronger evidence than §7 and is not meant to be.
#
# The threshold is CALIBRATED, not assumed. Each model is also run against itself
# at a second seed, and the builder-versus-arm discrepancy has to sit inside the
# larger of those two floors. See `cb_calibrated_row`.
#
# Bit-identical chains are reported but NOT required, and the reason is worth
# stating plainly: the two engines compute the same gradient through a different
# sequence of floating-point operations. The trajectories therefore part company in
# the last bit of the first leapfrog step, and NUTS amplifies that. A demand for
# identical draws would be a demand for identical arithmetic, which is not what
# "the same model" means.

CB_POSTERIOR = DataFrame(arm = String[], parameters = Int[],
                         max_z_identified = Float64[], mean_z_conv = Float64[],
                         lambda_reldiff = Float64[], noise_floor = Float64[])

if CB_RUN_POSTERIOR
    for (label, built, arm) in CB_ARMS
        fs     = CB_FEATURES[label]
        oos_df = CB_FOLDS[CB_FOLD].oos_df
        @printf("  fitting arm %-12s — %d x %d chains per model, %d warm-up + %d draws each ...\n",
                label, CB_NUTS_GROUPS, CB_NUTS_CHAINS, CB_NUTS_WARMUP, CB_NUTS_SAMPLES)

        # Four chains: each model twice. The two same-model pairs are the negative
        # controls that set the noise floor; the cross pair is the test.
        nuts(m, sd) = cb_nuts(m, fs; seed = sd, chains = CB_NUTS_CHAINS,
                              warmup = CB_NUTS_WARMUP, samples = CB_NUTS_SAMPLES)
        # Several independent chain GROUPS per model. The within-model pairs are the
        # negative controls; one estimate of the noise floor is not enough, because
        # the floor is itself a maximum over correlated parameters and varies from
        # replicate to replicate by tens of percent.
        B = [nuts(built, CB_NUTS_SEED + k) for k in 0:(CB_NUTS_GROUPS - 1)]
        A = [nuts(arm,   CB_NUTS_SEED + k) for k in 0:(CB_NUTS_GROUPS - 1)]

        pairs(G) = [(G[i], G[j]) for i in 1:length(G) for j in (i + 1):length(G)]
        stats_of(x, y) = cb_posterior_stats(cb_posterior_parity(x, y)[2])
        null_b = [stats_of(x, y) for (x, y) in pairs(B)]
        null_a = [stats_of(x, y) for (x, y) in pairs(A)]

        rows, table = cb_posterior_parity(B[1], A[1])
        test = cb_posterior_stats(table)

        # Prediction parity, on the fold's genuine t+1 fixtures.
        extract_row, extract_max = cb_extraction_parity(built, arm, fs, oos_df, B[1])
        λ(m, ch) = cb_lambda_means(m, fs, oos_df, ch)
        λB = [λ(built, ch) for ch in B]
        λA = [λ(arm,   ch) for ch in A]
        λ_test   = cb_lambda_discrepancy(λB[1], λA[1])
        λ_null_b = maximum(cb_lambda_discrepancy(x, y) for (x, y) in pairs(λB))
        λ_null_a = maximum(cb_lambda_discrepancy(x, y) for (x, y) in pairs(λA))

        @assert sl_gate_table("9. Posterior parity — arm $label", vcat(
            cb_convergence_row("composable chains converged", B[1]),
            cb_convergence_row("hand-written chains converged", A[1]),
            rows[1:2], extract_row,
            cb_calibrated_row("identified parameters: max |z|",
                              test.max_z_identified,
                              maximum(x.max_z_identified for x in null_b),
                              maximum(x.max_z_identified for x in null_a);
                              floor_value = 4.0),
            cb_result("comparable parameters",
                      test.n_comparable >= 0.5 * test.n_total,
                      "$(test.n_comparable) of $(test.n_total) reached R-hat <= 1.01 in BOTH models"),
            cb_calibrated_row("converged parameters: mean |z|",
                              test.mean_z_comparable,
                              maximum(x.mean_z_comparable for x in null_b),
                              maximum(x.mean_z_comparable for x in null_a);
                              floor_value = 1.0),
            cb_calibrated_row("OOS posterior-mean rates: mean relative difference",
                              λ_test, λ_null_b, λ_null_a; floor_value = 0.01),
        ))

        push!(CB_POSTERIOR, (label, nrow(table), test.max_z_identified,
                             test.mean_z_comparable, λ_test,
                             max(maximum(x.max_z_identified for x in null_b),
                                 maximum(x.max_z_identified for x in null_a))))

        # The identified parameters, side by side, so the comparison is readable
        # and not only asserted.
        println("     ", rpad("parameter", 16), rpad("builder mean", 16),
                rpad("arm mean", 16), rpad("sd", 12), "|z|")
        for r in eachrow(table[table.identified .& table.converged, :])
            @printf("     %-16s %-16.6f %-16.6f %-12.6f %.2f\n",
                    r.parameter, r.mean_builder, r.mean_arm, r.sd_builder, r.z)
        end
        @printf("     %d OOS fixtures priced; mean relative difference in posterior-mean λ = %.3e\n",
                nrow(oos_df), λ_test)
        @printf("     same draws through both extractors: max relative difference = %.3e\n\n",
                extract_max)
    end
else
    println("  §9 SKIPPED (CB_RUN_POSTERIOR = false) — density parity in §7 stands alone.\n")
end


# %% ===========================================================================
# 10. What the builder can now do that the arms cannot
# ==============================================================================
# The four arms above are four of the models this builder spans. The point of the
# design is the ones nobody has written an engine for.

println("=" ^ 78)
println("10. REACHABLE WITHOUT WRITING AN ENGINE")
println("=" ^ 78)

CB_EXTRA = [
    ("arm 00 exactly, guard removed",
     build(add!(cb_base_builder(), NoGuard()))),
    ("wealth as a LEVEL covariate",
     build(add!(cb_base_builder(), WealthCovariate(role = LevelRole())))),
    ("distance, no team dynamics (H2 ablation)",
     build(add!(add!(CountModelBuilder(), CB_INTERCEPTION, CB_HOME_ADV,
                     CB_PG.StaticZeroDynamics(days_half_life = CB_HALF_LIFE)),
                DistanceCovariate()))),
    ("negative binomial + both covariates",
     build(add!(cb_base_builder(), WealthCovariate(), DistanceCovariate(),
                CB_PG.GlobalDispersion()))),
    ("hierarchical monthly level + team home advantage + wealth",
     build(add!(add!(CountModelBuilder(),
                     CB_PG.HierarchicalMonthlyInterception(),
                     CB_PG.HierarchicalTeamHomeAdvantage(),
                     CB_DYNAMICS),
                WealthCovariate()))),
]

let fs = CB_FEATURES["04 joint"], n_teams = Int(fs.data[:n_teams])
    @printf("  %-44s %-14s %s\n", "configuration", "family", "parameters")
    for (label, m) in CB_EXTRA
        fam = m isa PoissonCountModel ? "Poisson" : "NegBin"
        @printf("  %-44s %-14s %d\n", label, fam, cb_parameter_count(m, n_teams))
    end
end

println("\n  Each row above is a `build()` call, not a file. The four arms in §3")
println("  cost 5 hand-written `@model` functions between them; these cost none.\n")


# %% ===========================================================================
# 11. Summary
# ==============================================================================
println("=" ^ 78)
println("SUMMARY — composable builder vs hand-written arms, fold $(CB_FOLD)")
println("=" ^ 78)
@printf("  %-12s %-10s %-12s %-14s %-14s %s\n",
        "arm", "density", "grad ratio", "max |z| ident.", "|z| floor", "OOS λ rel. diff")
for (label, built, arm) in CB_ARMS
    g = CB_GRADIENT[CB_GRADIENT.arm .== label, :]
    p = CB_POSTERIOR[CB_POSTERIOR.arm .== label, :]
    @printf("  %-12s %-10s %-12s %-14s %-14s %s\n",
            label,
            CB_DENSITY[label] ? "exact" : "FAIL",
            isempty(g) ? "—" : @sprintf("%.3f", g.ratio[1]),
            isempty(p) ? "—" : @sprintf("%.2f", p.max_z_identified[1]),
            isempty(p) ? "—" : @sprintf("%.2f", p.noise_floor[1]),
            isempty(p) ? "—" : @sprintf("%.2e", p.lambda_reldiff[1]))
end
println("=" ^ 78)
println()
println("  Nothing here selects a model. Arms 02-04 own that question; this run")
println("  only establishes that asking it through the builder asks the same question.")
println()
