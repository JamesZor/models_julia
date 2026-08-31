# ==============================================================================
# r93 — Feature synergy, collinearity, and what survives squad wealth
# ==============================================================================
#
# WHAT THIS IS
#   A joint study of every covariate the unified builder can currently attach on
#   Scottish tiers 56/57:
#
#     wealth          raw starting-XI log market value differential
#     prod_wealth     the same, age-weighted by a Richards production curve
#     distance        away-travel burden between the two grounds
#     pxg_supremacy   point-in-time proxy-xG form, supremacy role
#     pxg_level       the same feature, level role
#     pxg_rapm        starting-XI stint-RAPM differential
#
#   THREE QUESTIONS.
#     1. How much do these six overlap? (correlation matrix, VIF)
#     2. What does each carry on its own? (univariate signal)
#     3. What survives once squad wealth is already in the model? — the only question
#        that decides whether a new covariate earns its parameter, because wealth is
#        the incumbent and a covariate that merely re-derives it is free to drop.
#
# WHAT THIS IS NOT
#   Not a model comparison. These are linear associations with the realised
#   scoreline, not out-of-sample log loss under the count model; r40 is that.
#   A covariate can look weak here and still help, because the engine sees it
#   alongside team dynamics that this flat regression does not have.
#
# ⚠ EVERYTHING IS MEASURED HELD-OUT.
#   Two of these features FIT something (the RAPM ridge, the shot-xG cell table).
#   Scored on the matches they were fitted on, a goal-differential RAPM target
#   scores r = 0.42; scored held-out it scores r = 0.004. All of it was leakage.
#   Every number below comes from the last 20% of the fixture list, built from
#   features that only ever saw the first 80%.
#
# USAGE
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r93_feature_synergy_and_correlations.jl")
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics

include(joinpath(@__DIR__, "l93_eda_toolkit.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R93_SEGMENT = Data.ScottishLower()
const R93_HISTORY_FRAC = 0.80

# The six columns, each named by the design key its extractor emits.
const R93_SPEC = [
    (name = "wealth",        key = :flat_delta_wealth_logsum,     config = LogSumWealthFeature()),
    (name = "prod_wealth",   key = :flat_delta_production_wealth, config = ProductionWealthFeature()),
    (name = "distance",      key = :flat_distance,                config = Features.DistanceFeature()),
    (name = "pxg_supremacy", key = :flat_pxg_supremacy,           config = PxGFeature()),
    (name = "pxg_level",     key = :flat_pxg_level,               config = PxGFeature()),
    (name = "pxg_rapm",      key = :flat_pxg_rapm,                config = PxGRapmFeature()),
]

eda_banner("r93 · feature synergy, collinearity and incremental value")
println("  held-out fraction: ", round(100 * (1 - R93_HISTORY_FRAC)), "% of the fixture list")

# %%
# ==============================================================================
# 3. Data snapshot and joint feature build
# ==============================================================================

eda_section("1/6", "Building all six design columns on one common split")

ds = Data.load_datastore_cached(R93_SEGMENT; max_age_hours = 100_000)
@printf("  matches %d | lineups %d | bbc_events %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.bbc_events))

# One build with every feature attached, so all six columns share one fold, one
# ordering and one held-out mask. Building them separately would silently compare
# columns aligned to different match orders.
configs = Features.AbstractFeatureConfig[]
seen = Set{Symbol}()
for spec in R93_SPEC
    key = Symbol(nameof(typeof(spec.config)))
    key in seen && continue          # PxGFeature serves both the supremacy and level columns
    push!(seen, key)
    push!(configs, spec.config)
end

held = eda_holdout(ds, configs; history_frac = R93_HISTORY_FRAC)
@printf("  history %d matches | held-out %d matches\n", held.n_history, held.n_target)

frame = held.frame[held.mask, :]
columns = Vector{Float64}[]
names_used = String[]
for spec in R93_SPEC
    if !haskey(held.fs.data, spec.key)
        @warn "r93: $(spec.name) produced no column ($(spec.key)); skipping"
        continue
    end
    push!(columns, Float64.(held.fs.data[spec.key])[held.mask])
    push!(names_used, spec.name)
end
@printf("  columns resolved: %s\n", join(names_used, ", "))

usable = frame.played
frame = frame[usable, :]
columns = [c[usable] for c in columns]
@printf("  scored on %d played held-out matches\n", nrow(frame))

eda_print_describe([eda_describe(names_used[i], columns[i]) for i in eachindex(columns)];
                   title = "\n  Design columns (held-out block):")

# %%
# ==============================================================================
# 4. Correlation and multicollinearity
# ==============================================================================

eda_section("2/6", "Cross-feature correlation")

eda_print_corr(names_used, eda_corr_matrix(names_used, columns; method = :pearson);
               title = "Pearson:", flag = 0.60)
println()
eda_print_corr(names_used, eda_corr_matrix(names_used, columns; method = :spearman);
               title = "Spearman:", flag = 0.60)

eda_section("3/6", "Variance inflation")

vif = eda_vif(names_used, columns)
@printf("  %-16s | %10s | %10s | %s\n", "feature", "R² on rest", "VIF", "")
println("  " * repeat('-', 62))
for r in eachrow(sort(vif, :vif, rev = true))
    flag = r.vif >= 10 ? "  [SEVERE]" : r.vif >= 5 ? "  [NOTE]" : ""
    @printf("  %-16s | %10.4f | %10.3f |%s\n", r.feature, r.r2, r.vif, flag)
end
println("\n  VIF above 5 is worth noting; above 10 the two covariates are measuring the same")
println("  thing and the engine cannot separate their weights. A supremacy/level PAIR from")
println("  one feature is expected to be near-orthogonal by construction — if it is not,")
println("  the two roles are not doing different jobs.")

# %%
# ==============================================================================
# 5. Univariate signal
# ==============================================================================

eda_section("4/6", "What each covariate carries on its own")

eda_print_signal([eda_signal(names_used[i], columns[i], frame) for i in eachindex(columns)];
                 title = "Held-out association with the realised scoreline:")

println("\n  Univariate OLS against goal supremacy:")
@printf("  %-16s | %10s | %8s | %8s | %9s\n", "feature", "beta", "se", "t", "R²")
println("  " * repeat('-', 62))
for i in eachindex(columns)
    fit = eda_ols(reshape(columns[i], length(columns[i]), 1), frame.supremacy)
    @printf("  %-16s | %+10.4f | %8.4f | %+8.3f | %9.4f\n",
            names_used[i], fit.beta[2], fit.se[2], fit.t[2], fit.r2)
end

# %%
# ==============================================================================
# 6. Residual explanation beyond squad wealth
# ==============================================================================

eda_section("5/6", "What survives squad wealth")

wealth_idx = findfirst(==("wealth"), names_used)
if wealth_idx === nothing
    println("  [SKIP] the wealth column is unavailable on this store.")
else
    wealth = columns[wealth_idx]
    base = eda_ols(reshape(wealth, length(wealth), 1), frame.supremacy)
    @printf("  Baseline — goal supremacy ~ wealth: R² = %.4f (beta %+.4f, t %+.2f, n %d)\n\n",
            base.r2, base.beta[2], base.t[2], base.n)

    println("  Each candidate is correlated with the WEALTH RESIDUAL, then added to the")
    println("  wealth-only model. `dR²` is the incremental variance it explains beyond")
    println("  wealth — the quantity that decides whether it is worth a parameter.\n")
    @printf("  %-16s | %12s | %10s | %10s | %9s | %8s\n",
            "feature", "r(residual)", "joint R²", "dR²", "beta", "t")
    println("  " * repeat('-', 78))
    incremental = NamedTuple[]
    for i in eachindex(columns)
        i == wealth_idx && continue
        joint = eda_ols(hcat(wealth, columns[i]), frame.supremacy)
        d = joint.r2 - base.r2
        push!(incremental, (feature = names_used[i], dr2 = d))
        @printf("  %-16s | %+12.4f | %10.4f | %+10.4f | %+9.4f | %+8.2f\n",
                names_used[i], eda_pearson(columns[i], base.resid),
                joint.r2, d, joint.beta[3], joint.t[3])
    end

    println("\n  Same test against the PRODUCTION wealth baseline (the stronger incumbent):")
    prod_idx = findfirst(==("prod_wealth"), names_used)
    if prod_idx !== nothing
        pbase = eda_ols(reshape(columns[prod_idx], length(columns[prod_idx]), 1), frame.supremacy)
        @printf("  baseline R² = %.4f\n", pbase.r2)
        @printf("  %-16s | %12s | %10s | %10s | %8s\n",
                "feature", "r(residual)", "joint R²", "dR²", "t")
        println("  " * repeat('-', 68))
        for i in eachindex(columns)
            i == prod_idx && continue
            joint = eda_ols(hcat(columns[prod_idx], columns[i]), frame.supremacy)
            @printf("  %-16s | %+12.4f | %10.4f | %+10.4f | %+8.2f\n",
                    names_used[i], eda_pearson(columns[i], pbase.resid),
                    joint.r2, joint.r2 - pbase.r2, joint.t[3])
        end
    end
end

# %%
# ==============================================================================
# 7. The joint model
# ==============================================================================

eda_section("6/6", "All six together")

design = reduce(hcat, columns)
joint = eda_ols(design, frame.supremacy)
@printf("  goal supremacy ~ all six: R² = %.4f, adjusted R² = %.4f, n = %d\n\n",
        joint.r2, joint.adj_r2, joint.n)
@printf("  %-16s | %10s | %8s | %8s | %s\n", "feature", "beta", "se", "t", "")
println("  " * repeat('-', 62))
@printf("  %-16s | %+10.4f | %8.4f | %+8.3f |\n",
        "(intercept)", joint.beta[1], joint.se[1], joint.t[1])
for i in eachindex(names_used)
    mark = abs(joint.t[i + 1]) >= 2 ? "  significant" : ""
    @printf("  %-16s | %+10.4f | %8.4f | %+8.3f |%s\n",
            names_used[i], joint.beta[i + 1], joint.se[i + 1], joint.t[i + 1], mark)
end

condition = cond(hcat(ones(size(design, 1)), design))
@printf("\n  design condition number: %.1f %s\n", condition,
        condition > 30 ? "[NOTE] above 30 indicates collinearity is affecting the solve" : "")

println("\n  Total-goals model, for the level role:")
joint_total = eda_ols(design, frame.total)
@printf("  total goals ~ all six: R² = %.4f\n", joint_total.r2)
for i in eachindex(names_used)
    abs(joint_total.t[i + 1]) >= 2 || continue
    @printf("    %-16s beta %+8.4f (t %+.2f)\n",
            names_used[i], joint_total.beta[i + 1], joint_total.t[i + 1])
end

println()
eda_rule(100, '-')
println("  Read this alongside r40. A flat linear regression on the scoreline is a WEAK")
println("  proxy for what the count model does: the engine already carries team strength in")
println("  dyn.α/dyn.β, so a covariate's job is to explain what those cannot. A covariate")
println("  with a small dR² here can still earn its parameter there — and one with a large")
println("  dR² here may simply be re-deriving team strength the engine already has.")
eda_rule(100, '=')
