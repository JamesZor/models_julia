# ==============================================================================
# r50 — What is a player rating made of, and what should RAPM be normalised by?
# ==============================================================================
#
# WHAT THIS IS
#   The descriptive half of the player-normalisation stream. It establishes the facts
#   that r51's bench then acts on.
#
#   FOUR QUESTIONS.
#     1. Is the SofaScore rating position-normalised? (and league-normalised?)
#     2. How much of it do age and market value actually explain?
#     3. Does OUR RAPM carry the same structure, or a different one?
#     4. Where does RAPM agree with the SofaScore yardstick, and where does it not?
#
# WHY ENGLAND. Only tiers 1/2/3/84 carry a reference rating, market values and dates of
# birth together. Scottish Upper has ratings but zero wealth data; Scottish Lower — the
# deployment target — has wealth but no ratings at all. See l50's header for the table.
#
# WHAT THIS IS NOT
#   Not a feature. Nothing here changes src/, and no covariate is built. r51 does the
#   comparison; this runner only measures.
#
# USAGE
#   source .env
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r50_rating_structure.jl")
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
include(joinpath(@__DIR__, "l50_player_norm.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R50_TARGET = :y_xg
const R50_LAMBDAS = (200.0, 5000.0)      # shipped, and near the reliability plateau
const R50_HALF_LIFE = 730.0
const R50_MIN_SOFA = 10                  # ratings a player needs before he is a yardstick
const R50_KAPPA = 20.0

eda_banner("r50 · rating structure and what RAPM should be normalised by")

# %%
# ==============================================================================
# 3. The reference laboratory
# ==============================================================================

eda_section("1/6", "The English reference store")

ds = l50_store()
@printf("  matches %d | lineup rows %d | incidents %d | shot events %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.incidents), nrow(ds.bbc_events))
@printf("  rating %.1f%% | market value %.1f%% | date of birth %.1f%%\n",
        100 * count(!ismissing, ds.lineups.rating) / nrow(ds.lineups),
        100 * count(!ismissing, ds.lineups.proposed_market_value) / nrow(ds.lineups),
        100 * count(!ismissing, ds.lineups.date_of_birth_timestamp) / nrow(ds.lineups))

prep = Features.pm_prepared(ds)
segments = prep.segments
@printf("  stints %d, live-text covered %d (%.1f%%) — Scottish Lower manages 56.6%%\n",
        nrow(segments), count(segments.covered),
        100 * count(segments.covered) / nrow(segments))

# %%
# ==============================================================================
# 4. Is the SofaScore rating position-normalised?
# ==============================================================================

eda_section("2/6", "The yardstick's own construction")

starters = NamedTuple[]
for r in eachrow(ds.lineups)
    coalesce(r.is_substitute, false) && continue
    ismissing(r.rating) && continue
    push!(starters, (
        tier = Int(r.tournament_id),
        position = Features.pm_clean_position(r.position),
        rating = Float64(r.rating),
        value = ismissing(r.proposed_market_value) ? NaN : Float64(r.proposed_market_value),
        dob = ismissing(r.date_of_birth_timestamp) ? NaN : Float64(r.date_of_birth_timestamp),
        match_id = Int(r.match_id),
    ))
end
starter_frame = DataFrame(starters)
kickoff_of = Dict(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
starter_frame.age = [
    (isnan(r.dob) || !haskey(kickoff_of, r.match_id)) ? NaN :
    (datetime2unix(DateTime(kickoff_of[r.match_id])) - r.dob) / (365.25 * 86_400)
    for r in eachrow(starter_frame)]
starter_frame = starter_frame[
    (starter_frame.position .!= "?") .&
    ((isnan.(starter_frame.age)) .| ((starter_frame.age .> 14) .& (starter_frame.age .< 46))), :]

@printf("  %d rated starter rows\n\n", nrow(starter_frame))
@printf("  %-6s | %8s | %10s | %8s | %8s | %8s | %8s\n",
        "pos", "n", "mean", "sd", "p10", "p50", "p90")
println("  " * repeat('-', 72))
for position in L50_POSITIONS
    s = starter_frame[starter_frame.position .== position, :]
    nrow(s) == 0 && continue
    @printf("  %-6s | %8d | %10.4f | %8.4f | %8.2f | %8.2f | %8.2f\n",
            position, nrow(s), mean(s.rating), std(s.rating),
            quantile(s.rating, 0.10), median(s.rating), quantile(s.rating, 0.90))
end
let means = [mean(starter_frame[starter_frame.position .== p, :rating])
             for p in L50_POSITIONS if any(starter_frame.position .== p)]
    @printf("\n  spread of position means: %.4f  against a pooled sd of %.4f\n",
            maximum(means) - minimum(means), std(starter_frame.rating))
    println("  A spread far below the sd is the signature of a POSITION-NORMALISED score:")
    println("  each position is graded on its own curve, not against the others.")
end

println("\n  By tier — is the scale league-invariant too?")
@printf("  %-22s | %-4s | %8s | %9s | %8s\n", "tier", "pos", "n", "mean", "sd")
println("  " * repeat('-', 62))
for tier in sort(unique(starter_frame.tier)), position in L50_POSITIONS
    s = starter_frame[(starter_frame.tier .== tier) .& (starter_frame.position .== position), :]
    nrow(s) < 50 && continue
    @printf("  %-22s | %-4s | %8d | %9.4f | %8.4f\n",
            get(L50_TIER_NAMES, tier, "?"), position, nrow(s), mean(s.rating), std(s.rating))
end
println("\n  [READ] If a Premier League mean equals a League Two mean, the rating cannot be")
println("  an absolute quality measure — it is a within-league, within-position relative")
println("  score. Only its DISPERSION is allowed to differ by position.")

# %%
# ==============================================================================
# 5. How much do age and wealth explain?
# ==============================================================================

eda_section("3/6", "Age and wealth against the rating")

@printf("  %-6s | %8s | %10s | %11s | %11s | %12s\n",
        "pos", "n", "r(age)", "rho(age)", "r(logValue)", "rho(logValue)")
println("  " * repeat('-', 76))
for position in L50_POSITIONS
    s = starter_frame[starter_frame.position .== position, :]
    nrow(s) < 100 && continue
    ok_age = .!isnan.(s.age)
    ok_val = .!isnan.(s.value) .& (s.value .> 0)
    @printf("  %-6s | %8d | %10s | %11s | %11s | %12s\n", position, nrow(s),
            eda_fmt(eda_pearson(s.age[ok_age], s.rating[ok_age]), "%+.4f"),
            eda_fmt(eda_spearman(s.age[ok_age], s.rating[ok_age]), "%+.4f"),
            eda_fmt(eda_pearson(log.(s.value[ok_val]), s.rating[ok_val]), "%+.4f"),
            eda_fmt(eda_spearman(log.(s.value[ok_val]), s.rating[ok_val]), "%+.4f"))
end

println("\n  Mean rating by age band (all positions):")
for lo in 16:2:38
    s = starter_frame[(.!isnan.(starter_frame.age)) .&
                      (starter_frame.age .>= lo) .& (starter_frame.age .< lo + 2), :]
    nrow(s) < 300 && continue
    bar = repeat('▇', max(0, round(Int, (mean(s.rating) - 6.6) * 120)))
    @printf("    %2d–%2d  n=%7d  %.4f  %s\n", lo, lo + 2, nrow(s), mean(s.rating), bar)
end

let ok = .!isnan.(starter_frame.age) .& .!isnan.(starter_frame.value) .& (starter_frame.value .> 0)
    s = starter_frame[ok, :]
    X = hcat(s.age, s.age .^ 2, log.(s.value))
    fit = eda_ols(X, s.rating)
    @printf("\n  OLS rating ~ age + age² + log(value):  R² = %.5f  (n = %d)\n", fit.r2, fit.n)
    println("  [READ] Whatever that R² is, it bounds how much a demographic normalisation of")
    println("  the RATING could possibly change. A small value does not make age and wealth")
    println("  useless — it makes them ORTHOGONAL, and therefore additive rather than")
    println("  redundant information.")
end

# %%
# ==============================================================================
# 6. RAPM's own structure
# ==============================================================================

eda_section("4/6", "Does RAPM carry the same structure?")

comp_sets = Features.competition_sets(ds)
T_rating = maximum(segments.match_date)
exposure = Features.player_exposure(segments)

fits = Dict{Float64,DataFrame}()
for lambda in R50_LAMBDAS
    t0 = time()
    fit = Features.fit_ratings(segments; target = R50_TARGET, λ = lambda, w_sim = 0.0,
                               half_life = R50_HALF_LIFE, T_rating = T_rating,
                               comp_sets = comp_sets)
    @printf("  fitted lambda = %6.0f over %d covered stints in %.1fs — %d players\n",
            lambda, count(segments.covered), time() - t0, fit === nothing ? 0 : nrow(fit))
    fit === nothing || (fits[lambda] = fit)
end

frames = Dict{Float64,DataFrame}()
for (lambda, fit) in fits
    frames[lambda] = l50_player_frame(ds, fit, exposure; reference_date = T_rating)
end

let frame = frames[maximum(keys(frames))]
    rated = frame[frame.n_sofa .>= R50_MIN_SOFA, :]
    @printf("\n  %d players carry at least %d SofaScore ratings.\n\n", nrow(rated), R50_MIN_SOFA)
    eda_print_describe([
        eda_describe("RAPM", rated.rapm),
        eda_describe("SofaScore mean", rated.sofa_mean),
        eda_describe("age (years)", rated.age),
        eda_describe("log market value", rated.log_value),
        eda_describe("stints", rated.n_segments),
    ]; title = "  Player-level frame:")

    println("\n  RAPM dispersion by position — is RAPM itself position-normalised?")
    @printf("  %-6s | %8s | %12s | %12s | %12s\n", "pos", "n", "mean RAPM", "sd RAPM", "sd SofaScore")
    println("  " * repeat('-', 60))
    for position in L50_POSITIONS
        s = rated[rated.position .== position, :]
        nrow(s) < 15 && continue
        @printf("  %-6s | %8d | %+12.6f | %12.6f | %12.4f\n",
                position, nrow(s), mean(s.rapm), std(s.rapm), std(s.sofa_mean))
    end
    println("\n  [READ] RAPM is zero-centred by construction, so its MEANS will look aligned.")
    println("  The question is the SPREAD: if one position's ratings vary far more than")
    println("  another's, summing eleven of them weights that position more heavily than")
    println("  intended — silently, and with no way for the engine to correct it.")

    println("\n  RAPM against the same demographics:")
    @printf("  %-6s | %8s | %11s | %13s\n", "pos", "n", "r(age)", "r(logValue)")
    println("  " * repeat('-', 46))
    for position in L50_POSITIONS
        s = rated[rated.position .== position, :]
        nrow(s) < 15 && continue
        ok_age = .!isnan.(s.age); ok_val = .!isnan.(s.log_value)
        @printf("  %-6s | %8d | %11s | %13s\n", position, nrow(s),
                eda_fmt(eda_pearson(s.age[ok_age], s.rapm[ok_age]), "%+.4f"),
                eda_fmt(eda_pearson(s.log_value[ok_val], s.rapm[ok_val]), "%+.4f"))
    end
end

# %%
# ==============================================================================
# 7. Where does RAPM agree with the yardstick?
# ==============================================================================

eda_section("5/6", "RAPM against the SofaScore rating, by position")

println("  This is the measurement the whole stream turns on. A position where RAPM and")
println("  the reference disagree is a position whose contribution to the starting-XI sum")
println("  is noise.\n")

for lambda in sort(collect(keys(frames)))
    frame = frames[lambda]
    rated = frame[frame.n_sofa .>= R50_MIN_SOFA, :]
    println("  lambda = $(Int(lambda)):")
    @printf("  %-6s | %7s | %12s | %12s | %14s\n",
            "pos", "n", "r(RAPM,SS)", "rho", "r, >=40 stints")
    println("  " * repeat('-', 62))
    for position in (L50_POSITIONS..., "ALL")
        s = position == "ALL" ? rated : rated[rated.position .== position, :]
        nrow(s) < 15 && continue
        hi = s[s.n_segments .>= 40, :]
        @printf("  %-6s | %7d | %12s | %12s | %14s\n", position, nrow(s),
                eda_fmt(eda_pearson(s.rapm, s.sofa_mean), "%+.4f"),
                eda_fmt(eda_spearman(s.rapm, s.sofa_mean), "%+.4f"),
                nrow(hi) >= 15 ? eda_fmt(eda_pearson(hi.rapm, hi.sofa_mean), "%+.4f") : "—")
    end
    println()
end

# %%
# ==============================================================================
# 8. What the facts imply
# ==============================================================================

eda_section("6/6", "Implications for r51")

let frame = frames[maximum(keys(frames))]
    rated = frame[frame.n_sofa .>= R50_MIN_SOFA, :]
    per_pos = Dict(p => (n = count(==(p), rated.position),
                         r = eda_pearson(rated[rated.position .== p, :rapm],
                                         rated[rated.position .== p, :sofa_mean]),
                         sd = std(rated[rated.position .== p, :rapm]))
                   for p in L50_POSITIONS if count(==(p), rated.position) >= 15)
    for (p, v) in sort(collect(per_pos), by = x -> -(isnan(x[2].r) ? -Inf : x[2].r))
        verdict = isnan(v.r) ? "—" :
                  v.r >= 0.35 ? "carries real signal" :
                  v.r >= 0.15 ? "weak but positive" : "NO SIGNAL — contributes noise"
        @printf("    %s (n=%4d, sd=%.5f) : r = %s  %s\n",
                p, v.n, v.sd, eda_fmt(v.r, "%+.4f"), verdict)
    end
    println()
    println("  Candidate normalisations for r51 follow directly:")
    println("    · drop or downweight any position with no signal")
    println("    · standardise within position, so no position dominates the XI sum by spread")
    println("    · use age and value as a PRIOR for sparse players rather than as a residual,")
    println("      if section 3 showed them near-orthogonal to the rating")
end
eda_rule(100, '=')
