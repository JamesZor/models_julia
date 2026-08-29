# current_development/scottish_lower/r13_verify_production_wealth_feature.jl
#
# This is a deterministic feature diagnostic, not a model-training or betting study.
#
# Question: does the production wealth transform preserve the starting-XI wealth
# contract while strengthening its simple association with Scottish Lower scores?
# Control: raw starting-XI log-sum wealth on the identical fixtures and valuations.
# Decision evidence: curve shape, per-match deltas, correlation, rank correlation,
# and the deviance of a two-sided Poisson GLM. No posterior claim is made here.
#
# Data contract: finished SofaScore matches in tournaments 56 and 57. The lineup
# row itself is treated as the observation stamp when no separate valuation stamp
# exists, matching the production feature's PIT guard.
#
# Usage:
#   julia --project -t 8 current_development/scottish_lower/r13_verify_production_wealth_feature.jl

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using GLM
using LibPQ
using Printf
using Statistics
import StatsBase

const R13_FEATURES = BayesianFootball.Features
const R13_DATA = BayesianFootball.Data
const R13_TOURNAMENT_IDS = (56, 57)
const R13_INSPECTION_ROWS = 15

# %%
# ==============================================================================
# 2. Candidate curve configuration
# ==============================================================================

const R13_CURVES = (
    ("Richards", RichardsSigmoid()),
    ("Gamma", ShiftedGamma()),
    ("Gaussian", GaussianPrime()),
)
const R13_PRODUCTION_CONFIG = ProductionWealthFeature(
    curve=RichardsSigmoid(),
    fallback_default=100_000.0,
    fallback_age=26.5,
    log_scale=1.0,
)
const R13_RAW_CONFIG = LogSumWealthFeature(
    fallback_default=100_000.0,
    log_scale=1.0,
)

function r13_bar(value::Real; width::Int=24)
    n = round(Int, clamp(Float64(value), 0.0, 1.0) * width)
    return rpad(repeat("#", n), width)
end

function r13_print_curves()
    println("\n", "="^104)
    println("1. CANDIDATE AGE-PRODUCTION CURVES")
    println("="^104)
    @printf("%5s | %10s | %10s | %10s\n", "Age", "Richards", "Gamma", "Gaussian")
    println("-"^104)
    for age in 16.0:2.0:40.0
        values = map(curve -> age_weight(curve[2], age), R13_CURVES)
        @printf("%5.1f | %10.6f | %10.6f | %10.6f\n",
                age, values[1], values[2], values[3])
        for (label, value) in zip(first.(R13_CURVES), values)
            @printf("      | %-8s [%s]\n", label, r13_bar(value))
        end
    end
end

# %%
# ==============================================================================
# 3. Real Scottish Lower data snapshot
# ==============================================================================

const R13_SQL = """
    SELECT
        m.match_id,
        m.tournament_id,
        m.start_timestamp,
        m.home_team,
        m.away_team,
        m.home_score,
        m.away_score,
        CASE WHEN l.is_home_team THEN 'home' ELSE 'away' END AS team_side,
        l.player_id,
        l.substitute AS is_substitute,
        l.proposed_market_value,
        (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint AS date_of_birth_timestamp
    FROM sofascore.matches m
    JOIN sofascore.match_player_lineups l ON l.match_id = m.match_id
    WHERE m.tournament_id IN (56, 57)
      AND m.status_type = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY m.start_timestamp, m.match_id, l.is_home_team DESC, l.player_id
"""

function r13_load_snapshot()
    haskey(ENV, "BF_DB_URL") || error(
        "BF_DB_URL is not set. Add it to .env or export it before running r13.")
    connection = LibPQ.Connection(ENV["BF_DB_URL"])
    try
        rows = DataFrame(LibPQ.execute(connection, R13_SQL))
        nrow(rows) > 0 || error("the Scottish Lower lineup query returned no rows")
        matches = unique(select(
            rows, :match_id, :tournament_id, :start_timestamp,
            :home_team, :away_team, :home_score, :away_score))
        lineups = select(
            rows, :match_id, :team_side, :player_id, :is_substitute,
            :proposed_market_value, :date_of_birth_timestamp)
        empty = DataFrame()
        datastore = R13_DATA.DataStore(
            R13_DATA.ScottishLower(), matches,
            empty, empty, lineups, empty, empty, empty, empty)
        return (; datastore, matches, lineups)
    finally
        close(connection)
    end
end

# %%
# ==============================================================================
# 4. Raw and production feature construction
# ==============================================================================

function r13_build_comparison(snapshot)
    ordered_matches = sort(snapshot.matches, :start_timestamp)
    ordered_ids = Int.(ordered_matches.match_id)
    team_map = Dict{String,Int}()

    raw_data = Dict{Symbol,Any}()
    production_data = Dict{Symbol,Any}()
    R13_FEATURES.add_feature!(
        raw_data, R13_RAW_CONFIG, ordered_ids, team_map, snapshot.datastore)
    R13_FEATURES.add_feature!(
        production_data, R13_PRODUCTION_CONFIG,
        ordered_ids, team_map, snapshot.datastore)

    comparison = DataFrame(
        match_id=ordered_ids,
        start_timestamp=ordered_matches.start_timestamp,
        home_team=String.(ordered_matches.home_team),
        away_team=String.(ordered_matches.away_team),
        home_score=Int.(ordered_matches.home_score),
        away_score=Int.(ordered_matches.away_score),
        goal_difference=Int.(ordered_matches.home_score .- ordered_matches.away_score),
        delta_raw=raw_data[:flat_delta_wealth_logsum],
        delta_production=production_data[:flat_delta_production_wealth],
        raw_fallback=raw_data[:flat_wealth_fallback],
        production_fallback=production_data[:flat_production_wealth_fallback],
    )
    comparison.delta_change = comparison.delta_production .- comparison.delta_raw

    # Diagnostics compare exactly the same complete-lineup fixtures. Neutral
    # fallbacks remain visible in the inventory but do not masquerade as signal.
    usable = subset(
        comparison,
        [:raw_fallback, :production_fallback] =>
            ByRow((raw, production) -> raw == 0 && production == 0),
    )
    nrow(usable) > 2 || error("fewer than three jointly usable matches")
    return (; comparison, usable)
end

function r13_print_inventory(snapshot, comparison)
    starters = count(x -> !coalesce(x, false), snapshot.lineups.is_substitute)
    missing_dob = count(ismissing, snapshot.lineups.date_of_birth_timestamp)
    println("\n", "="^104)
    println("2. DATA AND FEATURE INVENTORY")
    println("="^104)
    println("Matches queried:                    ", nrow(snapshot.matches))
    println("Lineup rows queried:                ", nrow(snapshot.lineups))
    println("Rows treated as starters:           ", starters)
    println("Rows using fallback age (missing):  ", missing_dob)
    println("Raw-wealth neutral fixtures:        ", sum(comparison.raw_fallback))
    println("Production-wealth neutral fixtures: ", sum(comparison.production_fallback))
end

# %%
# ==============================================================================
# 5. Side-by-side real-match inspection
# ==============================================================================

function r13_print_inspection(usable::DataFrame)
    n = min(R13_INSPECTION_ROWS, nrow(usable))
    rows = usable[(nrow(usable) - n + 1):nrow(usable), :]
    println("\n", "="^132)
    println("3. LATEST COMPLETE-LINEUP MATCHES: RAW VS AGE-ADJUSTED PRODUCTION WEALTH")
    println("="^132)
    @printf("%-10s %-10s %-22s %-22s %5s %11s %11s %11s\n",
            "Date", "Match ID", "Home", "Away", "GD",
            "ΔW raw", "ΔW prod", "prod-raw")
    println("-"^132)
    for row in eachrow(rows)
        @printf("%-10s %-10d %-22s %-22s %+5d %+11.5f %+11.5f %+11.5f\n",
                string(Date(row.start_timestamp)), row.match_id,
                first(row.home_team, min(length(row.home_team), 22)),
                first(row.away_team, min(length(row.away_team), 22)),
                row.goal_difference, row.delta_raw,
                row.delta_production, row.delta_change)
    end
end

# %%
# ==============================================================================
# 6. Signal diagnostics
# ==============================================================================

function r13_poisson_deviance(delta::Vector{Float64}, usable::DataFrame)
    n = nrow(usable)
    glm_data = DataFrame(
        goals=Float64.(vcat(usable.home_score, usable.away_score)),
        home_indicator=vcat(ones(Float64, n), zeros(Float64, n)),
        wealth_signal=vcat(delta, .-delta),
    )
    fit = glm(
        @formula(goals ~ home_indicator + wealth_signal),
        glm_data, Poisson(), LogLink())
    return deviance(fit)
end

function r13_print_diagnostics(usable::DataFrame)
    goal_difference = Float64.(usable.goal_difference)
    raw = Vector{Float64}(usable.delta_raw)
    production = Vector{Float64}(usable.delta_production)

    raw_pearson = cor(goal_difference, raw)
    production_pearson = cor(goal_difference, production)
    raw_spearman = StatsBase.corspearman(goal_difference, raw)
    production_spearman = StatsBase.corspearman(goal_difference, production)
    raw_deviance = r13_poisson_deviance(raw, usable)
    production_deviance = r13_poisson_deviance(production, usable)

    println("\n", "="^104)
    println("4. SIGNAL DIAGNOSTICS ON IDENTICAL COMPLETE-LINEUP FIXTURES")
    println("="^104)
    @printf("Usable matches                         %10d\n", nrow(usable))
    @printf("Pearson r: raw wealth                 %+10.5f\n", raw_pearson)
    @printf("Pearson r: production wealth          %+10.5f\n", production_pearson)
    @printf("Pearson improvement                   %+10.5f\n",
            production_pearson - raw_pearson)
    @printf("Spearman ρ: raw wealth                %+10.5f\n", raw_spearman)
    @printf("Spearman ρ: production wealth         %+10.5f\n", production_spearman)
    @printf("Spearman improvement                  %+10.5f\n",
            production_spearman - raw_spearman)
    @printf("Two-sided Poisson deviance: raw       %10.3f\n", raw_deviance)
    @printf("Two-sided Poisson deviance: production%10.3f\n", production_deviance)
    @printf("Deviance drop vs raw (positive better)%10.3f\n",
            raw_deviance - production_deviance)
end

# %%
# ==============================================================================
# 7. Execute deterministic diagnostic
# ==============================================================================

r13_print_curves()
r13_snapshot = r13_load_snapshot()
r13_results = r13_build_comparison(r13_snapshot)
r13_print_inventory(r13_snapshot, r13_results.comparison)
r13_print_inspection(r13_results.usable)
r13_print_diagnostics(r13_results.usable)
