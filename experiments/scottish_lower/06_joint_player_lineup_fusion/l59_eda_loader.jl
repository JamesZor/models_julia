# ==============================================================================
# Loader — deterministic ridge machinery for the six candidate FORMULATIONS
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# WHAT THIS IS
#   A fast, closed-form stand-in for the six MCMC candidates in `l60_loader.jl`.
#   Each candidate's log-rate structure is mapped onto one ridge design matrix, so a
#   whole formulation can be scored on held-out fixtures in seconds instead of hours.
#
# WHAT THIS IS NOT
#   It is not the model. Three deliberate simplifications are made, and every claim
#   made from this loader must be read against them:
#
#     1. IDENTITY LINK. The count models use `log λ`; this ridge is linear in the
#        rate. It preserves the ORDERING of a supremacy signal, not its scale.
#     2. STATIC TEAM STATE. `m05`'s `TimeDecayDynamics(180 d)` becomes a pair of
#        static attack/defence dummies fit on the history block. That is a strictly
#        weaker control: it cannot track a team that improves inside the window.
#     3. NO GAMMA ARM. The two-arm `JointGammaPoissonObservation` has no ridge
#        analogue, so the proxy-xG arm is absent. Scoring official xG supremacy as a
#        second target is the compensation, not a reproduction.
#
#   Consequently this loader RANKS formulations and detects dead signal. It does not
#   estimate the production models' predictive performance.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using Serialization
using Statistics
using StatsBase: tiedrank

const J59_DATA = BayesianFootball.Data
const J59_FEATURES = BayesianFootball.Features
const J59_MODELS = BayesianFootball.Models

const J59_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const J59_ENGLISH_CACHE = joinpath(J59_ROOT, "current_development", "scottish_lower",
                                   "l50_english_store.jls")
const J59_XG_CACHE = joinpath(J59_ROOT, "current_development", "scottish_lower",
                              "l92_pxg_validation_pull.jls")

const J59_TIERS_ENGLAND_SCOTLAND = [1, 2, 3, 84, 54, 55, 56, 57]
const J59_TIERS_SCOTLAND = [54, 55, 56, 57]
const J59_TIERS_LOWER = [56, 57]

# The candidates' fixed bench weight, taken from `l60_loader.jl` — not re-tuned here.
const J59_BENCH_WEIGHT = 0.10
const J59_RIDGE_PENALTY = 1.0e-3

# The English cache was serialized under this Main-level type name; the same
# loader-only segment must exist again before `deserialize` can rebuild it.
struct EnglishTiers <: J59_DATA.DataTournemantSegment end
J59_DATA.tournament_ids(::EnglishTiers) = [1, 2, 3, 84]

struct JointPlayerEDASegment <: J59_DATA.DataTournemantSegment
    tiers::Vector{Int}
end
J59_DATA.tournament_ids(segment::JointPlayerEDASegment) = segment.tiers

# ==============================================================================
# 1. Local snapshots
# ==============================================================================
function j59_vcat(frames::AbstractDataFrame...)
    usable = DataFrame[DataFrame(frame) for frame in frames if nrow(frame) > 0]
    isempty(usable) && return DataFrame()
    return vcat(usable...; cols = :union)
end

"Load the eight-tier England + Scotland snapshot from local caches only."
function j59_full_store()
    isfile(J59_ENGLISH_CACHE) || error("missing local English cache: $J59_ENGLISH_CACHE")
    english = deserialize(J59_ENGLISH_CACHE)
    upper = J59_DATA.load_datastore_cached(J59_DATA.ScottishUpper(); max_age_hours = 100_000)
    lower = J59_DATA.load_datastore_cached(J59_DATA.ScottishLower(); max_age_hours = 100_000)
    return J59_DATA.DataStore(
        JointPlayerEDASegment(copy(J59_TIERS_ENGLAND_SCOTLAND)),
        j59_vcat(english.matches, upper.matches, lower.matches),
        j59_vcat(english.statistics, upper.statistics, lower.statistics),
        j59_vcat(english.odds, upper.odds, lower.odds),
        j59_vcat(english.lineups, upper.lineups, lower.lineups),
        j59_vcat(english.incidents, upper.incidents, lower.incidents),
        j59_vcat(english.betfair_odds, upper.betfair_odds, lower.betfair_odds),
        j59_vcat(english.bbc, upper.bbc, lower.bbc),
        j59_vcat(english.bbc_events, upper.bbc_events, lower.bbc_events),
    )
end

function j59_subset_store(ds::J59_DATA.DataStore, tiers::Vector{Int})
    tier_set = Set(tiers)
    matches = ds.matches[in.(Int.(ds.matches.tournament_id), Ref(tier_set)), :]
    ids = Set(Int.(matches.match_id))
    take(frame) = nrow(frame) == 0 || !hasproperty(frame, :match_id) ? DataFrame(frame) :
                  frame[in.(Int.(frame.match_id), Ref(ids)), :]
    return J59_DATA.DataStore(
        JointPlayerEDASegment(copy(tiers)), matches,
        take(ds.statistics), take(ds.odds), take(ds.lineups), take(ds.incidents),
        take(ds.betfair_odds), take(ds.bbc), take(ds.bbc_events),
    )
end

"Official SofaScore xG cached by the independent r92 validation pull."
function j59_official_xg()
    isfile(J59_XG_CACHE) || error("missing local official-xG cache: $J59_XG_CACHE")
    pull = deserialize(J59_XG_CACHE)
    return Dict{Int,Tuple{Float64,Float64}}(
        Int(row.match_id) => (Float64(row.xg_home), Float64(row.xg_away))
        for row in eachrow(pull.official)
        if Float64(row.xg_home) + Float64(row.xg_away) > 0.0
    )
end

# ==============================================================================
# 2. One point-in-time feature bundle per scope
# ==============================================================================
#
# The split is the filtration. Everything downstream — both RAPM vectors, the
# standardisations inside them, and every ridge coefficient — is fit on `history_ids`
# alone. `target_ids` are scored and never fitted.

"Chronological (or target-season) split plus every covariate the six candidates need."
function j59_bundle(ds::J59_DATA.DataStore;
                    history_fraction::Float64 = 0.80,
                    target_seasons::Union{Nothing,Vector{String}} = nothing)
    ordered_all = sort(ds.matches, [:match_date, :match_id])
    if target_seasons === nothing
        ordered = ordered_all
        cut = clamp(floor(Int, history_fraction * nrow(ordered)), 1, nrow(ordered) - 1)
    else
        target_set = Set(target_seasons)
        target_rows = in.(coalesce.(ordered_all.season, ""), Ref(target_set))
        any(target_rows) || error("none of target seasons $(target_seasons) exist in this scope")
        first_target_date = minimum(ordered_all.match_date[target_rows])
        keep = (ordered_all.match_date .< first_target_date) .| target_rows
        ordered = ordered_all[keep, :]
        cut = count(ordered.match_date .< first_target_date)
        cut > 0 && cut < nrow(ordered) || error("target-season split has no history or no target")
    end

    ids = Int.(ordered.match_id)
    history_ids = ids[1:cut]

    # Both RAPM targets write the SAME keys, so each needs its own dictionary.
    shots = Dict{Symbol,Any}(:history_match_ids => history_ids)
    J59_FEATURES.add_feature!(
        shots,
        J59_FEATURES.ShotsPlusMinusFeature(
            w_sim = 0.0, λ = 1000.0, half_life_days = 730.0, fit_on = :history),
        ids, Dict{String,Int}(), ds)

    pxg = Dict{Symbol,Any}(:history_match_ids => history_ids)
    J59_FEATURES.add_feature!(
        pxg,
        J59_FEATURES.XGPlusMinusFeature(
            w_sim = 0.0, λ = 200.0, half_life_days = 730.0, fit_on = :history),
        ids, Dict{String,Int}(), ds)

    covariates = Dict{Symbol,Any}(:history_match_ids => history_ids)
    J59_FEATURES.add_feature!(
        covariates,
        J59_MODELS.ProductionWealthFeature(curve = J59_MODELS.RichardsSigmoid(23.0, 0.80, 2.0)),
        ids, Dict{String,Int}(), ds)
    J59_FEATURES.add_feature!(
        covariates, J59_FEATURES.DistanceFeature(metric = :log_dist_z),
        ids, Dict{String,Int}(), ds)

    teams = sort!(unique(vcat(String.(ordered.home_team), String.(ordered.away_team))))
    team_index = Dict{String,Int}(team => i for (i, team) in enumerate(teams))
    home_team = Int[team_index[String(t)] for t in ordered.home_team]
    away_team = Int[team_index[String(t)] for t in ordered.away_team]

    return (; ordered, ids, cut, history_ids, target_ids = ids[(cut + 1):end],
            shots, pxg, covariates, teams, team_index, home_team, away_team)
end

"Scored outcomes and the masks that decide which fixtures are fittable or scorable."
function j59_outcomes(bundle, official_xg::Dict{Int,Tuple{Float64,Float64}})
    ordered = bundle.ordered
    n = nrow(ordered)
    home_goals = Float64.(coalesce.(ordered.home_score, 0))
    away_goals = Float64.(coalesce.(ordered.away_score, 0))
    played = .!ismissing.(ordered.home_score) .& .!ismissing.(ordered.away_score)
    xg_home = fill(NaN, n)
    xg_away = fill(NaN, n)
    for (i, id) in enumerate(bundle.ids)
        haskey(official_xg, id) || continue
        xg_home[i], xg_away[i] = official_xg[id]
    end
    xg_available = isfinite.(xg_home) .& isfinite.(xg_away)
    target = falses(n)
    target[(bundle.cut + 1):end] .= true
    return (; home_goals, away_goals, played, xg_home, xg_away, xg_available, target)
end

# ==============================================================================
# 3. The six formulations as ridge designs
# ==============================================================================
#
# `home`/`away` are the per-side player-rating columns; they enter the two rates
# symmetrically (own rating, opponent rating), exactly as `PlayerLineupDynamics`
# does. `supremacy` holds the antisymmetric covariates: `covariate_sides(SupremacyRole(), q)`
# is `(q, -q)`, so the home rate gets `+q` and the away rate `-q`. `team_effects`
# switches on the static attack/defence dummies that stand in for team dynamics.

struct J59Formulation
    name::String
    home::Matrix{Float64}
    away::Matrix{Float64}
    supremacy::Matrix{Float64}
    team_effects::Bool
end

j59_column(data, key) = Vector{Float64}(data[key])
j59_none(n::Int) = Matrix{Float64}(undef, n, 0)

"Starters plus a fixed-weight bench, matching `BenchWeightedPlayerAggregation`."
function j59_bench_sides(data; bench_weight::Float64 = J59_BENCH_WEIGHT)
    home = j59_column(data, :flat_home_outfield_rating) .+
           bench_weight .* j59_column(data, :flat_home_bench_rating)
    away = j59_column(data, :flat_away_outfield_rating) .+
           bench_weight .* j59_column(data, :flat_away_bench_rating)
    return reshape(home, :, 1), reshape(away, :, 1)
end

function j59_outfield_sides(data)
    return reshape(j59_column(data, :flat_home_outfield_rating), :, 1),
           reshape(j59_column(data, :flat_away_outfield_rating), :, 1)
end

"Map every `l60_loader.jl` candidate onto its ridge analogue, in leaderboard order."
function j59_formulations(bundle; bench_weight::Float64 = J59_BENCH_WEIGHT)
    n = length(bundle.ids)
    wealth = reshape(j59_column(bundle.covariates, :flat_delta_production_wealth), :, 1)
    distance = reshape(j59_column(bundle.covariates, :flat_distance), :, 1)

    shots_out_h, shots_out_a = j59_outfield_sides(bundle.shots)
    shots_bench_h, shots_bench_a = j59_bench_sides(bundle.shots; bench_weight)
    pxg_bench_h, pxg_bench_a = j59_bench_sides(bundle.pxg; bench_weight)

    return J59Formulation[
        # Team state, no lineup: the control the player arms must beat.
        J59Formulation("m05_joint_production_wealth",
                       j59_none(n), j59_none(n), wealth, true),
        J59Formulation("m09_joint_player_shots_outfield",
                       shots_out_h, shots_out_a, j59_none(n), false),
        J59Formulation("m10_joint_player_shots_bench",
                       shots_bench_h, shots_bench_a, j59_none(n), false),
        J59Formulation("m11_joint_player_pxg_bench",
                       pxg_bench_h, pxg_bench_a, j59_none(n), false),
        J59Formulation("m12_joint_hybrid_synergy",
                       shots_bench_h, shots_bench_a, wealth, false),
        J59Formulation("m13_joint_composite",
                       shots_bench_h, shots_bench_a, hcat(wealth, distance), false),
    ]
end

"""
    j59_design(formulation, bundle, indices)

Stack the two rates of each selected fixture into one design. Rows `1:n` are the
home rates and rows `n+1:2n` the away rates, so a single coefficient vector fits
both sides under the model's own symmetry.
"""
function j59_design(formulation::J59Formulation, bundle, indices::Vector{Int})
    n = length(indices)
    p = size(formulation.home, 2)
    q = size(formulation.supremacy, 2)
    t = formulation.team_effects ? length(bundle.teams) : 0
    design = zeros(Float64, 2n, 2 + 2p + q + 2t)

    @views begin
        design[:, 1] .= 1.0
        design[1:n, 2] .= 1.0
        if p > 0
            design[1:n, 3:(2 + p)] .= formulation.home[indices, :]
            design[1:n, (3 + p):(2 + 2p)] .= formulation.away[indices, :]
            design[(n + 1):(2n), 3:(2 + p)] .= formulation.away[indices, :]
            design[(n + 1):(2n), (3 + p):(2 + 2p)] .= formulation.home[indices, :]
        end
        if q > 0
            offset = 2 + 2p
            design[1:n, (offset + 1):(offset + q)] .= formulation.supremacy[indices, :]
            design[(n + 1):(2n), (offset + 1):(offset + q)] .= .-formulation.supremacy[indices, :]
        end
    end

    if t > 0
        attack = 2 + 2p + q
        defence = attack + t
        for (row, i) in enumerate(indices)
            home_team = bundle.home_team[i]
            away_team = bundle.away_team[i]
            design[row, attack + home_team] = 1.0
            design[row, defence + away_team] = 1.0
            design[n + row, attack + away_team] = 1.0
            design[n + row, defence + home_team] = 1.0
        end
    end
    return design
end

"Ridge on everything except the intercept and home advantage; those two are free."
function j59_ridge_fit(formulation::J59Formulation, bundle, home_target::Vector{Float64},
                       away_target::Vector{Float64}, mask::BitVector;
                       penalty::Float64 = J59_RIDGE_PENALTY)
    indices = findall(mask)
    design = j59_design(formulation, bundle, indices)
    response = vcat(home_target[indices], away_target[indices])
    gram = transpose(design) * design
    regularizer = Matrix{Float64}(I, size(gram)) .* penalty
    regularizer[1, 1] = 0.0
    regularizer[2, 2] = 0.0
    return (gram + regularizer) \ (transpose(design) * response)
end

function j59_predict(formulation::J59Formulation, bundle, coefficients::Vector{Float64})
    indices = collect(1:length(bundle.ids))
    prediction = j59_design(formulation, bundle, indices) * coefficients
    n = length(indices)
    return (; home = prediction[1:n], away = prediction[(n + 1):(2n)])
end

# ==============================================================================
# 4. Held-out scoring
# ==============================================================================
j59_pearson(x, y) = length(x) < 3 || std(x) == 0.0 || std(y) == 0.0 ? NaN : cor(x, y)
j59_spearman(x, y) = j59_pearson(tiedrank(x), tiedrank(y))

function j59_metrics(name, prediction, home_target, away_target, mask::BitVector)
    indices = findall(mask)
    predicted = prediction.home[indices] .- prediction.away[indices]
    observed = home_target[indices] .- away_target[indices]
    residual = predicted .- observed
    denominator = sum(abs2, observed .- mean(observed))
    # R2 against the HELD-OUT mean. A negative value means the formulation is worse
    # than predicting "every fixture is a draw on average"; that is the real gate.
    r2 = denominator > 0.0 ? 1.0 - sum(abs2, residual) / denominator : NaN
    return (;
        formulation = name,
        n = length(indices),
        r = j59_pearson(predicted, observed),
        rho = j59_spearman(predicted, observed),
        mae = mean(abs, residual),
        r2,
    )
end

"Score every formulation on held-out goal supremacy and official-xG supremacy."
function j59_leaderboard(ds::J59_DATA.DataStore;
                         history_fraction::Float64 = 0.80,
                         target_seasons::Union{Nothing,Vector{String}} = nothing,
                         bench_weight::Float64 = J59_BENCH_WEIGHT,
                         official_xg::Union{Nothing,Dict{Int,Tuple{Float64,Float64}}} = nothing)
    bundle = j59_bundle(ds; history_fraction, target_seasons)
    outcomes = j59_outcomes(bundle, official_xg === nothing ? j59_official_xg() : official_xg)
    formulations = j59_formulations(bundle; bench_weight)

    goal_train = BitVector(.!outcomes.target .& outcomes.played)
    goal_test = BitVector(outcomes.target .& outcomes.played)
    xg_train = BitVector(.!outcomes.target .& outcomes.xg_available)
    xg_test = BitVector(outcomes.target .& outcomes.xg_available)

    rows = NamedTuple[]
    for formulation in formulations
        goal_fit = j59_ridge_fit(formulation, bundle, outcomes.home_goals,
                                 outcomes.away_goals, goal_train)
        goal_prediction = j59_predict(formulation, bundle, goal_fit)
        push!(rows, merge((target = "goal supremacy",), j59_metrics(
            formulation.name, goal_prediction, outcomes.home_goals,
            outcomes.away_goals, goal_test)))

        if count(xg_train) >= 30 && count(xg_test) >= 20
            xg_fit = j59_ridge_fit(formulation, bundle, outcomes.xg_home,
                                   outcomes.xg_away, xg_train)
            xg_prediction = j59_predict(formulation, bundle, xg_fit)
            push!(rows, merge((target = "official xG supremacy",), j59_metrics(
                formulation.name, xg_prediction, outcomes.xg_home,
                outcomes.xg_away, xg_test)))
        end
    end
    return (; table = DataFrame(rows), bundle, outcomes)
end

# ==============================================================================
# 5. Coverage diagnostics
# ==============================================================================
#
# Every one of these can silently turn a candidate into a copy of a simpler one:
# a zero RAPM vector makes m09-m13 an intercept model, a zero wealth column makes
# m12 into m10, and a constant distance column makes m13 into m12. They are reported
# beside the leaderboard so a null result is never mistaken for a fair test.

function j59_coverage(bundle)
    target = falses(length(bundle.ids))
    target[(bundle.cut + 1):end] .= true
    share(v) = count(v) / max(length(v), 1)

    shots_h = j59_column(bundle.shots, :flat_home_outfield_rating)
    shots_a = j59_column(bundle.shots, :flat_away_outfield_rating)
    pxg_h = j59_column(bundle.pxg, :flat_home_outfield_rating)
    pxg_a = j59_column(bundle.pxg, :flat_away_outfield_rating)
    wealth = j59_column(bundle.covariates, :flat_delta_production_wealth)
    distance = j59_column(bundle.covariates, :flat_distance)
    fallback = Vector{Int}(bundle.covariates[:flat_distance_fallback])

    return (;
        n_matches = length(bundle.ids),
        n_history = bundle.cut,
        n_target = length(bundle.ids) - bundle.cut,
        shots_rated_share = share((shots_h .!= 0.0) .| (shots_a .!= 0.0)),
        pxg_rated_share = share((pxg_h .!= 0.0) .| (pxg_a .!= 0.0)),
        wealth_share = share(wealth .!= 0.0),
        distance_fallback_share = share(fallback .== 1),
        distance_sd = std(distance),
        target_shots_rated_share = share(((shots_h .!= 0.0) .| (shots_a .!= 0.0))[target]),
        target_wealth_share = share((wealth .!= 0.0)[target]),
    )
end

j59_print_table(table::DataFrame) = (show(table; allrows = true, allcols = true,
                                          truncate = 40); println())
