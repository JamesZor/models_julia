# ==============================================================================
# Loader — shared multi-tier lineup-formulation EDA machinery
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using LibPQ
using LinearAlgebra
using Serialization
using Statistics
using StatsBase: tiedrank

const PLD_DATA = BayesianFootball.Data
const PLD_FEATURES = BayesianFootball.Features
const PLD_TIERS_A = [1, 2, 3, 84, 54, 55, 56, 57]
const PLD_TIERS_B = [54, 55, 56, 57]
const PLD_TIERS_C = [56, 57]
const PLD_ENGLISH_CACHE = joinpath(@__DIR__, "..", "scottish_lower", "l50_english_store.jls")
const PLD_XG_CACHE = joinpath(@__DIR__, "..", "scottish_lower", "l92_pxg_validation_pull.jls")

# The existing English cache was serialized by l50 with this Main-level type name.
# Re-declaring the same loader-only segment is required before deserialization.
struct EnglishTiers <: PLD_DATA.DataTournemantSegment end
PLD_DATA.tournament_ids(::EnglishTiers) = [1, 2, 3, 84]

struct PlayerLineupEDASegment <: PLD_DATA.DataTournemantSegment
    tiers::Vector{Int}
end
PLD_DATA.tournament_ids(segment::PlayerLineupEDASegment) = segment.tiers

function pld_vcat(frames::AbstractDataFrame...)
    usable = DataFrame[DataFrame(frame) for frame in frames if nrow(frame) > 0]
    isempty(usable) && return DataFrame()
    return vcat(usable...; cols=:union)
end

"Load the eight-tier local snapshot without fetching or sampling remotely."
function pld_scope_a_store()
    isfile(PLD_ENGLISH_CACHE) || error("missing local English cache: $PLD_ENGLISH_CACHE")
    english = deserialize(PLD_ENGLISH_CACHE)
    upper = PLD_DATA.load_datastore_cached(PLD_DATA.ScottishUpper(); max_age_hours=100_000)
    lower = PLD_DATA.load_datastore_cached(PLD_DATA.ScottishLower(); max_age_hours=100_000)
    return PLD_DATA.DataStore(
        PlayerLineupEDASegment(copy(PLD_TIERS_A)),
        pld_vcat(english.matches, upper.matches, lower.matches),
        pld_vcat(english.statistics, upper.statistics, lower.statistics),
        pld_vcat(english.odds, upper.odds, lower.odds),
        pld_vcat(english.lineups, upper.lineups, lower.lineups),
        pld_vcat(english.incidents, upper.incidents, lower.incidents),
        pld_vcat(english.betfair_odds, upper.betfair_odds, lower.betfair_odds),
        pld_vcat(english.bbc, upper.bbc, lower.bbc),
        pld_vcat(english.bbc_events, upper.bbc_events, lower.bbc_events),
    )
end

function pld_subset_store(ds::PLD_DATA.DataStore, tiers::Vector{Int})
    tier_set = Set(tiers)
    matches = ds.matches[in.(Int.(ds.matches.tournament_id), Ref(tier_set)), :]
    ids = Set(Int.(matches.match_id))
    take(frame) = nrow(frame) == 0 || !hasproperty(frame, :match_id) ? DataFrame(frame) :
                  frame[in.(Int.(frame.match_id), Ref(ids)), :]
    return PLD_DATA.DataStore(
        PlayerLineupEDASegment(copy(tiers)), matches,
        take(ds.statistics), take(ds.odds), take(ds.lineups), take(ds.incidents),
        take(ds.betfair_odds), take(ds.bbc), take(ds.bbc_events),
    )
end

"Official SofaScore xG cached by the independent r92 validation stream."
function pld_official_xg()
    isfile(PLD_XG_CACHE) || error("missing local official-xG cache: $PLD_XG_CACHE")
    pull = deserialize(PLD_XG_CACHE)
    return Dict{Int,Tuple{Float64,Float64}}(
        Int(row.match_id) => (Float64(row.xg_home), Float64(row.xg_away))
        for row in eachrow(pull.official)
        if Float64(row.xg_home) + Float64(row.xg_away) > 0.0
    )
end

pld_pm_config(::Val{:y_xg}) = PLD_FEATURES.XGPlusMinusFeature()
pld_pm_config(::Val{:y_shots}) = PLD_FEATURES.ShotsPlusMinusFeature()

function pld_feature_data(ds::PLD_DATA.DataStore, target::Symbol;
                          history_fraction::Float64=0.80,
                          target_seasons::Union{Nothing,Vector{String}}=nothing)
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
    data = Dict{Symbol,Any}(:history_match_ids => history_ids)
    config = pld_pm_config(Val(target))
    PLD_FEATURES.add_feature!(data, config, ids, Dict{String,Int}(), ds)
    return (; data, ordered, ids, cut, history_ids, target_ids=ids[(cut + 1):end])
end

function pld_outcomes(feature_run, official_xg::Dict{Int,Tuple{Float64,Float64}})
    ordered = feature_run.ordered
    n = nrow(ordered)
    home_goals = Float64.(coalesce.(ordered.home_score, 0))
    away_goals = Float64.(coalesce.(ordered.away_score, 0))
    played = .!ismissing.(ordered.home_score) .& .!ismissing.(ordered.away_score)
    xg_home = fill(NaN, n)
    xg_away = fill(NaN, n)
    for (i, id) in enumerate(feature_run.ids)
        haskey(official_xg, id) || continue
        xg_home[i], xg_away[i] = official_xg[id]
    end
    xg_available = isfinite.(xg_home) .& isfinite.(xg_away)
    target = falses(n)
    target[(feature_run.cut + 1):end] .= true
    return (; home_goals, away_goals, played, xg_home, xg_away, xg_available, target)
end

struct PLDRepresentation
    name::String
    home::Matrix{Float64}
    away::Matrix{Float64}
end

pld_matrix(data, keys) = hcat((Vector{Float64}(data[key]) for key in keys)...)

function pld_representations(data; bench_weight::Float64=0.25)
    out_h = Vector{Float64}(data[:flat_home_outfield_rating])
    out_a = Vector{Float64}(data[:flat_away_outfield_rating])
    bench_h = Vector{Float64}(data[:flat_home_bench_rating])
    bench_a = Vector{Float64}(data[:flat_away_bench_rating])
    return PLDRepresentation[
        PLDRepresentation("outfield starters", reshape(out_h, :, 1), reshape(out_a, :, 1)),
        PLDRepresentation("starters + bench", reshape(out_h .+ bench_weight .* bench_h, :, 1),
                          reshape(out_a .+ bench_weight .* bench_a, :, 1)),
        PLDRepresentation("positional vectors",
            pld_matrix(data, (:flat_home_F_rating, :flat_home_M_rating, :flat_home_D_rating,
                              :flat_home_bench_F_rating, :flat_home_bench_M_rating,
                              :flat_home_bench_D_rating)),
            pld_matrix(data, (:flat_away_F_rating, :flat_away_M_rating, :flat_away_D_rating,
                              :flat_away_bench_F_rating, :flat_away_bench_M_rating,
                              :flat_away_bench_D_rating))),
        PLDRepresentation("expected minutes",
            reshape(Vector{Float64}(data[:flat_home_minute_weighted_rating]), :, 1),
            reshape(Vector{Float64}(data[:flat_away_minute_weighted_rating]), :, 1)),
    ]
end

function pld_rate_design(representation::PLDRepresentation, indices::Vector{Int})
    p = size(representation.home, 2)
    n = length(indices)
    design = Matrix{Float64}(undef, 2n, 2 + 2p)
    @views begin
        design[1:n, 1] .= 1.0
        design[(n + 1):(2n), 1] .= 1.0
        design[1:n, 2] .= 1.0
        design[(n + 1):(2n), 2] .= 0.0
        design[1:n, 3:(2 + p)] .= representation.home[indices, :]
        design[1:n, (3 + p):(2 + 2p)] .= representation.away[indices, :]
        design[(n + 1):(2n), 3:(2 + p)] .= representation.away[indices, :]
        design[(n + 1):(2n), (3 + p):(2 + 2p)] .= representation.home[indices, :]
    end
    return design
end

function pld_ridge_fit(representation::PLDRepresentation, home_target::Vector{Float64},
                       away_target::Vector{Float64}, mask::BitVector; penalty::Float64=1.0e-3)
    indices = findall(mask)
    design = pld_rate_design(representation, indices)
    response = vcat(home_target[indices], away_target[indices])
    gram = transpose(design) * design
    regularizer = Matrix{Float64}(I, size(gram)) .* penalty
    regularizer[1, 1] = 0.0
    regularizer[2, 2] = 0.0
    return (gram + regularizer) \ (transpose(design) * response)
end

function pld_predict(representation::PLDRepresentation, coefficients::Vector{Float64})
    indices = collect(1:size(representation.home, 1))
    prediction = pld_rate_design(representation, indices) * coefficients
    n = length(indices)
    return (; home=prediction[1:n], away=prediction[(n + 1):(2n)])
end

pld_pearson(x, y) = length(x) < 3 || std(x) == 0.0 || std(y) == 0.0 ? NaN : cor(x, y)
pld_spearman(x, y) = pld_pearson(tiedrank(x), tiedrank(y))

function pld_metrics(name, prediction, home_target, away_target, mask::BitVector)
    indices = findall(mask)
    predicted = prediction.home[indices] .- prediction.away[indices]
    observed = home_target[indices] .- away_target[indices]
    residual = predicted .- observed
    denominator = sum(abs2, observed .- mean(observed))
    r2 = denominator > 0.0 ? 1.0 - sum(abs2, residual) / denominator : NaN
    return (;
        formulation=name,
        n=length(indices),
        r=pld_pearson(predicted, observed),
        rho=pld_spearman(predicted, observed),
        mae=mean(abs, residual),
        r2,
    )
end

function pld_leaderboard(ds::PLD_DATA.DataStore, target::Symbol;
                         history_fraction::Float64=0.80, bench_weight::Float64=0.25,
                         target_seasons::Union{Nothing,Vector{String}}=nothing)
    run = pld_feature_data(ds, target; history_fraction, target_seasons)
    outcomes = pld_outcomes(run, pld_official_xg())
    representations = pld_representations(run.data; bench_weight)
    goal_train = BitVector(.!outcomes.target .& outcomes.played)
    goal_test = BitVector(outcomes.target .& outcomes.played)
    xg_train = BitVector(.!outcomes.target .& outcomes.xg_available)
    xg_test = BitVector(outcomes.target .& outcomes.xg_available)
    rows = NamedTuple[]
    for representation in representations
        goal_fit = pld_ridge_fit(representation, outcomes.home_goals,
                                 outcomes.away_goals, goal_train)
        goal_prediction = pld_predict(representation, goal_fit)
        push!(rows, merge((target="scoreline",), pld_metrics(
            representation.name, goal_prediction, outcomes.home_goals,
            outcomes.away_goals, goal_test)))
        if count(xg_train) >= 30 && count(xg_test) >= 20
            xg_fit = pld_ridge_fit(representation, outcomes.xg_home,
                                   outcomes.xg_away, xg_train)
            xg_prediction = pld_predict(representation, xg_fit)
            push!(rows, merge((target="official SofaScore xG",), pld_metrics(
                representation.name, xg_prediction, outcomes.xg_home,
                outcomes.xg_away, xg_test)))
        end
    end
    return (; table=DataFrame(rows), run, outcomes)
end

function pld_select_bench_weight(run, outcomes; grid=collect(0.10:0.025:0.35))
    n_history = run.cut
    validation_start = max(1, floor(Int, 0.90 * n_history))
    train = falses(length(run.ids)); train[1:(validation_start - 1)] .= true
    validation = falses(length(run.ids)); validation[validation_start:n_history] .= true
    train .&= outcomes.played
    validation .&= outcomes.played
    rows = NamedTuple[]
    for weight in grid
        representation = pld_representations(run.data; bench_weight=weight)[2]
        fit = pld_ridge_fit(representation, outcomes.home_goals, outcomes.away_goals,
                            BitVector(train))
        prediction = pld_predict(representation, fit)
        metric = pld_metrics("starters + bench", prediction, outcomes.home_goals,
                             outcomes.away_goals, BitVector(validation))
        push!(rows, merge((w_bench=weight,), metric))
    end
    table = DataFrame(rows)
    winner = table.w_bench[argmax(table.r2)]
    return (; winner, table)
end

function pld_print_table(table::DataFrame)
    show(table; allrows=true, allcols=true, truncate=40)
    println()
end
