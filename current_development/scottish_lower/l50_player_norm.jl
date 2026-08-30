# ==============================================================================
# l50 — LOADER: the English reference laboratory and the normalisation library
# ==============================================================================
#
# Definitions only, no execution. Shared by r50 (structure) and r51 (bench).
#
# WHY ENGLAND. Three things have to co-exist to study how a player rating should be
# normalised: a REFERENCE rating, the DEMOGRAPHICS to normalise by, and the live text
# that RAPM is built from. Measured coverage across the eight tiers in the store:
#
#   tier          rating    market value    date of birth    live text
#   1  ENG PL      73.8%        97.2%          100.0%          yes
#   2  ENG Champ   76.1%        96.0%          100.0%          yes
#   3  ENG L1      81.8%        94.3%           99.9%          yes
#   84 ENG L2      80.7%        92.6%           99.7%          yes
#   54 SCO Prem    73.9%         0.0%            0.0%          yes
#   55 SCO Champ   39.9%         0.0%            0.0%          yes
#   56 SCO L1       0.0%        63.1%           99.1%          yes
#   57 SCO L2       0.0%        52.7%           98.5%          yes
#
# Only the English tiers have all three. Scottish Upper has ratings but NO market value
# or date of birth at all; Scottish Lower — the deployment target — has wealth but no
# ratings, which is the entire reason this study has to borrow a laboratory.
#
# ⚠ WHAT THE SOFASCORE RATING IS. Measured on 218,478 starter rows across tiers
# 1/2/3/84/54/55, the mean rating by position is M 6.9176, D 6.9136, F 6.9063,
# G 6.9100 — a spread of 0.011 against a standard deviation of 0.55. It is
# POSITION-NORMALISED, and league-normalised too (Premier League D 6.909 vs
# League Two D 6.891). It is therefore a WITHIN-CONTEXT relative score, not an
# absolute quality measure, and only its DISPERSION differs by position
# (G 0.68 > F 0.62 > D/M 0.55). Any comparison against it must be made within
# position, which is most of the point of this stream.
#
# CREDENTIALS: `BF_DB_URL` only. Nothing is committed here.
# ==============================================================================

using DataFrames
using Dates
using LibPQ
using Printf
using Serialization
using Statistics
using StatsBase: tiedrank

const L50_DATA = BayesianFootball.Data
const L50_FEATURES = BayesianFootball.Features

"""The four English tiers, as a segment the production fetchers understand."""
struct EnglishTiers <: L50_DATA.DataTournemantSegment end
L50_DATA.tournament_ids(::EnglishTiers) = [1, 2, 3, 84]

const L50_TIER_NAMES = Dict(1 => "ENG Premier League", 2 => "ENG Championship",
                            3 => "ENG League One", 84 => "ENG League Two")
const L50_CACHE = joinpath(@__DIR__, "l50_english_store.jls")
const L50_POSITIONS = ("G", "D", "M", "F")

# ==============================================================================
# 1. THE REFERENCE STORE
# ==============================================================================

"""
    l50_store(; force = false) -> DataStore

Only the four domains this study needs — matches, lineups, incidents and BBC shot
commentary — fetched through the PRODUCTION fetchers so every schema, coercion and QA
rule is identical to what the deployed feature sees. Odds, Betfair and match statistics
are deliberately left empty: they are large and nothing here reads them.
"""
function l50_store(; force::Bool = false)
    if !force && isfile(L50_CACHE)
        @info "l50: loading cached English store from $(basename(L50_CACHE))"
        return deserialize(L50_CACHE)
    end
    url = get(ENV, "BF_DB_URL") do
        error("BF_DB_URL is not set. Export it (or source .env) before running r50/r51.")
    end
    segment = EnglishTiers()
    @info "l50: fetching tiers $(L50_DATA.tournament_ids(segment)) — slow path, cached afterwards"
    conn = LibPQ.Connection(url)
    try
        matches    = L50_DATA.load_data(conn, segment, L50_DATA.MatchesData())
        lineups    = L50_DATA.load_data(conn, segment, L50_DATA.LineUpsData())
        incidents  = L50_DATA.load_data(conn, segment, L50_DATA.IncidentsData())
        bbc_events = L50_DATA.load_data(conn, segment, L50_DATA.BBCEventsData())
        empty = DataFrame()
        ds = L50_DATA.DataStore(segment, matches, empty, empty, lineups, incidents,
                                empty, empty, bbc_events)
        serialize(L50_CACHE, ds)
        @info "l50: cached $(nrow(matches)) matches, $(nrow(lineups)) lineup rows"
        return ds
    finally
        close(conn)
    end
end

# ==============================================================================
# 2. THE PLAYER FRAME
# ==============================================================================

"""
    l50_player_frame(ds, ratings, exposure; match_ids, reference_date) -> DataFrame

One row per rated player: the fitted RAPM, the SofaScore yardstick, and the three
covariates a normalisation might use — age at `reference_date`, log market value, and
modal position.

`match_ids` restricts which matches contribute to the SofaScore mean and the modal
position, so a held-out evaluation can build this from the history block alone.
"""
function l50_player_frame(ds::L50_DATA.DataStore, ratings::DataFrame,
                          exposure::DataFrame;
                          match_ids::Union{Nothing,Set{Int}} = nothing,
                          reference_date::Date)
    ratings_by = Dict{Int,Vector{Float64}}()
    position_counts = Dict{Int,Dict{String,Int}}()
    value_of = Dict{Int,Float64}()
    dob_of = Dict{Int,Float64}()
    name_of = Dict{Int,String}()
    minutes_of = Dict{Int,Float64}()

    columns = propertynames(ds.lineups)
    value_col = :proposed_market_value in columns ? :proposed_market_value :
                (:market_value in columns ? :market_value : nothing)

    for row in eachrow(ds.lineups)
        ismissing(row.player_id) && continue
        match_ids === nothing || Int(row.match_id) in match_ids || continue
        coalesce(row.is_substitute, false) && continue
        pid = Int(row.player_id)

        ismissing(row.rating) || push!(get!(ratings_by, pid, Float64[]), Float64(row.rating))
        counts = get!(position_counts, pid, Dict{String,Int}())
        p = L50_FEATURES.pm_clean_position(row.position)
        counts[p] = get(counts, p, 0) + 1
        ismissing(row.player_name) || (name_of[pid] = String(row.player_name))
        if value_col !== nothing && !ismissing(row[value_col])
            v = try Float64(row[value_col]) catch; NaN end
            isfinite(v) && v > 0 && (value_of[pid] = v)
        end
        if :date_of_birth_timestamp in columns && !ismissing(row.date_of_birth_timestamp)
            d = try Float64(row.date_of_birth_timestamp) catch; NaN end
            isfinite(d) && d > 0 && (dob_of[pid] = d)
        end
        if :minutes_played in columns && !ismissing(row.minutes_played)
            m = try Float64(row.minutes_played) catch; 0.0 end
            minutes_of[pid] = get(minutes_of, pid, 0.0) + max(m, 0.0)
        end
    end

    segments_of = Dict{Int,Float64}(
        Int(r.player_id) => Float64(r.n_segments) for r in eachrow(exposure))
    stint_minutes = Dict{Int,Float64}(
        Int(r.player_id) => Float64(r.minutes) for r in eachrow(exposure))
    reference_unix = datetime2unix(DateTime(reference_date))

    rows = NamedTuple[]
    for r in eachrow(ratings)
        pid = Int(r.player_id)
        age = haskey(dob_of, pid) ? (reference_unix - dob_of[pid]) / (365.25 * 86_400) : NaN
        (isfinite(age) && 14 < age < 46) || (age = NaN)
        value = get(value_of, pid, NaN)
        sofa = get(ratings_by, pid, Float64[])
        push!(rows, (
            player_id = pid,
            name = get(name_of, pid, "player $pid"),
            rapm = Float64(r.rapm),
            position = haskey(position_counts, pid) ? argmax(position_counts[pid]) : "?",
            age = age,
            log_value = (isfinite(value) && value > 0) ? log(value) : NaN,
            value = value,
            sofa_mean = isempty(sofa) ? NaN : mean(sofa),
            n_sofa = length(sofa),
            n_segments = get(segments_of, pid, 0.0),
            stint_minutes = get(stint_minutes, pid, 0.0),
        ))
    end
    return DataFrame(rows)
end

# ==============================================================================
# 3. THE NORMALISATION LIBRARY
# ==============================================================================
#
# Every strategy maps the player frame to one adjusted rating per row. They are
# deliberately written as independent transforms rather than composed knobs, so the
# bench compares whole strategies rather than a partially-explored product grid.
#
# ⚠ ALL OF THEM MUST BE FITTABLE ON HISTORY ALONE. Anything that estimates something
# (a position mean, a demographic regression) takes it from the frame it is handed,
# which the bench builds from the history block only.

"Exposure shrink toward the neutral player — what production already does."
l50_exposure_weight(n::Real, kappa::Real) = n / (n + kappa)

"Fit `y ~ 1 + age + age^2 + log_value` within each position; return fitted values."
function l50_demographic_fit(frame::DataFrame, y::Vector{Float64})
    fitted = fill(NaN, nrow(frame))
    for position in L50_POSITIONS
        idx = findall(i -> frame.position[i] == position &&
                           isfinite(y[i]) && isfinite(frame.age[i]) &&
                           isfinite(frame.log_value[i]), 1:nrow(frame))
        length(idx) < 30 && continue
        age = frame.age[idx]
        X = hcat(age, age .^ 2, frame.log_value[idx])
        fit = eda_ols(X, y[idx])
        fitted[idx] = y[idx] .- fit.resid
    end
    # Positions with too few players fall back to the position mean, or the global mean.
    for position in L50_POSITIONS
        idx = findall(i -> frame.position[i] == position && isnan(fitted[i]), 1:nrow(frame))
        isempty(idx) && continue
        pool = [y[i] for i in 1:nrow(frame) if frame.position[i] == position && isfinite(y[i])]
        fitted[idx] .= isempty(pool) ? 0.0 : mean(pool)
    end
    fitted[isnan.(fitted)] .= 0.0
    return fitted
end

"Within-position standardisation. Zero variance or an unknown position leaves the value alone."
function l50_position_z(frame::DataFrame, y::Vector{Float64})
    out = copy(y)
    for position in L50_POSITIONS
        idx = findall(i -> frame.position[i] == position && isfinite(y[i]), 1:nrow(frame))
        length(idx) < 10 && continue
        mu = mean(y[idx]); sd = std(y[idx])
        sd < 1e-12 && continue
        out[idx] .= (y[idx] .- mu) ./ sd
    end
    out[.!isfinite.(out)] .= 0.0
    return out
end

"Within-position normal scores (van der Waerden) — the rank-robust cousin of `l50_position_z`."
function l50_position_rank(frame::DataFrame, y::Vector{Float64})
    out = zeros(length(y))
    for position in L50_POSITIONS
        idx = findall(i -> frame.position[i] == position && isfinite(y[i]), 1:nrow(frame))
        length(idx) < 10 && continue
        n = length(idx)
        ranks = tiedrank(y[idx])
        # Probit of the plotting position; sqrt(2)*erfinv is the standard normal quantile.
        out[idx] .= [sqrt(2) * _l50_erfinv(2 * (r / (n + 1)) - 1) for r in ranks]
    end
    return out
end

# Small, dependency-free inverse error function (Giles' rational approximation).
function _l50_erfinv(x::Float64)
    x = clamp(x, -0.999999, 0.999999)
    w = -log((1 - x) * (1 + x))
    if w < 5.0
        w -= 2.5
        p = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
                  -0.00125372503, -0.00417768164, 0.246640727, 1.50140941)
            p = p * w + c
        end
        return p * x
    else
        w = sqrt(w) - 3.0
        p = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
                  -0.0076224613, 0.00943887047, 1.00167406, 2.83297682)
            p = p * w + c
        end
        return p * x
    end
end

"""
    l50_strategies(kappa) -> Vector{(name, description, fn)}

`fn(frame) -> Vector{Float64}`, the adjusted rating per row.
"""
function l50_strategies(; kappa::Float64 = 20.0)
    exposure_shrunk(frame) = frame.rapm .* l50_exposure_weight.(frame.n_segments, kappa)
    keeper_mask(frame) = [p == "G" ? 0.0 : 1.0 for p in frame.position]

    return [
        ("raw", "ridge coefficient, no adjustment at all",
            frame -> copy(frame.rapm)),
        ("exposure", "PRODUCTION BASELINE: shrink by n/(n+kappa)",
            exposure_shrunk),
        ("drop_gk", "exposure shrink, goalkeepers contribute zero",
            frame -> exposure_shrunk(frame) .* keeper_mask(frame)),
        ("zpos", "standardise within modal position",
            frame -> l50_position_z(frame, exposure_shrunk(frame))),
        ("zpos_nogk", "standardise within position AND drop goalkeepers",
            frame -> l50_position_z(frame, exposure_shrunk(frame)) .* keeper_mask(frame)),
        ("rank_pos", "within-position normal scores (rank-robust)",
            frame -> l50_position_rank(frame, exposure_shrunk(frame)) .* keeper_mask(frame)),
        ("resid_demo", "residual after regressing out age, age^2 and log value, per position",
            function (frame)
                y = exposure_shrunk(frame)
                return y .- l50_demographic_fit(frame, y)
            end),
        ("prior_demo", "shrink toward the demographic prediction instead of toward zero",
            function (frame)
                y = exposure_shrunk(frame)
                m = l50_demographic_fit(frame, y)
                w = l50_exposure_weight.(frame.n_segments, kappa)
                return w .* y .+ (1 .- w) .* m
            end),
        ("prior_zpos", "demographic prior, then standardise within position, no keeper",
            function (frame)
                y = exposure_shrunk(frame)
                m = l50_demographic_fit(frame, y)
                w = l50_exposure_weight.(frame.n_segments, kappa)
                blended = w .* y .+ (1 .- w) .* m
                return l50_position_z(frame, blended) .* keeper_mask(frame)
            end),
    ]
end

# ==============================================================================
# 4. FROM ADJUSTED RATINGS TO A MATCH COLUMN
# ==============================================================================

"""
    l50_xi_column(ds, adjusted, ordered_ids; min_rated) -> (column, available)

The starting-XI differential under an arbitrary adjusted-rating map. Mirrors
`Features.pxg_rapm_deltas` in every respect except that the rating is supplied rather
than read from the ridge, so a strategy can be swapped without touching the feature.
"""
function l50_xi_column(ds::L50_DATA.DataStore, adjusted::Dict{Int,Float64},
                       ordered_ids; min_rated::Int = 3)
    sums = Dict{Tuple{Int,Bool},Float64}()
    counts = Dict{Tuple{Int,Bool},Int}()
    for row in eachrow(ds.lineups)
        (ismissing(row.player_id) || ismissing(row.team_side)) && continue
        coalesce(row.is_substitute, false) && continue
        side = lowercase(String(row.team_side))
        side in ("home", "away") || continue
        value = get(adjusted, Int(row.player_id), 0.0)
        (isfinite(value) && value != 0.0) || continue
        key = (Int(row.match_id), side == "home")
        sums[key] = get(sums, key, 0.0) + value
        counts[key] = get(counts, key, 0) + 1
    end

    column = zeros(Float64, length(ordered_ids))
    available = zeros(Float64, length(ordered_ids))
    for (i, id) in enumerate(ordered_ids)
        mid = Int(id)
        h_n = get(counts, (mid, true), 0)
        a_n = get(counts, (mid, false), 0)
        (h_n >= min_rated && a_n >= min_rated) || continue
        delta = get(sums, (mid, true), 0.0) - get(sums, (mid, false), 0.0)
        isfinite(delta) || continue
        column[i] = delta
        available[i] = 1.0
    end
    # Standardise so strategies on wildly different scales are comparable.
    live = column[available .> 0]
    scale = length(live) >= 10 ? std(live) : 1.0
    (isfinite(scale) && scale > 1e-9) || (scale = 1.0)
    return column ./ scale, available
end
