# src/data/fetchers/schemas.jl

using Dates
using InlineStrings

const NameType = String63

const MATCHES_COLS_TYPES = Dict(
    :tournament_id => Int64,
    :season_id => Int64,
    :season => String7,
    :match_id => Int64,
    :tournament_slug => String15,
    :home_team => String31,
    :away_team => String31,
    :home_score => Float64,
    :away_score => Float64,
    :home_score_ht => Float64,
    :away_score_ht => Float64,
    :winner_code => Float64,
    :match_date => Date,
    :round => Float64,
    :has_xg => Bool,
    :has_stats => Bool,
    :match_hour => Int64,
    :match_dayofweek => Int64,
    :match_month => Int64
)

# BBC shot counts. Only the join keys are guaranteed non-missing; the four count columns stay
# `Union{Missing, Float64}` on purpose (per-match usability is a feature-layer decision).
const BBC_COLS_TYPES = Dict{Symbol, Type}(
    :match_id => Int64,
    :tournament_id => Int64
)

# ... (Include INCIDENTS_COLS_TYPES and ODDS_COLS_TYPES here) ...
