#=
l01_roleneutral_helpers.jl — Option-1 EDA helpers: out-of-position effect on ROLE-NEUTRAL output.

Why: Gate 3 (l00) tested the off-modal effect on the SofaScore *rating* and got ~0 — because the
rating is computed conditional on the position played, so it is blind to out-of-position penalty by
construction (deep-research batch-20260627-130505-1a266db9; see RESEARCH_role_aware_ideas.md). Here we
re-test on OBJECTIVE per-player outputs (xG, xA, bigChanceCreated, touchesInOppBox, shots, defensive
actions …) that a role-conditioned rating cannot hide.

KEY TRICK — role standardisation. A striker played at CB mechanically posts low xG; that's a baseline
shift, not a skill drop. So we z-score each target WITHIN position first (z = (x − mean_pos)/sd_pos),
then run the within-player FE regression `z ~ off_modal + controls`. The off_modal coef then reads:
"when out of his modal position, does the player perform BELOW that role's norm relative to himself?"
Negative & |t|≥2 = a genuine role-fit penalty the single rating misses.

Builds on l00 (prepare_starter_lineups, add_modal_position!, add_opponent_strength!, demean!).

    include("current_development/position_aware_ratings/l00_position_helpers.jl")
    include("current_development/position_aware_ratings/l01_roleneutral_helpers.jl")
=#

using BayesianFootball
using DataFrames
using Statistics
using GLM

const Data = BayesianFootball.Data

# columns that are NOT role-neutral output targets (ids, meta, the rating itself)
const TARGET_BLOCKLIST = Set(Symbol.([
    "match_id", "player_id", "season_id", "tournament_id", "shirt_number",
    "minutes_played", "rating", "is_substitute", "is_captain"]))

"""
    candidate_target_columns(ds; min_coverage=0.3) -> Vector{Symbol}

Discover numeric per-player stat columns in `ds.lineups` (schema differs by league: base xG/xA + an
unstacked SofaScore JSON stat block) with non-missing coverage ≥ `min_coverage`, minus the blocklist.
Returns the real column names so we don't have to guess JSON key spellings.
"""
function candidate_target_columns(ds::Data.DataStore; min_coverage::Float64=0.3)
    lu = ds.lineups
    out = Symbol[]
    for c in names(lu)
        sym = Symbol(c)
        sym in TARGET_BLOCKLIST && continue
        col = lu[!, c]
        (nonmissingtype(eltype(col)) <: Real) || continue   # numeric only (skips strings/dates)
        nonmissingtype(eltype(col)) === Bool && continue     # skip flag columns
        mean((!ismissing).(col)) >= min_coverage || continue
        push!(out, sym)
    end
    return out
end

target_coverage(ds::Data.DataStore, t::Symbol) = round(100 * mean((!ismissing).(ds.lineups[!, t])), digits=1)

"""
    attach_targets(df, ds, targets) -> DataFrame

Left-join the chosen target columns from `ds.lineups` onto the prepared starter frame by
(match_id, player_id). One lineup row per player-match, so the join is 1:1.
"""
function attach_targets(df::DataFrame, ds::Data.DataStore, targets::Vector{Symbol})
    t = select(ds.lineups, unique([:match_id, :player_id; targets])...)
    return leftjoin(df, t, on = [:match_id, :player_id])
end

"""
    add_role_zscores!(df, targets; min_pos_n=5) -> DataFrame

For each target add `z_<target>` = the value standardised WITHIN the player's position over real-position
rows (mean 0, sd 1 per position). Removes the mechanical position-baseline so the regression isolates
role-fit. Rows with a missing target or a too-thin position get `missing` z (excluded downstream).
"""
function add_role_zscores!(df::DataFrame, targets::Vector{Symbol}; min_pos_n::Int=5)
    for t in targets
        stats = Dict{String,Tuple{Float64,Float64}}()
        for sub in groupby(df[df.pos_is_real, :], :pos)
            v = collect(skipmissing(sub[!, t]))
            length(v) >= min_pos_n || continue
            s = std(v)
            stats[String(first(sub.pos))] = (mean(v), s > 0 ? s : 1.0)
        end
        z = Vector{Union{Missing,Float64}}(missing, nrow(df))
        for i in 1:nrow(df)
            (df.pos_is_real[i] && !ismissing(df[i, t]) && haskey(stats, String(df.pos[i]))) || continue
            (m, s) = stats[String(df.pos[i])]
            z[i] = (Float64(df[i, t]) - m) / s
        end
        df[!, Symbol("z_", t)] = z
    end
    return df
end

"""
    role_fit_regression(df, zcol) -> NamedTuple | nothing

Within-player FE (group-demean) regression `zcol ~ off_modal + is_home + minutes + opp_strength`.
Returns the off_modal effect (coef in role-sd units, t, p, n). Negative & |t|≥2 ⇒ players genuinely
underperform the role's norm when out of position — a signal the single rating misses.
"""
function role_fit_regression(df::DataFrame, zcol::Symbol)
    mask = df.pos_is_real .& (.!ismissing).(df.is_off_modal) .&
           (.!ismissing).(df[!, zcol]) .& (!isnan).(coalesce.(df.opp_strength, NaN))
    d = df[mask, :]
    nrow(d) < 50 && return nothing

    dd = DataFrame(
        player_id = d.player_id,
        y         = Float64.(d[!, zcol]),
        off_modal = Float64.(coalesce.(d.is_off_modal, false)),
        is_home   = Float64.(d.team_side .== "home"),
        minutes   = Float64.(coalesce.(d.minutes_played, 0.0)),
        opp_str   = Float64.(d.opp_strength),
    )
    for c in (:y, :off_modal, :is_home, :minutes, :opp_str); demean!(dd, c); end
    m  = lm(@formula(y ~ 0 + off_modal + is_home + minutes + opp_str), dd)
    ct = coeftable(m)
    r  = findfirst(==("off_modal"), ct.rownms)
    (n = nrow(d), coef = round(ct.cols[1][r], digits=4), t = round(ct.cols[3][r], digits=2),
     p = round(ct.cols[4][r], digits=4))
end
