# ==============================================================================
# l92 — LOADER: cross-league proxy-vs-official-xG validation data and regimes
# ==============================================================================
#
# Definitions only, no execution. Paired with r92.
#
# WHY THIS EXISTS. The pxG feature is DEPLOYED on Scottish tiers 56/57, which have
# no official xG at all — that absence is the entire reason the proxy was built.
# So it cannot be validated where it is used. It CAN be validated on the six tiers
# that carry both BBC live text and SofaScore xG:
#
#   1  England Premier League        1703 matches with both
#   2  England Championship          1660
#   3  England League One            2256
#   84 England League Two            2232
#   54 Scottish Premiership           218
#   55 Scottish Championship          553
#                                    ----
#                                    8622 matches = 17244 team-match observations
#
# THE QUESTION THAT MATTERS IS TRANSFER, NOT FIT. A cell table fitted and scored on
# the same tier will always look good. The deployed table is fitted on 56/57's own
# shots, so the honest test is: fit where we deploy, score where we can measure.
# `l92_regimes` builds exactly that comparison, with pooled and leave-one-tier-out
# as the upper and middle references.
#
# CREDENTIALS. `BF_DB_URL` only; nothing is committed here. The pull is cached to a
# `.jls` beside this file so re-runs cost nothing and the numbers in the report can
# be reproduced without the database.
# ==============================================================================

using DataFrames
using Dates
using LibPQ
using Serialization
using Statistics

const L92_TOURNAMENTS = Dict{Int,String}(
    1  => "ENG Premier League",
    2  => "ENG Championship",
    3  => "ENG League One",
    84 => "ENG League Two",
    54 => "SCO Premiership",
    55 => "SCO Championship",
    56 => "SCO League One",
    57 => "SCO League Two",
)

# Ordered so every table reads top tier down, England then Scotland.
const L92_TIER_ORDER = [1, 2, 3, 84, 54, 55, 56, 57]

const L92_CACHE = joinpath(@__DIR__, "l92_pxg_validation_pull.jls")

# The shot event types the src fetcher considers attempts. Kept in lockstep with
# src/Data/fetchers/sql/bbc_events.jl::BBC_SHOT_EVENTS — this pull must see exactly
# what the production feature sees, or the validation measures a different model.
const L92_SHOT_EVENTS = ["goal", "attempt_missed", "attempt_saved", "attempt_blocked",
                         "post", "penalty_missed", "penalty_saved"]

# ==============================================================================
# 1. THE PULL
# ==============================================================================

"""
    l92_fetch(; force = false) -> NamedTuple(events, official, matches)

Three frames for the eight tiers, cached to `L92_CACHE`.

  * `events`   — one row per attempt, with the same three-way `-fc`-normalised side
                 attribution the production fetcher uses. Getting this wrong silently
                 assigns every unmatched slug to AWAY and reverses home advantage;
                 see the note in src/Data/fetchers/sql/bbc_events.jl.
  * `official` — SofaScore `expectedGoals`, period ALL, both sides present.
  * `matches`  — tier, season, kickoff and the scoreline.
"""
function l92_fetch(; force::Bool = false)
    if !force && isfile(L92_CACHE)
        @info "l92: loading cached pull from $(basename(L92_CACHE))"
        return deserialize(L92_CACHE)
    end

    url = get(ENV, "BF_DB_URL") do
        error("BF_DB_URL is not set. Export it (or source .env) before running r92:\n" *
              "  export BF_DB_URL=\"postgresql://<user>:<password>@<host>:5433/betdb\"")
    end
    tiers = "{" * join(L92_TIER_ORDER, ",") * "}"
    events_in = join(("'" * e * "'" for e in L92_SHOT_EVENTS), ", ")

    @info "l92: pulling from the database (this is the slow path; the result is cached)"
    conn = LibPQ.Connection(url)
    try
        events = DataFrame(LibPQ.execute(conn, """
            SELECT lt.match_id,
                   m.tournament_id,
                   lt.time,
                   lt.added_time,
                   lt.event_type,
                   CASE
                     WHEN regexp_replace(lt.team, '-fc\$', '') =
                          regexp_replace(mm.bbc_home_slug, '-fc\$', '') THEN true
                     WHEN regexp_replace(lt.team, '-fc\$', '') =
                          regexp_replace(mm.bbc_away_slug, '-fc\$', '') THEN false
                     ELSE NULL
                   END AS is_home_event,
                   lt.text
            FROM bbc.live_text lt
            JOIN sofascore.matches m ON m.match_id = lt.match_id
            JOIN bbc.match_meta mm   ON mm.match_id = lt.match_id
            WHERE m.tournament_id = ANY(\$1)
              AND lt.event_type IN ($events_in)
            ORDER BY lt.match_id, lt.post_index
            """, [tiers]))

        # DISTINCT ON because `match_statistics` carries repeat scrapes: 10,324 rows for
        # 8,394 distinct matches. Ordering by the pair sum DESC makes a genuine
        # measurement win over a zero-filled duplicate of the same match.
        official = DataFrame(LibPQ.execute(conn, """
            SELECT DISTINCT ON (s.match_id)
                   s.match_id,
                   s.home_value::float8 AS xg_home,
                   s.away_value::float8 AS xg_away
            FROM sofascore.match_statistics s
            JOIN sofascore.matches m ON m.match_id = s.match_id
            WHERE m.tournament_id = ANY(\$1)
              AND s.stat_key = 'expectedGoals' AND s.period = 'ALL'
              AND s.home_value IS NOT NULL AND s.away_value IS NOT NULL
            ORDER BY s.match_id, (s.home_value + s.away_value) DESC
            """, [tiers]))

        matches = DataFrame(LibPQ.execute(conn, """
            SELECT m.match_id, m.tournament_id, se.name AS season,
                   m.start_timestamp, m.home_score, m.away_score
            FROM sofascore.matches m
            LEFT JOIN sofascore.seasons se
                   ON se.season_id = m.season_id AND se.tournament_id = m.tournament_id
            WHERE m.tournament_id = ANY(\$1)
            """, [tiers]))

        pull = (events = events, official = official, matches = matches,
                pulled_at = now())
        serialize(L92_CACHE, pull)
        @info "l92: cached $(nrow(events)) attempts, $(nrow(official)) official-xG matches"
        return pull
    finally
        close(conn)
    end
end

# ==============================================================================
# 2. THE SHOT TABLE
# ==============================================================================

"""
    l92_shot_table(pull) -> DataFrame

Parse every attempt into the production descriptor set (`zone`, `body_part`,
`context`, `is_penalty`, `parsed`) using `Features.parse_shot` — the SAME parser the
deployed feature uses, not a re-implementation. Attempts whose side could not be
resolved are dropped, and the drop rate is reported by the caller.
"""
function l92_shot_table(pull)
    events = pull.events
    nrow(events) == 0 && return DataFrame()

    parsed = EDA_FEATURES.parse_shot.(String.(events.event_type), events.text)
    shots = DataFrame(
        match_id      = Int.(events.match_id),
        tournament_id = Int.(events.tournament_id),
        event_type    = String.(events.event_type),
        is_home       = events.is_home_event,
        zone          = [p.zone       for p in parsed],
        body_part     = [p.body_part  for p in parsed],
        context       = [p.context    for p in parsed],
        is_penalty    = [p.is_penalty for p in parsed],
        parsed        = [p.parsed     for p in parsed],
    )
    shots.is_goal      = in.(shots.event_type, Ref(EDA_FEATURES.PM_GOAL_EVENTS))
    shots.is_on_target = in.(shots.event_type, Ref(EDA_FEATURES.PM_ON_TARGET_EVENTS))
    return shots
end

# ==============================================================================
# 3. TEAM-MATCH AGGREGATION
# ==============================================================================

"""
    l92_team_frame(shots, pull; model, label) -> DataFrame

Two rows per match — one per side — carrying the proxy pxG under `model`, the shot
count, and the official xG. This is the unit every metric in r92 is computed on.

The naive `shots * league mean xG per shot` control is carried alongside, because
the question "does parsing zone / body part / context buy anything?" cannot be
answered without it: if the control matches the proxy, the parser is decoration.
"""
function l92_team_frame(shots::DataFrame, pull; model, label::AbstractString)
    nrow(shots) == 0 && return DataFrame()
    usable = shots[.!ismissing.(shots.is_home), :]
    predicted = EDA_FEATURES.predict_xg(model, usable)
    per_shot = isempty(predicted) ? 0.0 : mean(predicted)

    totals = Dict{Tuple{Int,Bool},Tuple{Float64,Int,Int}}()   # (pxg, shots, on target)
    for (i, r) in enumerate(eachrow(usable))
        key = (Int(r.match_id), r.is_home === true)
        pxg, n, sot = get(totals, key, (0.0, 0, 0))
        totals[key] = (pxg + predicted[i], n + 1, sot + (r.is_on_target ? 1 : 0))
    end

    official = Dict{Int,Tuple{Float64,Float64}}(
        Int(r.match_id) => (Float64(r.xg_home), Float64(r.xg_away))
        for r in eachrow(pull.official))
    meta = Dict{Int,Any}(Int(r.match_id) => r for r in eachrow(pull.matches))

    rows = NamedTuple[]
    for ((match_id, is_home), (pxg, n_shots, n_sot)) in totals
        haskey(official, match_id) || continue
        haskey(meta, match_id) || continue
        m = meta[match_id]
        xg_h, xg_a = official[match_id]
        push!(rows, (
            match_id = match_id,
            tournament_id = Int(m.tournament_id),
            tier = get(L92_TOURNAMENTS, Int(m.tournament_id), "?"),
            season = String(coalesce(m.season, "?")),
            is_home = is_home,
            regime = String(label),
            pxg = pxg,
            pxg_shot_count = n_shots * per_shot,
            shots = n_shots,
            sot = n_sot,
            official_xg = is_home ? xg_h : xg_a,
            # ⚠ ZERO-FILL. A match whose official xG is 0.000 on BOTH sides is not a
            # measurement — it is a placeholder row. It cannot be a real observation
            # here because every match in this frame HAS live-text attempts, and an
            # attempt cannot carry zero xG. English League One and League Two are
            # 59-60% zero-filled this way; scoring against those rows drags their
            # apparent correlation from ~0.85 to ~0.30 and invents a +0.85 xG bias.
            # Flagged, not dropped, so the runner can report the exclusion.
            official_live = !(xg_h == 0.0 && xg_a == 0.0),
            goals = eda_num(is_home ? m.home_score : m.away_score),
        ))
    end
    return sort!(DataFrame(rows), [:match_id, :is_home])
end

"""
    l92_match_frame(team_frame) -> DataFrame

One row per match: the two sides' totals, the match total, and the SUPREMACY
difference. The difference is the decision-relevant quantity — it is what the
`PxGCovariate` supremacy role is built from — and a proxy can track the total well
while tracking the difference badly.
"""
function l92_match_frame(team_frame::DataFrame)
    nrow(team_frame) == 0 && return DataFrame()
    rows = NamedTuple[]
    for g in groupby(team_frame, :match_id)
        nrow(g) == 2 || continue
        home = g[g.is_home .== true, :]
        away = g[g.is_home .== false, :]
        (nrow(home) == 1 && nrow(away) == 1) || continue
        h, a = home[1, :], away[1, :]
        push!(rows, (
            match_id = h.match_id,
            tournament_id = h.tournament_id,
            tier = h.tier,
            season = h.season,
            regime = h.regime,
            official_live = h.official_live && a.official_live,
            pxg_total = h.pxg + a.pxg,
            pxg_diff = h.pxg - a.pxg,
            official_total = h.official_xg + a.official_xg,
            official_diff = h.official_xg - a.official_xg,
            goal_diff = h.goals - a.goals,
            goal_total = h.goals + a.goals,
        ))
    end
    return DataFrame(rows)
end

# ==============================================================================
# 4. FITTING REGIMES
# ==============================================================================

"""
    l92_fit_regimes(shots; k) -> Vector{NamedTuple(label, model, scope)}

Three ways to fit the zonal cell table, in increasing honesty:

  * `pooled`      — fitted on every tier. The optimistic ceiling; each tier is partly
                    scored on a table that saw its own shots.
  * `loto/<tier>` — leave-one-tier-out. Fitted on the other seven, scored on the held
                    out tier. The standard transfer measure.
  * `sco_lower`   — fitted on tiers 56/57 ONLY. This is the table the deployed
                    ScottishLower feature actually builds, so this row is the one that
                    licenses (or refuses) the production claim. Note 56/57 have no
                    official xG, so this regime can only ever be SCORED elsewhere.

`k` is the empirical-Bayes pseudo-count, passed straight through to `fit_shot_xg`.
"""
function l92_fit_regimes(shots::DataFrame; k::Float64 = 25.0)
    regimes = NamedTuple[]
    push!(regimes, (label = "pooled", scope = :all,
                    model = EDA_FEATURES.fit_shot_xg(shots; k = k)))

    lower = shots[in.(shots.tournament_id, Ref(Set([56, 57]))), :]
    if nrow(lower) > 0
        push!(regimes, (label = "sco_lower", scope = :deployed,
                        model = EDA_FEATURES.fit_shot_xg(lower; k = k)))
    end

    for tier in L92_TIER_ORDER
        rest = shots[shots.tournament_id .!= tier, :]
        nrow(rest) > 0 || continue
        push!(regimes, (label = "loto/$(tier)", scope = tier,
                        model = EDA_FEATURES.fit_shot_xg(rest; k = k)))
    end
    return regimes
end

"""
    l92_live(frame) -> DataFrame

Drop the zero-filled official rows. Every agreement metric in r92 runs through this;
scoring against a placeholder is not a measurement of the proxy.
"""
l92_live(frame::DataFrame) =
    nrow(frame) == 0 ? frame : frame[frame.official_live, :]

"""
    l92_zero_fill_report(frame) -> DataFrame

Per tier: how much of the official reference is a zero-filled placeholder. This is a
property of the SofaScore scrape, not of the proxy, and it must be reported before any
agreement number is believed.
"""
function l92_zero_fill_report(frame::DataFrame)
    nrow(frame) == 0 && return DataFrame()
    rows = NamedTuple[]
    for tier in L92_TIER_ORDER
        sub = frame[frame.tournament_id .== tier, :]
        nrow(sub) == 0 && continue
        push!(rows, (tier = get(L92_TOURNAMENTS, tier, "?"),
                     n = nrow(sub),
                     live = count(sub.official_live),
                     zero_filled = count(.!sub.official_live),
                     zero_share = count(.!sub.official_live) / nrow(sub)))
    end
    return DataFrame(rows)
end

"""
    l92_cell_table(model; top) -> DataFrame

The fitted cell table, most-probable first: what the model believes a shot from each
(zone, body part, context) is worth.
"""
function l92_cell_table(model; top::Int = 30)
    rows = [(zone = String(k[1]), body_part = String(k[2]), context = String(k[3]), xg = v)
            for (k, v) in model.cells]
    isempty(rows) && return DataFrame()
    frame = sort!(DataFrame(rows), :xg, rev = true)
    return first(frame, min(top, nrow(frame)))
end
