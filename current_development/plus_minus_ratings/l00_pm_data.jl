# current_development/plus_minus_ratings/l00_pm_data.jl
#
# LOADER (temporary module). WP1 — raw data pulls for the RAPM stream.
#
# Three tables, one `.jls` cache each, all four Scottish tiers (54 Prem, 55 Champ,
# 56 League One, 57 League Two) in a single pull:
#
#   1. lineups   — who was on the teamsheet, position, minutes, (SofaScore rating), player xG
#   2. incidents — WHEN players entered/left and when goals/cards happened, WITH PLAYER IDS
#   3. live_text — BBC Opta-style commentary rows (shot events + their descriptors)
#
# WHY ITS OWN SQL (do not "just use the DataStore"):
#   * `src/Data/fetchers/sql/incidents.jl` extracts only the *slug* out of the incident jsonb
#     (`data->'playerIn'->>'slug'`). Segment reconstruction needs the *id*
#     (`data->'playerIn'->>'id'`), which joins to `match_player_lineups.player_id`. Slugs are
#     not a safe join key across the name variants in these leagues.
#   * `DataTournemantSegment` has no singleton covering 54/55, and adding one to `src/` is
#     out of scope for this stream (see the plan's "out of scope" section).
#   * The DataStore pipeline hauls odds/betfair/statistics we do not need here.
#
# DATA HAZARDS BAKED INTO THE QUERIES (verified 2026-07-23, see NOTES.md §"Data facts"):
#   * `minutes_played` is IDENTICALLY 0 for seasons 20/21–22/23 and NULL for most of 25/26.
#     It is a CROSS-CHECK, never the base for segments. Incidents are the base.
#   * tier 56 has incident holes (23/24: 71/180 matches, 25/26: 16/175). BBC `live_text` is
#     the fallback there — but its `substitution` rows carry NO player column, only free text.
#   * `incidentClass` distinguishes yellow / red / yellowRed. A `yellowRed` IS a dismissal.
#
# Downstream: `r00_data_qa.jl` (the WP1 gate), then `l01_segments.jl`.

using DataFrames
using Dates
using LibPQ
using Serialization

const PM_TIERS = [54, 55, 56, 57]

# Upper tiers carry the SofaScore rating we validate against; lower tiers are the target.
pm_tier_group(t::Integer) = t in (54, 55) ? :upper : :lower

# ==========================================
# 0. CONNECTION
# ==========================================
"""
    pm_connect() -> LibPQ.Connection

Opens a connection from `BF_DB_URL`. Off the home network the betdb MCP is unreachable, so
this is expected to run on the server (`/root/BayesianFootball`) via the kaimon REPL, where
the database is local.
"""
function pm_connect()
    url = get(ENV, "BF_DB_URL") do
        error("BF_DB_URL is not set. Export it before loading data, e.g.\n" *
              "  export BF_DB_URL=\"postgresql://admin:<password>@<host>:5433/betdb\"")
    end
    return LibPQ.Connection(url)
end

# ==========================================
# 1. LINEUPS
# ==========================================
"""
    fetch_pm_lineups(conn; tournaments) -> DataFrame

One row per (match, player) teamsheet entry. `is_starter` is `NOT substitute` — note that in
SofaScore `substitute=true` means "named on the bench", which includes UNUSED substitutes
(they show `minutes_played = 0`), so it is not the same thing as "came on".

`rating` is null throughout tiers 56/57 by construction — that null is the entire reason this
stream exists.
"""
function fetch_pm_lineups(conn; tournaments::Vector{Int} = PM_TIERS)
    t_in = join(tournaments, ",")
    sql = """
    SELECT m.match_id,
           m.tournament_id,
           s.year                AS season,
           m.start_timestamp,
           m.home_score, m.away_score,
           m.injury_time1, m.injury_time2,
           m.home_team, m.away_team,
           l.player_id,
           l.player_name,
           l.team_id,
           l.is_home_team,
           l.position,
           l.substitute          AS is_bench,
           NOT l.substitute      AS is_starter,
           l.minutes_played,
           l.rating,
           l.expected_goals,
           l.expected_assists,
           l.goals
    FROM sofascore.matches m
    JOIN sofascore.seasons s
      ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    JOIN sofascore.match_player_lineups l ON l.match_id = m.match_id
    WHERE m.tournament_id IN ($t_in)
    ORDER BY m.start_timestamp, m.match_id, l.is_home_team DESC, l.substitute, l.player_id
    """
    return DataFrame(LibPQ.execute(conn, sql))
end

# ==========================================
# 2. INCIDENTS  (player IDs pulled out of the jsonb)
# ==========================================
"""
    fetch_pm_incidents(conn; tournaments) -> DataFrame

Substitutions, goals and cards with **player ids**, which is what the repo's own fetcher does
not give us. Columns:

  * substitution → `player_in_id`, `player_out_id`
  * goal         → `player_id` (scorer), `assist_id`, `incident_class` ∈ {regular, penalty,
                   ownGoal, goal}. NB an `ownGoal` credits the *conceding* side — handled in
                   `l03_targets.jl`, not here.
  * card         → `player_id`, `incident_class` ∈ {yellow, red, yellowRed}

`is_home` is the side the incident belongs to. `time` is the match minute; `added_time` is the
stoppage-time offset (null outside stoppage time).
"""
function fetch_pm_incidents(conn; tournaments::Vector{Int} = PM_TIERS)
    t_in = join(tournaments, ",")
    sql = """
    SELECT m.match_id,
           m.tournament_id,
           s.year                                        AS season,
           i.id                                          AS incident_id,
           i.incident_type,
           i.time,
           i.added_time,
           i.is_home,
           i.data ->> 'incidentClass'                    AS incident_class,
           (i.data -> 'player'    ->> 'id')::int         AS player_id,
           (i.data -> 'playerIn'  ->> 'id')::int         AS player_in_id,
           (i.data -> 'playerOut' ->> 'id')::int         AS player_out_id,
           (i.data -> 'assist1'   ->> 'id')::int         AS assist_id,
           (i.data ->> 'rescinded')::boolean             AS rescinded
    FROM sofascore.match_incidents i
    JOIN sofascore.matches m ON m.match_id = i.match_id
    JOIN sofascore.seasons s
      ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    WHERE m.tournament_id IN ($t_in)
      AND i.incident_type IN ('substitution', 'goal', 'card')
    ORDER BY m.match_id, i.time NULLS FIRST, i.id
    """
    return DataFrame(LibPQ.execute(conn, sql))
end

# ==========================================
# 3. BBC LIVE TEXT
# ==========================================
# Shot-bearing event types. `post` = hit the woodwork (a real attempt, no goal).
const PM_SHOT_EVENTS = ["goal", "attempt_missed", "attempt_saved", "attempt_blocked", "post",
                        "penalty_missed", "penalty_saved"]
# On target = keeper had to deal with it, or it went in.
const PM_SOT_EVENTS  = ["goal", "attempt_saved", "penalty_saved"]

"""
    fetch_pm_livetext(conn; tournaments, event_types=nothing) -> DataFrame

BBC commentary rows. `text` is retained because it is the ONLY place several things live:
the shot descriptor (zone / body part / set-piece context, parsed in `l02_shot_parser.jl`) and
the substitution player names (the `player` column is null on `substitution` rows).

Pass `event_types` to restrict; default is everything, since WP1's QA needs the full picture.
"""
function fetch_pm_livetext(conn; tournaments::Vector{Int} = PM_TIERS,
                           event_types::Union{Nothing, Vector{String}} = nothing)
    t_in = join(tournaments, ",")
    ev_clause = if event_types === nothing
        ""
    else
        "AND lt.event_type IN (" * join(("'" * e * "'" for e in event_types), ",") * ")"
    end
    sql = """
    SELECT lt.match_id,
           m.tournament_id,
           s.year         AS season,
           lt.post_index,
           lt.time,
           lt.added_time,
           lt.event_type,
           lt.team,
           lt.team_bbc,
           -- Deterministic side mapping. Do NOT infer the side from the running score — that
           -- fails on goalless matches and on own goals.
           --
           -- THREE-WAY, NOT BOOLEAN. The obvious `(lt.team = mm.bbc_home_slug)` is WRONG: SQL
           -- returns FALSE (not NULL) whenever the slug matches neither side, so every
           -- unmatched slug is silently attributed to AWAY. That hit **7,073 of 45,201 shot
           -- rows (15.6%)** and reversed the measured home shot advantage — our home mean came
           -- out 9.21 vs BBC's ground-truth 11.16, which surfaced as a NEGATIVE home-advantage
           -- coefficient in the first WP5 run.
           --
           -- The cause is a slug variant: `dundee-fc` vs `dundee`, `clyde-fc` vs `clyde`,
           -- `queens-park-fc` vs `queens-park`. Normalising the trailing `-fc` resolves ALL
           -- 7,073 (zero left unmatched), so these are recovered rather than dropped.
           CASE
             WHEN regexp_replace(lt.team, '-fc\$', '') =
                  regexp_replace(mm.bbc_home_slug, '-fc\$', '') THEN true
             WHEN regexp_replace(lt.team, '-fc\$', '') =
                  regexp_replace(mm.bbc_away_slug, '-fc\$', '') THEN false
             ELSE NULL
           END AS is_home_event,
           lt.player,
           lt.area,
           lt.home_score,
           lt.away_score,
           lt.text
    FROM bbc.live_text lt
    JOIN sofascore.matches m ON m.match_id = lt.match_id
    JOIN sofascore.seasons s
      ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    JOIN bbc.match_meta mm ON mm.match_id = lt.match_id
    WHERE m.tournament_id IN ($t_in) $ev_clause
    ORDER BY lt.match_id, lt.post_index
    """
    return DataFrame(LibPQ.execute(conn, sql))
end

# ==========================================
# 4. CACHE + EAGER LOAD
# ==========================================
const PM_CACHE_DIR = @__DIR__

pm_cache_path(name::AbstractString) = joinpath(PM_CACHE_DIR, "pm_$(name).jls")

const PM_LINEUPS   = Ref{DataFrame}()
const PM_INCIDENTS = Ref{DataFrame}()
const PM_LIVETEXT  = Ref{DataFrame}()

"""
    ensure_pm_data!(; tournaments=PM_TIERS, refresh=false)

Populate `PM_LINEUPS`, `PM_INCIDENTS`, `PM_LIVETEXT` from disk cache, falling back to a single
database round-trip. Call once at the top of a runner — the DB must never be touched from
inside a per-fold code path.
"""
function ensure_pm_data!(; tournaments::Vector{Int} = PM_TIERS, refresh::Bool = false)
    specs = (("lineups",   PM_LINEUPS,   fetch_pm_lineups),
             ("incidents", PM_INCIDENTS, fetch_pm_incidents),
             ("livetext",  PM_LIVETEXT,  fetch_pm_livetext))

    missing_specs = [sp for sp in specs
                     if refresh || !(isassigned(sp[2]) || isfile(pm_cache_path(sp[1])))]

    conn = isempty(missing_specs) ? nothing : pm_connect()
    try
        for (name, ref, fetcher) in specs
            isassigned(ref) && !refresh && continue
            path = pm_cache_path(name)
            if !refresh && isfile(path)
                @info "[pm_data] loading $name from cache"
                ref[] = deserialize(path)
            else
                @info "[pm_data] fetching $name from betdb (tiers $(tournaments))"
                df = fetcher(conn; tournaments = tournaments)
                serialize(path, df)
                ref[] = df
                @info "[pm_data] $name: $(nrow(df)) rows cached → $(basename(path))"
            end
        end
    finally
        conn === nothing || close(conn)
    end
    return (lineups = PM_LINEUPS[], incidents = PM_INCIDENTS[], livetext = PM_LIVETEXT[])
end

# ==========================================
# 5. SMALL SHARED HELPERS
# ==========================================
"""
    pm_match_meta(lineups) -> DataFrame

One row per match: tier, season, kickoff, final score. Derived from the lineups pull so that
every downstream file agrees on the match universe.
"""
function pm_match_meta(lineups::DataFrame = PM_LINEUPS[])
    return unique(select(lineups, :match_id, :tournament_id, :season, :start_timestamp,
                                  :home_score, :away_score, :injury_time1, :injury_time2))
end

"""
    pm_club_map(; lineups) -> DataFrame(player_id, club)

Each player's club: the modal value of `is_home_team ? home_team : away_team` over his
appearances.

**DO NOT USE `match_player_lineups.team_id` FOR THIS.** It is not a stable club identifier —
there are **626 distinct `team_id` values across tiers 54–57 for ~44 actual clubs**, and a single
id (e.g. 2351, mostly Rangers) carries dozens of different club names across its rows. Grouping
~880 players by their modal `team_id` produced **451 groups**, i.e. an average group size under
two, which inflates any "share of variance explained by team" statistic almost to the ceiling
purely through degrees of freedom. That error overstated the team-loading figure in the first
WP7 pass (0.38–0.54 reported, 0.21–0.39 actual). The name-derived label gives the correct **44**.
"""
function pm_club_map(; lineups::DataFrame = PM_LINEUPS[])
    d = Dict{Int, Dict{String, Int}}()
    for r in eachrow(lineups)
        ismissing(r.player_id) && continue
        nm = r.is_home_team === true ? r.home_team : r.away_team
        ismissing(nm) && continue
        c = get!(d, Int(r.player_id), Dict{String, Int}())
        s = String(nm); c[s] = get(c, s, 0) + 1
    end
    ids = collect(keys(d))
    return DataFrame(player_id = ids,
                     club = [argmax(d[i]) for i in ids])
end

"""
    pm_clean_position(p) -> String

Collapse SofaScore's position strings to the G/D/M/F buckets the rating vectors use.

DELIBERATELY DIFFERENT from `src/features/extractors/player_extractors.jl::clean_pos`, which
silently defaults missing to `"M"`. That default poisoned the `position_aware_ratings` EDA
(NOTES.md, 2026-06-27), so here an unknown position stays `"U"` and is counted, not hidden.
"""
function pm_clean_position(p)
    (ismissing(p) || p == "") && return "U"
    s = uppercase(String(p))
    startswith(s, "G") && return "G"
    startswith(s, "D") && return "D"
    startswith(s, "M") && return "M"
    (startswith(s, "F") || startswith(s, "A")) && return "F"
    return "U"
end

"""
    pm_full_time(row_time, added_time) -> Float64

Match minute including stoppage time. Both BBC and SofaScore report `time` as the minute the
clock had reached, with `added_time` separate; a 90+3 event is `time=90, added_time=3`.
"""
pm_full_time(t, at) = (ismissing(t) ? NaN : Float64(t)) +
                      (ismissing(at) ? 0.0 : Float64(at))
