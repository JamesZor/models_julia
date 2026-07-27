# src/data/fetchers/sql/bbc_events.jl
#
# RAW BBC shot-bearing commentary events — one row per attempt, as opposed to `ds.bbc`, which is
# per-MATCH totals from `bbc.match_stats`. The two are deliberately separate domains: the funnel
# engine (DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel) depends on `ds.bbc` unchanged.
#
# Graduated from current_development/plus_minus_ratings/l00_pm_data.jl::fetch_pm_livetext. It feeds
# the plus-minus (RAPM) rating family in src/features/plus_minus/, which needs to place each shot
# on the match clock inside a personnel segment — something a per-match total cannot do.
#
# Coverage: Scottish tiers only (54-57, same as `bbc.match_meta`), and only from 23/24, when BBC
# live text starts. Every other segment resolves to an empty DataFrame, and the PM extractors emit
# zero ratings rather than erroring.
#
# WHY ONLY SHOT EVENTS: the four PM targets consume attempts (`y_shots`, `y_sot`, `y_xg`) or goals
# from `ds.incidents` (`y_goals`). Substitutions come from `ds.incidents` too — BBC's `substitution`
# rows carry no player column, only free text, and WP3 measured BBC shooter names resolving to a
# player_id only 93.2% of the time, so the name route was explicitly rejected. Restricting here
# keeps the domain at ~45k rows across all four tiers instead of ~500k.
#
# `text` is retained because it is the ONLY place the shot descriptor lives (zone / body part /
# set-piece context). It is parsed in src/features/plus_minus/shot_parser.jl.

const BBC_SHOT_EVENTS = ["goal", "attempt_missed", "attempt_saved", "attempt_blocked", "post",
                         "penalty_missed", "penalty_saved"]

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::BBCEventsData)
    # BBC_SHOT_EVENTS is a compile-time constant, so interpolating it is safe; the tournament ids
    # are bound as a parameter like every other fetcher.
    ev_in = join(("'" * e * "'" for e in BBC_SHOT_EVENTS), ", ")
    query = """
    SELECT lt.match_id,
           m.tournament_id,
           lt.time,
           lt.added_time,
           lt.event_type,
           -- Deterministic side mapping. Do NOT infer the side from the running score — that
           -- fails on goalless matches and on own goals.
           --
           -- THREE-WAY, NOT BOOLEAN. The obvious `(lt.team = mm.bbc_home_slug)` is WRONG: SQL
           -- returns FALSE (not NULL) whenever the slug matches neither side, so every unmatched
           -- slug is silently attributed to AWAY. That hit 7,073 of 45,201 shot rows (15.6%) in
           -- the research and REVERSED the measured home shot advantage, surfacing as a negative
           -- home-advantage coefficient in the ridge fit.
           --
           -- The cause is a slug variant: `dundee-fc` vs `dundee`, `clyde-fc` vs `clyde`,
           -- `queens-park-fc` vs `queens-park`. Normalising the trailing `-fc` resolves ALL 7,073
           -- (zero left unmatched), so these are recovered rather than dropped.
           CASE
             WHEN regexp_replace(lt.team, '-fc\$', '') =
                  regexp_replace(mm.bbc_home_slug, '-fc\$', '') THEN true
             WHEN regexp_replace(lt.team, '-fc\$', '') =
                  regexp_replace(mm.bbc_away_slug, '-fc\$', '') THEN false
             ELSE NULL
           END AS is_home_event,
           lt.player,
           lt.text
    FROM bbc.live_text lt
    JOIN sofascore.matches m ON m.match_id = lt.match_id
    JOIN bbc.match_meta mm   ON mm.match_id = lt.match_id
    WHERE m.tournament_id = ANY(\$1)
      AND lt.event_type IN ($ev_in)
    ORDER BY lt.match_id, lt.post_index
    """
    try
        return DataFrame(LibPQ.execute(conn, query, [t_ids]))
    catch e
        @warn "Failed to fetch BBCEventsData: $(e)"
        return DataFrame()
    end
end

const BBC_EVENTS_SCHEMA = Dict{Symbol, Type}(
    :match_id      => Int64,
    :tournament_id => Int64,
    :time          => Union{Missing, Int32},
    :added_time    => Union{Missing, Int32},
    :event_type    => InlineStrings.String31,
    :is_home_event => Union{Missing, Bool},
    :player        => Union{Missing, String},
    :text          => Union{Missing, String},
)

function process_data(df::DataFrame, ::BBCEventsData)
    apply_schema!(df, BBC_EVENTS_SCHEMA)
    return df
end

function validate_data(df::DataFrame, ::BBCEventsData)
    nrow(df) == 0 && return true

    # The `-fc` normalisation above is supposed to leave NOTHING unattributed beyond the ~2.4% of
    # shots whose BBC slug matches neither side for other reasons (measured in WP3). A jump here
    # means a slug-format drift on a re-scrape, and it silently biases every shot-based PM target.
    unattributed = count(ismissing, df.is_home_event) / nrow(df)
    if unattributed > 0.05
        @warn "BBCEventsData: $(round(100 * unattributed, digits=1))% of shots have no side " *
              "(expected ~2.4%) — check the bbc_home_slug / bbc_away_slug formats"
        return false
    end
    return true
end
