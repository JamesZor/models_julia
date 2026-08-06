# src/MatchDay/implementations/sources.jl
#
# Fixture sources, identity resolvers, lineup sources.

export SofaScoreEvents, ExplicitFixtures, MatchMetaCrosswalk, ResolverChain,
       ProvisionalDB, LastHistorical, JsonPin, SourceChain

# ===================================================================
# Fixture sources
# ===================================================================

"""
    SofaScoreEvents(; horizon = Hour(36))

Unstarted fixtures from `sofascore.events` whose kick-off falls in `[as_of, as_of + horizon)`.

A horizon rather than a calendar day, deliberately: the prototype used
`start_timestamp >= EXTRACT(EPOCH FROM CURRENT_DATE)`, which puts a late kick-off on the wrong
side of a UTC midnight the moment the DB and the fixture disagree about the day.
"""
Base.@kwdef struct SofaScoreEvents <: AbstractFixtureSource
    horizon::Period = Hour(36)
end

function fixtures(s::SofaScoreEvents, segment, as_of::DateTime)
    t_ids = Data.tournament_ids(segment)
    lo = Int(round(datetime2unix(as_of)))
    hi = Int(round(datetime2unix(as_of + s.horizon)))
    df = _query(FIXTURES_SQL, (t_ids, lo, hi))
    return Fixture[Fixture(Int(r.match_id), String(r.home_team), String(r.away_team),
                           unix2datetime(r.start_timestamp), Int(r.tournament_id))
                   for r in eachrow(df)]
end

"""
    ExplicitFixtures(fixtures)

A fixed list. The replay and test source -- it is how a past match day is re-run without
depending on what `sofascore.events` says today.
"""
struct ExplicitFixtures <: AbstractFixtureSource
    list::Vector{Fixture}
end

fixtures(s::ExplicitFixtures, _segment, as_of::DateTime) =
    Fixture[f for f in s.list if f.kickoff >= as_of]

# ===================================================================
# Identity resolvers
# ===================================================================

"""
    MatchMetaCrosswalk(; require_verified = true, markets = nothing)

Looks the fixture up in `betfair.match_meta`. **Not a matcher** -- see `AbstractIdentityResolver`.

Failure modes, all reported rather than thrown:
* `:absent_from_crosswalk` -- no row. The resolution job has not seen this fixture. This is the
  common case for anything recent: the job stopped around 2026-06-27, after which resolution is
  0%, having been 100% before.
* `:not_verified` -- a row exists but `is_verified` is false.
* `:no_markets` -- the event resolved but `betfair_live.market_metadata` has no markets for it.
"""
Base.@kwdef struct MatchMetaCrosswalk <: AbstractIdentityResolver
    require_verified::Bool = true
end

function resolve(r::MatchMetaCrosswalk, f::Fixture)
    df = _query(IDENTITY_SQL, (f.m_id,))
    isempty(df) && return Unresolved(f, :absent_from_crosswalk)

    verified = any(skipmissing(df.is_verified))
    (r.require_verified && !verified) && return Unresolved(f, :not_verified)

    ev = String(first(skipmissing(df.betfair_event_id)))
    mk = Dict{String,String}()
    for row in eachrow(df)
        (ismissing(row.market_id) || ismissing(row.market_type)) && continue
        mk[String(row.market_type)] = String(row.market_id)
    end
    isempty(mk) && return Unresolved(f, :no_markets)
    return Resolved(f, ev, mk, verified)
end

"""
    ResolverChain(resolvers...)

First success wins; if all fail, the **first** reason is reported, because it is the one from
the most authoritative source.
"""
struct ResolverChain{T<:Tuple} <: AbstractIdentityResolver
    resolvers::T
end
ResolverChain(rs::AbstractIdentityResolver...) = ResolverChain(rs)

function resolve(c::ResolverChain, f::Fixture)
    first_fail = nothing
    for r in c.resolvers
        out = resolve(r, f)
        out isa Resolved && return out
        first_fail === nothing && (first_fail = out)
    end
    return first_fail === nothing ? Unresolved(f, :no_resolver) : first_fail
end

# ===================================================================
# Lineup sources
# ===================================================================

"""
    ProvisionalDB()

`sofascore.lineup_provisional`, filtered to rows scraped at or before `as_of`.

Health warning, measured: `confirmed` has never been true for any match in this table. Every
scrape so far has run 4.4-5.8 hours before kick-off and SofaScore publishes the confirmed XI
about an hour out, so what this returns is a *predicted* XI. The scraper is correct; it has
simply never been invoked inside the window where the answer changes. Treat
`kickoff - scraped_at` as the usable signal, not `confirmed`.
"""
struct ProvisionalDB <: AbstractLineupSource end

function lineup(::ProvisionalDB, f::Fixture, as_of::DateTime)
    df = _query(LINEUP_SQL, (f.m_id, as_of))
    isempty(df) && return nothing

    home = Player[]; away = Player[]
    for r in eachrow(df)
        p = Player(Int(r.player_id),
                   ismissing(r.player_name) ? "Unknown" : String(r.player_name),
                   clean_position(ismissing(r.position) ? "M" : String(r.position)),
                   coalesce(r.substitute, false))
        push!(coalesce(r.is_home_team, true) ? home : away, p)
    end
    (isempty(home) || isempty(away)) && return nothing

    return Lineup(home, away, any(coalesce.(df.confirmed, false)), :provisional,
                  maximum(DateTime.(df.scraped_at)))
end

"""
    LastHistorical(ds)

Each team's most recent completed XI from `ds.lineups`. The floor of the chain: always answers,
never fresh. `compare_matchday_lineups` in the prototype measured how far this moves the model's
positional-sum inputs versus a provisional XI, and that comparison is still worth porting.
"""
struct LastHistorical <: AbstractLineupSource
    ds::Any
end
LastHistorical() = LastHistorical(nothing)

function lineup(s::LastHistorical, f::Fixture, as_of::DateTime)
    s.ds === nothing && return nothing
    h = _last_xi(s.ds, f.home, as_of)
    a = _last_xi(s.ds, f.away, as_of)
    (isempty(h) || isempty(a)) && return nothing
    return Lineup(h, a, false, :last_historical, as_of)
end

function _last_xi(ds, team::AbstractString, as_of::DateTime)
    m = ds.matches
    rows = findall(i -> (m.home_team[i] == team || m.away_team[i] == team) &&
                        DateTime(m.match_date[i]) <= as_of, 1:nrow(m))
    isempty(rows) && return Player[]
    i = rows[argmax(m.match_date[rows])]
    mid, side = m.match_id[i], m.home_team[i] == team ? "home" : "away"

    lu = ds.lineups
    sel = findall(j -> lu.match_id[j] == mid && String(lu.team_side[j]) == side, 1:nrow(lu))
    return Player[Player(Int(lu.player_id[j]),
                         ismissing(lu.player_name[j]) ? "Unknown" : String(lu.player_name[j]),
                         clean_position(ismissing(lu.position[j]) ? "M" : String(lu.position[j])),
                         coalesce(lu.is_substitute[j], false)) for j in sel]
end

"""
    JsonPin(dir)

A manually pinned XI at `<dir>/<match_id>.json`, in SofaScore's own response shape. Top of the
chain so a human can override every automated source for one fixture.
"""
struct JsonPin <: AbstractLineupSource
    dir::String
end

function lineup(s::JsonPin, f::Fixture, ::DateTime)
    path = joinpath(s.dir, "$(f.m_id).json")
    isfile(path) || return nothing
    data = try
        JSON3.read(read(path, String))
    catch e
        @warn "JsonPin: unparseable lineup file" path exception = e
        return nothing
    end
    (haskey(data, :home) && haskey(data, :away)) || return nothing
    pull(side) = Player[Player(Int(p.player.id), String(p.player.name),
                               clean_position(String(p.position)), Bool(p.substitute))
                        for p in side.players]
    return Lineup(pull(data.home), pull(data.away),
                  get(data, :confirmed, false), :json_pin, unix2datetime(0))
end

"""
    SourceChain(sources...)

First source that answers wins. This is the prototype's tiering, preserved verbatim in order --
manual pin, then announced XI, then last completed XI -- because each tier is strictly less
informative than the one above and the fallback never fails outright.

Distinct from `GateChain`, which is conjunctive. Two different combinators, deliberately.
"""
struct SourceChain{T<:Tuple} <: AbstractLineupSource
    sources::T
end
SourceChain(ss::AbstractLineupSource...) = SourceChain(ss)

function lineup(c::SourceChain, f::Fixture, as_of::DateTime)
    for s in c.sources
        out = lineup(s, f, as_of)
        out === nothing || return out
    end
    return nothing
end

"Normalise a raw position label to `:G`, `:D`, `:M`, `:F`. Unknown labels become `:M`."
function clean_position(pos::AbstractString)
    p = uppercase(strip(pos))
    (p == "G" || p == "GK" || p == "GOALKEEPER") && return :G
    (p == "D" || p == "DF" || p == "DEFENDER")   && return :D
    (p == "F" || p == "FW" || p == "A" || p == "FORWARD") && return :F
    return :M
end
