# src/MatchDay/implementations/gates.jl
#
# Refuse-to-bet conditions.
#
# The justification for this layer is not per-bet filtering -- Portfolio's AbstractSelectionFilter
# already does that. It is that the three most likely things to be wrong on any given match day
# are a dead identity resolver, a dead order-book drain and a stale XI, and NONE of them
# currently produces a visible symptom. An empty stake sheet looks identical to "no bets today".

export IdentityResolved, MaxBookAge, ConfirmedXI, MinMatched, MaxLineupAge, GateChain

"""
    GateChain(gates...)

Conjunctive: runs **every** gate and concatenates reasons rather than short-circuiting. The
second reason is usually the informative one -- "unresolved" plus "book 4 days old" is a dead
collector, whereas "unresolved" alone is a dead resolver.
"""
struct GateChain{T<:Tuple} <: AbstractReadinessGate
    gates::T
end
GateChain(gs::AbstractReadinessGate...) = GateChain(gs)

function ready(c::GateChain, card::FixtureCard)
    reasons = Pair{Symbol,String}[]
    for g in c.gates
        out = ready(g, card)
        out isa Blocked && append!(reasons, out.reasons)
    end
    return isempty(reasons) ? Ready() : Blocked(reasons)
end

"""
    IdentityResolved()

Blocks a fixture we cannot map onto exchange markets. Blocking rather than dropping is the
point: an unresolved fixture that is silently filtered out is invisible, and invisible failures
are how a dead resolver goes unnoticed for six weeks.
"""
struct IdentityResolved <: AbstractReadinessGate end

ready(::IdentityResolved, c::FixtureCard{Resolved}) = Ready()
ready(::IdentityResolved, c::FixtureCard{Unresolved}) =
    Blocked([:identity => "unresolved ($(c.identity.reason)) -- no betfair.match_meta row, " *
                          "the resolution job has probably not run for this fixture"])

"""
    MaxBookAge(max_age)

Blocks when the newest quote is older than `max_age` relative to `as_of`.

This is the gate that would have caught the drain outage: the last order-book tick was
2026-08-02 while 99 markets opened after it, so anything priced since would have been staking
against a book that had stopped moving.

Reads `card.book_age`, set by the pipeline after quoting; a card that was never quoted is
blocked rather than passed.
"""
struct MaxBookAge <: AbstractReadinessGate
    max_age::Period
end

function ready(g::MaxBookAge, c::FixtureCard)
    age = get(_card_meta(c), :book_age, nothing)
    age === nothing && return Blocked([:book => "no quotes retrieved"])
    return age <= g.max_age ? Ready() :
           Blocked([:book => "stale book: newest tick $(canonicalize(age)) before as_of " *
                             "(limit $(canonicalize(g.max_age)))"])
end

"""
    ConfirmedXI(; blocking = false)

Non-blocking by default, and that default is measured rather than chosen: `confirmed` has never
been true for any match in `sofascore.lineup_provisional`, so a blocking version blocks 100% of
fixtures. It becomes usable once the scraper is re-invoked inside the last hour before kick-off.
"""
Base.@kwdef struct ConfirmedXI <: AbstractReadinessGate
    blocking::Bool = false
end

function ready(g::ConfirmedXI, c::FixtureCard)
    lu = c.lineup
    (lu !== nothing && lu.confirmed) && return Ready()
    msg = lu === nothing ? "no lineup at all" : "lineup is a predicted XI ($(lu.source))"
    return g.blocking ? Blocked([:lineup => msg]) : Ready()
end

"""
    MaxLineupAge(max_age; blocking = false)

How long before kick-off the XI was scraped. The usable version of `ConfirmedXI` while
`confirmed` is never set: an XI scraped 5 hours out is a guess, one scraped 40 minutes out is
close to the real thing.
"""
Base.@kwdef struct MaxLineupAge <: AbstractReadinessGate
    max_age::Period = Hour(2)
    blocking::Bool  = false
end

function ready(g::MaxLineupAge, c::FixtureCard)
    lu = c.lineup
    lu === nothing && return g.blocking ? Blocked([:lineup => "no lineup"]) : Ready()
    lead = c.fixture.kickoff - lu.scraped_at
    lead <= g.max_age && return Ready()
    msg = "XI scraped $(canonicalize(lead)) before kick-off (limit $(canonicalize(g.max_age)))"
    return g.blocking ? Blocked([:lineup => msg]) : Ready()
end

"""
    MinMatched(minimum; blocking = false)

Liquidity floor on `market_matched`.

Non-blocking by default because the column is NULL in 62% of `order_book_1m` rows -- a blocking
gate would be blind more often than it fires, and a gate that silently passes on missing data
is worse than no gate. Reported so the number is at least visible.
"""
Base.@kwdef struct MinMatched <: AbstractReadinessGate
    minimum::Float64 = 500.0
    blocking::Bool   = false
end

function ready(g::MinMatched, c::FixtureCard)
    m = get(_card_meta(c), :max_matched, nothing)
    (m === nothing || isnan(m)) &&
        return g.blocking ? Blocked([:liquidity => "market_matched unavailable"]) : Ready()
    m >= g.minimum && return Ready()
    msg = "matched $(round(m, digits=0)) below minimum $(g.minimum)"
    return g.blocking ? Blocked([:liquidity => msg]) : Ready()
end

# Gate metadata is attached by the pipeline rather than carried on FixtureCard, so that adding a
# gate never requires widening the domain type.
const _CARD_META = IdDict{FixtureCard,Dict{Symbol,Any}}()
_card_meta(c::FixtureCard) = get(_CARD_META, c, Dict{Symbol,Any}())
_set_card_meta!(c::FixtureCard, k::Symbol, v) =
    (get!(_CARD_META, c, Dict{Symbol,Any}())[k] = v)
