# r05_extending.jl -- adding your own components without touching src/.
#
# Same claim as the portfolio runbook, same recipe:
#   1. subtype the seam          struct Mine <: MD.AbstractThing ... end
#   2. implement its ONE method  MD.the_method(m::Mine, ...) = ...
#   3. drop it into a MatchDaySpec and run
#
# Step 2 uses the QUALIFIED name (`MD.ready`, not `ready`). A bare `ready(...)` defines a new
# function in Main that the module never calls -- a silent no-op, and the most common way to get
# this wrong.

include("_setup.jl")

as_of = DateTime(2026, 6, 19, 17, 15)
spec  = replay_spec(Date(2026, 6, 19))

# ===================================================================
# EXAMPLE 1: a gate -- "don't bet a fixture whose XI I have never seen"
# ===================================================================
#
# Seam:     AbstractReadinessGate
# Contract: ready(gate, card) -> Ready | Blocked
#
# Gates are the cheapest place to encode an operational rule, because they run after everything
# is known and their reasons are reported rather than swallowed.

struct RequireProvisional <: MD.AbstractReadinessGate end

function MD.ready(::RequireProvisional, c::MD.FixtureCard)
    lu = c.lineup
    (lu !== nothing && lu.source === :provisional) && return MD.Ready()
    src = lu === nothing ? "none" : string(lu.source)
    return MD.Blocked([:lineup => "lineup source is $src, wanted :provisional"])
end

# ===================================================================
# EXAMPLE 2: a quote rule -- pay the spread
# ===================================================================
#
# Seam:     AbstractQuoteRule
# Contract: quote_price(rule, levels, side) -> Float64
#
# A crude execution-realism test: assume you never get the top of the book and always fill one
# level worse. If an edge survives that, it might survive a real fill.

struct SecondLevel <: MD.AbstractQuoteRule end

function MD.quote_price(::SecondLevel, b::MD.BookLevels, side::Symbol)
    v = side === :back ? b.back : b.lay
    length(v) >= 2 && return v[2]
    return isempty(v) ? NaN : v[1]
end

# ===================================================================
# EXAMPLE 3: a stake rounding rule -- round to whole pence
# ===================================================================
#
# Seam:     AbstractStakeRounding
# Contract: round_stake(rule, stake, instrument) -> Float64  (0.0 drops the leg)

Base.@kwdef struct PenceRounding <: MD.AbstractStakeRounding
    minimum::Float64 = 1.0
end

function MD.round_stake(r::PenceRounding, stake::Real, inst::MD.Instrument)
    stake <= 0 && return 0.0
    MD.venue_stake(inst, stake) < r.minimum && return 0.0
    return round(Float64(stake), digits = 2)
end

# ===================================================================
# Race them
# ===================================================================

println("\n", "="^90, "\n=== EXTENSIONS ===\n", "="^90)

race = DataFrame(variant = String[], bets = Int[], blocked = Int[],
                 total_risk = Float64[], mean_odds = Float64[])

function add!(name, sp)
    r = MD.match_day(sp, SYS, DD.Ireland(), expr, ds; as_of = as_of, bankroll = 1000.0)
    push!(race, (name, nrow(r.sheet), length(r.blocked),
                 round(sum(r.sheet.risk; init = 0.0), digits = 2),
                 isempty(r.sheet) ? 0.0 : round(mean(r.sheet.odds), digits = 3)))
end

_spec(; kw...) = MD.MatchDaySpec(; fixtures = spec.fixtures, lineups = LINEUPS,
                                  gate = GATES, markets = MARKETS, kw...)

add!("baseline",            _spec())
add!("DirectBackOnly",      _spec(instrument = MD.DirectBackOnly()))
add!("SecondLevel quotes",  _spec(quote_rule = MD.SecondLevel()))
add!("RequireProvisional",  _spec(gate = MD.GateChain(MD.IdentityResolved(), RequireProvisional())))
add!("PenceRounding",       _spec(rounding = PenceRounding()))

show(race, allrows = true, allcols = true); println()

println("""

  Read `mean_odds` between the first two rows: BestOfBackLay never prices a position worse than
  DirectBackOnly, so it can only move that number up. `SecondLevel` moves it down, and by how
  much is a rough measure of what the spread would cost you on a bad fill.

  Three new components, zero edits to src/. Two rules:
    1. extend with the QUALIFIED name (MD.ready, MD.quote_price, MD.round_stake).
    2. know which stage your component sits in -- a gate runs after pricing and can only
       remove; a quote rule runs before and changes every number downstream.""")

# ===================================================================
# What is deliberately NOT extensible yet
# ===================================================================

println("""

$("="^90)
  Declared seams with no working implementation. They ERROR or DEFER rather than quietly
  returning empty, which is the same convention src/Portfolio uses for NetMarketCommission and
  VolTargetCap:

    RedisLive              errors -- Redis is down and its data is drained into Postgres anyway,
                           so ArchivedOrderBook reads the same feed
    MarketPillarFromBook   defers -- needs the engine's exact feature key, which should be read
                           off the engine rather than guessed. Not needed by the current smile
                           engine, whose extract_parameters consumes no market feature.
    SizedBestOfBackLay     not built -- size checking is out of scope; BestOfBackLay's leverage
                           cap covers the dangerous cases on price alone
$("="^90)""")
