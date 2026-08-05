# r05_extending.jl -- adding your own components without touching src/.
#
# The claim the module makes is "add a struct + one method". This file tests that claim by
# adding three things from a runner file: a filter, a trust model, and a price policy.
#
# The recipe every time:
#   1. subtype the abstract seam        struct Mine <: PF.AbstractThing ... end
#   2. implement its ONE method         PF.the_method(m::Mine, ...) = ...
#   3. drop it into a PolicySpec/BookSpec and run
#
# Note step 2 uses the QUALIFIED name (`PF.keep`, not `keep`). Writing `keep(...)` here would
# define a NEW function in Main that the module never calls -- a silent no-op, and the single
# most common way to get this wrong.

include("_setup.jl")

spec = PF.BookSpec(markets = MARKETS)
CACHE = joinpath(@__DIR__, "books_$(string(PF.book_cache_key(spec), base = 16)).jls")
books = isfile(CACHE) ? deserialize(CACHE) :
        (b = PF.build_books(spec, latents_df, expr, odds, ds); serialize(CACHE, b); b)
slates = PF.group(PF.DailySlate(), books)

baseline = PF.path_metrics(PF.simulate(PF.PolicySpec(), slates))

# ===================================================================
# EXAMPLE 1: a filter -- "only back prices in a sane range"
# ===================================================================
#
# Seam:     AbstractSelectionFilter
# Contract: keep(f, sel, stake, ctx) -> Bool
#
# Filters run LAST and can only remove exposure; they never resize what survives.

struct OddsBand <: PF.AbstractSelectionFilter
    lo::Float64
    hi::Float64
end

PF.keep(f::OddsBand, s::PF.Selection, ::Real, ::PF.SlateContext) = f.lo <= s.odds_used <= f.hi

# ===================================================================
# EXAMPLE 2: a trust model -- "believe the model less on thin markets"
# ===================================================================
#
# Seam:     AbstractTrustModel
# Contract: trust_for(t, sel, ctx) -> Float64 in [0,1]
#
# A plausible idea worth testing: extreme prices come from the least liquid markets, where the
# closing quote is often a single trade, so the market probability there is noisy and the model
# deserves MORE weight -- or the price is untradeable and it deserves less. This encodes the
# second reading.

struct LiquidityTrust <: PF.AbstractTrustModel
    core::Float64      # weight on prices near evens
    tail::Float64      # weight on extreme prices
    cutoff::Float64    # decimal odds beyond which we call it a tail
end

function PF.trust_for(t::LiquidityTrust, s::PF.Selection, ::PF.SlateContext)
    extreme = s.odds_used >= t.cutoff || s.odds_used <= 1 + 1 / (t.cutoff - 1)
    return extreme ? t.tail : t.core
end

# ===================================================================
# EXAMPLE 3: a price policy -- crossing the spread
# ===================================================================
#
# Seam:     AbstractPricePolicy
# Contract: settlement_odds(p, d, overround) -> Float64
#
# This one lives in a BookSpec, so it CHANGES THE CACHE KEY and forces a rebuild. That is the
# distinction to internalise: policy components are free, book components are not.

struct HaircutPrice <: PF.AbstractPricePolicy
    ticks::Float64      # fraction of the price given up to cross the spread
end

PF.settlement_odds(p::HaircutPrice, d::Real, ov::Real) = d * min(ov, 1.0) * (1 - p.ticks)

# ===================================================================
# Race them
# ===================================================================

println("\n", "="^92, "\n=== POLICY-LEVEL EXTENSIONS (no rebuild needed) ===\n", "="^92)

race = DataFrame(policy = String[], n_bets = Int[], roi = Float64[], final = Float64[],
                 growth = Float64[], mdd = Float64[])

function add!(name, pol)
    t = PF.simulate(pol, slates)
    m = PF.path_metrics(t)
    push!(race, (name, m.n_bets, round(m.roi, digits = 2), round(m.final, digits = 3),
                 round(m.growth_per_slate, digits = 5), round(m.mdd, digits = 1)))
end

add!("baseline",            PF.PolicySpec())
add!("OddsBand(1.5, 6.0)",  PF.PolicySpec(filter = OddsBand(1.5, 6.0)))
add!("OddsBand(1.2, 15.0)", PF.PolicySpec(filter = OddsBand(1.2, 15.0)))
add!("LiquidityTrust",      PF.PolicySpec(trust = LiquidityTrust(0.35, 0.05, 6.0)))
# filters compose with FilterChain -- conjunction, all must keep
add!("band + min edge",     PF.PolicySpec(filter = PF.FilterChain(OddsBand(1.5, 6.0),
                                                                  PF.MinEdge(0.02))))
println(race)

println("""

  Note none of that rebuilt a book: trust and filter are PolicySpec fields, so all five rows
  came off the same cached Vector{MatchBook}.""")

# ===================================================================
# The book-level extension DOES rebuild
# ===================================================================

println("\n", "="^92, "\n=== BOOK-LEVEL EXTENSION (rebuild required) ===\n", "="^92)

spec_hc = PF.BookSpec(markets = MARKETS, price = HaircutPrice(0.01),
                      shrink = PF.NoShrinkage())     # shrinkage off to keep the demo quick
@printf("  cache key  DeArb %s\n             Haircut %s   <- different, so rebuild\n",
        string(PF.book_cache_key(spec), base = 16),
        string(PF.book_cache_key(spec_hc), base = 16))

books_hc = @time PF.build_books(spec_hc, latents_df, expr, odds, ds)
let m0 = PF.path_metrics(PF.simulate(PF.PolicySpec(), slates; use_shrink = false)),
    m1 = PF.path_metrics(PF.simulate(PF.PolicySpec(),
                                     PF.group(PF.DailySlate(), books_hc); use_shrink = false))
    @printf("\n  DeArb              ROI %.2f%%  final %.3fx\n", m0.roi, m0.final)
    @printf("  Haircut(1%% of price) ROI %.2f%%  final %.3fx\n", m1.roi, m1.final)
    println("""
  A 1% haircut is a crude stand-in for crossing the spread. If it erases the edge, the edge
  was never executable -- which on a market with one trade per closing window is the question
  that matters most.""")
end

# ===================================================================
# What you did NOT have to do
# ===================================================================
println("""

$("="^92)
  Three new components, zero edits to src/. No registry, no if/else in the solver, no enum to
  extend. The dispatch table is the registry.

  Two rules to remember:
    1. extend with the QUALIFIED name (PF.keep, PF.trust_for, PF.settlement_odds).
       A bare `keep(...)` defines a new function in Main and is silently never called.
    2. know which config your component lives in. PolicySpec = free. BookSpec = rebuild.
$("="^92)""")
