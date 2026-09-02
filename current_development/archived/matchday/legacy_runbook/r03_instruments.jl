# r03_instruments.jl -- back, lay, and why Portfolio never had to learn the difference.
#
# On a two-runner market every position has TWO instruments. Backing Over 2.5 and laying
# Under 2.5 are the same bet; you should always take whichever prices better.
#
# The trick that makes this cheap is measuring the position in RISK rather than in stake.

include("_setup.jl")

# ===================================================================
# 1. The arithmetic
# ===================================================================
#
#   lay `Under` at d with backer stake b   ->   risk = b(d-1),  win = b
#   set risk = s, so b = s/(d-1)           ->   win/risk = 1/(d-1)
#   a back at D has win/risk = D-1         ->   D = 1 + 1/(d-1) = d/(d-1)

println("\n", "="^88, "\n=== BACK/LAY EQUIVALENCE ===\n", "="^88)

d, s = 1.26, 1.0
D    = MD.lay_to_back(d)
b    = s / (d - 1)

@printf("  lay Under at %.2f, risking £%.2f\n", d, s)
@printf("    post £%.4f with the backer\n", b)
@printf("    Over hits  -> +£%.4f      Under hits -> -£%.2f\n", b, s)
@printf("  back Over at the effective price %.4f, staking £%.2f\n", D, s)
@printf("    Over hits  -> +£%.4f      Under hits -> -£%.2f\n", s * (D - 1), s)
println("\n  Identical. That is the whole trick.")

println("""

  CONSEQUENCE: if the instrument layer hands Portfolio (effective_odds, risk), then the payoff
  matrix, KellyLogUtility, BakerMcHale, SlateDrawdown and FixedCap are ALREADY denominated in
  risk and work unchanged. FixedCap sums liability by construction -- it does not need to be
  made "liability aware", it always was. Only the order ticket differs.""")

# ===================================================================
# 2. On a real book
# ===================================================================

f  = MD.Fixture(15238109, "bohemian", "dundalk-fc", DateTime(2026, 6, 19, 19, 0), 79)
id = MD.resolve(MD.MatchMetaCrosswalk(), f)
as_of = DateTime(2026, 6, 19, 17, 15)
book  = MD.quotes(MD.ArchivedOrderBook(), id, as_of)
ks, q = collect(keys(book)), MD.BestAvailable()

rows = NamedTuple[]
for k in sort(ks, by = x -> (x.group, x.line, string(x.selection)))
    c = MD.complement_of(k, ks)
    direct = MD.instrument(MD.DirectBackOnly(), k, c, book, q)
    best   = MD.instrument(MD.BestOfBackLay(),  k, c, book, q)
    (direct === nothing || best === nothing) && continue
    push!(rows, (sel = "$(k.group)$(k.line == 0.0 ? "" : " $(k.line)")_$(k.selection)",
                 direct = round(direct.odds, digits = 3),
                 chosen = round(best.odds, digits = 3), side = best.side,
                 gain_pct = round(100 * (best.odds / direct.odds - 1), digits = 3),
                 leverage = round(best.leverage, digits = 2)))
end
cmp = DataFrame(rows)

println("\n", "="^88, "\n=== bohemian v dundalk, 2026-06-19 17:15 ===\n", "="^88)
show(cmp, allrows = true, allcols = true); println()

@printf("\n  %d of %d selections priced better by laying the complement\n",
        count(==(:lay), cmp.side), nrow(cmp))
@printf("  mean gain where lay wins: %.3f%%\n",
        mean(cmp.gain_pct[cmp.side .== :lay]))

# ===================================================================
# 3. Why it is worth more in Scotland than in Ireland
# ===================================================================
#
# Measured over 43,796 uncrossed two-sided snapshots. The MEDIAN gain is ~0 -- the book is
# arbitrage-free, so usually the two instruments agree. Taking the better one is a free OPTION,
# and E[max(0, gain)] is where the value shows up:
#
#   competition             O/U1.5  O/U2.5  O/U3.5 | back overround @3.5
#   Scottish League Two      0.13%   1.09%   6.43% |  7.94%
#   Scottish League One      0.20%   0.97%   3.48% |  6.63%
#   Scottish Championship    0.20%   0.93%   5.07% |  4.33%
#   Scottish Premiership     0.29%   0.47%   1.94% |  1.71%
#   Irish Division 1         0.30%   0.43%   1.88% |  2.13%
#   Irish Premier Division   0.28%   0.37%   1.13% |  1.39%
#
# It tracks book width almost monotonically: worth most exactly where the book is worst. And
# it is asymmetric by side -- quote a longshot by LAYING ITS COMPLEMENT, because the complement
# is a near-certainty and therefore tightly priced while the longshot's own back book is wide.

# ===================================================================
# 4. The guard, and why it replaces a depth query
# ===================================================================
#
# The synthetic needs backer stake risk/(d-1), which blows up as d -> 1: laying Under 0.5 at
# 1.02 needs £50 posted for £1 of risk. Those are also the lines where the measured "gain"
# looked implausible (O/U 5.5 at 46%, O/U 0.5 at 23%) -- an empty back book, not an edge.
#
# `max_leverage` rejects them on price alone. No volume data required, which is what makes
# skipping the depth check safe.

println("\n", "="^88, "\n=== THE LEVERAGE CAP ===\n", "="^88)

thin = Dict((group = "OverUnder", line = 0.5, selection = :over_05)  =>
                MD.BookLevels([12.0], [50.0], [13.0], [50.0], 1000.0, as_of),
            (group = "OverUnder", line = 0.5, selection = :under_05) =>
                MD.BookLevels([1.02], [50.0], [1.03], [50.0], 1000.0, as_of))
tk = collect(keys(thin))
k  = (group = "OverUnder", line = 0.5, selection = :over_05)

for lev in (20.0, 1e6)
    i = MD.instrument(MD.BestOfBackLay(max_leverage = lev), k, MD.complement_of(k, tk), thin, q)
    @printf("  max_leverage %-8s -> %s at %.2f  (leverage %.1fx)\n",
            lev == 1e6 ? "off" : string(Int(lev)), uppercase(string(i.side)), i.odds, i.leverage)
end

println("""

  Uncapped it "improves" 12.0 to 34.3 by laying a 1.03 shot -- a 33x-leveraged position off a
  book that is almost certainly empty. Capped, it takes the direct back at 12.0 and the artifact
  disappears without a single size query.""")

# ===================================================================
# 5. The minimum-stake interaction
# ===================================================================
#
# Betfair's £1 minimum applies to the VENUE STAKE, not to your risk. A lay at a short price
# therefore clears the minimum with far less at risk -- so the morphism does not only buy a
# better price, it buys SMALLER MINIMUM POSITIONS, which matters most at a small bankroll.

println("\n", "="^88, "\n=== £1 MINIMUM: RISK vs VENUE STAKE ===\n", "="^88)

lay  = MD.Instrument(k, MD.lay_to_back(1.26), :lay, 1.26, 1 / 0.26)
back = MD.Instrument(k, 4.80, :back, 4.80, 1.0)
rule = MD.FloorOrDrop(minimum = 1.0)

for (name, inst) in (("back @ 4.80", back), ("lay @ 1.26", lay))
    for risk in (0.20, 0.50, 1.00)
        @printf("  %-12s risk £%.2f -> place £%.2f -> %s\n", name, risk,
                MD.venue_stake(inst, risk),
                MD.round_stake(rule, risk, inst) > 0 ? "KEPT" : "dropped")
    end
end

println("""

  Same 50p of risk: dropped as a back, kept as a lay. On a £15 bankroll where every leg is
  pennies, that is the difference between a placeable book and an empty one.""")
