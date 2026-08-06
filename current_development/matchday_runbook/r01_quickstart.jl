# r01_quickstart.jl -- the whole pipeline, on a real match day.
#
# MatchDay's job in one sentence: manufacture the two inputs Portfolio needs -- a `latents_df` of
# posterior draws and an `odds_df` of prices -- for fixtures that have not been played, and
# refuse loudly when it cannot.
#
# It does no staking maths. Allocation, shrinkage, drawdown and the exposure cap are all
# src/Portfolio and are reached through one call to `stake_sheet`.

include("_setup.jl")

# ===================================================================
# 1. One call
# ===================================================================
#
# `as_of` is the instant being priced. It is a parameter, never `now()` read inside a stage --
# that is what makes a past match day replayable and a live decision auditable.
#
# Here we replay 19 June 2026 at 17:15, which is the exact moment the paper track in
# `match_day_inference/paper_tracks/ireland_19_06_26.md` was generated. Kick-off was 18:45.

as_of = DateTime(2026, 6, 19, 17, 15)
spec  = replay_spec(Date(2026, 6, 19))

res = MD.match_day(spec, SYS, DD.Ireland(), expr, ds; as_of = as_of, bankroll = 15.0)

println("\n", "="^90)
println(res)
println("="^90)

# Expect roughly:
#   MatchDayResult(23 bets, 5 priced, 0 blocked, as_of 2026-06-19T17:15:00)
#
# and a warning from select_split about conditioning on split 29 of 31. That warning is real and
# unresolved -- see r04.

# ===================================================================
# 2. Read the exposure before you read the bets
# ===================================================================
#
# Same discipline as the portfolio runbook: individual stakes cannot ruin you, simultaneous
# exposure can.

println("\n--- slate summary ---")
println(PF.slate_summary(res.sheet))

# ===================================================================
# 3. The sheet
# ===================================================================
#
# Columns MatchDay adds on top of Portfolio's:
#
#   side         :back or :lay -- HOW the position is expressed
#   venue_odds   the price shown on the exchange for that side
#   risk         what you actually have at stake (Portfolio's `stake`, after the minimum rule)
#   venue_stake  what you PLACE. For a lay that is risk/(d-1), not risk.
#
# `odds` is the EFFECTIVE price of the position whichever side it is on, denominated so a unit
# of stake is a unit of risk. That is why Portfolio never needed to learn what a lay is.

println("\n--- stake sheet ---")
show(select(res.sheet, :match_id, :group, :line, :selection, :side,
            :odds        => ByRow(x -> round(x, digits = 3)) => :eff_odds,
            :venue_odds  => ByRow(x -> round(x, digits = 3)) => :venue,
            :edge        => ByRow(x -> round(x, digits = 4)) => :edge,
            :risk        => ByRow(x -> round(x, digits = 2)) => :risk,
            :venue_stake => ByRow(x -> round(x, digits = 2)) => :place),
     allrows = true, allcols = true)
println()

# ===================================================================
# 4. What you would actually place
# ===================================================================

println("\n--- order tickets ---")
for r in eachrow(first(sort(res.sheet, :risk, rev = true), 5))
    t = MD.order_ticket(r)
    @printf("  %-9s %-4s %-10s %s @ %.2f   stake £%.2f   liability £%.2f\n",
            t.market, t.line == 0.0 ? "" : string(t.line), string(t.selection),
            uppercase(string(t.side)), t.price, t.stake, t.liability)
end

println("""

  Note `stake` and `liability` differ only for a lay: you post the backer's stake and your
  liability is what is at risk. For a back they are the same number.""")

# ===================================================================
# 5. The bit that will surprise you
# ===================================================================
#
# The default rounding is NoMinimum -- research mode, which ignores the exchange floor.
# Betfair's minimum is £1. Apply it and see what survives.

println("\n", "="^90, "\n=== THE £1 MINIMUM ===\n", "="^90)

for (label, bankroll) in (("£15", 15.0), ("£100", 100.0), ("£1000", 1000.0))
    sp = MD.MatchDaySpec(fixtures = spec.fixtures, lineups = LINEUPS, gate = GATES,
                         markets = MARKETS, rounding = MD.FloorOrDrop(minimum = 1.0))
    r  = MD.match_day(sp, SYS, DD.Ireland(), expr, ds; as_of = as_of, bankroll = bankroll)
    @printf("  bankroll %-6s -> %2d placeable bets (of %d priced)\n",
            label, nrow(r.sheet), nrow(res.sheet))
end

println("""

  At £15 the answer is ZERO. Every one of the 23 legs is between 1p and 31p, because a 25%
  exposure cap over 5 fixtures leaves ~£3.75 to spread across 23 positions.

  That is not a bug and no amount of tuning fixes it: £15 is simply below the operating
  threshold for a 5-match slate. Either raise the bankroll, cut the book to a handful of
  selections, or accept FloorOrRoundUp and the over-staking it implies.""")
