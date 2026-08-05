# r03_matchday_stakes.jl -- pricing fixtures you have NOT played yet.
#
# THE MATCH-DAY RECIPE
# ====================
# Three inputs, one call:
#
#   1. latents_df   posterior summaries for the UPCOMING fixtures.
#                   NOT from Experiments.extract_oos_predictions -- that only covers matches
#                   already inside a CV fold. For live fixtures use the match-day inference
#                   pipeline:
#                       include("current_development/match_day_inference/loader.jl")
#                       todays = get_todays_matches(ds, ...)
#                       latents = compute_todays_matches_latents(ds, expr, todays, LINEUPS_DIR)
#                   (that step needs today's lineups; see match_day_inference/fetch_lineups.sh)
#
#   2. odds_df      current prices, in the ds.odds schema:
#                       :match_id, :market_name, :market_line, :selection, :odds_close
#                   Any source with that schema works -- the historical Betfair summary here,
#                   or a live feed (match_day_inference/src/live_betting.jl reads redis).
#
#   3. sys          a PortfolioSystem: which markets, how to price, how much to trust, how much
#                   drawdown to budget, what to cap simultaneous exposure at.
#
#   sheet = PF.stake_sheet(sys, latents_df, expr, odds_df, ds; bankroll = 1000.0)
#
# The sheet is produced by the SAME code path that was audited against history. There is one
# difference and only one: a match-day book has `settle == nothing`, so it can be staked but
# not simulated. `simulate` refuses unsettled books rather than quietly scoring them as losses.
#
# Below we simulate match day honestly: take a real card, and price it as if the results were
# unknown.

include("_setup.jl")

# ===================================================================
# 1. The system you are betting
# ===================================================================

sys = PF.PortfolioSystem(
    PF.BookSpec(markets = MARKETS),                  # DeArb + Kelly + BakerMcHale + 2% commission
    PF.PolicySpec(trust = PF.FlatTrust(0.25),
                  risk  = PF.SlateDrawdown(23.0),    # ~20% drawdown budget at 1% probability
                  cap   = PF.FixedCap(0.25)),        # never more than 25% live at once
)

BANKROLL = 1_000.0

# ===================================================================
# 2. Pick the card
# ===================================================================
#
# In production: whatever kicks off today. Here: the busiest historical match day, because a
# full Saturday card is where simultaneous exposure actually bites.

_have = Set(latents_df.match_id)
_by_date = Dict{Date,Vector{Int}}()
for r in eachrow(ds.matches)
    r.match_id in _have || continue
    push!(get!(_by_date, Date(r.match_date), Int[]), Int(r.match_id))
end
target_date = argmax(d -> length(_by_date[d]), collect(keys(_by_date)))
card = latents_df[in.(latents_df.match_id, Ref(Set(_by_date[target_date]))), :]

@info "match day" date = target_date fixtures = nrow(card)

# ===================================================================
# 3. One call
# ===================================================================

sheet = PF.stake_sheet(sys, card, expr, odds, ds; bankroll = BANKROLL)

# ===================================================================
# 4. Check the SLATE before you look at the bets
# ===================================================================
#
# Exposure is the number that can ruin you. Individual stakes cannot.

println("\n", "="^78, "\n=== SLATE SUMMARY -- read this first ===\n", "="^78)
summary = PF.slate_summary(sheet)
summary.total_stake = round.(summary.total_stake, digits = 2)
summary.exposure    = round.(summary.exposure, digits = 4)
summary.k_risk      = round.(summary.k_risk, digits = 4)
println(summary)
@printf("\n  %.1f%% of bankroll live simultaneously. Drawdown budget cut stakes to %.1f%% of\n",
        100 * first(summary.exposure), 100 * first(summary.k_risk))
println("  what full Kelly wanted. Hard cap bound: ", first(summary.capped))

# ===================================================================
# 5. The stake sheet
# ===================================================================

disp = select(sheet, :match_id, :family, :selection, :odds, :p_model, :p_market, :edge,
              :frac, :stake)
for c in (:odds, :p_model, :p_market, :edge); disp[!, c] = round.(disp[!, c], digits = 3); end
disp.frac  = round.(disp.frac .* 100, digits = 2)
disp.stake = round.(disp.stake, digits = 2)

println("\n", "="^90, "\n=== STAKE SHEET  (bankroll $(BANKROLL)) ===\n", "="^90)
println(disp)

println("""

  You will see rows with NEGATIVE edge that still carry a stake. That is not a bug.

  This is a PORTFOLIO Kelly solve, not a list of independent value bets: the allocator
  maximises expected log-growth over the whole 144-state score grid at once, so it will happily
  take a small negative-edge position when it hedges a larger correlated one in the same match
  (typically a draw against a big home/away position). Judge the sheet per match, not per row.
  If you want only standalone value, add a `MinEdge` filter to the policy -- but expect growth
  to fall, because you have removed the hedges.""")

# ===================================================================
# 6. Proof it is really the match-day path
# ===================================================================
#
# Same fixtures, built the two ways. The stakes must be identical -- the ONLY difference
# between backtest and match-day is whether a settlement vector is attached.

println("\n", "="^78, "\n=== BACKTEST vs MATCH-DAY: same stakes ===\n", "="^78)
let bt = PF.build_books(sys.book, card, expr, odds, ds; require_result = true),
    md = PF.build_books(sys.book, card, expr, odds, ds; require_result = false)
    @printf("  settled books   %d  (all have a settle vector: %s)\n",
            length(bt), all(PF.is_settled, bt))
    @printf("  unsettled build %d  (settle attached where a result exists: %s)\n",
            length(md), all(PF.is_settled, md))
    @printf("  max |Δ a_kelly| %.3e   <- must be 0\n",
            maximum(maximum(abs.(b.a_kelly .- m.a_kelly)) for (b, m) in zip(bt, md)))
    # and simulate refuses a book with no result
    fake = PF.MatchBook(md[1].m_id, md[1].date, md[1].sels, md[1].p_grid, md[1].R, nothing,
                        md[1].a_kelly, md[1].k_shrink, md[1].kkt, md[1].converged)
    ok = try
        PF.simulate(sys.policy, [PF.Slate(fake.date, [fake])]); false
    catch
        true
    end
    println("  simulate refuses an unsettled book: ", ok)
end

println("""

  Before acting on a sheet like this:
   * `odds` is the de-arbed TRADED price from a 20-minute pre-kick-off window. On this league
     the median O/U and BTTS market has ONE trade in that window. It is not a quote you can
     necessarily get filled at, and there is no back/lay spread modelled anywhere. Against a
     LIVE feed you should be reading the back price and its available size, not a traded mid.
   * `edge` is against the vig-removed market price. It is not a forecast of profit.
   * the risk factor was solved under the MODEL's probabilities. If the model is miscalibrated,
     the drawdown guarantee is decorative -- realised drawdown has been running ~1.15x nominal.""")
