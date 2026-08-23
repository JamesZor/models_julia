# r02_replay.jl -- the capability the prototype never had.
#
# The old match-day code could only be exercised on a live Saturday. That is why none of it was
# ever validated: there was no way to run it twice, no way to run it against a known outcome,
# and no way to test a gate without waiting for something to break.
#
# `betfair_live.order_book_1m` archives the same Redis feed production reads, at 1-minute
# resolution. So `as_of` can be moved backwards and the whole pipeline re-run as of any instant.
# That is the single highest-value structural property of this module.

include("_setup.jl")

# ===================================================================
# 1. What is replayable
# ===================================================================
#
# Replay needs BOTH an order book and a resolvable match_id. Written as a query, not a fixed
# list, because the corpus grows every match week.

const CORPUS_SQL = """
SELECT DISTINCT mm.match_id, md.event_id
FROM betfair_live.order_book_1m o
JOIN betfair_live.market_metadata md USING (market_id)
JOIN betfair.match_meta mm ON mm.betfair_event_id = md.event_id AND mm.match_id IS NOT NULL;
"""

corpus = MD._query(CORPUS_SQL)
have   = Set(corpus.match_id)
days   = sort(unique(Date.(filter(r -> r.match_id in have, ds.matches).match_date)))

@printf("\nreplay corpus: %d matches over %d days, %s .. %s\n",
        length(have), length(days), first(days), last(days))

println("""
  Bounded at the far end by order_book_1m starting 2026-05-29, and at the near end by the
  identity resolver dying around 2026-06-27 -- resolution was 100% before that and is 0% after.
  Restarting that job is what unlocks everything since.""")

# ===================================================================
# 2. The two-snapshot workflow, automated
# ===================================================================
#
# The 12 Jun paper track priced at 17:38 (pre-lineup) and RE-PRICED at 20:11 (XIs confirmed,
# ~kickoff), then compared. It found the totals edges held through team news and that every
# skipped 1X2 selection drifted FURTHER from the model.
#
# That was done by hand, in a browser, because the infrastructure only ever captured snapshot
# one. Here it is a loop.

day = Date(2026, 6, 19)
spec = replay_spec(day)
snapshots = [DateTime(day, Time(h, m)) for (h, m) in ((13, 0), (15, 0), (17, 15), (18, 30))]

rows = NamedTuple[]
for t in snapshots
    r = MD.match_day(spec, SYS, DD.Ireland(), expr, ds; as_of = t, bankroll = 1000.0)
    isempty(r.sheet) && continue
    for b in eachrow(r.sheet)
        push!(rows, (as_of = t, match_id = b.match_id,
                     sel = "$(b.group)$(b.line == 0.0 ? "" : " $(b.line)")_$(b.selection)",
                     odds = b.odds, side = b.side, p_model = b.p_model, edge = b.edge))
    end
end
trace = DataFrame(rows)

println("\n--- how prices moved through the afternoon ---")
piv = unstack(select(trace, :sel, :as_of, :odds), :sel, :as_of, :odds; combine = first)
show(first(piv, 15), allcols = true)
println()

# ===================================================================
# 3. Did the market move toward us or away?
# ===================================================================
#
# CLV is the measurement instrument that matters here. On a sample this size, P/L cannot
# distinguish a real edge from luck -- the Portfolio backtest's ROI interval includes zero over
# 628 matches. Price movement between our entry and the close is far higher powered.
#
# Convention: a price that SHORTENS after we take it means the market came toward us (+CLV).

if !isempty(trace)
    early, late = minimum(trace.as_of), maximum(trace.as_of)
    e = select(filter(:as_of => ==(early), trace), :match_id, :sel, :odds => :odds_early)
    l = select(filter(:as_of => ==(late),  trace), :match_id, :sel, :odds => :odds_late)
    clv = innerjoin(e, l, on = [:match_id, :sel])
    clv.move_pct = round.(100 .* (clv.odds_late ./ clv.odds_early .- 1), digits = 2)
    clv.toward_us = clv.move_pct .< 0

    @printf("\n  %d selections priced at both %s and %s\n", nrow(clv),
            Dates.format(early, "HH:MM"), Dates.format(late, "HH:MM"))
    @printf("  moved toward us: %.0f%%   median move %.2f%%\n",
            100mean(clv.toward_us), median(clv.move_pct))
    show(sort(clv, :move_pct), allrows = false, allcols = true)
    println()
end

println("""

  One match day proves nothing -- this is the mechanism, not the result. Loop it over the whole
  corpus to get a number worth quoting, and re-run it as the corpus grows.""")

# ===================================================================
# 4. Why replay is not the backtest, and must not be compared to it
# ===================================================================
#
# `last_price_traded` is NULL in 100% of order_book_1m rows -- the Redis drain writes prices and
# volumes only. So replay prices off the BOOK (what you could actually have taken) while the
# Portfolio backtest settles at the TRADED price from betfair.odds_history.
#
# Those are different quantities. The book price is the honest one -- the backtest's own
# diagnostics show a 1% execution haircut costs ~24% of cumulative gain -- but a replay number
# and a backtest number are not comparable, and saying so is not a caveat you can drop.

println("\n  reminder: replay prices off the book; the backtest settles at traded prices.")
println("  Different quantities. Never put them in the same table without a note.")
