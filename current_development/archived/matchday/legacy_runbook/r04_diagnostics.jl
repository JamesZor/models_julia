# r04_diagnostics.jl -- what to check before you place anything.
#
# Every check here corresponds to something that is currently broken, or to a failure mode that
# produced a plausible-looking empty result rather than an error.
#
# The recurring theme on this project is silent emptiness. Run this whenever a match day
# produces fewer bets than you expected -- ESPECIALLY when it produces none.

include("_setup.jl")

as_of = DateTime(2026, 6, 19, 17, 15)
spec  = replay_spec(Date(2026, 6, 19))
res   = MD.match_day(spec, SYS, DD.Ireland(), expr, ds; as_of = as_of, bankroll = 1000.0)

# ===================================================================
# 1. "No bets" vs "broken": always read this first
# ===================================================================
#
# A refusal is a VALUE, not an absence. `blocked_report` is the difference between
# "nothing qualified today" and "three background jobs are dead".

println("\n", "="^90, "\n=== 1. REFUSALS ===\n", "="^90)
@printf("  priced %d  |  blocked %d  |  bets %d\n",
        length(res.cards) - length(res.blocked), length(res.blocked), nrow(res.sheet))
rep = MD.blocked_report(res)
isempty(rep) ? println("  nothing blocked.") : show(rep, allrows = true, allcols = true)
println()

# Now the same call on a recent fixture, where the infrastructure IS broken:
recent = filter(r -> Date(r.match_date) == maximum(Date.(ds.matches.match_date)), ds.matches)
f2 = MD.Fixture(Int(recent.match_id[1]), String(recent.home_team[1]), String(recent.away_team[1]),
                DateTime(maximum(Date.(ds.matches.match_date)), Time(18, 45)), 79)
spec2 = MD.MatchDaySpec(fixtures = MD.ExplicitFixtures([f2]), lineups = LINEUPS,
                        gate = GATES, markets = MARKETS)
res2 = MD.match_day(spec2, SYS, DD.Ireland(), expr, ds;
                    as_of = f2.kickoff - Hour(2), bankroll = 1000.0)

println("\n  --- the same pipeline on the most recent fixture in the store ---")
show(MD.blocked_report(res2), allrows = true, allcols = true)
println("""

  Two reasons, not one. GateChain is CONJUNCTIVE and collects every failure rather than
  short-circuiting, because the second reason is usually the informative one:
  "unresolved" alone is a dead resolver; "unresolved" AND "no quotes" is a dead collector too.""")

# ===================================================================
# 2. Are the background jobs alive?
# ===================================================================
#
# Three jobs feed this module and all three have died at least once. None of them raises
# anything when it stops -- they just stop producing rows.

println("\n", "="^90, "\n=== 2. UPSTREAM JOB HEALTH ===\n", "="^90)

health = MD._query("""
SELECT 'identity resolver'   AS job, MAX(kickoff_time)::text AS last_output FROM betfair.match_meta
UNION ALL
SELECT 'order-book drain',        MAX(ts)::text          FROM betfair_live.order_book_1m
UNION ALL
SELECT 'provisional lineups',     MAX(scraped_at)::text  FROM sofascore.lineup_provisional;
""")
show(health, allrows = true, allcols = true); println()

lag = MD._query("""
SELECT COUNT(*) AS live_events,
       COUNT(mm.match_id) AS resolved,
       ROUND(100.0*COUNT(mm.match_id)/NULLIF(COUNT(*),0),1) AS pct
FROM (SELECT DISTINCT event_id FROM betfair_live.market_metadata) l
LEFT JOIN betfair.match_meta mm ON mm.betfair_event_id = l.event_id AND mm.match_id IS NOT NULL;
""")
@printf("\n  identity resolution across the live feed: %s of %s (%s%%)\n",
        lag.resolved[1], lag.live_events[1], lag.pct[1])
println("""
  When the resolver runs it resolves 100%. A low number here is a STOPPED JOB, not a matching
  problem -- do not reach for a fuzzy matcher.""")

# ===================================================================
# 3. Is the lineup a real XI or a guess?
# ===================================================================
#
# `confirmed` has never been true for any match in the table, because every scrape has run
# 4.4-5.8h before kick-off and SofaScore publishes the confirmed XI about an hour out. So
# `confirmed` is useless as a signal and LEAD TIME is what you read instead.

println("\n", "="^90, "\n=== 3. LINEUP FRESHNESS ===\n", "="^90)
for c in res.cards
    lu = c.lineup
    if lu === nothing
        @printf("  %-40s no lineup\n", "$(c.fixture.home) v $(c.fixture.away)")
    else
        @printf("  %-40s %-16s %4.1fh before KO  confirmed=%s\n",
                "$(c.fixture.home) v $(c.fixture.away)", lu.source,
                Dates.value(c.fixture.kickoff - lu.scraped_at) / 3.6e6, lu.confirmed)
    end
end
println("""
  A `last_historical` source means the provisional scrape never ran for that fixture and the
  model is being fed last week's XI. That is not fatal but it is worth knowing.""")

# ===================================================================
# 4. Which split are we conditioning on?  (UNRESOLVED)
# ===================================================================
#
# The chain is picked by position: index `length(training_results)` into a boundary list that is
# REBUILT at inference time and no longer has the same length. `select_split` reports the
# mismatch instead of hiding it, but reporting is not fixing.

println("\n", "="^90, "\n=== 4. SPLIT SELECTION -- KNOWN DEFECT ===\n", "="^90)
bnds = DD.create_id_boundaries(ds, expr.config.splitter)
sel  = MD.select_split(expr, bnds; strict = false)
@printf("  trained splits %d | boundaries today %d | conditioning on %d\n",
        length(expr.training_results), length(bnds), sel.idx)
isempty(sel.warning) ? println("  aligned.") : println("  ", sel.warning)

println("""

  Two problems, one certain and one conditional:
    CERTAIN     the most recent window(s) are not used at all -- today's fixtures are priced off
                a posterior that stops short.
    CONDITIONAL the pairing is only CORRECT if the splitter appends boundaries rather than
                recomputing them. If it recomputes, index N names a different window in the
                chain than in the features: a posterior fitted on one period applied to
                another period's covariates.

  Splits should be named by boundary, not by a positional index into a list whose length changes
  underneath them. This is worth fixing before the module prices anything you actually bet, and
  it may also affect Experiments.extract_oos_predictions.""")

# ===================================================================
# 5. Coverage: is every fixture actually representable?
# ===================================================================
#
# `check_coverage` asserts PER FIXTURE, not per feature. `haskey(fs.data, :player_ratings_map)`
# is true straight out of training and says nothing about today -- a fixture missing from that
# map is priced off an empty Dict, i.e. zero player strength, silently.

println("\n", "="^90, "\n=== 5. FEATURE COVERAGE ===\n", "="^90)
println("  (matchday_latents calls check_coverage internally and throws with the specific")
println("   teams or match_ids missing -- no output here means every fixture was representable.)")

# ===================================================================
# 6. Is the price defensible?
# ===================================================================

println("\n", "="^90, "\n=== 6. PRICING ===\n", "="^90)
if !isempty(res.sheet)
    n_lay = count(==(:lay), res.sheet.side)
    @printf("  legs via lay: %d of %d (%.0f%%)\n", n_lay, nrow(res.sheet),
            100n_lay / nrow(res.sheet))
    @printf("  max leverage used: %.2fx\n",
            maximum(r.side == :lay ? r.venue_stake / r.risk : 1.0 for r in eachrow(res.sheet)))
    @printf("  effective odds never below venue back price: %s\n",
            all(r.side == :lay || r.odds <= r.venue_odds + 1e-9 for r in eachrow(res.sheet)))
end
println("""
  A leverage figure near the cap means the book was thin on that line. Nothing here should ever
  show a lay chosen at WORSE effective odds than the direct back -- BestOfBackLay takes the max,
  so that would be a bug.""")
