# r06_slate_ledger_console.jl
#
# THE WHOLE LIVE LOOP, on one past Scottish match day: canonical fit -> slate -> paper ledger ->
# operator console.
#
#     canonical_fit ─► price_slate ─► insert_slate! ─► execute_slate_batch! ─► submit_slate!
#                          │                                                        │
#                          └────────────► serve_console ◄─── slate_snapshot ────────┘
#                                                                                   │
#                                                                          settle_slate!
#
# Deliberately does NOT include `_setup.jl`. That file targets Ireland Premier (79) with
# `src_sup40_sw40` off a filesystem experiment tree; this runner targets ScottishLower (56, 57)
# with a canonical fit read from `mcmc_experiments`, which is the seam Phase 1 added. Two
# leagues, two engines, two storage backends -- sharing a setup would hide all three differences.
#
# Needs: `.cache/datastore_ScottishLower.jls`, `BF_DB_URL` (betdb), and reachability of
# `mcmc-beast:5432` via `~/.pgpass`. Every step says what it needs before it does anything.

# ===================================================================
# 1. Packages
# ===================================================================
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, UUIDs
import LibPQ

const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const TT = BayesianFootball.Training

# ===================================================================
# 2. Configuration
# ===================================================================
#
# `AS_OF` is T-25 on the 2026-08-08 14:00 UTC card -- the instant the blueprint recommends
# pricing at, because it is the first moment the confirmed XI exists (the scraper lands it at
# T-13..T-42) and still 13 minutes clear of the T-12 submission target.
#
# The order-book collector covered this card from T-120, so it is one of the few Scottish match
# days that can be replayed at all. See RESEARCH_MATCHDAY_ARCHITECTURE.md §4.3.

const AS_OF     = DateTime(2026, 8, 8, 13, 35)
const KO_DAY    = Date(2026, 8, 8)
const BANKROLL  = 2_400.0
const ACCOUNT   = "runbook_scottish"

# `scottish_lower_poisson_2426` looks like the obvious choice and is NOT: every one of its runs
# stores a two-parameter SYNTHETIC chain (`synthetic_intercept`, `synthetic_home_advantage`) from
# a smoke test, so `extract_parameters` fails on a missing `inter.μ`. The runs carrying a real
# 50-parameter posterior are in `scottish_lower_joint_2426` and `scottish_lower_player_grid_2426`.
# Checked with `canonical_fit(...).fit.folds[1].chain |> names`; do that before trusting a name.
const EXPERIMENT = "scottish_lower_joint_2426"
const RUN_NAME   = "m00_poisson_control"

# Paper, not production: this runner writes real rows, and it should not write them into the
# ledger a real Saturday is using.
const SCHEMA = "paper_runbook"

# ===================================================================
# 3. Data
# ===================================================================
@info "loading ScottishLower DataStore (uses .cache/ if warm)"
ds = DD.load_datastore_cached(DD.ScottishLower())
@info "datastore" matches = nrow(ds.matches)

# Replay needs an EXPLICIT fixture list. `SofaScoreEvents` filters `status_type = 'notstarted'`,
# which is false for a match that has since been played -- so on a replay it returns nothing and
# the whole thing looks like a quiet Saturday rather than a wrong query.
fixtures = let c = MD.MatchDay._conn()
    df = DataFrame(LibPQ.execute(c, """
        SELECT match_id, home_team, away_team, start_timestamp, tournament_id
        FROM sofascore.events
        WHERE tournament_id = ANY(\$1) AND start_timestamp >= \$2 AND start_timestamp < \$3
        ORDER BY start_timestamp;""",
        ([56, 57], Int(round(datetime2unix(DateTime(KO_DAY)))),
                   Int(round(datetime2unix(DateTime(KO_DAY) + Day(1)))))))
    close(c)
    MD.Fixture[MD.Fixture(Int(r.match_id), String(r.home_team), String(r.away_team),
                          unix2datetime(r.start_timestamp), Int(r.tournament_id))
               for r in eachrow(df)]
end
@info "fixtures on the card" n = length(fixtures)

# ===================================================================
# 4. The spec and the staking system
# ===================================================================
#
# Three departures from `MatchDaySpec`'s defaults, each measured rather than chosen:
#
#   * `ResolverChain(MatchMetaCrosswalk, LiveNameMatch)` -- the crosswalk job resolves 100% when
#     it runs and 0% when it does not, and it has not run since 2026-08-29. The fallback is
#     opt-in precisely so a dead crosswalk stays visible in the reason field.
#   * `MaxBookAge(Minute(10))` rather than 30 -- at T-25 against a 1-minute drain, a 10-minute-old
#     book means the drain has stalled, which is the most likely thing to be wrong on any given
#     Saturday.
#   * `MaxSpread(0.08)` -- the gate that catches Scottish League Two BTTS: a 9-tick book with
#     ~£170 resting and ~£25 ever matched. `MinMatched` alone waves it through.

spec = MD.MatchDaySpec(
    fixtures   = MD.ExplicitFixtures(fixtures),
    identity   = MD.ResolverChain(MD.MatchMetaCrosswalk(), MD.LiveNameMatch()),
    lineups    = MD.SourceChain(MD.ProvisionalDB(), MD.LastHistorical(ds)),
    book       = MD.ArchivedOrderBook(max_age = Hour(2)),
    instrument = MD.BestOfBackLay(),
    rounding   = MD.FloorOrDrop(minimum = 1.0),
    gate       = MD.GateChain(MD.IdentityResolved(),
                              MD.MaxBookAge(Minute(10)),
                              MD.MaxSpread(0.08),
                              MD.MinMatched(minimum = 20.0)),
    markets    = MD.canonical_markets())

# SlateDrawdown(20.0) is λ = 20, NOT a 20% drawdown: λ = log β / log D, and the default λ = 23
# targets a real 20% floor because realised drawdown overshoots the nominal by ~1.15x. λ = 20 is
# therefore slightly MORE aggressive than a 20% floor. Deliberate is fine; accidental is not.
sys = PF.PortfolioSystem(
    PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb()),
    PF.PolicySpec(risk = PF.SlateDrawdown(20.0), cap = PF.FixedCap(0.25),
                  trust = PF.FlatTrust(1.0)))

# ===================================================================
# 5. The canonical fit
# ===================================================================
cf = MD.canonical_fit(TT.PostgresStorage(EXPERIMENT), RUN_NAME)
MD.matchday_fit_report(cf)

# ===================================================================
# 6. Pricing the slate
# ===================================================================
#
# EXPECT A REFUSAL on the full card. Ross County and Airdrieonians moved into League One for
# 26/27 and are absent from the chosen fold's `team_map`, so `check_coverage` refuses rather than
# pricing them at league average. That is the guard working, not a failure -- and it is the
# reason this runner catches it and reports rather than wrapping it in a filter and moving on.

slate = try
    MD.price_slate(spec, sys, DD.ScottishLower(), cf, ds;
                   as_of = AS_OF, bankroll = BANKROLL, account_id = ACCOUNT)
catch e
    println("\nREFUSED, correctly:\n  ", sprint(showerror, e), "\n")
    unknown = ["ross-county", "airdrieonians"]
    covered = [f for f in fixtures if !(f.home in unknown || f.away in unknown)]
    @info "re-pricing on the fixtures this fold covers" n = length(covered)
    spec2 = MD.MatchDaySpec(fixtures = MD.ExplicitFixtures(covered), identity = spec.identity,
                            lineups = spec.lineups, book = spec.book,
                            instrument = spec.instrument, rounding = spec.rounding,
                            gate = spec.gate, markets = spec.markets)
    MD.price_slate(spec2, sys, DD.ScottishLower(), cf, ds;
                   as_of = AS_OF, bankroll = BANKROLL, account_id = ACCOUNT)
end

# EXPOSURE FIRST, ALWAYS. A sheet is a list of attractive-looking prices; the only number that
# says whether the vector is safe to commit is what fraction of the bankroll settles at once and
# whether FixedCap had to bind to get it there.
println("\n=== BATCH HEADER ===")
for (k, v) in pairs(MD.slate_batch_summary(slate))
    println("  ", rpad(k, 18), v)
end

println("\n=== TOP LEGS BY RISK ===")
show(stdout, MIME"text/plain"(),
     first(sort(slate.sheet, :risk, rev = true), 10)[:,
        [:match_id, :group, :line, :selection, :side, :venue_selection, :venue_odds,
         :p_model, :p_market, :edge, :risk, :venue_stake, :depth_touch, :fill_confidence]])

println("\n\n=== BLOCKED (read this before concluding there were no bets) ===")
show(stdout, MIME"text/plain"(),
     MD.blocked_report(MD.MatchDayResult(slate.sheet, slate.cards, slate.blocked,
                                         slate.odds, slate.instruments, slate.as_of)))

# ===================================================================
# 7. The paper ledger
# ===================================================================
conn = MD.paper_connection()
MD.migrate_paper_schema!(conn; schema = SCHEMA)
MD.ensure_account!(conn, MD.PaperAccount(account_id = ACCOUNT, opening_balance = BANKROLL,
                                         balance = BANKROLL, max_slate_exposure = 0.25);
                   schema = SCHEMA)

# `insert_slate!` returns the id the DATABASE holds. On a re-run of the same `as_of` that is the
# existing slate, not the one we just generated -- which is exactly what stops a restarted
# match-day process from double-staking.
slate_id = MD.insert_slate!(conn, slate; schema = SCHEMA, run_name = RUN_NAME)
orders   = MD.orders_to_paper(slate; slate_id = slate_id)
MD.insert_orders!(conn, orders; schema = SCHEMA)
@info "ledger: slate written" slate_id n_orders = length(orders)

# THE ATOM. One SELECT ... FOR UPDATE, one exposure assert, the whole vector or nothing.
res = MD.execute_slate_batch!(conn, ACCOUNT, slate_id; schema = SCHEMA)
println("\n=== RESERVATION ===")
println("  status    ", res.status)
println("  reserved  ", @sprintf("%.2f", res.reserved))
println("  admitted  ", res.n_admitted, "   refused ", res.n_refused)
println("  balance   ", @sprintf("%.2f", res.account.balance),
        "   reserved ", @sprintf("%.2f", res.account.reserved))
isempty(res.reason) || println("  reason    ", res.reason)

# Submission is embarrassingly parallel AFTER the reservation, because the submitters touch no
# account row. `TouchOnly` is the honest fill model: it takes what is resting at the touch and
# lets the rest expire, which is what resting an order actually does.
fills = MD.submit_slate!(conn, slate_id, slate.books, MD.TouchOnly(); schema = SCHEMA)
println("\n=== EXECUTION (TouchOnly) ===")
println("  matched ", fills.n_matched, "  partial ", fills.n_partial,
        "  unfilled ", fills.n_unfilled)
println("  risk filled ", @sprintf("%.2f", fills.risk_filled),
        " of ", @sprintf("%.2f", slate.total_risk))

rec = MD.reconcile_account(conn, ACCOUNT; schema = SCHEMA)
println("\n=== RECONCILIATION === ", rec.ok ? "OK" : "FAILED")
rec.ok || @error "the ledger does not explain the balance" rec

# ===================================================================
# 8. Settlement
# ===================================================================
results = let c = MD.MatchDay._conn()
    df = DataFrame(LibPQ.execute(c, """
        SELECT match_id, home_score, away_score FROM sofascore.matches
        WHERE match_id = ANY(\$1) AND home_score IS NOT NULL;""",
        ([Int(m) for m in unique(slate.sheet.match_id)],)))
    close(c)
    Dict{Int,Tuple{Int,Int}}(Int(r.match_id) => (Int(r.home_score), Int(r.away_score))
                             for r in eachrow(df))
end
@info "results available" n = length(results)

if !isempty(results)
    st = MD.settle_slate!(conn, slate_id, results; schema = SCHEMA)
    println("\n=== SETTLEMENT ===")
    println("  settled ", st.n_settled, " legs   net PnL ", @sprintf("%+.2f", st.total_pnl))
    println("  balance ", @sprintf("%.2f", st.account.balance),
            "   reserved ", @sprintf("%.2f", st.account.reserved))
    println("  reconciles: ", MD.reconcile_account(conn, ACCOUNT; schema = SCHEMA).ok)
end

# ===================================================================
# 9. The operator console
# ===================================================================
#
# Loopback only. This page can commit a slate; reach it over an SSH tunnel or a tailnet rather
# than binding it to whatever network the machine happens to be on.
#
#   ssh -N -L 8080:127.0.0.1:8080 archpc      then open http://127.0.0.1:8080

account = MD.account_row(conn, ACCOUNT; schema = SCHEMA)
status  = MD._parse_batch(String(first(MD.slate_row(conn, slate_id; schema = SCHEMA)).batch_status))

state = MD.ConsoleState(
    () -> MD.slate_snapshot(slate, MD.account_row(conn, ACCOUNT; schema = SCHEMA);
                            status = status);
    # The browser is not in the trust path: it POSTs an INTENT and the server performs the same
    # transaction a script would.
    on_execute = () -> begin
        r = MD.execute_slate_batch!(conn, ACCOUNT, slate_id; schema = SCHEMA)
        (ok = r.status === MD.RESERVED, note = "reserved $(r.n_admitted) legs", error = r.reason)
    end,
    on_kill = () -> begin
        r = MD.kill_slate!(conn, slate_id; schema = SCHEMA)
        (ok = true, note = "killed, released $(abs(r.reserved))")
    end)

# Uncomment to serve. Left commented so a scripted run of this file terminates.
# MD.serve_console(state; port = 8080)
# @info "console on http://127.0.0.1:8080 -- MD.stop_console!(state) to stop"

println("\n=== CONSOLE SNAPSHOT (what the page renders) ===")
snap = state.snapshot()
println("  batch  : ", snap.batch.n_legs, " legs over ", snap.batch.n_fixtures, " fixtures, ",
        snap.batch.exposure_pct, "% of ", snap.batch.cap_pct, "% cap, k=", snap.batch.k_risk)
println("  account: equity ", snap.account.equity, "  reserved ", snap.account.reserved)
println("  cards  : ", length(snap.cards), ", sorted by EV%:")
for c in first(snap.cards, 5)
    @printf("    %-28s %+6.2f%%  £%6.2f  %d legs\n",
            c.home * " v " * c.away, c.ev_pct, c.risk, c.n_legs)
end
println("  blocked: ", length(snap.blocked))

close(conn)
@info "r06 complete" slate_id schema = SCHEMA
