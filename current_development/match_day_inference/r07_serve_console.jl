# r07_serve_console.jl
#
# Standalone operator console for replaying the 2026-08-08 Scottish Lower slate.
# It binds all interfaces on port 8085 so an operator on the LAN can open the console.
# This is paper-ledger only; execution and kill requests retain the ledger transactions
# enforced by MatchDay rather than trusting the browser.

# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball
using DataFrames, Dates
import LibPQ

const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const TT = BayesianFootball.Training

# ===================================================================
# 2. Configuration
# ===================================================================
const R07_AS_OF = DateTime(2026, 8, 8, 13, 35)
const R07_KO_DAY = Date(2026, 8, 8)
const R07_BANKROLL = 2_400.0
const R07_ACCOUNT = "runbook_scottish"
const R07_EXPERIMENT = "scottish_lower_joint_2426"
const R07_RUN_NAME = "m00_poisson_control"
const R07_SCHEMA = "paper_runbook"
const R07_HOST = "0.0.0.0"
const R07_PORT = 8085

# ===================================================================
# 3. Data snapshot and explicit replay fixtures
# ===================================================================
@info "loading ScottishLower DataStore (uses .cache/ if warm)"
ds = DD.load_datastore_cached(DD.ScottishLower())

# Replays require explicit fixtures: SofaScore's normal live query only returns matches
# still marked `notstarted`, which this historical card no longer is.
fixtures = let conn = MD.MatchDay._conn()
    frame = DataFrame(LibPQ.execute(conn, """
        SELECT match_id, home_team, away_team, start_timestamp, tournament_id
        FROM sofascore.events
        WHERE tournament_id = ANY(\$1) AND start_timestamp >= \$2 AND start_timestamp < \$3
        ORDER BY start_timestamp;""",
        ([56, 57], Int(round(datetime2unix(DateTime(R07_KO_DAY)))),
                   Int(round(datetime2unix(DateTime(R07_KO_DAY) + Day(1)))))))
    close(conn)
    MD.Fixture[MD.Fixture(Int(row.match_id), String(row.home_team), String(row.away_team),
                          unix2datetime(row.start_timestamp), Int(row.tournament_id))
               for row in eachrow(frame)]
end
@info "fixtures on the card" n = length(fixtures)

# ===================================================================
# 4. Match-day specification and portfolio policy
# ===================================================================
spec = MD.MatchDaySpec(
    fixtures = MD.ExplicitFixtures(fixtures),
    identity = MD.ResolverChain(MD.MatchMetaCrosswalk(), MD.LiveNameMatch()),
    lineups = MD.SourceChain(MD.ProvisionalDB(), MD.LastHistorical(ds)),
    book = MD.ArchivedOrderBook(max_age = Hour(2)),
    instrument = MD.BestOfBackLay(),
    rounding = MD.FloorOrDrop(minimum = 1.0),
    gate = MD.GateChain(MD.IdentityResolved(), MD.MaxBookAge(Minute(10)),
                        MD.MaxSpread(0.08), MD.MinMatched(minimum = 20.0)),
    markets = MD.canonical_markets())

system = PF.PortfolioSystem(
    PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb()),
    MD.canonical_scottish_lower_policy())

# ===================================================================
# 5. Canonical fit, coverage, and slate pricing
# ===================================================================
canonical = MD.canonical_fit(TT.PostgresStorage(R07_EXPERIMENT), R07_RUN_NAME)
MD.matchday_fit_report(canonical)

# `price_slate` materialises feature maps then calls `check_coverage` before posterior
# extraction, refusing fixtures that cannot be represented by the selected trained fold.
slate = try
    MD.price_slate(spec, system, DD.ScottishLower(), canonical, ds;
                   as_of = R07_AS_OF, bankroll = R07_BANKROLL, account_id = R07_ACCOUNT)
catch error
    println("\nInitial coverage refusal:\n  ", sprint(showerror, error), "\n")
    unknown = ["ross-county", "airdrieonians"]
    covered = [fixture for fixture in fixtures if !(fixture.home in unknown || fixture.away in unknown)]
    @info "re-pricing only covered fixtures" n = length(covered)
    covered_spec = MD.MatchDaySpec(fixtures = MD.ExplicitFixtures(covered), identity = spec.identity,
                                   lineups = spec.lineups, book = spec.book,
                                   instrument = spec.instrument, rounding = spec.rounding,
                                   gate = spec.gate, markets = spec.markets)
    MD.price_slate(covered_spec, system, DD.ScottishLower(), canonical, ds;
                   as_of = R07_AS_OF, bankroll = R07_BANKROLL, account_id = R07_ACCOUNT)
end

# ===================================================================
# 6. Paper ledger
# ===================================================================
conn = MD.paper_connection()
MD.migrate_paper_schema!(conn; schema = R07_SCHEMA)
MD.ensure_account!(conn, MD.PaperAccount(account_id = R07_ACCOUNT,
                                         opening_balance = R07_BANKROLL,
                                         balance = R07_BANKROLL,
                                         max_slate_exposure = 0.25);
                   schema = R07_SCHEMA)

slate_id = MD.insert_slate!(conn, slate; schema = R07_SCHEMA, run_name = R07_RUN_NAME)
orders = MD.orders_to_paper(slate; slate_id = slate_id)
MD.insert_orders!(conn, orders; schema = R07_SCHEMA)
@info "paper slate ready" slate_id n_orders = length(orders)

# ===================================================================
# 7. Operator console
# ===================================================================
status = MD._parse_batch(String(first(MD.slate_row(conn, slate_id; schema = R07_SCHEMA)).batch_status))
state = MD.ConsoleState(
    () -> MD.slate_snapshot(slate, MD.account_row(conn, R07_ACCOUNT; schema = R07_SCHEMA);
                            status = status),
    on_execute = () -> begin
        result = MD.execute_slate_batch!(conn, R07_ACCOUNT, slate_id; schema = R07_SCHEMA)
        (ok = result.status === MD.RESERVED,
         note = "Reserved $(result.n_admitted) legs", error = result.reason)
    end,
    on_kill = () -> begin
        MD.kill_slate!(conn, slate_id; schema = R07_SCHEMA)
        (ok = true, note = "Killed slate", error = nothing)
    end)

MD.serve_console(state; host = R07_HOST, port = R07_PORT)
println("\n" * "="^70)
println("  MatchDay Slate Console is LIVE!")
println("  Local URL : http://localhost:8085")
println("  LAN URL   : http://192.168.1.88:8085")
println("  Press Ctrl+C to stop the server.")
println("="^70 * "\n")

try
    wait()
catch error
    error isa InterruptException || rethrow()
finally
    MD.stop_console!(state)
    close(conn)
    println("\nConsole stopped.")
end
