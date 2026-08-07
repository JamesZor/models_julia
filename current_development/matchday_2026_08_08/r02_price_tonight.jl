# current_development/matchday_2026_08_08/r02_price_tonight.jl
#
# Price a live card and emit a paper stake sheet.
#
#   julia --project -t 16 current_development/matchday_2026_08_08/r02_price_tonight.jl
#
# PAPER ONLY. Nothing here places an order; `order_ticket` rows are written to CSV so they can be
# graded after the fact against the result.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS RUN DISCOVERED (2026-08-07, and why the spec below is not the default one)
# ---------------------------------------------------------------------------------------------
#
# 1. THE CROSSWALK WAS DEAD, AND WAS REPAIRED MID-RUN. At 16:09 UTC `betfair.match_meta` held 0
#    rows for all 9 of tonight's fixtures (the job stopped 2026-06-27), so every fixture came
#    back `:absent_from_crosswalk`, `IdentityResolved()` blocked the whole card and the sheet was
#    empty. By 16:31 the backfill had run: 9/9 rows, all `is_verified`, 11 markets each.
#
#    `ResolverChain(MatchMetaCrosswalk(), LiveNameMatch())` is kept anyway. The crosswalk is the
#    authority and now answers first, so `LiveNameMatch` should never fire -- the count warned
#    at the end of each segment is the check that it didn't. It exists so that the next time the
#    job stops, the card degrades to a name match instead of to silence.
#
# 2. THE COLLECTOR ONLY CARRIES TODAY. `betfair_live.market_metadata` had markets for tonight's
#    9 events and none of Saturday's 22. A weekend card therefore has to be priced one match day
#    at a time -- re-run this file on Saturday morning and again on Sunday. `horizon = Hour(6)`
#    below keeps the fixture list and the market list in agreement, so Saturday's fixtures are
#    not pulled in tonight only to be blocked for having no book.
#
# 3. `LastHistorical()` NEEDS THE DATASTORE. The default spec's chain is
#    `SourceChain(ProvisionalDB(), LastHistorical())`, and the no-argument constructor sets
#    `ds = nothing`, which makes the fallback return `nothing` unconditionally. Tonight the 3
#    Premier Division fixtures have a provisional XI (22 rows each) and the 5 First Division
#    fixtures have none -- so with the default chain those 5 get no lineup, `RatingsFromTracker`
#    skips them, and `check_coverage` aborts the WHOLE segment. `LastHistorical(ds)` is not
#    optional here.
#
# 4. THE XI IS A PREDICTION, AND ONLY FOR SOME FIXTURES. At T-2.8h only tournament 79 had one
#    (3 fixtures); 718 and 55 returned nothing because SofaScore had not published theirs yet,
#    not because they were not requested. `confirmed` has still never been true in this table,
#    so `MaxLineupAge` is reported rather than blocking and the 718 fixtures are priced off
#    `LastHistorical` -- last completed XI, i.e. last week's team. A T-60min poller is armed, so
#    a re-run after ~17:30 UTC should have strictly better lineups than this one.
#
# 5. `market_matched` IS NOW POPULATED, so `MinMatched` has teeth for the first time. It is
#    stored x10000 and `book.jl` already unscales it, so the 500.0 below is a real GBP 500.
#    Left non-blocking on this run: it has never actually been exercised against live data, and
#    finding out what it would have refused is more useful than having it refuse it.

using BayesianFootball
using DataFrames, Dates, CSV, Statistics
using ThreadPinning

pinthreads(:cores)

const Data      = BayesianFootball.Data
const E         = BayesianFootball.Experiments
const MD        = BayesianFootball.MatchDay
const PF        = BayesianFootball.Portfolio

const EXP_DIR   = "./data/matchday_wknd_0808/"
const OUT_DIR   = "./data/paper_trades/"
mkpath(OUT_DIR)

# `now(UTC)`, never `now()`. Kick-offs come from `unix2datetime` and `open_date` is a UTC
# timestamptz; a BST local clock would put every gate an hour out and silently drop the 18:45s.
const AS_OF     = now(UTC)
const BANKROLL  = 1000.0

@info "match day" as_of = AS_OF bankroll = BANKROLL

# ===================================================================
# 1. Spec -- see the header for why each override exists
# ===================================================================
function spec_for(ds)
    MD.MatchDaySpec(
        fixtures   = MD.SofaScoreEvents(horizon = Hour(6)),
        identity   = MD.ResolverChain(MD.MatchMetaCrosswalk(), MD.LiveNameMatch()),
        lineups    = MD.SourceChain(MD.ProvisionalDB(), MD.LastHistorical(ds)),
        gate       = MD.GateChain(MD.IdentityResolved(),
                                  MD.MaxBookAge(Minute(30)),
                                  MD.MinMatched(500.0, false),
                                  MD.MaxLineupAge(Hour(8), false)))
end

# ===================================================================
# 2. Staking policy
# ===================================================================
# Every number here is carried over from the staking work rather than chosen fresh:
#
# * FlatTrust(0.5)  -- the simulation's verdict was to stake at a fixed w = 0.5 and keep a
#                      learned trust model only as a junk-model alarm. Every attempt to LEARN
#                      per-selection trust lost money out of sample.
# * FixedCap(0.25)  -- the portfolio cap on simultaneous exposure is the dominant lever in the
#                      whole system. Independent per-bet Kelly went bankrupt on the same book.
# * NO MinEdge     -- deliberately absent. A flat probability floor is not scale-free: 0.03
#                      points is ~3.5% of a fair stake's return at odds 1.16 and ~20% at 6.60,
#                      so it demands most edge exactly where the price is longest. Worse, on a
#                      COMPRESSED model it selects for the fixtures whose market sits furthest
#                      from the model's near-constant output -- which is how tournament 718 came
#                      to hold 64% of the risk on the 2026-08-07 card. It amplified the bias it
#                      should have filtered. Kelly already sizes a small edge small; that is the
#                      principled version of the same intent, so the floor is redundant as well
#                      as mis-shaped. If a floor is ever wanted again, make it EV-based
#                      (p_model*odds - 1) so it is scale-free.
# * MarketWhitelist -- totals and BTTS only. On the one out-of-sample test available, per-line
#                      curation put trust at ~0 on 1X2 and ~0.5 on totals/BTTS, and the 1X2
#                      family bled. Set CURATED = false to price the full book and see for
#                      yourself; that is the more informative run on a smoke test, and the
#                      curated one is the one you would actually stake.
const CURATED = true

const TOTALS_BTTS = Set{Tuple{String,Float64,Symbol}}(
    vcat([("BTTS", 0.0, :btts_yes), ("BTTS", 0.0, :btts_no)],
         [("OverUnder", i + 0.5, Symbol("over_",  replace(string(i + 0.5), "." => ""))) for i in 0:4],
         [("OverUnder", i + 0.5, Symbol("under_", replace(string(i + 0.5), "." => ""))) for i in 0:4]))

filt = CURATED ? PF.MarketWhitelist(TOTALS_BTTS) : PF.KeepAll()

const SYS = PF.PortfolioSystem(
    PF.BookSpec(markets = MD.MatchDaySpec().markets),
    PF.PolicySpec(trust  = PF.FlatTrust(0.5),
                  risk   = PF.SlateDrawdown(lambda = 23.0, mode = :sequential),
                  cap    = PF.FixedCap(0.25),
                  filter = filt))

# ===================================================================
# 3. Run each segment
# ===================================================================
# One `match_day` call per segment because a call takes one DataStore, one experiment and one
# segment. Tonight that is IrelandAll (8 fixtures) and ScottishUpper (1); on Saturday the
# ScottishLower funnel run joins them and needs no lineups at all.
const RUNS = [
    (name = "IrelandAll",    seg = Data.IrelandAll(),
     path = EXP_DIR * "ire_pooled_poisson_outfield_20260807_001923"),
    (name = "ScottishUpper", seg = Data.ScottishUpper(),
     path = EXP_DIR * "scot_upper_poisson_outfield_20260807_011126"),
    (name = "ScottishLower", seg = Data.ScottishLower(),
     path = EXP_DIR * "scot_lower_funnel_20260807_012812"),
]

sheets  = DataFrame[]
blocked = DataFrame[]

for r in RUNS
    @info "=== $(r.name) ==="
    try
        ds   = Data.load_datastore_cached(r.seg)
        expr = E.load_experiment(r.path)
        res  = MD.match_day(spec_for(ds), SYS, r.seg, expr, ds;
                            as_of = AS_OF, bankroll = BANKROLL)

        @info "$(r.name) cards" total = length(res.cards) blocked = length(res.blocked) sheet_rows = nrow(res.sheet)

        # A refusal is a value. Read this BEFORE concluding there were no bets: an empty sheet
        # because the gate refused everything and an empty sheet because the model found no edge
        # are the same DataFrame otherwise.
        br = MD.blocked_report(res)
        isempty(br) || (println("--- blocked: $(r.name) ---"); show(br, allrows = true, allcols = true); println())
        isempty(br) || push!(blocked, insertcols!(br, 1, :segment => r.name))

        if !isempty(res.sheet)
            push!(sheets, insertcols!(copy(res.sheet), 1, :segment => r.name))
        end

        # Which identities came from the fallback rather than the crosswalk. `verified` is false
        # for every name-matched fixture by construction, which is exactly what makes them
        # countable here.
        nm = count(c -> c.identity isa MD.Resolved && !c.identity.verified, res.cards)
        nm == 0 || @warn "$(r.name): $nm fixture(s) resolved by LiveNameMatch, not the crosswalk"
    catch e
        @error "FAILED $(r.name)" exception = (e, catch_backtrace())
    end
end

# ===================================================================
# 4. Paper sheet
# ===================================================================
if isempty(sheets)
    @warn "no stake sheet produced -- check the blocked report above before assuming 'no edge'"
else
    sheet = reduce(vcat, sheets)
    stamp = Dates.format(AS_OF, "yyyymmdd_HHMM")

    println("\n=== PAPER STAKE SHEET  ($(nrow(sheet)) legs, bankroll $(BANKROLL)) ===")
    show(sheet, allrows = true, allcols = true); println()

    @info "book summary" legs = nrow(sheet) total_risk = round(sum(sheet.risk), digits = 2) exposure_pct =
        round(100 * sum(sheet.risk) / BANKROLL, digits = 2) families = combine(groupby(sheet, :group), nrow => :n)

    CSV.write(joinpath(OUT_DIR, "sheet_$(stamp).csv"), sheet)

    tickets = DataFrame([MD.order_ticket(row) for row in eachrow(sheet)])
    CSV.write(joinpath(OUT_DIR, "tickets_$(stamp).csv"), tickets)
    println("\n=== ORDER TICKETS (paper) ===")
    show(tickets, allrows = true, allcols = true); println()

    @info "written" sheet = joinpath(OUT_DIR, "sheet_$(stamp).csv") tickets = joinpath(OUT_DIR, "tickets_$(stamp).csv")
end

isempty(blocked) || CSV.write(joinpath(OUT_DIR, "blocked_$(Dates.format(AS_OF, "yyyymmdd_HHMM")).csv"),
                              reduce(vcat, blocked))

# ===================================================================
# 5. What to check before trusting the sheet
# ===================================================================
# * blocked report empty?  If not, the reason distinguishes a dead feed from a quiet market.
# * how many fixtures came from LiveNameMatch rather than the crosswalk (warned above)?
# * `slate_exposure` at or near 0.25 means the cap bound and the sheet is cap-shaped, not
#   edge-shaped -- the ordering is still information but the sizes are not.
# * `p_model` vs `p_market`: an edge above ~0.15 on a thin exchange line is far more likely to
#   be a stale quote than a real disagreement.
