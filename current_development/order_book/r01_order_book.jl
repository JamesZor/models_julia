# current_development/order_book/r01_order_book.jl
# Runner: interactively fetch and explore Betfair order book data
# Run from project root: julia --project current_development/order_book/r01_order_book.jl
# Or paste into REPL after: using Pkg; Pkg.activate(".")

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Dates
using LibPQ

include(joinpath(@__DIR__, "l01_order_book.jl"))

# ---------------------------------------------------------------------------
# DB connection — reuse BayesianFootball.Data types
# ---------------------------------------------------------------------------

db_url    = get(ENV, "BF_DB_URL", "postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
db_config = BayesianFootball.Data.DBConfig(db_url)
conn      = BayesianFootball.Data.connect_to_db(db_config)

ob = try
    # Adjust date range to the period you want to explore
    raw = fetch_order_book_raw(conn;
        date_from = DateTime(2026, 5, 1),
        date_to   = DateTime(2026, 6, 1))

    println("Raw rows fetched: ", nrow(raw))
    isempty(raw) && error("No rows returned — check table names and date range")

    ob = process_order_book(raw)
    println("Processed rows:   ", nrow(ob))
    ob
finally
    close(conn)
end

# ---------------------------------------------------------------------------
# Basic inspection
# ---------------------------------------------------------------------------

println("\nColumns: ", names(ob))
println("\nFirst 5 rows:")
display(first(ob, 5))

println("\nMarket types present:")
display(combine(groupby(ob, [:market_name, :market_line]), nrow => :count))

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

# 1. Price scale: 1X2 home best bid should be in [1.5, 5.0] for most matches
match_odds_home = filter(r -> r.market_name == "1X2" && r.selection == :home, ob)
if !isempty(match_odds_home)
    lo, hi = extrema(skipmissing(filter(!isnan, match_odds_home.bid_price_1)))
    println("\n[Sanity] 1X2 home bid_price_1 range: ($lo, $hi)")
    println("  Expected ~[1.5, 5.0] for typical football odds")
    println("  If values look wrong, adjust _PRICE_SCALE in l01_order_book.jl")
    display(first(match_odds_home[:, [:home_team, :away_team, :ts, :bid_price_1, :ask_price_1, :minutes_to_kickoff]], 5))
end

# 2. minutes_to_kickoff — should be negative for pre-match snapshots
println("\n[Sanity] minutes_to_kickoff range: ", extrema(ob.minutes_to_kickoff))
println("  Expect negative values for pre-match data")

# 3. Correct score symbol mapping
cs = filter(r -> r.market_name == "CorrectScore", ob)
if !isempty(cs)
    println("\n[Sanity] CorrectScore selections (first 10): ", unique(cs.selection)[1:min(10, end)])
end
