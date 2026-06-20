# current_development/order_book/r02_ob_eda.jl
# EDA runner for Betfair order book data
# Assumes ob DataFrame is already loaded (run r01_order_book.jl first)
# Or run standalone — uncomment the data-loading block below

using Plots
gr()   # GR backend — fast, saves to file and opens a window if display available
using Plots
unicodeplots()

include(joinpath(@__DIR__, "l01_order_book.jl"))
include(joinpath(@__DIR__, "l02_ob_plots.jl"))

# ---------------------------------------------------------------------------
# Optional: load data fresh (skip if ob is already in scope from r01)
# ---------------------------------------------------------------------------
# using Pkg; Pkg.activate(".")
# using BayesianFootball, DataFrames, Dates, LibPQ
# db_url    = get(ENV, "BF_DB_URL", "postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
# db_config = BayesianFootball.Data.DBConfig(db_url)
# conn      = BayesianFootball.Data.connect_to_db(db_config)
# raw = try
#     fetch_order_book_raw(conn; date_from=DateTime(2026,5,1), date_to=DateTime(2026,6,1))
# finally
#     close(conn)
# end
# ob = process_order_book(raw)

# ---------------------------------------------------------------------------
# Step 1: Get a lay of the land
# ---------------------------------------------------------------------------

println("=== Available matches and markets ===")
display(list_matches(ob))

# Pick a match to explore — update these to suit what you see above
HOME = "Shamrock Rovers"
AWAY = "St Patricks"

match_df = filter_match(ob, HOME, AWAY)
println("\nMarkets available for $HOME vs $AWAY:")
display(unique(match_df[:, [:market_name, :market_line, :selection]]))

# ---------------------------------------------------------------------------
# Step 2: 1X2 market — all selections mid-price
# ---------------------------------------------------------------------------

fig_1x2_prices = plot_market_prices(ob, HOME, AWAY, "1X2")
fig = plot_market_prices(ob, HOME, AWAY, "1X2"; time_window=(-60, 0))
plot!(fig, extra_kwargs=Dict(:subplot => (; width=1000, height=250, canvas=:block, border=:bold)))
display(fig)

# savefig(fig_1x2_prices, joinpath(@__DIR__, "plots/$(HOME)_$(AWAY)_1X2_prices.png"))
display(fig_1x2_prices)
fig_1x2_prices

# ---------------------------------------------------------------------------
# Step 3: Drill into one selection — price ladder + depth
# ---------------------------------------------------------------------------

fig_home = plot_ob_selection(ob, HOME, AWAY, "1X2", :home)
savefig(fig_home, joinpath(@__DIR__, "plots/$(HOME)_$(AWAY)_1X2_home_ob.png"))
display(fig_home)

fig_away = plot_ob_selection(ob, HOME, AWAY, "1X2", :away)
savefig(fig_away, joinpath(@__DIR__, "plots/$(HOME)_$(AWAY)_1X2_away_ob.png"))
display(fig_away)


plot_ob_selection(ob, HOME, AWAY, "OverUnder", Symbol("over_25 goals"))
# ---------------------------------------------------------------------------
# Step 4: Spread tightening — does liquidity improve pre-kickoff?
# ---------------------------------------------------------------------------

fig_spread = plot_spread(ob, HOME, AWAY, "1X2")
savefig(fig_spread, joinpath(@__DIR__, "plots/$(HOME)_$(AWAY)_1X2_spread.png"))
display(fig_spread)

# ---------------------------------------------------------------------------
# Step 5: OverUnder market (if available for this match)
# Check which lines exist first
# ---------------------------------------------------------------------------

ou_markets = filter(r -> r.market_name == "OverUnder" && r.home_team == HOME, ob)
ou_lines   = sort(unique(ou_markets.market_line))
println("\nOverUnder lines available: $ou_lines")

if !isempty(ou_lines)
    line = ou_lines[1]   # pick the first line, e.g. 2.5
    fig_ou = plot_ob_selection(ob, HOME, AWAY, "OverUnder", Symbol("over_$(Int(line*10))"))
    # savefig(fig_ou, joinpath(@__DIR__, "plots/$(HOME)_$(AWAY)_OU$(line)_ob.png"))
    display(fig_ou)

    fig_ou_spread = plot_spread(ob, HOME, AWAY, "OverUnder")
    # savefig(fig_ou_spread, joinpath(@__DIR__, "plots/$(HOME)_$(AWAY)_OU_spread.png"))
    display(fig_ou_spread)
end

# ---------------------------------------------------------------------------
# Step 6: Quick stats — spread and depth in last 60 mins pre-kickoff
# ---------------------------------------------------------------------------

println("\n=== Market quality: last 60 mins pre-kickoff ($(HOME) vs $(AWAY)) ===")
window = filter(r ->
    r.home_team == HOME &&
    r.away_team == AWAY &&
    r.market_name == "1X2" &&
    r.minutes_to_kickoff >= -60 &&
    r.minutes_to_kickoff <= 0,
    ob)

if !isempty(window)
    for (sel, grp) in pairs(groupby(window, :selection))
        spread = grp.ask_price_1 .- grp.bid_price_1
        depth  = grp.bid_vol_1 .+ grp.ask_vol_1
        @printf("  %-12s  spread: mean=%.4f  min=%.4f    depth L1: mean=£%6.0f\n",
            sel.selection, mean(spread), minimum(spread), mean(depth))
    end
end

