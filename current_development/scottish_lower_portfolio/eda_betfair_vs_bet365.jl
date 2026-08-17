# current_development/scottish_lower_portfolio/eda_betfair_vs_bet365.jl
#
# EXPLORATORY DATA ANALYSIS: Betfair Exchange vs Bet365 (SofaScore) Odds
# Target: Scottish Lower Leagues (56 League One, 57 League Two)
#
# Analyses:
# 1. Season-by-Season Match Coverage (Bet365 vs Betfair)
# 2. Market-Level Liquidity & Tick Depth in Closing Window [-20min, 0min] & [-60min, 0min]
# 3. Vig / Overround Breakdown (How much margin does Bet365 extract vs Betfair + 2% comm?)
# 4. Direct Quoted Price & Yield Comparison (Betfair Net vs Bet365 Decimal Odds)
# 5. Hybrid Exchange-First Fallback Routing Feasibility

using BayesianFootball
using DataFrames, Dates, Statistics, Printf

const DD = BayesianFootball.Data

println("="^95)
println("EDA: BETFAIR EXCHANGE VS BET365 ODDS ANALYSIS (Scottish Lower)")
println("="^95)

# 1. Load DataStore
ds = DD.load_datastore_cached(DD.ScottishLower(), max_age_hours = 720)
matches_df = ds.matches
b365_df    = ds.odds
bf_long    = ds.betfair_odds

n_matches_total = nrow(matches_df)
@info "Loaded ScottishLower DataStore" n_matches=n_matches_total n_b365_rows=nrow(b365_df) n_bf_ticks=nrow(bf_long)

# -------------------------------------------------------------------
# SECTION 1: SEASON-BY-SEASON MATCH COVERAGE
# -------------------------------------------------------------------
println("\n", "="^95)
println("SECTION 1: SEASON-BY-SEASON MATCH & MARKET COVERAGE")
println("="^95)

# Attach season label to matches
function _season_label(dt::TimeType)
    y = year(dt)
    m = month(dt)
    return m >= 7 ? "$(y % 100)/$((y+1) % 100)" : "$((y-1) % 100)/$(y % 100)"
end

matches_df.season_str = _season_label.(matches_df.match_date)

bf_match_ids = Set(bf_long.match_id)
b365_match_ids = Set(b365_df.match_id)

coverage_by_season = combine(groupby(matches_df, :season_str)) do sub
    n_tot = nrow(sub)
    n_b365 = count(id in b365_match_ids for id in sub.match_id)
    n_bf   = count(id in bf_match_ids for id in sub.match_id)
    (
        total_matches = n_tot,
        b365_matches  = n_b365,
        b365_pct      = round(100 * n_b365 / n_tot, digits = 1),
        bf_matches    = n_bf,
        bf_pct        = round(100 * n_bf / n_tot, digits = 1)
    )
end
sort!(coverage_by_season, :season_str)
show(coverage_by_season; allrows = true, allcols = true, truncate = 0)
println()

# -------------------------------------------------------------------
# SECTION 2: BETFAIR MARKET-LEVEL LIQUIDITY IN CLOSING WINDOWS
# -------------------------------------------------------------------
println("\n", "="^95)
println("SECTION 2: BETFAIR CLOSING LIQUIDITY & TICK COUNTS")
println("="^95)

# Analyze ticks in closing windows: [-20min, 0min] and [-60min, 0min]
bf_close20 = filter(r -> -20.0 <= r.minutes_to_kickoff <= 0.0, bf_long)
bf_close60 = filter(r -> -60.0 <= r.minutes_to_kickoff <= 0.0, bf_long)

println("Total Betfair Ticks in Dataset       : $(nrow(bf_long))")
println("Ticks in Close Window [-60min, 0min] : $(nrow(bf_close60)) ($(round(100 * nrow(bf_close60)/nrow(bf_long), digits=1))%)")
println("Ticks in Close Window [-20min, 0min] : $(nrow(bf_close20)) ($(round(100 * nrow(bf_close20)/nrow(bf_long), digits=1))%)")

# Group by market family
market_liq = combine(groupby(bf_close20, :market_name)) do sub
    n_matches_active = length(unique(sub.match_id))
    (
        n_ticks_20m       = nrow(sub),
        matches_active    = n_matches_active,
        mean_ticks_match  = round(nrow(sub) / max(1, n_matches_active), digits = 1),
        median_ticks_match= round(median(combine(groupby(sub, :match_id), nrow => :n).n), digits = 1)
    )
end
show(market_liq; allrows = true, allcols = true, truncate = 0)
println()

# -------------------------------------------------------------------
# SECTION 3: VIG / OVERROUND COMPARISON (Bet365 vs Betfair)
# -------------------------------------------------------------------
println("\n", "="^95)
println("SECTION 3: VIG & OVERROUND COMPARISON (How much margin does Bet365 take?)")
println("="^95)

# Summarize Betfair markets in closing window
bf_summary_20m = DD.summarize_betfair_market(ds; open_window=(-1440.0, -1380.0), close_window=(-20.0, 0.0))
bf_summary_60m = DD.summarize_betfair_market(ds; open_window=(-1440.0, -1380.0), close_window=(-60.0, 0.0))

# Compare Overround by Market on Bet365
b365_valid = filter(r -> !ismissing(r.overround_close) && r.overround_close > 1.0, b365_df)
b365_vig_by_mkt = combine(groupby(b365_valid, :market_name)) do sub
    (
        n_quotes        = nrow(sub),
        mean_overround  = round(mean(sub.overround_close), digits = 4),
        mean_vig_pct    = round((mean(sub.overround_close) - 1.0) * 100, digits = 2),
        median_vig_pct  = round((median(sub.overround_close) - 1.0) * 100, digits = 2)
    )
end
println("\n--- Bet365 Closing Vig by Market ---")
show(b365_vig_by_mkt; allrows = true, allcols = true, truncate = 0)
println()

if nrow(bf_summary_20m) > 0
    bf_vig_by_mkt = combine(groupby(bf_summary_20m, :market_name)) do sub
        (
            n_quotes        = nrow(sub),
            mean_overround  = round(mean(sub.overround_close), digits = 4),
            mean_vig_pct    = round((mean(sub.overround_close) - 1.0) * 100, digits = 2),
            median_vig_pct  = round((median(sub.overround_close) - 1.0) * 100, digits = 2)
        )
    end
    println("\n--- Betfair Closing Overround by Market (Raw exchange book before commission) ---")
    show(bf_vig_by_mkt; allrows = true, allcols = true, truncate = 0)
    println()
end

# -------------------------------------------------------------------
# SECTION 4: HEAD-TO-HEAD MATCHED ODDS COMPARISON
# -------------------------------------------------------------------
println("\n", "="^95)
println("SECTION 4: HEAD-TO-HEAD MATCHED QUOTES (Betfair Net of 2% Comm vs Bet365)")
println("="^95)

if nrow(bf_summary_20m) > 0
    # Join on match_id, market_name, market_line, selection
    matched_quotes = innerjoin(
        b365_df[:, [:match_id, :market_name, :market_line, :selection, :odds_close]],
        bf_summary_20m[:, [:match_id, :market_name, :market_line, :selection, :odds_close]],
        on = [:match_id, :market_name, :market_line, :selection],
        makeunique = true
    )
    rename!(matched_quotes, :odds_close => :odds_b365, :odds_close_1 => :odds_bf_raw)
    
    # Betfair Net of 2% commission on winning bets: odds_net = 1 + (odds - 1) * 0.98
    matched_quotes.odds_bf_net = 1.0 .+ (matched_quotes.odds_bf_raw .- 1.0) .* 0.98
    matched_quotes.delta_odds  = matched_quotes.odds_bf_net .- matched_quotes.odds_b365
    matched_quotes.pct_better  = ((matched_quotes.odds_bf_net ./ matched_quotes.odds_b365) .- 1.0) .* 100
    
    matched_summary = combine(groupby(matched_quotes, :market_name)) do sub
        (
            n_matched_quotes  = nrow(sub),
            mean_b365_odds    = round(mean(sub.odds_b365), digits = 2),
            mean_bf_net_odds  = round(mean(sub.odds_bf_net), digits = 2),
            mean_odds_diff    = round(mean(sub.delta_odds), digits = 3),
            mean_gain_pct     = round(mean(sub.pct_better), digits = 2),
            pct_bf_better     = round(100 * count(sub.delta_odds .> 0.001) / nrow(sub), digits = 1),
            pct_b365_better   = round(100 * count(sub.delta_odds .< -0.001) / nrow(sub), digits = 1)
        )
    end
    println("\n--- Head-to-Head Quoted Odds Comparison by Market ---")
    show(matched_summary; allrows = true, allcols = true, truncate = 0)
    println()
    
    # Breakdown by odds tier (Favorites vs Longshots)
    matched_quotes.tier = ifelse.(matched_quotes.odds_b365 .< 2.0, "1. Favorite (< 2.0)",
                          ifelse.(matched_quotes.odds_b365 .<= 3.5, "2. Mid-Range (2.0 - 3.5)", "3. Longshot (> 3.5)"))
    
    tier_summary = combine(groupby(matched_quotes, :tier)) do sub
        (
            n_quotes         = nrow(sub),
            mean_b365        = round(mean(sub.odds_b365), digits = 2),
            mean_bf_net      = round(mean(sub.odds_bf_net), digits = 2),
            mean_gain_pct    = round(mean(sub.pct_better), digits = 2),
            pct_bf_better    = round(100 * count(sub.delta_odds .> 0.001) / nrow(sub), digits = 1)
        )
    end
    sort!(tier_summary, :tier)
    println("\n--- Head-to-Head Odds Comparison by Price Tier ---")
    show(tier_summary; allrows = true, allcols = true, truncate = 0)
    println()
end

# -------------------------------------------------------------------
# SECTION 5: HYBRID EXECUTION SUMMARY & RECOMMENDATION
# -------------------------------------------------------------------
println("\n", "="^95)
println("SECTION 5: HYBRID EXCHANGE-FIRST ROUTING ASSESSMENT")
println("="^95)

println("EDA Execution Complete.")
