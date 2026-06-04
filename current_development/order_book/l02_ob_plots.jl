# current_development/order_book/l02_ob_plots.jl
# Plotting helpers for order book EDA

using Plots
using DataFrames
using Statistics

# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

"""
Return all rows for a given match (by home/away team names).
"""
function filter_match(ob::DataFrame, home_team::String, away_team::String)
    filter(r -> r.home_team == home_team && r.away_team == away_team, ob)
end

"""
Return rows for a specific market + selection within a match DataFrame.
"""
function filter_selection(df::DataFrame, market_name::String, selection::Symbol)
    filter(r -> r.market_name == market_name && r.selection == selection, df)
end

# ---------------------------------------------------------------------------
# Core plot: price ladder + volume for one selection over time
# ---------------------------------------------------------------------------

"""
    plot_ob_selection(ob, home_team, away_team, market_name, selection;
                      pre_kickoff_only=true, save_path=nothing)

Two-panel chart for one selection:
  Top:    bid/ask price ladder (L1 solid, L2+L3 faded) with spread shaded.
  Bottom: total bid and ask depth (sum of L1–L3 volumes).

X-axis is minutes_to_kickoff (negative = before match).
"""
function plot_ob_selection(
    ob::DataFrame,
    home_team::String,
    away_team::String,
    market_name::String,
    selection::Symbol;
    time_window::Union{Tuple{Real, Real}, Nothing} = nothing,
    pre_kickoff_only::Bool = true,
    save_path::Union{String, Nothing} = nothing,
)
    match_df = filter_match(ob, home_team, away_team)
    df = filter_selection(match_df, market_name, selection)
    isempty(df) && error("No data found for $home_team vs $away_team | $market_name | $selection")

    if !isnothing(time_window)
        filter!(r -> time_window[1] <= r.minutes_to_kickoff <= time_window[2], df)
    elseif pre_kickoff_only
        filter!(r -> r.minutes_to_kickoff <= 0, df)
    end
    sort!(df, :minutes_to_kickoff)

    x = df.minutes_to_kickoff

    mid_price = (df.bid_price_1 .+ df.ask_price_1) ./ 2
    spread    = df.ask_price_1 .- df.bid_price_1

    bid_depth = map(r -> sum(filter(!isnan, [r.bid_vol_1, r.bid_vol_2, r.bid_vol_3])), eachrow(df))
    ask_depth = map(r -> sum(filter(!isnan, [r.ask_vol_1, r.ask_vol_2, r.ask_vol_3])), eachrow(df))

    title_str = "$home_team vs $away_team  |  $market_name  |  $(selection)"

    # --- Price panel ---
    p1 = plot(;
        title    = title_str,
        ylabel   = "Decimal Odds",
        xlabel   = "",
        xgrid    = true,
        legend   = :topleft,
        titlefontsize = 10,
    )

    # Shaded spread between best bid and best ask
    plot!(p1, x, df.ask_price_1;
        fillrange = df.bid_price_1,
        fillalpha = 0.15,
        fillcolor = :grey,
        linewidth = 0,
        label     = "spread",
        color     = :grey,
    )

    # L2 bid/ask as faded lines
    valid_l2 = .!isnan.(df.bid_price_2)
    if any(valid_l2)
        plot!(p1, x[valid_l2], df.bid_price_2[valid_l2]; color=:royalblue, alpha=0.35, linewidth=1, label="bid L2", linestyle=:dash)
        plot!(p1, x[valid_l2], df.ask_price_2[valid_l2]; color=:firebrick, alpha=0.35, linewidth=1, label="ask L2", linestyle=:dash)
    end

    # Best bid (back) and best ask (lay) — bold
    plot!(p1, x, df.bid_price_1; color=:royalblue, linewidth=2.5, label="bid L1 (back)")
    plot!(p1, x, df.ask_price_1; color=:firebrick,  linewidth=2.5, label="ask L1 (lay)")

    # Mid price
    plot!(p1, x, mid_price; color=:black, linewidth=1.5, linestyle=:dot, label="mid")

    vline!(p1, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff")

    # --- Volume / depth panel ---
    p2 = plot(;
        ylabel  = "Depth (£)",
        xlabel  = "Minutes to Kickoff",
        xgrid   = true,
        legend  = :topleft,
    )

    plot!(p2, x, bid_depth; color=:royalblue, linewidth=2, label="bid depth (L1–L3)", fillrange=0, fillalpha=0.15)
    plot!(p2, x, ask_depth; color=:firebrick,  linewidth=2, label="ask depth (L1–L3)", fillrange=0, fillalpha=0.15)
    vline!(p2, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff")

    fig = plot(p1, p2; layout=(2, 1), size=(1000, 650), left_margin=5Plots.mm, bottom_margin=4Plots.mm)

    if !isnothing(save_path)
        savefig(fig, save_path)
        println("Saved: $save_path")
    end

    return fig
end

# ---------------------------------------------------------------------------
# Market overview: all selections in one market, price only
# ---------------------------------------------------------------------------

"""
    plot_market_prices(ob, home_team, away_team, market_name;
                       pre_kickoff_only=true, save_path=nothing)

Mid-price time series for every selection in a market on one chart.
Useful for seeing how a 1X2 or OverUnder market moves as a whole.
"""
function plot_market_prices(
    ob::DataFrame,
    home_team::String,
    away_team::String,
    market_name::String;
    time_window::Union{Tuple{Real, Real}, Nothing} = nothing,
    pre_kickoff_only::Bool = true,
    save_path::Union{String, Nothing} = nothing,
)
    match_df = filter_match(ob, home_team, away_team)
    mkt_df   = filter(r -> r.market_name == market_name, match_df)
    isempty(mkt_df) && error("No data for $market_name in $home_team vs $away_team")

    if !isnothing(time_window)
        filter!(r -> time_window[1] <= r.minutes_to_kickoff <= time_window[2], mkt_df)
    elseif pre_kickoff_only
        filter!(r -> r.minutes_to_kickoff <= 0, mkt_df)
    end

    p = plot(;
        title     = "$home_team vs $away_team  |  $market_name  — mid prices",
        xlabel    = "Minutes to Kickoff",
        ylabel    = "Decimal Odds (mid)",
        legend    = :outertopright,
        size      = (1000, 450),
        titlefontsize = 10,
    )

    for (sel, grp) in pairs(groupby(sort(mkt_df, :minutes_to_kickoff), :selection))
        mid = (grp.bid_price_1 .+ grp.ask_price_1) ./ 2
        plot!(p, grp.minutes_to_kickoff, mid; label=string(sel.selection), linewidth=2)
    end

    vline!(p, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff")

    !isnothing(save_path) && savefig(p, save_path)
    return p
end

# ---------------------------------------------------------------------------
# Spread evolution: how tight is the market over time?
# ---------------------------------------------------------------------------

"""
    plot_spread(ob, home_team, away_team, market_name;
                pre_kickoff_only=true, save_path=nothing)

Bid-ask spread (ask_L1 - bid_L1) for every selection in a market.
Tightening spread = improving liquidity as kickoff approaches.
"""
function plot_spread(
    ob::DataFrame,
    home_team::String,
    away_team::String,
    market_name::String;
    time_window::Union{Tuple{Real, Real}, Nothing} = nothing,
    pre_kickoff_only::Bool = true,
    save_path::Union{String, Nothing} = nothing,
)
    match_df = filter_match(ob, home_team, away_team)
    mkt_df   = filter(r -> r.market_name == market_name, match_df)
    isempty(mkt_df) && error("No data for $market_name in $home_team vs $away_team")

    if !isnothing(time_window)
        filter!(r -> time_window[1] <= r.minutes_to_kickoff <= time_window[2], mkt_df)
    elseif pre_kickoff_only
        filter!(r -> r.minutes_to_kickoff <= 0, mkt_df)
    end

    p = plot(;
        title     = "$home_team vs $away_team  |  $market_name  — bid-ask spread",
        xlabel    = "Minutes to Kickoff",
        ylabel    = "Spread (ask L1 − bid L1)",
        legend    = :outertopright,
        size      = (1000, 400),
        titlefontsize = 10,
    )

    for (sel, grp) in pairs(groupby(sort(mkt_df, :minutes_to_kickoff), :selection))
        spread = grp.ask_price_1 .- grp.bid_price_1
        plot!(p, grp.minutes_to_kickoff, spread; label=string(sel.selection), linewidth=2)
    end

    vline!(p, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff")
    hline!(p, [0.0]; color=:grey, linestyle=:dot, linewidth=1, label="")

    !isnothing(save_path) && savefig(p, save_path)
    return p
end

# ---------------------------------------------------------------------------
# Helper: list available matches in ob
# ---------------------------------------------------------------------------

function list_matches(ob::DataFrame)
    unique(ob[:, [:home_team, :away_team, :market_name, :market_line]])
end
