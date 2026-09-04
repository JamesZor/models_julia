# ==============================================================================
# l02 — The T−25 point-in-time book
# ==============================================================================
#
# Loader. Definitions only. Pairs with `r03_t25_calibration_and_portfolio.jl`.
#
# ------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ------------------------------------------------------------------------------
#
# Every number in `l01` / `r01` / `r02` is priced at the Betfair CLOSE. A closing
# price is not available at the moment a bet is struck, so all of it is an upper
# bound — stated at the top of both runners and in README §3, and never resolved.
# This file resolves it.
#
# `MatchDay`'s execution band is T−25 to T−12 (AGENTS.md §7.2), so T−25 is the
# EARLIEST instant at which a slate is committed and therefore the most
# conservative honest choice. Everything here is built from ticks stamped at or
# before that instant.
#
# ------------------------------------------------------------------------------
# WHY NOT `summarize_odds(..., window = (-30.0, -25.0))`
# ------------------------------------------------------------------------------
#
# Because the archive is not dense enough that far out, and a TWA over a window
# silently papers over it. MEASURED on `ds.betfair_odds` (599,529 ticks, 1,641
# fixtures):
#
#   window            (match, market, line, selection) groups     median ticks
#   (−30,  −25)                     8,644                              1
#   (−60,  −25)                    17,129                              1
#   (−240, −25)                    26,820                              2
#   ( −20,   0)  ← the close       26,341                              —
#
# A five-minute window at T−25 carries a THIRD of the close book's coverage, and
# a median of one tick — so the "time-weighted average" is one number with a
# weight. Widening the window to recover coverage makes the estimate an average
# over prices up to four hours old, which is not the price on the screen at T−25
# either.
#
# So this file does what the replay console does instead (AGENTS.md §7.1): it
# takes the LAST TICK AT OR BEFORE the cutoff, per selection, and carries its
# STALENESS as a column. That is the price that was actually showing. A tick from
# after the cutoff is unreachable rather than merely unqueried, and a book whose
# freshest tick is two hours old is refused by a gate rather than averaged into
# something that looks current.
#
# ------------------------------------------------------------------------------
# WHAT THIS FIXES ON THE WAY PAST
# ------------------------------------------------------------------------------
#
# `l01_betfair_closing_odds` de-vigs by normalising within (match, market, line)
# whatever selections happen to be present, which is degenerate on a one-sided
# quote: a lone `over_05` normalises to a fair probability of exactly 1.0. That
# defect is documented in README §5.6 and cost the O/U 0.5 ladder its place in the
# Phase 2 book. This builder REQUIRES the market's full selection set before it
# de-vigs, so the defect cannot occur here for any line. `l01`'s builder is left
# untouched so r01 and r02 stay reproducible.
#
# ------------------------------------------------------------------------------
# THE COLUMN NAMES ARE THE SCHEMA'S, NOT A CLAIM ABOUT WHEN
# ------------------------------------------------------------------------------
#
# `Evaluation.build_odds_view` and `Portfolio.build_odds_index` both read
# `:odds_close`, `:prob_fair_close` and `:prob_implied_close` by name. A T−25 book
# must therefore carry those names to pass through the pipeline unmodified — and a
# frame named "close" that holds a T−25 price is exactly the sort of thing that
# gets mixed up three weeks later. So every frame this builder returns also carries
# `:as_of_minutes` and `:staleness_minutes`, and `assert_book_as_of` turns the
# mix-up into an error at the call site rather than a wrong number in a table.
# ==============================================================================

# %%
# ===================================================================
# 1. Packages
# ===================================================================

using BayesianFootball
using DataFrames
using Printf
using Statistics


# %%
# ===================================================================
# 2. Configuration
# ===================================================================

"""
    PointInTimeBookConfig(; as_of_minutes, max_staleness_minutes,
                            overround_limits, require_complete_markets)

| field | is |
|---|---|
| `as_of_minutes` | the cutoff, in minutes relative to kick-off. `-25.0` is the start of MatchDay's execution band; `0.0` reproduces a close-like book from the same code path |
| `max_staleness_minutes` | refuse a selection whose freshest tick at or before the cutoff is older than this. The gate that stops a four-hour-old price being priced as current |
| `overround_limits` | the same cross-selection sanity filter `Data.summarize_odds` applies. Pre-match books are wider than closing ones, so this is expected to bind harder here and the runner reports how much |
| `require_complete_markets` | de-vig only a market whose full selection set is present. See the header |

The default `max_staleness_minutes = 90.0` is not a tuned value; it is the
loosest bound at which "the price on the screen" is still a defensible
description. `r03` reports the staleness distribution so it can be argued with.
"""
Base.@kwdef struct PointInTimeBookConfig
    as_of_minutes::Float64 = -25.0
    max_staleness_minutes::Float64 = 90.0
    overround_limits::Tuple{Float64,Float64} = (0.9, 1.10)
    require_complete_markets::Bool = true
end

"How many selections a complete market of this name and line must carry."
function expected_selection_count(market_name::AbstractString, ::Real)
    market_name == "1X2" && return 3
    market_name == "OverUnder" && return 2
    market_name == "BTTS" && return 2
    market_name == "DoubleChance" && return 3
    market_name == "DrawNoBet" && return 2
    return 0            # 0 means "no completeness contract known"; see below
end


# %%
# ===================================================================
# 3. The point-in-time price
# ===================================================================

"""
    point_in_time_prices(betfair_long; config) -> DataFrame

The last traded price at or before `config.as_of_minutes`, per
(match, market, line, selection), with its staleness.

`<=`, not `<`: a tick stamped exactly at the cutoff is visible at the cutoff.
Everything after it is unreachable — the frame is filtered before the group-wise
`argmax`, so there is no path by which a later tick can be selected.

Returns one row per selection with `odds_close` (the schema's name for the price
this book carries — see the header), `tick_minutes`, `staleness_minutes` and
`n_ticks_before` (how much history stood behind that price).
"""
function point_in_time_prices(betfair_long::AbstractDataFrame;
                              config::PointInTimeBookConfig = PointInTimeBookConfig())
    for c in (:match_id, :market_name, :market_line, :selection,
              :minutes_to_kickoff, :traded_price)
        hasproperty(betfair_long, c) || error(
            "point_in_time_prices: the long Betfair frame has no :$c. Pass " *
            "`ds.betfair_odds`, which `Data.unpack_betfair_odds` builds with these columns.")
    end

    as_of = config.as_of_minutes
    visible = filter(r -> r.minutes_to_kickoff <= as_of &&
                          isfinite(r.traded_price) && r.traded_price > 1.0,
                     betfair_long)
    nrow(visible) == 0 && error(
        "point_in_time_prices: no tick at or before T$(as_of). The archive cannot " *
        "answer this cutoff.")

    picked = combine(groupby(visible, [:match_id, :market_name, :market_line, :selection])) do sdf
        i = argmax(sdf.minutes_to_kickoff)
        return (odds_close = Float64(sdf.traded_price[i]),
                tick_minutes = Float64(sdf.minutes_to_kickoff[i]),
                n_ticks_before = nrow(sdf))
    end
    picked.staleness_minutes = as_of .- picked.tick_minutes
    picked.as_of_minutes = fill(as_of, nrow(picked))
    picked.match_id = Int.(picked.match_id)
    picked.market_name = String.(picked.market_name)
    picked.market_line = Float64.(picked.market_line)
    picked.selection = Symbol.(picked.selection)
    return picked
end

"""
    devig_book!(prices; config) -> (DataFrame, DataFrame)

De-vig within (match, market, line) and apply the completeness and overround
gates. Returns the surviving book and a refusal frame naming what was dropped and
why, one row per refused market.

COMPLETENESS IS CHECKED BEFORE NORMALISATION, not after. Normalising a one-sided
quote produces a fair probability of 1.0 and no error, which is how the O/U 0.5
ladder came to be scored against a fabricated price in r01 (README §5.6). A market
whose family has no known selection count (`expected_selection_count` returns 0 —
correct score, Asian handicap) is passed through only when
`require_complete_markets` is off, because there is no contract to check it
against.
"""
function devig_book!(prices::AbstractDataFrame;
                     config::PointInTimeBookConfig = PointInTimeBookConfig())
    keep = DataFrame[]
    refused = NamedTuple[]

    for g in groupby(prices, [:match_id, :market_name, :market_line])
        name = first(g.market_name)
        line = first(g.market_line)
        mid = first(g.match_id)
        want = expected_selection_count(name, line)

        if config.require_complete_markets
            if want == 0
                push!(refused, (; match_id = mid, market_name = name, market_line = line,
                                n_selections = nrow(g), overround = NaN,
                                max_staleness = maximum(g.staleness_minutes),
                                reason = "no completeness contract for this market family"))
                continue
            elseif nrow(g) != want
                push!(refused, (; match_id = mid, market_name = name, market_line = line,
                                n_selections = nrow(g), overround = NaN,
                                max_staleness = maximum(g.staleness_minutes),
                                reason = "incomplete market ($(nrow(g)) of $want selections)"))
                continue
            end
        end

        stale = maximum(g.staleness_minutes)
        if stale > config.max_staleness_minutes
            push!(refused, (; match_id = mid, market_name = name, market_line = line,
                            n_selections = nrow(g), overround = NaN,
                            max_staleness = stale,
                            reason = @sprintf("stalest side %.0f min exceeds %.0f",
                                              stale, config.max_staleness_minutes)))
            continue
        end

        implied = 1.0 ./ g.odds_close
        overround = sum(implied)
        lo, hi = config.overround_limits
        if !(lo <= overround <= hi)
            push!(refused, (; match_id = mid, market_name = name, market_line = line,
                            n_selections = nrow(g), overround = overround,
                            max_staleness = stale,
                            reason = @sprintf("overround %.4f outside [%.2f, %.2f]",
                                              overround, lo, hi)))
            continue
        end

        out = DataFrame(g)
        out.prob_implied_close = implied
        out.prob_fair_close = implied ./ overround
        out.overround = fill(overround, nrow(out))
        push!(keep, out)
    end

    book = isempty(keep) ? similar(prices, 0) : vcat(keep...)
    nrow(book) > 0 && sort!(book, [:match_id, :market_name, :market_line, :selection])
    return book, isempty(refused) ? DataFrame() : DataFrame(refused)
end

"""
    point_in_time_book(ds; config) -> (book, refusals)

The whole builder: last visible tick → completeness → staleness → overround →
de-vig → realised outcome.

`is_winner` is joined from `ds.odds`, which is the settlement record and carries
no price, so joining it cannot leak a price the cutoff could not see.
"""
function point_in_time_book(ds; config::PointInTimeBookConfig = PointInTimeBookConfig())
    prices = point_in_time_prices(ds.betfair_odds; config = config)
    book, refusals = devig_book!(prices; config = config)
    nrow(book) == 0 && return book, refusals

    outcome_cols = [:match_id, :market_name, :market_line, :selection, :is_winner]
    winners = unique(select(ds.odds, outcome_cols))
    winners.match_id = Int.(winners.match_id)
    winners.market_name = String.(winners.market_name)
    winners.market_line = Float64.(winners.market_line)
    winners.selection = Symbol.(winners.selection)
    book = leftjoin(book, winners;
                    on = [:match_id, :market_name, :market_line, :selection])
    sort!(book, [:match_id, :market_name, :market_line, :selection])
    return book, refusals
end

"""
    assert_book_as_of(book, expected_minutes)

Refuse a book that was not built at the instant the caller believes it was.

The pipeline's column names say "close" whatever the cutoff (see the header), so
this is the only thing standing between a T−25 experiment and a table of closing
prices with a T−25 caption. Call it at every point a book crosses into scoring or
staking.
"""
function assert_book_as_of(book::AbstractDataFrame, expected_minutes::Real)
    hasproperty(book, :as_of_minutes) || error(
        "assert_book_as_of: this frame carries no :as_of_minutes, so it was not " *
        "built by `point_in_time_book`. A closing book from `l01` is not " *
        "interchangeable with a point-in-time one; the column names are the same " *
        "and the prices are not.")
    got = unique(book.as_of_minutes)
    length(got) == 1 || error(
        "assert_book_as_of: this frame mixes cutoffs $(got). One book, one instant.")
    isapprox(first(got), Float64(expected_minutes); atol = 1e-9) || error(
        "assert_book_as_of: expected T$(expected_minutes), got T$(first(got)).")
    return book
end


# %%
# ===================================================================
# 4. Diagnostics
# ===================================================================

"""
    book_coverage(book, refusals) -> NamedTuple

What survived, and the staleness of what did. `p90_staleness` is the number that
says whether "the price on the screen at T−25" is a fair description of this book
or a generous one.
"""
function book_coverage(book::AbstractDataFrame, refusals::AbstractDataFrame)
    nrow(book) == 0 && return (; n_rows = 0, n_fixtures = 0, n_markets = 0,
                               median_staleness = NaN, p90_staleness = NaN,
                               max_staleness = NaN, median_overround = NaN,
                               n_refused_markets = nrow(refusals))
    mk = nrow(unique(select(book, [:match_id, :market_name, :market_line])))
    return (; n_rows = nrow(book),
            n_fixtures = length(unique(book.match_id)),
            n_markets = mk,
            median_staleness = median(book.staleness_minutes),
            p90_staleness = quantile(book.staleness_minutes, 0.90),
            max_staleness = maximum(book.staleness_minutes),
            median_overround = median(book.overround),
            n_refused_markets = nrow(refusals))
end

"Refusal reasons and counts, most frequent first. Reasons carry their numbers."
function refusal_summary(refusals::AbstractDataFrame)
    nrow(refusals) == 0 && return Pair{String,Int}[]
    # Collapse the numeric detail so the reasons group; the frame keeps the detail.
    kind(r) = startswith(r, "incomplete market") ? "incomplete market" :
              startswith(r, "stalest side") ? "stale beyond the bound" :
              startswith(r, "overround") ? "overround outside limits" : r
    counts = Dict{String,Int}()
    for r in refusals.reason
        k = kind(r)
        counts[k] = get(counts, k, 0) + 1
    end
    return sort!(collect(counts), by = last, rev = true)
end

"""
    book_drift(pit_book, close_book) -> DataFrame

The same selection at two instants, side by side: the T−25 price, the closing
price, the log price drift, and the fair-probability move.

This is the closing-line-value question asked of the BOOK rather than of a bet:
a selection whose price shortened between T−25 and the close was one the market
came to agree with. It is also the diagnostic that says how much of a
close-priced backtest's edge was information the bettor could not have had.
"""
function book_drift(pit_book::AbstractDataFrame, close_book::AbstractDataFrame)
    key = [:match_id, :market_name, :market_line, :selection]
    a = select(pit_book, key..., :odds_close => :odds_pit,
               :prob_fair_close => :fair_pit, :staleness_minutes)
    b = select(close_book, key..., :odds_close => :odds_close_final,
               :prob_fair_close => :fair_close)
    j = innerjoin(a, b; on = key)
    nrow(j) == 0 && return j
    j.log_price_drift = log.(j.odds_close_final ./ j.odds_pit)
    j.fair_drift = j.fair_close .- j.fair_pit
    return j
end
