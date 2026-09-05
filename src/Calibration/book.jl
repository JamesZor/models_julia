# ==============================================================================
# src/Calibration/book.jl — the tradeable point-in-time book
# ==============================================================================
#
# Graduated from `current_development/calibration_generative_eda/l02_point_in_time_book.jl`.
#
# ------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ------------------------------------------------------------------------------
#
# A calibrator is only worth deploying if it can be run against a price a bettor could
# actually have taken. `MatchDay`'s execution band is T-25 to T-12 (AGENTS.md §7.2), so
# T-25 is the EARLIEST instant at which a slate is committed and therefore the most
# conservative honest choice. Everything here is built from ticks stamped at or before
# that instant.
#
# ------------------------------------------------------------------------------
# WHY NOT `summarize_odds(..., window = (-30.0, -25.0))`
# ------------------------------------------------------------------------------
#
# Because the archive is not dense enough that far out, and a TWA over a window silently
# papers over it. MEASURED on `ds.betfair_odds` (599,529 ticks, 1,641 fixtures):
#
#   window          (match, market, line, selection) groups     median ticks
#   (-30,  -25)                   8,644                              1
#   (-60,  -25)                  17,129                              1
#   (-240, -25)                  26,820                              2
#   ( -20,   0)  <- the close    26,341                              -
#
# A five-minute window at T-25 carries a THIRD of the close book's coverage at a median
# of ONE tick, so its "time-weighted average" is one number with a weight. Widening the
# window to recover coverage makes the estimate an average over prices up to four hours
# old, which is not the price on the screen at T-25 either.
#
# So this builder does what the replay console does (AGENTS.md §7.1): the LAST TICK AT OR
# BEFORE the cutoff, per selection, carrying its STALENESS as a column. A tick from after
# the cutoff is unreachable rather than merely unqueried, and a book whose freshest tick
# is two hours old is refused by a gate rather than averaged into something that looks
# current.
#
# ------------------------------------------------------------------------------
# WHAT THIS FIXES ON THE WAY PAST
# ------------------------------------------------------------------------------
#
# The prototype's closing-book builder de-vigs by normalising within
# (match, market, line) whatever selections happen to be present, which is DEGENERATE on
# a one-sided quote: a lone `over_05` normalises to a fair probability of exactly 1.0. On
# the Scottish Lower archive the O/U 0.5 ladder is quoted 982 over against 408 under —
# 574 one-sided fixtures — and the symptom is unmissable: the closing line's own LogLoss
# on that family is 1.31832 against the model's 0.21098, where every other scored line is
# paired to within three rows. Staking against a fabricated fair price of 1.0 would
# manufacture an edge out of a de-vigging artefact.
#
# This builder REQUIRES the market's full selection set BEFORE it de-vigs, so the defect
# cannot occur here for any line at any cutoff.
#
# ------------------------------------------------------------------------------
# THE COLUMN NAMES ARE THE SCHEMA'S, NOT A CLAIM ABOUT WHEN
# ------------------------------------------------------------------------------
#
# `Evaluation.build_odds_view` and `Portfolio.build_odds_index` both read `:odds_close`,
# `:prob_fair_close` and `:prob_implied_close` BY NAME. A T-25 book must therefore carry
# those names to pass through the pipeline unmodified — and a frame named "close" holding
# a T-25 price is exactly the sort of thing that gets mixed up three weeks later. So
# every frame this builder returns also carries `:as_of_minutes` and
# `:staleness_minutes`, and `assert_book_as_of` turns the mix-up into an error at the
# call site rather than a wrong number in a table.
# ==============================================================================


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

"""
    PointInTimeBookConfig(; as_of_minutes = -25.0, max_staleness_minutes = 90.0,
                            overround_limits = (0.9, 1.10),
                            require_complete_markets = true)

| field | is |
|---|---|
| `as_of_minutes` | the cutoff, in minutes relative to kick-off. `-25.0` is the start of MatchDay's execution band; `0.0` reproduces a close-like book from the same code path |
| `max_staleness_minutes` | refuse a selection whose freshest tick at or before the cutoff is older than this. The gate that stops a four-hour-old price being priced as current |
| `overround_limits` | the cross-selection sanity filter. Pre-match books are wider than closing ones, so this binds harder here |
| `require_complete_markets` | de-vig only a market whose full selection set is present. See the file header |

`max_staleness_minutes = 90.0` is not a tuned value; it is the loosest bound at which
"the price on the screen" is still a defensible description. `book_coverage` reports the
staleness distribution so it can be argued with — measured at T-25 on Scottish Lower:
1,572 of 1,627 fixtures, median staleness 8 minutes.
"""
Base.@kwdef struct PointInTimeBookConfig
    as_of_minutes::Float64 = -25.0
    max_staleness_minutes::Float64 = 90.0
    overround_limits::Tuple{Float64, Float64} = (0.9, 1.10)
    require_complete_markets::Bool = true
end

"""
    expected_selection_count(market_name, market_line) -> Int

How many selections a COMPLETE market of this name must carry.

`0` means "no completeness contract known" (correct score, Asian handicap), and such a
market is refused under `require_complete_markets` rather than passed through — there is
no contract to check it against, and de-vigging an unknown-arity market is the exact
operation that fabricated the O/U 0.5 price.
"""
function expected_selection_count(market_name::AbstractString, ::Real)
    market_name == "1X2" && return 3
    market_name == "OverUnder" && return 2
    market_name == "BTTS" && return 2
    market_name == "DoubleChance" && return 3
    market_name == "DrawNoBet" && return 2
    return 0
end


# ==============================================================================
# 2. THE POINT-IN-TIME PRICE
# ==============================================================================

"""
    point_in_time_prices(betfair_long; config) -> DataFrame

The last traded price at or before `config.as_of_minutes`, per
(match, market, line, selection), with its staleness.

`<=`, not `<`: a tick stamped exactly at the cutoff is visible at the cutoff. Everything
after it is unreachable — the frame is filtered BEFORE the group-wise `argmax`, so there
is no path by which a later tick can be selected.

Returns one row per selection with `odds_close` (the schema's name for the price this
book carries — see the file header), `tick_minutes`, `staleness_minutes` and
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
    devig_book!(prices; config) -> (book, refusals)

De-vig within (match, market, line) and apply the completeness, staleness and overround
gates. Returns the surviving book and a refusal frame naming what was dropped and why,
one row per refused market.

COMPLETENESS IS CHECKED BEFORE NORMALISATION, not after. Normalising a one-sided quote
produces a fair probability of 1.0 and no error; see the file header for what that cost.
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

The whole builder: last visible tick -> completeness -> staleness -> overround -> de-vig
-> realised outcome.

`is_winner` is joined from `ds.odds`, which is the settlement record and carries no
price, so joining it cannot leak a price the cutoff could not see.

The returned frame is directly consumable by `Evaluation.build_odds_view`,
`Portfolio.build_odds_index` and [`invert_market_rates`](@ref) — it carries `:odds_close`,
`:prob_implied_close`, `:prob_fair_close` and `:is_winner` under the schema's own names,
plus `:as_of_minutes` and `:staleness_minutes` which say when it is from.
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
    assert_book_as_of(book, expected_minutes) -> book

Refuse a book that was not built at the instant the caller believes it was.

The pipeline's column names say "close" whatever the cutoff (see the file header), so
this is the only thing standing between a T-25 experiment and a table of closing prices
with a T-25 caption. `calibrate_fit` calls it; call it at every other point a book
crosses into scoring or staking.
"""
function assert_book_as_of(book::AbstractDataFrame, expected_minutes::Real)
    hasproperty(book, :as_of_minutes) || error(
        "assert_book_as_of: this frame carries no :as_of_minutes, so it was not built by " *
        "`point_in_time_book`. A closing book is not interchangeable with a point-in-time " *
        "one; the column names are the same and the prices are not.")
    got = unique(book.as_of_minutes)
    length(got) == 1 || error(
        "assert_book_as_of: this frame mixes cutoffs $(got). One book, one instant.")
    isapprox(first(got), Float64(expected_minutes); atol = 1e-9) || error(
        "assert_book_as_of: expected T$(expected_minutes), got T$(first(got)).")
    return book
end


# ==============================================================================
# 3. DIAGNOSTICS
# ==============================================================================

"""
    book_coverage(book, refusals) -> NamedTuple

What survived, and the staleness of what did. `p90_staleness` is the number that says
whether "the price on the screen at T-25" is a fair description of this book or a
generous one.
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

"""
    book_refusal_summary(refusals) -> Vector{Pair{String, Int}}

Refusal reasons and counts, most frequent first, with the numeric detail collapsed so the
reasons group. The frame keeps the detail.

A gate that refuses 40% of a book for one reason is a configuration problem; one that
refuses 2% across four reasons is the book being thin. The two look identical in a
coverage percentage, which is why this exists.
"""
function book_refusal_summary(refusals::AbstractDataFrame)
    nrow(refusals) == 0 && return Pair{String, Int}[]
    kind(r) = startswith(r, "incomplete market") ? "incomplete market" :
              startswith(r, "stalest side") ? "stale beyond the bound" :
              startswith(r, "overround") ? "overround outside limits" : r
    counts = Dict{String, Int}()
    for r in refusals.reason
        k = kind(r)
        counts[k] = get(counts, k, 0) + 1
    end
    return sort!(collect(counts), by = last, rev = true)
end

"""
    book_drift(pit_book, close_book) -> DataFrame

The same selection at two instants side by side: the point-in-time price, the closing
price, the log price drift, and the fair-probability move.

This is the closing-line-value question asked of the BOOK rather than of a bet: a
selection whose price shortened between T-25 and the close was one the market came to
agree with.
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


# ==============================================================================
# 4. CLOSING-LINE VALUE
# ==============================================================================
#
# CLV is the diagnostic a T-25 strategy earns and a close-priced one cannot even ask for.
# A backtest settled on realised outcomes is a sample of a few hundred binary draws; CLV
# asks a different and much better-powered question — did the market subsequently move
# TOWARD the bet? — and it answers it on every bet struck, whether it won or lost.
#
# It is not a substitute for P&L. A strategy can have positive CLV and lose money over a
# season, and the reverse. It is evidence about EDGE, where settlement is evidence about
# edge plus variance.

"""
    bet_clv(bets, drift) -> DataFrame

Join a portfolio bet ledger (`result.trajectory.bets`) to the point-in-time -> close
price drift, one row per bet.

`clv_pct` is `100*(odds_taken/odds_close - 1)`: positive means the bet was struck at a
longer price than the market closed at, i.e. the market moved toward the bet.
`fair_gain` is the same thing in probability space.

A bet whose selection is absent from the drift frame — struck on a market the close book
did not carry — is DROPPED and counted by `clv_summary`, never imputed. There is no
closing price to compare it to and inventing one would put the answer in the question.
"""
function bet_clv(bets::AbstractDataFrame, drift::AbstractDataFrame)
    nrow(bets) == 0 && return DataFrame()
    (hasproperty(bets, :match_id) && hasproperty(bets, :selection)) || error(
        "bet_clv: expected a `result.trajectory.bets` ledger with :match_id and :selection.")
    d = select(drift, [:match_id, :selection, :odds_pit, :odds_close_final,
                       :fair_pit, :fair_close, :staleness_minutes])
    j = innerjoin(bets, unique(d, [:match_id, :selection]); on = [:match_id, :selection])
    nrow(j) == 0 && return j
    j.clv_pct = 100 .* (j.odds ./ j.odds_close_final .- 1.0)
    j.fair_gain = j.fair_close .- j.fair_pit
    return j
end

"""
    clv_summary(bets, clv) -> NamedTuple

Headline CLV, stake-weighted as well as flat.

The stake-weighted figure is the one that matters: a strategy with positive CLV on its
small bets and negative CLV on its large ones has negative CLV where the money is, and
the flat mean hides it. It is also the figure `calibration_runs.clv_weighted_pct` stores.
"""
function clv_summary(bets::AbstractDataFrame, clv::AbstractDataFrame)
    n_bets = nrow(bets)
    nrow(clv) == 0 && return (; n_bets, n_matched = 0, n_unmatched = n_bets,
                              mean_clv_pct = NaN, median_clv_pct = NaN,
                              stake_weighted_clv_pct = NaN, pct_positive = NaN,
                              mean_fair_gain = NaN)
    w = clv.stake
    return (; n_bets, n_matched = nrow(clv), n_unmatched = n_bets - nrow(clv),
            mean_clv_pct = mean(clv.clv_pct),
            median_clv_pct = median(clv.clv_pct),
            stake_weighted_clv_pct = sum(w .* clv.clv_pct) / sum(w),
            pct_positive = 100 * count(>(0.0), clv.clv_pct) / nrow(clv),
            mean_fair_gain = mean(clv.fair_gain))
end


# ==============================================================================
# 5. THE CLOSING BOOK, FROM THE SAME CODE PATH
# ==============================================================================

"""
    closing_book(ds; window = (-20.0, 0.0)) -> DataFrame

The Betfair exchange close: time-weighted over `window`, de-vigged within
(match, market, line), joined to the realised outcome, and carrying
`as_of_minutes = 0.0` so [`assert_book_as_of`](@ref) can tell it apart from a
point-in-time book.

DE-VIGGING HERE IS NOT COMPLETENESS-CHECKED, because a TWA close book has no per-tick
staleness to gate on and the normalisation is over whatever the window held. That is the
defect described in the file header, and it is why `calibrate_fit` prefers
[`point_in_time_book`](@ref). This function exists so a close-priced comparison can be
run from production code rather than from the prototype, and it names the risk here
rather than in a table three weeks later: **do not stake a one-sided ladder off this
frame.** `drop_one_sided_markets` removes them if you must.
"""
function closing_book(ds; window::Tuple{Float64, Float64} = (-20.0, 0.0))
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = window)
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    transform!(
        groupby(odds, [:match_id, :market_name, :market_line]),
        :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close,
    )
    odds.as_of_minutes = fill(0.0, nrow(odds))
    odds.staleness_minutes = fill(0.0, nrow(odds))

    outcome_cols = [:match_id, :market_name, :market_line, :selection, :is_winner]
    winners = unique(select(ds.odds, outcome_cols))
    odds = leftjoin(odds, winners;
                    on = [:match_id, :market_name, :market_line, :selection])
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"""
    drop_one_sided_markets(book) -> (book, dropped)

Remove every (match, market, line) group whose selection count does not match
[`expected_selection_count`](@ref), and return what was removed.

The completeness gate `devig_book!` applies BEFORE normalising, offered here as a
post-hoc filter for a book that was built without it — a `closing_book`, or a frame from
outside this module. It cannot undo a fabricated `prob_fair_close`; it can only stop one
being staked.
"""
function drop_one_sided_markets(book::AbstractDataFrame)
    keep = trues(nrow(book))
    dropped = NamedTuple[]
    for g in groupby(book, [:match_id, :market_name, :market_line])
        want = expected_selection_count(first(g.market_name), first(g.market_line))
        (want != 0 && nrow(g) == want) && continue
        for i in parentindices(g)[1]
            keep[i] = false
        end
        push!(dropped, (; match_id = first(g.match_id), market_name = first(g.market_name),
                        market_line = first(g.market_line), n_selections = nrow(g),
                        expected = want))
    end
    return book[keep, :], isempty(dropped) ? DataFrame() : DataFrame(dropped)
end
