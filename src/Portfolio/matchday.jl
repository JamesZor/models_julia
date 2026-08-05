# src/Portfolio/matchday.jl
#
# Live use: fixtures that have not been played yet.
#
# Backtesting and match-day differ in exactly one respect -- a backtest book carries a
# settlement vector and a match-day book does not. Everything else (pricing, the payoff matrix,
# the allocator, shrinkage, trust, the drawdown budget, the exposure cap) is identical and
# shared, which is the point: the sheet you bet from is produced by the same code path that was
# audited against history, not by a parallel reimplementation.

export is_settled, stake_sheet, slate_summary

"Has this fixture been played and graded?"
is_settled(b::MatchBook) = b.settle !== nothing

"""
    stake_sheet(sys, latents_df, expr, odds_df, ds; bankroll = 1.0) -> DataFrame

The match-day entry point. Produces one row per bet to place.

`latents_df` holds posterior summaries for the fixtures you want to price -- for upcoming
matches these come from the match-day inference pipeline
(`current_development/match_day_inference/src/inference.jl`), not from
`Experiments.extract_oos_predictions`, which only covers matches already in a CV fold.

`odds_df` must carry `:match_id, :market_name, :market_line, :selection, :odds_close`. Any
source with that schema works -- the historical Betfair summary, or a live price feed.

`bankroll` scales `frac` into a `stake` column in your own currency.

Risk is solved per slate, so all fixtures settling together share one drawdown budget and one
exposure cap. That is the whole reason this is not a per-match loop.
"""
function stake_sheet(sys::PortfolioSystem, latents_df::DataFrame, expr,
                     odds_df::DataFrame, ds; bankroll::Real = 1.0)
    books  = build_books(sys.book, latents_df, expr, odds_df, ds; require_result = false)
    isempty(books) && return _empty_sheet()
    slates = group(sys.policy.grouping, books)

    rows = NamedTuple[]
    for (t, sl) in enumerate(slates)
        ctx   = SlateContext(t, sl.window, Float64(bankroll))
        alloc = stake_slate(sys.policy, sl, ctx)
        for (i, b) in enumerate(sl.books), j in eachindex(b.sels)
            f = alloc.stakes[i][j]
            f > 0 || continue
            s = b.sels[j]
            push!(rows, (slate = sl.window, match_id = b.m_id, family = s.family,
                         group = s.group, line = s.line, selection = s.selection,
                         odds_quoted = s.odds_quoted, odds = s.odds_used,
                         p_model = s.p_model, p_market = s.p_market,
                         edge = s.p_model - s.p_market,
                         frac = f, stake = f * bankroll,
                         k_risk = alloc.k_risk, slate_exposure = alloc.exposure,
                         capped = alloc.capped, settled = is_settled(b)))
        end
    end
    isempty(rows) && return _empty_sheet()
    return sort!(DataFrame(rows), [:slate, :stake], rev = [false, true])
end

stake_sheet(sys::PortfolioSystem, latents, expr, odds_df::DataFrame, ds; kw...) =
    stake_sheet(sys, latents.df, expr, odds_df, ds; kw...)

_empty_sheet() = DataFrame(slate = Date[], match_id = Int[], family = String[], group = String[],
                           line = Float64[], selection = Symbol[], odds_quoted = Float64[],
                           odds = Float64[], p_model = Float64[], p_market = Float64[],
                           edge = Float64[], frac = Float64[], stake = Float64[],
                           k_risk = Float64[], slate_exposure = Float64[], capped = Bool[],
                           settled = Bool[])

"""
    slate_summary(sheet) -> DataFrame

Per-settlement-window totals: how many bets, how much of the bankroll is live at once, what the
drawdown budget did, and whether the hard cap bound. Check this before the sheet itself --
exposure is the number that can ruin you, individual stakes are not.
"""
function slate_summary(sheet::DataFrame)
    isempty(sheet) && return DataFrame()
    g = combine(groupby(sheet, :slate),
                :match_id => (x -> length(unique(x))) => :fixtures,
                nrow => :bets,
                :stake => sum => :total_stake,
                :frac => sum => :exposure,
                :k_risk => first => :k_risk,
                :capped => first => :capped)
    return g
end
