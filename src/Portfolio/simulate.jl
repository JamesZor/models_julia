# src/Portfolio/simulate.jl
#
# Stage C: walk the slates forward, settle each one against a common bankroll, compound once.

export simulate

"""
    simulate(policy, slates; use_shrink = true, scale = 1.0) -> Trajectory

Chronological, simultaneous same-slate settlement, one compounding step per slate.

Two assertions guard the failures the prototype shipped with:

* slates must be sorted. Final wealth is order-invariant but drawdown, Ulcer, Calmar and Martin
  are not -- on the same 628-match book, Martin ranged 52 to 144 across random orderings of the
  identical returns.
* every book must be settled -- you cannot backtest an unplayed fixture.
* a slate cannot lose more than the bankroll. Guaranteed by `FixedCap`'s `(0,1)` constraint;
  asserted anyway so a future cap implementation cannot quietly break it.
"""
function simulate(policy::PolicySpec, slates::Vector{Slate};
                  use_shrink::Bool = true, scale::Float64 = 1.0)
    @assert issorted(slates, by = s -> s.window) "slates must be chronological"
    @assert all(is_settled(b) for sl in slates for b in sl.books) """
        simulate needs settled books: at least one fixture has no result.
        Build with require_result = true (the default) for a backtest; unsettled books are for
        stake_sheet only."""

    bank     = 1.0
    hist     = Float64[1.0]
    dates    = Date[]
    slate_pl = Float64[]
    ks       = Float64[]
    expo     = Float64[]
    n_capped = 0
    tot_stake = 0.0
    tot_pl    = 0.0
    rows = NamedTuple[]

    for (t, sl) in enumerate(slates)
        ctx = SlateContext(t, sl.window, bank)
        alloc = stake_slate(policy, sl, ctx; use_shrink = use_shrink, scale = scale)
        push!(ks, alloc.k_risk); push!(expo, alloc.exposure)
        alloc.capped && (n_capped += 1)

        pl, stk = 0.0, 0.0
        for (i, b) in enumerate(sl.books), j in eachindex(b.sels)
            s = alloc.stakes[i][j]
            s > 0 || continue
            r = s * b.settle[j]
            stk += s; pl += r
            push!(rows, (match_id = b.m_id, date = b.date, family = b.sels[j].family,
                         selection = b.sels[j].selection, odds = b.sels[j].odds_used,
                         stake = s, pnl = r, payoff = b.settle[j],
                         p_model = b.sels[j].p_model, p_market = b.sels[j].p_market))
        end

        @assert pl > -1.0 "slate $(sl.window) lost more than the bankroll (pl = $pl); " *
                          "the exposure cap is not doing its job"
        bank *= (1.0 + pl)
        push!(hist, bank); push!(dates, sl.window); push!(slate_pl, pl)
        tot_stake += stk; tot_pl += pl
    end

    bets = isempty(rows) ? _empty_bets() : DataFrame(rows)
    return Trajectory(hist, dates, slate_pl, ks, expo, n_capped, tot_stake, tot_pl, bets)
end

simulate(sys::PortfolioSystem, slates::Vector{Slate}; kw...) = simulate(sys.policy, slates; kw...)

_empty_bets() = DataFrame(match_id = Int[], date = Date[], family = String[],
                          selection = Symbol[], odds = Float64[], stake = Float64[],
                          pnl = Float64[], payoff = Float64[],
                          p_model = Float64[], p_market = Float64[])
