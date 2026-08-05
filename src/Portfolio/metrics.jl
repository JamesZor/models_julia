# src/Portfolio/metrics.jl
#
# Stage D. Path statistics are computed here because they need the slate-level bankroll series;
# anything that only needs an equity curve or a per-bet frame is delegated to BackTesting rather
# than reimplemented.

export path_metrics, bootstrap_roi, report

"""
    path_metrics(t::Trajectory) -> NamedTuple

Final wealth, flat ROI, mean log-growth per slate, max drawdown, Ulcer, Calmar, Martin, plus
exposure and capping diagnostics.

Only meaningful on a chronologically ordered trajectory -- `simulate` enforces that. Final
wealth and ROI are order-invariant; everything drawdown-based is not.
"""
function path_metrics(t::Trajectory)
    bk = t.bankroll
    rm = accumulate(max, bk)
    dd = (bk .- rm) ./ rm .* 100
    ui = max(sqrt(mean(dd .^ 2)), 1e-9)
    tr = (bk[end] - 1.0) * 100
    mdd = minimum(dd)

    return (final            = bk[end],
            roi              = t.total_stake > 0 ? 100 * t.total_pl / t.total_stake : 0.0,
            growth_per_slate = isempty(t.slate_pl) ? 0.0 : mean(log.(1.0 .+ t.slate_pl)),
            mdd              = mdd,
            ulcer            = ui,
            calmar           = mdd < 0 ? tr / abs(mdd) : 0.0,
            martin           = tr / ui,
            n_slates         = length(t.slate_pl),
            n_bets           = nrow(t.bets),
            mean_exposure    = isempty(t.exposure) ? 0.0 : mean(t.exposure),
            max_exposure     = isempty(t.exposure) ? 0.0 : maximum(t.exposure),
            worst_slate      = isempty(t.slate_pl) ? 0.0 : minimum(t.slate_pl),
            mean_k_risk      = isempty(t.k_risk) ? 1.0 : mean(t.k_risk),
            n_capped         = t.n_capped)
end

"""
    bootstrap_roi(bets; B = 4000, seed = 1) -> (lo, hi, sd)

Percentile CI on flat ROI, resampled **by match**.

Resampling individual bets would understate the interval badly: several bets on one match share
one scoreline, so they are strongly dependent. On the reference book this is the difference
between an interval that excludes zero and one that does not.
"""
function bootstrap_roi(bets::DataFrame; B::Int = 4000, seed::Int = 1)
    (isempty(bets) || sum(bets.stake) <= 0) && return (lo = 0.0, hi = 0.0, sd = 0.0)
    rng  = Random.MersenneTwister(seed)
    mids = unique(bets.match_id)
    idxs = Dict(m => findall(==(m), bets.match_id) for m in mids)
    v = Vector{Float64}(undef, B)
    for b in 1:B
        sel = Int[]
        for _ in eachindex(mids)
            append!(sel, idxs[mids[rand(rng, 1:length(mids))]])
        end
        v[b] = 100 * sum(bets.pnl[sel]) / sum(bets.stake[sel])
    end
    return (lo = quantile(v, 0.025), hi = quantile(v, 0.975), sd = std(v))
end

"""
    report(t::Trajectory, metrics = AbstractWealthMetric[]) -> NamedTuple

`path_metrics` plus any `BackTesting.AbstractWealthMetric`s evaluated on the slate-level
bankroll curve, plus the match-clustered ROI interval.
"""
function report(t::Trajectory, metrics::Vector = [])
    base = path_metrics(t)
    extra = NamedTuple()
    for m in metrics
        extra = merge(extra, NamedTuple{(Symbol(BackTesting.metric_name(m)),)}(
            (BackTesting.compute_metric(m, t.bankroll),)))
    end
    ci = bootstrap_roi(t.bets)
    return merge(base, extra, (roi_ci_lo = ci.lo, roi_ci_hi = ci.hi))
end

"""
    attribution(t::Trajectory) -> DataFrame

Stake, P/L, ROI and hit rate per selection family. The first thing to look at when a headline
number moves: on the reference book 83% of the profit came from 1X2, a family on which the model
has no measurable log-loss advantage over the market.
"""
function attribution(t::Trajectory)
    isempty(t.bets) && return DataFrame()
    g = combine(groupby(t.bets, :family),
                nrow => :n,
                :stake => sum => :stake,
                :pnl => sum => :pnl,
                :odds => median => :med_odds,
                :payoff => (x -> mean(x .> 0)) => :hit)
    g.roi = 100 .* g.pnl ./ g.stake
    return sort!(g, :pnl, rev = true)
end
export attribution
