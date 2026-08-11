# current_development/orderbook_layer2/l02_l2_ledger.jl
#
# The judge side: one long frame, and a tearsheet over it.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ---------------------------------------------------------------------------------------------
#
# `BacktestLedger` + `generate_tearsheet` is the half of Layer 1 that makes two results from
# different months comparable: a long frame with a fixed schema, a `groupby`, and a set of
# pluggable metrics that never learn anything about the experiment that produced their input.
#
# `Layer2Ledger` is the same shape with the Layer-2 axes bolted on. It deliberately keeps Layer
# 1's column names (`:stake, :pnl, :is_winner, :odds, :selection`) so that every existing metric
# — the six wealth metrics and `BernoulliGammaHurdle` — works on it UNCHANGED. That is what
# makes an L2 growth number comparable to the `hurdle_G_emp` figures in `b21`.
#
# ---------------------------------------------------------------------------------------------
# THE ONE DELIBERATE DEVIATION FROM `generate_tearsheet`
# ---------------------------------------------------------------------------------------------
#
# `BackTesting._compute_wealth_metrics` builds its equity curve as `cumsum(pnl)`.
#
# That is arithmetic accumulation, which is correct for FLAT staking and wrong for everything
# this layer does. Portfolio stakes a FRACTION of a bankroll and compounds once per settlement
# window (`simulate`: `bank *= (1 + pl)`), so a Sharpe or Calmar computed off `cumsum` is
# measuring a strategy nobody ran. The error is silent and flatters early losses.
#
# So `l2_tearsheet` builds the curve the way `Portfolio.simulate` does — group by slate, sum the
# fractional P/L, compound — and hands THAT to the same `compute_metric` calls. The bridge
# already exists: `Portfolio.report` (`src/Portfolio/metrics.jl:73`) evaluates
# `BackTesting.AbstractWealthMetric` against `Trajectory.bankroll`, and reproducing its numbers
# exactly is this file's acceptance gate.
#
# ---------------------------------------------------------------------------------------------
# UNITS (get this wrong and every wealth number is wrong)
# ---------------------------------------------------------------------------------------------
#
#   :stake   FRACTION of bankroll, matching `Trajectory.bets.stake` and `stake_sheet`'s `:frac`.
#   :payoff  UNIT payoff — net return per unit staked. +(odds-1)*(1-commission) on a win,
#            -1 on a loss, 0 on a push.
#   :pnl     `stake * payoff`, therefore also a fraction of bankroll. This is what compounds.
#   :stake_cash / :pnl_cash  the same two in currency, for reading. Never used in a metric.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS CORPUS CANNOT SUPPORT (measured, see NOTES.md)
# ---------------------------------------------------------------------------------------------
#
# The Ireland corpus is **12 settlement windows**. `simulate` compounds once per slate, so every
# drawdown-path metric — max drawdown, Calmar, Ulcer, Burke, Sterling — is being estimated from
# 12 points. They are computed here because they are free, and `PATH_METRICS_UNRELIABLE` names
# them so a reader cannot mistake a number for evidence. Report ROI, growth and the
# ledger-level metrics in `l03` instead.

using DataFrames, Dates, Statistics, Printf

_bt() = BayesianFootball.BackTesting

# ===================================================================
# 1. The frame
# ===================================================================

"""
    Layer2Ledger

Every leg, at every snapshot, under every policy — the analogue of `BackTesting.BacktestLedger`.

One row per `(match_id, group, line, selection, as_of, policy_name, arm)`.

Schema, in four blocks:

**Layer-1 compatible** (do not rename — the shared metrics read these)
`:stake` `:pnl` `:payoff` `:is_winner` `:odds` `:selection` `:family` `:p_model` `:p_market`

**identity** `:match_id` `:slate` `:group` `:line` `:tournament_id` `:stake_cash` `:pnl_cash`

**Layer-2 axes** `:as_of` `:mins_to_ko` `:entry_bucket` `:policy_name` `:arm` `:entry_name`

**price** `:back` `:back_size` `:lay` `:lay_size` `:rel_spread` `:matched` `:odds_close_final`
`:fair_close`

`:odds_close_final` and `:fair_close` are the SAME selection's price and de-vigged probability at
the last pre-kickoff snapshot, broadcast onto every earlier row. They are what make
`PriceDrift` and `ClosingLineValue` computable from a single row, and they are the reason the
ledger is stamped once at build time rather than joined at metric time — a metric that has to
re-derive the close is a metric that can silently disagree with another one.
"""
struct Layer2Ledger
    df::DataFrame
end

Base.show(io::IO, l::Layer2Ledger) = show(io, l.df)
Base.show(io::IO, m::MIME"text/plain", l::Layer2Ledger) = show(io, m, l.df)
Base.length(l::Layer2Ledger)  = nrow(l.df)
Base.isempty(l::Layer2Ledger)  = isempty(l.df)
Base.getindex(l::Layer2Ledger, args...) = getindex(l.df, args...)
DataFrames.nrow(l::Layer2Ledger) = nrow(l.df)

"Wealth metrics whose estimator needs more settlement windows than this corpus has."
const PATH_METRICS_UNRELIABLE = [:CalmarRatio, :BurkeRatio, :SterlingRatio, :SortinoRatio]

"Minimum settlement windows before a drawdown-path metric is worth reading at all."
const MIN_SLATES_FOR_PATH = 25

# ===================================================================
# 2. Entry buckets
# ===================================================================

"""
    ENTRY_BUCKET_EDGES

Upper edges in minutes-to-kickoff. Chosen to match the buckets the corpus was MEASURED on
(see NOTES.md), so a tearsheet row can be read straight against the spread/size table rather
than against a re-bucketing of it.
"""
const ENTRY_BUCKET_EDGES = [5.0, 15.0, 30.0, 60.0, 120.0, 180.0]
const ENTRY_BUCKET_NAMES = ["0-5m", "5-15m", "15-30m", "30-60m", "60-120m", "120-180m", "180m+"]

"Label a lead time. Ordered lexicographically by construction so `sort` gives clock order."
function entry_bucket(mins_to_ko::Real)
    i = searchsortedfirst(ENTRY_BUCKET_EDGES, Float64(mins_to_ko))
    return @sprintf("%d_%s", i, ENTRY_BUCKET_NAMES[min(i, length(ENTRY_BUCKET_NAMES))])
end

add_entry_buckets!(df::AbstractDataFrame) =
    (df.entry_bucket = entry_bucket.(df.mins_to_ko); df)

# ===================================================================
# 3. The compounded curve
# ===================================================================

"""
    l2_curve(sub_df) -> Vector{Float64}

The bankroll series `Portfolio.simulate` would have produced from these legs.

Groups by `:slate`, sums fractional P/L within each window, and compounds. Starts at 1.0 so the
series length is `n_slates + 1`, matching `Trajectory.bankroll` exactly.

Slates are sorted before compounding, without exception. Final wealth is order-invariant but
every drawdown statistic is not: on the reference 628-match book, Martin ranged 52 to 144 across
random orderings of the identical returns.
"""
function l2_curve(sub::AbstractDataFrame)
    isempty(sub) && return [1.0]
    g = combine(groupby(sub, :slate), :pnl => sum => :pl)
    sort!(g, :slate)
    bank = 1.0
    out  = Float64[1.0]
    for pl in g.pl
        bank *= (1.0 + pl)
        push!(out, bank)
    end
    return out
end

"""
    l2_path_metrics(sub_df) -> NamedTuple

`Portfolio.path_metrics` computed from a ledger slice rather than from a `Trajectory`, so that a
tearsheet row and a `Portfolio.report` on the same legs agree. Same field names on purpose.
"""
function l2_path_metrics(sub::AbstractDataFrame)
    bk = l2_curve(sub)
    n_slates = length(bk) - 1
    rm  = accumulate(max, bk)
    dd  = (bk .- rm) ./ rm .* 100
    ui  = max(sqrt(mean(dd .^ 2)), 1e-9)
    tr  = (bk[end] - 1.0) * 100
    mdd = minimum(dd)
    stk = sum(sub.stake)
    slate_pl = n_slates == 0 ? Float64[] :
               sort(combine(groupby(sub, :slate), :pnl => sum => :pl), :slate).pl

    return (final            = bk[end],
            roi              = stk > 0 ? 100 * sum(sub.pnl) / stk : 0.0,
            growth_per_slate = isempty(slate_pl) ? 0.0 : mean(log.(1.0 .+ slate_pl)),
            mdd              = mdd,
            ulcer            = ui,
            calmar           = mdd < 0 ? tr / abs(mdd) : 0.0,
            martin           = tr / ui,
            n_slates         = n_slates,
            path_reliable    = n_slates >= MIN_SLATES_FOR_PATH)
end

# ===================================================================
# 4. The tearsheet
# ===================================================================

"""
    l2_tearsheet(ledger; groupby_cols, metrics, dist_metrics, bootstrap) -> DataFrame

Group, then judge. The Layer-2 analogue of `BackTesting.generate_tearsheet`.

Default grouping is the two axes this stream exists to compare — entry rule and time bucket.
Swap in `[:policy_name]` for the trust race, `[:family]` for attribution, `[:arm]` to read the
frozen/live decomposition.

`metrics` are `BackTesting.AbstractWealthMetric`s and are evaluated on the COMPOUNDED curve (see
the header). `dist_metrics` are `BackTesting.AbstractDistributionalMetric`s and receive the raw
slice, which is how `l03`'s price-level metrics get access to the book columns.

`bootstrap = true` adds a match-clustered ROI interval per group via `Portfolio.bootstrap_roi`.
On this corpus that interval is the honest headline and the point estimate is not — resampling
individual legs instead of matches would understate it badly, because several legs on one match
share one scoreline.
"""
function l2_tearsheet(l::Layer2Ledger;
                      groupby_cols::Vector{Symbol} = [:entry_name, :entry_bucket],
                      metrics = _default_wealth_metrics(),
                      dist_metrics = _default_dist_metrics(),
                      bootstrap::Bool = true)
    df = l.df
    isempty(df) && return DataFrame()

    cols = filter(c -> hasproperty(df, c), groupby_cols)
    isempty(cols) && error("l2_tearsheet: none of $groupby_cols present in the ledger")

    out = combine(groupby(df, cols)) do sub
        stats = _l2_basic_stats(sub)
        path  = l2_path_metrics(sub)
        wealth = Dict{Symbol,Float64}()
        curve  = l2_curve(sub)
        for m in metrics
            wealth[_bt().metric_symbol(m)] = round(_bt().compute_metric(m, curve), digits = 4)
        end
        dist = Dict{Symbol,Any}()
        for m in dist_metrics
            for (k, v) in pairs(_bt().compute_distributional_metric(m, sub))
                dist[k] = v
            end
        end
        ci = bootstrap ? _clustered_roi(sub) : (roi_ci_lo = NaN, roi_ci_hi = NaN)
        merge(stats, path, ci, wealth, dist)
    end

    return sort!(out, cols)
end

function _l2_basic_stats(sub::AbstractDataFrame)
    active = sub.stake .> 1e-9
    n      = count(active)
    stk    = sum(sub.stake[active]; init = 0.0)
    pl     = sum(sub.pnl[active]; init = 0.0)
    wins   = count(i -> active[i] && coalesce(sub.is_winner[i], false), eachindex(active))
    return (legs        = nrow(sub),
            bets        = n,
            matches     = length(unique(sub.match_id)),
            turnover    = round(stk, digits = 4),
            profit      = round(pl, digits = 4),
            win_rate    = n > 0 ? round(100 * wins / n, digits = 1) : 0.0,
            med_odds    = n > 0 ? round(median(sub.odds[active]), digits = 3) : NaN,
            med_lead    = hasproperty(sub, :mins_to_ko) && n > 0 ?
                          round(median(sub.mins_to_ko[active]), digits = 1) : NaN)
end

"Match-clustered ROI interval, delegated to Portfolio so both layers use one implementation."
function _clustered_roi(sub::AbstractDataFrame)
    bets = DataFrame(match_id = sub.match_id, stake = sub.stake, pnl = sub.pnl)
    ci = BayesianFootball.Portfolio.bootstrap_roi(bets)
    return (roi_ci_lo = round(ci.lo, digits = 2), roi_ci_hi = round(ci.hi, digits = 2))
end

_default_wealth_metrics() = [_bt().CumulativeWealth(), _bt().SharpeRatio()]
_default_dist_metrics()   = [_bt().BernoulliGammaHurdle()]

# ===================================================================
# 5. Attribution and the health warning
# ===================================================================

"""
    l2_attribution(ledger; by = :family) -> DataFrame

Stake, P/L, ROI and hit rate per selection family — `Portfolio.attribution` over a ledger.

The first thing to look at when a headline moves. On the reference book 83% of profit came from
1X2, a family the model has no measurable advantage on, which is the single clearest example of
why a headline number is not a finding.
"""
function l2_attribution(l::Layer2Ledger; by::Symbol = :family)
    isempty(l.df) && return DataFrame()
    g = combine(groupby(l.df, by),
                nrow => :legs,
                :stake  => sum    => :stake,
                :pnl    => sum    => :pnl,
                :odds   => median => :med_odds,
                :payoff => (x -> mean(x .> 0)) => :hit)
    g.roi = 100 .* g.pnl ./ g.stake
    return sort!(g, :pnl, rev = true)
end

"""
    path_warning(tearsheet) -> String

The sentence that must accompany any drawdown-derived column from this corpus.

Returns `""` when every group had enough settlement windows, so it is safe to interpolate
unconditionally into a report.
"""
function path_warning(ts::DataFrame)
    (isempty(ts) || !hasproperty(ts, :n_slates)) && return ""
    worst = minimum(ts.n_slates)
    worst >= MIN_SLATES_FOR_PATH && return ""
    return "PATH METRICS UNRELIABLE: smallest group has $worst settlement windows " *
           "(need >= $MIN_SLATES_FOR_PATH). Columns $(join(PATH_METRICS_UNRELIABLE, ", ")) " *
           "plus :mdd, :ulcer, :calmar, :martin are noise here — read :roi, " *
           ":growth_per_slate, the clustered CI and the l03 price metrics instead."
end
