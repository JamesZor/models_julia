# current_development/orderbook_layer2/l03_l2_metrics.jl
#
# Layer-2 metrics: what a 12-slate corpus can actually answer.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ---------------------------------------------------------------------------------------------
#
# The Ireland order-book corpus is 81 fixtures in 12 settlement windows. After curation that is
# ~500 graded legs, ~60 per bucket on an 8-bucket timing curve. Any P/L difference between entry
# times will sit inside its own confidence interval, and every drawdown-path metric is being
# estimated from 12 points.
#
# The response is not a bigger sweep. It is a better estimator.
#
# The same replay produces observations at three very different rates:
#
#   ~100,000   PRICE observations   (81 fixtures x 9 markets x ~2.5 selections x ~28 snapshots)
#   ~1,000     MODEL-PICKED legs    (only the selections the model wanted, at each snapshot)
#   ~500       GRADED P/L legs      (one entry per leg, settled)
#
# `PriceDrift` reads the first, `ClosingLineValue` the second, `FillCost` the first with a
# notional stake attached. All three subtype `BackTesting.AbstractDistributionalMetric`, so they
# drop into `l2_tearsheet` — and into Layer 1's `generate_tearsheet` — with no plumbing.
#
# ---------------------------------------------------------------------------------------------
# THE ESTIMATOR HIERARCHY, AND HOW TO READ A DISAGREEMENT
# ---------------------------------------------------------------------------------------------
#
#   PriceDrift          no model at all. Answers "when is the book cheapest to trade".
#                       If this is flat, entry time cannot matter for execution reasons, and any
#                       P/L difference across buckets is noise or model-edge decay.
#   ClosingLineValue    the model's own picks, judged against the closing consensus. Answers
#                       "is the edge real, and does it survive being taken early".
#   BernoulliGammaHurdle / ROI   the outcome. Lowest power. CONFIRMS or CONTRADICTS; it does not
#                       adjudicate on its own at this n.
#
# When P/L disagrees with CLV at this sample size, believe CLV. That is not a stylistic
# preference: CLV has ~2x the observations and vastly lower variance per observation, because it
# is not carrying the binomial noise of whether the ball went in.
#
# ---------------------------------------------------------------------------------------------
# CLUSTERING IS NOT OPTIONAL
# ---------------------------------------------------------------------------------------------
#
# Nine markets on one fixture share one scoreline, and ~28 snapshots of one market share one
# order book. Treating those as independent would shrink every interval by roughly sqrt(28) and
# manufacture significance out of nothing. Every interval below resamples MATCHES.

using DataFrames, Dates, Statistics, Random, Printf

# ===================================================================
# 0. Shared machinery
# ===================================================================

"""
    _cluster_boot(v, match_ids; B = 4000, seed = 1) -> (lo, hi, sd)

Percentile CI on `mean(v)`, resampling by match.

Mirrors `Portfolio.bootstrap_roi`'s clustering exactly — same resampling scheme, same B, same
seed default — so a CLV interval and an ROI interval from the same slice are directly
comparable rather than merely similar.
"""
function _cluster_boot(v::AbstractVector{<:Real}, match_ids::AbstractVector;
                       B::Int = 4000, seed::Int = 1)
    (isempty(v) || length(v) != length(match_ids)) && return (lo = NaN, hi = NaN, sd = NaN)
    rng  = Random.MersenneTwister(seed)
    mids = unique(match_ids)
    idxs = Dict(m => findall(==(m), match_ids) for m in mids)
    n    = length(mids)
    out  = Vector{Float64}(undef, B)
    @inbounds for b in 1:B
        acc, cnt = 0.0, 0
        for _ in 1:n
            for i in idxs[mids[rand(rng, 1:n)]]
                acc += v[i]; cnt += 1
            end
        end
        out[b] = cnt == 0 ? NaN : acc / cnt
    end
    u = filter(!isnan, out)
    isempty(u) && return (lo = NaN, hi = NaN, sd = NaN)
    return (lo = quantile(u, 0.025), hi = quantile(u, 0.975), sd = std(u))
end

"Best (first) level of a depth column that may hold either a vector of levels or a scalar."
_top(x) = x isa AbstractVector ? (isempty(x) ? NaN : Float64(first(x))) : Float64(x)

"All levels of a depth column, as a vector, whatever it was stored as."
_levels(x) = x isa AbstractVector ? Float64.(x) : [Float64(x)]

_finite(v) = filter(x -> isfinite(x), v)

# ===================================================================
# 1. PriceDrift — the high-power, model-free estimator
# ===================================================================

"""
    PriceDrift <: AbstractDistributionalMetric

How the price moved between entry and the close, with no model involved.

`drift = log(odds_close_final / odds_entry)`, evaluated per row, from the BACKER's point of view:

  * `drift > 0` — the price got LONGER after entry. Waiting would have paid; you entered early.
  * `drift < 0` — the price SHORTENED. The market came toward you and entering early paid.

This is the estimator with ~100k observations, so it is the one that can actually resolve an
entry-time curve. Read it first. If `mean_drift` is flat across `:entry_bucket`, then execution
timing does not matter for price reasons and any P/L gradient must be model-edge decay or noise
— which is a finding, and a cheap one.

Reported per group: mean and median drift, the share of legs where waiting would have paid, and
a match-clustered interval on the mean.

# Fields
- `min_odds::Float64` — ignore legs priced below this. Betfair's tick grid is coarse at short
  prices, so a 1.01 → 1.02 move is a 1% "drift" that is really one tick of granularity.
"""
Base.@kwdef struct PriceDrift <: BayesianFootball.BackTesting.AbstractDistributionalMetric
    min_odds::Float64 = 1.10
    bootstrap::Bool   = true
end

BayesianFootball.BackTesting.metric_description(::PriceDrift) =
    "log(odds_close / odds_entry) per leg, backer's sign convention. Model-free; the " *
    "highest-power estimator of whether entry time matters for execution."

function BayesianFootball.BackTesting.compute_distributional_metric(m::PriceDrift,
                                                                    sub::AbstractDataFrame)
    (isempty(sub) || !hasproperty(sub, :odds_close_final)) && return _empty_drift()

    ok = findall(i -> begin
            oe, oc = sub.odds[i], sub.odds_close_final[i]
            isfinite(oe) && isfinite(oc) && oe >= m.min_odds && oc > 1.0 && oe > 1.0
        end, 1:nrow(sub))
    isempty(ok) && return _empty_drift()

    d = [log(sub.odds_close_final[i] / sub.odds[i]) for i in ok]
    ci = m.bootstrap ? _cluster_boot(d, sub.match_id[ok]) : (lo = NaN, hi = NaN, sd = NaN)

    return (drift_n         = length(d),
            drift_mean      = round(mean(d), digits = 5),
            drift_med       = round(median(d), digits = 5),
            drift_wait_paid = round(100 * count(>(0), d) / length(d), digits = 1),
            drift_ci_lo     = round(ci.lo, digits = 5),
            drift_ci_hi     = round(ci.hi, digits = 5))
end

_empty_drift() = (drift_n = 0, drift_mean = NaN, drift_med = NaN,
                  drift_wait_paid = NaN, drift_ci_lo = NaN, drift_ci_hi = NaN)

# ===================================================================
# 2. ClosingLineValue — is the edge real?
# ===================================================================

"""
    ClosingLineValue <: AbstractDistributionalMetric

The model's picks, judged against the closing consensus rather than against the result.

Two readings, both reported:

  * **fair CLV** `log(odds_entry * fair_close)` — the log expected value of a unit bet at the
    price taken, evaluated at the DE-VIGGED closing probability. Positive means the price taken
    was +EV against the market's own final opinion. This is the one to quote: it is denominated
    in log-growth, so it is directly comparable to `growth_per_slate`.
  * **beat rate** the share of legs where `odds_entry > odds_close` — the folk version. Reported
    because it is what everyone else quotes, and because a large gap between the two means the
    vig is doing the work rather than the model.

Why this instead of P/L: CLV strips the binomial noise of whether the ball went in. At ~1k legs
it resolves differences that ~500 graded legs cannot. When the two disagree here, believe CLV.

The honest caveat, stated because it bounds every conclusion drawn from this metric: CLV
measures agreement with the closing market, so a model that is *right* where the market is
*wrong* is penalised. It is evidence that an edge is REAL, not that it is PROFITABLE. That is
why `BernoulliGammaHurdle` still runs alongside it.
"""
Base.@kwdef struct ClosingLineValue <: BayesianFootball.BackTesting.AbstractDistributionalMetric
    min_odds::Float64 = 1.10
    bootstrap::Bool   = true
end

BayesianFootball.BackTesting.metric_description(::ClosingLineValue) =
    "log(odds_entry * fair_close) per leg — log-EV of the taken price against the de-vigged " *
    "closing probability. Evidence that an edge is real; not that it is profitable."

function BayesianFootball.BackTesting.compute_distributional_metric(m::ClosingLineValue,
                                                                    sub::AbstractDataFrame)
    (isempty(sub) || !hasproperty(sub, :fair_close)) && return _empty_clv()

    ok = findall(i -> begin
            oe, fc = sub.odds[i], sub.fair_close[i]
            isfinite(oe) && isfinite(fc) && oe >= m.min_odds && 0 < fc < 1
        end, 1:nrow(sub))
    isempty(ok) && return _empty_clv()

    clv = [log(sub.odds[i] * sub.fair_close[i]) for i in ok]
    ci  = m.bootstrap ? _cluster_boot(clv, sub.match_id[ok]) : (lo = NaN, hi = NaN, sd = NaN)

    beat = if hasproperty(sub, :odds_close_final)
        j = filter(i -> isfinite(sub.odds_close_final[i]), ok)
        isempty(j) ? NaN :
            round(100 * count(i -> sub.odds[i] > sub.odds_close_final[i], j) / length(j), digits = 1)
    else
        NaN
    end

    return (clv_n      = length(clv),
            clv_mean   = round(mean(clv), digits = 5),
            clv_med    = round(median(clv), digits = 5),
            clv_ci_lo  = round(ci.lo, digits = 5),
            clv_ci_hi  = round(ci.hi, digits = 5),
            clv_pos    = round(100 * count(>(0), clv) / length(clv), digits = 1),
            beat_close = beat)
end

_empty_clv() = (clv_n = 0, clv_mean = NaN, clv_med = NaN, clv_ci_lo = NaN,
                clv_ci_hi = NaN, clv_pos = NaN, beat_close = NaN)

# ===================================================================
# 3. FillCost — what the book actually charges you
# ===================================================================

"""
    FillCost(; stakes, ...) <: AbstractDistributionalMetric

What it costs to take the price you were quoted, at several notional order sizes.

Three components, all per leg:

  * **half-spread** `(mid - best_back) / mid` — the immediate cost of crossing to the bid.
  * **size shortfall** the share of legs where the top of book cannot absorb the order.
  * **slippage** the VWAP cost of walking down the ladder to fill the whole order, minus the
    half-spread. Zero whenever the top level absorbs it.

`stakes` is a vector on purpose. The corpus says top-of-book size quadruples from T-240 to the
off (£1,906 → £7,641 on MATCH_ODDS) while spread barely moves, so the pre-registered hypothesis
is that **entry time costs capacity, not price** — and therefore that these curves should be
FLAT in `:entry_bucket` at small stakes and separate only at large ones. Passing one stake size
would make that untestable; the shape across sizes is the whole result.

Sizes are in the same currency as the book's volumes (GBP here).

**Do not build this on `market_matched`.** That column is NULL on every Ireland fixture before
2026-08-02 (23.8% of rows) — a gate built on it would silently apply to 18 of 81 fixtures.
Top-of-book size is 100% populated and is what this metric uses.
"""
Base.@kwdef struct FillCost <: BayesianFootball.BackTesting.AbstractDistributionalMetric
    stakes::Vector{Float64} = [10.0, 100.0, 1000.0]
    min_odds::Float64       = 1.10
end

BayesianFootball.BackTesting.metric_description(m::FillCost) =
    "Half-spread, size shortfall and VWAP slippage at notional stakes " *
    join(m.stakes, "/") * ". Tests whether entry time costs price or capacity."

function BayesianFootball.BackTesting.compute_distributional_metric(m::FillCost,
                                                                    sub::AbstractDataFrame)
    (isempty(sub) || !hasproperty(sub, :back) || !hasproperty(sub, :lay)) &&
        return _empty_fill(m)

    hs   = Float64[]
    slip = Dict(s => Float64[] for s in m.stakes)
    shrt = Dict(s => Int[]     for s in m.stakes)

    for r in eachrow(sub)
        bb, bl = _top(r.back), _top(r.lay)
        (isfinite(bb) && isfinite(bl) && bb >= m.min_odds && bl > bb) || continue
        mid = (bb + bl) / 2
        push!(hs, (mid - bb) / mid)

        prices = _finite(_levels(r.back))
        sizes  = hasproperty(sub, :back_size) ? _finite(_levels(r.back_size)) : Float64[]
        n      = min(length(prices), length(sizes))

        for s in m.stakes
            if n == 0
                push!(shrt[s], 0); push!(slip[s], 0.0); continue
            end
            push!(shrt[s], sizes[1] < s ? 1 : 0)
            push!(slip[s], (_vwap(prices, sizes, s, n) - bb) / bb * -1)  # cost, so sign-flipped
        end
    end

    isempty(hs) && return _empty_fill(m)

    base = (fill_n = length(hs), half_spread = round(mean(hs), digits = 5))
    for s in m.stakes
        k = Int(round(s))
        base = merge(base, NamedTuple{(Symbol("slip_$k"), Symbol("short_$k"))}((
            round(mean(slip[s]), digits = 5),
            round(100 * mean(shrt[s]), digits = 1))))
    end
    return base
end

"""
    _vwap(prices, sizes, stake, n) -> Float64

Volume-weighted average odds of walking `stake` down the back ladder.

For a back bet you commit money and receive `stake_i * price_i`, so the effective odds are
`sum(filled_i * price_i) / stake`. Any unfilled remainder is charged at the WORST available
level rather than dropped — dropping it would report a book that cannot fill your order as
cheaper than one that can, which is exactly backwards.
"""
function _vwap(prices::Vector{Float64}, sizes::Vector{Float64}, stake::Float64, n::Int)
    left, ret = stake, 0.0
    @inbounds for i in 1:n
        take = min(left, sizes[i])
        ret += take * prices[i]
        left -= take
        left <= 0 && break
    end
    left > 0 && (ret += left * prices[n])       # remainder at the worst level seen
    return ret / stake
end

function _empty_fill(m::FillCost)
    base = (fill_n = 0, half_spread = NaN)
    for s in m.stakes
        k = Int(round(s))
        base = merge(base, NamedTuple{(Symbol("slip_$k"), Symbol("short_$k"))}((NaN, NaN)))
    end
    return base
end

# ===================================================================
# 4. The Layer-2 default set
# ===================================================================

"""
    l2_metrics(; stakes) -> Vector{AbstractDistributionalMetric}

The three Layer-2 metrics in the order they should be READ, not the order they run:
drift (model-free, ~100k obs) → CLV (model's picks, ~1k) → fill cost (execution).

Pair with `BackTesting.BernoulliGammaHurdle()` for the outcome side, which `l2_tearsheet`
includes by default.
"""
l2_metrics(; stakes::Vector{Float64} = [10.0, 100.0, 1000.0]) =
    [PriceDrift(), ClosingLineValue(), FillCost(stakes = stakes)]
