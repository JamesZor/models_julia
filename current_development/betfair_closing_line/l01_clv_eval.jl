# current_development/betfair_closing_line/l01_clv_eval.jl
#
# Loader: time-series evaluation of Betfair (free-tier, last-traded-only) prices
# against a Bayesian model's posterior predictive distributions.
#
# Research emphasis: EDGE / ENTRY TIMING.
#   - Stage 0: build a horizon-resolved, vig-removed market panel (LOCF on sparse ticks)
#   - Stage 1: edge decay over time   (log-loss / Brier model vs market, bootstrap CI)
#   - Stage 2: does the model anticipate the closing-line move? (CLV alpha)
#   - Stage 3: CLV -> realised P&L (filtered-bet ROI / log-growth by entry horizon)
#   - Stage 4: microstructure rigour (Roll effective spread) + PIT calibration check
#
# Data-model note: a last-traded price is a piecewise-constant STEP process — a price is
# the market state until the next trade overwrites it. So a horizon with "no tick" is not
# missing; we Last-Observation-Carry-Forward (LOCF) the most recent trade. Within a trailing
# window we time-weight (TWA) to denoise the bid-ask bounce. The unobservable half-spread is
# quantified via Roll (1984).
#
# Reuses (mirrored locally where trivial, for prototype robustness):
#   - TWA estimator           ~ src/Data/betfair_util.jl::estimate_price(TWAEstimator)
#   - vig removal / overround ~ src/Data/betfair_util.jl::summarize_odds / summarize_betfair_market
#   - log-loss                ~ src/evaluation/metrics_methods/logloss.jl::calc_logloss
#   - PIT (market quantile)   ~ src/evaluation/metrics_methods/miq.jl::get_miq
#   - edge regression pattern ~ src/evaluation/metrics_methods/glm_edge.jl
# Grading reuses BayesianFootball.Data.grade_selection directly (single source of truth).

using DataFrames
using Statistics
using Random
using GLM
using HypothesisTests
using Distributions
using Printf
using Plots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# The five best backtest growth-rate selections.
const TARGET_SELECTIONS = [:over_25, :under_25, :under_35, :btts_yes, :under_15]

# Pre-game horizons in minutes_to_kickoff (negative = before kickoff).
const DEFAULT_HORIZONS = [-1440.0, -720.0, -360.0, -180.0, -90.0, -45.0, -20.0, -5.0]

# Closing reference (the line we measure CLV against).
const CLOSE_HORIZON = -2.0
const CLOSE_WINDOW  = 20.0

# ---------------------------------------------------------------------------
# Thin scoring helpers (identical to the src versions; kept local so the
# prototype runs even if internal, non-exported names move)
# ---------------------------------------------------------------------------

# binary cross-entropy; lower is better
_logloss(p::Real, y::Real) = (pc = clamp(p, 1e-15, 1.0 - 1e-15); -(y * log(pc) + (1.0 - y) * log(1.0 - pc)))

# Brier score
_brier(p::Real, y::Real) = (p - y)^2

# Probability Integral Transform: fraction of posterior draws <= the market prob
_miq(samples::AbstractVector{<:Real}, mprob::Real) = sum(<=(mprob), samples) / length(samples)

# Time-weighted average price (mirrors TWAEstimator)
function _twa(prices::AbstractVector, mins::AbstractVector)
    length(prices) == 1 && return float(prices[1])
    p = sortperm(mins)
    sp = float.(prices[p]); sm = float.(mins[p])
    durations = diff(sm)
    md = isempty(durations) ? 0.0 : mean(durations)
    push!(durations, md)
    total = sum(durations)
    total <= 0 && return mean(sp)        # all ticks at the same instant
    return sum(sp .* durations) / total
end

# bootstrap percentile CI for the mean of x
function _boot_ci(x::AbstractVector{<:Real}, n_boot::Int; α::Float64 = 0.05, rng = Random.default_rng())
    n = length(x)
    n == 0 && return (NaN, NaN)
    means = Vector{Float64}(undef, n_boot)
    @inbounds for b in 1:n_boot
        s = 0.0
        for _ in 1:n
            s += x[rand(rng, 1:n)]
        end
        means[b] = s / n
    end
    return (quantile(means, α / 2), quantile(means, 1 - α / 2))
end

# ---------------------------------------------------------------------------
# Stage 0a: snapshot the market at a single horizon τ (with LOCF)
# ---------------------------------------------------------------------------

"""
    snapshot_at(bf_long, τ; window_width, overround_limits) -> DataFrame

Vig-removed market snapshot at horizon `τ` (minutes_to_kickoff, negative).
For each (match, market, line, selection): TWA over the trailing window
`[τ-window_width, τ]`; if that window is empty, LOCF the most recent tick `≤ τ`.
Overround is computed per (match, market, line) and used to strip the vig.

Columns: match_id, market_name, market_line, selection, odds, prob_implied,
         prob_fair, overround, n_ticks, locf, horizon.
"""
function snapshot_at(
    bf_long::DataFrame,
    τ::Float64;
    window_width::Float64 = 60.0,
    overround_limits::Tuple{Float64,Float64} = (0.90, 1.10),
)
    pre = filter(r -> r.minutes_to_kickoff <= τ, bf_long)
    isempty(pre) && return DataFrame()

    price_df = combine(groupby(pre, [:match_id, :market_name, :market_line, :selection])) do sdf
        win = filter(r -> r.minutes_to_kickoff >= τ - window_width, sdf)
        if nrow(win) >= 1
            (odds = _twa(win.traded_price, win.minutes_to_kickoff), n_ticks = nrow(win), locf = false)
        else
            j = argmax(sdf.minutes_to_kickoff)         # most recent trade <= τ
            (odds = float(sdf.traded_price[j]), n_ticks = 0, locf = true)
        end
    end

    # overround per market instance, then strip vig
    transform!(groupby(price_df, [:match_id, :market_name, :market_line]),
        :odds => (o -> sum(1.0 ./ o)) => :overround)
    filter!(r -> overround_limits[1] <= r.overround <= overround_limits[2], price_df)

    price_df.prob_implied = 1.0 ./ price_df.odds
    price_df.prob_fair    = price_df.prob_implied ./ price_df.overround
    price_df.horizon     .= τ
    return price_df
end

# ---------------------------------------------------------------------------
# Stage 0b: model scalars + full panel
# ---------------------------------------------------------------------------

"""Reduce the PPD posterior arrays to scalar mean/median probabilities."""
function model_scalars(ppd)
    return select(
        transform(ppd.df,
            :distribution => ByRow(mean)   => :prob_model_mean,
            :distribution => ByRow(median) => :prob_model_median),
        :match_id, :market_name, :market_line, :selection,
        :prob_model_mean, :prob_model_median)
end

"""Grade a panel's `is_winner` via the canonical Data.grade_selection, keeping NaNs out."""
function grade_panel!(panel::DataFrame, matches::DataFrame)
    m = select(matches, :match_id, :home_score, :away_score)
    leftjoin!(panel, m, on = :match_id)
    panel.is_winner = BayesianFootball.Data.grade_selection.(
        panel.market_name, panel.market_line, panel.selection,
        panel.home_score, panel.away_score)
    select!(panel, Not([:home_score, :away_score]))
    return panel
end

"""
    build_clv_panel(ppd, ds; kwargs...) -> DataFrame

Long panel — one row per (match, target-selection, horizon) — carrying the model
scalar probs, the horizon market `prob_fair`/`odds`, the closing `prob_fair_close`/
`odds_close`, and graded `is_winner`. Full markets are summarised (so vig removal is
correct) before filtering down to `targets`.
"""
function build_clv_panel(
    ppd, ds;
    horizons::Vector{Float64} = DEFAULT_HORIZONS,
    window_width::Float64 = 60.0,
    targets::Vector{Symbol} = TARGET_SELECTIONS,
    close_horizon::Float64 = CLOSE_HORIZON,
    close_window::Float64 = CLOSE_WINDOW,
)
    mp = model_scalars(ppd)
    bf = ds.betfair_odds

    close_snap = snapshot_at(bf, close_horizon; window_width = close_window)
    close_ref  = select(close_snap,
        :match_id, :market_name, :market_line, :selection,
        :odds => :odds_close, :prob_fair => :prob_fair_close)

    frames = DataFrame[]
    for τ in horizons
        snap = snapshot_at(bf, τ; window_width = window_width)
        isempty(snap) && continue
        j = innerjoin(snap, mp, on = [:match_id, :market_name, :market_line, :selection])
        leftjoin!(j, close_ref, on = [:match_id, :market_name, :market_line, :selection])
        push!(frames, j)
    end
    isempty(frames) && error("No market snapshots built — check betfair coverage / horizons.")

    panel = reduce(vcat, frames)
    grade_panel!(panel, ds.matches)
    filter!(r -> r.selection in targets, panel)
    return panel
end

"""Per (selection, horizon) coverage diagnostics: match count, LOCF fraction, mean ticks."""
function coverage_table(panel::DataFrame)
    ct = combine(groupby(panel, [:selection, :horizon]),
        nrow => :n_matches,
        :locf    => mean => :locf_frac,
        :n_ticks => mean => :mean_ticks)
    return sort!(ct, [:selection, :horizon])
end

# ---------------------------------------------------------------------------
# Stage 1: edge decay over time (HEADLINE)
# ---------------------------------------------------------------------------

"""
    edge_by_horizon(panel; group, n_boot) -> DataFrame

Per group (default `[:horizon]`, pass `[:selection, :horizon]` for a breakdown):
mean model vs market log-loss and Brier, their `diff = model - market`
(negative ⇒ model better), and a bootstrap CI on `diff_ll`.
"""
function edge_by_horizon(panel::DataFrame; group::Vector{Symbol} = [:horizon], n_boot::Int = 2000)
    df = dropmissing(panel, [:prob_fair, :prob_model_mean, :is_winner])
    rows = DataFrame()
    for sub in groupby(df, group)
        y   = Float64.(sub.is_winner)
        llm = _logloss.(sub.prob_model_mean, y)
        llk = _logloss.(sub.prob_fair, y)
        lo, hi = _boot_ci(llm .- llk, n_boot)
        nt = NamedTuple(k => sub[1, k] for k in group)
        push!(rows, merge(nt, (
            n           = nrow(sub),
            model_ll    = mean(llm),
            market_ll   = mean(llk),
            diff_ll     = mean(llm) - mean(llk),
            diff_ll_lo  = lo,
            diff_ll_hi  = hi,
            model_brier = mean(_brier.(sub.prob_model_mean, y)),
            market_brier= mean(_brier.(sub.prob_fair, y)),
            diff_brier  = mean(_brier.(sub.prob_model_mean, y)) - mean(_brier.(sub.prob_fair, y)),
        )); promote = true, cols = :union)
    end
    return sort!(rows, group)
end

# ---------------------------------------------------------------------------
# Stage 2: does the model anticipate the closing-line move? (CLV alpha)
# ---------------------------------------------------------------------------

"""
    clv_alpha(panel; group) -> DataFrame

Per horizon:
  model_signal = prob_model - prob_fair_τ ;  realized_move = prob_fair_close - prob_fair_τ
  β from OLS realized_move ~ model_signal  (β>0 ⇒ model predicts the line's direction)
  directional hit-rate of sign(model_signal)==sign(move) + binomial test vs 0.5
  mean CLV = mean log(odds_τ / odds_close)  (>0 ⇒ took a bigger price than close)
"""
function clv_alpha(panel::DataFrame; group::Vector{Symbol} = [:horizon])
    df = dropmissing(panel, [:prob_fair, :prob_fair_close, :prob_model_mean, :odds, :odds_close])
    df.model_signal  = df.prob_model_mean .- df.prob_fair
    df.realized_move = df.prob_fair_close .- df.prob_fair
    df.clv           = log.(df.odds ./ df.odds_close)

    rows = DataFrame()
    for sub in groupby(df, group)
        n = nrow(sub)
        if n >= 10 && std(sub.model_signal) > 1e-9
            m  = lm(@formula(realized_move ~ model_signal), sub)
            ct = coeftable(m)
            β  = coef(m)[2]; se = stderror(m)[2]; pβ = ct.cols[4][2]
        else
            β = NaN; se = NaN; pβ = NaN
        end
        mask = sub.model_signal .!= 0.0
        nb   = count(mask)
        hits = count(sign.(sub.model_signal[mask]) .== sign.(sub.realized_move[mask]))
        hr   = nb > 0 ? hits / nb : NaN
        hp   = nb > 0 ? pvalue(BinomialTest(hits, nb, 0.5)) : NaN
        nt = NamedTuple(k => sub[1, k] for k in group)
        push!(rows, merge(nt, (
            n = n, beta = β, beta_se = se, beta_p = pβ,
            hit_rate = hr, hit_n = nb, hit_p = hp,
            mean_clv = mean(sub.clv),
        )); promote = true, cols = :union)
    end
    return sort!(rows, group)
end

# ---------------------------------------------------------------------------
# Stage 3: CLV -> realised P&L by entry horizon
# ---------------------------------------------------------------------------

"""
    entry_timing_pnl(panel; edge_threshold, group) -> DataFrame

Flat-stake filtered-bet sim. Enter at horizon τ when prob_model - prob_fair_τ > threshold;
settle on is_winner at the horizon odds. Reports n_bets, ROI (mean(odds*y - 1)),
mean log-growth (geometric, equal-weighted), and hit rate.
"""
function entry_timing_pnl(panel::DataFrame; edge_threshold::Float64 = 0.02, group::Vector{Symbol} = [:horizon])
    df = dropmissing(panel, [:prob_fair, :prob_model_mean, :is_winner, :odds])
    df.edge = df.prob_model_mean .- df.prob_fair
    bets = filter(r -> r.edge > edge_threshold, df)

    rows = DataFrame()
    for sub in groupby(bets, group)
        y   = Float64.(sub.is_winner)
        ret = sub.odds .* y .- 1.0
        nt  = NamedTuple(k => sub[1, k] for k in group)
        push!(rows, merge(nt, (
            n_bets          = nrow(sub),
            roi             = mean(ret),
            mean_log_growth = mean(log.(1.0 .+ clamp.(ret, -0.999, Inf))),
            hit             = mean(y),
            avg_odds        = mean(sub.odds),
        )); promote = true, cols = :union)
    end
    return sort!(rows, group)
end

# ---------------------------------------------------------------------------
# Stage 4a: Roll (1984) effective-spread estimate
# ---------------------------------------------------------------------------

"""
    roll_spread(bf_long; targets) -> DataFrame

Roll's implied effective spread from last-traded returns:
  s ≈ 2·√(−Cov(Δp_t, Δp_{t−1}))   (in decimal-odds units)
Pools the lag-1 autocovariance of tick-to-tick price changes across matches per selection.
A negative covariance is the signature of bid-ask bounce; positive ⇒ estimator undefined (NaN).
"""
function roll_spread(bf_long::DataFrame; targets::Vector{Symbol} = TARGET_SELECTIONS)
    rows = DataFrame()
    for sub in groupby(filter(r -> r.selection in targets, bf_long), :selection)
        covs = Float64[]
        for g in groupby(sub, :match_id)
            s = sort(g, :minutes_to_kickoff)
            length(s.traded_price) < 3 && continue
            dp = diff(s.traded_price)
            length(dp) < 2 && continue
            push!(covs, Statistics.cov(dp[1:end-1], dp[2:end]))
        end
        c = isempty(covs) ? NaN : mean(covs)
        push!(rows, (
            selection   = sub.selection[1],
            roll_cov    = c,
            roll_spread = (!isnan(c) && c < 0) ? 2 * sqrt(-c) : NaN,
            n_matches   = length(covs),
        ); promote = true)
    end
    return rows
end

# ---------------------------------------------------------------------------
# Stage 4b: PIT calibration of the posterior vs the closing line (supporting check)
# ---------------------------------------------------------------------------

"""
    pit_calibration(ppd, ds; kwargs...) -> NamedTuple

PIT = fraction of posterior draws ≤ closing fair prob (per row). Under a posterior that is
calibrated to the closing line, PIT ~ Uniform(0,1). Returns the PIT vector, a KS test vs
U(0,1), and a central-credible-interval coverage table (empirical vs nominal).
"""
function pit_calibration(
    ppd, ds;
    close_horizon::Float64 = CLOSE_HORIZON,
    close_window::Float64 = CLOSE_WINDOW,
    targets::Vector{Symbol} = TARGET_SELECTIONS,
    levels::Vector{Float64} = [0.5, 0.8, 0.95],
)
    snap = snapshot_at(ds.betfair_odds, close_horizon; window_width = close_window)
    j = innerjoin(snap,
        select(ppd.df, :match_id, :market_name, :market_line, :selection, :distribution),
        on = [:match_id, :market_name, :market_line, :selection])
    filter!(r -> r.selection in targets, j)

    j.pit = [_miq(Float64.(d), p) for (d, p) in zip(j.distribution, j.prob_fair)]

    ks_d, ks_p = NaN, NaN
    try
        ks = ApproximateOneSampleKSTest(j.pit, Uniform(0, 1))
        ks_d, ks_p = ks.δ, pvalue(ks)
    catch err
        @warn "KS test failed" err
    end

    cov = DataFrame()
    for lvl in levels
        a = (1 - lvl) / 2
        c = mean([quantile(Float64.(d), a) <= p <= quantile(Float64.(d), 1 - a)
                  for (d, p) in zip(j.distribution, j.prob_fair)])
        push!(cov, (level = lvl, empirical_coverage = c, n = nrow(j)); promote = true)
    end

    return (pit = j.pit, ks_d = ks_d, ks_p = ks_p, coverage = cov, n = nrow(j))
end

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

function plot_edge_decay(edge_df::DataFrame; save_path = nothing)
    d = sort(edge_df, :horizon)
    p = plot(d.horizon, d.diff_ll;
        ribbon = (d.diff_ll .- d.diff_ll_lo, d.diff_ll_hi .- d.diff_ll),
        marker = :circle, linewidth = 2, label = "log-loss diff (model − market)",
        title = "Edge decay: model advantage vs horizon",
        xlabel = "Minutes to kickoff", ylabel = "Δ log-loss (negative = model better)",
        legend = :topright, titlefontsize = 10, size = (900, 450))
    hline!(p, [0.0]; color = :grey, linestyle = :dash, label = "")
    !isnothing(save_path) && savefig(p, save_path)
    return p
end

function plot_clv(alpha_df::DataFrame; save_path = nothing)
    d = sort(alpha_df, :horizon)
    p1 = plot(d.horizon, d.beta; marker = :circle, linewidth = 2, label = "β (move ~ signal)",
        ylabel = "β: line move per unit signal", title = "Does the model predict the line move?",
        titlefontsize = 10, legend = :topright)
    hline!(p1, [0.0]; color = :grey, linestyle = :dash, label = "")
    p2 = plot(d.horizon, d.mean_clv; marker = :diamond, linewidth = 2, color = :seagreen,
        label = "mean log-CLV vs close", xlabel = "Minutes to kickoff",
        ylabel = "mean log(odds_τ / odds_close)", legend = :topright)
    hline!(p2, [0.0]; color = :grey, linestyle = :dash, label = "")
    p = plot(p1, p2; layout = (2, 1), size = (900, 650))
    !isnothing(save_path) && savefig(p, save_path)
    return p
end

function plot_pnl(pnl_df::DataFrame; save_path = nothing)
    d = sort(pnl_df, :horizon)
    p = plot(d.horizon, d.roi; marker = :circle, linewidth = 2, label = "ROI",
        title = "Entry-timing P&L by horizon", xlabel = "Minutes to kickoff",
        ylabel = "Flat-stake ROI", legend = :topright, titlefontsize = 10, size = (900, 450))
    hline!(p, [0.0]; color = :grey, linestyle = :dash, label = "")
    !isnothing(save_path) && savefig(p, save_path)
    return p
end

function plot_pit(pit::AbstractVector; save_path = nothing)
    p = histogram(pit; bins = 20, normalize = true, label = "PIT", color = :steelblue,
        title = "PIT of posterior vs closing line (flat = calibrated)",
        xlabel = "PIT = P(posterior ≤ market)", ylabel = "density",
        titlefontsize = 10, size = (900, 400))
    hline!(p, [1.0]; color = :red, linestyle = :dash, label = "Uniform(0,1)")
    !isnothing(save_path) && savefig(p, save_path)
    return p
end
