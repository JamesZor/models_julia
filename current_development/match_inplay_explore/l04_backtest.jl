#=
l04_backtest.jl  —  In-play betting backtest for the intensity model.

Turns the fitted intensity model into per-bin market probabilities (Over/Under + 1X2),
compares to Betfair in-play prices, places value bets (fractional Kelly, min-edge), and
settles on real outcomes net of commission.

CRITICAL EXECUTION REALISM (see in_play_backtest_report.md):
  - `run_backtest` fills at the price AS-OF the bin (latest trade <= t_w). This is
    OPTIMISTIC: a stale last-traded price can pre-date a goal the model already conditions
    on → fake "edge" (lookahead). ROI scales with staleness.
  - `run_backtest_fwd` fills at the NEXT available price after the signal (forward
    execution). This is the realistic measure; the apparent edge collapses to ~noise.

Inputs come from the l01 panel (per-bin score/time/red cards + pregame λ join) and the
fitted-model posterior-mean coefficients ᾱ, β̄ with the l03 standardisation (x_center/x_scale).
Needs in scope: `panel`, `bf`, `te_ids`, `finmap`, `inp` (x_center/x_scale), `ᾱ`, `β̄`,
and l01's `latest_prices`.
=#

using DataFrames
using Distributions
using LinearAlgebra
using Statistics

# Final scores lookup: match_id => (home, away)
make_finmap(ds) = Dict(Int(r.match_id) => (Int(r.home_score), Int(r.away_score))
                       for r in eachrow(ds.matches) if !ismissing(r.home_score))

"Posterior-mean predicted remaining-goal MEAN for one side (uses l03 standardisation)."
function predict_mu_side(ᾱ, β̄, x_center, x_scale, t_m, is_home, trailing, leading, man_adv, logpg)
    rf = max((90.0 - t_m) / 90.0, 0.05)
    x  = [t_m, t_m^2, Float64(is_home), Float64(trailing), Float64(leading), Float64(man_adv), logpg]
    xs = (x .- x_center) ./ x_scale
    return exp(clamp(ᾱ + dot(β̄, xs) + log(rf), -20.0, 20.0))
end

"Model probabilities for OU(0.5..5.5) + 1X2 at a bin, given current score & red cards."
function model_probs(ᾱ, β̄, xc, xs_, t_m, gh, ga, hr, ar, pg_h, pg_a; G = 12)
    μ_h = predict_mu_side(ᾱ, β̄, xc, xs_, t_m, 1, gh < ga, gh > ga, ar - hr, log(pg_h))
    μ_a = predict_mu_side(ᾱ, β̄, xc, xs_, t_m, 0, ga < gh, ga > gh, hr - ar, log(pg_a))
    μ_tot = μ_h + μ_a; T = gh + ga
    probs = Dict{Symbol,Float64}()
    for k in 0:5
        need = (k + 1) - T
        p_over = need <= 0 ? 1.0 : ccdf(Poisson(μ_tot), need - 1)
        probs[Symbol("over_$(k)5")]  = p_over
        probs[Symbol("under_$(k)5")] = 1.0 - p_over
    end
    ph = pdf.(Poisson(μ_h), 0:G); pa = pdf.(Poisson(μ_a), 0:G)
    p_home = p_draw = p_away = 0.0
    for i in 0:G, j in 0:G
        w = ph[i+1] * pa[j+1]; fh = gh + i; fa = ga + j
        fh > fa ? (p_home += w) : fh == fa ? (p_draw += w) : (p_away += w)
    end
    probs[:home] = p_home; probs[:draw] = p_draw; probs[:away] = p_away
    return probs
end

"Settle a selection given the final score."
function settle(sel::Symbol, fh, fa)
    s = String(sel); T = fh + fa
    startswith(s, "over_")  && return T > parse(Int, s[6:6]) + 0.5
    startswith(s, "under_") && return T < parse(Int, s[7:7]) + 0.5
    sel === :home && return fh > fa
    sel === :draw && return fh == fa
    sel === :away && return fh < fa
    return false
end

is_ou(sel) = (s = String(sel); startswith(s, "over_") || startswith(s, "under_"))

"Forward fill: FIRST traded price per selection in [t_w+lag, t_w+lag+window] (realistic execution)."
function forward_prices(bf_match, t_w; lag = 0.5, window = 6.0)
    out = Dict{Symbol,Float64}()
    sub = filter(r -> (t_w + lag) <= r.minutes_to_kickoff <= (t_w + lag + window), bf_match)
    isempty(sub) && return out
    for g in groupby(sub, :selection)
        r = first(sort(g, :minutes_to_kickoff)); out[r.selection] = r.traded_price
    end
    return out
end

"""
    run_backtest(panel, bf, te_ids, finmap, ᾱ, β̄, inp; kwargs...) -> DataFrame of bets

One value bet per (match, selection) at the first qualifying bin. `mode = :asof` fills at
the latest price ≤ t_w (optimistic/stale); `mode = :forward` fills at the next price after
the signal (realistic). Back-bet EV with commission on net winnings; fractional-Kelly stake.
"""
function run_backtest(panel, bf, te_ids, finmap, ᾱ, β̄, inp;
                      mode = :forward, comm = 0.05, min_edge = 0.05, kelly_frac = 0.25,
                      staleness = 2.0, lag = 0.5, window = 6.0, haircut = 0.0,
                      sels = nothing, dmin = 1.1, dmax = 21.0)
    xc, xs_ = inp.x_center, inp.x_scale
    bets = NamedTuple[]
    for mid in te_ids
        haskey(finmap, mid) || continue
        fh, fa = finmap[mid]
        bf_match = subset(bf, :match_id => ByRow(==(mid)),
                              :minutes_to_kickoff => ByRow(x -> 0.0 < x <= 130))
        bins = sort(subset(panel, :match_id => ByRow(==(mid))), :t_w)
        placed = Set{Symbol}()
        for r in eachrow(bins)
            (ismissing(r.pg_λ_h) || r.t_m > 80) && continue
            prices = mode === :forward ? forward_prices(bf_match, r.t_w; lag = lag, window = window) :
                                         latest_prices(bf_match, r.t_w; staleness = staleness)
            isempty(prices) && continue
            mp = model_probs(ᾱ, β̄, xc, xs_, r.t_m, r.gh, r.ga, r.home_reds, r.away_reds, r.pg_λ_h, r.pg_λ_a)
            for (sel, d0) in prices
                (sel in placed) && continue
                (sels !== nothing && !(sel in sels)) && continue
                haskey(mp, sel) || continue
                d = 1 + (d0 - 1) * (1 - haircut)
                (d < dmin || d > dmax) && continue
                q  = mp[sel]
                ev = q * (d - 1) * (1 - comm) - (1 - q)
                ev > min_edge || continue
                f = (q * d - 1) / (d - 1)
                stake = clamp(kelly_frac * f, 0.0, 1.0)
                stake <= 0 && continue
                won = settle(sel, fh, fa)
                pnl = won ? (d - 1) * (1 - comm) * stake : -stake
                push!(bets, (match_id = mid, t_m = r.t_m, market = is_ou(sel) ? :OU : :X12,
                             selection = sel, price = d, model_p = q, edge = ev,
                             stake = stake, won = won, pnl = pnl))
                push!(placed, sel)
            end
        end
    end
    return DataFrame(bets)
end

roi(b) = isempty(b) ? NaN : sum(b.pnl) / sum(b.stake)
