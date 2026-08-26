# ==============================================================================
# Model 01 — GATE 6 : EVALUATION
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 5 established that the prices are the prices the model implies. Gate 6 asks
# whether those prices are any good, against outcomes that actually happened.
#
# THE PASS CRITERION IS NOT "BEATS THE MARKET".
#
# That has to be said first, because it is the easiest way to build a gate that
# throws away a working model. Prior work on this book had the model losing
# narrowly to market on 1X2 in proper-scoring terms while still producing positive
# CLV and growth. A gate set at "beats market log loss" would have binned it.
#
# Gate 6 passes when the model is not BROKEN: calibrated within reason, no line
# catastrophically worse than market, probabilities well-formed, fixture sets
# identical. Beating the market on a line is recorded as a FINDING, not a pass.
#
# Four metrics, deliberately answering different questions:
#
#   log_loss   how good is the probability of what happened?          needs market
#              Proper, but only comparable against a baseline.
#
#   lpd        how good is the density of the actual SCORELINE?       market-free
#              log (1/S) Σ_s P(y_h, y_a | θ_s) — averages the likelihood over the
#              posterior rather than plugging in its mean, so it is the only
#              metric here that rewards having the right uncertainty. This is the
#              one that ranks model VARIANTS against each other.
#
#   rqr        is the predictive distribution the right SHAPE?        market-free
#              Randomised quantile residuals (Dunn & Smyth 1996). Under a correctly
#              specified model these are exactly N(0,1). Mean ≠ 0 is bias; sd ≠ 1
#              is mis-dispersion — sd < 1 means the model is too confident. Prior
#              work found this model family running at roughly half the market's
#              dispersion, so this is the direct test of that.
#
#   glm_edge   does the model know anything the MARKET doesn't?       needs market
#              Encompassing regression: y ~ logit(p_market) + logit(p_model).
#              A model can be worse than the market in absolute log loss and still
#              carry incremental information — that combination is exactly what a
#              profitable contrarian model looks like, and log loss alone hides it.
#              This is the metric closest to the question gate 7 will answer.
#
# ==============================================================================

using BayesianFootball
using DataFrames
using Distributions
using GLM
using Random
using Statistics
using Printf

const Eval_Data = BayesianFootball.Data


# ==============================================================================
# 1. Books
# ==============================================================================

"""
    tp_book_markets(contract) -> Vector{AbstractMarket}

The contract's book as market objects. Nothing outside this list is ever scored —
adding a market here is the only way to widen the ranking, which keeps the book a
declared choice rather than whatever happened to be in the odds table.
"""
function tp_book_markets(contract::SLContract)
    ms = Any[Markets.Market1X2(), Markets.MarketBTTS()]
    append!(ms, [Markets.MarketOverUnder(l) for l in contract.totals_lines])
    return ms
end

"""
    tp_market_book(odds_df, contract) -> DataFrame

De-vigged closing probabilities from a table in `ds.odds` schema, restricted to the
contract book.

De-vigging is PROPORTIONAL (`prob_implied_close / overround_close`), which the
package computes upstream. On this league the 1X2 overround is ~10.65%, where
proportional de-vigging is known to bias against favourites; Shin or power
de-vigging would give a different baseline. That shifts the BASELINE, not the
model, so it moves every Δ reported. Recorded here so the number is read with it.
"""
function tp_market_book(odds_df::AbstractDataFrame, contract::SLContract)
    wanted = Set{Tuple{String,Float64}}()
    for m in tp_book_markets(contract)
        push!(wanted, (Eval_Data.market_group(m), Float64(Eval_Data.market_line(m))))
    end

    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, odds_df)
    isempty(df) && return DataFrame()

    out = DataFrame(
        match_id  = Int.(df.match_id),
        market    = String.(df.market_name),
        line      = Float64.(df.market_line),
        selection = Symbol.(df.selection),
        p_market  = Float64.(df.prob_implied_close) ./ Float64.(df.overround_close),
        is_winner = Bool.(coalesce.(df.is_winner, false)),
    )
    return out
end

"""
    tp_model_book(model, latents, ds, contract; seed) -> (book, fixtures)

Model probabilities for the contract book, plus the per-fixture quantities the
market-free metrics need.

Streams one fixture at a time. The full posterior grid for 360 fixtures is
360 x 12 x 12 x 2000 Float64 ≈ 414 MB, so it is computed, harvested and discarded
rather than held.

`book`     one row per (fixture, selection): p_model
`fixtures` one row per fixture: lpd, and the randomised quantile residuals

Model probabilities are NORMALISED per fixture by the grid's total mass. Gate 5
measured that mass at 1 - 4.4e-5, so this is a tiny correction — but market
probabilities de-vig to exactly 1, and scoring a sub-stochastic vector against a
stochastic one would charge the model for truncation. The normalisation magnitude
is returned so it cannot hide.

RQR uses the ANALYTIC NegBin CDF, not the grid marginal: randomised quantile
residuals are precisely a statement about tail behaviour, and the grid's tail is
truncated at 11 goals. Using the grid would make the model look better in the tail
than it is.
"""
function tp_model_book(model, latents, ds, contract::SLContract; seed::Int = 20260826)
    rng     = Random.MersenneTwister(seed)
    mg      = contract.max_goals
    markets = tp_book_markets(contract)

    scores = select(ds.matches, :match_id, :home_score, :away_score)
    ldf    = innerjoin(latents.df, scores, on = :match_id)

    rows  = NamedTuple[]
    fixes = NamedTuple[]

    for row in eachrow(ldf)
        (ismissing(row.home_score) || ismissing(row.away_score)) && continue
        yh, ya = Int(row.home_score), Int(row.away_score)

        params = Predictions.extract_params(model, row)
        S      = Predictions.compute_score_matrix(model, params; max_goals = mg)
        n_s    = size(S.data, 3)
        mass   = [sum(S.data[:, :, k]) for k in 1:n_s]

        for m in markets
            probs = Predictions.compute_market_probs(S, m)
            for (_, sel) in pairs(Eval_Data.outcomes(m))
                push!(rows, (
                    match_id  = Int(row.match_id),
                    market    = Eval_Data.market_group(m),
                    line      = Float64(Eval_Data.market_line(m)),
                    selection = sel,
                    p_model   = mean(probs[sel] ./ mass),
                ))
            end
        end

        # ---- LPD: the density of the SCORELINE that actually happened, averaged
        # over the posterior. Averaging the density (not plugging in the posterior
        # mean) is what makes this reward correct uncertainty rather than a good
        # point estimate.
        lpd = if yh < mg && ya < mg
            log(mean(S.data[yh + 1, ya + 1, :] ./ mass))
        else
            NaN     # scoreline off the grid; counted separately, never silently dropped
        end

        # ---- RQR, analytic marginals.
        rqr_h = _tp_rqr(rng, row.r_h, row.λ_h, yh)
        rqr_a = _tp_rqr(rng, row.r_a, row.λ_a, ya)

        push!(fixes, (
            match_id = Int(row.match_id),
            home_score = yh, away_score = ya,
            lpd = lpd, rqr_h = rqr_h, rqr_a = rqr_a,
            mass = mean(mass),
            is_draw = yh == ya,
            p_draw = 0.0,      # filled below from the book
        ))
    end

    book = DataFrame(rows)
    fx   = DataFrame(fixes)

    draws = filter(r -> r.market == "1X2" && r.selection == :draw, book)
    dmap  = Dict(r.match_id => r.p_model for r in eachrow(draws))
    fx.p_draw = [get(dmap, id, NaN) for id in fx.match_id]

    return (book, fx)
end

"""
    _tp_rqr(rng, r_draws, λ_draws, y) -> Float64

One randomised quantile residual for an observed count.

For a DISCRETE distribution the probability integral transform does not give a
uniform, because the CDF jumps. Dunn & Smyth's fix is to draw uniformly within the
jump: u ~ U(F(y-1), F(y)), then Φ⁻¹(u). Under correct specification the result is
exactly N(0,1) — not approximately.

The predictive CDF is averaged over posterior draws, so this is the residual of the
POSTERIOR PREDICTIVE, not of a plug-in fit.
"""
function _tp_rqr(rng, r_draws, λ_draws, y::Int)
    nb(r, λ) = NegativeBinomial(r, r / (r + λ))
    f_lo = y == 0 ? 0.0 : mean(cdf(nb(r_draws[k], λ_draws[k]), y - 1) for k in eachindex(λ_draws))
    f_hi = mean(cdf(nb(r_draws[k], λ_draws[k]), y) for k in eachindex(λ_draws))
    u    = f_lo + rand(rng) * (f_hi - f_lo)
    return quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
end

"""
    tp_join_books(model_book, market_books) -> DataFrame

Inner join on (match_id, market, line, selection).

INNER deliberately: a selection priced by one side and not the other is not
comparable, and silently keeping it would score the model and the market on
different fixture sets. Gate 6b reports exactly what the join dropped.
"""
function tp_join_books(model_book::AbstractDataFrame, market_books::Dict{String,DataFrame})
    out = copy(model_book)
    for (name, mb) in market_books
        cols = select(mb, [:match_id, :market, :line, :selection, :p_market, :is_winner])
        rename!(cols, :p_market => Symbol("p_", name), :is_winner => Symbol("win_", name))
        out = innerjoin(out, cols, on = [:match_id, :market, :line, :selection])
    end
    return out
end


# ==============================================================================
# 2. Metrics
# ==============================================================================

_tp_clampp(p) = clamp(p, 1e-9, 1 - 1e-9)
_tp_logit(p)  = log(_tp_clampp(p) / (1 - _tp_clampp(p)))

"Log loss of `p` against binary `y`. Mean, so it is comparable across lines of different size."
tp_log_loss(p::AbstractVector, y::AbstractVector) =
    -mean(yi ? log(_tp_clampp(pi)) : log(1 - _tp_clampp(pi)) for (pi, yi) in zip(p, y))

"""
    tp_paired_delta(p_a, p_b, y) -> (; Δ, se, t)

Difference in log loss between two forecasters on the SAME outcomes.

Paired, not two independent means. The two score identical fixtures, so the
statistic is the per-fixture difference and its standard error; treating them as
independent samples inflates the interval until it says nothing. This is the
difference between "the model is worse, t = -4.5" and "the model is worse,
CI [-0.04, +0.03]".
"""
function tp_paired_delta(p_a::AbstractVector, p_b::AbstractVector, y::AbstractVector)
    la = [yi ? -log(_tp_clampp(pi)) : -log(1 - _tp_clampp(pi)) for (pi, yi) in zip(p_a, y)]
    lb = [yi ? -log(_tp_clampp(pi)) : -log(1 - _tp_clampp(pi)) for (pi, yi) in zip(p_b, y)]
    d  = la .- lb
    se = std(d) / sqrt(length(d))
    return (Δ = mean(d), se = se, t = se > 0 ? mean(d) / se : 0.0)
end

"Brier score (mean squared error of the probability)."
tp_brier(p::AbstractVector, y::AbstractVector) = mean((p .- Float64.(y)).^2)

"""
    tp_glm_edge(df, p_model_col, p_market_col, y_col) -> NamedTuple

Two logistic regressions per line.

CALIBRATION   y ~ logit(p_model)
    slope 1 and intercept 0 is perfect. Slope < 1 means over-confident: the model's
    probabilities are too far from the base rate and should be shrunk toward it.

ENCOMPASSING  y ~ logit(p_market) + logit(p_model)
    Does the model carry information the market does not? A positive, significant
    coefficient on the model term says yes EVEN IF the model's absolute log loss is
    worse than the market's — and that combination is precisely what a profitable
    contrarian model looks like. Log loss alone cannot see it.

    β_model ≈ 0  the model adds nothing; abstain.
    β_market ≈ 0 the market adds nothing given the model (implausible; check for a bug).
"""
function tp_glm_edge(df::AbstractDataFrame; p_model = :p_model, p_market = :p_bet365, y = :is_winner)
    d = DataFrame(
        y  = Float64.(df[!, y]),
        lm = _tp_logit.(df[!, p_model]),
        lo = _tp_logit.(df[!, p_market]),
    )
    n = nrow(d)
    (n < 30 || length(unique(d.y)) < 2) &&
        return (n = n, slope = NaN, intercept = NaN, β_model = NaN, se_model = NaN,
                z_model = NaN, β_market = NaN)

    cal = glm(@formula(y ~ lm),      d, Binomial(), LogitLink())
    enc = glm(@formula(y ~ lo + lm), d, Binomial(), LogitLink())

    ccal = coef(cal); cenc = coef(enc); senc = stderror(enc)
    return (n = n,
            intercept = ccal[1], slope = ccal[2],
            β_market  = cenc[2],
            β_model   = cenc[3], se_model = senc[3],
            z_model   = senc[3] > 0 ? cenc[3] / senc[3] : NaN)
end


# ==============================================================================
# 3. GATE 6a — Book integrity
# ==============================================================================

"""
    tp_gate_book_integrity(market_book, contract; label) -> Vector

Before any score is computed: is the market book itself well-formed?

Scoring against a corrupt outcome column produces confident nonsense, and the
failure is invisible in the output. This repository has one documented instance —
Double Chance marks 1 of 2 selections as a winner and halves its fair
probabilities, which reversed a headline result once. DC is deliberately not in the
contract book; this gate is what would catch it if someone added it.
"""
function tp_gate_book_integrity(mb::AbstractDataFrame, contract::SLContract; label::String = "")
    out = Any[]
    g = groupby(mb, [:match_id, :market, :line])

    winners = combine(g, :is_winner => sum => :n_win)
    bad_w   = filter(r -> r.n_win != 1, winners)
    push!(out, (
        name   = "exactly one winner per market",
        pass   = isempty(bad_w),
        detail = isempty(bad_w) ? "$(nrow(winners)) fixture-markets, all with 1 winner" :
                 "$(nrow(bad_w)) fixture-markets with $(unique(bad_w.n_win)) winners",
    ))

    sums   = combine(g, :p_market => sum => :s)
    worst  = isempty(sums) ? 0.0 : maximum(abs.(sums.s .- 1.0))
    push!(out, (
        name   = "de-vigged probabilities sum to 1",
        pass   = worst <= 1e-9,
        detail = @sprintf("max |Σp - 1| = %.3e over %d fixture-markets", worst, nrow(sums)),
    ))

    push!(out, (
        name   = "probabilities in (0, 1)",
        pass   = all(0 .< mb.p_market .< 1),
        detail = @sprintf("range [%.4f, %.4f]", minimum(mb.p_market), maximum(mb.p_market)),
    ))

    want = Set((Eval_Data.market_group(m), Float64(Eval_Data.market_line(m))) for m in tp_book_markets(contract))
    got  = Set((r.market, r.line) for r in eachrow(unique(select(mb, [:market, :line]))))
    push!(out, (
        name   = "book is the contract book",
        pass   = got == want,
        detail = "$(length(got)) of $(length(want)) markets present" *
                 (isempty(setdiff(want, got)) ? "" : "; missing $(collect(setdiff(want, got)))"),
    ))

    return out
end


# ==============================================================================
# 4. GATE 6b — Alignment
# ==============================================================================

"""
    tp_gate_alignment(joined, model_book, market_books) -> Vector

Model and market must be scored on identical fixture x selection sets, asserted
BEFORE any ranking is printed.

Two forecasters scored on different sets are not comparable, and nothing in the
output reveals it: both columns are full, both numbers are plausible, and the
comparison is meaningless. The join is inner, so this reports what it dropped.
"""
function tp_gate_alignment(joined::AbstractDataFrame, model_book::AbstractDataFrame,
                           market_books::Dict{String,DataFrame})
    out = Any[]
    key(df) = Set((r.match_id, r.market, r.line, r.selection) for r in eachrow(df))
    k_model = key(model_book)

    push!(out, (
        name   = "joined set is non-empty",
        pass   = nrow(joined) > 0,
        detail = "$(nrow(joined)) rows, $(length(unique(joined.match_id))) fixtures",
    ))

    for (name, mb) in market_books
        k_mkt  = key(mb)
        common = intersect(k_model, k_mkt)
        push!(out, (
            name   = "coverage vs $name",
            pass   = length(common) > 0,
            detail = @sprintf("%d shared; model-only %d, %s-only %d",
                              length(common), length(setdiff(k_model, k_mkt)),
                              name, length(setdiff(k_mkt, k_model))),
        ))
    end

    # The outcome column must agree between baselines. If Bet365 and Betfair disagree
    # about who won, one of the graders is wrong and every score below is suspect.
    wincols = [c for c in names(joined) if startswith(String(c), "win_")]
    if length(wincols) > 1
        disagree = count(r -> length(unique([r[c] for c in wincols])) > 1, eachrow(joined))
        push!(out, (
            name   = "graders agree on outcomes",
            pass   = disagree == 0,
            detail = disagree == 0 ? "all $(nrow(joined)) rows agree" : "$disagree rows disagree",
        ))
    end

    push!(out, (
        name   = "model probabilities well-formed",
        pass   = all(isfinite, joined.p_model) && all(0 .< joined.p_model .< 1),
        detail = @sprintf("range [%.5f, %.5f]", minimum(joined.p_model), maximum(joined.p_model)),
    ))

    return out
end


# ==============================================================================
# 5. GATE 6c — Scores, per line
# ==============================================================================

"""
    tp_score_table(joined; baseline) -> DataFrame

Per-line log loss, Brier, and the paired difference against a baseline.

PER LINE, never aggregated across the selections of one market. Aggregating over
selections is what reversed the APM headline once: a market whose selections are
not independent events (Double Chance) or whose base rates differ wildly (1X2
draw versus home) produces a pooled number that means nothing.

FIXTURE-WEIGHTED, not fold-averaged. Pooling every fixture into one table does this
automatically; the trap is only in averaging per-fold scores, where a 2-fixture OOS
block would count as much as a 24-fixture one. `tp_fold_weighting_check` shows the
size of that difference rather than asserting it does not matter.
"""
function tp_score_table(joined::AbstractDataFrame; baseline::Symbol = :p_bet365)
    wincol = Symbol("win_", String(baseline)[3:end])
    rows = NamedTuple[]

    for g in groupby(joined, [:market, :line, :selection])
        y  = collect(g[!, wincol])
        pm = collect(g.p_model)
        pb = collect(g[!, baseline])
        d  = tp_paired_delta(pm, pb, y)

        push!(rows, (
            market    = g.market[1],
            line      = g.line[1],
            selection = g.selection[1],
            n         = nrow(g),
            base_rate = round(mean(y), digits = 3),
            ll_model  = round(tp_log_loss(pm, y), digits = 4),
            ll_market = round(tp_log_loss(pb, y), digits = 4),
            Δll       = round(d.Δ, digits = 4),
            t         = round(d.t, digits = 2),
            brier_model  = round(tp_brier(pm, y), digits = 4),
            brier_market = round(tp_brier(pb, y), digits = 4),
        ))
    end
    return sort!(DataFrame(rows), [:market, :line, :selection])
end

"""
    tp_fold_weighting_check(joined, folds; baseline) -> NamedTuple

Pooled versus fold-averaged log loss, side by side.

Not a gate — a demonstration. OOS blocks here range from 2 to 24 fixtures, and the
point is to show how far a fold average drifts from the fixture-weighted number,
rather than to assert in a comment that it would.
"""
function tp_fold_weighting_check(joined::AbstractDataFrame, folds; baseline::Symbol = :p_bet365)
    wincol = Symbol("win_", String(baseline)[3:end])
    fmap = Dict{Int,Int}()
    for f in folds, id in f.oos_df.match_id
        fmap[Int(id)] = f.idx
    end
    j = copy(joined)
    j.fold = [get(fmap, id, 0) for id in j.match_id]

    onex = filter(r -> r.market == "1X2", j)
    pooled = tp_log_loss(onex.p_model, onex[!, wincol])

    per_fold = combine(groupby(onex, :fold),
                       [:p_model, wincol] => ((p, y) -> tp_log_loss(p, y)) => :ll,
                       nrow => :n)
    per_fold = filter(r -> r.fold > 0, per_fold)

    return (pooled_1x2 = round(pooled, digits = 4),
            fold_averaged_1x2 = round(mean(per_fold.ll), digits = 4),
            difference = round(mean(per_fold.ll) - pooled, digits = 4),
            fold_sizes = extrema(per_fold.n),
            n_folds = nrow(per_fold))
end


# ==============================================================================
# 6. GATE 6d — Shape: RQR and LPD
# ==============================================================================

"""
    tp_gate_shape(fixtures; mean_tol, sd_lo, sd_hi) -> Vector

Is the predictive distribution the right shape, independently of any market?

Randomised quantile residuals are EXACTLY N(0,1) under correct specification, not
approximately, so both moments are meaningful:

    mean ≠ 0   bias — the model systematically over- or under-predicts goals
    sd   < 1   OVER-CONFIDENT — predictive spread too narrow
    sd   > 1   under-confident

The sd band is deliberately wide (0.85-1.15). A structural model of football is not
going to be perfectly dispersed, and prior work on this book found dispersion at
roughly half the market's; the gate is there to catch a model that is badly wrong,
not to demand one that is perfect.

LPD is reported, never gated. In isolation a log density has no scale — it only
means something compared against another model on the same fixtures, which is what
it exists for.
"""
function tp_gate_shape(fx::AbstractDataFrame; mean_tol = 0.15, sd_lo = 0.85, sd_hi = 1.15)
    out = Any[]
    r = vcat(fx.rqr_h, fx.rqr_a)
    r = filter(isfinite, r)

    m, s = mean(r), std(r)
    push!(out, (
        name   = "RQR mean (bias)",
        pass   = abs(m) <= mean_tol,
        detail = @sprintf("%.4f over %d residuals (target 0, tol ±%.2f)", m, length(r), mean_tol),
    ))
    push!(out, (
        name   = "RQR sd (dispersion)",
        pass   = sd_lo <= s <= sd_hi,
        detail = @sprintf("%.4f (target 1, band [%.2f, %.2f]) — %s", s, sd_lo, sd_hi,
                          s < sd_lo ? "OVER-confident" : s > sd_hi ? "under-confident" : "well dispersed"),
    ))

    mh, ma = mean(filter(isfinite, fx.rqr_h)), mean(filter(isfinite, fx.rqr_a))
    push!(out, (
        name   = "RQR home/away symmetry",
        pass   = abs(mh - ma) <= 2 * mean_tol,
        detail = @sprintf("home %.4f, away %.4f, gap %.4f", mh, ma, abs(mh - ma)),
    ))

    off = count(!isfinite, fx.lpd)
    push!(out, (
        name   = "scorelines on the grid",
        pass   = off == 0,
        detail = off == 0 ? "all $(nrow(fx)) scorelines within the grid" :
                            "$off scoreline(s) off the grid — LPD undefined there",
    ))

    lp = filter(isfinite, fx.lpd)
    push!(out, (
        name   = "LPD (reported, not gated)",
        pass   = true,
        detail = @sprintf("mean %.4f, total %.1f over %d fixtures — compare across variants only",
                          mean(lp), sum(lp), length(lp)),
    ))

    push!(out, (
        name   = "grid mass (truncation correction)",
        pass   = abs(1 - mean(fx.mass)) <= 1e-3,
        detail = @sprintf("mean grid mass %.8f — model probs divided by this before scoring",
                          mean(fx.mass)),
    ))

    return out
end

"""
    tp_gate_draw_deficit(fx) -> Vector

Gate 5 observed mean predicted draw ≈ 0.243 against an empirical Scottish L1/L2
rate of roughly 0.25-0.27. Conditionally independent goals with no Dixon-Coles term
under-predict draws, and this is what that looks like.

Tested rather than assumed, because if the deficit is real it is the strongest
available argument for a DC or copula variant — and if it is not, that argument
should not be made.
"""
function tp_gate_draw_deficit(fx::AbstractDataFrame)
    n         = nrow(fx)
    observed  = mean(fx.is_draw)
    predicted = mean(filter(isfinite, fx.p_draw))
    se        = sqrt(observed * (1 - observed) / n)
    z         = se > 0 ? (observed - predicted) / se : 0.0

    return Any[(
        name   = "draw rate matches prediction",
        pass   = abs(z) <= 2.0,
        detail = @sprintf("observed %.4f vs predicted %.4f over %d fixtures, z = %+.2f — %s",
                          observed, predicted, n, z,
                          abs(z) <= 2.0 ? "no significant deficit" :
                                          "SIGNIFICANT: evidence for a Dixon-Coles / copula term"),
    )]
end

"""
    tp_edge_table(joined; baseline) -> DataFrame

Calibration and encompassing regressions per line — the metric closest to the
question gate 7 answers.

`slope` is the calibration slope: below 1 means over-confident and argues for
shrinking probabilities toward the base rate.

`β_model` is the coefficient on the model's logit GIVEN the market's. Positive and
significant means the model carries information the market does not — which can be
true even where `Δll` is positive (the model losing on absolute log loss). That
combination is what a profitable contrarian model looks like, and it is invisible
to proper scoring alone.
"""
function tp_edge_table(joined::AbstractDataFrame; baseline::Symbol = :p_bet365)
    wincol = Symbol("win_", String(baseline)[3:end])
    rows = NamedTuple[]
    for g in groupby(joined, [:market, :line, :selection])
        e = tp_glm_edge(g; p_model = :p_model, p_market = baseline, y = wincol)
        push!(rows, (
            market    = g.market[1],
            line      = g.line[1],
            selection = g.selection[1],
            n         = e.n,
            slope     = round(e.slope, digits = 3),
            intercept = round(e.intercept, digits = 3),
            β_market  = round(e.β_market, digits = 3),
            β_model   = round(e.β_model, digits = 3),
            z_model   = round(e.z_model, digits = 2),
        ))
    end
    return sort!(DataFrame(rows), [:market, :line, :selection])
end

"""
    tp_gate_not_broken(scores, edges; max_delta, slope_lo, slope_hi) -> Vector

The pass condition for gate 6 as a whole.

NOT "beats the market". Prior work on this book had the model losing narrowly on
1X2 log loss while producing positive CLV and growth; a gate demanding it win would
have discarded a working model. What is required is that it is not BROKEN:
calibrated within reason, and nowhere catastrophically worse than the market.

Beating the market anywhere is recorded as a finding, not a pass condition.
"""
function tp_gate_not_broken(scores::AbstractDataFrame, edges::AbstractDataFrame;
                            max_delta = 0.02, slope_lo = 0.7, slope_hi = 1.3)
    out = Any[]

    worst_i = argmax(scores.Δll)
    push!(out, (
        name   = "no line catastrophically worse than market",
        pass   = maximum(scores.Δll) <= max_delta,
        detail = @sprintf("worst Δll %+.4f on %s %s (t %+.2f), threshold +%.2f",
                          scores.Δll[worst_i], scores.market[worst_i],
                          string(scores.selection[worst_i]), scores.t[worst_i], max_delta),
    ))

    sl = filter(isfinite, edges.slope)
    push!(out, (
        name   = "calibration slopes sane",
        pass   = all(slope_lo .<= sl .<= slope_hi),
        detail = @sprintf("range [%.3f, %.3f], band [%.2f, %.2f]; %d/%d lines inside",
                          minimum(sl), maximum(sl), slope_lo, slope_hi,
                          count(s -> slope_lo <= s <= slope_hi, sl), length(sl)),
    ))

    beat = filter(r -> r.Δll < 0, scores)
    push!(out, (
        name   = "lines beating market (finding, not a criterion)",
        pass   = true,
        detail = isempty(beat) ? "none" :
                 join(["$(r.market) $(r.selection) $(r.Δll)" for r in eachrow(beat)], ", "),
    ))

    inf = filter(r -> isfinite(r.z_model) && r.z_model > 2, edges)
    push!(out, (
        name   = "lines with incremental information (finding)",
        pass   = true,
        detail = isempty(inf) ? "none at z > 2" :
                 join(["$(r.market) $(r.selection) z=$(r.z_model)" for r in eachrow(inf)], ", "),
    ))

    return out
end
