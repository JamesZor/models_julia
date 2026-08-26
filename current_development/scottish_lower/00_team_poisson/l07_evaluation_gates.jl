# ==============================================================================
# Model 00 — GATE 6 : EVALUATION (Pure Poisson)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Covers Gate 6 for Model 00 (Pure Poisson):
#   6a: Book integrity (Bet365 de-vigged close and Betfair close)
#   6b: Alignment between model prices and market prices
#   6c: Shape diagnostics (Randomised Quantile Residuals & LPD for pure Poisson)
#   6d: Draw rate verification
#   6e: Proper scoring (per-line paired log loss, Brier) & calibration
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

function tp00_book_markets(contract::SLContract)
    ms = Any[Markets.Market1X2(), Markets.MarketBTTS()]
    append!(ms, [Markets.MarketOverUnder(l) for l in contract.totals_lines])
    return ms
end

function tp00_market_book(odds_df::AbstractDataFrame, contract::SLContract;
                          ids::Union{Nothing,AbstractSet} = nothing)
    wanted = Set{Tuple{String,Float64}}()
    for m in tp00_book_markets(contract)
        push!(wanted, (Eval_Data.market_group(m), Float64(Eval_Data.market_line(m))))
    end

    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, odds_df)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))
    isempty(df) && return DataFrame()

    return DataFrame(
        match_id  = Int.(df.match_id),
        market    = String.(df.market_name),
        line      = Float64.(df.market_line),
        selection = Symbol.(df.selection),
        p_market  = Float64.(df.prob_implied_close) ./ Float64.(df.overround_close),
        is_winner = Bool.(coalesce.(df.is_winner, false)),
    )
end

function _tp00_rqr(rng, λ_draws, y::Int)
    f_lo = y == 0 ? 0.0 : mean(cdf(Poisson(λ_draws[k]), y - 1) for k in eachindex(λ_draws))
    f_hi = mean(cdf(Poisson(λ_draws[k]), y) for k in eachindex(λ_draws))
    u    = f_lo + rand(rng) * (f_hi - f_lo)
    return quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
end

function tp00_betfair_book(ds, contract::SLContract, grading::AbstractDataFrame;
                           ids::Union{Nothing,AbstractSet} = nothing,
                           window = (-20.0, 0.0))
    D   = BayesianFootball.Data
    raw = D.summarize_odds(ds.betfair_odds, D.TWAEstimator(); window = window)
    isempty(raw) && return DataFrame()

    wanted = Set((D.market_group(m), Float64(D.market_line(m))) for m in tp00_book_markets(contract))
    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, raw)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))
    "is_sane" in names(df) && (df = filter(r -> coalesce(r.is_sane, true), df))
    isempty(df) && return DataFrame()

    out = DataFrame(
        match_id   = Int.(df.match_id),
        market     = String.(df.market_name),
        line       = Float64.(df.market_line),
        selection  = Symbol.(df.selection),
        p_market   = (1 ./ Float64.(df.odds)) ./ Float64.(df.overround),
        odds_close = Float64.(df.odds),
    )
    return innerjoin(out, select(grading, [:match_id, :market, :line, :selection, :is_winner]),
                     on = [:match_id, :market, :line, :selection])
end

function tp00_drop_incomplete(bf_book::AbstractDataFrame)
    isempty(bf_book) && return (DataFrame(), 0)
    
    # Check completeness per market
    gdf = groupby(bf_book, [:match_id, :market, :line])
    complete_keys = Tuple{Int,String,Float64}[]
    n_partial = 0

    for ((mid, mkt, line), sub) in pairs(gdf)
        exp_legs = (mkt == "1X2") ? 3 : 2
        if nrow(sub) == exp_legs
            push!(complete_keys, (mid, mkt, line))
        else
            n_partial += 1
        end
    end

    cset = Set(complete_keys)
    filtered = filter(r -> (r.match_id, r.market, r.line) in cset, bf_book)
    return (DataFrame(filtered), n_partial)
end

function tp00_model_book(model::DynamicPoissonGoalsTimeDecayModel, latents, ds, contract::SLContract; seed::Int = 20260826)
    rng     = Random.MersenneTwister(seed)
    mg      = contract.max_goals
    markets = tp00_book_markets(contract)

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

        lpd = if yh < mg && ya < mg
            log(mean(S.data[yh + 1, ya + 1, :] ./ mass))
        else
            NaN
        end

        rqr_h = _tp00_rqr(rng, row.λ_h, yh)
        rqr_a = _tp00_rqr(rng, row.λ_a, ya)

        push!(fixes, (
            match_id = Int(row.match_id),
            home_score = yh, away_score = ya,
            lpd = lpd, rqr_h = rqr_h, rqr_a = rqr_a,
            mass = mean(mass),
            is_draw = yh == ya,
            p_draw = 0.0,
        ))
    end

    book = DataFrame(rows)
    fx   = DataFrame(fixes)

    draws = filter(r -> r.market == "1X2" && r.selection == :draw, book)
    dmap  = Dict(r.match_id => r.p_model for r in eachrow(draws))
    fx.p_draw = [get(dmap, id, NaN) for id in fx.match_id]

    return (book, fx)
end

function tp00_join_books(model_book::AbstractDataFrame, market_books::Dict{String,DataFrame})
    out = Dict{String,DataFrame}()
    for (name, mb) in market_books
        isempty(mb) && (out[name] = DataFrame(); continue)
        cols = select(mb, [:match_id, :market, :line, :selection, :p_market, :is_winner])
        out[name] = innerjoin(model_book, cols, on = [:match_id, :market, :line, :selection])
    end
    return out
end


# ==============================================================================
# 2. Gate Assertions & Metrics
# ==============================================================================

_tp00_clampp(p) = clamp(p, 1e-9, 1 - 1e-9)
_tp00_logit(p)  = log(_tp00_clampp(p) / (1 - _tp00_clampp(p)))

tp00_log_loss(p::AbstractVector, y::AbstractVector) =
    -mean(yi ? log(_tp00_clampp(pi)) : log(1 - _tp00_clampp(pi)) for (pi, yi) in zip(p, y))

function tp00_paired_delta(p_a::AbstractVector, p_b::AbstractVector, y::AbstractVector)
    la = [yi ? -log(_tp00_clampp(pi)) : -log(1 - _tp00_clampp(pi)) for (pi, yi) in zip(p_a, y)]
    lb = [yi ? -log(_tp00_clampp(pi)) : -log(1 - _tp00_clampp(pi)) for (pi, yi) in zip(p_b, y)]
    d  = la .- lb
    se = std(d) / sqrt(length(d))
    return (Δ = mean(d), se = se, t = se > 0 ? mean(d) / se : 0.0)
end

tp00_brier(p::AbstractVector, y::AbstractVector) = mean((p .- Float64.(y)).^2)

function tp00_gate_book_integrity(book::DataFrame, contract::SLContract)
    results = []
    push!(results, (
        name   = "book non-empty",
        pass   = nrow(book) > 0,
        detail = "$(nrow(book)) rows, $(length(unique(book.match_id))) fixtures",
    ))
    return results
end

function tp00_gate_alignment(joined::Dict{String,DataFrame}, model_book::DataFrame)
    results = []
    for (name, jdf) in joined
        push!(results, (
            name   = "aligned vs $name",
            pass   = nrow(jdf) > 0,
            detail = "$(nrow(jdf)) rows aligned across $(length(unique(jdf.match_id))) fixtures",
        ))
    end
    return results
end

function tp00_gate_shape(fx::DataFrame)
    rqr = vcat(fx.rqr_h, fx.rqr_a)
    rqr_clean = filter(isfinite, rqr)
    m = mean(rqr_clean)
    s = std(rqr_clean)

    results = []
    push!(results, (
        name   = "RQR mean ≈ 0 (bias)",
        pass   = abs(m) <= 0.15,
        detail = @sprintf("RQR mean = %.4f (target 0.0)", m),
    ))
    push!(results, (
        name   = "RQR sd ≈ 1 (dispersion)",
        pass   = 0.80 <= s <= 1.25,
        detail = @sprintf("RQR sd = %.4f (target 1.0)", s),
    ))
    return results
end

function tp00_gate_draw_deficit(fx::DataFrame)
    obs_draw = mean(fx.is_draw)
    pred_draw = mean(fx.p_draw)
    diff = pred_draw - obs_draw

    results = []
    push!(results, (
        name   = "draw calibration",
        pass   = abs(diff) <= 0.06,
        detail = @sprintf("observed %.4f vs predicted %.4f (diff %.4f)", obs_draw, pred_draw, diff),
    ))
    return results
end

function tp00_score_table(df::DataFrame)
    gdf = groupby(df, [:market, :line, :selection])
    out = NamedTuple[]
    for ((mkt, line, sel), sub) in pairs(gdf)
        y = sub.is_winner
        pm = sub.p_model
        pk = sub.p_market
        pd = tp00_paired_delta(pm, pk, y)
        push!(out, (
            market    = mkt,
            line      = line,
            selection = sel,
            n         = nrow(sub),
            base_rate = mean(y),
            ll_model  = tp00_log_loss(pm, y),
            ll_market = tp00_log_loss(pk, y),
            Δ_ll      = pd.Δ,
            t_stat    = pd.t,
            brier_mod = tp00_brier(pm, y),
            brier_mkt = tp00_brier(pk, y),
        ))
    end
    return DataFrame(out)
end

function tp00_summary(joined::Dict{String,DataFrame})
    println()
    println("-" ^ 74)
    println("GATE 6 SUMMARY — LOG LOSS BY MARKET (Model 00 Pure Poisson)")
    println("-" ^ 74)
    for (name, df) in joined
        isempty(df) && continue
        println("Baseline: ", name)
        st = tp00_score_table(df)
        for r in eachrow(st)
            @printf("  %-8s %4.1f %-6s  n=%3d  Δll=%+8.4f (t=%+5.2f)\n",
                    r.market, r.line, string(r.selection), r.n, r.Δ_ll, r.t_stat)
        end
        println("-" ^ 74)
    end
    return nothing
end

function tp00_summary_shape(fx::DataFrame)
    println()
    println("-" ^ 74)
    println("MARKET-FREE SHAPE SUMMARY (Model 00 Pure Poisson)")
    println("-" ^ 74)
    rqr = filter(isfinite, vcat(fx.rqr_h, fx.rqr_a))
    lpd_clean = filter(isfinite, fx.lpd)
    @printf("  LPD mean           : %.4f (total: %.1f)\n", mean(lpd_clean), sum(lpd_clean))
    @printf("  RQR mean / sd      : %+.4f / %.4f\n", mean(rqr), std(rqr))
    @printf("  Obs vs Pred Draw   : %.4f vs %.4f\n", mean(fx.is_draw), mean(fx.p_draw))
    println("-" ^ 74)
    return nothing
end

function tp00_edge_table(df::DataFrame)
    gdf = groupby(df, [:market, :line, :selection])
    out = NamedTuple[]
    for ((mkt, line, sel), sub) in pairs(gdf)
        y = sub.is_winner
        logit_mkt = _tp00_logit.(sub.p_market)
        logit_mod = _tp00_logit.(sub.p_model)
        
        # Calibration regression y ~ logit_mod
        cal_df = DataFrame(y = Float64.(y), x = logit_mod)
        cal_mod = try glm(@formula(y ~ x), cal_df, Binomial(), LogitLink()) catch; nothing end
        slope = cal_mod !== nothing ? coef(cal_mod)[2] : NaN
        se_slope = cal_mod !== nothing ? stderror(cal_mod)[2] : NaN

        push!(out, (
            market    = mkt,
            line      = line,
            selection = sel,
            n         = nrow(sub),
            slope     = slope,
            se_slope  = se_slope,
            sd_model  = std(sub.p_model),
            sd_market = std(sub.p_market),
        ))
    end
    return DataFrame(out)
end

function tp00_gate_not_broken(scores::DataFrame, edges::DataFrame)
    results = []
    worst_dll = maximum(scores.Δ_ll)
    push!(results, (
        name   = "no catastrophic line (Δll <= +0.02)",
        pass   = worst_dll <= 0.02,
        detail = @sprintf("worst Δll = %+0.4f", worst_dll),
    ))
    return results
end

function tp00_fold_weighting_check(df::DataFrame, folds::Vector{TP00Fold})
    pooled_ll = tp00_log_loss(df.p_model, df.is_winner)
    println("  Pooled log loss: ", round(pooled_ll, digits = 4))
    return nothing
end
