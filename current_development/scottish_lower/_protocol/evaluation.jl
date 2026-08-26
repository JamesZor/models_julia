# ==============================================================================
# Gate 6 — evaluation
# ==============================================================================
# This layer is distribution-neutral.  All model-specific probabilities, score
# grids, CDF bounds, and reference calculations are obtained through adapter hooks.

using DataFrames
using Distributions
using GLM
using Random
using Statistics
using Printf

const SLEvalData = BayesianFootball.Data

"""Build the proportional de-vigged bookmaker close for the declared book."""
function sl_market_book(odds_df::AbstractDataFrame, contract::SLContract;
                        ids::Union{Nothing, AbstractSet} = nothing)
    wanted = Set((SLEvalData.market_group(m), Float64(SLEvalData.market_line(m)))
                 for m in sl_book_markets(contract))
    source = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, odds_df)
    ids === nothing || (source = filter(r -> Int(r.match_id) in ids, source))
    isempty(source) && return DataFrame()
    return DataFrame(match_id = Int.(source.match_id), market = String.(source.market_name),
        line = Float64.(source.market_line), selection = Symbol.(source.selection),
        p_market = Float64.(source.prob_implied_close) ./ Float64.(source.overround_close),
        is_winner = Bool.(coalesce.(source.is_winner, false)))
end

"""Drop incomplete fixture-markets before probabilities are scored."""
function sl_drop_incomplete(book::AbstractDataFrame)
    isempty(book) && return (DataFrame(), 0)
    kept = DataFrame[]
    dropped = 0
    for group in groupby(book, [:match_id, :market, :line])
        expected = group.market[1] == "1X2" ? 3 : 2
        if nrow(group) == expected
            push!(kept, DataFrame(group))
        else
            dropped += 1
        end
    end
    return (isempty(kept) ? DataFrame() : vcat(kept...), dropped)
end

"""Apply the capability-declared grid normalisation policy to draw probabilities."""
function _sl_normalized(values, masses, adapter::AbstractSLModelAdapter)
    _sl_grid_is_normalized(adapter) && return values
    return values ./ masses
end

"""
    sl_model_book(adapter, latents, ds, contract; seed=20260826) -> (book, fixtures)

Stream posterior score grids one fixture at a time.  Market probabilities are
normalised only when the adapter declares a truncated/sub-stochastic grid.  LPD is
posterior-predictive scoreline density; RQR uses adapter CDF bounds rather than a
truncated grid marginal.
"""
function sl_model_book(adapter::AbstractSLModelAdapter, latents, ds, contract::SLContract;
                       seed::Int = 20260826)
    rng = Random.MersenneTwister(seed)
    scores = select(ds.matches, :match_id, :home_score, :away_score)
    latent_rows = innerjoin(latents.df, scores, on = :match_id)
    book_rows = NamedTuple[]
    fixture_rows = NamedTuple[]

    for row in eachrow(latent_rows)
        (ismissing(row.home_score) || ismissing(row.away_score)) && continue
        home, away = Int(row.home_score), Int(row.away_score)
        score = sl_score_matrix(adapter, row; max_goals = contract.max_goals)
        grid = SLPredictions.score_matrix_data(score)
        masses = [sum(grid[:, :, draw]) for draw in axes(grid, 3)]
        any(x -> !isfinite(x) || x <= 0, masses) && continue

        for market in sl_book_markets(contract)
            quoted = SLPredictions.compute_market_probs(score, market)
            for (_, selection) in pairs(SLEvalData.outcomes(market))
                probability = mean(_sl_normalized(quoted[selection], masses, adapter))
                push!(book_rows, (match_id = Int(row.match_id),
                    market = SLEvalData.market_group(market),
                    line = Float64(SLEvalData.market_line(market)), selection, p_model = probability))
            end
        end

        lpd = home < contract.max_goals && away < contract.max_goals ?
            log(mean(_sl_normalized(grid[home + 1, away + 1, :], masses, adapter))) : NaN
        rqr(side, observed) = begin
            lower, upper = sl_marginal_cdf_bounds(adapter, side, row, observed)
            lo, hi = mean(lower), mean(upper)
            u = lo + rand(rng) * (hi - lo)
            quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
        end
        push!(fixture_rows, (match_id = Int(row.match_id), home_score = home, away_score = away,
            lpd, rqr_h = rqr(:home, home), rqr_a = rqr(:away, away), mass = mean(masses),
            is_draw = home == away, p_draw = NaN))
    end

    book, fixtures = DataFrame(book_rows), DataFrame(fixture_rows)
    isempty(fixtures) && return (book, fixtures)
    draws = filter(r -> r.market == "1X2" && r.selection == :draw, book)
    draw_map = Dict(r.match_id => r.p_model for r in eachrow(draws))
    fixtures.p_draw = [get(draw_map, id, NaN) for id in fixtures.match_id]
    return (book, fixtures)
end

"""
    sl_betfair_book(ds, contract, grading; ids=nothing, window=(-20.0, 0.0))

Build an exchange close directly from the close window.  It intentionally does not
require an opening-window observation, because an unavailable opening quote must
not discard a valid close.
"""
function sl_betfair_book(ds, contract::SLContract, grading::AbstractDataFrame;
                         ids::Union{Nothing, AbstractSet} = nothing, window = (-20.0, 0.0))
    data = BayesianFootball.Data
    raw = data.summarize_odds(ds.betfair_odds, data.TWAEstimator(); window = window)
    isempty(raw) && return DataFrame()
    wanted = Set((data.market_group(m), Float64(data.market_line(m))) for m in sl_book_markets(contract))
    source = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, raw)
    ids === nothing || (source = filter(r -> Int(r.match_id) in ids, source))
    "is_sane" in names(source) && (source = filter(r -> coalesce(r.is_sane, true), source))
    isempty(source) && return DataFrame()
    close = DataFrame(match_id = Int.(source.match_id), market = String.(source.market_name),
        line = Float64.(source.market_line), selection = Symbol.(source.selection),
        p_market = (1 ./ Float64.(source.odds)) ./ Float64.(source.overround),
        odds_close = Float64.(source.odds))
    return innerjoin(close, select(grading, [:match_id, :market, :line, :selection, :is_winner]),
                     on = [:match_id, :market, :line, :selection])
end

"""Align each baseline independently, preserving its own fixture coverage."""
function sl_join_books(model_book::AbstractDataFrame, books::Dict{String, <:AbstractDataFrame})
    aligned = Dict{String, DataFrame}()
    for (name, book) in books
        aligned[name] = isempty(book) ? DataFrame() : innerjoin(model_book,
            select(book, [:match_id, :market, :line, :selection, :p_market, :is_winner]),
            on = [:match_id, :market, :line, :selection])
    end
    return aligned
end

_sl_clamp(p) = clamp(p, 1e-9, 1 - 1e-9)
sl_log_loss(p, y) = -mean(yi ? log(_sl_clamp(pi)) : log(1 - _sl_clamp(pi)) for (pi, yi) in zip(p, y))
sl_brier(p, y) = mean((p .- Float64.(y)) .^ 2)

"""Paired model-minus-baseline log-loss difference on identical outcomes."""
function sl_paired_delta(model_probability, market_probability, winner)
    isempty(model_probability) && return (; Δ = NaN, se = NaN, t = NaN)
    differences = [(yi ? -log(_sl_clamp(pm)) : -log(1 - _sl_clamp(pm))) -
                   (yi ? -log(_sl_clamp(pb)) : -log(1 - _sl_clamp(pb)))
                   for (pm, pb, yi) in zip(model_probability, market_probability, winner)]
    se = std(differences) / sqrt(length(differences))
    return (; Δ = mean(differences), se, t = se > 0 ? mean(differences) / se : 0.0)
end

"""Calibration and encompassing logistic regressions for one selection line."""
function _sl_glm_edge(group::AbstractDataFrame)
    data = DataFrame(y = Float64.(group.is_winner), lm = _sl_logit.(group.p_model),
                     lb = _sl_logit.(group.p_market))
    if nrow(data) < 30 || length(unique(data.y)) < 2
        return (; n = nrow(data), slope = NaN, z_slope = NaN, β_model = NaN, z_model = NaN)
    end
    calibration = glm(@formula(y ~ lm), data, Binomial(), LogitLink())
    encompassing = glm(@formula(y ~ lb + lm), data, Binomial(), LogitLink())
    c, s = coef(calibration), stderror(calibration)
    e, es = coef(encompassing), stderror(encompassing)
    return (; n = nrow(data), slope = c[2], z_slope = s[2] > 0 ? (c[2] - 1) / s[2] : NaN,
            β_model = e[3], z_model = es[3] > 0 ? e[3] / es[3] : NaN)
end
_sl_logit(p) = log(_sl_clamp(p) / (1 - _sl_clamp(p)))

"""Gate 6a: winner grading, probability, completeness, and contract-book integrity."""
function sl_gate_book_integrity(book::AbstractDataFrame, contract::SLContract)
    isempty(book) && return [sl_result("book non-empty", false, "no rows")]
    groups = groupby(book, [:match_id, :market, :line])
    winners = combine(groups, :is_winner => sum => :n_winners)
    sums = combine(groups, :p_market => sum => :probability_sum)
    complete = all(nrow(g) == (g.market[1] == "1X2" ? 3 : 2) for g in groups)
    wanted = Set((SLEvalData.market_group(m), Float64(SLEvalData.market_line(m))) for m in sl_book_markets(contract))
    found = Set((r.market, r.line) for r in eachrow(unique(select(book, [:market, :line]))))
    return Any[
        sl_result("exactly one winner per market", all(winners.n_winners .== 1), "$(nrow(winners)) fixture-markets"),
        sl_result("de-vigged probabilities sum to 1", maximum(abs.(sums.probability_sum .- 1)) <= 1e-9,
                  @sprintf("max %.3e", maximum(abs.(sums.probability_sum .- 1)))),
        sl_result("probabilities in (0, 1)", all(0 .< book.p_market .< 1), "market probability range checked"),
        sl_result("markets have all their selections", complete, "$(length(groups)) markets"),
        sl_result("book is the contract book", found == wanted, "$(length(found))/$(length(wanted)) markets"),
    ]
end

"""Gate 6b: baseline-specific fixture coverage, grader agreement, and model range."""
function sl_gate_alignment(joined::Dict, model_book::AbstractDataFrame; min_coverage::Float64 = 0.80)
    checks = Any[]
    fixture_count = isempty(model_book) ? 0 : length(unique(model_book.match_id))
    push!(checks, sl_result("model book built", fixture_count > 0, "$(fixture_count) fixtures"))
    for (name, data) in sort(collect(joined); by = first)
        covered = isempty(data) ? 0 : length(unique(data.match_id))
        coverage = fixture_count == 0 ? 0.0 : covered / fixture_count
        push!(checks, sl_result("fixture coverage vs $name", coverage >= min_coverage,
            @sprintf("%d/%d fixtures (%.1f%%)", covered, fixture_count, 100coverage)))
        if !isempty(data)
            lines = combine(groupby(data, [:market, :line]), :match_id => (x -> length(unique(x))) => :fixtures)
            detail = join(["$(r.market) $(r.line): $(r.fixtures)" for r in eachrow(lines)], " | ")
            push!(checks, sl_result("per-line coverage vs $name", true, detail))
        end
    end
    names_ = collect(keys(joined))
    if length(names_) > 1 && !isempty(joined[names_[1]]) && !isempty(joined[names_[2]])
        left = select(joined[names_[1]], [:match_id, :market, :line, :selection, :is_winner])
        right = select(joined[names_[2]], [:match_id, :market, :line, :selection, :is_winner])
        overlap = innerjoin(left, right, on = [:match_id, :market, :line, :selection], makeunique = true)
        disagreements = count(r -> r.is_winner != r.is_winner_1, eachrow(overlap))
        push!(checks, sl_result("graders agree where they overlap", disagreements == 0,
            "$(nrow(overlap)) shared rows, $disagreements disagreements"))
    end
    probabilities = isempty(model_book) ? Float64[] : model_book.p_model
    push!(checks, sl_result("model probabilities well-formed", !isempty(probabilities) &&
        all(isfinite, probabilities) && all(0 .< probabilities .< 1), "model probability range checked"))
    return checks
end

"""Per-selection proper scores, paired rather than independent comparisons."""
function sl_score_table(joined::AbstractDataFrame)
    rows = NamedTuple[]
    for group in groupby(joined, [:market, :line, :selection])
        delta = sl_paired_delta(group.p_model, group.p_market, group.is_winner)
        edge = _sl_glm_edge(group)
        push!(rows, (market = group.market[1], line = group.line[1], selection = group.selection[1], n = nrow(group),
            base_rate = mean(group.is_winner), ll_model = sl_log_loss(group.p_model, group.is_winner),
            ll_market = sl_log_loss(group.p_market, group.is_winner), Δll = delta.Δ, t = delta.t,
            brier_model = sl_brier(group.p_model, group.is_winner), brier_market = sl_brier(group.p_market, group.is_winner),
            slope = edge.slope, z_slope = edge.z_slope, β_model = edge.β_model, z_model = edge.z_model))
    end
    return isempty(rows) ? DataFrame() : sort!(DataFrame(rows), [:market, :line, :selection])
end

"""Gate 6c: posterior-predictive RQR, LPD, grid mass, and draw-rate checks."""
function sl_gate_shape(fixtures::AbstractDataFrame; mean_tol = 0.15, sd_lo = 0.85, sd_hi = 1.15)
    isempty(fixtures) && return [sl_result("shape inputs", false, "empty")]
    residuals = filter(isfinite, vcat(fixtures.rqr_h, fixtures.rqr_a))
    isempty(residuals) && return [sl_result("shape inputs", false, "no finite residuals")]
    home = filter(isfinite, fixtures.rqr_h); away = filter(isfinite, fixtures.rqr_a)
    lpd = filter(isfinite, fixtures.lpd)
    observed_draw = mean(fixtures.is_draw)
    predicted_draw = mean(filter(isfinite, fixtures.p_draw))
    draw_se = sqrt(observed_draw * (1 - observed_draw) / nrow(fixtures))
    draw_z = draw_se > 0 ? (observed_draw - predicted_draw) / draw_se : 0.0
    return Any[
        sl_result("RQR mean (bias)", abs(mean(residuals)) <= mean_tol, @sprintf("%.4f", mean(residuals))),
        sl_result("RQR sd (dispersion)", sd_lo <= std(residuals) <= sd_hi, @sprintf("%.4f", std(residuals))),
        sl_result("RQR home/away symmetry", abs(mean(home) - mean(away)) <= 2mean_tol,
                  @sprintf("home %.4f away %.4f", mean(home), mean(away))),
        sl_result("scorelines on the grid", length(lpd) == nrow(fixtures), "$(length(lpd))/$(nrow(fixtures))"),
        sl_result("LPD (reported, not gated)", true, @sprintf("mean %.4f total %.2f", mean(lpd), sum(lpd))),
        sl_result("grid mass (truncation correction)", all(isfinite, fixtures.mass) && minimum(fixtures.mass) > 0,
                  @sprintf("mean %.8f", mean(fixtures.mass))),
        sl_result("draw rate matches prediction", abs(draw_z) <= 2, @sprintf("z = %+.2f", draw_z)),
    ]
end

"""Gate 6d: no catastrophic proper-score loss or significant calibration failure."""
function sl_gate_not_broken(scores::AbstractDataFrame; max_delta = 0.02)
    isempty(scores) && return [sl_result("score table", false, "empty")]
    finite_delta = filter(isfinite, scores.Δll)
    finite_slope = filter(isfinite, scores.z_slope)
    isempty(finite_delta) && return [sl_result("score table", false, "no finite scores")]
    return Any[
        sl_result("no line catastrophically worse than market", maximum(finite_delta) <= max_delta,
                  @sprintf("worst Δll %.4f", maximum(finite_delta))),
        sl_result("calibration slopes not significantly off", all(abs.(finite_slope) .<= 2),
                  "$(count(z -> abs(z) <= 2, finite_slope))/$(length(finite_slope)) lines within two standard errors"),
    ]
end

"""Run the consolidated evaluation gate and append adapter-owned referee checks."""
function sl_gate_evaluation(adapter::AbstractSLModelAdapter, book::AbstractDataFrame,
                            fixtures::AbstractDataFrame, joined::Dict)
    checks = vcat(sl_gate_book_integrity(book, sl_contract()), sl_gate_alignment(joined, book),
                  sl_gate_shape(fixtures))
    return vcat(checks, sl_adapter_check(adapter, :evaluation, book, fixtures, joined))
end

"""Compare pooled fixture weighting with an unweighted fold average for 1X2."""
function sl_fold_weighting_check(joined::AbstractDataFrame, folds)
    fold_for_match = Dict{Int, Int}()
    for fold in folds, match_id in fold.oos_df.match_id
        fold_for_match[Int(match_id)] = fold.idx
    end
    data = filter(r -> r.market == "1X2", copy(joined))
    isempty(data) && return (; pooled_1x2 = NaN, fold_averaged_1x2 = NaN,
                              difference = NaN, fold_sizes = (0, 0), n_folds = 0)
    data.fold = [get(fold_for_match, Int(id), 0) for id in data.match_id]
    pooled = sl_log_loss(data.p_model, data.is_winner)
    per_fold = combine(groupby(filter(r -> r.fold > 0, data), :fold),
        [:p_model, :is_winner] => ((p, y) -> sl_log_loss(p, y)) => :loss, nrow => :n)
    isempty(per_fold) && return (; pooled_1x2 = pooled, fold_averaged_1x2 = NaN,
                                  difference = NaN, fold_sizes = (0, 0), n_folds = 0)
    average = mean(per_fold.loss)
    return (; pooled_1x2 = pooled, fold_averaged_1x2 = average,
            difference = average - pooled, fold_sizes = extrema(per_fold.n), n_folds = nrow(per_fold))
end

"""Calibration and market-encompassing regression diagnostics per selection line."""
function sl_edge_table(joined::AbstractDataFrame)
    rows = NamedTuple[]
    for group in groupby(joined, [:market, :line, :selection])
        edge = _sl_glm_edge(group)
        push!(rows, (market = group.market[1], line = group.line[1], selection = group.selection[1],
            n = edge.n, sd_model = std(group.p_model), sd_market = std(group.p_market),
            slope = edge.slope, z_slope = edge.z_slope, β_model = edge.β_model, z_model = edge.z_model))
    end
    return isempty(rows) ? DataFrame() : sort!(DataFrame(rows), [:market, :line, :selection])
end

"""Multiclass market and whole-book summary; each fixture is paired by its id."""
function sl_summary(joined::Dict{String, DataFrame}; baselines = sort(collect(keys(joined))))
    rows = NamedTuple[]
    for baseline in baselines
        data = joined[baseline]
        isempty(data) && continue
        for group in groupby(data, [:market, :line])
            model_loss = Float64[]
            market_loss = Float64[]
            for fixture in groupby(group, :match_id)
                winner = findfirst(fixture.is_winner)
                winner === nothing && continue
                push!(model_loss, -log(_sl_clamp(fixture.p_model[winner])))
                push!(market_loss, -log(_sl_clamp(fixture.p_market[winner])))
            end
            isempty(model_loss) && continue
            delta = sl_paired_delta(exp.(-model_loss), exp.(-market_loss), trues(length(model_loss)))
            label = group.market[1] * (group.line[1] == 0.0 ? "" : " $(group.line[1])")
            push!(rows, (baseline, market = label, n = length(model_loss),
                ll_model = mean(model_loss), ll_mkt = mean(market_loss), Δ = mean(model_loss .- market_loss), t = delta.t))
        end
    end
    return isempty(rows) ? DataFrame() : sort!(DataFrame(rows), [:baseline, :market])
end

"""One-line market-free shape summary for comparing model variants."""
function sl_summary_shape(fixtures::AbstractDataFrame)
    residuals = filter(isfinite, vcat(fixtures.rqr_h, fixtures.rqr_a))
    lpd = filter(isfinite, fixtures.lpd)
    draws = filter(isfinite, fixtures.p_draw)
    return DataFrame(metric = ["RQR mean", "RQR sd", "LPD mean", "LPD total", "draw observed", "draw predicted"],
        value = [mean(residuals), std(residuals), mean(lpd), sum(lpd), mean(fixtures.is_draw), mean(draws)])
end
