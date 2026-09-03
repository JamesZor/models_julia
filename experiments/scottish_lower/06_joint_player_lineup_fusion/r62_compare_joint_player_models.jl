# ==============================================================================
# r62 — Proper-score and Betfair closing-calibration comparison
# ==============================================================================

using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l60_loader.jl"))

const R62_EVAL = BayesianFootball.Evaluation

"Convert Betfair TWA closes into the evaluator's enriched long odds contract."
function r62_betfair_closing_odds(ds)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    transform!(
        groupby(odds, [:match_id, :market_name, :market_line]),
        :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close,
    )

    outcome_cols = [:match_id, :market_name, :market_line, :selection, :is_winner]
    winners = unique(select(ds.odds, outcome_cols))
    odds = leftjoin(odds, winners;
                    on = [:match_id, :market_name, :market_line, :selection])
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

function r62_curve_frame(name, source, curve)
    return DataFrame(
        model = fill(String(name), length(curve)),
        source = fill(String(source), length(curve)),
        bin = collect(1:length(curve)),
        lower = curve.edges[1:end-1],
        upper = curve.edges[2:end],
        n = curve.counts,
        predicted = curve.mean_predicted,
        observed = curve.observed,
    )
end

println("\n" * "="^110)
println(" EXPERIMENT 06 · PROPER SCORING AND BETFAIR CLOSING CALIBRATION")
println("="^110)

fits = Dict(name => load_fit(db, name) for name in L60_MODEL_NAMES)
for name in L60_MODEL_NAMES
    f = fits[name]
    if !f.diagnostics.passed
        @warn "Model $(name) did not pass strict convergence gating" rhat=f.diagnostics.max_rhat ess=f.diagnostics.min_ess_bulk divergences=f.diagnostics.n_divergent
    else
        println("  [PASS] $(name): R̂=$(round(f.diagnostics.max_rhat, digits=4)), ESS=$(f.diagnostics.min_ess_bulk), div=$(f.diagnostics.n_divergent)")
    end
end

bf_odds = r62_betfair_closing_odds(ds)
println("  Betfair rows: $(nrow(bf_odds)) across $(length(unique(bf_odds.match_id))) matches")

metrics = R62_EVAL.AbstractScoringRule[
    R62_EVAL.LogLoss(),
    R62_EVAL.CRPS(),
    R62_EVAL.PredictionScore(),
]
summary_rows = NamedTuple[]
curve_frames = DataFrame[]

for name in L60_MODEL_NAMES
    fit = fits[name]
    ctx = build_evaluation_context(
        fit_latents(fit), bf_odds, ds.matches, metrics;
        threaded = true,
    )
    prediction = evaluate_predictions(ctx; n_bins = 10)
    crps = R62_EVAL.compute_metric(R62_EVAL.CRPS(), ctx)
    model_curve = calibration_curve(ctx; n_bins = 10, source = :model)
    market_curve = calibration_curve(ctx; n_bins = 10, source = :market)
    push!(curve_frames, r62_curve_frame(name, :model, model_curve))
    push!(curve_frames, r62_curve_frame(name, :betfair, market_curve))

    push!(summary_rows, (
        model = name,
        logloss = prediction.model.logloss,
        betfair_logloss = prediction.market.logloss,
        delta_logloss = prediction.model.logloss - prediction.market.logloss,
        brier = prediction.model.brier,
        betfair_brier = prediction.market.brier,
        crps = crps.all.mean,
        rps = prediction.model.rps,
        betfair_rps = prediction.market.rps,
        ece = prediction.model.ece,
        betfair_ece = prediction.market.ece,
        mce = prediction.model.mce,
        n_obs = prediction.model.n_obs,
    ))
end

summary = sort(DataFrame(summary_rows), :logloss)
curves = vcat(curve_frames...)
output_dir = joinpath(@__DIR__, "results")
mkpath(output_dir)
CSV.write(joinpath(output_dir, "r62_proper_scores.csv"), summary)
CSV.write(joinpath(output_dir, "r62_calibration_curves.csv"), curves)

println("\n" * "="^154)
@printf(" %-45s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %7s\n",
        "Model", "LogLoss", "BF LL", "Brier", "CRPS", "RPS", "ECE", "BF ECE", "N")
println("-"^154)
for row in eachrow(summary)
    @printf(" %-45s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %7d\n",
            row.model, row.logloss, row.betfair_logloss, row.brier, row.crps,
            row.rps, row.ece, row.betfair_ece, row.n_obs)
end
println("="^154)
println("Saved summary and ten-bin model/Betfair calibration curves under $output_dir")
