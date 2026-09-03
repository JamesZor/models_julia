# ==============================================================================
# compare_scottish_experiments.jl — unified cross-paradigm benchmark
# ==============================================================================
#
# Four model generations, one bench. Each generation was developed in its own experiment
# directory against its own leaderboard; this runner puts the champion and the control of
# each side by side on three axes that the per-experiment runners never share:
#
#   A. RQR      — Dunn-Smyth randomized quantile residuals of the GOALS marginal. Answers
#                 "did the likelihood upgrade actually fix the dispersion?", which no proper
#                 score answers: log loss rewards sharpness and calibration jointly and can
#                 improve while the count distribution stays wrong.
#   B. GLM edge — reliability, ECE/MCE against the Betfair close, and the logistic
#                 recalibration (Platt) shift a Kelly stake would have needed.
#   C. Portfolio— fractional-Kelly performance at the exchange close.
#
# ONE PRICE SOURCE, RECOMPUTED. `portfolio_runs` is NOT read for dimension C, deliberately.
# Experiments 01/03 priced off `ds.odds` (bookmaker close, overround intact) and 05/06 off
# the Betfair exchange close; the two reach opposite conclusions from identical posteriors
# (`03_joint_gamma_poisson/NOTES.md` §2). Ranking those rows against each other would compare
# price sources and call it a model comparison. Every generation is therefore re-simulated
# here under one BookSpec, one PolicySpec and one price source. The persisted rows are still
# reported alongside, tagged with their own `odds_source`, as provenance.
#
# THIS RUNNER LAUNCHES NO MCMC. It reads completed runs from PostgreSQL.
#
# Usage:
#
#   julia --project -t 16 experiments/scottish_lower/compare_scottish_experiments.jl
#
# Outputs:
#   experiments/scottish_lower/results/unified_paradigm_comparison.csv
#   experiments/scottish_lower/results/unified_reliability_curves.csv
#   experiments/scottish_lower/results/unified_rqr_residuals.csv
#   experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md

using BayesianFootball
using CSV
using DataFrames
using Dates
using Distributions
using GLM
using HypothesisTests
using LibPQ
using LinearAlgebra
using Printf
using Random
using Statistics
using StatsBase
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const CMP_PORTFOLIO = BayesianFootball.Portfolio
const CMP_EVAL = BayesianFootball.Evaluation

# Randomized quantile residuals are randomized. A fixed seed makes the reported moments and
# normality p-values reproducible; the conclusions do not depend on the draw, but the fourth
# decimal place does, and an irreproducible table invites re-running until it agrees.
const CMP_SEED = 20260903

const CMP_OUTPUT_DIR = joinpath(@__DIR__, "results")
const CMP_REPORT_PATH = joinpath(@__DIR__, "UNIFIED_PARADIGM_REPORT.md")

# ==============================================================================
# 1. The bench — champion and control per generation
# ==============================================================================

struct Candidate
    generation::Int
    paradigm::String
    experiment::String
    run_name::String
    role::Symbol          # :control or :champion
    likelihood::String
end

const CMP_CANDIDATES = Candidate[
    Candidate(1, "Poisson", "scottish_lower_poisson_2426",
              "m00_baseline", :control, "Poisson"),
    Candidate(1, "Poisson", "scottish_lower_poisson_2426",
              "m05_production_wealth", :champion, "Poisson"),
    Candidate(2, "Negative Binomial", "scottish_lower_negbin_2426",
              "m00_negbin_baseline", :control, "NegBin"),
    Candidate(2, "Negative Binomial", "scottish_lower_negbin_2426",
              "m05_negbin_production_wealth", :champion, "NegBin"),
    Candidate(3, "Two-arm joint Gamma-Poisson", "scottish_lower_joint_2426",
              "m00_joint_baseline", :control, "Joint Gamma-Poisson"),
    Candidate(3, "Two-arm joint Gamma-Poisson", "scottish_lower_joint_2426",
              "m05_joint_production_wealth", :champion, "Joint Gamma-Poisson"),
    Candidate(4, "Joint player-lineup hybrid", "scottish_lower_joint_player_2426",
              "m12_joint_hybrid_synergy", :control, "Joint Gamma-Poisson"),
    Candidate(4, "Joint player-lineup hybrid", "scottish_lower_joint_player_2426",
              "m13_joint_composite", :champion, "Joint Gamma-Poisson"),
]

cmp_label(c::Candidate) = "G$(c.generation) $(c.run_name)"

# ==============================================================================
# 2. Shared price source and portfolio recipe
# ==============================================================================

"Betfair exchange close, time-weighted over the last 20 minutes before kickoff."
function cmp_betfair_closing_odds(ds::Data.DataStore)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"The evaluator wants the same frame enriched with implied and vig-free closing probabilities."
function cmp_enrich_odds(odds::DataFrame, ds::Data.DataStore)
    enriched = copy(odds)
    enriched.prob_implied_close = 1.0 ./ enriched.odds_close
    transform!(
        groupby(enriched, [:match_id, :market_name, :market_line]),
        :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close,
    )
    outcome_cols = [:match_id, :market_name, :market_line, :selection, :is_winner]
    winners = unique(select(ds.odds, outcome_cols))
    enriched = leftjoin(enriched, winners;
                        on = [:match_id, :market_name, :market_line, :selection])
    sort!(enriched, [:match_id, :market_name, :market_line, :selection])
    return enriched
end

cmp_book_spec() = BookSpec(
    markets = Data.MarketConfig(Data.AbstractMarket[
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS(),
    ]),
    price = DeArb(),
    allocator = KellyLogUtility(),
    shrink = CMP_PORTFOLIO.FractionalKelly(0.30),
    exec = ExecutionConfig(
        commission = PerBetCommission(0.02),
        budget = 0.99,
        min_selection_stake = 0.001,
    ),
)

cmp_policy_spec() = PolicySpec(
    trust = FlatTrust(1.0),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(0.20),
    grouping = DailySlate(),
)

# ==============================================================================
# 3. Dimension A — Dunn-Smyth randomized quantile residuals
# ==============================================================================

"""
    cmp_predictive_cdf(latents, row, y) -> Float64

`F(y)` under the POSTERIOR PREDICTIVE, i.e. `(1/S) Σ_s F(y | θ_s)`, not `F(y | θ̄)`.

The distinction is the whole point of the exercise. Averaging the parameters first and then
taking one CDF discards the posterior spread in λ, and that spread is itself a source of
predictive overdispersion — a Poisson model with a wide posterior on λ has a predictive
variance above its predictive mean even though every draw is equidispersed. Collapsing to the
posterior mean would therefore charge the LIKELIHOOD for dispersion the model already
handles, and make the Poisson generation look worse than it is.
"""
function cmp_predictive_cdf(λ::AbstractVector{<:Real}, r::Union{Nothing,AbstractVector{<:Real}},
                            y::Int)
    y < 0 && return 0.0
    n = length(λ)
    acc = 0.0
    if r === nothing
        @inbounds for s in 1:n
            acc += cdf(Poisson(λ[s]), y)
        end
    else
        # RobustNegativeBinomial(r, μ) is NegativeBinomial(r, r / (r + μ)) — mean μ, variance
        # μ + μ²/r. Distributions' parameterisation is used here for its tested `cdf`.
        @inbounds for s in 1:n
            acc += cdf(NegativeBinomial(r[s], r[s] / (r[s] + λ[s])), y)
        end
    end
    return acc / n
end

"""
    cmp_rqr(fit, ds; rng) -> NamedTuple

Randomized quantile residuals for the home and away goal marginals of every out-of-sample
fixture the fit holds.

    u_i ~ Uniform(F(y_i - 1), F(y_i));   r_i = Φ⁻¹(u_i)

Under a correctly specified count model the `r_i` are exactly standard normal — mean 0,
variance 1. Variance above 1 is the signature of an unmodelled overdispersion; below 1,
of a predictive distribution that is too wide.
"""
function cmp_rqr(fit::Fit, ds::Data.DataStore; rng::AbstractRNG)
    latents = fit.latents
    latents isa CountLatents || error(
        "cmp_rqr expects CountLatents; got $(typeof(latents)). The RQR axis is defined on " *
        "the two count marginals and has no meaning for another latent shape.")

    scores = Dict{Int,Tuple{Int,Int}}()
    for row in eachrow(ds.matches)
        (ismissing(row.home_score) || ismissing(row.away_score)) && continue
        scores[Int(row.match_id)] = (Int(row.home_score), Int(row.away_score))
    end

    negbin = observation_family(latents) === :negbin
    ids = latents.match_ids
    residual_home = Float64[]
    residual_away = Float64[]
    kept_ids = Int[]
    sizehint!(residual_home, length(ids))
    sizehint!(residual_away, length(ids))

    for i in eachindex(ids)
        outcome = get(scores, ids[i], nothing)
        outcome === nothing && continue
        gh, ga = outcome

        λh = @view latents.λ_home[i, :]
        λa = @view latents.λ_away[i, :]
        rh = negbin ? (@view latents.observation_params.r_h[i, :]) : nothing
        ra = negbin ? (@view latents.observation_params.r_a[i, :]) : nothing

        for (λ, r, y, sink) in ((λh, rh, gh, residual_home), (λa, ra, ga, residual_away))
            lower = cmp_predictive_cdf(λ, r, y - 1)
            upper = cmp_predictive_cdf(λ, r, y)
            u = lower + (upper - lower) * rand(rng)
            # Φ⁻¹ is infinite at the open ends; a fixture in the extreme tail would otherwise
            # poison every moment downstream with an Inf.
            u = clamp(u, 1e-12, 1 - 1e-12)
            push!(sink, quantile(Normal(), u))
        end
        push!(kept_ids, ids[i])
    end

    pooled = vcat(residual_home, residual_away)
    isempty(pooled) && error("cmp_rqr scored no fixtures; the fit and DataStore do not overlap.")

    ks = ExactOneSampleKSTest(pooled, Normal())
    ad = OneSampleADTest(pooled, Normal())

    return (;
        residual_home, residual_away, pooled, match_ids = kept_ids,
        n = length(pooled),
        mean = mean(pooled),
        var = var(pooled),
        skewness = StatsBase.skewness(pooled),
        kurtosis = StatsBase.kurtosis(pooled),      # excess kurtosis; 0 under normality
        var_home = var(residual_home),
        var_away = var(residual_away),
        ks_stat = ks.δ,
        ks_pvalue = pvalue(ks),
        ad_stat = ad.A²,
        ad_pvalue = pvalue(ad),
    )
end

# ==============================================================================
# 4. Dimension B — GLM calibration edge
# ==============================================================================

cmp_logit(p) = log(p / (1 - p))
cmp_logistic(x) = 1 / (1 + exp(-x))

"""
    cmp_glm_edge(rows) -> NamedTuple

Fit `y ~ Bernoulli(logistic(α + β · logit(p_model)))` over the scored selections — the Platt
recalibration a Layer-2 shift model performs — and report what it had to do.

`α ≈ 0, β ≈ 1` is a model that needs no correction. `β < 1` is the overconfidence signature:
the model's probabilities are too extreme, and fractional Kelly, which is linear in the edge
`p - 1/o` near the origin, overstakes in direct proportion. `α ≠ 0` is a uniform directional
bias.

The recalibrated ECE reported alongside is IN-SAMPLE — the GLM saw these outcomes. It is a
measure of how much miscalibration is of a shape a two-parameter shift can absorb, not an
out-of-sample claim about a deployed calibrator.
"""
function cmp_glm_edge(rows::AbstractVector{<:CMP_EVAL.EvaluationRow}; n_bins::Int = 10)
    p = Float64[]
    y = Float64[]
    for r in rows
        r.outcome < 0 && continue
        isfinite(r.model_prob) || continue
        push!(p, clamp(r.model_prob, 1e-6, 1 - 1e-6))
        push!(y, CMP_EVAL.row_y(r))
    end
    length(p) < 50 && return (; n = length(p), alpha = NaN, beta = NaN,
                                alpha_se = NaN, beta_se = NaN,
                                ece_raw = NaN, ece_calibrated = NaN, ece_recovered = NaN)

    frame = DataFrame(y = y, lp = cmp_logit.(p))
    model = glm(@formula(y ~ lp), frame, Binomial(), LogitLink())
    coefs = coef(model)
    errors = stderror(model)
    calibrated = cmp_logistic.(coefs[1] .+ coefs[2] .* frame.lp)

    ece_raw = cmp_binned_ece(p, y; n_bins = n_bins)
    ece_cal = cmp_binned_ece(calibrated, y; n_bins = n_bins)

    return (; n = length(p), alpha = coefs[1], beta = coefs[2],
              alpha_se = errors[1], beta_se = errors[2],
              ece_raw = ece_raw, ece_calibrated = ece_cal,
              ece_recovered = ece_raw - ece_cal)
end

"Count-weighted mean reliability gap over equal-width probability bins."
function cmp_binned_ece(p::AbstractVector{<:Real}, y::AbstractVector{<:Real}; n_bins::Int = 10)
    counts = zeros(Int, n_bins)
    sum_p = zeros(Float64, n_bins)
    sum_y = zeros(Float64, n_bins)
    for i in eachindex(p)
        b = clamp(floor(Int, p[i] * n_bins) + 1, 1, n_bins)
        counts[b] += 1
        sum_p[b] += p[i]
        sum_y[b] += y[i]
    end
    total = sum(counts)
    total == 0 && return NaN
    acc = 0.0
    for b in 1:n_bins
        counts[b] == 0 && continue
        acc += counts[b] * abs(sum_y[b] / counts[b] - sum_p[b] / counts[b])
    end
    return acc / total
end

function cmp_curve_frame(label::AbstractString, source::AbstractString, curve)
    return DataFrame(
        model = fill(String(label), length(curve)),
        source = fill(String(source), length(curve)),
        bin = collect(1:length(curve)),
        lower = curve.edges[1:end-1],
        upper = curve.edges[2:end],
        n = curve.counts,
        predicted = curve.mean_predicted,
        observed = curve.observed,
    )
end

# ==============================================================================
# 5. Dimension C — persisted portfolio provenance
# ==============================================================================

"Every persisted portfolio row for a run, with the price source it was built on."
function cmp_persisted_portfolios(db::PostgresStorage, run_name::String)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            SELECT COALESCE(pr.metadata->>'odds_source', 'unrecorded') AS odds_source,
                   pr.portfolio_run_id, pr.n_bets, pr.total_return_pct, pr.flat_roi_pct,
                   pr.roi_1x2_pct, pr.max_drawdown_pct, pr.sharpe_ann, pr.win_rate
            FROM portfolio_runs pr
            JOIN runs r ON r.run_id = pr.model_run_id
            WHERE r.experiment_name = \$1 AND r.name = \$2
            ORDER BY pr.id DESC;
        """, (db.experiment_name, run_name))
        try
            return DataFrame(result)
        finally
            close(result)
        end
    finally
        close(conn)
    end
end

# ==============================================================================
# 6. Workflow
# ==============================================================================

function cmp_compare_scottish_experiments()
    println("\n", "="^120)
    println(" SCOTTISH LOWER — UNIFIED CROSS-PARADIGM BENCHMARK (RQR · GLM EDGE · BETFAIR PORTFOLIO)")
    println("="^120)

    rng = Xoshiro(CMP_SEED)
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    println("  DataStore : $(nrow(ds.matches)) matches, $(nrow(ds.odds)) bookmaker odds rows")

    bf_odds = cmp_betfair_closing_odds(ds)
    bf_eval_odds = cmp_enrich_odds(bf_odds, ds)
    println("  Betfair   : $(nrow(bf_odds)) closing rows across " *
            "$(length(unique(bf_odds.match_id))) matches")

    book_spec = cmp_book_spec()
    policy_spec = cmp_policy_spec()
    println("  Portfolio : FractionalKelly(0.30), SlateDrawdown(23), FixedCap(0.20), " *
            "2% commission, Betfair TWA close")

    storages = Dict{String,PostgresStorage}()
    for candidate in CMP_CANDIDATES
        get!(storages, candidate.experiment) do
            PostgresStorage(candidate.experiment)
        end
    end

    rows = NamedTuple[]
    curve_frames = DataFrame[]
    residual_frames = DataFrame[]
    persisted_frames = DataFrame[]

    for candidate in CMP_CANDIDATES
        label = cmp_label(candidate)
        println("\n", "-"^120)
        println(" $label  [$(candidate.paradigm), $(candidate.role)]")
        println("-"^120)

        db = storages[candidate.experiment]
        fit = load_fit(db, candidate.run_name)
        converged = fit.diagnostics.passed
        converged || @warn "$label did not pass strict convergence gating" rhat = fit.diagnostics.max_rhat ess = fit.diagnostics.min_ess_bulk divergences = fit.diagnostics.n_divergent
        println("  folds=$(length(fit)) OOS=$(n_matches(fit.latents)) " *
                "family=$(observation_family(fit.latents)) " *
                "R̂=$(round(fit.diagnostics.max_rhat, digits = 4)) " *
                "ESS=$(round(fit.diagnostics.min_ess_bulk, digits = 1)) " *
                "div=$(fit.diagnostics.n_divergent)")

        # --- A. RQR -----------------------------------------------------------------
        rqr = cmp_rqr(fit, ds; rng = rng)
        @printf("  [A] RQR      n=%d  mean=%+.4f  var=%.4f  skew=%+.4f  exkurt=%+.4f  KS p=%.4g  AD p=%.4g\n",
                rqr.n, rqr.mean, rqr.var, rqr.skewness, rqr.kurtosis,
                rqr.ks_pvalue, rqr.ad_pvalue)
        push!(residual_frames, DataFrame(
            model = fill(label, rqr.n),
            generation = fill(candidate.generation, rqr.n),
            side = [fill("home", length(rqr.residual_home));
                    fill("away", length(rqr.residual_away))],
            match_id = [rqr.match_ids; rqr.match_ids],
            residual = rqr.pooled,
        ))

        # --- B. GLM calibration edge --------------------------------------------------
        ctx = build_evaluation_context(
            fit_latents(fit), bf_eval_odds, ds.matches,
            CMP_EVAL.AbstractScoringRule[CMP_EVAL.LogLoss(), CMP_EVAL.PredictionScore()];
            threaded = true,
        )
        prediction = evaluate_predictions(ctx; n_bins = 10)
        eval_rows = evaluation_rows(ctx)
        glm_edge = cmp_glm_edge(eval_rows; n_bins = 10)
        model_curve = calibration_curve(ctx; n_bins = 10, source = :model)
        market_curve = calibration_curve(ctx; n_bins = 10, source = :market)
        push!(curve_frames, cmp_curve_frame(label, "model", model_curve))
        push!(curve_frames, cmp_curve_frame(label, "betfair", market_curve))
        @printf("  [B] GLM edge ECE=%.4f (BF %.4f)  MCE=%.4f  a=%+.4f  b=%.4f  post-GLM ECE=%.4f\n",
                prediction.model.ece, prediction.market.ece, prediction.model.mce,
                glm_edge.alpha, glm_edge.beta, glm_edge.ece_calibrated)

        # --- C. Portfolio -------------------------------------------------------------
        result, books, report = run_portfolio_simulation(
            book_spec, policy_spec, fit, bf_odds, ds;
            bootstrap = false,
            require_converged = false,
            quiet = true,
        )
        summary = result.summary
        @printf("  [C] Betfair  bets=%d  return=%+.2f%%  flat ROI=%+.2f%%  1X2 ROI=%+.2f%%  MDD=%.2f%%  Sharpe=%.3f  win=%.2f%%\n",
                summary.n_bets, summary.total_return_pct, summary.roi, summary.roi_1x2,
                summary.mdd, summary.sharpe_ann, 100 * summary.win_rate)

        persisted = cmp_persisted_portfolios(db, candidate.run_name)
        if nrow(persisted) > 0
            insertcols!(persisted, 1, :model => fill(label, nrow(persisted)))
            push!(persisted_frames, persisted)
            for row in eachrow(unique(persisted, :odds_source))
                @printf("      persisted [%-30s] return=%+.2f%%  Sharpe=%.3f  bets=%d\n",
                        row.odds_source, Float64(row.total_return_pct),
                        Float64(row.sharpe_ann), Int(row.n_bets))
            end
        end

        push!(rows, (
            generation = candidate.generation,
            paradigm = candidate.paradigm,
            experiment = candidate.experiment,
            model = candidate.run_name,
            role = String(candidate.role),
            likelihood = candidate.likelihood,
            n_folds = length(fit),
            n_oos_matches = n_matches(fit.latents),
            converged = converged,
            max_rhat = fit.diagnostics.max_rhat,
            min_ess_bulk = fit.diagnostics.min_ess_bulk,
            n_divergent = fit.diagnostics.n_divergent,
            # A
            rqr_n = rqr.n,
            rqr_mean = rqr.mean,
            rqr_var = rqr.var,
            rqr_var_home = rqr.var_home,
            rqr_var_away = rqr.var_away,
            rqr_skewness = rqr.skewness,
            rqr_excess_kurtosis = rqr.kurtosis,
            rqr_ks_stat = rqr.ks_stat,
            rqr_ks_pvalue = rqr.ks_pvalue,
            rqr_ad_stat = rqr.ad_stat,
            rqr_ad_pvalue = rqr.ad_pvalue,
            # B
            logloss = prediction.model.logloss,
            betfair_logloss = prediction.market.logloss,
            delta_logloss = prediction.model.logloss - prediction.market.logloss,
            brier = prediction.model.brier,
            betfair_brier = prediction.market.brier,
            rps = prediction.model.rps,
            betfair_rps = prediction.market.rps,
            ece = prediction.model.ece,
            betfair_ece = prediction.market.ece,
            mce = prediction.model.mce,
            betfair_mce = prediction.market.mce,
            n_scored = prediction.model.n_obs,
            glm_alpha = glm_edge.alpha,
            glm_alpha_se = glm_edge.alpha_se,
            glm_beta = glm_edge.beta,
            glm_beta_se = glm_edge.beta_se,
            glm_ece_raw = glm_edge.ece_raw,
            glm_ece_calibrated = glm_edge.ece_calibrated,
            glm_ece_recovered = glm_edge.ece_recovered,
            # C
            n_books = length(books),
            n_skipped = n_skipped(report),
            n_bets = summary.n_bets,
            total_return_pct = summary.total_return_pct,
            flat_roi_pct = summary.roi,
            roi_1x2_pct = summary.roi_1x2,
            max_drawdown_pct = summary.mdd,
            sharpe_ann = summary.sharpe_ann,
            win_rate = summary.win_rate,
        ))
    end

    summary_df = DataFrame(rows)
    mkpath(CMP_OUTPUT_DIR)
    CSV.write(joinpath(CMP_OUTPUT_DIR, "unified_paradigm_comparison.csv"), summary_df)
    CSV.write(joinpath(CMP_OUTPUT_DIR, "unified_reliability_curves.csv"),
              vcat(curve_frames...))
    CSV.write(joinpath(CMP_OUTPUT_DIR, "unified_rqr_residuals.csv"),
              vcat(residual_frames...))
    persisted_df = isempty(persisted_frames) ? DataFrame() :
                   vcat(persisted_frames...; cols = :union)
    isempty(persisted_df) ||
        CSV.write(joinpath(CMP_OUTPUT_DIR, "unified_persisted_portfolios.csv"), persisted_df)

    cmp_print_tables(summary_df)
    cmp_write_report(summary_df, persisted_df, book_spec, policy_spec, bf_odds, ds)

    println("\nWrote:")
    println("  ", joinpath(CMP_OUTPUT_DIR, "unified_paradigm_comparison.csv"))
    println("  ", joinpath(CMP_OUTPUT_DIR, "unified_reliability_curves.csv"))
    println("  ", joinpath(CMP_OUTPUT_DIR, "unified_rqr_residuals.csv"))
    isempty(persisted_df) ||
        println("  ", joinpath(CMP_OUTPUT_DIR, "unified_persisted_portfolios.csv"))
    println("  ", CMP_REPORT_PATH)

    return (; summary = summary_df, curves = vcat(curve_frames...),
              residuals = vcat(residual_frames...), persisted = persisted_df)
end

# ==============================================================================
# 7. Console rendering
# ==============================================================================

function cmp_print_tables(df::DataFrame)
    println("\n", "="^128)
    println(" A · RANDOMIZED QUANTILE RESIDUALS — goals marginal, posterior-predictive CDF")
    println("   target: mean 0, variance 1, skew 0, excess kurtosis 0; p > 0.05 fails to reject normality")
    println("="^128)
    @printf(" %-2s %-30s | %-20s | %7s | %8s | %8s | %8s | %8s | %9s | %9s\n",
            "G", "Model", "Likelihood", "N", "Mean", "Var", "Skew", "ExKurt", "KS p", "AD p")
    println("-"^128)
    for r in eachrow(df)
        @printf(" %-2d %-30s | %-20s | %7d | %+8.4f | %8.4f | %+8.4f | %+8.4f | %9.3g | %9.3g\n",
                r.generation, r.model, r.likelihood, r.rqr_n, r.rqr_mean, r.rqr_var,
                r.rqr_skewness, r.rqr_excess_kurtosis, r.rqr_ks_pvalue, r.rqr_ad_pvalue)
    end

    println("\n", "="^128)
    println(" B · GLM CALIBRATION EDGE vs BETFAIR CLOSE")
    println("   α = bias shift, β = slope (β<1 ⇒ overconfident ⇒ Kelly overstakes); post-ECE is in-sample Platt")
    println("="^128)
    @printf(" %-2s %-30s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s\n",
            "G", "Model", "LogLoss", "BF LL", "ECE", "BF ECE", "MCE", "α", "β", "post-ECE")
    println("-"^128)
    for r in eachrow(df)
        @printf(" %-2d %-30s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %+8.4f | %8.4f | %8.4f\n",
                r.generation, r.model, r.logloss, r.betfair_logloss, r.ece, r.betfair_ece,
                r.mce, r.glm_alpha, r.glm_beta, r.glm_ece_calibrated)
    end

    println("\n", "="^128)
    println(" C · BETFAIR PORTFOLIO — one BookSpec, one PolicySpec, one price source, recomputed here")
    println("="^128)
    @printf(" %-2s %-30s | %6s | %10s | %9s | %9s | %9s | %8s | %8s\n",
            "G", "Model", "Bets", "Return %", "Flat ROI", "1X2 ROI", "Max DD", "Sharpe", "Win %")
    println("-"^128)
    for r in eachrow(sort(df, :total_return_pct; rev = true))
        @printf(" %-2d %-30s | %6d | %+10.2f | %+9.2f | %+9.2f | %9.2f | %8.3f | %8.2f\n",
                r.generation, r.model, r.n_bets, r.total_return_pct, r.flat_roi_pct,
                r.roi_1x2_pct, r.max_drawdown_pct, r.sharpe_ann, 100 * r.win_rate)
    end
    println("="^128)
end

# ==============================================================================
# 8. Markdown report
# ==============================================================================

function cmp_md_table(header::Vector{String}, rows::Vector{Vector{String}})
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", join(fill(" :--- ", length(header)), "|"), "|")
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
    return String(take!(io))
end

f4(x) = ismissing(x) || !isfinite(x) ? "—" : @sprintf("%.4f", x)
f2(x) = ismissing(x) || !isfinite(x) ? "—" : @sprintf("%.2f", x)
fp(x) = ismissing(x) || !isfinite(x) ? "—" : (x < 1e-4 ? @sprintf("%.1e", x) : @sprintf("%.4f", x))

function cmp_write_report(df::DataFrame, persisted::DataFrame, book_spec, policy_spec,
                          bf_odds::DataFrame, ds::Data.DataStore)
    best_var = df[argmin(abs.(df.rqr_var .- 1.0)), :]
    worst_var = df[argmax(abs.(df.rqr_var .- 1.0)), :]
    best_return = df[argmax(df.total_return_pct), :]
    best_logloss = df[argmin(df.logloss), :]
    best_ece = df[argmin(df.ece), :]

    io = IOBuffer()
    println(io, "# Scottish Lower — Unified Cross-Paradigm Report")
    println(io)
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            " by `experiments/scottish_lower/compare_scottish_experiments.jl`.")
    println(io)
    println(io, "Four model generations, champion and control each, on three axes: the ",
            "randomized quantile residuals of the goals marginal, the GLM calibration shift ",
            "against the Betfair close, and fractional-Kelly portfolio performance.")
    println(io)

    println(io, "## 0. What is and is not comparable here")
    println(io)
    println(io, "Dimension C is **recomputed**, not read from `portfolio_runs`. Experiments 01 ",
            "and 03 priced their persisted portfolios off `ds.odds` — the bookmaker close, ",
            "overround intact — while 05 and 06 priced off the Betfair exchange close. Those ",
            "two sources reach opposite conclusions from identical posteriors (see ",
            "`03_joint_gamma_poisson/NOTES.md` §2), so ranking the persisted rows against each ",
            "other would compare price sources and report the result as a model comparison.")
    println(io)
    println(io, "Every row below was therefore re-simulated under one recipe:")
    println(io)
    println(io, "- **BookSpec** — 1X2, Over/Under 2.5, BTTS; `DeArb` pricing; ",
            "`KellyLogUtility`; `FractionalKelly(0.30)`; 2% per-bet commission; 0.99 budget.")
    println(io, "- **PolicySpec** — `FlatTrust(1.0)`, `SlateDrawdown(23.0)`, `FixedCap(0.20)`, ",
            "`DailySlate()`.")
    println(io, "- **Prices** — Betfair exchange close, time-weighted over [−20 min, kickoff]: ",
            nrow(bf_odds), " rows across ", length(unique(bf_odds.match_id)), " matches.")
    println(io)
    println(io, "The RQR draw is seeded (`CMP_SEED = ", CMP_SEED, "`) so the moments and ",
            "normality p-values reproduce exactly.")
    println(io)
    cmp_write_fold_caveat(io, df)

    println(io, "## 1. The bench")
    println(io)
    print(io, cmp_md_table(
        ["Gen", "Paradigm", "Model", "Role", "Likelihood", "Experiment", "Folds", "OOS", "R̂", "Div"],
        [[string(r.generation), r.paradigm, "`$(r.model)`", r.role, r.likelihood,
          "`$(r.experiment)`", string(r.n_folds), string(r.n_oos_matches),
          f4(r.max_rhat), string(r.n_divergent)] for r in eachrow(df)]))
    println(io)

    println(io, "## 2. Dimension A — randomized quantile residuals")
    println(io)
    println(io, "Dunn-Smyth residuals of the home and away goal counts, pooled. For each ",
            "fixture `u ~ Uniform(F(y−1), F(y))` and `r = Φ⁻¹(u)`, where `F` is the ",
            "**posterior predictive** CDF `(1/S) Σ_s F(·|θ_s)` — averaging the CDFs, not the ",
            "parameters. That distinction matters: a Poisson model with a wide posterior on λ ",
            "already has predictive variance above its mean, and collapsing to `F(·|θ̄)` would ",
            "charge the likelihood for dispersion the model handles.")
    println(io)
    println(io, "Under correct specification the residuals are exactly standard normal. ",
            "**Variance > 1 is unmodelled overdispersion.**")
    println(io)
    print(io, cmp_md_table(
        ["Gen", "Model", "Likelihood", "N", "Mean", "Var", "Var (H)", "Var (A)", "Skew",
         "Ex. kurt", "KS p", "AD p"],
        [[string(r.generation), "`$(r.model)`", r.likelihood, string(r.rqr_n),
          f4(r.rqr_mean), f4(r.rqr_var), f4(r.rqr_var_home), f4(r.rqr_var_away),
          f4(r.rqr_skewness), f4(r.rqr_excess_kurtosis),
          fp(r.rqr_ks_pvalue), fp(r.rqr_ad_pvalue)] for r in eachrow(df)]))
    println(io)
    println(io, "Closest to unit variance: **`", best_var.model, "`** (Gen ",
            best_var.generation, ", var ", f4(best_var.rqr_var), "). Furthest: **`",
            worst_var.model, "`** (Gen ", worst_var.generation, ", var ",
            f4(worst_var.rqr_var), ").")
    println(io)
    cmp_write_dispersion_verdict(io, df)

    println(io, "## 3. Dimension B — GLM calibration edge")
    println(io)
    println(io, "Scored against the Betfair close over 1X2, O/U 2.5 and BTTS. `α` and `β` come ",
            "from the logistic recalibration `y ~ Bernoulli(logistic(α + β·logit(p̂)))` — the ",
            "Platt shift a Layer-2 calibrator applies. `β < 1` means the model's probabilities ",
            "are too extreme; fractional Kelly is near-linear in the edge `p − 1/o`, so an ",
            "overconfident posterior overstakes in direct proportion.")
    println(io)
    println(io, "`post-ECE` is **in-sample** — the GLM saw these outcomes. Read it as how much ",
            "of the miscalibration has a shape two parameters can absorb, not as an ",
            "out-of-sample claim about a deployed calibrator.")
    println(io)
    print(io, cmp_md_table(
        ["Gen", "Model", "LogLoss", "BF LogLoss", "Δ", "Brier", "RPS", "ECE", "BF ECE",
         "MCE", "α", "β", "post-ECE", "N"],
        [[string(r.generation), "`$(r.model)`", f4(r.logloss), f4(r.betfair_logloss),
          f4(r.delta_logloss), f4(r.brier), f4(r.rps), f4(r.ece), f4(r.betfair_ece),
          f4(r.mce), f4(r.glm_alpha), f4(r.glm_beta), f4(r.glm_ece_calibrated),
          string(r.n_scored)] for r in eachrow(df)]))
    println(io)
    println(io, "Best log loss: **`", best_logloss.model, "`** (", f4(best_logloss.logloss),
            ", Betfair ", f4(best_logloss.betfair_logloss), "). Best ECE: **`",
            best_ece.model, "`** (", f4(best_ece.ece), ").")
    println(io)
    println(io, "Ten-bin reliability curves for every model and for the closing line are in ",
            "`results/unified_reliability_curves.csv`.")
    println(io)

    println(io, "## 4. Dimension C — Betfair portfolio")
    println(io)
    print(io, cmp_md_table(
        ["Gen", "Model", "Bets", "Return %", "Flat ROI %", "1X2 ROI %", "Max DD %",
         "Sharpe (ann)", "Win rate %"],
        [[string(r.generation), "`$(r.model)`", string(r.n_bets), f2(r.total_return_pct),
          f2(r.flat_roi_pct), f2(r.roi_1x2_pct), f2(r.max_drawdown_pct),
          f4(r.sharpe_ann), f2(100 * r.win_rate)]
         for r in eachrow(sort(df, :total_return_pct; rev = true))]))
    println(io)
    println(io, "Best return: **`", best_return.model, "`** (Gen ", best_return.generation,
            ", ", f2(best_return.total_return_pct), "%, Sharpe ",
            f4(best_return.sharpe_ann), ").")
    println(io)

    if nrow(persisted) > 0
        println(io, "### 4.1 Persisted `portfolio_runs`, for provenance only")
        println(io)
        println(io, "These are the rows each experiment wrote at its own time, under its own ",
                "price source. They are **not** comparable across generations; the table above ",
                "is. A row tagged `unrecorded` predates the `odds_source` metadata convention ",
                "and was priced off `ds.odds`.")
        println(io)
        latest = unique(persisted, [:model, :odds_source])
        print(io, cmp_md_table(
            ["Model", "Price source", "Bets", "Return %", "Flat ROI %", "Max DD %", "Sharpe"],
            [[r.model, "`$(r.odds_source)`", string(Int(r.n_bets)),
              f2(Float64(r.total_return_pct)), f2(Float64(r.flat_roi_pct)),
              f2(Float64(r.max_drawdown_pct)), f4(Float64(r.sharpe_ann))]
             for r in eachrow(sort(latest, [:model, :odds_source]))]))
        println(io)
    end

    println(io, "## 5. Reading the three axes together")
    println(io)
    cmp_write_synthesis(io, df)

    println(io, "## 6. Reproducing this")
    println(io)
    println(io, "```bash")
    println(io, "julia --project -t 16 experiments/scottish_lower/compare_scottish_experiments.jl")
    println(io, "```")
    println(io)
    println(io, "No MCMC is launched; every fit is loaded from PostgreSQL. Artefacts:")
    println(io)
    println(io, "- `results/unified_paradigm_comparison.csv` — one row per model, all three axes")
    println(io, "- `results/unified_reliability_curves.csv` — ten-bin model and Betfair curves")
    println(io, "- `results/unified_rqr_residuals.csv` — every residual, for plotting")
    nrow(persisted) > 0 &&
        println(io, "- `results/unified_persisted_portfolios.csv` — the historical rows")

    open(CMP_REPORT_PATH, "w") do handle
        write(handle, String(take!(io)))
    end
    return CMP_REPORT_PATH
end

"""
    cmp_write_fold_caveat(io, df)

Say so, in the report, when the bench is not evaluated over one common fixture set.

Runs extended into a later season (`r23`, `r37`) carry more folds and more out-of-sample
fixtures than the ones that were not. Every table here would still render, and the extra
fixtures would quietly widen one row's evaluation window relative to its neighbours' — a
difference that shows up only as a larger `N`. It is small, it is not a defect, and it is
exactly the kind of thing a reader should be told rather than left to notice.
"""
function cmp_write_fold_caveat(io::IO, df::DataFrame)
    folds = unique(df.n_folds)
    length(folds) == 1 && return nothing

    baseline = minimum(df.n_folds)
    extended = df[df.n_folds .> baseline, :]
    println(io, "> **The bench is not fold-uniform.** ",
            join(["`" * r.model * "` (" * string(r.n_folds) * " folds, " *
                  string(r.n_oos_matches) * " OOS fixtures)" for r in eachrow(extended)], ", "),
            " ", nrow(extended) == 1 ? "has" : "have",
            " been extended into a later season, against ", baseline,
            " folds elsewhere. That row is therefore scored over a slightly wider window; ",
            "the `N` column in §2 and `OOS` in §1 show it. The difference is small but it is ",
            "not nothing, so the row is not exactly comparable with the rest.")
    println(io)
    return nothing
end

"Does the likelihood upgrade actually move the residual variance toward 1?"
function cmp_write_dispersion_verdict(io::IO, df::DataFrame)
    println(io, "### 2.1 Did the likelihood upgrades resolve the overdispersion?")
    println(io)
    by_gen = combine(groupby(df, [:generation, :paradigm]),
                     :rqr_var => mean => :mean_var,
                     :rqr_ad_pvalue => maximum => :best_ad_p)
    sort!(by_gen, :generation)
    print(io, cmp_md_table(
        ["Gen", "Paradigm", "Mean RQR variance", "|var − 1|", "Best AD p"],
        [[string(r.generation), r.paradigm, f4(r.mean_var), f4(abs(r.mean_var - 1)),
          fp(r.best_ad_p)] for r in eachrow(by_gen)]))
    println(io)

    poisson_rows = df[df.likelihood .== "Poisson", :]
    negbin_rows = df[df.likelihood .== "NegBin", :]
    if nrow(poisson_rows) > 0 && nrow(negbin_rows) > 0
        pv = mean(poisson_rows.rqr_var)
        nv = mean(negbin_rows.rqr_var)
        direction = abs(nv - 1) < abs(pv - 1) ? "toward" : "away from"
        println(io, "Pure Poisson averages a residual variance of ", f4(pv),
                "; the negative binomial averages ", f4(nv), " — a move **", direction,
                "** unit variance of ", f4(abs(abs(pv - 1) - abs(nv - 1))), ".")
        println(io)
    end
    return nothing
end

function cmp_write_synthesis(io::IO, df::DataFrame)
    ranked_rqr = sort(df, :rqr_var; by = v -> abs(v - 1))
    ranked_ll = sort(df, :logloss)
    ranked_return = sort(df, :total_return_pct; rev = true)

    print(io, cmp_md_table(
        ["Rank", "Best dispersion (|var−1|)", "Best log loss", "Best return"],
        [[string(i), "`$(ranked_rqr.model[i])`", "`$(ranked_ll.model[i])`",
          "`$(ranked_return.model[i])`"] for i in 1:nrow(df)]))
    println(io)

    agree_top = ranked_rqr.model[1] == ranked_return.model[1]
    println(io, agree_top ?
        "The dispersion winner and the money winner are the same model." :
        "**The dispersion winner and the money winner are different models** — `" *
        ranked_rqr.model[1] * "` has the best-specified count distribution, `" *
        ranked_return.model[1] * "` makes the most money. A correctly specified " *
        "likelihood is not the same claim as an exploitable edge at the closing line.")
    println(io)

    correlation_ll_return = cor(df.logloss, df.total_return_pct)
    correlation_var_return = cor(abs.(df.rqr_var .- 1), df.total_return_pct)
    println(io, "Across these ", nrow(df), " models, log loss correlates with total return at ",
            f4(correlation_ll_return), " and |RQR variance − 1| at ",
            f4(correlation_var_return), ". With ", nrow(df),
            " points neither is an estimate to lean on; they are reported so the table is ",
            "not read as if it established one.")
    println(io)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cmp_compare_scottish_experiments()
end
