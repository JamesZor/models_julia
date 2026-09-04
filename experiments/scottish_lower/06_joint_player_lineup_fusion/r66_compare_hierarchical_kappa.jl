# ==============================================================================
# r66 — Hierarchical team-kappa vs shared-kappa: proper scores, GLM edge, and
#       the finishing-factor posterior
# ==============================================================================
#
# WHAT THIS ANSWERS. The 40-fold production grid (r65) produced two hierarchical
# candidates. Each has a shared-κ control that differs from it in EXACTLY ONE
# component, so every number below is a paired contrast, never a leaderboard:
#
#     m05_joint_production_wealth  →  m05_hierarchical_kappa
#     m12_joint_hybrid_synergy     →  m12_hierarchical_kappa
#
# FOUR QUESTIONS, IN THE ORDER THEY HAVE TO BE ASKED.
#
#   §6  Do the hierarchical fits price the held-out season better? LogLoss, CRPS,
#       Brier, RPS, ECE and MCE against the Betfair TWA closing line, over the
#       same 710 fixtures and the same 2,899 scored observations.
#
#   §7  Is any difference bigger than the noise? A proper-score mean over 2,899
#       observations is NOT 2,899 independent numbers — the eleven selections on
#       one fixture share one scoreline. The paired LogLoss difference is therefore
#       tested with the MATCH as the unit, which is the only clustering under which
#       "significant" means anything here.
#
#   §8  Does the model carry information the closing line does not? The GLM edge
#       regression logit P(win) = β₀ + β_mkt·p_fair + β_edge·(p_model − p_fair).
#       β_edge > 0 says the model's disagreement with the market is informative.
#       Standard errors are reported BOTH naively and clustered by match; the naive
#       ones are printed only so the size of the dependence correction is visible.
#
#   §9  Did σ_κ find anything? The whole hierarchy is one scale parameter. If its
#       posterior sits on its prior and no team's δ_κ separates from zero, then the
#       component shrank to the common mean and §6–§8 are measuring nothing but the
#       cost of carrying n_teams + 1 extra parameters.
#
# HOW TO RUN
#   julia --project -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r66_compare_hierarchical_kappa.jl
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and shared experiment state
# ==============================================================================
using BayesianFootball
using CSV
using DataFrames
using Dates
using Distributions
using GLM
using LinearAlgebra
using Printf
using Statistics
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# Brings `ds`, `db`, both candidate sets, the run manifest, the Betfair closing-line
# frame, and the §0 artefact compatibility shim without which the two shared-κ controls
# — fitted before `HierarchicalKappa` existed — cannot be deserialized at all.
include(joinpath(@__DIR__, "l66_hierarchical_kappa_eval_loader.jl"))

const R66_EVAL = BayesianFootball.Evaluation
const R66_PG = BayesianFootball.Models.PreGame
const R66_FEATURES = BayesianFootball.Features
const R66_INF = BayesianFootball.Training.Inference

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
#
# The run manifest, the fold and fixture counts and the output directory all come from
# `l66`; only what is specific to scoring is set here.
const R66_NAMES = L66_NAMES
const R66_ARMS = L66_ARMS
const R66_HIERARCHICAL = [L66_CANDIDATE_NAME[arm] for arm in L66_ARMS]
const R66_EXPECTED_FOLDS = L66_EXPECTED_FOLDS
const R66_EXPECTED_OOS = L66_EXPECTED_OOS
const R66_OUTPUT_DIR = L66_OUTPUT_DIR

const R66_HPDI_PROB = 0.90

# The temporal splits. `:all` is the headline; the seasons are what says whether a
# difference is a stable property or one season's luck.
const R66_SPLITS = [:all, Symbol("24/25"), Symbol("25/26")]

# %%
# ==============================================================================
# 3. Proper scoring
# ==============================================================================
r66_metrics() = R66_EVAL.AbstractScoringRule[
    R66_EVAL.LogLoss(), R66_EVAL.CRPS(), R66_EVAL.PredictionScore()]

"""
    r66_context(latents, odds, matches) -> EvaluationContext

Both the odds frame AND the match frame are filtered for a temporal split. Filtering
only the odds would leave CRPS — which walks the latent panel and reads outcomes from
`matches` — scoring the whole sample under a season's label.
"""
r66_context(latents, odds, matches) =
    build_evaluation_context(latents, odds, matches, r66_metrics(); threaded = true)

function r66_curve_frame(model, split, source, curve)
    n = length(curve.counts)
    return DataFrame(
        model = fill(String(model), n),
        split = fill(String(split), n),
        source = fill(String(source), n),
        bin = collect(1:n),
        lower = curve.edges[1:end-1],
        upper = curve.edges[2:end],
        n = curve.counts,
        predicted = curve.mean_predicted,
        observed = curve.observed,
    )
end

# %%
# ==============================================================================
# 4. Statistics — clustered inference
# ==============================================================================
#
# EVERY inferential number in this file clusters on `match_id`. 2,899 scored
# observations come from 710 fixtures; a home win, a draw and an away price on one
# fixture are three views of one scoreline. Treating them as independent divides the
# standard error by roughly √4 and turns "no effect" into "p < 0.01".

"""
    r66_cluster_ttest(values, clusters) -> (mean, se, t, p, n_clusters)

A one-sample t-test on the CLUSTER means. Equivalent to a cluster-robust test of
`mean(values) == 0` and easier to state: the unit of evidence is the fixture.
"""
function r66_cluster_ttest(x::AbstractVector{<:Real}, clusters::AbstractVector)
    by_cluster = Dict{Any,Vector{Float64}}()
    for (v, c) in zip(x, clusters)
        push!(get!(by_cluster, c, Float64[]), Float64(v))
    end
    means = [mean(v) for v in Base.values(by_cluster)]
    g = length(means)
    g < 2 && return (mean = isempty(x) ? NaN : mean(x), se = NaN, t = NaN,
                     p = NaN, n_clusters = g)
    m = mean(means)
    se = std(means) / sqrt(g)
    t = m / se
    p = 2 * ccdf(TDist(g - 1), abs(t))
    return (mean = m, se = se, t = t, p = p, n_clusters = g)
end

"""
    r66_logit_clustered(X, y, clusters) -> (β, se_naive, se_cluster, ...)

Logistic regression with a CR1 cluster-robust sandwich covariance.

    V = B · M · B,   B = (X'WX)⁻¹,   W = diag(p(1−p)),
    M = Σ_g (Σ_{i∈g} xᵢuᵢ)(Σ_{i∈g} xᵢuᵢ)',   u = y − p,

scaled by the usual finite-cluster correction `G/(G−1) · (n−1)/(n−k)`. The point
estimate is GLM.jl's; only the covariance is replaced, so β is exactly what a reader
reproducing this with `glm(...)` will get.
"""
function r66_logit_clustered(fitted, X::AbstractMatrix, y::AbstractVector,
                             clusters::AbstractVector)
    β = coef(fitted)
    se_naive = stderror(fitted)
    η = X * β
    p = 1.0 ./ (1.0 .+ exp.(.-η))
    u = y .- p
    w = p .* (1.0 .- p)

    n, k = size(X)
    bread = inv(Symmetric(X' * (w .* X)))

    ids = unique(clusters)
    G = length(ids)
    index = Dict(id => i for (i, id) in enumerate(ids))
    scores = zeros(Float64, G, k)
    @inbounds for i in 1:n
        g = index[clusters[i]]
        for j in 1:k
            scores[g, j] += X[i, j] * u[i]
        end
    end
    meat = scores' * scores
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V = correction .* (bread * meat * bread)
    se_cluster = sqrt.(max.(diag(V), 0.0))

    z_naive = β ./ se_naive
    z_cluster = β ./ se_cluster
    return (; β, se_naive, se_cluster,
              z_naive, z_cluster,
              p_naive = 2 .* ccdf.(Normal(), abs.(z_naive)),
              p_cluster = 2 .* ccdf.(Normal(), abs.(z_cluster)),
              n_obs = n, n_clusters = G)
end

"""
    r66_glm_edge(frame, model, split, subset) -> NamedTuple

    logit P(win) = β₀ + β_mkt·p_fair_close + β_edge·(p_model − p_fair_close)

`β_edge` is the whole question. Under a market that already prices everything the
model knows, the model's disagreement is noise and β_edge = 0. A positive, clustered-
significant β_edge says the disagreement moves the outcome in its own direction.

`β_mkt` near its own well-calibrated value is a sanity anchor, not a finding.
"""
function r66_glm_edge(frame::AbstractDataFrame, model, split, subset)
    n = nrow(frame)
    blank = (model = String(model), split = String(split), subset = String(subset),
             n_obs = n, n_clusters = 0,
             beta_intercept = NaN, se_intercept = NaN,
             beta_mkt = NaN, se_mkt = NaN, z_mkt = NaN, p_mkt = NaN,
             beta_edge = NaN, se_edge_naive = NaN, se_edge = NaN,
             z_edge_naive = NaN, z_edge = NaN, p_edge_naive = NaN, p_edge = NaN,
             mean_spread = NaN, sd_spread = NaN, deviance = NaN, nulldeviance = NaN,
             pseudo_r2 = NaN)
    n < 30 && return blank

    fitted = glm(@formula(y ~ prob_fair + spread), frame, Binomial(), LogitLink())
    X = hcat(ones(n), frame.prob_fair, frame.spread)
    robust = r66_logit_clustered(fitted, X, Float64.(frame.y), frame.match_id)

    return (model = String(model), split = String(split), subset = String(subset),
            n_obs = n, n_clusters = robust.n_clusters,
            beta_intercept = robust.β[1], se_intercept = robust.se_cluster[1],
            beta_mkt = robust.β[2], se_mkt = robust.se_cluster[2],
            z_mkt = robust.z_cluster[2], p_mkt = robust.p_cluster[2],
            beta_edge = robust.β[3],
            se_edge_naive = robust.se_naive[3], se_edge = robust.se_cluster[3],
            z_edge_naive = robust.z_naive[3], z_edge = robust.z_cluster[3],
            p_edge_naive = robust.p_naive[3], p_edge = robust.p_cluster[3],
            mean_spread = mean(frame.spread), sd_spread = std(frame.spread),
            deviance = deviance(fitted), nulldeviance = nulldeviance(fitted),
            pseudo_r2 = 1 - deviance(fitted) / nulldeviance(fitted))
end

# %%
# ==============================================================================
# 5. Finishing-factor posterior
# ==============================================================================
"""
    r66_chain_n_teams(chain) -> Int

The team count a fold's chain carries, READ OFF THE CHAIN rather than assumed.

A hierarchical chain states it directly in `obs.κ_team_raw[1:n_teams]`; a shared-κ one
does not carry the block at all, so the team-indexed dynamics `dyn.raw_a[1:n_teams]` is
the fallback. Both are counted rather than taken from a FeatureSet because the number
that matters is the one the sampler actually used.
"""
function r66_chain_n_teams(chain)
    names_str = String.(names(chain))
    hier = count(c -> startswith(c, "obs.κ_team_raw["), names_str)
    hier > 0 && return hier
    dyn = count(c -> startswith(c, "dyn.raw_a["), names_str)
    dyn > 0 || error("chain carries neither obs.κ_team_raw nor dyn.raw_a; cannot size teams")
    return dyn
end

r66_draws(chain, sym) = vec(Array(chain[sym]))

"""
    r66_fold_kappa(name, model, fit) -> DataFrame

One row per fold: the league finishing factor, and — for a hierarchical fit — the
spread σ_κ and how many teams SEPARATED, meaning their 90% HPDI on δ_κ excludes zero.

The separation count is what decides the component. A σ_κ posterior that is merely
non-zero is compatible with the hierarchy fitting Poisson noise; a team whose interval
clears zero is a claim about that team.
"""
function r66_fold_kappa(name::AbstractString, model, fit)
    hierarchical = model.observation.kappa isa HierarchicalKappa
    rows = NamedTuple[]
    for fold in fit.folds
        chain = fold.chain
        n_teams = r66_chain_n_teams(chain)
        κ_global = exp.(r66_draws(chain, Symbol("obs.log_κ")))
        κ_lo, κ_hi = cb_hpdi(κ_global, R66_HPDI_PROB)
        ν = r66_draws(chain, Symbol("obs.ν"))

        σ_mean = NaN; σ_lo = NaN; σ_hi = NaN; p_sigma_gt = NaN
        n_separated = 0; δ_max = NaN; δ_min = NaN
        if hierarchical
            k = R66_PG.extract_kappa(chain, model.observation, n_teams;
                                     prob = R66_HPDI_PROB)
            σ = k.σ_κ
            σ_mean = mean(σ)
            σ_lo, σ_hi = cb_hpdi(σ, R66_HPDI_PROB)
            p_sigma_gt = mean(>(0.05), σ)
            n_separated = count(r -> r.δ_lo > 0.0 || r.δ_hi < 0.0, eachrow(k.summary))
            δ_max = maximum(k.summary.δ_mean)
            δ_min = minimum(k.summary.δ_mean)
        end

        push!(rows, (
            model = String(name), fold = fold.fold, n_teams = n_teams,
            n_draws = length(κ_global),
            kappa_global_mean = mean(κ_global), kappa_global_lo = κ_lo,
            kappa_global_hi = κ_hi,
            nu_mean = mean(ν),
            sigma_kappa_mean = σ_mean, sigma_kappa_lo = σ_lo, sigma_kappa_hi = σ_hi,
            p_sigma_gt_005 = p_sigma_gt,
            n_teams_separated = n_separated,
            delta_max = δ_max, delta_min = δ_min,
        ))
    end
    return DataFrame(rows)
end

"""
    r66_team_kappa(name, model, fold_index, boundary) -> DataFrame

The named per-team finishing table for one fold. Names come from the FeatureSet's
`team_map`, rebuilt for that boundary alone — a numbered row cannot be checked against
anything a reader knows about the league.
"""
function r66_team_kappa(name::AbstractString, model, fit, fold_index::Int, boundaries)
    feature_sets = R66_FEATURES.create_features(
        [boundaries[fold_index]], ds, model, l64_production_splitter)
    feature_set = first(feature_sets[1])
    n_teams = Int(feature_set.data[:n_teams])
    chain = fit.folds[fold_index].chain
    n_teams == r66_chain_n_teams(chain) || error(
        "$name fold $fold_index: FeatureSet has $n_teams teams, chain carries " *
        "$(r66_chain_n_teams(chain)); the boundary rebuild is not the fitted one")

    k = R66_PG.extract_kappa(chain, model.observation, n_teams;
                             team_map = feature_set.data[:team_map], prob = R66_HPDI_PROB)
    out = copy(k.summary)
    insertcols!(out, 1, :model => fill(String(name), nrow(out)))
    insertcols!(out, 2, :fold => fill(fold_index, nrow(out)))
    out.separated = (out.δ_lo .> 0.0) .| (out.δ_hi .< 0.0)
    return out
end

# %%
# ==============================================================================
# 6. Load the fits
# ==============================================================================
println("\n" * "="^118)
println(" EXPERIMENT 06 · HIERARCHICAL TEAM KAPPA · 40-FOLD OUT-OF-SAMPLE EVALUATION")
println("="^118)
println("  database  : ", db)
println("  threads   : ", Threads.nthreads())
println("  started   : ", Dates.now())

models_by_name = l66_models_by_name()

fits = Dict{String,Any}()
run_ids = Dict{String,UUID}()
for name in R66_NAMES
    fit, resolved = l66_load_fit(db, name)
    fits[name] = fit
    run_ids[name] = resolved
    @printf("  loaded %-30s run %s  R̂ %.4f  ESS %d/%d  div %d  %s\n",
            name, string(resolved), fit.diagnostics.max_rhat,
            fit.diagnostics.min_ess_bulk, fit.diagnostics.min_ess_tail,
            fit.diagnostics.n_divergent,
            fit.diagnostics.passed ? "PASS" : "FAIL")
end

# %%
# ==============================================================================
# 7. Fold-level convergence, straight from the experiment database
# ==============================================================================
conn = R66_INF._db_connect(db)
fold_diag = try
    placeholders = join(["\$$(i)::uuid" for i in 1:length(R66_NAMES)], ", ")
    R66_INF._db_rows(conn, """
        SELECT r.name AS model, f.fold_idx AS fold, f.r_hat_max, f.ess_bulk_min,
               f.ess_tail_min, f.divergences, f.converged, f.runtime_seconds
        FROM fold_results f JOIN runs r ON r.run_id = f.run_id
        WHERE f.run_id IN ($placeholders)
        ORDER BY r.name, f.fold_idx;
    """, Tuple(string(run_ids[n]) for n in R66_NAMES))
finally
    close(conn)
end
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_fold_diagnostics.csv"), fold_diag)

convergence = combine(groupby(fold_diag, :model),
    :r_hat_max => maximum => :max_rhat,
    :ess_bulk_min => minimum => :min_ess_bulk,
    :ess_tail_min => minimum => :min_ess_tail,
    :divergences => sum => :divergences,
    :runtime_seconds => sum => :wallclock_seconds,
    nrow => :n_folds)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_convergence_summary.csv"), convergence)

println("\n" * "-"^118)
println(" CONVERGENCE (40 folds × 4 chains × 800 retained draws per model)")
println("-"^118)
@printf(" %-30s | %6s | %8s | %8s | %8s | %5s | %10s\n",
        "Model", "Folds", "max R̂", "ESS bulk", "ESS tail", "Div", "Hours")
for row in eachrow(sort(convergence, :model))
    @printf(" %-30s | %6d | %8.5f | %8d | %8d | %5d | %10.2f\n",
            row.model, row.n_folds, row.max_rhat, row.min_ess_bulk,
            row.min_ess_tail, row.divergences, row.wallclock_seconds / 3600)
end

# %%
# ==============================================================================
# 8. Proper scoring rules over the three temporal splits
# ==============================================================================
bf_odds = l66_betfair_closing_odds(ds)
@printf("\n  Betfair TWA closes: %d rows across %d matches\n",
        nrow(bf_odds), length(unique(bf_odds.match_id)))

season_of = l66_season_of(ds)
date_of = l66_date_of(ds)

split_matches = Dict{Symbol,DataFrame}()
split_odds = Dict{Symbol,DataFrame}()
for split in R66_SPLITS
    if split === :all
        split_matches[split] = ds.matches
        split_odds[split] = bf_odds
    else
        season = String(split)
        ids = Set(Int(r.match_id) for r in eachrow(ds.matches) if String(r.season) == season)
        split_matches[split] = filter(:match_id => id -> Int(id) in ids, ds.matches)
        split_odds[split] = filter(:match_id => id -> Int(id) in ids, bf_odds)
    end
end

score_rows = NamedTuple[]
curve_frames = DataFrame[]
glm_rows = NamedTuple[]
row_frames = DataFrame[]

market_lookup = Dict((Int(r.match_id), r.selection) =>
                     (l66_family(r.market_name, r.market_line), r.odds_close)
                     for r in eachrow(bf_odds))

for name in R66_NAMES
    fit = fits[name]
    latents = fit_latents(fit)
    for split in R66_SPLITS
        ctx = r66_context(latents, split_odds[split], split_matches[split])
        alignment = verify_alignment(ctx)
        alignment.ok || error("$name/$split alignment failed: $alignment")

        prediction = evaluate_predictions(ctx; n_bins = 10)
        crps = R66_EVAL.compute_metric(R66_EVAL.CRPS(), ctx)

        push!(score_rows, (
            model = name, split = String(split),
            n_obs = prediction.model.n_obs,
            n_matches = alignment.n_priced_fixtures,
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
            crps = crps.all.mean,
            crps_home = crps.home.mean,
            crps_away = crps.away.mean,
        ))

        if split === :all
            push!(curve_frames,
                  r66_curve_frame(name, split, :model, calibration_curve(ctx; n_bins = 10, source = :model)))
            push!(curve_frames,
                  r66_curve_frame(name, split, :betfair, calibration_curve(ctx; n_bins = 10, source = :market)))
        end

        # The per-observation frame every inferential number below is built from.
        rows = R66_EVAL.evaluation_rows(ctx)
        frame = DataFrame(
            model = fill(name, length(rows)),
            match_id = [r.match_id for r in rows],
            selection = [r.selection for r in rows],
            prob_model = [r.model_prob for r in rows],
            prob_fair = [r.market_prob for r in rows],
            y = [Float64(r.outcome) for r in rows],
        )
        frame.spread = frame.prob_model .- frame.prob_fair
        frame.family = [get(market_lookup, (m, s), ("unknown", NaN))[1]
                        for (m, s) in zip(frame.match_id, frame.selection)]
        frame.odds_close = [get(market_lookup, (m, s), ("unknown", NaN))[2]
                            for (m, s) in zip(frame.match_id, frame.selection)]
        frame.season = [get(season_of, m, "unknown") for m in frame.match_id]
        frame.match_date = [get(date_of, m, Date(1970, 1, 1)) for m in frame.match_id]
        frame.logloss_model = -(frame.y .* log.(frame.prob_model) .+
                                (1 .- frame.y) .* log.(1 .- frame.prob_model))
        frame.logloss_market = -(frame.y .* log.(frame.prob_fair) .+
                                 (1 .- frame.y) .* log.(1 .- frame.prob_fair))
        frame.split = fill(String(split), nrow(frame))

        push!(glm_rows, r66_glm_edge(frame, name, split, "all"))
        for fam in sort(unique(frame.family))
            sub = filter(:family => ==(fam), frame)
            push!(glm_rows, r66_glm_edge(sub, name, split, fam))
        end
        split === :all && push!(row_frames, frame)
    end
    println("  scored $name over $(length(R66_SPLITS)) splits")
end

scores = DataFrame(score_rows)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_proper_scores.csv"), scores)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_calibration_curves.csv"), vcat(curve_frames...))
glm_table = DataFrame(glm_rows)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_glm_edge.csv"), glm_table)

observations = vcat(row_frames...)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_scored_observations.csv"), observations)

println("\n" * "="^160)
println(" PROPER SCORES vs BETFAIR TWA CLOSE — all 710 held-out fixtures")
println("="^160)
@printf(" %-30s | %-6s | %6s | %8s | %8s | %8s | %8s | %8s | %8s | %8s\n",
        "Model", "Split", "N", "LogLoss", "ΔBF LL", "Brier", "RPS", "ECE", "MCE", "CRPS")
println("-"^160)
for split in R66_SPLITS
    sub = sort(filter(:split => ==(String(split)), scores), :logloss)
    for row in eachrow(sub)
        @printf(" %-30s | %-6s | %6d | %8.5f | %+8.5f | %8.5f | %8.5f | %8.5f | %8.5f | %8.5f\n",
                row.model, row.split, row.n_obs, row.logloss, row.delta_logloss,
                row.brier, row.rps, row.ece, row.mce, row.crps)
    end
    bf = first(sub)
    @printf(" %-30s | %-6s | %6d | %8.5f | %8s | %8.5f | %8.5f | %8.5f | %8.5f | %8s\n",
            "Betfair closing line", String(split), bf.n_obs, bf.betfair_logloss,
            "-", bf.betfair_brier, bf.betfair_rps, bf.betfair_ece, bf.betfair_mce, "-")
    println("-"^160)
end

# %%
# ==============================================================================
# 9. Paired contrasts — is the hierarchical difference bigger than the noise?
# ==============================================================================
paired_rows = NamedTuple[]
for arm in R66_ARMS
    control = L66_CONTROL_NAME[arm]
    candidate = L66_CANDIDATE_NAME[arm]
    left = filter(:model => ==(control), observations)
    right = filter(:model => ==(candidate), observations)
    sort!(left, [:match_id, :selection]); sort!(right, [:match_id, :selection])
    (left.match_id == right.match_id && left.selection == right.selection) || error(
        "$arm: the control and the candidate did not score the same observation set")

    for split in R66_SPLITS
        keep = split === :all ? trues(nrow(left)) : (left.season .== String(split))
        Δll = right.logloss_model[keep] .- left.logloss_model[keep]
        Δspread = abs.(right.spread[keep]) .- abs.(left.spread[keep])
        test = r66_cluster_ttest(Δll, left.match_id[keep])
        spread_test = r66_cluster_ttest(Δspread, left.match_id[keep])
        push!(paired_rows, (
            arm = arm, control = control, candidate = candidate, split = String(split),
            n_obs = count(keep), n_clusters = test.n_clusters,
            delta_logloss = test.mean, se = test.se, t = test.t, p_value = test.p,
            control_logloss = mean(left.logloss_model[keep]),
            candidate_logloss = mean(right.logloss_model[keep]),
            delta_abs_spread = spread_test.mean, p_abs_spread = spread_test.p,
            mean_abs_prob_shift = mean(abs.(right.prob_model[keep] .- left.prob_model[keep])),
            max_abs_prob_shift = maximum(abs.(right.prob_model[keep] .- left.prob_model[keep])),
            corr_prob = cor(right.prob_model[keep], left.prob_model[keep]),
        ))
    end
end
paired = DataFrame(paired_rows)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_paired_contrasts.csv"), paired)

println("\n" * "="^150)
println(" PAIRED LOGLOSS CONTRAST — hierarchical minus shared, clustered on fixture")
println("="^150)
@printf(" %-4s | %-6s | %6s | %6s | %10s | %10s | %11s | %9s | %8s | %9s\n",
        "Arm", "Split", "Nobs", "Ncl", "Control", "Hier", "Δ LogLoss", "SE", "t", "p")
println("-"^150)
for row in eachrow(paired)
    @printf(" %-4s | %-6s | %6d | %6d | %10.5f | %10.5f | %+11.6f | %9.6f | %+8.3f | %9.4f\n",
            row.arm, row.split, row.n_obs, row.n_clusters, row.control_logloss,
            row.candidate_logloss, row.delta_logloss, row.se, row.t, row.p_value)
end
println("="^150)
println(" A NEGATIVE Δ favours the hierarchical model. p is two-sided with the fixture as the unit.")

# %%
# ==============================================================================
# 10. GLM edge
# ==============================================================================
println("\n" * "="^158)
println(" GLM EDGE:  logit P(win) = β₀ + β_mkt·p_fair + β_edge·(p_model − p_fair)")
println("="^158)
@printf(" %-30s | %-6s | %-7s | %6s | %9s | %9s | %8s | %8s | %9s\n",
        "Model", "Split", "Subset", "N", "β_mkt", "β_edge", "SE(cl)", "z(cl)", "p(cl)")
println("-"^158)
for row in eachrow(filter(r -> r.subset in ("all", "1X2", "OU2.5"), glm_table))
    @printf(" %-30s | %-6s | %-7s | %6d | %9.4f | %+9.4f | %8.4f | %+8.3f | %9.4f\n",
            row.model, row.split, row.subset, row.n_obs, row.beta_mkt, row.beta_edge,
            row.se_edge, row.z_edge, row.p_edge)
end
println("="^158)

# %%
# ==============================================================================
# 11. Finishing-factor posterior
# ==============================================================================
boundaries = Data.create_id_boundaries(ds, l64_production_splitter)
length(boundaries) == R66_EXPECTED_FOLDS || error(
    "splitter produced $(length(boundaries)) boundaries; expected $R66_EXPECTED_FOLDS")

kappa_frames = DataFrame[]
for name in R66_NAMES
    push!(kappa_frames, r66_fold_kappa(name, models_by_name[name], fits[name]))
end
kappa_folds = vcat(kappa_frames...)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_kappa_by_fold.csv"), kappa_folds)

team_frames = DataFrame[]
for name in R66_HIERARCHICAL
    push!(team_frames,
          r66_team_kappa(name, models_by_name[name], fits[name], R66_EXPECTED_FOLDS, boundaries))
end
team_kappa = vcat(team_frames...)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_team_kappa_final_fold.csv"), team_kappa)

kappa_summary = combine(groupby(kappa_folds, :model),
    :kappa_global_mean => mean => :kappa_global,
    :kappa_global_lo => mean => :kappa_global_lo,
    :kappa_global_hi => mean => :kappa_global_hi,
    :nu_mean => mean => :nu,
    :sigma_kappa_mean => mean => :sigma_kappa,
    :sigma_kappa_lo => mean => :sigma_kappa_lo,
    :sigma_kappa_hi => mean => :sigma_kappa_hi,
    :p_sigma_gt_005 => mean => :p_sigma_gt_005,
    :n_teams_separated => sum => :folds_team_separations,
    :n_teams_separated => maximum => :max_separated_in_a_fold,
    :delta_max => maximum => :delta_max,
    :delta_min => minimum => :delta_min)
CSV.write(joinpath(R66_OUTPUT_DIR, "r66_kappa_summary.csv"), kappa_summary)

println("\n" * "="^150)
println(" FINISHING FACTOR — posterior over 40 folds (fold means of per-fold posterior summaries)")
println("="^150)
@printf(" %-30s | %-24s | %-24s | %8s | %7s | %9s\n",
        "Model", "κ_league (90% HPDI)", "σ_κ (90% HPDI)", "P(σ>.05)", "Sep.", "ν")
println("-"^150)
for row in eachrow(sort(kappa_summary, :model))
    σ_text = isnan(row.sigma_kappa) ? "— (shared κ)" :
        @sprintf("%.4f [%.4f, %.4f]", row.sigma_kappa, row.sigma_kappa_lo, row.sigma_kappa_hi)
    @printf(" %-30s | %.4f [%.4f, %.4f] | %-24s | %8s | %7d | %9.4f\n",
            row.model, row.kappa_global, row.kappa_global_lo, row.kappa_global_hi,
            σ_text,
            isnan(row.p_sigma_gt_005) ? "—" : @sprintf("%.3f", row.p_sigma_gt_005),
            row.folds_team_separations, row.nu)
end
println("="^150)
println(" `Sep.` counts (fold, team) pairs whose 90% HPDI on δ_κ excludes zero, over all 40 folds.")

println("\n TEAM FINISHING FACTORS — final fold (fullest history)")
for name in R66_HIERARCHICAL
    sub = filter(:model => ==(name), team_kappa)
    println("\n  $name  ($(nrow(sub)) teams, $(count(sub.separated)) separated at 90%)")
    @printf("   %-28s | %9s | %-20s | %8s | %8s\n",
            "Team", "δ_κ", "90% HPDI", "κ_team", "P(δ>0)")
    for row in eachrow(first(sub, 5))
        @printf("   %-28s | %+9.4f | [%+.4f, %+.4f] | %8.4f | %8.3f\n",
                row.team, row.δ_mean, row.δ_lo, row.δ_hi, row.κ_mean, row.p_over)
    end
    println("   ...")
    for row in eachrow(last(sub, 5))
        @printf("   %-28s | %+9.4f | [%+.4f, %+.4f] | %8.4f | %8.3f\n",
                row.team, row.δ_mean, row.δ_lo, row.δ_hi, row.κ_mean, row.p_over)
    end
end

println("\nCSVs written under $R66_OUTPUT_DIR")
println("Finished: ", Dates.now())
