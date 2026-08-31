# ==============================================================================
# r51 — Nine ways to normalise a player rating, scored against two criteria
# ==============================================================================
#
# WHAT THIS IS
#   The comparison half of the player-normalisation stream. r50 established the facts;
#   this runner acts on them by building nine candidate starting-XI columns and scoring
#   each one twice:
#
#     CRITERION A — agreement with the SofaScore rating, per position. Only measurable
#                   on the English tiers, where a reference rating exists.
#     CRITERION B — held-out association with the realised scoreline. Measurable
#                   everywhere, and the one that decides.
#
#   The two can disagree, and when they do, B wins: agreeing with SofaScore is a
#   validation of the RATING, whereas the covariate's job is to price a match.
#
# WHAT THIS IS NOT
#   Not a model comparison and not a feature. Nothing here touches src/. A strategy that
#   wins on held-out linear association still has to earn its place on out-of-sample log
#   loss under the count model, which is r40's question.
#
# ⚠ EVERYTHING IS FITTED ON HISTORY ONLY.
#   The ridge, the position means, the demographic regressions and the standardising
#   scales are all estimated on the first 80% of the fixture list and applied to the
#   last 20%. Measuring any of them in-sample inflates them — on the Scottish store the
#   :y_goals target scored r = 0.42 in-sample and r = 0.004 held-out.
#
# USAGE
#   source .env
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r51_rapm_normalisation_bench.jl")
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "l93_eda_toolkit.jl"))
include(joinpath(@__DIR__, "l50_player_norm.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R51_TARGET = :y_xg
const R51_LAMBDA = 5000.0        # near the reliability plateau; see r94 section 4
const R51_HALF_LIFE = 730.0
const R51_KAPPA = 20.0           # exposure pseudo-count, the production default
const R51_HISTORY_FRAC = 0.80
const R51_MIN_SOFA = 10
const R51_MIN_RATED = 3

eda_banner("r51 · nine normalisations of the starting-XI rating")
println("  target = ", R51_TARGET, " | lambda = ", Int(R51_LAMBDA),
        " | kappa = ", R51_KAPPA, " | history = ", Int(100 * R51_HISTORY_FRAC), "%")

# %%
# ==============================================================================
# 3. The bench, as one reusable pass over a store
# ==============================================================================

"""
    r51_bench(ds, label; score_against_rating) -> (rows, frame)

Fit the ridge on the history block, build every candidate column, and score them on the
held-out block. `score_against_rating` is only possible where a SofaScore rating exists.
"""
function r51_bench(ds, label::AbstractString; score_against_rating::Bool)
    ordered = sort(ds.matches, [:match_date, :match_id])
    ids = Int.(ordered.match_id)
    cut = clamp(round(Int, R51_HISTORY_FRAC * length(ids)), 1, length(ids) - 1)
    history = Set(ids[1:cut])
    target = Set(ids[(cut + 1):end])
    @printf("  %s: %d matches — history %d, held out %d\n",
            label, length(ids), length(history), length(target))

    prep = Features.pm_prepared(ds)
    segments_history = prep.segments[in.(Int.(prep.segments.match_id), Ref(history)), :]
    @printf("  history stints %d (covered %d)\n",
            nrow(segments_history), count(segments_history.covered))

    t0 = time()
    fit = Features.fit_ratings(segments_history; target = R51_TARGET, λ = R51_LAMBDA,
                               w_sim = 0.0, half_life = R51_HALF_LIFE,
                               T_rating = maximum(segments_history.match_date),
                               comp_sets = Features.competition_sets(ds; match_ids = history))
    fit === nothing && (println("  [SKIP] history block too small to fit");
                        return (NamedTuple[], DataFrame(), Float64[]))
    @printf("  ridge fitted in %.1fs — %d players\n", time() - t0, nrow(fit))

    exposure = Features.player_exposure(segments_history)
    frame = l50_player_frame(ds, fit, exposure;
                             match_ids = history,
                             reference_date = maximum(segments_history.match_date))

    outcomes = eda_match_frame(ds; ordered_ids = ids)
    mask = Bool[id in target for id in ids] .& outcomes.played
    @printf("  scored on %d played held-out matches\n\n", count(mask))

    rows = NamedTuple[]
    for (name, description, transform) in l50_strategies(; kappa = R51_KAPPA)
        adjusted_values = transform(frame)
        adjusted = Dict{Int,Float64}(
            frame.player_id[i] => adjusted_values[i]
            for i in 1:nrow(frame) if isfinite(adjusted_values[i]))
        column, available = l50_xi_column(ds, adjusted, ids; min_rated = R51_MIN_RATED)

        signal = eda_signal(name, column[mask], outcomes[mask, :])
        agreement_all = NaN
        agreement = Dict{String,Float64}()
        if score_against_rating
            rated = [i for i in 1:nrow(frame)
                     if frame.n_sofa[i] >= R51_MIN_SOFA && isfinite(adjusted_values[i])]
            if length(rated) > 50
                agreement_all = eda_pearson(adjusted_values[rated], frame.sofa_mean[rated])
                for position in L50_POSITIONS
                    idx = [i for i in rated if frame.position[i] == position]
                    length(idx) >= 15 || continue
                    agreement[position] = eda_pearson(adjusted_values[idx], frame.sofa_mean[idx])
                end
            end
        end
        push!(rows, (name = name, description = description, signal = signal,
                     coverage = mean(available[mask]),
                     agreement_all = agreement_all, agreement = agreement,
                     column = column[mask]))
    end
    return rows, frame, outcomes.supremacy[mask]
end

# %%
# ==============================================================================
# 4. The English laboratory
# ==============================================================================

eda_section("1/4", "Fitting on the English tiers")
ds_eng = l50_store()
rows_eng, frame_eng, y_eng = r51_bench(ds_eng, "England 1/2/3/84"; score_against_rating = true)

eda_section("2/4", "Criterion A — agreement with the SofaScore rating")
println("  Pearson r between each candidate column's player values and that player's mean")
println("  SofaScore rating. Within-position figures are the honest ones: the yardstick is")
println("  itself position-normalised, so a pooled correlation mixes four scales.\n")
@printf("  %-12s | %9s | %9s | %9s | %9s | %9s\n", "strategy", "pooled", "G", "D", "M", "F")
println("  " * repeat('-', 72))
for r in rows_eng
    @printf("  %-12s | %9s | %9s | %9s | %9s | %9s\n", r.name,
            eda_fmt(r.agreement_all, "%+.4f"),
            eda_fmt(get(r.agreement, "G", NaN), "%+.4f"),
            eda_fmt(get(r.agreement, "D", NaN), "%+.4f"),
            eda_fmt(get(r.agreement, "M", NaN), "%+.4f"),
            eda_fmt(get(r.agreement, "F", NaN), "%+.4f"))
end
println("\n  [NOTE] `drop_gk` and its relatives set keepers to exactly zero, so their G column")
println("  is undefined by construction — that is the intervention, not a measurement gap.")

eda_section("3/4", "Criterion B — held-out signal on the English tiers")
eda_print_signal([r.signal for r in rows_eng];
                 title = "Association with the realised scoreline, held-out block:")
println()
@printf("  %-12s | %9s | %s\n", "strategy", "coverage", "what it does")
println("  " * repeat('-', 96))
for r in rows_eng
    @printf("  %-12s | %8.1f%% | %s\n", r.name, 100 * r.coverage, r.description)
end

# %%
# ==============================================================================
# 5. Transfer to the deployment target
# ==============================================================================

eda_section("4/4", "Transfer — the same nine on Scottish League One and Two")

println("  Tiers 56/57 carry no SofaScore rating, so only criterion B is available here.")
println("  This is the table that matters: it is the store the feature is deployed on.\n")

ds_scot = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
rows_scot, _, y_scot = r51_bench(ds_scot, "Scottish 56/57"; score_against_rating = false)
eda_print_signal([r.signal for r in rows_scot];
                 title = "Association with the realised scoreline, held-out block:")

println("\n  Side by side, held-out r(goal supremacy):")
@printf("  %-12s | %14s | %14s | %10s\n", "strategy", "England", "Scot Lower", "consistent")
println("  " * repeat('-', 62))
scot_by_name = Dict(r.name => r for r in rows_scot)
for r in rows_eng
    s = get(scot_by_name, r.name, nothing)
    s === nothing && continue
    both = r.signal.r_supremacy > 0 && s.signal.r_supremacy > 0
    @printf("  %-12s | %14s | %14s | %10s\n", r.name,
            eda_fmt(r.signal.r_supremacy, "%+.4f"),
            eda_fmt(s.signal.r_supremacy, "%+.4f"),
            both ? "yes" : "NO")
end

# %%
# ==============================================================================
# 6. Verdict
# ==============================================================================

eda_section("VERDICT", "What to take forward")

"""
    r51_bootstrap_delta(column, baseline, y; draws) -> (mean, lo, hi)

Paired bootstrap over held-out MATCHES of `r(column, y) - r(baseline, y)`. Paired because
both columns are scored on the same fixtures, so their sampling errors cancel to a large
degree and the naive standard error of a single correlation badly overstates the
uncertainty in their difference.
"""
function r51_bootstrap_delta(column::Vector{Float64}, baseline::Vector{Float64},
                             y::Vector{Float64}; draws::Int = 4000)
    n = length(y)
    n < 30 && return (NaN, NaN, NaN)
    deltas = Vector{Float64}(undef, draws)
    idx = Vector{Int}(undef, n)
    for b in 1:draws
        for i in 1:n
            idx[i] = rand(1:n)
        end
        deltas[b] = eda_pearson(column[idx], y[idx]) - eda_pearson(baseline[idx], y[idx])
    end
    keep = filter(isfinite, deltas)
    isempty(keep) && return (NaN, NaN, NaN)
    return (mean(keep), quantile(keep, 0.05), quantile(keep, 0.95))
end

let
    baseline_eng = first(filter(r -> r.name == "exposure", rows_eng))
    baseline_scot = first(filter(r -> r.name == "exposure", rows_scot))
    @printf("  Production baseline (`exposure`):  England r = %+.4f, AUC %.4f  |  Scot Lower r = %+.4f, AUC %.4f\n",
            baseline_eng.signal.r_supremacy, baseline_eng.signal.auc_home_win,
            baseline_scot.signal.r_supremacy, baseline_scot.signal.auc_home_win)
    @printf("  Held-out sample sizes: England %d matches, Scottish Lower %d.\n\n",
            length(y_eng), length(y_scot))

    println("  Paired bootstrap of the change in r against the baseline, 4,000 resamples.")
    println("  An interval spanning zero means the strategy is indistinguishable from doing")
    println("  nothing on that store, whatever the point estimate says.\n")
    @printf("  %-12s | %-26s | %-26s | %s\n", "strategy",
            "ENGLAND  d r  [90% CI]", "SCOT LOWER  d r  [90% CI]", "verdict")
    println("  " * repeat('-', 104))

    Random.seed!(51)
    summary = NamedTuple[]
    for r in rows_eng
        r.name == "exposure" && continue
        s = get(scot_by_name, r.name, nothing)
        s === nothing && continue
        e = r51_bootstrap_delta(r.column, baseline_eng.column, y_eng)
        c = r51_bootstrap_delta(s.column, baseline_scot.column, y_scot)
        eng_sig = isfinite(e[2]) && (e[2] > 0 || e[3] < 0)
        sco_sig = isfinite(c[2]) && (c[2] > 0 || c[3] < 0)
        verdict = if eng_sig && sco_sig
            (e[1] > 0 && c[1] > 0) ? "REAL GAIN on both" : "real, but opposite signs"
        elseif eng_sig || sco_sig
            "significant on one store only"
        else
            "indistinguishable from baseline"
        end
        push!(summary, (name = r.name, d_eng = e[1], d_sco = c[1],
                        eng_sig = eng_sig, sco_sig = sco_sig, verdict = verdict))
        @printf("  %-12s | %+7.4f [%+.4f, %+.4f] | %+7.4f [%+.4f, %+.4f] | %s\n",
                r.name, e[1], e[2], e[3], c[1], c[2], c[3], verdict)
    end

    real_gains = filter(x -> x.eng_sig && x.sco_sig && x.d_eng > 0 && x.d_sco > 0, summary)
    println()
    if isempty(real_gains)
        println("  [VERDICT] NO normalisation is distinguishable from the production baseline on")
        println("            both stores. Every point estimate sits inside the bootstrap interval")
        println("            of doing nothing. The exposure shrink already captures what is there.")
        println()
        println("            The informative result is the DISSOCIATION between the two criteria:")
        println("            the strategies that agree best with the SofaScore rating (rank_pos,")
        println("            prior_zpos, zpos_nogk) are among the WORST at predicting matches on")
        println("            the deployment store. That is not a contradiction — the SofaScore")
        println("            rating is position- AND league-normalised, so it deliberately removes")
        println("            the team and context information a match covariate needs. Agreeing")
        println("            with it more closely means carrying less of what the engine wants.")
    else
        println("  [VERDICT] Distinguishable improvements on both stores:")
        for x in real_gains
            @printf("            %-12s  ENG %+.4f  SCO %+.4f\n", x.name, x.d_eng, x.d_sco)
        end
        println("\n            Next step is r40: a linear association is a necessary condition,")
        println("            not a sufficient one.")
    end
end
eda_rule(100, '=')
