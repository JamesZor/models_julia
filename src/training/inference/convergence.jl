# src/training/inference/convergence.jl
#
# MCMC convergence telemetry.
#
#     audit_convergence(folds) -> ConvergenceSummary
#
# No `DataStore`, no splitter, no feature sets, no team names — the folds and nothing
# else. That is the structural difference from `Experiments.Diagnostics`, whose
# `extract_chains` re-derives boundaries and rebuilds feature sets from a live
# `DataStore` purely to resolve `dyn.raw_a[7]` into `"Queen's Park"`. Team-name
# resolution is a report about parameter VALUES; it is deliberately not reimplemented
# here, and `Experiments.Diagnostics` still provides it for callers that want it.
#
# THE SIX METRICS, AND WHY EACH IS HERE
#
#   R-hat        between/within-chain variance ratio. Detects chains that explored
#                different regions. Blind to a single chain, and blind to all four
#                chains being stuck in the same wrong place.
#
#   bulk ESS     independent-draw equivalent for the BODY of the distribution — what a
#                posterior mean is worth.
#
#   tail ESS     the same for the 5%/95% quantiles. Reported SEPARATELY because they
#                fail separately, and the tail is the half that prices a 4-goal correct
#                score. Bulk 2000 with tail 90 is a trustworthy mean and untrustworthy
#                extremes; one number would hide that.
#
#   divergences  the sampler reporting it could not integrate the trajectory. The only
#                metric here that indicates BIAS rather than variance: divergent
#                transitions cluster in high-curvature regions, so the missing draws are
#                systematically from a specific part of the posterior. `check_convergence`
#                omits this entirely, so a fold with 400 divergences and R-hat 1.003
#                passes it cleanly.
#
#   tree depth   trajectories hitting `max_depth` were TRUNCATED, not completed. A
#                performance gate on the CAP RATE, not on depth itself.
#
#   BFMI         whether momentum resampling is moving the chain through energy space.
#                Unlike divergences it can be low while every other metric looks fine.
#
# An optimisation fold (MAP/MLE) has no R-hat: over one draw it is undefined, not 1.0.
# Rather than reporting a NaN that a downstream `<` silently turns into `false`, such
# folds are marked `applicable = false`, excluded from every reduction, and counted.

# ==============================================================================
# 1. PER-FOLD TELEMETRY
# ==============================================================================

"A fold whose sampler returned a point estimate. Every metric absent, and said so."
inapplicable_fold(fold::Integer, n_params::Integer = 0) =
    FoldConvergence(Int(fold), false, Int(n_params), 1, 1,
                    NaN, :none, NaN, :none, NaN, :none,
                    0, 0, NaN, 0, 0, NaN, NaN)

"""
    bfmi(energy::AbstractMatrix) -> Vector{Float64}

Bayesian Fraction of Missing Information, one value per chain, from the
`hamiltonian_energy` internal (`n_draws × n_chains`).

    E-BFMI = Σ_{n≥2} (E_n − E_{n−1})²  /  Σ_n (E_n − Ē)²

Implemented rather than called because `MCMCChains` 7.7 does not export `bfmi` — it
lives in `MCMCDiagnosticTools`, which is not a direct dependency of this project. Adding
a dependency to compute a two-line formula would be the larger change.

Returns an empty vector for fewer than two draws, where the numerator has no terms.
"""
function bfmi(energy::AbstractMatrix{<:Real})
    n_draws, n_chains = size(energy)
    n_draws < 2 && return Float64[]
    out = Vector{Float64}(undef, n_chains)
    for c in 1:n_chains
        num = 0.0
        @inbounds for n in 2:n_draws
            d = energy[n, c] - energy[n - 1, c]
            num += d * d
        end
        μ = 0.0
        @inbounds for n in 1:n_draws
            μ += energy[n, c]
        end
        μ /= n_draws
        den = 0.0
        @inbounds for n in 1:n_draws
            d = energy[n, c] - μ
            den += d * d
        end
        out[c] = den > 0 ? num / den : NaN
    end
    return out
end

"An internal column of a chain as an `(n_draws × n_chains)` matrix, or `nothing`."
function _inf_internal(chain::Chains, name::Symbol)
    try
        name in MCMCChains.names(chain, :internals) || return nothing
        return Array(chain[name])
    catch
        return nothing
    end
end

"""
    audit_fold(fold::Integer, chain; max_depth = 10) -> FoldConvergence
    audit_fold(f::FoldFit; max_depth = 10)           -> FoldConvergence

Every metric for one chain. Never throws: a diagnostic that can abort a completed
six-hour run is worse than one that reports `NaN`, so each metric is computed inside its
own guard and a failure to compute one leaves the others intact.

`max_depth` is the sampler's configured cap, needed to count saturating trajectories.
`fit_model` reads it off the sampler config; the default is `NUTSConfig`'s own.
"""
function audit_fold(fold::Integer, chain::Chains; max_depth::Integer = 10)
    n_draws, _, n_chains = size(chain)
    params = MCMCChains.names(chain, :parameters)
    n_params = length(params)

    n_draws < 2 && return inapplicable_fold(fold, n_params)

    max_rhat, rhat_param = _inf_worst(chain, :rhat, :max)
    min_bulk, bulk_param = _inf_worst(chain, :ess_bulk, :min)
    min_tail, tail_param = _inf_worst(chain, :ess_tail, :min)

    # `:numerical_error` is what AdvancedHMC records through Turing. A sampler that does
    # not record it reports 0 of 0 rather than 0 of N, so the rate is NaN and the gate
    # abstains instead of passing on absent evidence.
    div = _inf_internal(chain, :numerical_error)
    n_transitions = n_draws * n_chains
    n_divergent = div === nothing ? 0 : count(x -> x > 0, div)
    div_rate = div === nothing ? NaN : n_divergent / n_transitions

    td = _inf_internal(chain, :tree_depth)
    max_td = td === nothing ? 0 : Int(maximum(td))
    n_capped = td === nothing ? 0 : count(>=(max_depth), td)
    td_rate = td === nothing ? NaN : n_capped / n_transitions

    energy = _inf_internal(chain, :hamiltonian_energy)
    bf = energy === nothing ? Float64[] : bfmi(energy)
    min_bfmi = isempty(bf) ? NaN : minimum(bf)

    return FoldConvergence(Int(fold), true, n_params, n_draws, n_chains,
                           max_rhat, rhat_param,
                           min_bulk, bulk_param,
                           min_tail, tail_param,
                           n_divergent, n_transitions, div_rate,
                           max_td, n_capped, td_rate,
                           min_bfmi)
end

audit_fold(fold::Integer, chain; max_depth::Integer = 10) = inapplicable_fold(fold, 0)

audit_fold(f::FoldFit; max_depth::Integer = 10) =
    audit_fold(f.fold, f.chain; max_depth = max_depth)

"""
    _inf_worst(chain, which, dir) -> (value, parameter)

The extreme of one diagnostic over every parameter, and the parameter that attains it.

`NaN` entries are SKIPPED rather than propagated. A constant site (a `Dirac`, a fixed
league offset in a single-league fold) has zero variance and every diagnostic returns
`NaN` for it; letting that NaN win the `maximum` would fail the R-hat gate on a
parameter that has nothing to converge.
"""
function _inf_worst(chain::Chains, which::Symbol, dir::Symbol)
    vals, params = try
        if which === :rhat
            df = DataFrame(MCMCChains.rhat(chain))
            (df.rhat, df.parameters)
        else
            kind = which === :ess_bulk ? :bulk : :tail
            df = DataFrame(MCMCChains.ess(chain; kind = kind))
            (df.ess, df.parameters)
        end
    catch
        return (NaN, :none)
    end

    best = dir === :max ? -Inf : Inf
    best_p = :none
    for (v, p) in zip(vals, params)
        (v === missing || !isfinite(v)) && continue
        if (dir === :max && v > best) || (dir === :min && v < best)
            best = Float64(v)
            best_p = Symbol(p)
        end
    end
    return (isfinite(best) ? best : NaN, best_p)
end


# ==============================================================================
# 2. RUN-LEVEL SUMMARY
# ==============================================================================

"""
    audit_convergence(folds; thresholds = ConvergenceThresholds(), max_depth = 10)
        -> ConvergenceSummary
    audit_convergence(fit::Fit; kwargs...) -> ConvergenceSummary

The telemetry module's public entry point. Takes the folds, returns the verdict.

Deliberately takes `Vector{<:FoldFit}` and NOTHING else — no `DataStore`, no splitter,
no model — which is what makes it callable on a `Fit` loaded from disk with no database
in sight.
"""
function audit_convergence(folds::Vector{<:FoldFit};
                           thresholds::ConvergenceThresholds = ConvergenceThresholds(),
                           max_depth::Integer = 10)
    per = [audit_fold(f; max_depth = max_depth) for f in folds]
    return summarise_convergence(per; thresholds = thresholds)
end

audit_convergence(f::Fit; kwargs...) = audit_convergence(getfield(f, :folds); kwargs...)

"""
    summarise_convergence(per_fold; thresholds = ConvergenceThresholds())
        -> ConvergenceSummary

Reduce per-fold telemetry to the run verdict. Split out from `audit_convergence` so a
caller can re-gate an existing summary against different thresholds without touching the
chains — `summarise_convergence(summary.folds; thresholds = stricter)`.
"""
function summarise_convergence(per::Vector{FoldConvergence};
                               thresholds::ConvergenceThresholds = ConvergenceThresholds())
    ok = filter(f -> f.applicable, per)

    max_rhat, rhat_fold = _inf_reduce(ok, f -> f.max_rhat,     f -> f.fold, :max)
    min_bulk, bulk_fold = _inf_reduce(ok, f -> f.min_ess_bulk, f -> f.fold, :min)
    min_tail, tail_fold = _inf_reduce(ok, f -> f.min_ess_tail, f -> f.fold, :min)
    min_bfmi, bfmi_fold = _inf_reduce(ok, f -> f.min_bfmi,     f -> f.fold, :min)

    n_div   = sum(f -> f.n_divergent, ok; init = 0)
    n_trans = sum(f -> f.n_transitions, ok; init = 0)
    measured_div = any(f -> isfinite(f.divergence_rate), ok)
    div_rate = measured_div && n_trans > 0 ? n_div / n_trans : NaN

    max_td   = maximum(f -> f.max_tree_depth, ok; init = 0)
    n_capped = sum(f -> f.n_depth_capped, ok; init = 0)
    measured_td = any(f -> isfinite(f.treedepth_rate), ok)
    td_rate = measured_td && n_trans > 0 ? n_capped / n_trans : NaN

    failures  = String[]
    failed    = String[]
    abstained = String[]
    gate!(name, v, t, dir, msg) =
        _inf_record_gate!(failures, failed, abstained, name, v, t, dir, msg)

    gate!("R-hat", max_rhat, thresholds.max_rhat, :below,
          @sprintf("max R-hat %.4f at fold %d (gate < %.3f)",
                   max_rhat, rhat_fold, thresholds.max_rhat))
    gate!("bulk ESS", min_bulk, thresholds.min_ess, :above,
          @sprintf("min bulk ESS %.1f at fold %d (gate > %.0f)",
                   min_bulk, bulk_fold, thresholds.min_ess))
    gate!("tail ESS", min_tail, thresholds.min_ess, :above,
          @sprintf("min tail ESS %.1f at fold %d (gate > %.0f)",
                   min_tail, tail_fold, thresholds.min_ess))
    gate!("divergences", div_rate, thresholds.max_divergence_rate, :below,
          @sprintf("%d divergent of %d transitions = %.3f%% (gate < %.3f%%)",
                   n_div, n_trans, 100 * div_rate,
                   100 * thresholds.max_divergence_rate))
    gate!("BFMI", min_bfmi, thresholds.min_bfmi, :above,
          @sprintf("min BFMI %.3f at fold %d (gate > %.2f)",
                   min_bfmi, bfmi_fold, thresholds.min_bfmi))
    gate!("tree depth", td_rate, thresholds.max_treedepth_rate, :below,
          @sprintf("%d of %d transitions saturated depth %d = %.2f%% (gate < %.1f%%)",
                   n_capped, n_trans, max_td, 100 * td_rate,
                   100 * thresholds.max_treedepth_rate))

    return ConvergenceSummary(per, thresholds, length(per), length(ok),
                              max_rhat, rhat_fold,
                              min_bulk, bulk_fold,
                              min_tail, tail_fold,
                              n_div, n_trans, div_rate,
                              max_td, n_capped, td_rate,
                              min_bfmi, bfmi_fold,
                              isempty(failures), failures, failed, abstained)
end

"Extreme of `val` over folds, and the `key` of the fold attaining it. NaN-skipping."
function _inf_reduce(folds, val, key, dir::Symbol)
    best = dir === :max ? -Inf : Inf
    best_k = 0
    for f in folds
        v = val(f)
        isfinite(v) || continue
        if (dir === :max && v > best) || (dir === :min && v < best)
            best = v
            best_k = key(f)
        end
    end
    return (isfinite(best) ? best : NaN, best_k)
end

"""
    _inf_record_gate!(failures, failed, abstained, name, value, threshold, dir, message)

Record one gate as passed, failed, or abstained-for-want-of-a-measurement.

A failure lands in BOTH lists: the sentence in `failures`, the bare gate name in
`failed_gates`. Keeping the name separately is what lets a caller write
`"BFMI" in summary.failed_gates` instead of `occursin("BFMI", …)` over prose.
"""
function _inf_record_gate!(failures::Vector{String}, failed::Vector{String},
                           abstained::Vector{String},
                           name::String, value::Float64, threshold::Float64,
                           direction::Symbol, message::String)
    if !isfinite(value)
        push!(abstained, name)
        return nothing
    end
    ok = direction === :below ? value < threshold : value > threshold
    if !ok
        push!(failures, message)
        push!(failed, name)
    end
    return nothing
end


# ==============================================================================
# 3. DISPLAY
# ==============================================================================

"""
    convergence_table(summary; io = stdout)

The per-fold table. One line per fold, worst-offending parameter named where the fold
failed its own gate.
"""
function convergence_table(s::ConvergenceSummary; io::IO = stdout)
    println(io, "  fold  R-hat    ESS-bulk  ESS-tail   div        depth-cap   BFMI    worst")
    println(io, "  ", "-"^76)
    for f in s.folds
        if !f.applicable
            @printf(io, "  %4d  %-8s %-9s %-10s %-10s %-11s %-7s %s\n",
                    f.fold, "—", "—", "—", "—", "—", "—", "point estimate")
            continue
        end
        worst = _inf_fold_worst(f, s.thresholds)
        @printf(io, "  %4d  %-8.4f %-9.1f %-10.1f %-10s %-11s %-7s %s\n",
                f.fold, f.max_rhat, f.min_ess_bulk, f.min_ess_tail,
                isfinite(f.divergence_rate) ?
                    @sprintf("%d/%d", f.n_divergent, f.n_transitions) : "—",
                isfinite(f.treedepth_rate) ?
                    @sprintf("%d @ %d", f.n_depth_capped, f.max_tree_depth) : "—",
                isfinite(f.min_bfmi) ? @sprintf("%.3f", f.min_bfmi) : "—",
                worst)
    end
    return nothing
end

"The gate this fold is furthest from clearing, for the table's last column."
function _inf_fold_worst(f::FoldConvergence, g::ConvergenceThresholds)
    isfinite(f.max_rhat) && f.max_rhat >= g.max_rhat &&
        return "rhat " * string(f.worst_rhat_param)
    isfinite(f.min_ess_bulk) && f.min_ess_bulk <= g.min_ess &&
        return "ess-bulk " * string(f.worst_ess_bulk_param)
    isfinite(f.min_ess_tail) && f.min_ess_tail <= g.min_ess &&
        return "ess-tail " * string(f.worst_ess_tail_param)
    isfinite(f.divergence_rate) && f.divergence_rate >= g.max_divergence_rate &&
        return "divergences"
    isfinite(f.min_bfmi) && f.min_bfmi <= g.min_bfmi && return "bfmi"
    return "ok"
end

"The one-line verdict `Fit`'s own `show` prints."
diagnostics_line(s::ConvergenceSummary) = string(
    s.passed ? "PASS" : "FAIL",
    @sprintf("  R̂≤%.4f  ESS≥%.0f  div %d/%d",
             s.max_rhat, min(s.min_ess_bulk, s.min_ess_tail),
             s.n_divergent, s.n_transitions))

function Base.show(io::IO, s::ConvergenceSummary)
    print(io, "ConvergenceSummary(", s.passed ? "PASS" : "FAIL",
          ", ", s.n_folds, " folds, R̂≤", @sprintf("%.4f", s.max_rhat),
          ", ESS≥", @sprintf("%.0f", min(s.min_ess_bulk, s.min_ess_tail)),
          ", div ", s.n_divergent, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::ConvergenceSummary)
    println(io, "ConvergenceSummary  —  ", s.passed ? "PASS" : "FAIL")
    println(io, "  folds          : ", s.n_folds,
                " (", s.n_applicable, " sampled, ",
                s.n_folds - s.n_applicable, " point estimates)")
    @printf(io, "  max R-hat      : %.4f      (fold %d, gate < %.3f)\n",
            s.max_rhat, s.worst_rhat_fold, s.thresholds.max_rhat)
    @printf(io, "  min bulk ESS   : %.1f       (fold %d, gate > %.0f)\n",
            s.min_ess_bulk, s.worst_ess_bulk_fold, s.thresholds.min_ess)
    @printf(io, "  min tail ESS   : %.1f       (fold %d, gate > %.0f)\n",
            s.min_ess_tail, s.worst_ess_tail_fold, s.thresholds.min_ess)
    if isfinite(s.divergence_rate)
        @printf(io, "  divergences    : %d / %d = %.3f%%   (gate < %.3f%%)\n",
                s.n_divergent, s.n_transitions, 100 * s.divergence_rate,
                100 * s.thresholds.max_divergence_rate)
    else
        println(io, "  divergences    : not recorded by this sampler")
    end
    if isfinite(s.treedepth_rate)
        @printf(io, "  depth capped   : %d / %d = %.2f%% at depth %d  (gate < %.1f%%)\n",
                s.n_depth_capped, s.n_transitions, 100 * s.treedepth_rate,
                s.max_tree_depth, 100 * s.thresholds.max_treedepth_rate)
    else
        println(io, "  depth capped   : not recorded by this sampler")
    end
    if isfinite(s.min_bfmi)
        @printf(io, "  min BFMI       : %.3f       (fold %d, gate > %.2f)\n",
                s.min_bfmi, s.worst_bfmi_fold, s.thresholds.min_bfmi)
    else
        println(io, "  min BFMI       : no energy record")
    end
    isempty(s.abstained) ||
        println(io, "  abstained      : ", join(s.abstained, ", "),
                    "  (metric absent, gate neither passed nor failed)")
    if !isempty(s.failures)
        println(io, "  failures:")
        for f in s.failures
            println(io, "    ✗ ", f)
        end
    end
    return nothing
end
