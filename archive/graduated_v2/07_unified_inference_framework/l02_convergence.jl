# ==============================================================================
# 07 — UNIFIED INFERENCE FRAMEWORK : CONVERGENCE TELEMETRY
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# WHAT THIS FILE REPLACES
# ------------------------------------------------------------------------------
#
# `src/experiments/diagnostics/` — seven files, ~600 lines:
#
#   types.jl        `ExperimentChains`, `ChainDiagnostic`, `StabilityDiagnostic`,
#                   all three of which are `struct X; df::DataFrame; end`.
#   extraction.jl   `extract_chains` — 240 lines that rebuild the SPLITTER and the
#                   FEATURE SETS from a `DataStore`, then walk eight
#                   `hasproperty(model, :x_config)` branches to map every parameter
#                   back to a team name.
#   convergence.jl  `check_convergence` — 6 lines: `filter(row -> !isnan(row.rhat), …)`.
#   stability.jl, display.jl, utils.jl, diagnostics.jl
#
# Two things are wrong with that, and only one of them is size.
#
#   1. IT NEEDS A `DataStore`. `extract_chains(ds, exp_results)` re-derives boundaries
#      and features purely to resolve `dyn.raw_a[7]` into `"Queen's Park"`. That is a
#      REPORTING concern. It means the question "did fold 9 converge" cannot be asked
#      of a saved run without a live database connection — so in practice it is not
#      asked at all.
#
#   2. IT ONLY LOOKS AT R-HAT AND ESS, and only at whichever parameters its eight
#      branches happened to reach. A NUTS run's most informative failure signals are
#      DIVERGENCES and BFMI, and neither appears anywhere in that directory. A fold
#      with 400 divergent transitions and R-hat 1.003 passes `check_convergence`
#      cleanly, and its posterior is wrong in exactly the region that matters.
#
# ------------------------------------------------------------------------------
# WHAT REPLACES IT
# ------------------------------------------------------------------------------
#
# One function over the chains and nothing else:
#
#     audit_convergence(folds) -> ConvergenceSummary
#
# No `DataStore`, no splitter, no feature sets, no team names. Six numbers per fold,
# reduced to six numbers per run, and a `passed::Bool` that is a conjunction of
# explicit thresholds rather than a judgement call.
#
# The team-name mapping is NOT reimplemented here and is not missed: it belongs to a
# report about parameter VALUES (`extract_chains`' actual job — it also collects means
# and standard deviations for stability analysis), not to a question about whether the
# sampler worked. Deleting the coupling is the point.
#
# ------------------------------------------------------------------------------
# THE SIX METRICS, AND WHY EACH ONE IS THERE
# ------------------------------------------------------------------------------
#
#   R-hat        between/within-chain variance ratio. Detects chains that explored
#                different regions. Blind to a single chain, and blind to all four
#                chains being stuck in the same wrong place.
#
#   bulk ESS     independent-draw equivalent for the BODY of the distribution — what a
#                posterior mean is worth.
#
#   tail ESS     the same for the 5%/95% quantiles. Reported SEPARATELY from bulk
#                because they fail separately, and the tail is the half that prices a
#                4-goal correct-score line. A run with bulk ESS 2000 and tail ESS 90
#                has a trustworthy mean and untrustworthy extremes; one number would
#                hide that.
#
#   divergences  the sampler reporting that it could not integrate the trajectory. The
#                only metric here that indicates BIAS rather than variance: divergent
#                transitions cluster in high-curvature regions, so the draws that are
#                missing are systematically the ones from a specific part of the
#                posterior. This is the metric the legacy directory omits entirely.
#
#   tree depth   trajectories hitting `max_depth` were TRUNCATED, not completed. Not
#                an error — the run is still valid — but it means each draw cost the
#                maximum and bought less than it should have. A gate on the CAP RATE,
#                not on depth itself.
#
#   BFMI         Bayesian Fraction of Missing Information. Whether the momentum
#                resampling is actually moving the chain through energy space. Low BFMI
#                means the sampler is exploring a heavy-tailed posterior with a
#                step size calibrated for a light one, and — unlike divergences — it
#                can be low while every other metric looks perfect.
#
# ------------------------------------------------------------------------------
# WHAT AN OPTIMISATION FIT DOES HERE
# ------------------------------------------------------------------------------
#
# `MAPConfig` / `MLEConfig` return a point estimate. R-hat over one draw is undefined,
# not 1.0. Rather than reporting a NaN that a downstream `<` silently converts into
# `false`, such folds are marked `applicable = false`, excluded from every reduction,
# and COUNTED, so a summary can say "3 of 12 folds were point estimates" instead of
# quietly averaging over nine.
#
# ==============================================================================

using MCMCChains
using Printf
using Statistics

include(joinpath(@__DIR__, "l01_types.jl"))


# ==============================================================================
# 1. GATES
# ==============================================================================

"""
    ConvergenceGates(; max_rhat = 1.01, min_ess = 400.0,
                       max_divergence_rate = 0.001, min_bfmi = 0.30,
                       max_treedepth_rate = 0.05)

The thresholds `passed` is a conjunction of.

The four the briefing names are the Stan/`bayesplot` community defaults, and are not
arbitrary:

  * `max_rhat = 1.01` — the 2019 Vehtari et al. revision of the older 1.1, which was
    shown to pass chains that had visibly not mixed.
  * `min_ess = 400` — 100 effective draws per chain at 4 chains; below this the Monte
    Carlo standard error on a mean is a material fraction of the posterior SD.
  * `max_divergence_rate = 0.001` — the briefing's 0.10%. Stricter than the usual
    "any divergence is worth investigating" advice in one direction and looser in the
    other; it is a THRESHOLD, and the count is reported next to it so a reader can
    apply their own.
  * `min_bfmi = 0.30` — Betancourt's threshold for "reparameterise this model".

`max_treedepth_rate` is added here, not in the briefing: tree-depth saturation is
reported by every NUTS run and ignoring it entirely would mean collecting the number
and then not using it. It is a PERFORMANCE gate, not a correctness one, and it is the
one gate whose failure does not invalidate the posterior.
"""
Base.@kwdef struct ConvergenceGates
    max_rhat::Float64 = 1.01
    min_ess::Float64 = 400.0
    max_divergence_rate::Float64 = 0.001
    min_bfmi::Float64 = 0.30
    max_treedepth_rate::Float64 = 0.05
end


# ==============================================================================
# 2. PER-FOLD TELEMETRY
# ==============================================================================

"""
    FoldConvergence

One fold's six metrics, plus the parameter name that produced each worst case.

Carrying `worst_rhat_param` and friends is what makes the summary ACTIONABLE. "max
R-hat 1.34" sends a reader back to the chains; "max R-hat 1.34 at `dyn.σ_a`" names a
non-centred scale parameter and, usually, the fix.

`applicable = false` marks a point-estimate fold (MAP/MLE, or any chain with one draw),
whose metrics are all `NaN`/`0` and are excluded from every reduction in §3.
"""
struct FoldConvergence
    fold::Int
    applicable::Bool
    n_params::Int
    n_draws::Int
    n_chains::Int
    max_rhat::Float64
    worst_rhat_param::Symbol
    min_ess_bulk::Float64
    worst_ess_bulk_param::Symbol
    min_ess_tail::Float64
    worst_ess_tail_param::Symbol
    n_divergent::Int
    n_transitions::Int
    divergence_rate::Float64
    max_tree_depth::Int
    n_depth_capped::Int
    treedepth_rate::Float64
    min_bfmi::Float64
end

"A fold whose sampler returned a point estimate. Every metric absent, and said so."
inapplicable_fold(fold::Integer, n_params::Integer = 0) =
    FoldConvergence(Int(fold), false, Int(n_params), 1, 1,
                    NaN, :none, NaN, :none, NaN, :none,
                    0, 0, NaN, 0, 0, NaN, NaN)

"""
    bfmi(energy::AbstractMatrix) -> Vector{Float64}

Bayesian Fraction of Missing Information, one value per chain, from the
`hamiltonian_energy` internal (`(n_draws × n_chains)`).

    E-BFMI = Σ_{n≥2} (E_n − E_{n−1})²  /  Σ_n (E_n − Ē)²

Implemented here rather than called, because `MCMCChains` 7.7 does not export `bfmi`
(it lives in `MCMCDiagnosticTools`, which is not a direct dependency of this project —
`Project.toml` has neither the package nor a compat bound for it). Adding a dependency
to compute a two-line formula would be the larger change.

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
function _uif_internal(chain::Chains, name::Symbol)
    try
        name in MCMCChains.names(chain, :internals) || return nothing
        return Array(chain[name])
    catch
        return nothing
    end
end

"""
    audit_fold(fold::Int, chain; max_depth = 10) -> FoldConvergence

Every metric for one chain. Never throws: a diagnostic that can abort a completed
six-hour run is worse than a diagnostic that reports `NaN`, so each metric is computed
inside its own guard and a failure to compute one leaves the others intact.

`max_depth` is the sampler's configured cap, needed to count saturating trajectories.
`fit_model` reads it off the sampler config; the default is `NUTSConfig`'s own.
"""
function audit_fold(fold::Integer, chain::Chains; max_depth::Integer = 10)
    n_draws, _, n_chains = size(chain)
    params = MCMCChains.names(chain, :parameters)
    n_params = length(params)

    n_draws < 2 && return inapplicable_fold(fold, n_params)

    # --- R-hat -----------------------------------------------------------------
    max_rhat, rhat_param = _uif_worst(chain, :rhat, :max)
    # --- ESS -------------------------------------------------------------------
    min_bulk, bulk_param = _uif_worst(chain, :ess_bulk, :min)
    min_tail, tail_param = _uif_worst(chain, :ess_tail, :min)

    # --- divergences -----------------------------------------------------------
    #
    # `:numerical_error` is what AdvancedHMC records through Turing. A run under a
    # sampler that does not record it (ADVI, SGLD) reports 0 of 0 rather than 0 of N,
    # so the rate is NaN and the gate abstains instead of passing on absent evidence.
    div = _uif_internal(chain, :numerical_error)
    n_transitions = n_draws * n_chains
    n_divergent = div === nothing ? 0 : count(x -> x > 0, div)
    div_rate = div === nothing ? NaN : n_divergent / n_transitions

    # --- tree depth ------------------------------------------------------------
    td = _uif_internal(chain, :tree_depth)
    max_td = td === nothing ? 0 : Int(maximum(td))
    n_capped = td === nothing ? 0 : count(>=(max_depth), td)
    td_rate = td === nothing ? NaN : n_capped / n_transitions

    # --- BFMI ------------------------------------------------------------------
    energy = _uif_internal(chain, :hamiltonian_energy)
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
    _uif_worst(chain, which, dir) -> (value, parameter)

The extreme of one diagnostic over every parameter, and the parameter that attains it.

`NaN` entries are SKIPPED rather than propagated. A constant site (a `Dirac`, a fixed
league offset in a single-league fold) has zero variance and every diagnostic returns
`NaN` for it; letting that NaN win the `maximum` would fail the R-hat gate on a
parameter that has nothing to converge.
"""
function _uif_worst(chain::Chains, which::Symbol, dir::Symbol)
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
# 3. RUN-LEVEL SUMMARY
# ==============================================================================

"""
    ConvergenceSummary

The reduction over folds, and the verdict.

`passed` is a CONJUNCTION over the gates, evaluated only on applicable folds.
`failures` gives one human-readable line per fallen gate, with the number that did it;
`failed_gates` gives the same set as bare gate NAMES, for a caller that wants to branch
on which one rather than parse a sentence. A summary that says `passed = false` can
always answer "because of what", in both registers.

ABSTENTION. A gate whose metric is `NaN` for every fold — no divergence record because
the sampler does not emit one, no energy record because it is not Hamiltonian — is
neither passed nor failed: it is listed in `abstained`. Treating an unmeasured gate as
passed would let a sampler earn a clean bill of health by recording nothing, which is
precisely backwards.
"""
struct ConvergenceSummary
    folds::Vector{FoldConvergence}
    gates::ConvergenceGates
    n_folds::Int
    n_applicable::Int
    max_rhat::Float64
    worst_rhat_fold::Int
    min_ess_bulk::Float64
    worst_ess_bulk_fold::Int
    min_ess_tail::Float64
    worst_ess_tail_fold::Int
    n_divergent::Int
    n_transitions::Int
    divergence_rate::Float64
    max_tree_depth::Int
    n_depth_capped::Int
    treedepth_rate::Float64
    min_bfmi::Float64
    worst_bfmi_fold::Int
    passed::Bool
    failures::Vector{String}
    failed_gates::Vector{String}
    abstained::Vector{String}
end

"""
    audit_convergence(folds; gates = ConvergenceGates(), max_depth = 10)
        -> ConvergenceSummary

The whole telemetry module's public entry point. Takes the folds, returns the verdict.

Deliberately takes `Vector{<:FoldFit}` and NOTHING else — no `DataStore`, no splitter,
no model. That is the entire structural change from `Experiments.Diagnostics`, and it
is what makes this callable on a `Fit` loaded from disk with no database in sight.
"""
function audit_convergence(folds::Vector{<:FoldFit};
                           gates::ConvergenceGates = ConvergenceGates(),
                           max_depth::Integer = 10)
    per = [audit_fold(f; max_depth = max_depth) for f in folds]
    return summarise_convergence(per; gates = gates)
end

audit_convergence(f::Fit; kwargs...) = audit_convergence(getfield(f, :folds); kwargs...)

"""
    summarise_convergence(per_fold; gates = ConvergenceGates()) -> ConvergenceSummary

Reduce per-fold telemetry to the run verdict. Split out from `audit_convergence` so a
caller can re-gate an existing summary against different thresholds without touching
the chains — `summarise_convergence(summary.folds; gates = stricter)`.
"""
function summarise_convergence(per::Vector{FoldConvergence};
                               gates::ConvergenceGates = ConvergenceGates())
    ok = filter(f -> f.applicable, per)

    max_rhat, rhat_fold = _uif_reduce(ok, f -> f.max_rhat,     f -> f.fold, :max)
    min_bulk, bulk_fold = _uif_reduce(ok, f -> f.min_ess_bulk, f -> f.fold, :min)
    min_tail, tail_fold = _uif_reduce(ok, f -> f.min_ess_tail, f -> f.fold, :min)
    min_bfmi, bfmi_fold = _uif_reduce(ok, f -> f.min_bfmi,     f -> f.fold, :min)

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
    _uif_gate!(f, a, name, v, t, dir, msg) =
        _uif_record_gate!(f, failed, a, name, v, t, dir, msg)

    _uif_gate!(failures, abstained, "R-hat", max_rhat, gates.max_rhat, :below,
               @sprintf("max R-hat %.4f at fold %d (gate < %.3f)",
                        max_rhat, rhat_fold, gates.max_rhat))
    _uif_gate!(failures, abstained, "bulk ESS", min_bulk, gates.min_ess, :above,
               @sprintf("min bulk ESS %.1f at fold %d (gate > %.0f)",
                        min_bulk, bulk_fold, gates.min_ess))
    _uif_gate!(failures, abstained, "tail ESS", min_tail, gates.min_ess, :above,
               @sprintf("min tail ESS %.1f at fold %d (gate > %.0f)",
                        min_tail, tail_fold, gates.min_ess))
    _uif_gate!(failures, abstained, "divergences", div_rate, gates.max_divergence_rate,
               :below,
               @sprintf("%d divergent of %d transitions = %.3f%% (gate < %.3f%%)",
                        n_div, n_trans, 100 * div_rate, 100 * gates.max_divergence_rate))
    _uif_gate!(failures, abstained, "BFMI", min_bfmi, gates.min_bfmi, :above,
               @sprintf("min BFMI %.3f at fold %d (gate > %.2f)",
                        min_bfmi, bfmi_fold, gates.min_bfmi))
    _uif_gate!(failures, abstained, "tree depth", td_rate, gates.max_treedepth_rate,
               :below,
               @sprintf("%d of %d transitions saturated depth %d = %.2f%% (gate < %.1f%%)",
                        n_capped, n_trans, max_td, 100 * td_rate,
                        100 * gates.max_treedepth_rate))

    return ConvergenceSummary(per, gates, length(per), length(ok),
                              max_rhat, rhat_fold,
                              min_bulk, bulk_fold,
                              min_tail, tail_fold,
                              n_div, n_trans, div_rate,
                              max_td, n_capped, td_rate,
                              min_bfmi, bfmi_fold,
                              isempty(failures), failures, failed, abstained)
end

"Extreme of `val` over folds, and the `key` of the fold attaining it. NaN-skipping."
function _uif_reduce(folds, val, key, dir::Symbol)
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
    _uif_record_gate!(failures, failed, abstained, name, value, threshold, dir, message)

Record one gate as passed, failed, or abstained-for-want-of-a-measurement.

A failure lands in BOTH lists: the sentence in `failures`, the bare gate name in
`failed_gates`. Keeping the name separately is what lets a caller write
`"BFMI" in summary.failed_gates` instead of `occursin("BFMI", …)` over prose that may
be reworded.
"""
function _uif_record_gate!(failures::Vector{String}, failed::Vector{String},
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
# 4. DISPLAY
# ==============================================================================

"""
    convergence_table(s; io = stdout)

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
        worst = _uif_fold_worst(f, s.gates)
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

"The name of whichever gate this fold is furthest from clearing, for the table's last column."
function _uif_fold_worst(f::FoldConvergence, g::ConvergenceGates)
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
            s.max_rhat, s.worst_rhat_fold, s.gates.max_rhat)
    @printf(io, "  min bulk ESS   : %.1f       (fold %d, gate > %.0f)\n",
            s.min_ess_bulk, s.worst_ess_bulk_fold, s.gates.min_ess)
    @printf(io, "  min tail ESS   : %.1f       (fold %d, gate > %.0f)\n",
            s.min_ess_tail, s.worst_ess_tail_fold, s.gates.min_ess)
    if isfinite(s.divergence_rate)
        @printf(io, "  divergences    : %d / %d = %.3f%%   (gate < %.3f%%)\n",
                s.n_divergent, s.n_transitions, 100 * s.divergence_rate,
                100 * s.gates.max_divergence_rate)
    else
        println(io, "  divergences    : not recorded by this sampler")
    end
    if isfinite(s.treedepth_rate)
        @printf(io, "  depth capped   : %d / %d = %.2f%% at depth %d  (gate < %.1f%%)\n",
                s.n_depth_capped, s.n_transitions, 100 * s.treedepth_rate,
                s.max_tree_depth, 100 * s.gates.max_treedepth_rate)
    else
        println(io, "  depth capped   : not recorded by this sampler")
    end
    if isfinite(s.min_bfmi)
        @printf(io, "  min BFMI       : %.3f       (fold %d, gate > %.2f)\n",
                s.min_bfmi, s.worst_bfmi_fold, s.gates.min_bfmi)
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

# The one-line form `Fit`'s own `show` uses (l01 §8 declares the fallback).
_uif_diag_line(s::ConvergenceSummary) = string(
    s.passed ? "PASS" : "FAIL",
    @sprintf("  R̂≤%.4f  ESS≥%.0f  div %d/%d",
             s.max_rhat, min(s.min_ess_bulk, s.min_ess_tail),
             s.n_divergent, s.n_transitions))
