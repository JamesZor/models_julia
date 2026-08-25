# ==============================================================================
# Model 01 — GATE 3 : SAMPLING
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 3 asks three separate questions, in increasing cost:
#
#   (a) equation parity   is the Turing model the model MODEL.md documents?
#   (b) gradient health   is the AD path correct and fast enough to sample?
#   (c) smoke run         does it actually sample, and converge?
#
# (a) is the load-bearing one. It compares DynamicPPL's log density against
# l02_equations.jl — an independent implementation written from the documentation
# rather than from the engine. If those agree, the fitted model and the documented
# model are the same object, and gate 4 can then check the priced model against
# the same reference.
#
# The smoke run persists a real chain through src/experiments. That artifact is
# the input to gate 4 — nothing here is throwaway.
#
# ==============================================================================

using BayesianFootball
using DynamicPPL
using LogDensityProblems
using ForwardDiff
using ReverseDiff
using LinearAlgebra
using Random
using Statistics
using Printf

const PGm = BayesianFootball.Models.PreGame


# ==============================================================================
# 1. Prior draws
# ==============================================================================

"""
    tp_params_from_varinfo(vi) -> TPParams

Read a draw out of a `VarInfo` BY NAME.

Deliberately not by position. A positional mapping would silently reinterpret the
vector if DynamicPPL ever reordered sites, which is precisely the class of error
this gate exists to detect.
"""
function tp_params_from_varinfo(vi)
    v = Dict(string(k) => vi[k] for k in keys(vi))
    return TPParams(
        μ     = v["inter.μ"],
        log_r = v["disp.log_r"],
        γ     = v["ha.γ_global"],
        σ_a   = v["dyn.σ_a"],
        σ_d   = v["dyn.σ_d"],
        raw_a = Vector{Float64}(v["dyn.raw_a"]),
        raw_d = Vector{Float64}(v["dyn.raw_d"]),
    )
end

"""
    tp_prior_draw(model, fs; seed) -> NamedTuple

One seeded prior draw, returned in both representations: the flat vector
DynamicPPL scores, and the named `TPParams` l02 scores.
"""
function tp_prior_draw(model, fs; seed::Int = 20260825)
    turing_model = PGm.build_turing_model(model, fs)
    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(turing_model)
    turing_model(vi)

    return (
        turing_model = turing_model,
        vi           = vi,
        θ            = copy(vi[:]),
        params       = tp_params_from_varinfo(vi),
    )
end

"""
    tp_logdensity_fn(turing_model) -> Function

`θ -> log density`, in the model's ORIGINAL space.

`DynamicPPL.VarInfo(model)` is unlinked, so no Jacobian corrections appear and the
Gamma-distributed scales are scored directly. Verified 2026-08-25: parity with
`tp_logjoint` is exact, which would not hold if the space were linked.
"""
function tp_logdensity_fn(turing_model)
    lf = DynamicPPL.LogDensityFunction(turing_model)
    return x -> LogDensityProblems.logdensity(lf, x)
end


# ==============================================================================
# 2. GATE 3a — Equation parity
# ==============================================================================

"""
    tp_gate_equation_parity(model, fs; seeds)

Compare DynamicPPL's log density against `l02_equations.jl` at several independent
prior draws.

Several draws, not one: a single point can agree by coincidence — for instance if
a term that should scale a parameter happens to be near 1. Independent draws are
used rather than perturbations of one draw so that no assumption is made about the
ordering of the flat parameter vector.
"""
function tp_gate_equation_parity(model, fs; seeds = [20260825, 7, 991])
    tp_assert_default(model)
    data = tp_equation_data(fs)
    hl   = model.dynamics_config.days_half_life

    diffs = Float64[]
    for s in seeds
        draw = tp_prior_draw(model, fs; seed = s)
        f    = tp_logdensity_fn(draw.turing_model)
        push!(diffs, f(draw.θ) - tp_logjoint(draw.params, data, model))
    end

    worst = maximum(abs, diffs)
    results = [(
        name   = "log density parity (Turing vs l02)",
        pass   = worst <= 1e-8,
        detail = @sprintf("max |Δ| = %.3e over %d prior draws", worst, length(seeds)),
    )]

    # The sampled-site manifest must match what MODEL.md claims and what gate 4
    # will later read out of the chain.
    draw = tp_prior_draw(model, fs; seed = first(seeds))
    observed = Set(string.(keys(draw.vi)))
    expected = Set(["inter.μ", "disp.log_r", "ha.γ_global",
                    "dyn.σ_a", "dyn.σ_d", "dyn.raw_a", "dyn.raw_d"])
    push!(results, (
        name   = "sampled-site manifest",
        pass   = observed == expected,
        detail = observed == expected ? "$(length(observed)) sites, as documented" :
                 "unexpected: $(sort(collect(symdiff(observed, expected))))",
    ))

    n_teams = Int(fs.data[:n_teams])
    push!(results, (
        name   = "parameter count",
        pass   = length(draw.θ) == 5 + 2 * n_teams,
        detail = "$(length(draw.θ)) = 5 scalars + 2 x $n_teams team effects",
    ))

    return results
end


# ==============================================================================
# 3. GATE 3b — Gradient health
# ==============================================================================

_tp_relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

"""
    tp_gate_gradients(model, fs; seed, probes) -> (results, artifacts)

Four independent routes to the same gradient:

  fresh ReverseDiff     re-records control flow every call
  compiled ReverseDiff  the tape NUTS will actually replay
  ForwardDiff           a structurally different AD mode
  finite differences    no AD at all

A compiled tape is a STATIC recording. If the model contains a data-dependent
branch, the tape silently keeps the branch taken at record time and stays wrong at
other parameter values — which is why agreement is also checked at perturbed
points, not only where the tape was recorded.

Thresholds follow the repository's AD guide: fresh/compiled 1e-8, AD-mode 1e-6,
finite differences 1e-4.
"""
function tp_gate_gradients(model, fs; seed::Int = 20260825, probes = [0.001, -0.002, 0.003])
    draw = tp_prior_draw(model, fs; seed = seed)
    f    = tp_logdensity_fn(draw.turing_model)
    θ    = draw.θ

    results = []

    push!(results, (
        name   = "log density finite at prior draw",
        pass   = isfinite(f(θ)),
        detail = @sprintf("logdensity = %.4f", f(θ)),
    ))

    compile_s = @elapsed tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))

    g_compiled = similar(θ)
    ReverseDiff.gradient!(g_compiled, tape, θ)
    g_fresh   = ReverseDiff.gradient(f, θ)
    g_forward = ForwardDiff.gradient(f, θ)

    push!(results, (
        name   = "gradients finite",
        pass   = all(isfinite, g_compiled) && all(isfinite, g_fresh) && all(isfinite, g_forward),
        detail = "compiled / fresh / ForwardDiff all finite",
    ))

    e_tape = _tp_relerr(g_fresh, g_compiled)
    push!(results, (
        name   = "compiled tape == fresh ReverseDiff",
        pass   = e_tape <= 1e-8,
        detail = @sprintf("relerr = %.3e", e_tape),
    ))

    e_mode = _tp_relerr(g_compiled, g_forward)
    push!(results, (
        name   = "ReverseDiff == ForwardDiff",
        pass   = e_mode <= 1e-6,
        detail = @sprintf("relerr = %.3e", e_mode),
    ))

    # Finite differences at indices spread across the parameter vector.
    n_teams = Int(fs.data[:n_teams])
    idx = unique([1, 4, 6, 6 + n_teams ÷ 2, length(θ)])
    worst_fd = 0.0
    for k in idx
        ε = 1e-6
        xp = copy(θ); xp[k] += ε
        xm = copy(θ); xm[k] -= ε
        central = (f(xp) - f(xm)) / (2ε)
        worst_fd = max(worst_fd, abs(g_compiled[k] - central))
    end
    push!(results, (
        name   = "finite differences agree",
        pass   = worst_fd <= 1e-4,
        detail = @sprintf("max |Δ| = %.3e at indices %s", worst_fd, string(idx)),
    ))

    # Static-tape safety away from the recording point.
    worst_probe = 0.0
    for δ in probes
        θp = θ .+ δ .* sin.(collect(eachindex(θ)))
        gp = similar(θp)
        ReverseDiff.gradient!(gp, tape, θp)
        worst_probe = max(worst_probe, _tp_relerr(ReverseDiff.gradient(f, θp), gp))
    end
    push!(results, (
        name   = "static tape safe at perturbed points",
        pass   = worst_probe <= 1e-8,
        detail = @sprintf("max relerr = %.3e over %d probes", worst_probe, length(probes)),
    ))

    # Timing is reported, never gating: a slow model is a cost, a wrong one is a bug.
    g_bench = similar(θ)
    times = Float64[]
    for _ in 1:50
        push!(times, @elapsed ReverseDiff.gradient!(g_bench, tape, θ))
    end
    med_ms = median(times) * 1e3
    push!(results, (
        name   = "compiled gradient latency",
        pass   = true,
        detail = @sprintf("median %.3f ms (compile %.2f s) — target < 1 ms", med_ms, compile_s),
    ))

    return (results, (; θ, tape, f, median_ms = med_ms))
end
