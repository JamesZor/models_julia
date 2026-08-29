# ==============================================================================
# 05 — Composable Count Model Builder : THE PARITY HARNESS
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# WHAT IT IS FOR. A composable builder is only worth having if the model it
# assembles is the SAME MODEL as the hand-written one it replaces. "Looks
# equivalent" is not a claim; these are.
#
#   cb_layout_parity     the two models declare the same parameters in the same
#                        ORDER, so a θ vector means the same thing to both
#   cb_density_parity    evaluated at the SAME θ, the two log-densities agree
#   cb_clamp_headroom    the [-10, 10] clamp is inactive at the compared draws,
#                        so parity with arm 00 (which has no clamp) is exact and
#                        not an artefact of both saturating
#   cb_gradient_report   compiled-tape gradient health and latency
#   cb_posterior_parity  same seed, same sampler, same fold → same chain
#
# Feeding ONE θ to BOTH models is the whole trick. Drawing separately from each
# and comparing summaries would only show the two densities are similar; this
# shows they are the same function.
#
# ==============================================================================

using Random
using Statistics
using LinearAlgebra
using Printf
using DataFrames
import DynamicPPL          # see l03_engine.jl — `using` would shadow DataFrames.subset
using LogDensityProblems
using ReverseDiff
using ForwardDiff
using MCMCChains
using Turing

include(joinpath(@__DIR__, "l04_equations.jl"))


# ==============================================================================
# 1. Draws and densities
# ==============================================================================

"One seeded prior draw, in the model's own (unlinked) parameter space."
function cb_prior_draw(model, feature_set; seed::Int = 20260825)
    tm = CB_PG.build_turing_model(model, feature_set)
    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(tm)
    tm(vi)
    return (; tm, vi, θ = copy(vi[:]), sites = collect(string.(keys(vi))))
end

"`θ -> logdensity`, unlinked, no Jacobian."
function cb_logdensity_fn(tm)
    lf = DynamicPPL.LogDensityFunction(tm)
    return x -> LogDensityProblems.logdensity(lf, x)
end

"""
Site names differ where they legitimately differ: the hand-written arms sample a
top-level `w_wealth`, the builder samples `wealth.w` inside a named covariate
submodel. Everything else must match exactly, including order.
"""
const CB_SITE_ALIASES = Dict("w_wealth" => "wealth.w", "w_dist" => "distance.w")

cb_canonical_sites(sites) = [get(CB_SITE_ALIASES, s, s) for s in sites]


# ==============================================================================
# 2. Layout, density, and clamp-headroom parity
# ==============================================================================

"""
    cb_layout_parity(builder_model, arm_model, fs) -> result row

Both models must place the same parameters at the same indices of θ. If they do
not, the density comparison below is meaningless — it would be evaluating two
different functions at two different points and reporting the difference of the
values.
"""
function cb_layout_parity(builder_model, arm_model, feature_set; seed::Int = 20260825)
    b = cb_prior_draw(builder_model, feature_set; seed)
    a = cb_prior_draw(arm_model, feature_set; seed)
    same_sites = cb_canonical_sites(b.sites) == cb_canonical_sites(a.sites)
    same_len   = length(b.θ) == length(a.θ)
    detail = same_sites && same_len ?
        "$(length(b.θ)) parameters, $(length(b.sites)) grouped sites, identical order" :
        "builder $(cb_canonical_sites(b.sites)) vs arm $(cb_canonical_sites(a.sites))"
    return cb_result("θ layout identical", same_sites && same_len, detail)
end

"""
    cb_density_parity(builder_model, arm_model, fs; seeds) -> result row

Draw θ from the builder model, evaluate BOTH log-densities at that same θ, over
several prior draws. Reports the worst absolute difference.
"""
function cb_density_parity(builder_model, arm_model, feature_set;
                           seeds = [20260825, 7, 991, 20240101])
    tm_arm = CB_PG.build_turing_model(arm_model, feature_set)
    f_arm  = cb_logdensity_fn(tm_arm)

    diffs = Float64[]
    for s in seeds
        d = cb_prior_draw(builder_model, feature_set; seed = s)
        push!(diffs, cb_logdensity_fn(d.tm)(d.θ) - f_arm(d.θ))
    end
    worst = maximum(abs, diffs)
    return cb_result("log density == hand-written arm", worst == 0.0,
        @sprintf("max |Δ| = %.3e over %d prior draws%s", worst, length(seeds),
                 worst == 0.0 ? " (bit-identical)" : ""))
end

"""
    cb_reference_parity(model, fs; seeds) -> result row

The builder's engine against `l04_equations.jl`, the independent re-derivation.
This is the check that the ENGINE is right; `cb_density_parity` is the check that
it is the same as the arm. Both matter: agreeing with a wrong arm is not a result.
"""
function cb_reference_parity(model, feature_set; seeds = [20260825, 7, 991, 20240101])
    data  = cb_equation_data(model, feature_set)
    diffs = Float64[]
    for s in seeds
        d = cb_prior_draw(model, feature_set; seed = s)
        push!(diffs, cb_logdensity_fn(d.tm)(d.θ) -
                     cb_logjoint(model, cb_params_from_varinfo(model, d.vi), data))
    end
    worst = maximum(abs, diffs)
    return cb_result("log density == independent reference", worst <= 1e-9,
        @sprintf("max |Δ| = %.3e over %d prior draws", worst, length(seeds)))
end

"""
    cb_clamp_headroom(model, fs; seeds) -> result row

Arm 00's engine has no `clamp`; the composable engine always clamps to [-10, 10].
Those are the same function only where the clamp does not bind. This reports the
largest |η| reached at the compared draws, so "bit-identical" is a statement about
the region actually tested rather than a hope.
"""
function cb_clamp_headroom(model, feature_set; seeds = [20260825, 7, 991, 20240101])
    data  = cb_equation_data(model, feature_set)
    worst = 0.0
    for s in seeds
        d = cb_prior_draw(model, feature_set; seed = s)
        p = cb_params_from_varinfo(model, d.vi)
        α, β = cb_team_effects(p)
        n = length(data.home)
        q_h = zeros(n); q_a = zeros(n)
        for k in eachindex(data.x)
            qh, qa = covariate_sides(data.roles[k], p.w[k] .* data.x[k])
            q_h .+= qh; q_a .+= qa
        end
        η_h = p.μ .+ p.γ .+ α[data.home] .+ β[data.away] .+ q_h
        η_a = p.μ        .+ α[data.away] .+ β[data.home] .+ q_a
        worst = max(worst, maximum(abs, η_h), maximum(abs, η_a))
    end
    return cb_result("clamp never binds at compared draws", worst < 10.0,
        @sprintf("max |η| = %.3f against a bound of 10.0", worst))
end


# ==============================================================================
# 3. Gradient health and latency
# ==============================================================================

_cb_relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

"""
    cb_gradient_report(model, fs; seed, reps) -> (rows, median_ms)

The AD guide's contract, checked rather than assumed: a compiled ReverseDiff tape
that agrees with a fresh tape and with ForwardDiff, stays correct at perturbed
points, and evaluates in under a millisecond.

The perturbed-point probe is the one that matters for a COMPOSABLE engine. A
compiled tape silently follows whichever branch it recorded, so a design that
smuggled a value-dependent branch into the covariate walk would pass every other
check here and produce quietly wrong gradients. It does not, because the walk is
unrolled on TYPES.
"""
function cb_gradient_report(model, feature_set; seed::Int = 20260825, reps::Int = 50,
                            probes = [0.001, -0.002, 0.003])
    d = cb_prior_draw(model, feature_set; seed)
    f, θ = cb_logdensity_fn(d.tm), d.θ

    compile_s = @elapsed tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))
    g_tape = similar(θ); ReverseDiff.gradient!(g_tape, tape, θ)
    g_fresh   = ReverseDiff.gradient(f, θ)
    g_forward = ForwardDiff.gradient(f, θ)

    rows = Any[
        cb_result("log density finite", isfinite(f(θ)), @sprintf("%.4f", f(θ))),
        cb_result("gradients finite",
                  all(isfinite, g_tape) && all(isfinite, g_fresh) && all(isfinite, g_forward),
                  "compiled / fresh / ForwardDiff"),
        cb_result("compiled tape == fresh ReverseDiff", _cb_relerr(g_fresh, g_tape) <= 1e-8,
                  @sprintf("relerr = %.3e", _cb_relerr(g_fresh, g_tape))),
        cb_result("ReverseDiff == ForwardDiff", _cb_relerr(g_tape, g_forward) <= 1e-6,
                  @sprintf("relerr = %.3e", _cb_relerr(g_tape, g_forward))),
    ]

    worst_probe = 0.0
    for δ in probes
        θp = θ .+ δ .* sin.(collect(eachindex(θ)))
        gp = similar(θp); ReverseDiff.gradient!(gp, tape, θp)
        worst_probe = max(worst_probe, _cb_relerr(ReverseDiff.gradient(f, θp), gp))
    end
    push!(rows, cb_result("static tape safe at perturbed points", worst_probe <= 1e-8,
        @sprintf("max relerr = %.3e over %d probes", worst_probe, length(probes))))

    g = similar(θ)
    times = [@elapsed ReverseDiff.gradient!(g, tape, θ) for _ in 1:reps]
    median_ms = median(times) * 1e3
    push!(rows, cb_result("compiled gradient < 0.65 ms", median_ms < 0.65,
        @sprintf("median %.3f ms over %d reps (tape compile %.2f s)", median_ms, reps, compile_s)))

    return (rows, median_ms)
end


# ==============================================================================
# 4. Posterior parity
# ==============================================================================

"""
    cb_nuts(model, fs; seed, warmup, samples, chains) -> Chains

`chains` NUTS chains from one explicitly seeded RNG, sampled in parallel.

Several chains rather than one, because everything §9 compares is a Monte-Carlo
standard error, and a single-chain MCSE for a slowly-mixing scale parameter is
itself so noisy that it can fail a model against a copy of itself. Several chains
also make R-hat available, so "the two posteriors differ" and "neither chain has
converged" stop being the same observation.

The two models are NOT started from a shared point. They name their covariate
sites differently (`wealth.w` versus `w_wealth`), so a shared start would have to
be translated between two schemas — and independent starts are the stronger test:
chains that begin in different places and land on the same posterior have shown
something a shared start cannot.
"""
function cb_nuts(model, feature_set; seed::Int = 20260825, warmup::Int = 500,
                 samples::Int = 1000, accept::Float64 = 0.65, max_depth::Int = 10,
                 chains::Int = 4)
    tm  = CB_PG.build_turing_model(model, feature_set)
    alg = NUTS(warmup, accept; max_depth = max_depth,
               adtype = Turing.AutoReverseDiff(compile = true))
    return sample(Random.Xoshiro(seed), tm, alg, MCMCThreads(), warmup + samples, chains;
                  progress = false, verbose = false)
end

"""
    cb_convergence_row(label, chain) -> result row

R-hat across chains. Reported before any comparison, because a disagreement
between two unconverged chains says nothing about the two models.
"""
function cb_convergence_row(label::AbstractString, chain::Chains)
    stats = DataFrame(MCMCChains.ess_rhat(chain))
    rhat  = collect(skipmissing(stats.rhat))
    ess   = collect(skipmissing(stats.ess))
    worst = isempty(rhat) ? NaN : maximum(rhat)
    # REPORTED, not asserted. A model cannot be blamed for a sampler that failed to
    # converge on it, and §7 has already proved the two targets are the same
    # function; what convergence decides here is which parameters are eligible to
    # be COMPARED, which `cb_posterior_parity` handles by excluding the rest.
    return cb_result(label, true,
        @sprintf("max R-hat = %.4f%s, min ESS = %.0f, median ESS = %.0f over %d chains",
                 worst, worst <= 1.01 ? "" : "  (NOT CONVERGED)",
                 minimum(ess), median(ess), size(chain, 3)))
end

"""
    cb_posterior_parity(chain_builder, chain_arm) -> (rows, comparison table)

Per-parameter posterior comparison. Site names are canonicalised first, so
`w_wealth` in the arm's chain is compared against `wealth.w` in the builder's.

Reports the strong claim (identical draws) and, separately, the weak one
(posterior means within a Monte-Carlo standard error), because only the second is
still meaningful if a future sampler change breaks bit-reproducibility.
"""
function cb_posterior_parity(chain_b::Chains, chain_a::Chains; z_max::Float64 = 4.0)
    names_b = cb_canonical_sites(String.(names(chain_b, :parameters)))
    names_a = cb_canonical_sites(String.(names(chain_a, :parameters)))
    shared  = intersect(names_b, names_a)

    rows = Any[cb_result("posterior parameter sets match",
        sort(names_b) == sort(names_a),
        "$(length(shared)) shared of $(length(names_b)) / $(length(names_a))")]

    draws_bld = Array(chain_b)
    draws_arm = Array(chain_a)
    idx_b = Dict(n => i for (i, n) in enumerate(names_b))
    idx_a = Dict(n => i for (i, n) in enumerate(names_a))
    mcse_b = _cb_mcse_map(chain_b)
    mcse_a = _cb_mcse_map(chain_a)
    rhat_b = _cb_rhat_map(chain_b)
    rhat_a = _cb_rhat_map(chain_a)

    table = DataFrame(parameter = String[], identified = Bool[], converged = Bool[],
                      mean_builder = Float64[], mean_arm = Float64[],
                      sd_builder = Float64[], sd_arm = Float64[],
                      abs_mean_diff = Float64[], se = Float64[], z = Float64[],
                      max_draw_diff = Float64[])
    for n in shared
        cb = draws_bld[:, idx_b[n]]
        ca = draws_arm[:, idx_a[n]]
        # Monte-Carlo standard error of the DIFFERENCE of two independent chains.
        # `std / sqrt(n_draws)` would be wrong by the square root of the
        # autocorrelation time — roughly an order of magnitude for `dyn.raw_*` —
        # and would fail a comparison of a model against ITSELF.
        se = sqrt(get(mcse_b, n, NaN)^2 + get(mcse_a, n, NaN)^2)
        d  = abs(mean(cb) - mean(ca))
        conv = get(rhat_b, n, Inf) <= 1.01 && get(rhat_a, n, Inf) <= 1.01
        push!(table, (n, cb_is_identified(n), conv, mean(cb), mean(ca), std(cb), std(ca),
                      d, se, d / se,
                      length(cb) == length(ca) ? maximum(abs, cb .- ca) : NaN))
    end

    worst_draw = isempty(table) ? NaN : maximum(table.max_draw_diff)
    push!(rows, cb_result("chains bit-identical (reported, not required)", true,
        worst_draw == 0.0 ? "identical draws" :
        @sprintf("max |Δ draw| = %.3e — expected: same density, different op order",
                 worst_draw)))

    return (rows, table)
end

"Per-parameter R-hat, keyed by canonical site name."
function _cb_rhat_map(chain::Chains)
    out = Dict{String, Float64}()
    stats = DataFrame(MCMCChains.ess_rhat(chain))
    for row in eachrow(stats)
        r = row.rhat
        out[first(cb_canonical_sites([String(row.parameters)]))] =
            (r === missing || isnan(r)) ? Inf : Float64(r)
    end
    return out
end

"""
    cb_is_identified(name) -> Bool

Is this a parameter the data actually pins down?

`dyn.raw_a[i]` / `dyn.raw_d[i]` are the non-centred z-scores of an exchangeable,
zero-sum team effect. Individually they sit close to their `Normal(0,1)` prior,
mix slowly, and their single-chain MCSE is the least reliable number in the
summary. A maximum taken over fifty of them is dominated by whichever one mixed
worst, not by whether two implementations agree — which is why §9 tests the
identified block and reports the rest through an aggregate instead.
"""
cb_is_identified(name::AbstractString) =
    !(occursin("raw_a[", name) || occursin("raw_d[", name) ||
      occursin("raw_month[", name) || occursin("_raw[", name))

"""
    cb_calibrated_row(z_test, z_null_builder, z_null_arm; floor_z) -> result row

The gate that decides §9.

A raw threshold on max |z| cannot be set from theory. The statistic is a maximum
over ~50 correlated parameters of a ratio whose denominator — a single-chain MCSE
for a non-centred z-score — is itself noisy, so its null distribution has a long
right tail and depends on how well each chain happens to mix.

So the null is MEASURED, twice: each model is compared against ITSELF at a second
seed, a comparison that is true by construction. On fold 1 those two floors are
genuinely different (the composable engine reaches roughly 1.4x the median ESS of
the hand-written one at the same budget), which is exactly why calibrating against
only one of them would be rigged. The claim is then falsifiable in the only way
that means anything here: the builder-versus-arm discrepancy must sit inside the
seed-to-seed noise the two models already produce on their own.
"""
function cb_calibrated_row(label::AbstractString, test::Float64,
                           null_b::Float64, null_a::Float64;
                           floor_value::Float64 = 4.0, fmt = "%.2f")
    null      = max(null_b, null_a)
    threshold = max(floor_value, null)
    return cb_result(label, test <= threshold,
        @sprintf("test %.3f; noise floor: builder-vs-builder %.3f, arm-vs-arm %.3f; threshold %.3f",
                 test, null_b, null_a, threshold))
end

"The three summary statistics §9 compares, all computed the same way for the test
pair and for each same-model control pair."
function cb_posterior_stats(table::DataFrame)
    # Only parameters BOTH chains converged on. Comparing posterior means from a
    # chain with R-hat 1.3 is comparing two numbers neither of which estimates
    # anything; excluding them is the correct handling, and the count is reported
    # so the exclusion is never silent.
    ok    = table[table.converged, :]
    ident = ok[ok.identified, :]
    return (
        max_z_identified = isempty(ident) ? 0.0 : maximum(ident.z),
        mean_z_comparable = isempty(ok) ? 0.0 : mean(ok.z),
        n_comparable = nrow(ok),
        n_total = nrow(table),
    )
end


# ==============================================================================
# 4b. Prediction parity
# ==============================================================================

"""
    cb_lambda_means(model, fs, df, chain) -> Matrix{Float64}

Posterior-mean (λ_h, λ_a) per fixture, straight out of `extract_parameters`.

This is the quantity the rest of the system consumes — the score grid, the book,
the stake — so it is the one worth comparing. It also exercises the DERIVED
extractor end to end: the covariate walk, the out-of-sample design columns, and
the chain-site names the builder invented for itself.
"""
function cb_lambda_means(model, feature_set, df, chain::Chains)
    priced = CB_PG.extract_parameters(model, df, feature_set, chain)
    out = Matrix{Float64}(undef, nrow(df), 2)
    for (i, row) in enumerate(eachrow(df))
        p = priced[Int(row.match_id)]
        out[i, 1] = mean(p.λ_h)
        out[i, 2] = mean(p.λ_a)
    end
    return out
end

"Relative differences between two sets of posterior-mean rates."
cb_lambda_reldiff(A::Matrix{Float64}, B::Matrix{Float64}) =
    abs.(A .- B) ./ ((abs.(A) .+ abs.(B)) ./ 2)

"Mean relative difference — stable under Monte-Carlo noise, unlike the maximum."
cb_lambda_discrepancy(A, B) = mean(cb_lambda_reldiff(A, B))

"""
    cb_extraction_parity(builder_model, arm_model, fs, df, chain) -> (row, maxrel)

The EXACT extraction test, and the one that carries the weight.

Both extractors are handed the SAME posterior draws — the builder's chain, with
its covariate sites renamed to the arm's convention — so Monte-Carlo noise is
removed entirely and any difference is arithmetic. The builder's `extract_parameters`
is fully derived (it walks the covariate tuple and reads site names it invented);
the arm's is hand-written. They must produce the same rates for the same fixtures.

The Monte-Carlo λ comparison alongside it can only ever say "within noise". This
says "identical".
"""
function cb_extraction_parity(builder_model, arm_model, feature_set, df, chain::Chains)
    aliases = Dict(v => k for (k, v) in CB_SITE_ALIASES)          # wealth.w -> w_wealth
    present = Dict(k => v for (k, v) in aliases if String(k) in String.(names(chain)))
    chain_arm = isempty(present) ? chain : MCMCChains.replacenames(chain, present)

    A = cb_lambda_means(builder_model, feature_set, df, chain)
    B = cb_lambda_means(arm_model,     feature_set, df, chain_arm)
    maxrel = maximum(cb_lambda_reldiff(A, B))
    return (cb_result("extraction: one chain, two extractors, same rates",
                      maxrel <= 1e-12,
                      @sprintf("max relative difference = %.3e over %d fixtures x 2 sides",
                               maxrel, size(A, 1))),
            maxrel)
end


"""
Per-parameter Monte-Carlo standard error, autocorrelation-aware. Falls back to the
naive `std / sqrt(n)` only where MCMCChains cannot produce an estimate, which makes
the test STRICTER there, never weaker.
"""
function _cb_mcse_map(chain::Chains)
    out = Dict{String, Float64}()
    stats = DataFrame(MCMCChains.mcse(chain))
    col = :mcse in propertynames(stats) ? :mcse : propertynames(stats)[2]
    for row in eachrow(stats)
        out[first(cb_canonical_sites([String(row.parameters)]))] = Float64(row[col])
    end
    draws = Array(chain)
    for (i, n) in enumerate(cb_canonical_sites(String.(names(chain, :parameters))))
        v = get(out, n, NaN)
        (isnan(v) || v <= 0.0) && (out[n] = std(draws[:, i]) / sqrt(size(draws, 1)))
    end
    return out
end



# ==============================================================================
# 5. Feature derivation
# ==============================================================================

"""
    cb_feature_parity(builder_model, arm_model) -> result row

`required_features` is DERIVED for the builder model and HAND-WRITTEN for the arm.
They must agree, or the two models are being fitted to different data and every
other comparison is void.
"""
function cb_feature_parity(builder_model, arm_model)
    fb = sort(String.(nameof.(typeof.(CB_Features.required_features(builder_model)))))
    fa = sort(String.(nameof.(typeof.(CB_Features.required_features(arm_model)))))
    return cb_result("required_features derived == hand-written", fb == fa,
        fb == fa ? join(fb, ", ") : "builder $(fb)\n            arm $(fa)")
end
