#=
LOADER l02 — MPT strategies as staking_layer POLICIES + the paper's evaluation protocol.

Makes every solver in l01_mpt_portfolio.jl runnable inside the existing `run_race` harness, so
the MPT/Markowitz family is compared against the incumbent Kelly policies on the SAME OOS stream,
the same trust model, the same cap, and the same reporting — a clean solver A/B rather than a
parallel backtest with its own subtly different accounting.

THE IMPORTANT DESIGN POINT
`MPTPolicy` reuses UnifiedPolicy's preprocessing verbatim:

    trust-blend targets → coherent IPF tilt → tilted 144-state p → SOLVER → fraction ω

Only the final step differs. That matters for correctness, not just tidiness: on a smile engine
the plain grid's totals marginals do NOT match the smile PPD, so solving on `m.pbar` invents edge
on the O/U ladder. The tilt imprints the model's actual per-line probabilities onto the grid
first, which is why every policy here goes through it. If you solve on raw `pbar` you get the
inflated-growth artefact.

    l01_mpt_portfolio.jl  must be included first (solvers)
    staking_layer/src/loader.jl  must be included first (tilt, trust, run_race)
=#

using Statistics, Random, Printf

# ------------------------------------------------------------------
# 1. The policy
# ------------------------------------------------------------------

"""
    MPTPolicy(; solve, opts, trust, cap, ω, cycles)

A staking policy whose portfolio step is any `(p, R; cap, kwargs...) -> f` solver.

  `solve`   the solver function (solve_P, solve_msharpe, solve_mpt, ...)
  `opts`    NamedTuple of solver-specific kwargs, e.g. `(γ=0.5,)` or `(α=0.3, β=0.1)`
  `trust`   any AbstractTrustModel — hold this FIXED across the roster to isolate the solver
  `cap`     portfolio cap Σf ≤ cap
  `ω`       §5.2 fractioning applied AFTER the solve (1.0 = no fractioning)
  `cycles`  IPF cycles for the coherent tilt (50 is the certified value)
"""
Base.@kwdef struct MPTPolicy <: AbstractStakingPolicy
    solve::Function = solve_P
    opts::NamedTuple = NamedTuple()
    trust::AbstractTrustModel = CuratedTrust()
    cap::Float64 = 0.2
    ω::Float64 = 1.0
    cycles::Int = 50
end

needs_trust(::MPTPolicy) = true

function stake_for(p::MPTPolicy, m::StakingMatch, model_sel, model_dists, fitted;
                   rng=Random.default_rng())
    w    = trust_weights(fitted, m)
    mult = coherent_multiplier(m.pbar, blend_targets(model_sel, m.q_mkt, w); cycles=p.cycles)
    pv   = normalize_mult(m.pbar, mult)
    f    = p.solve(pv, m.R; cap=p.cap, p.opts...)
    return frac(f, p.ω)
end

# ------------------------------------------------------------------
# 2. The roster
# ------------------------------------------------------------------

"""
    mpt_roster(; trust=CuratedTrust(), cap=0.2, iters=1200)

The paper's Table 1 strategies, all on one trust model and one cap so the ONLY difference is the
optimiser. `iters` is lowered from the solver default (4000) because the race re-solves for every
match × policy; the ascent exits on its own tolerance well before that in practice.
"""
function mpt_roster(; trust=CuratedTrust(), cap::Float64=0.2, iters::Int=1200)
    P(solve, opts=NamedTuple(); ω=1.0) =
        MPTPolicy(solve=solve, opts=merge(opts, (iters=iters,)), trust=trust, cap=cap, ω=ω)
    return [
        "Kelly"          => P(solve_P),
        "KellyFrac(.5)"  => P(solve_P; ω=0.5),
        "QuadKelly"      => P(solve_quad_kelly),
        "MPT(γ=0.5)"     => P(solve_mpt, (γ=0.5,)),
        "MPT(γ=2)"       => P(solve_mpt, (γ=2.0,)),
        "MPT(γ=10)"      => P(solve_mpt, (γ=10.0,)),
        "MSharpe"        => P(solve_msharpe),
        "MSharpeFrac"    => P(solve_msharpe; ω=0.5),
        "KellyDrawdown"  => P(solve_kelly_drawdown, (α=0.3, β=0.1, outer=4)),
        "KellyRobust.05" => P(solve_kelly_dro, (η=0.05,)),
        "KellyRobust.10" => P(solve_kelly_dro, (η=0.10,)),
    ]
end

"Reference policies from the incumbent staking layer, for context in the same table."
reference_roster(; cap::Float64=0.2) = [
    "FLAT_1pct"   => FlatPolicy(),
    "PB_BK_cap02" => PerBetKellyPolicy(cap=cap),
]

# ------------------------------------------------------------------
# 3. The paper's evaluation protocol (§6.2)
# ------------------------------------------------------------------

"""
    bootstrap_wealth(logw; reps=1000, drop=0.1, ruin_floor=1e-4, seed=1)

Paper §6.2: 1000 runs, each a random reshuffle of the match sequence with 10% of matches dropped;
report the distribution of final wealth plus ruin %. Ruin = wealth touching 0.01% of the initial
bank at any point (§6.2.2).

APPROXIMATION, stated plainly: this reshuffles a FIXED per-match log-return sequence rather than
re-solving under each new order. Stakes are order-independent except through the trust refit
cadence, so this captures sequencing risk (the thing the protocol is for) but not the second-order
effect of the trust model seeing a different history. Re-solving per shuffle is ~100× the cost;
do that only if a result turns on it.
"""
function bootstrap_wealth(logw::Vector{Float64}; reps::Int=1000, drop::Float64=0.1,
                          ruin_floor::Float64=1e-4, seed::Int=1)
    n = length(logw)
    k = max(round(Int, n * (1 - drop)), 1)
    rng = Xoshiro(seed)
    finals = Vector{Float64}(undef, reps)
    ruins = 0
    idx = collect(1:n)
    for r in 1:reps
        shuffle!(rng, idx)
        cw = cumsum(view(logw, view(idx, 1:k)))
        finals[r] = exp(cw[end])
        minimum(cw) < log(ruin_floor) && (ruins += 1)
    end
    return (median=median(finals), mean=mean(finals), min=minimum(finals), max=maximum(finals),
            sigma=std(finals), q5=quantile(finals, 0.05), ruin_pct=100 * ruins / reps)
end

"""
    protocol_rows(rs; reps=1000, drop=0.1)

Table in the paper's format (§6.2.2) for every policy in a completed race, plus the §6.2.1
selection flag: eligible iff Q5 > 0.9, and among eligible the best median(W_f) wins.
"""
function protocol_rows(rs; reps::Int=1000, drop::Float64=0.1, seed::Int=1)
    rows = String[]
    push!(rows, @sprintf("%-16s %10s %10s %10s %10s %10s %8s %6s",
                         "strategy", "median(Wf)", "mean(Wf)", "min(Wi)", "max(Wi)",
                         "sigma(Wf)", "ruin %", "Q5>.9"))
    stats = Dict{String,Any}()
    for nm in rs.names
        b = bootstrap_wealth(rs.logw[nm]; reps=reps, drop=drop, seed=seed)
        stats[nm] = b
        push!(rows, @sprintf("%-16s %10.4f %10.4f %10.2e %10.2f %10.4f %8.1f %6s",
                             nm, b.median, b.mean, b.min, b.max, b.sigma, b.ruin_pct,
                             b.q5 > 0.9 ? "yes" : "-"))
    end
    elig = [nm for nm in rs.names if stats[nm].q5 > 0.9]
    push!(rows, "")
    if isempty(elig)
        push!(rows, "§6.2.1 selection: NO strategy meets Q5 > 0.9 — none is safe on this book.")
    else
        best = elig[argmax([stats[nm].median for nm in elig])]
        push!(rows, @sprintf("§6.2.1 selection: %s  (median Wf = %.4f, Q5 = %.4f)",
                             best, stats[best].median, stats[best].q5))
    end
    return rows
end

# ------------------------------------------------------------------
# 4. Coherence guard
# ------------------------------------------------------------------

"""
    coherence_report(loaded; n=25, cycles=50)

Diagnostic for the artefact that inflates growth on smile engines: how far the PLAIN grid's
per-selection marginals sit from the model's actual per-line probabilities, and how much implied
edge that gap manufactures. Run this BEFORE trusting any race result.

`raw_edge` uses `MMASK' * pbar` (wrong on a smile engine); `tilted_edge` uses the coherent grid
(right). A large gap on the OverUnder rows is the smoking gun.
"""
function coherence_report(loaded; n::Int=25, cycles::Int=50)
    ms = loaded.matches[1:min(n, length(loaded.matches))]
    raw_gap = zeros(11); raw_b = zeros(11); tilt_b = zeros(11); cnt = zeros(Int, 11)
    for (i, m) in enumerate(ms)
        msel = loaded.model_sel[i]
        praw = MMASK' * m.pbar
        mult = coherent_multiplier(m.pbar, blend_targets(msel, m.q_mkt, ones(7)); cycles=cycles)
        ptil = MMASK' * normalize_mult(m.pbar, mult)
        for k in 1:11
            m.d[k] > 1.0 || continue
            raw_gap[k] += abs(praw[k] - msel[k])
            raw_b[k]   += praw[k] * m.d[k] - 1.0
            tilt_b[k]  += ptil[k] * m.d[k] - 1.0
            cnt[k] += 1
        end
    end
    safe(v) = [cnt[k] > 0 ? v[k] / cnt[k] : NaN for k in 1:11]
    rows = String[]
    push!(rows, @sprintf("%-10s %10s %12s %12s %7s", "sel", "|grid−PPD|", "raw_edge", "tilted_edge", "n"))
    for k in 1:11
        push!(rows, @sprintf("%-10s %10.4f %+12.4f %+12.4f %7d",
                             SEL_NAMES[k], safe(raw_gap)[k], safe(raw_b)[k], safe(tilt_b)[k], cnt[k]))
    end
    return rows
end
