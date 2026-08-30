# ==============================================================================
# 06 — TYPED POSTERIOR LATENTS : THE PARITY AND ALLOCATION HARNESS
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# WHAT THIS FILE IS FOR
# ------------------------------------------------------------------------------
#
# A container swap has exactly one acceptance criterion: THE PRICES DO NOT MOVE.
# Not "the prices barely move", not "the prices agree to five decimal places" — a
# betting model whose numbers shift by 1e-9 across a refactor cannot be reconciled
# against its own history, and nobody will be able to say later whether an edge that
# disappeared was the market or the merge.
#
# So this file measures the distance between the typed path and the legacy path in
# UNITS OF LAST PLACE (ULP), not in absolute error. The difference matters:
#
#   * |Δ| < 1e-12 on a probability of 0.42 permits ~4500 ULP of drift. A systematic
#     reassociation would pass that threshold everywhere and still be a real change.
#   * 0 ULP means the two implementations produced the identical bit pattern. There
#     is nothing left to be uncertain about.
#
# The briefing asks for |Δ| < 1e-12. That threshold is reported and enforced, and the
# ULP column sits next to it so a passing-but-drifting row cannot hide.
#
# ------------------------------------------------------------------------------
# WHAT "LEGACY" MEANS HERE
# ------------------------------------------------------------------------------
#
# The live `src` code, called through its real entry points:
#
#     Predictions.extract_params(model, row)          src/predictions/interface.jl:10
#     Predictions.compute_score_matrix(model, params) src/predictions/score_computation/*
#     Predictions.compute_market_probs(S, market)     src/predictions/market_inference/*
#
# Not a transcription of them into this directory. If the `src` kernels change, these
# comparisons change with them and start failing, which is the entire value of
# pointing at the real functions.
#
# ==============================================================================

using DataFrames
using Dates
using MCMCChains
using Printf
using Random
using Statistics
using BayesianFootball

include(joinpath(@__DIR__, "l03_score_grids.jl"))


# ==============================================================================
# 1. FLOATING-POINT DISTANCE
# ==============================================================================

"""
    ulp_distance(a, b) -> Int64

Number of representable `Float64` values strictly between `a` and `b`, plus one; `0`
when the bit patterns are identical.

Works across zero and across the sign boundary by mapping each `Float64` to a
monotone integer key: non-negative floats keep their bit pattern (IEEE-754 orders them
the same way as their bits), negative floats are reflected through `typemin`. `-0.0`
and `+0.0` both map to `0`, so they are 0 ULP apart, which is the useful answer.

`NaN` against anything is `typemax(Int64)` — a parity harness must treat a NaN as
maximally wrong, never as "not comparable".
"""
@inline _ulp_key(x::Float64) = (b = reinterpret(Int64, x); b < 0 ? (typemin(Int64) - b) : b)

function ulp_distance(a::Float64, b::Float64)
    (isnan(a) || isnan(b)) && return typemax(Int64)
    a == b && return Int64(0)
    (isinf(a) || isinf(b)) && return typemax(Int64)
    d = abs(widen(_ulp_key(a)) - widen(_ulp_key(b)))
    return d > widen(typemax(Int64)) ? typemax(Int64) : Int64(d)
end


# ==============================================================================
# 2. THE REPORT ROW
# ==============================================================================

"""
    ParityRow

One comparison of two equally-shaped collections of `Float64`.

A row passes only when BOTH budgets are met: `max_abs <= tol` AND `max_ulp <= ulp_budget`.

THE ULP BUDGET IS THE ONE THAT BITES, and it exists because the absolute one does not.
A one-ULP perturbation of a single λ propagates into a score grid as roughly 1e-17 of
absolute error and 16 ULP — comfortably inside a 1e-12 tolerance, and unmistakable in
ULP. A harness that gated only on `tol` would report "pass" while the numbers had in
fact changed, which is precisely the failure this whole prototype is meant to make
impossible. (That scenario is not hypothetical: it is what an earlier version of this
file did, found by mutation-testing the harness against itself.)

`ulp_budget` defaults to 0 — bit-identity — and is a parameter rather than a constant
only so a future comparison that legitimately cannot be bit-exact (a different
summation order forced by a different algorithm, say) has to state its budget out loud
instead of loosening `tol` until the row goes green.

`n` guards against the most embarrassing way to pass a parity test, which is to compare
nothing at all: `tpl_parity_table` fails any row with `n == 0`.
"""
struct ParityRow
    check::String
    n::Int
    max_abs::Float64
    max_ulp::Int64
    tol::Float64
    ulp_budget::Int64
    pass::Bool
end

"""
    tpl_compare(check, a, b; tol = 1e-12, ulp_budget = 0) -> ParityRow

Compare two collections elementwise. Shape mismatch is a failure, not an error, so a
report can show every problem at once rather than stopping at the first.
"""
function tpl_compare(check::AbstractString, a, b;
                     tol::Float64 = 1e-12, ulp_budget::Integer = 0)
    ub = Int64(ulp_budget)
    if length(a) != length(b)
        return ParityRow(check, 0, Inf, typemax(Int64), tol, ub, false)
    end
    isempty(a) && return ParityRow(check, 0, 0.0, Int64(0), tol, ub, false)

    max_abs = 0.0
    max_ulp = Int64(0)
    for (x, y) in zip(a, b)
        fx = Float64(x)
        fy = Float64(y)
        d  = abs(fx - fy)
        (isnan(fx) || isnan(fy)) && (d = Inf)
        d > max_abs && (max_abs = d)
        u = ulp_distance(fx, fy)
        u > max_ulp && (max_ulp = u)
    end
    return ParityRow(check, length(a), max_abs, max_ulp, tol, ub,
                     max_abs <= tol && max_ulp <= ub)
end

"""
    tpl_parity_table(rows; title) -> Bool

Print the report and return whether everything passed. An empty comparison (`n == 0`)
is reported as a failure with an explicit marker: a harness that silently compares
zero elements is worse than one that fails.
"""
function tpl_parity_table(rows::Vector{ParityRow}; title::AbstractString = "PARITY")
    width = maximum(length(r.check) for r in rows; init = 20)
    rule  = "-"^(width + 52)
    println()
    println("  ", title)
    println("  ", rule)
    @printf("  %-*s %10s %12s %9s %8s  %s\n", width,
            "check", "compared", "max |Δ|", "max ULP", "ULP bgt", "verdict")
    println("  ", rule)
    ok = true
    for r in rows
        verdict = if r.n == 0
            "FAIL (nothing compared)"
        elseif r.pass
            "pass"
        elseif r.max_abs > r.tol
            "FAIL (|Δ| > tol)"
        else
            "FAIL (ULP over budget)"
        end
        r.pass && r.n > 0 || (ok = false)
        absstr = isfinite(r.max_abs) ? @sprintf("%.3e", r.max_abs) : "   Inf"
        ulpstr = r.max_ulp == typemax(Int64) ? "  NaN/Inf" : string(r.max_ulp)
        @printf("  %-*s %10d %12s %9s %8d  %s\n", width,
                r.check, r.n, absstr, ulpstr, r.ulp_budget, verdict)
    end
    println("  ", rule)
    return ok
end


# ==============================================================================
# 3. THE LEGACY LATENTS DATAFRAME, BUILT THE WAY `src` BUILDS IT
# ==============================================================================

"""
    tpl_legacy_latents_df(raw::Dict{Int,NamedTuple}, ids) -> DataFrame

Reproduce `Experiments._latent_state_dict_to_df` (src/experiments/post_processing.jl:205)
EXACTLY, including its `Vector{Any}` columns, in `ids` row order.

The `Vector{Any}` is not a detail to tidy up in passing — it is the thing under
measurement. A DataFrame built the obvious way here would have `Vector{Vector{Float64}}`
columns, would be faster and smaller than the one production actually holds, and would
quietly understate every number in the comparison. So this mirrors the real constructor,
right down to the column type.

(The row ORDER differs from the original, which iterates `keys(raw)`. That order is a
hash artefact and is not reproducible between runs; fixing it to `ids` is the one
deviation, and it cannot affect a per-row comparison.)
"""
function tpl_legacy_latents_df(raw::AbstractDict{Int, <:NamedTuple}, ids::Vector{Int})
    isempty(ids) && return DataFrame()
    cols = Dict{Symbol, Vector{Any}}(:match_id => Vector{Any}(ids))
    for p in keys(raw[first(ids)])
        cols[p] = Any[raw[id][p] for id in ids]
    end
    return DataFrame(cols)
end

"""
    tpl_dataframe_bytes(df) -> (bytes, objects)

Deep size of a latents DataFrame: the column vectors PLUS every posterior sample array
held in their cells, and a count of the distinct heap objects involved.

DISTINCT OBJECTS, COUNTED ONCE. Some engines put the SAME array object on every row —
the global smile shape `φ` is one matrix shared by every fixture
(goals_smile_league.jl:208-212). Counting it per row would inflate the legacy side by a
factor of `n_matches` and flatter the typed container by exactly the amount this
comparison is supposed to measure honestly. Identity is tracked by `objectid`, so a
shared array is charged once, which is what it actually costs.

`Base.summarysize` is not used because it also walks the DataFrame's index and metadata
machinery, which the typed container does not replace and should not be credited for.
"""
function tpl_dataframe_bytes(df::AbstractDataFrame)
    bytes = 0
    objects = 0
    seen = Set{UInt}()
    for col in eachcol(df)
        bytes += sizeof(col)
        objects += 1
        for cell in col
            cell isa AbstractArray || continue
            oid = objectid(cell)
            oid in seen && continue
            push!(seen, oid)
            bytes += sizeof(cell)
            objects += 1
        end
    end
    return (bytes, objects)
end


# ==============================================================================
# 4. PACKING PARITY  —  is the typed container the same numbers?
# ==============================================================================

"""
    parity_packing(l, raw, ids, field_map; tol) -> Vector{ParityRow}

Read every cell of every parameter matrix back against the source sample vectors.

`field_map` pairs a container matrix name with the legacy `NamedTuple` field it came
from, e.g. `[:λ_home => :λ_h, :λ_away => :λ_a]`.

This is the check that the (n_matches × n_draws) re-layout in `tpl_stack` did not
transpose, truncate, or misalign anything. It is a full sweep — `n_matches × n_draws`
comparisons per field — because a partial sweep would miss exactly the corner cases
(the last fixture, the last draw) that an off-by-one produces.
"""
function parity_packing(l::AbstractPosteriorLatents,
                        raw::AbstractDict{Int, <:NamedTuple},
                        ids::Vector{Int},
                        field_map::Vector{Pair{Symbol, Symbol}};
                        tol::Float64 = 1e-12)
    mats = latent_matrices(l)
    rows = ParityRow[]
    for (mat_name, legacy_name) in field_map
        M = mats[mat_name]
        typed  = Float64[]
        legacy = Float64[]
        sizehint!(typed,  length(M))
        sizehint!(legacy, length(M))
        for (i, id) in enumerate(ids)
            v = raw[id][legacy_name]
            for k in 1:size(M, 2)
                push!(typed,  M[i, k])
                push!(legacy, v[k])
            end
        end
        push!(rows, tpl_compare("pack $(mat_name) <- :$(legacy_name)", typed, legacy; tol))
    end
    return rows
end


# ==============================================================================
# 5. SCORE-GRID PARITY  —  do the two kernels produce the same tensor?
# ==============================================================================

"""
    legacy_score_tensor(model, legacy_df, i; max_goals) -> Array{Float64,3}

Run the production prediction path for one fixture:
`extract_params` -> `compute_score_matrix` -> the raw `[h × a × draws]` tensor.

Goes through `Predictions.score_matrix_data` rather than `.data`, so a
`SmileScoreMatrix` (whose grid is nested one level deeper) is unwrapped by the same
accessor the rest of the pipeline uses.
"""
function legacy_score_tensor(model, legacy_df::AbstractDataFrame, i::Integer;
                             max_goals::Integer = TPL_MAX_GOALS)
    row    = legacy_df[i, :]
    params = TPL_Pred.extract_params(model, row)
    S      = TPL_Pred.compute_score_matrix(model, params; max_goals = max_goals)
    return TPL_Pred.score_matrix_data(S)
end

"""
    parity_score_grids(model, l, legacy_df; fixtures, max_goals, tol) -> Vector{ParityRow}

Compare the typed kernel's grid against the legacy kernel's, fixture by fixture.

Reported per fixture rather than pooled. A pooled maximum tells you something is
wrong; a per-fixture table tells you WHICH fixture, and whether the failure is one
outlier or all of them — which is the difference between a bug in the kernel and a bug
in one fixture's parameters.
"""
function parity_score_grids(model, l::AbstractPosteriorLatents, legacy_df::AbstractDataFrame;
                            fixtures = 1:n_matches(l),
                            max_goals::Integer = TPL_MAX_GOALS,
                            tol::Float64 = 1e-12)
    ws   = GridWorkspace(max_goals)
    S    = alloc_score_grid(l, max_goals)
    rows = ParityRow[]
    for i in fixtures
        compute_score_grid!(S, ws, l, Int(i))
        L = legacy_score_tensor(model, legacy_df, i; max_goals = max_goals)
        push!(rows, tpl_compare("grid  fixture $(l.match_ids[i])", S, L; tol))
    end
    return rows
end

"""
    parity_grid_mass(l; fixtures, max_goals) -> Vector{ParityRow}

Independent sanity check that has nothing to do with the legacy path: does each
`12×12` slice sum to a probability?

A truncated grid sums to `P(H ≤ 11)·P(A ≤ 11)`, i.e. slightly below 1 (the
recombination family renormalises and sums to exactly 1 — see l03 §4). Either way it
must be in `(0, 1]`. This catches a class of failure a parity test cannot: if BOTH
kernels were wrong in the same way, they would agree at 0 ULP and this row would still
fail.
"""
function parity_grid_mass(l::AbstractPosteriorLatents;
                          fixtures = 1:n_matches(l),
                          max_goals::Integer = TPL_MAX_GOALS)
    ws   = GridWorkspace(max_goals)
    S    = alloc_score_grid(l, max_goals)
    rows = ParityRow[]
    worst_low  = 1.0
    worst_high = 0.0
    n = 0
    for i in fixtures
        compute_score_grid!(S, ws, l, Int(i))
        for k in 1:size(S, 3)
            m = sum(view(S, :, :, k))
            worst_low  = min(worst_low, m)
            worst_high = max(worst_high, m)
            n += 1
        end
    end
    ok = worst_low > 0.0 && worst_high <= 1.0 + 1e-12
    push!(rows, ParityRow(@sprintf("grid mass in (0,1]  [%.9f, %.9f]", worst_low, worst_high),
                          n, ok ? 0.0 : 1.0, Int64(0), 1e-12, Int64(0), ok))
    return rows
end


# ==============================================================================
# 6. MARKET-PRICE PARITY  —  the number that actually gets staked
# ==============================================================================

"""
    parity_market_prices(model, l, legacy_df, markets; fixtures, max_goals, tol)

Compare `price_market` against `Predictions.compute_market_probs` for every
(fixture, market, outcome).

Compared per OUTCOME, not per market: 1X2 home and 1X2 away accumulate over disjoint
triangles of the grid in different orders, and pooling them would let a failure in one
be masked by the other's scale.

Note the legacy comparison is made against the SAME container type the production path
would build — a `SmileScoreMatrix` when the model is a smile engine — so the O/U route
being compared is the smile route, not the grid fallback.
"""
function parity_market_prices(model, l::AbstractPosteriorLatents,
                              legacy_df::AbstractDataFrame, markets;
                              fixtures = 1:n_matches(l),
                              max_goals::Integer = TPL_MAX_GOALS,
                              tol::Float64 = 1e-12)
    ws   = GridWorkspace(max_goals)
    S    = alloc_score_grid(l, max_goals)
    rows = ParityRow[]

    # Accumulate per (market, outcome) across fixtures; one row each.
    acc_typed  = Dict{String, Vector{Float64}}()
    acc_legacy = Dict{String, Vector{Float64}}()
    order      = String[]

    for i in fixtures
        typed_container = _tpl_typed_container(l, S, ws, Int(i))

        legacy_row    = legacy_df[i, :]
        legacy_params = TPL_Pred.extract_params(model, legacy_row)
        legacy_S      = TPL_Pred.compute_score_matrix(model, legacy_params; max_goals = max_goals)

        for m in markets
            typed_probs  = price_market(typed_container, m)
            legacy_probs = TPL_Pred.compute_market_probs(legacy_S, m)
            for sel in market_keys(m)
                key = "$(TPL_D.market_group(m)) $(TPL_D.market_line(m)) :$(sel)"
                haskey(acc_typed, key) || (push!(order, key);
                                           acc_typed[key]  = Float64[];
                                           acc_legacy[key] = Float64[])
                append!(acc_typed[key],  typed_probs[sel])
                append!(acc_legacy[key], legacy_probs[sel])
            end
        end
    end

    for key in order
        push!(rows, tpl_compare(key, acc_typed[key], acc_legacy[key]; tol))
    end
    return rows
end

"Build the pricing container the typed path would hand to `price_market`."
function _tpl_typed_container(l::AbstractPosteriorLatents, S, ws, i::Int)
    compute_score_grid!(S, ws, l, i)
    return S
end

function _tpl_typed_container(l::SmileLatents, S, ws, i::Int)
    compute_score_grid!(S, ws, l, i)
    buf = alloc_smile_buffers(l)
    fill_smile_buffers!(buf.λ_tot, buf.φ, l, i)
    return SmileScoreGrid(S, buf.λ_tot, buf.φ, l.strikes)
end


# ==============================================================================
# 7. ALLOCATION AUDIT
# ==============================================================================

"""
    AllocRow

One `@allocated` measurement of a hot-path operation, and whether it hit zero.
"""
struct AllocRow
    op::String
    bytes::Int
    budget::Int
    pass::Bool
end

"""
    tpl_measure_alloc(op, budget, f) -> AllocRow

Measure `f()`'s heap allocation after warming it up.

The warm-up call is mandatory and not a courtesy: the FIRST call to any Julia method
allocates for compilation, so `@allocated` on a cold function measures the compiler,
not the kernel. Two warm-ups, because a method's first specialisation can trigger a
second round of inference on a callee.

`@allocated` is measured on a `Ref`-free zero-argument closure whose captures are all
concretely typed, so the closure itself does not contribute — verified by the
`baseline` row in `allocation_audit`, which measures an empty closure through the same
path and must also read 0.
"""
function tpl_measure_alloc(op::AbstractString, budget::Int, f)
    f()
    f()
    bytes = @allocated f()
    return AllocRow(String(op), Int(bytes), budget, bytes <= budget)
end

"""
    allocation_audit(l; max_goals, markets) -> Vector{AllocRow}

Measure every in-place hot-path operation on `l`.

WHAT IS AND IS NOT BEING CLAIMED. The claim is that the STEADY-STATE kernels —
score-grid fill and market pricing into caller-owned buffers — allocate nothing. It is
NOT that the containers are free to build: `extract_latents` allocates the matrices, by
design and exactly once per fold. Reporting the two together would be dishonest, so the
setup allocations are shown separately in `r01_demo.jl` §10 as a memory comparison
rather than folded in here as a pass/fail.
"""
function allocation_audit(l::AbstractPosteriorLatents;
                          max_goals::Integer = TPL_MAX_GOALS,
                          markets = (Market1X2(), MarketBTTS(), MarketOverUnder(2.5)))
    ws = GridWorkspace(max_goals)
    S  = alloc_score_grid(l, max_goals)
    i  = 1

    rows = AllocRow[]
    push!(rows, tpl_measure_alloc("baseline (empty closure)", 0, () -> nothing))
    push!(rows, tpl_measure_alloc("compute_score_grid!  [$(nameof(typeof(l)))]", 0,
                                  () -> compute_score_grid!(S, ws, l, i)))

    container = _tpl_typed_container(l, S, ws, i)
    if l isa SmileLatents
        buf = alloc_smile_buffers(l)
        push!(rows, tpl_measure_alloc("fill_smile_buffers!", 0,
                                      () -> fill_smile_buffers!(buf.λ_tot, buf.φ, l, i)))
    end

    for m in markets
        book = alloc_market_book(m, n_draws(l))
        push!(rows, tpl_measure_alloc("price_market!  $(m)", 0,
                                      () -> price_market!(book, container, m)))
    end
    return rows
end

"""
    tpl_alloc_table(rows; title) -> Bool

Print the allocation audit and return whether every row met its budget.
"""
function tpl_alloc_table(rows::Vector{AllocRow}; title::AbstractString = "ALLOCATION AUDIT")
    width = maximum(length(r.op) for r in rows; init = 20)
    println()
    println("  ", title)
    println("  ", "-"^(width + 32))
    @printf("  %-*s %12s %8s  %s\n", width, "operation", "bytes", "budget", "verdict")
    println("  ", "-"^(width + 32))
    ok = true
    for r in rows
        r.pass || (ok = false)
        @printf("  %-*s %12d %8d  %s\n", width, r.op, r.bytes, r.budget, r.pass ? "pass" : "FAIL")
    end
    println("  ", "-"^(width + 32))
    return ok
end


# ==============================================================================
# 8. TIMING AND MEMORY
# ==============================================================================
#
# Not a benchmark suite. These numbers exist to answer one question — "is the typed
# path at least as fast, and materially smaller?" — and they are reported as ratios
# with the sample count visible, so a reader can see how much to trust them.

struct TimingRow
    op::String
    seconds::Float64
    per_fixture_μs::Float64
end

"""
    tpl_time(op, n, f) -> TimingRow

Time `f()` over `n` fixtures after two warm-up calls. Best of 3 runs: the minimum is
the right statistic for a deterministic kernel, where every deviation upward is
measurement noise (GC, scheduling) and none is signal.
"""
function tpl_time(op::AbstractString, n::Int, f)
    f()
    f()
    best = Inf
    for _ in 1:3
        t0 = time_ns()
        f()
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return TimingRow(String(op), best, n == 0 ? NaN : best / n * 1e6)
end

function tpl_timing_table(rows::Vector{TimingRow}; title::AbstractString = "TIMING")
    width = maximum(length(r.op) for r in rows; init = 20)
    println()
    println("  ", title)
    println("  ", "-"^(width + 32))
    @printf("  %-*s %12s %14s\n", width, "operation", "seconds", "µs/fixture")
    println("  ", "-"^(width + 32))
    for r in rows
        @printf("  %-*s %12.4f %14.1f\n", width, r.op, r.seconds, r.per_fixture_μs)
    end
    println("  ", "-"^(width + 32))
    return nothing
end

"""
    memory_comparison(label, l, legacy_df) -> Nothing

Print the storage the typed container needs against the storage the equivalent
`latents.df` needs, in bytes and in heap objects.

THE OBJECT COUNT IS THE NUMBER THAT MATTERS, and it is the one that explains the
timing. A `latents.df` holds `n_matches × n_parameters` separately-allocated sample
vectors — each an independent cache miss, an independent GC root, and an independent
`Any`-boxed dynamic dispatch on access. The typed container holds one matrix per
parameter, whatever the fold size, so that count does not grow with the fold.

The BYTE ratio is the less interesting of the two and is not always in the typed
container's favour: `SmileLatents` materialises the per-fixture `φ` dimension that the
engines currently share across every row, and pays for it (see `l01_latents.jl` §5 for
what that buys). Reported either way.
"""
function memory_comparison(label::AbstractString, l::AbstractPosteriorLatents,
                           legacy_df::AbstractDataFrame)
    tb, to = latent_bytes(l), latent_allocations(l)
    lb, lo = tpl_dataframe_bytes(legacy_df)
    println()
    println("  MEMORY — $label  ($(n_matches(l)) fixtures × $(n_draws(l)) draws)")
    println("  ", "-"^62)
    @printf("  %-28s %14s %14s\n", "", "bytes", "heap objects")
    @printf("  %-28s %14s %14d\n", "legacy latents.df", _tpl_human_bytes(lb), lo)
    @printf("  %-28s %14s %14d\n", "typed container", _tpl_human_bytes(tb), to)
    @printf("  %-28s %13.2fx %13.2fx\n", "ratio (legacy / typed)",
            tb == 0 ? NaN : lb / tb, to == 0 ? NaN : lo / to)
    println("  ", "-"^62)
    return nothing
end


# ==============================================================================
# 9. DETERMINISTIC SYNTHETIC POSTERIORS
# ==============================================================================
#
# WHY SYNTHETIC, AND WHAT IS AND IS NOT THEREFORE TESTED.
#
# The briefing forbids running MCMC grids for this prototype, and it is right to: what
# is under test is a CONTAINER, and a container cannot tell whether the numbers in it
# came from a converged NUTS run or from `randn`. Fitting a model would add hours and
# a source of run-to-run variation while testing nothing extra.
#
# So the chains here are drawn from priors with a fixed seed. Everything downstream is
# the REAL code path: the real `extract_parameters` for each engine, the real
# `Predictions` kernels, the real market types. The synthetic part is the posterior
# and nothing else.
#
# WHAT THIS DOES NOT COVER, stated so nobody mistakes a green table for more than it
# is: nothing here shows that any model FITS anything, or that its OOS prices are
# good. It shows that moving the posterior from a DataFrame into a typed matrix does
# not change a single price. That is the entire claim.

"""
    tpl_synthetic_site(name, n, rng) -> Vector{Float64}

Draw `n` values for one chain site, from a distribution appropriate to what that site
means in the engines.

Not cosmetic. A `σ` site drawn from an unconstrained `Normal` goes negative, and
`extract_dynamics` multiplies raw z-scores by it, producing sign-flipped ratings and
intensities that exercise the clamp on every draw — which would make the parity table
a test of `clamp` rather than of the kernels. A `log_r` drawn far from its prior makes
every negative-binomial marginal numerically degenerate. The rules below reproduce
each site's own prior closely enough that the resulting λ land in the 0.5-3.0 range a
real fold produces.

Matched in order, first hit wins, so the specific patterns precede the generic ones.
"""
function tpl_synthetic_site(name::AbstractString, n::Int, rng::AbstractRNG)
    has(s) = occursin(s, name)

    # --- specific sites, ahead of the generic `raw` rule -----------------------
    has("log_κ_raw")   && return 0.10 .* randn(rng, n)              # filldist(Normal(0, 0.10))
    has("δ_league_raw") && return 0.20 .* randn(rng, n)             # filldist(Normal(0, 0.2))
    has("γ_league_raw") && return 0.20 .* randn(rng, n)
    has("log_φ")       && return 0.05 .* randn(rng, n)              # smile shape, near-flat
    has("pen_base_μ")  && return -1.60 .+ 0.20 .* randn(rng, n)
    has("ha_pen")      && return 0.15 .+ 0.10 .* randn(rng, n)
    has("ν_xg")        && return 3.50 .+ 0.30 .* abs.(randn(rng, n))
    has("δ_r_home")    && return 0.20 .+ 0.05 .* randn(rng, n)
    has("log_r")       && return 1.50 .+ 0.15 .* randn(rng, n)      # r ≈ 4.5, mild overdispersion

    # --- scales: must be strictly positive ------------------------------------
    has("σ")           && return 0.15 .+ 0.05 .* abs.(randn(rng, n))

    # --- structural ------------------------------------------------------------
    (has("μ_base") || name == "inter.μ")   && return 0.15 .+ 0.05 .* randn(rng, n)
    (has("γ_global") || has("γ_base"))     && return 0.25 .+ 0.05 .* randn(rng, n)
    (has("w_wealth") || endswith(name, ".w")) && return 0.10 .+ 0.03 .* abs.(randn(rng, n))

    # --- non-centred z-scores --------------------------------------------------
    has("raw")         && return randn(rng, n)

    return 0.20 .* randn(rng, n)
end

"""
    tpl_synthetic_chain(colnames; n_draws, n_chains, seed) -> Chains

A deterministic `MCMCChains.Chains` with exactly the requested sites.

`n_chains > 1` is worth using wherever the engine supports it, because it exercises
the `size(chain,1) * size(chain,3)` flattening that every `extract_*` component
performs — a flattening two of them get wrong (see `tpl_multichain_warning`).
"""
function tpl_synthetic_chain(colnames::Vector{String};
                             n_draws::Int, n_chains::Int = 1, seed::Int = 20240601)
    rng  = MersenneTwister(seed)
    vals = Array{Float64, 3}(undef, n_draws, length(colnames), n_chains)
    for c in 1:n_chains, (j, name) in enumerate(colnames)
        vals[:, j, c] = tpl_synthetic_site(name, n_draws, rng)
    end
    return Chains(vals, Symbol.(colnames))
end

"""
    tpl_multichain_warning() -> String

The reason the recombination family is exercised with ONE chain.

`extract_recombination` (src/models/pregame/components/recombination.jl:54, :63) and
`extract_squad_wealth` / `extract_pxg_observation` size their outputs with
`size(chain, 1)` — samples per chain — where every other extractor in the repository
uses `size(chain, 1) * size(chain, 3)`, the flattened total. With more than one chain
those vectors come out `n_chains` times too short and the per-match broadcast against
the full-length `dyn.α` either throws or, worse, silently recycles.

Out of scope to fix here: this prototype is a container change, and quietly repairing
a sampler-facing extractor inside it would make the parity tables incomparable to the
production path they are supposed to be validating. Recorded, and reproduced in
`r01_demo.jl` §6.
"""
tpl_multichain_warning() =
    "extract_recombination / extract_squad_wealth / extract_pxg_observation size their " *
    "output with size(chain,1), not size(chain,1)*size(chain,3) — single-chain only."

"""
    tpl_synthetic_fixtures(n; n_teams, seed, first_date) -> DataFrame

`n` held-out fixtures over `n_teams` teams, with every column the engines' OOS
extractors read:

    match_id, home_team, away_team, match_date, season_idx, delta_wealth_logsum

`delta_wealth_logsum` is materialised on the frame rather than left to the covariate's
point-in-time bridge, which is the documented way to price a hypothetical lineup
(`05_composable_count_builder/l01_components.jl`, `covariate_oos`). It also keeps this
demo free of any dependency on cached valuation data.

Home and away teams are drawn so that no fixture is a team against itself, and the
dates walk forward through distinct months so the monthly interception component
actually varies across fixtures instead of being silently constant.
"""
function tpl_synthetic_fixtures(n::Int; n_teams::Int = 8, seed::Int = 424242,
                                first_date::Date = Date(2025, 1, 5))
    rng   = MersenneTwister(seed)
    teams = ["TEAM_$(lpad(i, 2, '0'))" for i in 1:n_teams]

    home = Vector{String}(undef, n)
    away = Vector{String}(undef, n)
    for i in 1:n
        h = rand(rng, 1:n_teams)
        a = rand(rng, 1:n_teams)
        while a == h
            a = rand(rng, 1:n_teams)
        end
        home[i] = teams[h]
        away[i] = teams[a]
    end

    return DataFrame(
        match_id            = collect(9_000_001:(9_000_000 + n)),
        home_team           = home,
        away_team           = away,
        match_date          = [first_date + Day(11 * (i - 1)) for i in 1:n],
        season_idx          = fill(2, n),
        delta_wealth_logsum = round.(0.35 .* randn(rng, n), digits = 6),
    )
end

"""
    tpl_team_map(n_teams) -> Dict{String, Int}

The `team_map` a real `FeatureSet` carries, over the same names
`tpl_synthetic_fixtures` generates.
"""
tpl_team_map(n_teams::Int) =
    Dict("TEAM_$(lpad(i, 2, '0'))" => i for i in 1:n_teams)

"""
    tpl_feature_set(; pairs...) -> FeatureSet

A `FeatureSet` carrying exactly the metadata the extractor under test reads, and
nothing else.

Deliberately minimal. An extractor that silently reaches for a key this does not
supply raises a `KeyError` here, at the top of a five-second demo, rather than being
discovered when a real fold is already three hours into a run.
"""
tpl_feature_set(; pairs...) =
    BayesianFootball.TypesInterfaces.FeatureSet(Dict{Symbol, Any}(pairs))
