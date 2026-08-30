# ==============================================================================
# 08 — UNIFIED EVALUATION FRAMEWORK : THE METRIC KERNELS
# ==============================================================================
#
# Every number this framework reports is computed here.
#
# ------------------------------------------------------------------------------
# WHAT THIS REPLACES, AND WHAT IT COSTS TODAY
# ------------------------------------------------------------------------------
#
# `src/evaluation/metrics_methods/*.jl` all have the same four-step shape:
#
#     ppd            = Predictions.model_inference(latents_raw)        # (1)
#     model_features = transform(ppd.df, :distribution => ByRow(mean))  # (2)
#     analysis_df    = innerjoin(ds.odds, model_features, on = [4 cols]) # (3)
#     dropmissing!(analysis_df, [...])                                   # (4)
#
# and each of the four is paid ONCE PER METRIC:
#
#   (1) prices every market in `DEFAULT_MARKET_CONFIG` — 40-odd of them, including
#       the whole Asian-handicap ladder — for a metric that wants three. It also
#       builds a `Vector{Float64}` of posterior draws per fixture per selection and
#       stores them in a `Vector{Any}` DataFrame column, which is the fragmentation
#       `06_typed_posterior_latents` exists to remove, reintroduced downstream of it.
#
#   (2) collapses every one of those vectors to a mean — including the ~90% of them
#       the metric will discard three lines later.
#
#   (3) is a four-column hash join between a ~50,000-row odds frame and a
#       ~80,000-row PPD frame, materialising a third frame with every column of both.
#
#   (4) copies it again.
#
# `Predictions._PPD_CACHE` memoises (1) on `objectid(latents.df)`, so a second metric
# on the SAME object is cheaper — but `evaluate_experiments` calls
# `extract_oos_predictions` once per experiment and the cache key is an `objectid`,
# so any reload misses. And nothing memoises (3).
#
# ------------------------------------------------------------------------------
# THE REPLACEMENT
# ------------------------------------------------------------------------------
#
# Three dense, typed indexes, built ONCE for a whole batch of metrics:
#
#     OddsView             ds.odds, as six concretely-typed parallel vectors
#     MatchOutcomes        match_id → (home_score, away_score)
#     MarketProbabilities  a (n_draws × n_fixtures × n_selections) tensor, priced by
#                          06's zero-allocation kernels over the markets the metrics
#                          ACTUALLY named
#
# and every kernel is then a single pass over the odds rows with two integer lookups.
# No join, no `dropmissing`, no intermediate frame, no `Vector{Any}`.
#
# ------------------------------------------------------------------------------
# THE PARITY CONSTRAINT, AND WHAT IT DICTATES
# ------------------------------------------------------------------------------
#
# These kernels must produce the SAME FLOAT64 as the legacy ones, not merely the same
# real number. Two consequences run through the whole file:
#
#   * ROW ORDER IS PRESERVED. Every aggregate is a `mean` or `sum` over a vector, and
#     floating-point addition is not associative, so the rows must be accumulated in
#     the order `innerjoin(ds.odds, …)` produced them — which is `ds.odds`' own order.
#     Hence "walk the odds table", not "walk the fixtures".
#
#   * THE SCALAR FORMULAE ARE COPIED, NOT REDERIVED. `calc_logloss`, `calc_lpd_samples`,
#     `compute_crps` and `compute_rqr` below are the `src` bodies, clamp constants and
#     all. Where a body is written differently it is because the `src` version
#     allocates per call and the rewrite is documented, term for term, at the site.
#
# ==============================================================================

include(joinpath(@__DIR__, "l01_types.jl"))

using GLM
using HypothesisTests
using LinearAlgebra
using Random


# ==============================================================================
# 1. THE ODDS VIEW
# ==============================================================================
#
# `ds.odds` is a DataFrame whose numeric columns are `Union{Missing, Float64}` and
# whose `match_id` is `Int32` (`src/Data/fetchers/sql/odds.jl:41-47`). A kernel that
# reads it row by row through `eachrow` pays a dynamic dispatch per cell.
#
# This converts the six columns any metric can want into concrete vectors, ONCE. The
# conversion is O(n) over a frame the legacy path hash-joins O(n) times per metric, so
# it is strictly cheaper even before the join is counted.
#
# MISSINGNESS IS A BITVECTOR, NOT A NaN SENTINEL. `dropmissing!(df, :prob_fair_close)`
# drops `missing` and KEEPS a genuine `NaN`; collapsing the two would change which rows
# a metric scores. The `present` masks below reproduce `dropmissing!` exactly.

"""
    _ue_float_column(df, name) -> (Vector{Float64}, BitVector)

A frame column as values plus a presence mask. Absent columns come back all-missing
rather than raising, so a store built without `odds_close` still supports `LogLoss`.
"""
function _ue_float_column(df::AbstractDataFrame, name::Symbol)
    n = nrow(df)
    vals = Vector{Float64}(undef, n)
    present = falses(n)
    hasproperty(df, name) || return vals, present
    col = getproperty(df, name)
    @inbounds for i in 1:n
        x = col[i]
        if x === missing
            vals[i] = NaN
        else
            vals[i] = Float64(x)
            present[i] = true
        end
    end
    return vals, present
end

"""
    OddsView(odds_df)

`ds.odds`, flattened into the vectors every kernel reads.

`is_winner` is `Int8`: `1` won, `0` lost, `-1` the row's outcome is unknown. The
three-valued encoding is what lets a kernel reproduce `dropmissing!(df, :is_winner)`
with an integer comparison instead of a frame copy.
"""
struct OddsView
    n::Int
    match_id::Vector{Int}
    selection::Vector{Symbol}
    market_name::Vector{String}
    market_line::Vector{Float64}
    prob_fair_close::Vector{Float64}
    has_fair::BitVector
    prob_implied_close::Vector{Float64}
    has_implied::BitVector
    odds_close::Vector{Float64}
    has_odds_close::BitVector
    is_winner::Vector{Int8}
end

function OddsView(df::AbstractDataFrame)
    n = nrow(df)
    for c in (:match_id, :market_name, :market_line, :selection)
        hasproperty(df, c) || error(
            "OddsView: the odds frame has no `$c` column. Expected the enriched long " *
            "form that `Data.process_data(::OddsData)` produces — match_id, " *
            "market_name, market_line, selection, prob_fair_close, is_winner.")
    end

    mid = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        mid[i] = Int(df.match_id[i])
    end

    sel = Vector{Symbol}(undef, n)
    @inbounds for i in 1:n
        s = df.selection[i]
        sel[i] = s isa Symbol ? s : Symbol(s)
    end

    mname = Vector{String}(undef, n)
    @inbounds for i in 1:n
        mname[i] = String(df.market_name[i])
    end

    mline = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        x = df.market_line[i]
        mline[i] = x === missing ? NaN : Float64(x)
    end

    fair, has_fair       = _ue_float_column(df, :prob_fair_close)
    impl, has_impl       = _ue_float_column(df, :prob_implied_close)
    oclose, has_oclose   = _ue_float_column(df, :odds_close)

    win = fill(Int8(-1), n)
    if hasproperty(df, :is_winner)
        col = df.is_winner
        @inbounds for i in 1:n
            x = col[i]
            x === missing && continue
            win[i] = Int8(x == true ? 1 : 0)
        end
    end

    return OddsView(n, mid, sel, mname, mline, fair, has_fair,
                    impl, has_impl, oclose, has_oclose, win)
end

Base.length(o::OddsView) = o.n

Base.show(io::IO, o::OddsView) =
    print(io, "OddsView(", o.n, " rows, ", length(unique(o.selection)), " selections)")


# ==============================================================================
# 2. THE OUTCOME INDEX
# ==============================================================================

"""
    MatchOutcomes(matches_df)

`match_id → (home_score, away_score)` for every fixture whose result is recorded.

A fixture with a missing score is ABSENT rather than present-with-a-sentinel, so a
kernel that asks for it gets `nothing` and skips the row. `src`'s CRPS and RQR
`innerjoin` against `ds.matches` and do not drop missing scores, which means a store
holding a not-yet-played fixture makes `compute_crps(missing, …)` raise
`MethodError` several minutes into an evaluation. Recorded in `README.md`; the
behaviour here is to skip and report the count.
"""
struct MatchOutcomes
    scores::Dict{Int, Tuple{Int, Int}}
end

function MatchOutcomes(df::AbstractDataFrame)
    d = Dict{Int, Tuple{Int, Int}}()
    (hasproperty(df, :match_id) && hasproperty(df, :home_score) &&
     hasproperty(df, :away_score)) || return MatchOutcomes(d)
    sizehint!(d, nrow(df))
    @inbounds for i in 1:nrow(df)
        h = df.home_score[i]
        a = df.away_score[i]
        (h === missing || a === missing) && continue
        d[Int(df.match_id[i])] = (Int(h), Int(a))
    end
    return MatchOutcomes(d)
end

@inline outcome_of(o::MatchOutcomes, mid::Integer) = get(o.scores, Int(mid), nothing)

Base.length(o::MatchOutcomes) = length(o.scores)
Base.show(io::IO, o::MatchOutcomes) = print(io, "MatchOutcomes(", length(o.scores), " fixtures)")


# ==============================================================================
# 3. THE PRICED POSTERIOR
# ==============================================================================
#
# The replacement for `Predictions.model_inference` on the evaluation path.

"""
    MarketProbabilities

Every posterior market probability the batch needs, in ONE tensor.

| field        | shape                                       |
|--------------|---------------------------------------------|
| `draws`      | `n_draws × n_fixtures × n_selections`        |
| `means`      | `n_fixtures × n_selections`                  |
| `match_ids`  | `n_fixtures`, the latent container's order   |
| `selections` | `n_selections`, `market_keys` order          |

DRAW-MAJOR ON PURPOSE. `view(draws, :, i, c)` is then CONTIGUOUS, which matters twice:
it is what LPD and MIQ iterate, and a contiguous `SubArray` reduces with the same
pairwise blocking Base uses on a `Vector`, so `mean(view(...))` is bit-identical to
`mean` of the `Vector{Float64}` the legacy PPD frame stored. Parity with `src` depends
on that, and §6 of `r01_demo.jl` checks it rather than assuming it.

`draws` is EMPTY (`0 × 0 × 0`) when every metric in the batch only needs means. LogLoss
and GLMEdge do; LPD and MIQ do not. On a 900-fixture fold at 1,200 draws and eleven
selections that is 95 MB not allocated.
"""
struct MarketProbabilities
    markets::Vector{AbstractMarket}
    match_ids::Vector{Int}
    selections::Vector{Symbol}
    row_of::Dict{Int, Int}
    col_of::Dict{Symbol, Int}
    draws::Array{Float64, 3}
    means::Matrix{Float64}
    n_draws::Int
    keep_draws::Bool
end

Base.show(io::IO, p::MarketProbabilities) = print(
    io, "MarketProbabilities(", length(p.match_ids), " fixtures × ",
    length(p.selections), " selections × ", p.n_draws, " draws",
    p.keep_draws ? "" : ", means only", ")")

"Bytes the tensor holds. The number to set against the legacy PPD frame's."
probability_bytes(p::MarketProbabilities) = sizeof(p.draws) + sizeof(p.means)

@inline function _ue_locate(p::MarketProbabilities, mid::Integer, sel::Symbol)
    r = get(p.row_of, Int(mid), 0)
    r == 0 && return (0, 0)
    c = get(p.col_of, sel, 0)
    return (r, c)
end

"""
    prob_mean(p, match_id, selection) -> Float64 or nothing

The model's mean probability, or `nothing` when this framework did not price that
(fixture, selection) — the exact condition under which `innerjoin` would have dropped
the row.
"""
@inline function prob_mean(p::MarketProbabilities, mid::Integer, sel::Symbol)
    r, c = _ue_locate(p, mid, sel)
    (r == 0 || c == 0) && return nothing
    return @inbounds p.means[r, c]
end

"""
    prob_draws(p, match_id, selection) -> contiguous view or nothing

The full posterior of that probability. `nothing` if unpriced; raises if the tensor was
built means-only, because silently returning the mean would turn LPD into log-loss.
"""
@inline function prob_draws(p::MarketProbabilities, mid::Integer, sel::Symbol)
    p.keep_draws || error(
        "prob_draws: this MarketProbabilities was built means-only. Pass " *
        "`keep_draws = true`, or include a metric whose `needs_draws` is true.")
    r, c = _ue_locate(p, mid, sel)
    (r == 0 || c == 0) && return nothing
    return @inbounds view(p.draws, :, r, c)
end

# --- 3.1 the pricing sweep ------------------------------------------------------
#
# One `GridWorkspace` and one destination grid per WORKER, exactly as
# `06/l03_score_grids.jl` prescribes — not per fixture, and not per market.

"The per-fixture pricing target: the grid, or a `SmileScoreGrid` for the smile family."
struct _UESmileTarget{L}
    latents::L
    grid::Array{Float64, 3}
    holder::SmileScoreGrid
end

_ue_target(l::AbstractPosteriorLatents, S::Array{Float64,3}) = S

function _ue_target(l::SmileLatents, S::Array{Float64,3})
    buf = alloc_smile_buffers(l)
    return _UESmileTarget(l, S, SmileScoreGrid(S, buf.λ_tot, buf.φ, copy(l.strikes)))
end

@inline _ue_price_target(t::Array{Float64,3}, l, i) = t

@inline function _ue_price_target(t::_UESmileTarget, l::SmileLatents, i::Int)
    fill_smile_buffers!(t.holder.λ_tot, t.holder.φ, l, i)
    return t.holder
end

_ue_grid_of(t::Array{Float64,3}) = t
_ue_grid_of(t::_UESmileTarget) = t.grid

"""
    market_probabilities(latents, markets; keep_draws = true,
                         max_goals = TPL_MAX_GOALS, threaded = true) -> MarketProbabilities

Price every fixture in `latents` on every market in `markets`.

The inner two loops are `06`'s `compute_score_grid!` and `price_market!`, which measure
zero bytes; everything this function allocates it allocates ONCE, before the sweep.

THREADED OVER FIXTURES, WITH A WORKSPACE PER CHUNK. Fixtures are independent and each
writes a disjoint slice of the output, so the result is bit-identical to the sequential
one — the thread count changes the wall clock and nothing else. Chunked with `@spawn`
rather than `@threads` so the workspace binding does not depend on `threadid()`, which
is not stable across a task yield.
"""
function market_probabilities(l::AbstractPosteriorLatents,
                              markets::AbstractVector;
                              keep_draws::Bool = true,
                              max_goals::Integer = TPL_MAX_GOALS,
                              threaded::Bool = true)
    mkts = AbstractMarket[m for m in markets]
    isempty(mkts) && return MarketProbabilities(
        mkts, Int[], Symbol[], Dict{Int,Int}(), Dict{Symbol,Int}(),
        Array{Float64,3}(undef, 0, 0, 0), Matrix{Float64}(undef, 0, 0), n_draws(l), false)

    ids = copy(latent_match_ids(l))
    nf  = length(ids)
    nd  = n_draws(l)

    sels = Symbol[]
    for m in mkts, k in market_keys(m)
        k in sels && error(
            "market_probabilities: selection :$k is claimed by two markets in the " *
            "list. Every selection symbol must name exactly one market — see " *
            "`market_for_selection`.")
        push!(sels, k)
    end
    ns = length(sels)

    means = Matrix{Float64}(undef, nf, ns)
    draws = keep_draws ? Array{Float64,3}(undef, nd, nf, ns) :
                         Array{Float64,3}(undef, 0, 0, 0)

    # The column each market's outcomes write into, precomputed so the hot loop does
    # no symbol lookup.
    offsets = Int[]
    let acc = 0
        for m in mkts
            push!(offsets, acc)
            acc += market_arity(m)
        end
    end

    function sweep(lo::Int, hi::Int)
        ws    = GridWorkspace(max_goals)
        S     = alloc_score_grid(l, max_goals)
        tgt   = _ue_target(l, S)
        books = Tuple(alloc_market_book(m, nd) for m in mkts)
        for i in lo:hi
            compute_score_grid!(_ue_grid_of(tgt), ws, l, i)
            priced = _ue_price_target(tgt, l, i)
            for (j, m) in enumerate(mkts)
                book = books[j]
                price_market!(book, priced, m)
                off = offsets[j]
                for s in 1:length(book)
                    v = book[s]
                    c = off + s
                    means[i, c] = mean(v)
                    if keep_draws
                        @inbounds copyto!(view(draws, :, i, c), v)
                    end
                end
            end
        end
        return nothing
    end

    nchunks = threaded ? min(Threads.nthreads(), nf) : 1
    if nchunks <= 1
        sweep(1, nf)
    else
        per = cld(nf, nchunks)
        @sync for t in 1:nchunks
            lo = (t - 1) * per + 1
            hi = min(t * per, nf)
            lo > hi && continue
            Threads.@spawn sweep(lo, hi)
        end
    end

    return MarketProbabilities(
        mkts, ids, sels,
        Dict{Int,Int}(id => i for (i, id) in enumerate(ids)),
        Dict{Symbol,Int}(s => c for (c, s) in enumerate(sels)),
        draws, means, nd, keep_draws)
end


# ==============================================================================
# 4. THE EVALUATION CONTEXT
# ==============================================================================

"""
    EvaluationContext(latents, odds, outcomes, probs)

Everything a batch of metrics reads, built once and shared.

This is the structural fix. `evaluate_experiments` gives every metric the same
`latents` object and lets each one rebuild the PPD, the join and the cleaned frame for
itself; here the three indexes are built once from the union of what the metrics
declared (`scored_markets`, `needs_outcomes`, `needs_draws`) and the kernels read them.
"""
struct EvaluationContext{L<:AbstractPosteriorLatents}
    latents::L
    odds::OddsView
    outcomes::MatchOutcomes
    probs::MarketProbabilities
    max_goals::Int
end

function Base.show(io::IO, c::EvaluationContext)
    print(io, "EvaluationContext(", n_matches(c.latents), " fixtures, ",
          c.odds.n, " odds rows, ", c.probs, ")")
end

"""
    evaluation_context(latents, odds_df, matches_df, metrics; max_goals, threaded)

Build the shared indexes for `metrics`.

The market list is the UNION of every rule's `scored_markets`, deduplicated. Two rules
that want overlapping markets price them once between them; a batch of `CRPS` and `RQR`
alone prices nothing at all, because neither reads a market.
"""
function evaluation_context(l::AbstractPosteriorLatents,
                            odds_df::AbstractDataFrame,
                            matches_df::AbstractDataFrame,
                            metrics::AbstractVector;
                            max_goals::Integer = TPL_MAX_GOALS,
                            threaded::Bool = true)
    mkts = AbstractMarket[]
    for m in metrics, mk in scored_markets(m)
        any(x -> x == mk, mkts) || push!(mkts, mk)
    end
    keep = any(needs_draws, metrics) && !isempty(mkts)
    want_outcomes = any(needs_outcomes, metrics)

    odds = OddsView(isempty(mkts) ? similar(odds_df, 0) : odds_df)
    outs = want_outcomes ? MatchOutcomes(matches_df) : MatchOutcomes(Dict{Int,Tuple{Int,Int}}())
    probs = market_probabilities(l, mkts; keep_draws = keep,
                                 max_goals = max_goals, threaded = threaded)

    return EvaluationContext(l, odds, outs, probs, Int(max_goals))
end

evaluation_context(l::AbstractPosteriorLatents, odds_df, matches_df,
                   metric::AbstractScoringRule; kwargs...) =
    evaluation_context(l, odds_df, matches_df, [metric]; kwargs...)

"""
    _ue_selection_filter(metric) -> (Set{Symbol} or nothing)

`nothing` means "no filter" — the empty-`selections` case, which is what `src` means by
`isempty(metric.selections)`. A `Set` is used rather than the vector because the filter
is tested once per odds row and the vectors are short but the frames are not.
"""
function _ue_selection_filter(m::AbstractScoringRule)
    sels = scored_selections(m)
    return isempty(sels) ? nothing : Set{Symbol}(sels)
end

@inline _ue_passes(::Nothing, ::Symbol) = true
@inline _ue_passes(f::Set{Symbol}, s::Symbol) = s in f


# ==============================================================================
# 5. THE SCALAR FORMULAE
# ==============================================================================
#
# Copied from `src/evaluation/metrics_methods/`, clamp constants and all. Any
# divergence here is a parity failure, so they are kept together, short, and adjacent
# to the line they came from.

"""
    calc_logloss(p, y) -> Float64

Binary cross-entropy. `src/evaluation/metrics_methods/logloss.jl:49`.
"""
@inline function calc_logloss(p::Float64, y::Float64)
    p_clamped = clamp(p, 1e-15, 1.0 - 1e-15)
    return -(y * log(p_clamped) + (1.0 - y) * log(1.0 - p_clamped))
end

"""
    calc_lpd_scalar(p, y) -> Float64

`src/evaluation/metrics_methods/lpd.jl:76`.
"""
@inline function calc_lpd_scalar(p::Float64, y::Float64)
    p_clamped = clamp(p, 1e-15, 1.0 - 1e-15)
    return y == 1.0 ? log(p_clamped) : log(1.0 - p_clamped)
end

"""
    calc_lpd_samples!(scratch_a, scratch_b, samples, y) -> Float64

`log( (1/S) Σ_s p(y | θ^s) )` by log-sum-exp, into caller-supplied scratch.

TERM FOR TERM `src/evaluation/metrics_methods/lpd.jl:61-69`, which is

    log_liks = log.(clamp.(samples, 1e-15, 1-1e-15))     # or 1 .- samples for y = 0
    lmax     = maximum(log_liks)
    lmax + log(mean(exp.(log_liks .- lmax)))

with the two broadcast temporaries hoisted into reusable buffers. The reductions are
still Base's `maximum` and `mean` over a `Vector{Float64}` of the same length in the
same order, so the result is bit-identical — which is the reason the buffers are
`Vector`s and not, say, an on-line accumulation.
"""
function calc_lpd_samples!(log_liks::Vector{Float64}, expbuf::Vector{Float64},
                           samples::AbstractVector{Float64}, y::Float64)
    n = length(samples)
    length(log_liks) >= n || resize!(log_liks, n)
    length(expbuf) >= n || resize!(expbuf, n)
    ll = view(log_liks, 1:n)
    eb = view(expbuf, 1:n)
    if y == 1.0
        @inbounds for k in 1:n
            ll[k] = log(clamp(samples[k], 1e-15, 1.0 - 1e-15))
        end
    else
        @inbounds for k in 1:n
            ll[k] = log(clamp(1.0 - samples[k], 1e-15, 1.0 - 1e-15))
        end
    end
    lmax = maximum(ll)
    @inbounds for k in 1:n
        eb[k] = exp(ll[k] - lmax)
    end
    return lmax + log(mean(eb))
end

"""
    compute_crps(y, λ, r; max_goals = 30) -> Float64

`src/evaluation/metrics_methods/crps.jl:31`. `r = Inf` (or `NaN`) selects Poisson.
"""
function compute_crps(y::Real, λ::Real, r_disp::Real; max_goals::Integer = 30)
    dist = if isinf(r_disp) || isnan(r_disp)
        Poisson(λ)
    else
        NegativeBinomial(r_disp, r_disp / (r_disp + λ))
    end
    crps_value = 0.0
    for x in 0:max_goals
        F_x = cdf(dist, x)
        indicator = x >= y ? 1.0 : 0.0
        crps_value += (F_x - indicator)^2
    end
    return crps_value
end

"""
    compute_rqr(y, dist, rng) -> Float64

`src/evaluation/metrics_methods/rqr.jl:39`, with the distribution passed in rather than
reconstructed from an `Inf` sentinel (§6.4) and with an EXPLICIT rng.

    u ~ Uniform(F(y−1), F(y)),   r = Φ⁻¹(clamp(u, 1e-7, 1−1e-7))

One `rand` per call, so a seeded rng reproduces `src`'s sequence exactly provided the
calls happen in the same order — which `metric_rqr` guarantees (all home residuals in
fixture order, then all away).
"""
function compute_rqr(y::Integer, dist::UnivariateDistribution, rng::AbstractRNG)
    cdf_lower = y > 0 ? cdf(dist, y - 1) : 0.0
    cdf_upper = cdf(dist, y)
    u = rand(rng, Uniform(cdf_lower, cdf_upper))
    u = clamp(u, 1e-7, 1.0 - 1e-7)
    return quantile(Normal(0, 1), u)
end

"""
    summarize_stats(x) -> DistributionStats

`src/evaluation/metrics_methods/rqr.jl:71`. Fewer than three points makes
`ShapiroWilkTest` throw, so that case reports `NaN` for `W` and `p` and keeps the
moments, rather than losing the whole row.
"""
function summarize_stats(x::Vector{Float64})::DistributionStats
    n = length(x)
    if n < 3
        return DistributionStats(n == 0 ? NaN : mean(x), n < 2 ? NaN : std(x),
                                 NaN, NaN, NaN, NaN)
    end
    w, p = try
        sw = ShapiroWilkTest(x)
        (sw.W, pvalue(sw))
    catch
        (NaN, NaN)
    end
    return DistributionStats(mean(x), std(x), skewness(x), kurtosis(x), w, p)
end

"""
    get_miq(samples, market_prob) -> Float64 or missing

`src/evaluation/metrics_methods/miq.jl:51`, without the broadcast temporary.
"""
@inline function get_miq(samples::AbstractVector{Float64}, market_prob::Float64)
    isnan(market_prob) && return missing
    c = 0
    @inbounds for k in eachindex(samples)
        samples[k] <= market_prob && (c += 1)
    end
    return c / length(samples)
end

"""
    evaluate_group_edge(quantiles, is_winner) -> MIQStats

`src/evaluation/metrics_methods/miq.jl:60`.
"""
function evaluate_group_edge(q::Vector{Float64}, won::Vector{Bool})::MIQStats
    winners = q[won]
    losers  = q[.!won]
    if length(winners) < 2 || length(losers) < 2
        return MIQStats(missing, missing, missing, missing, missing,
                        length(winners), length(losers))
    end
    mean_gap = mean(losers) - mean(winners)
    ks = ApproximateTwoSampleKSTest(winners, losers)
    return MIQStats(mean(q), std(q), mean_gap, ks.δ, pvalue(ks),
                    length(winners), length(losers))
end


# ==============================================================================
# 6. MARGINALS BY DISPATCH
# ==============================================================================
#
# The `hasproperty(df, :r)` cascade of `crps.jl:51` and `rqr.jl:58`, turned into method
# dispatch on the container type. A `CountLatents{T, Nothing}` has no dispersion field
# to probe for, so there is no branch that could pick the wrong density and no `Inf`
# sentinel standing in for "there is no r".
#
# The parameters are POSTERIOR MEANS, matching `src` (`mean.(joined.λ_h)`), and are
# taken with `mean(view(M, i, :))` — a strided `SubArray` that Base reduces with the
# same pairwise blocking as the contiguous `Vector` the legacy frame stored, so the two
# means are the same Float64 and not merely the same number.

"Posterior mean of parameter row `i`. Zero allocations; bit-identical to `mean(M[i,:])`."
@inline posterior_mean(M::AbstractMatrix{Float64}, i::Integer) = mean(view(M, i, :))

"""
    marginals(latents, i) -> (home::UnivariateDistribution, away::UnivariateDistribution)

The plug-in marginal goal distributions for fixture row `i`.

| container                       | marginal                              |
|---------------------------------|---------------------------------------|
| `CountLatents{T, Nothing}`      | `Poisson(λ̄)`                          |
| `CountLatents{T, <:NamedTuple}` | `NegativeBinomial(r̄, r̄/(r̄+λ̄))`        |
| `RecombLatents{T}`              | `Poisson(λ̄_open + λ̄_pen + λ̄_og)`      |
| `SmileLatents{T, Nothing}`      | `Poisson(λ̄)` on the GRID intensities  |
| `SmileLatents{T, <:NamedTuple}` | `NegativeBinomial` on the grid ones    |

The smile methods deliberately read `λ_home`/`λ_away` and NOT `λ_tot·φ`: CRPS and RQR
are per-SIDE goal-count diagnostics and the smile curve prices a TOTAL. Using the
pricing intensity here would test a different distribution from the one being reported.
"""
function marginals end

marginals(l::CountLatents{Float64, Nothing}, i::Integer) =
    (Poisson(posterior_mean(l.λ_home, i)), Poisson(posterior_mean(l.λ_away, i)))

function marginals(l::CountLatents{Float64, <:NamedTuple}, i::Integer)
    λh = posterior_mean(l.λ_home, i)
    λa = posterior_mean(l.λ_away, i)
    rh = posterior_mean(l.observation_params.r_h, i)
    ra = posterior_mean(l.observation_params.r_a, i)
    return (NegativeBinomial(rh, rh / (rh + λh)),
            NegativeBinomial(ra, ra / (ra + λa)))
end

function marginals(l::RecombLatents{Float64}, i::Integer)
    nd = n_draws(l)
    sh = 0.0
    sa = 0.0
    @inbounds for k in 1:nd
        sh += recomb_total_home(l, i, k)
        sa += recomb_total_away(l, i, k)
    end
    return (Poisson(sh / nd), Poisson(sa / nd))
end

marginals(l::SmileLatents{Float64, Nothing}, i::Integer) =
    (Poisson(posterior_mean(l.λ_home, i)), Poisson(posterior_mean(l.λ_away, i)))

function marginals(l::SmileLatents{Float64, <:NamedTuple}, i::Integer)
    λh = posterior_mean(l.λ_home, i)
    λa = posterior_mean(l.λ_away, i)
    rh = posterior_mean(l.observation_params.r_h, i)
    ra = posterior_mean(l.observation_params.r_a, i)
    return (NegativeBinomial(rh, rh / (rh + λh)),
            NegativeBinomial(ra, ra / (ra + λa)))
end

"""
    crps_parameters(latents, i) -> (λ_h, r_h, λ_a, r_a)

The four numbers `compute_crps` takes, with `Inf` for "no dispersion" so the shared
`src` formula can be called verbatim. Separate from `marginals` because
`compute_crps` builds its own distribution and reproducing that construction here is
what keeps the parity exact.
"""
crps_parameters(l::CountLatents{Float64, Nothing}, i::Integer) =
    (posterior_mean(l.λ_home, i), Inf, posterior_mean(l.λ_away, i), Inf)

crps_parameters(l::CountLatents{Float64, <:NamedTuple}, i::Integer) =
    (posterior_mean(l.λ_home, i), posterior_mean(l.observation_params.r_h, i),
     posterior_mean(l.λ_away, i), posterior_mean(l.observation_params.r_a, i))

crps_parameters(l::SmileLatents{Float64, Nothing}, i::Integer) =
    (posterior_mean(l.λ_home, i), Inf, posterior_mean(l.λ_away, i), Inf)

crps_parameters(l::SmileLatents{Float64, <:NamedTuple}, i::Integer) =
    (posterior_mean(l.λ_home, i), posterior_mean(l.observation_params.r_h, i),
     posterior_mean(l.λ_away, i), posterior_mean(l.observation_params.r_a, i))

function crps_parameters(l::RecombLatents{Float64}, i::Integer)
    dh, da = marginals(l, i)
    return (mean(dh), Inf, mean(da), Inf)
end


# ==============================================================================
# 7. THE KERNELS
# ==============================================================================
#
# One pass over the odds rows, in odds-frame order, with two integer lookups per row.
# `compute_metric(metric, ctx)` is the real entry point; §8 wraps it.

"""
    compute_metric(metric, ctx::EvaluationContext) -> AbstractEvaluationResult

The kernel. Every other `compute_metric` signature builds a context and calls this.
"""
function compute_metric end

# --- 7.1 LogLoss ---------------------------------------------------------------

function compute_metric(m::LogLoss, ctx::EvaluationContext)::LogLossResult
    o = ctx.odds
    p = ctx.probs
    filt = _ue_selection_filter(m)

    model_ll  = Float64[]
    market_ll = Float64[]
    sizehint!(model_ll, o.n)
    sizehint!(market_ll, o.n)

    @inbounds for i in 1:o.n
        o.has_fair[i] || continue            # dropmissing!(:prob_fair_close)
        o.is_winner[i] < 0 && continue       # dropmissing!(:is_winner)
        sel = o.selection[i]
        _ue_passes(filt, sel) || continue
        p̂ = prob_mean(p, o.match_id[i], sel)
        p̂ === nothing && continue            # the innerjoin would have dropped this
        y = Float64(o.is_winner[i])
        push!(model_ll,  calc_logloss(p̂, y))
        push!(market_ll, calc_logloss(o.prob_fair_close[i], y))
    end

    n = length(model_ll)
    n == 0 && return LogLossResult(LogLossComponent(NaN, NaN, NaN, 0))
    mm = mean(model_ll)
    mk = mean(market_ll)
    return LogLossResult(LogLossComponent(mm, mk, mm - mk, n))
end

# --- 7.2 LPD -------------------------------------------------------------------

function compute_metric(m::LPD, ctx::EvaluationContext)::LPDResult
    return m.target === :score ? _ue_lpd_score(m, ctx) : _ue_lpd_market(m, ctx)
end

function _ue_lpd_market(m::LPD, ctx::EvaluationContext)::LPDResult
    o = ctx.odds
    p = ctx.probs
    filt = _ue_selection_filter(m)

    lpd_model  = Float64[]
    lpd_market = Float64[]
    sizehint!(lpd_model, o.n)
    sizehint!(lpd_market, o.n)

    log_liks = Vector{Float64}(undef, p.n_draws)
    expbuf   = Vector{Float64}(undef, p.n_draws)

    @inbounds for i in 1:o.n
        o.has_fair[i] || continue
        o.is_winner[i] < 0 && continue
        sel = o.selection[i]
        _ue_passes(filt, sel) || continue
        v = prob_draws(p, o.match_id[i], sel)
        v === nothing && continue
        y = Float64(o.is_winner[i])
        push!(lpd_model,  calc_lpd_samples!(log_liks, expbuf, v, y))
        push!(lpd_market, calc_lpd_scalar(o.prob_fair_close[i], y))
    end

    return _ue_pack_lpd(lpd_model, lpd_market)
end

"""
    _ue_lpd_score(metric, ctx)

The briefing's LPD: the joint log posterior predictive density of the REALISED
scoreline.

    LPD_i = log( (1/S) Σ_s S_i[g_h+1, g_a+1, s] )

read straight off `06`'s score grid, one fixture at a time, with one grid and one
workspace for the whole sweep. There is no market baseline (§3.2 of `l01_types.jl`), so
`market_lpd` and `diff_lpd` are `NaN`.

A scoreline beyond the grid's truncation cannot be evaluated — the grid stops at
`max_goals` — and such a fixture is SKIPPED and counted out of `n_obs` rather than
scored as `log(0)`. At the repository's `max_goals = 12` that is a 12-goal side.
"""
function _ue_lpd_score(m::LPD, ctx::EvaluationContext)::LPDResult
    l  = ctx.latents
    mg = ctx.max_goals
    ws = GridWorkspace(mg)
    S  = alloc_score_grid(l, mg)
    ids = latent_match_ids(l)
    nd  = n_draws(l)

    out = Float64[]
    sizehint!(out, length(ids))
    probs = Vector{Float64}(undef, nd)

    @inbounds for i in eachindex(ids)
        sc = outcome_of(ctx.outcomes, ids[i])
        sc === nothing && continue
        gh, ga = sc
        (0 <= gh < mg && 0 <= ga < mg) || continue
        compute_score_grid!(S, ws, l, i)
        for k in 1:nd
            probs[k] = S[gh + 1, ga + 1, k]
        end
        # log-mean-exp is unnecessary here: these are probabilities, not log-densities,
        # and the mean of `nd` numbers in [0,1] cannot overflow. The clamp is the same
        # 1e-15 floor `calc_lpd_samples` applies, so a fixture the model gave zero mass
        # scores −34.5 rather than −Inf and does not poison the mean.
        push!(out, log(clamp(mean(probs), 1e-15, Inf)))
    end

    market = fill(NaN, length(out))
    return _ue_pack_lpd(out, market)
end

function _ue_pack_lpd(model::Vector{Float64}, market::Vector{Float64})::LPDResult
    n = length(model)
    n == 0 && return LPDResult(LPDComponent(NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0))
    mm = mean(model)
    mk = isempty(market) || all(isnan, market) ? NaN : mean(market)
    return LPDResult(LPDComponent(
        mm,
        n < 2 ? NaN : std(model),
        n < 3 ? NaN : skewness(model),
        n < 4 ? NaN : kurtosis(model),
        mk,
        mm - mk,
        sum(model),
        n))
end

# --- 7.3 CRPS ------------------------------------------------------------------

function compute_metric(m::CRPS, ctx::EvaluationContext)::CRPSResults
    l = ctx.latents
    ids = latent_match_ids(l)

    crps_home = Float64[]
    crps_away = Float64[]
    sizehint!(crps_home, length(ids))
    sizehint!(crps_away, length(ids))

    @inbounds for i in eachindex(ids)
        sc = outcome_of(ctx.outcomes, ids[i])
        sc === nothing && continue
        gh, ga = sc
        λh, rh, λa, ra = crps_parameters(l, i)
        push!(crps_home, compute_crps(gh, λh, rh; max_goals = m.max_goals))
        push!(crps_away, compute_crps(ga, λa, ra; max_goals = m.max_goals))
    end

    isempty(crps_home) && return CRPSResults(CRPSComponent(NaN), CRPSComponent(NaN),
                                             CRPSComponent(NaN))
    crps_all = (crps_home .+ crps_away) ./ 2.0
    return CRPSResults(CRPSComponent(mean(crps_home)),
                       CRPSComponent(mean(crps_away)),
                       CRPSComponent(mean(crps_all)))
end

# --- 7.4 RQR -------------------------------------------------------------------

"""
    compute_metric(::RQR, ctx) -> RQRResult

`n_sims` independent randomisations, each summarised in full; the SUMMARIES are
averaged (see `RQR`'s docstring for why not the residuals).

THE CALL ORDER IS LOAD-BEARING. Within a replicate every home residual is drawn first,
in fixture order, then every away one — the order `rqr.jl:108-109`'s two broadcasts
produce. A seeded rng therefore reproduces `src`'s draw sequence exactly at
`n_sims = 1`, which is what makes `l05_parity.jl`'s RQR row a real comparison rather
than a comparison of two different random samples.

The marginals are built ONCE and reused across replicates. They do not depend on the
randomisation, and rebuilding a `NegativeBinomial` per replicate per fixture is the
whole cost of a 1,000-replicate run.

`rng` DEFAULTS TO A FRESH `Xoshiro(seed)` — a private stream, so evaluating a metric
never perturbs the caller's global RNG and two evaluations of the same fit agree. It is
a keyword rather than a field so `l05_parity.jl` can hand in a snapshot of the seeded
GLOBAL stream, which is the only stream `src`'s unseeded `rand` can be made to use.
"""
function compute_metric(m::RQR, ctx::EvaluationContext;
                        rng::AbstractRNG = Random.Xoshiro(m.seed))::RQRResult
    l = ctx.latents
    ids = latent_match_ids(l)

    dists_h = UnivariateDistribution[]
    dists_a = UnivariateDistribution[]
    goals_h = Int[]
    goals_a = Int[]
    for i in eachindex(ids)
        sc = outcome_of(ctx.outcomes, ids[i])
        sc === nothing && continue
        dh, da = marginals(l, i)
        push!(dists_h, dh); push!(dists_a, da)
        push!(goals_h, sc[1]); push!(goals_a, sc[2])
    end

    nobs = length(goals_h)
    nan6 = DistributionStats(NaN, NaN, NaN, NaN, NaN, NaN)
    nobs == 0 && return RQRResult(nan6, nan6, nan6)

    n_sims = max(m.n_sims, 1)

    acc_h = zeros(Float64, 6)
    acc_a = zeros(Float64, 6)
    acc_all = zeros(Float64, 6)
    rh = Vector{Float64}(undef, nobs)
    ra = Vector{Float64}(undef, nobs)

    for _ in 1:n_sims
        @inbounds for i in 1:nobs
            rh[i] = compute_rqr(goals_h[i], dists_h[i], rng)
        end
        @inbounds for i in 1:nobs
            ra[i] = compute_rqr(goals_a[i], dists_a[i], rng)
        end
        _ue_accumulate!(acc_h,   summarize_stats(copy(rh)))
        _ue_accumulate!(acc_a,   summarize_stats(copy(ra)))
        _ue_accumulate!(acc_all, summarize_stats(vcat(rh, ra)))
    end

    return RQRResult(_ue_mean_stats(acc_h, n_sims),
                     _ue_mean_stats(acc_a, n_sims),
                     _ue_mean_stats(acc_all, n_sims))
end

function _ue_accumulate!(acc::Vector{Float64}, s::DistributionStats)
    acc[1] += s.mean
    acc[2] += s.std
    acc[3] += s.skewness
    acc[4] += s.kurtosis
    acc[5] += s.shapiro_w
    acc[6] += s.shapiro_p
    return acc
end

# `n == 1` divides by one, which is exact, so a single-replicate run returns the
# summary itself rather than a rounded copy of it. That is what the parity row needs.
_ue_mean_stats(acc::Vector{Float64}, n::Int) =
    DistributionStats(acc[1] / n, acc[2] / n, acc[3] / n,
                      acc[4] / n, acc[5] / n, acc[6] / n)

# --- 7.5 GLMEdge ---------------------------------------------------------------

"""
    compute_metric(::GLMEdge, ctx) -> GLMEdgeResult

`GLM.glm` needs a table, so this one builds a DataFrame — but a three-column one of
exactly the scored rows, not a join of the whole odds frame with the whole PPD.

THE `n_obs < 10` GUARD IS CHECKED BEFORE THE ODDS FILTER, as in `glm_edge.jl:75-82`.
That ordering is `src`'s and it is preserved for parity, even though it means a metric
with 12 rows of which 9 have a missing `odds_close` reaches `glm` with 3 rows and
raises. The behaviour is recorded in `README.md` rather than changed here.
"""
function compute_metric(m::GLMEdge, ctx::EvaluationContext)::GLMEdgeResult
    o = ctx.odds
    p = ctx.probs
    filt = _ue_selection_filter(m)

    prob_fair   = Float64[]
    spread_fair = Float64[]
    Y           = Float64[]
    n_prefilter = 0

    @inbounds for i in 1:o.n
        o.has_fair[i] || continue
        sel = o.selection[i]
        _ue_passes(filt, sel) || continue
        p̂ = prob_mean(p, o.match_id[i], sel)
        p̂ === nothing && continue
        n_prefilter += 1
        # `dropmissing!(analysis_df, [:odds_close, :is_winner])`, glm_edge.jl:82
        o.has_odds_close[i] || continue
        o.is_winner[i] < 0 && continue
        sf = p̂ - o.prob_fair_close[i]
        abs(sf) < m.min_edge && continue
        push!(prob_fair, o.prob_fair_close[i])
        push!(spread_fair, sf)
        push!(Y, Float64(o.is_winner[i]))
    end

    empty_coef = GLMCoefComponent(NaN, NaN, NaN, NaN)
    if n_prefilter < 10
        @warn "GLMEdge: not enough observations ($n_prefilter) for selections " *
              "$(scored_selections(m)). Returning NaNs."
        return GLMEdgeResult(empty_coef, empty_coef, empty_coef, n_prefilter)
    end

    n = length(Y)
    n < 3 && return GLMEdgeResult(empty_coef, empty_coef, empty_coef, n)

    df = DataFrame(Y = Y, prob_fair_close = prob_fair, spread_fair = spread_fair)
    reg = try
        glm(@formula(Y ~ prob_fair_close + spread_fair), df, Binomial(), LogitLink())
    catch e
        @warn "GLMEdge: the logistic fit failed" exception = e
        return GLMEdgeResult(empty_coef, empty_coef, empty_coef, n)
    end

    ct = coeftable(reg)
    function grab(name::String)
        idx = findfirst(==(name), ct.rownms)
        idx === nothing && return empty_coef
        return GLMCoefComponent(ct.cols[1][idx], ct.cols[2][idx],
                                ct.cols[3][idx], ct.cols[4][idx])
    end

    return GLMEdgeResult(grab("(Intercept)"), grab("prob_fair_close"),
                         grab("spread_fair"), n)
end

# --- 7.6 MIQ -------------------------------------------------------------------

function compute_metric(m::MIQ, ctx::EvaluationContext)::MIQResult
    o = ctx.odds
    p = ctx.probs

    q_all   = Float64[]
    won_all = Bool[]
    by_sel  = Dict{Symbol, Tuple{Vector{Float64}, Vector{Bool}}}()
    for s in p.selections
        by_sel[s] = (Float64[], Bool[])
    end

    @inbounds for i in 1:o.n
        sel = o.selection[i]
        v = prob_draws(p, o.match_id[i], sel)
        v === nothing && continue
        # `get_miq` returns `missing` for an absent market probability; `dropmissing`
        # on :market_quantile (miq.jl:117) removes exactly those rows.
        o.has_fair[i] || continue
        q = get_miq(v, o.prob_fair_close[i])
        q === missing && continue
        # `is_winner` is compared with `.== true` / `.== false` (miq.jl:61-62), so a
        # missing outcome falls into NEITHER group. Reproduced by skipping it.
        w = o.is_winner[i]
        w < 0 && continue
        won = w == 1
        push!(q_all, q); push!(won_all, won)
        bucket = by_sel[sel]
        push!(bucket[1], q); push!(bucket[2], won)
    end

    empty_stats = MIQStats(missing, missing, missing, missing, missing, 0, 0)
    stats_for(s::Symbol) = haskey(by_sel, s) ?
        evaluate_group_edge(by_sel[s][1], by_sel[s][2]) : empty_stats

    return MIQResult(
        evaluate_group_edge(q_all, won_all),
        (stats_for(s) for s in MIQ_FIELD_SELECTIONS)...)
end


# ==============================================================================
# 8. THE PUBLIC ENTRY POINTS
# ==============================================================================
#
# Everything above takes a prebuilt `EvaluationContext`. These build one.
#
# CONVERGENCE GATING LIVES HERE TOO, not only in the batch runner. A single
# `compute_metric(metric, fit, ds)` on a run whose R-hat is 1.4 is exactly as wrong as
# a batch of them, and `require_converged = true` refuses it by raising rather than by
# returning a plausible number.

"""
    compute_metric(metric, latents::AbstractPosteriorLatents, odds_df, matches_df; …)

The briefing's second signature. Builds a context for this one metric and scores it.

For more than one metric use `evaluate_fits` (or build an `EvaluationContext`
yourself) — the pricing sweep is the expensive part and it is shared there.
"""
function compute_metric(metric::AbstractScoringRule,
                        latents::AbstractPosteriorLatents,
                        odds_df::AbstractDataFrame,
                        matches_df::AbstractDataFrame;
                        max_goals::Integer = TPL_MAX_GOALS,
                        threaded::Bool = true)
    ctx = evaluation_context(latents, odds_df, matches_df, [metric];
                             max_goals = max_goals, threaded = threaded)
    return compute_metric(metric, ctx)
end

"""
    ConvergenceRefusal

Raised when `require_converged = true` and the fit did not pass its gates.

An exception rather than a `NaN` result: a metric computed on a chain that did not mix
is not a worse number, it is not a number, and a leaderboard that ranks it alongside
converged runs is worse than one that is missing a row.
"""
struct ConvergenceRefusal <: Exception
    fit::String
    failed_gates::Vector{String}
    detail::Vector{String}
end

function Base.showerror(io::IO, e::ConvergenceRefusal)
    print(io, "ConvergenceRefusal: fit `", e.fit, "` did not converge — failed gate(s): ",
          isempty(e.failed_gates) ? "unknown" : join(e.failed_gates, ", "), ".")
    for d in e.detail
        print(io, "\n    ", d)
    end
    print(io, "\n  Pass `require_converged = false` to score it anyway (the result will ",
              "be flagged, not trusted).")
end

"""
    convergence_verdict(fit) -> (passed::Bool, failed_gates, detail)

Read `fit.diagnostics` without assuming it is a `ConvergenceSummary`.

A `Fit` built by `fit_model` always carries one. A `Fit` reconstructed from an
old serialisation, or one someone constructed by hand, may carry anything — and this
function's contract is that it never throws, because a missing audit must degrade to
"unknown", not to a crash inside the gate that exists to prevent crashes.

An UNKNOWN verdict counts as NOT PASSED under `require_converged = true`. Treating an
unmeasured run as converged would let a container earn a clean bill of health by
recording nothing, which is the same abstention rule `07/l02_convergence.jl` applies
to individual gates.
"""
function convergence_verdict(fit)
    diag = try
        getfield(fit, :diagnostics)
    catch
        nothing
    end
    diag isa ConvergenceSummary || return (false, ["no audit"],
        ["this container carries no ConvergenceSummary — re-run `fit_model`, or " *
         "`audit_convergence(fit)` if you have the folds."])
    return (diag.passed, copy(diag.failed_gates), copy(diag.failures))
end

"""
    fit_latents(fit) -> AbstractPosteriorLatents

The typed OOS container a `Fit` carries, with an actionable error when it does not.
"""
function fit_latents(fit)
    lat = try
        getfield(fit, :latents)
    catch
        nothing
    end
    lat isa AbstractPosteriorLatents && return lat
    lat === nothing && error(
        "this Fit carries no typed latents, so there is nothing to score. `fit_model` " *
        "records the reason in `fit.config.tags` when a model's family is not " *
        "registered with `latent_family` (06/l02_extract.jl §1).")
    error("expected a typed posterior container, got a $(typeof(lat)).")
end

"""
    compute_metric(metric, fit::Fit, ds::DataStore; require_converged = false, …)

The briefing's first signature: score one run, straight off its typed latents.

No `extract_oos_predictions`, no re-derived boundaries, no rebuilt feature sets — the
container was extracted by the run that produced it (`07`) and is read, not recomputed.
"""
function compute_metric(metric::AbstractScoringRule, fit::Fit, ds::UE_D.DataStore;
                        require_converged::Bool = false,
                        max_goals::Integer = TPL_MAX_GOALS,
                        threaded::Bool = true)
    if require_converged
        passed, gates, detail = convergence_verdict(fit)
        passed || throw(ConvergenceRefusal(fit_name(fit), gates, detail))
    end
    return compute_metric(metric, fit_latents(fit), ds.odds, ds.matches;
                          max_goals = max_goals, threaded = threaded)
end

"""
    compute_metric(metric, fit::Fit, ds::DataStore, latents) -> result

The LEGACY four-argument shape (`src/evaluation/interfaces.jl:8`), preserved so a call
site that already holds its latents does not have to change.

`latents` may be a typed container, an `Experiments.LatentStates`, this framework's
`LatentStates`, or a raw legacy `DataFrame`. `as_typed_latents` reconciles them; where
it cannot — the recombination family cannot be rebuilt from a legacy frame, because
`λ_total − λ_open` is one equation in two unknowns (06/README) — it says so.
"""
function compute_metric(metric::AbstractScoringRule, fit, ds::UE_D.DataStore, latents;
                        require_converged::Bool = false,
                        max_goals::Integer = TPL_MAX_GOALS,
                        threaded::Bool = true)
    if require_converged
        passed, gates, detail = convergence_verdict(fit)
        passed || throw(ConvergenceRefusal(string(_ue_fit_name(fit)), gates, detail))
    end
    typed = as_typed_latents(latents, _ue_fit_model(fit))
    return compute_metric(metric, typed, ds.odds, ds.matches;
                          max_goals = max_goals, threaded = threaded)
end

"""
    as_typed_latents(latents, model) -> AbstractPosteriorLatents

Normalise whatever a legacy caller passed into a typed container.

| given                                   | how                                     |
|-----------------------------------------|-----------------------------------------|
| an `AbstractPosteriorLatents`           | returned as-is                          |
| anything with a `.latents` container    | unwrapped (both `LatentStates` shapes)  |
| anything with a `.df`                   | `latents_from_legacy_dataframe`         |
| a `DataFrame`                           | `latents_from_legacy_dataframe`         |

The `DataFrame` route needs the MODEL, because the family — and therefore the schema —
is a property of the engine and not of the columns: a frame with `λ_h`, `λ_a` and no
`r` is a Poisson container for one model and a mis-extracted NegBin one for another.
"""
function as_typed_latents(latents, model = nothing)
    latents isa AbstractPosteriorLatents && return latents
    if hasproperty(latents, :latents)
        inner = getproperty(latents, :latents)
        inner isa AbstractPosteriorLatents && return inner
    end
    df = latents isa AbstractDataFrame ? latents :
         hasproperty(latents, :df) ? getproperty(latents, :df) : nothing
    df === nothing && error(
        "as_typed_latents: cannot read a $(typeof(latents)). Expected a typed " *
        "container (06), a LatentStates, or a legacy DataFrame.")
    m = model
    if m === nothing && hasproperty(latents, :model)
        m = getproperty(latents, :model)
    end
    m === nothing && error(
        "as_typed_latents: rebuilding a typed container from a legacy DataFrame needs " *
        "the model — the family determines the schema. Pass the model, or hand over " *
        "`fit.latents` instead.")
    return latents_from_legacy_dataframe(m, df)
end

# Duck-typed readers, so the four-argument bridge works on a `Fit`, on a genuine
# `BayesianFootball.Experiments.ExperimentResults`, and on anything else that carries a
# `.config` — which is every container this repository has ever put a model in.
_ue_fit_model(fit) = hasproperty(fit, :config) && hasproperty(fit.config, :model) ?
                     fit.config.model : nothing
_ue_fit_name(fit)  = hasproperty(fit, :config) && hasproperty(fit.config, :name) ?
                     String(fit.config.name) : string(typeof(fit))
