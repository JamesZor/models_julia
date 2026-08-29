# src/evaluation/pricing.jl
#
# The replacement for `Predictions.model_inference` on the evaluation path.
#
# `model_inference` prices every market in `DEFAULT_MARKET_CONFIG` and stores each
# fixture's posterior as a `Vector{Float64}` inside a `Vector{Any}` DataFrame column.
# Here the same posterior is priced over the markets the metrics ACTUALLY named, into
# one dense `(n_draws × n_fixtures × n_selections)` tensor, by the zero-allocation
# kernels in `src/predictions/score_grids/kernels.jl`.
#
# THE ALLOCATION CONTRACT. Everything this file allocates it allocates ONCE, before the
# sweep: one `EvaluationWorkspace` per worker, one destination tensor for the batch.
# `price_match_markets!` — the per-fixture unit — measures 0 bytes, and the test suite
# asserts that against a warmed baseline rather than assuming it.
#
# The reason to care is the same reason `06`'s kernels care: a 900-fixture fold at 1,200
# draws over eleven selections is ~95 MB of posterior probabilities, and allocating it a
# fixture at a time is what schedules the garbage collector in the middle of a sweep.

export alloc_evaluation_workspace, price_match_markets!, market_probabilities


# ==============================================================================
# 1. THE WORKSPACE
# ==============================================================================

"""
    alloc_evaluation_workspace(latents, markets; max_goals = 12) -> EvaluationWorkspace

One workspace per WORKER — not one per fixture, and not one per market.

Holds the score-grid destination, the marginal-PMF scratch, one market book per market,
and (for the smile family) the `SmileScoreGrid` holder its pricer needs. `markets` and
`books` are frozen into `Tuple`s so the per-fixture loop over markets unrolls at compile
time instead of dispatching per element of an abstractly-typed vector.
"""
function alloc_evaluation_workspace(l::AbstractPosteriorLatents,
                                    markets::AbstractVector;
                                    max_goals::Integer = Predictions.TPL_MAX_GOALS)
    mg = Int(max_goals)
    nd = n_draws(l)
    grid = alloc_score_grid(l, mg)
    ws = GridWorkspace(mg)
    mkts = Tuple(markets)
    books = Tuple(alloc_market_book(m, nd) for m in markets)

    offsets = Int[]
    let acc = 0
        for m in markets
            push!(offsets, acc)
            acc += Predictions.market_arity(m)
        end
    end

    smile = _alloc_smile_holder(l, grid)
    return EvaluationWorkspace(grid, ws, books, mkts, offsets, smile, mg)
end

# Dispatch, not a flag: the ordinary containers emit no test for a smile holder they
# cannot have, and `EvaluationWorkspace`'s third type parameter records which it is.
_alloc_smile_holder(::AbstractPosteriorLatents, ::Array{Float64,3}) = nothing

function _alloc_smile_holder(l::SmileLatents, grid::Array{Float64,3})
    buf = alloc_smile_buffers(l)
    return SmileScoreGrid(grid, buf.λ_tot, buf.φ, copy(l.strikes))
end


# ==============================================================================
# 2. PRICING ONE FIXTURE
# ==============================================================================

"""
    price_match_markets!(probs, wsp, latents, i) -> probs

Price fixture row `i` on every market in the workspace, writing into `probs.means` (and
`probs.draws` when the tensor keeps them). 0 bytes.

The two inner steps are `compute_score_grid!` and `price_market!`, both of which measure
zero allocations; this function adds a `mean` over each book vector and, when draws are
kept, a `copyto!` into a contiguous tensor column.

`i` indexes the LATENT CONTAINER, and `probs.match_ids[i]` is the fixture it belongs to.
The two share an order by construction — `market_probabilities` copies the container's
own `latent_match_ids` — and `verify_alignment` checks that they still do.
"""
function price_match_markets!(probs::MarketProbabilities,
                              wsp::EvaluationWorkspace,
                              l::AbstractPosteriorLatents,
                              i::Int)
    compute_score_grid!(wsp.grid, wsp.ws, l, i)
    target = _price_target(wsp, l, i)
    _price_books!(probs, wsp, target, i, wsp.markets, wsp.books, 1)
    return probs
end

# The ordinary families price straight off the grid; the smile family prices off a
# `SmileScoreGrid` whose per-fixture buffers have to be refilled first.
@inline _price_target(wsp::EvaluationWorkspace{M,B,Nothing}, l, i) where {M,B} = wsp.grid

@inline function _price_target(wsp::EvaluationWorkspace{M,B,SmileScoreGrid},
                               l::SmileLatents, i::Int) where {M,B}
    fill_smile_buffers!(wsp.smile.λ_tot, wsp.smile.φ, l, i)
    return wsp.smile
end

# Tuple recursion rather than a loop over a `Vector{AbstractMarket}`: the market count
# and types are in the workspace's type, so this unrolls into straight-line code and the
# `price_market!` call sites are concrete.
@inline _price_books!(::MarketProbabilities, ::EvaluationWorkspace, _, ::Int,
                      ::Tuple{}, ::Tuple{}, ::Int) = nothing

@inline function _price_books!(probs::MarketProbabilities, wsp::EvaluationWorkspace,
                               target, i::Int, mkts::Tuple, books::Tuple, j::Int)
    book = first(books)
    price_market!(book, target, first(mkts))
    off = wsp.offsets[j]
    @inbounds for s in eachindex(book)
        v = book[s]
        c = off + s
        probs.means[i, c] = mean(v)
        if probs.keep_draws
            copyto!(view(probs.draws, :, i, c), v)
        end
    end
    return _price_books!(probs, wsp, target, i, Base.tail(mkts), Base.tail(books), j + 1)
end


# ==============================================================================
# 3. THE BATCH SWEEP
# ==============================================================================

"""
    market_probabilities(latents, markets; keep_draws = true, max_goals = 12,
                         threaded = true) -> MarketProbabilities

Price every fixture in `latents` on every market in `markets`.

THREADED OVER FIXTURES, WITH A WORKSPACE PER CHUNK. Fixtures are independent and each
writes a disjoint slice of the output, so the result is bit-identical to the sequential
one — the thread count changes the wall clock and nothing else. Chunked with `@spawn`
rather than `@threads` so the workspace binding does not depend on `threadid()`, which
is not stable across a task yield.

`keep_draws = false` skips the `n_draws × n_fixtures × n_selections` tensor entirely.
LogLoss and GLMEdge need only means; LPD and MIQ need the draws.
"""
function market_probabilities(l::AbstractPosteriorLatents,
                              markets::AbstractVector;
                              keep_draws::Bool = true,
                              max_goals::Integer = Predictions.TPL_MAX_GOALS,
                              threaded::Bool = true)
    mkts = AbstractMarket[m for m in markets]
    isempty(mkts) && return MarketProbabilities(
        mkts, Int[], Symbol[], Dict{Int,Int}(), Dict{Symbol,Int}(),
        Array{Float64,3}(undef, 0, 0, 0), Matrix{Float64}(undef, 0, 0),
        n_draws(l), false)

    ids = copy(latent_match_ids(l))
    nf = length(ids)
    nd = n_draws(l)

    sels = Symbol[]
    for m in mkts, k in market_keys(m)
        k in sels && error(
            "market_probabilities: selection :$k is claimed by two markets in the " *
            "list. Every selection symbol must name exactly one market — see " *
            "`market_for_selection`.")
        push!(sels, k)
    end

    probs = MarketProbabilities(
        mkts, ids, sels,
        Dict{Int,Int}(id => i for (i, id) in enumerate(ids)),
        Dict{Symbol,Int}(s => c for (c, s) in enumerate(sels)),
        keep_draws ? Array{Float64,3}(undef, nd, nf, length(sels)) :
                     Array{Float64,3}(undef, 0, 0, 0),
        Matrix{Float64}(undef, nf, length(sels)),
        nd, keep_draws)

    function sweep(lo::Int, hi::Int)
        wsp = alloc_evaluation_workspace(l, mkts; max_goals = max_goals)
        for i in lo:hi
            price_match_markets!(probs, wsp, l, i)
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

    return probs
end
