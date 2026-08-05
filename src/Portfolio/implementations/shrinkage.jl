# src/Portfolio/implementations/shrinkage.jl

export NoShrinkage, FractionalKelly, BakerMcHale

"Keep the point-estimate allocation as-is."
struct NoShrinkage <: AbstractShrinkage end
shrink_factor(::NoShrinkage, ::Any, ::AbstractMatrix, ::AbstractVector,
              ::AbstractAllocator, ::ExecutionConfig; seed_offset::Integer = 0) = 1.0

"Fixed fraction of full Kelly. `FractionalKelly(0.5)` is the folk half-Kelly."
struct FractionalKelly <: AbstractShrinkage
    k::Float64
    function FractionalKelly(k::Real)
        0.0 <= k <= 1.0 || throw(ArgumentError("k must be in [0,1]: $k"))
        new(Float64(k))
    end
end
shrink_factor(s::FractionalKelly, ::Any, ::AbstractMatrix, ::AbstractVector,
              ::AbstractAllocator, ::ExecutionConfig; seed_offset::Integer = 0) = s.k

"""
    BakerMcHale(; n_draws = 128, grid = 0.0:0.02:1.0, seed = 20260805)

Baker & McHale (2013) shrinkage under parameter uncertainty, generalised to the
non-mutually-exclusive portfolio.

Re-solves the allocator on each posterior draw `q_j`, then picks the single `k` maximising

    U(k) = (1/m) sum_j  sum_w  p_w log(1 + k * R_w(a*(q_j)))

`U` is strictly concave in `k` (its second derivative is a sum of strictly negative terms), so
the grid argmax is the global optimum and no solver is needed for the outer loop.

The point of this is that the naive allocation optimises against the posterior *mean* and so
inherits none of the spread: on ScottishLower, 35% of matches have negative expected growth at
full Kelly once the posterior is integrated over, and the median `k*` is 0.64.

This is the multi-bet generalisation of `Signals.BayesianKelly`
(`src/signals/implementations/kelly.jl`), which solves the same problem for a single isolated
bet. The two are deliberately separate: this one needs the joint payoff matrix, that one does
not.

Caveat worth knowing: `k*` prices only the uncertainty the model *knows about*. It is not a
substitute for the drawdown budget, and empirically it is still over-levered relative to a
flat quarter-Kelly.
"""
Base.@kwdef struct BakerMcHale <: AbstractShrinkage
    n_draws::Int = 128
    grid::Vector{Float64} = collect(0.0:0.02:1.0)
    seed::Int = 20260805
end

function shrink_factor(s::BakerMcHale, score_matrix, R::AbstractMatrix{Float64},
                       p_true::AbstractVector{Float64}, alloc::AbstractAllocator,
                       exec::ExecutionConfig; seed_offset::Integer = 0)
    size(R, 2) == 0 && return 1.0
    n_samples = size(score_matrix.data, 3)
    n_samples <= 1 && return 1.0

    rng   = Random.MersenneTwister(s.seed + Int(seed_offset))
    draws = Random.randperm(rng, n_samples)[1:min(s.n_draws, n_samples)]

    # portfolio return vector implied by each draw's own optimal allocation
    port = Vector{Vector{Float64}}(undef, length(draws))
    @inbounds for (i, j) in enumerate(draws)
        q = vec(score_matrix.data[:, :, j])          # copy -> concrete Vector{Float64}
        q ./= sum(q)
        port[i] = R * allocate(alloc, q, R, exec).a
    end

    best_k, best_u = 0.0, -Inf
    for k in s.grid
        u = 0.0
        for r in port
            u += dot(p_true, log.(max.(1.0 .+ k .* r, 1e-12)))
        end
        if u > best_u
            best_u, best_k = u, k
        end
    end
    return best_k
end
