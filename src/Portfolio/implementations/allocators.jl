# src/Portfolio/implementations/allocators.jl

export KellyLogUtility

"""
    KellyLogUtility()

Jacot & Mochkovitch (2023) eq. (12)-(14): the Kelly criterion for *non-mutually-exclusive* bets.

    maximise   G(a) = sum_w p_w log(1 + R_w' a)
    subject to 0 <= a_j <= max_selection_stake,  sum(a) <= budget

`G` is concave and the feasible set convex, so the KKT point is the global optimum.

The budget is enforced with a log-barrier that appears in **both** the objective and the
gradient. That symmetry matters: the prototype put a hard `Inf` in the objective only, so the
optimiser had no interior signal steering it away from the cliff and the line search had to
discover the boundary by backtracking.
"""
struct KellyLogUtility <: AbstractAllocator end

function allocate(::KellyLogUtility, p::AbstractVector{Float64}, R::AbstractMatrix{Float64},
                  exec::ExecutionConfig)
    n = size(R, 2)
    n == 0 && return (a = Float64[], kkt = 0.0, converged = true)

    B, mu, ub = exec.budget, exec.barrier_mu, exec.max_selection_stake

    function obj(a)
        s = sum(a)
        s >= B && return Inf
        w = 1.0 .+ R * a
        any(<=(1e-10), w) && return Inf
        return -dot(p, log.(w)) - mu * log(B - s)
    end

    function grad!(g, a)
        s = sum(a)
        w = 1.0 .+ R * a
        if s >= B || any(<=(1e-10), w)
            fill!(g, 1e6)
            return g
        end
        g .= -(R' * (p ./ w)) .+ mu / (B - s)
        return g
    end

    res = Optim.optimize(obj, grad!, zeros(n), fill(ub, n), fill(1e-3, n),
                         Optim.Fminbox(Optim.LBFGS()))
    a = copy(Optim.minimizer(res))
    a[a .< exec.min_selection_stake] .= 0.0

    return (a = a, kkt = kkt_residual(a, p, R, exec), converged = Optim.converged(res))
end

"""
    kkt_residual(a, p, R, exec) -> Float64

Worst first-order-condition violation of `a` on the *unbarriered* problem.

For `min -G(a)` s.t. `a >= 0`, `a <= ub`, `sum(a) <= B`, stationarity is
`grad_j + nu - mu_lo,j + mu_hi,j = 0` with `nu >= 0` and `nu * (sum(a) - B) = 0`. Every interior
coordinate must therefore share the *same* multiplier `nu = -grad_j`; the residual is the spread
of the gradient over interior coordinates plus the sign conditions at the bounds.

Checking `abs(grad_j)` directly -- the obvious thing -- is wrong whenever the budget binds, and
reports a healthy solve as a failure.
"""
function kkt_residual(a::Vector{Float64}, p::AbstractVector{Float64},
                      R::AbstractMatrix{Float64}, exec::ExecutionConfig)
    isempty(a) && return 0.0
    ub, B = exec.max_selection_stake, exec.budget
    w  = 1.0 .+ R * a
    any(<=(0.0), w) && return Inf
    gr = -(R' * (p ./ w))

    interior = [j for j in eachindex(a) if a[j] > 0.0 && a[j] < ub - 1e-9]
    nu = (sum(a) < B - 1e-6 || isempty(interior)) ? 0.0 : max(0.0, -mean(gr[interior]))

    kkt = 0.0
    @inbounds for j in eachindex(a)
        if a[j] <= 0.0
            kkt = max(kkt, max(0.0, -(gr[j] + nu)))      # gr_j + nu >= 0
        elseif a[j] >= ub - 1e-9
            kkt = max(kkt, max(0.0, gr[j] + nu))         # gr_j + nu <= 0
        else
            kkt = max(kkt, abs(gr[j] + nu))              # gr_j + nu == 0
        end
    end
    return kkt
end

"Expected log-growth of `a` under belief `p`; `-Inf` outside the wealth-positive region."
function growth(a::AbstractVector{Float64}, p::AbstractVector{Float64},
                R::AbstractMatrix{Float64})
    w = 1.0 .+ R * a
    any(<=(1e-12), w) && return -Inf
    return dot(p, log.(w))
end
