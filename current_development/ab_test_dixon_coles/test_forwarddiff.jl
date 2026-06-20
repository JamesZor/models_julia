using ForwardDiff
using Distributions
using BenchmarkTools

function test_clamp(λ_goals_h, λ_goals_a, ρ, idx_00, length_home)
    max_rho = min.(1.0 ./ (λ_goals_h .* λ_goals_a) .- 1e-4, 1.0 - 1e-4)
    min_rho = max.(-1.0 ./ λ_goals_h .+ 1e-4, -1.0 ./ λ_goals_a .+ 1e-4)
    ρ_match = clamp.(ρ, min_rho, max_rho)

    τ_term = ones(eltype(λ_goals_h), length_home)
    if !isempty(idx_00) 
        τ_term[idx_00] = 1.0 .- (λ_goals_h[idx_00] .* λ_goals_a[idx_00] .* ρ_match[idx_00]) 
    end
    return sum(τ_term)
end

h = [3.0, 4.0, 5.0]
a = [2.0, 3.0, 4.0]
ρ_val = 0.3
idx = [1, 2]
len = 3

# Test forward diff
f(x) = test_clamp(h .* x[1], a .* x[2], ρ_val * x[3], idx, len)
println(ForwardDiff.gradient(f, [1.0, 1.0, 1.0]))
