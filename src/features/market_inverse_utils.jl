# src/features/market_inverse_utils.jl
using Distributions
using Optim
using LinearAlgebra # For diag, tril, triu

using ..MyDistributions: FrankCopulaNegBin, DoubleNegativeBinomial

# ---------------------------------------------------------
# 1. Core Mathematical Helpers
# ---------------------------------------------------------

function dixon_coles_tau(i::Int, j::Int, λ_h::Float64, λ_a::Float64, ρ::Float64)::Float64
    if i == 0 && j == 0 
        return 1.0 - (λ_h * λ_a * ρ)
    elseif i == 0 && j == 1 
        return 1.0 + (λ_h * ρ)
    elseif i == 1 && j == 0 
        return 1.0 + (λ_a * ρ)
    elseif i == 1 && j == 1 
        return 1.0 - ρ
    else 
        return 1.0 
    end
end

function _build_probability_matrix_dixon(λh::Float64, λa::Float64, ρ::Float64, max_goals::Integer)::Matrix{Float64}
    P = zeros(Float64, max_goals + 1, max_goals + 1)
    dist_h = Poisson(λh)
    dist_a = Poisson(λa)
    for j in 0:max_goals
        for i in 0:max_goals
            P[i+1, j+1] = pdf(dist_h, i) * pdf(dist_a, j) * dixon_coles_tau(i, j, λh, λa, ρ)
        end
    end
    # Ensure all values are positive and sum to 1
    P = max.(P, 0.0)
    return P ./ sum(P)
end

function _build_probability_matrix_frank_negbin(λh::Float64, λa::Float64, rh::Float64, ra::Float64, κ::Float64, max_goals::Integer)::Matrix{Float64}
    dist = FrankCopulaNegBin(rh, λh, ra, λa, κ)
    P = zeros(Float64, max_goals + 1, max_goals + 1)
    for j in 0:max_goals
        for i in 0:max_goals
            P[i+1, j+1] = exp(logpdf(dist, i, j))
        end
    end
    P = max.(P, 0.0)
    return P ./ sum(P)
end

function _build_probability_matrix_double_negbin(λh::Float64, λa::Float64, rh::Float64, ra::Float64, max_goals::Integer)::Matrix{Float64}
    dist = DoubleNegativeBinomial(λh, λa, rh, ra)
    P = zeros(Float64, max_goals + 1, max_goals + 1)
    for j in 0:max_goals
        for i in 0:max_goals
            P[i+1, j+1] = exp(logpdf(dist, [i, j]))
        end
    end
    P = max.(P, 0.0)
    return P ./ sum(P)
end

function _build_probability_matrix_dixon_negbin(λh::Float64, λa::Float64, rh::Float64, ra::Float64, ρ::Float64, max_goals::Integer)::Matrix{Float64}
    dist = DoubleNegativeBinomial(λh, λa, rh, ra)
    P = zeros(Float64, max_goals + 1, max_goals + 1)
    for j in 0:max_goals
        for i in 0:max_goals
            P[i+1, j+1] = exp(logpdf(dist, [i, j])) * dixon_coles_tau(i, j, λh, λa, ρ)
        end
    end
    P = max.(P, 0.0)
    return P ./ sum(P)
end

# ---------------------------------------------------------
# 2. Market Error Calculators (Using Multiple Dispatch)
# ---------------------------------------------------------

function _calculate_error(::Val{:result_1x2}, P::Matrix{Float64}, targets::Dict{Symbol,Float64})
    err = 0.0
    if haskey(targets, :home) err += (sum(tril(P, -1)) - targets[:home])^2 end
    if haskey(targets, :draw) err += (sum(diag(P)) - targets[:draw])^2 end
    if haskey(targets, :away) err += (sum(triu(P, 1)) - targets[:away])^2 end
    return err
end

function _calculate_error(::Val{:btts}, P::Matrix{Float64}, targets::Dict{Symbol,Float64})
    err = 0.0
    if haskey(targets, :btts_yes) || haskey(targets, :btts_no)
        prob_btts = sum(@views P[2:end, 2:end]) 
        if haskey(targets, :btts_yes) err += (prob_btts - targets[:btts_yes])^2 end
        if haskey(targets, :btts_no)  err += ((1.0 - prob_btts) - targets[:btts_no])^2 end
    end 
    return err
end

function _calculate_error_underover(k::Integer, P::Matrix{Float64}, targets::Dict{Symbol,Float64})
    err = 0.0
    over_key = Symbol("over_$(k)5")
    under_key = Symbol("under_$(k)5")

    if haskey(targets, over_key) || haskey(targets, under_key)
        prob_under = 0.0
        max_goals = size(P, 1) - 1 
        for j in 0:max_goals
            for i in 0:max_goals
                if (i + j) <= k
                    prob_under += P[i+1, j+1]
                end
            end
        end
        prob_over = 1.0 - prob_under
        if haskey(targets, over_key)  err += (prob_over - targets[over_key])^2 end
        if haskey(targets, under_key) err += (prob_under - targets[under_key])^2 end
    end
    return err
end

function _calculate_error(::Val{:uo}, P::Matrix{Float64}, targets::Dict{Symbol,Float64}; min_k::Integer=0, max_k::Integer=8)
    err = 0.0
    for k in min_k:max_k
        err += _calculate_error_underover(k, P, targets)
    end
    return err
end

# Handle individual over/under lines explicitly for the dispatch
for k in 0:9
    @eval function _calculate_error(::Val{$(QuoteNode(Symbol("over_$(k)5")))}, P::Matrix{Float64}, targets::Dict{Symbol,Float64})
        return _calculate_error_underover($k, P, targets)
    end
    @eval function _calculate_error(::Val{$(QuoteNode(Symbol("under_$(k)5")))}, P::Matrix{Float64}, targets::Dict{Symbol,Float64})
        return _calculate_error_underover($k, P, targets)
    end
end

_calculate_error(::Val{T}, P::Matrix{Float64}, targets::Dict{Symbol,Float64}) where T = 0.0

# ---------------------------------------------------------
# 3. Multiple Dispatch Architecture for Configs
# ---------------------------------------------------------

get_initial_guess(::DoublePoissonMarketFeature) = [log(1.5), log(1.0)]
get_initial_guess(::DixonColesMarketFeature) = [log(1.5), log(1.0), 0.05]
get_initial_guess(::RegularizedFrankCopulaMarketFeature) = [log(1.5), log(1.0), log(15.0), log(15.0), 0.1]
get_initial_guess(::RegularizedDoubleNegativeBinomialMarketFeature) = [log(1.5), log(1.0), log(15.0), log(15.0)]
get_initial_guess(::RegularizedDixonColesNegativeBinomialMarketFeature) = [log(1.5), log(1.0), log(15.0), log(15.0), 0.05]

function build_probability_matrix(::DoublePoissonMarketFeature, θ::Vector{Float64}, max_goals::Int)
    λh, λa = exp(θ[1]), exp(θ[2])
    return _build_probability_matrix_dixon(λh, λa, 0.0, max_goals)
end

function build_probability_matrix(::DixonColesMarketFeature, θ::Vector{Float64}, max_goals::Int)
    λh, λa, ρ = exp(θ[1]), exp(θ[2]), 0.3 * tanh(θ[3])
    return _build_probability_matrix_dixon(λh, λa, ρ, max_goals)
end

function build_probability_matrix(::RegularizedFrankCopulaMarketFeature, θ::Vector{Float64}, max_goals::Int)
    λh, λa, rh, ra, κ = exp(θ[1]), exp(θ[2]), exp(θ[3]), exp(θ[4]), θ[5]
    return _build_probability_matrix_frank_negbin(λh, λa, rh, ra, κ, max_goals)
end

function build_probability_matrix(::RegularizedDoubleNegativeBinomialMarketFeature, θ::Vector{Float64}, max_goals::Int)
    λh, λa, rh, ra = exp(θ[1]), exp(θ[2]), exp(θ[3]), exp(θ[4])
    return _build_probability_matrix_double_negbin(λh, λa, rh, ra, max_goals)
end

function build_probability_matrix(::RegularizedDixonColesNegativeBinomialMarketFeature, θ::Vector{Float64}, max_goals::Int)
    λh, λa, rh, ra, ρ = exp(θ[1]), exp(θ[2]), exp(θ[3]), exp(θ[4]), 0.3 * tanh(θ[5])
    return _build_probability_matrix_dixon_negbin(λh, λa, rh, ra, ρ, max_goals)
end

extract_parameters(::DoublePoissonMarketFeature, θ) = (λ_home=exp(θ[1]), λ_away=exp(θ[2]), ρ=0.0)
extract_parameters(::DixonColesMarketFeature, θ) = (λ_home=exp(θ[1]), λ_away=exp(θ[2]), ρ=0.3 * tanh(θ[3]))
extract_parameters(::RegularizedFrankCopulaMarketFeature, θ) = (λ_home=exp(θ[1]), λ_away=exp(θ[2]), r_home=exp(θ[3]), r_away=exp(θ[4]), κ=θ[5])
extract_parameters(::RegularizedDoubleNegativeBinomialMarketFeature, θ) = (λ_home=exp(θ[1]), λ_away=exp(θ[2]), r_home=exp(θ[3]), r_away=exp(θ[4]))
extract_parameters(::RegularizedDixonColesNegativeBinomialMarketFeature, θ) = (λ_home=exp(θ[1]), λ_away=exp(θ[2]), r_home=exp(θ[3]), r_away=exp(θ[4]), ρ=0.3 * tanh(θ[5]))

compute_loss_penalty(config::AbstractMarketFeatureConfig, θ::Vector{Float64}) = 0.0

function compute_loss_penalty(config::RegularizedFrankCopulaMarketFeature, θ::Vector{Float64})
    rh, ra = exp(θ[3]), exp(θ[4])
    penalty = (log(rh) - log(config.prior_r))^2 + (log(ra) - log(config.prior_r))^2
    return config.penalty_weight * penalty
end

function compute_loss_penalty(config::RegularizedDoubleNegativeBinomialMarketFeature, θ::Vector{Float64})
    rh, ra = exp(θ[3]), exp(θ[4])
    penalty = (log(rh) - log(config.prior_r))^2 + (log(ra) - log(config.prior_r))^2
    return config.penalty_weight * penalty
end

function compute_loss_penalty(config::RegularizedDixonColesNegativeBinomialMarketFeature, θ::Vector{Float64})
    rh, ra = exp(θ[3]), exp(θ[4])
    penalty = (log(rh) - log(config.prior_r))^2 + (log(ra) - log(config.prior_r))^2
    return config.penalty_weight * penalty
end

# ---------------------------------------------------------
# 4. The Main Wrapper & Optimizer
# ---------------------------------------------------------

function fit_market_implied_parameters(match_df, config::AbstractMarketFeatureConfig; max_goals=10)
    targets = Dict{Symbol, Float64}(row.selection => row.prob_fair_close for row in eachrow(match_df))

    function loss(θ::Vector{Float64})
        P = build_probability_matrix(config, θ, max_goals)
        sse = 0.0
        for line in config.lines
            sse += _calculate_error(Val(line), P, targets)
        end
        return sse + compute_loss_penalty(config, θ)
    end

    result = optimize(loss, get_initial_guess(config), NelderMead())
    
    return (
        match_id = first(match_df.match_id),
        minimizer = Optim.minimizer(result)
    )
end
