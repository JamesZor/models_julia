module RebuildEquations

using SpecialFunctions: loggamma
using Statistics: mean

export PRIMITIVE_PARAMETER_FIELDS, DERIVED_PARAMETER_FIELDS, validate_primitive_parameters,
       validate_feature_set, primitive_parameter_length, flatten_primitives, unflatten_primitives,
       transformed_parameters, equation_data, component_rates, weighted_data_loglikelihood,
       weighted_data_loglikelihood_scalar, predictive_component_rates

"""The complete sampled-parameter contract; all other quantities are deterministic."""
const PRIMITIVE_PARAMETER_FIELDS = (:zA, :zD, :kappa_A, :kappa_D, :mu_Y, :Delta,
    :zM, :xi_M, :b_Y, :pen_base, :pen_home, :q_pen, :lambda_og)
const DERIVED_PARAMETER_FIELDS = (:tau_A, :tau_D, :alpha, :beta, :L, :sigma_M, :M)
const _RATE_FLOOR = 1e-6
const _LOG_RATE_BOUND = 20.0

# Hard clamp has parameter-dependent control flow that a compiled ReverseDiff tape records
# statically. Smooth saturation is branch-free and keeps log rates in (-bound, bound).
_saturate_log_rate(x) = _LOG_RATE_BOUND * tanh(x / _LOG_RATE_BOUND)

"""Validate dimensions and supports before entering generic differentiable functions."""
function validate_primitive_parameters(p; n_teams::Integer)
    keys(p) == PRIMITIVE_PARAMETER_FIELDS || throw(ArgumentError("primitive fields must be exactly $(PRIMITIVE_PARAMETER_FIELDS)"))
    J = Int(n_teams); J > 0 || throw(ArgumentError("n_teams must be positive"))
    length(p.zA) == J || throw(DimensionMismatch("zA must have length $J"))
    length(p.zD) == J || throw(DimensionMismatch("zD must have length $J"))
    length(p.zM) == 12 || throw(DimensionMismatch("zM must have length 12"))
    all(isfinite, p.zA) && all(isfinite, p.zD) && all(isfinite, p.zM) || throw(ArgumentError("raw effects must be finite"))
    for k in (:kappa_A, :kappa_D, :mu_Y, :Delta, :xi_M, :b_Y, :pen_base, :pen_home)
        isfinite(getproperty(p, k)) || throw(ArgumentError("$k must be finite"))
    end
    isfinite(p.q_pen) && 0 < p.q_pen < 1 || throw(ArgumentError("q_pen must be finite and strictly between zero and one"))
    isfinite(p.lambda_og) && p.lambda_og > 0 || throw(ArgumentError("lambda_og must be finite and positive"))
    return p
end

"""Validate the concrete Stage-3 FeatureSet data contract, outside the hot path."""
function validate_feature_set(fs)
    J = Int(fs[:n_teams]); J > 0 || throw(ArgumentError("n_teams must be positive"))
    names = (:home_team,:away_team,:Y_home,:Y_away,:A_home,:A_away,:C_home,:C_away,:O_home,:O_away,:month_ids,:league_ids,:weights)
    n = length(fs[:home_team]); n > 0 || throw(ArgumentError("feature set is empty"))
    all(length(fs[k]) == n for k in names) || throw(DimensionMismatch("feature vectors must have common length"))
    all(fs[k] isa Vector{Int} for k in names[1:12]) || throw(ArgumentError("indices and observations must be Vector{Int}"))
    fs[:weights] isa Vector{Float64} || throw(ArgumentError("weights must be Vector{Float64}"))
    all(x -> 1 <= x <= J, fs[:home_team]) && all(x -> 1 <= x <= J, fs[:away_team]) || throw(ArgumentError("team index out of range"))
    all(x -> 1 <= x <= 12, fs[:month_ids]) || throw(ArgumentError("month index out of range"))
    all(x -> 1 <= x <= 2, fs[:league_ids]) || throw(ArgumentError("league index out of range"))
    all(>=(0), fs[:Y_home]) && all(>=(0), fs[:Y_away]) && all(>=(0), fs[:A_home]) && all(>=(0), fs[:A_away]) && all(>=(0), fs[:C_home]) && all(>=(0), fs[:C_away]) && all(>=(0), fs[:O_home]) && all(>=(0), fs[:O_away]) || throw(ArgumentError("component observations must be non-negative"))
    all(fs[:C_home] .<= fs[:A_home]) && all(fs[:C_away] .<= fs[:A_away]) || throw(ArgumentError("converted penalties must satisfy C <= A"))
    all(isfinite, fs[:weights]) && all(w -> 0 < w <= 1, fs[:weights]) || throw(ArgumentError("weights must be finite in (0, 1]"))
    return fs
end

primitive_parameter_length(J::Integer) = 2 * Int(J) + 22
function flatten_primitives(p)
    (p.zA..., p.zD..., p.kappa_A, p.kappa_D, p.mu_Y, p.Delta, p.zM...,
     p.xi_M, p.b_Y, p.pen_base, p.pen_home, p.q_pen, p.lambda_og)
end
function unflatten_primitives(x::AbstractVector, J::Integer)
    n = primitive_parameter_length(J); length(x) == n || throw(DimensionMismatch("expected $n primitive values"))
    j = Int(J); k = 1
    zA = collect(view(x, k:k+j-1)); k += j; zD = collect(view(x, k:k+j-1)); k += j
    kappa_A, kappa_D, mu_Y, Delta = x[k], x[k+1], x[k+2], x[k+3]; k += 4
    zM = collect(view(x, k:k+11)); k += 12
    return (zA=zA, zD=zD, kappa_A=kappa_A, kappa_D=kappa_D, mu_Y=mu_Y, Delta=Delta,
        zM=zM, xi_M=x[k], b_Y=x[k+1], pen_base=x[k+2], pen_home=x[k+3], q_pen=x[k+4], lambda_og=x[k+5])
end

"""Generic, non-mutating deterministic parameter transforms.  Inputs are assumed validated."""
function transformed_parameters(p)
    tau_A, tau_D, sigma_M = exp(p.kappa_A), exp(p.kappa_D), exp(p.xi_M)
    alpha = tau_A .* (p.zA .- mean(p.zA))
    beta = tau_D .* (p.zD .- mean(p.zD)) # positive beta is defensive vulnerability
    league = (p.Delta / 2, -p.Delta / 2)
    month = sigma_M .* (p.zM .- mean(p.zM))
    return (tau_A=tau_A, tau_D=tau_D, alpha=alpha, beta=beta, L=league,
        sigma_M=sigma_M, M=month)
end

"""Extract the concrete, array-only data contract used by differentiable equations and Turing.

This is the sole `FeatureSet`/dictionary boundary. Call it before AD tracing; the
hot methods below accept this typed `NamedTuple` and never look up `fs[:key]`.
"""
function equation_data(fs)
    validate_feature_set(fs)
    data = (home_team=fs[:home_team], away_team=fs[:away_team],
        Y_home=fs[:Y_home], Y_away=fs[:Y_away], A_home=fs[:A_home], A_away=fs[:A_away],
        C_home=fs[:C_home], C_away=fs[:C_away], O_home=fs[:O_home], O_away=fs[:O_away],
        month_ids=fs[:month_ids], league_ids=fs[:league_ids], weights=fs[:weights],
        n_teams=Int(fs[:n_teams]))
    all(v isa Vector{Int} for v in values(data)[1:12]) || throw(ArgumentError("equation indices/observations must be concrete Vector{Int}"))
    data.weights isa Vector{Float64} || throw(ArgumentError("equation weights must be concrete Vector{Float64}"))
    return data
end

"""Vectorized training component rates; `data` is a concrete `equation_data` NamedTuple."""
function component_rates(data::NamedTuple, p)
    t = transformed_parameters(p)
    home, away = data.home_team, data.away_team
    common = p.mu_Y .+ getindex.(Ref(t.L), data.league_ids) .+ view(t.M, data.month_ids)
    eta_home = common .+ p.b_Y .+ view(t.alpha, home) .+ view(t.beta, away)
    eta_away = common .+ view(t.alpha, away) .+ view(t.beta, home)
    lambda_Y_home = exp.(_saturate_log_rate.(eta_home)) .+ _RATE_FLOOR
    lambda_Y_away = exp.(_saturate_log_rate.(eta_away)) .+ _RATE_FLOOR
    lambda_pen_home = exp(_saturate_log_rate(p.pen_base + p.pen_home)) + _RATE_FLOOR
    lambda_pen_away = exp(_saturate_log_rate(p.pen_base)) + _RATE_FLOOR
    return (transforms=t, eta_Y_home=eta_home, eta_Y_away=eta_away,
        lambda_Y_home=lambda_Y_home, lambda_Y_away=lambda_Y_away,
        lambda_pen_home=lambda_pen_home, lambda_pen_away=lambda_pen_away,
        q_pen=p.q_pen, lambda_og=p.lambda_og)
end
component_rates(fs, p) = component_rates(equation_data(fs), p)

_poisson_ll(y, lambda) = y .* log.(lambda) .- lambda .- loggamma.(y .+ 1)
_binomial_ll(c, a, q) = loggamma.(a .+ 1) .- loggamma.(c .+ 1) .- loggamma.(a .- c .+ 1) .+ c .* log(q) .+ (a .- c) .* log1p(-q)

"""Complete weighted data likelihood only; `data` is `equation_data(fs)` in AD hot paths."""
function weighted_data_loglikelihood(data::NamedTuple, p)
    r = component_rates(data, p); w = data.weights
    side_home = _poisson_ll(data.Y_home, r.lambda_Y_home) .+ _poisson_ll(data.A_home, r.lambda_pen_home) .+ _binomial_ll(data.C_home, data.A_home, r.q_pen) .+ _poisson_ll(data.O_home, r.lambda_og)
    side_away = _poisson_ll(data.Y_away, r.lambda_Y_away) .+ _poisson_ll(data.A_away, r.lambda_pen_away) .+ _binomial_ll(data.C_away, data.A_away, r.q_pen) .+ _poisson_ll(data.O_away, r.lambda_og)
    return sum(w .* (side_home .+ side_away))
end
weighted_data_loglikelihood(fs, p) = weighted_data_loglikelihood(equation_data(fs), p)

"""Float64 scalar reference only; it intentionally uses loops and is not an AD/model hot path."""
function weighted_data_loglikelihood_scalar(fs, p)
    r = component_rates(fs, p); total = 0.0
    for i in eachindex(fs[:weights])
        pois(y, λ) = y * log(λ) - λ - loggamma(y + 1)
        binom(c, a, q) = loggamma(a + 1) - loggamma(c + 1) - loggamma(a - c + 1) + c * log(q) + (a - c) * log1p(-q)
        total += fs[:weights][i] * (pois(fs[:Y_home][i], r.lambda_Y_home[i]) + pois(fs[:A_home][i], r.lambda_pen_home) + binom(fs[:C_home][i], fs[:A_home][i], r.q_pen) + pois(fs[:O_home][i], r.lambda_og) + pois(fs[:Y_away][i], r.lambda_Y_away[i]) + pois(fs[:A_away][i], r.lambda_pen_away) + binom(fs[:C_away][i], fs[:A_away][i], r.q_pen) + pois(fs[:O_away][i], r.lambda_og))
    end
    total
end

"""Pure per-fixture predictive component rates (no final-score convolution). Team vectors may include population-fallback zeros."""
function predictive_component_rates(p, league_ids, month_ids, alpha_home, beta_home, alpha_away, beta_away)
    t = transformed_parameters(p)
    common = p.mu_Y .+ getindex.(Ref(t.L), league_ids) .+ view(t.M, month_ids)
    eta_home = common .+ p.b_Y .+ alpha_home .+ beta_away
    eta_away = common .+ alpha_away .+ beta_home
    lambda_Y_home = exp.(_saturate_log_rate.(eta_home)) .+ _RATE_FLOOR
    lambda_Y_away = exp.(_saturate_log_rate.(eta_away)) .+ _RATE_FLOOR
    lambda_pen_home = exp(_saturate_log_rate(p.pen_base + p.pen_home)) + _RATE_FLOOR
    lambda_pen_away = exp(_saturate_log_rate(p.pen_base)) + _RATE_FLOOR
    return (lambda_Y_home=lambda_Y_home, lambda_Y_away=lambda_Y_away,
        lambda_converted_pen_home=p.q_pen * lambda_pen_home,
        lambda_converted_pen_away=p.q_pen * lambda_pen_away, lambda_og=p.lambda_og)
end

end # module
