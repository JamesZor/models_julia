#=
current_development/score_matrix_calibration/l01_score_matrix_calibration.jl

Loader file for the Score Matrix Calibrator.
Implements the Exponential Tilt / MaxEnt shift on the 144xS score matrix
based on fitted logit bias parameters.

Also implements functions to fit the bias on all data and via a time-decay walk-forward process.
=#

using DataFrames
using LogExpFunctions # for logit / logistic
using Statistics
using Dates
using BayesianFootball

const GG = 12
const HGRID = vec([h for h in 0:GG-1, a in 0:GG-1])
const AGRID = vec([a for h in 0:GG-1, a in 0:GG-1])

"""
    mask_for(mname, mline, sel)
Binary indicator over the 144 grid states for a market.
"""
function mask_for(mname, mline, sel)
    s = String(sel)
    if mname == "1X2"
        s == "home" && return HGRID .> AGRID
        s == "draw" && return HGRID .== AGRID
        return HGRID .< AGRID
    elseif mname == "OverUnder"
        return startswith(s, "over") ? (HGRID .+ AGRID .> mline) : (HGRID .+ AGRID .< mline)
    elseif mname == "BTTS"
        yes = (HGRID .>= 1) .& (AGRID .>= 1)
        return s == "btts_yes" ? yes : .!yes
    end
    error("unknown market $mname")
end

"""
    tilt_score_matrix!(P::Matrix{Float64}, masks::Vector{BitVector}, gammas::Vector{Float64})
Applies an exponential tilt to the 144xS score matrix `P` and normalizes the columns.
P_calib(ω) ∝ P_raw(ω) * exp(Σ γ_m I_m(ω))
"""
function tilt_score_matrix!(P::Matrix{Float64}, masks::Vector{BitVector}, gammas::Vector{Float64})
    S = size(P, 2)
    # Build the multiplier vector for all 144 states
    multiplier = ones(144)
    for (i, mask) in enumerate(masks)
        if gammas[i] != 0.0
            multiplier[mask] .*= exp(gammas[i])
        end
    end
    
    # Apply to the matrix and renormalize
    for s in 1:S
        P[:, s] .*= multiplier
        P[:, s] ./= sum(P[:, s])
    end
    return P
end

# ------------------------------------------------------------------------------
# INTERCEPT-ONLY LOGISTIC SHIFT (the calibration γ)
# ------------------------------------------------------------------------------
# γ solves the (optionally weighted) intercept-only logistic MLE with the model's
# logit as a fixed offset:   Σ w_i·( logistic(γ + logit(model_p_i)) − y_i ) = 0.
# This is monotone increasing in γ, so a bisection is exact and robust.
#
# The target `y` is the reference we calibrate the per-line MEAN onto:
#   • target = actual outcomes (is_winner ∈ {0,1})  ⇒ calibrate to REALITY.
#   • target = de-vigged market prob (prob_fair_close ∈ (0,1)) ⇒ strip model−market skew.
# For a binary y this reproduces the GLM intercept-only fit exactly; for a fractional
# y (the market prob) it is the natural quasi-likelihood analogue and needs no GLM.
function _fit_shift(y::Vector{Float64}, logit_p::Vector{Float64}, w::Vector{Float64})
    f(γ) = sum(w .* (logistic.(γ .+ logit_p) .- y))   # increasing in γ; root = MLE
    lo, hi = -20.0, 20.0
    f(lo) > 0.0 && return lo                            # degenerate: mean model ≫ target
    f(hi) < 0.0 && return hi
    for _ in 1:100
        m = 0.5 * (lo + hi)
        f(m) > 0.0 ? (hi = m) : (lo = m)
        hi - lo < 1e-10 && break
    end
    return 0.5 * (lo + hi)
end

"""
    fit_global_bias(df_market::DataFrame; target::Symbol = :is_winner)
Global (pooled) calibration shift γ centering the tilted per-line mean onto `target`.
`df_market` must contain `prob_model` and the `target` column (`is_winner` for reality,
`prob_fair_close` for the market). Returns the scalar γ.
"""
function fit_global_bias(df_market::DataFrame; target::Symbol = :is_winner)
    df_fit = dropmissing(df_market, [target, :prob_model])
    nrow(df_fit) == 0 && return 0.0
    eps = 1e-6
    y = clamp.(Float64.(df_fit[!, target]), eps, 1.0 - eps)
    o = logit.(clamp.(Float64.(df_fit.prob_model), eps, 1.0 - eps))
    return _fit_shift(y, o, ones(length(y)))
end

"""
    fit_walk_forward_bias(df_market::DataFrame; half_life_days = 90.0, target = :is_winner)
Walk-forward calibration shift γ per match, fit on ONLY past data with an exponential
time decay (half-life in days). Same `target` semantics as `fit_global_bias`.
Returns a Dict mapping `match_id` -> γ (0.0 until `min_history` past matches exist).
"""
function fit_walk_forward_bias(df_market::DataFrame; half_life_days::Float64 = 90.0,
                               target::Symbol = :is_winner)
    df_fit = dropmissing(df_market, [target, :prob_model, :match_date])
    nrow(df_fit) == 0 && return Dict{eltype(df_market.match_id), Float64}()

    eps = 1e-6
    df_fit = copy(df_fit)
    sort!(df_fit, :match_date)
    y_all = clamp.(Float64.(df_fit[!, target]), eps, 1.0 - eps)
    o_all = logit.(clamp.(Float64.(df_fit.prob_model), eps, 1.0 - eps))
    dates = df_fit.match_date

    match_gammas = Dict{eltype(df_market.match_id), Float64}()
    min_history = 20
    decay_rate = log(2) / half_life_days

    for i in 1:nrow(df_fit)
        mid = df_fit.match_id[i]
        # strictly-past rows (guards same-day leakage)
        past = findall(dates[1:(i-1)] .< dates[i])
        if length(past) < min_history
            match_gammas[mid] = 0.0
            continue
        end
        days_diff = (dates[i] .- dates[past]) ./ Dates.Day(1)
        wts = exp.(-decay_rate .* days_diff)
        match_gammas[mid] = _fit_shift(y_all[past], o_all[past], wts)
    end
    return match_gammas
end
