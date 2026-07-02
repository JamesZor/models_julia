#=
current_development/score_matrix_calibration/l01_score_matrix_calibration.jl

Loader file for the Score Matrix Calibrator.
Implements the Exponential Tilt / MaxEnt shift on the 144xS score matrix
based on fitted logit bias parameters.

Also implements functions to fit the bias on all data and via a time-decay walk-forward process.
=#

using DataFrames
using GLM
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

"""
    fit_global_bias(df_market::DataFrame)
Fits a simple intercept-only logistic regression on the provided market data.
Returns the fitted gamma.
Data should contain `is_winner` (1.0/0.0) and `prob_model` (the raw model mean probability).
"""
function fit_global_bias(df_market::DataFrame)
    df_fit = dropmissing(df_market, [:is_winner, :prob_model])
    nrow(df_fit) == 0 && return 0.0
    
    eps = 1e-6
    df_fit.actual = Float64.(df_fit.is_winner)
    df_fit.logit_prob = logit.(clamp.(Float64.(df_fit.prob_model), eps, 1.0 - eps))
    
    form = @formula(actual ~ 1)
    glm_model = glm(form, df_fit, Binomial(), LogitLink(), offset=df_fit.logit_prob)
    
    return coef(glm_model)[1]
end

"""
    fit_walk_forward_bias(df_market::DataFrame; half_life_days::Float64 = 60.0)
Calculates the walking forward bias gamma for each match using ONLY past data.
Observations are weighted with an exponential time decay.
Returns a Dict mapping `match_id` -> gamma.
"""
function fit_walk_forward_bias(df_market::DataFrame; half_life_days::Float64 = 90.0)
    df_fit = dropmissing(df_market, [:is_winner, :prob_model, :match_date])
    nrow(df_fit) == 0 && return Dict{String, Float64}()
    
    eps = 1e-6
    df_fit.actual = Float64.(df_fit.is_winner)
    df_fit.logit_prob = logit.(clamp.(Float64.(df_fit.prob_model), eps, 1.0 - eps))
    
    # Sort by date
    sort!(df_fit, :match_date)
    
    match_gammas = Dict{String, Float64}()
    
    # Start predicting only when we have at least N past matches
    min_history = 20
    
    decay_rate = log(2) / half_life_days
    
    for i in 1:nrow(df_fit)
        target_match = df_fit[i, :]
        
        if i <= min_history
            match_gammas[target_match.match_id] = 0.0
            continue
        end
        
        # All data strictly before the current date
        past_data = df_fit[1:(i-1), :]
        past_data = past_data[past_data.match_date .< target_match.match_date, :]
        
        if nrow(past_data) < min_history
            match_gammas[target_match.match_id] = 0.0
            continue
        end
        
        # Calculate time weights
        days_diff = (target_match.match_date .- past_data.match_date) ./ Dates.Day(1)
        wts = exp.(-decay_rate .* days_diff)
        
        form = @formula(actual ~ 1)
        
        try
            glm_model = glm(form, past_data, Binomial(), LogitLink(), wts=wts, offset=past_data.logit_prob)
            match_gammas[target_match.match_id] = coef(glm_model)[1]
        catch e
            match_gammas[target_match.match_id] = 0.0
        end
    end
    
    return match_gammas
end
