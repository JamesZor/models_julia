# src/Calibration/shift_models/fitted_logit.jl
#
# DEPRECATED, for the same reason as `basic_logit.jl`: a per-team logit offset is still a
# SELECTION-level shift, so the shifted board is still not a scoreline distribution.

"""
    TeamBiasLogitShift()

**DEPRECATED — use [`GenerativeRateCalibrator`](@ref).**

A logit offset per team plus an intercept. Coherence across derivative markets fails
exactly as it does for [`BasicLogitShift`](@ref); the per-team resolution does not change
that.
"""
struct TeamBiasLogitShift <: AbstractLayerTwoModel
    # Can hold hyperparameters like L2 Regularization penalty later if needed
    function TeamBiasLogitShift()
        _warn_selection_level_deprecated("TeamBiasLogitShift")
        return new()
    end
end

struct FittedTeamBiasShift
    c_base::Float64
    team_betas::Dict{String, Float64} # Maps "Arsenal" -> +0.12
    prob_col::Symbol
end

function fit_calibrator(model::TeamBiasLogitShift, data::DataFrame, config::CalibrationConfig)
    dropmissing!(data, :is_winner)
    eps = 1e-6
    prob_data = data[!, config.prob_col]
    
    # 1. Setup the basic fit dataframe
    df_fit = DataFrame(
        actual = Float64.(data.is_winner),
        logit_prob = logit.(clamp.(prob_data, eps, 1.0 - eps))
    )
    
    # 2. Get all unique teams
    all_teams = unique(vcat(data.home_team, data.away_team))
    
    # 3. Drop one team to avoid the Dummy Variable Trap (Baseline team)
    baseline_team = pop!(all_teams) 
    
    # 4. Create the indicator (Delta) columns for the remaining teams
    # 1.0 if the team is home OR away, 0.0 otherwise
    for t in all_teams
        df_fit[!, Symbol("team_", t)] = Float64.((data.home_team .== t) .| (data.away_team .== t))
    end
    
    # 5. Dynamically build the GLM Formula: actual ~ 1 + team_A + team_B + ...
    # We use StatsModels.term to build this programmatically
    team_terms = [term(Symbol("team_", t)) for t in all_teams]
    rhs = sum(team_terms)
    form = term(:actual) ~ term(1) + rhs
    
    # 6. Fit the model
    glm_model = glm(form, df_fit, Binomial(), LogitLink(), offset=df_fit.logit_prob)
    
    # 7. Extract the coefficients into a clean Dictionary
    coefs = coef(glm_model)
    c_base = coefs[1] # Intercept
    
    team_betas = Dict{String, Float64}()
    # The baseline team has exactly 0.0 shift relative to the intercept
    team_betas[baseline_team] = 0.0 
    
    for (i, t) in enumerate(all_teams)
        # +1 because index 1 is the intercept
        team_betas[t] = coefs[i + 1] 
    end
    
    return FittedTeamBiasShift(c_base, team_betas, config.prob_col)
end


function apply_calibration(fitted_model::FittedTeamBiasShift, new_data::DataFrame)
    eps = 1e-6
    prob_data = new_data[!, fitted_model.prob_col]
    
    # 1. Calculate the specific shift for every single row
    # c_base + beta_home + beta_away
    row_shifts = map(eachrow(new_data)) do r
        home_beta = get(fitted_model.team_betas, r.home_team, 0.0)
        away_beta = get(fitted_model.team_betas, r.away_team, 0.0)
        return fitted_model.c_base + home_beta + away_beta
    end
    
    # 2. Apply the specific shifts to the scalars
    shifted_scalars = logistic.(logit.(clamp.(prob_data, eps, 1.0 - eps)) .+ row_shifts)
    
    # 3. Apply the specific shifts to the MCMC distributions
    shifted_dists = map(enumerate(new_data.distribution)) do (i, dist)
        logistic.(logit.(clamp.(dist, eps, 1.0 - eps)) .+ row_shifts[i])
    end
    
    return shifted_scalars, shifted_dists
end
