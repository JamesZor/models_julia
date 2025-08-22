# Type definitions for prediction system
"""
    ParameterDistribution

Stores the posterior distribution summary for a single parameter.
"""
struct ParameterDistribution
    mean::Float64
    median::Float64
    quantiles::Vector{Float64}  # Values at specified quantile levels
    quantile_levels::Vector{Float64}  # The quantile levels (e.g., [0.1, 0.2, ..., 0.9])
    samples::Union{Vector{Float64}, Nothing}  # Optional: keep raw samples for later use
end

"""
    TeamParameters

Stores all parameter distributions for a single team.
"""
struct TeamParameters
    team_name::String
    attack_ht::ParameterDistribution
    defense_ht::ParameterDistribution
    attack_ft::ParameterDistribution
    defense_ft::ParameterDistribution
end

"""
    ModelParameters

Complete parameter set from a trained model.
"""
struct ModelParameters
    teams::Dict{String, TeamParameters}
    home_advantage_ht::ParameterDistribution
    home_advantage_ft::ParameterDistribution
    team_order::Vector{String}  # Preserve original ordering
end

"""
    PredictionConfig

Configuration for prediction generation.
"""
struct PredictionConfig
    quantile_levels::Vector{Float64}  # Which quantiles to compute
    keep_samples::Bool  # Whether to store raw samples
    n_simulations::Int  # For probabilistic predictions
    max_goals::Int  # Maximum goals to consider in probability calculations
    
    function PredictionConfig(;
        quantile_levels = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
        keep_samples = false,
        n_simulations = 1000,
        max_goals = 10
    )
        new(quantile_levels, keep_samples, n_simulations, max_goals)
    end
end

"""
    MatchPrediction

Prediction for a single match including uncertainty.
"""
struct MatchPrediction
    match_id::Union{Int, String}
    home_team::String
    away_team::String
    
    # Expected goals with uncertainty
    xG_home_ht::ParameterDistribution
    xG_away_ht::ParameterDistribution
    xG_home_ft::ParameterDistribution
    xG_away_ft::ParameterDistribution
    
    # Outcome probabilities
    prob_home_win::Float64
    prob_draw::Float64
    prob_away_win::Float64
    
    # Score probabilities matrix
    score_matrix::Union{Matrix{Float64}, Nothing}
end


##################################################
# Functions for extracting parameters with full distributions
##################################################
#
"""
    extract_parameter_distribution(samples, quantile_levels, keep_samples)

Extract distribution summary from parameter samples.
"""
function extract_parameter_distribution(
    samples::Vector{Float64},
    quantile_levels::Vector{Float64},
    keep_samples::Bool = false
)::ParameterDistribution
    
    quantiles = [quantile(samples, q) for q in quantile_levels]
    
    return ParameterDistribution(
        mean(samples),
        median(samples),
        quantiles,
        quantile_levels,
        keep_samples ? samples : nothing
    )
end

"""
    extract_model_parameters(chains, mapping, config)

Extract all model parameters with distributions.
"""
function extract_model_parameters(
    chains::TrainedChains,
    mapping::MappedData,
    config::PredictionConfig
)::ModelParameters
    
    teams_in_order = sort(collect(keys(mapping.team)), by = team -> mapping.team[team])
    n_teams = length(teams_in_order)
    
    function extract_all_from_chain(chain)
        # Extract raw samples
        n_samples = size(chain, 1)
        log_α_raw_samples = zeros(n_samples, n_teams)
        log_β_raw_samples = zeros(n_samples, n_teams)
        
        for i in 1:n_teams
            log_α_raw_samples[:, i] = vec(chain[Symbol("log_α_raw[$i]")])
            log_β_raw_samples[:, i] = vec(chain[Symbol("log_β_raw[$i]")])
        end
        
        # Apply centering
        log_α_centered = log_α_raw_samples .- mean(log_α_raw_samples, dims=2)
        log_β_centered = log_β_raw_samples .- mean(log_β_raw_samples, dims=2)
        
        # Transform to original scale
        α_samples = exp.(log_α_centered)
        β_samples = exp.(log_β_centered)
        
        # Home advantage
        log_γ_samples = vec(chain[:log_γ])
        γ_samples = exp.(log_γ_samples)
        
        return α_samples, β_samples, γ_samples
    end
    
    # Extract samples for HT and FT
    α_ht_samples, β_ht_samples, γ_ht_samples = extract_all_from_chain(chains.ht)
    α_ft_samples, β_ft_samples, γ_ft_samples = extract_all_from_chain(chains.ft)
    
    # Create team parameters
    team_params = Dict{String, TeamParameters}()
    
    for (i, team_name) in enumerate(teams_in_order)
        team_params[team_name] = TeamParameters(
            team_name,
            extract_parameter_distribution(α_ht_samples[:, i], config.quantile_levels, config.keep_samples),
            extract_parameter_distribution(β_ht_samples[:, i], config.quantile_levels, config.keep_samples),
            extract_parameter_distribution(α_ft_samples[:, i], config.quantile_levels, config.keep_samples),
            extract_parameter_distribution(β_ft_samples[:, i], config.quantile_levels, config.keep_samples)
        )
    end
    
    # Home advantage parameters
    home_adv_ht = extract_parameter_distribution(γ_ht_samples, config.quantile_levels, config.keep_samples)
    home_adv_ft = extract_parameter_distribution(γ_ft_samples, config.quantile_levels, config.keep_samples)
    
    return ModelParameters(team_params, home_adv_ht, home_adv_ft, teams_in_order)
end

"""
    get_parameter_at_quantile(dist, q)

Get parameter value at specific quantile level.
"""
function get_parameter_at_quantile(dist::ParameterDistribution, q::Float64)
    idx = findfirst(==(q), dist.quantile_levels)
    if isnothing(idx)
        error("Quantile level $q not found in distribution")
    end
    return dist.quantiles[idx]
end



##################################################
# Predictions
##################################################
# Prediction functions for Maher model

"""
    predict_xG_distribution(home_params, away_params, home_adv, config)

Calculate xG distribution for a match.
"""
function predict_xG_distribution(
    home_attack::ParameterDistribution,
    away_defense::ParameterDistribution,
    away_attack::ParameterDistribution,
    home_defense::ParameterDistribution,
    home_advantage::ParameterDistribution,
    config::PredictionConfig
)
    
    # Calculate xG at different quantiles
    xG_home_quantiles = Float64[]
    xG_away_quantiles = Float64[]
    
    for q in config.quantile_levels
        # Home xG = home_attack * away_defense * home_advantage
        h_att = get_parameter_at_quantile(home_attack, q)
        a_def = get_parameter_at_quantile(away_defense, q)
        h_adv = get_parameter_at_quantile(home_advantage, q)
        push!(xG_home_quantiles, h_att * a_def * h_adv)
        
        # Away xG = away_attack * home_defense
        a_att = get_parameter_at_quantile(away_attack, q)
        h_def = get_parameter_at_quantile(home_defense, q)
        push!(xG_away_quantiles, a_att * h_def)
    end
    
    # Mean xG using mean parameters
    xG_home_mean = home_attack.mean * away_defense.mean * home_advantage.mean
    xG_away_mean = away_attack.mean * home_defense.mean
    
    # Median xG
    xG_home_median = home_attack.median * away_defense.median * home_advantage.median
    xG_away_median = away_attack.median * home_defense.median
    
    return (
        home = ParameterDistribution(
            xG_home_mean,
            xG_home_median,
            xG_home_quantiles,
            config.quantile_levels,
            nothing
        ),
        away = ParameterDistribution(
            xG_away_mean,
            xG_away_median,
            xG_away_quantiles,
            config.quantile_levels,
            nothing
        )
    )
end

"""
    predict_match_probabilistic(home_team, away_team, model_params, config)

Generate probabilistic predictions for a match.
"""
function predict_match_probabilistic(
    home_team::String,
    away_team::String,
    model_params::ModelParameters,
    config::PredictionConfig
)::Union{MatchPrediction, Nothing}
    
    # Check teams exist
    if !haskey(model_params.teams, home_team) || !haskey(model_params.teams, away_team)
        return nothing
    end
    
    home_params = model_params.teams[home_team]
    away_params = model_params.teams[away_team]
    
    # Calculate xG distributions for HT
    xG_ht = predict_xG_distribution(
        home_params.attack_ht,
        away_params.defense_ht,
        away_params.attack_ht,
        home_params.defense_ht,
        model_params.home_advantage_ht,
        config
    )
    
    # Calculate xG distributions for FT
    xG_ft = predict_xG_distribution(
        home_params.attack_ft,
        away_params.defense_ft,
        away_params.attack_ft,
        home_params.defense_ft,
        model_params.home_advantage_ft,
        config
    )
    
    # Calculate outcome probabilities using median xG
    probs = calculate_match_probabilities(xG_ft.home.median, xG_ft.away.median, config.max_goals)
    
    return MatchPrediction(
        "",  # match_id to be filled later
        home_team,
        away_team,
        xG_ht.home,
        xG_ht.away,
        xG_ft.home,
        xG_ft.away,
        probs.home_win,
        probs.draw,
        probs.away_win,
        probs.score_matrix
    )
end

"""
    calculate_match_probabilities(xG_home, xG_away, max_goals)

Calculate match outcome probabilities from expected goals.
"""
function calculate_match_probabilities(xG_home::Float64, xG_away::Float64, max_goals::Int = 10)
    # Calculate score probabilities
    probs = zeros(max_goals+1, max_goals+1)
    for h in 0:max_goals
        for a in 0:max_goals
            probs[h+1, a+1] = pdf(Poisson(xG_home), h) * pdf(Poisson(xG_away), a)
        end
    end
    
    # Match outcome probabilities
    home_win = sum(probs[h, a] for h in 2:(max_goals+1) for a in 1:(h-1))
    draw = sum(probs[i, i] for i in 1:(max_goals+1))
    away_win = sum(probs[h, a] for h in 1:max_goals for a in (h+1):(max_goals+1))
    
    return (
        home_win = home_win,
        draw = draw,
        away_win = away_win,
        score_matrix = probs
    )
end

"""
    simulate_match_outcomes(home_params, away_params, home_adv, n_sims)

Simulate match outcomes using parameter uncertainty.
"""
function simulate_match_outcomes(
    home_params::TeamParameters,
    away_params::TeamParameters,
    home_adv::ParameterDistribution,
    n_sims::Int = 1000
)
    
    # If we have raw samples, use them; otherwise use quantiles
    home_wins = 0
    draws = 0
    away_wins = 0
    
    for _ in 1:n_sims
        # Sample from parameter distributions
        # This is simplified - ideally we'd sample from the actual posterior
        # For now, randomly pick a quantile level
        q_idx = rand(1:length(home_adv.quantile_levels))
        q = home_adv.quantile_levels[q_idx]
        
        h_att = home_params.attack_ft.quantiles[q_idx]
        a_def = away_params.defense_ft.quantiles[q_idx]
        h_adv = home_adv.quantiles[q_idx]
        a_att = away_params.attack_ft.quantiles[q_idx]
        h_def = home_params.defense_ft.quantiles[q_idx]
        
        λ_home = h_att * a_def * h_adv
        λ_away = a_att * h_def
        
        # Simulate goals
        home_goals = rand(Poisson(λ_home))
        away_goals = rand(Poisson(λ_away))
        
        if home_goals > away_goals
            home_wins += 1
        elseif home_goals == away_goals
            draws += 1
        else
            away_wins += 1
        end
    end
    
    return (
        home_win = home_wins / n_sims,
        draw = draws / n_sims,
        away_win = away_wins / n_sims
    )
end



# High-level workflows for prediction

"""
    predict_round_with_uncertainty(split_chains, next_round_matches, mapping, config)

Predict all matches in the next round with uncertainty quantification.
"""
function predict_round_with_uncertainty(
    split_chains::TrainedChains,
    next_round_matches::DataFrame,
    mapping::MappedData,
    config::PredictionConfig
)::DataFrame
    
    # Extract model parameters with distributions
    model_params = extract_model_parameters(split_chains, mapping, config)
    
    predictions = DataFrame()
    
    for row in eachrow(next_round_matches)
        pred = predict_match_probabilistic(
            row.home_team,
            row.away_team,
            model_params,
            config
        )
        
        if !isnothing(pred)
            # Create row with prediction intervals
            push!(predictions, (
                match_id = row.match_id,
                home_team = row.home_team,
                away_team = row.away_team,
                round = row.round,
                # Actual results
                actual_home_ht = row.home_score_ht,
                actual_away_ht = row.away_score_ht,
                actual_home_ft = row.home_score,
                actual_away_ft = row.away_score,
                # Point predictions (median)
                pred_xG_home_ht = pred.xG_home_ht.median,
                pred_xG_away_ht = pred.xG_away_ht.median,
                pred_xG_home_ft = pred.xG_home_ft.median,
                pred_xG_away_ft = pred.xG_away_ft.median,
                # Prediction intervals (95% CI)
                xG_home_ft_lower = pred.xG_home_ft.quantiles[1],  # 2.5%
                xG_home_ft_upper = pred.xG_home_ft.quantiles[end],  # 97.5%
                xG_away_ft_lower = pred.xG_away_ft.quantiles[1],
                xG_away_ft_upper = pred.xG_away_ft.quantiles[end],
                # Outcome probabilities
                prob_home_win = pred.prob_home_win,
                prob_draw = pred.prob_draw,
                prob_away_win = pred.prob_away_win
            ))
        end
    end
    
    return predictions
end

"""
    create_prediction_summary(predictions_df)

Create a summary of predictions with uncertainty metrics.
"""
function create_prediction_summary(predictions_df::DataFrame)
    # Calculate coverage of prediction intervals
    in_interval_home = sum(
        (predictions_df.actual_home_ft .>= predictions_df.xG_home_ft_lower) .&
        (predictions_df.actual_home_ft .<= predictions_df.xG_home_ft_upper)
    )
    in_interval_away = sum(
        (predictions_df.actual_away_ft .>= predictions_df.xG_away_ft_lower) .&
        (predictions_df.actual_away_ft .<= predictions_df.xG_away_ft_upper)
    )
    
    n_matches = nrow(predictions_df)
    
    # Width of prediction intervals
    interval_width_home = mean(predictions_df.xG_home_ft_upper - predictions_df.xG_home_ft_lower)
    interval_width_away = mean(predictions_df.xG_away_ft_upper - predictions_df.xG_away_ft_lower)
    
    # Probabilistic scoring (Brier score for outcome predictions)
    actual_outcomes = String[]
    for row in eachrow(predictions_df)
        if row.actual_home_ft > row.actual_away_ft
            push!(actual_outcomes, "H")
        elseif row.actual_home_ft == row.actual_away_ft
            push!(actual_outcomes, "D")
        else
            push!(actual_outcomes, "A")
        end
    end
    
    brier_score = 0.0
    for (i, outcome) in enumerate(actual_outcomes)
        p_h = predictions_df.prob_home_win[i]
        p_d = predictions_df.prob_draw[i]
        p_a = predictions_df.prob_away_win[i]
        
        if outcome == "H"
            brier_score += (1 - p_h)^2 + p_d^2 + p_a^2
        elseif outcome == "D"
            brier_score += p_h^2 + (1 - p_d)^2 + p_a^2
        else  # Away win
            brier_score += p_h^2 + p_d^2 + (1 - p_a)^2
        end
    end
    brier_score /= n_matches
    
    return (
        n_matches = n_matches,
        interval_coverage = (
            home = in_interval_home / n_matches,
            away = in_interval_away / n_matches
        ),
        interval_width = (
            home = interval_width_home,
            away = interval_width_away
        ),
        brier_score = brier_score,
        mae = (
            home_ft = mean(abs.(predictions_df.actual_home_ft - predictions_df.pred_xG_home_ft)),
            away_ft = mean(abs.(predictions_df.actual_away_ft - predictions_df.pred_xG_away_ft))
        )
    )
end

"""
    analyze_parameter_evolution(experiment_result, team_name, config)

Track how a team's parameter distributions evolve across CV splits.
"""
function analyze_parameter_evolution(
    experiment_result::ExperimentResult,
    team_name::String,
    config::PredictionConfig
)::DataFrame
    
    evolution = DataFrame()
    
    for (i, chains) in enumerate(experiment_result.chains_sequence)
        model_params = extract_model_parameters(chains, experiment_result.mapping, config)
        
        if haskey(model_params.teams, team_name)
            team_params = model_params.teams[team_name]
            
            push!(evolution, (
                split = i,
                round_info = chains.round_info,
                # Median values
                attack_ft_median = team_params.attack_ft.median,
                defense_ft_median = team_params.defense_ft.median,
                # Uncertainty (IQR)
                attack_ft_iqr = team_params.attack_ft.quantiles[5] - team_params.attack_ft.quantiles[3],  # 75% - 25%
                defense_ft_iqr = team_params.defense_ft.quantiles[5] - team_params.defense_ft.quantiles[3],
                # 95% CI width
                attack_ft_ci_width = team_params.attack_ft.quantiles[end] - team_params.attack_ft.quantiles[1],
                defense_ft_ci_width = team_params.defense_ft.quantiles[end] - team_params.defense_ft.quantiles[1]
            ))
        end
    end
    
    return evolution
end
