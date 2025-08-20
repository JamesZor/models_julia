# Match prediction functions
"""
    extract_team_parameters(chains, mapping)

Extract team parameters (attack, defense, home advantage) from MCMC chains.
"""
function extract_team_parameters(chains::TrainedChains, mapping::MappedData)
    # Get teams in training order
    teams_in_order = sort(collect(keys(mapping.team)), by = team -> mapping.team[team])
    n_teams = length(teams_in_order)
    
    function extract_from_chain(chain)
        # Extract raw samples
        log_α_raw_samples = zeros(size(chain, 1), n_teams)
        log_β_raw_samples = zeros(size(chain, 1), n_teams)
        
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
        
        # Get posterior medians
        α_median = [median(α_samples[:, i]) for i in 1:n_teams]
        β_median = [median(β_samples[:, i]) for i in 1:n_teams]
        
        # Home advantage
        log_γ_samples = vec(chain[:log_γ])
        γ_median = median(exp.(log_γ_samples))
        
        return α_median, β_median, γ_median
    end
    
    # Extract for HT and FT
    α_ht, β_ht, γ_ht = extract_from_chain(chains.ht)
    α_ft, β_ft, γ_ft = extract_from_chain(chains.ft)
    
    # Create parameter DataFrame
    return DataFrame(
        team = teams_in_order,
        attack_ht = α_ht,
        defense_ht = β_ht,
        attack_ft = α_ft,
        defense_ft = β_ft,
        home_adv_ht = fill(γ_ht, n_teams),  # Same for all teams
        home_adv_ft = fill(γ_ft, n_teams)
    )
end

"""
    predict_match_xG(home_team, away_team, params_df)

Predict expected goals for a single match.
"""
function predict_match_xG(home_team::AbstractString, away_team::AbstractString, params_df::DataFrame)
    # Get team parameters
    home_params = filter(row -> row.team == home_team, params_df)
    away_params = filter(row -> row.team == away_team, params_df)
    
    if nrow(home_params) == 0 || nrow(away_params) == 0
        return nothing  # Team not in training data
    end
    
    home_params = home_params[1, :]
    away_params = away_params[1, :]
    
    # Calculate expected goals
    xG_home_ht = home_params.attack_ht * away_params.defense_ht * home_params.home_adv_ht
    xG_away_ht = away_params.attack_ht * home_params.defense_ht
    
    xG_home_ft = home_params.attack_ft * away_params.defense_ft * home_params.home_adv_ft
    xG_away_ft = away_params.attack_ft * home_params.defense_ft
    
    return (
        xG_home_ht = xG_home_ht,
        xG_away_ht = xG_away_ht,
        xG_home_ft = xG_home_ft,
        xG_away_ft = xG_away_ft
    )
end

"""
    predict_round_matches(split_chains, next_round_matches, mapping)

Predict all matches in the next round using the current split's trained model.
"""
function predict_round_matches(
    split_chains::TrainedChains,
    next_round_matches::DataFrame,
    mapping::MappedData
)
    # Extract parameters from this split's chains
    params = extract_team_parameters(split_chains, mapping)
    
    predictions = DataFrame()
    
    for row in eachrow(next_round_matches)
        pred = predict_match_xG(row.home_team, row.away_team, params)
        
        if !isnothing(pred)
            push!(predictions, (
                match_id = row.match_id,
                home_team = row.home_team,
                away_team = row.away_team,
                round = row.round,
                actual_home_ht = row.home_score_ht,
                actual_away_ht = row.away_score_ht,
                actual_home_ft = row.home_score,
                actual_away_ft = row.away_score,
                pred_xG_home_ht = pred.xG_home_ht,
                pred_xG_away_ht = pred.xG_away_ht,
                pred_xG_home_ft = pred.xG_home_ft,
                pred_xG_away_ft = pred.xG_away_ft
            ))
        end
    end
    
    return predictions
end

"""
    predict_all_rounds(experiment_result, data_store, cv_config)

Generate predictions for all rounds using the expanding window CV approach.
"""
function predict_all_rounds(
    experiment_result::ExperimentResult,
    data_store::DataStore,
    cv_config::TimeSeriesSplitsConfig
)
    all_predictions = DataFrame()
    
    # Get target season data
    target_matches = filter(row -> row.season == cv_config.target_season, data_store.matches)
    rounds = sort(unique(target_matches.round))
    
    # For each split, predict the corresponding round
    for (i, split_chains) in enumerate(experiment_result.chains_sequence)
        if i == 1
            # First split (base only) predicts round 1
            next_round = rounds[1]
        elseif i <= length(rounds)
            # Subsequent splits predict the next round
            next_round = rounds[i]
        else
            break  # No more rounds to predict
        end
        
        # Get matches for this round
        round_matches = filter(row -> row.round == next_round, target_matches)
        
        if nrow(round_matches) > 0
            round_preds = predict_round_matches(
                split_chains,
                round_matches,
                experiment_result.mapping
            )
            
            if nrow(round_preds) > 0
                round_preds[!, :split_idx] .= i
                round_preds[!, :split_info] .= split_chains.round_info
                append!(all_predictions, round_preds)
            end
        end
    end
    
    return all_predictions
end

"""
    calculate_match_probabilities(xG_home, xG_away, max_goals=10)

Calculate match outcome probabilities from expected goals.
"""
function calculate_match_probabilities(xG_home::Float64, xG_away::Float64; max_goals::Int=10)
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
