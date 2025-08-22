# Proper probabilistic match predictions using full posterior
"""
Extract all samples for parameters matching a base string, e.g., "log_α_raw".
"""
function extract_samples(chain::Chains, base::String)
    nms = names(chain)
    matches = filter(n -> occursin(base, String(n)), nms)
    arr = Array(chain[matches])
    reshape(arr, :, size(arr, 2))  # flatten iterations × chains in first dimension
end

"""
    extract_posterior_samples(chain, mapping)
    
Extract all posterior samples preserving correlations.
"""
function extract_posterior_samples(chain, mapping::MappedData)
    teams_in_order = sort(collect(keys(mapping.team)), by = team -> mapping.team[team])
    n_teams = length(teams_in_order)
    n_samples = size(chain, 1) * size(chain, 3)  # samples × chains
    
    # Extract raw samples
    log_α_raw = extract_samples(chain, "log_α_raw")
    log_β_raw = extract_samples(chain, "log_β_raw")
    log_γ = vec(Array(chain[:log_γ]))
    
    # Process each sample maintaining correlations
    α_samples = zeros(n_samples, n_teams)
    β_samples = zeros(n_samples, n_teams)
    γ_samples = exp.(log_γ)
    
    # Centering: subtract row means from each element in the row
    log_α_centered = log_α_raw .- mean(log_α_raw, dims=2)
    log_β_centered = log_β_raw .- mean(log_β_raw, dims=2)

    # Exponentiation
    α_samples = exp.(log_α_centered)
    β_samples = exp.(log_β_centered)

    return (
        α = α_samples,
        β = β_samples,
        γ = γ_samples,
        teams = teams_in_order
    )
end

function compute_xScore(λ_home::Number, λ_away::Number, max_goals::Int64)
  p = zeros(max_goals+1, max_goals+1)
  for h in 0:max_goals, a in 0:max_goals
      p[h+1,a+1] = pdf(Poisson(λ_home), h) * pdf(Poisson(λ_away), a)
  end 
  return p
end

function calculate_1x2(p::Matrix{Float64}) 
  hw = 0.0
  dr = 0.0
  aw = 0.0
  for h in 1:(size(p,1)), a in 1:(size(p,2))
    if h > a
      hw += p[h, a]
    elseif h == a
      dr += p[h,a]
    else
      aw += p[h, a]
    end 
  end 
  return (
    hw = hw,
    dr = dr, 
    aw = aw 
   )
end

function calculate_correct_score(p::Matrix{Float64}, max_display_goals::Int64=3) 
  correct_scores =  p[1:max_display_goals+1, 1:max_display_goals+1]
  # Initialize the three 'other' probabilities
  any_other_home_win = 0.0
  any_other_away_win = 0.0
  any_other_draw = 0.0
  max_goals = size(p,1)-1

  for h in 0:max_goals, a in 0:max_goals
      # Check if the score is not one of the explicitly listed ones (0-0 to 3-3)
      if h > max_display_goals || a > max_display_goals
          # Home Win
          if h > a
              any_other_home_win += p[h+1, a+1]
          # Away Win
          elseif a > h
              any_other_away_win += p[h+1, a+1]
          # Draw
          else # h == a
              any_other_draw += p[h+1, a+1]
          end
      end
  end
  return ( 
      correct_scores = reshape(correct_scores, 1,:),
      other_home_win = any_other_home_win, 
      other_away_win = any_other_away_win,
      other_draw = any_other_draw
     )
end

function calculate_under_prob(p::Matrix{Float64}, threshold::Int)
    prob_under = 0.0
    max_goals = size(p,1)-1

    for h in 0:max_goals, a in 0:max_goals
          # Calculate the total goals for the score h-a
          total_goals = h + a
          
          # Check if the total goals are under the threshold
          if total_goals <= threshold
              prob_under += p[h+1, a+1]
        end
    end
    return prob_under
end

module Data
  struct MatchHalfChainPredicts 
    home::AbstractVector{Float64}
    draw::AbstractVector{Float64}
    away::AbstractVector{Float64}

    correct_score::AbstractArray{Float64}
    other_home_win_score::AbstractVector{Float64} 
    other_away_win_score::AbstractVector{Float64} 
    other_draw_score::AbstractVector{Float64} 

    under_05::AbstractVector{Float64} 
    under_15::AbstractVector{Float64} 
    under_25::AbstractVector{Float64} 
    under_35::AbstractVector{Float64} 
  end


  struct MatchPredict
    ht::MatchHalfChainPredicts
    ft::MatchHalfChainPredicts
  end
end




function predict_match_chain(
    home_team::AbstractString, 
    away_team::AbstractString,
    chain::Chains,
    mapping::MappedData,
  )
  n_samples = size(chain, 1) * size(chain, 3)  # samples × chains

  posterior_samples = extract_posterior_samples(chain, mapping)

  home_idx = findfirst(==(home_team), posterior_samples.teams)
  away_idx = findfirst(==(away_team), posterior_samples.teams)

  # define lines 
  home_win_probs = zeros(n_samples)
  draw_probs     = zeros(n_samples)
  away_win_probs = zeros(n_samples)

  correct_score =  zeros(n_samples, 16)
  other_home_win_score = zeros(n_samples)
  other_away_win_score = zeros(n_samples)
  other_draw_score = zeros(n_samples)

  under_05 = zeros(n_samples)
  under_15 = zeros(n_samples)
  under_25 = zeros(n_samples)
  under_35 = zeros(n_samples)

  for i in 1:n_samples
    α_h = posterior_samples.α[i, home_idx]
    β_h = posterior_samples.β[i, home_idx]
    α_a = posterior_samples.α[i, away_idx]
    β_a = posterior_samples.β[i, away_idx]
    γ = posterior_samples.γ[i]

    # compute xG
    λ_home = α_h * β_a * γ
    λ_away = α_a * β_h

    p = compute_xScore(λ_home, λ_away, 10)
    hda = calculate_1x2(p)
    cs = calculate_correct_score(p,3)

    u_05 = calculate_under_prob(p,0)
    u_15 = calculate_under_prob(p,1)
    u_25 = calculate_under_prob(p,2)
    u_35 = calculate_under_prob(p,3)


    # assign 
    home_win_probs[i] = hda.hw 
    draw_probs[i] = hda.dr 
    away_win_probs[i] = hda.aw 

    correct_score[i, :] = cs.correct_scores 
    other_home_win_score[i] = cs.other_home_win 
    other_away_win_score[i] = cs.other_away_win 
    other_draw_score[i] = cs.other_draw 

    under_05[i] = u_05
    under_15[i] = u_15
    under_25[i] = u_25
    under_35[i] = u_35

  end

    return Data.MatchHalfChainPredicts( 
  home_win_probs,
  draw_probs,
  away_win_probs,
  correct_score,
  other_home_win_score,
  other_away_win_score,
  other_draw_score,
  under_05,
  under_15,
  under_25,
  under_35,
 )
end

function predict_match_ft_ht_chain(
    home_team::AbstractString,
    away_team::AbstractString,
    round_chains::TrainedChains,
    mapping::MappedData) 

  ht_predict = predict_match_chain(home_team,away_team, round_chains.ht, mapping)
  ft_predict = predict_match_chain(home_team,away_team, round_chains.ft, mapping)
  return Data.MatchPredict(ht_predict, ft_predict)
end










#########################################

"""
    predict_match_full_posterior(home_team, away_team, posterior_samples)
    
Generate match predictions using full posterior distribution.
"""
function predict_match_full_posterior(
    home_team::AbstractString,
    away_team::AbstractString,
    posterior_samples::NamedTuple,
    max_goals::Int = 10
)
    # Find team indices
    home_idx = findfirst(==(home_team), posterior_samples.teams)
    away_idx = findfirst(==(away_team), posterior_samples.teams)
    
    if isnothing(home_idx) || isnothing(away_idx)
        return nothing
    end
    
    n_samples = size(posterior_samples.α, 1)
    
    # Store results
    xG_home_samples = Float64[]
    xG_away_samples = Float64[]
    home_wins = 0
    draws = 0
    away_wins = 0
    
    # Score distribution accumulator
    score_dist = zeros(max_goals+1, max_goals+1)
    
    # Process each posterior sample
    for i in 1:n_samples
        # Get parameters for this iteration
        α_h = posterior_samples.α[i, home_idx]
        β_h = posterior_samples.β[i, home_idx]
        α_a = posterior_samples.α[i, away_idx]
        β_a = posterior_samples.β[i, away_idx]
        γ = posterior_samples.γ[i]
        
# Compute expected goals
        λ_home = α_h * β_a * γ
        λ_away = α_a * β_h
        
        push!(xG_home_samples, λ_home)
        push!(xG_away_samples, λ_away)
        
        # Calculate outcome probabilities for this parameter set
        for h in 0:max_goals
            for a in 0:max_goals
                p = pdf(Poisson(λ_home), h) * pdf(Poisson(λ_away), a)
                score_dist[h+1, a+1] += p / n_samples
                
                if h > a
                    home_wins += p / n_samples
                elseif h == a
                    draws += p / n_samples
                else
                    away_wins += p / n_samples
                end
            end
        end
    end
    
    # Calculate xG distribution statistics
    xG_home_quantiles = [quantile(xG_home_samples, q) for q in [0.025, 0.25, 0.5, 0.75, 0.975]]
    xG_away_quantiles = [quantile(xG_away_samples, q) for q in [0.025, 0.25, 0.5, 0.75, 0.975]]
    
    return (
        xG_home = (
            mean = mean(xG_home_samples),
            median = xG_home_quantiles[3],
            q025 = xG_home_quantiles[1],
            q975 = xG_home_quantiles[5],
            iqr = xG_home_quantiles[4] - xG_home_quantiles[2],
            samples = xG_home_samples
        ),
        xG_away = (
            mean = mean(xG_away_samples),
            median = xG_away_quantiles[3],
            q025 = xG_away_quantiles[1],
            q975 = xG_away_quantiles[5],
            iqr = xG_away_quantiles[4] - xG_away_quantiles[2],
            samples = xG_away_samples
        ),
        prob_home_win = home_wins,
        prob_draw = draws,
        prob_away_win = away_wins,
        score_distribution = score_dist
    )
end




function predict_match_ht_ft(
    home_team::AbstractString,
    away_team::AbstractString,
    chain::TrainedChains,
    mapping::MappedData,
)
  ht_predict = predict_match_full_posterior(home_team, away_team, extract_posterior_samples(chain.ht, mapping), 6)
  ft_predict = predict_match_full_posterior(home_team, away_team, extract_posterior_samples(chain.ft, mapping), 6)

  return (
    ht=ht_predict, 
    ft=ft_predict 
   )
end


function predict_round(
    split_chain::TrainedChains, 
    next_round_matches::DataFrame,
    mapping::MappedData
  ) 
  predictions = DataFrame() 

  for row in eachrow(next_round_matches)
    match_predict = predict_match_ht_ft(
                            row.home_team,
                            row.away_team,
                            split_chain,
                            mapping)
    if !isnothing(match_predict)
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
              # point predictions 
              pred_xG_home_ht_mean = match_predict.ht.xG_home.mean, 
              pred_xG_away_ht_mean = match_predict.ht.xG_away.mean, 
              pred_xG_home_ft_mean = match_predict.ft.xG_home.mean, 
              pred_xG_away_ft_mean = match_predict.ft.xG_away.mean, 

              home_ht = match_predict.ht.prob_home_win,
              away_ht = match_predict.ht.prob_away_win,
              draw_ht = match_predict.ht. prob_draw, 

              home_ft = match_predict.ft.prob_home_win,
              away_ft = match_predict.ft.prob_away_win,
              draw_ft = match_predict.ft. prob_draw, 
             )
           )
        end
      end 
    return predictions
end


"""
    calibration_analysis(predictions, actuals, n_bins=10)
    
Analyze calibration of probabilistic predictions.
"""
function calibration_analysis(predictions::DataFrame, n_bins::Int = 10)
    # For outcome probabilities
    actual_outcomes = String[]
    pred_home_probs = Float64[]
    pred_draw_probs = Float64[]
    pred_away_probs = Float64[]
    
    for row in eachrow(predictions)
        # Actual outcome
        if row.actual_home_ft > row.actual_away_ft
            push!(actual_outcomes, "H")
        elseif row.actual_home_ft == row.actual_away_ft
            push!(actual_outcomes, "D")
        else
            push!(actual_outcomes, "A")
        end
        
        push!(pred_home_probs, row.prob_home_win)
        push!(pred_draw_probs, row.prob_draw)
        push!(pred_away_probs, row.prob_away_win)
    end
    
    # Bin predictions and calculate calibration
    calibration_home = calculate_calibration(pred_home_probs, actual_outcomes .== "H", n_bins)
    calibration_draw = calculate_calibration(pred_draw_probs, actual_outcomes .== "D", n_bins)
    calibration_away = calculate_calibration(pred_away_probs, actual_outcomes .== "A", n_bins)
    
    # For xG predictions - check interval calibration
    xG_calibration = check_interval_calibration(predictions)
    
    return (
        outcome_calibration = (
            home = calibration_home,
            draw = calibration_draw,
            away = calibration_away
        ),
        xG_calibration = xG_calibration
    )
end

function calculate_calibration(predicted_probs::Vector, actual_outcomes::BitVector, n_bins::Int)
    bins = range(0, 1, length=n_bins+1)
    calibration_data = []
    
    for i in 1:(n_bins)
        mask = (predicted_probs .>= bins[i]) .& (predicted_probs .< bins[i+1])
        if sum(mask) > 0
            mean_pred = mean(predicted_probs[mask])
            actual_freq = mean(actual_outcomes[mask])
            n = sum(mask)
            
            push!(calibration_data, (
                bin_center = (bins[i] + bins[i+1]) / 2,
                mean_predicted = mean_pred,
                actual_frequency = actual_freq,
                n_samples = n,
                se = sqrt(actual_freq * (1 - actual_freq) / n)  # Standard error
            ))
        end
    end
    
    return calibration_data
end

function check_interval_calibration(predictions::DataFrame)
    # Check coverage at different confidence levels
    coverage_levels = [0.5, 0.8, 0.9, 0.95]
    results = []
    
    for level in coverage_levels
        # Calculate what quantiles we need
        lower_q = (1 - level) / 2
        upper_q = 1 - lower_q
        
        # Check coverage (this assumes you have these columns)
        if hasproperty(predictions, :xG_home_q025)
            # Interpolate to get the right quantiles
            # For now, use closest available
            if level == 0.95
                in_interval = sum(
                    (predictions.actual_home_ft .>= predictions.xG_home_q025) .&
                    (predictions.actual_home_ft .<= predictions.xG_home_q975)
                )
            else
                # Would need more quantiles stored
                continue
            end
            
            coverage = in_interval / nrow(predictions)
            push!(results, (
                confidence_level = level,
                actual_coverage = coverage,
                expected_coverage = level,
                deviation = coverage - level
            ))
        end
    end
    
    return results
end

"""
    roc_analysis_by_uncertainty(predictions)
    
Analyze prediction accuracy stratified by uncertainty level.
"""
function roc_analysis_by_uncertainty(predictions::DataFrame)
    # Calculate prediction uncertainty (width of prediction interval)
    predictions.uncertainty_home = predictions.xG_home_q975 - predictions.xG_home_q025
    predictions.uncertainty_away = predictions.xG_away_q975 - predictions.xG_away_q025
    
    # Stratify by uncertainty quantiles
    uncertainty_quantiles = [0.25, 0.5, 0.75]
    uncertainty_thresholds = quantile(predictions.uncertainty_home, uncertainty_quantiles)
    
    results = []
    
    for i in 1:length(uncertainty_thresholds)
        if i == 1
            mask = predictions.uncertainty_home .<= uncertainty_thresholds[i]
            label = "Low uncertainty"
        elseif i == length(uncertainty_thresholds)
            mask = predictions.uncertainty_home .> uncertainty_thresholds[i-1]
            label = "High uncertainty"
        else
            mask = (predictions.uncertainty_home .> uncertainty_thresholds[i-1]) .&
                   (predictions.uncertainty_home .<= uncertainty_thresholds[i])
            label = "Medium uncertainty"
        end
        
        subset = predictions[mask, :]
        
        # Calculate metrics for this subset
        mae_home = mean(abs.(subset.actual_home_ft - subset.pred_xG_home_ft))
        mae_away = mean(abs.(subset.actual_away_ft - subset.pred_xG_away_ft))
        
        # Classification accuracy
        predicted_outcomes = String[]
        actual_outcomes = String[]
        
        for row in eachrow(subset)
            # Predicted
            if row.prob_home_win > row.prob_draw && row.prob_home_win > row.prob_away_win
                push!(predicted_outcomes, "H")
            elseif row.prob_draw > row.prob_away_win
                push!(predicted_outcomes, "D")
            else
                push!(predicted_outcomes, "A")
            end
            
            # Actual
            if row.actual_home_ft > row.actual_away_ft
                push!(actual_outcomes, "H")
            elseif row.actual_home_ft == row.actual_away_ft
                push!(actual_outcomes, "D")
            else
                push!(actual_outcomes, "A")
            end
        end
        
        accuracy = mean(predicted_outcomes .== actual_outcomes)
        
        push!(results, (
            uncertainty_level = label,
            n_matches = nrow(subset),
            mae_home = mae_home,
            mae_away = mae_away,
            classification_accuracy = accuracy,
            mean_uncertainty = mean(subset.uncertainty_home)
        ))
    end
    
    return results
end
