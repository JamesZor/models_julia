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

    return MatchHalfChainPredicts( 
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
  return MatchPredict(ht_predict, ft_predict)
end
