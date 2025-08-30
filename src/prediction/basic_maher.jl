
# prediction/basic_maher.jl
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
    
    # Centering: subtract row means from each element in the row
    log_α_centered = log_α_raw .- mean(log_α_raw, dims=2)
    log_β_centered = log_β_raw .- mean(log_β_raw, dims=2)
    
    # Exponentiation
    α_samples = exp.(log_α_centered)
    β_samples = exp.(log_β_centered)
    γ_samples = exp.(log_γ)
    
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
    return (hw = hw, dr = dr, aw = aw)
end

function calculate_correct_score_dict_ht(p::Matrix{Float64})
    # Initialize dict with all score combinations
    correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    
    # Standard scores (0-0 through 2-2)
    for h in 0:2, a in 0:2
        correct_scores[(h, a)] = p[h+1, a+1]
    end
    
    # Calculate any_unquoted probability
    any_unquoted = 0.0
    max_goals = size(p,1)-1
    for h in 0:max_goals, a in 0:max_goals
        if h > 2 || a > 2
            any_unquoted += p[h+1, a+1]
        end
    end
    correct_scores["any_unquoted"] = any_unquoted
    
    return correct_scores
end

function calculate_correct_score_dict_ft(p::Matrix{Float64})
    # Initialize dict with all score combinations
    correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    
    # Standard scores (0-0 through 3-3)
    for h in 0:3, a in 0:3
        correct_scores[(h, a)] = p[h+1, a+1]
    end
    
    # Calculate other scores
    other_home_win = 0.0
    other_away_win = 0.0
    other_draw = 0.0
    max_goals = size(p,1)-1
    
    for h in 0:max_goals, a in 0:max_goals
        if h > 3 || a > 3
            if h > a
                other_home_win += p[h+1, a+1]
            elseif a > h
                other_away_win += p[h+1, a+1]
            else  # h == a
                other_draw += p[h+1, a+1]
            end
        end
    end
    
    correct_scores["other_home_win"] = other_home_win
    correct_scores["other_away_win"] = other_away_win
    correct_scores["other_draw"] = other_draw
    
    return correct_scores
end

function calculate_under_prob(p::Matrix{Float64}, threshold::Int)
    prob_under = 0.0
    max_goals = size(p,1)-1
    for h in 0:max_goals, a in 0:max_goals
        total_goals = h + a
        if total_goals <= threshold
            prob_under += p[h+1, a+1]
        end
    end
    return prob_under
end

function calculate_btts(p::Matrix{Float64})
    btts_prob = 0.0
    max_goals = size(p,1)-1
    for h in 1:max_goals, a in 1:max_goals  # Start from 1 (both teams score)
        btts_prob += p[h+1, a+1]
    end
    return btts_prob
end

function predict_match_chain_ht(
    home_team::AbstractString, 
    away_team::AbstractString,
    chain::Chains,
    mapping::MappedData
)
    n_samples = size(chain, 1) * size(chain, 3)
    posterior_samples = extract_posterior_samples(chain, mapping)
    home_idx = findfirst(==(home_team), posterior_samples.teams)
    away_idx = findfirst(==(away_team), posterior_samples.teams)
    
    # Initialize result vectors
    λ_home = zeros(n_samples)
    λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples)
    draw_probs = zeros(n_samples)
    away_win_probs = zeros(n_samples)
    
    # Initialize correct score dict of vectors
    cs_keys = vcat(
        [(h, a) for h in 0:2 for a in 0:2],
        ["any_unquoted"]
    )
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(
        key => zeros(n_samples) for key in cs_keys
    )
    
    under_05 = zeros(n_samples)
    under_15 = zeros(n_samples)
    under_25 = zeros(n_samples)
    
    for i in 1:n_samples
        α_h = posterior_samples.α[i, home_idx]
        β_h = posterior_samples.β[i, home_idx]
        α_a = posterior_samples.α[i, away_idx]
        β_a = posterior_samples.β[i, away_idx]
        γ = posterior_samples.γ[i]
        
        λ_home[i] = α_h * β_a * γ
        λ_away[i] = α_a * β_h
        p = compute_xScore(λ_home[i], λ_away[i], 10)
        
        hda = calculate_1x2(p)
        cs = calculate_correct_score_dict_ht(p)
        
        home_win_probs[i] = hda.hw
        draw_probs[i] = hda.dr
        away_win_probs[i] = hda.aw
        
        # Assign correct scores
        for (key, value) in cs
            correct_score[key][i] = value
        end
        
        under_05[i] = calculate_under_prob(p, 0)
        under_15[i] = calculate_under_prob(p, 1)
        under_25[i] = calculate_under_prob(p, 2)
    end
    
    return Predictions.MatchHTPredictions(
        λ_home,
        λ_away,
        home_win_probs,
        draw_probs,
        away_win_probs,
        correct_score,
        under_05,
        under_15,
        under_25
    )
end

function predict_match_chain_ft(
    home_team::AbstractString, 
    away_team::AbstractString,
    chain::Chains,
    mapping::MappedData
)
    n_samples = size(chain, 1) * size(chain, 3)
    posterior_samples = extract_posterior_samples(chain, mapping)
    home_idx = findfirst(==(home_team), posterior_samples.teams)
    away_idx = findfirst(==(away_team), posterior_samples.teams)
    
    # Initialize result vectors
    λ_home = zeros(n_samples)
    λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples)
    draw_probs = zeros(n_samples)
    away_win_probs = zeros(n_samples)
    
    # Initialize correct score dict of vectors
    cs_keys = vcat(
        [(h, a) for h in 0:3 for a in 0:3],
        ["other_home_win", "other_away_win", "other_draw"]
    )
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(
        key => zeros(n_samples) for key in cs_keys
    )
    
    under_05 = zeros(n_samples)
    under_15 = zeros(n_samples)
    under_25 = zeros(n_samples)
    under_35 = zeros(n_samples)
    btts = zeros(n_samples)
    
    for i in 1:n_samples
        α_h = posterior_samples.α[i, home_idx]
        β_h = posterior_samples.β[i, home_idx]
        α_a = posterior_samples.α[i, away_idx]
        β_a = posterior_samples.β[i, away_idx]
        γ = posterior_samples.γ[i]
        
        λ_home[i] = α_h * β_a * γ
        λ_away[i] = α_a * β_h
        p = compute_xScore(λ_home[i], λ_away[i], 10)
        
        hda = calculate_1x2(p)
        cs = calculate_correct_score_dict_ft(p)
        
        home_win_probs[i] = hda.hw
        draw_probs[i] = hda.dr
        away_win_probs[i] = hda.aw
        
        # Assign correct scores
        for (key, value) in cs
            correct_score[key][i] = value
        end
        
        under_05[i] = calculate_under_prob(p, 0)
        under_15[i] = calculate_under_prob(p, 1)
        under_25[i] = calculate_under_prob(p, 2)
        under_35[i] = calculate_under_prob(p, 3)
        btts[i] = calculate_btts(p)
    end
    
    return Predictions.MatchFTPredictions(
        λ_home,
        λ_away,
        home_win_probs,
        draw_probs,
        away_win_probs,
        correct_score,
        under_05,
        under_15,
        under_25,
        under_35,
        btts
    )
end

function predict_match_ft_ht_chain(
    home_team::AbstractString,
    away_team::AbstractString,
    round_chains::TrainedChains,
    mapping::MappedData
) 
    ht_predict = predict_match_chain_ht(home_team, away_team, round_chains.ht, mapping)
    ft_predict = predict_match_chain_ft(home_team, away_team, round_chains.ft, mapping)
    return Predictions.MatchLinePredictions(ht_predict, ft_predict)
end

function predict_round_chains( 
    round_chain_split::TrainedChains,
    round_matches::Union{SubDataFrame,DataFrame},
    mapping::MappedData
)
    return Dict(
        row.match_id => predict_match_ft_ht_chain(
            row.home_team, 
            row.away_team, 
            round_chain_split, 
            mapping
        ) for row in eachrow(round_matches)
    )
end

function predict_target_season(
    target_matches::DataFrame,
    results::ExperimentResult,
    mapping::MappedData
)
    grouped_matches = groupby(target_matches, :round)
    n_threads = nthreads()
    
    # Each thread gets its own dictionary
    thread_results = Vector{Dict{Int64, Predictions.MatchLinePredictions}}(undef, n_threads)
    
    @threads for tid in 1:n_threads
        local_dict = Dict{Int64, Predictions.MatchLinePredictions}()
        
        # Thread tid processes every n_threads-th round
        for round_idx in tid:n_threads:length(grouped_matches)
            round_predictions = predict_round_chains(
                results.chains_sequence[round_idx],
                grouped_matches[(round=round_idx,)],
                mapping
            )
            merge!(local_dict, round_predictions)
        end
        thread_results[tid] = local_dict
    end
    # Merge all thread dictionaries
    return merge(thread_results...)
end
