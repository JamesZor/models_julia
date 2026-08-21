# current_development/scottish_lower/open_play/l03_recombination_models.jl
#
# LOADER: Two-Stage Recombination Models (Poisson Proof-of-Concept & Negative Binomial)
#
# Implements:
# 1. Branch A: Analytical Empirical Bayes Penalty & Referee Shrinkage Estimator
# 2. Branch B: Integrated Co-Trained Turing Bayesian Model (Open Play Goals + Matchday Penalties)
# 3. Discrete Probability Convolution & Moment-Matched Score Matrix Recombination
# 4. Strictly conforming to docs/turing_ad_performance_guide.md (zero scalar loops, SIMD broadcasting)

using Turing
using DynamicPPL
using Distributions
using LinearAlgebra
using Statistics
using DataFrames
using Dates

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Predictions = BayesianFootball.Predictions

# ==============================================================================
# 1. BRANCH A: ANALYTICAL EMPIRICAL BAYES PENALTY & REFEREE SHRINKAGE
# ==============================================================================

struct EmpiricalBayesPenaltyEstimator
    base_pen_rate::Float64
    ha_pen::Float64
    ref_strictness::Dict{Int, Float64}
    team_draw_rates::Dict{Int, Float64}
    team_concede_rates::Dict{Int, Float64}
    p_conv::Float64
    lambda_og::Float64
end

function fit_empirical_bayes_penalties(ds::DataStore, match_ids::Vector{Int};
                                      shrink_ref_k::Float64 = 10.0,
                                      shrink_team_k::Float64 = 15.0)::EmpiricalBayesPenaltyEstimator
    df_clean, df_ref = build_open_play_target_dataset(ds)
    m_sub = filter(r -> r.match_id in match_ids, df_clean)
    
    tot_matches = nrow(m_sub)
    if tot_matches == 0
        return EmpiricalBayesPenaltyEstimator(0.136, 0.19, Dict{Int,Float64}(), Dict{Int,Float64}(), Dict{Int,Float64}(), 0.768, 0.0276)
    end
    
    tot_home_pens = sum(m_sub.home_pen_awarded)
    tot_away_pens = sum(m_sub.away_pen_awarded)
    tot_pens = tot_home_pens + tot_away_pens
    
    base_rate = max(1e-4, tot_pens / (2.0 * tot_matches))
    ha_rate   = tot_home_pens > 0 && tot_away_pens > 0 ? log(tot_home_pens / tot_away_pens) / 2.0 : 0.19
    
    # 1. Referee Empirical Bayes Shrinkage (Gamma-Poisson / Log Normal)
    ref_counts = combine(groupby(m_sub, :referee_id),
        nrow => :matches,
        [:home_pen_awarded, :away_pen_awarded] => ((h, a) -> sum(h) + sum(a)) => :pens
    )
    ref_strictness = Dict{Int, Float64}()
    for r in eachrow(ref_counts)
        r.referee_id == 0 && continue
        # Empirical Bayes shrinkage towards base_rate
        obs_rate = (r.pens + shrink_ref_k * (2.0 * base_rate)) / (2.0 * r.matches + shrink_ref_k)
        ref_strictness[r.referee_id] = log(obs_rate / (2.0 * base_rate))
    end
    
    # 2. Team Penalty Draw & Concede Shrinkage
    team_draw = Dict{Int, Float64}()
    team_concede = Dict{Int, Float64}()
    
    home_grp = combine(groupby(m_sub, :home_team_id), nrow => :n_h, :home_pen_awarded => sum => :pens_h_draw, :away_pen_awarded => sum => :pens_h_concede)
    away_grp = combine(groupby(m_sub, :away_team_id), nrow => :n_a, :away_pen_awarded => sum => :pens_a_draw, :home_pen_awarded => sum => :pens_a_concede)
    
    all_teams = unique(vcat(m_sub.home_team_id, m_sub.away_team_id))
    for tid in all_teams
        h_row = filter(r -> r.home_team_id == tid, home_grp)
        a_row = filter(r -> r.away_team_id == tid, away_grp)
        
        n_m = (nrow(h_row) > 0 ? h_row.n_h[1] : 0) + (nrow(a_row) > 0 ? a_row.n_a[1] : 0)
        p_draw = (nrow(h_row) > 0 ? h_row.pens_h_draw[1] : 0) + (nrow(a_row) > 0 ? a_row.pens_a_draw[1] : 0)
        p_concede = (nrow(h_row) > 0 ? h_row.pens_h_concede[1] : 0) + (nrow(a_row) > 0 ? a_row.pens_a_concede[1] : 0)
        
        if n_m > 0
            shrunk_draw = (p_draw + shrink_team_k * base_rate) / (n_m + shrink_team_k)
            shrunk_concede = (p_concede + shrink_team_k * base_rate) / (n_m + shrink_team_k)
            team_draw[tid] = log(shrunk_draw / base_rate)
            team_concede[tid] = log(shrunk_concede / base_rate)
        else
            team_draw[tid] = 0.0
            team_concede[tid] = 0.0
        end
    end
    
    return EmpiricalBayesPenaltyEstimator(
        base_rate, ha_rate, ref_strictness, team_draw, team_concede, 0.768, 0.0276
    )
end

function compute_match_noise_intensity(eb::EmpiricalBayesPenaltyEstimator, home_team_id::Int, away_team_id::Int, ref_id::Int)
    gamma_ref = get(eb.ref_strictness, ref_id, 0.0)
    alpha_draw_h = get(eb.team_draw_rates, home_team_id, 0.0)
    beta_conc_a  = get(eb.team_concede_rates, away_team_id, 0.0)
    
    alpha_draw_a = get(eb.team_draw_rates, away_team_id, 0.0)
    beta_conc_h  = get(eb.team_concede_rates, home_team_id, 0.0)
    
    log_pen_h = clamp(log(eb.base_pen_rate) + eb.ha_pen + gamma_ref + alpha_draw_h + beta_conc_a, -10.0, 2.0)
    log_pen_a = clamp(log(eb.base_pen_rate) - eb.ha_pen + gamma_ref + alpha_draw_a + beta_conc_h, -10.0, 2.0)
    
    lambda_pen_h = exp(log_pen_h)
    lambda_pen_a = exp(log_pen_a)
    
    lambda_noise_h = (eb.p_conv * lambda_pen_h) + eb.lambda_og
    lambda_noise_a = (eb.p_conv * lambda_pen_a) + eb.lambda_og
    
    return lambda_noise_h, lambda_noise_a
end

# ==============================================================================
# 2. DISCRETE PROBABILITY CONVOLUTION & MOMENT MATCHING RECOMBINATION
# ==============================================================================

"""
    reconstruct_score_matrix_discrete_conv(mu_open_h, mu_open_a, lambda_noise_h, lambda_noise_a;
                                           dist=:poisson, r_h=Inf, r_a=Inf, max_goals=10)

Performs exact 1D discrete convolution (P_open * P_noise) on home and away marginals
and builds the normalized (max_goals+1) x (max_goals+1) joint score matrix.
"""
function reconstruct_score_matrix_discrete_conv(mu_open_h::Float64, mu_open_a::Float64,
                                                lambda_noise_h::Float64, lambda_noise_a::Float64;
                                                dist::Symbol = :poisson,
                                                r_h::Float64 = Inf, r_a::Float64 = Inf,
                                                max_goals::Int = 10)::Matrix{Float64}
    # 1. Home Marginals
    p_open_h = dist == :negbin && isfinite(r_h) && r_h > 0.0 ?
        [pdf(NegativeBinomial2(mu_open_h, r_h), k) for k in 0:max_goals] :
        [pdf(Poisson(mu_open_h), k) for k in 0:max_goals]
    p_noise_h = [pdf(Poisson(lambda_noise_h), k) for k in 0:max_goals]
    
    p_tot_h = zeros(Float64, max_goals + 1)
    for k in 0:max_goals
        for m in 0:k
            p_tot_h[k + 1] += p_open_h[m + 1] * p_noise_h[k - m + 1]
        end
    end
    p_tot_h ./= sum(p_tot_h)
    
    # 2. Away Marginals
    p_open_a = dist == :negbin && isfinite(r_a) && r_a > 0.0 ?
        [pdf(NegativeBinomial2(mu_open_a, r_a), k) for k in 0:max_goals] :
        [pdf(Poisson(mu_open_a), k) for k in 0:max_goals]
    p_noise_a = [pdf(Poisson(lambda_noise_a), k) for k in 0:max_goals]
    
    p_tot_a = zeros(Float64, max_goals + 1)
    for k in 0:max_goals
        for m in 0:k
            p_tot_a[k + 1] += p_open_a[m + 1] * p_noise_a[k - m + 1]
        end
    end
    p_tot_a ./= sum(p_tot_a)
    
    # 3. Outer Product Score Matrix
    S = p_tot_h * p_tot_a'
    return S ./ sum(S)
end

"""
    reconstruct_score_matrix_moment_match(mu_open_h, mu_open_a, lambda_noise_h, lambda_noise_a;
                                          dist=:poisson, r_h=Inf, r_a=Inf, max_goals=10)

Moment-matched approximation: matches mean E[Y] = \mu_{open} + \lambda_{noise}
and variance Var(Y) = Var(Y_{open}) + \lambda_{noise}.
"""
function reconstruct_score_matrix_moment_match(mu_open_h::Float64, mu_open_a::Float64,
                                               lambda_noise_h::Float64, lambda_noise_a::Float64;
                                               dist::Symbol = :poisson,
                                               r_h::Float64 = Inf, r_a::Float64 = Inf,
                                               max_goals::Int = 10)::Matrix{Float64}
    mu_tot_h = mu_open_h + lambda_noise_h
    mu_tot_a = mu_open_a + lambda_noise_a
    
    if dist == :poisson || !isfinite(r_h) || r_h <= 0.0
        p_h = [pdf(Poisson(mu_tot_h), k) for k in 0:max_goals]
        p_a = [pdf(Poisson(mu_tot_a), k) for k in 0:max_goals]
    else
        var_h = (mu_open_h + (mu_open_h^2 / r_h)) + lambda_noise_h
        var_a = (mu_open_a + (mu_open_a^2 / r_a)) + lambda_noise_a
        
        r_tot_h = (var_h > mu_tot_h + 1e-4) ? (mu_tot_h^2) / (var_h - mu_tot_h) : 100.0
        r_tot_a = (var_a > mu_tot_a + 1e-4) ? (mu_tot_a^2) / (var_a - mu_tot_a) : 100.0
        
        p_h = [pdf(NegativeBinomial2(mu_tot_h, r_tot_h), k) for k in 0:max_goals]
        p_a = [pdf(NegativeBinomial2(mu_tot_a, r_tot_a), k) for k in 0:max_goals]
    end
    
    p_h ./= sum(p_h)
    p_a ./= sum(p_a)
    
    S = p_h * p_a'
    return S ./ sum(S)
end

function score_matrix_divergence(S1::Matrix{Float64}, S2::Matrix{Float64})
    l1_diff = sum(abs.(S1 .- S2)) / 2.0 # Total variation distance
    kl_div = 0.0
    for i in 1:size(S1, 1), j in 1:size(S1, 2)
        p = clamp(S1[i, j], 1e-12, 1.0)
        q = clamp(S2[i, j], 1e-12, 1.0)
        kl_div += p * log(p / q)
    end
    return (; total_variation = l1_diff, kl_divergence = kl_div)
end

# ==============================================================================
# 3. BRANCH B: INTEGRATED CO-TRAINED TURING ENGINE (POISSON PROOF-OF-CONCEPT)
# ==============================================================================

struct TeamGoalsPoissonOpenPlayModel <: PreGame.AbstractPreGameModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    name::String
end
TeamGoalsPoissonOpenPlayModel(; dynamics_config = PreGame.TimeDecayDynamics(days_half_life=365.0), name="goals_pois_open_play") =
    TeamGoalsPoissonOpenPlayModel(dynamics_config, name)

struct TeamGoalsRecombIntegratedPoissonModel <: PreGame.AbstractPreGameModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    name::String
end
TeamGoalsRecombIntegratedPoissonModel(; dynamics_config = PreGame.TimeDecayDynamics(days_half_life=365.0), name="recomb_pois_integrated_bayes") =
    TeamGoalsRecombIntegratedPoissonModel(dynamics_config, name)

# --- FeatureSet Overload for Integrated Model ---
function Features.create_features(
    boundaries::Vector{Features.GroupBoundary},
    ds::Data.DataStore,
    model::Union{TeamGoalsPoissonOpenPlayModel, TeamGoalsRecombIntegratedPoissonModel}
)
    df_clean, df_ref = build_open_play_target_dataset(ds)
    all_refs = unique(filter(x -> x > 0, df_clean.referee_id))
    ref_map = Dict(r => idx for (idx, r) in enumerate(all_refs))
    
    map(boundaries) do b
        m = filter(r -> r.match_id in b.train_match_ids, df_clean)
        
        home_ids = Vector{Int}(m.home_team_id)
        away_ids = Vector{Int}(m.away_team_id)
        
        home_open_goals = Vector{Int}(m.home_goals_np_nog)
        away_open_goals = Vector{Int}(m.away_goals_np_nog)
        
        home_pens = Vector{Int}(m.home_pen_awarded)
        away_pens = Vector{Int}(m.away_pen_awarded)
        
        ref_indices = [get(ref_map, r, 0) for r in m.referee_id]
        ref_mask    = Float64.(ref_indices .> 0)
        ref_ids_clamped = [idx > 0 ? idx : 1 for idx in ref_indices]
        
        date_deltas = Vector{Float64}(b.train_dates)
        weights     = 0.5 .^ (date_deltas ./ model.dynamics_config.days_half_life)
        
        all_teams = sort(unique(vcat(home_ids, away_ids)))
        team_map  = Dict(t => idx for (idx, t) in enumerate(all_teams))
        
        h_idx = [team_map[t] for t in home_ids]
        a_idx = [team_map[t] for t in away_ids]
        
        month_indices  = month.(m.match_date)
        league_indices = ones(Int, length(home_ids))
        
        Features.FeatureSet(
            b,
            Dict{Symbol, Any}(
                :home_team_indices   => h_idx,
                :away_team_indices   => a_idx,
                :month_indices       => month_indices,
                :league_indices      => league_indices,
                :home_open_goals     => home_open_goals,
                :away_open_goals     => away_open_goals,
                :home_pens           => home_pens,
                :away_pens           => away_pens,
                :ref_indices         => ref_ids_clamped,
                :ref_mask            => ref_mask,
                :match_weights       => weights,
                :n_teams             => length(all_teams),
                :n_refs              => max(1, length(all_refs)),
                :n_months            => 12,
                :n_leagues           => 1,
                :team_map            => team_map,
                :ref_map             => ref_map,
                :clean_df            => df_clean
            )
        )
    end
end

# --- Vectorized Turing Model: Pure Open Play Poisson ---
@model function _turing_goals_poisson_open_play(
    home_indices::Vector{Int},
    away_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    home_open_goals::Vector{Int},
    away_open_goals::Vector{Int},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_months::Int,
    n_leagues::Int
)
    # 1. Priors
    base_mu ~ Normal(0.2, 0.5)
    ha_home ~ Normal(0.2, 0.15)
    
    delta_month ~ filldist(Normal(0.0, 0.1), n_months)
    delta_league ~ filldist(Normal(0.0, 0.1), n_leagues)
    
    raw_alpha ~ filldist(Normal(0.0, 0.3), n_teams)
    raw_beta  ~ filldist(Normal(0.0, 0.3), n_teams)
    
    alpha = raw_alpha .- mean(raw_alpha)
    beta  = raw_beta  .- mean(raw_beta)
    
    # 2. Vectorized Intensity
    int_m = base_mu .+ view(delta_month, month_indices) .+ view(delta_league, league_indices)
    
    log_mu_h = clamp.(int_m .+ ha_home .+ view(alpha, home_indices) .- view(beta, away_indices), -10.0, 10.0)
    log_mu_a = clamp.(int_m .+ view(alpha, away_indices) .- view(beta, home_indices), -10.0, 10.0)
    
    mu_h = exp.(log_mu_h) .+ 1e-6
    mu_a = exp.(log_mu_a) .+ 1e-6
    
    # 3. Vectorized Likelihood
    ll_h = logpdf.(Poisson.(mu_h), home_open_goals)
    ll_a = logpdf.(Poisson.(mu_a), away_open_goals)
    
    Turing.@addlogprob! sum((ll_h .+ ll_a) .* match_weights)
end

# --- Vectorized Turing Model: Integrated Open Play Goals + Referee Penalty Co-Trained Engine ---
@model function _turing_goals_recomb_integrated_poisson(
    home_indices::Vector{Int},
    away_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    home_open_goals::Vector{Int},
    away_open_goals::Vector{Int},
    home_pens::Vector{Int},
    away_pens::Vector{Int},
    ref_indices::Vector{Int},
    ref_mask::Vector{Float64},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_refs::Int,
    n_months::Int,
    n_leagues::Int
)
    # 1. Open Play Priors
    base_mu ~ Normal(0.2, 0.5)
    ha_home ~ Normal(0.2, 0.15)
    
    delta_month ~ filldist(Normal(0.0, 0.1), n_months)
    delta_league ~ filldist(Normal(0.0, 0.1), n_leagues)
    
    raw_alpha ~ filldist(Normal(0.0, 0.3), n_teams)
    raw_beta  ~ filldist(Normal(0.0, 0.3), n_teams)
    
    alpha = raw_alpha .- mean(raw_alpha)
    beta  = raw_beta  .- mean(raw_beta)
    
    # 2. Penalty Sub-Model Priors
    pen_base_mu ~ Normal(-2.0, 0.5) # log(0.136) ~ -2.0
    ha_pen      ~ Normal(0.19, 0.1) # Home whistle bias
    
    sigma_ref   ~ Exponential(1.0)
    raw_gamma_ref ~ filldist(Normal(0.0, 1.0), n_refs)
    gamma_ref   = raw_gamma_ref .* sigma_ref
    
    alpha_pen_draw ~ filldist(Normal(0.0, 0.2), n_teams)
    beta_pen_foul  ~ filldist(Normal(0.0, 0.2), n_teams)
    
    # 3. Vectorized Open Play Intensity
    int_m = base_mu .+ view(delta_month, month_indices) .+ view(delta_league, league_indices)
    
    log_mu_h = clamp.(int_m .+ ha_home .+ view(alpha, home_indices) .- view(beta, away_indices), -10.0, 10.0)
    log_mu_a = clamp.(int_m .+ view(alpha, away_indices) .- view(beta, home_indices), -10.0, 10.0)
    
    mu_h = exp.(log_mu_h) .+ 1e-6
    mu_a = exp.(log_mu_a) .+ 1e-6
    
    # 4. Vectorized Penalty Intensity
    log_pen_h = clamp.(pen_base_mu .+ ha_pen .+ view(gamma_ref, ref_indices) .+ view(alpha_pen_draw, home_indices) .+ view(beta_pen_foul, away_indices), -10.0, 5.0)
    log_pen_a = clamp.(pen_base_mu .- ha_pen .+ view(gamma_ref, ref_indices) .+ view(alpha_pen_draw, away_indices) .+ view(beta_pen_foul, home_indices), -10.0, 5.0)
    
    lambda_pen_h = exp.(log_pen_h) .+ 1e-6
    lambda_pen_a = exp.(log_pen_a) .+ 1e-6
    
    # 5. Combined Likelihood
    ll_open_h = logpdf.(Poisson.(mu_h), home_open_goals)
    ll_open_a = logpdf.(Poisson.(mu_a), away_open_goals)
    
    ll_pen_h  = logpdf.(Poisson.(lambda_pen_h), home_pens)
    ll_pen_a  = logpdf.(Poisson.(lambda_pen_a), away_pens)
    
    ll_open_tot = (ll_open_h .+ ll_open_a) .* match_weights
    ll_pen_tot  = (ll_pen_h .+ ll_pen_a) .* ref_mask .* match_weights
    
    Turing.@addlogprob! sum(ll_open_tot .+ ll_pen_tot)
end

# --- Builder Implementations ---
function PreGame.build_turing_model(model::TeamGoalsPoissonOpenPlayModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    return _turing_goals_poisson_open_play(
        d[:home_team_indices],
        d[:away_team_indices],
        d[:month_indices],
        d[:league_indices],
        d[:home_open_goals],
        d[:away_open_goals],
        d[:match_weights],
        d[:n_teams],
        d[:n_months],
        d[:n_leagues]
    )
end

function PreGame.build_turing_model(model::TeamGoalsRecombIntegratedPoissonModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    return _turing_goals_recomb_integrated_poisson(
        d[:home_team_indices],
        d[:away_team_indices],
        d[:month_indices],
        d[:league_indices],
        d[:home_open_goals],
        d[:away_open_goals],
        d[:home_pens],
        d[:away_pens],
        d[:ref_indices],
        d[:ref_mask],
        d[:match_weights],
        d[:n_teams],
        d[:n_refs],
        d[:n_months],
        d[:n_leagues]
    )
end

# --- Parameter Extraction Interface ---
function PreGame.extract_parameters(model::TeamGoalsPoissonOpenPlayModel, chain::Chains, feature_set::Features.FeatureSet)
    d = feature_set.data
    team_map = d[:team_map]
    n_teams  = d[:n_teams]
    
    base_mu_samples = vec(Array(chain["base_mu"]))
    ha_samples      = vec(Array(chain["ha_home"]))
    
    alpha_dict = Dict{Int, Vector{Float64}}()
    beta_dict  = Dict{Int, Vector{Float64}}()
    
    raw_alpha_mat = Array(chain[["raw_alpha[$i]" for i in 1:n_teams]])
    raw_beta_mat  = Array(chain[["raw_beta[$i]" for i in 1:n_teams]])
    
    alpha_mat = raw_alpha_mat .- mean(raw_alpha_mat, dims=2)
    beta_mat  = raw_beta_mat  .- mean(raw_beta_mat, dims=2)
    
    for (team_id, idx) in team_map
        alpha_dict[team_id] = alpha_mat[:, idx]
        beta_dict[team_id]  = beta_mat[:, idx]
    end
    
    return Dict{Symbol, Any}(
        :base_mu    => base_mu_samples,
        :ha_home    => ha_samples,
        :alpha      => alpha_dict,
        :beta       => beta_dict,
        :team_map   => team_map
    )
end

function PreGame.extract_parameters(model::TeamGoalsRecombIntegratedPoissonModel, chain::Chains, feature_set::Features.FeatureSet)
    d = feature_set.data
    team_map = d[:team_map]
    ref_map  = d[:ref_map]
    n_teams  = d[:n_teams]
    n_refs   = d[:n_refs]
    
    base_mu_samples = vec(Array(chain["base_mu"]))
    ha_samples      = vec(Array(chain["ha_home"]))
    
    pen_base_mu_samples = vec(Array(chain["pen_base_mu"]))
    ha_pen_samples      = vec(Array(chain["ha_pen"]))
    sigma_ref_samples   = vec(Array(chain["sigma_ref"]))
    
    alpha_dict = Dict{Int, Vector{Float64}}()
    beta_dict  = Dict{Int, Vector{Float64}}()
    
    raw_alpha_mat = Array(chain[["raw_alpha[$i]" for i in 1:n_teams]])
    raw_beta_mat  = Array(chain[["raw_beta[$i]" for i in 1:n_teams]])
    
    alpha_mat = raw_alpha_mat .- mean(raw_alpha_mat, dims=2)
    beta_mat  = raw_beta_mat  .- mean(raw_beta_mat, dims=2)
    
    for (team_id, idx) in team_map
        alpha_dict[team_id] = alpha_mat[:, idx]
        beta_dict[team_id]  = beta_mat[:, idx]
    end
    
    gamma_ref_dict = Dict{Int, Vector{Float64}}()
    if n_refs > 0 && haskey(chain, "raw_gamma_ref[1]")
        raw_gamma_mat = Array(chain[["raw_gamma_ref[$i]" for i in 1:n_refs]])
        gamma_mat = raw_gamma_mat .* sigma_ref_samples
        for (ref_id, idx) in ref_map
            gamma_ref_dict[ref_id] = gamma_mat[:, idx]
        end
    end
    
    alpha_pen_draw_dict = Dict{Int, Vector{Float64}}()
    beta_pen_foul_dict  = Dict{Int, Vector{Float64}}()
    if haskey(chain, "alpha_pen_draw[1]")
        apd_mat = Array(chain[["alpha_pen_draw[$i]" for i in 1:n_teams]])
        bpf_mat = Array(chain[["beta_pen_foul[$i]" for i in 1:n_teams]])
        for (team_id, idx) in team_map
            alpha_pen_draw_dict[team_id] = apd_mat[:, idx]
            beta_pen_foul_dict[team_id]  = bpf_mat[:, idx]
        end
    end
    
    return Dict{Symbol, Any}(
        :base_mu        => base_mu_samples,
        :ha_home        => ha_samples,
        :alpha          => alpha_dict,
        :beta           => beta_dict,
        :pen_base_mu    => pen_base_mu_samples,
        :ha_pen         => ha_pen_samples,
        :sigma_ref      => sigma_ref_samples,
        :gamma_ref      => gamma_ref_dict,
        :alpha_pen_draw => alpha_pen_draw_dict,
        :beta_pen_foul  => beta_pen_foul_dict,
        :team_map       => team_map,
        :ref_map        => ref_map
    )
end

# --- Score Matrix & Prediction Overloads ---
function Predictions.compute_score_matrix(model::TeamGoalsPoissonOpenPlayModel, r::DataFrameRow; max_goals::Int = 10)
    mu_h = Float64(r.λ_h)
    mu_a = Float64(r.λ_a)
    
    # Baseline open-play Poisson score matrix (uniform own goal baseline)
    p_h = [pdf(Poisson(mu_h), k) for k in 0:max_goals]
    p_a = [pdf(Poisson(mu_a), k) for k in 0:max_goals]
    p_h ./= sum(p_h)
    p_a ./= sum(p_a)
    
    S = p_h * p_a'
    return S ./ sum(S)
end

function Predictions.compute_score_matrix(model::TeamGoalsRecombIntegratedPoissonModel, r::DataFrameRow; max_goals::Int = 10)
    mu_h = Float64(r.λ_h)
    mu_a = Float64(r.λ_a)
    
    lambda_noise_h = hasproperty(r, :lambda_noise_h) ? Float64(r.lambda_noise_h) : 0.136 * 0.768 + 0.0276
    lambda_noise_a = hasproperty(r, :lambda_noise_a) ? Float64(r.lambda_noise_a) : 0.136 * 0.768 + 0.0276
    
    return reconstruct_score_matrix_discrete_conv(mu_h, mu_a, lambda_noise_h, lambda_noise_a; dist=:poisson, max_goals=max_goals)
end

println("✓ l03_recombination_models.jl loaded (Empirical Bayes + Integrated Turing Engines + Discrete Convolution)")
