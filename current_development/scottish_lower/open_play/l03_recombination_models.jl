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
using SpecialFunctions: loggamma

using BayesianFootball.MyDistributions: RobustNegativeBinomial

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Predictions = BayesianFootball.Predictions

const SCOTTISH_HOMEAWAY_DISPERSION = PreGame.HomeAwayDispersion(
    log_r     = Normal(2.6, 0.5),
    δ_r_home  = Normal(0.6, 0.5)
)

function _negbin_precompute(goals::Vector{Int}, w::Vector{Float64})
    max_k = isempty(goals) ? 0 : maximum(goals)
    if max_k > 0
        N_j = Float64[sum(w[goals .> j]) for j in 0:(max_k - 1)]
        j_offsets = Float64.(0:(max_k - 1))
    else
        N_j = Float64[]
        j_offsets = Float64[]
    end
    return (
        w = w,
        c_g_lin = w .* goals,
        S_w = sum(w),
        N_j = N_j,
        j_offsets = j_offsets
    )
end

@inline function _negbin_vector_loglik(
    goals::Vector{Int},
    log_λ::AbstractVector,
    r::Real,
    nb::NamedTuple
)
    lr = log(max(r, 1e-6))
    λ  = exp.(log_λ)

    # 1. Scalar r*log(r) block
    ll_r = nb.S_w * r * lr

    # 2. Integer recurrence log(r + j) terms (O(max_goals), zero loggamma, zero allocations)
    ll_gamma = isempty(nb.N_j) ? 0.0 : sum(nb.N_j .* log.(r .+ nb.j_offsets))

    # 3. Vector (k + r) * log(r + λ)
    ll_denom = sum(nb.w .* (goals .+ r) .* log.(r .+ λ))

    # 4. Vector k * log(λ)
    ll_numer = sum(nb.c_g_lin .* log_λ)

    return ll_r + ll_gamma + ll_numer - ll_denom
end

const _TARGET_CACHE = Dict{UInt64, Tuple{DataFrame, Vector{Int}}}()

# ==============================================================================
# 0. OPEN-PLAY DATASET BUILDER
# ==============================================================================

function build_open_play_target_dataset(ds::Data.DataStore)
    k = objectid(ds.matches)
    if haskey(_TARGET_CACHE, k)
        return _TARGET_CACHE[k]
    end

    df = extract_open_play_match_data(ds; include_referees=true)
    
    # Map team names to unique integer team IDs
    all_team_names = sort(unique(vcat(df.home_team, df.away_team)))
    team_name_map  = Dict(t => idx for (idx, t) in enumerate(all_team_names))
    
    df.home_team_id = [team_name_map[t] for t in df.home_team]
    df.away_team_id = [team_name_map[t] for t in df.away_team]
    
    # Add home_pen_awarded and away_pen_awarded
    df.home_pen_awarded = df.pen_scored_h .+ df.pen_missed_h
    df.away_pen_awarded = df.pen_scored_a .+ df.pen_missed_a
    
    # Add home_goals_np_nog and away_goals_np_nog
    df.home_goals_np_nog = df.y_np_nog_h
    df.away_goals_np_nog = df.y_np_nog_a
    
    # Map referee_name -> integer referee_id
    if hasproperty(df, :referee_name)
        unique_refs = unique(filter(x -> x != "Unknown", df.referee_name))
        ref_id_map = Dict(r => idx for (idx, r) in enumerate(unique_refs))
        df.referee_id = [get(ref_id_map, r, 0) for r in df.referee_name]
    else
        df.referee_id = fill(0, nrow(df))
    end
    
    res = (df, unique(filter(x -> x > 0, df.referee_id)))
    _TARGET_CACHE[k] = res
    return res
end

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

function fit_empirical_bayes_penalties(ds::Data.DataStore, match_ids::Vector{Int};
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

raw"""
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
# ==============================================================================
# 3. BRANCH B: INTEGRATED CO-TRAINED TURING ENGINE (POISSON PROOF-OF-CONCEPT)
# ==============================================================================

struct TeamGoalsPoissonModel <: PreGame.AbstractTimeDecayTeamModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    name::String
end
TeamGoalsPoissonModel(; dynamics_config = PreGame.TimeDecayDynamics(days_half_life=365.0), name="goals_pois_ctl") =
    TeamGoalsPoissonModel(dynamics_config, name)

struct TeamGoalsPoissonOpenPlayModel <: PreGame.AbstractTimeDecayTeamModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    name::String
end
TeamGoalsPoissonOpenPlayModel(; dynamics_config = PreGame.TimeDecayDynamics(days_half_life=365.0), name="goals_pois_open_play") =
    TeamGoalsPoissonOpenPlayModel(dynamics_config, name)

struct TeamGoalsRecombIntegratedPoissonModel <: PreGame.AbstractTimeDecayTeamModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    name::String
end
TeamGoalsRecombIntegratedPoissonModel(; dynamics_config = PreGame.TimeDecayDynamics(days_half_life=365.0), name="recomb_pois_integrated_bayes") =
    TeamGoalsRecombIntegratedPoissonModel(dynamics_config, name)

struct TeamGoalsRecombIntegratedNegBinModel <: PreGame.AbstractTimeDecayTeamModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    dispersion_config::PreGame.AbstractDispersionConfig
    name::String
end
TeamGoalsRecombIntegratedNegBinModel(; 
    dynamics_config = PreGame.TimeDecayDynamics(days_half_life=365.0),
    dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
    name = "recomb_negbin_integrated"
) = TeamGoalsRecombIntegratedNegBinModel(dynamics_config, dispersion_config, name)

const AbstractScottishModel = Union{
    TeamGoalsPoissonModel,
    TeamGoalsPoissonOpenPlayModel,
    TeamGoalsRecombIntegratedPoissonModel,
    TeamGoalsRecombIntegratedNegBinModel
}
const AbstractScottishPoissonModel = AbstractScottishModel

# --- FeatureSet Overload for Integrated Model ---
function _build_recomb_features(b::Data.SplitBoundary, ds::Data.DataStore, model::AbstractScottishModel)
    df_clean, df_ref = build_open_play_target_dataset(ds)
    all_refs = unique(filter(x -> x > 0, df_clean.referee_id))
    ref_map = Dict(r => idx for (idx, r) in enumerate(all_refs))
    
    m = filter(r -> r.match_id in b.history_match_ids, df_clean)
    sort!(m, :match_date)
    
    home_ids = Vector{Int}(m.home_team_id)
    away_ids = Vector{Int}(m.away_team_id)
    
    home_gross_goals = Vector{Int}(coalesce.(m.home_score, 0))
    away_gross_goals = Vector{Int}(coalesce.(m.away_score, 0))
    
    home_open_goals = Vector{Int}(m.home_goals_np_nog)
    away_open_goals = Vector{Int}(m.away_goals_np_nog)
    
    home_pens = Vector{Int}(m.home_pen_awarded)
    away_pens = Vector{Int}(m.away_pen_awarded)
    
    ref_indices = [get(ref_map, r, 0) for r in m.referee_id]
    ref_mask    = Float64.(ref_indices .> 0)
    ref_ids_clamped = [idx > 0 ? idx : 1 for idx in ref_indices]
    
    max_date = maximum(m.match_date)
    date_deltas = [Float64(Dates.value(max_date - d)) for d in m.match_date]
    weights     = 0.5 .^ (date_deltas ./ model.dynamics_config.days_half_life)
    
    nb_h = _negbin_precompute(home_open_goals, weights)
    nb_a = _negbin_precompute(away_open_goals, weights)
    
    all_teams = sort(unique(vcat(home_ids, away_ids)))
    team_map  = Dict(t => idx for (idx, t) in enumerate(all_teams))
    
    h_idx = [team_map[t] for t in home_ids]
    a_idx = [team_map[t] for t in away_ids]
    
    month_indices  = month.(m.match_date)
    league_indices = ones(Int, length(home_ids))
    
    return Features.FeatureSet(
        Dict{Symbol, Any}(
            :home_team_indices   => h_idx,
            :away_team_indices   => a_idx,
            :month_indices       => month_indices,
            :league_indices      => league_indices,
            :home_gross_goals    => home_gross_goals,
            :away_gross_goals    => away_gross_goals,
            :home_open_goals     => home_open_goals,
            :away_open_goals     => away_open_goals,
            :home_pens           => home_pens,
            :away_pens           => away_pens,
            :ref_indices         => ref_ids_clamped,
            :ref_mask            => ref_mask,
            :match_weights       => weights,
            :nb_h                => nb_h,
            :nb_a                => nb_a,
            :n_teams             => length(all_teams),
            :n_refs              => max(1, length(all_refs)),
            :n_months            => 12,
            :n_leagues           => 1,
            :team_map            => team_map,
            :ref_map             => ref_map,
            :clean_df            => df_clean,
            :boundary            => b
        )
    )
end

function Features.create_features(
    splits::Vector{<:Tuple{Data.SplitBoundary, <:Any}},
    ds::Data.DataStore,
    model::AbstractScottishModel,
    dynamics_col::Symbol = :match_month
)
    raw_vector = [
        (_build_recomb_features(boundary, ds, model), meta)
        for (boundary, meta) in splits
    ]
    return Features.FeatureCollection(raw_vector)
end

function Features.create_features(
    boundary::Data.SplitBoundary,
    ds::Data.DataStore,
    model::AbstractScottishModel,
    dynamics_col::Symbol = :match_month
)
    return _build_recomb_features(boundary, ds, model)
end

# --- Vectorized Turing Model: Baseline Gross Goals Poisson ---
@model function _turing_goals_poisson_control(
    home_indices::Vector{Int},
    away_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    home_gross_goals::Vector{Int},
    away_gross_goals::Vector{Int},
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
    ll_h = logpdf.(Poisson.(mu_h), home_gross_goals)
    ll_a = logpdf.(Poisson.(mu_a), away_gross_goals)
    
    Turing.@addlogprob! sum((ll_h .+ ll_a) .* match_weights)
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

# --- Vectorized Turing Model: Integrated Negative Binomial Recombination ---
@model function _turing_goals_recomb_integrated_negbin(
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
    nb_h::NamedTuple,
    nb_a::NamedTuple,
    n_teams::Int,
    n_refs::Int,
    n_months::Int,
    n_leagues::Int
)
    # 1. Priors: Open Play Skill & Interception
    base_mu ~ Normal(0.0, 0.3)
    ha_home ~ Normal(0.2, 0.1)
    
    tau_alpha ~ truncated(Normal(0.0, 0.2), 0.0, Inf)
    tau_beta  ~ truncated(Normal(0.0, 0.2), 0.0, Inf)
    
    raw_alpha ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_beta  ~ filldist(Normal(0.0, 1.0), n_teams)
    
    alpha = (raw_alpha .- mean(raw_alpha)) .* tau_alpha
    beta  = (raw_beta  .- mean(raw_beta))  .* tau_beta
    
    delta_month  ~ filldist(Normal(0.0, 0.05), n_months)
    delta_league ~ filldist(Normal(0.0, 0.1), n_leagues)
    
    # 2. Dispersion Parameters (Scottish Home/Away Asymmetry)
    log_r ~ Normal(2.6, 0.5)
    delta_r_home ~ Normal(0.6, 0.5)
    r_a = exp(log_r)
    r_h = exp(log_r + delta_r_home)
    
    # 3. Penalty & Referee Whistle Submodel
    pen_base_mu ~ Normal(-2.0, 0.3)
    ha_pen      ~ Normal(0.1, 0.1)
    sigma_ref   ~ truncated(Normal(0.0, 0.3), 0.0, Inf)
    
    raw_gamma_ref ~ filldist(Normal(0.0, 1.0), n_refs)
    gamma_ref = (raw_gamma_ref .- mean(raw_gamma_ref)) .* sigma_ref
    
    sigma_team_pen ~ truncated(Normal(0.0, 0.2), 0.0, Inf)
    raw_alpha_pen  ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_beta_foul  ~ filldist(Normal(0.0, 1.0), n_teams)
    
    alpha_pen_draw = (raw_alpha_pen .- mean(raw_alpha_pen)) .* sigma_team_pen
    beta_pen_foul  = (raw_beta_foul  .- mean(raw_beta_foul))  .* sigma_team_pen
    
    # 4. Vectorized Open Play Intensity
    int_m = base_mu .+ view(delta_month, month_indices) .+ view(delta_league, league_indices)
    
    log_mu_h = clamp.(int_m .+ ha_home .+ view(alpha, home_indices) .- view(beta, away_indices), -10.0, 10.0)
    log_mu_a = clamp.(int_m .+ view(alpha, away_indices) .- view(beta, home_indices), -10.0, 10.0)
    
    # 5. Vectorized Penalty Intensity
    log_pen_h = clamp.(pen_base_mu .+ ha_pen .+ view(gamma_ref, ref_indices) .+ view(alpha_pen_draw, home_indices) .+ view(beta_pen_foul, away_indices), -10.0, 5.0)
    log_pen_a = clamp.(pen_base_mu .- ha_pen .+ view(gamma_ref, ref_indices) .+ view(alpha_pen_draw, away_indices) .+ view(beta_pen_foul, home_indices), -10.0, 5.0)
    
    lambda_pen_h = exp.(log_pen_h) .+ 1e-6
    lambda_pen_a = exp.(log_pen_a) .+ 1e-6
    
    # 6. Combined Likelihood
    ll_open_h = _negbin_vector_loglik(home_open_goals, log_mu_h, r_h, nb_h)
    ll_open_a = _negbin_vector_loglik(away_open_goals, log_mu_a, r_a, nb_a)
    
    ll_pen_h  = logpdf.(Poisson.(lambda_pen_h), home_pens)
    ll_pen_a  = logpdf.(Poisson.(lambda_pen_a), away_pens)
    
    ll_pen_tot = sum((ll_pen_h .+ ll_pen_a) .* ref_mask .* match_weights)
    
    Turing.@addlogprob! (ll_open_h + ll_open_a + ll_pen_tot)
end

# --- Builder Implementations ---
function PreGame.build_turing_model(model::TeamGoalsPoissonModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    return _turing_goals_poisson_control(
        d[:home_team_indices],
        d[:away_team_indices],
        d[:month_indices],
        d[:league_indices],
        d[:home_gross_goals],
        d[:away_gross_goals],
        d[:match_weights],
        d[:n_teams],
        d[:n_months],
        d[:n_leagues]
    )
end

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

function PreGame.build_turing_model(model::TeamGoalsRecombIntegratedNegBinModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    return _turing_goals_recomb_integrated_negbin(
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
        d[:nb_h],
        d[:nb_a],
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
    if n_refs > 0 && _has_param(chain, "raw_gamma_ref[1]")
        raw_gamma_mat = Array(chain[["raw_gamma_ref[$i]" for i in 1:n_refs]])
        gamma_mat = raw_gamma_mat .* sigma_ref_samples
        for (ref_id, idx) in ref_map
            gamma_ref_dict[ref_id] = gamma_mat[:, idx]
        end
    end
    
    alpha_pen_draw_dict = Dict{Int, Vector{Float64}}()
    beta_pen_foul_dict  = Dict{Int, Vector{Float64}}()
    if _has_param(chain, "alpha_pen_draw[1]")
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

_has_param(chain::Chains, p::String) = Symbol(p) in names(chain) || p in string.(names(chain))

function PreGame.extract_parameters(
    model::Union{TeamGoalsPoissonModel, TeamGoalsPoissonOpenPlayModel},
    df::AbstractDataFrame,
    feature_set::Features.FeatureSet,
    chain::Chains
)
    data = feature_set.data
    team_map = data[:team_map]
    n_teams = data[:n_teams]
    n_months = data[:n_months]
    n_leagues = data[:n_leagues]
    
    base_mu = vec(Array(chain["base_mu"]))
    ha_home = vec(Array(chain["ha_home"]))
    n_samples = length(base_mu)
    
    raw_alpha_mat = Array(chain[["raw_alpha[$i]" for i in 1:n_teams]])
    raw_beta_mat  = Array(chain[["raw_beta[$i]" for i in 1:n_teams]])
    alpha_mat = raw_alpha_mat .- mean(raw_alpha_mat, dims=2)
    beta_mat  = raw_beta_mat  .- mean(raw_beta_mat, dims=2)
    
    delta_month_mat  = _has_param(chain, "delta_month[1]") ? Array(chain[["delta_month[$i]" for i in 1:n_months]]) : zeros(n_samples, n_months)
    delta_league_mat = _has_param(chain, "delta_league[1]") ? Array(chain[["delta_league[$i]" for i in 1:n_leagues]]) : zeros(n_samples, n_leagues)
    
    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        mid = Int(row.match_id)
        h_id = hasproperty(row, :home_team_id) ? Int(row.home_team_id) : (hasproperty(row, :home_team) ? get(team_map, row.home_team, -1) : -1)
        a_id = hasproperty(row, :away_team_id) ? Int(row.away_team_id) : (hasproperty(row, :away_team) ? get(team_map, row.away_team, -1) : -1)
        
        h_idx = get(team_map, h_id, -1)
        a_idx = get(team_map, a_id, -1)
        
        α_h = h_idx > 0 ? alpha_mat[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? beta_mat[:, h_idx]  : zeros(n_samples)
        α_a = a_idx > 0 ? alpha_mat[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? beta_mat[:, a_idx]  : zeros(n_samples)
        
        m_idx = month(row.match_date)
        l_idx = hasproperty(row, :tournament_id) && row.tournament_id == 57 ? 2 : 1
        
        δ_m = (m_idx >= 1 && m_idx <= n_months) ? delta_month_mat[:, m_idx] : zeros(n_samples)
        δ_l = (l_idx >= 1 && l_idx <= n_leagues) ? delta_league_mat[:, l_idx] : zeros(n_samples)
        
        int_m = base_mu .+ δ_m .+ δ_l
        λ_h = exp.(int_m .+ ha_home .+ α_h .- β_a)
        λ_a = exp.(int_m .+           α_a .- β_h)
        
        results[mid] = (;
            λ_h = λ_h,
            λ_a = λ_a,
            r_h = fill(100.0, n_samples),
            r_a = fill(100.0, n_samples),
            true_xg_h = λ_h,
            true_xg_a = λ_a
        )
    end
    return results
end

function PreGame.extract_parameters(
    model::TeamGoalsRecombIntegratedPoissonModel,
    df::AbstractDataFrame,
    feature_set::Features.FeatureSet,
    chain::Chains
)
    data = feature_set.data
    team_map = data[:team_map]
    ref_map  = data[:ref_map]
    n_teams  = data[:n_teams]
    n_refs   = data[:n_refs]
    n_months = data[:n_months]
    n_leagues = data[:n_leagues]
    
    base_mu = vec(Array(chain["base_mu"]))
    ha_home = vec(Array(chain["ha_home"]))
    n_samples = length(base_mu)
    
    raw_alpha_mat = Array(chain[["raw_alpha[$i]" for i in 1:n_teams]])
    raw_beta_mat  = Array(chain[["raw_beta[$i]" for i in 1:n_teams]])
    alpha_mat = raw_alpha_mat .- mean(raw_alpha_mat, dims=2)
    beta_mat  = raw_beta_mat  .- mean(raw_beta_mat, dims=2)
    
    delta_month_mat  = _has_param(chain, "delta_month[1]") ? Array(chain[["delta_month[$i]" for i in 1:n_months]]) : zeros(n_samples, n_months)
    delta_league_mat = _has_param(chain, "delta_league[1]") ? Array(chain[["delta_league[$i]" for i in 1:n_leagues]]) : zeros(n_samples, n_leagues)
    
    pen_base_mu = vec(Array(chain["pen_base_mu"]))
    ha_pen      = vec(Array(chain["ha_pen"]))
    sigma_ref   = vec(Array(chain["sigma_ref"]))
    
    raw_gamma_mat = (n_refs > 0 && _has_param(chain, "raw_gamma_ref[1]")) ? Array(chain[["raw_gamma_ref[$i]" for i in 1:n_refs]]) : zeros(n_samples, n_refs)
    gamma_mat = raw_gamma_mat .* sigma_ref
    
    apd_mat = _has_param(chain, "alpha_pen_draw[1]") ? Array(chain[["alpha_pen_draw[$i]" for i in 1:n_teams]]) : zeros(n_samples, n_teams)
    bpf_mat = _has_param(chain, "beta_pen_foul[1]") ? Array(chain[["beta_pen_foul[$i]" for i in 1:n_teams]]) : zeros(n_samples, n_teams)
    
    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        mid = Int(row.match_id)
        h_id = hasproperty(row, :home_team_id) ? Int(row.home_team_id) : (hasproperty(row, :home_team) ? get(team_map, row.home_team, -1) : -1)
        a_id = hasproperty(row, :away_team_id) ? Int(row.away_team_id) : (hasproperty(row, :away_team) ? get(team_map, row.away_team, -1) : -1)
        
        h_idx = get(team_map, h_id, -1)
        a_idx = get(team_map, a_id, -1)
        
        α_h = h_idx > 0 ? alpha_mat[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? beta_mat[:, h_idx]  : zeros(n_samples)
        α_a = a_idx > 0 ? alpha_mat[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? beta_mat[:, a_idx]  : zeros(n_samples)
        
        m_idx = month(row.match_date)
        l_idx = hasproperty(row, :tournament_id) && row.tournament_id == 57 ? 2 : 1
        
        δ_m = (m_idx >= 1 && m_idx <= n_months) ? delta_month_mat[:, m_idx] : zeros(n_samples)
        δ_l = (l_idx >= 1 && l_idx <= n_leagues) ? delta_league_mat[:, l_idx] : zeros(n_samples)
        
        int_m = base_mu .+ δ_m .+ δ_l
        λ_h = exp.(int_m .+ ha_home .+ α_h .- β_a)
        λ_a = exp.(int_m .+           α_a .- β_h)
        
        # Referee Penalty Intensity
        ref_id = hasproperty(row, :referee_id) && !ismissing(row.referee_id) ? Int(row.referee_id) : -1
        r_idx  = get(ref_map, ref_id, -1)
        γ_ref  = r_idx > 0 ? gamma_mat[:, r_idx] : zeros(n_samples)
        
        apd_h = h_idx > 0 ? apd_mat[:, h_idx] : zeros(n_samples)
        apd_a = a_idx > 0 ? apd_mat[:, a_idx] : zeros(n_samples)
        bpf_h = h_idx > 0 ? bpf_mat[:, h_idx] : zeros(n_samples)
        bpf_a = a_idx > 0 ? bpf_mat[:, a_idx] : zeros(n_samples)
        
        log_pen_h = pen_base_mu .+ ha_pen .+ γ_ref .+ apd_h .+ bpf_a
        log_pen_a = pen_base_mu .- ha_pen .+ γ_ref .+ apd_a .+ bpf_h
        
        lambda_pen_h = exp.(log_pen_h)
        lambda_pen_a = exp.(log_pen_a)
        
        # Noise goal intensity (conversion ~ 0.768 + own goals ~ 0.0276)
        lambda_noise_h = (0.768 .* lambda_pen_h) .+ 0.0276
        lambda_noise_a = (0.768 .* lambda_pen_a) .+ 0.0276
        
        # Total expected goals for scoring rules & portfolio books
        λ_tot_h = λ_h .+ lambda_noise_h
        λ_tot_a = λ_a .+ lambda_noise_a
        
        results[mid] = (;
            λ_h = λ_tot_h,
            λ_a = λ_tot_a,
            r_h = fill(100.0, n_samples),
            r_a = fill(100.0, n_samples),
            true_xg_h = λ_tot_h,
            true_xg_a = λ_tot_a,
            λ_open_h = λ_h,
            λ_open_a = λ_a,
            lambda_noise_h = lambda_noise_h,
            lambda_noise_a = lambda_noise_a
        )
    end
    return results
end

# --- Score Matrix & Prediction Overloads ---
function Predictions.extract_params(model::Union{TeamGoalsPoissonModel, TeamGoalsPoissonOpenPlayModel}, row)
    return (
        λ_h = row.λ_h isa AbstractVector ? row.λ_h : [row.λ_h],
        λ_a = row.λ_a isa AbstractVector ? row.λ_a : [row.λ_a]
    )
end

function Predictions.compute_score_matrix(model::Union{TeamGoalsPoissonModel, TeamGoalsPoissonOpenPlayModel}, params; max_goals::Int = 12)
    p = params isa DataFrameRow ? Predictions.extract_params(model, params) : params
    λ_h = p.λ_h
    λ_a = p.λ_a
    n_samples = length(λ_h)
    
    S = zeros(Float64, max_goals, max_goals, n_samples)
    for k in 1:n_samples
        mu_h = λ_h[k]
        mu_a = λ_a[k]
        p_h = [pdf(Poisson(mu_h), g) for g in 0:max_goals-1]
        p_a = [pdf(Poisson(mu_a), g) for g in 0:max_goals-1]
        p_h ./= sum(p_h)
        p_a ./= sum(p_a)
        S[:, :, k] = p_h * p_a'
    end
    return Predictions.ScoreMatrix(S)
end

Predictions.compute_score_matrix(model::Union{TeamGoalsPoissonModel, TeamGoalsPoissonOpenPlayModel}, r::DataFrameRow; max_goals::Int = 12) = Predictions.compute_score_matrix(model, Predictions.extract_params(model, r); max_goals=max_goals)

function Predictions.extract_params(model::TeamGoalsRecombIntegratedPoissonModel, row)
    n_s = length(row.λ_h)
    ln_h = hasproperty(row, :lambda_noise_h) ? (row.lambda_noise_h isa AbstractVector ? row.lambda_noise_h : fill(Float64(row.lambda_noise_h), n_s)) : fill(0.136 * 0.768 + 0.0276, n_s)
    ln_a = hasproperty(row, :lambda_noise_a) ? (row.lambda_noise_a isa AbstractVector ? row.lambda_noise_a : fill(Float64(row.lambda_noise_a), n_s)) : fill(0.136 * 0.768 + 0.0276, n_s)
    return (
        λ_h = row.λ_h isa AbstractVector ? row.λ_h : [row.λ_h],
        λ_a = row.λ_a isa AbstractVector ? row.λ_a : [row.λ_a],
        λ_open_h = hasproperty(row, :λ_open_h) ? (row.λ_open_h isa AbstractVector ? row.λ_open_h : [row.λ_open_h]) : row.λ_h,
        λ_open_a = hasproperty(row, :λ_open_a) ? (row.λ_open_a isa AbstractVector ? row.λ_open_a : [row.λ_open_a]) : row.λ_a,
        lambda_noise_h = ln_h,
        lambda_noise_a = ln_a
    )
end

function Predictions.compute_score_matrix(model::TeamGoalsRecombIntegratedPoissonModel, params; max_goals::Int = 12)
    p = params isa DataFrameRow ? Predictions.extract_params(model, params) : params
    λ_open_h = p.λ_open_h
    λ_open_a = p.λ_open_a
    ln_h = p.lambda_noise_h
    ln_a = p.lambda_noise_a
    n_samples = length(λ_open_h)
    
    S = zeros(Float64, max_goals, max_goals, n_samples)
    for k in 1:n_samples
        mu_open_h = λ_open_h[k]
        mu_open_a = λ_open_a[k]
        mu_noise_h = ln_h[k]
        mu_noise_a = ln_a[k]
        
        p_open_h  = [pdf(Poisson(mu_open_h), g) for g in 0:max_goals-1]
        p_noise_h = [pdf(Poisson(mu_noise_h), g) for g in 0:max_goals-1]
        p_open_a  = [pdf(Poisson(mu_open_a), g) for g in 0:max_goals-1]
        p_noise_a = [pdf(Poisson(mu_noise_a), g) for g in 0:max_goals-1]
        
        # Convolve: P(Y_total = g) = sum_{m=0}^g P(Y_open = m) * P(Y_noise = g - m)
        p_tot_h = [sum(p_open_h[m+1] * p_noise_h[g - m + 1] for m in 0:g) for g in 0:max_goals-1]
        p_tot_a = [sum(p_open_a[m+1] * p_noise_a[g - m + 1] for m in 0:g) for g in 0:max_goals-1]
        
        p_tot_h ./= sum(p_tot_h)
        p_tot_a ./= sum(p_tot_a)
        
        S[:, :, k] = p_tot_h * p_tot_a'
    end
    return Predictions.ScoreMatrix(S)
end

Predictions.compute_score_matrix(model::TeamGoalsRecombIntegratedPoissonModel, r::DataFrameRow; max_goals::Int = 12) = Predictions.compute_score_matrix(model, Predictions.extract_params(model, r); max_goals=max_goals)

function PreGame.extract_parameters(model::TeamGoalsRecombIntegratedNegBinModel, chain::Chains, feature_set::Features.FeatureSet)
    d = feature_set.data
    team_map = d[:team_map]
    ref_map  = d[:ref_map]
    n_teams  = d[:n_teams]
    n_refs   = d[:n_refs]
    
    base_mu_samples = vec(Array(chain["base_mu"]))
    ha_samples      = vec(Array(chain["ha_home"]))
    
    log_r_samples        = vec(Array(chain["log_r"]))
    delta_r_home_samples = vec(Array(chain["delta_r_home"]))
    r_a_samples          = exp.(log_r_samples)
    r_h_samples          = exp.(log_r_samples .+ delta_r_home_samples)
    
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
    if n_refs > 0 && _has_param(chain, "raw_gamma_ref[1]")
        raw_gamma_mat = Array(chain[["raw_gamma_ref[$i]" for i in 1:n_refs]])
        gamma_mat = raw_gamma_mat .* sigma_ref_samples
        for (ref_id, idx) in ref_map
            gamma_ref_dict[ref_id] = gamma_mat[:, idx]
        end
    end
    
    alpha_pen_draw_dict = Dict{Int, Vector{Float64}}()
    beta_pen_foul_dict  = Dict{Int, Vector{Float64}}()
    if _has_param(chain, "alpha_pen_draw[1]")
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
        :r_h            => r_h_samples,
        :r_a            => r_a_samples,
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

function PreGame.extract_parameters(
    model::TeamGoalsRecombIntegratedNegBinModel,
    df::AbstractDataFrame,
    feature_set::Features.FeatureSet,
    chain::Chains
)
    data = feature_set.data
    team_map = data[:team_map]
    ref_map  = data[:ref_map]
    n_teams  = data[:n_teams]
    n_refs   = data[:n_refs]
    n_months = data[:n_months]
    n_leagues = data[:n_leagues]
    
    base_mu = vec(Array(chain["base_mu"]))
    ha_home = vec(Array(chain["ha_home"]))
    n_samples = length(base_mu)
    
    log_r        = vec(Array(chain["log_r"]))
    delta_r_home = vec(Array(chain["delta_r_home"]))
    r_a          = exp.(log_r)
    r_h          = exp.(log_r .+ delta_r_home)
    
    raw_alpha_mat = Array(chain[["raw_alpha[$i]" for i in 1:n_teams]])
    raw_beta_mat  = Array(chain[["raw_beta[$i]" for i in 1:n_teams]])
    alpha_mat = raw_alpha_mat .- mean(raw_alpha_mat, dims=2)
    beta_mat  = raw_beta_mat  .- mean(raw_beta_mat, dims=2)
    
    delta_month_mat  = _has_param(chain, "delta_month[1]") ? Array(chain[["delta_month[$i]" for i in 1:n_months]]) : zeros(n_samples, n_months)
    delta_league_mat = _has_param(chain, "delta_league[1]") ? Array(chain[["delta_league[$i]" for i in 1:n_leagues]]) : zeros(n_samples, n_leagues)
    
    pen_base_mu = vec(Array(chain["pen_base_mu"]))
    ha_pen      = vec(Array(chain["ha_pen"]))
    sigma_ref   = vec(Array(chain["sigma_ref"]))
    
    raw_gamma_mat = (n_refs > 0 && _has_param(chain, "raw_gamma_ref[1]")) ? Array(chain[["raw_gamma_ref[$i]" for i in 1:n_refs]]) : zeros(n_samples, n_refs)
    gamma_mat = raw_gamma_mat .* sigma_ref
    
    apd_mat = _has_param(chain, "alpha_pen_draw[1]") ? Array(chain[["alpha_pen_draw[$i]" for i in 1:n_teams]]) : zeros(n_samples, n_teams)
    bpf_mat = _has_param(chain, "beta_pen_foul[1]") ? Array(chain[["beta_pen_foul[$i]" for i in 1:n_teams]]) : zeros(n_samples, n_teams)
    
    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        mid = Int(row.match_id)
        h_id = hasproperty(row, :home_team_id) ? Int(row.home_team_id) : (hasproperty(row, :home_team) ? get(team_map, row.home_team, -1) : -1)
        a_id = hasproperty(row, :away_team_id) ? Int(row.away_team_id) : (hasproperty(row, :away_team) ? get(team_map, row.away_team, -1) : -1)
        
        h_idx = get(team_map, h_id, -1)
        a_idx = get(team_map, a_id, -1)
        
        α_h = h_idx > 0 ? alpha_mat[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? beta_mat[:, h_idx]  : zeros(n_samples)
        α_a = a_idx > 0 ? alpha_mat[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? beta_mat[:, a_idx]  : zeros(n_samples)
        
        m_idx = month(row.match_date)
        l_idx = hasproperty(row, :tournament_id) && row.tournament_id == 57 ? 2 : 1
        
        δ_m = (m_idx >= 1 && m_idx <= n_months) ? delta_month_mat[:, m_idx] : zeros(n_samples)
        δ_l = (l_idx >= 1 && l_idx <= n_leagues) ? delta_league_mat[:, l_idx] : zeros(n_samples)
        
        int_m = base_mu .+ δ_m .+ δ_l
        λ_h = exp.(int_m .+ ha_home .+ α_h .- β_a)
        λ_a = exp.(int_m .+           α_a .- β_h)
        
        # Referee Penalty Intensity
        ref_id = hasproperty(row, :referee_id) && !ismissing(row.referee_id) ? Int(row.referee_id) : -1
        r_idx  = get(ref_map, ref_id, -1)
        γ_ref  = r_idx > 0 ? gamma_mat[:, r_idx] : zeros(n_samples)
        
        apd_h = h_idx > 0 ? apd_mat[:, h_idx] : zeros(n_samples)
        apd_a = a_idx > 0 ? apd_mat[:, a_idx] : zeros(n_samples)
        bpf_h = h_idx > 0 ? bpf_mat[:, h_idx] : zeros(n_samples)
        bpf_a = a_idx > 0 ? bpf_mat[:, a_idx] : zeros(n_samples)
        
        log_pen_h = pen_base_mu .+ ha_pen .+ γ_ref .+ apd_h .+ bpf_a
        log_pen_a = pen_base_mu .- ha_pen .+ γ_ref .+ apd_a .+ bpf_h
        
        lambda_pen_h = exp.(log_pen_h)
        lambda_pen_a = exp.(log_pen_a)
        
        # Noise goal intensity (conversion ~ 0.768 + own goals ~ 0.0276)
        lambda_noise_h = (0.768 .* lambda_pen_h) .+ 0.0276
        lambda_noise_a = (0.768 .* lambda_pen_a) .+ 0.0276
        
        # Total expected goals for scoring rules & portfolio books
        λ_tot_h = λ_h .+ lambda_noise_h
        λ_tot_a = λ_a .+ lambda_noise_a
        
        results[mid] = (;
            λ_h = λ_tot_h,
            λ_a = λ_tot_a,
            r_h = r_h,
            r_a = r_a,
            true_xg_h = λ_tot_h,
            true_xg_a = λ_tot_a,
            λ_open_h = λ_h,
            λ_open_a = λ_a,
            lambda_noise_h = lambda_noise_h,
            lambda_noise_a = lambda_noise_a
        )
    end
    return results
end

function Predictions.extract_params(model::TeamGoalsRecombIntegratedNegBinModel, row)
    n_s = length(row.λ_h)
    ln_h = hasproperty(row, :lambda_noise_h) ? (row.lambda_noise_h isa AbstractVector ? row.lambda_noise_h : fill(Float64(row.lambda_noise_h), n_s)) : fill(0.136 * 0.768 + 0.0276, n_s)
    ln_a = hasproperty(row, :lambda_noise_a) ? (row.lambda_noise_a isa AbstractVector ? row.lambda_noise_a : fill(Float64(row.lambda_noise_a), n_s)) : fill(0.136 * 0.768 + 0.0276, n_s)
    rh = hasproperty(row, :r_h) ? (row.r_h isa AbstractVector ? row.r_h : fill(Float64(row.r_h), n_s)) : fill(20.0, n_s)
    ra = hasproperty(row, :r_a) ? (row.r_a isa AbstractVector ? row.r_a : fill(Float64(row.r_a), n_s)) : fill(10.0, n_s)
    return (
        λ_h = row.λ_h isa AbstractVector ? row.λ_h : [row.λ_h],
        λ_a = row.λ_a isa AbstractVector ? row.λ_a : [row.λ_a],
        λ_open_h = hasproperty(row, :λ_open_h) ? (row.λ_open_h isa AbstractVector ? row.λ_open_h : [row.λ_open_h]) : row.λ_h,
        λ_open_a = hasproperty(row, :λ_open_a) ? (row.λ_open_a isa AbstractVector ? row.λ_open_a : [row.λ_open_a]) : row.λ_a,
        r_h = rh,
        r_a = ra,
        lambda_noise_h = ln_h,
        lambda_noise_a = ln_a
    )
end

function Predictions.compute_score_matrix(model::TeamGoalsRecombIntegratedNegBinModel, params; max_goals::Int = 12)
    p = params isa DataFrameRow ? Predictions.extract_params(model, params) : params
    λ_open_h = p.λ_open_h
    λ_open_a = p.λ_open_a
    r_h = p.r_h
    r_a = p.r_a
    ln_h = p.lambda_noise_h
    ln_a = p.lambda_noise_a
    n_samples = length(λ_open_h)
    
    S = zeros(Float64, max_goals, max_goals, n_samples)
    for k in 1:n_samples
        mu_open_h = λ_open_h[k]
        mu_open_a = λ_open_a[k]
        rk_h = r_h[k]
        rk_a = r_a[k]
        mu_noise_h = ln_h[k]
        mu_noise_a = ln_a[k]
        
        # Probabilities for NegBin: prob = r / (r + mu)
        prob_h = rk_h / (rk_h + mu_open_h)
        prob_a = rk_a / (rk_a + mu_open_a)
        
        d_nb_h = RobustNegativeBinomial(rk_h, prob_h)
        d_nb_a = RobustNegativeBinomial(rk_a, prob_a)
        
        p_open_h  = [pdf(d_nb_h, g) for g in 0:max_goals-1]
        p_open_a  = [pdf(d_nb_a, g) for g in 0:max_goals-1]
        
        p_noise_h = [pdf(Poisson(mu_noise_h), g) for g in 0:max_goals-1]
        p_noise_a = [pdf(Poisson(mu_noise_a), g) for g in 0:max_goals-1]
        
        # Convolve: P(Y_total = g) = sum_{m=0}^g P(Y_open = m) * P(Y_noise = g - m)
        p_tot_h = [sum(p_open_h[m+1] * p_noise_h[g - m + 1] for m in 0:g) for g in 0:max_goals-1]
        p_tot_a = [sum(p_open_a[m+1] * p_noise_a[g - m + 1] for m in 0:g) for g in 0:max_goals-1]
        
        p_tot_h ./= sum(p_tot_h)
        p_tot_a ./= sum(p_tot_a)
        
        S[:, :, k] = p_tot_h * p_tot_a'
    end
    return Predictions.ScoreMatrix(S)
end

Predictions.compute_score_matrix(model::TeamGoalsRecombIntegratedNegBinModel, r::DataFrameRow; max_goals::Int = 12) = Predictions.compute_score_matrix(model, Predictions.extract_params(model, r); max_goals=max_goals)

println("✓ l03_recombination_models.jl loaded (Empirical Bayes + Integrated Turing Engines + Discrete Convolution)")
