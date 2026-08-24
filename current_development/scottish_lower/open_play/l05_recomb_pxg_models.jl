# ==============================================================================
# l05_recomb_pxg_models.jl
#
# Integrated Open-Play Proxy xG (pxG) Recombination + Squad Wealth Engine
#
# Mathematical Architecture:
# 1. Open-Play Tactical Intensity:
#    log μ_open_h = base_μ + δ_month + δ_league + ha_home + α_h - β_a + w_wealth * ΔW
#    log μ_open_a = base_μ + δ_month + δ_league + α_a - β_h - w_wealth * ΔW
#
# 2. Co-Training Multi-Task Likelihoods:
#    A. Continuous Open-Play pxG:
#       pxG_open_h ~ Gamma(ν_xg, μ_open_h / ν_xg)  (masked with binary mask)
#    B. Discrete Realized Open-Play Goals:
#       Y_open_h ~ Poisson(κ_h * μ_open_h)
#    C. Discrete Penalty Awards:
#       N_pen_h ~ Poisson(λ_pen_h)
#       where log λ_pen_h = pen_base_μ + ha_pen + γ_ref + α_pen_draw_h + β_pen_foul_a
#
# 3. Out-of-Sample Score Matrix Recombination:
#    μ_total_h = κ_h * μ_open_h + q_pen * λ_pen_h + λ_og
#    P(H=h, A=a) = [ P(Y_total_h = h) * P(Y_total_a = a) ] * τ_DC(h, a; ρ)
# ==============================================================================

using Turing
using DynamicPPL
using Distributions
using DataFrames
using Dates
using Statistics
using Printf
using LibPQ
using Serialization

using BayesianFootball
const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Predictions = BayesianFootball.Predictions
const Data        = BayesianFootball.Data
const Samplers    = BayesianFootball.Samplers

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l03_recombination_models.jl")
include("l04_recomb_wealth_models.jl")

# ==============================================================================
# SECTION 1: STRUCT DEFINITION
# ==============================================================================

"""
    TeamPxGRecombWealthIntegratedModel <: PreGame.AbstractTimeDecayTeamModel

Integrated Open-Play Proxy xG (pxG) Recombination Model with Starting-XI Squad Wealth.
"""
struct TeamPxGRecombWealthIntegratedModel <: PreGame.AbstractTimeDecayTeamModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    w_wealth_prior::Distribution
    ν_xg_prior::Distribution
    name::String
end

TeamPxGRecombWealthIntegratedModel(; 
    dynamics_config = PreGame.TimeDecayDynamics(days_half_life=365.0),
    w_wealth_prior = truncated(Normal(0.10, 0.05), lower = 0.0),
    ν_xg_prior = truncated(Normal(3.5, 0.5), lower = 0.5),
    name = "recomb_pxg_wealth_integrated"
) = TeamPxGRecombWealthIntegratedModel(dynamics_config, w_wealth_prior, ν_xg_prior, name)

# ==============================================================================
# SECTION 2: FEATURE SET BUILDER
# ==============================================================================

function _build_recomb_pxg_wealth_features(b::Data.SplitBoundary, ds::Data.DataStore, model::TeamPxGRecombWealthIntegratedModel)
    df_clean, df_ref = build_open_play_target_dataset(ds)
    all_refs = unique(filter(x -> x > 0, df_clean.referee_id))
    ref_map = Dict(r => idx for (idx, r) in enumerate(all_refs))
    
    # 1. Starting-XI Wealth Differential
    wealth_map = build_scottish_match_wealth_map(ds)
    
    # 2. Clean Open-Play pxG (Zonal Empirical Bayes excluding penalty shots)
    pxg_df = aggregate_clean_pxg_by_match(ds; k = 25.0)
    pxg_h_map = Dict{Int32, Float64}()
    pxg_a_map = Dict{Int32, Float64}()
    
    if !isempty(pxg_df)
        for r in eachrow(pxg_df)
            m_id = Int32(r.match_id)
            if hasproperty(r, :clean_pxg_h) && isfinite(r.clean_pxg_h) && r.clean_pxg_h > 0.0
                pxg_h_map[m_id] = Float64(r.clean_pxg_h)
            end
            if hasproperty(r, :clean_pxg_a) && isfinite(r.clean_pxg_a) && r.clean_pxg_a > 0.0
                pxg_a_map[m_id] = Float64(r.clean_pxg_a)
            end
        end
    end
    
    # Filter boundary history matches
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
    
    # Referee indexing & masking
    ref_indices = [get(ref_map, r, 0) for r in m.referee_id]
    ref_mask    = Float64.(ref_indices .> 0)
    ref_ids_clamped = [idx > 0 ? idx : 1 for idx in ref_indices]
    
    # Wealth differential
    wealth_diff = Float64[get(wealth_map, mid, 0.0) for mid in m.match_id]
    
    # Open-play pxG vectors & binary masks (impute NaN -> 1.0 for AD stability)
    pxg_h_raw  = [get(pxg_h_map, Int32(mid), NaN) for mid in m.match_id]
    pxg_a_raw  = [get(pxg_a_map, Int32(mid), NaN) for mid in m.match_id]
    
    mask_pxg_h = Float64[isfinite(v) && v > 0.0 ? 1.0 : 0.0 for v in pxg_h_raw]
    mask_pxg_a = Float64[isfinite(v) && v > 0.0 ? 1.0 : 0.0 for v in pxg_a_raw]
    
    pxg_h_safe = Float64[isfinite(v) && v > 0.0 ? v : 1.0 for v in pxg_h_raw]
    pxg_a_safe = Float64[isfinite(v) && v > 0.0 ? v : 1.0 for v in pxg_a_raw]
    
    # Half-life time decay weights
    max_date = maximum(m.match_date)
    date_deltas = [Float64(Dates.value(max_date - d)) for d in m.match_date]
    weights     = 0.5 .^ (date_deltas ./ model.dynamics_config.days_half_life)
    
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
            :pxg_open_h          => pxg_h_safe,
            :pxg_open_a          => pxg_a_safe,
            :mask_pxg_h          => mask_pxg_h,
            :mask_pxg_a          => mask_pxg_a,
            :ref_indices         => ref_ids_clamped,
            :ref_mask            => ref_mask,
            :wealth_diff         => wealth_diff,
            :match_weights       => weights,
            :n_teams             => length(all_teams),
            :n_refs              => max(1, length(all_refs)),
            :n_months            => 12,
            :n_leagues           => 1,
            :team_map            => team_map,
            :ref_map             => ref_map,
            :wealth_map          => wealth_map,
            :clean_df            => df_clean,
            :boundary            => b
        )
    )
end

function Features.create_features(
    splits::Vector{<:Tuple{Data.SplitBoundary, <:Any}},
    ds::Data.DataStore,
    model::TeamPxGRecombWealthIntegratedModel,
    dynamics_col::Symbol = :match_month
)
    raw_vector = [
        (_build_recomb_pxg_wealth_features(boundary, ds, model), meta)
        for (boundary, meta) in splits
    ]
    return Features.FeatureCollection(raw_vector)
end

function Features.create_features(
    boundary::Data.SplitBoundary,
    ds::Data.DataStore,
    model::TeamPxGRecombWealthIntegratedModel,
    dynamics_col::Symbol = :match_month
)
    return _build_recomb_pxg_wealth_features(boundary, ds, model)
end

# ==============================================================================
# SECTION 3: TURING MODEL SPECIFICATION
# ==============================================================================

@model function _turing_recomb_pxg_wealth(
    home_indices::Vector{Int},
    away_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    home_open_goals::Vector{Int},
    away_open_goals::Vector{Int},
    home_pens::Vector{Int},
    away_pens::Vector{Int},
    pxg_open_h::Vector{Float64},
    pxg_open_a::Vector{Float64},
    mask_pxg_h::Vector{Float64},
    mask_pxg_a::Vector{Float64},
    ref_indices::Vector{Int},
    ref_mask::Vector{Float64},
    wealth_diff::Vector{Float64},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_refs::Int,
    n_months::Int,
    n_leagues::Int,
    w_wealth_prior::Distribution,
    ν_xg_prior::Distribution
)
    # 1. Open Play Baseline & Hierarchical Tactical Strengths
    base_mu     ~ Normal(0.15, 0.3)
    ha_home     ~ Normal(0.20, 0.1)
    
    tau_alpha   ~ truncated(Normal(0.0, 0.2), 0.0, Inf)
    tau_beta    ~ truncated(Normal(0.0, 0.2), 0.0, Inf)
    
    raw_alpha   ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_beta    ~ filldist(Normal(0.0, 1.0), n_teams)
    
    alpha       = (raw_alpha .- mean(raw_alpha)) .* tau_alpha
    beta        = (raw_beta  .- mean(raw_beta))  .* tau_beta
    
    delta_month  ~ filldist(Normal(0.0, 0.05), n_months)
    delta_league ~ filldist(Normal(0.0, 0.1), n_leagues)
    
    # 2. Squad Wealth Sensitivity
    w_wealth ~ w_wealth_prior
    w_shift  = w_wealth .* wealth_diff

    # 3. Finishing Efficiency (Kappa) & pxG Precision (nu_xg)
    ν_xg      ~ ν_xg_prior
    raw_kappa ~ filldist(Normal(0.0, 0.10), n_teams)
    kappa     = exp.(raw_kappa)

    # 4. Penalty Sub-Model Priors
    pen_base_mu ~ Normal(-2.0, 0.5)
    ha_pen      ~ Normal(0.19, 0.1)
    
    sigma_ref     ~ Exponential(1.0)
    raw_gamma_ref ~ filldist(Normal(0.0, 1.0), n_refs)
    gamma_ref     = raw_gamma_ref .* sigma_ref
    
    alpha_pen_draw ~ filldist(Normal(0.0, 0.2), n_teams)
    beta_pen_foul  ~ filldist(Normal(0.0, 0.2), n_teams)
    
    # 5. Vectorized Open Play Intensity
    int_m = base_mu .+ view(delta_month, month_indices) .+ view(delta_league, league_indices)
    
    log_mu_h = clamp.(int_m .+ ha_home .+ view(alpha, home_indices) .- view(beta, away_indices) .+ w_shift, -10.0, 10.0)
    log_mu_a = clamp.(int_m .+ view(alpha, away_indices) .- view(beta, home_indices) .- w_shift, -10.0, 10.0)
    
    mu_h = exp.(log_mu_h) .+ 1e-6
    mu_a = exp.(log_mu_a) .+ 1e-6
    
    # 6. Vectorized Penalty Intensity
    log_pen_h = clamp.(pen_base_mu .+ ha_pen .+ view(gamma_ref, ref_indices) .+ view(alpha_pen_draw, home_indices) .+ view(beta_pen_foul, away_indices), -10.0, 5.0)
    log_pen_a = clamp.(pen_base_mu .- ha_pen .+ view(gamma_ref, ref_indices) .+ view(alpha_pen_draw, away_indices) .+ view(beta_pen_foul, home_indices), -10.0, 5.0)
    
    lambda_pen_h = exp.(log_pen_h) .+ 1e-6
    lambda_pen_a = exp.(log_pen_a) .+ 1e-6
    
    # 7. Multi-Task Co-Training Likelihood
    # A. Realized Open-Play Goals (Finishing Kappa * mu_open)
    ll_open_h = logpdf.(Poisson.(view(kappa, home_indices) .* mu_h), home_open_goals)
    ll_open_a = logpdf.(Poisson.(view(kappa, away_indices) .* mu_a), away_open_goals)
    
    # B. Continuous Open-Play pxG Likelihood (Gamma(nu_xg, mu_open / nu_xg))
    ll_pxg_h = logpdf.(Gamma.(ν_xg, mu_h ./ ν_xg), pxg_open_h) .* mask_pxg_h
    ll_pxg_a = logpdf.(Gamma.(ν_xg, mu_a ./ ν_xg), pxg_open_a) .* mask_pxg_a
    
    # C. Realized Penalties Likelihood
    ll_pen_h  = logpdf.(Poisson.(lambda_pen_h), home_pens)
    ll_pen_a  = logpdf.(Poisson.(lambda_pen_a), away_pens)
    
    ll_open_tot = (ll_open_h .+ ll_open_a) .* match_weights
    ll_pxg_tot  = (ll_pxg_h  .+ ll_pxg_a)  .* match_weights
    ll_pen_tot  = (ll_pen_h  .+ ll_pen_a)  .* ref_mask .* match_weights
    
    Turing.@addlogprob! sum(ll_open_tot .+ ll_pxg_tot .+ ll_pen_tot)
end

function PreGame.build_turing_model(
    model::TeamPxGRecombWealthIntegratedModel,
    feature_set
)
    f = feature_set.data
    return _turing_recomb_pxg_wealth(
        f[:home_team_indices],
        f[:away_team_indices],
        f[:month_indices],
        f[:league_indices],
        f[:home_open_goals],
        f[:away_open_goals],
        f[:home_pens],
        f[:away_pens],
        f[:pxg_open_h],
        f[:pxg_open_a],
        f[:mask_pxg_h],
        f[:mask_pxg_a],
        f[:ref_indices],
        f[:ref_mask],
        f[:wealth_diff],
        f[:match_weights],
        f[:n_teams],
        f[:n_refs],
        f[:n_months],
        f[:n_leagues],
        model.w_wealth_prior,
        model.ν_xg_prior
    )
end

# ==============================================================================
# SECTION 4: OUT-OF-SAMPLE PREDICTIONS & SCORE CONVOLUTION
# ==============================================================================

function PreGame.extract_parameters(
    model::TeamPxGRecombWealthIntegratedModel,
    df::AbstractDataFrame,
    feature_set,
    chain::Chains
)
    data = feature_set.data
    team_map   = data[:team_map]
    ref_map    = data[:ref_map]
    wealth_map = data[:wealth_map]
    n_teams    = data[:n_teams]
    n_refs     = data[:n_refs]
    n_months   = data[:n_months]
    n_leagues  = data[:n_leagues]
    
    base_mu  = vec(Array(chain["base_mu"]))
    ha_home  = vec(Array(chain["ha_home"]))
    w_wealth = vec(Array(chain["w_wealth"]))
    n_samples = length(base_mu)
    
    effects = _tau_scaled_team_effects(chain, n_teams; context="TeamPxGRecombWealthIntegratedModel extractor")
    alpha_mat, beta_mat = effects.alpha, effects.beta
    
    raw_kappa_mat = _has_param(chain, "raw_kappa[1]") ? Array(chain[["raw_kappa[$i]" for i in 1:n_teams]]) : zeros(n_samples, n_teams)
    kappa_mat = exp.(raw_kappa_mat)
    
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
        
        κ_h = h_idx > 0 ? kappa_mat[:, h_idx] : ones(n_samples)
        κ_a = a_idx > 0 ? kappa_mat[:, a_idx] : ones(n_samples)
        
        dw = get(wealth_map, mid, 0.0)
        w_shift = w_wealth .* dw
        
        m_idx = month(row.match_date)
        l_idx = hasproperty(row, :tournament_id) && row.tournament_id == 57 ? 2 : 1
        
        δ_m = (m_idx >= 1 && m_idx <= n_months) ? delta_month_mat[:, m_idx] : zeros(n_samples)
        δ_l = (l_idx >= 1 && l_idx <= n_leagues) ? delta_league_mat[:, l_idx] : zeros(n_samples)
        
        int_m = base_mu .+ δ_m .+ δ_l
        λ_h = exp.(int_m .+ ha_home .+ α_h .- β_a .+ w_shift)
        λ_a = exp.(int_m .+           α_a .- β_h .- w_shift)
        
        # Finishing kappa applied to open-play goal rate
        mu_open_h = κ_h .* λ_h
        mu_open_a = κ_a .* λ_a
        
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
        
        lambda_total_h = mu_open_h .+ lambda_noise_h
        lambda_total_a = mu_open_a .+ lambda_noise_a
        
        results[mid] = (;
            λ_h = lambda_total_h,
            λ_a = lambda_total_a,
            r_h = fill(100.0, n_samples),
            r_a = fill(100.0, n_samples),
            true_xg_h = λ_h,
            true_xg_a = λ_a,
            lambda_pen_h = lambda_pen_h,
            lambda_pen_a = lambda_pen_a,
            lambda_open_h = mu_open_h,
            lambda_open_a = mu_open_a
        )
    end
    return results
end

function Predictions.extract_params(
    model::TeamPxGRecombWealthIntegratedModel,
    feature_set::Features.FeatureSet,
    chain::Chains
)
    F = feature_set.data
    b = F[:boundary]
    df_clean = F[:clean_df]
    target_matches = filter(r -> r.match_id in b.target_match_ids, df_clean)
    
    team_map   = F[:team_map]
    ref_map    = F[:ref_map]
    wealth_map = F[:wealth_map]
    n_teams    = F[:n_teams]
    n_refs     = F[:n_refs]
    n_months   = F[:n_months]
    n_leagues  = F[:n_leagues]
    
    base_mu  = vec(Array(chain["base_mu"]))
    ha_home  = vec(Array(chain["ha_home"]))
    w_wealth = vec(Array(chain["w_wealth"]))
    n_samples = length(base_mu)
    
    effects = _tau_scaled_team_effects(chain, n_teams; context="TeamPxGRecombWealthIntegratedModel extractor")
    alpha_mat, beta_mat = effects.alpha, effects.beta
    
    raw_kappa_mat = _has_param(chain, "raw_kappa[1]") ? Array(chain[["raw_kappa[$i]" for i in 1:n_teams]]) : zeros(n_samples, n_teams)
    kappa_mat = exp.(raw_kappa_mat)
    
    delta_month_mat  = _has_param(chain, "delta_month[1]") ? Array(chain[["delta_month[$i]" for i in 1:n_months]]) : zeros(n_samples, n_months)
    delta_league_mat = _has_param(chain, "delta_league[1]") ? Array(chain[["delta_league[$i]" for i in 1:n_leagues]]) : zeros(n_samples, n_leagues)
    
    pen_base_mu = vec(Array(chain["pen_base_mu"]))
    ha_pen      = vec(Array(chain["ha_pen"]))
    sigma_ref   = vec(Array(chain["sigma_ref"]))
    
    raw_gamma_mat = (n_refs > 0 && _has_param(chain, "raw_gamma_ref[1]")) ? Array(chain[["raw_gamma_ref[$i]" for i in 1:n_refs]]) : zeros(n_samples, n_refs)
    gamma_mat = raw_gamma_mat .* sigma_ref
    
    apd_mat = _has_param(chain, "alpha_pen_draw[1]") ? Array(chain[["alpha_pen_draw[$i]" for i in 1:n_teams]]) : zeros(n_samples, n_teams)
    bpf_mat = _has_param(chain, "beta_pen_foul[1]") ? Array(chain[["beta_pen_foul[$i]" for i in 1:n_teams]]) : zeros(n_samples, n_teams)
    
    out_df = DataFrame(
        match_id            = Int[],
        mu_open_h_samples   = Vector{Float64}[],
        mu_open_a_samples   = Vector{Float64}[],
        lambda_pen_h_samples= Vector{Float64}[],
        lambda_pen_a_samples= Vector{Float64}[],
        q_pen_samples       = Vector{Float64}[],
        rho_samples         = Vector{Float64}[]
    )
    
    for row in eachrow(target_matches)
        m_id = row.match_id
        h_id = row.home_team_id
        a_id = row.away_team_id
        r_id = row.referee_id
        
        h_idx = get(team_map, h_id, -1)
        a_idx = get(team_map, a_id, -1)
        r_idx = get(ref_map, r_id, -1)
        
        α_h = h_idx > 0 ? alpha_mat[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? beta_mat[:, h_idx]  : zeros(n_samples)
        α_a = a_idx > 0 ? alpha_mat[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? beta_mat[:, a_idx]  : zeros(n_samples)
        
        κ_h = h_idx > 0 ? kappa_mat[:, h_idx] : ones(n_samples)
        κ_a = a_idx > 0 ? kappa_mat[:, a_idx] : ones(n_samples)
        
        dw = get(wealth_map, m_id, 0.0)
        w_shift = w_wealth .* dw
        
        m_idx = month(row.match_date)
        l_idx = hasproperty(row, :tournament_id) && row.tournament_id == 57 ? 2 : 1
        
        δ_m = (m_idx >= 1 && m_idx <= n_months) ? delta_month_mat[:, m_idx] : zeros(n_samples)
        δ_l = (l_idx >= 1 && l_idx <= n_leagues) ? delta_league_mat[:, l_idx] : zeros(n_samples)
        
        int_m = base_mu .+ δ_m .+ δ_l
        λ_h = exp.(int_m .+ ha_home .+ α_h .- β_a .+ w_shift)
        λ_a = exp.(int_m .+           α_a .- β_h .- w_shift)
        
        mu_open_h = κ_h .* λ_h
        mu_open_a = κ_a .* λ_a
        
        # Penalty intensities
        γ_ref = r_idx > 0 ? gamma_mat[:, r_idx] : zeros(n_samples)
        
        apd_h = h_idx > 0 ? apd_mat[:, h_idx] : zeros(n_samples)
        apd_a = a_idx > 0 ? apd_mat[:, a_idx] : zeros(n_samples)
        bpf_h = h_idx > 0 ? bpf_mat[:, h_idx] : zeros(n_samples)
        bpf_a = a_idx > 0 ? bpf_mat[:, a_idx] : zeros(n_samples)
        
        lambda_pen_h = exp.(pen_base_mu .+ ha_pen .+ γ_ref .+ apd_h .+ bpf_a)
        lambda_pen_a = exp.(pen_base_mu .- ha_pen .+ γ_ref .+ apd_a .+ bpf_h)
        
        q_pen_samples = fill(0.768, n_samples)
        rho_samples   = fill(-0.05, n_samples)
        
        push!(out_df, (
            match_id             = m_id,
            mu_open_h_samples    = mu_open_h,
            mu_open_a_samples    = mu_open_a,
            lambda_pen_h_samples = lambda_pen_h,
            lambda_pen_a_samples = lambda_pen_a,
            q_pen_samples        = q_pen_samples,
            rho_samples          = rho_samples
        ))
    end
    
    return Predictions.LatentStates(out_df, model)
end

function Predictions.extract_params(model::TeamPxGRecombWealthIntegratedModel, row::DataFrameRow)
    ln_h = hasproperty(row, :lambda_noise_h) ? (row.lambda_noise_h isa AbstractVector ? row.lambda_noise_h : [row.lambda_noise_h]) : (hasproperty(row, :lambda_pen_h) ? (0.768 .* (row.lambda_pen_h isa AbstractVector ? row.lambda_pen_h : [row.lambda_pen_h])) .+ 0.0276 : fill(0.10, length(row.λ_h)))
    ln_a = hasproperty(row, :lambda_noise_a) ? (row.lambda_noise_a isa AbstractVector ? row.lambda_noise_a : [row.lambda_noise_a]) : (hasproperty(row, :lambda_pen_a) ? (0.768 .* (row.lambda_pen_a isa AbstractVector ? row.lambda_pen_a : [row.lambda_pen_a])) .+ 0.0276 : fill(0.10, length(row.λ_a)))
    
    return (
        λ_open_h = hasproperty(row, :λ_open_h) ? (row.λ_open_h isa AbstractVector ? row.λ_open_h : [row.λ_open_h]) : (hasproperty(row, :lambda_open_h) ? (row.lambda_open_h isa AbstractVector ? row.lambda_open_h : [row.lambda_open_h]) : row.λ_h),
        λ_open_a = hasproperty(row, :λ_open_a) ? (row.λ_open_a isa AbstractVector ? row.λ_open_a : [row.λ_open_a]) : (hasproperty(row, :lambda_open_a) ? (row.lambda_open_a isa AbstractVector ? row.lambda_open_a : [row.lambda_open_a]) : row.λ_a),
        lambda_noise_h = ln_h,
        lambda_noise_a = ln_a
    )
end

function Predictions.compute_score_matrix(model::TeamPxGRecombWealthIntegratedModel, params; max_goals::Int = 12)
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
        
        p_open_h  = [pdf(Poisson(max(1e-4, mu_open_h)), g) for g in 0:max_goals-1]
        p_noise_h = [pdf(Poisson(max(1e-4, mu_noise_h)), g) for g in 0:max_goals-1]
        p_open_a  = [pdf(Poisson(max(1e-4, mu_open_a)), g) for g in 0:max_goals-1]
        p_noise_a = [pdf(Poisson(max(1e-4, mu_noise_a)), g) for g in 0:max_goals-1]
        
        # Discrete Poisson Convolution
        p_tot_h = [sum(p_open_h[m+1] * p_noise_h[g - m + 1] for m in 0:g) for g in 0:max_goals-1]
        p_tot_a = [sum(p_open_a[m+1] * p_noise_a[g - m + 1] for m in 0:g) for g in 0:max_goals-1]
        
        p_tot_h ./= sum(p_tot_h)
        p_tot_a ./= sum(p_tot_a)
        
        S[:, :, k] = p_tot_h * p_tot_a'
    end
    return Predictions.ScoreMatrix(S)
end

Predictions.compute_score_matrix(model::TeamPxGRecombWealthIntegratedModel, r::DataFrameRow; max_goals::Int = 12) = Predictions.compute_score_matrix(model, Predictions.extract_params(model, r); max_goals=max_goals)

println("✓ l05_recomb_pxg_models.jl loaded (Open-Play Proxy xG + Squad Wealth + Officiating Submodel)")
