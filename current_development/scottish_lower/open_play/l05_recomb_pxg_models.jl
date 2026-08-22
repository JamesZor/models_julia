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
# SECTION 4: POSTERIOR PREDICTION & SCORE MATRIX RECOMBINATION
# ==============================================================================

function Predictions.extract_params(
    model::TeamPxGRecombWealthIntegratedModel,
    chain::MCMCChains.Chains,
    feature_set
)
    f = feature_set.data
    team_map = f[:team_map]
    ref_map  = f[:ref_map]
    wealth_map = f[:wealth_map]
    
    return Dict(
        :chain      => chain,
        :team_map   => team_map,
        :ref_map    => ref_map,
        :wealth_map => wealth_map,
        :model      => model
    )
end

function Predictions.extract_params(
    model::TeamPxGRecombWealthIntegratedModel,
    row::DataFrameRow
)
    return row.params
end

function Predictions.compute_score_matrix(
    model::TeamPxGRecombWealthIntegratedModel,
    params::Dict;
    max_goals::Int = 12,
    ρ::Float64 = -0.05
)
    chain      = params[:chain]
    team_map   = params[:team_map]
    ref_map    = params[:ref_map]
    wealth_map = params[:wealth_map]
    
    h_tid = get(params, :home_team_id, 0)
    a_tid = get(params, :away_team_id, 0)
    m_id  = get(params, :match_id, 0)
    r_id  = get(params, :referee_id, 0)
    
    h_idx = get(team_map, h_tid, 0)
    a_idx = get(team_map, a_tid, 0)
    r_idx = get(ref_map, r_id, 0)
    
    delta_w = get(wealth_map, Int(m_id), 0.0)
    
    # Extract posterior draws
    base_mu = vec(Array(chain[:base_mu]))
    ha_home = vec(Array(chain[:ha_home]))
    w_w     = vec(Array(chain[:w_wealth]))
    n_draws = length(base_mu)
    
    alpha_h = h_idx > 0 && haskey(chain, Symbol("raw_alpha[$h_idx]")) ? vec(Array(chain[Symbol("raw_alpha[$h_idx]")])) : zeros(n_draws)
    alpha_a = a_idx > 0 && haskey(chain, Symbol("raw_alpha[$a_idx]")) ? vec(Array(chain[Symbol("raw_alpha[$a_idx]")])) : zeros(n_draws)
    beta_h  = h_idx > 0 && haskey(chain, Symbol("raw_beta[$h_idx]"))  ? vec(Array(chain[Symbol("raw_beta[$h_idx]")]))  : zeros(n_draws)
    beta_a  = a_idx > 0 && haskey(chain, Symbol("raw_beta[$a_idx]"))  ? vec(Array(chain[Symbol("raw_beta[$a_idx]")]))  : zeros(n_draws)
    
    tau_a = vec(Array(chain[:tau_alpha]))
    tau_b = vec(Array(chain[:tau_beta]))
    
    # Finishing kappa
    raw_kap_h = h_idx > 0 && haskey(chain, Symbol("raw_kappa[$h_idx]")) ? vec(Array(chain[Symbol("raw_kappa[$h_idx]")])) : zeros(n_draws)
    raw_kap_a = a_idx > 0 && haskey(chain, Symbol("raw_kappa[$a_idx]")) ? vec(Array(chain[Symbol("raw_kappa[$a_idx]")])) : zeros(n_draws)
    kappa_h = exp.(raw_kap_h)
    kappa_a = exp.(raw_kap_a)
    
    # Penalty submodel
    pen_base_mu = vec(Array(chain[:pen_base_mu]))
    ha_pen      = vec(Array(chain[:ha_pen]))
    sigma_ref   = vec(Array(chain[:sigma_ref]))
    gamma_r     = r_idx > 0 && haskey(chain, Symbol("raw_gamma_ref[$r_idx]")) ? vec(Array(chain[Symbol("raw_gamma_ref[$r_idx]")])) .* sigma_ref : zeros(n_draws)
    
    alpha_pen_h = h_idx > 0 && haskey(chain, Symbol("alpha_pen_draw[$h_idx]")) ? vec(Array(chain[Symbol("alpha_pen_draw[$h_idx]")])) : zeros(n_draws)
    alpha_pen_a = a_idx > 0 && haskey(chain, Symbol("alpha_pen_draw[$a_idx]")) ? vec(Array(chain[Symbol("alpha_pen_draw[$a_idx]")])) : zeros(n_draws)
    beta_pen_h  = h_idx > 0 && haskey(chain, Symbol("beta_pen_foul[$h_idx]"))  ? vec(Array(chain[Symbol("beta_pen_foul[$h_idx]")]))  : zeros(n_draws)
    beta_pen_a  = a_idx > 0 && haskey(chain, Symbol("beta_pen_foul[$a_idx]"))  ? vec(Array(chain[Symbol("beta_pen_foul[$a_idx]")]))  : zeros(n_draws)
    
    # Latent open-play & penalty intensities
    log_mu_open_h = clamp.(base_mu .+ ha_home .+ (alpha_h .* tau_a) .- (beta_a .* tau_b) .+ (w_w .* delta_w), -10.0, 10.0)
    log_mu_open_a = clamp.(base_mu .+            (alpha_a .* tau_a) .- (beta_h .* tau_b) .- (w_w .* delta_w), -10.0, 10.0)
    
    mu_open_h = exp.(log_mu_open_h)
    mu_open_a = exp.(log_mu_open_a)
    
    log_pen_h = clamp.(pen_base_mu .+ ha_pen .+ gamma_r .+ alpha_pen_h .+ beta_pen_a, -10.0, 5.0)
    log_pen_a = clamp.(pen_base_mu .- ha_pen .+ gamma_r .+ alpha_pen_a .+ beta_pen_h, -10.0, 5.0)
    
    lambda_pen_h = exp.(log_pen_h)
    lambda_pen_a = exp.(log_pen_a)
    
    # Analytical total rates via Poisson convolution
    # mu_total = kappa * mu_open + q_pen * lambda_pen + lambda_og
    q_pen = 0.768
    lambda_og = 0.0276
    
    mu_tot_h = (kappa_h .* mu_open_h) .+ (q_pen .* lambda_pen_h) .+ lambda_og
    mu_tot_a = (kappa_a .* mu_open_a) .+ (q_pen .* lambda_pen_a) .+ lambda_og
    
    # Construct 3D score matrix (max_goals x max_goals x n_draws)
    mat = zeros(Float64, max_goals, max_goals, n_draws)
    
    @inbounds for s in 1:n_draws
        lh = mu_tot_h[s]
        la = mu_tot_a[s]
        
        # Marginal Poisson probabilities
        p_h = [pdf(Poisson(lh), g) for g in 0:(max_goals - 1)]
        p_a = [pdf(Poisson(la), g) for g in 0:(max_goals - 1)]
        
        # Outer product with Dixon-Coles adjustment
        for h in 1:max_goals, a in 1:max_goals
            gh = h - 1
            ga = a - 1
            prob = p_h[h] * p_a[a]
            
            # Dixon-Coles low score adjustment
            if gh == 0 && ga == 0
                prob *= (1.0 - lh * la * ρ)
            elseif gh == 0 && ga == 1
                prob *= (1.0 + lh * ρ)
            elseif gh == 1 && ga == 0
                prob *= (1.0 + la * ρ)
            elseif gh == 1 && ga == 1
                prob *= (1.0 - ρ)
            end
            mat[h, a, s] = max(0.0, prob)
        end
        
        # Normalize sample slice
        tot = sum(view(mat, :, :, s))
        if tot > 0.0
            mat[:, :, s] ./= tot
        end
    end
    
    return Predictions.ScoreMatrix(mat)
end

println("✓ l05_recomb_pxg_models.jl loaded (Open-Play Proxy xG + Squad Wealth + Officiating Submodel)")
