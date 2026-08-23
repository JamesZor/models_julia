# current_development/scottish_lower/corners/l05_corner_recomb_pipeline.jl
#
# LOADER: 4-Way Goal & Corner Recombination Framework for Walk-Forward Evaluation
#
# Decomposes total match goals into:
#   y_goals = y_open_play + y_penalties + y_own_goals + y_corner_goals
#
# Implements:
# 1. TeamGoalsCornerRecombIntegratedModel <: AbstractTimeDecayTeamModel
# 2. Vectorized Turing Bayesian Model with Robust Negative Binomial corner count generation
# 3. Discrete Poisson Convolution Score Matrix Generator
# 4. Latent extraction and out-of-sample prediction hooks for Experiments.run_experiment

using Turing
using DynamicPPL
using Distributions
using LinearAlgebra
using Statistics
using DataFrames
using Dates
using SpecialFunctions: loggamma
using StatsFuns: logistic, log1pexp

using BayesianFootball.MyDistributions: RobustNegativeBinomial

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Predictions = BayesianFootball.Predictions
const Data        = BayesianFootball.Data

include("l01_corner_data.jl")

# ==============================================================================
# 1. MODEL STRUCT DEFINITIONS
# ==============================================================================

struct TeamGoalsCornerRecombIntegratedModel <: PreGame.AbstractTimeDecayTeamModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    interception_config::PreGame.AbstractInterceptionConfig
    homeadvantage_config::PreGame.AbstractHomeAdvantageConfig
    name::String
end

function TeamGoalsCornerRecombIntegratedModel(;
    dynamics_config      = PreGame.TimeDecayDynamics(days_half_life = 365.0),
    interception_config  = PreGame.GlobalInterception(),
    homeadvantage_config = PreGame.GlobalHomeAdvantage(),
    name                 = "recomb_corner_integrated"
)
    return TeamGoalsCornerRecombIntegratedModel(
        dynamics_config,
        interception_config,
        homeadvantage_config,
        name
    )
end

# ==============================================================================
# 2. FEATURE EXTRACTION PIPELINE
# ==============================================================================

const _CORNER_CACHE = Dict{UInt64, DataFrame}()

function get_cached_corner_clean_df()
    k = UInt64(101)
    if haskey(_CORNER_CACHE, k)
        return _CORNER_CACHE[k]
    end
    df = fetch_scottish_corner_dataset()
    _CORNER_CACHE[k] = df
    return df
end

function _build_corner_recomb_features(b::Data.SplitBoundary, ds::Data.DataStore, model::TeamGoalsCornerRecombIntegratedModel)
    df_all = get_cached_corner_clean_df()
    m = filter(r -> r.match_id in b.history_match_ids, df_all)
    sort!(m, :match_datetime)

    all_teams = sort(unique(vcat(m.home_team, m.away_team)))
    team_map  = Dict(t => idx for (idx, t) in enumerate(all_teams))
    n_teams   = length(all_teams)

    h_idx = [team_map[t] for t in m.home_team]
    a_idx = [team_map[t] for t in m.away_team]

    y_op_h = Vector{Int}(m.open_goals_h)
    y_op_a = Vector{Int}(m.open_goals_a)

    corners_h = Vector{Int}(m.corners_h)
    corners_a = Vector{Int}(m.corners_a)

    corner_goals_h = Vector{Int}(m.corner_goals_h)
    corner_goals_a = Vector{Int}(m.corner_goals_a)

    # Precompute log-combinations for binomial likelihood
    log_binom_h = Float64[loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
                          for (n, k) in zip(corners_h, corner_goals_h)]
    log_binom_a = Float64[loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
                          for (n, k) in zip(corners_a, corner_goals_a)]

    max_date = maximum(m.match_date)
    date_deltas = [Float64(Dates.value(max_date - d)) for d in m.match_date]
    weights     = 0.5 .^ (date_deltas ./ model.dynamics_config.days_half_life)

    month_indices  = month.(m.match_date)
    league_indices = [r.tournament_id == 57 ? 2 : 1 for r in eachrow(m)]

    return Features.FeatureSet(
        Dict{Symbol, Any}(
            :home_team_indices   => h_idx,
            :away_team_indices   => a_idx,
            :month_indices       => month_indices,
            :league_indices      => league_indices,
            :y_op_h              => y_op_h,
            :y_op_a              => y_op_a,
            :corners_h           => corners_h,
            :corners_a           => corners_a,
            :corner_goals_h      => corner_goals_h,
            :corner_goals_a      => corner_goals_a,
            :log_binom_h         => log_binom_h,
            :log_binom_a         => log_binom_a,
            :match_weights       => weights,
            :n_teams             => n_teams,
            :n_months            => 12,
            :n_leagues           => 2,
            :team_map            => team_map,
            :boundary            => b
        )
    )
end

function Features.create_features(
    splits::Vector{<:Tuple{Data.SplitBoundary, <:Any}},
    ds::Data.DataStore,
    model::TeamGoalsCornerRecombIntegratedModel,
    dynamics_col::Symbol = :match_month
)
    raw_vector = [
        (_build_corner_recomb_features(boundary, ds, model), meta)
        for (boundary, meta) in splits
    ]
    return Features.FeatureCollection(raw_vector)
end

function Features.create_features(
    boundary::Data.SplitBoundary,
    ds::Data.DataStore,
    model::TeamGoalsCornerRecombIntegratedModel,
    dynamics_col::Symbol = :match_month
)
    return _build_corner_recomb_features(boundary, ds, model)
end

# ==============================================================================
# 3. VECTORIZED TURING MASTER ENGINE
# ==============================================================================

@model function _turing_corner_recomb_integrated(
    home_indices::Vector{Int},
    away_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    y_op_h::Vector{Int},
    y_op_a::Vector{Int},
    corners_h::Vector{Int},
    corners_a::Vector{Int},
    corner_goals_h::Vector{Int},
    corner_goals_a::Vector{Int},
    log_binom_h::Vector{Float64},
    log_binom_a::Vector{Float64},
    match_weights::Vector{Float64},
    n_teams::Int,
    config::TeamGoalsCornerRecombIntegratedModel
)
    # -------------------------------------------------------------
    # 1. Open-Play Tactical Goals Submodel
    # -------------------------------------------------------------
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, 1, 12))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))

    ha_val = view(ha, home_indices)
    int_m  = inter.μ_base[1] .+ inter.δ_month[month_indices]

    att_h = view(dyn.α, home_indices)
    def_h = view(dyn.β, home_indices)
    att_a = view(dyn.α, away_indices)
    def_a = view(dyn.β, away_indices)

    log_μ_op_h = int_m .+ ha_val .+ att_h .- def_a
    log_μ_op_a = int_m .+           att_a .- def_h

    ll_op_h = (y_op_h .* log_μ_op_h) .- exp.(log_μ_op_h)
    ll_op_a = (y_op_a .* log_μ_op_a) .- exp.(log_μ_op_a)
    @addlogprob! sum((ll_op_h .+ ll_op_a) .* match_weights)

    # -------------------------------------------------------------
    # 2. Corner Count Generation Submodel (Robust Negative Binomial)
    # -------------------------------------------------------------
    μ_c_base ~ Normal(1.45, 0.20)
    γ_ha_c   ~ Normal(0.13, 0.05)
    log_ϕ_c  ~ Normal(2.0, 0.5)
    ϕ_c      = exp(clamp(log_ϕ_c, -1.0, 4.0))

    α_c_raw ~ filldist(Normal(0, 0.25), n_teams)
    β_c_raw ~ filldist(Normal(0, 0.25), n_teams)
    α_c     = α_c_raw .- mean(α_c_raw)
    β_c     = β_c_raw .- mean(β_c_raw)

    att_c_h = view(α_c, home_indices)
    def_c_h = view(β_c, home_indices)
    att_c_a = view(α_c, away_indices)
    def_c_a = view(β_c, away_indices)

    log_λ_c_h = μ_c_base .+ γ_ha_c .+ att_c_h .- def_c_a
    log_λ_c_a = μ_c_base .+           att_c_a .- def_c_h

    λ_c_h = exp.(log_λ_c_h)
    λ_c_a = exp.(log_λ_c_a)

    dist_c_h = RobustNegativeBinomial.(ϕ_c, λ_c_h)
    dist_c_a = RobustNegativeBinomial.(ϕ_c, λ_c_a)

    ll_c_h = logpdf.(dist_c_h, corners_h)
    ll_c_a = logpdf.(dist_c_a, corners_a)
    @addlogprob! sum((ll_c_h .+ ll_c_a) .* match_weights)

    # -------------------------------------------------------------
    # 3. Corner Goal Conversion Submodel (Zero-Allocation Logit-Binomial)
    # -------------------------------------------------------------
    σ_conv_att ~ truncated(Normal(0.10, 0.08), 0.0, 0.5)
    σ_conv_def ~ truncated(Normal(0.10, 0.08), 0.0, 0.5)

    z_att_raw ~ filldist(Normal(0, 1), n_teams)
    z_def_raw ~ filldist(Normal(0, 1), n_teams)
    z_att     = z_att_raw .- mean(z_att_raw)
    z_def     = z_def_raw .- mean(z_def_raw)

    z_att_h = view(z_att, home_indices)
    z_def_h = view(z_def, home_indices)
    z_att_a = view(z_att, away_indices)
    def_z_a = view(z_def, away_indices)

    logit_q_h = -3.23 .+ (σ_conv_att .* z_att_h) .- (σ_conv_def .* def_z_a)
    logit_q_a = -3.23 .+ (σ_conv_att .* z_att_a) .- (σ_conv_def .* z_def_h)

    ll_conv_h = log_binom_h .+ (corner_goals_h .* logit_q_h) .- (corners_h .* log1pexp.(logit_q_h))
    ll_conv_a = log_binom_a .+ (corner_goals_a .* logit_q_a) .- (corners_a .* log1pexp.(logit_q_a))
    @addlogprob! sum((ll_conv_h .+ ll_conv_a) .* match_weights)

    return nothing
end

function PreGame.build_turing_model(model::TeamGoalsCornerRecombIntegratedModel, features::Features.FeatureSet)
    d = features.data
    return _turing_corner_recomb_integrated(
        d[:home_team_indices],
        d[:away_team_indices],
        d[:month_indices],
        d[:league_indices],
        d[:y_op_h],
        d[:y_op_a],
        d[:corners_h],
        d[:corners_a],
        d[:corner_goals_h],
        d[:corner_goals_a],
        d[:log_binom_h],
        d[:log_binom_a],
        d[:match_weights],
        d[:n_teams],
        model
    )
end

# ==============================================================================
# 4. LATENT STATE EXTRACTION & SCORE MATRIX COMPUTATION
# ==============================================================================

function _has_param(chain::Chains, p::String)
    return Symbol(p) in names(chain, :parameters)
end

function PreGame.extract_parameters(
    model::TeamGoalsCornerRecombIntegratedModel,
    df::AbstractDataFrame,
    feature_set,
    chain::Chains
)
    features = feature_set isa Tuple ? feature_set[1] : feature_set
    n_samples = size(chain, 1) * size(chain, 3)
    d = features.data
    team_map = d[:team_map]
    n_teams  = d[:n_teams]

    # Open Play Latents
    base_mu = _has_param(chain, "inter.μ") ? vec(Array(chain["inter.μ"])) : (_has_param(chain, "inter.μ_base[1]") ? vec(Array(chain["inter.μ_base[1]"])) : zeros(n_samples))
    ha_val  = _has_param(chain, "ha.γ_global") ? vec(Array(chain["ha.γ_global"])) : zeros(n_samples)

    sigma_a = vec(Array(chain["dyn.σ_a"]))
    sigma_d = vec(Array(chain["dyn.σ_d"]))

    raw_a = Array(chain[["dyn.raw_a[$i]" for i in 1:n_teams]])
    raw_d = Array(chain[["dyn.raw_d[$i]" for i in 1:n_teams]])

    scaled_a = raw_a .* sigma_a
    scaled_d = raw_d .* sigma_d
    alpha_mat = scaled_a .- mean(scaled_a, dims=2)
    beta_mat  = scaled_d .- mean(scaled_d, dims=2)

    # Corner Generation Latents
    mu_c_base = vec(Array(chain["μ_c_base"]))
    gamma_ha_c = vec(Array(chain["γ_ha_c"]))

    raw_ac = Array(chain[["α_c_raw[$i]" for i in 1:n_teams]])
    raw_bc = Array(chain[["β_c_raw[$i]" for i in 1:n_teams]])
    alpha_c_mat = raw_ac .- mean(raw_ac, dims=2)
    beta_c_mat  = raw_bc .- mean(raw_bc, dims=2)

    # Corner Conversion Latents
    sigma_conv_att = vec(Array(chain["σ_conv_att"]))
    sigma_conv_def = vec(Array(chain["σ_conv_def"]))

    raw_za = Array(chain[["z_att_raw[$i]" for i in 1:n_teams]])
    raw_zd = Array(chain[["z_def_raw[$i]" for i in 1:n_teams]])
    za_mat = raw_za .- mean(raw_za, dims=2)
    zd_mat = raw_zd .- mean(raw_zd, dims=2)

    delta_month_mat = _has_param(chain, "inter.δ_month[1]") ?
        Array(chain[["inter.δ_month[$i]" for i in 1:12]]) : zeros(n_samples, 12)

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        mid = Int(row.match_id)
        h_name = string(row.home_team)
        a_name = string(row.away_team)

        h_idx = get(team_map, h_name, -1)
        a_idx = get(team_map, a_name, -1)

        α_h = h_idx > 0 ? alpha_mat[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? beta_mat[:, h_idx]  : zeros(n_samples)
        α_a = a_idx > 0 ? alpha_mat[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? beta_mat[:, a_idx]  : zeros(n_samples)

        m_idx = month(row.match_date)
        δ_m = (m_idx >= 1 && m_idx <= 12) ? delta_month_mat[:, m_idx] : zeros(n_samples)

        # 1. Open Play Intensity
        int_m = base_mu .+ δ_m
        λ_op_h = exp.(int_m .+ ha_val .+ α_h .- β_a)
        λ_op_a = exp.(int_m .+           α_a .- β_h)

        # 2. Corner Intensity
        α_c_h = h_idx > 0 ? alpha_c_mat[:, h_idx] : zeros(n_samples)
        β_c_h = h_idx > 0 ? beta_c_mat[:, h_idx]  : zeros(n_samples)
        α_c_a = a_idx > 0 ? alpha_c_mat[:, a_idx] : zeros(n_samples)
        β_c_a = a_idx > 0 ? beta_c_mat[:, a_idx]  : zeros(n_samples)

        λ_c_h = exp.(mu_c_base .+ gamma_ha_c .+ α_c_h .- β_c_a)
        λ_c_a = exp.(mu_c_base .+               α_c_a .- β_c_h)

        # 3. Corner Goal Conversion
        z_att_h = h_idx > 0 ? za_mat[:, h_idx] : zeros(n_samples)
        z_def_h = h_idx > 0 ? zd_mat[:, h_idx] : zeros(n_samples)
        z_att_a = a_idx > 0 ? za_mat[:, a_idx] : zeros(n_samples)
        z_def_a = a_idx > 0 ? zd_mat[:, a_idx] : zeros(n_samples)

        logit_q_h = -3.23 .+ (sigma_conv_att .* z_att_h) .- (sigma_conv_def .* z_def_a)
        logit_q_a = -3.23 .+ (sigma_conv_att .* z_att_a) .- (sigma_conv_def .* z_def_h)
        q_c_h     = logistic.(logit_q_h)
        q_c_a     = logistic.(logit_q_a)

        # 4. Corner Goals & Noise Rates
        μ_cg_h = q_c_h .* λ_c_h
        μ_cg_a = q_c_a .* λ_c_a

        μ_noise_h = fill(0.78 * 0.219 + 0.063, n_samples)
        μ_noise_a = fill(0.78 * 0.219 + 0.063, n_samples)

        # Total Goals
        λ_tot_h = λ_op_h .+ μ_cg_h .+ μ_noise_h
        λ_tot_a = λ_op_a .+ μ_cg_a .+ μ_noise_a

        results[mid] = (;
            λ_h = λ_tot_h,
            λ_a = λ_tot_a,
            r_h = fill(100.0, n_samples),
            r_a = fill(100.0, n_samples),
            true_xg_h = λ_tot_h,
            true_xg_a = λ_tot_a,
            λ_op_h = λ_op_h,
            λ_op_a = λ_op_a,
            λ_c_h = λ_c_h,
            λ_c_a = λ_c_a,
            q_c_h = q_c_h,
            q_c_a = q_c_a
        )
    end
    return results
end

# --- Score Matrix & Prediction Overloads ---
function Predictions.extract_params(model::TeamGoalsCornerRecombIntegratedModel, row)
    return (
        λ_h = row.λ_h isa AbstractVector ? row.λ_h : [row.λ_h],
        λ_a = row.λ_a isa AbstractVector ? row.λ_a : [row.λ_a],
        λ_op_h = hasproperty(row, :λ_op_h) ? (row.λ_op_h isa AbstractVector ? row.λ_op_h : [row.λ_op_h]) : [row.λ_h],
        λ_op_a = hasproperty(row, :λ_op_a) ? (row.λ_op_a isa AbstractVector ? row.λ_op_a : [row.λ_op_a]) : [row.λ_a],
        λ_c_h  = hasproperty(row, :λ_c_h) ? (row.λ_c_h isa AbstractVector ? row.λ_c_h : [row.λ_c_h]) : [5.0],
        λ_c_a  = hasproperty(row, :λ_c_a) ? (row.λ_c_a isa AbstractVector ? row.λ_c_a : [row.λ_c_a]) : [4.5],
        q_c_h  = hasproperty(row, :q_c_h) ? (row.q_c_h isa AbstractVector ? row.q_c_h : [row.q_c_h]) : [0.038],
        q_c_a  = hasproperty(row, :q_c_a) ? (row.q_c_a isa AbstractVector ? row.q_c_a : [row.q_c_a]) : [0.038]
    )
end

function _poisson_pmf_vec(λ::Float64, max_g::Int)
    p = zeros(Float64, max_g + 1)
    p[1] = exp(-λ)
    for k in 1:max_g
        p[k + 1] = p[k] * λ / k
    end
    return p
end

function _convolve_pmfs(p1::Vector{Float64}, p2::Vector{Float64}, max_g::Int)
    p_out = zeros(Float64, max_g + 1)
    for i in 0:max_g
        for j in 0:(max_g - i)
            p_out[i + j + 1] += p1[i + 1] * p2[j + 1]
        end
    end
    return p_out
end

function Predictions.compute_score_matrix(model::TeamGoalsCornerRecombIntegratedModel, params; max_goals::Int = 12)
    p = params isa DataFrameRow ? Predictions.extract_params(model, params) : params

    n_samples = length(p.λ_h)
    dim = max_goals + 1
    S = zeros(Float64, dim, dim, n_samples)

    μ_pen = 0.78 * 0.219 # 0.1708
    μ_og  = 0.0630
    p_noise = _poisson_pmf_vec(μ_pen + μ_og, max_goals)

    for k in 1:n_samples
        μ_op_h = p.λ_op_h[k]
        μ_op_a = p.λ_op_a[k]
        λ_c_h  = p.λ_c_h[k]
        λ_c_a  = p.λ_c_a[k]
        q_c_h  = p.q_c_h[k]
        q_c_a  = p.q_c_a[k]

        # 1. PMFs for Open Play
        p_op_h = _poisson_pmf_vec(μ_op_h, max_goals)
        p_op_a = _poisson_pmf_vec(μ_op_a, max_goals)

        # 2. PMFs for Corner Goals
        μ_cg_h = q_c_h * λ_c_h
        μ_cg_a = q_c_a * λ_c_a
        p_cg_h = _poisson_pmf_vec(μ_cg_h, max_goals)
        p_cg_a = _poisson_pmf_vec(μ_cg_a, max_goals)

        # 3. Exact Discrete Poisson Convolution
        p_tot_h = _convolve_pmfs(_convolve_pmfs(p_op_h, p_noise, max_goals), p_cg_h, max_goals)
        p_tot_a = _convolve_pmfs(_convolve_pmfs(p_op_a, p_noise, max_goals), p_cg_a, max_goals)

        # Normalize marginals
        p_tot_h ./= sum(p_tot_h)
        p_tot_a ./= sum(p_tot_a)

        # 2D Joint Score Slice
        for j in 1:dim
            pj = p_tot_a[j]
            for i in 1:dim
                S[i, j, k] = p_tot_h[i] * pj
            end
        end
    end

    return Predictions.ScoreMatrix(S)
end

Predictions.compute_score_matrix(model::TeamGoalsCornerRecombIntegratedModel, r::DataFrameRow; max_goals::Int = 12) =
    Predictions.compute_score_matrix(model, Predictions.extract_params(model, r); max_goals=max_goals)
