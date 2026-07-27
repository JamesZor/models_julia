# current_development/split_market_pillar/l09_hier_iso_poisson.jl
#
# LOADER (temporary module). Single self-contained model:
#   HierIsoDoublePoissonModel — the ISOTROPIC market-pillar double-Poisson (src
#   DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel), but with a HIERARCHICAL market σ.
#
# The src iso model anchors BOTH home & away log-rates to the market-implied log-λ with ONE sampled
# scalar σ_market:  ll = logpdf(Normal(log_λ_h+log κ_h, σ_market), market_log_λ_h) + (away) , scaled by
# market_weight. This model keeps that pillar but makes σ per-match / per-side & HIERARCHICAL
# (NON-CENTERED, to avoid Neal's funnel). Iso has NO strikes (it is not the smile), so the only natural
# grouping is per-TEAM (does the market price some teams' rate more reliably than others?), plus an
# optional global home/away anchor-tightness offset δ_side:
#
#   log σ_h(m) = log_σ_base + per_team·(τ_team · z_team[home_m]) + per_side·(+δ_side)
#   log σ_a(m) = log_σ_base + per_team·(τ_team · z_team[away_m]) + per_side·(−δ_side)
#   σ_{h,a}    = clamp(exp(·), 0.01, 5.0)
#
# z_team ~ N(0,1)^n_teams (same latent whether the team is home/away — "team T is priced to within σ_T"),
# τ_team half-normal hyper-scale, δ_side ~ N(0,0.25), log_σ_base ~ N(log .1, .5) (centred on src σ_market).
# per_team / per_side are 0/1 FLAGS gating each term's CONTRIBUTION; the latents are ALWAYS sampled
# (graph stays static → AD-safe, no in-@model branching) but a gated-off term just draws from its prior.
#
# PURPOSE: feasibility / convergence smoke test of learning the iso market-anchor σ hierarchically, the
# iso analogue of l08's hierarchical smile σ (see [[hierarchical-smile-sigma-null]]). Watch τ_team / δ_side
# and R-hat: τ→0 = no heterogeneity (collapses to the global scalar σ = src iso); τ large + bad R-hat =
# funnel (tighten the τ prior). PREDICTION is identical to the src iso model (the σ hierarchy is a
# training-only regulariser on λ), so it is a plain double-Poisson outer-product score.
#
# STANDALONE — does NOT depend on l03/l08. Include on its own:
#   include("current_development/split_market_pillar/l09_hier_iso_poisson.jl")

using Turing
using Distributions
using DataFrames

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

# market-inversion sanity (mirror the src builder); identical def if l03 already loaded
_iso_mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct HierIsoDoublePoissonModel{
    I<:PreGame.AbstractInterceptionConfig,
    P<:PreGame.OutfieldPlayerDynamicsConfig,
    D<:PreGame.AbstractDispersionConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    K<:PreGame.AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig
  } <: PreGame.AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P
      dispersion_config::D            # config-compat; unused by the Poisson likelihood
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5)
      # --- hierarchical market-σ priors ---
      log_sigma_base_prior::Distribution = Normal(log(0.1), 0.5)   # global anchor scale (centred on src σ_market≈0.1)
      tau_team_prior::Distribution       = truncated(Normal(0.0, 0.3), lower=0.0)   # per-team spread
      delta_side_prior::Distribution     = Normal(0.0, 0.25)        # home/away anchor-tightness offset
      sigma_per_team::Bool               = true     # include δ[team]
      sigma_per_side::Bool               = false    # include ±δ_side (home tighter/looser than away)
      market_on::Bool                    = true
      market_weight::Float64             = 1.0
end

_iso_market_active(config) = config.market_on ? 1.0 : 0.0

# ==========================================
# 2. THE TURING ENGINE  (= src iso double-Poisson + hierarchical market σ)
# ==========================================
@model function build_hier_iso_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    home_G_ratings::Vector{Float64}, home_D_ratings::Vector{Float64},
    home_M_ratings::Vector{Float64}, home_F_ratings::Vector{Float64},
    away_G_ratings::Vector{Float64}, away_D_ratings::Vector{Float64},
    away_M_ratings::Vector{Float64}, away_F_ratings::Vector{Float64},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    xg_mask::Vector{Float64},
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_mask::Vector{Float64},
    market_active::Float64,
    market_weight::Float64,
    per_team_on::Float64,
    per_side_on::Float64,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::HierIsoDoublePoissonModel
)
    # --- 1. LOAD COMPONENTS ---
    ν_xg ~ config.ν_xg

    # hierarchical market-σ hyperparameters (non-centered)
    log_σ_base ~ config.log_sigma_base_prior
    τ_team     ~ config.tau_team_prior
    δ_side     ~ config.delta_side_prior
    z_team     ~ filldist(Normal(0.0, 1.0), n_teams)

    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))

    # --- 2. VECTORIZED INDEXING & MATH ---
    base_rating = config.player_ratings_feature.tracker.prior_mean
    h_G_c = home_G_ratings .- base_rating
    h_O_c = (home_D_ratings .+ home_M_ratings .+ home_F_ratings) .- (10.0 * base_rating)
    a_G_c = away_G_ratings .- base_rating
    a_O_c = (away_D_ratings .+ away_M_ratings .+ away_F_ratings) .- (10.0 * base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    int_m    = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    home_adv = view(ha, home_team_indices)
    log_λ_h = clamp.(int_m .+ att_h .+ def_a .+ home_adv, -20.0, 20.0)
    log_λ_a = clamp.(int_m .+ att_a .+ def_h, -20.0, 20.0)

    kap_h = view(kap, home_team_indices)
    kap_a = view(kap, away_team_indices)
    λ_h = kap_h .* exp.(log_λ_h) .+ 1e-6
    λ_a = kap_a .* exp.(log_λ_a) .+ 1e-6

    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- Pillar B: Actual Goals (Poisson) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) ---
    xg_rate_h = exp.(log_λ_h) .+ 1e-6
    xg_rate_a = exp.(log_λ_a) .+ 1e-6
    xg_rate_h = ifelse.(isnan.(xg_rate_h) .| isinf.(xg_rate_h), one.(xg_rate_h), xg_rate_h)
    xg_rate_a = ifelse.(isnan.(xg_rate_a) .| isinf.(xg_rate_a), one.(xg_rate_a), xg_rate_a)
    ll_xg_h = logpdf.(Gamma.(ν_xg, xg_rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, xg_rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C: ISOTROPIC MARKET (Normal), σ HIERARCHICAL per-team / per-side ---
    # log σ_h = log_σ_base + per_team·τ_team·z_team[h] + per_side·(+δ_side)
    # log σ_a = log_σ_base + per_team·τ_team·z_team[a] + per_side·(−δ_side)
    z_h = view(z_team, home_team_indices)
    z_a = view(z_team, away_team_indices)
    log_σ_h = log_σ_base .+ per_team_on .* (τ_team .* z_h) .+ per_side_on .* δ_side
    log_σ_a = log_σ_base .+ per_team_on .* (τ_team .* z_a) .- per_side_on .* δ_side
    σ_h = clamp.(exp.(log_σ_h), 0.01, 5.0)
    σ_a = clamp.(exp.(log_σ_a), 0.01, 5.0)

    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
    ll_market_h = logpdf.(Normal.(market_rate_h, σ_h), market_log_λ_h)
    ll_market_a = logpdf.(Normal.(market_rate_a, σ_a), market_log_λ_a)
    Turing.@addlogprob! market_active * market_weight *
        sum((ll_market_h .+ ll_market_a) .* match_weights .* market_mask)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::HierIsoDoublePoissonModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.XGFeature(),
        model.market_feature_config,
        model.player_ratings_feature,
        Features.TimeIndicesFeature()
    ]
end

function PreGame.build_turing_model(config::HierIsoDoublePoissonModel, feature_set)
    data = feature_set.data

    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12

    date_deltas   = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.player_dynamics_config.days_half_life)

    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    month_idx  = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    hG = Vector{Float64}(data[:flat_home_G_rating]); hD = Vector{Float64}(data[:flat_home_D_rating])
    hM = Vector{Float64}(data[:flat_home_M_rating]); hF = Vector{Float64}(data[:flat_home_F_rating])
    aG = Vector{Float64}(data[:flat_away_G_rating]); aD = Vector{Float64}(data[:flat_away_D_rating])
    aM = Vector{Float64}(data[:flat_away_M_rating]); aF = Vector{Float64}(data[:flat_away_F_rating])

    # xG (Gamma support x>0): mask requires BOTH sides present; floor present 0.0 to ε
    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    away_xg_raw = coalesce.(data[:flat_away_xg], NaN)
    xg_mask = Float64.(.!isnan.(home_xg_raw) .& .!isnan.(away_xg_raw))
    home_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in home_xg_raw]
    away_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in away_xg_raw]

    # market pillar: both implied rates present & in a plausible range
    market_mask  = Float64.(_iso_mok.(data[:flat_market_λ_home]) .& _iso_mok.(data[:flat_market_λ_away]))
    market_log_h = [_iso_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_home]]
    market_log_a = [_iso_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_away]]

    return build_hier_iso_engine(
        home_ids, away_ids, season_ids, month_idx,
        home_goals, away_goals, match_weights,
        hG, hD, hM, hF, aG, aD, aM, aF,
        home_xg, away_xg, xg_mask,
        market_log_h, market_log_a, market_mask,
        _iso_market_active(config), config.market_weight,
        config.sigma_per_team ? 1.0 : 0.0,
        config.sigma_per_side ? 1.0 : 0.0,
        n_teams, n_seasons, n_months, config
    )
end

# ==========================================
# 4. THE EXTRACTOR  (prediction-identical to src iso: λ_h, λ_a)
# ==========================================
function PreGame.extract_parameters(model::HierIsoDoublePoissonModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams])
    team_map    = data[:team_map]
    ratings_map = data[:player_ratings_map]

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3)
    base_r = model.player_ratings_feature.tracker.prior_mean
    ρ_vec  = zeros(n_samples)

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        mid  = Int(row.match_id)
        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        m_ratings = get(ratings_map, mid, Dict())
        h_G = get(m_ratings, ("home","G"), 0.0); h_D = get(m_ratings, ("home","D"), 0.0)
        h_M = get(m_ratings, ("home","M"), 0.0); h_F = get(m_ratings, ("home","F"), 0.0)
        a_G = get(m_ratings, ("away","G"), 0.0); a_D = get(m_ratings, ("away","D"), 0.0)
        a_M = get(m_ratings, ("away","M"), 0.0); a_F = get(m_ratings, ("away","F"), 0.0)

        h_G_c = h_G - base_r; h_O_c = (h_D + h_M + h_F) - (10.0 * base_r)
        a_G_c = a_G - base_r; a_O_c = (a_D + a_M + a_F) - (10.0 * base_r)

        att_h = (p_dyn_nt.w_G_att .* h_G_c) .+ (p_dyn_nt.w_Outfield_att .* h_O_c)
        def_h = (p_dyn_nt.w_G_def .* h_G_c) .+ (p_dyn_nt.w_Outfield_def .* h_O_c)
        att_a = (p_dyn_nt.w_G_att .* a_G_c) .+ (p_dyn_nt.w_Outfield_att .* a_O_c)
        def_a = (p_dyn_nt.w_G_def .* a_G_c) .+ (p_dyn_nt.w_Outfield_def .* a_O_c)

        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)
        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = hasproperty(row, :month_idx) ? Int(row.month_idx) : 1
        μ_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        att_a .+ def_h, -20.0, 20.0)

        λ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        results[mid] = (;
            λ_h, λ_a,
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = ρ_vec, ρ = ρ_vec,
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
        )
    end
    return results
end

# ==========================================
# 5. PREDICTION — plain double-Poisson outer product (σ hierarchy is training-only)
#    (own methods so we don't have to touch the src AbstractDoublePoissonPlayerModels Union)
# ==========================================
Pred.extract_params(::HierIsoDoublePoissonModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)

function Pred.compute_score_matrix(::HierIsoDoublePoissonModel, params; max_goals::Int=12)
    λ_h, λ_a = params.λ_h, params.λ_a
    n_samples = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n_samples)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals - 1)
    @inbounds for k in 1:n_samples
        d_h = Poisson(λ_h[k]); d_a = Poisson(λ_a[k])
        @. p_h = pdf(d_h, goals)
        @. p_a = pdf(d_a, goals)
        for j in 1:max_goals
            pj = p_a[j]
            for i in 1:max_goals
                S[i, j, k] = p_h[i] * pj
            end
        end
    end
    return Pred.ScoreMatrix(S)
end

println("[l09] hierarchical-σ iso loader ready: HierIsoDoublePoissonModel " *
        "{src iso double-Poisson market pillar + hierarchical market σ = log_σ_base + per_team·δ[team] + per_side·±δ_side}; " *
        "flags sigma_per_team / sigma_per_side; knob market_weight. STANDALONE.")
