# current_development/kappa_def/l01_kappa_def_models.jl
#
# LOADER (temporary module). One self-contained model with a κ-structure switch:
#   KappaDefDoublePoissonModel — double-Poisson {goals + xG + outfield ratings}, market OFF.
#
#   kappa_mode = :attack_only  V0 control — the exact current HierarchicalTeamKappa
#                              (softplus multiplicative, attack-indexed):
#                              λ_h = κ[home]·exp(log_λ_h)
#   kappa_mode = :net          V2 Dixon-Coles-style net conversion strength (n params):
#                              λ_h = exp(log_λ_h + κ0 + τ_net·(δc[home] − δc[away]))
#   kappa_mode = :attdef       V1 full Maher split on the conversion residual (2n params):
#                              λ_h = exp(log_λ_h + κ0 + τ_att·zc_att[home] − τ_def·zc_def[away])
#
# κ enters the GOALS pillar only; the xG pillar uses the raw exp(log_λ), so κ is the
# xG→goals conversion residual and κ0 is identified by the goals/xG contrast.
# δ/z vectors are CENTRED in-model (smooth, AD-safe) — kills the mean flat-direction
# (:net) / the κ0 confound (:attdef). Non-centred τ·z parameterization throughout.
#
# Dispatch: this struct is not in the src score Unions ⇒ explicit extract_params /
# compute_score_matrix overrides below (Poisson route, same as l02 in split_market_pillar).

using Turing
using Distributions
using DataFrames
using Statistics
using MCMCChains

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct KappaDefDoublePoissonModel{
    I<:PreGame.AbstractInterceptionConfig,
    P<:PreGame.OutfieldPlayerDynamicsConfig,
    D<:PreGame.AbstractDispersionConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    K<:PreGame.AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig
  } <: PreGame.AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P
      dispersion_config::D            # carried for pipeline-compat; unused (Poisson)
      homeadvantage_config::H
      kappa_config::K                 # used ONLY when kappa_mode == :attack_only (V0)
      player_ratings_feature::R
      ν_xg::Distribution   = truncated(Normal(3.0, 0.5), lower=0.5)
      kappa_mode::Symbol   = :attack_only     # :attack_only | :net | :attdef
      κ0_prior::Distribution = Normal(0.0, 0.15)                     # global log-conversion
      τ_prior::Distribution  = truncated(Normal(0.0, 0.10), lower=0.0)  # team-spread scale(s)
end

_centre(v) = v .- mean(v)

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_kappa_def_engine(
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
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::KappaDefDoublePoissonModel
)
    # ---- components ----
    ν_xg  ~ config.ν_xg
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))

    # ---- κ structure (static branch on config — data-side, AD-safe) ----
    local logκ_h, logκ_a
    if config.kappa_mode === :attack_only
        kap ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
        logκ_h = log.(view(kap, home_team_indices))
        logκ_a = log.(view(kap, away_team_indices))
    elseif config.kappa_mode === :net
        κ0    ~ config.κ0_prior
        τ_net ~ config.τ_prior
        δ_net ~ filldist(Normal(0, 1), n_teams)
        d = τ_net .* _centre(δ_net)
        logκ_h = κ0 .+ view(d, home_team_indices) .- view(d, away_team_indices)
        logκ_a = κ0 .+ view(d, away_team_indices) .- view(d, home_team_indices)
    else  # :attdef
        κ0    ~ config.κ0_prior
        τ_att ~ config.τ_prior
        τ_def ~ config.τ_prior
        z_att ~ filldist(Normal(0, 1), n_teams)
        z_def ~ filldist(Normal(0, 1), n_teams)
        av = τ_att .* _centre(z_att)
        dv = τ_def .* _centre(z_def)
        logκ_h = κ0 .+ view(av, home_team_indices) .- view(dv, away_team_indices)
        logκ_a = κ0 .+ view(av, away_team_indices) .- view(dv, home_team_indices)
    end

    # ---- ratings → rates (identical to the src no-market engine) ----
    base_rating = config.player_ratings_feature.tracker.prior_mean
    h_G_c = home_G_ratings .- base_rating
    h_O_c = (home_D_ratings .+ home_M_ratings .+ home_F_ratings) .- (10.0 * base_rating)
    a_G_c = away_G_ratings .- base_rating
    a_O_c = (away_D_ratings .+ away_M_ratings .+ away_F_ratings) .- (10.0 * base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)

    log_λ_h = clamp.(int_m .+ view(ha, home_team_indices) .+ att_h .+ def_a, -20.0, 20.0)
    log_λ_a = clamp.(int_m                                .+ att_a .+ def_h, -20.0, 20.0)

    # ---- goals pillar (κ applied on log scale) ----
    λ_h = exp.(clamp.(log_λ_h .+ logκ_h, -20.0, 20.0)) .+ 1e-6
    λ_a = exp.(clamp.(log_λ_a .+ logκ_a, -20.0, 20.0)) .+ 1e-6

    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # ---- xG pillar (raw rate; sanitized independently — NaN would reach Gamma before -Inf) ----
    xg_rate_h = exp.(log_λ_h) .+ 1e-6
    xg_rate_a = exp.(log_λ_a) .+ 1e-6
    xg_rate_h = ifelse.(isnan.(xg_rate_h) .| isinf.(xg_rate_h), one.(xg_rate_h), xg_rate_h)
    xg_rate_a = ifelse.(isnan.(xg_rate_a) .| isinf.(xg_rate_a), one.(xg_rate_a), xg_rate_a)

    ll_xg_h = logpdf.(Gamma.(ν_xg, xg_rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, xg_rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::KappaDefDoublePoissonModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.XGFeature(),
        model.player_ratings_feature,
        Features.TimeIndicesFeature()
    ]
end

function _unpack_xg_kd(data)
    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    away_xg_raw = coalesce.(data[:flat_away_xg], NaN)
    xg_mask = Float64.(.!isnan.(home_xg_raw) .& .!isnan.(away_xg_raw))
    home_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in home_xg_raw]
    away_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in away_xg_raw]
    return home_xg, away_xg, xg_mask
end

function PreGame.build_turing_model(config::KappaDefDoublePoissonModel, feature_set)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12

    date_deltas   = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.player_dynamics_config.days_half_life)

    home_xg, away_xg, xg_mask = _unpack_xg_kd(data)

    return build_kappa_def_engine(
        Vector{Int}(data[:flat_home_ids]), Vector{Int}(data[:flat_away_ids]),
        Vector{Int}(data[:season_indices]), Vector{Int}(data[:flat_months]),
        Vector{Int}(data[:flat_home_goals]), Vector{Int}(data[:flat_away_goals]),
        match_weights,
        Vector{Float64}(data[:flat_home_G_rating]), Vector{Float64}(data[:flat_home_D_rating]),
        Vector{Float64}(data[:flat_home_M_rating]), Vector{Float64}(data[:flat_home_F_rating]),
        Vector{Float64}(data[:flat_away_G_rating]), Vector{Float64}(data[:flat_away_D_rating]),
        Vector{Float64}(data[:flat_away_M_rating]), Vector{Float64}(data[:flat_away_F_rating]),
        home_xg, away_xg, xg_mask,
        n_teams, n_seasons, n_months, config
    )
end

# ==========================================
# 4. κ RECONSTRUCTION FROM CHAINS (shared by extractor + runner diagnostics)
# ==========================================
_vecp(chain, s) = vec(Array(chain[Symbol(s)]))

"per-sample per-team log-κ CONTRIBUTION matrices (att, def): logκ_h = κ0 + att[:,h] − def[:,a]"
function kappa_logmats(model::KappaDefDoublePoissonModel, chain, n_teams::Int)
    ns = size(chain, 1) * size(chain, 3)
    if model.kappa_mode === :attack_only
        κm = PreGame.extract_kappa(chain, model.kappa_config, n_teams)     # multiplicative
        return (κ0 = zeros(ns), att = log.(κm), def = zeros(ns, n_teams))
    elseif model.kappa_mode === :net
        κ0 = _vecp(chain, "κ0"); τ = _vecp(chain, "τ_net")
        δ  = reduce(hcat, (_vecp(chain, "δ_net[$i]") for i in 1:n_teams))
        δc = (δ .- mean(δ, dims=2)) .* τ
        return (κ0 = κ0, att = δc, def = δc)          # net: same vector both sides
    else
        κ0 = _vecp(chain, "κ0")
        τa = _vecp(chain, "τ_att"); τd = _vecp(chain, "τ_def")
        za = reduce(hcat, (_vecp(chain, "z_att[$i]") for i in 1:n_teams))
        zd = reduce(hcat, (_vecp(chain, "z_def[$i]") for i in 1:n_teams))
        return (κ0 = κ0,
                att = (za .- mean(za, dims=2)) .* τa,
                def = (zd .- mean(zd, dims=2)) .* τd)
    end
end

"runner-facing per-team summary: posterior mean/std of the att/def MULTIPLIERS exp(±contrib)"
function kappa_team_summary(model::KappaDefDoublePoissonModel, chain, n_teams::Int;
                            team_names::Vector{String}=String[])
    km = kappa_logmats(model, chain, n_teams)
    names_ = isempty(team_names) ? ["team_$i" for i in 1:n_teams] : team_names
    att_m = exp.(km.att); def_m = exp.(km.def)
    df = DataFrame(team = names_,
                   att_mult = round.(vec(mean(att_m, dims=1)), digits=4),
                   att_sd   = round.(vec(std(att_m, dims=1)), digits=4),
                   def_mult = round.(vec(mean(def_m, dims=1)), digits=4),
                   def_sd   = round.(vec(std(def_m, dims=1)), digits=4))
    globals = (κ0_conv = round(mean(exp.(km.κ0)), digits=4),
               att_spread = round(maximum(df.att_mult) - minimum(df.att_mult), digits=4),
               def_spread = round(maximum(df.def_mult) - minimum(df.def_mult), digits=4),
               attdef_cor = model.kappa_mode === :attdef ?
                            round(cor(df.att_mult, df.def_mult), digits=3) : NaN)
    return df, globals
end

# ==========================================
# 5. THE EXTRACTOR (for PPD / eval, mirrors the src no-market extractor)
# ==========================================
function PreGame.extract_parameters(model::KappaDefDoublePoissonModel, df, feature_set, chain)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams])
    team_map    = data[:team_map]
    ratings_map = data[:player_ratings_map]

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)
    km       = kappa_logmats(model, chain, n_teams)

    n_samples = size(chain, 1) * size(chain, 3)
    ρ_vec = zeros(n_samples)
    base_r = model.player_ratings_feature.tracker.prior_mean

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        m_ratings = get(ratings_map, Int(row.match_id), Dict())
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
        logκ_h = km.κ0 .+ (h_id > 0 ? km.att[:, h_id] : zeros(n_samples)) .-
                          (a_id > 0 ? km.def[:, a_id] : zeros(n_samples))
        logκ_a = km.κ0 .+ (a_id > 0 ? km.att[:, a_id] : zeros(n_samples)) .-
                          (h_id > 0 ? km.def[:, h_id] : zeros(n_samples))

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = hasproperty(row, :month_idx) ? Int(row.month_idx) : 1
        μ_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        att_a .+ def_h, -20.0, 20.0)

        λ_h = exp.(clamp.(log_λ_h .+ logκ_h, -20.0, 20.0)) .+ 1e-6
        λ_a = exp.(clamp.(log_λ_a .+ logκ_a, -20.0, 20.0)) .+ 1e-6

        results[Int(row.match_id)] = (;
            λ_h, λ_a,
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = ρ_vec, ρ = ρ_vec,
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
        )
    end
    return results
end

# ==========================================
# 6. PREDICTION OVERRIDES (Poisson route; struct not in src Unions)
# ==========================================
function _poisson_score_kd(λ_h, λ_a; max_goals::Int=12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        @. p_h = pdf(Poisson(λ_h[k]), goals)
        @. p_a = pdf(Poisson(λ_a[k]), goals)
        for j in 1:max_goals, i in 1:max_goals
            S[i, j, k] = p_h[i] * p_a[j]
        end
    end
    return Pred.ScoreMatrix(S)
end

Pred.extract_params(::KappaDefDoublePoissonModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.compute_score_matrix(::KappaDefDoublePoissonModel, params; max_goals::Int=12) =
    _poisson_score_kd(params.λ_h, params.λ_a; max_goals)

println("[l01] kappa_def loader ready: KappaDefDoublePoissonModel " *
        "{goals + xG + outfield, market OFF}; kappa_mode ∈ (:attack_only, :net, :attdef)")
