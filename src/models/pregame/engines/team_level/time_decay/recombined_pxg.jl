# src/models/pregame/engines/team_level/time_decay/recombined_pxg.jl
#
# Master Engine: Integrated Open-Play Proxy xG (pxG) Recombination + Squad Wealth Engine

using Turing
using DynamicPPL
using Distributions
using DataFrames
using Dates
using Statistics
using MCMCChains

Base.@kwdef struct DynamicPxGRecombModel{
    I<:AbstractInterceptionConfig,
    T<:AbstractDynamicsConfig,
    H<:AbstractHomeAdvantageConfig,
    W<:AbstractSquadWealthConfig,
    X<:AbstractPxGObservationConfig,
    R<:AbstractRecombinationConfig
} <: AbstractTimeDecayTeamModel
    interception_config::I   = HierarchicalMonthlyInterception()
    dynamics_config::T       = TimeDecayDynamics(days_half_life = 365.0)
    homeadvantage_config::H  = GlobalHomeAdvantage()
    wealth_config::W         = LinearSquadWealthConfig()
    pxg_config::X            = GammaPxGObservationConfig()
    recomb_config::R         = HierarchicalOfficiatingConfig()
    name::String             = "dynamic_pxg_recomb"
end

function Features.required_features(model::DynamicPxGRecombModel)
    req = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.OpenPlayGoalsFeature(),
        Features.OpenPlayPxGFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.LeagueFeature(),
        Features.TimeIndicesFeature()
    ]
    if !(model.wealth_config isa NoSquadWealthConfig)
        push!(req, Features.SquadWealthFeature())
    end
    if model.recomb_config isa HierarchicalOfficiatingConfig
        push!(req, Features.RefereeOfficiatingFeature())
    end
    return req
end

# Turing @model with multi-task pxG + Open-Play Goals + Penalty Awards
@model function build_recombined_pxg_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    ref_indices::Vector{Int},
    delta_wealth::Vector{Float64},
    pxg_open_h::Vector{Float64},
    pxg_open_a::Vector{Float64},
    mask_pxg_h::Vector{Float64},
    mask_pxg_a::Vector{Float64},
    y_open_h::Vector{Int},
    y_open_a::Vector{Int},
    n_pen_h::Vector{Int},
    n_pen_a::Vector{Int},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_leagues::Int,
    n_refs::Int,
    config::DynamicPxGRecombModel
)
    # 1. Base Components
    inter ~ to_submodel(build_interception(config.interception_config, 1, 12))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # League fixed effects (sum-to-zero)
    δ_league_raw ~ filldist(Normal(0, 0.2), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)

    # 2. Starting-XI Wealth Shift
    w_shift ~ to_submodel(build_squad_wealth(config.wealth_config, delta_wealth))

    # 3. Vectorized Open-Play Tactical Intensity
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)

    base_mu = inter.μ_base[1]
    month_eff = inter.δ_month[month_indices]
    league_eff = δ_league[league_indices]
    ha_eff = view(ha, home_team_indices)

    log_μ_open_h = base_mu .+ month_eff .+ league_eff .+ ha_eff .+ att_h .- def_a .+ w_shift
    log_μ_open_a = base_mu .+ month_eff .+ league_eff .+           att_a .- def_h .- w_shift

    μ_open_h = exp.(clamp.(log_μ_open_h, -5.0, 4.0))
    μ_open_a = exp.(clamp.(log_μ_open_a, -5.0, 4.0))

    # 4. Continuous pxG Likelihood (Gamma with binary mask)
    if config.pxg_config isa GammaPxGObservationConfig
        ν_xg ~ config.pxg_config.ν_xg_prior
        scale_pxg_h = μ_open_h ./ ν_xg
        scale_pxg_a = μ_open_a ./ ν_xg

        ll_pxg_h = logpdf.(Gamma.(ν_xg, scale_pxg_h), pxg_open_h)
        ll_pxg_a = logpdf.(Gamma.(ν_xg, scale_pxg_a), pxg_open_a)

        Turing.@addlogprob! sum(ll_pxg_h .* mask_pxg_h .* match_weights)
        Turing.@addlogprob! sum(ll_pxg_a .* mask_pxg_a .* match_weights)
    end

    # 5. Team Finishing Factors & Realized Open-Play Goals
    log_κ_raw ~ filldist(Normal(0, 0.10), n_teams)
    log_κ = log_κ_raw .- mean(log_κ_raw)
    κ_team = exp.(clamp.(log_κ, -0.50, 0.50))

    κ_h = view(κ_team, home_team_indices)
    κ_a = view(κ_team, away_team_indices)

    μ_goal_open_h = μ_open_h .* κ_h
    μ_goal_open_a = μ_open_a .* κ_a

    ll_open_h = logpdf.(Poisson.(μ_goal_open_h), y_open_h)
    ll_open_a = logpdf.(Poisson.(μ_goal_open_a), y_open_a)
    Turing.@addlogprob! sum(ll_open_h .* match_weights)
    Turing.@addlogprob! sum(ll_open_a .* match_weights)

    # 6. Officiating & Penalty Submodel
    if config.recomb_config isa HierarchicalOfficiatingConfig
        officiating ~ to_submodel(build_penalty_officiating(config.recomb_config, n_refs))
        γ_ref_h = view(officiating.γ_ref, ref_indices)
        γ_ref_a = view(officiating.γ_ref, ref_indices)

        log_λ_pen_h = officiating.pen_base_μ .+ officiating.ha_pen .+ γ_ref_h
        log_λ_pen_a = officiating.pen_base_μ .+                       γ_ref_a

        λ_pen_h = exp.(clamp.(log_λ_pen_h, -6.0, 2.0))
        λ_pen_a = exp.(clamp.(log_λ_pen_a, -6.0, 2.0))

        ll_pen_h = logpdf.(Poisson.(λ_pen_h), n_pen_h)
        ll_pen_a = logpdf.(Poisson.(λ_pen_a), n_pen_a)
        Turing.@addlogprob! sum(ll_pen_h .* match_weights)
        Turing.@addlogprob! sum(ll_pen_a .* match_weights)
    end
end

function build_turing_model(model::DynamicPxGRecombModel, feature_set::Features.FeatureSet)
    data = feature_set.data

    n_teams    = Int(data[:n_teams])
    n_leagues  = get(data, :n_leagues, 1)
    n_refs     = get(data, :n_referees, 1)

    date_deltas = Vector{Int}(data[:dates])
    match_weights = calculate_match_weights(date_deltas, model.dynamics_config.days_half_life)

    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    month_ids  = Vector{Int}(data[:flat_month_ids])
    league_ids = get(data, :flat_league_ids, ones(Int, length(home_ids)))
    ref_ids    = get(data, :flat_referee_ids, ones(Int, length(home_ids)))
    delta_w    = get(data, :flat_delta_wealth, zeros(Float64, length(home_ids)))

    pxg_h      = Vector{Float64}(data[:flat_pxg_h])
    pxg_a      = Vector{Float64}(data[:flat_pxg_a])
    mask_h     = Vector{Float64}(data[:flat_mask_pxg_h])
    mask_a     = Vector{Float64}(data[:flat_mask_pxg_a])

    y_open_h   = Vector{Int}(data[:flat_y_open_h])
    y_open_a   = Vector{Int}(data[:flat_y_open_a])
    n_pen_h    = get(data, :flat_pen_awarded_h, zeros(Int, length(home_ids)))
    n_pen_a    = get(data, :flat_pen_awarded_a, zeros(Int, length(home_ids)))

    return build_recombined_pxg_engine(
        home_ids, away_ids, month_ids, league_ids, ref_ids,
        delta_w, pxg_h, pxg_a, mask_h, mask_a,
        y_open_h, y_open_a, n_pen_h, n_pen_a,
        match_weights, n_teams, n_leagues, n_refs, model
    )
end

function extract_parameters(model::DynamicPxGRecombModel, chain::Chains, feature_set::Features.FeatureSet)
    data = feature_set.data
    n_teams = Int(data[:n_teams])

    inter = extract_interception(chain, model.interception_config, 1, 12)
    ha    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn   = extract_dynamics(chain, model.dynamics_config, n_teams)
    w_val = extract_squad_wealth(chain, model.wealth_config)
    ν_val = extract_pxg_observation(chain, model.pxg_config)
    recomb = extract_recombination(chain, model.recomb_config)

    # Extract finishing factor κ
    κ_sym = [Symbol("log_κ_raw[$i]") for i in 1:n_teams]
    all_κ_present = all(haskey(chain, s) for s in κ_sym)
    if all_κ_present
        raw_mat = Array(chain[κ_sym])
        centered = raw_mat .- mean(raw_mat, dims=2)
        κ_mat = exp.(clamp.(centered, -0.50, 0.50))
    else
        κ_mat = ones(Float64, size(chain, 1), n_teams)
    end

    return (
        base_mu  = inter.μ_base,
        month_eff = inter.δ_month,
        ha_home  = ha,
        alpha    = dyn.α,
        beta     = dyn.β,
        w_wealth = w_val,
        ν_xg     = ν_val,
        kappa    = κ_mat,
        recomb   = recomb
    )
end

function extract_parameters(
    model::DynamicPxGRecombModel,
    df::AbstractDataFrame,
    feature_set::Features.FeatureSet,
    chain::Chains
)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    team_map  = data[:team_map]
    n_leagues = get(data, :n_leagues, 1)

    inter = extract_interception(chain, model.interception_config, 1, 12)
    ha_mat = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn = extract_dynamics(chain, model.dynamics_config, n_teams)
    w_val = extract_squad_wealth(chain, model.wealth_config)
    recomb = extract_recombination(chain, model.recomb_config)

    n_samples = size(chain, 1) * size(chain, 3)

    # Finishing factor κ
    κ_sym = [Symbol("log_κ_raw[$i]") for i in 1:n_teams]
    all_κ_present = all(haskey(chain, s) for s in κ_sym)
    if all_κ_present
        raw_mat = Array(chain[κ_sym])
        centered = raw_mat .- mean(raw_mat, dims=2)
        κ_mat = exp.(clamp.(centered, -0.50, 0.50))
    else
        κ_mat = ones(Float64, n_samples, n_teams)
    end

    league_sym = [Symbol("δ_league_raw[$i]") for i in 1:n_leagues]
    has_league = all(haskey(chain, s) for s in league_sym)
    if has_league
        raw_l = Array(chain[league_sym])
        delta_league = raw_l .- mean(raw_l, dims=2)
    else
        delta_league = zeros(Float64, n_samples, n_leagues)
    end

    wealth_lookup = get(data, :wealth_lookup, Dict{Int, Float64}())
    league_lookup = get(data, :league_lookup, Dict{Int, Int}())

    results = Dict{Int, NamedTuple}()

    for row in eachrow(df)
        mid = Int(row.match_id)

        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        α_h = h_idx > 0 ? dyn.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn.β[:, a_idx] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)
        κ_h = h_idx > 0 ? κ_mat[:, h_idx]  : ones(n_samples)
        κ_a = a_idx > 0 ? κ_mat[:, a_idx]  : ones(n_samples)

        m_idx = month(row.match_date)
        l_idx = get(league_lookup, mid, 1)

        dw = get(wealth_lookup, mid, 0.0)
        w_shift = w_val .* dw

        inter_match = inter.μ_base[:, 1] .+ inter.δ_month[:, m_idx] .+ delta_league[:, min(l_idx, n_leagues)]

        log_μ_open_h = clamp.(inter_match .+ γ_h .+ α_h .- β_a .+ w_shift, -5.0, 4.0)
        log_μ_open_a = clamp.(inter_match .+        α_a .- β_h .- w_shift, -5.0, 4.0)

        μ_open_h = exp.(log_μ_open_h)
        μ_open_a = exp.(log_μ_open_a)

        # Apply finishing efficiency to open play goals
        μ_goal_open_h = μ_open_h .* κ_h
        μ_goal_open_a = μ_open_a .* κ_a

        # Officiating / Recombination rates
        q_pen   = hasproperty(recomb, :pen_conv) ? recomb.pen_conv : fill(0.768, n_samples)
        og_rate = hasproperty(recomb, :og_rate) ? recomb.og_rate : fill(0.0276, n_samples)

        if hasproperty(recomb, :pen_base_μ)
            λ_pen_h = exp.(clamp.(recomb.pen_base_μ .+ recomb.ha_pen, -6.0, 2.0))
            λ_pen_a = exp.(clamp.(recomb.pen_base_μ, -6.0, 2.0))
        else
            λ_pen_h = fill(0.207, n_samples)
            λ_pen_a = fill(0.207, n_samples)
        end

        μ_total_h = μ_goal_open_h .+ (q_pen .* λ_pen_h) .+ og_rate
        μ_total_a = μ_goal_open_a .+ (q_pen .* λ_pen_a) .+ og_rate

        results[mid] = (;
            λ_h = μ_total_h,
            λ_a = μ_total_a,
            μ_open_h = μ_open_h,
            μ_open_a = μ_open_a,
            κ_h = κ_h,
            κ_a = κ_a,
            λ_pen_h = λ_pen_h,
            λ_pen_a = λ_pen_a,
            true_xg_h = μ_total_h,
            true_xg_a = μ_total_a
        )
    end

    return results
end
