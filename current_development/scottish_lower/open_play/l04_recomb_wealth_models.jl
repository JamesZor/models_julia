# ==============================================================================
# l04_recomb_wealth_models.jl
#
# Integrated Poisson Recombination + Starting-XI Squad Wealth Model (Scottish Lower)
#
# Mathematical Architecture:
# 1. Open-Play Vectorized Poisson Intensity:
#    log μ_open_h = base_μ + δ_month + δ_league + ha_home + α_h - β_a + w_wealth * ΔW
#    log μ_open_a = base_μ + δ_month + δ_league + α_a - β_h - w_wealth * ΔW
# 2. Penalty Submodel:
#    log λ_pen_h = base_μ_pen + γ_ref + α_pen_draw_h + β_pen_foul_a
#    log λ_pen_a = base_μ_pen + γ_ref + α_pen_draw_a + β_pen_foul_h
#    q ~ Beta(76, 24)
# 3. Discrete Score Convolution & Dixon-Coles Adjustment:
#    P(H=h, A=a) = [ Σ Σ Poisson(h-k_h|μ_open_h) Poisson(a-k_a|μ_open_a) P(k_h) P(k_a) ] * τ(h, a; ρ)
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

# ==============================================================================
# SECTION 1: WEALTH DATA EXTRACTION & STANDARDIZATION
# ==============================================================================

function wealth_db_connect()
    url = get(ENV, "BF_DB_URL", "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@archpc:5433/betdb")
    return LibPQ.Connection(url)
end

function fetch_scottish_player_valuations(conn::LibPQ.Connection; tournament_ids=[386, 56, 57])
    sql = """
    WITH scottish_players AS (
        SELECT DISTINCT l.player_id
        FROM sofascore.match_player_lineups l
        JOIN sofascore.matches m ON l.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
        
        UNION
        
        SELECT DISTINCT l.sofascore_player_id AS player_id
        FROM bbc.match_lineup l
        JOIN sofascore.matches m ON l.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
          AND l.sofascore_player_id IS NOT NULL
    )
    SELECT DISTINCT ON (v.player_id)
        v.player_id,
        v.player_name,
        v.player_position,
        v.market_value
    FROM (
        SELECT 
            (data->'player'->>'id')::int AS player_id,
            data->'player'->>'name' AS player_name,
            data->'player'->>'position' AS player_position,
            (data->'player'->'proposedMarketValueRaw'->>'value')::numeric AS market_value
        FROM sofascore.match_incidents
        WHERE data->'player'->'proposedMarketValueRaw'->>'value' IS NOT NULL
        
        UNION ALL
        
        SELECT 
            (data->'playerIn'->>'id')::int AS player_id,
            data->'playerIn'->>'name' AS player_name,
            data->'playerIn'->>'position' AS player_position,
            (data->'playerIn'->'proposedMarketValueRaw'->>'value')::numeric AS market_value
        FROM sofascore.match_incidents
        WHERE data->'playerIn'->'proposedMarketValueRaw'->>'value' IS NOT NULL
        
        UNION ALL
        
        SELECT 
            (data->'playerOut'->>'id')::int AS player_id,
            data->'playerOut'->>'name' AS player_name,
            data->'playerOut'->>'position' AS player_position,
            (data->'playerOut'->'proposedMarketValueRaw'->>'value')::numeric AS market_value
        FROM sofascore.match_incidents
        WHERE data->'playerOut'->'proposedMarketValueRaw'->>'value' IS NOT NULL
    ) v
    JOIN scottish_players sp ON v.player_id = sp.player_id
    WHERE v.market_value IS NOT NULL AND v.market_value > 0
    ORDER BY v.player_id, v.market_value DESC
    """
    res = LibPQ.execute(conn, sql, [tournament_ids])
    return DataFrame(res)
end

function get_cached_scottish_valuation_catalog()
    cache_path = joinpath(@__DIR__, "cache", "scottish_val_catalog.jls")
    if isfile(cache_path)
        return deserialize(cache_path)
    else
        mkpath(dirname(cache_path))
        local val_cat
        try
            conn = wealth_db_connect()
            val_cat = fetch_scottish_player_valuations(conn, tournament_ids=[386, 56, 57])
            close(conn)
        catch e
            @warn "Failed to connect to PostgreSQL for valuations: $e. Falling back to empty catalog."
            val_cat = DataFrame(player_id=Int[], market_value=Float64[])
        end
        serialize(cache_path, val_cat)
        return val_cat
    end
end

function build_scottish_match_wealth_map(ds::Data.DataStore; fallback_default::Float64 = 100_000.0)
    val_cat = get_cached_scottish_valuation_catalog()
    val_map = Dict(Int(r.player_id) => Float64(r.market_value) for r in eachrow(val_cat))
    
    lineup_sub_col = :is_substitute in propertynames(ds.lineups) ? :is_substitute : (:substitute in propertynames(ds.lineups) ? :substitute : nothing)
    starters = lineup_sub_col !== nothing ? filter(r -> !coalesce(r[lineup_sub_col], false), copy(ds.lineups)) : copy(ds.lineups)
    
    # Map player market value with fallback
    pos_medians = Dict("G" => 80_000.0, "D" => 100_000.0, "M" => 110_000.0, "F" => 120_000.0)
    pos_col = :position in propertynames(starters) ? :position : nothing
    
    starters.val = [
        let v = get(val_map, coalesce(r.player_id, 0), NaN)
            if !isnan(v) && v > 0
                v
            elseif pos_col !== nothing && !ismissing(r[pos_col])
                get(pos_medians, String(r[pos_col]), fallback_default)
            else
                fallback_default
            end
        end
        for r in eachrow(starters)
    ]
    
    # Identify home/away
    team_side_col = :team_side in propertynames(starters) ? :team_side : (:is_home_team in propertynames(starters) ? :is_home_team : :is_home)
    is_home_expr(r) = (team_side_col == :team_side) ? (r.team_side == "home") : Bool(r[team_side_col])
    
    starters.is_home_bool = is_home_expr.(eachrow(starters))
    
    home_starters = filter(r -> r.is_home_bool, starters)
    away_starters = filter(r -> !r.is_home_bool, starters)
    
    home_agg = combine(groupby(home_starters, :match_id), :val => (v -> mean(log.(v))) => :log_w_h)
    away_agg = combine(groupby(away_starters, :match_id), :val => (v -> mean(log.(v))) => :log_w_a)
    
    joined = innerjoin(home_agg, away_agg, on = :match_id)
    
    # Standardize
    all_log_w = vcat(joined.log_w_h, joined.log_w_a)
    mu_w  = isempty(all_log_w) ? 0.0 : mean(all_log_w)
    std_w = isempty(all_log_w) ? 1.0 : std(all_log_w)
    std_w = std_w == 0.0 || isnan(std_w) ? 1.0 : std_w
    
    joined.delta_w = ((joined.log_w_h .- mu_w) ./ std_w) .- ((joined.log_w_a .- mu_w) ./ std_w)
    
    return Dict(Int(r.match_id) => Float64(r.delta_w) for r in eachrow(joined))
end

# ==============================================================================
# SECTION 2: STRUCT DEFINITIONS & FEATURE PIPELINE
# ==============================================================================

"""
    TeamGoalsRecombIntegratedPoisWealthModel <: PreGame.AbstractTimeDecayTeamModel
"""
struct TeamGoalsRecombIntegratedPoisWealthModel <: PreGame.AbstractTimeDecayTeamModel
    dynamics_config::PreGame.AbstractDynamicsConfig
    w_wealth_prior::Distribution
    name::String
end

TeamGoalsRecombIntegratedPoisWealthModel(; 
    dynamics_config = PreGame.TimeDecayDynamics(days_half_life=365.0),
    w_wealth_prior = truncated(Normal(0.10, 0.05), lower = 0.0),
    name = "recomb_pois_wealth_integrated"
) = TeamGoalsRecombIntegratedPoisWealthModel(dynamics_config, w_wealth_prior, name)

# FeatureSet builder with Starting-XI Wealth
function _build_recomb_wealth_features(b::Data.SplitBoundary, ds::Data.DataStore, model::TeamGoalsRecombIntegratedPoisWealthModel)
    df_clean, df_ref = build_open_play_target_dataset(ds)
    all_refs = unique(filter(x -> x > 0, df_clean.referee_id))
    ref_map = Dict(r => idx for (idx, r) in enumerate(all_refs))
    
    wealth_map = build_scottish_match_wealth_map(ds)
    
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
    
    wealth_diff = Float64[get(wealth_map, mid, 0.0) for mid in m.match_id]
    
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
    model::TeamGoalsRecombIntegratedPoisWealthModel,
    dynamics_col::Symbol = :match_month
)
    raw_vector = [
        (_build_recomb_wealth_features(boundary, ds, model), meta)
        for (boundary, meta) in splits
    ]
    return Features.FeatureCollection(raw_vector)
end

function Features.create_features(
    boundary::Data.SplitBoundary,
    ds::Data.DataStore,
    model::TeamGoalsRecombIntegratedPoisWealthModel,
    dynamics_col::Symbol = :match_month
)
    return _build_recomb_wealth_features(boundary, ds, model)
end

# ==============================================================================
# SECTION 3: TURING MODEL SPECIFICATION
# ==============================================================================

@model function _turing_goals_recomb_pois_wealth(
    home_indices::Vector{Int},
    away_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    home_open_goals::Vector{Int},
    away_open_goals::Vector{Int},
    home_pens_awarded::Vector{Int},
    away_pens_awarded::Vector{Int},
    ref_indices::Vector{Int},
    ref_mask::Vector{Float64},
    wealth_diff::Vector{Float64},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_refs::Int,
    n_months::Int,
    n_leagues::Int,
    w_wealth_prior::Distribution
)
    # 1. Priors: Open Play
    base_mu_open ~ Normal(0.15, 0.5)
    ha_home      ~ Normal(0.20, 0.15)
    
    delta_month  ~ filldist(Normal(0.0, 0.1), n_months)
    delta_league ~ filldist(Normal(0.0, 0.1), n_leagues)
    
    sigma_team   ~ truncated(Normal(0.3, 0.15), lower = 0.05)
    raw_alpha    ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_beta     ~ filldist(Normal(0.0, 1.0), n_teams)
    
    alpha = (raw_alpha .- mean(raw_alpha)) .* sigma_team
    beta  = (raw_beta  .- mean(raw_beta))  .* sigma_team
    
    # Squad Wealth Sensitivity
    w_wealth ~ w_wealth_prior
    w_shift = w_wealth .* wealth_diff

    # 2. Priors: Penalty Submodel
    base_mu_pen    ~ Normal(-2.0, 0.5)
    sigma_ref      ~ truncated(Normal(0.2, 0.1), lower = 0.01)
    raw_gamma_ref  ~ filldist(Normal(0.0, 1.0), n_refs)
    gamma_ref      = (raw_gamma_ref .- mean(raw_gamma_ref)) .* sigma_ref
    
    sigma_team_pen ~ truncated(Normal(0.15, 0.1), lower = 0.01)
    raw_alpha_pen  ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_beta_pen   ~ filldist(Normal(0.0, 1.0), n_teams)
    alpha_pen      = (raw_alpha_pen .- mean(raw_alpha_pen)) .* sigma_team_pen
    beta_pen       = (raw_beta_pen  .- mean(raw_beta_pen))  .* sigma_team_pen
    
    # 3. Vectorized Open-Play Likelihood
    int_open = base_mu_open .+ view(delta_month, month_indices) .+ view(delta_league, league_indices)
    
    log_mu_open_h = clamp.(int_open .+ ha_home .+ view(alpha, home_indices) .- view(beta, away_indices) .+ w_shift, -10.0, 10.0)
    log_mu_open_a = clamp.(int_open .+ view(alpha, away_indices) .- view(beta, home_indices) .- w_shift, -10.0, 10.0)
    
    mu_open_h = exp.(log_mu_open_h) .+ 1e-6
    mu_open_a = exp.(log_mu_open_a) .+ 1e-6
    
    ll_open_h = logpdf.(Poisson.(mu_open_h), home_open_goals)
    ll_open_a = logpdf.(Poisson.(mu_open_a), away_open_goals)
    
    # 4. Vectorized Penalty Awarded Likelihood
    ref_eff = view(gamma_ref, ref_indices) .* ref_mask
    
    log_lambda_pen_h = clamp.(base_mu_pen .+ ref_eff .+ view(alpha_pen, home_indices) .+ view(beta_pen, away_indices), -10.0, 5.0)
    log_lambda_pen_a = clamp.(base_mu_pen .+ ref_eff .+ view(alpha_pen, away_indices) .+ view(beta_pen, home_indices), -10.0, 5.0)
    
    lambda_pen_h = exp.(log_lambda_pen_h) .+ 1e-6
    lambda_pen_a = exp.(log_lambda_pen_a) .+ 1e-6
    
    ll_pen_h = logpdf.(Poisson.(lambda_pen_h), home_pens_awarded)
    ll_pen_a = logpdf.(Poisson.(lambda_pen_a), away_pens_awarded)
    
    # 5. Combined Likelihood
    Turing.@addlogprob! sum((ll_open_h .+ ll_open_a .+ ll_pen_h .+ ll_pen_a) .* match_weights)
end

function PreGame.build_turing_model(
    model::TeamGoalsRecombIntegratedPoisWealthModel,
    feature_set
)
    f = feature_set.data
    return _turing_goals_recomb_pois_wealth(
        f[:home_team_indices],
        f[:away_team_indices],
        f[:month_indices],
        f[:league_indices],
        f[:home_open_goals],
        f[:away_open_goals],
        f[:home_pens],
        f[:away_pens],
        f[:ref_indices],
        f[:ref_mask],
        f[:wealth_diff],
        f[:match_weights],
        f[:n_teams],
        f[:n_refs],
        f[:n_months],
        f[:n_leagues],
        model.w_wealth_prior
    )
end

# ==============================================================================
# SECTION 4: OUT-OF-SAMPLE PREDICTIONS & SCORE CONVOLUTION
# ==============================================================================

function Predictions.extract_params(
    model::TeamGoalsRecombIntegratedPoisWealthModel,
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
    
    n_samples = size(chain, 1) * size(chain, 3)
    
    # 1. Open-play parameter extraction
    base_mu_open_samples = vec(Array(chain["base_mu_open"]))
    ha_home_samples      = vec(Array(chain["ha_home"]))
    w_wealth_samples     = vec(Array(chain["w_wealth"]))
    
    raw_alpha_samples    = Array(chain[[Symbol("raw_alpha[$i]") for i in 1:F[:n_teams]]])
    raw_beta_samples     = Array(chain[[Symbol("raw_beta[$i]") for i in 1:F[:n_teams]]])
    sigma_team_samples   = vec(Array(chain["sigma_team"]))
    
    alpha_mat = zeros(Float64, n_samples, F[:n_teams])
    beta_mat  = zeros(Float64, n_samples, F[:n_teams])
    for s in 1:n_samples
        ra = raw_alpha_samples[s, :]
        rb = raw_beta_samples[s, :]
        alpha_mat[s, :] = (ra .- mean(ra)) .* sigma_team_samples[s]
        beta_mat[s, :]  = (rb .- mean(rb))  .* sigma_team_samples[s]
    end
    
    # 2. Penalty submodel extraction
    base_mu_pen_samples    = vec(Array(chain["base_mu_pen"]))
    sigma_ref_samples      = vec(Array(chain["sigma_ref"]))
    raw_gamma_ref_samples  = Array(chain[[Symbol("raw_gamma_ref[$i]") for i in 1:F[:n_refs]]])
    
    gamma_ref_mat = zeros(Float64, n_samples, F[:n_refs])
    for s in 1:n_samples
        rg = raw_gamma_ref_samples[s, :]
        gamma_ref_mat[s, :] = (rg .- mean(rg)) .* sigma_ref_samples[s]
    end
    
    sigma_team_pen_samples = vec(Array(chain["sigma_team_pen"]))
    raw_alpha_pen_samples  = Array(chain[[Symbol("raw_alpha_pen[$i]") for i in 1:F[:n_teams]]])
    raw_beta_pen_samples   = Array(chain[[Symbol("raw_beta_pen[$i]") for i in 1:F[:n_teams]]])
    
    alpha_pen_mat = zeros(Float64, n_samples, F[:n_teams])
    beta_pen_mat  = zeros(Float64, n_samples, F[:n_teams])
    for s in 1:n_samples
        rap = raw_alpha_pen_samples[s, :]
        rbp = raw_beta_pen_samples[s, :]
        alpha_pen_mat[s, :] = (rap .- mean(rap)) .* sigma_team_pen_samples[s]
        beta_pen_mat[s, :]  = (rbp .- mean(rbp))  .* sigma_team_pen_samples[s]
    end
    
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
        
        h_idx = get(team_map, h_id, 0)
        a_idx = get(team_map, a_id, 0)
        r_idx = get(ref_map, r_id, 0)
        
        dw = get(wealth_map, m_id, 0.0)
        w_shift = w_wealth_samples .* dw
        
        # Open Play intensities
        alpha_h = h_idx > 0 ? alpha_mat[:, h_idx] : zeros(Float64, n_samples)
        beta_h  = h_idx > 0 ? beta_mat[:, h_idx]  : zeros(Float64, n_samples)
        alpha_a = a_idx > 0 ? alpha_mat[:, a_idx] : zeros(Float64, n_samples)
        beta_a  = a_idx > 0 ? beta_mat[:, a_idx]  : zeros(Float64, n_samples)
        
        m_idx = month(row.match_date)
        delta_m_samples = vec(Array(chain[Symbol("delta_month[$m_idx]")]))
        delta_l_samples = vec(Array(chain[Symbol("delta_league[1]")]))
        
        int_open = base_mu_open_samples .+ delta_m_samples .+ delta_l_samples
        
        mu_open_h = exp.(int_open .+ ha_home_samples .+ alpha_h .- beta_a .+ w_shift)
        mu_open_a = exp.(int_open .+ alpha_a .- beta_h .- w_shift)
        
        # Penalty intensities
        ref_eff = r_idx > 0 ? gamma_ref_mat[:, r_idx] : zeros(Float64, n_samples)
        
        alpha_pen_h = h_idx > 0 ? alpha_pen_mat[:, h_idx] : zeros(Float64, n_samples)
        beta_pen_h  = h_idx > 0 ? beta_pen_mat[:, h_idx]  : zeros(Float64, n_samples)
        alpha_pen_a = a_idx > 0 ? alpha_pen_mat[:, a_idx] : zeros(Float64, n_samples)
        beta_pen_a  = a_idx > 0 ? beta_pen_mat[:, a_idx]  : zeros(Float64, n_samples)
        
        lambda_pen_h = exp.(base_mu_pen_samples .+ ref_eff .+ alpha_pen_h .+ beta_pen_a)
        lambda_pen_a = exp.(base_mu_pen_samples .+ ref_eff .+ alpha_pen_a .+ beta_pen_h)
        
        # Fixed empirical conversion rate & Dixon-Coles rho
        q_pen_samples = fill(0.76, n_samples)
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

function Predictions.compute_score_matrix(
    model::TeamGoalsRecombIntegratedPoisWealthModel,
    params;
    max_goals::Int = 12
)
    mu_open_h    = params.mu_open_h
    mu_open_a    = params.mu_open_a
    lambda_pen_h = params.lambda_pen_h
    lambda_pen_a = params.lambda_pen_a
    q_pen        = get(params, :q_pen, 0.76)
    rho          = get(params, :rho, -0.05)

    # 1. Open Play Marginals: Poisson
    d_open_h = Poisson(max(Float64(mu_open_h), 1e-4))
    d_open_a = Poisson(max(Float64(mu_open_a), 1e-4))

    p_open_h = [pdf(d_open_h, g) for g in 0:max_goals]
    p_open_a = [pdf(d_open_a, g) for g in 0:max_goals]

    # 2. Penalty Marginals
    p_pen_h = _compute_penalty_goal_probs(lambda_pen_h, q_pen, max_goals)
    p_pen_a = _compute_penalty_goal_probs(lambda_pen_a, q_pen, max_goals)

    # 3. Discrete Convolution
    p_h = _discrete_convolve(p_open_h, p_pen_h, max_goals)
    p_a = _discrete_convolve(p_open_a, p_pen_a, max_goals)

    # 4. Joint Matrix with Dixon-Coles Low-Score Adjustment
    S = zeros(Float64, max_goals + 1, max_goals + 1)
    for h in 0:max_goals, a in 0:max_goals
        tau = _dixon_coles_tau(h, a, mu_open_h, mu_open_a, rho)
        S[h + 1, a + 1] = p_h[h + 1] * p_a[a + 1] * tau
    end

    total_p = sum(S)
    if total_p > 0.0
        S ./= total_p
    end

    return S
end

println("✓ l04_recomb_wealth_models.jl loaded (Integrated Poisson Recombination + Starting-XI Squad Wealth Model)")
