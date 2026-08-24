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
    team_name_to_index = _team_name_to_existing_index(team_map, df_clean)
    league_map = _pooled_legacy_league_map(ds)
    
    month_indices  = month.(m.match_date)
    league_indices = [_oos_league_index(row, league_map, 1) for row in eachrow(m)]
    
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
            :team_name_to_index  => team_name_to_index,
            :league_map          => league_map,
            :league_encoding     => :pooled_legacy_one_column,
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
    home_pens::Vector{Int},
    away_pens::Vector{Int},
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
    # 1. Open Play Priors
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
    
    # Squad Wealth Sensitivity
    w_wealth ~ w_wealth_prior
    w_shift  = w_wealth .* wealth_diff

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
    
    log_mu_h = clamp.(int_m .+ ha_home .+ view(alpha, home_indices) .- view(beta, away_indices) .+ w_shift, -10.0, 10.0)
    log_mu_a = clamp.(int_m .+ view(alpha, away_indices) .- view(beta, home_indices) .- w_shift, -10.0, 10.0)
    
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

function PreGame.extract_parameters(
    model::TeamGoalsRecombIntegratedPoisWealthModel,
    df::AbstractDataFrame,
    feature_set,
    chain::Chains
)
    data = feature_set.data
    team_map   = data[:team_map]
    team_name_to_index = _feature_team_name_to_index(data)
    league_map = _feature_league_map(data)
    unknowns = Dict{String,Int}()
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
    
    effects = _tau_scaled_team_effects(chain, n_teams; context="TeamGoalsRecombIntegratedPoisWealthModel extractor")
    alpha_mat, beta_mat = effects.alpha, effects.beta
    
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
        h_idx = _oos_team_index(row, :home, team_map, team_name_to_index, unknowns)
        a_idx = _oos_team_index(row, :away, team_map, team_name_to_index, unknowns)
        
        α_h = h_idx > 0 ? alpha_mat[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? beta_mat[:, h_idx]  : zeros(n_samples)
        α_a = a_idx > 0 ? alpha_mat[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? beta_mat[:, a_idx]  : zeros(n_samples)
        
        dw = get(wealth_map, mid, 0.0)
        w_shift = w_wealth .* dw
        
        m_idx = month(row.match_date)
        l_idx = _oos_league_index(row, league_map, n_leagues)
        
        δ_m = (m_idx >= 1 && m_idx <= n_months) ? delta_month_mat[:, m_idx] : zeros(n_samples)
        δ_l = (l_idx >= 1 && l_idx <= n_leagues) ? delta_league_mat[:, l_idx] : zeros(n_samples)
        
        int_m = base_mu .+ δ_m .+ δ_l
        λ_h = exp.(int_m .+ ha_home .+ α_h .- β_a .+ w_shift)
        λ_a = exp.(int_m .+           α_a .- β_h .- w_shift)
        
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
        
        lambda_total_h = λ_h .+ lambda_noise_h
        lambda_total_a = λ_a .+ lambda_noise_a
        
        results[mid] = (;
            λ_h = lambda_total_h,
            λ_a = lambda_total_a,
            r_h = fill(100.0, n_samples),
            r_a = fill(100.0, n_samples),
            true_xg_h = λ_h,
            true_xg_a = λ_a,
            lambda_pen_h = lambda_pen_h,
            lambda_pen_a = lambda_pen_a,
            lambda_open_h = λ_h,
            lambda_open_a = λ_a
        )
    end
    _warn_oos_unknown_teams!(unknowns, "$(typeof(model)) extraction")
    return results
end

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
    team_name_to_index = _feature_team_name_to_index(F)
    league_map = _feature_league_map(F)
    unknowns = Dict{String,Int}()
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
    
    effects = _tau_scaled_team_effects(chain, n_teams; context="TeamGoalsRecombIntegratedPoisWealthModel extractor")
    alpha_mat, beta_mat = effects.alpha, effects.beta
    
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
        r_id = row.referee_id
        h_idx = _oos_team_index(row, :home, team_map, team_name_to_index, unknowns)
        a_idx = _oos_team_index(row, :away, team_map, team_name_to_index, unknowns)
        r_idx = get(ref_map, r_id, -1)
        
        α_h = h_idx > 0 ? alpha_mat[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? beta_mat[:, h_idx]  : zeros(n_samples)
        α_a = a_idx > 0 ? alpha_mat[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? beta_mat[:, a_idx]  : zeros(n_samples)
        
        dw = get(wealth_map, m_id, 0.0)
        w_shift = w_wealth .* dw
        
        m_idx = month(row.match_date)
        l_idx = _oos_league_index(row, league_map, n_leagues)
        
        δ_m = (m_idx >= 1 && m_idx <= n_months) ? delta_month_mat[:, m_idx] : zeros(n_samples)
        δ_l = (l_idx >= 1 && l_idx <= n_leagues) ? delta_league_mat[:, l_idx] : zeros(n_samples)
        
        int_m = base_mu .+ δ_m .+ δ_l
        mu_open_h = exp.(int_m .+ ha_home .+ α_h .- β_a .+ w_shift)
        mu_open_a = exp.(int_m .+           α_a .- β_h .- w_shift)
        
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
    
    _warn_oos_unknown_teams!(unknowns, "$(typeof(model)) FeatureSet prediction extraction")
    return Predictions.LatentStates(out_df, model)
end

function Predictions.extract_params(model::TeamGoalsRecombIntegratedPoisWealthModel, row::DataFrameRow)
    ln_h = hasproperty(row, :lambda_noise_h) ? (row.lambda_noise_h isa AbstractVector ? row.lambda_noise_h : [row.lambda_noise_h]) : (hasproperty(row, :lambda_pen_h) ? (0.768 .* (row.lambda_pen_h isa AbstractVector ? row.lambda_pen_h : [row.lambda_pen_h])) .+ 0.0276 : fill(0.10, length(row.λ_h)))
    ln_a = hasproperty(row, :lambda_noise_a) ? (row.lambda_noise_a isa AbstractVector ? row.lambda_noise_a : [row.lambda_noise_a]) : (hasproperty(row, :lambda_pen_a) ? (0.768 .* (row.lambda_pen_a isa AbstractVector ? row.lambda_pen_a : [row.lambda_pen_a])) .+ 0.0276 : fill(0.10, length(row.λ_a)))
    
    return (
        λ_open_h = hasproperty(row, :λ_open_h) ? (row.λ_open_h isa AbstractVector ? row.λ_open_h : [row.λ_open_h]) : (hasproperty(row, :lambda_open_h) ? (row.lambda_open_h isa AbstractVector ? row.lambda_open_h : [row.lambda_open_h]) : row.λ_h),
        λ_open_a = hasproperty(row, :λ_open_a) ? (row.λ_open_a isa AbstractVector ? row.λ_open_a : [row.λ_open_a]) : (hasproperty(row, :lambda_open_a) ? (row.lambda_open_a isa AbstractVector ? row.lambda_open_a : [row.lambda_open_a]) : row.λ_a),
        lambda_noise_h = ln_h,
        lambda_noise_a = ln_a
    )
end

function Predictions.compute_score_matrix(model::TeamGoalsRecombIntegratedPoisWealthModel, params; max_goals::Int = 12)
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
        
        # Convolve: P(Y_total = g) = sum_{m=0}^g P(Y_open = m) * P(Y_noise = g - m)
        p_tot_h = [sum(p_open_h[m+1] * p_noise_h[g - m + 1] for m in 0:g) for g in 0:max_goals-1]
        p_tot_a = [sum(p_open_a[m+1] * p_noise_a[g - m + 1] for m in 0:g) for g in 0:max_goals-1]
        
        p_tot_h ./= sum(p_tot_h)
        p_tot_a ./= sum(p_tot_a)
        
        S[:, :, k] = p_tot_h * p_tot_a'
    end
    return Predictions.ScoreMatrix(S)
end

Predictions.compute_score_matrix(model::TeamGoalsRecombIntegratedPoisWealthModel, r::DataFrameRow; max_goals::Int = 12) = Predictions.compute_score_matrix(model, Predictions.extract_params(model, r); max_goals=max_goals)

println("✓ l04_recomb_wealth_models.jl loaded (Integrated Poisson Recombination + Starting-XI Squad Wealth Model)")
