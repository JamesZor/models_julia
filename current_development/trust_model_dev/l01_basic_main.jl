using BayesianFootball
using BayesianFootball.Features
using BayesianFootball.Training
using BayesianFootball.Data.Markets # <-- Native market typing
using DataFrames
using Statistics
using Turing
using LogExpFunctions

# 1. The Ecosystem Hook
abstract type AbstractTrustModel <: BayesianFootball.TypesInterfaces.AbstractFootballModel end
abstract type AbstractTrustComponent end

# 2. Component Configs (Modularity)
Base.@kwdef struct GlobalTrustConfig <: AbstractTrustComponent
    use_clv_prior::Bool = true
    default_mean::Float64 = -1.0
end

Base.@kwdef struct HierarchicalTeamTrustConfig <: AbstractTrustComponent
    sigma_prior::Float64 = 0.5
end

# 3. Master Trust Model Config
Base.@kwdef struct TrustModelConfig <: AbstractTrustModel
    # Now uses the native MarketConfig! (fully qualified to prevent namespace issues)
    market_config::BayesianFootball.Data.Markets.MarketConfig = BayesianFootball.Data.Markets.MarketConfig(BayesianFootball.Data.Markets.MarketOverUnder(2.5)) 
    
    global_config::GlobalTrustConfig = GlobalTrustConfig()
    team_config::Union{Nothing, HierarchicalTeamTrustConfig} = HierarchicalTeamTrustConfig()
end

# 4. Pipeline Config
Base.@kwdef struct TrustPipelineConfig
    name::String
    model::AbstractTrustModel
    training_config::TrainingConfig # Reusing the NUTS/Walk-Forward engine!
end



# ---------------------------------------------------------
# A. Custom Feature Definitions
# ---------------------------------------------------------

# This single feature config handles everything, ensuring arrays never misalign
struct TrustDatasetFeature <: Features.AbstractFeatureConfig
    l1_ppd_df::DataFrame 
    markets::Vector{BayesianFootball.Data.Markets.AbstractMarket}
end

# ---------------------------------------------------------
# C. The Extractors (The Monkey-Patch)
# ---------------------------------------------------------

function compute_is_winner(selection::String, home_score::Int, away_score::Int)
    if selection == "home"
        return home_score > away_score
    elseif selection == "away"
        return away_score > home_score
    elseif selection == "draw"
        return home_score == away_score
    elseif selection == "btts_yes"
        return home_score > 0 && away_score > 0
    elseif selection == "btts_no"
        return home_score == 0 || away_score == 0
    elseif startswith(selection, "under_")
        line = parse(Float64, replace(selection, "under_" => "")) / 10.0
        return Float64(home_score + away_score) < line
    elseif startswith(selection, "over_")
        line = parse(Float64, replace(selection, "over_" => "")) / 10.0
        return Float64(home_score + away_score) > line
    end
    return missing
end

function Features.add_feature!(
    F_data::Dict, 
    config::TrustDatasetFeature, 
    ordered_ids, 
    team_map::Dict, 
    ds::BayesianFootball.Data.DataStore
)
    # 1. 🚨 CRITICAL AXIOM PRESERVATION & NO DOUBLE-COUNTING: 
    # We only extract the FIRST outcome of every market (e.g. :home, :over_25, :btts_yes).
    # Since they are symmetric inverses, training a Bernoulli likelihood on BOTH sides 
    # would artificially double the sample size and incorrectly shrink the posterior variance!
    outcome_to_market_id = Dict{String, Int}()
    target_strs = String[]
    
    for (i, m) in enumerate(config.markets)
        # Extract ONLY the first outcome defined in the market's interface
        first_out = first(values(BayesianFootball.Data.Markets.outcomes(m)))
        out_str = String(first_out)
        
        push!(target_strs, out_str)
        outcome_to_market_id[out_str] = i # Map this single outcome back to the parent market ID
    end
    unique!(target_strs)
    
    # 2. Extract odds (Now only 1 row per market per match!)
    odds_df = subset(ds.odds, 
        :match_id => ByRow(in(ordered_ids)),
        :selection => ByRow(s -> String(s) ∈ target_strs)
    )
    
    # 3. Extract L1 predictions
    l1_df = subset(config.l1_ppd_df, 
        :match_id => ByRow(in(ordered_ids)),
        :selection => ByRow(s -> String(s) ∈ target_strs)
    )
    
    # 4. Inner join to guarantee perfect alignment
    combined = innerjoin(odds_df, l1_df, on=[:match_id, :selection], makeunique=true)
    
    # 5. Extract Team IDs & Scores
    match_team_map = Dict(row.match_id => (row.home_team, row.away_team) for row in eachrow(ds.matches))
    score_map = Dict(row.match_id => (row.home_score, row.away_score) for row in eachrow(ds.matches))
    
    # 6. Backfill missing `is_winner`
    resolved_winners = map(eachrow(combined)) do row
        if ismissing(row.is_winner)
            hs, as = score_map[row.match_id]
            if ismissing(hs) || ismissing(as)
                return missing
            end
            return compute_is_winner(String(row.selection), Int(hs), Int(as))
        else
            return row.is_winner
        end
    end
    combined.is_winner = resolved_winners
    filter!(row -> !ismissing(row.is_winner), combined)
    
    # 7. Populate F_data
    F_data[:q_market] = combined.prob_fair_close
    F_data[:is_winner] = convert(Vector{Int}, combined.is_winner)
    F_data[:p_model] = [mean(dist) for dist in combined.distribution]
    
    # Keep the raw string so we know what selection this row is
    F_data[:market_name_str] = String.(combined.selection)
    
    # Map the outcome string back to its PARENT MARKET'S integer index!
    F_data[:market_index] = [outcome_to_market_id[String(sel)] for sel in combined.selection]
    
    F_data[:home_team_index] = [team_map[match_team_map[id][1]] for id in combined.match_id]
    F_data[:away_team_index] = [team_map[match_team_map[id][2]] for id in combined.match_id]
    
    # Total markets is the number of AbstractMarkets passed in, not the number of individual selections!
    F_data[:n_markets] = length(config.markets)
end


# =========================================================
# 3. Stage 3: The Turing Engine (@model)
# =========================================================

# ---------------------------------------------------------


@model function trust_engine(
    is_winner::Vector{Int}, 
    p_model::Vector{Float64}, 
    q_market::Vector{Float64},
    home_team_index::Vector{Int},
    away_team_index::Vector{Int},
    market_index::Vector{Int},
    n_teams::Int,
    n_markets::Int,
    config::TrustModelConfig
)
    # 1. Global Intercept (w0 array per market)
    if config.global_config.use_clv_prior
        w0 ~ filldist(Normal(config.global_config.default_mean, 1.0), n_markets)
    else
        w0 ~ filldist(Normal(config.global_config.default_mean, 1.0), n_markets)
    end
    
    # 2. Hierarchical Team Effects (array per team)
    if config.team_config === nothing
        team_effects = zeros(n_teams)
    else
        σ_team ~ truncated(Normal(0.0, config.team_config.sigma_prior), lower=0.0)
        team_z ~ filldist(Normal(0, 1), n_teams)
        team_effects = team_z .* σ_team
    end
    
    # 3. Compute the match-specific weights (η) for every single row
    # Vectorized computation for extreme speed!
    η = w0[market_index] .+ team_effects[home_team_index] .+ team_effects[away_team_index]
    
    # 4. Map η to [0, 1] using logistic
    w_l = logistic.(η)
    
    # 5. Blend the probabilities! 
    p_tilde = w_l .* p_model .+ (1.0 .- w_l) .* q_market
    
    # Prevent explosions in NUTS gradient
    p_tilde_clamped = clamp.(p_tilde, 1e-6, 1.0 - 1e-6)
    
    # 6. The Likelihood
    Turing.@addlogprob! sum(logpdf.(Bernoulli.(p_tilde_clamped), is_winner))
end

# =========================================================
# 4. Stage 3.5: The Training API Hook
# =========================================================

# This is the magic function that the `Training` orchestrator natively calls!
# It unpacks the FeatureSet and passes it cleanly to our @model.
function BayesianFootball.Models.PreGame.build_turing_model(model::TrustModelConfig, f::Features.FeatureSet)
    return trust_engine(
        # Safely convert the Union{Missing, Bool} to strict Vector{Int} for Turing
        Int.(coalesce.(f.data[:is_winner], 0)),
        f.data[:p_model],
        f.data[:q_market],
        f.data[:home_team_index],
        f.data[:away_team_index],
        f.data[:market_index],
        f.data[:n_teams],
        f.data[:n_markets],
        model
    )
end
