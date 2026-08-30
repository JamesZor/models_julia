"""
This module brings together all the different model types.
"""
module Models

# Models now ONLY depends on the central TypesInterfaces module for its contracts
# and shared structs. It has NO dependency on Features.
using ..TypesInterfaces
using ..MyDistributions
using ..Features

# --- Include and export sub-modules ---
include("pregame/pregame-module.jl")
include("ingame/ingame-module.jl")
include("latents/latents-module.jl")
using .Latents
using .PreGame: CountModelBuilder, PoissonCountModel, NegBinCountModel,
    ComposableCountModel, AbstractCovariateRole, SupremacyRole, LevelRole,
    AbstractCovariateConfig, LogSumWealthFeature, SLFPLogSumWealthFeature,
    AbstractAgeWeightingCurve, RichardsSigmoid, ShiftedGamma, GaussianPrime,
    age_weight, ProductionWealthFeature, WealthCovariate,
    ProductionWealthCovariate, DistanceCovariate,
    covariate_name, covariate_role, covariate_prior, covariate_features,
    covariate_column, covariate_oos, covariate_sides,
    AbstractRateGuard, ClampGuard, NoGuard,
    AbstractObservationConfig, PoissonObservation, NegativeBinomialObservation,
    DixonColesCorrelation, FrankCopulaCorrelation,
    add!, add, replace!, validate, build, build_count_model,
    cb_varinfo_sites, cb_chain_columns, cb_parameter_count,
    GlobalInterception, SeasonalInterception, HierarchicalMonthlyInterception,
    GlobalHomeAdvantage, HierarchicalTeamHomeAdvantage, HierarchicalLeagueHomeAdvantage,
    TimeDecayDynamics, StaticZeroDynamics, PositionalPlayerDynamics,
    GlobalDispersion, HomeAwayDispersion

# Expose the sub-modules and typed posterior API to the rest of the package.
export PreGame, InGame, Latents
export AbstractPosteriorLatents, CountLatents, RecombLatents, SmileLatents
export AbstractLatentFamily, PoissonCountFamily, NegBinCountFamily,
       RecombinationFamily, SmilePoissonFamily, SmileNegBinFamily
export n_matches, n_draws, n_strikes, latent_match_ids, latent_matrices,
       match_index, latent_bytes, latent_allocations, observation_family,
       recomb_total_home, recomb_total_away, smile_intensity
export extract_latents, latent_family, latents_from_legacy_dataframe,
       to_legacy_dataframe
export CountModelBuilder, PoissonCountModel, NegBinCountModel, ComposableCountModel
export AbstractCovariateRole, SupremacyRole, LevelRole
export AbstractCovariateConfig, LogSumWealthFeature, SLFPLogSumWealthFeature,
       AbstractAgeWeightingCurve, RichardsSigmoid, ShiftedGamma, GaussianPrime,
       age_weight, ProductionWealthFeature, WealthCovariate,
       ProductionWealthCovariate, DistanceCovariate
export covariate_name, covariate_role, covariate_prior, covariate_features,
       covariate_column, covariate_oos, covariate_sides
export AbstractRateGuard, ClampGuard, NoGuard
export AbstractObservationConfig, PoissonObservation, NegativeBinomialObservation,
       GlobalDispersion, HomeAwayDispersion,
       DixonColesCorrelation, FrankCopulaCorrelation
export add!, add, replace!, validate, build, build_count_model
export cb_varinfo_sites, cb_chain_columns, cb_parameter_count
export GlobalInterception, SeasonalInterception, HierarchicalMonthlyInterception
export GlobalHomeAdvantage, HierarchicalTeamHomeAdvantage, HierarchicalLeagueHomeAdvantage
export TimeDecayDynamics, StaticZeroDynamics, PositionalPlayerDynamics
export GlobalDispersion, HomeAwayDispersion
# We must re-export the contract function so other modules can use it.
export required_mapping_keys



export model_name, model_parameters

"""
    model_name(model::AbstractFootballModel)::String

Returns the simplified name of the model strategy (e.g., "StaticPoisson").
"""
function model_name(model::AbstractFootballModel)::String
    return string(nameof(typeof(model)))
end


"""
    _clean_param_str(s::String)

Removes module prefixes and type parameters to make config strings readable.
Example: "Distributions.Normal{Float64}(...)" -> "Normal(...)"
"""
function _clean_param_str(s::String)
    # 1. Remove common module prefixes
    s = replace(s, "Distributions." => "")
    s = replace(s, "BayesianFootball." => "")
    s = replace(s, "Base." => "")
    
    # 2. Remove Type Parameters like {Float64}
    # This regex matches "{Float64}" or "{Any}" but keeps the core type name
    s = replace(s, r"\{Float64\}" => "")
    s = replace(s, r"\{Any\}" => "")
    
    # Optional: Remove specific inner type noise if needed
    # s = replace(s, r"\{.*?\}" => "") # Aggressive removal of ALL {...}
    
    return s
end

"""
    model_parameters(model::AbstractFootballModel)::String

Returns a clean string representation of the model's configuration.
"""
function model_parameters(model::AbstractFootballModel)::String
    fields = fieldnames(typeof(model))
    if isempty(fields)
        return "standard"
    end
    
    params = String[]
    for f in fields
        val = getfield(model, f)
        
        # Convert the value to string, then clean it
        raw_str = string(val)
        clean_str = _clean_param_str(raw_str)
        
        push!(params, "$f=$clean_str")
    end
    
    return join(params, ", ")
end

end
