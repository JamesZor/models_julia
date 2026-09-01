module Builder

using Turing
using Distributions
using DataFrames
using Dates
using SpecialFunctions
using Statistics
using MCMCChains
import DynamicPPL

import ....Features
import ....TypesInterfaces

const CB_PG = parentmodule(@__MODULE__)
const CB_Features = Features
const CB_TI = TypesInterfaces

include("components.jl")
include("player_dynamics.jl")
include("builder.jl")
include("engine.jl")
include("equations.jl")

export CountModelBuilder, PoissonCountModel, NegBinCountModel, ComposableCountModel
export AbstractCovariateRole, SupremacyRole, LevelRole
export AbstractPlayerAggregation, OutfieldPlayerAggregation,
       BenchWeightedPlayerAggregation, PositionalPlayerAggregation,
       MinuteWeightedPlayerAggregation, PlayerLineupDynamics
export AbstractCovariateConfig, LogSumWealthFeature, SLFPLogSumWealthFeature,
       AbstractAgeWeightingCurve, RichardsSigmoid, ShiftedGamma, GaussianPrime,
       age_weight, ProductionWealthFeature, WealthCovariate,
       ProductionWealthCovariate, BenchDepthCovariate, DistanceCovariate,
       PxGCovariate, LateGameChanceCovariate, PxGRapmCovariate
export covariate_name, covariate_role, covariate_prior, covariate_features,
       covariate_column, covariate_oos, covariate_sides
export AbstractRateGuard, ClampGuard, NoGuard
export AbstractObservationConfig, PoissonObservation, NegativeBinomialObservation,
       NegBinObservation,
       GlobalDispersion, HomeAwayDispersion,
       DixonColesCorrelation, FrankCopulaCorrelation,
       JointGammaPoissonObservation, JointGammaPoissonDesign,
       CBPoissonFamilyObservation,
       observation_features, observation_design
export add!, add, replace!, validate, build, build_count_model
export cb_covariates, cb_covariate_names, cb_varinfo_sites, cb_chain_columns,
       cb_parameter_count

end
