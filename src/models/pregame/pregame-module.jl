# src/models/pregame/PreGame.jl

module PreGame

# We import the Types to extend them, but we don't need Reexport
using ...TypesInterfaces

# Macro libraries MUST be loaded at the top
using Turing, Distributions, DataFrames
using ..MyDistributions 
using ..Features
using LinearAlgebra
using Statistics
using Dates
using MCMCChains

# 2. feature_set updates & Architecture
include("./types.jl")
include("./components/dispersion.jl")
include("./components/interception.jl")
include("./components/home_advantage.jl")
include("./components/copula.jl")
include("./components/dixon_coles.jl")
include("./components/dynamics.jl")
include("./components/kappa.jl")
include("./display.jl")

# Team Level - Standard
include("./engines/team_level/standard/goals.jl")
include("./engines/team_level/standard/goals_market.jl")
include("./engines/team_level/standard/xg.jl")
include("./engines/team_level/standard/xg_market.jl")

# Team Level - Time Decay
include("./engines/team_level/time_decay/goals.jl")
include("./engines/team_level/time_decay/copula_goals.jl")
include("./engines/team_level/time_decay/goals_market.jl")
include("./engines/team_level/time_decay/goals_smile_league.jl")
include("./engines/team_level/time_decay/goals_funnel_league.jl")
include("./engines/team_level/time_decay/xg.jl")
include("./engines/team_level/time_decay/xg_market.jl")

# Player Level - Standard
include("./engines/player_level/standard/xg_market.jl")

# Player Level - Time Decay
include("./engines/player_level/time_decay/xg_market.jl")
include("./engines/player_level/time_decay/hierarchical_xg_market.jl")
include("./engines/player_level/time_decay/outfield_xg_market.jl")
include("./engines/player_level/time_decay/outfield_xg.jl")
include("./engines/player_level/time_decay/outfield_xg_dixon_coles.jl")
include("./engines/player_level/time_decay/fullposition_xg_dixon_coles.jl")
include("./engines/player_level/time_decay/outfield_xg_double_poisson.jl")
include("./engines/player_level/time_decay/outfield_xg_smile_double_poisson.jl")
include("./engines/player_level/time_decay/outfield_xg_double_negbin.jl")
include("./engines/player_level/time_decay/outfield_xg_double_poisson_no_market.jl")
include("./engines/player_level/time_decay/outfield_xg_dixon_coles_no_market.jl")
include("./engines/player_level/time_decay/outfield_bigchance_double_poisson.jl")

export DynamicGoalsModel, DynamicGoalsTimeDecayModel, DynamicMarketGoalsTimeDecayModel, DynamicXGModel, DynamicXGTimeDecayModel, DynamicMarketGoalsModel, DynamicMarketXGModel, DynamicMarketXGTimeDecayModel, DynamicMarketXGPlayerModel, DynamicMarketXGPlayerTimeDecayModel, DynamicMarketXGHierarchicalPlayerTimeDecayModel, DynamicMarketXGOutfieldPlayerTimeDecayModel, DynamicXGOutfieldPlayerTimeDecayModel, DynamicCopulaGoalsTimeDecayModel, DynamicDixonColesXGOutfieldPlayerTimeDecayModel, DynamicDixonColesXGFullPositionPlayerTimeDecayModel, DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel, DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel, DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel, DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel, DynamicDoubleNegBinXGOutfieldPlayerTimeDecayModel, DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel, DynamicDixonColesXGOutfieldPlayerTimeDecayNoMarketModel, DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel
export TimeDecayDynamics, PositionalPlayerDynamics, HierarchicalPlayerDynamicsConfig, OutfieldPlayerDynamicsConfig, HierarchicalFrankCopulaConfig, GlobalFrankCopulaConfig

##

export build_turing_model, extract_parameters

end # module
