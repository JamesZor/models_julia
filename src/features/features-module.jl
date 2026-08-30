# src/features/features-module.jl

"""
This module is responsible for transforming raw data from a DataStore
into a model-ready FeatureSet using the new relational SplitBoundary architecture.
"""
module Features

using CSV
using DataFrames
using Dates
using Statistics
using Base.Threads
using ..Data
using ..TypesInterfaces

export FeatureSet, create_features, required_features, add_feature!
export AbstractFeatureConfig, TeamIDsFeature, GoalsFeature, LeagueFeature, XGFeature, ShotsFeature, BigChanceFeature, ShotsInsideBoxFeature, FinalThirdEntriesFeature, TouchesInOppBoxFeature, ShotsFunnelFeature, MarketSmileFeature, TimeIndicesFeature, DatesFeature, MonthFeature, MidweekFeature, PlasticPitchFeature, DistanceFeature, AbstractRatingTracker, PlayerRatingsFeature
export LastValueTracker, WindowAverageTracker, EWMATracker, BayesianTracker
# Plus-minus (RAPM) rating family — one struct per PM target, all sharing one extractor.
export AbstractPlusMinusFeature, ShotsPlusMinusFeature, ShotsOnTargetPlusMinusFeature,
       GoalsPlusMinusFeature, XGPlusMinusFeature, pm_target, rating_base
# Recombination & Open-Play Features
export OpenPlayGoalsFeature, OpenPlayPxGFeature, SquadWealthFeature, RefereeOfficiatingFeature
# Point-in-time proxy-xG form and stint RAPM covariate feeds for the composable count builder.
export PxGFeature, PxGRapmFeature, pxg_match_observations, pxg_rapm_deltas

# Core Architecture
include("./model_requirements.jl")
include("./vocabulary.jl")
include("./map_builders.jl")
include("./builder.jl")

# Relational Extractors
include("./types.jl")
include("./trackers/last_value.jl")
include("./trackers/window_average.jl")
include("./trackers/ewma.jl")
include("./trackers/bayesian.jl")
include("./market_inverse_utils.jl")
include("./plus_minus/plus_minus.jl")
include("./extractors/core_extractors.jl")
include("./extractors/time_extractors.jl")
include("./extractors/distance_extractors.jl")
include("./extractors/stats_extractors.jl")
include("./extractors/bbc_extractors.jl")
include("./extractors/market_extractors.jl")
include("./extractors/player_extractors.jl")
include("./extractors/plus_minus_extractors.jl")
include("./extractors/open_play_extractors.jl")
# Unified-builder covariate feeds. Loaded after plus_minus/ and the shot parser, whose segment,
# shot-xG and ridge machinery both of these reuse verbatim.
include("./pxg.jl")
include("./pxg_rapm.jl")
include("./display.jl")

end # module
