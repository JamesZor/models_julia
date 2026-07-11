# src/features/features-module.jl

"""
This module is responsible for transforming raw data from a DataStore
into a model-ready FeatureSet using the new relational SplitBoundary architecture.
"""
module Features

using DataFrames
using Dates
using Base.Threads
using ..Data
using ..TypesInterfaces

export FeatureSet, create_features, required_features, add_feature!
export AbstractFeatureConfig, TeamIDsFeature, GoalsFeature, LeagueFeature, XGFeature, ShotsFeature, BigChanceFeature, ShotsInsideBoxFeature, FinalThirdEntriesFeature, TouchesInOppBoxFeature, MarketSmileFeature, TimeIndicesFeature, DatesFeature, MonthFeature, MidweekFeature, PlasticPitchFeature, AbstractRatingTracker, PlayerRatingsFeature
export LastValueTracker, WindowAverageTracker, EWMATracker, BayesianTracker

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
include("./extractors/core_extractors.jl")
include("./extractors/time_extractors.jl")
include("./extractors/stats_extractors.jl")
include("./extractors/market_extractors.jl")
include("./extractors/player_extractors.jl")
include("./display.jl")

end # module
