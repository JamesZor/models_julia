# current_development/MetaModels/src/types.jl

export AbstractMetaModel, AbstractMetaDynamicsConfig, AbstractMetaHierarchyConfig
export ConvexMixtureMetaModel, AffineCalibrationMetaModel
export MetaGRWDynamicsConfig, GlobalMetaHierarchyConfig, HierarchicalMetaTeamConfig
export MetaModelData

# --- Component Base Types ---
abstract type AbstractMetaDynamicsConfig end
abstract type AbstractMetaHierarchyConfig end

# --- Engine Base Type ---
# Taking inspiration from Distributions.jl, the configuration is encoded
# in the type parameters for zero-cost abstractions and multiple dispatch.
abstract type AbstractMetaModel{
    D<:AbstractMetaDynamicsConfig, 
    H<:AbstractMetaHierarchyConfig
} end

# --- Concrete Engine 1: Convex Mixture ---
"""
    ConvexMixtureMetaModel

Implements the Q_i = θ * p_i + (1-θ) * m_i mixture model.
"""
Base.@kwdef struct ConvexMixtureMetaModel{
    D<:AbstractMetaDynamicsConfig, 
    H<:AbstractMetaHierarchyConfig
} <: AbstractMetaModel{D, H}
    dynamics_config::D
    hierarchy_config::H
end

# --- Concrete Engine 2: Affine Calibration ---
"""
    AffineCalibrationMetaModel

Implements a fully Bayesian Dynamic Platt Scaling (Affine Logistic Shift).
logit(Q_i) = α_t + β_t * logit(p_L1_i)
"""
Base.@kwdef struct AffineCalibrationMetaModel{
    D<:AbstractMetaDynamicsConfig, 
    H<:AbstractMetaHierarchyConfig
} <: AbstractMetaModel{D, H}
    dynamics_config::D
    hierarchy_config::H
end

# --- Concrete Dynamics Components ---
Base.@kwdef struct MetaGRWDynamicsConfig <: AbstractMetaDynamicsConfig
    σ_prior::Float64 = 0.1 # Prior for the GRW drift
end

Base.@kwdef struct MetaTimeDecayDynamicsConfig <: AbstractMetaDynamicsConfig
    half_life_days::Float64 = 60.0
end

# --- Concrete Hierarchy Components ---
struct GlobalMetaHierarchyConfig <: AbstractMetaHierarchyConfig end

Base.@kwdef struct HierarchicalMetaTeamConfig <: AbstractMetaHierarchyConfig
    σ_team_prior::Float64 = 0.1 # Prior for the variance among team biases
end

# --- Data Container ---
"""
    MetaModelData
Data container passed into the Turing model.
"""
struct MetaModelData
    Y::Vector{Int}              # Binary outcome (0 or 1)
    p_L1::Vector{Float64}       # Layer 1 probability prediction
    m_i::Vector{Float64}        # Market probability (vig removed)
    W::Vector{Int}              # Time indices (e.g. weeks)
    home_idx::Vector{Int}       # Team index for home (if applicable)
    away_idx::Vector{Int}       # Team index for away (if applicable)
    n_weeks::Int
    n_teams::Int
end
