# src/Models/PreGame/types.jl

# ==========================================
# 1. ABSTRACT COMPONENT INTERFACES
# ==========================================
abstract type AbstractModelComponent end

abstract type AbstractDispersionConfig <: AbstractModelComponent end
abstract type AbstractHomeAdvantageConfig <: AbstractModelComponent end
abstract type AbstractKappaConfig <: AbstractModelComponent end
abstract type AbstractDynamicsConfig <: AbstractModelComponent end
abstract type AbstractInterceptionConfig <: AbstractModelComponent end
abstract type AbstractCopulaConfig <: AbstractModelComponent end
abstract type AbstractDixonColesConfig <: AbstractModelComponent end
abstract type AbstractRecombinationConfig <: AbstractModelComponent end
abstract type AbstractSquadWealthConfig <: AbstractModelComponent end
abstract type AbstractPxGObservationConfig <: AbstractModelComponent end

# ==========================================
# 2. MASTER ARCHITECTURE TYPES
# ==========================================

# Entity Level Abstractions
abstract type AbstractTeamModel <: AbstractNegBinModel end
abstract type AbstractPlayerModel <: AbstractNegBinModel end

# Implementation Style Abstractions (Team)
abstract type AbstractStandardTeamModel <: AbstractTeamModel end
abstract type AbstractTimeDecayTeamModel <: AbstractTeamModel end

# Implementation Style Abstractions (Player)
abstract type AbstractStandardPlayerModel <: AbstractPlayerModel end
abstract type AbstractTimeDecayPlayerModel <: AbstractPlayerModel end

# Engine model config structs have been moved to their respective engine files 
# in src/Models/PreGame/engines/ to keep them self-contained.
