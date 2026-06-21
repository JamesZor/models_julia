# src/data/fetchers/segments.jl

# ---------------------------------------------------------
# The Singleton Type Approach (The "Julia" Way for Dispatch)
# ---------------------------------------------------------
struct ScottishLower <: DataTournemantSegment end
struct Ireland       <: DataTournemantSegment end
struct IrelandFirstDivision <: DataTournemantSegment end
struct SouthKorea    <: DataTournemantSegment end
struct Norway    <: DataTournemantSegment end


# HACK: 2026-04-15 - for DataLegacy DataStore
struct DevLegacyDataStore <: DataTournemantSegment end 

"""
    tournament_ids(segment::DataTournemantSegment) -> Vector{Int}
Maps the segment type to the specific tournament IDs in the database.
"""
tournament_ids(::ScottishLower) = [56, 57]
tournament_ids(::Ireland)       = [79]
tournament_ids(::IrelandFirstDivision) = [718]
tournament_ids(::SouthKorea)    = [3284, 6230]
tournament_ids(::Norway)    = [5, 6]

# Fallback method to catch missing definitions
function tournament_ids(segment::DataTournemantSegment)
    error("tournament_ids not defined for segment: $(typeof(segment))")
end
