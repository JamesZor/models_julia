# current_development/match_day_inference/l00_matchday_utils.jl

# ==============================================================================
# ARCHITECTURE UPDATE (May 2026)
# ==============================================================================
# The monolithic draft in this file has been split into a clean, modular structure 
# under the current_development/match_day_inference/ directory to support player-level 
# time-decay models and local/database fallbacks for lineups.
#
# NEW FILE STRUCTURE:
# ├── loader.jl              # Includes dependencies and mini-src modules
# ├── r00_matchday_runner.jl # REPL-friendly execution script
# └── src/
#     ├── lineups.jl         # SofaScore JSON parsers and DB fallback lineups
#     ├── ratings.jl         # AbstractRatingTracker chronological ratings extraction
#     ├── inference.jl       # Parameter extraction and posterior PPD simulation
#     └── live_betting.jl    # Redis Betfair live odds streaming and Kelly staking
#
# To load the modules:
#   include("current_development/match_day_inference/loader.jl")
#
# To run the pipeline:
#   include("current_development/match_day_inference/r00_matchday_runner.jl")
# ==============================================================================

# For convenience, load loader.jl directly when including this file:
include("loader.jl")
