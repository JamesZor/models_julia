module Latents

using DataFrames
using Dates
using MCMCChains
using Statistics
using Printf

using ...TypesInterfaces
using ..PreGame

include("types.jl")
include("extract.jl")

export AbstractPosteriorLatents, CountLatents, RecombLatents, SmileLatents
export AbstractLatentFamily, PoissonCountFamily, NegBinCountFamily,
       RecombinationFamily, SmilePoissonFamily, SmileNegBinFamily
export n_matches, n_draws, n_strikes, latent_match_ids, latent_matrices,
       match_index, latent_bytes, latent_allocations, observation_family,
       recomb_total_home, recomb_total_away, smile_intensity
export extract_latents, latent_family, latents_from_legacy_dataframe,
       to_legacy_dataframe

end
