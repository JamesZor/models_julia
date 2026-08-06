# src/predictions/interface.jl

"""
    extract_params(model::AbstractFootballModel, row)::NamedTuple

Extracts the raw parameter vectors (e.g., λ_h, λ_a, ρ) from a single row 
of the LatentStates DataFrame. This acts as an adapter between the DataFrame 
structure and the math kernels.
"""
function extract_params end



function get_latent_column_symbols end

"""
    compute_score_matrix(model::AbstractFootballModel, params)::AbstractScoreMatrix

Generates the probability score matrix (Home x Away x Samples) given the 
extracted parameter vectors.
"""
function compute_score_matrix end

"""
    compute_market_probs(score_mat::AbstractScoreMatrix, market::AbstractMarket)

Calculates the probabilities (or outcome distributions) for a specific market 
given the score matrix. Returns a Dict of SelectionName => DistributionVector.
"""
function compute_market_probs end

"""
    score_matrix_data(score_mat::AbstractScoreMatrix)

Extracts the 3D score tensor `[max_h × max_a × n_samples]` from any `AbstractScoreMatrix`.
"""
function score_matrix_data end

