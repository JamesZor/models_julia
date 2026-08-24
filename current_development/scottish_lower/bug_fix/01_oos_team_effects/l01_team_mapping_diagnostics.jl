# Reusable diagnostics for bug 01: OOS team effects silently disappear.
#
# This loader deliberately does not modify BayesianFootball methods. It reproduces the existing
# lookup and constructs a candidate bridge so the defect can be measured before any fix is installed.
# Everything is namespaced to avoid collisions with the prototype loaders' Main-level constants.

module TeamMappingDiagnostics

using BayesianFootball
using DataFrames
using MCMCChains
using Statistics

const BFData     = BayesianFootball.Data
const BFFeatures = BayesianFootball.Features

export team_index_map,
       current_oos_team_id,
       current_oos_team_index,
       build_name_to_id,
       build_name_to_training_index,
       candidate_oos_team_index,
       mapping_comparison,
       mapping_summary,
       chain_matrix,
       chain_vector,
       raw_centered_team_effects,
       fitted_team_effects,
       posterior_team_contributions,
       contribution_summary

"Return the integer-keyed team map stored by the custom recombination feature builders."
team_index_map(feature_set) = feature_set.data[:team_map]

"Reproduce the current l03/l04/l05 fallback from an OOS row to an internal team ID."
function current_oos_team_id(row, side::Symbol, team_map::AbstractDict)
    id_col = side === :home ? :home_team_id : :away_team_id
    name_col = side === :home ? :home_team : :away_team

    if hasproperty(row, id_col)
        value = getproperty(row, id_col)
        return ismissing(value) ? -1 : Int(value)
    elseif hasproperty(row, name_col)
        # This is the suspected bug: team_map is Dict{Int,Int}, but value is a string.
        return get(team_map, getproperty(row, name_col), -1)
    end
    return -1
end

"Reproduce the second lookup performed by the current adapters."
function current_oos_team_index(row, side::Symbol, team_map::AbstractDict)
    team_id = current_oos_team_id(row, side, team_map)
    return get(team_map, team_id, -1)
end

"Build a stable team-name => internal-team-ID identity table from the cleaned match dataset."
function build_name_to_id(clean_df::AbstractDataFrame)
    required = [:home_team, :away_team, :home_team_id, :away_team_id]
    missing_cols = setdiff(required, propertynames(clean_df))
    isempty(missing_cols) || error("clean_df is missing columns: $(missing_cols)")

    name_to_ids = Dict{String, Set{Int}}()
    for row in eachrow(clean_df)
        push!(get!(name_to_ids, String(row.home_team), Set{Int}()), Int(row.home_team_id))
        push!(get!(name_to_ids, String(row.away_team), Set{Int}()), Int(row.away_team_id))
    end

    ambiguous = Dict(name => ids for (name, ids) in name_to_ids if length(ids) != 1)
    isempty(ambiguous) || error("Team names map to multiple internal IDs: $(ambiguous)")
    return Dict(name => only(ids) for (name, ids) in name_to_ids)
end

"Bridge OOS team names to posterior columns while preserving `-1` for teams absent from training."
function build_name_to_training_index(feature_set)
    team_map = team_index_map(feature_set)
    clean_df = feature_set.data[:clean_df]
    name_to_id = build_name_to_id(clean_df)
    return Dict(name => get(team_map, team_id, -1) for (name, team_id) in name_to_id)
end

candidate_oos_team_index(row, side::Symbol, name_to_index::AbstractDict) = begin
    name_col = side === :home ? :home_team : :away_team
    hasproperty(row, name_col) || return -1
    value = getproperty(row, name_col)
    ismissing(value) ? -1 : get(name_to_index, String(value), -1)
end

"Create a row-level comparison of current and candidate mapping behavior."
function mapping_comparison(oos_df::AbstractDataFrame, feature_set)
    team_map = team_index_map(feature_set)
    name_to_index = build_name_to_training_index(feature_set)

    out = DataFrame(
        match_id = Int[], home_team = String[], away_team = String[],
        current_home_idx = Int[], candidate_home_idx = Int[],
        current_away_idx = Int[], candidate_away_idx = Int[],
    )

    for row in eachrow(oos_df)
        push!(out, (
            Int(row.match_id), String(row.home_team), String(row.away_team),
            current_oos_team_index(row, :home, team_map),
            candidate_oos_team_index(row, :home, name_to_index),
            current_oos_team_index(row, :away, team_map),
            candidate_oos_team_index(row, :away, name_to_index),
        ))
    end
    out.current_both_known = (out.current_home_idx .> 0) .& (out.current_away_idx .> 0)
    out.candidate_both_known = (out.candidate_home_idx .> 0) .& (out.candidate_away_idx .> 0)
    return out
end

"Summarize a mapping comparison without hiding genuinely unseen teams."
function mapping_summary(comparison::AbstractDataFrame)
    n = nrow(comparison)
    n == 0 && return (
        matches = 0, current_known = 0, candidate_known = 0,
        current_known_pct = NaN, candidate_known_pct = NaN,
    )
    current_known = count(comparison.current_both_known)
    candidate_known = count(comparison.candidate_both_known)
    return (
        matches = n,
        current_known = current_known,
        candidate_known = candidate_known,
        current_known_pct = 100 * current_known / n,
        candidate_known_pct = 100 * candidate_known / n,
    )
end

"Convert selected MCMC chain columns into a samples-by-parameters matrix."
function chain_matrix(chain::MCMCChains.Chains, parameter_names::Vector{String})
    array = Array(chain[parameter_names])
    if ndims(array) == 2
        return Matrix(array)
    elseif ndims(array) == 3
        # MCMCChains layout: iterations × parameters × chains. Stack chains as sample rows.
        return reshape(permutedims(array, (1, 3, 2)), :, length(parameter_names))
    end
    error("Unexpected chain array dimensions: $(size(array))")
end

"Convert a scalar chain parameter into one vector containing all draws from all chains."
chain_vector(chain::MCMCChains.Chains, parameter_name::String) =
    vec(Array(chain[parameter_name]))

"Center raw team effects exactly as the custom model does before applying hierarchy scales."
function raw_centered_team_effects(chain::MCMCChains.Chains, n_teams::Int)
    alpha_raw = chain_matrix(chain, ["raw_alpha[$i]" for i in 1:n_teams])
    beta_raw = chain_matrix(chain, ["raw_beta[$i]" for i in 1:n_teams])
    return (
        alpha = alpha_raw .- mean(alpha_raw, dims = 2),
        beta = beta_raw .- mean(beta_raw, dims = 2),
    )
end

"Apply tau scales when present, reproducing fitted-model team effects rather than current extraction."
function fitted_team_effects(chain::MCMCChains.Chains, n_teams::Int)
    centered = raw_centered_team_effects(chain, n_teams)
    chain_names = Set(string.(names(chain)))
    has_tau = "tau_alpha" in chain_names && "tau_beta" in chain_names
    if !has_tau
        return (; centered..., has_tau = false)
    end

    tau_alpha = chain_vector(chain, "tau_alpha")
    tau_beta = chain_vector(chain, "tau_beta")
    size(centered.alpha, 1) == length(tau_alpha) || error("tau/alpha draw count mismatch")
    return (
        alpha = centered.alpha .* reshape(tau_alpha, :, 1),
        beta = centered.beta .* reshape(tau_beta, :, 1),
        has_tau = true,
    )
end

"Measure nonzero home/away team contributions for one OOS fold under a supplied effect matrix."
function posterior_team_contributions(
    oos_df::AbstractDataFrame,
    feature_set,
    effects::NamedTuple,
)
    comparison = mapping_comparison(oos_df, feature_set)
    rows = DataFrame(
        match_id = Int[], home_team = String[], away_team = String[],
        home_idx = Int[], away_idx = Int[],
        mean_home_team_term = Float64[], mean_away_team_term = Float64[],
        median_home_rate_multiplier = Float64[], median_away_rate_multiplier = Float64[],
    )

    for row in eachrow(comparison)
        h_idx, a_idx = row.candidate_home_idx, row.candidate_away_idx
        if h_idx <= 0 || a_idx <= 0
            push!(rows, (row.match_id, row.home_team, row.away_team, h_idx, a_idx,
                         NaN, NaN, NaN, NaN))
            continue
        end

        home_term = effects.alpha[:, h_idx] .- effects.beta[:, a_idx]
        away_term = effects.alpha[:, a_idx] .- effects.beta[:, h_idx]
        push!(rows, (
            row.match_id, row.home_team, row.away_team, h_idx, a_idx,
            mean(home_term), mean(away_term),
            median(exp.(home_term)), median(exp.(away_term)),
        ))
    end
    return rows
end

"Summary used to quantify the scale of team terms omitted by the current adapter."
function contribution_summary(contributions::AbstractDataFrame)
    known = filter(row -> isfinite(row.mean_home_team_term) && isfinite(row.mean_away_team_term), contributions)
    isempty(known) && return (known_matches = 0,)
    multipliers = vcat(known.median_home_rate_multiplier, known.median_away_rate_multiplier)
    terms = vcat(known.mean_home_team_term, known.mean_away_team_term)
    return (
        known_matches = nrow(known),
        mean_abs_log_rate_term = mean(abs.(terms)),
        multiplier_q05 = quantile(multipliers, 0.05),
        multiplier_median = median(multipliers),
        multiplier_q95 = quantile(multipliers, 0.95),
        max_abs_log_rate_term = maximum(abs.(terms)),
    )
end

end # module TeamMappingDiagnostics
