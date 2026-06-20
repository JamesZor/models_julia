# current_development/MetaModels/src/components/teams.jl

using Turing

"""
    build_meta_hierarchy(config::AbstractMetaHierarchyConfig, n_teams::Int)

Builds the hierarchical team-level bias component for the Meta Model.
"""
function build_meta_hierarchy end

@model function build_meta_hierarchy(::GlobalMetaHierarchyConfig, n_teams::Int)
    # Global means no team-level bias
    return zeros(n_teams)
end

@model function build_meta_hierarchy(config::HierarchicalMetaTeamConfig, n_teams::Int)
    # Non-centered hierarchical prior for team-specific biases
    σ_team ~ Exponential(config.σ_team_prior)
    z_team ~ filldist(Normal(0, 1), n_teams)
    
    δ_team = z_team .* σ_team
    return δ_team
end
