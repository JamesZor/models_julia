# Mapping functionality

function MappingFunctions(func::Function)
    return MappingFunctions(func, func)
end

function create_list_mapping(inputs::AbstractVector{<:AbstractString})::Dict{String, Int} 
    all_items = unique(inputs)
    items_to_idx = Dict(item => i for (i, item) in enumerate(all_items))
    return items_to_idx
end 

function MappedData(data::DataStore, transformer::MappingFunctions) 
    all_teams = unique(vcat(data.matches.home_team, data.matches.away_team))
    team_mapping = transformer.team_mapping_func(all_teams)
    
    all_leagues = unique(string.(data.matches.tournament_id))
    league_mapping = transformer.league_mapping_func(all_leagues) 
    
    return MappedData(team_mapping, league_mapping)
end
