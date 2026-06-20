# src/data/fetchers/sql/statistics.jl

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::StatisticsData)
    query = """
    SELECT DISTINCT 
        m.match_id,
        m.tournament_id,
        m.season_id,
        s.period,
        s.stat_key,
        s.home_value,
        s.away_value
    FROM sofascore.match_statistics s
    JOIN sofascore.matches m ON s.match_id = m.match_id
    WHERE m.tournament_id = ANY(\$1)
    """
    try
        return DataFrame(LibPQ.execute(conn, query, [t_ids]))
    catch e
        @warn "Failed to fetch StatisticsData: $(e)"
        return DataFrame()
    end
end

function process_data(df::DataFrame, ::StatisticsData)
    # 1. Unstack the HOME values
    home_wide = unstack(
        df,
        [:match_id, :tournament_id, :season_id, :period], 
        :stat_key,                 
        :home_value,                                      
        renamecols = x -> "$(x)_home"           
    )

    # 2. Unstack the AWAY values
    away_wide = unstack(
        df,
        [:match_id, :tournament_id, :season_id, :period], 
        :stat_key,                              
        :away_value,                                      
        renamecols = x -> "$(x)_away"                 
    )

    # 3. Join them back together
    final_df = innerjoin(
        home_wide, 
        away_wide, 
        on = [:match_id, :tournament_id, :season_id, :period]
    )

    # 4. Apply Schema
    schema = Dict{Symbol, Type}(
        :match_id => Int32,
        :tournament_id => Int32,
        :season_id => Int32,
        :period => InlineStrings.String31
    )
    # Everything else is Union{Missing, Float64}
    for col in names(final_df)
        sym = Symbol(col)
        if !haskey(schema, sym)
            schema[sym] = Union{Missing, Float64}
        end
    end
    apply_schema!(final_df, schema)
    
    return final_df
end

function validate_data(df::DataFrame, ::StatisticsData)
    if !("match_id" in names(df)) || !("period" in names(df))
        @error "StatisticsData QA Failed: Missing base identifiers."
        return false
    end
    return true
end
