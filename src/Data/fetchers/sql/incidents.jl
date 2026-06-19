# src/data/fetchers/sql/incidents.jl

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::IncidentsData)
    query = """
        SELECT 
            i.id, i.match_id, i.incident_type, i.time, i.is_home, i.added_time,
            i.data -> 'player' ->> 'slug' AS player_name,
            i.data -> 'playerIn' ->> 'slug' AS player_in_name,
            i.data -> 'playerOut' ->> 'slug' AS player_out_name,
            i.data -> 'assist1' ->> 'slug' AS assist1_name,
            i.data -> 'assist2' ->> 'name' AS assist2_name,
            i.data ->> 'incidentClass' AS incident_class,
            i.data ->> 'reason' AS reason,
            (i.data ->> 'injury')::boolean AS is_injury,
            (i.data ->> 'rescinded')::boolean AS rescinded,
            i.data ->> 'text' AS period_text,
            (i.data ->> 'timeSeconds')::numeric AS time_seconds
        FROM sofascore.match_incidents i
        JOIN sofascore.matches m ON i.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
    """
    try
        return DataFrame(LibPQ.execute(conn, query, [t_ids]))
    catch e
        @warn "Failed to fetch IncidentsData: $(e)"
        return DataFrame()
    end
end

const INCIDENTS_SCHEMA = Dict{Symbol, Type}(
    :id => Int32,
    :match_id => Int32,
    :incident_type => InlineStrings.String31,
    :time => Union{Missing, Int32},
    :is_home => Union{Missing, Bool},
    :added_time => Union{Missing, Int32},
    :player_name => Union{Missing, String},
    :player_in_name => Union{Missing, String},
    :player_out_name => Union{Missing, String},
    :assist1_name => Union{Missing, String},
    :assist2_name => Union{Missing, String},
    :incident_class => Union{Missing, InlineStrings.String31},
    :reason => Union{Missing, InlineStrings.String31},
    :is_injury => Union{Missing, Bool},
    :rescinded => Union{Missing, Bool},
    :period_text => Union{Missing, InlineStrings.String31},
    :time_seconds => Union{Missing, Float64}
)

function process_data(df::DataFrame, ::IncidentsData)
    apply_schema!(df, INCIDENTS_SCHEMA)
    return df
end

function validate_data(df::DataFrame, ::IncidentsData)
    if !("incident_type" in names(df))
        @error "IncidentsData QA Failed: Missing incident_type."
        return false
    end
    return true
end
