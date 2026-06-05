# current_development/l01_momentum.jl

using LibPQ
using DataFrames
using JSON3

"""
    connect_to_db(conn_str::String) -> LibPQ.Connection

Establish a connection to the SofaScore PostgreSQL database.
"""
function connect_to_db(conn_str::String="postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")::LibPQ.Connection
    return LibPQ.Connection(conn_str)
end

"""
    fetch_momentum_data(conn::LibPQ.Connection; tournament_ids::Union{Nothing, Vector{Int}}=nothing) -> DataFrame

Query the `match_graph` table (joined with `matches`) for tournament IDs.
If `tournament_ids` is `nothing` or empty, retrieve all available records to build a comprehensive feature set.
"""
function fetch_momentum_data(conn::LibPQ.Connection; tournament_ids::Union{Nothing, Vector{Int}}=nothing)::DataFrame
    if isnothing(tournament_ids) || isempty(tournament_ids)
        # Query all records
        query = """
        SELECT 
          mg.match_id,
          mg.points 
        FROM 
          match_graph as mg
        INNER JOIN 
          matches as mm on mg.match_id = mm.match_id
        ORDER BY 
          mg.match_id ASC
        """
        return DataFrame(LibPQ.execute(conn, query))
    else
        # Query for specific tournament IDs
        query = """
        SELECT 
          mg.match_id,
          mg.points 
        FROM 
          match_graph as mg
        INNER JOIN 
          matches as mm on mg.match_id = mm.match_id
        WHERE 
          mm.tournament_id = ANY(\$1)
        ORDER BY 
          mg.match_id ASC
        """
        return DataFrame(LibPQ.execute(conn, query, [tournament_ids]))
    end
end

"""
    parse_points_to_vector(points_str::Union{Missing, AbstractString}) -> Union{Missing, Vector{Int}}

Parse raw JSON string `points` into a 1-based momentum vector, where indices represent the rounded minutes.
"""
function parse_points_to_vector(points_str::Union{Missing, AbstractString})::Union{Missing, Vector{Int}}
    if ismissing(points_str) || isempty(strip(points_str))
        return missing
    end
    try
        parsed = JSON3.read(points_str)
        if isempty(parsed)
            return Int[]
        end
        
        idx_vals = [round(Int, pt.minute) for pt in parsed]
        max_idx = isempty(idx_vals) ? 1 : maximum(idx_vals)
        vec_len = max(1, max_idx)
        
        vec = zeros(Int, vec_len)
        for pt in parsed
            idx = max(1, round(Int, pt.minute))
            v = Int(pt.value)
            vec[idx] = v
        end
        return vec
    catch e
        @warn "Failed to parse points JSON" exception=e points_str
        return missing
    end
end

"""
    compute_time_weighted_auc(momentum_vector::Union{Missing, Vector{Int}}; decay_rate::Float64=0.03) -> Tuple{Union{Missing, Float64}, Union{Missing, Float64}}

Implement a time-weighted AUC function:
- Home team area: sum of max(0, v_t) * w_t
- Away team area: sum of max(0, -v_t) * w_t
- Time-weight w_t: exponential decay weighting later minutes higher, e.g., w_t = exp(-decay_rate * (T - t)).
Allow decay_rate to be customizable.
"""
function compute_time_weighted_auc(momentum_vector::Union{Missing, Vector{Int}}; decay_rate::Float64=0.03)::Tuple{Union{Missing, Float64}, Union{Missing, Float64}}
    if ismissing(momentum_vector)
        return missing, missing
    end
    T = length(momentum_vector)
    if T == 0
        return 0.0, 0.0
    end
    
    home_auc = 0.0
    away_auc = 0.0
    
    for t in 1:T
        v_t = momentum_vector[t]
        w_t = exp(-decay_rate * (T - t))
        home_auc += max(0.0, Float64(v_t)) * w_t
        away_auc += max(0.0, Float64(-v_t)) * w_t
    end
    
    return home_auc, away_auc
end

"""
    build_momentum_features(df::DataFrame; decay_rate::Float64=0.03) -> DataFrame

Takes raw fetched DataFrame (containing match_id and points), parses the points, computes the momentum AUCs,
and returns a DataFrame with columns: match_id, home_momentum_auc, away_momentum_auc.
"""
function build_momentum_features(df::DataFrame; decay_rate::Float64=0.03)::DataFrame
    # Parse points column to vectors
    momentum_vectors = parse_points_to_vector.(df.points)
    
    # Compute AUC for each momentum vector
    aucs = compute_time_weighted_auc.(momentum_vectors; decay_rate=decay_rate)
    
    # Extract home and away AUC
    home_aucs = [ismissing(auc) ? missing : auc[1] for auc in aucs]
    away_aucs = [ismissing(auc) ? missing : auc[2] for auc in aucs]
    
    # Construct resulting DataFrame
    res_df = DataFrame(
        match_id = df.match_id,
        home_momentum_auc = home_aucs,
        away_momentum_auc = away_aucs
    )
    
    return res_df
end
