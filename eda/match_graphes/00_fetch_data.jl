using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


using LibPQ




db_config = Data.DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")




function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int64})
    # Note: We join `markets` to get the `market_type` string
    query = """
    SELECT 
      mg.match_id,
      mg.points 
    FROM 
      match_graph as mg
    INNER join 
      matches as mm on mg.match_id = mm.match_id
    where 
      mm.tournament_id = ANY(\$1)
    order by 
      mg.match_id asc 
"""
    return DataFrame(LibPQ.execute(conn, query, [t_ids]))
end

function featch_match_graph_data(ds::Data.DataStore)
  db_config = Data.DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
  conn = Data.connect_to_db(db_config)
  df = fetch_data(conn, Data.tournament_ids(ds.segment))
  return df
end




df = featch_match_graph_data(ds)

using JSON3

# Function to parse the raw JSON string into a Dict of Minute -> Momentum Value
function parse_match_graph_to_dict(points_str::Union{Missing, String})
    ismissing(points_str) && return missing
    
    parsed = JSON3.read(points_str)
    points_dict = Dict{Float64, Int}()
    
    for pt in parsed
        points_dict[Float64(pt.minute)] = Int(pt.value)
    end
    
    return points_dict
end

# Function to convert the dictionary into a Vector where index = minute
function dict_to_momentum_vector(points_dict::Union{Missing, Dict{Float64, Int}})
    ismissing(points_dict) && return missing
    isempty(points_dict) && return Int[]
    
    # Find the maximum minute to determine vector size
    max_minute = maximum(keys(points_dict))
    vec_len = max(1, ceil(Int, max_minute))
    
    # Initialize the momentum vector with zeros (neutral momentum)
    vec = zeros(Int, vec_len)
    
    for (min_val, val) in points_dict
        # Map fractional minutes (like 45.5 or 90.5) to an integer index
        # Using `round(Int, min_val)` maps 45.5 -> 46, and 90.5 -> 90 or 91
        idx = round(Int, min_val)
        
        # Ensure index is at least 1 (in case there's a minute like 0.0 or 0.4)
        idx = max(1, idx)
        
        # If rounding pushes the index slightly out of bounds, expand the vector
        if idx > length(vec)
            push!(vec, val)
        else
            # We assign the value. If there's a collision (e.g. 45.5 and 46.0 both mapping to 46),
            # this will overwrite, which might be acceptable as 46.0 is the "true" 46th minute
            vec[idx] = val
        end
    end
    
    return vec
end

# 1. Parse string to Dict to preserve all raw fractional minutes (e.g., 45.5, 90.5)
df.points_dict = parse_match_graph_to_dict.(df.points)

# 2. Convert Dict to an indexed Vector (Index = Minute)
df.momentum_vector = dict_to_momentum_vector.(df.points_dict)

# Check the results
# display(df.momentum_vector)

