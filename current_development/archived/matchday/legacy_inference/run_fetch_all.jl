# current_development/match_day_inference/run_fetch_all.jl
using BayesianFootball
using DataFrames

include("loader.jl")

println("--- Querying today's matches from database ---")
try
    ds = Data.load_datastore_cached(Data.Ireland())
    df = fetch_todays_matches(ds)

    if isempty(df)
        println("No matches found starting today in the database.")
    else
        ids = df.match_id
        println("Found $(length(ids)) matches today: ", join(ids, ", "))
        
        # Paths
        script_path = joinpath(@__DIR__, "fetch_lineups.sh")
        
        # Run fetch_lineups.sh for all match IDs
        # Setting 'dir=@__DIR__' makes sure the script runs inside match_day_inference directory
        # so data is saved directly to current_development/match_day_inference/data/lineups/
        cmd = Cmd(["/bin/bash", script_path, [string(id) for id in ids]...], dir=@__DIR__)
        run(cmd)
    end
catch e
    @error "Failed to fetch matches or run lineups downloader: $e"
end
