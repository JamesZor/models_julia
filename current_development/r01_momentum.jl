# current_development/r01_momentum.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using CSV

# Include the loader/logic file
include("l01_momentum.jl")

println("--- Starting Momentum Feature Generation ---")

# 1. Establish connection to the database
conn_str = get(ENV, "BF_DB_URL", "postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
println("Connecting to database...")
conn = connect_to_db(conn_str)

try
    # 2. Fetch momentum data (all available records to build a comprehensive feature set)
    println("Fetching match graph points from database...")
    raw_df = fetch_momentum_data(conn)
    println("Fetched $(nrow(raw_df)) rows from match_graph.")
    
    # 3. Compute momentum features (home/away AUC)
    println("Computing time-weighted momentum AUC features...")
    features_df = build_momentum_features(raw_df; decay_rate=0.03)
    
    # 4. Verify the DataFrame is non-empty and has correct columns
    println("Verifying results...")
    if isempty(features_df)
        error("Error: Computed features DataFrame is empty!")
    end
    
    required_cols = [:match_id, :home_momentum_auc, :away_momentum_auc]
    for col in required_cols
        if !(col in propertynames(features_df))
            error("Error: Required column $col not found in computed features!")
        end
    end
    
    println("Verification passed! Schema check completed.")
    println("Sample of computed features:")
    println(first(features_df, 10))
    
    # 5. Save the resulting DataFrame
    output_path = joinpath(@__DIR__, "momentum_features.csv")
    println("Saving features to: $output_path")
    CSV.write(output_path, features_df)
    println("Saved $(nrow(features_df)) matches' features.")
    
finally
    # Always close database connection
    println("Closing database connection...")
    close(conn)
end

println("--- Momentum Feature Generation Completed Successfully ---")
