# Data loading functionality

function DataFiles(path::String)
    base_dir = path
    match = joinpath(path, "football_data_mixed_matches.csv")
    odds = joinpath(path, "odds.csv")
    return DataFiles(base_dir, match, odds)
end

function DataStore(data_files::DataFiles)
    matches = CSV.read(data_files.match, DataFrame, header=1)
    odds = CSV.read(data_files.odds, DataFrame, header=1)
    return DataStore(matches, odds)
end
