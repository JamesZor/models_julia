# CV splitting functionality

function TimeSeriesSplits(df, base_seasons, target_season, round_col=:round)
    base_indices = findall(row -> row.season in base_seasons, eachrow(df))
    target_data = filter(row -> row.season == target_season, df)
    target_rounds = sort(unique(target_data[!, round_col]))
    
    return TimeSeriesSplits(df, base_seasons, target_season, round_col, 
                           base_indices, target_rounds)
end

Base.length(splits::TimeSeriesSplits) = length(splits.target_rounds) + 1

function Base.iterate(splits::TimeSeriesSplits, state=1)
    if state > length(splits)
        return nothing
    end
    
    if state == 1
        # First split: just base data
        train_view = @view splits.df[splits.base_indices, :]
        round_info = "base_only"
        return ((train_view, round_info), state + 1)
    else
        # Subsequent splits: base + incremental rounds
        round_idx = state - 1
        current_rounds = splits.target_rounds[1:round_idx]
        
        # Get indices for target rounds from original DataFrame
        target_indices = findall(eachrow(splits.df)) do row
            row.season == splits.target_season && row[splits.round_col] in current_rounds
        end
        
        # Combine base and target indices  
        combined_indices = vcat(splits.base_indices, target_indices)
        
        train_view = @view splits.df[combined_indices, :]
        round_info = round_idx == 1 ? 
                    "base + round_$(splits.target_rounds[round_idx])" :
                    "base + rounds_1-$(splits.target_rounds[round_idx])"
        
        return ((train_view, round_info), state + 1)
    end
end

function time_series_splits(data::DataStore, cv_config::TimeSeriesSplitsConfig)
    return TimeSeriesSplits(data.matches, cv_config.base_seasons, 
                           cv_config.target_season, cv_config.round_col)
end
