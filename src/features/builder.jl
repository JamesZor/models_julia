# src/features/builder.jl

# ==============================================================================
# RELATIONAL FEATURE BUILDER (DataStore & SplitBoundary Architecture)
# ==============================================================================

"""
    create_features(splits, ds, model, dynamics_col)
The Macro Loop: Iterates over a vector of SplitBoundaries and metadata to 
produce a FeatureCollection.
"""
function create_features(
    splits::Vector{<:Tuple{Data.SplitBoundary, <:Any}}, 
    ds::Data.DataStore, 
    model::AbstractFootballModel,
    dynamics_col::Symbol = :match_month
)
    raw_vector = [
        (create_features(boundary, ds, model, dynamics_col), meta) 
        for (boundary, meta) in splits
    ]
    return FeatureCollection(raw_vector)
end

"""
    create_features(boundary, ds, model, dynamics_col)
The Micro Builder: Extracts all necessary data for a single fold using 
the relational mapping between SplitBoundary and DataStore.
"""
function create_features(
    boundary::Data.SplitBoundary, 
    ds::Data.DataStore, 
    model::AbstractFootballModel,
    dynamics_col::Symbol
)
    F_data = Dict{Symbol, Any}()
    
    # 1. COMBINE IDs for the full sequence (History + Target)
    all_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
    
    # 2. Extract just the matches for this specific fold
    # Using subset for clarity; ensure only the matches we need are processed
    matches_df = subset(ds.matches, :match_id => ByRow(id -> id in all_ids))
    
    # 3. BUILD VOCABULARY (Strings -> Integers)
    # We build the team_map based on all teams present in this specific split
    all_teams = unique(vcat(matches_df.home_team, matches_df.away_team))
    team_map = Dict(name => i for (i, name) in enumerate(sort(all_teams)))
    
    F_data[:n_teams] = length(team_map)
    F_data[:team_map] = team_map

    # 4. GENERATE TIME INDICES & SEASONAL MAPPING
    history_df = subset(matches_df, :match_id => ByRow(id -> id in boundary.history_match_ids))
    target_df  = subset(matches_df, :match_id => ByRow(id -> id in boundary.target_match_ids))
    
    # Order matters: History first, then Target
    ordered_df = vcat(history_df, target_df)
    ordered_ids = Int.(ordered_df.match_id)
    
    # --- Build Season Indices (For intercepts) ---
    unique_seasons = sort(unique(ordered_df.season))
    n_seasons = length(unique_seasons)
    season_map = Dict(s => i for (i, s) in enumerate(unique_seasons))
    
    F_data[:n_seasons] = n_seasons
    F_data[:season_indices] = Int[season_map[s] for s in ordered_df.season]

    # --- Build Time Indices ---
    history_groups = groupby(history_df, :season, sort=true)
    target_groups  = groupby(target_df, dynamics_col, sort=true)
    
    time_indices = Int[]
    t_idx = 1
    
    for g in history_groups
        append!(time_indices, fill(t_idx, nrow(g)))
        t_idx += 1
    end
    n_history = length(history_groups)
    
    for g in target_groups
        append!(time_indices, fill(t_idx, nrow(g)))
        t_idx += 1
    end
    n_target = length(target_groups)
    
    F_data[:time_indices] = time_indices
    F_data[:n_history_steps] = n_history
    F_data[:n_target_steps] = n_target
    F_data[:n_rounds] = n_history + n_target

    # --- Stash the fold split itself ---
    # Most extractors only need `ordered_ids`, but any feature that FITS something (rather than
    # looking it up) must know which of those matches it is allowed to learn from. The plus-minus
    # RAPM ridge is the first such feature: it fits on history only and applies the resulting rating
    # vector to the whole fold. Cheap and backward-compatible — nothing else reads these keys.
    F_data[:history_match_ids] = Set(Int.(boundary.history_match_ids))
    F_data[:target_match_ids]  = Set(Int.(boundary.target_match_ids))

    # 5. DYNAMIC PIPELINE
    # The model asks for features, and we dispatch to add_feature! overloads
    for config in required_features(model)
        add_feature!(F_data, config, ordered_ids, team_map, ds)
    end

    return FeatureSet(F_data)
end
