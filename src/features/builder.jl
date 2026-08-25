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
    splits::Vector{<:Tuple{Data.SplitBoundary,<:Any}},
    ds::Data.DataStore,
    model::AbstractFootballModel,
    dynamics_col::Symbol=:match_month,
)
    raw_vector = [
        (create_features(boundary, ds, model, dynamics_col), meta)
        for (boundary, meta) in splits
    ]
    return FeatureCollection(raw_vector)
end

"Build features using the same effective clock contract as the splitter."
function create_features(
    splits::Vector{<:Tuple{Data.SplitBoundary,<:Any}},
    ds::Data.DataStore,
    model::AbstractFootballModel,
    splitter::Union{Data.CVConfig,Data.GroupedCVConfig},
)
    raw_vector = [
        let feature_set = create_features(boundary, ds, model, splitter.dynamics_col)
            _align_splitter_time!(
                feature_set, boundary, meta, ds, splitter.dynamics_col)
            (feature_set, meta)
        end
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
    length(unique(all_ids)) == length(all_ids) || error(
        "SplitBoundary history/target match IDs must be unique and disjoint")
    id_set = Set(Int.(all_ids))

    # 2. Extract just the matches for this specific fold.
    matches_df = subset(ds.matches, :match_id => ByRow(id -> Int(id) in id_set))
    nrow(matches_df) == length(all_ids) || error(
        "SplitBoundary resolves $(nrow(matches_df)) rows for $(length(all_ids)) match IDs")
    
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
    # Assign row-by-row. Appending group-sized runs is incorrect whenever physical DataFrame
    # order is not already group-major (pooled stores are normally tournament-major).
    history_steps = sort(unique(history_df.season))
    history_state = Dict(step => i for (i, step) in enumerate(history_steps))
    history_indices = Int[history_state[row.season] for row in eachrow(history_df)]

    target_steps = sort(unique(target_df[!, dynamics_col]))
    target_state = Dict(step => i for (i, step) in enumerate(target_steps))
    target_indices = Int[
        length(history_steps) + target_state[row[dynamics_col]] for row in eachrow(target_df)
    ]

    F_data[:time_indices] = vcat(history_indices, target_indices)
    F_data[:n_history_steps] = length(history_steps)
    F_data[:n_target_steps] = length(target_steps)
    F_data[:n_rounds] = length(history_steps) + length(target_steps)
    F_data[:ordered_match_ids] = ordered_ids

    # --- Stash the fold split itself ---
    # Most extractors only need `ordered_ids`, but any feature that FITS something (rather than
    # looking it up) must know which of those matches it is allowed to learn from. The plus-minus
    # RAPM ridge is the first such feature (see `AbstractPlusMinusFeature.fit_on`). Cheap and
    # backward-compatible — nothing else reads these keys.
    F_data[:history_match_ids] = Set(Int.(boundary.history_match_ids))
    F_data[:target_match_ids]  = Set(Int.(boundary.target_match_ids))

    # 5. DYNAMIC PIPELINE
    # The model asks for features, and we dispatch to add_feature! overloads
    for config in required_features(model)
        add_feature!(F_data, config, ordered_ids, team_map, ds)
    end

    return FeatureSet(F_data)
end

"Align pooled target rows to the splitter's shared effective clock."
function _align_splitter_time!(
    feature_set::FeatureSet,
    boundary::Data.SplitBoundary,
    meta,
    ds::Data.DataStore,
    dynamics_col::Symbol,
)
    meta isa Data.GroupedSplitMetaData || return feature_set
    length(meta.tournament_ids) > 1 || return feature_set

    history_ids = Set(Int.(boundary.history_match_ids))
    target_ids = Set(Int.(boundary.target_match_ids))
    all_ids = union(history_ids, target_ids)
    matches_df = subset(ds.matches, :match_id => ByRow(id -> Int(id) in all_ids))
    history_df = subset(matches_df, :match_id => ByRow(id -> Int(id) in history_ids))
    target_df = subset(matches_df, :match_id => ByRow(id -> Int(id) in target_ids))

    history_steps = sort(unique(history_df.season))
    history_state = Dict(step => i for (i, step) in enumerate(history_steps))
    history_indices = Int[history_state[row.season] for row in eachrow(history_df)]

    raw_step_by_id = Data._effective_step_map(
        ds.matches, meta.tournament_ids, meta.target_season, dynamics_col)
    raw_target_steps = sort(unique(
        raw_step_by_id[Int(id)] for id in target_df.match_id))
    target_state = Dict(step => i for (i, step) in enumerate(raw_target_steps))
    target_indices = Int[
        length(history_steps) + target_state[raw_step_by_id[Int(id)]]
        for id in target_df.match_id
    ]

    feature_set.data[:time_indices] = vcat(history_indices, target_indices)
    feature_set.data[:n_history_steps] = length(history_steps)
    feature_set.data[:n_target_steps] = length(raw_target_steps)
    feature_set.data[:n_rounds] = length(history_steps) + length(raw_target_steps)
    feature_set.data[:ordered_match_ids] = Int.(vcat(history_df.match_id, target_df.match_id))
    feature_set.data[:effective_target_steps] = Dict(
        Int(id) => raw_step_by_id[Int(id)] for id in target_df.match_id)
    return feature_set
end
