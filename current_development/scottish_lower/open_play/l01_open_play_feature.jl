# current_development/scottish_lower/open_play/l01_open_play_feature.jl
#
# LOADER 1/2 — Open-Play Target Builder, Referees, & Clean Proxy xG Feature
#
# Removes non-systemic noise (Penalties and Own Goals) from:
#   1. Match Goal Targets: y_np_nog_h, y_np_nog_a
#   2. Shot Commentary Parser: Clean Open-Play pxG (excluding is_penalty shots)
#   3. Match Officials Extraction: Attaches referee name for officiating bias analysis

using DataFrames
using Statistics
using Distributions
using InlineStrings
using LibPQ

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

# ==============================================================================
# 1. REFEREE EXTRACTION
# ==============================================================================

"""
    fetch_match_referees(t_ids::Vector{Int} = [56, 57]) -> DataFrame

Fetches referee assignments from `bbc.match_officials` for the specified tournament IDs.
Returns DataFrame with columns: `:match_id`, `:referee_name`.
"""
function fetch_match_referees(t_ids::Vector{Int} = [56, 57])::DataFrame
    db_config = Data.DBConfig()
    conn = Data.connect_to_db(db_config)
    try
        query = """
            SELECT mo.match_id, mo.name AS referee_name
            FROM bbc.match_officials mo
            JOIN sofascore.matches m ON mo.match_id = m.match_id
            WHERE m.tournament_id = ANY(\$1) AND mo.role = 'referee'
        """
        df = DataFrame(LibPQ.execute(conn, query, [t_ids]))
        return df
    catch e
        @warn "Failed to fetch match referees: $e"
        return DataFrame(match_id = Int32[], referee_name = String[])
    finally
        close(conn)
    end
end

# ==============================================================================
# 2. OPEN-PLAY MATCH TARGET EXTRACTION
# ==============================================================================

"""
    extract_open_play_match_data(ds::Data.DataStore; include_referees::Bool = true) -> DataFrame

Extracts match-level penalty and own-goal counts from `ds.incidents` and computes
clean Non-Penalty, Non-Own-Goal (NP-NOG) open-play goal targets.

Returns a DataFrame with columns:
  :match_id, :season, :match_date, :home_team, :away_team,
  :home_score, :away_score,
  :pen_scored_h, :pen_scored_a,
  :pen_missed_h, :pen_missed_a,
  :og_for_h, :og_for_a,
  :y_np_nog_h, :y_np_nog_a,
  :referee_name (if include_referees=true)
"""
function extract_open_play_match_data(ds::Data.DataStore; include_referees::Bool = true)::DataFrame
    matches = copy(ds.matches)
    incidents = copy(ds.incidents)

    # Initialize count dictionaries
    pen_scored_h = Dict{Int32, Int}()
    pen_scored_a = Dict{Int32, Int}()
    pen_missed_h = Dict{Int32, Int}()
    pen_missed_a = Dict{Int32, Int}()
    og_for_h     = Dict{Int32, Int}()
    og_for_a     = Dict{Int32, Int}()
    cards_h      = Dict{Int32, Int}()
    cards_a      = Dict{Int32, Int}()

    for row in eachrow(incidents)
        m_id = Int32(row.match_id)
        i_type = ismissing(row.incident_type) ? "" : String(row.incident_type)
        i_class = ismissing(row.incident_class) ? "" : String(row.incident_class)
        is_home = coalesce(row.is_home, true)

        if i_type == "goal"
            if i_class == "penalty"
                if is_home
                    pen_scored_h[m_id] = get(pen_scored_h, m_id, 0) + 1
                else
                    pen_scored_a[m_id] = get(pen_scored_a, m_id, 0) + 1
                end
            elseif i_class == "ownGoal"
                if is_home
                    og_for_h[m_id] = get(og_for_h, m_id, 0) + 1
                else
                    og_for_a[m_id] = get(og_for_a, m_id, 0) + 1
                end
            end
        elseif i_type == "inGamePenalty"
            if is_home
                pen_missed_h[m_id] = get(pen_missed_h, m_id, 0) + 1
            else
                pen_missed_a[m_id] = get(pen_missed_a, m_id, 0) + 1
            end
        elseif i_type == "card"
            if is_home
                cards_h[m_id] = get(cards_h, m_id, 0) + 1
            else
                cards_a[m_id] = get(cards_a, m_id, 0) + 1
            end
        end
    end

    # Build vectors aligned with matches
    n = nrow(matches)
    ps_h = zeros(Int, n)
    ps_a = zeros(Int, n)
    pm_h = zeros(Int, n)
    pm_a = zeros(Int, n)
    og_h = zeros(Int, n)
    og_a = zeros(Int, n)
    c_h  = zeros(Int, n)
    c_a  = zeros(Int, n)
    y_np_h = zeros(Int, n)
    y_np_a = zeros(Int, n)

    for i in 1:n
        m_id = Int32(matches.match_id[i])
        ps_h[i] = get(pen_scored_h, m_id, 0)
        ps_a[i] = get(pen_scored_a, m_id, 0)
        pm_h[i] = get(pen_missed_h, m_id, 0)
        pm_a[i] = get(pen_missed_a, m_id, 0)
        og_h[i] = get(og_for_h, m_id, 0)
        og_a[i] = get(og_for_a, m_id, 0)
        c_h[i]  = get(cards_h, m_id, 0)
        c_a[i]  = get(cards_a, m_id, 0)

        # Non-penalty, non-own-goal open-play goals
        raw_h = coalesce(matches.home_score[i], 0)
        raw_a = coalesce(matches.away_score[i], 0)

        y_np_h[i] = max(0, raw_h - ps_h[i] - og_h[i])
        y_np_a[i] = max(0, raw_a - ps_a[i] - og_a[i])
    end

    df = DataFrame(
        match_id      = matches.match_id,
        tournament_id = matches.tournament_id,
        season        = matches.season,
        match_date    = matches.match_date,
        home_team     = matches.home_team,
        away_team     = matches.away_team,
        home_score    = matches.home_score,
        away_score    = matches.away_score,
        pen_scored_h  = ps_h,
        pen_scored_a  = ps_a,
        pen_missed_h  = pm_h,
        pen_missed_a  = pm_a,
        og_for_h      = og_h,
        og_for_a      = og_a,
        cards_h       = c_h,
        cards_a       = c_a,
        y_np_nog_h    = y_np_h,
        y_np_nog_a    = y_np_a
    )

    if include_referees
        ref_df = fetch_match_referees([56, 57])
        if !isempty(ref_df)
            df = leftjoin(df, ref_df, on=:match_id)
            # Fill missing referees with Unknown
            df.referee_name = coalesce.(df.referee_name, "Unknown")
        else
            df.referee_name = fill("Unknown", nrow(df))
        end
    end

    return df
end

# ==============================================================================
# 3. CLEAN OPEN-PLAY PROXY xG FEATURE
# ==============================================================================

"""
    CleanProxyXGFeature(; k = 25.0, fit_on = :global)

Constructs team-match proxy xG strictly from open-play and regular set-piece shots,
excluding penalty kick events (`is_penalty == false`).
"""
struct CleanProxyXGFeature <: Features.AbstractFeatureConfig
    k::Float64
    fit_on::Symbol
end
CleanProxyXGFeature(; k = 25.0, fit_on = :global) = CleanProxyXGFeature(k, fit_on)

"""
    build_clean_open_play_shots(ds::Data.DataStore) -> DataFrame

Parses BBC commentary shot events, strips penalties (`is_penalty == false`), and
predicts empirical-Bayes zonal xG.
"""
function build_clean_open_play_shots(ds::Data.DataStore; k = 25.0)::DataFrame
    all_shots = Features.build_shots(ds)
    if isempty(all_shots)
        return DataFrame()
    end

    # Filter out penalty kick events
    open_play_shots = filter(s -> !s.is_penalty, all_shots)

    # Fit empirical-Bayes cell table on open play shots only
    model = Features.fit_shot_xg(open_play_shots; k = k)
    
    # Predict open play xG
    open_play_shots[!, :pred_xg] = Features.predict_xg(model, open_play_shots)
    return open_play_shots
end

"""
    aggregate_clean_pxg_by_match(ds::Data.DataStore; k = 25.0) -> DataFrame

Aggregates clean open-play shot xG to team-match totals (:clean_pxg_h, :clean_pxg_a).
"""
function aggregate_clean_pxg_by_match(ds::Data.DataStore; k = 25.0)::DataFrame
    shots = build_clean_open_play_shots(ds; k = k)
    matches = copy(ds.matches)

    pxg_h_map = Dict{Int32, Float64}()
    pxg_a_map = Dict{Int32, Float64}()
    n_shots_h = Dict{Int32, Int}()
    n_shots_a = Dict{Int32, Int}()

    if !isempty(shots)
        for row in eachrow(shots)
            m_id = Int32(row.match_id)
            xg = coalesce(row.pred_xg, 0.0)
            is_home = coalesce(row.is_home, true)
            if is_home
                pxg_h_map[m_id] = get(pxg_h_map, m_id, 0.0) + xg
                n_shots_h[m_id] = get(n_shots_h, m_id, 0) + 1
            else
                pxg_a_map[m_id] = get(pxg_a_map, m_id, 0.0) + xg
                n_shots_a[m_id] = get(n_shots_a, m_id, 0) + 1
            end
        end
    end

    n = nrow(matches)
    clean_pxg_h = fill(NaN, n)
    clean_pxg_a = fill(NaN, n)
    shots_count_h = zeros(Int, n)
    shots_count_a = zeros(Int, n)

    for i in 1:n
        m_id = Int32(matches.match_id[i])
        if haskey(pxg_h_map, m_id)
            clean_pxg_h[i] = round(pxg_h_map[m_id], digits=4)
            shots_count_h[i] = n_shots_h[m_id]
        end
        if haskey(pxg_a_map, m_id)
            clean_pxg_a[i] = round(pxg_a_map[m_id], digits=4)
            shots_count_a[i] = n_shots_a[m_id]
        end
    end

    return DataFrame(
        match_id       = matches.match_id,
        clean_pxg_h    = clean_pxg_h,
        clean_pxg_a    = clean_pxg_a,
        clean_shots_h  = shots_count_h,
        clean_shots_a  = shots_count_a
    )
end
