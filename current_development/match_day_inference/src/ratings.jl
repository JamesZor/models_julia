# current_development/match_day_inference/src/ratings.jl

using DataFrames
using Statistics
using BayesianFootball.Features
import BayesianFootball.Features: AbstractRatingTracker, BayesianTracker, EWMATracker, LastValueTracker, WindowAverageTracker

# ==========================================
# 1. TRACKER-SPECIFIC LATEST RATING EXTRACITON
# ==========================================

function calculate_latest_player_rating(config::BayesianTracker, ratings::AbstractVector)
    n = length(ratings)
    curr_mean = config.prior_mean
    curr_var = config.prior_var
    for i in 1:n
        obs = ratings[i]
        if !ismissing(obs) && !isnan(obs)
            pred_var = curr_var + config.process_noise
            kalman_gain = pred_var / (pred_var + config.obs_var)
            curr_mean = curr_mean + kalman_gain * (obs - curr_mean)
            curr_var = (1.0 - kalman_gain) * pred_var
        else
            curr_var += config.process_noise
        end
    end
    return curr_mean
end

function calculate_latest_player_rating(config::EWMATracker, ratings::AbstractVector)
    n = length(ratings)
    current_val = NaN
    for i in 1:n
        obs = ratings[i]
        if !ismissing(obs) && !isnan(obs)
            if isnan(current_val)
                current_val = Float64(obs)
            else
                current_val = (config.alpha * obs) + ((1.0 - config.alpha) * current_val)
            end
        end
    end
    return current_val
end

function calculate_latest_player_rating(config::LastValueTracker, ratings::AbstractVector)
    clean_ratings = filter(x -> !ismissing(x) && !isnan(x), ratings)
    return isempty(clean_ratings) ? NaN : Float64(last(clean_ratings))
end

function calculate_latest_player_rating(config::WindowAverageTracker, ratings::AbstractVector)
    n = length(ratings)
    start_idx = max(1, n - config.window_size + 1)
    end_idx = n
    if end_idx >= start_idx
        window = ratings[start_idx:end_idx]
        clean_window = filter(x -> !ismissing(x) && !isnan(x), window)
        if !isempty(clean_window)
            return config.agg_func(clean_window)
        end
    end
    return NaN
end

# Generic fallback for any other tracker
function calculate_latest_player_rating(config::AbstractRatingTracker, ratings::AbstractVector)
    clean_ratings = filter(x -> !ismissing(x) && !isnan(x), ratings)
    return isempty(clean_ratings) ? NaN : Float64(mean(clean_ratings))
end

# ==========================================
# 2. RUN PIPELINE & BUILD RATINGS MAP
# ==========================================

"""
    clean_pos(pos::String)

Normalises raw position labels (SofaScore / DB variants) to one of "G","D","M","F".
Unknown labels default to "M".
"""
function clean_pos(pos::String)
    if pos == "G" || pos == "Goalkeeper" || pos == "GK"
        return "G"
    elseif pos == "D" || pos == "Defender" || pos == "DF"
        return "D"
    elseif pos == "M" || pos == "Midfielder" || pos == "MF"
        return "M"
    elseif pos == "F" || pos == "Forward" || pos == "FW" || pos == "A"
        return "F"
    else
        return "M" # Default to Midfielder
    end
end

"""
    get_latest_player_ratings(ds::Data.DataStore, tracker::AbstractRatingTracker)

Computes the latest ratings for all players based on the historical datastore and the rating tracker.
"""
function get_latest_player_ratings(ds::Data.DataStore, tracker::AbstractRatingTracker)
    lineups = DataFrames.select(ds.lineups, :match_id, :player_id, :team_side, :position, :rating, :minutes_played)
    matches_dates = DataFrames.select(ds.matches, :match_id, :match_date)
    df_lineups = innerjoin(lineups, matches_dates, on = :match_id)
    sort!(df_lineups, :match_date)

    valid_ratings = filter(x -> !ismissing(x) && !isnan(x), df_lineups.rating)
    global_avg = isempty(valid_ratings) ? 6.0 : mean(valid_ratings)

    gdf = groupby(df_lineups, :player_id)
    latest_ratings = Dict{Int, Float64}()

    for df_p in gdf
        player_id = df_p.player_id[1]
        latest_r = calculate_latest_player_rating(tracker, df_p.rating)
        
        if isnan(latest_r)
            if hasproperty(tracker, :prior_mean)
                latest_r = tracker.prior_mean
            else
                latest_r = global_avg
            end
        end
        latest_ratings[player_id] = latest_r
    end

    return latest_ratings, global_avg
end

"""
    build_matchday_ratings_map(ds::Data.DataStore, tracker::AbstractRatingTracker, todays_matches::AbstractDataFrame, json_dir::String)

Computes latest player ratings and aggregates them by position for each matchday lineup.
"""
function build_matchday_ratings_map(ds::Data.DataStore, tracker::AbstractRatingTracker, todays_matches::AbstractDataFrame, json_dir::String)
    println("└── [Ratings] Computing latest player ratings from historical lineups...")
    player_ratings, global_avg = get_latest_player_ratings(ds, tracker)
    println("    └─ Global average player rating: ", round(global_avg, digits=3))
    println("    └─ Total tracked players: ", length(player_ratings))

    ratings_map = Dict{Int, Dict{Tuple{String, String}, Float64}}()

    for row in eachrow(todays_matches)
        mid = Int(row.match_id)
        home = String(row.home_team)
        away = String(row.away_team)
        
        println("└── [Fixture] Match ID: $mid | $home vs $away")
        
        # Load lineups
        lineup = get_matchday_lineup(ds, mid, home, away, json_dir)
        
        m_ratings = Dict{Tuple{String, String}, Float64}()
        
        # Helper to compute positional sums for starters (substitute == false)
        for (side, players) in [("home", lineup.home), ("away", lineup.away)]
            # Filter to starters
            starters = filter(p -> !p.substitute, players)
            
            if isempty(starters)
                @warn "No starters found for $side ($side == home ? $home : $away) in match $mid! Using default 0.0 for all positions."
                for pos in ["G", "D", "M", "F"]
                    m_ratings[(side, pos)] = 0.0
                end
                continue
            end
            
            # Group starters by clean position and sum their ratings
            pos_sums = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
            
            for p in starters
                p_id = p.player_id
                c_pos = clean_pos(p.position)
                
                rating = get(player_ratings, p_id, global_avg)
                
                # Report debutants for transparency
                if !haskey(player_ratings, p_id)
                    println("    └─ [Debut] Player ID: $p_id | $(p.player_name) ($side, $c_pos) - setting to fallback: $global_avg")
                end
                
                pos_sums[c_pos] += rating
            end
            
            for (pos, val) in pos_sums
                m_ratings[(side, pos)] = val
            end
            
            # Print team summary
            println("    └─ $side starters (N=$(length(starters))): G=$(round(pos_sums["G"], digits=1)), D=$(round(pos_sums["D"], digits=1)), M=$(round(pos_sums["M"], digits=1)), F=$(round(pos_sums["F"], digits=1))")
        end
        
        ratings_map[mid] = m_ratings
    end

    return ratings_map
end

# ==========================================
# 3. LINEUP COMPARISON DIAGNOSTIC
# ==========================================

"""
    _starters_rating_table(players, player_ratings, global_avg)

Builds a DataFrame of starters (substitute == false) with their tracked rating and a
debut flag (player unseen in history → falls back to `global_avg`).
"""
function _starters_rating_table(players, player_ratings::Dict, global_avg::Float64)
    starters = filter(p -> !p.substitute, players)
    rows = map(starters) do p
        (
            player_id   = p.player_id,
            player_name = p.player_name,
            position    = clean_pos(p.position),
            rating      = round(get(player_ratings, p.player_id, global_avg), digits=3),
            debut       = !haskey(player_ratings, p.player_id),
        )
    end
    return isempty(rows) ? DataFrame(player_id=Int[], player_name=String[], position=String[], rating=Float64[], debut=Bool[]) :
                           DataFrame(rows)
end

_pos_sums(tbl::AbstractDataFrame) = Dict(pos => sum(tbl.rating[tbl.position .== pos]; init=0.0) for pos in ["G","D","M","F"])

"""
    compare_matchday_lineups(ds, tracker, match_id, home_team, away_team, json_dir; verbose=true)

For a single fixture, fetch BOTH the provisional XI (`sofascore.lineup_provisional`) and the
fallback "most recent historical XI", attach each player's tracked rating, and compare them.

Returns a NamedTuple:
  - `home_provisional`, `home_fallback`, `away_provisional`, `away_fallback` :: player-level DataFrames
  - `summary` :: positional-sum DataFrame (side, position, provisional, fallback, delta)

The positional sums are exactly what feed the model (per side, sum of starter ratings by
position), so `delta` shows how much the model's inputs shift between the two lineup sources.
"""
function compare_matchday_lineups(ds::Data.DataStore, tracker::AbstractRatingTracker,
                                  match_id::Int, home_team::String, away_team::String,
                                  json_dir::String; verbose::Bool=true)
    player_ratings, global_avg = get_latest_player_ratings(ds, tracker)

    prov = fetch_provisional_lineup(match_id)
    prov_lu = isnothing(prov) ? (home = [], away = []) : (home = prov.home, away = prov.away)
    fb_lu   = (home = get_most_recent_lineup(ds, home_team),
               away = get_most_recent_lineup(ds, away_team))

    tables = Dict{Tuple{String,Symbol}, DataFrame}()
    for (side, hometeam) in [("home", home_team), ("away", away_team)]
        prov_players = getfield(prov_lu, Symbol(side))
        fb_players   = getfield(fb_lu,   Symbol(side))
        tables[(side, :provisional)] = _starters_rating_table(prov_players, player_ratings, global_avg)
        tables[(side, :fallback)]    = _starters_rating_table(fb_players,   player_ratings, global_avg)
    end

    # Positional-sum summary (model inputs)
    summary_rows = NamedTuple[]
    for side in ["home", "away"]
        ps = _pos_sums(tables[(side, :provisional)])
        fs = _pos_sums(tables[(side, :fallback)])
        for pos in ["G","D","M","F"]
            push!(summary_rows, (side = side, position = pos,
                                 provisional = round(ps[pos], digits=2),
                                 fallback = round(fs[pos], digits=2),
                                 delta = round(ps[pos] - fs[pos], digits=2)))
        end
        push!(summary_rows, (side = side, position = "TOTAL",
                             provisional = round(sum(values(ps)), digits=2),
                             fallback = round(sum(values(fs)), digits=2),
                             delta = round(sum(values(ps)) - sum(values(fs)), digits=2)))
    end
    summary = DataFrame(summary_rows)

    if verbose
        println("\n", "="^90)
        println(" LINEUP COMPARISON | match $match_id | $home_team vs $away_team")
        println(" Global avg rating fallback: $(round(global_avg, digits=3))")
        if isnothing(prov)
            println(" ⚠ No provisional lineup in DB — provisional side is empty (would use fallback live).")
        elseif !prov.confirmed
            println(" ℹ Provisional lineup is NOT yet confirmed (predicted XI).")
        else
            println(" ✓ Provisional lineup is confirmed.")
        end
        println("="^90)
        for side in ["home", "away"]
            team = side == "home" ? home_team : away_team
            println("\n── $side: $team ──")
            println("  [provisional XI]")
            show(tables[(side, :provisional)]; allrows=true, allcols=true); println()
            println("  [fallback / most-recent XI]")
            show(tables[(side, :fallback)]; allrows=true, allcols=true); println()
        end
        println("\n── positional-sum summary (model inputs) ──")
        show(summary; allrows=true, allcols=true); println()
        println("="^90)
    end

    return (
        home_provisional = tables[("home", :provisional)],
        home_fallback    = tables[("home", :fallback)],
        away_provisional = tables[("away", :provisional)],
        away_fallback    = tables[("away", :fallback)],
        summary          = summary,
    )
end

"""
    compare_matchday_lineups(ds, tracker, match_row, json_dir; kwargs...)

Convenience method taking a `DataFrameRow` from `todays_matches`.
"""
function compare_matchday_lineups(ds::Data.DataStore, tracker::AbstractRatingTracker,
                                  match_row::DataFrameRow, json_dir::String; kwargs...)
    return compare_matchday_lineups(ds, tracker,
                                    Int(match_row.match_id),
                                    String(match_row.home_team),
                                    String(match_row.away_team),
                                    json_dir; kwargs...)
end
