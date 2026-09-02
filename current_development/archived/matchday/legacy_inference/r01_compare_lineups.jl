# current_development/match_day_inference/r01_compare_lineups.jl

# ==========================================
# 1. ENVIRONMENT & IMPORTS
# ==========================================
using Pkg
Pkg.activate(".")

using Revise
# Setup CPU thread pinning
using ThreadPinning
pinthreads(:cores)

# Include loader (assumes we are in BayesianFootball workspace)
include("loader.jl")
include("./current_development/match_day_inference/loader.jl")

using DataFrames
using PrettyTables
using Statistics
using Dates

# ==========================================
# 2. LOAD DATASTORE & EXPERIMENT
# ==========================================
println("\n=== 1. Loading Datastore & Experiment ===")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

save_dir::String = "./data/ab_test_hierarchical_player/"
saved_files = Experiments.list_experiments(save_dir, data_dir="")
if isempty(saved_files)
    error("No saved experiments found in $save_dir")
end
expr = Experiments.load_experiment(saved_files, 2)

model = expr.config.model
tracker = model.player_ratings_feature.tracker

# Extract current player ratings from historical lineups
player_ratings, global_avg = get_latest_player_ratings(ds, tracker)

# ==========================================
# 3. FETCH TODAY'S MATCHES
# ==========================================
println("\n=== 2. Fetching Today's Fixtures ===")
todays_matches = fetch_todays_matches(ds)
if isempty(todays_matches)
    println("⚠️ No matches scheduled for today!")
    exit(0)
end

println("Fetched $(nrow(todays_matches)) matches for today:")
for row in eachrow(todays_matches)
    println("  - [$(row.match_id)] $(row.home_team) vs $(row.away_team)")
end

# Helper to clean position
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

# Helper to get Sofascore rating with fallback to team average of rated players
function get_sofa_rating(p, team_starters, default_val=6.8)
    if isnothing(p)
        return 0.0
    end
    val = p.sofascore_rating
    if val > 0.0
        return val
    end
    valid_ratings = Float64[pl.sofascore_rating for pl in team_starters if pl.sofascore_rating > 0.0]
    return isempty(valid_ratings) ? default_val : mean(valid_ratings)
end

# ==========================================
# 4. LINEUP COMPARISON PIPELINE
# ==========================================
println("\n=== 3. Running Lineup Comparison & Analysis ===")

for row in eachrow(todays_matches)
    mid = Int(row.match_id)
    home = String(row.home_team)
    away = String(row.away_team)
    
    println("\n" * "="^80)
    println(" ⚽ MATCH: $home vs $away (ID: $mid)")
    println("="^80)
    
    # 1. Fetch live lineup from Sofascore
    print("  └─ Fetching live lineup from Sofascore API... ")
    live = fetch_lineup_from_sofascore(mid)
    if isnothing(live)
        println("❌ Not available (Match may be too far in the future or lineups not released yet).")
        println("     Comparison is skipped for this fixture.")
        continue
    end
    status_str = live.confirmed ? "✅ CONFIRMED LINEUP" : "⏳ PROVISIONAL LINEUP"
    println("Success! ($status_str)")
    
    # 2. Get fallback lineups from database
    fallback_home = get_most_recent_lineup(ds, home)
    fallback_away = get_most_recent_lineup(ds, away)
    
    # Analyze home and away sides
    for (side, live_players, fallback_players, team_name) in [
        ("Home", live.home, fallback_home, home),
        ("Away", live.away, fallback_away, away)
    ]
        println("\n  --- $side Team: $team_name ---")
        
        # Filter to starters
        live_starters = filter(p -> !p.substitute, live_players)
        fallback_starters = filter(p -> !p.substitute, fallback_players)
        
        if isempty(live_starters)
            println("    ⚠️ Live starters list is empty.")
            continue
        end
        if isempty(fallback_starters)
            println("    ⚠️ Fallback starters list is empty (no history).")
            continue
        end
        
        # Calculate overlap
        live_ids = [p.player_id for p in live_starters]
        fallback_ids = [p.player_id for p in fallback_starters]
        
        overlap_ids = intersect(live_ids, fallback_ids)
        overlap_count = length(overlap_ids)
        overlap_pct = (overlap_count / max(1, length(live_starters))) * 100
        
        # Rotated in (in live starters, but not in fallback starters)
        rotated_in = filter(p -> p.player_id ∉ fallback_ids, live_starters)
        # Rotated out (in fallback starters, but not in live starters)
        rotated_out = filter(p -> p.player_id ∉ live_ids, fallback_starters)
        
        # Print general stats
        println("    📊 Starter Overlap: $overlap_count/$(length(live_starters)) ($(round(overlap_pct, digits=1))%)")
        
        if isempty(rotated_in) && isempty(rotated_out)
            println("    🔄 No player rotations/changes compared to last game.")
        else
            println("    🔄 Lineup Rotations & Swaps (Left: Live In | Right: Fallback Out):")
            
            # Pair them up
            in_remaining = copy(rotated_in)
            out_remaining = copy(rotated_out)
            pairs = Any[]
            
            # 1. Match same positions first
            for pos in ["G", "D", "M", "F"]
                in_idx = findall(p -> clean_pos(p.position) == pos, in_remaining)
                out_idx = findall(p -> clean_pos(p.position) == pos, out_remaining)
                
                num_matches = min(length(in_idx), length(out_idx))
                for idx in 1:num_matches
                    push!(pairs, (in_remaining[in_idx[idx]], out_remaining[out_idx[idx]]))
                end
                
                matched_in_indices = Base.sort(in_idx[1:num_matches], rev=true)
                matched_out_indices = Base.sort(out_idx[1:num_matches], rev=true)
                
                for idx in matched_in_indices
                    deleteat!(in_remaining, idx)
                end
                for idx in matched_out_indices
                    deleteat!(out_remaining, idx)
                end
            end
            
            # 2. Match remaining players regardless of position (tactical cross-position swaps)
            num_cross = min(length(in_remaining), length(out_remaining))
            for idx in 1:num_cross
                push!(pairs, (in_remaining[idx], out_remaining[idx]))
            end
            
            if num_cross > 0
                deleteat!(in_remaining, 1:num_cross)
                deleteat!(out_remaining, 1:num_cross)
            end
            
            # 3. Add leftovers
            for p in in_remaining
                push!(pairs, (p, nothing))
            end
            for p in out_remaining
                push!(pairs, (nothing, p))
            end
            
            # 4. Print the paired swaps
            for (in_p, out_p) in pairs
                in_str = if !isnothing(in_p)
                    is_new = in_p.player_id ∉ keys(player_ratings)
                    rating_val = get(player_ratings, in_p.player_id, global_avg)
                    sofa_val = get_sofa_rating(in_p, live_starters)
                    debut_tag = is_new ? " (DEBUTANT 🌟)" : ""
                    "[$(in_p.player_id)] $(in_p.player_name) [$(in_p.position)] (Our: $(round(rating_val, digits=2)), Sofa: $(round(sofa_val, digits=2))$debut_tag)"
                else
                    "None"
                end
                
                out_str = if !isnothing(out_p)
                    rating_val = get(player_ratings, out_p.player_id, global_avg)
                    sofa_val = get_sofa_rating(out_p, fallback_starters)
                    "[$(out_p.player_id)] $(out_p.player_name) [$(out_p.position)] (Our: $(round(rating_val, digits=2)), Sofa: $(round(sofa_val, digits=2)))"
                else
                    "None"
                end
                
                println("       - $in_str <-> $out_str")
            end
        end
        
        # Aggregate Positional Ratings
        our_ratings_live = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
        our_ratings_fallback = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
        sofa_ratings_live = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
        sofa_ratings_fallback = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
        
        for p in live_starters
            c = clean_pos(p.position)
            our_rating = get(player_ratings, p.player_id, global_avg)
            sofa_rating = get_sofa_rating(p, live_starters)
            our_ratings_live[c] += our_rating
            sofa_ratings_live[c] += sofa_rating
        end
        
        for p in fallback_starters
            c = clean_pos(p.position)
            our_rating = get(player_ratings, p.player_id, global_avg)
            sofa_rating = get_sofa_rating(p, fallback_starters)
            our_ratings_fallback[c] += our_rating
            sofa_ratings_fallback[c] += sofa_rating
        end
        
        # Build comparison table
        comp_data = Matrix{Any}(undef, 4, 7)
        for (i, pos) in enumerate(["G", "D", "M", "F"])
            our_fall = our_ratings_fallback[pos]
            sofa_fall = sofa_ratings_fallback[pos]
            our_live = our_ratings_live[pos]
            sofa_live = sofa_ratings_live[pos]
            
            our_diff = our_live - our_fall
            sofa_diff = sofa_live - sofa_fall
            
            our_diff_str = our_diff > 0.05 ? "+$(round(our_diff, digits=2))" : 
                            (our_diff < -0.05 ? "$(round(our_diff, digits=2))" : "0.0")
            sofa_diff_str = sofa_diff > 0.05 ? "+$(round(sofa_diff, digits=2))" : 
                            (sofa_diff < -0.05 ? "$(round(sofa_diff, digits=2))" : "0.0")
            
            comp_data[i, 1] = pos
            comp_data[i, 2] = round(our_fall, digits=2)
            comp_data[i, 3] = round(sofa_fall, digits=2)
            comp_data[i, 4] = round(our_live, digits=2)
            comp_data[i, 5] = round(sofa_live, digits=2)
            comp_data[i, 6] = our_diff_str
            comp_data[i, 7] = sofa_diff_str
        end
        
        table_format = PrettyTables.TextTableFormat(borders = PrettyTables.text_table_borders__unicode_rounded)
        
        pretty_table(
            comp_data;
            column_labels = ["Pos", "Fallback (Our)", "Fallback (Sofa)", "Live (Our)", "Live (Sofa)", "Delta Our", "Delta Sofa"],
            table_format = table_format,
            alignment = [:c, :r, :r, :r, :r, :c, :c]
        )
    end
end



println("\n" * "="^80)
println("🏁 Lineup analysis complete!")
println("="^80)
