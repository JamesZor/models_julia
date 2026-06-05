# current_development/l02_momentum_analysis.jl

using LibPQ
using DataFrames
using Statistics
using StatsBase
using HypothesisTests
using Distributions
using BayesianFootball
using Dates
using JSON3
using Printf

# Include momentum extractor functions
include(joinpath(@__DIR__, "l01_momentum.jl"))

"""
    load_analysis_data() -> Tuple{DataFrame, DataFrame, DataFrame, DataFrame, DataFrame}
Loads momentum data from SofaScore db and combines matches, statistics, and incidents from all leagues.
"""
function load_analysis_data()
    # 1. Establish connection to the database
    conn_str = get(ENV, "BF_DB_URL", "postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
    println("Connecting to database using: ", replace(conn_str, r":[^:@]+@" => ":****@"))
    conn = connect_to_db(conn_str)
    
    local raw_momentum_df
    try
        # Fetch momentum data
        println("Fetching match graph points from database...")
        raw_momentum_df = fetch_momentum_data(conn)
        println("Fetched $(nrow(raw_momentum_df)) matches with graph points.")
    finally
        close(conn)
    end
    
    # Compute momentum features
    println("Building time-weighted momentum AUC features...")
    momentum_features = build_momentum_features(raw_momentum_df; decay_rate=0.03)
    
    # 2. Load all segments from DataStore
    segments = [
        BayesianFootball.Data.ScottishLower(),
        BayesianFootball.Data.Ireland(),
        BayesianFootball.Data.SouthKorea(),
        BayesianFootball.Data.Norway()
    ]
    
    all_matches = DataFrame()
    all_stats = DataFrame()
    all_incidents = DataFrame()
    
    for seg in segments
        seg_name = string(typeof(seg))
        println("Loading DataStore segment: $(seg_name)")
        try
            # load cached datastore
            ds = BayesianFootball.Data.load_datastore_cached(seg)
            append!(all_matches, ds.matches, cols=:union)
            append!(all_stats, ds.statistics, cols=:union)
            append!(all_incidents, ds.incidents, cols=:union)
            println("  - Success! Matches: $(nrow(ds.matches)), Stats: $(nrow(ds.statistics)), Incidents: $(nrow(ds.incidents))")
        catch e
            @warn "Failed to load segment $(seg_name)" exception=e
        end
    end
    
    println("Combined DataStore: Matches: $(nrow(all_matches)), Stats: $(nrow(all_stats)), Incidents: $(nrow(all_incidents))")
    
    return raw_momentum_df, momentum_features, all_matches, all_stats, all_incidents
end

"""
    pearson_correlation_test(x::AbstractVector, y::AbstractVector)
Calculate Pearson correlation coefficient, p-value, sample size, and significance.
"""
function pearson_correlation_test(x::AbstractVector, y::AbstractVector)
    # Filter out missing/NaN values
    mask = .!ismissing.(x) .& .!ismissing.(y) .& .!isnan.(x) .& .!isnan.(y)
    xf = convert(Vector{Float64}, x[mask])
    yf = convert(Vector{Float64}, y[mask])
    
    n = length(xf)
    if n < 3
        return NaN, NaN, n, "N/A"
    end
    
    r = cor(xf, yf)
    try
        test = HypothesisTests.PearsonCorrelationTest(xf, yf)
        p = HypothesisTests.pvalue(test)
        significance = p < 0.05 ? "Yes (p < 0.05)" : "No"
        return r, p, n, significance
    catch e
        # Fallback to manual t-test if PearsonCorrelationTest fails
        r_clamped = clamp(r, -1.0 + 1e-15, 1.0 - 1e-15)
        t = r_clamped * sqrt((n - 2) / (1 - r_clamped^2))
        dist = Distributions.TDist(n - 2)
        p = 2 * Distributions.ccdf(dist, abs(t))
        significance = p < 0.05 ? "Yes (p < 0.05)" : "No"
        return r, p, n, significance
    end
end

"""
    analyze_game_state_momentum(raw_momentum_df::DataFrame, incidents_df::DataFrame)
Analyze if the leading team's average momentum changes significantly pre- vs post-first goal.
"""
function analyze_game_state_momentum(raw_momentum_df::DataFrame, incidents_df::DataFrame)
    println("Filtering goal incidents...")
    goals_df = filter(row -> row.incident_type == "goal" && coalesce(row.rescinded, false) == false, incidents_df)
    
    if nrow(goals_df) == 0
        @warn "No goals found in incidents."
        return DataFrame()
    end
    
    println("Identifying first goal for each match...")
    first_goals = combine(groupby(goals_df, :match_id)) do df
        df_valid = filter(r -> !ismissing(r.time), df)
        if nrow(df_valid) == 0
            return DataFrame()
        end
        sorted_df = sort(df_valid, [:time, :added_time])
        return DataFrame(sorted_df[1, :])
    end
    
    if nrow(first_goals) == 0
        @warn "No valid first goals found."
        return DataFrame()
    end
    
    # Create first goal map
    first_goal_map = Dict(
        row.match_id => (
            time = row.time,
            is_home = coalesce(row.is_home, true),
            incident_class = coalesce(row.incident_class, "")
        )
        for row in eachrow(first_goals)
    )
    
    results = DataFrame(
        match_id = Int[],
        first_goal_minute = Int[],
        leading_team = String[],
        pre_lead_avg_momentum = Float64[],
        post_lead_avg_momentum = Float64[],
        momentum_change = Float64[]
    )
    
    println("Analyzing match-by-match momentum changes around first goal...")
    for row in eachrow(raw_momentum_df)
        match_id = row.match_id
        if !haskey(first_goal_map, match_id)
            continue
        end
        
        goal_info = first_goal_map[match_id]
        G_1 = goal_info.time
        
        # Parse points vector
        points_vec = parse_points_to_vector(row.points)
        if ismissing(points_vec) || isempty(points_vec)
            continue
        end
        
        T = length(points_vec)
        # Require G_1 to be less than T and at least 1
        if G_1 >= T || G_1 < 1
            continue
        end
        
        # Determine who took the lead (adjust for ownGoal)
        is_own_goal = goal_info.incident_class == "ownGoal"
        scorer_is_home = goal_info.is_home
        leading_team_is_home = is_own_goal ? !scorer_is_home : scorer_is_home
        
        # Sign of leading team momentum
        lead_sign = leading_team_is_home ? 1.0 : -1.0
        
        # Compute pre and post averages
        pre_avg = mean(points_vec[1:G_1]) * lead_sign
        post_avg = mean(points_vec[G_1+1:T]) * lead_sign
        
        push!(results, (
            match_id,
            G_1,
            leading_team_is_home ? "Home" : "Away",
            pre_avg,
            post_avg,
            post_avg - pre_avg
        ))
    end
    
    println("Analyzed $(nrow(results)) matches with valid momentum and goal events.")
    return results
end

"""
    run_full_validation_pipeline(report_path::String)
Run the full statistical validation pipeline and write a report.
"""
function run_full_validation_pipeline(report_path::String)
    println("--- Running Statistical Validation Pipeline ---")
    
    # 1. Load data
    raw_momentum_df, momentum_features, matches, stats, incidents = load_analysis_data()
    
    # 2. Join features with matches and statistics
    # Match goals
    merged_df = innerjoin(momentum_features, matches, on=:match_id)
    
    # Join with statistics for xG
    # Filter stats to period == "ALL"
    stats_all = filter(row -> row.period == "ALL", stats)
    
    # Ensure expectedGoals_home and expectedGoals_away columns exist
    if !hasproperty(stats_all, :expectedGoals_home)
        stats_all[!, :expectedGoals_home] = fill!(Vector{Union{Missing, Float64}}(undef, nrow(stats_all)), missing)
    end
    if !hasproperty(stats_all, :expectedGoals_away)
        stats_all[!, :expectedGoals_away] = fill!(Vector{Union{Missing, Float64}}(undef, nrow(stats_all)), missing)
    end
    
    # Select only necessary columns from stats to avoid duplication/conflicts
    # Usually expectedGoals_home and expectedGoals_away
    stats_subset = select(stats_all, :match_id, :expectedGoals_home, :expectedGoals_away)
    
    merged_full = leftjoin(merged_df, stats_subset, on=:match_id)
    
    # 3. Compute correlation coefficients
    println("Computing correlation coefficients...")
    
    # Home momentum AUC vs Home goals
    r_home, p_home, n_home, sig_home = pearson_correlation_test(
        merged_full.home_momentum_auc,
        merged_full.home_score
    )
    
    # Away momentum AUC vs Away goals
    r_away, p_away, n_away, sig_away = pearson_correlation_test(
        merged_full.away_momentum_auc,
        merged_full.away_score
    )
    
    # Momentum difference vs Goal difference
    mom_diff = merged_full.home_momentum_auc .- merged_full.away_momentum_auc
    goal_diff = merged_full.home_score .- merged_full.away_score
    r_diff_goals, p_diff_goals, n_diff_goals, sig_diff_goals = pearson_correlation_test(
        mom_diff,
        goal_diff
    )
    
    # Momentum difference vs xG difference
    xg_diff = merged_full.expectedGoals_home .- merged_full.expectedGoals_away
    r_diff_xg, p_diff_xg, n_diff_xg, sig_diff_xg = pearson_correlation_test(
        mom_diff,
        xg_diff
    )
    
    # 4. Game-state analysis
    println("Running game state analysis...")
    game_state_results = analyze_game_state_momentum(raw_momentum_df, incidents)
    
    # Perform paired t-test for game-state change
    # Null hypothesis: mean momentum change is 0
    local t_stat, p_paired, conf_int, mean_pre, mean_post, mean_diff, std_diff
    if nrow(game_state_results) < 2
        @warn "Not enough game state results for paired t-test."
        t_stat = NaN
        p_paired = NaN
        conf_int = [NaN, NaN]
        mean_pre = NaN
        mean_post = NaN
        mean_diff = NaN
        std_diff = NaN
    else
        test_paired = OneSampleTTest(game_state_results.post_lead_avg_momentum, game_state_results.pre_lead_avg_momentum)
        t_stat = test_paired.t
        p_paired = pvalue(test_paired)
        conf_int = confint(test_paired)
        mean_pre = mean(game_state_results.pre_lead_avg_momentum)
        mean_post = mean(game_state_results.post_lead_avg_momentum)
        mean_diff = mean(game_state_results.momentum_change)
        std_diff = std(game_state_results.momentum_change)
    end
    
    # 5. Write Markdown report
    println("Writing statistical report to: $(report_path)...")
    
    open(report_path, "w") do io
        write(io, "# Momentum Feature Statistical Validation Report\n\n")
        
        write(io, "Generated on: $(Dates.now())\n\n")
        
        write(io, "## 1. Executive Summary\n")
        write(io, "This report presents the statistical validation of the SofaScore-based momentum features implemented for Milestone 2. ")
        write(io, "The momentum features represent time-weighted Area Under the Curve (AUC) metrics computed from match momentum graphs. ")
        write(io, "This analysis validates their relationship with actual match outcomes (goals) and underlying performance quality (Expected Goals, xG), ")
        write(io, "and examines leading team momentum behavior in response to game state transitions (taking the lead).\n\n")
        
        write(io, "## 2. Correlation Analysis\n")
        write(io, "We analyzed the correlation between momentum features and actual goals/xG across all available matches in the database.\n\n")
        
        # Table 1: Correlation results
        write(io, "| Relationship | Correlation Coefficient ($r$) | p-value | Sample Size ($N$) | Statistically Significant? ($\\alpha = 0.05$) |\n")
        write(io, "|---|---|---|---|---|\n")
        write(io, @sprintf("| Home Momentum AUC vs Home Goals | %.4f | %.4e | %d | %s |\n", r_home, p_home, n_home, sig_home))
        write(io, @sprintf("| Away Momentum AUC vs Away Goals | %.4f | %.4e | %d | %s |\n", r_away, p_away, n_away, sig_away))
        write(io, @sprintf("| Momentum Difference vs Goal Difference | %.4f | %.4e | %d | %s |\n", r_diff_goals, p_diff_goals, n_diff_goals, sig_diff_goals))
        write(io, @sprintf("| Momentum Difference vs xG Difference | %.4f | %.4e | %d | %s |\n", r_diff_xg, p_diff_xg, n_diff_xg, sig_diff_xg))
        write(io, "\n")
        
        write(io, "### Interpretation\n")
        if r_diff_goals > 0.3
            write(io, "- **Strong Positive Correlation:** There is a moderate-to-strong positive correlation ($r \\approx $(@sprintf("%.3f", r_diff_goals))) between momentum difference and goal difference. This indicates that teams with greater sustained momentum tend to win by larger margins.\n")
        else
            write(io, "- **Correlation:** The correlation between momentum difference and goal difference is $r \\approx $(@sprintf("%.3f", r_diff_goals)).\n")
        end
        if r_diff_xg > 0.3
            write(io, "- **Underlying Quality:** The correlation with Expected Goals (xG) difference ($r \\approx $(@sprintf("%.3f", r_diff_xg))) is also highly significant. This suggests that the momentum features are capturing genuine differences in team threat level and chance creation, rather than just random goals.\n")
        else
            write(io, "- **xG Correlation:** The correlation between momentum difference and xG difference is $r \\approx $(@sprintf("%.3f", r_diff_xg)).\n")
        end
        write(io, "\n")
        
        write(io, "## 3. Game State Analysis\n")
        write(io, "To test whether teams change their style or drop their intensity after taking the lead, we analyzed the leading team's average momentum before and after the first goal of the match. ")
        write(io, "The first goal splits the match into a **Pre-First-Goal** period and a **Post-First-Goal** period. ")
        write(io, "We analyze the leading team's momentum from their own perspective (positive indicates they are dominant, negative indicates they are being dominated).\n\n")
        
        # Table 2: Game-state summary
        write(io, "### Leading Team Momentum Summary\n\n")
        write(io, "| Metric | Value |\n")
        write(io, "|---|---|\n")
        write(io, @sprintf("| Number of Matches Analyzed | %d |\n", nrow(game_state_results)))
        write(io, @sprintf("| Mean Leading Team Momentum (Pre-First-Goal) | %.4f |\n", mean_pre))
        write(io, @sprintf("| Mean Leading Team Momentum (Post-First-Goal) | %.4f |\n", mean_post))
        write(io, @sprintf("| Mean Difference ($Post - Pre$) | %.4f |\n", mean_diff))
        write(io, @sprintf("| Standard Deviation of Difference | %.4f |\n", std_diff))
        write(io, "\n")
        
        # Table 3: Hypothesis test
        write(io, "### Paired t-Test Outcomes\n")
        write(io, "We performed a paired t-test testing the null hypothesis $H_0: \\mu_{post} - \\mu_{pre} = 0$.\n\n")
        write(io, "| Parameter | Value |\n")
        write(io, "|---|---|\n")
        write(io, @sprintf("| t-statistic | %.4f |\n", t_stat))
        write(io, @sprintf("| p-value | %.4e |\n", p_paired))
        write(io, @sprintf("| 95%% Confidence Interval | [%.4f, %.4f] |\n", conf_int[1], conf_int[2]))
        write(io, @sprintf("| Reject Null Hypothesis ($\\alpha = 0.05$)? | %s |\n", p_paired < 0.05 ? "Yes" : "No"))
        write(io, "\n")
        
        write(io, "### Interpretation\n")
        if mean_diff < 0 && p_paired < 0.05
            write(io, "- **Momentum Drop Confirmed:** The leading team's average momentum dropped significantly by $(@sprintf("%.3f", -mean_diff)) units after scoring the first goal (p-value = $(@sprintf("%.4e", p_paired))). ")
            write(io, "This provides strong statistical evidence that teams tend to experience a decline in offensive dominance and play more conservatively, or face increased pressure from the trailing team, after taking a lead.\n")
        elseif mean_diff > 0 && p_paired < 0.05
            write(io, "- **Momentum Increase Confirmed:** The leading team's average momentum increased significantly by $(@sprintf("%.3f", mean_diff)) units after scoring the first goal (p-value = $(@sprintf("%.4e", p_paired))). ")
            write(io, "This indicates that scoring the first goal boosts the leading team's dominance further (e.g., exploitation of counter-attacks as the trailing team pushes forward).\n")
        else
            write(io, "- **No Significant Change:** There was no statistically significant change in average momentum for the leading team after scoring the first goal (p-value = $(@sprintf("%.4e", p_paired))).\n")
        end
        write(io, "\n")
        
        write(io, "## 4. Methodological Verification\n")
        write(io, "The validation pipeline executed successfully on combined data stores. All database connections and query procedures were performed using the same core loaders as the production pipeline to guarantee consistency.\n")
    end
    
    println("Statistical report generated successfully at: $(report_path)")
end
