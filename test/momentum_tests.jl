# test/momentum_tests.jl

using Test
using DataFrames

# Include the logic file directly from current_development
include("../current_development/l02_momentum_analysis.jl")

@testset "Momentum Features Module" begin

    @testset "Points JSON Parsing" begin
        # 1. Standard SofaScore JSON format with fractional minutes and positive/negative values
        points_json = "[{\"minute\":1.5,\"value\":10},{\"minute\":2.0,\"value\":-5},{\"minute\":5.0,\"value\":15}]"
        vec = parse_points_to_vector(points_json)
        @test vec isa Vector{Int}
        # Length of vector should correspond to the maximum rounded minute: max(1.5, 2.0, 5.0) -> max(2, 2, 5) -> 5
        @test length(vec) == 5
        @test vec[2] == -5 # 1.5 rounds to 2, but 2.0 also maps to 2 (overwrites or fits index 2)
        @test vec[5] == 15
        @test vec[3] == 0  # No value recorded at minute 3, should be 0

        # 2. Handles missing
        @test ismissing(parse_points_to_vector(missing))

        # 3. Handles empty string
        @test ismissing(parse_points_to_vector(""))
        @test ismissing(parse_points_to_vector("   "))

        # 4. Handles empty JSON array
        @test parse_points_to_vector("[]") == Int[]

        # 5. Handles invalid JSON
        @test ismissing(parse_points_to_vector("{invalid}"))

        # 6. Handles AbstractString (SubString)
        sub_json = SubString(points_json, 1, length(points_json))
        @test parse_points_to_vector(sub_json) isa Vector{Int}

        # 7. Prevents trailing-zeros bug (maximum rounded index dictates the length)
        trailing_zero_json = "[{\"minute\":89.4,\"value\":10}]"
        vec_tz = parse_points_to_vector(trailing_zero_json)
        @test length(vec_tz) == 89
        @test vec_tz[89] == 10
    end

    @testset "Time Weighted AUC Calculation" begin
        # 1. Simple vector: [10, -5, 15]
        # T = 3
        # t = 1: v_1 = 10, w_1 = exp(-0.03 * (3 - 1)) = exp(-0.06)
        # t = 2: v_2 = -5, w_2 = exp(-0.03 * (3 - 2)) = exp(-0.03)
        # t = 3: v_3 = 15, w_3 = exp(-0.03 * (3 - 3)) = exp(0.0) = 1.0
        
        # Home AUC: max(0, 10)*exp(-0.06) + max(0, -5)*exp(-0.03) + max(0, 15)*1.0
        #           = 10*exp(-0.06) + 0 + 15 = 10 * 0.94176453 + 15 = 24.4176453
        # Away AUC: max(0, -10)*exp(-0.06) + max(0, 5)*exp(-0.03) + max(0, -15)*1.0
        #           = 0 + 5*exp(-0.03) + 0 = 5 * 0.97044553 = 4.85222767
        
        vec = [10, -5, 15]
        home_auc, away_auc = compute_time_weighted_auc(vec; decay_rate=0.03)
        
        expected_home = 10.0 * exp(-0.03 * 2) + 15.0
        expected_away = 5.0 * exp(-0.03 * 1)
        
        @test home_auc ≈ expected_home atol=1e-6
        @test away_auc ≈ expected_away atol=1e-6

        # 2. Handles missing vector
        home_auc_m, away_auc_m = compute_time_weighted_auc(missing)
        @test ismissing(home_auc_m)
        @test ismissing(away_auc_m)

        # 3. Empty vector
        h_auc, a_auc = compute_time_weighted_auc(Int[])
        @test h_auc == 0.0
        @test a_auc == 0.0

        # 4. Custom decay rate (e.g. 0.05)
        home_auc_c, away_auc_c = compute_time_weighted_auc(vec; decay_rate=0.05)
        expected_home_c = 10.0 * exp(-0.05 * 2) + 15.0
        expected_away_c = 5.0 * exp(-0.05 * 1)
        @test home_auc_c ≈ expected_home_c atol=1e-6
        @test away_auc_c ≈ expected_away_c atol=1e-6
    end

    @testset "DataFrame Feature Builder" begin
        df = DataFrame(
            match_id = [101, 102, 103],
            points = [
                "[{\"minute\":1,\"value\":5}]",
                missing,
                "[{\"minute\":1,\"value\":-10}]"
            ]
        )
        
        res = build_momentum_features(df; decay_rate=0.03)
        @test names(res) == ["match_id", "home_momentum_auc", "away_momentum_auc"]
        @test nrow(res) == 3
        @test res.match_id == [101, 102, 103]
        
        # Match 101: vec = [5], T = 1. home_auc = 5*1.0 = 5.0, away_auc = 0.0
        @test res.home_momentum_auc[1] ≈ 5.0
        @test res.away_momentum_auc[1] ≈ 0.0
        
        # Match 102: missing points -> missing AUCs
        @test ismissing(res.home_momentum_auc[2])
        @test ismissing(res.away_momentum_auc[2])
        
        # Match 103: vec = [-10], T = 1. home_auc = 0.0, away_auc = 10*1.0 = 10.0
        @test res.home_momentum_auc[3] ≈ 0.0
        @test res.away_momentum_auc[3] ≈ 10.0
    end

    @testset "Momentum Statistical Validation" begin
        # 1. Test pearson_correlation_test
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0] # perfect correlation
        r, p, n, sig = pearson_correlation_test(x, y)
        @test r ≈ 1.0 atol=1e-5
        @test p < 0.05
        @test n == 5
        @test sig == "Yes (p < 0.05)"

        # Test with missing/NaN values
        x_m = [1.0, missing, 3.0, NaN, 5.0]
        y_m = [2.0, 4.0, 6.0, 8.0, 10.0]
        r_m, p_m, n_m, sig_m = pearson_correlation_test(x_m, y_m)
        @test r_m ≈ 1.0 atol=1e-5
        @test n_m == 3

        # 2. Test analyze_game_state_momentum with mock data
        raw_mom = DataFrame(
            match_id = [1, 2],
            points = [
                "[{\"minute\":1,\"value\":10},{\"minute\":2,\"value\":20},{\"minute\":3,\"value\":-10}]", # T=3
                "[{\"minute\":1,\"value\":-10},{\"minute\":2,\"value\":-20}]" # T=2
            ]
        )
        incidents = DataFrame(
            match_id = [1, 2],
            incident_type = ["goal", "card"],
            time = [2, 1],
            added_time = [missing, missing],
            is_home = [true, false],
            incident_class = [missing, missing],
            rescinded = [missing, missing]
        )
        
        res_gs = analyze_game_state_momentum(raw_mom, incidents)
        @test nrow(res_gs) == 1 # Match 2 has no goal, Match 1 has goal at minute 2 (G_1=2 < T=3)
        @test res_gs.match_id[1] == 1
        @test res_gs.first_goal_minute[1] == 2
        @test res_gs.leading_team[1] == "Home"
        @test res_gs.pre_lead_avg_momentum[1] ≈ 15.0
        @test res_gs.post_lead_avg_momentum[1] ≈ -10.0
        @test res_gs.momentum_change[1] ≈ -25.0
    end

end

