# test/data_tests.jl

using Test
using BayesianFootball
using DataFrames
using Dates
using InlineStrings

@testset "Data Module" begin
    
    @testset "Fractional to Decimal Parsing" begin
        # Valid cases
        @test BayesianFootball.Data.parse_fractional_to_decimal("1/1") == 2.0
        @test BayesianFootball.Data.parse_fractional_to_decimal("1/2") == 1.5
        @test BayesianFootball.Data.parse_fractional_to_decimal("5/2") == 3.5
        @test BayesianFootball.Data.parse_fractional_to_decimal("100/30") ≈ 4.333 atol=1e-3
        
        # Invalid cases
        @test BayesianFootball.Data.parse_fractional_to_decimal("SP") == 0.0
        @test BayesianFootball.Data.parse_fractional_to_decimal("1") == 0.0
        @test BayesianFootball.Data.parse_fractional_to_decimal(missing) == 0.0
    end

    @testset "Lineup market values" begin
        raw = DataFrame(
            tournament_id = [56, 56],
            season_id = [1, 1],
            match_id = [1001, 1001],
            team_side = ["home", "away"],
            player_id = [101, 202],
            proposed_market_value = Union{Missing, Int64}[250_000, missing],
            proposed_market_value_currency = Union{Missing, String}["EUR", missing],
            totalPass = Union{Missing, Float64}[42.0, missing],
        )

        processed = BayesianFootball.Data.process_data(
            raw, BayesianFootball.Data.LineUpsData())

        @test eltype(processed.proposed_market_value) == Union{Missing, Int64}
        @test isequal(processed.proposed_market_value, Union{Missing, Int64}[250_000, missing])
        @test eltype(processed.proposed_market_value_currency) == Union{Missing, String3}
        @test processed.proposed_market_value_currency[1] == "EUR"
        @test ismissing(processed.proposed_market_value_currency[2])
        @test "total_passes" in names(processed)
        @test eltype(processed.total_passes) == Union{Missing, Float64}
        @test isequal(processed.total_passes, Union{Missing, Float64}[42.0, missing])
    end

    @testset "Betfair Grading" begin
        # 1X2
        @test BayesianFootball.Data.grade_selection("1X2", 0.0, :home, 2, 1) == true
        @test BayesianFootball.Data.grade_selection("1X2", 0.0, :home, 1, 1) == false
        @test BayesianFootball.Data.grade_selection("1X2", 0.0, :draw, 1, 1) == true
        @test BayesianFootball.Data.grade_selection("1X2", 0.0, :away, 1, 2) == true
        
        # Over/Under
        @test BayesianFootball.Data.grade_selection("OverUnder", 2.5, :over_25, 2, 1) == true
        @test BayesianFootball.Data.grade_selection("OverUnder", 2.5, :over_25, 1, 1) == false
        @test BayesianFootball.Data.grade_selection("OverUnder", 2.5, :under_25, 1, 1) == true
        
        # BTTS
        @test BayesianFootball.Data.grade_selection("BTTS", 0.0, :btts_yes, 1, 1) == true
        @test BayesianFootball.Data.grade_selection("BTTS", 0.0, :btts_yes, 2, 0) == false
        @test BayesianFootball.Data.grade_selection("BTTS", 0.0, :btts_no, 2, 0) == true
        
        # Correct Score
        @test BayesianFootball.Data.grade_selection("CorrectScore", 0.0, :cs_21, 2, 1) == true
        @test BayesianFootball.Data.grade_selection("CorrectScore", 0.0, :cs_21, 1, 2) == false
        @test BayesianFootball.Data.grade_selection("CorrectScore", 0.0, :cs_any_other_home, 4, 1) == true
        @test BayesianFootball.Data.grade_selection("CorrectScore", 0.0, :cs_any_other_home, 3, 1) == false
        
        # Missing data
        @test ismissing(BayesianFootball.Data.grade_selection("1X2", 0.0, :home, missing, 1))
    end

    @testset "Preprocessing: Match Week Logic" begin
        df = DataFrame(
            tournament_id = [1, 1, 1, 1],
            season = [2022, 2022, 2022, 2023],
            # Matches on Mon, Tue, and next Mon
            match_date = [Date("2022-08-01"), Date("2022-08-02"), Date("2022-08-08"), Date("2023-08-01")] 
        )
        
        processed_df = BayesianFootball.Data.add_match_week_column(df)
        
        @test "match_week" in names(processed_df)
        # Aug 1 and Aug 2 are in the same week for the 2022 season (Week 1)
        @test processed_df.match_week[1] == 1
        @test processed_df.match_week[2] == 1
        # Aug 8 is the next week (Week 2)
        @test processed_df.match_week[3] == 2
        
        # 2023 season restarts the week counter
        @test processed_df.match_week[4] == 1
    end
end
