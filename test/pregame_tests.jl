using Test
using BayesianFootball

@testset "Models Module" begin
    PG = BayesianFootball.Models.PreGame
    model = PG.DynamicGoalsTimeDecayModel(
        interception_config=PG.GlobalInterception(),
        dynamics_config=PG.TimeDecayDynamics(days_half_life=180.0),
        dispersion_config=PG.GlobalDispersion(),
        homeadvantage_config=PG.GlobalHomeAdvantage(),
    )

    @test model isa PG.AbstractPregameModel
    @test model.dynamics_config.days_half_life == 180.0

    required = BayesianFootball.Features.required_features(model)
    @test any(feature -> feature isa BayesianFootball.Features.TeamIDsFeature, required)
    @test any(feature -> feature isa BayesianFootball.Features.GoalsFeature, required)
    @test any(feature -> feature isa BayesianFootball.Features.DatesFeature, required)
    @test any(feature -> feature isa BayesianFootball.Features.TimeIndicesFeature, required)

    @test PG.calculate_match_weights([0, 180, 360], 180.0) ≈ [1.0, 0.5, 0.25]
end
