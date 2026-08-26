# Model 03 loader: static Haversine travel-fatigue Poisson extension.
if !isdefined(@__MODULE__, :SLFeaturePoissonModel)
    include(joinpath(@__DIR__, "..", "02_poisson_wealth", "l00_feature_poisson.jl"))
end
if !isdefined(@__MODULE__, :ScottishDistanceFeature)
    include(joinpath(@__DIR__, "l00_distance_feature.jl"))
end
function tp03_model(; half_life_days=180.0, distance_prior=truncated(Normal(0.0, 0.10), lower=0.0))
    SLFeaturePoissonModel(interception_config=SLFP_PG.GlobalInterception(μ=Normal(0.2,0.1)),
        dynamics_config=SLFP_PG.TimeDecayDynamics(days_half_life=half_life_days, σ_att=Gamma(2.0,.15), σ_def=Gamma(2.0,.15)),
        homeadvantage_config=SLFP_PG.GlobalHomeAdvantage(γ_global=Normal(.2,.2)),
        distance_feature=ScottishDistanceFeature(metric=:log_dist_z), distance_prior=distance_prior)
end
