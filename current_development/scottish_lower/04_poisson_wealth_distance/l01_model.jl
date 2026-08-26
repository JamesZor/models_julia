# Model 04 loader: joint point-in-time squad wealth plus static travel fatigue.
if !isdefined(@__MODULE__, :SLFeaturePoissonModel)
    include(joinpath(@__DIR__, "..", "02_poisson_wealth", "l00_feature_poisson.jl"))
end
if !isdefined(@__MODULE__, :ScottishDistanceFeature)
    include(joinpath(@__DIR__, "..", "03_poisson_distance", "l00_distance_feature.jl"))
end
function tp04_model(; half_life_days=180.0, wealth_prior=truncated(Normal(0.10,.05), lower=0.0), distance_prior=truncated(Normal(0.04,.03), lower=0.0))
    DynamicPoissonWealthDistanceGoalsTimeDecayModel(interception_config=SLFP_PG.GlobalInterception(μ=Normal(0.2,0.1)),
        dynamics_config=SLFP_PG.TimeDecayDynamics(days_half_life=half_life_days, σ_att=Gamma(2.0,.15), σ_def=Gamma(2.0,.15)),
        homeadvantage_config=SLFP_PG.GlobalHomeAdvantage(γ_global=Normal(.2,.2)),
        wealth_feature=SLFPLogSumWealthFeature(), distance_feature=ScottishDistanceFeature(metric=:log_dist_z),
        w_wealth_prior=wealth_prior, w_dist_prior=distance_prior)
end
