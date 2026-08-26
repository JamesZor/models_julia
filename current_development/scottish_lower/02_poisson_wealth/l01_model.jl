# Model 02 loader: starting-XI point-in-time squad-wealth Poisson extension.
include(joinpath(@__DIR__, "l00_feature_poisson.jl"))
function tp02_model(; half_life_days=180.0, wealth_prior=truncated(Normal(0.0, 0.15), lower=0.0))
    SLFeaturePoissonModel(interception_config=SLFP_PG.GlobalInterception(μ=Normal(0.2,0.1)),
        dynamics_config=SLFP_PG.TimeDecayDynamics(days_half_life=half_life_days, σ_att=Gamma(2.0,.15), σ_def=Gamma(2.0,.15)),
        homeadvantage_config=SLFP_PG.GlobalHomeAdvantage(γ_global=Normal(.2,.2)),
        wealth_feature=SLFPOOSWealthFeature(), wealth_prior=wealth_prior)
end
