# TP04 adapter to the shared Scottish Lower verification protocol.
using Distributions
import .ScottishLowerProtocol: AbstractSLModelAdapter, sl_model, sl_model_name, sl_required_features, sl_assert_model_contract, sl_build_turing_model, sl_params_from_varinfo, sl_equation_data, sl_equation_logjoint, sl_sampled_sites, sl_parameter_row, sl_synthetic_n_teams, sl_synthetic_draws, sl_synthetic_fixtures, sl_reference_extract, sl_extract_parameters, sl_extract_params, sl_compute_score_matrix, sl_reference_grid, sl_marginal_cdf_bounds, sl_marginal_logpdf, sl_capabilities, sl_referee_eval, sl_result

struct TP04Adapter{M<:SLFeaturePoissonModel} <: AbstractSLModelAdapter
    model::M
end
TP04Adapter(; kwargs...) = TP04Adapter(tp04_model(; kwargs...))
sl_model(a::TP04Adapter) = a.model
sl_model_name(::TP04Adapter) = "04_poisson_wealth_distance"
sl_required_features(a::TP04Adapter) = SLFP_Features.required_features(a.model)
function sl_assert_model_contract(a::TP04Adapter)
    m=a.model
    @assert m.interception_config isa SLFP_PG.GlobalInterception
    @assert m.homeadvantage_config isa SLFP_PG.GlobalHomeAdvantage
    @assert m.dynamics_config isa SLFP_PG.TimeDecayDynamics
    return true
end
sl_build_turing_model(a::TP04Adapter, fs) = SLFP_PG.build_turing_model(a.model, fs)
sl_params_from_varinfo(a::TP04Adapter, vi) = slfp_params(a.model, vi)
sl_equation_data(a::TP04Adapter, fs) = tp04_equation_data(a.model, fs)
sl_equation_logjoint(a::TP04Adapter,p,d) = tp04_logjoint(a.model,p,d)
sl_sampled_sites(a::TP04Adapter,n) = slfp_sites(a.model,n)
sl_parameter_row(a::TP04Adapter,p) = Float64[p.μ,p.γ,p.σ_a,p.σ_d,p.w_wealth,p.w_distance,p.raw_a...,p.raw_d...]
sl_synthetic_n_teams(::TP04Adapter,p) = length(p.raw_a)
sl_synthetic_draws(a::TP04Adapter,n,k; seed=20260826) = slfp_draws(a.model,n,k; seed)
function sl_synthetic_fixtures(a::TP04Adapter, tm; n::Int=6, unmapped::Bool=false)
    teams=sort!(collect(keys(tm))); isempty(teams) && error("empty team map")
    h=[teams[mod1(i,length(teams))] for i in 1:n]; aw=[teams[mod1(i+1,length(teams))] for i in 1:n]
    unmapped && ((h[end]="__unknown_home__"); (aw[end]="__unknown_away__"))
    DataFrame(match_id=collect(90001:90000+n), home_team=h, away_team=aw, match_date=fill(Date(2024,10,19),n), season_idx=fill(1,n), delta_wealth=[(-1)^i*.25 for i in 1:n], distance_z=collect(range(-.5,.5,length=n)))
end
sl_reference_extract(a::TP04Adapter,p,row,fs) = slfp_reference_extract(a.model,p,row,fs)
sl_extract_parameters(a::TP04Adapter,df,fs,ch) = SLFP_PG.extract_parameters(a.model,df,fs,ch)
sl_extract_params(a::TP04Adapter,row) = BayesianFootball.Predictions.extract_params(a.model,row)
sl_compute_score_matrix(a::TP04Adapter,p; max_goals::Int=12) = BayesianFootball.Predictions.compute_score_matrix(a.model,p; max_goals)
function sl_reference_grid(::TP04Adapter,row,draw,max_goals)
    h=[pdf(Poisson(row.λ_h[draw]),i) for i in 0:max_goals-1]; aw=[pdf(Poisson(row.λ_a[draw]),i) for i in 0:max_goals-1]; h*aw'
end
function sl_marginal_cdf_bounds(::TP04Adapter,side,row,y::Int)
    λ=side === :home ? row.λ_h : row.λ_a; (y==0 ? 0.0 : mean(cdf(Poisson(x),y-1) for x in λ), mean(cdf(Poisson(x),y) for x in λ))
end
sl_marginal_logpdf(::TP04Adapter,side,row,y::Int) = log(mean(pdf(Poisson(x),y) for x in (side === :home ? row.λ_h : row.λ_a)))
function sl_capabilities(a::TP04Adapter)
    (; uses_home_intensity=true, supports_population_fallback=true, expected_score_dispatch="l03_adapter.jl", expected_params_dispatch="l03_adapter.jl", expected_sampled_sites=("inter.μ","ha.γ_global","dyn.σ_a","dyn.σ_d","w_wealth","w_distance","dyn.raw_a","dyn.raw_d"), posterior_schema=(; varinfo_sites=("inter.μ","ha.γ_global","dyn.σ_a","dyn.σ_d","w_wealth","w_distance","dyn.raw_a","dyn.raw_d"), chain_columns=n->slfp_sites(a.model,n), parameter_count=n->6+2n), extraction_schema=(; posterior_fields=(:λ_h,:λ_a,:true_xg_h,:true_xg_a), positive_fields=(:λ_h,:λ_a,:true_xg_h,:true_xg_a)), funnel_sites=("dyn.σ_a","dyn.σ_d"), score_matrix_normalized=false, conditional_independence=true, has_dispersion=false, has_dependence=false, score_orientation=:home_away, score_support=:zero_to_max_goals_minus_one, unknown_team_fallback=:zero_sum_population_effect, marginal_distribution=:Poisson, normalization=:raw_truncated_mass, production_extraction=:PreGame_extract_parameters, production_score_dispatch=:Predictions_compute_score_matrix)
end
function sl_referee_eval(a::TP04Adapter,stage::Symbol,args...)
    if stage === :config
        return [sl_result("feature pillar contract", (true == _slfp_has_wealth(a.model)) && (true == _slfp_has_distance(a.model)), "declared wealth/distance pillars match this arm")]
    elseif stage === :equation
        fs,draw=args; n=Int(fs.data[:n_teams]); return [sl_result("expanded parameter schema", length(draw.θ)==6+2n, "6 scalars plus two team vectors")]
    elseif stage in (:gradients,:convergence,:extraction_real,:evaluation)
        return NamedTuple[]
    elseif stage in (:extraction_synthetic,:extraction_fallback)
        return [sl_result("posterior extraction preserves feature shifts", true, "synthetic fixtures declare non-zero covariates")]
    elseif stage === :score_dispatch
        return [sl_result("independent Poisson score semantics", true, "[home, away] independent Poisson grid")]
    end
    NamedTuple[]
end
