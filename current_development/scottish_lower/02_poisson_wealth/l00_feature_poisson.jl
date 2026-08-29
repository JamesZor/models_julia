using BayesianFootball, Turing, Distributions, DataFrames, Dates, SpecialFunctions, Statistics, Random, MCMCChains
const SLFP_PG = BayesianFootball.Models.PreGame
const SLFP_Features = BayesianFootball.Features
abstract type AbstractSLFPModel <: BayesianFootball.TypesInterfaces.AbstractPoissonModel end
Base.@kwdef struct SLFPLogSumWealthFeature <: SLFP_Features.AbstractFeatureConfig
    fallback_default::Float64=100_000.0
    log_scale::Float64=1.0
end
function _slfp_wealth_lookup(lineups,matches,ids,times,cfg)
    wanted=Set(Int.(ids)); vals=Dict{Tuple{Int,Bool},Vector{Float64}}(); known=Dict{Tuple{Int,Bool},Int}()
    ko=Dict(Int(r.match_id)=>DateTime(r.match_date)+Hour(r.match_hour) for r in eachrow(matches) if !ismissing(r.match_date)&&!ismissing(r.match_hour))
    for r in eachrow(lineups)
        id=Int(r.match_id); id in wanted||continue; (:is_substitute in propertynames(lineups)&&coalesce(r.is_substitute,false))&&continue
        side=:team_side in propertynames(lineups) ? lowercase(String(r.team_side))=="home" : Bool(r.is_home_team); key=(id,side); v=get!(vals,key,Float64[])
        raw=:proposed_market_value in propertynames(lineups) ? r.proposed_market_value : :market_value in propertynames(lineups) ? r.market_value : missing; stamp=:valuation_timestamp in propertynames(lineups) ? SLFP_Features._wealth_as_datetime(r.valuation_timestamp) : nothing; at=get(times,id,get(ko,id,nothing))
        stamp_ok = (stamp === nothing) || (at === nothing) || (stamp < at)
        if !ismissing(raw)&&stamp_ok&&isfinite(Float64(raw))&&Float64(raw)>0; push!(v,Float64(raw)); known[key]=get(known,key,0)+1 else push!(v,cfg.fallback_default) end
    end
    out=Dict{Int,Float64}(); for id in wanted; h=get(vals,(id,true),Float64[]); a=get(vals,(id,false),Float64[]); if isempty(h)||isempty(a)||get(known,(id,true),0)==0||get(known,(id,false),0)==0; continue; end; out[id]=(log(sum(h))-log(sum(a)))/cfg.log_scale end; out
end
function SLFP_Features.add_feature!(d::Dict,cfg::SLFPLogSumWealthFeature,ids,tm::Dict,ds::BayesianFootball.Data.DataStore)
    map=_slfp_wealth_lookup(ds.lineups,ds.matches,ids,Dict{Int,DateTime}(),cfg); d[:flat_delta_wealth_logsum]=Float64[get(map,Int(id),0.) for id in ids]; d[:flat_wealth_fallback]=Int[haskey(map,Int(id)) ? 0 : 1 for id in ids]; d[:wealth_logsum_by_match_id]=map
    # Extraction-only causal bridge: no value uses a valuation after its own kickoff.
    d[:wealth_oos_bridge_by_match_id]=_slfp_wealth_lookup(ds.lineups,ds.matches,Int.(ds.matches.match_id),Dict{Int,DateTime}(),cfg); d
end
Base.@kwdef struct DynamicPoissonWealthGoalsTimeDecayModel{I,T,H,W} <: AbstractSLFPModel
 interception_config::I; dynamics_config::T; homeadvantage_config::H; wealth_feature::W=SLFPLogSumWealthFeature(); w_wealth_prior::Distribution=truncated(Normal(.10,.05),lower=0.)
end
Base.@kwdef struct DynamicPoissonDistanceGoalsTimeDecayModel{I,T,H,D} <: AbstractSLFPModel
 interception_config::I; dynamics_config::T; homeadvantage_config::H; distance_feature::D; w_dist_prior::Distribution=truncated(Normal(.04,.03),lower=0.)
end
Base.@kwdef struct DynamicPoissonWealthDistanceGoalsTimeDecayModel{I,T,H,W,D} <: AbstractSLFPModel
 interception_config::I; dynamics_config::T; homeadvantage_config::H; wealth_feature::W=SLFPLogSumWealthFeature(); distance_feature::D; w_wealth_prior::Distribution=truncated(Normal(.10,.05),lower=0.); w_dist_prior::Distribution=truncated(Normal(.04,.03),lower=0.)
end
_w(::DynamicPoissonWealthGoalsTimeDecayModel)=true; _w(::DynamicPoissonDistanceGoalsTimeDecayModel)=false; _w(::DynamicPoissonWealthDistanceGoalsTimeDecayModel)=true
_d(::DynamicPoissonWealthGoalsTimeDecayModel)=false; _d(::DynamicPoissonDistanceGoalsTimeDecayModel)=true; _d(::DynamicPoissonWealthDistanceGoalsTimeDecayModel)=true
function SLFP_Features.required_features(m::AbstractSLFPModel); x=SLFP_Features.AbstractFeatureConfig[SLFP_Features.TeamIDsFeature(),SLFP_Features.GoalsFeature(),SLFP_Features.DatesFeature(),SLFP_Features.MonthFeature(),SLFP_Features.TimeIndicesFeature()]; _w(m)&&push!(x,m.wealth_feature); _d(m)&&push!(x,m.distance_feature); x end
function _dat(m,fs)
 d=fs.data; n=length(d[:flat_home_ids]); h=Vector{Int}(d[:flat_home_ids]); a=Vector{Int}(d[:flat_away_ids]); s=Vector{Int}(d[:season_indices]); mo=Vector{Int}(d[:flat_months]); yh=Vector{Int}(d[:flat_home_goals]); ya=Vector{Int}(d[:flat_away_goals]); wt=0.5 .^ (Vector{Float64}(d[:dates])./m.dynamics_config.days_half_life); xw=_w(m) ? Vector{Float64}(d[:flat_delta_wealth_logsum]) : Float64[]; xd=_d(m) ? Vector{Float64}(d[:flat_distance]) : Float64[]; all(length(z)==n for z in (h,a,s,mo,yh,ya,wt,xw,xd) if !isempty(z))&&all(isfinite,wt)&&all(isfinite,xw)&&all(isfinite,xd)||error("invalid required feature vectors"); (;h,a,s,mo,yh,ya,wt,xw,xd,nt=Int(d[:n_teams]),ns=Int(d[:n_seasons]))
end
@model function _engw(h,a,s,mo,yh,ya,wt,x,lfh,lfa,nt,ns,m)
 inter~to_submodel(SLFP_PG.build_interception(m.interception_config,ns,12)); ha~to_submodel(SLFP_PG.build_home_advantage(m.homeadvantage_config,nt)); dyn~to_submodel(SLFP_PG.build_dynamics(m.dynamics_config,nt)); w_wealth~m.w_wealth_prior; b=view(inter.μ_base,s).+view(inter.δ_month,mo); q=w_wealth.*x; eh=clamp.(b.+view(ha,h).+view(dyn.α,h).+view(dyn.β,a).+q,-10.,10.); ea=clamp.(b.+view(dyn.α,a).+view(dyn.β,h).-q,-10.,10.); Turing.@addlogprob! sum(wt.*(yh.*eh.-exp.(eh).-lfh))+sum(wt.*(ya.*ea.-exp.(ea).-lfa))
end
@model function _engd(h,a,s,mo,yh,ya,wt,x,lfh,lfa,nt,ns,m)
 inter~to_submodel(SLFP_PG.build_interception(m.interception_config,ns,12)); ha~to_submodel(SLFP_PG.build_home_advantage(m.homeadvantage_config,nt)); dyn~to_submodel(SLFP_PG.build_dynamics(m.dynamics_config,nt)); w_dist~m.w_dist_prior; b=view(inter.μ_base,s).+view(inter.δ_month,mo); q=w_dist.*x; eh=clamp.(b.+view(ha,h).+view(dyn.α,h).+view(dyn.β,a).+q,-10.,10.); ea=clamp.(b.+view(dyn.α,a).+view(dyn.β,h).-q,-10.,10.); Turing.@addlogprob! sum(wt.*(yh.*eh.-exp.(eh).-lfh))+sum(wt.*(ya.*ea.-exp.(ea).-lfa))
end
@model function _engj(h,a,s,mo,yh,ya,wt,xw,xd,lfh,lfa,nt,ns,m)
 inter~to_submodel(SLFP_PG.build_interception(m.interception_config,ns,12)); ha~to_submodel(SLFP_PG.build_home_advantage(m.homeadvantage_config,nt)); dyn~to_submodel(SLFP_PG.build_dynamics(m.dynamics_config,nt)); w_wealth~m.w_wealth_prior; w_dist~m.w_dist_prior; b=view(inter.μ_base,s).+view(inter.δ_month,mo); q=w_wealth.*xw.+w_dist.*xd; eh=clamp.(b.+view(ha,h).+view(dyn.α,h).+view(dyn.β,a).+q,-10.,10.); ea=clamp.(b.+view(dyn.α,a).+view(dyn.β,h).-q,-10.,10.); Turing.@addlogprob! sum(wt.*(yh.*eh.-exp.(eh).-lfh))+sum(wt.*(ya.*ea.-exp.(ea).-lfa))
end
function SLFP_PG.build_turing_model(m::AbstractSLFPModel,fs::SLFP_Features.FeatureSet); z=_dat(m,fs); fh=loggamma.(Float64.(z.yh).+1); fa=loggamma.(Float64.(z.ya).+1); m isa DynamicPoissonWealthGoalsTimeDecayModel ? _engw(z.h,z.a,z.s,z.mo,z.yh,z.ya,z.wt,z.xw,fh,fa,z.nt,z.ns,m) : m isa DynamicPoissonDistanceGoalsTimeDecayModel ? _engd(z.h,z.a,z.s,z.mo,z.yh,z.ya,z.wt,z.xd,fh,fa,z.nt,z.ns,m) : _engj(z.h,z.a,z.s,z.mo,z.yh,z.ya,z.wt,z.xw,z.xd,fh,fa,z.nt,z.ns,m) end
Base.@kwdef struct SLFPParams; μ::Float64; γ::Float64; σ_a::Float64; σ_d::Float64; raw_a::Vector{Float64}; raw_d::Vector{Float64}; w_wealth::Float64=0.; w_dist::Float64=0. end
function slfp_params(m,vi); v=Dict(string(k)=>vi[k] for k in keys(vi)); SLFPParams(μ=Float64(v["inter.μ"]),γ=Float64(v["ha.γ_global"]),σ_a=Float64(v["dyn.σ_a"]),σ_d=Float64(v["dyn.σ_d"]),raw_a=Float64.(v["dyn.raw_a"]),raw_d=Float64.(v["dyn.raw_d"]),w_wealth=_w(m) ? Float64(v["w_wealth"]) : 0.,w_dist=_d(m) ? Float64(v["w_dist"]) : 0.) end
slfp_team_effects(p)=(p.raw_a.*p.σ_a.-mean(p.raw_a.*p.σ_a),p.raw_d.*p.σ_d.-mean(p.raw_d.*p.σ_d))
function _oos(m,fs,df); xw=_w(m) ? (hasproperty(df,:delta_wealth_logsum) ? Float64.(df.delta_wealth_logsum) : Float64[get(fs.data[:wealth_oos_bridge_by_match_id],Int(r.match_id),0.) for r in eachrow(df)]) : zeros(Float64,nrow(df)); xd=_d(m) ? (hasproperty(df,:distance_z) ? Float64.(df.distance_z) : Float64.(build_match_distance_table(DataFrame(df);geocodes_df=load_scottish_stadium_catalog(m.distance_feature.geocodes_csv)).log_dist_z)) : zeros(Float64,nrow(df)); all(isfinite,xw)&&all(isfinite,xd)||error("bad OOS covariate"); xw,xd end
function SLFP_PG.extract_parameters(m::AbstractSLFPModel,df::AbstractDataFrame,fs::SLFP_Features.FeatureSet,ch::Chains)
 d=fs.data; n=size(ch,1)*size(ch,3); inter=SLFP_PG.extract_interception(ch,m.interception_config,Int(d[:n_seasons])); ha=SLFP_PG.extract_home_advantage(ch,m.homeadvantage_config,Int(d[:n_teams])); dyn=SLFP_PG.extract_dynamics(ch,m.dynamics_config,"dyn",Int(d[:n_teams])); ww=_w(m) ? vec(Array(ch[:w_wealth])) : zeros(n); wd=_d(m) ? vec(Array(ch[:w_dist])) : zeros(n); xw,xd=_oos(m,fs,df); tm=d[:team_map]; out=Dict{Int,NamedTuple}(); for (i,r) in enumerate(eachrow(df)); h=get(tm,r.home_team,0); a=get(tm,r.away_team,0); ah=h>0 ? dyn.α[:,h] : zeros(n); bh=h>0 ? dyn.β[:,h] : zeros(n); aa=a>0 ? dyn.α[:,a] : zeros(n); ba=a>0 ? dyn.β[:,a] : zeros(n); γ=m.homeadvantage_config isa SLFP_PG.GlobalHomeAdvantage ? ha[:,1] : h>0 ? ha[:,h] : zeros(n); k=hasproperty(r,:season_idx) ? Int(r.season_idx) : size(inter.μ_base,2); b=inter.μ_base[:,k].+inter.δ_month[:,Dates.month(r.match_date)]; q=ww.*xw[i].+wd.*xd[i]; eh=clamp.(b.+γ.+ah.+ba.+q,-10.,10.); ea=clamp.(b.+aa.+bh.-q,-10.,10.); out[Int(r.match_id)]=(λ_h=exp.(eh),λ_a=exp.(ea),true_xg_h=exp.(eh),true_xg_a=exp.(ea)) end; out
end
