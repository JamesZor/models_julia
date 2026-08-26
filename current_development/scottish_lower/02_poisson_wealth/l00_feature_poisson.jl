# Shared implementation for Scottish Lower Poisson feature extensions.
# Included by each arm-local l01_model.jl; definitions only.

using BayesianFootball
using Turing
using Distributions
using DataFrames
using Dates
using SpecialFunctions
using Statistics
using Random
using MCMCChains

const SLFP_PG = BayesianFootball.Models.PreGame
const SLFP_Features = BayesianFootball.Features

"Pure Poisson team model with optional, signed home-versus-away feature pillars."
Base.@kwdef struct SLFeaturePoissonModel{I,T,H,W,D} <: BayesianFootball.TypesInterfaces.AbstractPoissonModel
    interception_config::I
    dynamics_config::T
    homeadvantage_config::H
    wealth_feature::W = nothing
    distance_feature::D = nothing
    wealth_prior::Distribution = truncated(Normal(0.0, 0.15), lower = 0.0)
    distance_prior::Distribution = truncated(Normal(0.0, 0.10), lower = 0.0)
end

_slfp_has_wealth(m) = m.wealth_feature !== nothing
_slfp_has_distance(m) = m.distance_feature !== nothing

"Objective wealth feature: log ratio of imputed starting-XI value sums."
Base.@kwdef struct SLFPLogSumWealthFeature <: SLFP_Features.AbstractFeatureConfig
    fallback_default::Float64 = 100_000.0
    log_scale::Float64 = 1.0
end

function _slfp_wealth_lookup(lineups, matches, ids, decision_times, cfg::SLFPLogSumWealthFeature)
    values = Dict{Tuple{Int,Bool},Vector{Float64}}(); known = Dict{Tuple{Int,Bool},Int}()
    wanted = Set(Int.(ids)); kickoff = Dict(Int(r.match_id)=>DateTime(r.match_date)+Hour(r.match_hour) for r in eachrow(matches) if !ismissing(r.match_date) && !ismissing(r.match_hour))
    for r in eachrow(lineups)
        id = Int(r.match_id); id in wanted || continue
        (:is_substitute in propertynames(lineups) && coalesce(r.is_substitute,false)) && continue
        side = :team_side in propertynames(lineups) ? lowercase(String(r.team_side)) == "home" : Bool(r.is_home_team)
        key=(id,side); vals=get!(values,key,Float64[])
        raw = :market_value in propertynames(lineups) ? r.market_value : missing
        stamp = :valuation_timestamp in propertynames(lineups) ? r.valuation_timestamp : missing
        decision = get(decision_times,id,get(kickoff,id,nothing))
        safe = !ismissing(raw) && !ismissing(stamp) && decision !== nothing && SLFP_Features._wealth_as_datetime(stamp) !== nothing && SLFP_Features._wealth_as_datetime(stamp) < decision
        if safe && isfinite(Float64(raw)) && Float64(raw)>0
            push!(vals,Float64(raw)); known[key]=get(known,key,0)+1
        else
            push!(vals,cfg.fallback_default)
        end
    end
    out=Dict{Int,Float64}()
    for id in wanted
        h=get(values,(id,true),Float64[]); a=get(values,(id,false),Float64[])
        isempty(h) || isempty(a) || get(known,(id,true),0)==0 || get(known,(id,false),0)==0 || continue
        out[id]=(log(sum(h))-log(sum(a)))/cfg.log_scale
    end
    out
end

function SLFP_Features.add_feature!(data::Dict, cfg::SLFPLogSumWealthFeature, ordered_ids, team_map::Dict, ds::BayesianFootball.Data.DataStore)
    cfg.fallback_default > 0 && cfg.log_scale > 0 || throw(ArgumentError("positive wealth fallback and scale required"))
    fitted = _slfp_wealth_lookup(ds.lineups, ds.matches, ordered_ids, Dict{Int,DateTime}(), cfg)
    data[:flat_delta_wealth_logsum] = Float64[get(fitted,Int(id),0.0) for id in ordered_ids]
    data[:flat_wealth_fallback] = Int[haskey(fitted,Int(id)) ? 0 : 1 for id in ordered_ids]
    data[:wealth_logsum_by_match_id] = fitted
    return data
end

Base.@kwdef struct OOSCovariateSnapshot
    match_ids::Vector{Int}; delta_wealth_logsum::Vector{Float64}; wealth_fallback::Vector{Int}
    distance_z::Vector{Float64}; distance_fallback::Vector{Int}; decision_times::Vector{DateTime}
    feature_config_hash::String; source_fingerprint::String
end

function build_oos_covariate_snapshot(model, oos_df, ds; decision_time_by_match::AbstractDict{Int,DateTime})
    ids=Int.(oos_df.match_id); length(unique(ids))==length(ids) || error("duplicate OOS ids")
    times=DateTime[get(decision_time_by_match,id,error("missing decision time for $id")) for id in ids]
    cfg=model.wealth_feature; wealth=_slfp_has_wealth(model) ? _slfp_wealth_lookup(ds.lineups,ds.matches,ids,decision_time_by_match,cfg) : Dict{Int,Float64}()
    w=Float64[get(wealth,id,0.0) for id in ids]; wf=Int[haskey(wealth,id) ? 0 : 1 for id in ids]
    dist=zeros(Float64,length(ids)); dfallback=zeros(Int,length(ids))
    if _slfp_has_distance(model)
        table=build_match_distance_table(DataFrame(oos_df); geocodes_df=load_scottish_stadium_catalog(model.distance_feature.geocodes_csv)); dist.=Float64.(table.log_dist_z); dfallback.=Int.(table.distance_fallback)
    end
    OOSCovariateSnapshot(ids,w,wf,dist,dfallback,times,string(hash((cfg,model.distance_feature))),string(hash((ids,w,wf,dist,dfallback,times))))
end
function attach_oos_covariates(oos_df, snap::OOSCovariateSnapshot)
    Int.(oos_df.match_id)==snap.match_ids || error("snapshot IDs do not exactly match OOS order")
    out=DataFrame(oos_df); out.delta_wealth_logsum=snap.delta_wealth_logsum; out.wealth_fallback=snap.wealth_fallback; out.distance_z=snap.distance_z; out.distance_fallback=snap.distance_fallback; out
end

_slfp_weight(deltas, half_life) = 0.5 .^ (deltas ./ half_life)

function SLFP_Features.required_features(model::SLFeaturePoissonModel)
    out = SLFP_Features.AbstractFeatureConfig[
        SLFP_Features.TeamIDsFeature(), SLFP_Features.GoalsFeature(),
        SLFP_Features.DatesFeature(), SLFP_Features.MonthFeature(),
        SLFP_Features.TimeIndicesFeature(),
    ]
    _slfp_has_wealth(model) && push!(out, model.wealth_feature)
    _slfp_has_distance(model) && push!(out, model.distance_feature)
    return out
end

@model function slfp_engine(home_ids::Vector{Int}, away_ids::Vector{Int}, season_ids::Vector{Int},
                            month_ids::Vector{Int}, home_goals::Vector{Int}, away_goals::Vector{Int},
                            weights::Vector{Float64}, wealth::Vector{Float64}, distance::Vector{Float64},
                            n_teams::Int, n_seasons::Int, n_months::Int, wealth_on::Float64,
                            distance_on::Float64, config::SLFeaturePoissonModel)
    inter ~ to_submodel(SLFP_PG.build_interception(config.interception_config, n_seasons, n_months))
    ha ~ to_submodel(SLFP_PG.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn ~ to_submodel(SLFP_PG.build_dynamics(config.dynamics_config, n_teams))
    w_wealth ~ config.wealth_prior
    w_distance ~ config.distance_prior

    feature_shift = wealth_on .* w_wealth .* wealth .+ distance_on .* w_distance .* distance
    inter_match = inter.μ_base[season_ids] .+ inter.δ_month[month_ids]
    η_h = inter_match .+ ha[home_ids] .+ dyn.α[home_ids] .+ dyn.β[away_ids] .+ feature_shift
    η_a = inter_match .+ dyn.α[away_ids] .+ dyn.β[home_ids] .- feature_shift
    bad = any(isnan.(η_h)) || any(isnan.(η_a))
    η_h = ifelse.(isnan.(η_h), zero.(η_h), η_h)
    η_a = ifelse.(isnan.(η_a), zero.(η_a), η_a)
    Turing.@addlogprob! ifelse(bad, -Inf, 0.0)
    log_fact_h = loggamma.(Float64.(home_goals) .+ 1.0)
    log_fact_a = loggamma.(Float64.(away_goals) .+ 1.0)
    Turing.@addlogprob! sum(weights .* (home_goals .* η_h .- exp.(η_h) .- log_fact_h))
    Turing.@addlogprob! sum(weights .* (away_goals .* η_a .- exp.(η_a) .- log_fact_a))
end

function SLFP_PG.build_turing_model(model::SLFeaturePoissonModel, fs::SLFP_Features.FeatureSet)
    d = fs.data
    n = length(d[:flat_home_ids])
    wealth = _slfp_has_wealth(model) ? Vector{Float64}(d[:flat_delta_wealth_logsum]) : zeros(Float64, n)
    distance = _slfp_has_distance(model) ? Vector{Float64}(d[:flat_distance]) : zeros(Float64, n)
    all(isfinite, wealth) && all(isfinite, distance) || error("non-finite required feature")
    return slfp_engine(Vector{Int}(d[:flat_home_ids]), Vector{Int}(d[:flat_away_ids]),
        Vector{Int}(d[:season_indices]), Vector{Int}(d[:flat_months]),
        Vector{Int}(d[:flat_home_goals]), Vector{Int}(d[:flat_away_goals]),
        _slfp_weight(Float64.(d[:dates]), model.dynamics_config.days_half_life), wealth, distance,
        Int(d[:n_teams]), Int(d[:n_seasons]), 12, _slfp_has_wealth(model) ? 1.0 : 0.0,
        _slfp_has_distance(model) ? 1.0 : 0.0, model)
end

Base.@kwdef struct SLFPParams
    μ::Float64
    γ::Float64
    σ_a::Float64
    σ_d::Float64
    w_wealth::Float64
    w_distance::Float64
    raw_a::Vector{Float64}
    raw_d::Vector{Float64}
end

function slfp_team_effects(p::SLFPParams)
    a = p.raw_a .* p.σ_a
    d = p.raw_d .* p.σ_d
    return a .- mean(a), d .- mean(d)
end

function slfp_feature_values(model, fs, df)
    if _slfp_has_wealth(model)
        hasproperty(df, :delta_wealth_logsum) && hasproperty(df, :wealth_fallback) || error("OOS wealth snapshot columns are required")
        all(Int.(df.wealth_fallback) .∈ Ref((0,1))) || error("invalid wealth fallback flag")
        wealth = Float64.(df.delta_wealth_logsum)
    else
        wealth = zeros(Float64, nrow(df))
    end
    # Static distance is attached by the snapshot for OOS; synthetic rows declare it too.
    if _slfp_has_distance(model)
        hasproperty(df, :distance_z) && hasproperty(df, :distance_fallback) || error("OOS distance snapshot columns are required")
        all(Int.(df.distance_fallback) .∈ Ref((0,1))) || error("invalid distance fallback flag")
        distance = Float64.(df.distance_z)
    else
        distance = zeros(Float64, nrow(df))
    end
    all(isfinite, wealth) && all(isfinite, distance) || error("non-finite OOS covariate")
    return wealth, distance
    if _slfp_has_distance(model)
        if !isdefined(@__MODULE__, :ScottishDistanceFeature)
            include(joinpath(@__DIR__, "..", "03_poisson_distance", "l00_distance_feature.jl"))
        end
        table = build_match_distance_table(DataFrame(df); geocodes_df = load_scottish_stadium_catalog(model.distance_feature.geocodes_csv))
        metric = model.distance_feature.metric
        source = metric == :dist_z ? table.dist_z : metric == :hav_miles ? table.hav_miles :
                 metric == :road_miles ? table.road_miles : metric == :drive_minutes ? table.drive_minutes : table.log_dist_z
        distance .= Float64.(source)
    end
    return wealth, distance
end

function SLFP_PG.extract_parameters(model::SLFeaturePoissonModel, df::AbstractDataFrame, fs::SLFP_Features.FeatureSet, chain::Chains)
    data = fs.data; nteams = Int(data[:n_teams]); ns = size(chain, 1) * size(chain, 3)
    inter = SLFP_PG.extract_interception(chain, model.interception_config, Int(data[:n_seasons]))
    ha = SLFP_PG.extract_home_advantage(chain, model.homeadvantage_config, nteams)
    dyn = SLFP_PG.extract_dynamics(chain, model.dynamics_config, "dyn", nteams)
    w_w = vec(Array(chain[:w_wealth]))
    w_d = vec(Array(chain[:w_distance]))
    wealth, distance = slfp_feature_values(model, fs, df)
    tm = data[:team_map]; out = Dict{Int,NamedTuple}()
    for (i, row) in enumerate(eachrow(df))
        h = get(tm, row.home_team, 0); a = get(tm, row.away_team, 0)
        αh = h > 0 ? dyn.α[:, h] : zeros(ns); βh = h > 0 ? dyn.β[:, h] : zeros(ns)
        αa = a > 0 ? dyn.α[:, a] : zeros(ns); βa = a > 0 ? dyn.β[:, a] : zeros(ns)
        γ = h > 0 ? ha[:, min(h, size(ha, 2))] : (size(ha, 2) == 1 ? ha[:, 1] : zeros(ns))
        season = hasproperty(row, :season_idx) ? Int(row.season_idx) : size(inter.μ_base, 2)
        base = inter.μ_base[:, season] .+ inter.δ_month[:, Dates.month(row.match_date)]
        shift = w_w .* wealth[i] .+ w_d .* distance[i]
        λh = exp.(base .+ γ .+ αh .+ βa .+ shift)
        λa = exp.(base .+ αa .+ βh .- shift)
        out[Int(row.match_id)] = (; λ_h = λh, λ_a = λa, true_xg_h = λh, true_xg_a = λa)
    end
    out
end

function slfp_params(model, vi)
    v = Dict(string(k) => vi[k] for k in keys(vi))
    SLFPParams(μ=Float64(v["inter.μ"]), γ=Float64(v["ha.γ_global"]), σ_a=Float64(v["dyn.σ_a"]), σ_d=Float64(v["dyn.σ_d"]),
        w_wealth=Float64(v["w_wealth"]),
        w_distance=Float64(v["w_distance"]),
        raw_a=Float64.(v["dyn.raw_a"]), raw_d=Float64.(v["dyn.raw_d"]))
end

function slfp_reference_extract(model, p::SLFPParams, fixture, fs)
    α, β = slfp_team_effects(p); tm = fs.data[:team_map]
    h = get(tm, fixture.home_team, 0); a = get(tm, fixture.away_team, 0)
    αh = h > 0 ? α[h] : 0.0; βh = h > 0 ? β[h] : 0.0; αa = a > 0 ? α[a] : 0.0; βa = a > 0 ? β[a] : 0.0
    wealth, distance = slfp_feature_values(model, fs, DataFrame(fixture))
    shift = p.w_wealth * wealth[1] + p.w_distance * distance[1]
    λh = exp(p.μ + p.γ + αh + βa + shift); λa = exp(p.μ + αa + βh - shift)
    (; λ_h=λh, λ_a=λa, true_xg_h=λh, true_xg_a=λa)
end

function slfp_sites(model, n)
    # Both coefficients are sampled by the shared vectorised engine.  An inactive
    # pillar is exactly multiplied by zero, so its coefficient is prior-only.
    sites = String["inter.μ", "ha.γ_global", "dyn.σ_a", "dyn.σ_d", "w_wealth", "w_distance"]
    append!(sites, ["dyn.raw_a[$i]" for i in 1:n]); append!(sites, ["dyn.raw_d[$i]" for i in 1:n]); sites
end

function slfp_draws(model, n, draws; seed=20260826)
    rng = MersenneTwister(seed)
    [SLFPParams(μ=.2+.1randn(rng), γ=.2+.1randn(rng), σ_a=.2+.1rand(rng), σ_d=.2+.1rand(rng),
        w_wealth=_slfp_has_wealth(model) ? .1rand(rng) : 0.0, w_distance=_slfp_has_distance(model) ? .1rand(rng) : 0.0,
        raw_a=randn(rng,n), raw_d=randn(rng,n)) for _ in 1:draws]
end

function slfp_logjoint(model, p, data)
    α, β = slfp_team_effects(p)
    shift = p.w_wealth .* data.wealth .+ p.w_distance .* data.distance
    ηh = p.μ .+ p.γ .+ α[data.home] .+ β[data.away] .+ shift
    ηa = p.μ .+ α[data.away] .+ β[data.home] .- shift
    lp = logpdf(model.interception_config.μ, p.μ) + logpdf(model.homeadvantage_config.γ_global, p.γ) +
         logpdf(model.dynamics_config.σ_att, p.σ_a) + logpdf(model.dynamics_config.σ_def, p.σ_d) +
         sum(logpdf.(Normal(), p.raw_a)) + sum(logpdf.(Normal(), p.raw_d))
    lp += logpdf(model.wealth_prior, p.w_wealth)
    lp += logpdf(model.distance_prior, p.w_distance)
    lf_h = loggamma.(Float64.(data.yh) .+ 1.0); lf_a = loggamma.(Float64.(data.ya) .+ 1.0)
    lp + sum(data.weights .* (data.yh .* ηh .- exp.(ηh) .- lf_h)) + sum(data.weights .* (data.ya .* ηa .- exp.(ηa) .- lf_a))
end
