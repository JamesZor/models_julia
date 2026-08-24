module RebuildExtractionRecombination

using BayesianFootball
using DataFrames
using Dates
using MCMCChains
using Distributions
using Statistics

include(joinpath(@__DIR__, "l02_rebuild_features.jl"))
include(joinpath(@__DIR__, "l03_rebuild_equations.jl"))
include(joinpath(@__DIR__, "l04_rebuild_turing_model.jl"))
using .RebuildFeatures
using .RebuildEquations
using .RebuildTuringModel

import BayesianFootball.Models.PreGame
import BayesianFootball.Predictions

export ScottishLowerNPNOGRecombinedPoissonModel, primitive_turing_var_labels,
       validate_primitive_chain, stack_chain_draws, extract_primitive_draws,
       validate_inference_registry, direct_total_poisson_reference

"""Parameter-section labels must be exactly the model primitive manifest.
Sampler internals are deliberately not inspected or extracted."""
function validate_primitive_chain(chain::Chains, J::Integer)
    expected = primitive_turing_var_labels(J)
    observed = Symbol.(MCMCChains.names(chain, :parameters))
    observed == expected || throw(ArgumentError("primitive chain manifest mismatch; expected $expected, got $observed"))
    return expected
end

"""Select explicit labels and stack iteration × chain in chain-major draw order."""
function stack_chain_draws(chain::Chains, labels::AbstractVector{Symbol})
    available = Symbol.(MCMCChains.names(chain, :parameters))
    all(x -> x in available, labels) || throw(ArgumentError("requested labels are not all parameter labels"))
    # Index labels explicitly, rather than relying on the chain's incidental storage order.
    # `Array(chain)` in the pinned MCMCChains version is already flattened; use
    # the underlying labelled AxisArray to retain and explicitly control all axes.
    selected = Array(chain.value[:, labels, :])
    niter, nparam, nchains = size(selected)
    nparam == length(labels) || throw(ArgumentError("explicit chain selection changed label count"))
    return permutedims(selected, (1, 3, 2)) |> x -> reshape(x, niter * nchains, nparam)
end

function extract_primitive_draws(chain::Chains, J::Integer)
    labels = validate_primitive_chain(chain, J)
    raw = stack_chain_draws(chain, labels)
    D = size(raw, 1); p = [unflatten_primitives(vec(raw[d, :]), J) for d in 1:D]
    transforms = transformed_parameters.(p)
    # Scalars are vectors; team/month quantities retain draw as their first dimension.
    bundle = Dict{Symbol,Any}(:draw_count => D, :primitive_labels => labels,
        :zA => reduce(vcat, (reshape(collect(x), 1, :) for x in getproperty.(p, :zA))),
        :zD => reduce(vcat, (reshape(collect(x), 1, :) for x in getproperty.(p, :zD))),
        :zM => reduce(vcat, (reshape(collect(x), 1, :) for x in getproperty.(p, :zM))),
        :alpha => reduce(vcat, (reshape(collect(x), 1, :) for x in getproperty.(transforms, :alpha))),
        :beta => reduce(vcat, (reshape(collect(x), 1, :) for x in getproperty.(transforms, :beta))),
        :M => reduce(vcat, (reshape(collect(x), 1, :) for x in getproperty.(transforms, :M))),
        :L => reduce(vcat, (reshape(collect(x), 1, :) for x in getproperty.(transforms, :L))))
    for f in (:kappa_A,:kappa_D,:mu_Y,:Delta,:xi_M,:b_Y,:pen_base,:pen_home,:q_pen,:lambda_og)
        bundle[f] = Float64.(getproperty.(p, f))
    end
    for f in (:tau_A,:tau_D,:sigma_M)
        bundle[f] = Float64.(getproperty.(transforms, f))
    end
    return bundle
end

function PreGame.extract_parameters(model::ScottishLowerNPNOGRecombinedPoissonModel, chain::Chains, fs)
    get(fs, :registry_fingerprint, nothing) == model.registry_fingerprint || throw(ArgumentError("FeatureSet/model registry fingerprint mismatch"))
    return extract_primitive_draws(chain, Int(fs[:n_teams]))
end

# Canonical registry is authoritative. DataStore fields are expected to be provider slugs,
# but an exact provider display name is also accepted for callers that retained names.
function _checked_oos_identity(row, registry_row, side::Symbol, fs)
    id = RebuildFeatures._int(registry_row[Symbol(side, "_id")])
    slug, name = RebuildFeatures._text(registry_row[Symbol(side, "_slug")]), RebuildFeatures._text(registry_row[Symbol(side, "_name")])
    dscol = Symbol(side, "_team")
    if dscol in propertynames(row)
        supplied = RebuildFeatures._text(row[dscol])
        isempty(supplied) && throw(ArgumentError("missing DataStore $dscol for match $(row.match_id)"))
        supplied in (slug, name) || throw(ArgumentError("DataStore $dscol='$supplied' conflicts with registry '$slug'/'$name' for match $(row.match_id)"))
    end
    return resolve_oos_identity(fs; canonical_id=id, slug=slug, name=name)
end

"""Validate the immutable OOS registry without requiring it to equal training `fs`.

Training must retain its exact boundary-only registry for the feature adapter.  Inference
uses a separate immutable boundary∪next-step registry, but it must be the same model
version and configuration; only its registry fingerprint is intentionally different.
"""
function validate_inference_registry(inference_model::ScottishLowerNPNOGRecombinedPoissonModel,
    training_model::ScottishLowerNPNOGRecombinedPoissonModel, fs)
    registry_fingerprint(training_model.registry) == training_model.registry_fingerprint || throw(ArgumentError("training registry mutated"))
    registry_fingerprint(inference_model.registry) == inference_model.registry_fingerprint || throw(ArgumentError("inference registry mutated"))
    get(fs, :registry_fingerprint, nothing) == training_model.registry_fingerprint || throw(ArgumentError("FeatureSet/training registry fingerprint mismatch"))
    get(fs, :half_life_days, nothing) == training_model.half_life_days || throw(ArgumentError("FeatureSet/training half-life mismatch"))
    get(fs, :model_version, nothing) == training_model.version || throw(ArgumentError("FeatureSet/training model version mismatch"))
    get(fs, :own_goal_policy, nothing) == training_model.own_goal_policy || throw(ArgumentError("FeatureSet/training policy mismatch"))
    inference_model.version == training_model.version == get(fs, :model_version, nothing) || throw(ArgumentError("inference model version mismatch"))
    inference_model.half_life_days == training_model.half_life_days == get(fs, :half_life_days, nothing) || throw(ArgumentError("inference model half-life mismatch"))
    inference_model.own_goal_policy == training_model.own_goal_policy == get(fs, :own_goal_policy, nothing) || throw(ArgumentError("inference model policy mismatch"))
    return inference_model.registry_fingerprint
end

"""OOS extraction uses registry identity and stored history maps only; no outcome column is read."""
function PreGame.extract_parameters(inference_model::ScottishLowerNPNOGRecombinedPoissonModel,
    df::AbstractDataFrame, fs, chain::Chains, training_model::ScottishLowerNPNOGRecombinedPoissonModel)::Dict
    validate_inference_registry(inference_model, training_model, fs)
    # Do not use the model/chain/fs overload: its exact training fingerprint check is correct
    # for fitting but deliberately cannot apply to the separate inference registry.
    primitive = extract_primitive_draws(chain, Int(fs[:n_teams]))
    registry = Dict(Int(r.match_id) => r for r in eachrow(inference_model.registry))
    out = Dict{Int,Any}()
    fallback_counts = Dict{Symbol,Int}(:history_seen => 0, :target_only_population_fallback => 0, :unknown_identity => 0)
    for row in eachrow(df)
        mid = Int(row.match_id); haskey(registry, mid) || throw(ArgumentError("OOS match $mid absent from model registry"))
        rr = registry[mid]
        tournament = Int(rr.tournament_id)
        (:tournament_id in propertynames(row) && Int(row.tournament_id) != tournament) && throw(ArgumentError("OOS tournament conflicts with registry for match $mid"))
        haskey(RebuildFeatures.LEAGUE_MAP, tournament) || throw(ArgumentError("unknown tournament $tournament"))
        league = RebuildFeatures.LEAGUE_MAP[tournament]
        month_id = month(RebuildFeatures._date(row)); 1 <= month_id <= 12 || throw(ArgumentError("invalid month"))
        h = _checked_oos_identity(row, rr, :home, fs); a = _checked_oos_identity(row, rr, :away, fs)
        fallback_counts[h.status] = get(fallback_counts, h.status, 0) + 1
        fallback_counts[a.status] = get(fallback_counts, a.status, 0) + 1
        D = primitive[:draw_count]
        αh = h.column == 0 ? zeros(D) : copy(view(primitive[:alpha], :, h.column))
        βh = h.column == 0 ? zeros(D) : copy(view(primitive[:beta], :, h.column))
        αa = a.column == 0 ? zeros(D) : copy(view(primitive[:alpha], :, a.column))
        βa = a.column == 0 ? zeros(D) : copy(view(primitive[:beta], :, a.column))
        λYh=zeros(D); λYa=zeros(D); λPh=zeros(D); λPa=zeros(D); q=zeros(D); λog=zeros(D)
        for d in 1:D
            p = (zA=vec(primitive[:zA][d,:]), zD=vec(primitive[:zD][d,:]), kappa_A=primitive[:kappa_A][d], kappa_D=primitive[:kappa_D][d], mu_Y=primitive[:mu_Y][d], Delta=primitive[:Delta][d], zM=vec(primitive[:zM][d,:]), xi_M=primitive[:xi_M][d], b_Y=primitive[:b_Y][d], pen_base=primitive[:pen_base][d], pen_home=primitive[:pen_home][d], q_pen=primitive[:q_pen][d], lambda_og=primitive[:lambda_og][d])
            r = predictive_component_rates(p, [league], [month_id], [αh[d]], [βh[d]], [αa[d]], [βa[d]])
            λYh[d], λYa[d] = r.lambda_Y_home[1], r.lambda_Y_away[1]
            λPh[d], λPa[d], q[d], λog[d] = r.lambda_penalty_award_home, r.lambda_penalty_award_away, r.q_pen, r.lambda_og
        end
        cph, cpa = q .* λPh, q .* λPa
        out[mid] = Dict{Symbol,Any}(:match_id=>mid, :league_id=>league, :month_id=>month_id,
            :home_canonical_id=>h.canonical_id, :away_canonical_id=>a.canonical_id,
            :home_team_status=>h.status, :away_team_status=>a.status,
            :lambda_Y_home=>λYh, :lambda_Y_away=>λYa, :lambda_open_home=>λYh, :lambda_open_away=>λYa,
            :lambda_penalty_award_home=>λPh, :lambda_penalty_award_away=>λPa, :q_pen=>q,
            :lambda_converted_penalty_home=>cph, :lambda_converted_penalty_away=>cpa,
            :lambda_og_home=>λog, :lambda_og_away=>copy(λog),
            :lambda_h=>λYh .+ cph .+ λog, :lambda_a=>λYa .+ cpa .+ λog,
            :registry_fingerprint=>inference_model.registry_fingerprint, :provenance=>:registry_stored_history_map)
    end
    # Repeat one immutable summary on each latent row so standard Dict→DataFrame callers
    # retain diagnostics without a side channel or an outcome-derived lookup.
    diagnostic = (history_seen=fallback_counts[:history_seen],
        target_only_population_fallback=fallback_counts[:target_only_population_fallback],
        unknown_identity=fallback_counts[:unknown_identity])
    for latent in values(out)
        latent[:fallback_diagnostics] = diagnostic
    end
    return out
end

# Backward-compatible exact-registry path used by Stages 6–7. Genuine Stage-8 OOS
# passes distinct fitting and inference models through the five-argument method above.
function PreGame.extract_parameters(model::ScottishLowerNPNOGRecombinedPoissonModel,
    df::AbstractDataFrame, fs, chain::Chains)::Dict
    return PreGame.extract_parameters(model, df, fs, chain, model)
end

Predictions.extract_params(::ScottishLowerNPNOGRecombinedPoissonModel, row) = (
    lambda_Y_home=row.lambda_Y_home, lambda_Y_away=row.lambda_Y_away,
    lambda_converted_penalty_home=row.lambda_converted_penalty_home,
    lambda_converted_penalty_away=row.lambda_converted_penalty_away,
    lambda_og_home=row.lambda_og_home, lambda_og_away=row.lambda_og_away,
    λ_h=row.lambda_h, λ_a=row.lambda_a)
Predictions.get_latent_column_symbols(::ScottishLowerNPNOGRecombinedPoissonModel, df::AbstractDataFrame) =
    Symbol[:match_id, :lambda_Y_home, :lambda_Y_away, :lambda_converted_penalty_home, :lambda_converted_penalty_away, :lambda_og_home, :lambda_og_away, :lambda_h, :lambda_a]

"""Direct total-Poisson grid, retained solely as a test oracle for the component convolution."""
function direct_total_poisson_reference(params; max_goals::Int)
    D=length(params.λ_h); S=zeros(Float64,max_goals,max_goals,D)
    for d in 1:D
        ph=[pdf(Poisson(params.λ_h[d]), k) for k in 0:max_goals-1]; pa=[pdf(Poisson(params.λ_a[d]), k) for k in 0:max_goals-1]
        ph ./= sum(ph); pa ./= sum(pa); S[:,:,d] .= ph * pa'
    end
    return S
end

function _component_pmf(λ::Float64, K::Int)
    return [pdf(Poisson(λ), k) for k in 0:K-1]
end
function _convolve3(a,b,c,K)
    out=zeros(Float64,K)
    for i in 1:K, j in 1:K, k in 1:K
        g=i+j+k-3; g < K && (out[g+1] += a[i]*b[j]*c[k])
    end
    out ./= sum(out); return out
end
function Predictions.compute_score_matrix(::ScottishLowerNPNOGRecombinedPoissonModel, params; max_goals::Int=12, tail_tol::Real=1e-10)
    tail_tol > 0 || throw(ArgumentError("tail_tol must be positive")); D=length(params.λ_h); all(length(x)==D for x in (params.λ_a,params.lambda_Y_home,params.lambda_Y_away,params.lambda_converted_penalty_home,params.lambda_converted_penalty_away,params.lambda_og_home,params.lambda_og_away)) || throw(DimensionMismatch("draw vectors differ"))
    K=max(12,max_goals)
    while any(ccdf(Poisson(λ), K-1) > tail_tol for λ in vcat(Float64.(params.λ_h),Float64.(params.λ_a)))
        K += 1
    end
    S=zeros(Float64,K,K,D)
    for d in 1:D
        ph=_convolve3(_component_pmf(params.lambda_Y_home[d],K),_component_pmf(params.lambda_converted_penalty_home[d],K),_component_pmf(params.lambda_og_home[d],K),K)
        pa=_convolve3(_component_pmf(params.lambda_Y_away[d],K),_component_pmf(params.lambda_converted_penalty_away[d],K),_component_pmf(params.lambda_og_away[d],K),K)
        S[:,:,d] .= ph * pa'; S[:,:,d] ./= sum(S[:,:,d])
    end
    all(isfinite,S) && all(>=(0),S) && all(isapprox(sum(S[:,:,d]),1; atol=1e-12) for d in 1:D) || error("invalid score tensor")
    return Predictions.ScoreMatrix(S)
end

end # module
