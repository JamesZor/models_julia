module RebuildTuringModel

using BayesianFootball
using DataFrames
using Turing
using Distributions

include(joinpath(@__DIR__, "l02_rebuild_features.jl"))
include(joinpath(@__DIR__, "l03_rebuild_equations.jl"))
using .RebuildFeatures
using .RebuildEquations

import BayesianFootball.Features
import BayesianFootball.Models.PreGame

export ScottishLowerNPNOGRecombinedPoissonModel, rebuild_turing_engine,
       primitive_turing_var_labels

const MODEL_VERSION = "open_play_rebuild_turing_v1"

"""Stage-5, model-owned configuration with an immutable-by-convention registry snapshot.

The constructor validates a defensive registry copy and retains its deterministic
fingerprint.  The adapter rechecks it before every feature build, so a caller cannot
silently mutate the `DataFrame` held by this prototype model.
"""
struct ScottishLowerNPNOGRecombinedPoissonModel <: BayesianFootball.TypesInterfaces.AbstractPregameModel
    registry::DataFrame
    registry_fingerprint::String
    half_life_days::Float64
    own_goal_policy::Symbol
    version::String
end

function ScottishLowerNPNOGRecombinedPoissonModel(registry::DataFrame;
    half_life_days::Real=365.0, own_goal_policy::Symbol=:beneficiary,
    version::AbstractString=MODEL_VERSION)
    half_life_days > 0 || throw(ArgumentError("half_life_days must be positive"))
    own_goal_policy == :beneficiary || throw(ArgumentError("only audited :beneficiary policy is allowed"))
    snapshot = copy(registry)
    checked = validate_canonical_registry(snapshot, Int.(snapshot.match_id))
    return ScottishLowerNPNOGRecombinedPoissonModel(checked.registry,
        checked.manifest.registry_fingerprint, Float64(half_life_days), own_goal_policy,
        String(version))
end

"""Expected sampled DynamicPPL labels. Deterministic transforms are deliberately absent."""
function primitive_turing_var_labels(J::Integer)
    J > 0 || throw(ArgumentError("J must be positive"))
    return vcat([Symbol("zA[$j]") for j in 1:J], [Symbol("zD[$j]") for j in 1:J],
        [:kappa_A, :kappa_D, :mu_Y, :Delta], [Symbol("zM[$m]") for m in 1:12],
        [:xi_M, :b_Y, :pen_base, :pen_home, :q_pen, :lambda_og])
end

"""Specialized model-owned feature adapter; it intentionally bypasses generic extraction."""
function Features.create_features(boundary::BayesianFootball.Data.SplitBoundary,
    ds::BayesianFootball.Data.DataStore,
    model::ScottishLowerNPNOGRecombinedPoissonModel, dynamics_col::Symbol)
    # dynamics_col is accepted for the common Features API; this model's fixed day-decay
    # builder derives all time information from the boundary/cutoff instead.
    # `model.registry` is the one immutable canonical registry for the entire run.
    # A FeatureSet must nevertheless carry only its explicit fit boundary rows: the
    # fitted history plus the genuine next-step OOS target.  This preserves the
    # builder's strict no-extra-rows contract without creating per-fold models.
    registry_fingerprint(model.registry) == model.registry_fingerprint ||
        throw(ArgumentError("global model registry fingerprint changed after construction"))
    ids = sort!(unique(vcat(Int.(boundary.history_match_ids), Int.(boundary.target_match_ids))))
    subset = filter(:match_id => x -> Int(x) in Set(ids), model.registry)
    checked = validate_canonical_registry(subset, ids; ds=ds)
    fs = build_rebuild_feature_set(boundary, ds, checked.registry;
        half_life_days=model.half_life_days, own_goal_policy=model.own_goal_policy)
    fs[:registry_fingerprint] == checked.manifest.registry_fingerprint ||
        throw(ArgumentError("FeatureSet registry is not the exact fit-boundary subset"))
    # Distinguish the fold subset identity from the immutable global model identity.
    fs[:model_registry_fingerprint] = model.registry_fingerprint
    return fs
end

"""The specialized builder owns the complete contract; generic extractors must add nothing."""
Features.required_features(::ScottishLowerNPNOGRecombinedPoissonModel) = Features.AbstractFeatureConfig[]

# No loops, branches, missing handling, dictionary access, or tracked-array mutation here.
# `data` is the concrete `equation_data(fs)` NamedTuple constructed outside the model.
@model function rebuild_turing_engine(data::NamedTuple)
    J = data.n_teams
    zA ~ filldist(Normal(0, 1), J)
    zD ~ filldist(Normal(0, 1), J)
    kappa_A ~ Normal(log(0.35), 0.5)
    kappa_D ~ Normal(log(0.35), 0.5)
    mu_Y ~ Normal(log(1.2), 0.5)
    Delta ~ Normal(0, 0.5)
    zM ~ filldist(Normal(0, 1), 12)
    xi_M ~ Normal(log(0.2), 0.5)
    b_Y ~ Normal(0, 0.35)
    pen_base ~ Normal(log(0.12), 1)
    pen_home ~ Normal(0, 0.35)
    q_pen ~ Beta(8, 2)
    lambda_og ~ Gamma(2, 0.015) # Julia's second argument is scale.

    primitives = (zA=zA, zD=zD, kappa_A=kappa_A, kappa_D=kappa_D, mu_Y=mu_Y,
        Delta=Delta, zM=zM, xi_M=xi_M, b_Y=b_Y, pen_base=pen_base,
        pen_home=pen_home, q_pen=q_pen, lambda_og=lambda_og)
    Turing.@addlogprob! weighted_data_loglikelihood(data, primitives)
end

"""Validate the FeatureSet outside AD and bake its typed array contract into the Turing model."""
function PreGame.build_turing_model(model::ScottishLowerNPNOGRecombinedPoissonModel, fs)
    registry_fingerprint(model.registry) == model.registry_fingerprint ||
        throw(ArgumentError("global model registry fingerprint changed after construction"))
    get(fs, :model_registry_fingerprint, nothing) == model.registry_fingerprint ||
        throw(ArgumentError("FeatureSet/global model registry fingerprint mismatch"))
    # The fold fingerprint is intentionally different for a global model, but it
    # must still be an exact fingerprint of the FeatureSet's stored manifest.
    get(fs, :registry_fingerprint, nothing) == get(get(fs, :registry_manifest, nothing), :registry_fingerprint, nothing) ||
        throw(ArgumentError("FeatureSet fold registry manifest mismatch"))
    data = equation_data(fs)
    data.n_teams == length(fs[:team_ids]) || throw(DimensionMismatch("team IDs and equation team dimension differ"))
    return rebuild_turing_engine(data)
end

end # module
