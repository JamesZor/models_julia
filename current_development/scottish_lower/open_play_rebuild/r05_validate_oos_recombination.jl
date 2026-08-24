# Stage 6 notebook: deterministic extraction/recombination validation only.
# No sampling, persistence, database writes, or source edits occur in this notebook.

# %% BLOCK 1 — reuse the audited Stage-3 adapter and select metadata-only OOS fixtures
using BayesianFootball
using DataFrames
using Dates
using MCMCChains
using Distributions
using Statistics
const BFData = BayesianFootball.Data
const BFFeatures = BayesianFootball.Features
const BFPreGame = BayesianFootball.Models.PreGame
const BFPred = BayesianFootball.Predictions
include(joinpath(@__DIR__, "l05_rebuild_extraction_recombination.jl")); using .RebuildExtractionRecombination
# Stage-6 owns one coherent nested copy of the Stage-3/4 loader stack; do not include
# predecessors separately or Julia will create incompatible prototype-local model types.
const RebuildFeatures = RebuildExtractionRecombination.RebuildFeatures
const RebuildEquations = RebuildExtractionRecombination.RebuildEquations
using .RebuildExtractionRecombination.RebuildFeatures
using .RebuildExtractionRecombination.RebuildEquations

ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
splitter = BFData.GroupedCVConfig(tournament_groups=[[56,57]], target_seasons=["24/25","25/26"], history_seasons=2, dynamics_col=:match_biweek, warmup_period=0, stop_early=true)
boundaries = BFData.create_id_boundaries(ds, splitter)
by_match = Dict(Int(r.match_id) => Int(r.tournament_id) for r in eachrow(ds.matches))
fold = findlast(i -> Set(get(by_match, Int(id), -1) for id in first(boundaries[i]).history_match_ids) >= Set([56,57]), eachindex(boundaries))
isnothing(fold) && error("No pooled 56/57 history boundary")
boundary = first(boundaries[fold])
haskey(ENV, "BF_DB_URL") || error("BF_DB_URL is required for the existing read-only Stage-3 registry path")
registry = fetch_canonical_registry(vcat(Int.(boundary.history_match_ids), Int.(boundary.target_match_ids)))
model = ScottishLowerNPNOGRecombinedPoissonModel(registry)
fs = BFFeatures.create_features(boundary, ds, model, :match_biweek)
validate_feature_set(fs)
# Deliberately copy only identity/timing metadata. This notebook never reads target scores/events.
target = Set(Int.(boundary.target_match_ids))
meta = DataFrame(match_id=Int[], tournament_id=Int[], home_team=String[], away_team=String[], match_date=Date[])
for r in eachrow(ds.matches)
    Int(r.match_id) in target || continue
    push!(meta, (Int(r.match_id), Int(r.tournament_id), String(r.home_team), String(r.away_team), RebuildFeatures._date(r)))
end
registry_by_match = Dict(Int(r.match_id) => r for r in eachrow(registry))
known_known(r) = begin rr = registry_by_match[Int(r.match_id)]; RebuildFeatures._int(rr.home_id) in keys(fs[:team_map]) && RebuildFeatures._int(rr.away_id) in keys(fs[:team_map]) end
# Require a known-known metadata-only fixture from each source league.
chosen_ids = [Int(first(filter(r -> r.tournament_id == t && known_known(r), eachrow(meta))).match_id) for t in (56,57)]
oos = filter(:match_id => in(chosen_ids), meta)
# Include target-only East Kilbride whenever it is in this split, without using outcomes.
ek = filter(r -> r.home_team == "east-kilbride" || r.away_team == "east-kilbride", eachrow(meta))
!isempty(ek) && push!(oos, first(ek))
println("Stage 6 fold $fold: $(nrow(oos)) metadata-only OOS fixtures across $(unique(oos.tournament_id)).")

# %% BLOCK 2 — deterministic two-chain synthetic posterior and exact primitive manifest
J = fs[:n_teams]; labels = primitive_turing_var_labels(J); I, C = 12, 2
raw = Array{Float64}(undef, I, length(labels), C)
for c in 1:C, p in eachindex(labels), i in 1:I
    # Interior q/lambda values are assigned below; every value is nonconstant by draw and chain.
    raw[i,p,c] = 0.01 * p + 0.001 * i + 0.0001 * c
end
for (p, label) in enumerate(labels)
    label == :q_pen && (raw[:,p,:] .= [0.72 + 0.002i + 0.001c for i in 1:I, c in 1:C])
    label == :lambda_og && (raw[:,p,:] .= [0.025 + 0.0002i + 0.0001c for i in 1:I, c in 1:C])
end
chain = Chains(copy(raw), labels)
raw_before = copy(Array(chain))
@assert Symbol.(MCMCChains.names(chain, :parameters)) == labels
@assert validate_primitive_chain(chain, J) == labels
@assert size(stack_chain_draws(chain, labels)) == (I*C, length(labels))
bundle = BFPreGame.extract_parameters(model, chain, fs)
@assert bundle[:draw_count] == I*C
@assert all(size(bundle[k]) == (I*C,J) for k in (:zA,:zD,:alpha,:beta))
@assert size(bundle[:zM]) == size(bundle[:M]) == (I*C,12)
@assert all(length(bundle[k]) == I*C for k in (:tau_A,:tau_D,:sigma_M,:q_pen,:lambda_og))
@assert all(abs.(vec(sum(bundle[:alpha]; dims=2))) .< 1e-12)
@assert all(abs.(vec(sum(bundle[:beta]; dims=2))) .< 1e-12)
@assert all(abs.(vec(sum(bundle[:M]; dims=2))) .< 1e-12)
@assert Array(chain) == raw_before # extraction is read-only with respect to raw chain storage
println("Manifest/order, $(I*C) stacked draws, transforms, centering, and no-mutation passed.")

# %% BLOCK 3 — Stage-4 transform and OOS equation parity, including fallback/statuses
for d in (1, I*C)
    p = unflatten_primitives(vec(stack_chain_draws(chain, labels)[d,:]), J)
    t = transformed_parameters(p)
    @assert isapprox(t.alpha, bundle[:alpha][d,:]; atol=1e-12, rtol=0)
    @assert isapprox(t.beta, bundle[:beta][d,:]; atol=1e-12, rtol=0)
    @assert isapprox(t.M, bundle[:M][d,:]; atol=1e-12, rtol=0)
end
preds = BFPreGame.extract_parameters(model, oos, fs, chain)
@assert all(length(preds[Int(r.match_id)][:lambda_h]) == I*C for r in eachrow(oos))
@assert all(all(isfinite, preds[Int(r.match_id)][:lambda_h]) && all(isfinite, preds[Int(r.match_id)][:lambda_a]) for r in eachrow(oos))
@assert Set(preds[Int(r.match_id)][:league_id] for r in eachrow(oos)) == Set([1,2])
@assert all(preds[Int(r.match_id)][:home_team_status] in (:history_seen,:target_only_population_fallback,:unknown_identity) for r in eachrow(oos))
@assert all(preds[Int(r.match_id)][:away_team_status] in (:history_seen,:target_only_population_fallback,:unknown_identity) for r in eachrow(oos))
if !isempty(ek)
    @assert any(preds[Int(r.match_id)][:home_team_status] == :target_only_population_fallback || preds[Int(r.match_id)][:away_team_status] == :target_only_population_fallback for r in eachrow(oos) if Int(r.match_id) == Int(first(ek).match_id))
end
# Independently re-evaluate the Stage-4 predictive equation for a representative fixture/draw.
mid = Int(oos[1,:match_id]); x = preds[mid]; d = 1
p = unflatten_primitives(vec(stack_chain_draws(chain, labels)[d,:]), J)
hcol = resolve_oos_identity(fs; canonical_id=x[:home_canonical_id]).column; acol = resolve_oos_identity(fs; canonical_id=x[:away_canonical_id]).column
αh = hcol == 0 ? 0.0 : bundle[:alpha][d,hcol]; βh = hcol == 0 ? 0.0 : bundle[:beta][d,hcol]
αa = acol == 0 ? 0.0 : bundle[:alpha][d,acol]; βa = acol == 0 ? 0.0 : bundle[:beta][d,acol]
r = predictive_component_rates(p, [x[:league_id]], [x[:month_id]], [αh], [βh], [αa], [βa])
@assert abs(r.lambda_Y_home[1] - x[:lambda_Y_home][d]) <= 1e-10
@assert abs(r.lambda_Y_away[1] - x[:lambda_Y_away][d]) <= 1e-10
# Swapping distinct known identities must change at least one oriented NP-NOG rate.
r_swap = predictive_component_rates(p, [x[:league_id]], [x[:month_id]], [αa], [βa], [αh], [βh])
@assert max(abs(r.lambda_Y_home[1] - r_swap.lambda_Y_home[1]),
            abs(r.lambda_Y_away[1] - r_swap.lambda_Y_away[1])) > 1e-12
println("OOS registry/map resolution, 56→1/57→2, fallback statuses, and equation parity passed.")

# %% BLOCK 4 — explicit component convolution and ordinary Prediction inference
latent_rows = [(; match_id=mid, lambda_Y_home=x[:lambda_Y_home], lambda_Y_away=x[:lambda_Y_away], lambda_converted_penalty_home=x[:lambda_converted_penalty_home], lambda_converted_penalty_away=x[:lambda_converted_penalty_away], lambda_og_home=x[:lambda_og_home], lambda_og_away=x[:lambda_og_away], lambda_h=x[:lambda_h], lambda_a=x[:lambda_a]) for (mid, x) in sort(collect(preds); by=first)]
latent = DataFrame(latent_rows)
params = BFPred.extract_params(model, latent[1,:]); score = BFPred.compute_score_matrix(model, params; max_goals=12)
S = BFPred.score_matrix_data(score)
@assert all(isfinite,S) && all(>=(0),S) && all(isapprox(sum(S[:,:,d]),1; atol=1e-10) for d in axes(S,3))
@assert all(ccdf(Distributions.Poisson(λ), size(S,1)-1) <= 1e-10 for λ in vcat(params.λ_h,params.λ_a))
reference = direct_total_poisson_reference(params; max_goals=size(S,1))
@assert maximum(abs.(S .- reference)) <= 1e-10
ls = BayesianFootball.Experiments.LatentStates(latent, model)
ppd = BFPred.model_inference(ls; verbose=false)
@assert all(length(d) == I*C for d in ppd.df.distribution)
@assert Set(preds[mid][:league_id] for mid in keys(preds)) == Set([1,2])
println("Convolution/tail/reference and standard model_inference market path passed ($(nrow(ppd.df)) PPD rows).")
