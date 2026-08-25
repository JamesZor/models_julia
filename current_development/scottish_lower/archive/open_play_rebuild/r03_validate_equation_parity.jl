# Stage 4 notebook: pure equation parity only. No model, sampling, extraction, writes, or artifact output.

# %% BLOCK 1 — obtain the same read-only Stage-3 FeatureSet/registry path
using BayesianFootball
using DataFrames
using ForwardDiff
const BFData = BayesianFootball.Data
include(joinpath(@__DIR__, "l02_rebuild_features.jl")); using .RebuildFeatures
include(joinpath(@__DIR__, "l03_rebuild_equations.jl")); using .RebuildEquations

# %% BLOCK 0 — self-contained synthetic oracle/parity gate (no database access)
synthetic = Dict{Symbol,Any}(
    :n_teams => 2, :home_team => [1, 2, 1], :away_team => [2, 1, 2],
    :Y_home => [1, 0, 2], :Y_away => [0, 3, 1],
    :A_home => [1, 0, 2], :A_away => [0, 1, 1],
    :C_home => [1, 0, 1], :C_away => [0, 1, 0],
    :O_home => [0, 1, 0], :O_away => [1, 0, 0],
    :month_ids => [1, 6, 12], :league_ids => [1, 2, 1],
    :weights => [1.0, 0.7, 0.25])
p_synthetic = (zA=[-0.2, 0.2], zD=[0.1, -0.1], kappa_A=log(0.36),
    kappa_D=log(0.32), mu_Y=log(1.2), Delta=0.14,
    zM=collect(range(-0.22, 0.25; length=12)), xi_M=log(0.20), b_Y=0.08,
    pen_base=log(0.12), pen_home=0.04, q_pen=0.80, lambda_og=0.03)
data_synthetic = equation_data(synthetic)
@assert data_synthetic.league_sign == [1.0, -1.0, 1.0]
@assert all(x -> x isa Float64, values(data_synthetic)[15:end])
ll_synthetic = weighted_data_loglikelihood(data_synthetic, p_synthetic)
ll_synthetic_raw = weighted_data_loglikelihood_scalar(synthetic, p_synthetic)
@assert isapprox(ll_synthetic, ll_synthetic_raw; rtol=0, atol=1e-12)
θ_synthetic = collect(flatten_primitives(p_synthetic))
f_hot = x -> weighted_data_loglikelihood(data_synthetic, unflatten_primitives(x, 2))
f_raw = x -> weighted_data_loglikelihood_scalar(synthetic, unflatten_primitives(x, 2))
@assert isapprox(f_hot(θ_synthetic), f_raw(θ_synthetic); rtol=0, atol=1e-12)
@assert all(isapprox.(ForwardDiff.gradient(f_hot, θ_synthetic), ForwardDiff.gradient(f_raw, θ_synthetic);
    rtol=1e-10, atol=1e-10))
println("Synthetic raw-row likelihood and ForwardDiff gradient parity passed.")

ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
splitter = BFData.GroupedCVConfig(tournament_groups=[[56,57]], target_seasons=["24/25","25/26"],
    history_seasons=2, dynamics_col=:match_biweek, warmup_period=0, stop_early=true)
boundaries = BFData.create_id_boundaries(ds, splitter)
by_match = Dict(Int(r.match_id) => Int(r.tournament_id) for r in eachrow(ds.matches))
fold = findlast(i -> Set(get(by_match, Int(id), -1) for id in first(boundaries[i]).history_match_ids) >= Set([56,57]), eachindex(boundaries))
isnothing(fold) && error("No pooled 56/57 history boundary")
boundary = first(boundaries[fold])
haskey(ENV, "BF_DB_URL") || error("BF_DB_URL is required for the validated Stage-3 registry path")
registry = fetch_canonical_registry(vcat(Int.(boundary.history_match_ids), Int.(boundary.target_match_ids)))
fs = build_rebuild_feature_set(boundary, ds, registry; half_life_days=365, own_goal_policy=:beneficiary)
validate_feature_set(fs)
data = equation_data(fs)
@assert Set(fs[:league_ids]) == Set([1,2])
@assert data.league_sign == ifelse.(fs[:league_ids] .== 1, 1.0, -1.0)
@assert data.weights isa Vector{Float64} && all(x -> x isa Float64, values(data)[15:end])
println("Stage 4 boundary $fold: $(length(fs[:history_match_ids])) reconciled rows; $(fs[:n_teams]) teams.")

# %% BLOCK 2 — deterministic interior primitive contract and transforms
J = fs[:n_teams]
p = (zA=collect(range(-0.31, 0.29; length=J)), zD=collect(range(0.27, -0.33; length=J)),
    kappa_A=log(0.36), kappa_D=log(0.32), mu_Y=log(1.20), Delta=0.14,
    zM=collect(range(-0.22, 0.25; length=12)), xi_M=log(0.20), b_Y=0.08,
    pen_base=log(0.12), pen_home=0.04, q_pen=0.80, lambda_og=0.03)
validate_primitive_parameters(p; n_teams=J)
@assert length(flatten_primitives(p)) == primitive_parameter_length(J)
@assert unflatten_primitives(collect(flatten_primitives(p)), J) == p
p_before = deepcopy(p); fs_before = deepcopy(fs.data)
t = transformed_parameters(p)
@assert abs(sum(t.alpha)) <= 1e-14 && abs(sum(t.beta)) <= 1e-14 && abs(sum(t.M)) <= 1e-14
@assert t.L == (p.Delta / 2, -p.Delta / 2) && sum(t.L) == 0
@assert p == p_before && fs.data == fs_before # pure transforms/hot path do not mutate inputs
println("Primitive manifest, dimensions/support, flattening, and centered transforms passed.")

# %% BLOCK 3 — vectorized rates versus independent scalar per-row equations
r = component_rates(data, p)
for i in eachindex(fs[:weights]) # validation reference only, never a model hot path
    h, a, l, m = fs[:home_team][i], fs[:away_team][i], fs[:league_ids][i], fs[:month_ids][i]
    ηh = p.mu_Y + t.L[l] + t.M[m] + p.b_Y + t.alpha[h] + t.beta[a]
    ηa = p.mu_Y + t.L[l] + t.M[m] + t.alpha[a] + t.beta[h]
    @assert abs(r.eta_Y_home[i] - ηh) <= 1e-10 && abs(r.eta_Y_away[i] - ηa) <= 1e-10
    smooth_bound(x) = 20.0 * tanh(x / 20.0)
    @assert abs(r.lambda_Y_home[i] - (exp(smooth_bound(ηh)) + 1e-6)) <= 1e-10
    @assert abs(r.lambda_Y_away[i] - (exp(smooth_bound(ηa)) + 1e-6)) <= 1e-10
end
@assert all(isfinite, r.lambda_Y_home) && all(isfinite, r.lambda_Y_away)
@assert all(fs[:C_home] .<= fs[:A_home]) && all(fs[:C_away] .<= fs[:A_away])
@assert p == p_before && fs.data == fs_before
println("Both-league vectorized per-row rate parity and no-mutation checks passed.")

# %% BLOCK 4 — complete weighted data likelihood and thinning identities
ll_vector = weighted_data_loglikelihood(data, p)
ll_scalar = weighted_data_loglikelihood_scalar(fs, p)
@assert isfinite(ll_vector) && abs(ll_vector - ll_scalar) <= 1e-10
pred = predictive_component_rates(p, fs[:league_ids], fs[:month_ids],
    view(t.alpha, fs[:home_team]), view(t.beta, fs[:home_team]),
    view(t.alpha, fs[:away_team]), view(t.beta, fs[:away_team]))
@assert pred.lambda_Y_home == r.lambda_Y_home && pred.lambda_Y_away == r.lambda_Y_away
@assert pred.lambda_penalty_award_home == r.lambda_pen_home
@assert pred.lambda_penalty_award_away == r.lambda_pen_away
@assert pred.q_pen == p.q_pen && pred.lambda_converted_pen_home == p.q_pen * r.lambda_pen_home
@assert pred.lambda_converted_pen_away == p.q_pen * r.lambda_pen_away && pred.lambda_og == p.lambda_og
println("Weighted data-only likelihood parity: $ll_vector; Poisson-thinning identities passed.")

# %% BLOCK 5 — extreme finite smooth-saturation safety (interior supports retained)
p_extreme = merge(p, (mu_Y=1e6, Delta=-1e6, b_Y=1e6, pen_base=1e6, pen_home=-2e6,
    kappa_A=8.0, kappa_D=8.0, xi_M=8.0))
validate_primitive_parameters(p_extreme; n_teams=J)
r_extreme = component_rates(fs, p_extreme)
@assert all(isfinite, r_extreme.lambda_Y_home) && all(isfinite, r_extreme.lambda_Y_away)
@assert isfinite(r_extreme.lambda_pen_home) && isfinite(r_extreme.lambda_pen_away)
@assert all(>=(1e-6), r_extreme.lambda_Y_home) && r_extreme.lambda_pen_home >= 1e-6
println("Extreme smooth-saturation/floor safety passed.")

# %% BLOCK 6 — ForwardDiff gradient and central-difference spot checks for the pure likelihood
θ = collect(flatten_primitives(p))
f = x -> weighted_data_loglikelihood(fs, unflatten_primitives(x, J))
g = ForwardDiff.gradient(f, θ)
@assert all(isfinite, g)
for k in (1, J + 1, 2J + 3, 2J + 16, length(θ))
    ε = 1e-6
    xp, xm = copy(θ), copy(θ); xp[k] += ε; xm[k] -= ε
    central = (f(xp) - f(xm)) / (2ε)
    @assert isapprox(g[k], central; rtol=2e-5, atol=2e-5)
end
println("ForwardDiff finite gradient and central-difference spot checks passed.")
