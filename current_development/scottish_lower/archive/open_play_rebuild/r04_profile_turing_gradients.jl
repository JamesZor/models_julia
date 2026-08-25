# Stage 5 notebook: Turing/AD correctness and profiling only. No sampling, extraction,
# recombination, artifact writes, or database writes. Run the production-shaped blocks remotely.

# %% BLOCK 1 — obtain Stage-3 data through the new model-owned Features adapter
using BayesianFootball
using DataFrames
using Random
using LinearAlgebra
using Statistics
using ForwardDiff
using ReverseDiff
using DynamicPPL
using LogDensityProblems
using BenchmarkTools
using LinearAlgebra: BLAS
const BFData = BayesianFootball.Data
const BFFeatures = BayesianFootball.Features
const BFPreGame = BayesianFootball.Models.PreGame
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l02_rebuild_features.jl")); using .RebuildFeatures
include(joinpath(@__DIR__, "l03_rebuild_equations.jl")); using .RebuildEquations
include(joinpath(@__DIR__, "l04_rebuild_turing_model.jl")); using .RebuildTuringModel

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
model = ScottishLowerNPNOGRecombinedPoissonModel(registry; half_life_days=365)
fs = BFFeatures.create_features(boundary, ds, model, :match_biweek)
fs_direct = build_rebuild_feature_set(boundary, ds, registry; half_life_days=365, own_goal_policy=:beneficiary)
@assert fs.data == fs_direct.data
@assert fs[:registry_fingerprint] == model.registry_fingerprint == registry_fingerprint(registry)
validate_feature_set(fs)
println("Stage 5 boundary $fold: $(length(fs[:weights])) rows; $(fs[:n_teams]) teams; adapter parity passed.")

# %% BLOCK 2 — deterministic VarInfo initialization and sampled-variable manifest
Random.seed!(0x5A17E5)
turing_model = BFPreGame.build_turing_model(model, fs)
vi = DynamicPPL.VarInfo(turing_model)
turing_model(vi)
θ = copy(vi[:])
J = fs[:n_teams]
# DynamicPPL 0.38 stores vector-valued sites as one metadata key (zA/zD/zM),
# while Chain labels expand these groups to zA[1], etc. Check both stable facts.
expected_groups = Set(RebuildEquations.PRIMITIVE_PARAMETER_FIELDS)
observed_groups = Set(Symbol.(propertynames(vi.metadata)))
@assert observed_groups == expected_groups
@assert length(θ) == primitive_parameter_length(J)
@assert length(primitive_turing_var_labels(J)) == length(θ)
println("Primitive sampled-site manifest passed: $(sort!(collect(observed_groups))). Derived transforms are not sampled.")

# %% BLOCK 3 — log density, compiled/uncompiled ReverseDiff, ForwardDiff, finite differences
lf = DynamicPPL.LogDensityFunction(turing_model)
f = x -> LogDensityProblems.logdensity(lf, x)
@assert isfinite(f(θ))
raw_tape = ReverseDiff.GradientTape(f, θ)
compile_seconds = @elapsed compiled_tape = ReverseDiff.compile(raw_tape)
g_compiled = similar(θ)
ReverseDiff.gradient!(g_compiled, compiled_tape, θ)
# A fresh ReverseDiff call re-records control flow; replaying raw_tape would not be an
# independent oracle for static-tape safety.
g_fresh = ReverseDiff.gradient(f, θ)
g_forward = ForwardDiff.gradient(f, θ)
relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)
@assert all(isfinite, g_fresh) && all(isfinite, g_compiled) && all(isfinite, g_forward)
@assert relerr(g_fresh, g_compiled) <= 1e-8
@assert relerr(g_compiled, g_forward) <= 1e-6
for k in unique([1, J, 2J + 1, 2J + 5, length(θ)])
    ε = 1e-6
    xp, xm = copy(θ), copy(θ); xp[k] += ε; xm[k] -= ε
    central = (f(xp) - f(xm)) / (2ε)
    @assert isfinite(central) && isapprox(g_compiled[k], central; rtol=1e-4, atol=1e-4)
end
println("Finite log density; compiled/uncompiled ≤1e-8; ReverseDiff/ForwardDiff ≤1e-6; finite differences ≤1e-4.")

# %% BLOCK 4 — static-tape safety at nearby points and across the log-rate saturation regime
probe_points = [θ .+ δ .* sin.(collect(eachindex(θ))) for δ in (0.001, -0.002, 0.003)]
θ_saturated = copy(θ)
θ_saturated[2J + 3] = 25.0 # mu_Y: exercise the smooth upper saturation regime
push!(probe_points, θ_saturated)
for (point, θp) in enumerate(probe_points)
    @assert isfinite(f(θp))
    gp_fresh = ReverseDiff.gradient(f, θp) # newly recorded branch/control-flow oracle
    gp_compiled = similar(θp)
    ReverseDiff.gradient!(gp_compiled, compiled_tape, θp)
    gp_forward = ForwardDiff.gradient(f, θp)
    @assert all(isfinite, gp_fresh) && all(isfinite, gp_compiled) && all(isfinite, gp_forward)
    @assert relerr(gp_fresh, gp_compiled) <= 1e-8
    @assert relerr(gp_compiled, gp_forward) <= 1e-6
    println("Probe $point passed (fresh/compiled ReverseDiff and ForwardDiff agreement).")
end

# %% BLOCK 5 — compiled-tape performance report (correctness is never hidden by timing)
g_bench = similar(θ)
trial = @benchmark ReverseDiff.gradient!($g_bench, $compiled_tape, $θ)
med_ms = median(trial.times) / 1e6
p95_ms = quantile(trial.times, 0.95) / 1e6
println("compile=$(round(compile_seconds; digits=3)) s, median=$(round(med_ms; digits=3)) ms, p95=$(round(p95_ms; digits=3)) ms, allocations=$(trial.allocs), bytes=$(trial.memory)")
if med_ms < 1
    println("Performance target met (<1 ms).")
elseif med_ms < 3
    println("Performance acceptable (<3 ms); inspect allocations before promotion.")
elseif med_ms > 5
    println("INVESTIGATE: compiled gradient exceeds 5 ms; correctness gates above still passed.")
else
    println("Performance above acceptable target; investigate before promotion.")
end
