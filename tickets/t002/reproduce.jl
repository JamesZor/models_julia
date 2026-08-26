# tickets/t002/reproduce.jl
#
# T002 — deterministic reproducer and baseline measurement matrix.
#
# Canonical brief: docs/tickets/T002-scalar-taped-likelihood.md
#
# Run this BEFORE touching production code, and again after. Every number it
# prints is an acceptance criterion or the evidence behind one.
#
#   julia --project -t 16
#   include("tickets/t002/reproduce.jl")
#
# Threading is not required for correctness here, but the profiler samples all
# threads, so `-t 16` reproduces the environment the baseline was taken in.
#
# ==============================================================================

using BayesianFootball
using DataFrames, Distributions, Random, Printf, Statistics
using DynamicPPL, LogDensityProblems, ReverseDiff

const ROOT = pkgdir(BayesianFootball)
const SL   = joinpath(ROOT, "current_development", "scottish_lower")
const MD   = BayesianFootball.MyDistributions

# The protocol loaders are reused deliberately: gate 3a in l04 is the parity
# check that any fix must keep at exactly 0.0, so the fix should be measured
# with the same instrument that will judge it.
include(joinpath(SL, "_protocol", "config.jl"))
include(joinpath(SL, "01_team_poisson", "l01_model.jl"))
include(joinpath(SL, "01_team_poisson", "l02_equations.jl"))
include(joinpath(SL, "01_team_poisson", "l03_gates.jl"))
include(joinpath(SL, "01_team_poisson", "l04_sampling_gates.jl"))


# ==============================================================================
# 0. Fixtures
# ==============================================================================

contract = SLContract()
ds       = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
engine   = tp_model()
folds    = tp_build_folds(ds, contract)

splits     = [(f.boundary, f.meta) for f in folds]
collection = BayesianFootball.Features.create_features(splits, ds, engine, sl_splitter(contract))
features   = [fs for (fs, _) in collection]

fs_small = features[1]     # 720 matches  — season opener
fs_large = features[end]   # 1060 matches — largest fold


# ==============================================================================
# 1. BASELINE — the headline numbers
# ==============================================================================
#
# Acceptance criterion: n_inst must become independent of row count, and the
# fold-20 median must drop below 1 ms.

prof_small = tp_grad_profile(engine, fs_small)
prof_large = tp_grad_profile(engine, fs_large)

print(tp_profile_table(prof_small; label = "T002 baseline — fold 1  (720 matches)"))
print(tp_profile_table(prof_large; label = "T002 baseline — fold 20 (1060 matches)"))

@printf("\nTAPE SCALING  %d rows -> %d instructions | %d rows -> %d instructions\n",
        prof_small.n_rows, prof_small.n_inst, prof_large.n_rows, prof_large.n_inst)
@printf("  ratio %.2fx for %.2fx the rows  (target: ratio ~= 1.00, i.e. does not scale)\n",
        prof_large.n_inst / prof_small.n_inst, prof_large.n_rows / prof_small.n_rows)


# ==============================================================================
# 2. CAUSE (a) — `view` on a TrackedArray defeats vectorisation
# ==============================================================================
#
# This contradicts Rule 4 of docs/turing_ad_performance_guide.md, which the
# engines follow. Fixing the engines without fixing the guide reintroduces it.

Random.seed!(1)
idx_demo  = rand(1:23, 720)
base_demo = randn(23)
ninst(f, x) = length(ReverseDiff.GradientTape(f, x).tape)

n_view = ninst(θ -> sum(exp.(view(θ, idx_demo))), base_demo)
n_gidx = ninst(θ -> sum(exp.(θ[idx_demo])),       base_demo)

@printf("\nVIEW vs GETINDEX (720 rows, 23 parameters, identical value)\n")
@printf("  view(θ, idx)  %6d instructions\n", n_view)
@printf("  θ[idx]        %6d instructions\n", n_gidx)
@printf("  -> getindex is the FAST path; the guide says the opposite\n")


# ==============================================================================
# 3. CAUSE (b) — RobustNegativeBinomial crashes on the vectorised path
# ==============================================================================
#
# With `view` the operands are Array{TrackedReal} and everything is scalar-taped,
# which works but is slow. With `getindex` they are a true TrackedArray, the
# broadcast takes ReverseDiff's `tracker_∇broadcast` fast path, and that
# evaluates the kernel under ForwardDiff duals for EVERY broadcast argument --
# including the integer goal count. `Int(::Dual)` then throws.
#
# So (a) is currently masking (b). Fixing `view` alone will surface this crash.

data_eq = tp_equation_data(fs_small)
yh      = data_eq.home_goals
hidx    = data_eq.home_idx .+ 1
θ_demo  = vcat(3.1, randn(23) .* 0.1)

f_view(θ) = (r = exp(θ[1]); sum(logpdf.(MD.RobustNegativeBinomial.(r, exp.(view(θ, hidx))), yh)))
f_gidx(θ) = (r = exp(θ[1]); sum(logpdf.(MD.RobustNegativeBinomial.(r, exp.(θ[hidx])),       yh)))

@printf("\nFULL LIKELIHOOD CHAIN (720 rows)\n")
@printf("  values agree                     %s\n", f_view(θ_demo) ≈ f_gidx(θ_demo))
@printf("  view      %6d instructions\n", ninst(f_view, θ_demo))
@printf("  getindex  %6d instructions\n", ninst(f_gidx, θ_demo))

view_grad = try; all(isfinite, ReverseDiff.gradient(f_view, θ_demo)); catch e; "ERROR: $(typeof(e))"; end
gidx_grad = try; all(isfinite, ReverseDiff.gradient(f_gidx, θ_demo)); catch e; "ERROR: $(typeof(e))"; end
@printf("  view      gradient  %s\n", view_grad)
@printf("  getindex  gradient  %s   <-- src/MyDistributions/negative_binomial.jl:79\n", gidx_grad)


# ==============================================================================
# 4. ATTRIBUTION — engine vs distribution
# ==============================================================================
#
# Shows how much of the per-observation cost is the linear predictor and how much
# is our distribution, by holding the (slow) scalar path fixed and varying only
# the likelihood.

lam(θ) = exp.(view(θ, hidx))
g_lam(θ)    = sum(lam(θ))
g_pois(θ)   = sum(logpdf.(Poisson.(lam(θ)), yh))
g_stdnb(θ)  = (r = exp(θ[1]); λ = lam(θ); sum(logpdf.(NegativeBinomial.(r, r ./ (r .+ λ)), yh)))
g_robust(θ) = (r = exp(θ[1]); sum(logpdf.(MD.RobustNegativeBinomial.(r, lam(θ)), yh)))

n_obs = length(yh)
@printf("\nPER-OBSERVATION TAPE COST (scalar path held fixed)\n")
for (label, g) in (("lambda only (view + exp)", g_lam),
                   ("+ Poisson",               g_pois),
                   ("+ stdlib NegativeBinomial", g_stdnb),
                   ("+ RobustNegativeBinomial",  g_robust))
    @printf("  %-28s %6.1f nodes/obs\n", label, ninst(g, θ_demo) / n_obs)
end


# ==============================================================================
# 5. THE INVARIANT — must stay exactly 0.0 after any fix
# ==============================================================================
#
# tp_gate_equation_parity compares DynamicPPL's log density against an
# INDEPENDENT implementation written from the model documentation, not from the
# engine. It currently passes at max |Δ| = 0.000e+00 across three prior draws.
#
# This is the scope guard made executable: the fix may change how the density is
# computed, never what it evaluates to.

gate3a = tp_gate_equation_parity(engine, fs_small)
@assert sl_gate_table("3a. Equation parity (T002 invariant)", gate3a)
