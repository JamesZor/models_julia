# ==============================================================================
# MODEL 01 — TEAM-LEVEL BASELINE: PROTOCOL WALKTHROUGH
# ==============================================================================
#
# A staged correctness walkthrough: contract through persisted smoke extraction.
# It does not evaluate betting performance; run r02_evaluate.jl and
# r03_growth_clv.jl only after r01_train.jl has promoted a full grid artifact.
# Blocks are intentionally sendable in order and leave inspectable TP_* globals.
#

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Distributions
using Statistics

const TP01_ROOT = "current_development/scottish_lower"

include(joinpath(TP01_ROOT, "_protocol/ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
include(joinpath(TP01_ROOT, "01_team_poisson/l01_model.jl"))
include(joinpath(TP01_ROOT, "01_team_poisson/l02_equations.jl"))
include(joinpath(TP01_ROOT, "01_team_poisson/l03_adapter.jl"))


# %%
# ==============================================================================
# 2. GATE 0 — Contract and fold inventory
# ==============================================================================

TP01_contract = sl_contract()
sl_describe(TP01_contract)
TP01_ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
TP01_folds = sl_build_folds(TP01_ds, TP01_contract)
sl_fold_table(TP01_ds, TP01_folds)
TP01_gate0 = sl_gate_contract(TP01_ds, TP01_folds, TP01_contract)
@assert sl_gate_table("0. Contract", TP01_gate0)


# %%
# ==============================================================================
# 3. GATE 1 — Model configuration
# ==============================================================================
# The explicit constructor is the Model-01 scientific choice.  The adapter is the
# sole protocol boundary; it owns production dispatch and l02 reference parity.

TP01_model = tp_model(half_life_days = 180.0)
TP01_adapter = TP01Adapter(TP01_model)
tp_describe(TP01_model)
TP01_gate1 = sl_gate_config(TP01_adapter, TP01_contract)
@assert sl_gate_table("1. Config", TP01_gate1)


# %%
# ==============================================================================
# 4. GATE 2 — Features and filtration
# ==============================================================================

TP01_gate2, TP01_features = sl_gate_features(TP01_ds, TP01_folds, TP01_adapter, TP01_contract)
@assert sl_gate_table("2. Features", TP01_gate2)
# Inspect: TP01_features[1].data[:team_map]


# %%
# ==============================================================================
# 5. GATE 3a — Equation parity
# ==============================================================================

TP01_gate3a = sl_gate_equation_parity(TP01_adapter, TP01_features[1])
@assert sl_gate_table("3a. Equation parity", TP01_gate3a)


# %%
# ==============================================================================
# 6. GATE 3b — Gradient health
# ==============================================================================

TP01_gate3b, TP01_grad = sl_gate_gradients(TP01_adapter, TP01_features[1])
@assert sl_gate_table("3b. Gradient health", TP01_gate3b)
# Optional diagnostic: print(ScottishLowerProtocol.sl_profile_table(
#     ScottishLowerProtocol.sl_grad_profile(TP01_adapter, TP01_features[1])))


# %%
# ==============================================================================
# 7. GATE 3c — Persisted smoke run (MCMC)
# ==============================================================================
# Start Julia with the required server thread count before this block.  This is
# intentionally the only sampling block in the walkthrough.

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1)
@assert Threads.nthreads() == TP01_contract.queue_tasks

TP01_smoke_results, TP01_smoke_path = sl_run_experiment(
    TP01_ds, TP01_adapter, TP01_contract; smoke = true)
TP01_gate3c = sl_gate_convergence(
    TP01_smoke_results, TP01_adapter, TP01_contract; expected_folds = 1)
@assert sl_gate_table("3c. Smoke convergence", TP01_gate3c)
println("saved smoke artifact: ", TP01_smoke_path)


# %%
# ==============================================================================
# 8. GATE 4 — Extraction and persisted-artifact loading
# ==============================================================================

TP01_gate4a = sl_gate_extraction_synthetic(TP01_adapter, TP01_features[1])
@assert sl_gate_table("4a. Extraction parity (synthetic chain)", TP01_gate4a)

TP01_gate4c = sl_gate_extraction_fallbacks(TP01_adapter, TP01_features[1])
@assert sl_gate_table("4c. Extraction fallbacks", TP01_gate4c)

TP01_smoke_loaded = sl_load_experiment(TP01_smoke_path)
TP01_gate4b, TP01_smoke_latents = sl_gate_extraction_real(
    TP01_ds, TP01_smoke_loaded, TP01_adapter, TP01_contract)
@assert sl_gate_table("4b. Extraction plumbing (persisted smoke)", TP01_gate4b)


# %%
# ==============================================================================
# 9. GATE 5 — Score matrix
# ==============================================================================

TP01_gate5a = sl_gate_score_dispatch(
    TP01_adapter, first(eachrow(TP01_smoke_latents.df)); max_goals = TP01_contract.max_goals)
@assert sl_gate_table("5a. Score matrix dispatch", TP01_gate5a)
TP01_gate5b = sl_gate_score_grid(TP01_adapter, TP01_smoke_latents.df, TP01_contract)
@assert sl_gate_table("5b. Score matrix grid", TP01_gate5b)
TP01_gate5c = sl_gate_market_identities(TP01_adapter, TP01_smoke_latents.df, TP01_contract)
@assert sl_gate_table("5c. Market identities", TP01_gate5c)

# %%
# ==============================================================================
# 10. Full grid, persisted reload, and convergence   ***SAMPLES ALL FOLDS***
# ==============================================================================
TP01_GRID_RESULTS, TP01_GRID_PATH = sl_run_experiment(TP01_ds, TP01_adapter, TP01_contract; smoke = false)
TP01_GATE10 = sl_gate_convergence(TP01_GRID_RESULTS, TP01_adapter, TP01_contract; expected_folds = length(TP01_folds))
@assert sl_gate_table("3d. Grid convergence", TP01_GATE10)
TP01_GRID_LOADED = sl_load_experiment(TP01_GRID_PATH)
TP01_GRID_LATENTS = BayesianFootball.Experiments.extract_oos_predictions(TP01_ds, TP01_GRID_LOADED; force = true)

# %%
# ==============================================================================
# 11. GATE 6 — OOS evaluation at bookmaker and Betfair close
# ==============================================================================
TP01_OOS_IDS = Set(Int.(TP01_GRID_LATENTS.df.match_id))
TP01_BOOK_B365 = sl_market_book(TP01_ds.odds, TP01_contract; ids = TP01_OOS_IDS)
TP01_BOOK_BF, TP01_BF_PARTIAL = sl_drop_incomplete(ScottishLowerProtocol.sl_betfair_book(TP01_ds, TP01_contract, TP01_BOOK_B365; ids = TP01_OOS_IDS))
@assert sl_gate_table("6a. Bet365 book integrity", sl_gate_book_integrity(TP01_BOOK_B365, TP01_contract))
@assert sl_gate_table("6a. Betfair book integrity", sl_gate_book_integrity(TP01_BOOK_BF, TP01_contract))
TP01_MODEL_BOOK, TP01_FIXTURES = sl_model_book(TP01_adapter, TP01_GRID_LATENTS, TP01_ds, TP01_contract)
TP01_JOINED = sl_join_books(TP01_MODEL_BOOK, Dict("bet365" => TP01_BOOK_B365, "betfair" => TP01_BOOK_BF))
TP01_GATE6B = sl_gate_alignment(TP01_JOINED, TP01_MODEL_BOOK)
TP01_GATE6C = sl_gate_shape(TP01_FIXTURES)
TP01_SCORES_B365 = sl_score_table(TP01_JOINED["bet365"])
TP01_GATE6D = sl_gate_not_broken(TP01_SCORES_B365)
@assert sl_gate_table("6b. Alignment", TP01_GATE6B)
@assert sl_gate_table("6c. Shape (RQR / LPD / draw rate)", TP01_GATE6C)
@assert sl_gate_table("6d. Not broken (vs Bet365 close)", TP01_GATE6D)
TP01_EVALUATION = ScottishLowerProtocol.sl_summary(TP01_JOINED)
TP01_SHAPE_SUMMARY = ScottishLowerProtocol.sl_summary_shape(TP01_FIXTURES)

# %%
# ==============================================================================
# 12. GATE 7 — Betfair-close Portfolio-Kelly growth and CLV workflow
# ==============================================================================
TP01_BF_ODDS = sl_betfair_odds_df(TP01_ds, TP01_contract; ids = TP01_OOS_IDS)
TP01_BOOK_SPEC = sl_book_spec(TP01_contract)
TP01_BOOKS_BF = BayesianFootball.Portfolio.build_books(TP01_BOOK_SPEC, TP01_GRID_LATENTS.df, TP01_GRID_LOADED, TP01_BF_ODDS, TP01_ds)
TP01_GATE7A = sl_gate_books(TP01_BOOKS_BF, TP01_GRID_LATENTS.df, TP01_BF_ODDS)
@assert sl_gate_table("7a. Book construction", TP01_GATE7A)
TP01_POLICY = sl_growth_policies(TP01_contract)[1].policy
TP01_SLATES = BayesianFootball.Portfolio.group(TP01_POLICY.grouping, TP01_BOOKS_BF)
TP01_TRAJECTORY = BayesianFootball.Portfolio.simulate(TP01_POLICY, TP01_SLATES)
TP01_GATE7B = sl_gate_simulation(TP01_TRAJECTORY, TP01_SLATES, TP01_contract)
@assert sl_gate_table("7b. Simulation integrity", TP01_GATE7B)
TP01_GROWTH = sl_growth_table(TP01_BOOKS_BF, TP01_contract)
TP01_GATE7C = sl_gate_growth(TP01_GROWTH)
@assert sl_gate_table("7c. Growth verdict", TP01_GATE7C)
TP01_POLICY_SWEEP = ScottishLowerProtocol.sl_sweep_policy(TP01_BOOKS_BF, TP01_CONTRACT)
