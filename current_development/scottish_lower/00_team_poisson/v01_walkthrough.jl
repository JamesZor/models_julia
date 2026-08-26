# ==============================================================================
# MODEL 00 — PURE POISSON BASELINE : PROTOCOL WALKTHROUGH
# ==============================================================================
# Correctness walkthrough for the fitted, extracted, priced, evaluated, and
# Betfair-close-staked Model 00 posterior. Gates 0--7 are deliberately ordered:
# contract → config → features → sampling → extraction → pricing → evaluation → growth.
# Blocks 7 and 10 sample; do not run them locally.

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using DataFrames
using Distributions
using Statistics

const TP00_ROOT = "current_development/scottish_lower"

include(joinpath(TP00_ROOT, "_protocol/ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
include(joinpath(TP00_ROOT, "00_team_poisson/l01_model.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l02_equations.jl"))
include(joinpath(TP00_ROOT, "00_team_poisson/l03_adapter.jl"))

# %%
# ==============================================================================
# 2. GATE 0 — Contract and kickoff-filtered folds
# ==============================================================================
TP00_CONTRACT = sl_contract()
sl_describe(TP00_CONTRACT)
TP00_DS = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
TP00_FOLDS = sl_build_folds(TP00_DS, TP00_CONTRACT)
sl_fold_table(TP00_DS, TP00_FOLDS)
TP00_GATE0 = sl_gate_contract(TP00_DS, TP00_FOLDS, TP00_CONTRACT)
@assert sl_gate_table("0. Contract", TP00_GATE0)

# %%
# ==============================================================================
# 3. GATE 1 — Model configuration and protocol adapter
# ==============================================================================
TP00_ADAPTER = TP00Adapter(half_life_days = 180.0)
TP00_MODEL = sl_model(TP00_ADAPTER)
TP00_MODEL_HASH = sl_artifact_hash(TP00_ADAPTER, TP00_CONTRACT)
println("MODEL 00 Pure Poisson [", TP00_MODEL_HASH, "]")
TP00_GATE1 = sl_gate_config(TP00_ADAPTER, TP00_CONTRACT)
@assert sl_gate_table("1. Config", TP00_GATE1)

# %%
# ==============================================================================
# 4. GATE 2 — Features and anti-leakage
# ==============================================================================
TP00_GATE2, TP00_FEATURES = sl_gate_features(TP00_DS, TP00_FOLDS, TP00_ADAPTER, TP00_CONTRACT)
@assert sl_gate_table("2. Features", TP00_GATE2)

# %%
# ==============================================================================
# 5. GATE 3a — Independent equation parity
# ==============================================================================
TP00_GATE3A = sl_gate_equation_parity(TP00_ADAPTER, TP00_FEATURES[1])
@assert sl_gate_table("3a. Equation parity", TP00_GATE3A)

# %%
# ==============================================================================
# 6. GATE 3b — AD and gradient health
# ==============================================================================
TP00_GATE3B, TP00_GRAD = sl_gate_gradients(TP00_ADAPTER, TP00_FEATURES[1])
@assert sl_gate_table("3b. Gradient health", TP00_GATE3B)

# %%
# ==============================================================================
# 7. GATE 3c — Persisted smoke experiment   ***SAMPLES***
# ==============================================================================
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1)
TP00_SMOKE_RESULTS, TP00_SMOKE_PATH = sl_run_experiment(TP00_DS, TP00_ADAPTER, TP00_CONTRACT; smoke = true)
TP00_GATE3C = sl_gate_convergence(TP00_SMOKE_RESULTS, TP00_ADAPTER, TP00_CONTRACT; expected_folds = 1)
@assert sl_gate_table("3c. Smoke convergence", TP00_GATE3C)
println("Saved smoke artifact: ", TP00_SMOKE_PATH)

# %%
# ==============================================================================
# 8. GATE 4 — Extraction from synthetic and persisted chains
# ==============================================================================
TP00_GATE4A = sl_gate_extraction_synthetic(TP00_ADAPTER, TP00_FEATURES[1])
@assert sl_gate_table("4a. Extraction parity (synthetic chain)", TP00_GATE4A)
TP00_GATE4C = sl_gate_extraction_fallbacks(TP00_ADAPTER, TP00_FEATURES[1])
@assert sl_gate_table("4c. Extraction fallbacks", TP00_GATE4C)
TP00_SMOKE_LOADED = sl_load_experiment(TP00_SMOKE_PATH)
TP00_GATE4B, TP00_LATENTS = sl_gate_extraction_real(TP00_DS, TP00_SMOKE_LOADED, TP00_ADAPTER, TP00_CONTRACT)
@assert sl_gate_table("4b. Extraction plumbing (persisted chain)", TP00_GATE4B)

# %%
# ==============================================================================
# 9. GATE 5 — Posterior score matrices and market identities
# ==============================================================================
TP00_GATE5A = sl_gate_score_dispatch(TP00_ADAPTER, first(eachrow(TP00_LATENTS.df)); max_goals = TP00_CONTRACT.max_goals)
@assert sl_gate_table("5a. Score-matrix dispatch", TP00_GATE5A)
TP00_GATE5B = sl_gate_score_grid(TP00_ADAPTER, TP00_LATENTS.df, TP00_CONTRACT)
@assert sl_gate_table("5b. Score-matrix grid", TP00_GATE5B)
TP00_GATE5C = sl_gate_market_identities(TP00_ADAPTER, TP00_LATENTS.df, TP00_CONTRACT)
@assert sl_gate_table("5c. Market identities", TP00_GATE5C)

# %%
# ==============================================================================
# 10. Full grid, persisted reload, and convergence   ***SAMPLES ALL FOLDS***
# ==============================================================================
TP00_GRID_RESULTS, TP00_GRID_PATH = sl_run_experiment(TP00_DS, TP00_ADAPTER, TP00_CONTRACT; smoke = false)
TP00_GATE10 = sl_gate_convergence(TP00_GRID_RESULTS, TP00_ADAPTER, TP00_CONTRACT; expected_folds = length(TP00_FOLDS))
@assert sl_gate_table("3d. Grid convergence", TP00_GATE10)
TP00_GRID_LOADED = sl_load_experiment(TP00_GRID_PATH)
TP00_GRID_LATENTS = BayesianFootball.Experiments.extract_oos_predictions(TP00_DS, TP00_GRID_LOADED; force = true)

# %%
# ==============================================================================
# 11. GATE 6 — OOS evaluation at bookmaker and Betfair close
# ==============================================================================
TP00_OOS_IDS = Set(Int.(TP00_GRID_LATENTS.df.match_id))
TP00_BOOK_B365 = sl_market_book(TP00_DS.odds, TP00_CONTRACT; ids = TP00_OOS_IDS)
TP00_BOOK_BF, TP00_BF_PARTIAL = sl_drop_incomplete(ScottishLowerProtocol.sl_betfair_book(TP00_DS, TP00_CONTRACT, TP00_BOOK_B365; ids = TP00_OOS_IDS))
@assert sl_gate_table("6a. Bet365 book integrity", sl_gate_book_integrity(TP00_BOOK_B365, TP00_CONTRACT))
@assert sl_gate_table("6a. Betfair book integrity", sl_gate_book_integrity(TP00_BOOK_BF, TP00_CONTRACT))
TP00_MODEL_BOOK, TP00_FIXTURES = sl_model_book(TP00_ADAPTER, TP00_GRID_LATENTS, TP00_DS, TP00_CONTRACT)
TP00_JOINED = sl_join_books(TP00_MODEL_BOOK, Dict("bet365" => TP00_BOOK_B365, "betfair" => TP00_BOOK_BF))
TP00_GATE6B = sl_gate_alignment(TP00_JOINED, TP00_MODEL_BOOK)
TP00_GATE6C = sl_gate_shape(TP00_FIXTURES)
TP00_SCORES_B365 = sl_score_table(TP00_JOINED["bet365"])
TP00_GATE6D = sl_gate_not_broken(TP00_SCORES_B365)
@assert sl_gate_table("6b. Alignment", TP00_GATE6B)
@assert sl_gate_table("6c. Shape (RQR / LPD / draw rate)", TP00_GATE6C)
@assert sl_gate_table("6d. Not broken (vs Bet365 close)", TP00_GATE6D)
TP00_EVALUATION = ScottishLowerProtocol.sl_summary(TP00_JOINED)
TP00_SHAPE_SUMMARY = ScottishLowerProtocol.sl_summary_shape(TP00_FIXTURES)

# %%
# ==============================================================================
# 12. GATE 7 — Betfair-close Portfolio-Kelly growth and CLV workflow
# ==============================================================================
TP00_BF_ODDS = sl_betfair_odds_df(TP00_DS, TP00_CONTRACT; ids = TP00_OOS_IDS)
TP00_BOOK_SPEC = sl_book_spec(TP00_CONTRACT)
TP00_BOOKS_BF = BayesianFootball.Portfolio.build_books(TP00_BOOK_SPEC, TP00_GRID_LATENTS.df, TP00_GRID_LOADED, TP00_BF_ODDS, TP00_DS)
TP00_GATE7A = sl_gate_books(TP00_BOOKS_BF, TP00_GRID_LATENTS.df, TP00_BF_ODDS)
@assert sl_gate_table("7a. Book construction", TP00_GATE7A)
TP00_POLICY = sl_growth_policies(TP00_CONTRACT)[1].policy
TP00_SLATES = BayesianFootball.Portfolio.group(TP00_POLICY.grouping, TP00_BOOKS_BF)
TP00_TRAJECTORY = BayesianFootball.Portfolio.simulate(TP00_POLICY, TP00_SLATES)
TP00_GATE7B = sl_gate_simulation(TP00_TRAJECTORY, TP00_SLATES, TP00_CONTRACT)
@assert sl_gate_table("7b. Simulation integrity", TP00_GATE7B)
TP00_GROWTH = sl_growth_table(TP00_BOOKS_BF, TP00_CONTRACT)
TP00_GATE7C = sl_gate_growth(TP00_GROWTH)
@assert sl_gate_table("7c. Growth verdict", TP00_GATE7C)
TP00_POLICY_SWEEP = ScottishLowerProtocol.sl_sweep_policy(TP00_BOOKS_BF, TP00_CONTRACT)
