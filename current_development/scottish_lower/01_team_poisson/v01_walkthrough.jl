# ==============================================================================
# MODEL 01 — TEAM-LEVEL BASELINE : THE WALKTHROUGH
# ==============================================================================
#
# WHAT THIS IS
#   A correctness walkthrough of one model, one stage at a time. It answers:
#   is the model that gets FITTED the same model that gets DOCUMENTED and the
#   same model that gets PRICED?
#
# WHAT THIS IS NOT
#   Not a study, not a comparison, not a betting result. It produces no ranking
#   and no edge. Gates 6-7 do that, in r02/r03.
#
# HOW TO RUN
#   Send one numbered block at a time from nvim (kitty-runner). Blocks are
#   independently sendable in order; each leaves its output in a named global so
#   you can poke at it afterwards.
#
#   Blocks 0-2 are seconds. Block 3 contains the only MCMC in this file.
#
# GATE COVERAGE IN THIS FILE
#   [x] 0. Contract        [x] 1. Config        [x] 2. Features
#   [ ] 3. Sampling        [ ] 4. Extraction    [ ] 5. Score matrix
#   Blocks 3-5 are not yet written — see FINDINGS.md.
#
# ==============================================================================


# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Distributions
using Statistics

const TP_ROOT = "current_development/scottish_lower"

include(joinpath(TP_ROOT, "_protocol/config.jl"))
include(joinpath(TP_ROOT, "01_team_poisson/l01_model.jl"))
include(joinpath(TP_ROOT, "01_team_poisson/l02_equations.jl"))
include(joinpath(TP_ROOT, "01_team_poisson/l03_gates.jl"))


# %%
# ==============================================================================
# 2. GATE 0 — Contract
# ==============================================================================
#
# What is being modelled, on what data, over which folds. Read the fold table;
# it is the experiment's terms of reference.
#
# The column that matters is the last two: the last kickoff we FIT on must come
# strictly before the first kickoff we PREDICT. `target_match_ids` are fitted
# observations through step t, not a test set.

tp_contract = sl_contract()
sl_describe(tp_contract)

tp_ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

tp_folds = tp_build_folds(tp_ds, tp_contract)
tp_fold_table(tp_ds, tp_folds)

tp_gate0 = tp_gate_contract(tp_ds, tp_folds, tp_contract)
@assert sl_gate_table("0. Contract", tp_gate0)


# %%
# ==============================================================================
# 3. GATE 1 — Config
# ==============================================================================
#
# The component menu is printed so alternatives are visible. To try one, edit the
# `tp_engine = tp_model(...)` call below and re-run blocks 3 onward — nothing
# else needs changing.
#
# Note that l02_equations.jl implements the DEFAULT components only, and will
# refuse to compute for anything else. That refusal is the point: a component
# swap must not silently lose its parity reference.

tp_menu()

tp_engine = tp_model(
    half_life_days = 180.0,          # PROVISIONAL — src default, not Scottish evidence
)

tp_describe(tp_engine)

tp_gate1 = tp_gate_config(tp_engine, tp_contract)
@assert sl_gate_table("1. Config", tp_gate1)


# %%
# ==============================================================================
# 4. GATE 2 — Features
# ==============================================================================
#
# The anti-leakage gate. In order:
#
#   kickoff filtration   postponements can push a match past the prediction
#                        cutoff even when the nominal biweek looks clean
#   perturbation         adding future matches must not change an earlier fold's
#                        FeatureSet. Expected to pass trivially here (every
#                        extractor is a pure ID lookup); it is established on
#                        model 01 so that it is trustworthy on model 02, where
#                        features are FITTED and this test does real work
#   type purity          no missing, no NaN reaching an @model
#   team_map identity    String-keyed, matching what extract_parameters looks up.
#                        An Int-keyed map here is the 2026-08-24 defect
#   OOS coverage         unmapped sides are REPORTED, never absorbed

tp_gate2, tp_features = tp_gate_features(tp_ds, tp_folds, tp_engine, tp_contract)
@assert sl_gate_table("2. Features", tp_gate2)

# Poke at one fold:
#   tp_features[1].data[:n_teams]
#   sort(collect(tp_features[1].data[:team_map]), by = last)
