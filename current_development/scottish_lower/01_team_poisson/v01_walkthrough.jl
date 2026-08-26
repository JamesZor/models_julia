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
#   [x] 3. Sampling        [ ] 4. Extraction    [ ] 5. Score matrix
#   Blocks 4-5 are not yet written — see FINDINGS.md.
#
# BEFORE THE SMOKE RUN (block 7), the session must be started as
#   julia --project -t 16
# and set up with ThreadPinning + single-threaded BLAS. See
# docs/SERVER_AND_KAIMON.md § "Required setup before ANY sampling".
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
include(joinpath(TP_ROOT, "01_team_poisson/l04_sampling_gates.jl"))
include(joinpath(TP_ROOT, "01_team_poisson/l05_extraction_gates.jl"))
include(joinpath(TP_ROOT, "01_team_poisson/l06_score_matrix_gates.jl"))
include(joinpath(TP_ROOT, "01_team_poisson/l07_evaluation_gates.jl"))


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


# %%
# ==============================================================================
# 5. GATE 3a — Equation parity
# ==============================================================================
#
# The load-bearing gate. DynamicPPL scores the Turing model; l02_equations.jl
# scores an independent implementation written from MODEL.md. If these agree, the
# model being fitted is the model that is documented.
#
# Several independent prior draws, not one — a single point can agree by accident,
# for instance if a scale that should multiply a parameter happens to sit near 1.
#
# Verified 2026-08-25: max |Δ| = 0.0. Exact, not merely within tolerance.

tp_gate3a = tp_gate_equation_parity(tp_engine, tp_features[1])
@assert sl_gate_table("3a. Equation parity", tp_gate3a)


# %%
# ==============================================================================
# 6. GATE 3b — Gradient health
# ==============================================================================
#
# Four independent routes to the same gradient: fresh ReverseDiff, the compiled
# tape NUTS will actually replay, ForwardDiff, and finite differences.
#
# The compiled tape is a STATIC recording. A data-dependent branch inside the
# model would leave the tape frozen on whichever branch was taken at record time —
# still returning plausible numbers at other parameter values. That is why
# agreement is also checked at perturbed points, not only where it was recorded.
#
# Latency is reported, never gating: a slow model is a cost, a wrong one is a bug.

tp_gate3b, tp_grad = tp_gate_gradients(tp_engine, tp_features[1])
@assert sl_gate_table("3b. Gradient health", tp_gate3b)

# tp_grad.median_ms is fold 1 (720 rows). Later folds carry ~1060.


# %%
# ------------------------------------------------------------------------------
# 6b. Gradient profile   (diagnostic — not a gate, nothing here can fail)
# ------------------------------------------------------------------------------
#
# Run this when the 3b latency looks disappointing. It answers WHERE the time
# goes, which the gate deliberately does not.
#
# The number that matters is `tape instructions`. A correctly vectorised model
# has a tape whose size does not depend on row count; ours has 24.6 nodes per
# observation, and the self-time ranking is all tape bookkeeping rather than
# maths. Diagnosed 2026-08-26, raised as docs/tickets/T002.
#
# Takes ~30s (it compiles a second tape and profiles 1500 gradients).

tp_prof1 = tp_grad_profile(tp_engine, tp_features[1])
print(tp_profile_table(tp_prof1; label = "01_team_poisson — fold 1"))

# Compare against the largest fold to confirm the tape scales with rows (it should
# not, and currently does):
#
#   tp_prof20 = tp_grad_profile(tp_engine, tp_features[20])
#   (tp_prof1.n_inst, tp_prof20.n_inst)


# %%
# ==============================================================================
# 7. GATE 3c — Smoke run   ***THIS ONE SAMPLES***
# ==============================================================================
#
# One fold, four chains, persisted through src/experiments. The saved artifact is
# the input to gate 4 — do not delete it.
#
# REQUIRES the session to have been started with `julia --project -t 16` and set
# up as follows. Run this cell BEFORE tp_run_smoke; pinning after threads are
# already working does not move existing tasks.

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1)
@assert Threads.nthreads() == 16   # physical cores, not the 32 hyperthreads


# %%
# The fold here is the season-opening one (end_dynamics = 0): fitted on prior
# seasons only, predicting the first observed block. It is the smallest and least
# representative fold in the set, chosen because it is the only way to get exactly
# one fold through the real run_experiment path.
#
# So: passing this is NECESSARY, not sufficient.

tp_smoke_results, tp_smoke_path = tp_run_smoke(tp_ds, tp_engine, tp_contract)

tp_gate3c = tp_gate_convergence(tp_smoke_results, tp_contract; expected_folds = 1)
@assert sl_gate_table("3c. Smoke convergence", tp_gate3c)

println("saved to: ", tp_smoke_path)

# Poke at the chain:
#   tp_chain = first(tp_smoke_results.training_results)[1]
#   DataFrame(MCMCChains.summarystats(tp_chain))


# %%
# ==============================================================================
# 8. GATE 4 — Extraction
# ==============================================================================
#
# Gate 3a proved the FITTED model is the documented model. This asks the separate
# question: is the PRICED model the same one that was fitted?
#
# Different code path entirely. Training runs inside `@model` on TrackedArrays;
# extraction runs afterwards on an MCMCChains object, indexing by variable NAME
# and reassembling the arithmetic by hand. Nothing forces them to agree — and the
# 2026-08-24 audit found a prototype where they did not.
#
# 4a is exact and cheap: fabricate a chain with known parameters, price it, and
# compare against l02 — the same independent reference gate 3a used.

tp_gate4a = tp_gate_extraction_synthetic(tp_engine, tp_features[1])
@assert sl_gate_table("4a. Extraction parity (synthetic chain)", tp_gate4a)


# %%
# ------------------------------------------------------------------------------
# 4c. Fallbacks — what extraction does with what it has never seen
# ------------------------------------------------------------------------------
#
# Gate 2 found 4 of 720 OOS sides whose team is absent from the fitted fold
# (arbroath, inverness-caledonian-thistle). Extraction substitutes zeros for them.
# This measures whether those zeros are the right POPULATION values rather than
# assuming they are, and checks that the season/month fallbacks are inert for this
# component set.

tp_gate4c = tp_gate_extraction_fallbacks(tp_engine, tp_features[1])
@assert sl_gate_table("4c. Extraction fallbacks", tp_gate4c)


# %%
# ------------------------------------------------------------------------------
# 4b. Real chain, real loader
# ------------------------------------------------------------------------------
#
# Loads the gate 3c artifact back off DISK — not the in-memory object the smoke
# returned. The saved artifact is what gates 5-7 consume, so it is the thing that
# has to be readable.
#
# This cannot check arithmetic (the true λ is unknown). It checks plumbing: that
# the splitter reproduces the boundaries the experiment trained on, that the t+1
# fixtures are found and every one is priced, and that nothing is NaN or absurd.

tp_smoke_loaded = tp_load_smoke(tp_smoke_path)
tp_gate4b, tp_latents = tp_gate_extraction_real(tp_ds, tp_smoke_loaded, tp_contract)
@assert sl_gate_table("4b. Extraction plumbing (real chain)", tp_gate4b)

# tp_latents.df is the input to gate 5 — one row per OOS fixture, each carrying the
# full posterior for λ_h, λ_a, r_h, r_a.


# %%
# ==============================================================================
# 9. GATE 5 — Score matrix
# ==============================================================================
#
# The last translation before money: λ and r become a joint distribution over
# scorelines, and that grid becomes market probabilities.
#
# 5a is dispatch. Engines are routed to a pricer by abstract supertype and by
# Union membership, and an engine missing from the relevant Union does not error
# at definition time — it falls through to a default meant for another
# likelihood. So the gate asserts the resolved method BY FILE.

tp_gate5a = tp_gate_score_dispatch(tp_engine, first(eachrow(tp_latents.df));
                                   max_goals = tp_contract.max_goals)
@assert sl_gate_table("5a. Score matrix dispatch", tp_gate5a)


# %%
# ------------------------------------------------------------------------------
# 5b. The grid
# ------------------------------------------------------------------------------
#
# Parity against two stock NegativeBinomials built outside src, plus orientation
# and truncation mass.
#
# Orientation is the one that matters most. A transposed grid produces perfectly
# well-formed probabilities that are simply the wrong way round, and nothing
# downstream can detect it — it looks like a badly calibrated model, not a broken
# one. It is caught here only because γ > 0 makes the home and away marginals
# genuinely different.

tp_gate5b = tp_gate_score_grid(tp_engine, tp_latents.df, tp_contract)
@assert sl_gate_table("5b. Score matrix grid", tp_gate5b)


# %%
# ------------------------------------------------------------------------------
# 5c. Market identities
# ------------------------------------------------------------------------------
#
# Every market is a partition of the same grid, so each family must sum to the
# same total — and that total is `1 - truncation_mass`, NOT 1. Nothing in src
# normalises the NegBin grid, so asserting "1X2 sums to 1" would assert something
# false. Internal consistency is both true and the stronger test.

tp_gate5c = tp_gate_market_identities(tp_engine, tp_latents.df, tp_contract)
@assert sl_gate_table("5c. Market identities", tp_gate5c)


# %%
# ------------------------------------------------------------------------------
# First look at actual prices  (not a gate)
# ------------------------------------------------------------------------------
#
# The first point in the protocol where the model says something you can check
# against intuition. Worth reading before trusting gate 6.

tp_market_summary(tp_engine, tp_latents.df, tp_contract; n_rows = 8)


# %%
# ==============================================================================
# 10. THE FULL GRID   ***THIS SAMPLES ALL 20 FOLDS***
# ==============================================================================
#
# Everything up to here ran on ONE fold. Gates 6 and 7 need the whole development
# season, at the contract's grid budget rather than the smoke's cheaper one.
#
#   20 folds x 4 chains = 80 queued tasks, 800 warmup + 800 samples
#   concurrency 16 (physical cores — the queue flattens folds x chains)
#   → data/scottish_lower/01_team_poisson/<hash>/tp01_grid_<hash>_<timestamp>
#
# Same splitter gate 0 built the fold inventory from and gate 2 checked for
# leakage, so this trains on boundaries that have already been audited.
#
# REQUIRES the threading cell (block 6) to have been run in this session.
# Budget roughly 15-25 minutes: the smoke did 4 chains on 720 rows in 53s, and
# this is 1.6x the iterations on folds up to 1060 rows.
#
# Check the setup before committing to the run:
#
#   cfg = tp_grid_config(tp_engine, tp_contract;
#                        save_dir = sl_artifact_dir(tp_contract, "01_team_poisson",
#                                                   sl_hash(tp_engine)))
#   (cfg.name, cfg.save_dir, length(Data.create_id_boundaries(tp_ds, cfg.splitter)))

tp_grid_results, tp_grid_path = tp_run_grid(tp_ds, tp_engine, tp_contract)


# %%
# ------------------------------------------------------------------------------
# 10b. Grid convergence — the same diagnostics as 3c, across every fold
# ------------------------------------------------------------------------------
#
# 3c cleared ONE fold. A fold that fails here is a fold whose prices gate 6 must
# not score, so this runs before any evaluation.

tp_gate10 = tp_gate_convergence(tp_grid_results, tp_contract; expected_folds = length(tp_folds))
@assert sl_gate_table("6.0 Grid convergence (all folds)", tp_gate10)


# %%
# ------------------------------------------------------------------------------
# 10c. Extract OOS predictions for the whole season
# ------------------------------------------------------------------------------
#
# Same path gate 4b exercised, now across all 20 folds. Expect ~360 fixtures.

tp_grid_latents = Experiments.extract_oos_predictions(tp_ds, tp_grid_results; force = true)
nrow(tp_grid_latents.df)


# %%
# ==============================================================================
# 11. GATE 6 — Evaluation
# ==============================================================================
#
# THE PASS CRITERION IS NOT "BEATS THE MARKET". Prior work on this book had the
# model losing narrowly on 1X2 log loss while producing positive CLV and growth; a
# gate demanding it win would have discarded a working model. Gate 6 passes when
# the model is not BROKEN. Beating the market is recorded as a finding.
#
# 6a — book integrity, on BOTH baselines, before a single score is computed.
#
# Two traps this catches, both found here on 2026-08-26:
#   * a market missing selections de-vigs to p = 1.0, because the overround is
#     computed over what is PRESENT. Clamped, one losing p=1 costs ~20.7 log loss
#     while every other diagnostic reads healthy. 143 of 930 Betfair markets.
#   * is_winner contradicting the score (T004).

tp_oos_ids = Set(Int.(tp_grid_latents.df.match_id))

tp_mb_b365 = tp_market_book(tp_ds.odds, tp_contract; ids = tp_oos_ids)
tp_mb_bf, tp_n_partial = tp_drop_incomplete(
    tp_betfair_book(tp_ds, tp_contract, tp_mb_b365; ids = tp_oos_ids))

@assert sl_gate_table("6a. Book integrity (Bet365 close)", tp_gate_book_integrity(tp_mb_b365, tp_contract))
@assert sl_gate_table("6a. Book integrity (Betfair close)", tp_gate_book_integrity(tp_mb_bf, tp_contract))
tp_n_partial   # partial Betfair markets dropped — expect ~143


# %%
# ------------------------------------------------------------------------------
# 11b. The model book   (~20s: 360 fixtures x 2000 draws of score matrices)
# ------------------------------------------------------------------------------
#
# Streams one fixture at a time; the full posterior grid would be ~414 MB.
# Returns the market prices AND the per-fixture quantities the market-free metrics
# need (LPD, randomised quantile residuals).

tp_model_bk, tp_fx = tp_model_book(tp_engine, tp_grid_latents, tp_ds, tp_contract)


# %%
# ------------------------------------------------------------------------------
# 11c. Alignment — asserted BEFORE any ranking is printed
# ------------------------------------------------------------------------------
#
# Each baseline is scored on its OWN aligned subset. Joining every baseline into
# one table restricts to the intersection, and adding thin Betfair to thick Bet365
# once cut the evaluation from 4,658 rows to 96 while reporting PASS.

tp_books = Dict("bet365" => tp_mb_b365, "betfair" => tp_mb_bf)
tp_j     = tp_join_books(tp_model_bk, tp_books)
@assert sl_gate_table("6b. Alignment", tp_gate_alignment(tp_j, tp_model_bk))


# %%
# ------------------------------------------------------------------------------
# 11d. Shape — RQR and LPD, both market-free
# ------------------------------------------------------------------------------
#
# RQR is exactly N(0,1) under correct specification, so both moments mean
# something: mean ≠ 0 is bias, sd < 1 is over-confidence.
#
# Note RQR sd and the sd_model/sd_market column in 11f measure DIFFERENT things.
# RQR is the spread WITHIN a fixture's goal distribution; sd_model is the spread
# of probabilities BETWEEN fixtures. This model has the first right (0.9855) and
# the second at 0.55 of the market's — correct uncertainty about each match, too
# little conviction about which match is which.

@assert sl_gate_table("6c. Shape (RQR / LPD)", tp_gate_shape(tp_fx))
@assert sl_gate_table("6d. Draw deficit", tp_gate_draw_deficit(tp_fx))


# %%
# ------------------------------------------------------------------------------
# 11e. Proper scores, per line
# ------------------------------------------------------------------------------
#
# Per line, never aggregated across a market's selections. Paired differences, not
# two independent means — the two forecasters score the SAME fixtures, and treating
# them as independent inflates the interval until it says nothing.

tp_scores_b365 = tp_score_table(tp_j["bet365"])
tp_scores_bf   = tp_score_table(tp_j["betfair"])
tp_scores_b365


# %%
# ------------------------------------------------------------------------------
# 11e-summary. The whole book on one screen
# ------------------------------------------------------------------------------
#
# One row per market per baseline, scored with MULTICLASS log loss — minus the log
# of the probability given to whatever actually happened. Not per selection: for a
# two-outcome market yes/no give identical log loss, so listing both double-counts.
#
# Δ is model minus market, so NEGATIVE MEANS THE MODEL IS BETTER.
#
# Read the `n` column on every row. The BOOK row is only meaningful over fixtures
# priced in all six markets — on Betfair that is 2 of 320, because the exchange book
# is thin, so its BOOK row is noise while its per-market rows are fine.

tp_summary(tp_j)


# %%
# The market-free metrics. No baseline column because no baseline produces them: a
# bookmaker quotes market probabilities, not a distribution over scorelines. These
# exist to rank model VARIANTS against each other.

tp_summary_shape(tp_fx)


# %%
# ------------------------------------------------------------------------------
# 11f. glm_edge — does the model know anything the market does not?
# ------------------------------------------------------------------------------
#
# y ~ logit(p_market) + logit(p_model), per line. A model can be worse than the
# market in absolute log loss and still carry incremental information; that
# combination is what a profitable contrarian model looks like, and proper scoring
# alone cannot see it.
#
# Read `slope` together with `se_slope`: with sd_model as low as 0.008 the
# regression has almost no leverage, so the point estimate is noise. The gate tests
# significance, not a band.

tp_edges_b365 = tp_edge_table(tp_j["bet365"])
@assert sl_gate_table("6e. Not broken (vs Bet365 close)", tp_gate_not_broken(tp_scores_b365, tp_edges_b365))
tp_edges_b365


# %%
# ------------------------------------------------------------------------------
# 11g. Fold weighting — measured, not asserted
# ------------------------------------------------------------------------------
#
# OOS blocks range 2-24 fixtures, so a fold average lets a small block outvote a
# large one. Measured on this grid the difference is -0.0007, an order of magnitude
# below the Δll effects in play — so it does not change a conclusion HERE. Pooling
# stays the default because that is not guaranteed on a more uneven grid.

tp_fold_weighting_check(tp_j["bet365"], tp_folds)
