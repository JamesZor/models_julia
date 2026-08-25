# Stage 2 notebook runner: history-only incident/component audit.  No model or MCMC.

# %% BLOCK 1 — imports and loader
using BayesianFootball
using DataFrames
const BFData = BayesianFootball.Data
include(joinpath(@__DIR__, "l01_rebuild_data_contract.jl"))
using .RebuildDataContract

# %% BLOCK 2 — cached Scottish-Lower snapshot and a pooled (56/57) temporal boundary
# This is read-only: the runner deliberately has no save/export calls.
ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours = 10_000)
println("Loaded $(nrow(ds.matches)) finished matches and $(nrow(ds.incidents)) incident rows.")

splitter = BFData.GroupedCVConfig(
    tournament_groups = [[56, 57]], target_seasons = ["24/25", "25/26"],
    history_seasons = 2, dynamics_col = :match_biweek, warmup_period = 0, stop_early = true,
)
boundaries = BFData.create_id_boundaries(ds, splitter)

# Choose a genuine history set containing both source leagues, not simply `boundaries[end]`.
match_tournament = Dict(Int(r.match_id) => Int(r.tournament_id) for r in eachrow(ds.matches))
fold_index = findlast(eachindex(boundaries)) do i
    b = first(boundaries[i])
    Set(get(match_tournament, Int(id), -1) for id in b.history_match_ids) >= Set([56, 57])
end
isnothing(fold_index) && error("No temporal boundary has both leagues in history.")
boundary = first(boundaries[fold_index])
history_ids = Int.(boundary.history_match_ids)       # authoritative training/history IDs
# Passed only to prove the loader rejects history/target overlap; it is never processed.
target_ids = Int.(boundary.target_match_ids)
println("Boundary $fold_index: $(length(history_ids)) history IDs, $(length(target_ids)) target IDs.")

# %% BLOCK 3 — build ledger and per-league/season quarantine counts
report = audit_component_history(ds, history_ids; target_match_ids = target_ids)
println("\nHistory-only summary (both own-goal hypotheses are retained):")
display(report.summary)
println("\nInformative own-goal hypothesis evidence (own-goal matches with exactly one valid convention; no global policy selected):")
display(report.own_goal_hypothesis_evidence)
println("\nQuarantine counts by tournament / season / reason:")
display(report.quarantine_summary)
println("\nDiagnostics:")
display(report.diagnostics)

# %% BLOCK 4 — compare hypotheses without selecting one
comparison = select(report.ledger,
    :match_id, :official_G_h, :official_G_a,
    :own_goal_beneficiary_h, :own_goal_beneficiary_a,
    :own_goal_committing_h, :own_goal_committing_a,
    :residual_beneficiary_h, :residual_beneficiary_a,
    :residual_committing_h, :residual_committing_a,
    :beneficiary_nonnegative_ok, :committing_nonnegative_ok,
    :beneficiary_valid, :committing_valid, :quarantine_reasons)
comparison[!, :uniquely_selected_policy] = [select_own_goal_hypothesis(r) for r in eachrow(comparison)]
println("\nHypothesis comparison; `nothing` is intentional unless validated reconciliation uniquely supports a side convention:")
display(comparison)

# %% BLOCK 5 — review queue with provider event IDs (still no writes)
# A clean uniquely-resolved match is not problematic merely because its alternative failed.
problematic = filter(r -> !r.beneficiary_valid && !r.committing_valid || !isempty(r.quarantine_reasons), report.ledger)
println("\nProblematic match rows ($(nrow(problematic))):")
display(problematic)
problem_ids = Set(problematic.match_id)
problem_events = filter(r -> r.match_id in problem_ids, report.events)
println("\nEvents for problematic matches (deduplicated provider incident IDs):")
display(problem_events)
println("\nNo files were written. Review quarantines before any Stage 3 feature work.")
