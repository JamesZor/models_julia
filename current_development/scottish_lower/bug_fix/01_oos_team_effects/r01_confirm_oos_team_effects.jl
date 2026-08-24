# %% [markdown]
# # Issue 01 notebook — confirm that OOS team effects silently disappear
#
# Run this file block-by-block in a **fresh Julia REPL** from the repository root. The blocks are
# intentionally linear and verbose so they can be highlighted in Neovim and sent to a remote Kitty
# terminal. Do not `include` the complete runner: inspect each result before moving on.
#
# Investigation sequence:
# 1. Load the exact prototype model definitions and one saved experiment.
# 2. Recreate the temporal boundaries and one genuine next-period OOS fold.
# 3. Compare OOS row schema with the fitted feature map schema.
# 4. Reproduce the current lookup failure directly.
# 5. Measure the failure across every fold without rebuilding expensive pxG/wealth features.
# 6. Quantify posterior team terms omitted in one representative fold.
# 7. Prototype a name-to-training-index bridge without changing package dispatch.

# %%
# BLOCK 1 — Environment and prototype definitions
#
# This loads only the model family needed for the current pxG/wealth recombination champion. `l05`
# includes its l01/l03/l04 dependencies. Restart Julia before resending this block if a type
# redefinition error occurs.

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf
using MCMCChains

# IMPORTANT: l05 declares `const ROOT` in Main. Include it before creating any notebook path globals.
# If an earlier attempt already assigned `ROOT` as a non-constant, restart Julia before this block.
include(joinpath(pkgdir(BayesianFootball),
    "current_development/scottish_lower/open_play/l05_recomb_pxg_models.jl"))

PROJECT_ROOT = pkgdir(BayesianFootball)
OPEN_PLAY_DIR = joinpath(PROJECT_ROOT, "current_development/scottish_lower/open_play")
ISSUE_DIR = joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/01_oos_team_effects")

include(joinpath(ISSUE_DIR, "l01_team_mapping_diagnostics.jl"))
using .TeamMappingDiagnostics

BFData        = BayesianFootball.Data
BFFeatures    = BayesianFootball.Features
BFExperiments = BayesianFootball.Experiments
BFPreGame     = BayesianFootball.Models.PreGame

println("Loaded issue-01 diagnostics from: $ISSUE_DIR")

# %%
# BLOCK 2 — Load the datastore and identify the exact saved experiment
#
# We filter folders before deserialization so unrelated legacy model types do not generate JLD2
# reconstruction failures. If several candidates appear, inspect `candidate_table`. By default this
# chooses the lexicographically latest timestamp; set BF_BUGFIX_EXPERIMENT to pin another folder.

TARGET_PREFIX = "recomb_pxg_wealth_integrated_hl365_hs2"
ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours = 10_000)

all_folders = BFExperiments.list_experiments(
    "scottish_open_play_grid"; data_dir = joinpath(PROJECT_ROOT, "data"))
candidate_folders = sort(filter(path -> startswith(basename(path), TARGET_PREFIX), all_folders))
isempty(candidate_folders) && error("No experiment folder starts with '$TARGET_PREFIX'")

candidate_table = DataFrame(
    index = eachindex(candidate_folders),
    folder = basename.(candidate_folders),
    path = candidate_folders,
)
display(candidate_table)

requested_folder = get(ENV, "BF_BUGFIX_EXPERIMENT", "")
selected_folder = if isempty(requested_folder)
    @warn "BF_BUGFIX_EXPERIMENT is unset; selecting latest candidate" selected = basename(last(candidate_folders))
    last(candidate_folders)
else
    matches = filter(path -> basename(path) == requested_folder, candidate_folders)
    length(matches) == 1 || error("Requested folder '$requested_folder' was not found uniquely")
    only(matches)
end

exp = only(BFExperiments.load_experiments([selected_folder]))
println("Selected experiment: ", exp.config.name)
println("Artifact folder:      ", selected_folder)
println("Saved folds:          ", length(exp.training_results.items))

# %%
# BLOCK 3 — Recreate temporal boundaries and select one genuine OOS fold
#
# `target_match_ids` belong to the fitted fold. Production OOS extraction instead calls
# `get_next_matches`, which returns time_step + 1. We deliberately use that same production route.

boundaries = BFData.create_id_boundaries(ds, exp.config.splitter)
length(boundaries) == length(exp.training_results.items) ||
    error("Datastore/experiment fold count mismatch")

fold_sizes = DataFrame(fold = Int[], history = Int[], training_target = Int[], next_oos = Int[])
for i in eachindex(boundaries)
    boundary, _ = boundaries[i]
    next_df = BFData.get_next_matches(ds, boundaries[i], exp.config.splitter)
    push!(fold_sizes, (i, length(boundary.history_match_ids),
                       length(boundary.target_match_ids), nrow(next_df)))
end
display(fold_sizes)

# The final non-empty fold is representative and usually contains established teams. Change this
# manually after inspecting fold_sizes if a different fold is desired.
fold_index = findlast(fold_sizes.next_oos .> 0)
isnothing(fold_index) && error("No fold has next-period OOS matches")

boundary_tuple = boundaries[fold_index]
boundary = first(boundary_tuple)
oos_df = DataFrame(BFData.get_next_matches(ds, boundary_tuple, exp.config.splitter))
chain = exp.training_results.items[fold_index][1]

println("Selected fold $fold_index with $(nrow(oos_df)) genuine OOS matches")
display(first(oos_df, min(5, nrow(oos_df))))

# %%
# BLOCK 4 — Build one feature set and inspect the incompatible schemas
#
# This is the first potentially expensive block: pxG and wealth features are built once for the
# selected fold. We expect OOS rows to expose string team names, while the custom FeatureSet stores
# an integer-keyed map.

feature_set = BFFeatures.create_features(
    boundary, ds, exp.config.model, exp.config.splitter.dynamics_col)
team_map = team_index_map(feature_set)

schema_report = (
    oos_columns = propertynames(oos_df),
    has_home_team = :home_team in propertynames(oos_df),
    has_home_team_id = :home_team_id in propertynames(oos_df),
    team_map_type = typeof(team_map),
    team_map_key_type = keytype(team_map),
    team_map_value_type = valtype(team_map),
    n_training_teams = length(team_map),
)
println(schema_report)
println("Example map entries: ", first(collect(team_map), min(5, length(team_map))))

@assert schema_report.has_home_team
@assert !schema_report.has_home_team_id
@assert keytype(team_map) <: Integer

# %%
# BLOCK 5 — Reproduce current adapter lookup and compare with a candidate bridge
#
# `current_*_idx` follows the exact two-stage fallback in l03/l04/l05. `candidate_*_idx` uses the
# clean match identity table to bridge a team slug to the posterior column fitted for that team.

selected_mapping = mapping_comparison(oos_df, feature_set)
display(selected_mapping)
println("Selected-fold summary: ", mapping_summary(selected_mapping))

# Known OOS teams should map under the candidate bridge. The current adapter is expected to map none.
@assert count(selected_mapping.current_both_known) == 0
@assert count(selected_mapping.candidate_both_known) > 0

# %%
# BLOCK 6 — Behavioral proof: swapping known team names does not change current predictions
#
# We keep match ID, date, tournament, wealth lookup, and every other field fixed, then only swap the
# home/away team-name strings. If team effects were active, this should alter the extracted rates.
# Under the bug, both names map to -1 and rates remain bit-for-bit unchanged.

probe_original = copy(oos_df[1:1, :])
probe_swapped = copy(probe_original)
probe_swapped.home_team[1], probe_swapped.away_team[1] =
    probe_original.away_team[1], probe_original.home_team[1]

pred_original = BFPreGame.extract_parameters(
    exp.config.model, probe_original, feature_set, chain)[Int(probe_original.match_id[1])]
pred_swapped = BFPreGame.extract_parameters(
    exp.config.model, probe_swapped, feature_set, chain)[Int(probe_swapped.match_id[1])]

swap_test = (
    teams_original = (probe_original.home_team[1], probe_original.away_team[1]),
    teams_swapped = (probe_swapped.home_team[1], probe_swapped.away_team[1]),
    max_abs_open_home_change = maximum(abs.(pred_original.lambda_open_h .- pred_swapped.lambda_open_h)),
    max_abs_open_away_change = maximum(abs.(pred_original.lambda_open_a .- pred_swapped.lambda_open_a)),
    max_abs_total_home_change = maximum(abs.(pred_original.λ_h .- pred_swapped.λ_h)),
    max_abs_total_away_change = maximum(abs.(pred_original.λ_a .- pred_swapped.λ_a)),
)
println(swap_test)

@assert swap_test.max_abs_open_home_change == 0.0
@assert swap_test.max_abs_open_away_change == 0.0

# %%
# BLOCK 7 — Fast all-fold extent scan
#
# Rebuilding pxG and wealth FeatureSets for every fold is unnecessary to diagnose identity mapping.
# We reuse the clean identity table, reconstruct each fold's integer training map from history IDs,
# and inspect the same next-period rows used by production OOS extraction.

clean_df = feature_set.data[:clean_df]
name_to_id = build_name_to_id(clean_df)
all_fold_mapping = DataFrame(
    fold = Int[], matches = Int[], current_known = Int[], candidate_known = Int[],
    current_known_pct = Float64[], candidate_known_pct = Float64[],
    candidate_unknown_sides = Int[],
)

for i in eachindex(boundaries)
    btuple = boundaries[i]
    b = first(btuple)
    next_df = DataFrame(BFData.get_next_matches(ds, btuple, exp.config.splitter))
    isempty(next_df) && continue

    history_ids = Set(Int.(b.history_match_ids))
    history_rows = filter(row -> Int(row.match_id) in history_ids, clean_df)
    training_team_ids = sort(unique(vcat(
        Int.(history_rows.home_team_id), Int.(history_rows.away_team_id))))
    fold_team_map = Dict(team_id => idx for (idx, team_id) in enumerate(training_team_ids))

    current_known = 0
    candidate_known = 0
    candidate_unknown_sides = 0
    for row in eachrow(next_df)
        # ds.matches has no ID columns, so this reproduces the current failed string lookup.
        ch = current_oos_team_index(row, :home, fold_team_map)
        ca = current_oos_team_index(row, :away, fold_team_map)

        h_id = get(name_to_id, String(row.home_team), -1)
        a_id = get(name_to_id, String(row.away_team), -1)
        fh = get(fold_team_map, h_id, -1)
        fa = get(fold_team_map, a_id, -1)

        current_known += (ch > 0 && ca > 0)
        candidate_known += (fh > 0 && fa > 0)
        candidate_unknown_sides += (fh <= 0) + (fa <= 0)
    end

    n = nrow(next_df)
    push!(all_fold_mapping, (
        i, n, current_known, candidate_known,
        100 * current_known / n, 100 * candidate_known / n,
        candidate_unknown_sides,
    ))
end

display(all_fold_mapping)
all_fold_totals = (
    folds = nrow(all_fold_mapping),
    matches = sum(all_fold_mapping.matches),
    current_known_matches = sum(all_fold_mapping.current_known),
    candidate_known_matches = sum(all_fold_mapping.candidate_known),
    genuinely_unknown_sides = sum(all_fold_mapping.candidate_unknown_sides),
)
println("All-fold totals: ", all_fold_totals)

# This confirms the schema defect globally. Do not assert that candidate unknowns are zero: genuinely
# unseen/promoted teams require an explicit fallback and are a valid separate case.
@assert all_fold_totals.current_known_matches == 0
@assert all_fold_totals.candidate_known_matches > 0

# %%
# BLOCK 8 — Quantify omitted posterior team contributions in the selected fold
#
# First inspect centered raw effects (matching the current extractor's intended—but unreachable—team
# branch). Then inspect correctly scaled fitted effects when tau parameters exist. The latter overlaps
# audit issue 02 and is shown only to avoid mistaking the current extraction formula for ground truth.

n_teams = feature_set.data[:n_teams]
raw_effects = raw_centered_team_effects(chain, n_teams)
fitted_effects = fitted_team_effects(chain, n_teams)

raw_contributions = posterior_team_contributions(oos_df, feature_set, raw_effects)
fitted_contributions = posterior_team_contributions(oos_df, feature_set, fitted_effects)

println("Current-extractor-semantics contribution summary (mapping fixed only):")
println(contribution_summary(raw_contributions))
println("Fitted-model contribution summary (mapping + tau reconstruction):")
println(contribution_summary(fitted_contributions))
println("tau parameters present: ", fitted_effects.has_tau)

display(first(fitted_contributions, min(10, nrow(fitted_contributions))))

# A nonzero spread in multipliers establishes that silently replacing the terms with zero is material.
fitted_summary = contribution_summary(fitted_contributions)
@assert fitted_summary.known_matches > 0
@assert fitted_summary.max_abs_log_rate_term > 0

# %%
# BLOCK 9 — Candidate bridge contract (no global method patch yet)
#
# This validates the smallest identity fix: persist/use `team_name_to_index::Dict{String,Int}` for OOS
# extraction. It intentionally does not redefine `extract_parameters`; that belongs in the next
# implementation notebook after we review this evidence.

name_to_training_index = build_name_to_training_index(feature_set)
known_oos_names = unique(vcat(String.(oos_df.home_team), String.(oos_df.away_team)))
bridge_review = DataFrame(
    team_name = known_oos_names,
    training_index = [get(name_to_training_index, name, -1) for name in known_oos_names],
)
bridge_review.is_known = bridge_review.training_index .> 0
display(sort(bridge_review, :team_name))

valid_indices = filter(>(0), bridge_review.training_index)
@assert all((1 .<= valid_indices) .& (valid_indices .<= n_teams))
@assert length(unique(valid_indices)) == length(valid_indices)

println("Candidate contract passed for $(length(valid_indices)) known teams; ",
        "$(count(.!bridge_review.is_known)) teams require explicit unseen-team fallback.")

# %%
# BLOCK 10 — Persist compact outputs for review
#
# CSV files contain no posterior draws—only mapping counts and summary contributions. Commit them only
# if they are useful as durable evidence; otherwise leave them as local notebook output.

using CSV
CSV.write(joinpath(ISSUE_DIR, "mapping_all_folds.csv"), all_fold_mapping)
CSV.write(joinpath(ISSUE_DIR, "selected_fold_mapping.csv"), selected_mapping)
CSV.write(joinpath(ISSUE_DIR, "selected_fold_fitted_contributions.csv"), fitted_contributions)

println("Wrote diagnostic CSVs to $ISSUE_DIR")
println("Next: copy the compact summaries into FINDINGS.md before implementing any fix.")
