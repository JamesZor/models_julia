# %% [markdown]
# Issue 01 phase 2: validate an existing saved pxG champion chain without patching l03--l05.
# Send each block independently to a fresh remote Julia REPL. This is reconstruction only: no MCMC.

# %% BLOCK 1 -- load exactly the r01 prototype/artifact environment
using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, MCMCChains
include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower/open_play/l05_recomb_pxg_models.jl"))
PROJECT_ROOT = pkgdir(BayesianFootball)
ISSUE_DIR = joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/01_oos_team_effects")
include(joinpath(ISSUE_DIR, "l02_existing_chain_team_bridge.jl"))
using .ExistingChainTeamBridge
BFData=BayesianFootball.Data; BFFeatures=BayesianFootball.Features
BFExperiments=BayesianFootball.Experiments; BFPreGame=BayesianFootball.Models.PreGame
@assert bridge_self_check!() # ordering, known and unknown contracts; no chain columns reordered

# %% BLOCK 2 -- exact r01 saved experiment selection (pin BF_BUGFIX_EXPERIMENT for reproducibility)
TARGET_PREFIX="recomb_pxg_wealth_integrated_hl365_hs2"
ds=BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
folders=sort(filter(p->startswith(basename(p),TARGET_PREFIX), BFExperiments.list_experiments("scottish_open_play_grid"; data_dir=joinpath(PROJECT_ROOT,"data"))))
isempty(folders) && error("No matching artifact")
requested=get(ENV,"BF_BUGFIX_EXPERIMENT","")
folder=isempty(requested) ? last(folders) : only(filter(p->basename(p)==requested,folders))
experiment=only(BFExperiments.load_experiments([folder]))
println("artifact=",folder)

# %% BLOCK 3 -- exact r01 next-period OOS fold and its fitted FeatureSet
boundaries=BFData.create_id_boundaries(ds, experiment.config.splitter)
fold_index=findlast(i->nrow(BFData.get_next_matches(ds,boundaries[i],experiment.config.splitter))>0, eachindex(boundaries))
isnothing(fold_index) && error("No OOS fold")
boundary_tuple=boundaries[fold_index]; boundary=first(boundary_tuple)
oos_df=DataFrame(BFData.get_next_matches(ds,boundary_tuple,experiment.config.splitter))
feature_set=BFFeatures.create_features(boundary,ds,experiment.config.model,experiment.config.splitter.dynamics_col)
chain=experiment.training_results.items[fold_index][1]
n_draws=length(vec(Array(chain["base_mu"])))
println("fold=$fold_index, OOS matches=$(nrow(oos_df)), draws=$n_draws")

# %% BLOCK 4 -- old versus corrected identity maps and unknown-team report
bridge=build_name_to_existing_column(feature_set)
@assert assert_bridge_invariants!(feature_set,bridge) # every name retains its original fitted column
maps=mapping_comparison(oos_df,feature_set); display(maps)
diag=unknown_team_diagnostics(oos_df,bridge); println(diag)
@assert all((maps.corrected_home_idx .> 0) .| in.(maps.home_team, Ref(diag.unknown_names)))
@assert all((maps.corrected_away_idx .> 0) .| in.(maps.away_team, Ref(diag.unknown_names)))
@assert any((maps.legacy_home_idx .!= maps.corrected_home_idx) .| (maps.legacy_away_idx .!= maps.corrected_away_idx))

# %% BLOCK 5 -- current production latents (old mapping and old extraction semantics)
# This is intentionally a call to the unmodified l05 method, providing the baseline users currently receive.
old_latents=BFPreGame.extract_parameters(experiment.config.model,oos_df,feature_set,chain)

# %% BLOCK 6 -- mapping-only reconstruction: isolates issue 01 from the current l05 baseline
# Formulae/fallbacks are l05 dataframe extraction verbatim; only name → existing-column lookup changes.
mapping_only=reconstruct_pxg_mapping_only(oos_df,feature_set,chain; bridge=bridge)
@assert mapping_only.diagnostics == diag
println("mapping-only unknown-team diagnostics: ",mapping_only.diagnostics)

# %% BLOCK 7 -- old versus mapping-only: isolated evidence for issue 01
known_row=findfirst((maps.corrected_home_idx .> 0) .& (maps.corrected_away_idx .> 0))
isnothing(known_row) && error("No known-known OOS fixture; inspect diagnostics")
mid=maps.match_id[known_row]; old=old_latents[mid]; mapped=mapping_only.latents[mid]
mapping_comparison_result=(match_id=mid, teams=(maps.home_team[known_row],maps.away_team[known_row]),
  old_open=(mean(old.lambda_open_h),mean(old.lambda_open_a)), mapping_only_open=(mean(mapped.lambda_open_h),mean(mapped.lambda_open_a)),
  max_abs_open=(maximum(abs.(old.lambda_open_h.-mapped.lambda_open_h)),maximum(abs.(old.lambda_open_a.-mapped.lambda_open_a))),
  max_abs_total=(maximum(abs.(old.λ_h.-mapped.λ_h)),maximum(abs.(old.λ_a.-mapped.λ_a))))
println(mapping_comparison_result)
old_score=score_matrix(old); mapping_score=score_matrix(mapped)
println((mapping_only_score_matrix_max_abs=maximum(abs.(old_score.data.-mapping_score.data)), old_mass=mean(sum(old_score.data,dims=(1,2))), mapping_only_mass=mean(sum(mapping_score.data,dims=(1,2)))))
@assert maximum(abs.(old_score.data.-mapping_score.data)) > 0

# %% BLOCK 8 -- mapping-only versus fitted: deferred issue 02 / extraction-parity evidence
# This is NOT evidence attributable solely to issue 01: it adds tau, clamps/floors, and league-1 semantics.
fitted=reconstruct_pxg_fitted(oos_df,feature_set,chain; bridge=bridge)
@assert fitted.diagnostics == diag
fitted_latent=fitted.latents[mid]
fitted_comparison=(match_id=mid,
  mapping_only_open=(mean(mapped.lambda_open_h),mean(mapped.lambda_open_a)), fitted_open=(mean(fitted_latent.lambda_open_h),mean(fitted_latent.lambda_open_a)),
  max_abs_open=(maximum(abs.(mapped.lambda_open_h.-fitted_latent.lambda_open_h)),maximum(abs.(mapped.lambda_open_a.-fitted_latent.lambda_open_a))),
  max_abs_total=(maximum(abs.(mapped.λ_h.-fitted_latent.λ_h)),maximum(abs.(mapped.λ_a.-fitted_latent.λ_a))))
println(fitted_comparison)
fitted_score=score_matrix(fitted_latent)
println((fitted_score_matrix_max_abs_vs_mapping_only=maximum(abs.(mapping_score.data.-fitted_score.data)), mapping_only_mass=mean(sum(mapping_score.data,dims=(1,2))), fitted_mass=mean(sum(fitted_score.data,dims=(1,2)))))

# %% BLOCK 9 -- known-team swap sensitivity for issue 01 path (only names change)
probe=copy(oos_df[known_row:known_row,:]); swapped=copy(probe)
swapped.home_team[1],swapped.away_team[1]=probe.away_team[1],probe.home_team[1]
original=reconstruct_pxg_mapping_only(probe,feature_set,chain; bridge=bridge).latents[Int(probe.match_id[1])]
reversed=reconstruct_pxg_mapping_only(swapped,feature_set,chain; bridge=bridge).latents[Int(swapped.match_id[1])]
swap_sensitivity=(teams=(probe.home_team[1],probe.away_team[1]), max_open_h=maximum(abs.(original.lambda_open_h.-reversed.lambda_open_h)), max_open_a=maximum(abs.(original.lambda_open_a.-reversed.lambda_open_a)), max_pen_h=maximum(abs.(original.lambda_pen_h.-reversed.lambda_pen_h)), max_pen_a=maximum(abs.(original.lambda_pen_a.-reversed.lambda_pen_a)))
println(swap_sensitivity)
@assert swap_sensitivity.max_open_h > 0 || swap_sensitivity.max_open_a > 0
@assert swap_sensitivity.max_pen_h > 0 || swap_sensitivity.max_pen_a > 0

# %% BLOCK 10 -- compact review template (paste into FINDINGS.md)
println("""
artifact: $(basename(folder)); fold: $fold_index
unknown diagnostics: $diag
old vs mapping-only (issue 01): $mapping_comparison_result
mapping-only vs fitted (deferred issue 02/extraction parity): $fitted_comparison
mapping-only swap sensitivity: $swap_sensitivity
Caveat: mapping-only isolates the name bridge; fitted differences also include deferred transform parity.
""")
