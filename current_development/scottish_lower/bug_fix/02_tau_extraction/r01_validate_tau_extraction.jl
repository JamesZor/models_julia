# %% [markdown]
# Issue 02 phase 1 — no sampling and no production edits. Send blocks to a fresh beast REPL.

# %% BLOCK 1 -- loader environment; issue 01 owns the name -> existing-column identity bridge
using Revise
using BayesianFootball
using DataFrames, Statistics, MCMCChains
include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower/open_play/l05_recomb_pxg_models.jl"))
PROJECT_ROOT=pkgdir(BayesianFootball)
include(joinpath(PROJECT_ROOT,"current_development/scottish_lower/bug_fix/01_oos_team_effects/l02_existing_chain_team_bridge.jl"))
include(joinpath(PROJECT_ROOT,"current_development/scottish_lower/bug_fix/02_tau_extraction/l01_tau_extraction_diagnostics.jl"))
using .ExistingChainTeamBridge, .TauExtractionDiagnostics
@assert bridge_self_check!()
display(affected_method_manifest)

# %% BLOCK 2 -- pin the existing pxG champion; artifact loading/features can be expensive, but no MCMC occurs
BFData=BayesianFootball.Data; BFF=BayesianFootball.Features; BFE=BayesianFootball.Experiments; BFP=BayesianFootball.Models.PreGame
prefix="recomb_pxg_wealth_integrated_hl365_hs2"
ds=BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
folders=sort(filter(p->startswith(basename(p),prefix), BFE.list_experiments("scottish_open_play_grid";data_dir=joinpath(PROJECT_ROOT,"data"))))
requested=get(ENV,"BF_BUGFIX_EXPERIMENT","")
folder=isempty(requested) ? last(folders) : only(filter(p->basename(p)==requested,folders))
experiment=only(BFE.load_experiments([folder])); println("artifact=",basename(folder))

# %% BLOCK 3 -- same next-period OOS fold / saved chain, with validated issue-01 bridge
boundaries=BFData.create_id_boundaries(ds,experiment.config.splitter)
fold=findlast(i->nrow(BFData.get_next_matches(ds,boundaries[i],experiment.config.splitter))>0,eachindex(boundaries))
boundary_tuple=boundaries[fold]; oos_df=DataFrame(BFData.get_next_matches(ds,boundary_tuple,experiment.config.splitter))
fs=BFF.create_features(first(boundary_tuple),ds,experiment.config.model,experiment.config.splitter.dynamics_col)
chain=experiment.training_results.items[fold][1]; bridge=build_name_to_existing_column(fs)
@assert assert_bridge_invariants!(fs,bridge)
println((fold=fold,matches=nrow(oos_df),draws=length(chain_vector(chain,"base_mu")),unknown=unknown_team_diagnostics(oos_df,bridge)))

# %% BLOCK 4 -- shape-safe tau reconstruction, zero-sum checks, and posterior scale magnitude
# The loader handles 1-column and iterations x parameters x chains arrays, then stacks draws.
effects=tau_effects(chain,fs.data[:n_teams]); @assert assert_tau_reconstruction!(effects)
println((tau=tau_summary(effects),team_effects=team_effect_summary(effects)))
println("per-draw centered sums (max abs): ", maximum(abs.(sum(effects.centered_alpha,dims=2))), ", ", maximum(abs.(sum(effects.centered_beta,dims=2))))

# %% BLOCK 5 -- corrected mapping + CURRENT unscaled l05 arithmetic versus otherwise identical tau-only arithmetic
# No fitted-path clamps/floors and no training league remap are introduced: kappa, penalties, wealth,
# month, tournament-57 league fallback, and score construction remain the current l05 semantics.
mapping_only=reconstruct_pxg_mapping_only(oos_df,fs,chain;bridge=bridge)
tau_only=reconstruct_pxg_tau_only(oos_df,fs,chain;bridge=bridge)
@assert mapping_only.diagnostics == tau_only.diagnostics
known=findfirst(r->corrected_team_index(r,:home,bridge)>0 && corrected_team_index(r,:away,bridge)>0,eachrow(oos_df))
isnothing(known) && error("No known-known OOS fixture; choose another fold and record unknowns")
mid=Int(oos_df.match_id[known]); raw=mapping_only.latents[mid]; scaled=tau_only.latents[mid]
comparison=(match_id=mid,teams=(oos_df.home_team[known],oos_df.away_team[known]),
 raw_open=(mean(raw.lambda_open_h),mean(raw.lambda_open_a)),tau_open=(mean(scaled.lambda_open_h),mean(scaled.lambda_open_a)),
 max_abs_open=(maximum(abs.(raw.lambda_open_h.-scaled.lambda_open_h)),maximum(abs.(raw.lambda_open_a.-scaled.lambda_open_a))),
 max_abs_total=(maximum(abs.(raw.λ_h.-scaled.λ_h)),maximum(abs.(raw.λ_a.-scaled.λ_a))))
println(comparison)

# %% BLOCK 6 -- score-grid comparison; normalization is checked, not repaired
raw_score=score_matrix(raw); tau_score=score_matrix(scaled)
score_comparison=(max_abs=maximum(abs.(raw_score.data.-tau_score.data)),raw_mass=mean(sum(raw_score.data,dims=(1,2))),tau_mass=mean(sum(tau_score.data,dims=(1,2))))
@assert all(isapprox.(vec(sum(raw_score.data,dims=(1,2))),1;atol=1e-10))
@assert all(isapprox.(vec(sum(tau_score.data,dims=(1,2))),1;atol=1e-10))
println(score_comparison)

# %% BLOCK 7 -- known-team swap sensitivity under the same tau-only semantics
probe=copy(oos_df[known:known,:]); swapped=copy(probe)
swapped.home_team[1],swapped.away_team[1]=probe.away_team[1],probe.home_team[1]
a=reconstruct_pxg_tau_only(probe,fs,chain;bridge=bridge).latents[Int(probe.match_id[1])]
b=reconstruct_pxg_tau_only(swapped,fs,chain;bridge=bridge).latents[Int(swapped.match_id[1])]
swap=(teams=(probe.home_team[1],probe.away_team[1]),max_open_h=maximum(abs.(a.lambda_open_h.-b.lambda_open_h)),max_open_a=maximum(abs.(a.lambda_open_a.-b.lambda_open_a)),max_total_h=maximum(abs.(a.λ_h.-b.λ_h)),max_total_a=maximum(abs.(a.λ_a.-b.λ_a)))
println(swap); @assert swap.max_open_h>0 || swap.max_open_a>0

# %% BLOCK 8 -- paste this compact record into FINDINGS.md
println("""
artifact: $(basename(folder)); fold: $fold
unknowns: $(mapping_only.diagnostics)
tau posterior: $(tau_summary(effects))
team-effect/rate-multiplier posterior: $(team_effect_summary(effects))
current-unscaled vs tau-only: $comparison
score matrix: $score_comparison
known-team tau-only swap: $swap
Caveat: this isolates tau after the issue-01 bridge; it does not fix l03 NB penalty hierarchy, clamps/floors, or league parity.
""")
