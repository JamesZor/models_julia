# %% [markdown]
# Issue 05 phase-1 saved-chain diagnostic. No sampling; run blocks in the remote persistent REPL.
# It composes issue 01's name->original-chain-column bridge and l03's permanent tau helper.

# %% BLOCK 1 -- load code and diagnostic helpers
using Revise
using BayesianFootball
using DataFrames, Statistics, MCMCChains
const PROJECT_ROOT = pkgdir(BayesianFootball)
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/open_play/l03_recombination_models.jl"))
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/open_play/l05_recomb_pxg_models.jl"))
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/01_oos_team_effects/l02_existing_chain_team_bridge.jl"))
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/05_league_indexing/l01_league_index_diagnostics.jl"))
using .ExistingChainTeamBridge, .LeagueIndexDiagnostics
const BFData = BayesianFootball.Data
const BFFeatures = BayesianFootball.Features
const BFExperiments = BayesianFootball.Experiments
@assert bridge_self_check!()
@assert league_contract_self_check!()

# %% BLOCK 2 -- select the saved l05 pxG champion (pin BF_BUGFIX_EXPERIMENT to reproduce)
const TARGET_PREFIX = "recomb_pxg_wealth_integrated_hl365_hs2"
ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
folders = sort(filter(p -> startswith(basename(p), TARGET_PREFIX),
    BFExperiments.list_experiments("scottish_open_play_grid"; data_dir=joinpath(PROJECT_ROOT, "data"))))
isempty(folders) && error("No l05 pxG champion artifact matching $TARGET_PREFIX")
requested = get(ENV, "BF_BUGFIX_EXPERIMENT", "")
folder = isempty(requested) ? last(folders) : only(filter(p -> basename(p) == requested, folders))
experiment = only(BFExperiments.load_experiments([folder]))
println("artifact=", basename(folder))

# %% BLOCK 3 -- recreate one nonempty OOS fold and retain the saved chain unchanged
boundaries = BFData.create_id_boundaries(ds, experiment.config.splitter)
# Prefer the latest real fold containing both tournaments so neither parity assertion is vacuous.
function oos_for_fold(i)
    DataFrame(BFData.get_next_matches(ds, boundaries[i], experiment.config.splitter))
end
fold_index = findlast(i -> begin
    fold_df = oos_for_fold(i)
    nrow(fold_df) > 0 && Set(Int.(fold_df.tournament_id)) == Set((56, 57))
end, eachindex(boundaries))
isnothing(fold_index) && error("Artifact has no OOS fold containing both ScottishLower tournaments 56 and 57")
boundary_tuple = boundaries[fold_index]
oos_df = oos_for_fold(fold_index)
feature_set = BFFeatures.create_features(first(boundary_tuple), ds, experiment.config.model, experiment.config.splitter.dynamics_col)
chain = experiment.training_results.items[fold_index][1]
# Quantify coverage across the artifact, then retain the final nonempty fold for reconstruction.
fold_counts = DataFrame(fold=Int[], tournament_id=Int[], fixtures=Int[])
for i in eachindex(boundaries)
    fold_df = oos_for_fold(i)
    nrow(fold_df) == 0 && continue
    for row in eachrow(tournament_counts(fold_df))
        push!(fold_counts, (i, Int(row.tournament_id), Int(row.fixtures)))
    end
end
display(fold_counts)
println((fold=fold_index, fixtures=nrow(oos_df), counts=tournament_counts(oos_df)))

# %% BLOCK 4 -- enforce the legacy saved-artifact contract before any reconstruction
league_labels = filter(n -> startswith(String(n), "delta_league["), names(chain))
@assert length(league_labels) == 1 "This audit is for a one-column legacy pooled artifact; found $(league_labels)"
contract = LegacyPooledLeagueContract(length(league_labels))
@assert assert_league_contract!(contract, oos_df)
@assert league_column(contract, 56; semantics=:legacy_l05) == league_column(contract, 56; semantics=:artifact) == 1
@assert league_column(contract, 57; semantics=:legacy_l05) == 2
@assert league_column(contract, 57; semantics=:artifact) == 1
try
    league_column(contract, 999; semantics=:artifact)
    error("unknown tournament was silently accepted")
catch err
    occursin("unknown tournament_id=999", sprint(showerror, err)) || rethrow()
end

# %% BLOCK 5 -- posterior league magnitude (base_mu + delta is the pooled fitted log-rate)
delta = chain_vector(chain, "delta_league[1]")
base = chain_vector(chain, "base_mu")
league_posterior = (delta_league_1=posterior_summary(delta), exp_delta_league_1=posterior_summary(exp.(delta)),
                    base_mu=posterior_summary(base), base_mu_plus_delta=posterior_summary(base .+ delta))
println(league_posterior)

# %% BLOCK 6 -- reconstruct otherwise-identical legacy and artifact-compatible paths
bridge = build_name_to_existing_column(feature_set)
@assert assert_bridge_invariants!(feature_set, bridge)
reconstruction = reconstruct_l05_league_paths(oos_df, feature_set, chain; bridge=bridge)
legacy, candidate = reconstruction.paths[:legacy_l05], reconstruction.paths[:artifact]
println((unknown_teams=reconstruction.unknown_teams, tau_shapes=(size(reconstruction.effects.alpha), size(reconstruction.effects.beta))))

# %% BLOCK 7 -- exact semantic assertions: 56 unchanged; 57 open rates differ by exp(delta)
for row in eachrow(oos_df)
    mid, tid = Int(row.match_id), Int(row.tournament_id)
    current, pooled = legacy[mid], candidate[mid]
    if tid == 56
        @assert current.lambda_open_h == pooled.lambda_open_h
        @assert current.lambda_open_a == pooled.lambda_open_a
        @assert current.λ_h == pooled.λ_h && current.λ_a == pooled.λ_a
    elseif tid == 57
        # Compare open play only: penalty/noise is additive, so total-rate ratios are not exp(delta).
        @assert isapprox(pooled.lambda_open_h ./ current.lambda_open_h, exp.(delta); rtol=1e-11, atol=1e-12)
        @assert isapprox(pooled.lambda_open_a ./ current.lambda_open_a, exp.(delta); rtol=1e-11, atol=1e-12)
    else
        error("assert_league_contract! should have rejected tournament $tid")
    end
end
println("league semantic assertions passed")

# %% BLOCK 8 -- fixture-level deltas, normalized score grids, and straightforward market summaries
report = DataFrame(match_id=Int[], tournament_id=Int[], legacy_open_h=Float64[], pooled_open_h=Float64[],
                   legacy_open_a=Float64[], pooled_open_a=Float64[], legacy_total_h=Float64[], pooled_total_h=Float64[])
for row in eachrow(oos_df)
    x, y = legacy[Int(row.match_id)], candidate[Int(row.match_id)]
    push!(report, (Int(row.match_id), Int(row.tournament_id), mean(x.lambda_open_h), mean(y.lambda_open_h),
                   mean(x.lambda_open_a), mean(y.lambda_open_a), mean(x.λ_h), mean(y.λ_h)))
end
display(report)
for row in eachrow(report)
    x, y = legacy[row.match_id], candidate[row.match_id]
    gx, gy = score_grid(x), score_grid(y)
    @assert all(isapprox.(vec(sum(gx, dims=(1,2))), 1.0; atol=1e-12))
    @assert all(isapprox.(vec(sum(gy, dims=(1,2))), 1.0; atol=1e-12))
    println((match_id=row.match_id, tournament_id=row.tournament_id,
             open_rate_ratio=(home=mean(y.lambda_open_h ./ x.lambda_open_h), away=mean(y.lambda_open_a ./ x.lambda_open_a)),
             markets_legacy=market_summary(gx), markets_pooled=market_summary(gy)))
end

# %% BLOCK 9 -- pasteable FINDINGS template
println("""
artifact: $(basename(folder)); fold: $fold_index; counts: $(tournament_counts(oos_df))
posterior: $league_posterior
unknown teams: $(reconstruction.unknown_teams)
result: tournament 56 is draw-wise unchanged; tournament 57 open-play candidate/current is exp(delta_league[1]).
caveat: this diagnoses only league indexing with issue-01 mapping and issue-02 tau already composed; it does not patch l03-l05.
""")
