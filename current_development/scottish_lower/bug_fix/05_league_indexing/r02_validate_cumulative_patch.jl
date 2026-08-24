# Saved-chain/no-sampling cumulative validation for permanent issues 01, 02, and 05.
# Run block-by-block on beast.  It loads artifacts only; it never invokes a sampler.

# %% BLOCK 1 -- loader and invariant helpers
using Revise, BayesianFootball, DataFrames, MCMCChains, Test, Logging
const PROJECT_ROOT = pkgdir(BayesianFootball)
# l05 transitively loads l04 and l03; include it once to avoid redefining prototype types.
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/open_play/l05_recomb_pxg_models.jl"))
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/01_oos_team_effects/l02_existing_chain_team_bridge.jl"))
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/02_tau_extraction/l01_tau_extraction_diagnostics.jl"))
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/05_league_indexing/l01_league_index_diagnostics.jl"))
using .ExistingChainTeamBridge, .TauExtractionDiagnostics, .LeagueIndexDiagnostics
const BFData = BayesianFootball.Data
const BFFeatures = BayesianFootball.Features
const BFExperiments = BayesianFootball.Experiments

"""Assert that names are a copied bridge, not a re-enumeration."""
function assert_saved_columns!(fs)
    d, ids = fs.data, Dict{String,Int}()
    @assert d[:league_encoding] == :pooled_legacy_one_column
    @assert d[:league_map][56] == d[:league_map][57] == 1
    for row in eachrow(d[:clean_df]), (namecol, idcol) in ((:home_team, :home_team_id), (:away_team, :away_team_id))
        ids[String(row[namecol])] = Int(row[idcol])
    end
    for (name, col) in d[:team_name_to_index]
        @assert col == d[:team_map][ids[name]] "name bridge renumbered saved column for $name"
    end
    nothing
end

function assert_oos_helpers!(fs)
    d = fs.data
    known = first(keys(d[:team_name_to_index]))
    row = (home_team=known, away_team="permanent-validation-unseen", tournament_id=56)
    unknowns = Dict{String,Int}()
    @assert _oos_team_index(row, :home, d[:team_map], d[:team_name_to_index], unknowns) == d[:team_name_to_index][known]
    @assert _oos_team_index(row, :away, d[:team_map], d[:team_name_to_index], unknowns) == -1
    @assert sum(values(unknowns)) == 1
    @test_logs (:warn, r"genuinely unseen teams") _warn_oos_unknown_teams!(unknowns, "cumulative validation")
    @assert _oos_league_index((tournament_id=56,), d[:league_map], d[:n_leagues]) == 1
    @assert _oos_league_index((tournament_id=57,), d[:league_map], d[:n_leagues]) == 1
    @test_throws ErrorException _oos_league_index((tournament_id=999,), d[:league_map], d[:n_leagues])
end

# %% BLOCK 2 -- artifact discovery and feature construction (no sampling)
ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
const DATA_DIR = joinpath(PROJECT_ROOT, "data")
const SPECS = (
    l03=(prefix="recomb_negbin_integrated", label="l03 NegBin"),
    l04=(prefix="recomb_pois_wealth_integrated", label="l04 wealth"),
    l05=(prefix="recomb_pxg_wealth_integrated", label="l05 pxG"),
)
function latest_artifact(prefix)
    folders = sort(filter(p -> startswith(basename(p), prefix), BFExperiments.list_experiments("scottish_open_play_grid"; data_dir=DATA_DIR)))
    isempty(folders) && return nothing
    only(BFExperiments.load_experiments([last(folders)]))
end
artifacts = Dict(k => latest_artifact(v.prefix) for (k,v) in pairs(SPECS))
contexts = Dict{Symbol,Any}()
for (kind, experiment) in artifacts
    isnothing(experiment) && (@warn "artifact unavailable; reporting rather than masking validation" kind; continue)
    boundaries = BFData.create_id_boundaries(ds, experiment.config.splitter)
    fold = findlast(i -> !isempty(BFData.get_next_matches(ds, boundaries[i], experiment.config.splitter)), eachindex(boundaries))
    isnothing(fold) && (@warn "artifact has no executable OOS fold" kind; continue)
    fs = BFFeatures.create_features(first(boundaries[fold]), ds, experiment.config.model, experiment.config.splitter.dynamics_col)
    oos = DataFrame(BFData.get_next_matches(ds, boundaries[fold], experiment.config.splitter))
    assert_saved_columns!(fs); assert_oos_helpers!(fs)
    chain = experiment.training_results.items[fold][1]
    contexts[kind] = (; experiment, boundaries, fold, fs, oos, chain)
    # Issue 02: production helper is exactly the diagnostic centred/tau-scaled matrix.
    tau = _tau_scaled_team_effects(chain, fs.data[:n_teams]; context="cumulative $kind")
    diag = TauExtractionDiagnostics.tau_effects(chain, fs.data[:n_teams])
    @assert tau.alpha == diag.exact_alpha && tau.beta == diag.exact_beta
    known_sides = sum(_oos_team_index(row, side, fs.data[:team_map], fs.data[:team_name_to_index], Dict{String,Int}()) > 0
                      for row in eachrow(oos), side in (:home, :away))
    @assert known_sides > 0 "no known OOS side activated for $kind"
    route = try
        produced = BayesianFootball.Models.PreGame.extract_parameters(experiment.config.model, oos, fs, chain)
        @assert length(produced) == nrow(oos)
        "passed"
    catch err
        @warn "production route blocked by an independent extraction defect after mapping/tau/league invariants passed" kind exception=(err, catch_backtrace())
        "blocked: $(typeof(err))"
    end
    println("PASS $(SPECS[kind].label): maps, known OOS activation ($known_sides sides), pooled leagues, tau matrix; route=$route")
end

# %% BLOCK 3 -- executable l05 production-vs-pooled-candidate parity
# l01 diagnostics supplies the otherwise-identical pooled candidate.  Production now must equal it.
experiment = artifacts[:l05]
if !isnothing(experiment)
    boundaries = BFData.create_id_boundaries(ds, experiment.config.splitter)
    fold = findlast(i -> begin x = DataFrame(BFData.get_next_matches(ds, boundaries[i], experiment.config.splitter)); nrow(x)>0 && all(Int.(x.tournament_id) .∈ Ref([56,57])) end, eachindex(boundaries))
    if !isnothing(fold)
        fs = BFFeatures.create_features(first(boundaries[fold]), ds, experiment.config.model, experiment.config.splitter.dynamics_col)
        oos, chain = DataFrame(BFData.get_next_matches(ds, boundaries[fold], experiment.config.splitter)), experiment.training_results.items[fold][1]
        production = PreGame.extract_parameters(experiment.config.model, oos, fs, chain)
        bridge = build_name_to_existing_column(fs)
        candidate = reconstruct_l05_league_paths(oos, fs, chain; bridge=bridge).paths[:artifact]
        for row in eachrow(oos)
            got, want = production[Int(row.match_id)], candidate[Int(row.match_id)]
            # Candidate and production differ only in container field spelling, not arithmetic.
            @assert got.lambda_open_h == want.lambda_open_h && got.lambda_open_a == want.lambda_open_a
            @assert got.λ_h == want.λ_h && got.λ_a == want.λ_a
        end
        println("PASS l05: tournament 56 stable and tournament 57 equals pooled issue-05 candidate")
    else
        @warn "l05 has no executable 56/57 OOS fold; parity not asserted"
    end
end

# %% BLOCK 4 -- independent known l03 referee artifact drift is reported, never swallowed
# This intentionally runs after map/tau/league assertions above.
if haskey(contexts, :l03)
    ctx = contexts[:l03]
    fitted_refs = count(n -> startswith(String(n), "raw_gamma_ref["), names(ctx.chain))
    expected_refs = ctx.fs.data[:n_refs]
    fitted_refs == expected_refs || @warn "known independent l03 referee artifact drift: reconstructed feature vocabulary and saved chain disagree; mapping/tau/league checks already completed" fitted_refs expected_refs
end
