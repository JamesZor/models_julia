# %% [markdown]
# Issue 02 permanent-patch validation.  Existing saved chains only: no sampling.
# This recreates one OOS fold per saved l03 NegBin, l04 wealth, and l05 pxG artifact when available.
# IMPORTANT: production DataFrame OOS team-name -> team-id mapping remains issue 01 until that
# patch lands.  Therefore the exact assertions below compare the production tau helper against
# an independent diagnostic matrix reconstruction; OOS extractor calls are execution/shape smoke
# checks, not a claim that issue-01 mapping is fixed.

# %% BLOCK 1 -- loaders
using Revise
using BayesianFootball
using DataFrames, MCMCChains
include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower/open_play/l05_recomb_pxg_models.jl"))
const PROJECT_ROOT = pkgdir(BayesianFootball)
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/01_oos_team_effects/l02_existing_chain_team_bridge.jl"))
include(joinpath(PROJECT_ROOT, "current_development/scottish_lower/bug_fix/02_tau_extraction/l01_tau_extraction_diagnostics.jl"))
using .TauExtractionDiagnostics
const BFData = BayesianFootball.Data
const BFF = BayesianFootball.Features
const BFE = BayesianFootball.Experiments
const BFP = BayesianFootball.Models.PreGame

# %% BLOCK 2 -- select the newest matching saved artifact, or skip an unavailable type
const DATA_DIR = joinpath(PROJECT_ROOT, "data")
function latest_artifact(prefix::String)
    folders = sort(filter(p -> startswith(basename(p), prefix), BFE.list_experiments("scottish_open_play_grid"; data_dir=DATA_DIR)))
    isempty(folders) && return nothing
    requested = get(ENV, "BF_BUGFIX_" * uppercase(replace(prefix, '-' => '_')) * "_EXPERIMENT", "")
    return isempty(requested) ? last(folders) : only(filter(p -> basename(p) == requested, folders))
end

# %% BLOCK 3 -- recreate one fold and validate independent vs production tau matrices
function validate_artifact(prefix::String)
    folder = latest_artifact(prefix)
    isnothing(folder) && return println("SKIP $prefix: no saved artifact available")
    experiment = only(BFE.load_experiments([folder]))
    ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
    boundaries = BFData.create_id_boundaries(ds, experiment.config.splitter)
    fold = findlast(i -> nrow(BFData.get_next_matches(ds, boundaries[i], experiment.config.splitter)) > 0, eachindex(boundaries))
    isnothing(fold) && error("$prefix has no nonempty OOS fold")
    boundary = boundaries[fold]
    oos_df = DataFrame(BFData.get_next_matches(ds, boundary, experiment.config.splitter))
    fs = BFF.create_features(first(boundary), ds, experiment.config.model, experiment.config.splitter.dynamics_col)
    chain = experiment.training_results.items[fold][1]
    n_teams = fs.data[:n_teams]

    # Independent issue-02 diagnostic (not the production helper).
    diagnostic = tau_effects(chain, n_teams)
    @assert assert_tau_reconstruction!(diagnostic)
    production = _tau_scaled_team_effects(chain, n_teams; context="r02 $(basename(folder))")
    @assert production.alpha == diagnostic.exact_alpha
    @assert production.beta == diagnostic.exact_beta
    @assert size(production.alpha) == (length(vec(Array(chain["base_mu"]))), n_teams)

    # Exercise both affected production extraction routes without sampling.  The l03 chain route
    # exposes team effects directly, so it is also asserted against the independent matrices.
    if experiment.config.model isa TeamGoalsRecombIntegratedNegBinModel
        extracted = BFP.extract_parameters(experiment.config.model, chain, fs)
        for (team_id, idx) in fs.data[:team_map]
            @assert extracted[:alpha][team_id] == diagnostic.exact_alpha[:, idx]
            @assert extracted[:beta][team_id] == diagnostic.exact_beta[:, idx]
        end
    else
        # l04/l05 FeatureSet routes return latent states, not alpha/beta matrices; their shared
        # helper equality above is the exact reconstruction assertion.
        BayesianFootball.Predictions.extract_params(experiment.config.model, fs, chain)
    end

    # DataFrame OOS mapping remains issue 01; this is deliberately an execution/shape smoke check.
    latents = BFP.extract_parameters(experiment.config.model, oos_df, fs, chain)
    @assert length(latents) == nrow(oos_df)
    @assert all(length(v.λ_h) == size(production.alpha, 1) && length(v.λ_a) == size(production.alpha, 1) for v in values(latents))
    println((artifact=basename(folder), fold=fold, matches=nrow(oos_df), draws=size(production.alpha, 1), teams=n_teams, tau_helper_matches_independent_diagnostic=true))
end

# %% BLOCK 4 -- l03 NegBin, l04 wealth, then l05 pxG; each is skipped only when unavailable
validate_artifact("recomb_negbin_integrated")
validate_artifact("recomb_pois_wealth_integrated")
validate_artifact("recomb_pxg_wealth_integrated")

# %% [markdown]
# Interpretation: passing equality proves the permanent production reconstruction is exactly
# `(raw - rowmean) .* reshape(tau, :, 1)` for every saved draw and team column.  It does not
# resolve issue 01's DataFrame OOS mapping, and does not alter league, clamps, penalty/referee,
# wealth, or kappa behavior.
