# ==============================================================================
# r37 — Incremental extension of the Negative Binomial grid into the 2026/27 season
# ==============================================================================
#
# Sibling of `01_poisson_2426_grid/r23_extend_poisson_2627.jl`. `extend_fit` samples only the
# walk-forward folds the widened splitter adds; the 40 folds already in `runs` are loaded, not
# refitted, and the extended `Fit` is written back to the same immutable run UUID.
#
# THIS RUNNER LAUNCHES MCMC. Run it on mcmc-beast with 16 threads, and preview first:
#
#   julia --project -t 16 experiments/scottish_lower/02_negbin_2426_grid/r37_extend_negbin_2627.jl --preview
#   julia --project -t 16 experiments/scottish_lower/02_negbin_2426_grid/r37_extend_negbin_2627.jl
#
# `--refresh` forces a DataStore refetch from SQL (needs BF_DB_URL) instead of using the
# local cache; use it once the 26/27 season has fixtures the cache predates.
#
# `--preview` reports the fold delta, new fixture count and compute estimate, then stops
# without sampling. By default only `m00_negbin_baseline` is extended, matching r23's scope;
# pass model names as arguments to widen it, or `--all` for the whole grid.
#
# Requires r35_sync_to_postgres.jl to have run: the extension operates on a persisted run.

using BayesianFootball
using Dates, Printf, LinearAlgebra, ThreadPinning
import LibPQ

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const TT = BayesianFootball.Training
const DD = BayesianFootball.Data

const R37_EXPERIMENT = "scottish_lower_negbin_2426"
const R37_DEFAULT_MODELS = ["m00_negbin_baseline"]
const R37_ALL_MODELS = [
    "m00_negbin_baseline",
    "m02_negbin_wealth",
    "m03_negbin_distance",
    "m04_negbin_joint",
    "m05_negbin_production_wealth",
]

"The 24/26 splitter widened by one target season. Everything else is held fixed, so the
folds already in the database remain valid and only the 26/27 boundaries are new."
r37_splitter_2627() = DD.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25", "25/26", "26/27"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

function r37_parse_args(args)
    preview_only = "--preview" in args
    refresh = "--refresh" in args
    names = filter(a -> !startswith(a, "--"), args)
    models = "--all" in args ? copy(R37_ALL_MODELS) :
             isempty(names) ? copy(R37_DEFAULT_MODELS) : String.(names)
    unknown = setdiff(models, R37_ALL_MODELS)
    isempty(unknown) || error("Unknown model name(s): $(join(unknown, ", ")). " *
                              "Known: $(join(R37_ALL_MODELS, ", ")).")
    return (; preview_only, refresh, models)
end

"""
    r37_datastore(refresh) -> DataStore

The cache age is passed explicitly, and that is not incidental. `load_datastore_cached(ds)`
takes a 24-hour default, so an expired cache sends the runner down the SQL path — which needs
`BF_DB_URL`, absent from a non-interactive SSH environment on mcmc-beast, and fails after the
package load. Extending into a new season is also exactly the case where a stale store is
wrong: 26/27 fixtures the cache predates cannot become folds. So the choice is explicit.

`--refresh` requires `BF_DB_URL` and is checked before anything expensive happens.
"""
function r37_datastore(refresh::Bool)
    if refresh
        haskey(ENV, "BF_DB_URL") || error(
            "--refresh needs BF_DB_URL to reach the match database. Export it in the shell " *
            "that launches this runner (a non-interactive SSH session does not inherit it).")
        return DD.load_datastore_cached(DD.ScottishLower(); force = true)
    end
    return DD.load_datastore_cached(DD.ScottishLower(); max_age_hours = 100_000)
end

function r37_extend_negbin_2627(args = ARGS)
    options = r37_parse_args(args)

    println("="^90)
    println("  SCOTTISH LOWER NEGATIVE BINOMIAL: INCREMENTAL EXTENSION TO 2026/27 SEASON")
    println("="^90)
    println("  mode   : ", options.preview_only ? "PREVIEW ONLY (no sampling)" : "EXTEND (launches MCMC)")
    println("  models : ", join(options.models, ", "))
    println("  data   : ", options.refresh ? "forced refresh from SQL" : "cached DataStore")

    db = TT.PostgresStorage(R37_EXPERIMENT)
    println("  storage: ", db)
    ds = r37_datastore(options.refresh)
    seasons = sort(unique(String.(skipmissing(ds.matches.season))))
    println("  DataStore: $(length(ds.matches.match_id)) matches, seasons $(join(seasons, ", "))")
    if !("26/27" in seasons)
        @warn "The DataStore holds no 26/27 fixtures, so the widened splitter can add no " *
              "folds. Re-run with --refresh (needs BF_DB_URL) once the new season has data."
    end

    splitter_2627 = r37_splitter_2627()

    println("\n--- [1/3] Previewing extension ---")
    plans = Dict{String,Any}()
    for name in options.models
        println("\n  $name")
        plans[name] = TT.preview_extension(db, name, ds; splitter = splitter_2627)
    end

    total_new = sum(p.new_count for p in values(plans); init = 0)
    if options.preview_only
        println("\nPreview complete. $total_new new fold(s) would be sampled. No MCMC launched.")
        return (; db, ds, plans, extended = Dict{String,Any}())
    end
    if total_new == 0
        println("\nEvery selected run is already current for 26/27. Nothing to sample.")
        return (; db, ds, plans, extended = Dict{String,Any}())
    end

    println("\n--- [2/3] Executing extend_fit (QueuedExecution 16) ---")
    extended = Dict{String,Any}()
    for name in options.models
        plans[name].new_count == 0 && (println("  $name: up-to-date, skipped."); continue)
        println("\n  $name: sampling $(plans[name].new_count) new fold(s)...")
        extended[name] = TT.extend_fit(db, name, ds;
            splitter  = splitter_2627,
            execution = TT.QueuedExecution(16),
        )
        println("  $name: total folds now $(length(extended[name].folds))")
    end

    println("\n--- [3/3] Auditing database state ---")
    @printf("  %-30s | %6s | %9s | %8s | %9s | %5s\n",
            "Model", "Folds", "OOS fix.", "max R̂", "min ESS", "Div")
    println("  " * "-"^80)
    for name in options.models
        verified = TT.load_fit(db, name)
        # A NegBin run whose dispersion did not come back is a Poisson model under the wrong
        # name; it would still price and still look plausible, so check the family explicitly.
        observation_family(verified.latents) == :negbin || error(
            "$name reloaded as $(observation_family(verified.latents)) latents after extension; " *
            "the negative-binomial dispersion was lost.")
        @printf("  %-30s | %6d | %9d | %8.4f | %9.1f | %5d\n",
                name, length(verified.folds), n_matches(verified.latents),
                verified.diagnostics.max_rhat, verified.diagnostics.min_ess_bulk,
                verified.diagnostics.n_divergent)
    end
    println("="^90)

    return (; db, ds, plans, extended)
end

if abspath(PROGRAM_FILE) == @__FILE__
    r37_extend_negbin_2627()
end
