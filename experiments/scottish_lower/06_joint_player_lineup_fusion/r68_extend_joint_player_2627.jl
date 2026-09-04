# ==============================================================================
# r68 — Incremental extension of Joint Player Lineup models into the 2026/27 season
# ==============================================================================
#
# Sibling of `01_poisson_2426_grid/r23_extend_poisson_2627.jl` and
# `02_negbin_2426_grid/r37_extend_negbin_2627.jl`. `extend_fit` samples only the
# walk-forward folds the widened splitter adds (Folds 41-43); the 40 folds already in
# `runs` are loaded from PostgreSQL, not refitted, and the extended `Fit` is written back
# to the same immutable run UUID.
#
# THIS RUNNER LAUNCHES MCMC. Run it on mcmc-beast with 16 threads, and preview first:
#
#   julia --project -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r68_extend_joint_player_2627.jl --preview
#   julia --project -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r68_extend_joint_player_2627.jl
#
# Flags:
#   --preview   reports the fold delta, new fixture count and compute estimate, then stops without sampling.
#   --refresh   forces a DataStore refetch from SQL (needs BF_DB_URL) instead of using the local cache.
#   --all       extends all 3 candidate models (m05, m12, m13) instead of default (m12, m05).
#   <names...>  specific models to extend (e.g. m12_joint_hybrid_synergy).

using BayesianFootball
using Dates, Printf, LinearAlgebra, Serialization, ThreadPinning
using UUIDs
import LibPQ

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const TT = BayesianFootball.Training
const DD = BayesianFootball.Data
const MD = BayesianFootball.MatchDay
const PG = BayesianFootball.Models.PreGame

const R68_EXPERIMENT = "scottish_lower_joint_player_2426"

const R68_MODELS = Dict{String, UUID}(
    "m12_joint_hybrid_synergy"    => UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef"),
    "m05_joint_production_wealth" => UUID("ed541a7c-01e2-447e-a771-783517728d47"),
    "m13_joint_composite"         => UUID("5474e824-8c9d-4613-8e39-841426c3f80f"),
)

const R68_DEFAULT_MODELS = [
    "m12_joint_hybrid_synergy",
    "m05_joint_production_wealth",
]

const R68_ALL_MODELS = [
    "m12_joint_hybrid_synergy",
    "m05_joint_production_wealth",
    "m13_joint_composite",
]

# ==============================================================================
# Deserialization Compatibility Shim
# ==============================================================================
# Required because historical 24/25 + 25/26 runs were serialized before
# JointGammaPoissonObservation received its 4th type parameter (SharedKappa).
# This shim mirrors l66_hierarchical_kappa_eval_loader.jl §0.
function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   T::Type{<:PG.JointGammaPoissonObservation})
    T isa DataType && return invoke(
        Serialization.deserialize,
        Tuple{Serialization.AbstractSerializer, DataType}, s, T)

    fields = Any[]
    for _ in 1:3
        tag = Int32(read(s.io, UInt8)::UInt8)
        push!(fields, Serialization.handle_deserialize(s, tag))
    end
    return PG.JointGammaPoissonObservation(
        fields[1], fields[2], fields[3], PG.SharedKappa())
end

# The 24/26 splitter widened by one target season. Everything else is held fixed, so the
# folds already in the database remain valid and only the 26/27 boundaries are new.
r68_splitter_2627() = DD.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25", "25/26", "26/27"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

function r68_parse_args(args)
    preview_only = "--preview" in args
    refresh = "--refresh" in args
    names = filter(a -> !startswith(a, "--"), args)
    models = "--all" in args ? copy(R68_ALL_MODELS) :
             isempty(names) ? copy(R68_DEFAULT_MODELS) : String.(names)
    unknown = setdiff(models, R68_ALL_MODELS)
    isempty(unknown) || error("Unknown model name(s): $(join(unknown, ", ")). " *
                              "Known: $(join(R68_ALL_MODELS, ", ")).")
    return (; preview_only, refresh, models)
end

function r68_datastore(refresh::Bool)
    if refresh
        haskey(ENV, "BF_DB_URL") || error(
            "--refresh needs BF_DB_URL to reach the match database. Export it in the shell " *
            "that launches this runner.")
        return DD.load_datastore_cached(DD.ScottishLower(); force = true)
    end
    return DD.load_datastore_cached(DD.ScottishLower(); max_age_hours = 100_000)
end

function r68_extend_joint_player_2627(args = ARGS)
    options = r68_parse_args(args)

    println("="^95)
    println("  SCOTTISH LOWER JOINT-PLAYER MODELS: INCREMENTAL EXTENSION TO 2026/27 SEASON")
    println("="^95)
    println("  mode   : ", options.preview_only ? "PREVIEW ONLY (no sampling)" : "EXTEND (launches MCMC)")
    println("  models : ", join(options.models, ", "))
    println("  data   : ", options.refresh ? "forced refresh from SQL" : "cached DataStore")

    db = TT.PostgresStorage(R68_EXPERIMENT)
    println("  storage: ", db)
    ds = r68_datastore(options.refresh)
    seasons = sort(unique(String.(skipmissing(ds.matches.season))))
    println("  DataStore: $(length(ds.matches.match_id)) matches, seasons $(join(seasons, ", "))")
    if !("26/27" in seasons)
        @warn "The DataStore holds no 26/27 fixtures, so the widened splitter can add no " *
              "folds. Re-run with --refresh (needs BF_DB_URL) once the new season has data."
    end

    splitter_2627 = r68_splitter_2627()

    println("\n--- [1/3] Previewing extension ---")
    plans = Dict{String,Any}()
    for name in options.models
        uuid = R68_MODELS[name]
        println("\n  $name (UUID: $uuid)")
        plans[name] = TT.preview_extension(db, string(uuid), ds; splitter = splitter_2627)
    end

    total_new = sum(p.new_count for p in values(plans); init = 0)
    if options.preview_only
        println("\nPreview complete. $total_new new fold(s) would be sampled across $(length(options.models)) model(s). No MCMC launched.")
        return (; db, ds, plans, extended = Dict{String,Any}())
    end
    if total_new == 0
        println("\nEvery selected run is already current for 26/27. Nothing to sample.")
        return (; db, ds, plans, extended = Dict{String,Any}())
    end

    println("\n--- [2/3] Executing extend_fit (QueuedExecution 16) ---")
    extended = Dict{String,Any}()
    for name in options.models
        uuid = R68_MODELS[name]
        plans[name].new_count == 0 && (println("  $name: up-to-date, skipped."); continue)
        println("\n  $name ($uuid): sampling $(plans[name].new_count) new fold(s)...")
        t0 = time()
        extended[name] = TT.extend_fit(db, string(uuid), ds;
            splitter  = splitter_2627,
            execution = TT.QueuedExecution(16),
        )
        elapsed = time() - t0
        println("  $name: sampling completed in $(round(elapsed / 60, digits = 1)) min! Total folds now $(length(extended[name].folds)).")
    end

    println("\n--- [3/3] Auditing database state & MatchDay compatibility ---")
    @printf("  %-30s | %6s | %9s | %8s | %9s | %5s | %8s\n",
            "Model", "Folds", "OOS fix.", "max R̂", "min ESS", "Div", "MatchDay")
    println("  " * "-"^92)
    for name in options.models
        uuid = R68_MODELS[name]
        verified = TT.load_fit(db, string(uuid))
        
        # Test MatchDay canonical_fit loading and audit
        cf = MD.canonical_fit(db, string(uuid); require_converged = true)
        md_status = cf.converged ? "PASSED" : "FAILED"

        @printf("  %-30s | %6d | %9d | %8.4f | %9.1f | %5d | %8s\n",
                name, length(verified.folds), TT.n_matches(verified.latents),
                verified.diagnostics.max_rhat, verified.diagnostics.min_ess_bulk,
                verified.diagnostics.n_divergent, md_status)
    end
    println("="^95)
    println("  Incremental extension and database persistence verified successfully!")
    println("="^95)

    return (; db, ds, plans, extended)
end

if abspath(PROGRAM_FILE) == @__FILE__
    r68_extend_joint_player_2627()
end
