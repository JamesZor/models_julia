# Stage 8: REMOTE-ONLY native flattened fold × chain queue.
# Launch on beast only: julia --project -t16; this file intentionally does no local MCMC.

using BayesianFootball, DataFrames, Dates, MCMCChains, Statistics, Serialization, Random, LinearAlgebra, ThreadPinning, UUIDs
const BFData = BayesianFootball.Data; const BFFeatures = BayesianFootball.Features
const BFPreGame = BayesianFootball.Models.PreGame; const BFPred = BayesianFootball.Predictions
const BFTraining = BayesianFootball.Training; const BFSamplers = BayesianFootball.Samplers
include(joinpath(@__DIR__, "l07_rebuild_full_experiment.jl")); using .RebuildFullExperiment
using .RebuildFullExperiment.RebuildSampling
include(joinpath(@__DIR__, "l05_rebuild_extraction_recombination.jl")); using .RebuildExtractionRecombination
const RebuildFeatures = RebuildExtractionRecombination.RebuildFeatures
const RebuildEquations = RebuildExtractionRecombination.RebuildEquations
using .RebuildExtractionRecombination.RebuildFeatures
using .RebuildExtractionRecombination.RebuildEquations

Threads.nthreads() == 16 || error("Stage 8 is remote-only: launch julia --project -t16 (got $(Threads.nthreads()))")
BLAS.set_num_threads(1); BLAS.get_num_threads() == 1 || error("Stage 8 requires BLAS=1")
ThreadPinning.pinthreads(:cores)
haskey(ENV, "BF_DB_URL") || error("BF_DB_URL is required for read-only registry access and is never saved")
samples = parse(Int, get(ENV, "STAGE8_SAMPLES", "800")); warmup = parse(Int, get(ENV, "STAGE8_WARMUP", "800"))
expected_folds = parse(Int, get(ENV, "STAGE8_EXPECTED_FOLDS", "38")); dry_run = get(ENV, "STAGE8_DRY_RUN", "0") == "1"
prepare_only = get(ENV, "STAGE8_PREPARE_ONLY", "0") == "1"
max_tasks = parse(Int, get(ENV, "STAGE8_MAX_CONCURRENT_TASKS", "16"))
queue_seed = parse(Int, get(ENV, "STAGE8_SEED", "80808"))
samples > 0 && warmup > 0 && expected_folds > 0 && max_tasks > 0 || error("positive Stage 8 settings required")
max_tasks <= Threads.nthreads() || error("STAGE8_MAX_CONCURRENT_TASKS=$max_tasks exceeds Threads.nthreads()=$(Threads.nthreads())")
runid = get(ENV, "STAGE8_RUN_ID", Dates.format(now(UTC), "yyyymmddTHHMMSS") * "_" * string(rand(UInt), base=16))
outroot = abspath(get(ENV, "STAGE8_OUTPUT_DIR", joinpath("data", "scottish_open_play_rebuild")))
resume = get(ENV, "STAGE8_RESUME_DIR", ""); outdir = isempty(resume) ? joinpath(outroot, "stage8_" * runid) : abspath(resume)
if isempty(resume); ispath(outdir) && error("run directory exists: $outdir"); mkpath(outdir)
else; isdir(outdir) || error("resume directory does not exist: $outdir"); end

# Inventory and the one global immutable canonical registry.
ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
splitter = BFData.GroupedCVConfig(tournament_groups=[[56,57]], target_seasons=["24/25","25/26"], history_seasons=2, dynamics_col=:match_biweek, warmup_period=0, stop_early=true)
boundaries = BFData.create_id_boundaries(ds, splitter)
length(boundaries) == expected_folds || error("expected $expected_folds folds, got $(length(boundaries))")
oos_folds = true_oos_inventory(ds, boundaries, splitter); inventory = fold_inventory(oos_folds)
allids = sort!(unique(vcat([inference_ids(x.boundary, x.ids) for x in oos_folds]...)))
registry = fetch_canonical_registry(allids); registry_sha = registry_fingerprint(registry)
# This model, not a per-fold subset model, is used for every FeatureSet and extraction.
global_model = ScottishLowerNPNOGRecombinedPoissonModel(registry)
registry_fingerprint(global_model.registry) == registry_sha || error("global model registry changed at construction")
println("Stage8 native queue: folds=$(length(boundaries)), tasks=$(4length(boundaries)), concurrency=$max_tasks, registry=$registry_sha")

commit = try readchomp(`git rev-parse HEAD`) catch; "unavailable" end
run_manifest = immutable_manifest(run_id=runid, git_commit=commit, julia_version=string(VERSION), threads=Threads.nthreads(), blas_threads=1,
    splitter=(tournament_groups=[[56,57]],target_seasons=["24/25","25/26"],history_seasons=2,dynamics_col=:match_biweek,stop_early=true),
    expected_folds=expected_folds, registry_fingerprint=registry_sha,
    sampler=(samples=samples,warmup=warmup,chains=4,concurrency=max_tasks,init="PriorInit",max_depth=10,seed=queue_seed), folds=inventory)
manifest_path = joinpath(outdir, "run_manifest.jls")
if isfile(manifest_path)
    old=deserialize(manifest_path)
    same_sampler = old.sampler.samples == samples && old.sampler.warmup == warmup &&
        old.sampler.chains == 4 && old.sampler.init == "PriorInit" && old.sampler.max_depth == 10
    (old.stage == 8 && old.registry_fingerprint == registry_sha && old.folds == inventory && same_sampler) || error("resume manifest is not model/split/sampler exact")
else
    atomic_serialize(manifest_path, run_manifest)
end
queue_manifest_path = joinpath(outdir, "native_queue_manifest.jls")
if isfile(queue_manifest_path)
    queued_manifest = deserialize(queue_manifest_path)
    if get(queued_manifest, :git_commit, nothing) != commit || get(queued_manifest, :max_concurrent_tasks, nothing) != max_tasks
        mv(queue_manifest_path, queue_manifest_path * ".superseded-" * string(uuid4()))
    end
end
isfile(queue_manifest_path) || atomic_serialize(queue_manifest_path,
    (; stage=8, git_commit=commit, created_utc=string(now(UTC)), max_concurrent_tasks=max_tasks,
       queue_seed, source_run_manifest=basename(manifest_path), registry_fingerprint=registry_sha))

function stage8_progress!()
    states=Symbol[]
    for x in inventory
        d=joinpath(outdir,"fold_$(lpad(x.fold,2,'0'))")
        push!(states, isfile(joinpath(d,"fold_result.jls")) ? :pass : isfile(joinpath(d,"hard_gate_failure.jls")) ? :hardfail : isfile(joinpath(d,"error.jls")) ? :error : :pending)
    end
    summary=(updated_utc=string(now(UTC)),pass=count(==(:pass),states),hardfail=count(==(:hardfail),states),error=count(==(:error),states),pending=count(==(:pending),states),total=length(states))
    atomic_replace_serialize(joinpath(outdir,"progress.jls"), (;summary,states)); summary
end
if dry_run
    println("FINAL_STAGE8_DRY_RUN folds=$(length(boundaries)) registry=$registry_sha output=$outdir")
else
    checkpoint_dir=joinpath(outdir,"queued_checkpoints")
    cfg=BFTraining.TrainingConfig(BFSamplers.QueuedNUTSConfig(n_samples=samples,n_chains=4,n_warmup=warmup,accept_rate=0.8,max_depth=10,initialisation=PriorInit(),show_progress=false), BFTraining.Independent(parallel=true,max_concurrent_tasks=max_tasks),checkpoint_dir,false)

    # Homogeneous metadata is required by FeatureCollection and is also the checkpoint
    # provenance wrapper.  Native split checkpoints are exactly `(chain, metadata)`.
    match_dates = Dict(Int(r.match_id) => RebuildFeatures._date(r) for r in eachrow(ds.matches))
    function build_fold_context(x)
        cutoff = minimum(RebuildFeatures._date(r) for r in eachrow(x.rows))
        raw_fit_ids = boundary_ids(x.boundary)
        # `match_biweek` labels can overlap in actual kickoff time because of postponements.
        # Enforce the design contract by fitting only observations strictly before the
        # earliest next-step fixture; later-labelled-but-not-yet-played matches are excluded.
        fit_ids = sort!(Int[id for id in raw_fit_ids if match_dates[id] < cutoff])
        excluded_fit_ids = sort!(collect(setdiff(Set(raw_fit_ids), Set(fit_ids))))
        fit_boundary=BFData.SplitBoundary(x.boundary.fold_id,x.boundary.target_step,fit_ids,x.ids)
        fs=BFFeatures.create_features(fit_boundary,ds,global_model,:match_biweek)
        validate_feature_set(fs); fs[:model_registry_fingerprint] == registry_sha || error("fold $(x.fold) lost global registry identity")
        bh=boundary_sha256(fit_boundary)
        provenance=(true_oos_ids=Int.(x.ids),true_oos_ids_sha256=x.ids_sha256,prediction_step=Int(x.prediction_step),target_season=x.meta.target_season,target_time_step=Int(x.meta.time_step),cutoff_date=cutoff,excluded_not_yet_played_ids=excluded_fit_ids)
        meta=(original_meta=x.meta,fold_index=Int(x.fold),boundary_sha256=bh,oos_provenance=provenance)
        return (x=x,fit_boundary=fit_boundary,fs=fs,meta=meta,boundary_sha256=bh,excluded_fit_ids=excluded_fit_ids)

    end
    fold_context=build_fold_context.(oos_folds)
    feature_items=[(ctx.fs,ctx.meta) for ctx in fold_context]
    length(unique(typeof(ctx.meta) for ctx in fold_context)) == 1 || error("FeatureCollection metadata must be homogeneous")
    collection=BFFeatures.FeatureCollection(feature_items)

    # Reject stale/corrupt native checkpoints before Training sees them.  A valid old
    # custom combined chain is imported once into the project's atomic split checkpoint.
    function checkpoint_metadata_matches(stored, expected)
        stored isa NamedTuple || return false
        return get(stored,:fold_index,nothing) == expected.fold_index &&
            get(stored,:boundary_sha256,nothing) == expected.boundary_sha256 &&
            get(stored,:oos_provenance,nothing) == expected.oos_provenance
        # Do not compare `original_meta` by object identity: project metadata structs do
        # not define structural `==`, so a valid deserialized checkpoint would fail.
    end
    function valid_native_checkpoint(path, ctx)
        isfile(path) || return nothing
        z=try deserialize(path) catch; return nothing end
        (z isa Tuple && length(z)==2 && checkpoint_metadata_matches(z[2],ctx.meta) && z[1] isa Chains) || return nothing
        try
            validate_primitive_chain(z[1],Int(ctx.fs[:n_teams]))
            # `names(chain)` defaults to the parameter section and deliberately omits
            # sampler internals, so validate only retained iterations/chains here; the
            # exact primitive parameter manifest is already enforced above.
            size(z[1],1) == samples && size(z[1],3) == 4 || return nothing
            z
        catch; nothing end
    end
    for (i,ctx) in enumerate(fold_context)
        native=joinpath(checkpoint_dir,"split_$(lpad(i,3,'0')).jls")
        if isfile(native) && isnothing(valid_native_checkpoint(native,ctx))
            mv(native,native*".invalid-"*string(uuid4()))
        end
        # Recover audit-preserved checkpoints that an older identity-based metadata
        # validator incorrectly renamed. Restore only one uniquely valid candidate.
        if !isfile(native)
            candidates = filter(p -> startswith(basename(p), basename(native) * ".invalid-"), readdir(checkpoint_dir; join=true))
            candidates = filter(p -> !isnothing(valid_native_checkpoint(p,ctx)), candidates)
            length(candidates) == 1 && begin
                mv(only(candidates),native)
                println("Stage8 recovered native checkpoint for fold $(ctx.x.fold)")
            end
            length(candidates) > 1 && error("multiple valid recovery checkpoints for fold $(ctx.x.fold)")
        end
        isfile(native) && continue
        # Migration source is the pre-native runner's immutable combined artifact.
        fdir=joinpath(outdir,"fold_$(lpad(ctx.x.fold,2,'0'))"); oldchain=joinpath(fdir,"combined_chain.jls"); olddiag=joinpath(fdir,"diagnostics.jls")
        if isfile(oldchain) && isfile(olddiag)
            oldmeta=try deserialize(olddiag) catch; nothing end
            ch=try deserialize(oldchain) catch; nothing end
            exact = oldmeta isa NamedTuple && get(oldmeta,:boundary_sha256,nothing)==ctx.boundary_sha256 && get(oldmeta,:registry_fingerprint,nothing)==ctx.fs[:registry_fingerprint]
            if exact && ch isa Chains
                try
                    validate_primitive_chain(ch,Int(ctx.fs[:n_teams])); size(ch,1)==samples && size(ch,3)==4 || error("shape")
                    BFTraining.save_split_checkpoint(checkpoint_dir,i,(ch,ctx.meta)); println("Stage8 migrated fold $(ctx.x.fold) into native split checkpoint")
                catch err
                    @warn "Stage8 did not migrate invalid custom chain for fold $(ctx.x.fold)" exception=(err,catch_backtrace())
                end
            end
        end
    end

    # One native call: Independent/_train_queued flattens all 38 folds × 4 chains and
    # dynamically fills at most `max_tasks` single-thread chain tasks.
    if prepare_only
        migrated = count(i -> isfile(joinpath(checkpoint_dir,"split_$(lpad(i,3,'0')).jls")), eachindex(fold_context))
        println("FINAL_STAGE8_PREPARE_ONLY folds=$(length(collection)) valid_checkpoints=$migrated queue_tasks=$(4length(collection)) concurrency=$max_tasks output=$outdir")
    else
    Random.seed!(queue_seed)
    queued=BFTraining.train(global_model,cfg,collection)
    for (i,ctx) in enumerate(fold_context)
        fdir=joinpath(outdir,"fold_$(lpad(ctx.x.fold,2,'0'))"); mkpath(fdir)
        isfile(joinpath(fdir,"fold_result.jls")) && continue
        # TrainingResults intentionally omits failed entries in this project, so the
        # atomic split checkpoint is the indexed source of truth after the queue.
        result=valid_native_checkpoint(joinpath(checkpoint_dir,"split_$(lpad(i,3,'0')).jls"),ctx)
        if isnothing(result)
            println("Stage8 fold=$(ctx.x.fold) INCOMPLETE: native queue produced no valid completed split; resume uses $checkpoint_dir")
            continue
        end
        chain, stored_meta=result
        try
            checkpoint_metadata_matches(stored_meta,ctx.meta) || error("native checkpoint fold metadata mismatch")
            validate_primitive_chain(chain,Int(ctx.fs[:n_teams])); size(chain,1)==samples && size(chain,3)==4 || error("combined chain shape mismatch")
            diag=diagnostics(chain;max_depth=10)
            fold_manifest=(;stage=8,fold=ctx.x.fold,boundary_sha256=ctx.boundary_sha256,registry_fingerprint=ctx.fs[:registry_fingerprint],model_registry_fingerprint=registry_sha,fitted_match_ids=Int.(ctx.fit_boundary.history_match_ids),excluded_not_yet_played_ids=ctx.excluded_fit_ids,true_oos_ids=Int.(ctx.x.ids),true_oos_ids_sha256=ctx.x.ids_sha256,sampler=sampler_metadata(cfg.sampler),diagnostics=diag,native_checkpoint=joinpath("queued_checkpoints","split_$(lpad(i,3,'0')).jls"))
            isfile(joinpath(fdir,"diagnostics.jls")) || atomic_serialize(joinpath(fdir,"diagnostics.jls"),fold_manifest)
            if !hard_smoke_pass(diag)
                isfile(joinpath(fdir,"hard_gate_failure.jls")) || atomic_serialize(joinpath(fdir,"hard_gate_failure.jls"),(;fold_manifest,status=:hard_gate_failed))
                println("Stage8 fold=$(ctx.x.fold) HARD_GATE_FAILED; OOS blocked"); continue
            end
            preds=BFPreGame.extract_parameters(global_model,ctx.x.rows,ctx.fs,chain)
            rows=[(;match_id=mid,league_id=x[:league_id],home_team_status=x[:home_team_status],away_team_status=x[:away_team_status],provenance="stage8_true_next_step_metadata_only",fold_index=ctx.x.fold,boundary_sha256=ctx.boundary_sha256,lambda_h=x[:lambda_h],lambda_a=x[:lambda_a],lambda_Y_home=x[:lambda_Y_home],lambda_Y_away=x[:lambda_Y_away],lambda_converted_penalty_home=x[:lambda_converted_penalty_home],lambda_converted_penalty_away=x[:lambda_converted_penalty_away],lambda_og_home=x[:lambda_og_home],lambda_og_away=x[:lambda_og_away]) for (mid,x) in preds]
            length(rows)==length(ctx.x.ids) || error("missing true OOS extraction")
            latent=generic_dataframe(rows); dataframe_roundtrip_ok(latent) || error("generic DataFrame roundtrip failed")
            for row in eachrow(latent)
                S=BFPred.score_matrix_data(BFPred.compute_score_matrix(global_model,BFPred.extract_params(global_model,row);max_goals=12))
                all(isfinite,S) && all(>=(0),S) && all(isapprox(sum(S[:,:,d]),1;atol=1e-10) for d in axes(S,3)) || error("score mass failure")
            end
            ppd=BFPred.model_inference(BayesianFootball.Experiments.LatentStates(latent,global_model);verbose=false)
            all(length(d)==4samples for d in ppd.df.distribution) || error("PPD draw count failure")
            statuses=vcat(Symbol.(latent.home_team_status),Symbol.(latent.away_team_status)); fallback=count(==(:target_only_population_fallback),statuses)
            atomic_serialize(joinpath(fdir,"fold_result.jls"),(;fold_manifest,status=:pass,latent,ppd,fallback_audit=(fallback=fallback,total=length(statuses),rate=fallback/length(statuses))))
            println("Stage8 fold=$(ctx.x.fold) PASS")
        catch err
            io=IOBuffer(); Base.show_backtrace(io,catch_backtrace())
            artifact=(;stage=8,fold=ctx.x.fold,status=:post_queue_error,error=sanitized_error(err),backtrace=sanitized_error(String(take!(io))))
            isfile(joinpath(fdir,"error.jls")) || atomic_serialize(joinpath(fdir,"error.jls"),artifact)
            println("Stage8 fold=$(ctx.x.fold) ERROR $(artifact.error)")
        end
    end
    summary=stage8_progress!(); println("FINAL_STAGE8_SUMMARY pass=$(summary.pass) hardfail=$(summary.hardfail) error=$(summary.error) pending=$(summary.pending) output=$outdir")
    end # prepare_only
end
