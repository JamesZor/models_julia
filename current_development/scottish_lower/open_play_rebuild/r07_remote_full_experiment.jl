# Stage 8: REMOTE-ONLY full pooled-league temporal experiment.
# Launch exactly on beast: julia --project -t16; safe as notebook #%% cells or full include.

# %% 1 — topology, exact split inventory, and one canonical registry snapshot
using BayesianFootball, DataFrames, Dates, MCMCChains, Statistics, Serialization, Random, LinearAlgebra, ThreadPinning, UUIDs
const BFData = BayesianFootball.Data; const BFFeatures = BayesianFootball.Features
const BFPreGame = BayesianFootball.Models.PreGame; const BFPred = BayesianFootball.Predictions
const BFTraining = BayesianFootball.Training; const BFSamplers = BayesianFootball.Samplers
include(joinpath(@__DIR__, "l07_rebuild_full_experiment.jl")); using .RebuildFullExperiment
using .RebuildFullExperiment.RebuildSampling
include(joinpath(@__DIR__, "l05_rebuild_extraction_recombination.jl")); using .RebuildExtractionRecombination
const RebuildFeatures = RebuildExtractionRecombination.RebuildFeatures
using .RebuildExtractionRecombination.RebuildFeatures

Threads.nthreads() == 16 || error("Stage 8 is remote-only: launch julia --project -t16 (got $(Threads.nthreads()))")
BLAS.set_num_threads(1); BLAS.get_num_threads() == 1 || error("Stage 8 requires BLAS=1")
ThreadPinning.pinthreads(:cores)
haskey(ENV, "BF_DB_URL") || error("BF_DB_URL is required for read-only registry access and is never saved")
samples = parse(Int, get(ENV, "STAGE8_SAMPLES", "800")); warmup = parse(Int, get(ENV, "STAGE8_WARMUP", "800"))
expected_folds = parse(Int, get(ENV, "STAGE8_EXPECTED_FOLDS", "38")); fail_fast = get(ENV, "STAGE8_FAIL_FAST", "0") == "1"
dry_run = get(ENV, "STAGE8_DRY_RUN", "0") == "1"
stop_after_fold = parse(Int, get(ENV, "STAGE8_STOP_AFTER_FOLD", "0"))
samples > 0 && warmup > 0 && expected_folds > 0 || error("positive samples, warmup, and expected folds required")
runid = get(ENV, "STAGE8_RUN_ID", Dates.format(now(UTC), "yyyymmddTHHMMSS") * "_" * string(rand(UInt), base=16))
outroot = abspath(get(ENV, "STAGE8_OUTPUT_DIR", joinpath("data", "scottish_open_play_rebuild")))
resume = get(ENV, "STAGE8_RESUME_DIR", ""); outdir = isempty(resume) ? joinpath(outroot, "stage8_" * runid) : abspath(resume)
if isempty(resume); ispath(outdir) && error("run directory exists: $outdir"); mkpath(outdir)
else; isdir(outdir) || error("resume directory does not exist: $outdir"); end

ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
splitter = BFData.GroupedCVConfig(tournament_groups=[[56,57]], target_seasons=["24/25","25/26"], history_seasons=2, dynamics_col=:match_biweek, warmup_period=0, stop_early=true)
boundaries = BFData.create_id_boundaries(ds, splitter)
length(boundaries) == expected_folds || error("Stage 8 fold assertion failed: expected $expected_folds, got $(length(boundaries))")
# `target_match_ids` are cumulative training observations.  Genuine OOS is exactly
# get_next_matches(..., (boundary, meta), splitter), i.e. metadata at t + 1.
oos_folds = true_oos_inventory(ds, boundaries, splitter)
inventory = fold_inventory(oos_folds)
allids = sort!(unique(vcat([inference_ids(x.boundary, x.ids) for x in oos_folds]...)))
registry = fetch_canonical_registry(allids) # sole registry query: all training and true OOS IDs
registry_sha = registry_fingerprint(registry)
println("Stage8 topology threads=$(Threads.nthreads()) blas=$(BLAS.get_num_threads()) pinning=$(ThreadPinning.threadinfo())")
println("Stage8 folds=$(length(boundaries)) unique_registry_matches=$(length(allids)) registry_sha=$registry_sha")
for x in inventory; println("fold=$(x.fold) history=$(x.history_count) target=$(x.target_count) true_oos=$(x.oos_count) target=$(x.target_season)/t$(x.target_time_step) predict_t=$(x.prediction_step) sha=$(x.sha256) oos_sha=$(x.oos_ids_sha256)"); end

# %% 2 — immutable run manifest plus replaceable global progress (no credentials)
commit = try readchomp(`git rev-parse HEAD`) catch; "unavailable" end
run_manifest = immutable_manifest(run_id=runid, git_commit=commit, julia_version=string(VERSION), threads=16, blas_threads=1,
    splitter=(tournament_groups=[[56,57]], target_seasons=["24/25","25/26"], history_seasons=2, dynamics_col=:match_biweek, stop_early=true),
    expected_folds=expected_folds, registry_fingerprint=registry_sha, sampler=(samples=samples,warmup=warmup,chains=4,concurrency=4,init="PriorInit",max_depth=10), folds=inventory)
manifest_path = joinpath(outdir, "run_manifest.jls")
if isfile(manifest_path)
    old = deserialize(manifest_path)
    (old.stage == 8 && old.registry_fingerprint == registry_sha && old.folds == inventory && old.sampler.samples == samples && old.sampler.warmup == warmup) || error("resume manifest is not exact for this Stage 8 invocation")
else
    atomic_serialize(manifest_path, run_manifest)
end
function update_progress!()
    states = Symbol[]; fallback=0; identity_sides=0
    for x in inventory
        d=joinpath(outdir,"fold_$(lpad(x.fold,2,'0'))"); result=joinpath(d,"fold_result.jls")
        state = isfile(result) ? :pass : isfile(joinpath(d,"hard_gate_failure.jls")) ? :hardfail : isfile(joinpath(d,"error.jls")) ? :error : :pending
        push!(states,state)
        if state === :pass
            audit=deserialize(result).fallback_audit; fallback += audit.fallback; identity_sides += audit.total
        end
    end
    summary=(updated_utc=string(now(UTC)), pass=count(==(:pass),states), hardfail=count(==(:hardfail),states), error=count(==(:error),states), pending=count(==(:pending),states), total=length(states), fallback=fallback, identity_sides=identity_sides, fallback_rate=identity_sides == 0 ? missing : fallback/identity_sides)
    atomic_replace_serialize(joinpath(outdir,"progress.jls"), (;summary,states)); summary
end
if dry_run
    println("FINAL_STAGE8_DRY_RUN folds=$(length(boundaries)) unique_registry_matches=$(length(allids)) output=$outdir")
    # Manifest is the only allowed dry-run write; no fold directory, sampling, or OOS artifact is created.
    nothing
else

# %% 3 — sequential folds; each fold runs four prior-initialized chains concurrently and resumes missing valid checkpoints
cfg = BFTraining.TrainingConfig(BFSamplers.QueuedNUTSConfig(n_samples=samples,n_chains=4,n_warmup=warmup,accept_rate=0.8,max_depth=10,initialisation=PriorInit(),show_progress=false), BFTraining.Independent(parallel=true,max_concurrent_tasks=4), nothing, false)
for oos_fold in oos_folds
    fold, boundary, meta, oos_rows, oos_ids = oos_fold.fold, oos_fold.boundary, oos_fold.meta, oos_fold.rows, oos_fold.ids
    fdir=joinpath(outdir,"fold_$(lpad(fold,2,'0'))"); mkpath(fdir)
    isfile(joinpath(fdir,"fold_result.jls")) && (println("Stage8 fold=$fold already PASS"); update_progress!(); continue)
    isfile(joinpath(fdir,"hard_gate_failure.jls")) && (println("Stage8 fold=$fold already HARDFAIL"); update_progress!(); continue)
    try
        # Generic SplitBoundary.target_match_ids are cumulative observations through t.
        # The rebuild contract instead expects fitted IDs in `history` and held-out IDs in
        # `target`, so construct the exact walk-forward boundary explicitly: fit through t,
        # hold out only t+1, and derive decay cutoff from those genuine next fixtures.
        fit_boundary = BFData.SplitBoundary(boundary.fold_id, boundary.target_step,
            boundary_ids(boundary), oos_ids)
        fit_ids = inference_ids(boundary, oos_ids)
        model=ScottishLowerNPNOGRecombinedPoissonModel(registry_subset(registry, fit_ids))
        fs = BFFeatures.create_features(fit_boundary, ds, model, :match_biweek)
        validate_feature_set(fs); J=Int(fs[:n_teams]); bh=boundary_sha256(fit_boundary)
        seeds=[parse(Int,get(ENV,"STAGE8_SEED_$(fold)_$c",string(80_000+100fold+c))) for c in 1:4]
        checkpoints=Vector{Any}(undef,4); chains=Vector{Any}(undef,4); missing=Int[]
        for c in 1:4
            cppath=joinpath(fdir,"chain_$c.jls")
            cp=checkpoint_valid(cppath,bh,c,samples,J,validate_primitive_chain)
            if isnothing(cp)
                # Never overwrite an untrusted partial/stale checkpoint; retain it for audit.
                isfile(cppath) && mv(cppath, cppath * ".invalid-" * string(uuid4()))
                push!(missing,c)
            else
                checkpoints[c]=cp; chains[c]=cp.chain
            end
        end
        t0=time(); chain_lock=ReentrantLock()
        @sync for c in missing
            Threads.@spawn begin
                Random.seed!(seeds[c]); ct=time(); ch=BFTraining.train(model,cfg,fs;chain_id=c)
                validate_primitive_chain(ch,J); size(ch,1)==samples || error("fold $fold chain $c wrong retained draws")
                cp=(;chain=ch,chain_id=c,seed=seeds[c],samples=samples,boundary_sha256=bh,elapsed_seconds=time()-ct)
                atomic_serialize(joinpath(fdir,"chain_$c.jls"),cp)
                Base.lock(chain_lock) do; checkpoints[c]=cp; chains[c]=ch; end
            end
        end
        chain=cat(chains...;dims=3); validate_primitive_chain(chain,J); size(chain)==(samples,size(chain,2),4) || error("fold $fold combined chain shape mismatch")
        combined_path=joinpath(fdir,"combined_chain.jls")
        if isfile(combined_path) && !isempty(missing)
            mv(combined_path, combined_path * ".superseded-" * string(uuid4()))
        end
        isfile(combined_path) || atomic_serialize(combined_path,chain)
        diag=diagnostics(chain;max_depth=10)
        fold_manifest=(;stage=8,fold, boundary_sha256=bh, registry_fingerprint=fs[:registry_fingerprint], frozen_history_ids=Int.(boundary.history_match_ids), cumulative_target_training_ids=Int.(boundary.target_match_ids), fitted_match_ids=Int.(fit_boundary.history_match_ids), true_oos_ids=oos_ids, true_oos_count=length(oos_ids), true_oos_ids_sha256=oos_fold.ids_sha256, cutoff_date=fs[:cutoff_date], target_season=meta.target_season, target_time_step=meta.time_step, prediction_step=oos_fold.prediction_step, sampler=sampler_metadata(cfg.sampler),seeds,diagnostics=diag,chain_elapsed_seconds=[x.elapsed_seconds for x in checkpoints],sampling_wall_seconds=time()-t0)
        diagnostics_path=joinpath(fdir,"diagnostics.jls")
        if isfile(diagnostics_path) && !isempty(missing)
            mv(diagnostics_path, diagnostics_path * ".superseded-" * string(uuid4()))
        end
        isfile(diagnostics_path) || atomic_serialize(diagnostics_path,fold_manifest)
        if !hard_smoke_pass(diag)
            atomic_serialize(joinpath(fdir,"hard_gate_failure.jls"),(;fold_manifest,status=:hard_gate_failed))
            println("Stage8 fold=$fold HARD_GATE_FAILED; OOS blocked"); update_progress!(); continue
        end

        # %% 4 — genuine metadata-only t+1 OOS, generic persistence, score validation, and ordinary inference
        # `oos_rows` came only from get_next_matches above; never use cumulative target_match_ids here.
        preds=BFPreGame.extract_parameters(model, oos_rows, fs, chain)
        rows=NamedTuple[]
        for (mid,x) in preds
            push!(rows,(;match_id=mid,league_id=x[:league_id],home_team_status=x[:home_team_status],away_team_status=x[:away_team_status],provenance="stage8_true_next_step_metadata_only",fold_index=fold,boundary_sha256=bh,target_season=String(meta.target_season),target_time_step=Int(meta.time_step),prediction_step=Int(oos_fold.prediction_step),lambda_h=x[:lambda_h],lambda_a=x[:lambda_a],lambda_Y_home=x[:lambda_Y_home],lambda_Y_away=x[:lambda_Y_away],lambda_converted_penalty_home=x[:lambda_converted_penalty_home],lambda_converted_penalty_away=x[:lambda_converted_penalty_away],lambda_og_home=x[:lambda_og_home],lambda_og_away=x[:lambda_og_away]))
        end
        length(rows)==length(oos_ids) || error("fold $fold missing true OOS extraction")
        latent=generic_dataframe(rows); dataframe_roundtrip_ok(latent) || error("fold $fold generic DataFrame roundtrip failed")
        score_shapes=Tuple{Int,Int,Int}[]
        for row in eachrow(latent)
            S=BFPred.score_matrix_data(BFPred.compute_score_matrix(model,BFPred.extract_params(model,row);max_goals=12)) # adaptive tail is model-owned
            all(isfinite,S) && all(>=(0),S) && all(isapprox(sum(S[:,:,d]),1;atol=1e-10) for d in axes(S,3)) || error("fold $fold score mass failure match $(row.match_id)")
            push!(score_shapes,size(S))
        end
        ppd=BFPred.model_inference(BayesianFootball.Experiments.LatentStates(latent,model);verbose=false)
        all(length(d)==4samples for d in ppd.df.distribution) || error("fold $fold PPD draw count failure")
        statuses=vcat(Symbol.(latent.home_team_status),Symbol.(latent.away_team_status)); fallback=count(==(:target_only_population_fallback),statuses)
        atomic_serialize(joinpath(fdir,"fold_result.jls"),(;fold_manifest,status=:pass,latent,ppd,score_shapes,fallback_audit=(fallback=fallback,total=length(statuses),rate=fallback/length(statuses))))
        println("Stage8 fold=$fold PASS fallback_rate=$(fallback/length(statuses))")
    catch err
        bt=catch_backtrace(); io=IOBuffer(); Base.show_backtrace(io,bt)
        artifact=(;stage=8,fold,status=:unexpected_error,error=sanitized_error(err),backtrace=sanitized_error(String(take!(io))))
        !isfile(joinpath(fdir,"error.jls")) && atomic_serialize(joinpath(fdir,"error.jls"),artifact)
        println("Stage8 fold=$fold ERROR $(artifact.error)"); update_progress!(); fail_fast && rethrow()
    end
    update_progress!()
    stop_after_fold > 0 && fold >= stop_after_fold && begin
        println("Stage8 intentional stop after fold=$fold; resume from $outdir")
        break
    end
end
summary=update_progress!(); println("FINAL_STAGE8_SUMMARY pass=$(summary.pass) hardfail=$(summary.hardfail) error=$(summary.error) pending=$(summary.pending) fallback_rate=$(summary.fallback_rate) output=$outdir")
end
