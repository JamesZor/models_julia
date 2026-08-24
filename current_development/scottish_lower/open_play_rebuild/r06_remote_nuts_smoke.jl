# Stage 7: REMOTE-ONLY real-chain smoke. Launch exactly: julia --project -t16
# Safe for notebook cells and full `include`; it does not rebuild experiment/OOS caches.

# %% 1 — runtime guard, topology, and audited fold-38-style history-only construction
using BayesianFootball, DataFrames, Dates, MCMCChains, Distributions, Statistics, Serialization, Random, LinearAlgebra, Printf, ThreadPinning, SHA
const BFData = BayesianFootball.Data; const BFFeatures = BayesianFootball.Features
const BFPreGame = BayesianFootball.Models.PreGame; const BFPred = BayesianFootball.Predictions
const BFTraining = BayesianFootball.Training; const BFSamplers = BayesianFootball.Samplers
include(joinpath(@__DIR__, "l06_rebuild_sampling.jl")); using .RebuildSampling
include(joinpath(@__DIR__, "l05_rebuild_extraction_recombination.jl")); using .RebuildExtractionRecombination
const RebuildFeatures = RebuildExtractionRecombination.RebuildFeatures
const RebuildEquations = RebuildExtractionRecombination.RebuildEquations
using .RebuildExtractionRecombination.RebuildFeatures
using .RebuildExtractionRecombination.RebuildEquations

Threads.nthreads() == 16 || error("Stage 7 is remote-only and requires julia --project -t16; got $(Threads.nthreads())")
BLAS.set_num_threads(1); BLAS.get_num_threads() == 1 || error("BLAS must be single-threaded")
ThreadPinning.pinthreads(:cores)
println("Stage7 topology: julia_threads=$(Threads.nthreads()) blas_threads=$(BLAS.get_num_threads()) pinning=$(ThreadPinning.threadinfo())")
haskey(ENV, "BF_DB_URL") || error("BF_DB_URL required for read-only registry; it is never persisted")
nsamples = parse(Int, get(ENV, "STAGE7_SAMPLES", "800")); nwarmup = parse(Int, get(ENV, "STAGE7_WARMUP", "800"))
nsamples > 0 && nwarmup > 0 || error("samples and warmup must be positive")
runid = get(ENV, "STAGE7_RUN_ID", Dates.format(now(UTC), "yyyymmddTHHMMSS") * "_" * string(rand(UInt), base=16))
outroot = abspath(get(ENV, "STAGE7_OUTPUT_DIR", joinpath("data", "scottish_open_play_rebuild")))
resume_dir = get(ENV, "STAGE7_RESUME_DIR", "")
outdir = isempty(resume_dir) ? joinpath(outroot, "stage7_" * runid) : abspath(resume_dir)
if isempty(resume_dir)
    ispath(outdir) && error("run directory already exists: $outdir")
    mkpath(outdir)
else
    isdir(outdir) || error("resume directory does not exist: $outdir")
    ispath(joinpath(outdir, "manifest_diagnostics.jls")) && error("resume directory already has diagnostics manifest")
end

ds = BFData.load_datastore_cached(BFData.ScottishLower(), max_age_hours=10_000)
splitter = BFData.GroupedCVConfig(tournament_groups=[[56,57]], target_seasons=["24/25","25/26"], history_seasons=2, dynamics_col=:match_biweek, warmup_period=0, stop_early=true)
boundaries = BFData.create_id_boundaries(ds, splitter)
by_match = Dict(Int(r.match_id)=>Int(r.tournament_id) for r in eachrow(ds.matches))
fold = findlast(i -> Set(get(by_match, Int(id), -1) for id in first(boundaries[i]).history_match_ids) >= Set([56,57]), eachindex(boundaries))
isnothing(fold) && error("no pooled 56/57 history boundary")
boundary = first(boundaries[fold]); registry = fetch_canonical_registry(vcat(Int.(boundary.history_match_ids), Int.(boundary.target_match_ids)))
model = ScottishLowerNPNOGRecombinedPoissonModel(registry); fs = BFFeatures.create_features(boundary, ds, model, :match_biweek); validate_feature_set(fs)
J = Int(fs[:n_teams]); labels = primitive_turing_var_labels(J)

# %% 2 — four independent *single-threaded* project QueuedNUTS tasks and atomic checkpoints
cfg = BFTraining.TrainingConfig(BFSamplers.QueuedNUTSConfig(n_samples=nsamples, n_chains=4, n_warmup=nwarmup, accept_rate=0.8, max_depth=10, initialisation=PriorInit(), show_progress=false), BFTraining.Independent(parallel=true, max_concurrent_tasks=4), nothing, false)
seeds = [parse(Int, get(ENV, "STAGE7_SEED_$c", string(70_600 + c))) for c in 1:4]
elapsed = fill(NaN, 4)
if isempty(resume_dir)
    chains = Vector{Any}(undef, 4); lock = ReentrantLock(); sem = Base.Semaphore(4)
    sampling_wall_start = time()
    @sync for c in 1:4
        Threads.@spawn begin
            Base.acquire(sem); t0=time()
            try
                Random.seed!(seeds[c])
                ch = BFTraining.train(model, cfg, fs; chain_id=c)
                validate_primitive_chain(ch, J); size(ch, 1) == nsamples || error("chain $c retained wrong draw count")
                atomic_serialize(joinpath(outdir, "chain_$(c).jls"), (; chain=ch, chain_id=c, seed=seeds[c], elapsed_seconds=time()-t0))
                Base.lock(lock) do; chains[c]=ch; elapsed[c]=time()-t0; end
            finally
                Base.release(sem)
            end
        end
    end
    sampling_wall_seconds = time() - sampling_wall_start
    chain = cat(chains...; dims=3)
    atomic_serialize(joinpath(outdir, "combined_chain.jls"), chain)
else
    chain = deserialize(joinpath(outdir, "combined_chain.jls"))
    for c in 1:4
        checkpoint = deserialize(joinpath(outdir, "chain_$(c).jls"))
        elapsed[c] = checkpoint.elapsed_seconds
    end
    sampling_wall_seconds = maximum(elapsed)
end
validate_primitive_chain(chain, J)
size(chain, 1) == nsamples && size(chain, 3) == 4 || error("combined chain draw/chain shape mismatch: $(size(chain))")
bundle = BFPreGame.extract_parameters(model, chain, fs); bundle[:draw_count] == 4nsamples || error("primitive draw count mismatch")
all(size(bundle[k]) == (4nsamples, J) for k in (:zA,:zD,:alpha,:beta)) || error("team primitive/derived shape mismatch")
all(all(abs.(vec(sum(bundle[k]; dims=2))) .< 1e-10) for k in (:alpha,:beta,:M)) || error("derived centered sums failed")
diag = diagnostics(chain; max_depth=10)

# %% 3 — persist diagnostic manifest before any OOS promotion
commit = try readchomp(`git rev-parse HEAD`) catch; "unavailable" end
boundary_payload = join(vcat("H:" .* string.(Int.(boundary.history_match_ids)), "T:" .* string.(Int.(boundary.target_match_ids))), "\n")
boundary_sha256 = bytes2hex(sha256(codeunits(boundary_payload)))
manifest = (; stage=7, run_id=runid, created_utc=string(now(UTC)), git_commit=commit, julia_version=string(VERSION), threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(), model="ScottishLowerNPNOGRecombinedPoissonModel", model_version="open_play_rebuild_v1", registry_fingerprint=fs[:registry_fingerprint], boundary=(fold=fold, history_ids=Int.(boundary.history_match_ids), target_ids=Int.(boundary.target_match_ids), sha256=boundary_sha256,), quarantined_ids=fs[:quarantined_match_ids], sampler=sampler_metadata(cfg.sampler), seeds=seeds, chain_elapsed_seconds=elapsed, sampling_wall_seconds=sampling_wall_seconds, diagnostics=diag)
atomic_serialize(joinpath(outdir, "manifest_diagnostics.jls"), manifest)
println("Stage7 diagnostics: rhat=$(diag.max_rhat) bulk=$(diag.min_bulk_ess) tail=$(diag.min_tail_ess) div=$(diag.divergences) finite_lp=$(diag.finite_lp) depth=$(diag.max_tree_depth) cap_fraction=$(diag.depth_cap_fraction) bfmi=$(diag.bfmi)")
!ismissing(diag.depth_cap_fraction) && diag.depth_cap_fraction > 0.01 && @warn "investigate tree-depth cap fraction >1%" fraction=diag.depth_cap_fraction
if diag.bfmi !== missing && any(x -> !ismissing(x) && x < 0.3, diag.bfmi); @warn "investigate BFMI <0.3" bfmi=diag.bfmi; end
hard_smoke_pass(diag) || begin println("FINAL_STAGE7_SUMMARY status=HARD_GATE_FAILED output=$outdir"); error("hard convergence gate failed; chain and diagnostics were saved, OOS promotion blocked") end

# %% 4 — metadata-only OOS extraction and normal Predictions path (only after hard gate)
target = Set(Int.(boundary.target_match_ids)); meta = DataFrame(match_id=Int[], tournament_id=Int[], home_team=String[], away_team=String[], match_date=Date[])
for r in eachrow(ds.matches); Int(r.match_id) in target || continue; push!(meta, (Int(r.match_id),Int(r.tournament_id),String(r.home_team),String(r.away_team),RebuildFeatures._date(r))); end
rbm = Dict(Int(r.match_id)=>r for r in eachrow(registry)); known(r) = begin x=rbm[Int(r.match_id)]; RebuildFeatures._int(x.home_id) in keys(fs[:team_map]) && RebuildFeatures._int(x.away_id) in keys(fs[:team_map]) end
ids = [Int(first(filter(r -> r.tournament_id==t && known(r), eachrow(meta))).match_id) for t in (56,57)]; oos=filter(:match_id=>in(ids),meta)
ek=filter(r -> r.home_team=="east-kilbride" || r.away_team=="east-kilbride", eachrow(meta)); !isempty(ek) && push!(oos,first(ek))
preds=BFPreGame.extract_parameters(model,oos,fs,chain); all(length(x[:lambda_h])==4nsamples && all(isfinite,x[:lambda_h]) && all(isfinite,x[:lambda_a]) for x in values(preds)) || error("invalid OOS posterior draws")
all(x[:home_team_status] in (:history_seen,:target_only_population_fallback,:unknown_identity) && x[:away_team_status] in (:history_seen,:target_only_population_fallback,:unknown_identity) for x in values(preds)) || error("invalid OOS identity status")
!isempty(ek) && !any(x[:home_team_status] == :target_only_population_fallback || x[:away_team_status] == :target_only_population_fallback for (mid,x) in preds if mid == Int(first(ek).match_id)) && error("East Kilbride did not use target-only fallback")
rows=NamedTuple[]; for (mid,x) in preds; push!(rows,(;match_id=mid, league_id=x[:league_id], home_team_status=x[:home_team_status], away_team_status=x[:away_team_status], provenance="stage7_real_chain", lambda_h=x[:lambda_h], lambda_a=x[:lambda_a], lambda_Y_home=x[:lambda_Y_home], lambda_Y_away=x[:lambda_Y_away], lambda_converted_penalty_home=x[:lambda_converted_penalty_home], lambda_converted_penalty_away=x[:lambda_converted_penalty_away], lambda_og_home=x[:lambda_og_home], lambda_og_away=x[:lambda_og_away])); end
latent=generic_dataframe(rows); dataframe_roundtrip_ok(latent) || error("generic persistence DataFrame round-trip failed")
score_shapes = Tuple{Int,Int,Int}[]
for row in eachrow(latent)
    params = BFPred.extract_params(model, row)
    score = BFPred.compute_score_matrix(model, params; max_goals=12)
    S = BFPred.score_matrix_data(score)
    all(isfinite,S) && all(>=(0),S) && all(isapprox(sum(S[:,:,d]),1;atol=1e-10) for d in axes(S,3)) || error("score mass/tail failure for match $(row.match_id)")
    push!(score_shapes, size(S))
end
ppd=BFPred.model_inference(BayesianFootball.Experiments.LatentStates(latent,model);verbose=false); all(length(d)==4nsamples for d in ppd.df.distribution) || error("PPD distribution length failure")
atomic_serialize(joinpath(outdir,"oos_smoke.jls"),(; oos, latent, ppd, score_shapes))
preferred = diag.max_rhat <= 1.01 && diag.min_bulk_ess >= 400 && diag.min_tail_ess >= 400
println("FINAL_STAGE7_SUMMARY status=PASS draws=$(4nsamples) chains=4 rhat=$(diag.max_rhat) min_bulk=$(diag.min_bulk_ess) min_tail=$(diag.min_tail_ess) divergences=$(diag.divergences) preferred=$preferred wall_seconds=$(round(sampling_wall_seconds;digits=1)) output=$outdir")
