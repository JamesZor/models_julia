#=
r08_cv_compare.jl  —  Repeated k-fold model comparison with PAIRED significance.

Compares in-play model specs by mean ± SE across folds AND paired differences (same folds → a t-stat),
so we can tell a real improvement from sampling noise. (See heavytail_diagnosis_report.md for why a
single split is untrustworthy on ~253 matches.)

Validated finding (Ireland, 5-fold × 4 repeats = 20 folds):
  - game-state and momentum BOTH genuinely improve the goal-COUNT held-out log-likelihood
    (paired t = 3.15 and 2.15) ...
  - ... but NEITHER improves Over/Under MARKET calibration (ECE differences are noise / trivially worse).
  - mean_bias ≈ 0 ± 0.03 for all specs ⇒ the model is unbiased; the old single-split "OU under-prediction"
    was a goal-heavy test draw.

Run with threads:  julia --project -t 16  (pinthreads(:cores))
=#

using Revise, BayesianFootball
using DataFrames, GLM, Distributions, Statistics, Random
using ThreadPinning; pinthreads(:cores)

const Data        = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments
const Features    = BayesianFootball.Features

include("l01_inplay_inverse.jl")
include("l06_momentum_feature.jl")
include("l07_cv_harness.jl")

# ---- data + panel + momentum ----
ds = Data.load_datastore_cached(Data.Ireland()); bf = ds.betfair_odds
pg = Experiments.extract_oos_predictions(ds, Experiments.load_experiment(
        Experiments.list_experiments("./data/dixon_coles_ab/", data_dir=""), 1))
pg_tbl = DataFrame(match_id=Int.(pg.df.match_id),
                   pg_λ_h=[mean(Float64.(v)) for v in pg.df.λ_h], pg_λ_a=[mean(Float64.(v)) for v in pg.df.λ_a])
function build_panel(bf, ds, pg_tbl; bin_minutes=5.0, staleness=10.0, min_sel=6, mtk_max=130.0)
    ids = unique(subset(bf, :minutes_to_kickoff=>ByRow(x->0<x<=mtk_max)).match_id)
    parts = Vector{DataFrame}(undef, length(ids))
    Threads.@threads for k in eachindex(ids)
        local tr; try; tr=inplay_lambda_trace(bf,ds,Int(ids[k]); bin_minutes=bin_minutes,staleness=staleness,min_sel=min_sel,mtk_max=mtk_max)
        catch; tr=DataFrame(); end; parts[k]=tr
    end
    leftjoin(vcat([d for d in parts if nrow(d)>0]...), pg_tbl, on=:match_id)
end
panel = build_panel(bf, ds, pg_tbl)
fin = Dict(Int(r.match_id)=>(Int(r.home_score),Int(r.away_score)) for r in eachrow(ds.matches) if !ismissing(r.home_score))
mom_lookup = build_momentum_lookup(Data.tournament_ids(Data.Ireland()))
BINS = build_bins(panel, fin, mom_lookup)

# ---- specs to compare (add your own here) ----
specs = ["pregame_only" => @formula(rem_goals ~ is_home + log_pregame),
         "+game_state"  => @formula(rem_goals ~ t_m + t_m2 + is_home + trailing + leading + man_adv + log_pregame),
         "+momentum"    => @formula(rem_goals ~ t_m + t_m2 + is_home + trailing + leading + man_adv + log_pregame + momentum)]

folds = kfold_repeats(unique(BINS.match_id); k = 5, repeats = 4)
R = run_cv(specs, BINS, folds)

println("\n[CV summary — mean ± SE over $(length(folds)) folds]")
show(summarise_cv(R), allrows = true, truncate = 0); println()

println("\n[PAIRED differences on shared folds (t = mean/SE; |t|>2 ≈ real)]")
for (m, lbl) in ((:count_ll, "count logLL (higher better)"), (:ECE, "OU ECE (lower better)"))
    println("  $lbl:")
    println("    +game_state − pregame_only : ", paired_diff(R, m, "+game_state", "pregame_only"))
    println("    +momentum   − +game_state  : ", paired_diff(R, m, "+momentum", "+game_state"))
end
