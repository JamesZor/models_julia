#=
r08 — VERIFICATION of the two-layer funnel graduation into src/.

Checks that `DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel` (src) is a faithful, correctly
wired replacement for the r06 winner
`TeamFunnelFlexDPGoalsModel(cascade_weight=0, sot_on=false, p2_prior=Normal(logit(0.145), 0.5))`
(current_development/bbc_xg_proxy/l05_funnel_flex.jl), and that the new `ds.bbc` DataStore domain
behaves on both a BBC-covered and a BBC-less segment.

GATES (fast, ~2 min) — always run:
  G1  DataStore: ScottishLower has ds.bbc with 1968 rows; a no-BBC segment has an empty ds.bbc
      and still builds features for its own engines.
  G2  Feature: ShotsFunnelFeature emits Int counts with 0 dummies + a 0/1 Float64 mask, aligned
      to ordered_ids, and agrees value-for-value with the prototype's BBCFunnelFeature.
  G3  Likelihood equivalence: logjoint(new) − logjoint(old, minus its p₁ prior term) is CONSTANT
      across random parameter draws. That constant is the dropped log(y!) normaliser — the proof
      the sufficient-statistic reduction is exact.
  G4  Prediction dispatch: compute_score_matrix takes the Poisson path (no `:r`/NegBin error).

TRAIN SMOKE (~30-60 min) — opt in with ENV["FUNNEL_VERIFY_TRAIN"] = "1":
  G5  A one-season fold set trains under the default UniformInit; R-hat, p₂ ≈ 0.145, λ_s ≈ 10.
  G6  End-to-end pricing + evaluation runs with no dispatch error.

NOTE ON CACHES: adding `ds.bbc` invalidated every .cache/datastore_*.jls. This script always loads
with force=true.

Run on the server after `git pull`:
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    include(joinpath(pkgdir(BayesianFootball), "current_development/bbc_xg_proxy/r08_graduation_verify.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using Dates
using MCMCChains
using DynamicPPL
using StatsFuns: logit

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Data        = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Evaluation  = BayesianFootball.Evaluation

const ROOT = pkgdir(BayesianFootball)
_r(x, d=4) = round(x, digits=d)

const RESULTS = String[]
function gate(name::String, ok::Bool, detail::String = "")
    line = "$(ok ? "PASS" : "FAIL")  $(rpad(name, 34))  $detail"
    println(line); push!(RESULTS, line)
    return ok
end

println("\n", "="^78, "\n r08 — funnel graduation verification\n", "="^78)

# ==========================================================================================
# G1. DataStore: ds.bbc on a covered and an uncovered segment
# ==========================================================================================
println("\n[G1] DataStore...")

ds = Data.load_datastore_cached(Data.ScottishLower(); force = true)
bbc = ds.bbc
gate("G1a ScottishLower ds.bbc rows", nrow(bbc) == 1968, "nrow = $(nrow(bbc)) (expected 1968)")
gate("G1b ds.bbc columns",
     all(c -> c in names(bbc), ["match_id", "tournament_id", "shots_h", "shots_a", "sot_h", "sot_a"]),
     "$(names(bbc))")
gate("G1c ds.bbc shots non-missing",
     nrow(bbc) > 0 && count(ismissing, bbc.shots_h) == 0 && count(ismissing, bbc.shots_a) == 0,
     "missing h/a = $(count(ismissing, bbc.shots_h))/$(count(ismissing, bbc.shots_a))")
gate("G1d ds.bbc match_id unique", nrow(bbc) == length(unique(bbc.match_id)),
     "unique = $(length(unique(bbc.match_id)))")

# a segment with no BBC coverage must degrade to an empty frame, not error
ds_nobbc = Data.load_datastore_cached(Data.Ireland(); force = true)
gate("G1e no-BBC segment empty ds.bbc", nrow(ds_nobbc.bbc) == 0, "nrow = $(nrow(ds_nobbc.bbc))")
gate("G1f no-BBC segment still loads", nrow(ds_nobbc.matches) > 0,
     "matches = $(nrow(ds_nobbc.matches)), odds = $(nrow(ds_nobbc.odds))")

# ==========================================================================================
# G2. Feature: ShotsFunnelFeature vs the prototype's BBCFunnelFeature
# ==========================================================================================
println("\n[G2] Feature extraction...")

# the prototype loader (defines TeamFunnelFlexDPGoalsModel, BBCFunnelFeature, its own .jls cache)
include(joinpath(ROOT, "current_development/bbc_xg_proxy/l05_funnel_flex.jl"))

dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = 365.0)
m_new = PreGame.DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel(dynamics_config = dyn_cfg)
m_old = TeamFunnelFlexDPGoalsModel(dynamics_config = dyn_cfg, cascade_weight = 0.0,
                                   sot_on = false, p2_prior = Normal(logit(0.145), 0.5))

# one fold, cheap: build the standard splitter config and take its first boundary (no training)
cv_config = Data.GroupedCVConfig(
    tournament_groups = [Data.tournament_ids(ds.segment)],
    target_seasons    = ["25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = false)
boundary, _meta = first(Data.create_id_boundaries(ds, cv_config))
fs_new = Features.create_features(boundary, ds, m_new, :match_biweek)
fs_old = Features.create_features(boundary, ds, m_old, :match_biweek)

d_new, d_old = fs_new.data, fs_old.data
n_obs = length(d_new[:flat_home_ids])

gate("G2a counts are Int",
     d_new[:flat_home_shots_n] isa Vector{Int} && d_new[:flat_away_shots_n] isa Vector{Int},
     "$(eltype(d_new[:flat_home_shots_n]))")
gate("G2b mask is 0/1 Float64",
     d_new[:flat_funnel_mask_h] isa Vector{Float64} &&
     all(x -> x == 0.0 || x == 1.0, d_new[:flat_funnel_mask_h]) &&
     all(x -> x == 0.0 || x == 1.0, d_new[:flat_funnel_mask_a]),
     "coverage h = $(_r(mean(d_new[:flat_funnel_mask_h]), 3))")
gate("G2c lengths aligned",
     length(d_new[:flat_home_shots_n]) == n_obs && length(d_new[:flat_funnel_mask_h]) == n_obs,
     "n_obs = $n_obs")
gate("G2d no dummy leakage (masked ⇒ count 0)",
     all(i -> d_new[:flat_funnel_mask_h][i] == 1.0 || d_new[:flat_home_shots_n][i] == 0, 1:n_obs) &&
     all(i -> d_new[:flat_funnel_mask_a][i] == 1.0 || d_new[:flat_away_shots_n][i] == 0, 1:n_obs))
gate("G2e matches prototype extractor",
     d_new[:flat_home_shots_n] == d_old[:flat_home_shots_n] &&
     d_new[:flat_away_shots_n] == d_old[:flat_away_shots_n] &&
     d_new[:flat_funnel_mask_h] == d_old[:flat_funnel_mask_h] &&
     d_new[:flat_funnel_mask_a] == d_old[:flat_funnel_mask_a],
     "mean shots h = $(_r(mean(d_new[:flat_home_shots_n]), 2)) (expect ≈ 10-11)")

# the no-BBC path must produce an all-zero mask rather than throwing
F_empty = Dict{Symbol, Any}()
Features.add_feature!(F_empty, Features.ShotsFunnelFeature(),
                      Int.(ds_nobbc.matches.match_id[1:min(50, nrow(ds_nobbc.matches))]),
                      Dict{String, Int}(), ds_nobbc)
gate("G2f no-BBC segment ⇒ zero mask",
     all(iszero, F_empty[:flat_funnel_mask_h]) && all(iszero, F_empty[:flat_home_shots_n]))

# ==========================================================================================
# G3. Likelihood equivalence: new (src) vs old (l05, cw=0, sot_off)
# ==========================================================================================
# The src engine drops the log(y!) normalisers that `logpdf(Poisson(λ), y)` carries, and the old
# engine additionally samples an inert `p1_raw` (its likelihood contribution is gated to zero by
# sot_on=false, but its PRIOR still lands in the logjoint). Subtracting that prior term, the gap
# between the two logjoints must be a CONSTANT across parameter draws.
println("\n[G3] Likelihood equivalence...")

t_new = PreGame.build_turing_model(m_new, fs_new)
t_old = PreGame.build_turing_model(m_old, fs_old)

gaps = Float64[]
try
    for _ in 1:8
        vi = DynamicPPL.VarInfo(t_old)                       # fresh prior draw, contains p1_raw
        p1_val = vi[@varname(p1_raw)]
        lj_old = DynamicPPL.logjoint(t_old, vi) - logpdf(m_old.p1_prior, p1_val)
        lj_new = DynamicPPL.logjoint(t_new, vi)              # extra p1_raw in vi is ignored
        push!(gaps, lj_new - lj_old)
    end
    spread = maximum(gaps) - minimum(gaps)
    gate("G3 logjoint gap constant", spread < 1e-6,
         "gap = $(_r(mean(gaps), 3)), spread = $(_r(spread, 10)) (constant = dropped log y!)")
catch e
    gate("G3 logjoint gap constant", false, "evaluation failed: $(sprint(showerror, e))")
    @warn "G3 could not run — DynamicPPL API mismatch. Fall back to comparing short-NUTS " *
          "posteriors for p₂ and λ_s between the two engines."
end

# ==========================================================================================
# G4. Prediction dispatch — the AbstractNegBinModel trap
# ==========================================================================================
println("\n[G4] Prediction dispatch...")

fake_row = (λ_h = fill(1.4, 20), λ_a = fill(1.1, 20))
try
    params = Predictions.extract_params(m_new, fake_row)
    S = Predictions.compute_score_matrix(m_new, params; max_goals = 8)
    tot = sum(S.data[:, :, 1])
    gate("G4 Poisson score grid", isapprox(tot, 1.0; atol = 5e-3),
         "grid mass = $(_r(tot, 5)) (truncation at max_goals=8)")
catch e
    gate("G4 Poisson score grid", false, "dispatch error: $(sprint(showerror, e))")
end

# ==========================================================================================
# G5/G6. Train smoke + end-to-end eval (opt in)
# ==========================================================================================
if get(ENV, "FUNNEL_VERIFY_TRAIN", "0") == "1"
    println("\n[G5] Train smoke (this takes a while)...")
    save_dir = joinpath(ROOT, "data/funnel_graduation/")
    mkpath(save_dir)

    task = Experiments.create_experiment_task(
        ds, m_new, "funnel_src_smoke", save_dir;
        target_seasons = ["25/26"], history_seasons = 2, warmup_period = 0,
        dynamics_col = :match_biweek,
        samples = 600, warmup = 1000, chains = 4, use_queue = true, max_depth = 8)
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)

    items = res.training_results.items
    gate("G5a folds trained", length(items) > 0, "items = $(length(items))")

    worst = 0.0
    for it in items
        er = DataFrame(MCMCChains.ess_rhat(it[1]))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
        isempty(vals) || (worst = max(worst, maximum(vals)))
    end
    gate("G5b convergence", worst <= 1.05, "worst R-hat = $(_r(worst))")

    p2_draws = vcat([vec(Array(it[1][:p2_raw])) for it in items]...)
    p2_hat = mean(1 ./ (1 .+ exp.(-p2_draws)))
    gate("G5c p₂ ≈ 0.145", 0.11 <= p2_hat <= 0.18, "p₂ = $(_r(p2_hat))")

    println("\n[G6] End-to-end pricing + eval...")
    try
        ppd = Predictions.model_inference(ds, res)
        oos = Experiments.extract_oos_predictions(ds, res)
        gate("G6 pricing + eval", nrow(oos) > 0,
             "OOS rows = $(nrow(oos)), PPD built = $(ppd !== nothing)")
    catch e
        gate("G6 pricing + eval", false, "$(sprint(showerror, e))")
    end
else
    println("\n[G5/G6] skipped — set ENV[\"FUNNEL_VERIFY_TRAIN\"] = \"1\" to run the train smoke.")
end

# ==========================================================================================
println("\n", "="^78)
println(" SUMMARY")
println("="^78)
foreach(println, RESULTS)
n_fail = count(l -> startswith(l, "FAIL"), RESULTS)
println("\n", n_fail == 0 ? "ALL GATES PASSED" : "$n_fail GATE(S) FAILED")
open(joinpath(@__DIR__, "r08_verify_results.txt"), "w") do io
    foreach(l -> println(io, l), RESULTS)
    println(io, "\n", n_fail == 0 ? "ALL GATES PASSED" : "$n_fail GATE(S) FAILED")
end
