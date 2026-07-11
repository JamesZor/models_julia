#=
r06 — SMOKE runner for the GRADUATED team-level smile engine (Stage 4 verification).

Proof that l01's `TeamSmileDPGoalsModel` graduation into src is complete. This file builds the
model FROM SRC ONLY — it deliberately does NOT include(...) the l01 loader (that's the test).

⚠ EDIT FIRST (after r05): set SUP_W / SMILE_W / HL / HS to the Grid-B winner. Defaults below are
the Ireland keeper starting values — the smoke is valid either way, but run the winner config so
check 6 (loader-vs-src number cross-check) is meaningful if you enable it.

src pieces exercised:
  • Features.LeagueFeature + core extractor          (src/features/extractors/core_extractors.jl)
  • Features.MarketSmileFeature + inversion          (src/features/extractors/market_extractors.jl)
  • PreGame.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel
                                                     (…/team_level/time_decay/goals_smile_league.jl)
  • Predictions AbstractSmilePoissonEngines Union → SmileScoreMatrix O/U path
                                                     (src/predictions/score_computation/smile_poisson.jl)
  • Fixed (non-phantom) required_features of every *Market* engine.

Run on the server (kaimon REPL) after git pull + REPL RESTART (src struct changes):
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r06_smoke_src.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
# NB: NO include of l01 — the whole point is that src alone builds the model.

# ⚠ winner config from r05 (NOTES.md):
const SUP_W   = 1.0
const SMILE_W = 0.5
const HL      = 180.0
const HS      = 2

verdicts = Tuple{String, Bool}[]
_mark(name, ok) = (push!(verdicts, (name, ok)); println(ok ? "✅ $name" : "❌ $name"))

# ==========================================
# 1. DATA + MODEL (src struct only)
# ==========================================
println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_src_smoke/")
mkpath(save_dir)

season_strings = sort(unique(String.(ds.matches.season)))
TARGET = season_strings[end]

model = PreGame.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel(
    dynamics_config  = PreGame.TimeDecayDynamics(days_half_life = HL),
    supremacy_weight = SUP_W,
    smile_weight     = SMILE_W,
)

# ==========================================
# CHECK 1 — src-only build + feature plumbing
# ==========================================
println("\n", "="^60, "\nCHECK 1 — src-only build + features\n", "="^60)
_mark("1a. model is a src PreGame engine (no loader include)",
      model isa PreGame.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel)
reqs = Features.required_features(model)
_mark("1b. required_features declares Features.LeagueFeature",
      any(f -> f isa Features.LeagueFeature, reqs))
_mark("1c. required_features declares MarketSmileFeature",
      any(f -> f isa Features.MarketSmileFeature, reqs))
_mark("1d. src *Market* engines' required_features no longer throw (phantom fixed)",
      (try; Features.required_features(PreGame.DynamicMarketGoalsTimeDecayModel(
          interception_config  = PreGame.HierarchicalMonthlyInterception(),
          dynamics_config      = PreGame.TimeDecayDynamics(),
          dispersion_config    = PreGame.HomeAwayDispersion(),
          homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage())); true
       catch; false end))
_mark("1e. MarketLambdaFeature export is gone",
      !isdefined(Features, :MarketLambdaFeature))

# LeagueFeature extractor direct exercise
Fl = Dict{Symbol, Any}()
Features.add_feature!(Fl, Features.LeagueFeature(), ds.matches.match_id, Dict{Any,Any}(), ds)
_mark("1f. LeagueFeature extractor: n_leagues=2, both indices",
      Fl[:n_leagues] == 2 && sort(unique(Fl[:flat_league_ids])) == [1, 2])

# ==========================================
# 2. TRAIN — one short window (same spec as r01)
# ==========================================
println("\n", "="^60, "\nTRAIN — src engine, target=$TARGET\n", "="^60)
task = Experiments.create_experiment_task(
    ds, model, "r06_src_smile_smoke", save_dir;
    target_seasons  = [TARGET],
    history_seasons = HS,
    warmup_period   = 16,
    dynamics_col    = :match_biweek,
    samples         = 600,
    warmup          = 600,
    chains          = 4,
    use_queue       = true,
    max_depth       = 10,
)
res = Experiments.run_experiment(task)
Experiments.save_experiment(res)
_mark("2. trained (items=$(length(res.training_results.items)) > 0)",
      length(res.training_results.items) > 0)

# ==========================================
# CHECK 3 — convergence (raw chain incl. new params)
# ==========================================
println("\n", "="^60, "\nCHECK 3 — convergence\n", "="^60)
raw = res.training_results.items[1][1]
er = DataFrame(MCMCChains.ess_rhat(raw))
rcol = :rhat in propertynames(er) ? :rhat :
       first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
_rhat(p) = (rows = er[er.parameters .== Symbol(p), rcol]; isempty(rows) ? NaN : rows[1])
new_params = vcat(["σ_sup", "σ_smile", "δ_league_raw[1]", "δ_league_raw[2]"],
                  ["log_φ[$k]" for k in 1:5])
rhats = [(p, _rhat(p)) for p in new_params]
for (p, r) in rhats; println("    $(rpad(p, 18)) rhat=$(round(r, digits=4))"); end
vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
max_all = isempty(vals) ? NaN : maximum(vals)
_mark("3a. all new params present in chain", all(!isnan(r) for (_, r) in rhats))
_mark("3b. new-param R-hat ≤ 1.01", all(!isnan(r) && r <= 1.01 for (_, r) in rhats))
_mark("3c. global max R-hat ≤ 1.05 (=$(round(max_all, digits=4)))", max_all <= 1.05)

# ==========================================
# CHECK 4 — PPD end-to-end via the widened Union dispatch
# ==========================================
println("\n", "="^60, "\nCHECK 4 — PPD + smile pricing (Union dispatch)\n", "="^60)
ppd = Predictions.model_inference(ds, res)
_mark("4a. model_inference runs, non-empty PPD (no :r ArgumentError)",
      ppd isa Predictions.PPD && nrow(ppd) > 0)

latents = Experiments.extract_oos_predictions(ds, res)
ou_diffs = Float64[]
for row in eachrow(latents.df)
    params = Predictions.extract_params(model, row)
    S = Predictions.compute_score_matrix(model, params)
    S isa Predictions.SmileScoreMatrix || continue
    for K in 0:4
        mkt = Data.MarketOverUnder(K + 0.5)
        sp = Predictions.compute_market_probs(S, mkt)
        gp = Predictions.compute_market_probs(S.grid, mkt)
        ks = Data.outcomes(mkt)
        push!(ou_diffs, abs(mean(sp[ks.over]) - mean(gp[ks.over])))
    end
end
max_d = isempty(ou_diffs) ? 0.0 : maximum(ou_diffs)
_mark("4b. score matrix routes to SmileScoreMatrix", !isempty(ou_diffs))
_mark("4c. smile O/U ≠ grid O/U (max Δ=$(round(max_d, digits=4)))", max_d > 1e-4)

# ==========================================
# CHECK 5 — φ shape + δ_league read (context)
# ==========================================
println("\n", "="^60, "\nCHECK 5 — φ(K) + δ_league\n", "="^60)
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end
φ_ok = true
for k in 1:5
    d = exp.(_pool(res, "log_φ[$k]"))
    lo, hi = quantile(d, 0.05), quantile(d, 0.95)
    println("    O/U $(k-0.5):  φ=$(round(mean(d), digits=3))  90%=[$(round(lo, digits=3)), $(round(hi, digits=3))]")
    global φ_ok &= 0.7 < mean(d) < 1.3
end
_mark("5a. φ(K) means in a gentle band (0.7–1.3)", φ_ok)
gap = _pool(res, "δ_league_raw[1]") .- _pool(res, "δ_league_raw[2]")
println("    δ₅₆−δ₅₇: mean=$(round(mean(gap), digits=4)) " *
        "90%=[$(round(quantile(gap, 0.05), digits=4)), $(round(quantile(gap, 0.95), digits=4))]")
_mark("5b. δ_league finite & bounded", all(isfinite, gap) && abs(mean(gap)) < 0.3)

# ==========================================
# SUMMARY
# ==========================================
println("\n", "="^60, "\nR06 SRC-SMOKE SUMMARY\n", "="^60)
for (name, ok) in verdicts; println(ok ? "✅ $name" : "❌ $name"); end
n_pass = count(last, verdicts)
println("\n$(n_pass)/$(length(verdicts)) checks passed.")
println(n_pass == length(verdicts) ?
    ">> GRADUATION VERIFIED: src builds, trains and prices the team smile engine end-to-end.\n" *
    ">> Remaining: bake the r05 winner weights into the struct defaults + run Pkg.test()." :
    ">> SOME CHECKS FAILED — inspect before recording graduation.")
