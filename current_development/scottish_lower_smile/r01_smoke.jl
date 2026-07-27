#=
r01 — SMOKE runner for the l01 team-level engines on ScottishLower (few folds, one target season).

Verdict checks (✅/❌, r17/r20 pattern):
  1. required_features plumbing (LeagueFeature everywhere; market/smile features where due).
  2. LeagueFeature extraction: n_leagues=2, both indices present, lookup covers all matches.
  3. MarketSmileFeature inversion works on Scottish Bet365 odds (finite, rising median ladder).
  4. All 3 engines train without stall (short window: last season, biweek folds ≥ warmup 16).
  5. Convergence: global max R-hat ≤ 1.05; new params (δ_league_raw, σ_sup, σ_smile, log_φ,
     σ_market) R-hat ≤ 1.01.
  6. δ_league posterior read (informational): idx1(=56, League One) vs idx2(=57, League Two);
     expected sign positive (56 scores more), magnitude ~0.02–0.05.
  7. PPD end-to-end per engine (no :r ArgumentError); smile → SmileScoreMatrix and
     smile O/U ≠ grid O/U (φ genuinely priced).
  8. iso σ_market posterior not collapsed at the 0.01 lower bound.

Benchmark/eval odds = plain ds (SofaScore de-vigged close). NO Betfair swap (none for 56/57).

Run on the server (kaimon REPL) after git pull + REPL restart (fresh structs):
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r01_smoke.jl"))
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
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))

verdicts = Tuple{String, Bool}[]
_mark(name, ok) = (push!(verdicts, (name, ok)); println(ok ? "✅ $name" : "❌ $name"))

# ==========================================
# 1. DATA
# ==========================================
println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_smoke/")
mkpath(save_dir)

season_strings = sort(unique(String.(ds.matches.season)))
TARGET = season_strings[end]
println("[INFO] seasons: ", season_strings, " -> smoke target = ", TARGET)

# ==========================================
# 2. MODELS (defaults; smile at Ireland keeper weights as starting point)
# ==========================================
dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = 180.0)

m_dp = TeamDPGoalsModel(dynamics_config = dyn_cfg)
m_iso = TeamIsoDPGoalsModel(dynamics_config = dyn_cfg, market_weight = 1.0)
m_smile = TeamSmileDPGoalsModel(dynamics_config = dyn_cfg,
                                supremacy_weight = 1.0, smile_weight = 0.5)

specs = Tuple{String, Any}[
    ("none_pois_smoke",  m_dp),
    ("iso_pois_smoke",   m_iso),
    ("smile_pois_smoke", m_smile),
]

# ==========================================
# CHECK 1 — required_features plumbing
# ==========================================
println("\n", "="^60, "\nCHECK 1 — required_features\n", "="^60)
for (name, model) in specs
    reqs = Features.required_features(model)
    _mark("1. $name declares LeagueFeature", any(f -> f isa LeagueFeature, reqs))
end
_mark("1b. smile declares MarketSmileFeature",
      any(f -> f isa Features.MarketSmileFeature, Features.required_features(m_smile)))
_mark("1c. NB market engine required_features fixed (no phantom throw)",
      (try; Features.required_features(PreGame.DynamicMarketGoalsTimeDecayModel(
          interception_config  = PreGame.HierarchicalMonthlyInterception(),
          dynamics_config      = dyn_cfg,
          dispersion_config    = PreGame.HomeAwayDispersion(),
          homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage())); true
       catch; false end))

# ==========================================
# CHECK 2 — LeagueFeature extraction
# ==========================================
println("\n", "="^60, "\nCHECK 2 — LeagueFeature\n", "="^60)
Fl = Dict{Symbol, Any}()
all_ids = ds.matches.match_id
Features.add_feature!(Fl, LeagueFeature(), all_ids, Dict{Any,Any}(), ds)
_mark("2a. n_leagues == 2", Fl[:n_leagues] == 2)
_mark("2b. both league indices present",
      sort(unique(Fl[:flat_league_ids])) == [1, 2])
_mark("2c. league_lookup covers every match", length(Fl[:league_lookup]) == nrow(ds.matches))

# ==========================================
# CHECK 3 — smile inversion on Scottish odds
# ==========================================
println("\n", "="^60, "\nCHECK 3 — MarketSmileFeature on Bet365 (SofaScore) odds\n", "="^60)
Fs = Dict{Symbol, Any}()
odds_ids = unique(ds.odds.match_id)
Features.add_feature!(Fs, Features.MarketSmileFeature(Kmax=4), odds_ids, Dict{Any,Any}(), ds)
logΛ = Fs[:flat_smile_logΛ]; msk = Fs[:flat_smile_mask]
full_rows = findall(i -> all(msk[i, :] .> 0.5), 1:size(msk, 1))
Λ_med = isempty(full_rows) ? Float64[] : exp.(vec(median(logΛ[full_rows, :], dims=1)))
_mark("3a. full-ladder matches exist (n=$(length(full_rows)))", length(full_rows) > 100)
_mark("3b. Λ^mkt finite & positive", !isempty(full_rows) && all(isfinite, logΛ[full_rows, :]))
_mark("3c. median Λ^mkt(K) rises with strike", length(Λ_med) == 5 && all(diff(Λ_med) .> 0))
println("    median Λ^mkt(K) strikes 0.5→4.5: ", round.(Λ_med, digits=3))

# ==========================================
# 4. TRAIN — short window (last season, biweek, warmup_period=16 → few folds)
# ==========================================
println("\n", "="^60, "\nTRAIN — 3 engines, target=$TARGET\n", "="^60)
runs = Tuple{String, Any, Any}[]
for (name, model) in specs
    println("\n", "#"^68, "\n# RUN: $name\n", "#"^68)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = [TARGET],
            history_seasons = 2,
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
        push!(runs, (name, model, res))
        n_items = length(res.training_results.items)
        _mark("4. $name trained (items=$n_items > 0, no silent drop)", n_items > 0)
    catch e
        _mark("4. $name trained", false)
        @error "FAILED: $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# CHECK 5 — convergence (curated + raw chain for the new params)
# ==========================================
println("\n", "="^60, "\nCHECK 5 — convergence\n", "="^60)
for (name, model, res) in runs
    try
        chains_obj = Experiments.Diagnostics.extract_chains(ds, res)
        conv = Experiments.Diagnostics.check_convergence(chains_obj)
        worst = isempty(conv.df) ? NaN : maximum(skipmissing(conv.df.rhat))
        _mark("5a. $name global max R-hat ≤ 1.05 (=$(round(worst, digits=4)))", worst <= 1.05)

        # new-param R-hat from the raw first-fold chain (curated table drops engine-level params)
        raw = res.training_results.items[1][1]
        er = DataFrame(MCMCChains.ess_rhat(raw))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        _rhat(p) = (rows = er[er.parameters .== Symbol(p), rcol]; isempty(rows) ? NaN : rows[1])
        new_params = ["δ_league_raw[1]", "δ_league_raw[2]"]
        model isa TeamIsoDPGoalsModel   && push!(new_params, "σ_market")
        model isa TeamSmileDPGoalsModel && append!(new_params,
            vcat(["σ_sup", "σ_smile"], ["log_φ[$k]" for k in 1:5]))
        rhats = [(p, _rhat(p)) for p in new_params]
        for (p, r) in rhats; println("    $(rpad(p, 18)) rhat=$(round(r, digits=4))"); end
        ok = all(!isnan(r) && r <= 1.01 for (_, r) in rhats)
        _mark("5b. $name new-param R-hat ≤ 1.01", ok)
    catch e
        _mark("5. $name convergence check ran", false)
        @error "convergence check failed: $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# CHECK 6 — δ_league posterior read (pooled across folds; informational sign check)
# ==========================================
println("\n", "="^60, "\nCHECK 6 — δ_league (idx1=56 League One, idx2=57 League Two)\n", "="^60)
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end
for (name, model, res) in runs
    d1 = _pool(res, "δ_league_raw[1]"); d2 = _pool(res, "δ_league_raw[2]")
    isempty(d1) && (_mark("6. $name δ_league in chain", false); continue)
    gap = d1 .- d2   # centring cancels in the difference: δ_league[1] − δ_league[2] = raw1 − raw2
    println("  $name: δ₅₆−δ₅₇ mean=$(round(mean(gap), digits=4)) " *
            "90%=[$(round(quantile(gap, 0.05), digits=4)), $(round(quantile(gap, 0.95), digits=4))]" *
            "  (expected ≈ +0.05 rate gap; sign informational)")
    _mark("6. $name δ_league finite & bounded (|mean gap| < 0.3)",
          all(isfinite, gap) && abs(mean(gap)) < 0.3)
end

# ==========================================
# CHECK 7 — PPD end-to-end + smile ≠ grid O/U
# ==========================================
println("\n", "="^60, "\nCHECK 7 — PPD + smile pricing\n", "="^60)
for (name, model, res) in runs
    try
        ppd = Predictions.model_inference(ds, res)
        _mark("7a. $name PPD runs, non-empty (n=$(nrow(ppd)))",
              ppd isa Predictions.PPD && nrow(ppd) > 0)
    catch e
        _mark("7a. $name PPD runs", false)
        @error "PPD failed: $name" exception=(e, catch_backtrace())
    end
end
smile_run = findfirst(r -> r[2] isa TeamSmileDPGoalsModel, runs)
if smile_run !== nothing
    name, model, res = runs[smile_run]
    ou_diffs = Float64[]
    latents = Experiments.extract_oos_predictions(ds, res)
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
    _mark("7b. smile score matrix is SmileScoreMatrix", !isempty(ou_diffs))
    _mark("7c. smile O/U ≠ grid O/U (max Δ=$(round(max_d, digits=4)))", max_d > 1e-4)
end

# ==========================================
# CHECK 8 — iso σ_market not collapsed at the bound
# ==========================================
println("\n", "="^60, "\nCHECK 8 — iso σ_market\n", "="^60)
iso_run = findfirst(r -> r[2] isa TeamIsoDPGoalsModel, runs)
if iso_run !== nothing
    _, _, res = runs[iso_run]
    sm = _pool(res, "σ_market")
    println("  σ_market: mean=$(round(mean(sm), digits=4)) " *
            "90%=[$(round(quantile(sm, 0.05), digits=4)), $(round(quantile(sm, 0.95), digits=4))]")
    _mark("8. σ_market > 0.02 (not pinned at the 0.01 bound)", mean(sm) > 0.02)
end

# smile σ read (context; no gate)
if smile_run !== nothing
    _, _, res = runs[smile_run]
    for s in ("σ_sup", "σ_smile")
        v = _pool(res, s)
        isempty(v) || println("  $s: mean=$(round(mean(v), digits=4)) " *
            "90%=[$(round(quantile(v, 0.05), digits=4)), $(round(quantile(v, 0.95), digits=4))]")
    end
end

# ==========================================
# SUMMARY
# ==========================================
println("\n", "="^60, "\nR01 SMOKE SUMMARY\n", "="^60)
for (name, ok) in verdicts; println(ok ? "✅ $name" : "❌ $name"); end
n_pass = count(last, verdicts)
println("\n$(n_pass)/$(length(verdicts)) checks passed.")
println(n_pass == length(verdicts) ?
    ">> SMOKE PASSED: proceed to r02 Grid A (decay × history)." :
    ">> SOME CHECKS FAILED — fix before any grid run.")
