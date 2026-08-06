#=
r01 — SMOKE + RUNTIME CALIBRATION for the Scottish Upper (54/55) ladder.

Short window (last season, biweek folds ≥ warmup 16 → ~5 folds) over every candidate cell. This is a
HARD GATE: nothing goes into an overnight grid until it passes here, and cells that fail are CUT, not
debugged into the night.

Three jobs:

  A. FEATURE SANITY — the things that silently produce garbage rather than erroring:
     1. required_features plumbing (LeagueFeature everywhere it is due).
     2. LeagueFeature: n_leagues=2, both indices present, lookup covers all matches.
     3. ShotsFunnelFeature (ds.bbc) coverage — the funnel arm's only viable shot source.
     4. ⚠ RATING SCALE + MASK. The centred SofaScore rating must be O(1-3) and UNIMODAL. If the
        22/23 history block (tournament 55 has no ratings there) leaks through unmasked it shows up
        as a second mode near −65, and the ratings arm is worthless. Also reports the realised sd so
        the `w_sd = 0.05` prior can be checked against it.
     5. MarketSmileFeature inversion on Bet365 odds (finite, rising median ladder).

  B. CONVERGENCE — global max R-hat ≤ 1.05, new params ≤ 1.01, at max_depth=10 (never capped).

  C. RUNTIME CALIBRATION — wall-clock per cell, extrapolated to the grid:
         cell_wall ≈ smoke_wall × (grid_folds / smoke_folds)
     This is the number that decides whether `smile_pois` is affordable. On 56/57 smile cost ~20×
     the structural engine and had to be dropped.

  D. PPD end-to-end per engine (no `:r` ArgumentError from a missing dispatch-Union entry) and a
     δ_league read (informational — expect 54 > 55 on goal level, sign positive on idx1).

Benchmark = plain `ds` (de-vigged Bet365/SofaScore close). NO Betfair swap — 54/55 have none.

Run on the server (kaimon REPL) after git pull + REPL RESTART (src structs changed):
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_upper/r01_smoke.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning

pinthreads(:cores)

const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Evaluation  = BayesianFootball.Evaluation

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_upper/l01_upper.jl"))

verdicts = Tuple{String, Bool}[]
_mark(name, ok) = (push!(verdicts, (name, ok)); println(ok ? "✅ $name" : "❌ $name"))

# ==========================================
# 0. DATA + WINDOW
# ==========================================
println("[INFO] Loading ScottishUpper DataStore...")
ds = Data.load_datastore_cached(Data.ScottishUpper())
save_dir = joinpath(ROOT, "data/scottish_upper_smoke/")
mkpath(save_dir)

season_strings = sort(unique(String.(ds.matches.season)))
const TARGET = last(season_strings)
println("[INFO] seasons: ", season_strings, "   smoke target = ", TARGET)

# Grid window (what r02 will actually run) — needed for the runtime extrapolation.
const GRID_TARGETS = season_strings[max(1, end-1):end]
const GRID_HS      = 2

specs = family_specs(include_smile = true)
println("[INFO] ladder: ", join(first.(specs), ", "))

# ==========================================
# CHECK 1 — required_features plumbing
# ==========================================
println("\n", "="^60, "\nCHECK 1 — required_features\n", "="^60)
for (name, model) in specs
    try
        feats = Features.required_features(model)
        println("    $(rpad(name, 32)) ", join(string.(typeof.(feats)), ", "))
        _mark("1. $name required_features resolves", true)
    catch e
        _mark("1. $name required_features resolves", false)
        @error "required_features threw for $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# CHECK 2 — LeagueFeature (the pooled-division machinery)
# ==========================================
println("\n", "="^60, "\nCHECK 2 — LeagueFeature\n", "="^60)
Fl = Dict{Symbol, Any}()
all_ids = unique(ds.matches.match_id)
Features.add_feature!(Fl, Features.LeagueFeature(), all_ids, Dict{Any,Any}(), ds)
n_leagues = Int(Fl[:n_leagues])
lidx = Vector{Int}(Fl[:flat_league_ids])
_mark("2a. n_leagues == 2 (=$n_leagues)", n_leagues == 2)
_mark("2b. both league indices present", Set(unique(lidx)) == Set(1:n_leagues))
_mark("2c. league lookup covers every match", length(lidx) == length(all_ids) && all(lidx .> 0))
println("    league index counts: ", [count(==(i), lidx) for i in 1:n_leagues])

# ==========================================
# CHECK 3 — ShotsFunnelFeature (ds.bbc) — the funnel arm's input
# ==========================================
println("\n", "="^60, "\nCHECK 3 — ShotsFunnelFeature (BBC shots)\n", "="^60)
if nrow(ds.bbc) == 0
    _mark("3. ds.bbc non-empty", false)
else
    Ff = Dict{Symbol, Any}()
    Features.add_feature!(Ff, Features.ShotsFunnelFeature(), all_ids, Dict{Any,Any}(), ds)
    mh = Vector{Float64}(Ff[:flat_funnel_mask_h])
    ma = Vector{Float64}(Ff[:flat_funnel_mask_a])
    sh = Vector{Float64}(Ff[:flat_home_shots_n])
    cov = mean((mh .> 0.5) .& (ma .> 0.5))
    _mark("3a. BBC shot coverage (both sides) ≥ 95% (=$(round(100cov, digits=1))%)", cov >= 0.95)
    obs = sh[mh .> 0.5]
    if isempty(obs)
        _mark("3b. shot counts are plausible", false)
    else
        println("    home shots/match q05/50/95: ", round.(quantile(obs, [0.05, 0.5, 0.95]), digits=1))
        # shot_scale = log(10) is a FIXED offset on the log-rate; if the real level is far from ~10
        # the sampler starts on the wrong scale (that is exactly why the offset exists).
        _mark("3b. median shots/match in [5, 20] (shot_scale=log(10) is right)",
              5.0 <= median(obs) <= 20.0)
    end
end

# ==========================================
# CHECK 4 — ⚠ RATING SCALE + MASK (the arm most likely to be silently broken)
# ==========================================
println("\n", "="^60, "\nCHECK 4 — SofaScore rating pillar: scale + missing-data mask\n", "="^60)
Fr = Dict{Symbol, Any}()
Features.add_feature!(Fr, _ratings_feature(), all_ids, Dict{Any,Any}(), ds)
base = Features.rating_base(_ratings_feature())
D_h = Vector{Float64}(Fr[:flat_home_D_rating])
M_h = Vector{Float64}(Fr[:flat_home_M_rating])
F_h = Vector{Float64}(Fr[:flat_home_F_rating])
tot_h = D_h .+ M_h .+ F_h

rated = tot_h .> 0.0
println("    rating_base = ", base, "   sides with rated minutes: ",
        count(rated), "/", length(rated), " (", round(100mean(rated), digits=1), "%)")

# Masked centring, exactly as src `_pm_outfield` does it.
centred = [t > 0.0 ? t - 10.0 * base : 0.0 for t in tot_h]
c_rated = centred[rated]
if isempty(c_rated)
    _mark("4a. any rated sides at all", false)
else
    println("    centred rating (rated sides): mean=", round(mean(c_rated), digits=3),
            " sd=", round(std(c_rated), digits=3),
            " q05/50/95=", round.(quantile(c_rated, [0.05, 0.5, 0.95]), digits=2))
    # Sanity: a minute-weighted 10-man sum centred on 10·base should sit within a few units of 0.
    _mark("4a. centred rating is O(1-5), not O(60) [mask works]", maximum(abs, c_rated) < 30.0)
    # The failure signature we are hunting: a second mode near -10*base from unmasked missing data.
    n_disaster = count(x -> x < -30.0, centred)
    _mark("4b. no −10·base mode leaking through (n=$(n_disaster))", n_disaster == 0)
    # Prior check: pillar sd ≈ w_sd × rating sd should be a sensible fraction of a log-rate.
    w_sd = 0.05
    println("    ⇒ implied pillar sd at w_sd=$(w_sd): ", round(w_sd * std(c_rated), digits=4),
            " on log-λ  (want ~0.05-0.15; retune w_sd in l01 if far off)")
end

# Unrated sides are expected — the 22/23 block on tournament 55 has no ratings at all. Report the
# split by season so the history_seasons decision from r00 is corroborated on the FEATURE side.
season_of = Dict(Int(r.match_id) => String(r.season) for r in eachrow(ds.matches))
by_season = Dict{String, Vector{Bool}}()
for (i, mid) in enumerate(all_ids)
    s = get(season_of, Int(mid), "?")
    push!(get!(by_season, s, Bool[]), rated[i])
end
println("    rated-side share by season:")
for s in sort(collect(keys(by_season)))
    v = by_season[s]
    println("      ", rpad(s, 22), round(100mean(v), digits=1), "%  (n=", length(v), ")")
end

# ==========================================
# CHECK 5 — MarketSmileFeature inversion
# ==========================================
println("\n", "="^60, "\nCHECK 5 — MarketSmileFeature on Bet365 odds\n", "="^60)
Fs = Dict{Symbol, Any}()
odds_ids = unique(ds.odds.match_id)
Features.add_feature!(Fs, Features.MarketSmileFeature(Kmax = KMAX), odds_ids, Dict{Any,Any}(), ds)
logΛ = Fs[:flat_smile_logΛ]; msk_s = Fs[:flat_smile_mask]
full_rows = findall(i -> all(msk_s[i, :] .> 0.5), 1:size(msk_s, 1))
Λ_med = isempty(full_rows) ? Float64[] : exp.(vec(median(logΛ[full_rows, :], dims=1)))
_mark("5a. full-ladder matches exist (n=$(length(full_rows)))", length(full_rows) > 100)
_mark("5b. Λ^mkt finite & positive", !isempty(full_rows) && all(isfinite, logΛ[full_rows, :]))
_mark("5c. median Λ^mkt(K) rises with strike", length(Λ_med) == KMAX + 1 && all(diff(Λ_med) .> 0))
println("    median Λ^mkt(K) strikes 0.5→$(KMAX).5: ", round.(Λ_med, digits=3))

# ==========================================
# 6. TRAIN — short window + RUNTIME CALIBRATION
# ==========================================
println("\n", "="^60, "\nTRAIN — $(length(specs)) cells, target=$TARGET (smoke window)\n", "="^60)
runs = Tuple{String, Any, Any, Float64, Int}[]   # name, model, res, wall_seconds, n_folds
for (name, model) in specs
    println("\n", "#"^68, "\n# RUN: $name\n", "#"^68)
    t0 = time()
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = [TARGET],
            history_seasons = GRID_HS,
            warmup_period   = 16,
            dynamics_col    = :match_biweek,
            samples         = 600,
            warmup          = 600,
            chains          = 4,
            use_queue       = true,
            max_depth       = 10,     # NEVER capped — depth caps failed the ranking gate on 56/57
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        wall = time() - t0
        n_items = length(res.training_results.items)
        push!(runs, (name, model, res, wall, n_items))
        # A cell can "succeed" while holding no folds — always assert items, never trust the absence
        # of an exception.
        _mark("6. $name trained (items=$n_items > 0, no silent drop, $(round(wall/60, digits=1))m)",
              n_items > 0)
    catch e
        _mark("6. $name trained", false)
        @error "FAILED: $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# CHECK 7 — convergence
# ==========================================
println("\n", "="^60, "\nCHECK 7 — convergence\n", "="^60)
for (name, model, res, _, _) in runs
    try
        chains_obj = Experiments.Diagnostics.extract_chains(ds, res)
        conv = Experiments.Diagnostics.check_convergence(chains_obj)
        worst = isempty(conv.df) ? NaN : maximum(skipmissing(conv.df.rhat))
        _mark("7a. $name global max R-hat ≤ 1.05 (=$(round(worst, digits=4)))", worst <= 1.05)

        raw = res.training_results.items[1][1]
        er = DataFrame(MCMCChains.ess_rhat(raw))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        _rhat(p) = (rows = er[er.parameters .== Symbol(p), rcol]; isempty(rows) ? NaN : rows[1])

        new_params = ["δ_league_raw[1]", "δ_league_raw[2]"]
        model isa TeamIsoDPGoalsModel && push!(new_params, "σ_market")
        model isa PreGame.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel &&
            append!(new_params, vcat(["σ_sup", "σ_smile"], ["log_φ[$k]" for k in 1:(KMAX+1)]))
        model isa PreGame.DynamicGoalsPlusMinusLeagueTimeDecayModel &&
            append!(new_params, ["w_att", "w_def"])
        model isa PreGame.DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel &&
            push!(new_params, "p2_raw")

        rhats = [(p, _rhat(p)) for p in new_params]
        for (p, r) in rhats
            println("    $(rpad(p, 18)) rhat=", isnan(r) ? "—(absent)" : round(r, digits=4))
        end
        present = [(p, r) for (p, r) in rhats if !isnan(r)]
        _mark("7b. $name new-param R-hat ≤ 1.01", !isempty(present) && all(r <= 1.01 for (_, r) in present))
    catch e
        _mark("7. $name convergence read", false)
        @error "convergence read failed: $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# CHECK 8 — δ_league read (informational) + ratings-weight read
# ==========================================
println("\n", "="^60, "\nCHECK 8 — δ_league and pillar weights (informational)\n", "="^60)
for (name, model, res, _, _) in runs
    try
        raw = res.training_results.items[1][1]
        d1 = vec(Array(raw[Symbol("δ_league_raw[1]")]))
        d2 = vec(Array(raw[Symbol("δ_league_raw[2]")]))
        gap = (d1 .- d2) ./ 2       # zero-sum ⇒ per-league offset is half the raw difference
        println("    $(rpad(name, 34)) δ_gap(idx1−idx2)/2 = ", round(mean(gap), digits=4),
                "  90% CI [", round(quantile(gap, 0.05), digits=4), ", ",
                round(quantile(gap, 0.95), digits=4), "]")
        if model isa PreGame.DynamicGoalsPlusMinusLeagueTimeDecayModel
            wa = vec(Array(raw[:w_att])); wd = vec(Array(raw[:w_def]))
            println("        w_att = ", round(mean(wa), digits=4), " ± ", round(std(wa), digits=4),
                    "   w_def = ", round(mean(wd), digits=4), " ± ", round(std(wd), digits=4))
            # If the posterior just reproduces the prior sd (0.05), the pillar is unidentified here.
            println("        (prior sd 0.05 — posterior sd ≈ prior ⇒ pillar carries no information)")
        end
    catch e
        println("    $(rpad(name, 34)) δ_league read failed: ", typeof(e))
    end
end

# ==========================================
# CHECK 9 — PPD end-to-end
# ==========================================
println("\n", "="^60, "\nCHECK 9 — PPD end-to-end\n", "="^60)
for (name, model, res, _, _) in runs
    try
        ppd = Predictions.model_inference(ds, res)
        n = nrow(ppd.df)
        _mark("9. $name PPD generated (rows=$n)", n > 0)
    catch e
        _mark("9. $name PPD generated", false)
        @error "PPD failed: $name — check the score-computation dispatch Union" exception=(e, catch_backtrace())
    end
end

# ==========================================
# 10. RUNTIME BUDGET — the number that sizes r02
# ==========================================
println("\n", "="^60, "\n10. RUNTIME CALIBRATION → r02 budget\n", "="^60)
grid_folds = try
    cfg = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons    = GRID_TARGETS,
        history_seasons   = GRID_HS,
        dynamics_col      = :match_biweek,
        warmup_period     = 0,
    )
    length(Data.create_id_boundaries(ds, cfg))
catch
    -1
end
println("    grid window ", GRID_TARGETS, " hs=$GRID_HS warmup=0 → folds = ", grid_folds)
println()
println("    ", rpad("cell", 34), rpad("smoke_folds", 13), rpad("smoke_wall", 12), "projected r02 wall")
total_h = 0.0
for (name, _, _, wall, n_items) in runs
    proj_h = (grid_folds > 0 && n_items > 0) ? wall / 3600 * (grid_folds / n_items) : NaN
    isnan(proj_h) || (total_h += proj_h)
    println("    ", rpad(name, 34), rpad(n_items, 13),
            rpad(string(round(wall/60, digits=1), "m"), 12),
            isnan(proj_h) ? "—" : string(round(proj_h, digits=1), "h"))
end
println("    ", "-"^70)
println("    TOTAL projected r02 wall: ", round(total_h, digits=1), " h")
println()
println("    NOTE: r02 uses 800/300×4 vs this smoke's 600/600×4, so the projection is")
println("          approximate (fewer warmup, more sampling). Treat it as ±30%.")
println("    DECISION: if TOTAL > ~20h, drop `smile_pois` (last in the ladder, lowest expected")
println("              value — 56/57 found smile/supremacy add nothing to a team-level goals")
println("              engine). Do NOT cap max_depth to buy time; that failed the ranking gate.")

# ==========================================
# SUMMARY
# ==========================================
println("\n", "="^60, "\nSMOKE SUMMARY\n", "="^60)
n_ok = count(last, verdicts)
for (nm, ok) in verdicts
    println(ok ? "  ✅ " : "  ❌ ", nm)
end
println("\n", n_ok, "/", length(verdicts), " checks passed.")
println(n_ok == length(verdicts) ?
        "GATE PASSED → record the runtime table in NOTES.md and run r02_grid_family.jl." :
        "GATE FAILED → CUT the failing cells from family_specs() before r02. Do not debug into the night.")
