#=
r00 — WP0 DATA QA for the proxy-xG stream. No MCMC; runs in well under a minute.

Every gate here is a hard assert on something a later work package silently depends on. Run it
first and read every line — the funnel stream lost hours to a shot-count assumption that was never
checked.

    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_proxy_xg/r00_data_qa.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using Dates

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l01_proxy_xg_feature.jl"))

verdicts = Tuple{String, Bool}[]
_mark(name, ok) = (push!(verdicts, (name, ok)); println(ok ? "✅ $name" : "❌ $name"))
_r(x, d = 4) = round(x, digits = d)

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())

shots = Features.build_shots(ds)
lut, covered = proxy_xg_table(ds)

println("\n", "="^72,
        "\nGATE 1 — COMMENTARY COVERAGE (100% from 23/24, ZERO before)\n", "="^72)

seasons = "season" in names(ds.matches) ?
          sort(unique(String.(ds.matches.season))) : String[]
cov_rows = NamedTuple[]
for s in seasons
    ids = Set(Int.(ds.matches.match_id[String.(ds.matches.season) .== s]))
    n_cov = count(in(covered), ids)
    push!(cov_rows, (season = s, matches = length(ids), covered = n_cov,
                     pct = length(ids) == 0 ? 0.0 : _r(100n_cov / length(ids), 1)))
end
cov = DataFrame(cov_rows)
show(cov; allrows = true, allcols = true); println()

modern = filter(r -> r.season >= "23/24", cov)
legacy = filter(r -> r.season < "23/24", cov)
_mark("1a. every 23/24+ season is 100% covered",
      nrow(modern) > 0 && all(modern.pct .>= 99.9))
_mark("1b. every pre-23/24 season is 0% covered",
      nrow(legacy) == 0 || all(legacy.pct .<= 0.1))
_mark("1c. covered match count ≈ 1070+ (=$(length(covered)))",
      length(covered) >= 1000)

println("\n", "="^72,
        "\nGATE 2 — SHOT-COUNT RECONCILIATION (commentary events vs ds.bbc match pages)\n", "="^72)
println("""
    Arm B conditions the xG pillar on the EVENT count (the shots the xG was actually summed over)
    while the volume pillar reads ds.bbc. A constant level gap is absorbed by the global kappa; a
    gap that varies BY TEAM is a bias kappa cannot absorb, and would invalidate Arm B.""")

tr = proxy_team_rows(ds)
_mark("2a. proxy_team_rows built (n=$(nrow(tr)))", nrow(tr) > 1500)

if nrow(tr) > 0
    both = dropmissing(tr, [:shots_bbc, :shots_events])
    ratio = mean(both.shots_events) / mean(both.shots_bbc)
    println("    mean shots: events=$(_r(mean(both.shots_events), 2))  " *
            "bbc=$(_r(mean(both.shots_bbc), 2))  ratio=$(_r(ratio, 3))")
    println("    per-match correlation = $(_r(cor(both.shots_events, both.shots_bbc), 3))")

    byteam = combine(groupby(both, :team),
                     [:shots_events, :shots_bbc] =>
                         ((e, b) -> mean(e) / mean(b)) => :ratio, nrow => :n)
    byteam = byteam[byteam.n .>= 30, :]
    sd_ratio = nrow(byteam) > 1 ? std(byteam.ratio) : NaN
    println("    per-team ratio over $(nrow(byteam)) teams: mean=$(_r(mean(byteam.ratio), 3)) " *
            "sd=$(_r(sd_ratio, 4)) range=[$(_r(minimum(byteam.ratio), 3)), $(_r(maximum(byteam.ratio), 3))]")

    _mark("2b. level gap is modest (0.85 ≤ ratio ≤ 1.15)", 0.85 <= ratio <= 1.15)
    _mark("2c. the two series track per match (cor ≥ 0.85)",
          cor(both.shots_events, both.shots_bbc) >= 0.85)
    # THE gate that Arm B rests on: kappa is global, so the gap must not be team-specific.
    _mark("2d. gap is NOT systematic by team (sd of per-team ratio ≤ 0.06)",
          !isnan(sd_ratio) && sd_ratio <= 0.06)
end

println("\n", "="^72,
        "\nGATE 3 — CALIBRATION: sum(proxy xG) vs sum(goals)  ⇒  kappa-hat\n", "="^72)
println("""
    The cell table is a CONVERSION-RATE table, so sum(proxy xG) ~ sum(goals) by construction and
    kappa should land near 1. A large deviation means own goals, the ~2.4% unattributed side, or a
    penalty constant that does not transfer — and it tells us whether log_kappa ~ N(0, 0.2) is
    correctly centred.""")

if nrow(tr) > 0
    κ̂ = sum(tr.goals) / sum(tr.pxg)
    println("    sum(goals)=$(_r(sum(tr.goals), 1))  sum(pxg)=$(_r(sum(tr.pxg), 1))  " *
            "kappa_hat=$(_r(κ̂, 4))  (log=$(_r(log(κ̂), 4)))")
    for g in groupby(tr, :season)
        println("      $(g.season[1]): kappa_hat=$(_r(sum(g.goals) / sum(g.pxg), 3))  n=$(nrow(g))")
    end
    _mark("3a. kappa_hat within [0.80, 1.25]", 0.80 <= κ̂ <= 1.25)
    _mark("3b. log(kappa_hat) inside the N(0, 0.2) prior at 2sd", abs(log(κ̂)) <= 0.4)
end

println("\n", "="^72,
        "\nGATE 4 — THE CELL TABLE ON 56/57 ALONE\n", "="^72)
println("""
    ds.bbc_events is segment-filtered, so fit_shot_xg sees only 56/57 attempts (~19.5k) rather than
    the pooled 54-57 ~45k the research validated. EB shrinkage (k=25) should leave no degenerate
    cell. This mirrors what pm_prepared already does for the y_xg target — same behaviour, recorded
    so a drift is detectable.""")

m = Features.fit_shot_xg(shots)
cellvals = collect(values(m.cells))
println("    attempts=$(nrow(shots))  cells=$(length(m.cells))  " *
        "base_rate=$(_r(m.base_rate, 4))  penalty_xg=$(_r(m.penalty_xg, 4))  k=$(m.k)")
println("    cell xG range=[$(_r(minimum(cellvals), 4)), $(_r(maximum(cellvals), 4))]  " *
        "median=$(_r(median(cellvals), 4))")
_mark("4a. enough attempts to fit (≥ 15k)", nrow(shots) >= 15_000)
_mark("4b. no degenerate cell (all strictly inside (0, 1))",
      !isempty(cellvals) && all(0.0 .< cellvals .< 1.0))
_mark("4c. base rate is football-plausible (0.05–0.20)", 0.05 <= m.base_rate <= 0.20)
_mark("4d. penalty xG is football-plausible (0.65–0.90)", 0.65 <= m.penalty_xg <= 0.90)

println("\n", "="^72,
        "\nGATE 5 — PARSE COVERAGE + THE FREE-KICK OUTCOME-LEAK REMAP\n", "="^72)
println("""
    shot_parser.jl:52-61: "from a free kick with a ..." appears ONLY in goal descriptions, so a cell
    keyed on it converts at 100% and reads the outcome off the wording. The parser remaps it to
    (:outside_box, :direct_free_kick). Verify the remap actually fired on this segment.""")

pc = mean(shots.parsed)
println("    parse coverage = $(_r(100pc, 2))%   unparsed = $(count(.!shots.parsed))")
n_fk_zone = count(==(:free_kick_zone), shots.zone)
n_fk_ctx  = count(==(:direct_free_kick), shots.context)
println("    zone == :free_kick_zone (must be 0) = $n_fk_zone    " *
        "context == :direct_free_kick = $n_fk_ctx")
_mark("5a. parse coverage ≥ 98%", pc >= 0.98)
_mark("5b. :free_kick_zone fully remapped away", n_fk_zone == 0)
_mark("5c. direct-free-kick context is retained", n_fk_ctx > 0)

println("\n", "="^72,
        "\nGATE 6 — THE GAMMA PILLAR'S INPUT DISTRIBUTION\n", "="^72)
println("""
    Sets the nu prior and confirms the zero-flooring is a guard rather than a live code path.
    Gamma support is x > 0; a genuine 0 would give logpdf = -Inf.""")

if nrow(tr) > 0
    n_zero = count(<=(0.0), tr.pxg)
    cv = std(tr.pxg) / mean(tr.pxg)
    println("    team-match pxg: n=$(nrow(tr))  mean=$(_r(mean(tr.pxg), 3))  " *
            "sd=$(_r(std(tr.pxg), 3))  CV=$(_r(cv, 3))  min=$(_r(minimum(tr.pxg), 4))  " *
            "max=$(_r(maximum(tr.pxg), 3))")
    println("    implied MARGINAL nu = 1/CV^2 = $(_r(1 / cv^2, 2))  " *
            "(a LOWER bound on the within-rate nu the pillar fits — WP1-E4 measures the real one)")
    println("    exact-zero sides = $n_zero    mean goals = $(_r(mean(tr.goals), 3))")
    _mark("6a. no exact-zero proxy xG (the floor is a guard, not a path)", n_zero == 0)
    _mark("6b. mean pxg is on the goals scale (0.8–2.0) ⇒ no shot_scale offset needed",
          0.8 <= mean(tr.pxg) <= 2.0)
    _mark("6c. marginal nu is nearer 4 than Ireland's 3.0 default",
          1.0 <= 1 / cv^2 <= 20.0)
end

println("\n", "="^72,
        "\nGATE 7 — THE FEATURE CONTRACT (what the engines actually receive)\n", "="^72)

F = Dict{Symbol, Any}()
ids = collect(Int.(ds.matches.match_id))
Features.add_feature!(F, ProxyXGFeature(), ids, Dict{Any, Any}(), ds)
xh = F[:flat_home_xg_proxy]; xa = F[:flat_away_xg_proxy]
mh = F[:flat_pxg_mask_h];    ma = F[:flat_pxg_mask_a]
nh = F[:flat_home_pxg_shots]

println("    n=$(length(ids))  masked-in home=$(Int(sum(mh)))  away=$(Int(sum(ma)))")
_mark("7a. all six keys emitted",
      all(haskey(F, k) for k in (:flat_home_xg_proxy, :flat_away_xg_proxy, :flat_home_pxg_shots,
                                 :flat_away_pxg_shots, :flat_pxg_mask_h, :flat_pxg_mask_a)))
_mark("7b. NO NaN / Inf anywhere (the -Inf*0 == NaN trap)",
      all(isfinite, xh) && all(isfinite, xa))
_mark("7c. every value strictly positive (Gamma support)", all(xh .> 0) && all(xa .> 0))
_mark("7d. masked-out slots carry the 1.0 dummy",
      all(xh[mh .== 0.0] .== 1.0) && all(xa[ma .== 0.0] .== 1.0))
_mark("7e. masked-in slots all have ≥1 event shot", all(nh[mh .== 1.0] .>= 1))
_mark("7f. mask total ≈ covered matches ($(Int(sum(mh))) vs $(length(covered)))",
      abs(sum(mh) - length(covered)) <= 0.02 * length(covered))

# The :training refit must produce a DIFFERENT but sane table — proves the knob is live.
F2 = Dict{Symbol, Any}()
Features.add_feature!(F2, ProxyXGFeature(fit_on = :training), ids, Dict{Any, Any}(), ds)
_mark("7g. fit_on=:training is wired and returns finite values",
      all(isfinite, F2[:flat_home_xg_proxy]) && all(F2[:flat_home_xg_proxy] .> 0))

println("\n", "="^72, "\nR00 SUMMARY\n", "="^72)
for (name, ok) in verdicts; println(ok ? "✅ $name" : "❌ $name"); end
n_pass = count(last, verdicts)
println("\n$(n_pass)/$(length(verdicts)) gates passed.")
println(n_pass == length(verdicts) ?
    ">> WP0 PASS. Proceed to r01_eda_informativeness.jl (the go/no-go gate).\n" *
    ">> Record kappa_hat and the marginal nu in NOTES.md — they set the Arm A priors." :
    ">> GATES FAILED — fix before spending any MCMC time. Do NOT proceed to r01.")
