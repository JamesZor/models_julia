#=
RUNNER r03 — MPT vs Kelly over the full OOS stream (Ireland 2025-26, Betfair close).

The comparison runner. Same OOS matches, same trust model, same cap, same accounting as the
incumbent staking_layer race — the ONLY thing that varies across the roster is the portfolio
optimiser. Answers: does anything on the Markowitz side beat capped Kelly on real books, and does
the paper's ranking (fractional Kelly wins, MaxSharpe fragile, DRO safest) reproduce here.

Order of output:
  0  COHERENCE GUARD   — must pass before anything below means anything
  1  SEQUENTIAL RACE   — terminal wealth, G/match, max drawdown, ruin (the incumbent report)
  2  PER-FAMILY P/L    — where each optimiser makes or loses money
  3  PAPER PROTOCOL    — §6.2 bootstrap: median/mean/min/max/sigma/ruin% + the §6.2.1 pick

Run (server / kaimon, one warm session):
    include(".../staking_layer/src/loader.jl")
    include(".../staking_layer/preflight_real.jl")
    include(".../mpt_dev/l01_mpt_portfolio.jl")
    include(".../mpt_dev/l02_mpt_policies.jl")
    include(".../mpt_dev/r03_oos_race.jl")
=#
using BayesianFootball
using Printf, Statistics, DataFrames

const _MPT_DIR = joinpath(pkgdir(BayesianFootball), "current_development", "mpt_dev")
isdefined(Main, :STAKING_LAYER_DIR) ||
    include(joinpath(pkgdir(BayesianFootball), "current_development", "staking_layer", "src", "loader.jl"))
isdefined(Main, :build_real_inputs) || include(joinpath(STAKING_LAYER_DIR, "preflight_real.jl"))
isdefined(Main, :solve_msharpe)     || include(joinpath(_MPT_DIR, "l01_mpt_portfolio.jl"))
isdefined(Main, :MPTPolicy)         || include(joinpath(_MPT_DIR, "l02_mpt_policies.jl"))

"""
    run_mpt_race(; c=0.02, cap=0.2, trust=CuratedTrust(), refit_every=25, reps=1000)

Full MPT-vs-Kelly OOS comparison. Writes results/mpt_race_<tag>.txt and returns the race object.
"""
function run_mpt_race(; c::Float64=0.02, cap::Float64=0.2, trust=CuratedTrust(),
                        refit_every::Int=25, reps::Int=1000, iters::Int=1200,
                        outdir=joinpath(_MPT_DIR, "results"))
    mkpath(outdir)
    inp = build_real_inputs()
    loaded = load_matches(RealSource(lat=inp.lat, ppd=inp.ppd, odds_bf=inp.odds_bf,
                                     matches_df=inp.ds1.matches, c=c))

    lines = String[]
    push!(lines, "MPT vs KELLY — OOS race · n=$(length(loaded.matches)) · c=$c · cap=$cap · trust=$(typeof(trust))")

    # ---- 0. coherence guard -------------------------------------------------
    push!(lines, "", "="^78, "0. COHERENCE GUARD (plain grid vs model PPD, first 25 matches)", "="^78)
    append!(lines, coherence_report(loaded))
    push!(lines, "",
        "If |grid−PPD| is large on the OverUnder rows, the PLAIN grid invents totals edge and",
        "raw_edge will be far above tilted_edge. Every policy below solves on the TILTED grid,",
        "so the race is safe either way — but a big gap means r02's raw-pbar numbers were the",
        "artefact, not a finding.")

    # ---- 1. sequential race -------------------------------------------------
    roster = vcat(reference_roster(cap=cap), mpt_roster(trust=trust, cap=cap, iters=iters))
    @info "running race" n=length(loaded.matches) policies=length(roster)
    rs = run_race(loaded, roster; refit_every=refit_every, seed=1)

    push!(lines, "", "="^78, "1. SEQUENTIAL RACE (chronological, compounded)", "="^78)
    append!(lines, summary_rows(rs))
    push!(lines, "", "smile w=1 tilt max|Δover-prob| = $(rs.max_tilt_err)  (want ≈ 0)")

    # ---- 2. per-family P/L --------------------------------------------------
    push!(lines, "", "="^78, "2. PER-FAMILY P/L", "="^78)
    append!(lines, family_rows(rs))

    # ---- 3. paper protocol --------------------------------------------------
    push!(lines, "", "="^78, "3. PAPER PROTOCOL §6.2 ($reps reshuffles, 10% dropout)", "="^78)
    append!(lines, protocol_rows(rs; reps=reps))

    tag = @sprintf("c%03d_cap%03d", round(Int, c * 1000), round(Int, cap * 100))
    body = join(lines, "\n")
    write(joinpath(outdir, "mpt_race_$(tag).txt"), body)
    println(body)
    return (; rs, loaded)
end

out = run_mpt_race()
