#=
RUNNER r04 — EB vs Bayesian trust on the real core book (the new experiment the refactor enables).

Same 275-match src_sup40_sw40 book, identical everything downstream — only the AbstractTrustModel
swaps. Races:
  CURATED05   fixed w = [0,0,0,.5,.5,.5,.5]        (reference)
  TRUST_EB    EB point weights (l05)               (the incumbent)
  TRUST_BAYES BayesianTrust posterior-mean weights (l06)
  EB_dist     EB, distributional staking (draws averaged)
  BAYES_dist  BayesianTrust, distributional staking

Also dumps the per-unit trust posteriors at the final refit (EB grid mean vs Bayes mean ± 90% CI)
so the EB-vs-Bayes gap (Bayes pools harder — see the smoke test) is quantified on real data.

Run (server): include preflight_real.jl, then this file.
=#
using BayesianFootball
include(joinpath(@__DIR__, "src", "loader.jl"))
isdefined(Main, :build_real_inputs) || include(joinpath(@__DIR__, "preflight_real.jl"))
using Printf, Statistics

function run_and_report_trust(inp; c::Float64=0.02, outdir=joinpath(@__DIR__, "results"))
    mkpath(outdir)
    src = RealSource(lat=inp.lat, ppd=inp.ppd, odds_bf=inp.odds_bf, matches_df=inp.ds1.matches, c=c)
    loaded = load_matches(src)
    bt = BayesianTrust(nsamples=800, nadapt=500, seed=20260707)

    policies = [
        "CURATED05"   => UnifiedPolicy(trust=CuratedTrust(), cap=0.2),
        "TRUST_EB"    => UnifiedPolicy(trust=EBTrust(),      cap=0.2),
        "TRUST_BAYES" => UnifiedPolicy(trust=bt,             cap=0.2),
        "EB_dist"     => UnifiedPolicy(trust=EBTrust(),      cap=0.2, distributional=true, D=64),
        "BAYES_dist"  => UnifiedPolicy(trust=bt,             cap=0.2, distributional=true, D=64),
    ]
    rs = run_race(loaded, policies; refit_every=25, seed=1)

    # end-of-season posteriors on the FULL history (EB vs Bayes side by side)
    hist = TrustHist()
    for (m, msel) in zip(loaded.matches, loaded.model_sel); push_hist!(hist, m, msel); end
    feb = fit_trust(EBTrust(), hist); fb = fit_trust(bt, hist)

    lines = String[]
    push!(lines, "TRUST-MODEL RACE (EB vs Bayesian) — src_sup40_sw40 · n=$(rs.n) · c=$c")
    push!(lines, ""); append!(lines, summary_rows(rs))
    push!(lines, ""); push!(lines, "END-OF-SEASON PER-UNIT TRUST (EB mean vs Bayes mean ± 90% CI):")
    push!(lines, @sprintf("%-10s %8s %8s %18s", "unit", "EB_w", "Bayes", "Bayes 90% CI"))
    for u in 1:7
        d = sort(fb.wdraws[u, :]); lo = d[max(1, round(Int, 0.05 * length(d)))]; hi = d[round(Int, 0.95 * length(d))]
        push!(lines, @sprintf("%-10s %8.3f %8.3f     [%.3f, %.3f]", UNIT_NAMES[u], feb.w[u], fb.w[u], lo, hi))
    end
    body = join(lines, "\n"); write(joinpath(outdir, "e_trust_models.txt"), body)
    println(body); return rs
end

inp = build_real_inputs()
run_and_report_trust(inp)
