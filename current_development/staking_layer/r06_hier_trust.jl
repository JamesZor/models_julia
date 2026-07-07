#=
RUNNER r06 — hierarchical per-team trust (roadmap step 3) on the real src_sup40_sw40 book.

Two questions:
  (1) DIAGNOSIS — does per-unit trust genuinely vary by team? The σ_u posterior (between-team
      spread) is the principled version of r05's noisy across-team std: σ_u ≈ 0 ⇒ no real team
      variation on unit u (r05's spread was 28-obs noise); σ_u > 0 ⇒ that unit's trust differs by
      team. Expectation from r05: σ up on home/away (1X2), ≈0 on totals/BTTS.
  (2) DOES IT PAY? — race CURATED vs TRUST_EB vs HYBRID (EB totals/BTTS, 1X2 hard-abstained) vs
      TRUST_HIER (+ distributional). The hierarchy thesis: beat CURATED by trusting the model's
      1X2 only for teams where it's calibrated (r05: Bohemian high) and abstaining elsewhere.

Hierarchical NUTS is refit on a COARSE cadence (default every 50) to bound cost (~a handful of
fits). Run (server): include preflight_real.jl, then this file.
=#
using BayesianFootball
include(joinpath(pkgdir(BayesianFootball), "current_development", "staking_layer", "src", "loader.jl"))
isdefined(Main, :build_real_inputs) || include(joinpath(STAKING_LAYER_DIR, "preflight_real.jl"))
using Printf, Statistics

function run_hier(inp; c::Float64=0.02, refit_hier::Int=50, outdir=joinpath(STAKING_LAYER_DIR, "results"))
    mkpath(outdir)
    src = RealSource(lat=inp.lat, ppd=inp.ppd, odds_bf=inp.odds_bf, matches_df=inp.ds1.matches, c=c)
    loaded = load_matches(src); names = loaded.team_names

    # (1) end-of-season diagnosis on the full history
    hist = TrustHist()
    for (m, msel) in zip(loaded.matches, loaded.model_sel); push_hist!(hist, m, msel); end
    ht = HierarchicalTrust(nsamples=1000, nadapt=600, seed=20260707)
    fh = fit_trust(ht, hist)

    lines = String[]
    push!(lines, "HIERARCHICAL PER-TEAM TRUST — src_sup40_sw40 · Ireland · n=$(length(loaded.matches)) · c=$c")
    push!(lines, "")
    push!(lines, "(1) BETWEEN-TEAM SPREAD σ_u  — is there real per-team signal? (σ≈0 ⇒ no)")
    push!(lines, @sprintf("%-10s %10s %10s", "unit", "w0(pooled)", "σ_u"))
    for u in 1:7
        push!(lines, @sprintf("%-10s %10.3f %10.3f", UNIT_NAMES[u], fh.pooled_w[u], fh.σ[u]))
    end
    # per-team home-unit trust, shrunk by the hierarchy (compare to r05's unpooled fits)
    push!(lines, ""); push!(lines, "per-team HOME-unit trust (hierarchically shrunk):")
    order = sortperm(fh.wmean[1, :], rev=true)
    for t in order
        push!(lines, @sprintf("  %-22s home_w=%.3f  away_w=%.3f", names[fh.team_names_dense[t]], fh.wmean[1, t], fh.wmean[3, t]))
    end

    # (2) race
    policies = [
        "CURATED05"  => UnifiedPolicy(trust=CuratedTrust(), cap=0.2),
        "TRUST_EB"   => UnifiedPolicy(trust=EBTrust(), cap=0.2),
        "HYBRID"     => UnifiedPolicy(trust=OverrideTrust(EBTrust(), Dict(1=>0.0, 2=>0.0, 3=>0.0)), cap=0.2),
        "TRUST_HIER" => UnifiedPolicy(trust=ht, cap=0.2),
        "HIER_dist"  => UnifiedPolicy(trust=ht, cap=0.2, distributional=true, D=64),
    ]
    rs = run_race(loaded, policies; refit_every=refit_hier, seed=1)
    push!(lines, ""); push!(lines, "(2) RACE (hier refit every $refit_hier):"); append!(lines, summary_rows(rs))

    body = join(lines, "\n"); write(joinpath(outdir, "e_hier_trust.txt"), body)
    println(body); return (fh=fh, rs=rs)
end

inp = build_real_inputs()
run_hier(inp)
