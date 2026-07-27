#=
RUNNER r01 — simulated staking race (SimSource) through the unified harness.

A representative reproduction of the staking_sim E-series ordering on the new modular API: in a
world where the model has a genuine info edge on totals/BTTS but a BIASED, noisy supremacy (the
"sup-blind" regime that reproduces the real r10 1X2 bleed), the EB trust alarm should pull w down
on 1X2 and hold it on totals/BTTS, and CURATED ≻ TRUST_EB ≻ FLAT ≻ raw-U (which over-trusts the
bad 1X2 and bankrupts). This is a qualitative sanity check of the harness, not the authoritative
MC suite (that remains the staking_sim results until re-run cell-by-cell).

Run:  julia --project current_development/staking_layer/r01_sim_race.jl
=#
using BayesianFootball
include(joinpath(pkgdir(BayesianFootball), "current_development", "staking_layer", "src", "loader.jl"))
using Printf

# sup-blind world: good level (totals) info, junk supremacy → 1X2 should lose trust
cfg = SimConfig(n_matches=660, n_prehist=0, S=60,
                σ_mod_lvl=0.03, σ_mod_sup=0.14,      # tight level, noisy supremacy
                γ_tot=-0.08, γ_btts=0.12)            # per-line model bias (the tilt signature)
src = SimSource(cfg; seed=20260707)
loaded = load_matches(src)

policies = [
    "FLAT_1pct"   => FlatPolicy(),
    "PB_BK_cap02" => PerBetKellyPolicy(),
    "U_raw_cap02" => UnifiedPolicy(trust=FlatTrust(1.0),   cap=0.2),
    "TRUST05"     => UnifiedPolicy(trust=FlatTrust(0.5),   cap=0.2),
    "CURATED05"   => UnifiedPolicy(trust=CuratedTrust(),   cap=0.2),
    "TRUST_EB"    => UnifiedPolicy(trust=EBTrust(),        cap=0.2),
]
rs = run_race(loaded, policies; refit_every=30, seed=20260707)

outdir = joinpath(STAKING_LAYER_DIR, "results"); mkpath(outdir)
lines = String[]
push!(lines, "SIMULATED STAKING RACE — sup-blind world · n=$(rs.n) · SimSource(seed=20260707)")
push!(lines, "max_tilt_err = $(rs.max_tilt_err)")
push!(lines, ""); append!(lines, summary_rows(rs))
push!(lines, ""); push!(lines, "PER-FAMILY P/L:"); append!(lines, family_rows(rs))
push!(lines, ""); push!(lines, "EB TRUST w-TRAJECTORY:"); append!(lines, wtrace_rows(rs, "TRUST_EB"))
body = join(lines, "\n")
write(joinpath(outdir, "e_sim_race.txt"), body)
println(body)
println("\n[written] ", joinpath(outdir, "e_sim_race.txt"))
