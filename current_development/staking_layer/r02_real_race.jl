#=
RUNNER r02 — real-data staking race on src_sup40_sw40 (Ireland 2025-26 OOS, 275 Betfair-close).

The PARITY runner: reproduces staking_real/results/e_real_summary_c020.txt on the new modular API.
Expected (c=0.02): CURATED05 term_W ≈ 26.9, TRUST_EB ≈ 3.0 with home w → 0.18 / away → 0.33 and
totals/BTTS held ≈ 0.5, U_raw (w=1) RUINED, b21 sign agreement 11/11, max_tilt_err < 1e-6.

Run (server / kaimon, after the payload is present):
    include("current_development/staking_layer/preflight_real.jl")
    include("current_development/staking_layer/r02_real_race.jl")
=#
using BayesianFootball
include(joinpath(pkgdir(BayesianFootball), "current_development", "staking_layer", "src", "loader.jl"))
isdefined(Main, :build_real_inputs) || include(joinpath(STAKING_LAYER_DIR, "preflight_real.jl"))
using Printf

# b21 per-selection ROI% (src_sup40_sw40, BayesianKelly, Betfair) — adapter cross-check target.
const B21_ROI = Dict("home"=>-9.4, "draw"=>33.54, "away"=>22.69,
                     "over_15"=>-7.5, "under_15"=>42.02, "over_25"=>1.75, "under_25"=>12.21,
                     "over_35"=>16.03, "under_35"=>9.0, "btts_yes"=>32.6, "btts_no"=>48.22)

real_policies() = [
    "FLAT_1pct"        => FlatPolicy(),
    "PB_BK_cap02"      => PerBetKellyPolicy(),
    "U_raw_cap02"      => UnifiedPolicy(trust=FlatTrust(1.0), cap=0.2),
    "TRUST05_cap02"    => UnifiedPolicy(trust=FlatTrust(0.5), cap=0.2),
    "CURATED05_cap02"  => UnifiedPolicy(trust=CuratedTrust(), cap=0.2),
    "TRUST_EB_cap02"   => UnifiedPolicy(trust=EBTrust(),      cap=0.2),
]

"Run + report the real race at commission c; writes results/e_real_summary_cXXX.txt + w_trace CSV."
function run_and_report_real(inp, c::Float64; outdir=joinpath(STAKING_LAYER_DIR, "results"))
    mkpath(outdir)
    src = RealSource(lat=inp.lat, ppd=inp.ppd, odds_bf=inp.odds_bf, matches_df=inp.ds1.matches, c=c)
    loaded = load_matches(src)
    rs = run_race(loaded, real_policies(); refit_every=25, seed=1)

    tag = @sprintf("c%03d", round(Int, c * 1000))
    lines = String[]
    push!(lines, "REAL-DATA STAKING RACE — src_sup40_sw40 · Ireland 2025-26 · n=$(rs.n)")
    push!(lines, "commission c = $c   ·   smile w=1 tilt max|Δover-prob| = $(rs.max_tilt_err)")
    push!(lines, ""); append!(lines, summary_rows(rs))
    push!(lines, ""); push!(lines, "PER-FAMILY P/L:"); append!(lines, family_rows(rs))
    push!(lines, ""); push!(lines, "EB TRUST w-TRAJECTORY:"); append!(lines, wtrace_rows(rs, "TRUST_EB_cap02"))
    push!(lines, ""); push!(lines, "b21 CROSS-CHECK (PB_BK_cap02):"); append!(lines, crosscheck_rows(rs, "PB_BK_cap02", B21_ROI))
    body = join(lines, "\n"); write(joinpath(outdir, "e_real_summary_$(tag).txt"), body)

    io = IOBuffer(); println(io, "match," * join(UNIT_NAMES, ","))
    for (i, w) in rs.w_trace["TRUST_EB_cap02"]; println(io, string(i) * "," * join(string.(round.(w, digits=5)), ",")); end
    write(joinpath(outdir, "w_trace_$(tag).csv"), String(take!(io)))
    println(body); return rs
end

inp = build_real_inputs()
run_and_report_real(inp, 0.02)
run_and_report_real(inp, 0.0)
