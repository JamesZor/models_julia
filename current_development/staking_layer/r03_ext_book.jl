#=
RUNNER r03 — extended multi-market book on src_sup40_sw40 (all 7 families off the tilted grid).

Reproduces staking_real/results/e_ext_summary_c020.txt: the O/U ladder ADDS value (CS-excluded
CURATED ≈ 34× beats the core-11 ≈ 27×) but CorrectScore is a systematic drag (curate it out).
Each "strategy" is a trust model staked over the full ExtMatch book; `drop_fams` curates families.

Run (server): include preflight_real.jl, then this file.
=#
using BayesianFootball
include(joinpath(@__DIR__, "src", "loader.jl"))
isdefined(Main, :build_real_inputs) || include(joinpath(@__DIR__, "preflight_real.jl"))
using Printf

const CS = FAM_ID["CorrectScore"]

function run_and_report_ext(inp, c::Float64; outdir=joinpath(@__DIR__, "results"))
    mkpath(outdir)
    src = RealSource(lat=inp.lat, ppd=inp.ppd, odds_bf=inp.odds_bf, matches_df=inp.ds1.matches, c=c)
    ext = build_ext_books(src)

    specs = ["FLAT05"=>FlatTrust(0.5), "CURATED05"=>CuratedTrust(), "TRUST_EB"=>EBTrust()]
    rs_full = run_ext_race(ext, specs; cap=0.2, refit_every=25)                 # all families
    rs_nocs = run_ext_race(ext, specs; cap=0.2, refit_every=25, drop_fams=Set([CS]))  # CS curated out

    tag = @sprintf("c%03d", round(Int, c * 1000))
    lines = String[]
    push!(lines, "EXTENDED-BOOK RACE — src_sup40_sw40 · Ireland 2025-26 · n=$(rs_full.n) · c=$c")
    fam_present = sort(unique(vcat([em.fam for em in ext.matches]...)))
    push!(lines, "families present: " * join([FAM_LABEL[f] for f in fam_present], ", "))
    for (label, rs) in (("FULL BOOK (all families)", rs_full), ("CS-EXCLUDED", rs_nocs))
        push!(lines, ""); push!(lines, "="^60); push!(lines, label); push!(lines, "="^60)
        push!(lines, @sprintf("%-14s %12s %10s %8s", "strategy", "term_W", "G/match", "ruined"))
        for nm in rs.names
            sm = summarize_logw(rs.logw[nm])
            push!(lines, @sprintf("%-14s %12.4f %+10.5f %8s", nm, sm.terminal_W, sm.G_per_match, rs.ruined[nm] ? "YES" : "-"))
        end
        push!(lines, ""); push!(lines, @sprintf("%-14s %-14s %9s %9s", "strategy", "family", "profit", "turnover"))
        for nm in rs.names, f in fam_present
            t = rs.fam_turn[(nm, f)]; t <= 0 && continue
            push!(lines, @sprintf("%-14s %-14s %+9.4f %9.4f", nm, FAM_LABEL[f], rs.fam_profit[(nm, f)], t))
        end
    end
    body = join(lines, "\n"); write(joinpath(outdir, "e_ext_summary_$(tag).txt"), body)
    println(body); return (rs_full, rs_nocs)
end

inp = build_real_inputs()
run_and_report_ext(inp, 0.02)
run_and_report_ext(inp, 0.0)
