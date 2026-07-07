#=
RUNNER r05 — Step-0 EDA: does per-unit trust w vary by TEAM?

Before building the hierarchical trust model (roadmap step 3), confirm there is team-level signal
to model. Accumulate the full core-book TrustHist, then for each team fit EBTrust on the SUBSET of
observations where that team played (home perspective) and compare the per-unit w across teams.

If the per-team home-w's are tightly clustered around the pooled value, a hierarchical model will
(correctly) shrink them back — worth knowing before the modelling cost. If they spread, hierarchy
pays. This mirrors r04's "is the bias stationary?" question, but across teams instead of time.

Run (server): include preflight_real.jl, then this file.
=#
using BayesianFootball
include(joinpath(@__DIR__, "src", "loader.jl"))
isdefined(Main, :build_real_inputs) || include(joinpath(@__DIR__, "preflight_real.jl"))
using Printf, Statistics

"Filter a TrustHist to observations where `team` appears as home (per unit)."
function filter_hist_home(h::TrustHist, team::Int)
    g = TrustHist()
    for u in 1:7, i in eachindex(h.y[u])
        h.home[u][i] == team || continue
        push!(g.p[u], h.p[u][i]); push!(g.q[u], h.q[u][i]); push!(g.y[u], h.y[u][i])
        push!(g.home[u], h.home[u][i]); push!(g.away[u], h.away[u][i])
    end
    return g
end

function run_team_eda(inp; c::Float64=0.02, outdir=joinpath(@__DIR__, "results"))
    mkpath(outdir)
    src = RealSource(lat=inp.lat, ppd=inp.ppd, odds_bf=inp.odds_bf, matches_df=inp.ds1.matches, c=c)
    loaded = load_matches(src)
    names = loaded.team_names

    hist = TrustHist()
    for (m, msel) in zip(loaded.matches, loaded.model_sel); push_hist!(hist, m, msel); end
    pooled = fit_trust(EBTrust(), hist)

    rows = Tuple{String,Int,Vector{Float64}}[]
    for (tid, tname) in enumerate(names)
        gh = filter_hist_home(hist, tid)
        n = nobs(gh); n == 0 && continue
        fh = fit_trust(EBTrust(), gh)
        push!(rows, (tname, n, fh.w))
    end
    sort!(rows, by=r -> -r[2])

    lines = String[]
    push!(lines, "TEAM-LEVEL TRUST EDA — src_sup40_sw40 · Ireland · does w vary by team? (home matches)")
    push!(lines, @sprintf("pooled EB w: %s", join([@sprintf("%s=%.2f", UNIT_NAMES[u], pooled.w[u]) for u in 1:7], "  ")))
    push!(lines, ""); push!(lines, @sprintf("%-22s %5s  %s", "team (home)", "n", join([@sprintf("%9s", u) for u in UNIT_NAMES], "")))
    for (tname, n, w) in rows
        push!(lines, @sprintf("%-22s %5d  %s", tname, n, join([@sprintf("%9.2f", x) for x in w], "")))
    end
    # spread summary: std across teams per unit (signal for hierarchy)
    push!(lines, "")
    W = reduce(hcat, [w for (_, _, w) in rows])   # 7 × n_teams
    push!(lines, @sprintf("%-22s %5s  %s", "across-team std", "", join([@sprintf("%9.3f", std(W[u, :])) for u in 1:7], "")))
    push!(lines, "READ: a large across-team std on a unit ⇒ hierarchical w would separate that unit; " *
                 "small std ⇒ pooling dominates (hierarchy shrinks it back).")
    body = join(lines, "\n"); write(joinpath(outdir, "e_team_eda.txt"), body)
    println(body)
end

inp = build_real_inputs()
run_team_eda(inp)
