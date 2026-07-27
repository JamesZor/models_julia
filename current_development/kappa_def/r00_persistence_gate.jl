#=
r00 — PERSISTENCE GATE (EDA, no MCMC).

Question: does a team's xG→goals conversion residual PERSIST, per team, on the defensive
side? If not, κ_def is a free parameter for noise and the stream stops here
(the [[hierarchical-smile-sigma-null]] lesson).

Per match with xG present, two rows (one per team):
    att_res = goals_scored   − xg_for        (finishing above chance quality)
    def_res = goals_conceded − xg_against    (suppression beyond chance quality: keeper/blocks)

Two persistence reads, attack and defense separately:
  A. WITHIN-SEASON SPLIT-HALF — per team-season (≥ MIN_MATCHES), chronological odd/even
     halves; correlate half-1 vs half-2 team means across team-seasons. Permutation p
     (shuffle half-2 labels) — the honest small-n test.
  B. SEASON t → t+1 — per team-season means, correlate consecutive seasons per team.

Attack is the reference: finishing skill is known to be weakly persistent; defense must
clear the same bar to justify κ_def.

Run (local or server — cheap):
    include("current_development/kappa_def/r00_persistence_gate.jl")
=#

using BayesianFootball
using DataFrames
using Statistics
using Random
using Dates

const Data = BayesianFootball.Data

const MIN_MATCHES = 10      # per team-season for the split-half read
const N_PERM      = 4000

_season_of(m) = hasproperty(m, :season) ? string(m.season) : string(year(m.match_date))

"long residual table: one row per (team, match): att_res, def_res"
function residual_table(ds)
    xg = Dict(r.match_id => (r.expectedGoals_home, r.expectedGoals_away)
              for r in eachrow(ds.statistics) if r.period == "ALL")
    rows = NamedTuple[]
    for m in eachrow(ds.matches)
        haskey(xg, m.match_id) || continue
        xgh, xga = xg[m.match_id]
        (ismissing(xgh) || ismissing(xga)) && continue
        s = _season_of(m)
        push!(rows, (team=String(m.home_team), season=s, date=m.match_date,
                     att=Float64(m.home_score) - Float64(xgh),
                     def=Float64(m.away_score) - Float64(xga)))
        push!(rows, (team=String(m.away_team), season=s, date=m.match_date,
                     att=Float64(m.away_score) - Float64(xga),
                     def=Float64(m.home_score) - Float64(xgh)))
    end
    return DataFrame(rows)
end

"split-half correlation across team-seasons + permutation p-value"
function split_half(df, col)
    h1 = Float64[]; h2 = Float64[]
    for g in groupby(df, [:team, :season])
        nrow(g) < MIN_MATCHES && continue
        gs = sort(g, :date)
        odd  = gs[1:2:end, col]; even = gs[2:2:end, col]
        push!(h1, mean(odd)); push!(h2, mean(even))
    end
    n = length(h1)
    n < 8 && return (n=n, cor=NaN, p=NaN)
    c = cor(h1, h2)
    rng = Xoshiro(42)
    perm = count(_ -> abs(cor(h1, shuffle(rng, h2))) >= abs(c), 1:N_PERM) / N_PERM
    return (n=n, cor=round(c, digits=3), p=round(perm, digits=4))
end

"season t vs t+1 correlation of team-season means + permutation p"
function season_over_season(df, col)
    ts = combine(groupby(df, [:team, :season]), col => mean => :res, nrow => :n)
    ts = ts[ts.n .>= MIN_MATCHES, :]
    seasons = sort(unique(ts.season))
    x = Float64[]; y = Float64[]
    for i in 1:length(seasons)-1
        a = ts[ts.season .== seasons[i],   [:team, :res]]
        b = ts[ts.season .== seasons[i+1], [:team, :res]]
        j = innerjoin(a, b, on=:team, makeunique=true)
        append!(x, j.res); append!(y, j.res_1)
    end
    n = length(x)
    n < 8 && return (n=n, cor=NaN, p=NaN)
    c = cor(x, y)
    rng = Xoshiro(43)
    perm = count(_ -> abs(cor(x, shuffle(rng, y))) >= abs(c), 1:N_PERM) / N_PERM
    return (n=n, cor=round(c, digits=3), p=round(perm, digits=4))
end

# ==========================================
# RUN over segments
# ==========================================
segments = [Data.Ireland(), Data.IrelandFirstDivision()]

results = DataFrame(segment=String[], read=String[], side=String[],
                    n=Int[], cor=Float64[], p=Float64[])
for seg in segments
    tag = string(nameof(typeof(seg)))
    println("\n", "="^60, "\n  $tag\n", "="^60)
    ds = Data.load_datastore_cached(seg)
    df = residual_table(ds)
    println("matches with xG: $(nrow(df) ÷ 2)   team-seasons: ",
            nrow(combine(groupby(df, [:team, :season]), nrow => :n)))
    for (read, f) in [("split_half", split_half), ("season_t_t1", season_over_season)]
        for side in [:att, :def]
            r = f(df, side)
            push!(results, (tag, read, String(side), r.n, r.cor, r.p))
            println(rpad(read, 14), rpad(String(side), 5),
                    " n=", rpad(r.n, 5), " cor=", rpad(r.cor, 8), " p=", r.p)
        end
    end
end

println("\n", "█"^60, "\n  GATE VERDICT\n", "█"^60)
show(results; allrows=true, allcols=true)
println("""

[READ]
 • DEFENSE rows are the gate: cor > 0 with p < 0.10 on either read (either league) ⇒
   a team-specific defensive conversion residual is real ⇒ build/keep κ_def (:attdef, and
   :net as the cheap version). Record the verdict in EXPERIMENTS.md.
 • Both defense reads ≈ 0 ⇒ κ_def is a parameter for noise; the attack-only V0 is already
   the right model. Park the stream (r01 still worth one run as a convergence exercise).
 • ATTACK rows are the reference: if even attack doesn't persist here, the leagues are too
   thin/noisy for ANY per-team conversion latent and κ should arguably shrink harder.
""")
