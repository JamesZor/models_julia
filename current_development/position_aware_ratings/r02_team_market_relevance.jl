#=
r02_team_market_relevance.jl — Option-1 gate: does off-position deployment beat the MARKET on totals?

r01 showed a real per-player out-of-position penalty on ATTACKING output (shots-on-target/xG/goals) that
the rating hides. This asks the only question that matters for betting (per RESEARCH_role_aware_ideas.md):
does that translate to a TEAM-level signal carrying information BEYOND the de-vigged market on totals?

Pre-game signal (known at lineup time): per team, the COUNT of starters deployed off their modal position
(`n_off`), and the attacking-roles subset (`n_off_att`, played at M/F — where r01 located the penalty).
Per match, sum both teams. More off-position attackers ⇒ less attacking sharpness ⇒ fewer goals ⇒ the
OVER should hit LESS than the market implies. So the GLMEdge test:

    logit(is_winner_over) ~ market_logit + off_index          (de-vigged market as the control)

A NEGATIVE, significant coefficient on `off_index` for OVER lines = the signal predicts UNDER beyond the
market = a real edge. ≈0 / wrong sign = the market already prices lineup positions (the deep-research
caveat: lineup edges rarely survive the close) → no Option-1 build; move to Option 2 (RAPM).
Secondary: in-sample per-line LogLoss(market+signal) vs LogLoss(market) — indicative only (in-sample).

Server REPL: git pull, restart, then
    include("current_development/position_aware_ratings/r02_team_market_relevance.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using Printf
using GLM

include("current_development/position_aware_ratings/l00_position_helpers.jl")

const Data = BayesianFootball.Data

const SEGMENTS = [
    Data.Ireland(), Data.IrelandFirstDivision(), Data.SouthKorea(),
    Data.Norway(), Data.Veikkausliiga(), Data.ScottishLower(),
]
const MIN_APPS   = 5
const OVER_LINES = [:over_15, :over_25, :over_35]   # most balanced / liquid totals
seg_name(s) = string(nameof(typeof(s)))
_logit(p) = log(p / (1 - p))
_clampp(p) = clamp(p, 1e-6, 1 - 1e-6)
logloss(y, p) = -mean(y .* log.(_clampp.(p)) .+ (1 .- y) .* log.(1 .- _clampp.(p)))

# match-level off-position index from the prepared starter frame (needs is_off_modal from l00)
function match_off_index(df::DataFrame)
    d = copy(df)
    d.offmod = coalesce.(d.is_off_modal, false)                  # cold-start / unknown -> not off-modal
    d.att    = d.pos_is_real .& [p in ("M", "F") for p in d.pos_eda]
    tidx = combine(groupby(d, [:match_id, :team_side]),
                   :offmod => sum => :n_off,
                   [:offmod, :att] => ((o, a) -> sum(o .& a)) => :n_off_att)
    home = tidx[tidx.team_side .== "home", [:match_id, :n_off, :n_off_att]]
    away = tidx[tidx.team_side .== "away", [:match_id, :n_off, :n_off_att]]
    m = innerjoin(home, away, on = :match_id, renamecols = "_h" => "_a")
    m.off_total     = m.n_off_h     .+ m.n_off_a
    m.off_att_total = m.n_off_att_h .+ m.n_off_att_a
    return select(m, :match_id, :off_total, :off_att_total)
end

results = DataFrame(league=String[], line=Symbol[], signal=Symbol[], n=Int[],
                    coef=Float64[], z=Float64[], p=Float64[], dll=Float64[])

for seg in SEGMENTS
    name = seg_name(seg)
    println("\n", "#"^84, "\n# LEAGUE: $name\n", "#"^84)
    ds = try
        Data.load_datastore_cached(seg)
    catch e; println("[SKIP] load failed: $(sprint(showerror, e))"); continue; end

  try
    df = prepare_starter_lineups(ds; from_year=2023)
    if nrow(df) == 0; println("[SKIP] no 2023+ starters."); continue; end
    add_modal_position!(df; min_apps=MIN_APPS)
    midx = match_off_index(df)
    @printf("  matches with index=%d   off_total mean=%.2f (sd %.2f)   off_att_total mean=%.2f\n",
            nrow(midx), mean(midx.off_total), std(midx.off_total), mean(midx.off_att_total))

    ou = ds.odds[(ds.odds.market_name .== "OverUnder"), :]
    if nrow(ou) == 0; println("[SKIP] no OverUnder odds."); continue; end

    tbl = DataFrame(line=Symbol[], signal=Symbol[], n=Int[],
                    coef=Float64[], z=Float64[], p=Float64[], dll=Float64[])
    for line in OVER_LINES
        sel = ou[ou.selection .== line, [:match_id, :prob_fair_close, :is_winner]]
        nrow(sel) == 0 && continue
        dropmissing!(sel, [:prob_fair_close, :is_winner])
        d = innerjoin(sel, midx, on = :match_id)
        nrow(d) < 50 && continue
        d.y   = Float64.(d.is_winner)
        d.mkt = _logit.(_clampp.(Float64.(d.prob_fair_close)))
        ll_mkt = logloss(d.y, Float64.(d.prob_fair_close))

        for sig in (:off_total, :off_att_total)
            d.sig = Float64.(d[!, sig])
            std(d.sig) == 0 && continue
            m = glm(@formula(y ~ mkt + sig), d, Binomial(), LogitLink())
            ct = coeftable(m); r = findfirst(==("sig"), ct.rownms)
            pred = predict(m)
            dll  = round(logloss(d.y, pred) - ll_mkt, digits=5)   # <0 = improves on market (in-sample)
            row = (line, sig, nrow(d), round(ct.cols[1][r], digits=4),
                   round(ct.cols[3][r], digits=2), round(ct.cols[4][r], digits=4), dll)
            push!(tbl, row); push!(results, (name, row...))
        end
    end
    if isempty(tbl); println("  (no totals line had ≥50 matched matches)"); else
        println("\n  logit(over hit) ~ market_logit + signal   [coef<0 & |z|≥2 ⇒ off-position predicts UNDER beyond market]")
        show(tbl; allrows=true, allcols=true, truncate=0); println()
    end
  catch e
    println("[ERROR] $name failed: $(sprint(showerror, e))")
  end
end

# ============================================================================
# CROSS-LEAGUE SUMMARY
# ============================================================================
println("\n", "="^84, "\nCROSS-LEAGUE SUMMARY (signal coef on OVER outcome, market controlled)\n", "="^84)
if nrow(results) == 0
    println("no results.")
else
    g = combine(groupby(results, [:line, :signal]),
                :coef => (x -> round(mean(x), digits=4)) => :mean_coef,
                :z    => (x -> round(mean(x), digits=2)) => :mean_z,
                nrow  => :n_leagues,
                :z    => (x -> count(<=(-2), x)) => :n_neg_sig,
                :z    => (x -> count(>=(2),  x)) => :n_pos_sig,
                :dll  => (x -> round(mean(x), digits=5)) => :mean_dll)
    sort!(g, :mean_z)
    show(g; allrows=true, allcols=true, truncate=0); println()
end

println("""

$("="^84)
VERDICT:
 • Edge if a signal shows mean_coef<0, mean_z≤−2, negative & significant on MOST leagues (n_neg_sig),
   AND mean_dll<0 → off-position deployment predicts UNDER beyond the market → BUILD the Option-1 feature
   (fold the off-position attacking discount into the xG/goals engine; A/B per-line LogLoss + GLMEdge).
 • No edge if coefs ≈0 / mixed sign / |z|<2 → the market already prices public lineup positions
   (expected per deep-research) → drop Option-1, move to Option 2 (prior-informed Bayesian RAPM).
$("="^84)
""")
