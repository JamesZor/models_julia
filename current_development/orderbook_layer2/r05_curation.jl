# current_development/orderbook_layer2/r05_curation.jl
#
# WP5. Market curation, per-line trust, and matches to avoid.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS IS THE TRUST EXPERIMENT AND THE FLAT RACE IS NOT
# ---------------------------------------------------------------------------------------------
#
# `src/Portfolio/stake.jl:5-18` — `risk_factor` is homogeneous of degree 0. Once `SlateDrawdown`
# binds, multiplying every stake by a constant changes nothing: the risk model simply scales the
# book back up. Measured, trust 0.25 / 0.5 / 1.0 give identical wealth.
#
# So a flat-trust ladder is a provable no-op under the reference policy, and the only trust
# intervention that can do anything is a DIFFERENTIAL one — weights that differ across markets,
# which reshape the book rather than resize it. That is what this file measures.
#
# ---------------------------------------------------------------------------------------------
# PRE-REGISTRATION
# ---------------------------------------------------------------------------------------------
#
#   C1  Per-family skill is NOT uniform. 1X2 is the worst family and totals/BTTS the best.
#       Prior: the staking-layer stream's curated weights (0 on 1X2, ~0.5 on totals/BTTS) beat
#       both empirical-Bayes and flat weights, and r21 measured home `hurdle_G` at -0.042.
#
#   C2  Favourite-longshot: skill falls with price. Long selections are systematically overpriced,
#       so the model's positive claims on them are the market's margin, not information.
#       Prior: skipping odds above ~6 helps.
#
#   C3  **The tail test.** Skill does NOT fall monotonically with the size of the model's claimed
#       disagreement. The house position is that per-line expectation should sit on the market and
#       the edge lives in per-match deviations, so the tails should carry the information. The
#       competing reading — that a large disagreement is a large model error — predicts skill
#       collapsing in the outer edge bands. This is the cut that separates them, and it is the
#       one whose answer would change how the whole book is built.
#
#   C4  Wide-book matches are worse. A large mean relative spread marks a match nobody has priced
#       confidently, and it is where the margin is largest.
#
# Stated before running, because with 267 legs per league the difference between a finding and a
# story is whether it was written down first.
#
# ---------------------------------------------------------------------------------------------
# WHAT WOULD FALSIFY THE WHOLE EXERCISE
# ---------------------------------------------------------------------------------------------
#
# If per-family skill in 79 does not predict per-family skill in 718 — that is, if the curation
# derived on one league does nothing or hurts on the other — then there is no stable structure to
# curate on and per-market trust is not the lever. That is a real possible outcome and it is
# reported as the headline if it happens, not buried.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server; needs WP4's saved ledgers — no database, no Tier 1 rebuild)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/orderbook_layer2/r05_curation.jl")

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio

include(joinpath(@__DIR__, "l01_l2_experiment.jl"))
include(joinpath(@__DIR__, "l02_l2_ledger.jl"))
include(joinpath(@__DIR__, "l03_l2_metrics.jl"))
include(joinpath(@__DIR__, "l05_curation.jl"))

const LEDGER_DIR = "./data/l2_entry_timing"
const OUT_DIR    = "./data/l2_curation"

# ===================================================================
# Load — AtClose only, per WP4
# ===================================================================

"""
Load one league's staked ledger at the entry rule WP4 settled on.

`AtClose` only: WP4 measured CLV monotone in entry time across both leagues, so every other rung
is a strictly worse execution of the same opinion and mixing them would confound curation with
timing.
"""
function load_close(tag::String)
    p = joinpath(LEDGER_DIR, "$(tag)_entry.jls")
    isfile(p) || error("r05: missing $p — run r04_entry_timing.jl first")
    led = deserialize(p).ledger
    hasproperty(led, :entry_name) || error("r05: ledger has no :entry_name — re-run r04")
    d = copy(filter(r -> r.entry_name == "AtClose", led))
    return annotate!(d)
end

banner(s) = (println("\n", "="^92); println(s); println("="^92))

function show_df(title, df; n = 30)
    println("\n", title)
    isempty(df) ? println("  (empty)") :
        show(stdout, MIME"text/plain"(), first(df, min(n, nrow(df))))
    println()
end

# ===================================================================
# Per-league report
# ===================================================================

function league_report(tag::String, d::DataFrame)
    banner("WP5 CURATION — $tag   ($(nrow(d)) legs, $(length(unique(d.match_id))) matches)")

    u = usable(d)
    s = leg_skill(u)
    ci = _cluster_boot(s, u.match_id)
    @printf("\nOVERALL skill vs de-vigged close: %+.5f nats/leg  [%.5f, %.5f]   beats market on %.1f%% of legs\n",
            mean(s), ci.lo, ci.hi, 100 * count(>(0), s) / length(s))
    println("  (positive => the model knows something the closing market did not)")

    fam = skill_table(d, [:family])
    show_df("[C1] skill by family — the trust vector's evidence", fam)

    mkt = skill_table(d, [:market])
    show_df("[C1b] skill by market", mkt)

    ob = sort!(skill_table(d, [:odds_band]; min_legs = 5), :odds_band)
    show_df("[C2] skill by odds band — favourite-longshot", ob)

    eb = sort!(skill_table(d, [:edge_band]; min_legs = 5), :edge_band)
    show_df("[C3] skill by claimed disagreement — THE TAIL TEST", eb)

    m = match_table(d)
    show_df("[C4] worst and best matches", vcat(first(m, 5), last(m, 5)); n = 10)
    for c in (:spread, :max_claim, :longshot, :legs)
        show_df("[C4] matches by $c tercile", tercile_cut(m, c))
    end
    return (family = fam, market = mkt, odds = ob, edge = eb, matches = m)
end

# ===================================================================
# Cross-league validation
# ===================================================================

"""
    cross_validate(from_tag, from_df, to_tag, to_df) -> NamedTuple

Derive the trust vector on one league, apply it to the other, and report the held-out result.

This is the only number in the file that is not in-sample. Everything above chooses families using
the same legs it then scores them on; this does not.
"""
function cross_validate(from_tag, from_df, to_tag, to_df)
    banner("CROSS-LEAGUE: curation derived on $from_tag, tested on $to_tag")

    trust = derive_trust(from_df)
    ranked = sort([(k, v) for (k, v) in trust.table], by = x -> -x[2])
    println("\nderived trust vector ($(length(trust.table)) keyed selections, default $(trust.default)):")
    for (k, v) in ranked
        @printf("   %-22s line %5.1f  %-10s  w = %.2f\n", k[1], k[2], String(k[3]), v)
    end

    base = usable(to_df)
    hedged = apply_trust_oos(trust, to_df)
    isempty(hedged) && (println("\n  held-out ledger empty"); return (;))

    bs, hs = sum(base.stake), sum(hedged.stake)
    bp, hp = sum(base.pnl),   sum(hedged.pnl)
    @printf("\nheld-out (%s):\n", to_tag)
    @printf("  uncurated   stake %.4f   pnl %+.5f   roi %+.2f%%\n", bs, bp, 100 * bp / bs)
    @printf("  curated     stake %.4f   pnl %+.5f   roi %+.2f%%\n", hs, hp, 100 * hp / hs)
    @printf("  exposure kept %.1f%%,  recapped on %d legs\n",
            100 * hs / bs, hasproperty(hedged, :recapped) ? count(hedged.recapped) : 0)

    # Does the FAMILY RANKING transfer? That is the claim curation rests on, and it is cheaper to
    # believe than the P&L number, which inherits all of ROI's noise.
    a = skill_table(from_df, [:family]); b = skill_table(to_df, [:family])
    j = innerjoin(select(a, :family, :skill => :skill_from, :legs => :legs_from),
                  select(b, :family, :skill => :skill_to, :legs => :legs_to), on = :family)
    j = filter(r -> r.legs_from >= 8 && r.legs_to >= 8, j)
    if nrow(j) >= 4
        rho = cor(j.skill_from, j.skill_to)
        agree = count(r -> sign(r.skill_from) == sign(r.skill_to), eachrow(j))
        @printf("\nfamily skill transfer: r = %+.3f over %d families, sign agrees on %d/%d\n",
                rho, nrow(j), agree, nrow(j))
        show_df("  per-family, both leagues", sort(j, :skill_from, rev = true))
        println(rho > 0.3 ?
                "  => ranking transfers; curation has something stable to stand on" :
                "  => ranking does NOT transfer; per-market trust is not the lever here")
    else
        println("\n  too few shared families with >=8 legs to test transfer")
    end
    return (trust = trust, held_out = hedged, transfer = j)
end

# ===================================================================
# Run
# ===================================================================

banner("WP5 — market curation and match avoidance on the Ireland order book")
println("Estimator: skill = logscore(p_model) - logscore(fair_close), per leg, in nats.")
println("CLV is degenerate at the close (it equals minus the margin) and ROI cannot resolve")
println("15 families at 267 legs, so neither is used to choose anything here.")

d79  = load_close("ire79")
d718 = load_close("ire718")

r79  = league_report("ire79",  d79)
r718 = league_report("ire718", d718)

cv_a = cross_validate("ire79",  d79,  "ire718", d718)
cv_b = cross_validate("ire718", d718, "ire79",  d79)

mkpath(OUT_DIR)
serialize(joinpath(OUT_DIR, "curation.jls"),
          (ire79 = r79, ire718 = r718, cv_79_to_718 = cv_a, cv_718_to_79 = cv_b))

banner("WP5 complete — results in $OUT_DIR")
println("Read in this order: the transfer correlation first (does anything generalise), then")
println("[C3] the tail test, then [C1]. A family that only works in the league it was chosen in")
println("is not a finding.")
