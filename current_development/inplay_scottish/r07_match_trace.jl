#=
r07_match_trace.jl — high-resolution single-match trace: in-game λ and the full book,
model vs Betfair, minute by minute.

r06 evaluated on the 5-minute Betfair binning grid, which is the resolution the EXCHANGE
supports, not the resolution the MODEL supports. The model is a closed-form function of
(pregame λ, minute, score, red cards) — it can be evaluated anywhere. So here it is stepped
at 1 minute, with Betfair overlaid wherever there is a genuinely two-sided quote.

Emits, per match minute:
  * instantaneous goal rate λ_h(t), λ_a(t)  — the "in-game lambda rate" itself
  * integrated remaining intensity Λ(t), with a 90% band from the posterior draws
  * fair probability for 1X2, O/U 1.5 / 2.5 / 3.5 and BTTS
  * the Betfair vig-stripped probability for the same selection where it is two-sided

BETFAIR SIDE, HONESTLY: prints are thin (median gap ~1.05 min, p90 ~4.5), so a 1-minute
market series needs a short LOCF window — 2.5 min here. Where a line is SETTLED (e.g. Over
1.5 once two goals are in) the exchange stops trading it and the vig-strip degenerates; those
points are emitted with `settled = true` and must not be read as a market opinion. The model
is drawn throughout, the market only where it is real.

Run on the kaimon server session:
    include("current_development/inplay_scottish/r07_match_trace.jl")
=#

using DataFrames, Statistics, Serialization, Random, JSON3

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(dirname(@__DIR__), "match_inplay_explore", "l01_inplay_inverse.jl"))
for f in ("l01_nhpp_scottish.jl", "l02_ppd_compose.jl", "l04_bbc_timeline.jl",
          "l05_pregame_source.jl", "l09_ingame.jl")
    include(joinpath(@__DIR__, f))
end

OUT = joinpath(@__DIR__, "out")
const ENGINE = "funnel_apm_xg"
!(@isdefined model) && begin
    chain = deserialize(joinpath(OUT, "r04b_chain_$(ENGINE).jls"))
    mseqs = deserialize(joinpath(OUT, "r04b_mseqs_$(ENGINE).jls"))
    draws = pregame_draws(known_source(ENGINE), ds)
    model = InGameModel(ENGINE, chain, NHPPXConfig(), draws)
end

const TRACE_SELS = [:home, :draw, :away, :over_15, :under_15, :over_25, :under_25,
                    :over_35, :under_35, :btts_yes, :btts_no]

"Is this selection still undecided at match minute `t`?"
function still_live(sel::Symbol, goals, t)
    s = String(sel)
    tot = count(g -> g.t < t, goals)
    if startswith(s, "over_") || startswith(s, "under_")
        return tot <= parse(Int, s[end-1:end-1])
    elseif sel in (:btts_yes, :btts_no)
        return !(any(g -> g.home && g.t < t, goals) && any(g -> !g.home && g.t < t, goals))
    end
    return true
end

"""
    trace_match(ms; dt_model = 1.0, staleness = 2.5, n_pairs = 1200) -> NamedTuple

Model at `dt_model`-minute resolution; Betfair via LOCF on the same grid.
"""
function trace_match(ms; dt_model = 1.0, staleness = 2.5, n_pairs = 1200)
    bfm = subset(ds.betfair_odds, :match_id => ByRow(==(ms.mid)))
    cm = make_clock_map(anchor_goals(ds.betfair_odds, ds, ms.mid))

    # wall-clock grid → match minute (the clock map is monotone, so invert by scanning)
    grid = DataFrame(t_w = Float64[], t_m = Float64[])
    for t_w in 3.0:dt_model:118.0
        t_m = cm(t_w)
        (0.5 <= t_m <= 89.5) || continue
        push!(grid, (t_w, t_m))
    end
    unique!(grid, :t_m)

    rows = DataFrame(t_m = Float64[], t_w = Float64[], gh = Int[], ga = Int[],
                     rh = Int[], ra = Int[], lam_h = Float64[], lam_a = Float64[],
                     lam_rem = Float64[], lam_lo = Float64[], lam_hi = Float64[],
                     sel = Symbol[], p_model = Float64[], p_mkt = Float64[],
                     settled = Bool[])
    for r in eachrow(grid)
        t = r.t_m
        st = ingame_state(ms, ms.mid, t)
        rate = ingame_rate(model, ms.mid, t; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra)
        rem = ingame_remaining(model, ms.mid, t; gh = st.gh, ga = st.ga,
                               rh = st.rh, ra = st.ra, n_pairs = 600,
                               rng = Xoshiro(ms.mid + Int(round(t * 7))))
        tot = rem.Λ_h .+ rem.Λ_a
        book = ingame_book(model, ms.mid, t; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra,
                           n_pairs = n_pairs, rng = Xoshiro(ms.mid + Int(round(t * 13))))

        prices = latest_prices(bfm, r.t_w; staleness = staleness)
        mkt = Dict{Symbol, Float64}()
        if haskey(prices, :home) && haskey(prices, :draw) && haskey(prices, :away)
            fair = fair_match_df(prices)
            # two-sided only, per family group (see r06 header)
            grp(s) = (x = String(s); startswith(x, "over_") ? "ou_" * x[6:end] :
                      startswith(x, "under_") ? "ou_" * x[7:end] :
                      (s in (:btts_yes, :btts_no) ? "btts" : "x12"))
            gsum = Dict{String, Tuple{Int, Float64}}()
            for fr in eachrow(fair)
                g = grp(fr.selection); n, s = get(gsum, g, (0, 0.0))
                gsum[g] = (n + 1, s + fr.prob_fair_close)
            end
            for fr in eachrow(fair)
                g = grp(fr.selection); n, s = gsum[g]
                (g == "x12" || (n == 2 && 0.98 < s < 1.02)) || continue
                mkt[fr.selection] = fr.prob_fair_close
            end
        end

        for sel in TRACE_SELS
            haskey(book, sel) || continue
            push!(rows, (t, r.t_w, st.gh, st.ga, st.rh, st.ra,
                         rate.λ_h, rate.λ_a, mean(tot),
                         quantile(tot, 0.05), quantile(tot, 0.95),
                         sel, book[sel], get(mkt, sel, NaN),
                         !still_live(sel, ms.goals, t)))
        end
    end
    fh = count(g -> g.home, ms.goals); fa = count(g -> !g.home, ms.goals)
    mrow = only(subset(ds.matches, :match_id => ByRow(==(ms.mid))))
    (rows = rows, mid = ms.mid, home = String(mrow.home_team), away = String(mrow.away_team),
     final = (fh, fa),
     goals = [(t = g.t, home = g.home) for g in ms.goals],
     reds = [(t = c.t, home = c.home) for c in ms.reds],
     pregame = (λ_h = ms.pgh, λ_a = ms.pga))
end

# ---------------------------------------------------------------------------
# pick matches: one with a red card, one high-scoring, one that stays goalless late
# ---------------------------------------------------------------------------

bf_mids = Set(unique(subset(ds.betfair_odds, :minutes_to_kickoff =>
                            ByRow(x -> 0.0 < x <= 130.0)).match_id))
cands = [m for m in mseqs if m.mid in bf_mids]
n_prints(m) = nrow(subset(ds.betfair_odds, :match_id => ByRow(==(m.mid)),
                          :selection => ByRow(in([:home, :draw, :away]))))

pick_red   = argmax(m -> (length(m.reds) >= 1 ? 1 : 0) * 1_000_000 + n_prints(m), cands)
pick_goals = argmax(m -> (length(m.goals) >= 4 ? 1 : 0) * 1_000_000 + n_prints(m), cands)

traces = Dict("red" => trace_match(pick_red), "goals" => trace_match(pick_goals))

payload = Dict(k => Dict(
    "mid" => v.mid, "home" => v.home, "away" => v.away,
    "final" => [v.final[1], v.final[2]],
    "goals" => [Dict("t" => g.t, "home" => g.home) for g in v.goals],
    "reds"  => [Dict("t" => c.t, "home" => c.home) for c in v.reds],
    "pregame" => Dict("lam_h" => v.pregame.λ_h, "lam_a" => v.pregame.λ_a),
    "rows" => [Dict("t" => round(r.t_m, digits = 2), "gh" => r.gh, "ga" => r.ga,
                    "rh" => r.rh, "ra" => r.ra,
                    "lh" => round(r.lam_h, digits = 4), "la" => round(r.lam_a, digits = 4),
                    "lr" => round(r.lam_rem, digits = 3),
                    "lo" => round(r.lam_lo, digits = 3), "hi" => round(r.lam_hi, digits = 3),
                    "sel" => String(r.sel), "m" => round(r.p_model, digits = 4),
                    "k" => isnan(r.p_mkt) ? nothing : round(r.p_mkt, digits = 4),
                    "s" => r.settled) for r in eachrow(v.rows)])
    for (k, v) in traces)

write(joinpath(OUT, "r07_trace.json"), JSON3.write(payload))
R07 = (traces = traces,
       summary = [(k, v.mid, "$(v.home) $(v.final[1])-$(v.final[2]) $(v.away)",
                   nrow(v.rows) ÷ length(TRACE_SELS), length(v.goals), length(v.reds))
                  for (k, v) in traces])
@info "r07 done" R07.summary
