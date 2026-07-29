#=
r06_ingame_betfair.jl — the in-game model against Betfair historical.

Two jobs:
  (1) make the in-game λ / fair-price object usable (l09), and
  (2) show what it looks like next to the exchange, on real past matches.

This supersedes r02b in two ways: the pregame engine is now `funnel_apm_xg` (r02b used the
dead decay-grid latents), and the eval set is every Betfair match with pregame latents rather
than only t56 24/25.

TWO VALIDITY FILTERS ARE MANDATORY, both learned the hard way in r02b (2026-07-14):
  (a) LIVE selections only — a settled O/U line or BTTS trades stale, or not at all.
  (b) TWO-SIDED markets only — with thin prints an O/U line often has a single traded side
      in the window, and the vig-strip then normalises it to p_fair = 1.0. Those rows made
      raw market logloss look absurd (2.9 O/U, 5.9 BTTS). They are not market opinions.
Without both, every number below is meaningless.

Betfair is EVALUATION ONLY. The exchange here is thin (median ~49 in-play 1X2 prints per
match) and the parent Ireland stream already established there is no tradeable speed edge;
the point of this comparison is to show the model produces market-grade fair value, and to
price the ~⅓ of O/U rows and ~½ of BTTS rows where the exchange is one-sided or absent.

Run on the kaimon server session (-t 16):
    ENV["JULIA_PKG_PRECOMPILE_AUTO"]="0"; using BayesianFootball
    include("current_development/inplay_scottish/r06_ingame_betfair.jl")
    R06.agreement; R06.vs_reality; R06.calibration
=#

using DataFrames, Statistics, Serialization, Random

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(dirname(@__DIR__), "match_inplay_explore", "l01_inplay_inverse.jl"))
for f in ("l01_nhpp_scottish.jl", "l02_ppd_compose.jl", "l04_bbc_timeline.jl",
          "l05_pregame_source.jl", "l09_ingame.jl")
    include(joinpath(@__DIR__, f))
end

OUT = joinpath(@__DIR__, "out"); mkpath(OUT)

# ---------------------------------------------------------------------------
# §0 the model
# ---------------------------------------------------------------------------

const ENGINE = "funnel_apm_xg"
chain  = deserialize(joinpath(OUT, "r04b_chain_$(ENGINE).jls"))
mseqs  = deserialize(joinpath(OUT, "r04b_mseqs_$(ENGINE).jls"))
draws  = pregame_draws(known_source(ENGINE), ds)
model  = InGameModel(ENGINE, chain, NHPPXConfig(), draws)

K = kernel_scale(model)

# ---------------------------------------------------------------------------
# §1 eval set — every match with betfair in-play prints AND pregame latents
# ---------------------------------------------------------------------------

bf = ds.betfair_odds
bf_mids = Set(unique(subset(bf, :minutes_to_kickoff =>
                            ByRow(x -> 0.0 < x <= 130.0)).match_id))
eval_ms = [m for m in mseqs if m.mid in bf_mids]

coverage = combine(groupby(DataFrame(
    tournament_id = [m.tournament_id for m in eval_ms],
    season = [m.season for m in eval_ms]), [:tournament_id, :season]), nrow => :matches)
sort!(coverage, [:tournament_id, :season])

# ---------------------------------------------------------------------------
# §2 model vs market, per identifiable bin
# ---------------------------------------------------------------------------

rows = DataFrame(mid = Int[], t_w = Float64[], t_m = Float64[], sel = Symbol[],
                 p_model = Float64[], p_fair = Float64[], y = Int[],
                 gh = Int[], ga = Int[], reds = Int[], λ_rem = Float64[])

for ms in eval_ms
    bfm = subset(bf, :match_id => ByRow(==(ms.mid)))
    isempty(bfm) && continue
    cm = make_clock_map(anchor_goals(bf, ds, ms.mid))
    fh = count(g -> g.home, ms.goals); fa = count(g -> !g.home, ms.goals); tot = fh + fa
    truth = Dict(:home => Int(fh > fa), :draw => Int(fh == fa), :away => Int(fh < fa),
                 :btts_yes => Int(fh > 0 && fa > 0), :btts_no => Int(!(fh > 0 && fa > 0)))
    for k in 0:5
        truth[Symbol("over_$(k)5")] = Int(tot > k)
        truth[Symbol("under_$(k)5")] = Int(tot < k + 1)
    end
    for t_w in 10.0:5.0:110.0
        prices = latest_prices(bfm, t_w; staleness = 4.0)
        (haskey(prices, :home) && haskey(prices, :draw) && haskey(prices, :away) &&
         length(prices) >= 6) || continue
        t_m = cm(t_w); (1.0 <= t_m <= 85.0) || continue
        st = ingame_state(ms, ms.mid, t_m)
        book = ingame_book(model, ms.mid, t_m; gh = st.gh, ga = st.ga,
                           rh = st.rh, ra = st.ra, n_pairs = 800,
                           rng = Xoshiro(ms.mid + Int(round(t_w))))
        rem = ingame_remaining(model, ms.mid, t_m; gh = st.gh, ga = st.ga,
                               rh = st.rh, ra = st.ra, n_pairs = 400)
        λ_rem = mean(rem.Λ_h) + mean(rem.Λ_a)
        fair = fair_match_df(prices)
        for r in eachrow(fair)
            (haskey(book, r.selection) && haskey(truth, r.selection)) || continue
            push!(rows, (ms.mid, t_w, t_m, r.selection, book[r.selection],
                         r.prob_fair_close, truth[r.selection],
                         st.gh, st.ga, st.rh + st.ra, λ_rem))
        end
    end
end

fam(sel) = startswith(String(sel), "over") || startswith(String(sel), "under") ? :ou :
           (sel in (:home, :draw, :away) ? :x12 : :btts)
rows.family = fam.(rows.sel)

# --- the two mandatory filters (see header) ---------------------------------
gmap = Dict(m.mid => m.goals for m in eval_ms)
function is_live(r)
    g = gmap[r.mid]; tot = count(x -> x.t < r.t_m, g); s = String(r.sel)
    if startswith(s, "over_") || startswith(s, "under_")
        return tot <= parse(Int, s[end-1:end-1])
    elseif r.sel in (:btts_yes, :btts_no)
        return !(any(x -> x.home && x.t < r.t_m, g) && any(x -> !x.home && x.t < r.t_m, g))
    end
    return true
end
pairname(s) = (t = String(s); startswith(t, "over_") ? "ou_" * t[6:end] :
               startswith(t, "under_") ? "ou_" * t[7:end] :
               (s in (:btts_yes, :btts_no) ? "btts" : "x12"))
rows.live = map(is_live, eachrow(rows))
rows.grp = pairname.(rows.sel)
gsz = combine(groupby(rows, [:mid, :t_w, :grp]), nrow => :nsel, :p_fair => sum => :psum)
rows = leftjoin(rows, gsz, on = [:mid, :t_w, :grp])
rows.two_sided = map(r -> r.grp == "x12" || (r.nsel == 2 && 0.98 < r.psum < 1.02),
                     eachrow(rows))
raw_n = nrow(rows)
rows = subset(rows, :live => identity, :two_sided => identity)

# ---------------------------------------------------------------------------
# §3 the three summaries
# ---------------------------------------------------------------------------

ll(p, y) = -(y .* log.(clamp.(p, 1e-9, 1)) .+ (1 .- y) .* log.(clamp.(1 .- p, 1e-9, 1)))

agreement = combine(groupby(rows, :family), nrow => :n,
    [:p_model, :p_fair] => cor => :corr,
    [:p_model, :p_fair] => ((m, f) -> mean(abs.(m .- f))) => :mae,
    [:p_model, :p_fair] => ((m, f) -> mean(m .- f)) => :model_minus_mkt)

vs_reality = combine(groupby(rows, :family), nrow => :n,
    [:p_model, :y] => ((p, y) -> mean(ll(p, y))) => :logloss_model,
    [:p_fair, :y]  => ((p, y) -> mean(ll(p, y))) => :logloss_market,
    [:p_model, :p_fair, :y] => ((pm, pf, y) -> mean(ll(pf, y) .- ll(pm, y))) => :model_edge)

# MATCH-CLUSTERED t on the model-vs-market gap. The naive per-row t is inflated ~2-3x in
# this repo (see l07); the match is the independent unit.
gap_t = combine(groupby(rows, :family)) do g
    per = combine(groupby(DataFrame(mid = g.mid,
                    d = ll(g.p_fair, g.y) .- ll(g.p_model, g.y)), :mid), :d => mean => :d)
    (; n_matches = nrow(per), mean = mean(per.d),
       t = mean(per.d) / (std(per.d) / sqrt(nrow(per))))
end

# Reliability: is the model's stated probability the realised frequency?
calibration = combine(groupby(transform(rows,
        :p_model => ByRow(p -> clamp(floor(Int, p * 10) / 10 + 0.05, 0.05, 0.95)) => :bin),
        [:family, :bin]),
    nrow => :n, :p_model => mean => :p_mean, :y => mean => :y_rate)
sort!(calibration, [:family, :bin])

# By match phase, so the plot can show where the model and market diverge.
by_time = combine(groupby(transform(rows,
        :t_m => ByRow(t -> 15 * floor(Int, t / 15)) => :phase), [:family, :phase]),
    nrow => :n,
    [:p_model, :p_fair] => cor => :corr,
    [:p_model, :p_fair] => ((m, f) -> mean(abs.(m .- f))) => :mae,
    [:p_model, :y] => ((p, y) -> mean(ll(p, y))) => :logloss_model,
    [:p_fair, :y]  => ((p, y) -> mean(ll(p, y))) => :logloss_market)
sort!(by_time, [:family, :phase])

R06 = (engine = ENGINE, kernel = K, coverage = coverage,
       n_matches = length(unique(rows.mid)), n_eval_ms = length(eval_ms),
       n_bins = nrow(unique(select(rows, :mid, :t_w))),
       raw_rows = raw_n, kept_rows = nrow(rows),
       agreement = agreement, vs_reality = vs_reality, gap_t = gap_t,
       calibration = calibration, by_time = by_time)

serialize(joinpath(OUT, "r06_rows.jls"), rows)
serialize(joinpath(OUT, "r06_summary.jls"), R06)

@info "r06 done" R06.n_matches R06.n_bins R06.kernel
