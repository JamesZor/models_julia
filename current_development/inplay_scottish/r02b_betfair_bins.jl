#=
r02b_betfair_bins.jl — WP3 gate iii: composed in-play PPD vs Betfair identifiable bins.

Eval set: 56 24/25 (the only training-usable pair with betfair coverage). Per wall-clock
bin (5-min grid): LOCF prices (4-min staleness), require full 1X2 + ≥6 selections
(the l01 identifiability gate), vig-strip → fair probs; clock-map wall→match minute via
goal-jump anchoring; composed PPD at the same state; compare.

Questions answered:
  (a) does composed fair value TRACK the market where it exists (corr / MAE / bias)?
  (b) who prices REALITY better in those bins — model or market (per-family logloss)?
Betfair here is evaluation-only (thin market; concept-map closed in-play trading).
=#

using Serialization, Statistics, DataFrames, Random

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(dirname(@__DIR__), "match_inplay_explore", "l01_inplay_inverse.jl"))
include(joinpath(@__DIR__, "l01_nhpp_scottish.jl"))
include(joinpath(@__DIR__, "l02_ppd_compose.jl"))

OUT = joinpath(@__DIR__, "out")
chain = deserialize(joinpath(OUT, "r01_chain.jls"))
mseqs = deserialize(joinpath(OUT, "r01_mseqs.jls"))
config = NHPPXConfig()

const LATENTS_JLS = abspath(joinpath(dirname(@__DIR__), "..", "data",
                            "scottish_decay_grid", "latents_hl365_hs2.jls"))
latents_df = deserialize(LATENTS_JLS)
lat = Dict(r.match_id => (collect(r.λ_h), collect(r.λ_a)) for r in eachrow(latents_df))

eval_ms = [m for m in mseqs if m.tournament_id == 56 && m.season == "24/25"]
bf = ds.betfair_odds

rows = DataFrame(mid = Int[], t_w = Float64[], t_m = Float64[], sel = Symbol[],
                 p_model = Float64[], p_fair = Float64[], y = Int[])
n_skipped = 0
for ms in eval_ms
    bfm = subset(bf, :match_id => ByRow(==(ms.mid)))
    isempty(bfm) && (global n_skipped += 1; continue)
    anchors = anchor_goals(bf, ds, ms.mid)
    cm = make_clock_map(anchors)
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
        t_m = cm(t_w)
        (1.0 <= t_m <= 85.0) || continue
        st = (gh = count(g ->  g.home && g.t < t_m, ms.goals),
              ga = count(g -> !g.home && g.t < t_m, ms.goals),
              rh = count(c ->  c.home && c.t < t_m, ms.reds),
              ra = count(c -> !c.home && c.t < t_m, ms.reds))
        λh, λa = lat[ms.mid]
        ppd = inplay_ppd(chain, config, λh, λa; gh = st.gh, ga = st.ga,
                         reds_h = st.rh, reds_a = st.ra, t_now = t_m,
                         n_pairs = 800, rng = Xoshiro(ms.mid + Int(round(t_w))))
        fair = fair_match_df(prices)
        for r in eachrow(fair)
            haskey(ppd, r.selection) || continue
            haskey(truth, r.selection) || continue
            push!(rows, (ms.mid, t_w, t_m, r.selection,
                         mean(ppd[r.selection]), r.prob_fair_close, truth[r.selection]))
        end
    end
end

fam(sel) = startswith(String(sel), "over") || startswith(String(sel), "under") ? :ou :
           (sel in (:home, :draw, :away) ? :x12 : :btts)
rows.family = fam.(rows.sel)

# --- validity filters (verified necessary 2026-07-14) -----------------------
# (a) LIVE selections only: a settled OU line / BTTS trades stale or not at all.
# (b) TWO-SIDED markets only: with thin prints an OU line often has one traded
#     side in the LOCF window and the vig-strip normalises it to p_fair = 1.0 —
#     these rows made raw market logloss look absurd (2.9 OU / 5.9 BTTS).
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
rows = subset(rows, :live => identity, :two_sided => identity)
# ----------------------------------------------------------------------------
ll(p, y) = -(y .* log.(clamp.(p, 1e-9, 1)) .+ (1 .- y) .* log.(clamp.(1 .- p, 1e-9, 1)))

agreement = combine(groupby(rows, :family),
    nrow => :n,
    [:p_model, :p_fair] => cor => :corr,
    [:p_model, :p_fair] => ((m, f) -> mean(abs.(m .- f))) => :mae,
    [:p_model, :p_fair] => ((m, f) -> mean(m .- f)) => :model_minus_mkt)

vs_reality = combine(groupby(rows, :family),
    nrow => :n,
    [:p_model, :y] => ((p, y) -> mean(ll(p, y))) => :logloss_model,
    [:p_fair, :y]  => ((p, y) -> mean(ll(p, y))) => :logloss_market,
    [:p_model, :p_fair, :y] => ((pm, pf, y) -> mean(ll(pf, y) .- ll(pm, y))) => :model_edge)

# match-clustered t on the model-vs-market logloss gap (per family)
gap_t = combine(groupby(rows, :family)) do g
    per = combine(groupby(DataFrame(mid = g.mid, d = ll(g.p_fair, g.y) .- ll(g.p_model, g.y)), :mid),
                  :d => mean => :d)
    (; t = mean(per.d) / (std(per.d) / sqrt(nrow(per))), n_matches = nrow(per))
end

serialize(joinpath(OUT, "r02b_rows.jls"), rows)
R02B = (n_bins = nrow(unique(select(rows, :mid, :t_w))), n_matches = length(unique(rows.mid)),
        n_skipped = n_skipped, agreement = agreement, vs_reality = vs_reality, gap_t = gap_t)
@info "r02b done" R02B.n_bins R02B.n_matches
