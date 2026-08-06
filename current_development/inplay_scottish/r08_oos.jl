#=
r08_oos.jl — honest out-of-sample evaluation of the in-game model, on the pregame convention.

WHY. Every in-play number this stream has published — r02b's market tie, r06's tie on 376
matches — was produced by a multiplier chain fitted on the very matches it was then scored
on. The chain is 6 global parameters over ~27k slice rows so the optimism should be small,
but "should be small" is not a measurement. This runner measures it.

THE SPLIT, MATCHING THE PREGAME ENGINES. The pregame layer walks forward: fit on history,
predict the next block, never look ahead. The same discipline applied to the in-play
multiplier:

  * **Primary — season holdout.** Fit the NHPP on 24/25 ONLY, score 25/26. The pregame
    latents for 25/26 are already walk-forward OOS from the funnel run, so both stages are
    now out of sample and the number is clean end-to-end.
  * **Secondary — rolling monthly walk-forward.** For each test month, fit on every match
    strictly before it. Poisson MLE rather than NUTS (18 folds × NUTS is not worth it for a
    6-parameter model), which gives the stability picture across the whole period cheaply.

BBC IS WHAT MAKES THE TEST SET BIG. 25/26 has incidents for only 16 of 175 t56 matches, so
an incidents-only holdout would be almost entirely t57. BBC covers 25/26 in full — 350
matches — so the holdout is a real season, not a rump. This is the coverage argument the
four-arm race was designed to test, cashed out somewhere it actually matters.

The model uses the BBC clock (`kind = :expo`): fixed 18-bin [0,90] frame with each match's
real stoppage in the terminal bins.

Run on the kaimon server session (-t 16, pinthreads(:cores)):
    include("current_development/inplay_scottish/r08_oos.jl")
    R08.headline
=#

using DataFrames, Statistics, Serialization, Random, GLM, Dates

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(dirname(@__DIR__), "match_inplay_explore", "l01_inplay_inverse.jl"))
for f in ("l01_nhpp_scottish.jl", "l02_ppd_compose.jl", "l04_bbc_timeline.jl",
          "l05_pregame_source.jl", "l08_race.jl", "l09_ingame.jl")
    include(joinpath(@__DIR__, f))
end

OUT = joinpath(@__DIR__, "out"); mkpath(OUT)
const ENGINE = "funnel_apm_xg"

# ---------------------------------------------------------------------------
# §1 data — BBC events (full 25/26 coverage), funnel latents
# ---------------------------------------------------------------------------

seqs = deserialize(joinpath(OUT, "r04a_seqs.jls"))
stoppage = deserialize(joinpath(OUT, "r04a_stoppage.jls"))
draws = pregame_draws(known_source(ENGINE), ds)

# require_incidents = false: BBC is the event source, so incident coverage is irrelevant.
const ALL_PAIRS = Set([(56, "24/25"), (56, "25/26"), (57, "24/25"), (57, "25/26")])
allms = assemble_matches(ds, draws, ALL_PAIRS; seqs = seqs, require_incidents = false)

train_ms = [m for m in allms if m.season == "24/25"]
test_ms  = [m for m in allms if m.season == "25/26"]

split_stats = (train = length(train_ms), test = length(test_ms),
               train_goals = sum(length(m.goals) for m in train_ms),
               test_goals = sum(length(m.goals) for m in test_ms),
               by_pair = sort(combine(groupby(DataFrame(
                    t = [m.tournament_id for m in allms],
                    s = [m.season for m in allms]), [:t, :s]), nrow), [:t, :s]))

# ---------------------------------------------------------------------------
# §2 primary — fit on 24/25, score 25/26
# ---------------------------------------------------------------------------

config = NHPPXConfig(Tend = 90.0)              # BBC clock frame
train_sl = build_slices_bbc(train_ms; Δt = config.Δt, Tend = config.Tend)
full_sl  = build_slices_bbc(allms;    Δt = config.Δt, Tend = config.Tend)

oos_chain = Samplers.run_sampler(make_nhppx_model(train_sl, config),
    Samplers.NUTSConfig(n_samples = 600, n_chains = 4, n_warmup = 300,
                        max_depth = 8, show_progress = false))
# the in-sample comparator: same spec, fitted on everything (what r06 did)
ins_chain = Samplers.run_sampler(make_nhppx_model(full_sl, config),
    Samplers.NUTSConfig(n_samples = 600, n_chains = 4, n_warmup = 300,
                        max_depth = 8, show_progress = false))

post_cmp = DataFrame(param = Symbol[], oos_fit = Float64[], insample_fit = Float64[])
for p in (:α, :β, :γ_tr, :γ_ld, :γ_man, :σ_time)
    push!(post_cmp, (p, mean(_cv(oos_chain, p)), mean(_cv(ins_chain, p))))
end

mk(ch) = InGameModel(ENGINE, ch, config, draws; kind = :expo, stoppage = stoppage)
model_oos = mk(oos_chain); model_ins = mk(ins_chain)

# ---------------------------------------------------------------------------
# §3 outcome scoring on the held-out season
# ---------------------------------------------------------------------------

const SELS = vcat([:home, :draw, :away, :btts_yes, :btts_no],
                  [Symbol("over_$(k)5") for k in 0:3],
                  [Symbol("under_$(k)5") for k in 0:3])
famof(s) = s in (:home, :draw, :away) ? :x12 :
           s in (:btts_yes, :btts_no) ? :btts : :ou

function score_set(m::InGameModel, ms_list; checkpoints = (0.0, 30.0, 60.0, 80.0))
    rows = DataFrame(mid = Int[], t0 = Float64[], sel = Symbol[], p = Float64[], y = Int[])
    for ms in ms_list
        fh = count(g -> g.home, ms.goals); fa = count(g -> !g.home, ms.goals)
        tr = truth_of(fh, fa)
        for t0 in checkpoints
            st = ingame_state(ms, ms.mid, t0)
            bk = ingame_book(m, ms.mid, t0; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra,
                             n_pairs = 800, rng = Xoshiro(ms.mid + Int(t0)))
            for s in SELS
                haskey(bk, s) && haskey(tr, s) && push!(rows, (ms.mid, t0, s, bk[s], tr[s]))
            end
        end
    end
    rows.family = famof.(rows.sel)
    return rows
end

oos_rows = score_set(model_oos, test_ms)     # trained 24/25, scored 25/26  → HONEST
ins_rows = score_set(model_ins, test_ms)     # trained on everything        → OPTIMISTIC

_ll(p, y) = -(y * log(clamp(p, 1e-9, 1)) + (1 - y) * log(clamp(1 - p, 1e-9, 1)))
function fam_score(rows)
    combine(groupby(rows, [:t0, :family]), nrow => :n,
        [:p, :y] => ((p, y) -> mean(_ll.(p, y))) => :logloss,
        [:p, :y] => ((p, y) -> mean((p .- y) .^ 2)) => :brier,
        [:p, :y] => ((p, y) -> mean(p .- y)) => :bias)
end
oos_score = sort(fam_score(oos_rows), [:family, :t0])
ins_score = sort(fam_score(ins_rows), [:family, :t0])

# The optimism itself, match-clustered: how much better does the in-sample fit LOOK?
optimism = combine(groupby(innerjoin(
        select(oos_rows, :mid, :t0, :sel, :family, :p => :p_oos, :y),
        select(ins_rows, :mid, :t0, :sel, :p => :p_ins), on = [:mid, :t0, :sel]),
        :family)) do g
    per = combine(groupby(DataFrame(mid = g.mid,
            d = _ll.(g.p_oos, g.y) .- _ll.(g.p_ins, g.y)), :mid), :d => mean => :d)
    (; n_matches = nrow(per), logloss_gap = mean(per.d),
       t = mean(per.d) / (std(per.d) / sqrt(nrow(per))))
end

# Reliability on the held-out season — this is where the 1X2 favourite band gets tested.
oos_cal = sort(combine(groupby(transform(oos_rows,
        :p => ByRow(p -> clamp(floor(Int, p * 10) / 10 + 0.05, 0.05, 0.95)) => :bin),
        [:family, :bin]), nrow => :n, :p => mean => :p_mean, :y => mean => :y_rate),
    [:family, :bin])

# ---------------------------------------------------------------------------
# §4 held-out season vs Betfair
# ---------------------------------------------------------------------------

bf = ds.betfair_odds
bf_mids = Set(unique(subset(bf, :minutes_to_kickoff => ByRow(x -> 0.0 < x <= 130.0)).match_id))
bf_test = [m for m in test_ms if m.mid in bf_mids]

mkt_rows = DataFrame(mid = Int[], t_m = Float64[], sel = Symbol[],
                     p_model = Float64[], p_fair = Float64[], y = Int[])
for ms in bf_test
    bfm = subset(bf, :match_id => ByRow(==(ms.mid)))
    cm = make_clock_map(anchor_goals(bf, ds, ms.mid))
    fh = count(g -> g.home, ms.goals); fa = count(g -> !g.home, ms.goals)
    tr = truth_of(fh, fa)
    for t_w in 10.0:5.0:110.0
        prices = latest_prices(bfm, t_w; staleness = 4.0)
        (haskey(prices, :home) && haskey(prices, :draw) && haskey(prices, :away) &&
         length(prices) >= 6) || continue
        t_m = cm(t_w); (1.0 <= t_m <= 85.0) || continue
        st = ingame_state(ms, ms.mid, t_m)
        bk = ingame_book(model_oos, ms.mid, t_m; gh = st.gh, ga = st.ga,
                         rh = st.rh, ra = st.ra, n_pairs = 800,
                         rng = Xoshiro(ms.mid + Int(round(t_w))))
        fair = fair_match_df(prices)
        for r in eachrow(fair)
            (haskey(bk, r.selection) && haskey(tr, r.selection)) || continue
            push!(mkt_rows, (ms.mid, t_m, r.selection, bk[r.selection],
                             r.prob_fair_close, tr[r.selection]))
        end
    end
end
mkt_rows.family = famof.(mkt_rows.sel)

# the two mandatory r02b filters
gmap = Dict(m.mid => m.goals for m in bf_test)
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
mkt_rows.live = map(is_live, eachrow(mkt_rows)); mkt_rows.grp = pairname.(mkt_rows.sel)
gsz = combine(groupby(mkt_rows, [:mid, :t_m, :grp]), nrow => :nsel, :p_fair => sum => :psum)
mkt_rows = leftjoin(mkt_rows, gsz, on = [:mid, :t_m, :grp])
mkt_rows.two_sided = map(r -> r.grp == "x12" || (r.nsel == 2 && 0.98 < r.psum < 1.02),
                         eachrow(mkt_rows))
mkt_rows = subset(mkt_rows, :live => identity, :two_sided => identity)

vs_market = combine(groupby(mkt_rows, :family), nrow => :n,
    [:p_model, :p_fair] => cor => :corr,
    [:p_model, :p_fair] => ((m, f) -> mean(abs.(m .- f))) => :mae,
    [:p_model, :y] => ((p, y) -> mean(_ll.(p, y))) => :logloss_model,
    [:p_fair, :y]  => ((p, y) -> mean(_ll.(p, y))) => :logloss_market)
vs_market_t = combine(groupby(mkt_rows, :family)) do g
    per = combine(groupby(DataFrame(mid = g.mid,
            d = _ll.(g.p_fair, g.y) .- _ll.(g.p_model, g.y)), :mid), :d => mean => :d)
    (; n_matches = nrow(per), mean = mean(per.d),
       t = mean(per.d) / (std(per.d) / sqrt(nrow(per))))
end

# ---------------------------------------------------------------------------
# §5 secondary — rolling monthly walk-forward (Poisson MLE, cheap)
# ---------------------------------------------------------------------------

mdate = Dict(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
full_sl.ym = [(d = mdate[m]; 12 * year(d) + month(d)) for m in full_sl.match_id]
months = sort(unique(full_sl.ym))

walk = DataFrame(ym = Int[], n_train = Int[], n_test = Int[],
                 ll_walk = Float64[], ll_insample = Float64[])
for (i, ym) in enumerate(months)
    i <= 4 && continue                      # need a burn-in of history
    tr = subset(full_sl, :ym => ByRow(<(ym)))
    te = subset(full_sl, :ym => ByRow(==(ym)))
    (nrow(te) == 0 || sum(tr.y) < 100) && continue
    fw = glm(SPEC_FORMULAS[:full], tr, Poisson(), LogLink(); offset = tr.off)
    fi = glm(SPEC_FORMULAS[:full], full_sl, Poisson(), LogLink(); offset = full_sl.off)
    μw = GLM.predict(fw, te; offset = te.off); μi = GLM.predict(fi, te; offset = te.off)
    push!(walk, (ym, nrow(tr), nrow(te),
                 mean(logpdf.(Poisson.(max.(μw, 1e-9)), te.y)),
                 mean(logpdf.(Poisson.(max.(μi, 1e-9)), te.y))))
end
walk.gap = walk.ll_insample .- walk.ll_walk

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

R08 = (split = split_stats, post = post_cmp,
       oos_score = oos_score, ins_score = ins_score, optimism = optimism,
       calibration = oos_cal, vs_market = vs_market, vs_market_t = vs_market_t,
       n_market_matches = length(unique(mkt_rows.mid)),
       n_market_rows = nrow(mkt_rows), walk = walk,
       walk_gap = (mean = mean(walk.gap), max = maximum(walk.gap)),
       max_rhat = maximum(skipmissing(MCMCChains.summarystats(oos_chain)[:, :rhat])),
       kernel_oos = kernel_scale(model_oos), kernel_ins = kernel_scale(model_ins),
       headline = "trained 24/25 ($(length(train_ms)) matches) → scored 25/26 ($(length(test_ms)))")

serialize(joinpath(OUT, "r08_oos_chain.jls"), oos_chain)
serialize(joinpath(OUT, "r08_rows.jls"), (oos = oos_rows, ins = ins_rows, mkt = mkt_rows))
serialize(joinpath(OUT, "r08_summary.jls"), R08)
@info "r08 done" R08.headline
