#=
r09_calibrated.jl — does the calibration layer actually improve the held-out book?

Everything is fitted on 24/25 and tested on 25/26, the same split as r08, and the NHPP chain
is r08's (fitted on 24/25 only) — so nothing here has seen the test season.

FOUR ARMS, each adding one thing, so the credit is attributable:
  raw          — the model as it stands
  level        — + scalar level correction fitted on 24/25
  family       — + per-family logit calibration fitted on 24/25
  level+family — both

Plus two references that are NOT deployable and exist only to bound what is achievable:
  oracle_level — the level fitted ON the test season (how much of the level error is
                 forecastable from history at all, given the drift)
  rolling      — level from the trailing 120 matches, the form you would actually run

PRE-REGISTERED EXPECTATION. The level fix should help totals most (it is a totals error).
The family fix should help 1X2 (over-dispersed) and totals (under-dispersed) in OPPOSITE
directions. BTTS has ~700 rows in the calibration set, under the `min_n` guard for two of the
four checkpoints, so it may be left as identity — that is intended, not a bug.

Run on the kaimon server session:
    include("current_development/inplay_scottish/r09_calibrated.jl")
    R09.race
=#

using DataFrames, Statistics, Serialization, Random, Dates

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(dirname(@__DIR__), "match_inplay_explore", "l01_inplay_inverse.jl"))
for f in ("l01_nhpp_scottish.jl", "l02_ppd_compose.jl", "l04_bbc_timeline.jl",
          "l05_pregame_source.jl", "l08_race.jl", "l09_ingame.jl", "l10_calibrate.jl")
    include(joinpath(@__DIR__, f))
end

OUT = joinpath(@__DIR__, "out")
const ENGINE = "funnel_apm_xg"

seqs     = deserialize(joinpath(OUT, "r04a_seqs.jls"))
stoppage = deserialize(joinpath(OUT, "r04a_stoppage.jls"))
draws    = pregame_draws(known_source(ENGINE), ds)
oos_chain = deserialize(joinpath(OUT, "r08_oos_chain.jls"))     # fitted on 24/25 ONLY

const ALL_PAIRS = Set([(56, "24/25"), (56, "25/26"), (57, "24/25"), (57, "25/26")])
allms = assemble_matches(ds, draws, ALL_PAIRS; seqs = seqs, require_incidents = false)
train_ms = [m for m in allms if m.season == "24/25"]
test_ms  = [m for m in allms if m.season == "25/26"]

config = NHPPXConfig(Tend = 90.0)
base = InGameModel(ENGINE, oos_chain, config, draws; kind = :expo, stoppage = stoppage)

# ---------------------------------------------------------------------------
# §1 fit the corrections on 24/25
# ---------------------------------------------------------------------------

lvl_train  = fit_level(base, train_ms)                 # deployable
lvl_oracle = fit_level(base, test_ms)                  # reference only

m_raw   = base
m_level = with_level(base, lvl_train)
m_orc   = with_level(base, lvl_oracle)

cal_rows_raw = score_book(m_raw,   train_ms)           # calibration set, 24/25
cal_rows_lvl = score_book(m_level, train_ms)
cal_raw = fit_family_calibrator(cal_rows_raw)
cal_lvl = fit_family_calibrator(cal_rows_lvl)

cal_table = DataFrame(family = Symbol[], n_fit = Int[], a = Float64[], b = Float64[],
                      reading = String[])
for f in (:x12, :ou, :btts)
    b = cal_lvl.b[f]
    push!(cal_table, (f, cal_lvl.n[f], cal_lvl.a[f], b,
        b < 0.95 ? "shrink toward 0.5 (was over-confident)" :
        b > 1.05 ? "sharpen (was under-confident)" : "left as-is"))
end

# ---------------------------------------------------------------------------
# §2 score the held-out season
# ---------------------------------------------------------------------------

arms = Dict(
    "raw"          => score_book(m_raw,   test_ms),
    "level"        => score_book(m_level, test_ms),
    "family"       => score_book(m_raw,   test_ms; cal = cal_raw),
    "level+family" => score_book(m_level, test_ms; cal = cal_lvl),
    "oracle_level" => score_book(m_orc,   test_ms),
)

race = DataFrame(arm = String[], family = Symbol[], n = Int[],
                 logloss = Float64[], brier = Float64[])
for (nm, r) in arms, row in eachrow(score_summary(r))
    push!(race, (nm, row.family, row.n, row.logloss, row.brier))
end
sort!(race, [:family, :logloss])

gains = DataFrame(arm = String[], family = Symbol[], n_matches = Int[],
                  gain = Float64[], t = Float64[])
for nm in ("level", "family", "level+family", "oracle_level")
    for row in eachrow(paired_vs(arms[nm], arms["raw"]))
        push!(gains, (nm, row.family, row.n_matches, row.gain, row.t))
    end
end
sort!(gains, [:family, :arm])

# ---------------------------------------------------------------------------
# §3 did the level error actually shrink?
# ---------------------------------------------------------------------------

function level_check(m, ms_list; t0 = 0.0)
    p = 0.0; r = 0.0
    for ms in ms_list
        st = ingame_state(ms, ms.mid, t0)
        rem = ingame_remaining(m, ms.mid, t0; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra,
                               n_pairs = 400)
        p += st.gh + st.ga + mean(rem.Λ_h) + mean(rem.Λ_a); r += length(ms.goals)
    end
    (pred = p / length(ms_list), real = r / length(ms_list), hot_pct = 100 * (p / r - 1))
end

level_tbl = DataFrame(arm = String[], pred = Float64[], real = Float64[], hot_pct = Float64[])
for (nm, m) in (("raw", m_raw), ("level", m_level), ("oracle_level", m_orc))
    lc = level_check(m, test_ms); push!(level_tbl, (nm, lc.pred, lc.real, lc.hot_pct))
end

# rolling level: the deployable form, evaluated match by match in date order
mdate = Dict(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
sorted_all = sort(allms, by = m -> mdate[m.mid])
test_pos = [i for (i, m) in enumerate(sorted_all) if m.season == "25/26"]
roll_lv = Float64[]
for i in test_pos[1:max(1, length(test_pos) ÷ 25):end]      # sample, it is O(window) each
    push!(roll_lv, rolling_level(base, sorted_all, i; window = 120))
end
rolling_summary = (n = length(roll_lv), mean = mean(roll_lv),
                   min = minimum(roll_lv), max = maximum(roll_lv),
                   train_level = lvl_train, oracle_level = lvl_oracle)

# ---------------------------------------------------------------------------
# §4 reliability after calibration
# ---------------------------------------------------------------------------

relia(r) = sort(combine(groupby(transform(r,
        :p => ByRow(p -> clamp(floor(Int, p * 10) / 10 + 0.05, 0.05, 0.95)) => :bin),
        [:family, :bin]), nrow => :n, :p => mean => :p_mean, :y => mean => :y_rate),
    [:family, :bin])
relia_raw = relia(arms["raw"]); relia_cal = relia(arms["level+family"])

# max |p − y| over buckets with n ≥ 50 — the headline miscalibration number
maxdev(t) = combine(groupby(subset(t, :n => ByRow(>=(50))), :family),
    [:p_mean, :y_rate] => ((p, y) -> maximum(abs.(p .- y))) => :max_abs_dev)

R09 = (level_train = lvl_train, level_oracle = lvl_oracle,
       cal_table = cal_table, race = race, gains = gains, level_tbl = level_tbl,
       rolling = rolling_summary,
       relia_raw = relia_raw, relia_cal = relia_cal,
       maxdev_raw = maxdev(relia_raw), maxdev_cal = maxdev(relia_cal))

serialize(joinpath(OUT, "r09_calibrated.jls"), (; R09..., arms = arms))
@info "r09 done" R09.level_train R09.level_oracle
