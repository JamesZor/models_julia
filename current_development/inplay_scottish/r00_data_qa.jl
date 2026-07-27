#=
r00_data_qa.jl — WP1 data QA for the Scottish in-play stream (tournaments 56/57).

Answers, from the DataStore alone:
  §1 incident coverage per (tournament, season) + which seasons enter NHPP training
  §2 score-path reconstruction: do goal incidents reproduce the final score?
  §3 goal-minute sanity (added_time handling, stoppage mass) + red-card counts
  §4 betfair in-play density on 56: prints per market, inter-print gaps,
     share of 5-min bins with an identifiable full 1X2 (liquidity-audit repro)
  §5 clock-map anchoring on thin prints: jump-found rate, off1/off2 spread

Reuses ../match_inplay_explore/l01_inplay_inverse.jl (match_goals, match_reds,
anchor_goals, make_clock_map, latest_prices). Run on the homelab kaimon session:

    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    using BayesianFootball
    include("current_development/inplay_scottish/r00_data_qa.jl")
    QA.summary        # per-section DataFrames live in the QA NamedTuple
=#

using DataFrames
using Statistics

const BF = BayesianFootball

# ---------------------------------------------------------------------------
# §0 load
# ---------------------------------------------------------------------------

ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower())

include(joinpath(@__DIR__, "..", "match_inplay_explore", "l01_inplay_inverse.jl"))

const MATCH_COLS = [:match_id, :tournament_id, :season_id, :season,
                    :home_score, :away_score, :match_date]
matches = select(ds.matches, MATCH_COLS)

# ---------------------------------------------------------------------------
# §1 incident coverage per (tournament, season)
# ---------------------------------------------------------------------------

goal_counts = combine(
    groupby(subset(ds.incidents, :incident_type => ByRow(==("goal"))), :match_id),
    nrow => :n_goal_inc)

cov = leftjoin(matches, goal_counts, on = :match_id)
cov.n_goal_inc = coalesce.(cov.n_goal_inc, 0)
cov.total_goals = coalesce.(cov.home_score, 0) .+ coalesce.(cov.away_score, 0)
# a match "has incidents" if it has any incident row at all (0-0 games have none by goals)
inc_any = Set(unique(ds.incidents.match_id))
cov.has_inc = [m in inc_any for m in cov.match_id]

season_cov = combine(groupby(cov, [:tournament_id, :season_id, :season]),
    nrow => :matches,
    :has_inc => sum => :with_incidents,
    [:n_goal_inc, :total_goals] => ((n, t) -> sum(n .== t)) => :goal_count_ok,
) |> df -> sort(df, [:tournament_id, :season_id])
season_cov.usable = season_cov.with_incidents .>= 0.9 .* season_cov.matches

# ---------------------------------------------------------------------------
# §2 score-path reconstruction (matches with incidents only)
# ---------------------------------------------------------------------------

recon = DataFrame(match_id=Int[], tournament_id=Int[], season_id=Int[],
                  ok=Bool[], n_inc=Int[], n_final=Int[])
for r in eachrow(subset(cov, :has_inc => ByRow(identity)))
    g = match_goals(ds, r.match_id)
    gh = sum(g.is_home); ga = sum(.!g.is_home)
    ok = !ismissing(r.home_score) && gh == r.home_score && ga == r.away_score
    push!(recon, (r.match_id, r.tournament_id, r.season_id, ok, nrow(g), r.total_goals))
end
recon_summary = combine(groupby(recon, :tournament_id),
    nrow => :matches, :ok => sum => :exact, :ok => (x -> mean(x)) => :frac_ok)

# ---------------------------------------------------------------------------
# §3 goal-minute sanity + red cards
# ---------------------------------------------------------------------------

goals_inc = subset(ds.incidents, :incident_type => ByRow(==("goal")))
mm = [ _incident_minute(r) for r in eachrow(goals_inc) ]
minute_sanity = (
    n             = length(mm),
    min           = minimum(mm), max = maximum(mm),
    # SofaScore clamps stoppage goals to exactly 45/90 with added_time=0 in this feed
    # (verified 2026-07-14: all 129 t=45 and 302 t=90 goals have added_time=0), so the
    # terminal NHPP slices must carry extended exposure (injury_time1/2 or league mean).
    frac_clamp_45 = mean(mm .== 45),
    frac_clamp_90 = mean(mm .== 90),
    sentinel_999  = sum(coalesce.(goals_inc.added_time, 0) .== 999),
)

reds = DataFrame(match_id=Int[], n_reds=Int[])
for mid in unique(cov.match_id[cov.has_inc])
    r = match_reds(ds, mid)
    nrow(r) > 0 && push!(reds, (mid, nrow(r)))
end
red_summary = (matches_with_red = nrow(reds),
               total_reds = sum(reds.n_reds; init=0),
               rate = nrow(reds) / max(sum(cov.has_inc), 1))

# ---------------------------------------------------------------------------
# §4 betfair in-play density (56 has the coverage; 57 only ~140 matches)
# ---------------------------------------------------------------------------

bf = ds.betfair_odds
inplay = subset(bf, :minutes_to_kickoff => ByRow(x -> 0.0 < x <= 130.0))
inplay = leftjoin(inplay, select(matches, :match_id, :tournament_id), on = :match_id)

function per_match_density(df)
    combine(groupby(df, [:tournament_id, :match_id])) do g
        one_x2 = subset(g, :selection => ByRow(in([:home, :draw, :away])))
        ts = sort(unique(one_x2.minutes_to_kickoff))
        gaps = length(ts) >= 2 ? diff(ts) : Float64[]
        # 5-min bins from t_w=5..115 with a full identifiable 1X2 in-window (LOCF 4min)
        nbins = 0; okbins = 0
        for t_w in 5.0:5.0:115.0
            nbins += 1
            p = latest_prices(g, t_w; staleness = 4.0)
            (haskey(p, :home) && haskey(p, :draw) && haskey(p, :away) &&
             length(p) >= 6) && (okbins += 1)
        end
        (; n_prints_1x2 = length(ts),
           gap_p50 = isempty(gaps) ? NaN : quantile(gaps, 0.5),
           gap_p90 = isempty(gaps) ? NaN : quantile(gaps, 0.9),
           frac_id_bins = okbins / nbins)
    end
end

density = per_match_density(inplay)
density_summary = combine(groupby(density, :tournament_id),
    nrow => :matches,
    :n_prints_1x2 => median => :prints_med,
    :gap_p50 => (x -> median(filter(!isnan, x))) => :gap_p50_med,
    :gap_p90 => (x -> median(filter(!isnan, x))) => :gap_p90_med,
    :frac_id_bins => mean => :frac_identifiable_bins,
    :n_prints_1x2 => (x -> mean(x .== 0)) => :frac_no_inplay)

# ---------------------------------------------------------------------------
# §5 clock-map anchoring sanity (matches with ≥1 goal AND in-play prints)
# ---------------------------------------------------------------------------

anchor_qa = DataFrame(match_id=Int[], tournament_id=Int[], n_goals=Int[],
                      n_jump_found=Int[], off1=Float64[], off2=Float64[])
inplay_mids = Set(density.match_id[density.n_prints_1x2 .> 0])
for r in eachrow(subset(cov, :has_inc => ByRow(identity)))
    (r.match_id in inplay_mids && r.total_goals > 0) || continue
    anchors = anchor_goals(bf, ds, r.match_id)
    isempty(anchors) && continue
    # a goal counts as "jump found" if its anchor moved off the prior-expected position
    expected(a) = a.mm + (a.mm <= 45 ? 3.0 : 18.0)
    found = sum(abs(a.g_w - expected(a)) > 1e-9 for a in anchors)
    cm = make_clock_map(anchors)
    push!(anchor_qa, (r.match_id, r.tournament_id, length(anchors), found,
                      45.0 - cm(45.0), 90.0 + 18.0 - cm(90.0 + 18.0)))
end
anchor_summary = combine(groupby(anchor_qa, :tournament_id),
    nrow => :matches,
    [:n_jump_found, :n_goals] => ((f, n) -> sum(f) / sum(n)) => :jump_found_rate,
    :off1 => median => :off1_med, :off1 => std => :off1_sd,
    :off2 => median => :off2_med, :off2 => std => :off2_sd)

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

QA = (
    season_cov      = season_cov,
    recon_summary   = recon_summary,
    recon           = recon,
    minute_sanity   = minute_sanity,
    red_summary     = red_summary,
    density_summary = density_summary,
    density         = density,
    anchor_summary  = anchor_summary,
    anchor_qa       = anchor_qa,
    summary = "seasons usable for NHPP training: " *
              join(string.(season_cov.season[season_cov.usable]), ", "),
)

@info "r00 data QA done" QA.summary
QA.season_cov |> println
QA.recon_summary |> println
QA.density_summary |> println
QA.anchor_summary |> println
