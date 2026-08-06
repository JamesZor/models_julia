#=
r04a_bbc_timeline_qa.jl — WP-A runner: Gate A for the BBC commentary timeline.

Blocking checks (plan §WP-A):
  §1 coverage        — 1,070 matches with live text, ZERO null minutes
  §2 goal reconcile  — BBC goal counts vs the SofaScore final score, ≥ 92%, failures
                       ENUMERATED AND CLASSIFIED; also reports the slug route side by
                       side so the size of the own-goal correction is on the record
  §3 reds vs incidents — agreement + minute MAE on matches carrying both sources.
                       Reds are the dominant in-play effect (γ_man ×1.70), so a minute
                       disagreement > 2 min on more than 10% of reds is a STOP
  §4 subs vs incidents — same, informational (no MVP depends on subs yet)
  §5 side attribution — how many events the three-way slug CASE leaves unresolved

Run on the kaimon server session (-t 16):
    ENV["JULIA_PKG_PRECOMPILE_AUTO"]="0"; using BayesianFootball
    include("current_development/inplay_scottish/r04a_bbc_timeline_qa.jl")
    GATE_A.verdict
=#

using DataFrames, Statistics, LibPQ, Serialization

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(@__DIR__, "l04_bbc_timeline.jl"))

const T_IDS = [56, 57]

# ---------------------------------------------------------------------------
# §0 fetch
# ---------------------------------------------------------------------------

conn = bbc_conn()
timeline = fetch_bbc_timeline(conn, T_IDS)
resolve_sides!(timeline; matches = ds.matches)
stoppage = fetch_stoppage(conn, T_IDS)
seqs = build_event_seqs(timeline; stoppage = stoppage)
close(conn)

# ---------------------------------------------------------------------------
# §1 coverage + null minutes
# ---------------------------------------------------------------------------

coverage = combine(groupby(timeline, [:tournament_id, :season_id]),
    :match_id => (x -> length(unique(x))) => :matches,
    nrow => :events,
    :time => (x -> count(ismissing, x)) => :null_minutes)
sort!(coverage, [:tournament_id, :season_id])

cov_stats = (n_matches = length(seqs),
             n_events = nrow(timeline),
             null_minutes = count(ismissing, timeline.time),
             by_type = sort(combine(groupby(timeline, :event_type), nrow => :n),
                            :n, rev = true),
             score_breaks = metadata(timeline, "score_breaks"),
             text_recovered = metadata(timeline, "text_recovered"))

# ---------------------------------------------------------------------------
# §2 goal reconciliation (running-score route) + the slug route for contrast
# ---------------------------------------------------------------------------

recon = reconcile_goals(seqs, ds.matches)
recon_by_season = combine(groupby(recon, [:tournament_id, :season]),
    nrow => :n, :ok => sum => :ok, :ok => mean => :rate)
sort!(recon_by_season, [:tournament_id, :season])

# Slug-only counterfactual: attribute goals by the BBC team slug (own goals therefore
# land on the WRONG side, or nowhere at all). This is the ~92% the plan pre-registered.
slug_recon = let
    g = subset(timeline, :event_type => ByRow(==("goal")))
    cnt = combine(groupby(g, :match_id),
        :is_home_event => (s -> count(x -> x === true, s)) => :sh,
        :is_home_event => (s -> count(x -> x === false, s)) => :sa)
    j = innerjoin(select(ds.matches, :match_id, :home_score, :away_score), cnt, on = :match_id)
    mean((j.sh .== j.home_score) .& (j.sa .== j.away_score))
end

recon_failures = subset(recon, :ok => ByRow(!))
# Classify each failure: does BBC over-count, under-count, or misattribute?
recon_failures.delta_h = recon_failures.bh .- recon_failures.fh
recon_failures.delta_a = recon_failures.ba .- recon_failures.fa
recon_failures.kind = map(eachrow(recon_failures)) do r
    tb = r.bh + r.ba; tf = r.fh + r.fa
    tb == tf ? "misattribution (totals agree)" :
    tb  < tf ? "BBC missing $(tf - tb) goal(s) — truncated feed" :
               "BBC extra $(tb - tf) goal(s) — feed/final-score disagreement"
end

# ---------------------------------------------------------------------------
# §3/§4 cross-check vs ds.incidents on matches carrying both
# ---------------------------------------------------------------------------

red_agree, red_min, red_per = cross_check_events(seqs, ds, :red)
sub_agree, sub_min, sub_per = cross_check_events(seqs, ds, :sub)

# ---------------------------------------------------------------------------
# §5 side attribution
# ---------------------------------------------------------------------------

side_qa = combine(groupby(timeline, :event_type),
    nrow => :n,
    :is_home_event => (x -> count(ismissing, x)) => :slug_unresolved,
    :side => (x -> count(ismissing, x)) => :final_unresolved)
sort!(side_qa, :n, rev = true)

# ---------------------------------------------------------------------------
# §6 stoppage time per half (r00's open clamping problem)
# ---------------------------------------------------------------------------

# SofaScore has the columns but not the data — confirm rather than assume, because r00
# proposed injury_time1/2 as the fix.
sofa_inj = DataFrame(LibPQ.execute(bbc_conn(),
    "SELECT match_id, injury_time1, injury_time2 FROM sofascore.matches " *
    "WHERE tournament_id = ANY('{56,57}')"))
sofa_zero = (n = nrow(sofa_inj),
             frac_zero_h1 = mean(coalesce.(sofa_inj.injury_time1, 0) .== 0),
             frac_zero_h2 = mean(coalesce.(sofa_inj.injury_time2, 0) .== 0))

st_df = DataFrame(match_id = collect(keys(stoppage)),
                  at1 = [v.at1 for v in values(stoppage)],
                  at2 = [v.at2 for v in values(stoppage)])
stoppage_stats = (
    matches = nrow(st_df),
    at1 = (mean = mean(st_df.at1), median = median(st_df.at1),
           q05 = quantile(st_df.at1, 0.05), q95 = quantile(st_df.at1, 0.95)),
    at2 = (mean = mean(st_df.at2), median = median(st_df.at2),
           q05 = quantile(st_df.at2, 0.05), q95 = quantile(st_df.at2, 0.95)),
    # how much exposure the incumbent's flat Tend=95 proxy misses, per match
    h2_vs_flat5 = mean(st_df.at2 .- 5.0),
    h1_dropped_by_incumbent = mean(st_df.at1),
    sofascore_injury_time_usable = !(sofa_zero.frac_zero_h1 > 0.95),
    sofa_zero = sofa_zero,
)

# Clock sanity: under :incumbent, H1 stoppage events sort AFTER the H2 kickoff; under
# :bbc they must not. Count goals whose half flips between the two conventions.
seqs_inc = build_event_seqs(timeline; stoppage = stoppage, clock = :incumbent)
clock_qa = let nflip = 0, ntot = 0
    for (mid, s) in seqs
        si = seqs_inc[mid]
        for (a, b) in zip(s.goals, si.goals)
            ntot += 1
            ((a.t < 45) != (b.t < 45)) && (nflip += 1)
        end
    end
    (goals = ntot, half_flips_vs_incumbent = nflip,
     frac = nflip / max(ntot, 1),
     max_bbc_clock = maximum(maximum(g.t for g in s.goals; init = 0.0) for s in values(seqs)))
end

# Exposure vector sanity on a median match.
expo_example = bin_exposure(2.0, 5.0)

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

pass_cov   = cov_stats.n_matches == 1070 && cov_stats.null_minutes == 0
pass_recon = mean(recon.ok) >= 0.92
pass_reds  = isnan(red_min.frac_gt_tol) ? false : red_min.frac_gt_tol <= 0.10

GATE_A = (
    coverage = coverage, cov_stats = cov_stats,
    recon = recon, recon_by_season = recon_by_season,
    recon_rate = mean(recon.ok), slug_recon_rate = slug_recon,
    recon_failures = recon_failures,
    red_agree = red_agree, red_minutes = red_min, red_per = red_per,
    sub_agree = sub_agree, sub_minutes = sub_min,
    side_qa = side_qa,
    stoppage = stoppage, stoppage_stats = stoppage_stats,
    clock_qa = clock_qa, expo_example = expo_example,
    pass = (coverage = pass_cov, reconciliation = pass_recon, reds = pass_reds),
    verdict = (pass_cov && pass_recon && pass_reds) ? "GATE A PASS" : "GATE A FAIL",
)

OUT = joinpath(@__DIR__, "out"); mkpath(OUT)
serialize(joinpath(OUT, "r04a_seqs.jls"), seqs)
serialize(joinpath(OUT, "r04a_stoppage.jls"), stoppage)
serialize(joinpath(OUT, "r04a_gate.jls"),
          (; GATE_A.coverage, GATE_A.recon, GATE_A.recon_by_season, GATE_A.side_qa,
             GATE_A.recon_failures, GATE_A.pass, GATE_A.verdict))

@info "Gate A" GATE_A.verdict GATE_A.cov_stats.n_matches GATE_A.cov_stats.null_minutes GATE_A.recon_rate GATE_A.slug_recon_rate
