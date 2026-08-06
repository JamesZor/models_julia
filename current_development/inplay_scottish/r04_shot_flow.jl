#=
r04_shot_flow.jl — WP-C runner: Gate C for MVP-1 (shot-flow NHPP).

THE PRE-REGISTERED QUESTION. The incumbent's game-state term was a clean null on Scottish
goals (`state − time` paired t = 0.51). Is that a real absence, or a resolution failure at
1.4 goals per team-match? At 9.7 shots per team-match there is 7× the counting information.

  **Pre-registered win condition: |t| > 2 on the state term at SHOT resolution.**
  A null is a publishable result — it closes the question the incumbent left open — and
  must be recorded exactly as a positive would be. It is NOT to be talked up.

  §1 assemble BBC shot slices on both pregame arms
  §2 GLM cv_race at shot resolution (k=4, repeats=5, split by MATCH) + paired_diff
  §3 the head-to-head: the SAME race at goal resolution on the SAME matches, so the
     comparison is resolution and nothing else
  §4 Bayesian fit + checkpoint_bias at 60/75/85′ (incumbent: +0.020/+0.014/+0.003)
  §5 optional state-dependent conversion (does p2 move, or only volume?)

Run on the kaimon server session (-t 16, pinthreads(:cores)):
    ENV["JULIA_PKG_PRECOMPILE_AUTO"]="0"; using BayesianFootball
    include("current_development/inplay_scottish/r04_shot_flow.jl")
    GATE_C.verdict
=#

using DataFrames, Statistics, Serialization, Random, LibPQ

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(@__DIR__, "l01_nhpp_scottish.jl"))
include(joinpath(@__DIR__, "l02_ppd_compose.jl"))
include(joinpath(@__DIR__, "l04_bbc_timeline.jl"))
include(joinpath(@__DIR__, "l05_pregame_source.jl"))
include(joinpath(@__DIR__, "l06_shot_flow.jl"))

OUT = joinpath(@__DIR__, "out"); mkpath(OUT)

const TRAIN_PAIRS = Set([(56, "24/25"), (56, "25/26"),
                         (57, "23/24"), (57, "24/25"), (57, "25/26")])

# ---------------------------------------------------------------------------
# §1 data — BBC timeline + funnel latents
# ---------------------------------------------------------------------------

seqs = if isfile(joinpath(OUT, "r04a_seqs.jls"))
    deserialize(joinpath(OUT, "r04a_seqs.jls"))
else
    conn = bbc_conn()
    tl = fetch_bbc_timeline(conn, [56, 57])
    resolve_sides!(tl; matches = ds.matches)
    st = fetch_stoppage(conn, [56, 57])
    s = build_event_seqs(tl; stoppage = st)
    close(conn); s
end

draws = Dict(n => pregame_draws(known_source(n), ds)
             for n in ("funnel_apm_xg", "funnel_winner"))

mseqs = Dict(n => assemble_matches(ds, d, TRAIN_PAIRS; seqs = seqs)
             for (n, d) in draws)

shot_slices = Dict(n => build_shot_slices(m) for (n, m) in mseqs)
goal_slices = Dict(n => build_slices_bbc(m)  for (n, m) in mseqs)

data_stats = DataFrame(
    arm = collect(keys(mseqs)),
    matches = [length(m) for m in values(mseqs)],
    shots = [sum(shot_slices[n].y) for n in keys(mseqs)],
    goals = [sum(goal_slices[n].y) for n in keys(mseqs)],
    shots_per_team_match = [sum(shot_slices[n].y) / (2 * length(mseqs[n])) for n in keys(mseqs)],
    goals_per_team_match = [sum(goal_slices[n].y) / (2 * length(mseqs[n])) for n in keys(mseqs)],
    red_rows = [sum(shot_slices[n].man_adv .!= 0) for n in keys(mseqs)])

# ---------------------------------------------------------------------------
# §2/§3 GATE C — the state term, at both resolutions, on the same matches
# ---------------------------------------------------------------------------

function race_of(df)
    cv = cv_race(df)
    (cv = cv,
     time_vs_pg = paired_diff(cv, :time, :pg_only),
     state_vs_time = paired_diff(cv, :state, :time),
     full_vs_state = paired_diff(cv, :full, :state))
end

races = Dict{String, Any}()
for n in keys(mseqs)
    races["$(n)_shots"] = race_of(shot_slices[n])
    races["$(n)_goals"] = race_of(goal_slices[n])
end

gate_c_table = DataFrame(
    arm = String[], resolution = String[], counts = Int[],
    t_time = Float64[], t_state = Float64[], t_man = Float64[],
    state_mean = Float64[], state_se = Float64[])
for n in sort(collect(keys(mseqs))), res in ("shots", "goals")
    r = races["$(n)_$res"]
    df = res == "shots" ? shot_slices[n] : goal_slices[n]
    push!(gate_c_table, (n, res, sum(df.y),
                         r.time_vs_pg.t, r.state_vs_time.t, r.full_vs_state.t,
                         r.state_vs_time.mean, r.state_vs_time.se))
end

# ---------------------------------------------------------------------------
# §4 Bayesian fit + checkpoint bias (funnel_apm_xg only; the arm race is WP-F)
# ---------------------------------------------------------------------------

sf_config = ShotFlowConfig()
sf_chain = Samplers.run_sampler(
    make_shotflow_model(shot_slices["funnel_apm_xg"], sf_config),
    Samplers.NUTSConfig(n_samples = 600, n_chains = 4, n_warmup = 300,
                        max_depth = 8, show_progress = false))

sf_post = DataFrame(param = Symbol[], mean = Float64[], sd = Float64[],
                    q05 = Float64[], q95 = Float64[], rhat = Float64[])
sf_ss = MCMCChains.summarystats(sf_chain)
for p in (:α, :β, :γ_tr, :γ_ld, :γ_man, :σ_time)
    v = _cv(sf_chain, p)
    push!(sf_post, (p, mean(v), std(v), quantile(v, 0.05), quantile(v, 0.95),
                    sf_ss[p, :rhat]))
end

sf_bias = shot_checkpoint_bias(mseqs["funnel_apm_xg"], sf_chain, sf_config)

# ---------------------------------------------------------------------------
# §5 does CONVERSION move with state, or only volume?
# ---------------------------------------------------------------------------

p2_config = ShotFlowConfig(state_p2 = true)
p2_chain = Samplers.run_sampler(
    make_shotflow_model(shot_slices["funnel_apm_xg"], p2_config),
    Samplers.NUTSConfig(n_samples = 600, n_chains = 4, n_warmup = 300,
                        max_depth = 8, show_progress = false))
p2_ss = MCMCChains.summarystats(p2_chain)
p2_post = DataFrame(param = Symbol[], mean = Float64[], sd = Float64[],
                    q05 = Float64[], q95 = Float64[], rhat = Float64[])
for p in (:κ_0, :κ_tr, :κ_ld)
    v = _cv(p2_chain, p)
    push!(p2_post, (p, mean(v), std(v), quantile(v, 0.05), quantile(v, 0.95),
                    p2_ss[p, :rhat]))
end

# Raw empirical conversion by game state — the model-free version of the same question.
conv_by_state = let df = shot_slices["funnel_apm_xg"]
    df2 = copy(df)
    df2.state = [t == 1 ? "trailing" : l == 1 ? "leading" : "level"
                 for (t, l) in zip(df.trailing, df.leading)]
    combine(groupby(df2, :state), :y => sum => :shots, :y_goals => sum => :goals,
            [:y_goals, :y] => ((g, s) -> sum(g) / max(sum(s), 1)) => :conversion)
end

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

shot_t = maximum(abs(r.t_state) for r in eachrow(gate_c_table) if r.resolution == "shots")
goal_t = maximum(abs(r.t_state) for r in eachrow(gate_c_table) if r.resolution == "goals")

GATE_C = (
    data_stats = data_stats, races = races, table = gate_c_table,
    post = sf_post, bias = sf_bias, p2_post = p2_post, conv_by_state = conv_by_state,
    max_rhat = maximum(skipmissing(sf_ss[:, :rhat])),
    shot_state_t = shot_t, goal_state_t = goal_t,
    verdict = shot_t > 2 ? "GATE C PASS — state term resolves at shot resolution" :
                           "GATE C NULL — state does not move intensity, even at 7x counts",
)

serialize(joinpath(OUT, "r04_shot_chain.jls"), sf_chain)
serialize(joinpath(OUT, "r04_gate_c.jls"),
          (; GATE_C.data_stats, GATE_C.table, GATE_C.post, GATE_C.bias,
             GATE_C.p2_post, GATE_C.conv_by_state, GATE_C.verdict))

@info "Gate C" GATE_C.verdict GATE_C.shot_state_t GATE_C.goal_state_t
