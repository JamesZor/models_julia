#=
RUNNER — trust-LEARNING dynamics: learning rate × refit cadence (core-11 book, src_sup40_sw40).

Two hyperparameters of the EB trust ESTIMATOR were fixed arbitrarily in r01 (full memory,
refit every 25 matches). This sweeps them, framing cadence as TIME (Ireland ≈ 5 matches/week):

  • Learning rate A — forgetting HALF-LIFE H (weeks): fit_trust_eb(halflife=H·5). Short H = short
    memory = reacts fast to recent form/drift; Inf = full memory (r01 default).
  • Learning rate B — EMA STEP-SIZE α on each refit: w ← (1−α)·w + α·w_fit. α=1 = no smoothing
    (r01); small α = w crawls toward the new fit.
  • Refit CADENCE K (weeks): refit on week/round boundaries (dense ISO-week index from match_date,
    so the 97-day off-season gap doesn't distort cadence), every K weeks.

Answers: is the model's per-line bias STATIONARY (⇒ full memory + moderate cadence wins; fast
learning just adds noise — the staking-sim E2 "time-decay = no-op") or DRIFTING (⇒ short H / big α
/ weekly refit pays off)? Metrics per cell: terminal W, growth, maxDD, reactivity (final home w +
weeks-to-home-w<0.35) and stability (jitter = mean |Δw| between refits).

Only TRUST_EB_U_cap02 depends on these knobs (CURATED/FLAT are w-fixed → run once as reference).
Assumes `built` (core c=0.02 SimMatch book) + `matches_df` live in the session.
=#

using Statistics, Printf, Dates

include(joinpath(@__DIR__, "r01_race_src_sup40.jl"))

const WK_MATCHES = 5                    # Ireland Premier ≈ 5 matches / round (median, verified)
const HL_WEEKS   = [Inf, 8.0, 4.0, 2.0, 1.0]
const EMA_ALPHAS = [1.0, 0.7, 0.5, 0.3, 0.15]
const CADENCE_WK = [1, 2, 3, 4, 6, 8]

"Dense ISO-week index per built match (kickoff order), robust to the off-season gap."
function match_weeks(built, matches_df)
    md = Dict(r.match_id => Date(r.match_date) for r in eachrow(matches_df))
    keys = [year(md[mid]) * 53 + week(md[mid]) for mid in built.mids]
    uk = sort(unique(keys)); rank = Dict(k => i for (i, k) in enumerate(uk))
    return [rank[k] for k in keys]
end

"Match indices that begin a new K-week block ⇒ refit points (never match 1 — cold start)."
function refit_indices(weeks::Vector{Int}, K::Int)
    s = Set{Int}(); last = -1
    for (i, w) in enumerate(weeks)
        blk = (w - 1) ÷ K
        if blk != last
            i > 1 && push!(s, i); last = blk
        end
    end
    return s
end

"Reactivity + stability read from a run's w-trajectory (w_trace[1] is the cold start)."
function traj_metrics(rs; thresh=0.35)
    wt = rs.w_trace
    jit = length(wt) < 3 ? NaN :
          mean(mean(abs.(wt[k+1][2] .- wt[k][2])) for k in 2:(length(wt)-1))
    react = NaN
    for (i, w) in wt
        if w[1] < thresh; react = Float64(i); break; end
    end
    return (jitter=jit, react_match=react, home_final=rs.w_final[1])
end

_run_eb(built, weeks; hl, al, K) = run_real_season(
    built; strategies=["TRUST_EB_U_cap02"], halflife=hl, ema_alpha=al, refit_at=refit_indices(weeks, K))

"Sweep one learning-rate mechanism × cadence → rows of metrics."
function grid_run(built, weeks; mode::Symbol, params, Ks=CADENCE_WK)
    rows = NamedTuple[]
    for p in params, K in Ks
        hl = mode === :halflife ? (isinf(p) ? Inf : p * WK_MATCHES) : Inf
        al = mode === :ema ? p : 1.0
        rs = _run_eb(built, weeks; hl=hl, al=al, K=K)
        sm = summarize_logw(rs.logw["TRUST_EB_U_cap02"]); tm = traj_metrics(rs)
        push!(rows, (param=p, K=K, termW=sm.terminal_W, G=sm.G_per_match, maxDD=sm.max_dd,
                     jitter=tm.jitter, react=tm.react_match, home=tm.home_final,
                     n_refits=length(refit_indices(weeks, K))))
    end
    return rows
end

# ---------- reporting ----------

_pname(mode, p) = mode === :halflife ? (isinf(p) ? "H=Inf" : @sprintf("H=%gwk", p)) : @sprintf("α=%.2f", p)

"param × K matrix of one metric field."
function matrix_block(rows, params, Ks, field, label, mode; fmt="%8.3f")
    lines = ["", "$label  (rows = learning rate, cols = refit cadence in weeks)"]
    push!(lines, @sprintf("%-9s", "") * join([@sprintf("%8s", "K=$K") for K in Ks], ""))
    for p in params
        vals = [getfield(first(filter(x -> x.param == p && x.K == K, rows)), field) for K in Ks]
        push!(lines, @sprintf("%-9s", _pname(mode, p)) *
              join([isnan(v) ? @sprintf("%8s", "—") : Printf.format(Printf.Format(fmt), v) for v in vals], ""))
    end
    return lines
end

function run_and_report_wlearn(built, matches_df; outdir=joinpath(@__DIR__, "results"))
    mkpath(outdir)
    weeks = match_weeks(built, matches_df)
    rs_base = run_real_season(built; strategies=["TRUST_EB_U_cap02"])   # r01 default: H=Inf, α=1, refit_every=25
    sb = summarize_logw(rs_base.logw["TRUST_EB_U_cap02"]); tb = traj_metrics(rs_base)

    ghl = grid_run(built, weeks; mode=:halflife, params=HL_WEEKS)
    gem = grid_run(built, weeks; mode=:ema,      params=EMA_ALPHAS)

    lines = String["TRUST-LEARNING DYNAMICS — src_sup40_sw40 · core-11 book · c=0.02 · n=$(length(built.matches))",
                   "Ireland ≈ $(WK_MATCHES) matches/week; refit on week boundaries. TRUST_EB_U_cap02 only.",
                   @sprintf("BASELINE (r01: H=Inf, α=1, refit_every=25 ≈ 5wk): termW %.3f  G %+.5f  maxDD %.3f  home_w %.3f",
                            sb.terminal_W, sb.G_per_match, sb.max_dd, tb.home_final)]

    push!(lines, "", "#"^70, "# GRID A — forgetting HALF-LIFE H (weeks)  ×  cadence K (weeks)", "#"^70)
    append!(lines, matrix_block(ghl, HL_WEEKS, CADENCE_WK, :termW, "terminal W",  :halflife; fmt="%8.2f"))
    append!(lines, matrix_block(ghl, HL_WEEKS, CADENCE_WK, :G,     "G / match",   :halflife; fmt="%+8.4f"))
    append!(lines, matrix_block(ghl, HL_WEEKS, CADENCE_WK, :maxDD, "max drawdown",:halflife; fmt="%8.3f"))
    append!(lines, matrix_block(ghl, HL_WEEKS, CADENCE_WK, :jitter,"w jitter (stability↓)", :halflife; fmt="%8.4f"))
    append!(lines, matrix_block(ghl, HL_WEEKS, CADENCE_WK, :home,  "final home w (reactivity)", :halflife; fmt="%8.3f"))

    push!(lines, "", "#"^70, "# GRID B — EMA step-size α  ×  cadence K (weeks)", "#"^70)
    append!(lines, matrix_block(gem, EMA_ALPHAS, CADENCE_WK, :termW, "terminal W",  :ema; fmt="%8.2f"))
    append!(lines, matrix_block(gem, EMA_ALPHAS, CADENCE_WK, :G,     "G / match",   :ema; fmt="%+8.4f"))
    append!(lines, matrix_block(gem, EMA_ALPHAS, CADENCE_WK, :maxDD, "max drawdown",:ema; fmt="%8.3f"))
    append!(lines, matrix_block(gem, EMA_ALPHAS, CADENCE_WK, :jitter,"w jitter (stability↓)", :ema; fmt="%8.4f"))
    append!(lines, matrix_block(gem, EMA_ALPHAS, CADENCE_WK, :home,  "final home w (reactivity)", :ema; fmt="%8.3f"))

    body = join(lines, "\n"); write(joinpath(outdir, "e_wlearn.txt"), body)

    _csv(path, rows, pcol) = begin
        io = IOBuffer(); println(io, "$pcol,K_weeks,terminal_W,G_per_match,maxDD,jitter,react_match,home_w,n_refits")
        for r in rows
            println(io, @sprintf("%s,%d,%.6f,%.6f,%.6f,%.6f,%s,%.4f,%d",
                    isinf(r.param) ? "Inf" : string(r.param), r.K, r.termW, r.G, r.maxDD, r.jitter,
                    isnan(r.react) ? "" : string(Int(r.react)), r.home, r.n_refits))
        end
        write(path, String(take!(io)))
    end
    _csv(joinpath(outdir, "wlearn_halflife.csv"), ghl, "halflife_weeks")
    _csv(joinpath(outdir, "wlearn_ema.csv"),      gem, "ema_alpha")

    # example trajectories: baseline + fast/short-memory + slow-EMA + laggy
    examples = [("baseline_Hinf_5wk", rs_base),
                ("Hfast_1wk_K1", _run_eb(built, weeks; hl=1*WK_MATCHES, al=1.0, K=1)),
                ("EMAslow_a015_K1", _run_eb(built, weeks; hl=Inf, al=0.15, K=1)),
                ("laggy_Hinf_K8", _run_eb(built, weeks; hl=Inf, al=1.0, K=8))]
    io = IOBuffer(); println(io, "config,match," * join(UNIT_NAMES, ","))
    for (lbl, rs) in examples, (i, w) in rs.w_trace
        println(io, lbl * "," * string(i) * "," * join(string.(round.(w, digits=5)), ","))
    end
    write(joinpath(outdir, "w_trace_wlearn_examples.csv"), String(take!(io)))

    return (ghl=ghl, gem=gem, baseline=(termW=sb.terminal_W, G=sb.G_per_match, maxDD=sb.max_dd, home=tb.home_final),
            body=body, weeks=weeks)
end
