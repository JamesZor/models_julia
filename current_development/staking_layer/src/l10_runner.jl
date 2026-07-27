#=
LOADER l10 — the season race harness + metrics + reporting.

ONE runner for both sources (SimSource / RealSource) — it consumes the source's loaded stream
(matches, model_sel, model_dists) and a set of NAMED policies, maintains a single shared TrustHist,
refits each trust-bearing policy on a shared cadence (fixed-interval or an explicit week-boundary
set, with optional EMA smoothing), and compounds bankrolls sequentially with a ruin freeze. This
collapses the old run_season (sim) and run_real_season (real) into a single function.

`run_ext_race` is the analogous loop for the extended multi-market book (ExtMatch), where each
"policy" is just a trust model staked via the extended coherent tilt; `drop_fams` curates whole
market families out (e.g. CorrectScore). Depends on l01–l09.
=#

using Statistics
using Printf
using LinearAlgebra: dot
using Random

# ---------- metrics ----------

"Terminal wealth, growth/match, max drawdown from a log-return vector."
function summarize_logw(logw::Vector{Float64})
    cw = cumsum(logw)
    peak = accumulate(max, cw)
    return (terminal_W=exp(sum(logw)), G_per_match=mean(logw),
            max_dd=1.0 - exp(minimum(cw .- peak)))
end

# ---------- core-book race ----------

"""
    run_race(loaded, named_policies; refit_every=25, refit_at=nothing, ema_alpha=1.0,
             ruin_floor=0.01, seed=1, track=nothing)

`loaded` = a source's load_matches output. `named_policies` = Vector of `name => policy`.
Refit trust when `(i-1) % refit_every == 0` (i>1) OR `i ∈ refit_at`. `ema_alpha<1` smooths the
per-unit weights across refits (point estimate only). Returns per-policy logw/attribution/w-trace.
"""
function run_race(loaded, named_policies::AbstractVector;
                  refit_every::Int=25, refit_at::Union{Nothing,AbstractSet{Int}}=nothing,
                  ema_alpha::Float64=1.0, ruin_floor::Float64=0.01, seed::Int=1)
    matches, model_sel, model_dists = loaded.matches, loaded.model_sel, loaded.model_dists
    n = length(matches)
    names = [p.first for p in named_policies]
    pols  = Dict(p.first => p.second for p in named_policies)
    rng = Xoshiro(seed)

    hist = TrustHist()
    for (pm, ps) in zip(loaded.prehist.matches, loaded.prehist.model_sel)
        push_hist!(hist, pm, ps)
    end

    fitted = Dict{String,Any}(); wema = Dict{String,Vector{Float64}}()
    for nm in names
        needs_trust(pols[nm]) || continue
        f = fit_trust(pols[nm].trust, hist)
        fitted[nm] = f
        wema[nm] = hasproperty(f, :w) ? copy(f.w) : Float64[]
    end

    logw   = Dict(nm => Float64[] for nm in names)
    nbets  = Dict(nm => 0   for nm in names)
    turn   = Dict(nm => 0.0 for nm in names)
    ruined = Dict(nm => false for nm in names)
    cumW   = Dict(nm => 1.0 for nm in names)
    fam_profit = Dict((nm, f) => 0.0 for nm in names, f in 1:3)
    fam_turn   = Dict((nm, f) => 0.0 for nm in names, f in 1:3)
    fam_nbets  = Dict((nm, f) => 0   for nm in names, f in 1:3)
    sel_profit = Dict(nm => zeros(11) for nm in names)
    sel_turn   = Dict(nm => zeros(11) for nm in names)
    sel_nbets  = Dict(nm => zeros(Int, 11) for nm in names)
    sel_wins   = Dict(nm => zeros(Int, 11) for nm in names)

    w_trace = Dict(nm => Tuple{Int,Vector{Float64}}[] for nm in names if needs_trust(pols[nm]))
    for nm in keys(w_trace)
        hasproperty(fitted[nm], :w) && push!(w_trace[nm], (1, copy(fitted[nm].w)))
    end
    max_tilt_err = 0.0

    for i in 1:n
        m = matches[i]; msel = model_sel[i]; mdist = model_dists[i]

        do_refit = refit_at === nothing ? (i > 1 && (i - 1) % refit_every == 0) : (i in refit_at)
        if do_refit
            for nm in names
                needs_trust(pols[nm]) || continue
                f = fit_trust(pols[nm].trust, hist)
                if ema_alpha < 1.0 && hasproperty(f, :w)
                    wema[nm] = (1.0 - ema_alpha) .* wema[nm] .+ ema_alpha .* f.w
                    fitted[nm] = FittedConstantTrust(copy(wema[nm]))
                else
                    fitted[nm] = f
                end
                hasproperty(fitted[nm], :w) && push!(w_trace[nm], (i, copy(fitted[nm].w)))
            end
        end

        # verification: w=1 grid tilt reproduces the smile over-probs (units 4/6/8)
        mult_one = coherent_multiplier(m.pbar, blend_targets(msel, m.q_mkt, ones(7)); cycles=50)
        gtilt = normalize_mult(m.pbar, mult_one)
        for k in (4, 6, 8)
            max_tilt_err = max(max_tilt_err, abs(dot(Float64.(SEL_MASKS[k]), gtilt) - msel[k]))
        end

        for nm in names
            if ruined[nm]
                push!(logw[nm], 0.0); continue
            end
            a = stake_for(pols[nm], m, msel, mdist, get(fitted, nm, nothing); rng=rng)
            r = max(match_return(a, m), 1e-12)
            push!(logw[nm], log(r)); cumW[nm] *= r
            nbets[nm] += count(>(1e-8), a); turn[nm] += sum(a)
            is_perbet = pols[nm] isa PerBetKellyPolicy
            for k in 1:11
                a[k] <= 1e-8 && continue
                f = FAM_OF_SEL[k]
                rr = m.won[k] ? (m.d[k] - 1.0) : -1.0
                fam_profit[(nm, f)] += a[k] * rr; fam_turn[(nm, f)] += a[k]; fam_nbets[(nm, f)] += 1
                if is_perbet
                    sel_profit[nm][k] += a[k] * rr; sel_turn[nm][k] += a[k]
                    sel_nbets[nm][k] += 1; sel_wins[nm][k] += m.won[k]
                end
            end
            cumW[nm] < ruin_floor && (ruined[nm] = true)
        end

        push_hist!(hist, m, msel)   # settled AFTER betting — no leakage
    end

    return (; names, logw, nbets, turn, ruined, cumW,
            fam_profit, fam_turn, fam_nbets, sel_profit, sel_turn, sel_nbets, sel_wins,
            w_trace, max_tilt_err, n, pols)
end

# ---------- reporting ----------

function summary_rows(rs)
    rows = String[]
    push!(rows, @sprintf("%-20s %10s %12s %8s %7s %9s %8s", "policy", "term_W", "G/match±SE", "maxDD", "n_bets", "turnover", "ruined"))
    for nm in rs.names
        lw = rs.logw[nm]; sm = summarize_logw(lw); se = std(lw) / sqrt(length(lw))
        push!(rows, @sprintf("%-20s %10.4f  %+7.5f±%.5f %7.3f %7d %9.2f %8s",
              nm, sm.terminal_W, sm.G_per_match, se, sm.max_dd, rs.nbets[nm], rs.turn[nm],
              rs.ruined[nm] ? "YES" : "-"))
    end
    return rows
end

function family_rows(rs)
    fam_name = ("1X2", "totals", "BTTS")
    rows = String[]
    push!(rows, @sprintf("%-20s %-8s %9s %9s %8s %8s", "policy", "family", "profit", "turnover", "roi%", "n_bets"))
    for nm in rs.names, f in 1:3
        t = rs.fam_turn[(nm, f)]; p = rs.fam_profit[(nm, f)]
        push!(rows, @sprintf("%-20s %-8s %+9.4f %9.4f %+8.2f %8d", nm, fam_name[f], p, t, t > 0 ? 100p/t : 0.0, rs.fam_nbets[(nm, f)]))
    end
    return rows
end

"b21 adapter cross-check: a PerBetKelly policy's per-selection ROI% signs vs a reference dict."
function crosscheck_rows(rs, policy_name::String, b21::Dict)
    rows = String[]
    push!(rows, @sprintf("%-10s %9s %9s %7s %8s %6s", "selection", "PB_roi%", "b21_roi%", "sign", "PB_n", "wins"))
    agree = 0; tot = 0
    for k in 1:11
        nm = SEL_NAMES[k]
        roi = rs.sel_turn[policy_name][k] > 0 ? 100 * rs.sel_profit[policy_name][k] / rs.sel_turn[policy_name][k] : 0.0
        b = get(b21, nm, NaN)
        ok = !isnan(b) && sign(roi) == sign(b); tot += 1; agree += ok
        push!(rows, @sprintf("%-10s %+9.2f %+9.2f %7s %8d %6d", nm, roi, b, ok ? "OK" : "x", rs.sel_nbets[policy_name][k], rs.sel_wins[policy_name][k]))
    end
    push!(rows, @sprintf("sign agreement: %d/%d", agree, tot))
    return rows
end

"EB/Bayes w-trajectory for one policy: rows = refit points, cols = the 7 units."
function wtrace_rows(rs, policy_name::String)
    rows = String[]
    haskey(rs.w_trace, policy_name) || return ["(no w-trace for $policy_name)"]
    push!(rows, @sprintf("%6s  %s", "match", join([@sprintf("%9s", u) for u in UNIT_NAMES], "")))
    for (i, w) in rs.w_trace[policy_name]
        push!(rows, @sprintf("%6d  %s", i, join([@sprintf("%9.3f", x) for x in w], "")))
    end
    return rows
end

# ---------- extended-book race ----------

match_return_ext(a::Vector{Float64}, em::ExtMatch) = 1.0 + dot(a, em.settle)

"""
    run_ext_race(ext_loaded, trust_specs; cap=0.2, refit_every=25, drop_fams=Set{Int}(),
                 ruin_floor=0.01)

Each `trust_spec` = `name => AbstractTrustModel`, staked via the extended coherent tilt + capped
unified Kelly over the full book. `drop_fams` (FAM_ID values) curates whole families out.
"""
function run_ext_race(ext_loaded, trust_specs::AbstractVector;
                      cap::Float64=0.2, refit_every::Int=25, drop_fams=Set{Int}(),
                      ruin_floor::Float64=0.01)
    ems = ext_loaded.matches
    n = length(ems); names = [t.first for t in trust_specs]; models = Dict(t.first => t.second for t in trust_specs)
    hist = TrustHist()
    fitted = Dict(nm => fit_trust(models[nm], hist) for nm in names)
    logw = Dict(nm => Float64[] for nm in names); ruined = Dict(nm => false for nm in names); cumW = Dict(nm => 1.0 for nm in names)
    fam_profit = Dict((nm, f) => 0.0 for nm in names, f in 1:7); fam_turn = Dict((nm, f) => 0.0 for nm in names, f in 1:7)
    w_trace = Dict(nm => Tuple{Int,Vector{Float64}}[] for nm in names)

    for i in 1:n
        em = ems[i]
        if i > 1 && (i - 1) % refit_every == 0
            for nm in names
                fitted[nm] = fit_trust(models[nm], hist)
                hasproperty(fitted[nm], :w) && push!(w_trace[nm], (i, copy(fitted[nm].w)))
            end
        end
        keep = [k for k in 1:length(em.d) if !(em.fam[k] in drop_fams)]
        for nm in names
            if ruined[nm] || isempty(keep)
                push!(logw[nm], 0.0); continue
            end
            w = trust_weights(fitted[nm])          # match-free: ext book uses the 7 core units
            tp = ext_tilted_pbar(em, w)
            asub = solve_P(tp, em.R[:, keep]; cap=cap)
            a = zeros(length(em.d)); a[keep] = asub
            r = max(match_return_ext(a, em), 1e-12)
            push!(logw[nm], log(r)); cumW[nm] *= r
            for k in 1:length(em.d)
                a[k] <= 1e-8 && continue
                fam_profit[(nm, em.fam[k])] += a[k] * em.settle[k]; fam_turn[(nm, em.fam[k])] += a[k]
            end
            cumW[nm] < ruin_floor && (ruined[nm] = true)
        end
        push_hist_ext!(hist, em)
    end
    return (; names, logw, ruined, cumW, fam_profit, fam_turn, w_trace, n)
end
