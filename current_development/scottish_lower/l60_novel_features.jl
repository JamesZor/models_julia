# ==============================================================================
# l60 — LOADER: candidate match-level covariates and the evaluation harness
# ==============================================================================
#
# Definitions only, no execution. Paired with r60.
#
# THE THESIS. The shipped pxG covariate collapses a rich measurement into one raw
# rolling mean. Three things are thrown away in that collapse, and each is a candidate:
#
#   1. OPPONENT. `att_h` is the mean pxG a side created, unadjusted for who it played.
#      The engine adjusts GOALS for opponent through dyn.alpha/dyn.beta; nothing adjusts
#      pxG. Family A fits attack and defence ratings on the pxG matrix itself.
#   2. VOLUME vs QUALITY. pxG = shots x mean xG per shot. Shot volume is a persistent
#      team property; finishing quality is famously not. Averaging them into one number
#      makes a side that takes 20 poor shots look like one that takes 8 good ones.
#      Family B separates them.
#   3. OVER-PERFORMANCE. `goals - pxG` is the classic mean-reversion signal, and the
#      measurement ladder already computes both quantities before discarding one.
#      Family C keeps the residual.
#
# Two further families use data nothing in the repo currently touches:
#   4. SQUAD CONTINUITY — how much of today's XI started the last match.
#   5. MINUTES LOAD and REST — accumulated recent minutes, and days since the last game.
#
# EVERY CANDIDATE IS POINT-IN-TIME BY THE SAME RULE AS src/features/pxg.jl: matches are
# walked in kickoff order and fixtures sharing a kickoff are emitted as a group before
# any of them updates state. The opponent-adjusted solver inherits this for free by
# refitting once per distinct kickoff on strictly earlier matches only.
# ==============================================================================

using DataFrames
using Dates
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

const L60_FEATURES = BayesianFootball.Features
const L60_DATA = BayesianFootball.Data

# ==============================================================================
# 1. PER-MATCH OBSERVATIONS
# ==============================================================================

"""
    l60_records(ds; k) -> Dict{Int, NamedTuple}

`(pxg_h, pxg_a, shots_h, shots_a, goals_h, goals_a, covered)` per match.

`covered` marks fixtures whose pxG came from live-text commentary. Only those carry a
real shot count, so every volume- or quality-derived candidate is restricted to them —
a shot-count fallback would supply a denominator the numerator never saw.
"""
function l60_records(ds::L60_DATA.DataStore; k::Float64 = 25.0)
    shots = L60_FEATURES.build_shots(ds)
    per_match = Dict{Int, NTuple{4, Float64}}()   # pxg_h, pxg_a, n_h, n_a
    if nrow(shots) > 0
        model = L60_FEATURES.fit_shot_xg(shots; k = k)
        predicted = L60_FEATURES.predict_xg(model, shots)
        for (i, r) in enumerate(eachrow(shots))
            ismissing(r.is_home) && continue
            mid = Int(r.match_id)
            xg = predicted[i]
            isfinite(xg) || continue
            h_xg, a_xg, h_n, a_n = get(per_match, mid, (0.0, 0.0, 0.0, 0.0))
            per_match[mid] = r.is_home === true ? (h_xg + xg, a_xg, h_n + 1, a_n) :
                                                  (h_xg, a_xg + xg, h_n, a_n + 1)
        end
    end

    records = Dict{Int, NamedTuple}()
    for row in eachrow(ds.matches)
        mid = Int(row.match_id)
        gh = ismissing(row.home_score) ? NaN : Float64(row.home_score)
        ga = ismissing(row.away_score) ? NaN : Float64(row.away_score)
        (isfinite(gh) && isfinite(ga)) || continue
        if haskey(per_match, mid)
            h_xg, a_xg, h_n, a_n = per_match[mid]
            records[mid] = (pxg_h = h_xg, pxg_a = a_xg, shots_h = h_n, shots_a = a_n,
                            goals_h = gh, goals_a = ga, covered = true)
        else
            records[mid] = (pxg_h = NaN, pxg_a = NaN, shots_h = NaN, shots_a = NaN,
                            goals_h = gh, goals_a = ga, covered = false)
        end
    end
    return records
end

"Matches in kickoff order, with the tie groups the point-in-time rule needs."
function l60_schedule(ds::L60_DATA.DataStore)
    rows = [(id = Int(r.match_id),
             kickoff = L60_FEATURES._pxg_kickoff(r),
             home = String(r.home_team),
             away = String(r.away_team)) for r in eachrow(ds.matches)]
    sort!(rows, by = r -> (r.kickoff, r.id))
    return rows
end

# ==============================================================================
# 2. FAMILY A — OPPONENT-ADJUSTED pxG RATINGS
# ==============================================================================
#
#   log(pxG_h + eps) = mu + gamma + alpha_h + beta_a
#   log(pxG_a + eps) = mu       + alpha_a + beta_h
#
# solved as a ridge-penalised weighted least squares over every EARLIER match, refit
# once per distinct kickoff. alpha is attacking strength, beta is defensive leak, gamma
# is home advantage on the pxG scale. Identification needs a constraint — the ridge
# supplies it by shrinking alpha and beta toward zero, which also centres them.
#
# This is the same estimator shape as the goals engine's dynamics, applied to a
# measurement with two to three times the information per match. That is the whole
# hypothesis: pxG ratings should converge in fewer matches than goal ratings.

"""
    l60_adjusted_pxg(ds, records; half_life_days, lambda, min_matches) -> Dict{Int,NamedTuple}

`(sup, level, available)` per match — the opponent-adjusted expected-pxG difference and
total, in log space.
"""
function l60_adjusted_pxg(ds::L60_DATA.DataStore, records::Dict{Int, <:NamedTuple};
                          half_life_days::Float64 = 180.0,
                          lambda::Float64 = 4.0,
                          min_matches::Int = 40,
                          eps::Float64 = 0.15)
    schedule = l60_schedule(ds)
    teams = sort(unique(vcat([r.home for r in schedule], [r.away for r in schedule])))
    index = Dict(t => i for (i, t) in enumerate(teams))
    n_teams = length(teams)
    n_par = 2 * n_teams + 2                     # alpha, beta, gamma, mu

    out = Dict{Int, NamedTuple{(:sup, :level, :available), Tuple{Float64, Float64, Float64}}}()

    # Accumulated history: one row per team-match side.
    hist_att = Int[]; hist_def = Int[]; hist_home = Float64[]
    hist_y = Float64[]; hist_day = Float64[]

    i = 1
    n_rows = length(schedule)
    while i <= n_rows
        j = i
        while j <= n_rows && schedule[j].kickoff == schedule[i].kickoff
            j += 1
        end
        today = Dates.value(Date(schedule[i].kickoff))

        beta_hat = nothing
        if length(hist_y) >= 2 * min_matches
            weights = 0.5 .^ ((today .- hist_day) ./ half_life_days)
            m = length(hist_y)
            I = Int[]; J = Int[]; V = Float64[]
            for r in 1:m
                push!(I, r); push!(J, hist_att[r]);           push!(V, 1.0)
                push!(I, r); push!(J, n_teams + hist_def[r]); push!(V, 1.0)
                push!(I, r); push!(J, 2 * n_teams + 1);       push!(V, hist_home[r])
                push!(I, r); push!(J, 2 * n_teams + 2);       push!(V, 1.0)
            end
            X = sparse(I, J, V, m, n_par)
            W = Diagonal(weights)
            A = Matrix(Symmetric(Matrix(X' * W * X)))
            b = Vector(X' * (weights .* hist_y))
            # Penalise the team effects only; mu and gamma are nuisance parameters and
            # shrinking them would bias the level and the home advantage.
            penalty = vcat(ones(2 * n_teams), 0.0, 0.0)
            A .+= lambda .* Diagonal(penalty)
            beta_hat = try
                cholesky(Symmetric(A + 1e-9I)) \ b
            catch
                pinv(Symmetric(A)) * b
            end
        end

        for t in i:(j - 1)
            r = schedule[t]
            if beta_hat === nothing
                out[r.id] = (sup = 0.0, level = 0.0, available = 0.0)
                continue
            end
            h = index[r.home]; a = index[r.away]
            gamma = beta_hat[2 * n_teams + 1]
            mu = beta_hat[2 * n_teams + 2]
            eta_h = mu + gamma + beta_hat[h] + beta_hat[n_teams + a]
            eta_a = mu + beta_hat[a] + beta_hat[n_teams + h]
            (isfinite(eta_h) && isfinite(eta_a)) || (out[r.id] = (sup = 0.0, level = 0.0, available = 0.0); continue)
            out[r.id] = (sup = eta_h - eta_a, level = eta_h + eta_a - 2 * mu, available = 1.0)
        end

        for t in i:(j - 1)
            r = schedule[t]
            rec = get(records, r.id, nothing)
            (rec === nothing || !rec.covered) && continue
            (isfinite(rec.pxg_h) && isfinite(rec.pxg_a)) || continue
            h = index[r.home]; a = index[r.away]
            push!(hist_att, h); push!(hist_def, a); push!(hist_home, 1.0)
            push!(hist_y, log(rec.pxg_h + eps)); push!(hist_day, Float64(today))
            push!(hist_att, a); push!(hist_def, h); push!(hist_home, 0.0)
            push!(hist_y, log(rec.pxg_a + eps)); push!(hist_day, Float64(today))
        end
        i = j
    end
    return out
end

# ==============================================================================
# 3. FAMILIES B, C — ROLLING TEAM QUANTITIES
# ==============================================================================

"""
    l60_rolling(ds, records, extract; k, kappa, require_covered) -> Dict{Int,NamedTuple}

A generic point-in-time rolling walk. `extract(record) -> (for_home, against_home,
for_away, against_away)` names the quantity; the walk returns the four shrunk deviations
from the running league mean, plus the assembled supremacy and level.

Identical tie-group semantics to `Features._pxg_rolling_lookup`: a fixture never sees
itself or anything sharing its kickoff.
"""
function l60_rolling(ds::L60_DATA.DataStore, records::Dict{Int, <:NamedTuple}, extract;
                     k::Int = 8, kappa::Float64 = 3.0, require_covered::Bool = true,
                     min_matches::Int = 3)
    schedule = l60_schedule(ds)
    scored = Dict{String, Vector{Float64}}()
    conceded = Dict{String, Vector{Float64}}()
    base_sum = 0.0; base_n = 0
    out = Dict{Int, NamedTuple{(:att_h, :def_h, :att_a, :def_a, :sup, :level, :available),
                               NTuple{7, Float64}}}()

    shrunk(vals, baseline) = begin
        n = length(vals)
        m = k <= 0 ? n : min(k, n)
        total = 0.0
        for idx in 0:(m - 1)
            total += vals[n - idx]
        end
        (total + kappa * baseline) / (m + kappa)
    end

    i = 1; n_rows = length(schedule)
    while i <= n_rows
        j = i
        while j <= n_rows && schedule[j].kickoff == schedule[i].kickoff
            j += 1
        end
        baseline = base_n == 0 ? 0.0 : base_sum / base_n

        for t in i:(j - 1)
            r = schedule[t]
            hf = get(scored, r.home, Float64[]); ha = get(conceded, r.home, Float64[])
            af = get(scored, r.away, Float64[]); aa = get(conceded, r.away, Float64[])
            if length(hf) < min_matches || length(af) < min_matches
                out[r.id] = (att_h = 0.0, def_h = 0.0, att_a = 0.0, def_a = 0.0,
                             sup = 0.0, level = 0.0, available = 0.0)
                continue
            end
            att_h = shrunk(hf, baseline) - baseline
            def_h = shrunk(ha, baseline) - baseline
            att_a = shrunk(af, baseline) - baseline
            def_a = shrunk(aa, baseline) - baseline
            home_side = att_h + def_a
            away_side = att_a + def_h
            out[r.id] = (att_h = att_h, def_h = def_h, att_a = att_a, def_a = def_a,
                         sup = home_side - away_side, level = home_side + away_side,
                         available = 1.0)
        end

        for t in i:(j - 1)
            r = schedule[t]
            rec = get(records, r.id, nothing)
            rec === nothing && continue
            (require_covered && !rec.covered) && continue
            vals = extract(rec)
            vals === nothing && continue
            fh, ah, fa, aa_ = vals
            all(isfinite, (fh, ah, fa, aa_)) || continue
            push!(get!(scored, r.home, Float64[]), fh)
            push!(get!(conceded, r.home, Float64[]), ah)
            push!(get!(scored, r.away, Float64[]), fa)
            push!(get!(conceded, r.away, Float64[]), aa_)
            base_sum += fh + fa; base_n += 2
        end
        i = j
    end
    return out
end

# --- the extractors that define each candidate quantity -----------------------------
l60_pxg(rec)    = (rec.pxg_h, rec.pxg_a, rec.pxg_a, rec.pxg_h)
l60_volume(rec) = (rec.shots_h, rec.shots_a, rec.shots_a, rec.shots_h)
l60_quality(rec) = begin
    (rec.shots_h > 0 && rec.shots_a > 0) || return nothing
    (rec.pxg_h / rec.shots_h, rec.pxg_a / rec.shots_a,
     rec.pxg_a / rec.shots_a, rec.pxg_h / rec.shots_h)
end
"Goals minus pxG — positive means the side out-scored its chances and should regress."
l60_overperf(rec) = (rec.goals_h - rec.pxg_h, rec.goals_a - rec.pxg_a,
                     rec.goals_a - rec.pxg_a, rec.goals_h - rec.pxg_h)

# ==============================================================================
# 4. FAMILIES D, E — SQUAD CONTINUITY, LOAD AND REST
# ==============================================================================

"""
    l60_squad(ds) -> Dict{Int,NamedTuple}

`(continuity, rest, load, available)` per match, home minus away.

  * `continuity` — share of today's starting XI that also started the side's previous
    match. A settled team against a rotated one.
  * `rest` — days since the side's previous fixture, capped at 21 to keep an
    international break from dominating.
  * `load` — starting-XI minutes accumulated in the previous 14 days, per player.

All three read only fixtures strictly earlier than the current kickoff, and the previous
match is resolved from the ordered schedule rather than by date arithmetic, so a
same-kickoff pair cannot inform each other.
"""
function l60_squad(ds::L60_DATA.DataStore)
    schedule = l60_schedule(ds)
    starters = Dict{Int, Dict{Bool, Vector{Int}}}()
    minutes = Dict{Int, Dict{Int, Float64}}()
    columns = propertynames(ds.lineups)
    has_minutes = :minutes_played in columns

    for row in eachrow(ds.lineups)
        (ismissing(row.player_id) || ismissing(row.team_side)) && continue
        coalesce(row.is_substitute, false) && continue
        side = lowercase(String(row.team_side))
        side in ("home", "away") || continue
        mid = Int(row.match_id)
        push!(get!(get!(starters, mid, Dict{Bool, Vector{Int}}()), side == "home", Int[]),
              Int(row.player_id))
        if has_minutes && !ismissing(row.minutes_played)
            m = try Float64(row.minutes_played) catch; 0.0 end
            m > 0 && (get!(minutes, mid, Dict{Int, Float64}())[Int(row.player_id)] = m)
        end
    end

    last_xi = Dict{String, Vector{Int}}()
    last_day = Dict{String, Float64}()
    load_log = Dict{String, Vector{Tuple{Float64, Float64}}}()   # (day, minutes per starter)
    out = Dict{Int, NamedTuple{(:continuity, :rest, :load, :available), NTuple{4, Float64}}}()

    i = 1; n_rows = length(schedule)
    while i <= n_rows
        j = i
        while j <= n_rows && schedule[j].kickoff == schedule[i].kickoff
            j += 1
        end
        today = Float64(Dates.value(Date(schedule[i].kickoff)))

        for t in i:(j - 1)
            r = schedule[t]
            xi = get(starters, r.id, Dict{Bool, Vector{Int}}())
            home_xi = get(xi, true, Int[]); away_xi = get(xi, false, Int[])
            if length(home_xi) < 7 || length(away_xi) < 7 ||
               !haskey(last_xi, r.home) || !haskey(last_xi, r.away)
                out[r.id] = (continuity = 0.0, rest = 0.0, load = 0.0, available = 0.0)
                continue
            end
            share(now, before) = isempty(now) ? 0.0 :
                                 length(intersect(Set(now), Set(before))) / length(now)
            rest_of(team) = min(today - get(last_day, team, today), 21.0)
            load_of(team, xi_now) = begin
                entries = get(load_log, team, Tuple{Float64, Float64}[])
                recent = [m for (d, m) in entries if today - d <= 14.0]
                isempty(recent) ? 0.0 : sum(recent)
            end
            out[r.id] = (
                continuity = share(home_xi, last_xi[r.home]) - share(away_xi, last_xi[r.away]),
                rest = rest_of(r.home) - rest_of(r.away),
                load = (load_of(r.home, home_xi) - load_of(r.away, away_xi)) / 90.0,
                available = 1.0,
            )
        end

        for t in i:(j - 1)
            r = schedule[t]
            xi = get(starters, r.id, Dict{Bool, Vector{Int}}())
            for (team, is_home) in ((r.home, true), (r.away, false))
                side_xi = get(xi, is_home, Int[])
                isempty(side_xi) && continue
                last_xi[team] = side_xi
                last_day[team] = today
                per = get(minutes, r.id, Dict{Int, Float64}())
                total = isempty(per) ? 90.0 * length(side_xi) :
                        sum(get(per, p, 0.0) for p in side_xi)
                push!(get!(load_log, team, Tuple{Float64, Float64}[]),
                      (today, total / max(length(side_xi), 1)))
            end
        end
        i = j
    end
    return out
end

# ==============================================================================
# 5. EVALUATION HARNESS
# ==============================================================================

"""
    l60_evaluate(name, column, frame, train_mask, test_mask, baseline_columns) -> NamedTuple

Standalone association plus OUT-OF-SAMPLE incremental R-squared.

⚠ WHY THE COEFFICIENTS ARE FROZEN. In-sample, adding any column to an OLS fit raises
R-squared — mechanically, without exception, even for pure noise. A bootstrap that refits
on each resample therefore reports a strictly positive interval for every candidate ever
tested, which is exactly the false-positive pattern this harness produced on its first
run. Both models are fitted on the HISTORY block and scored with frozen coefficients on
the held-out block, so a useless column can and does score NEGATIVE.
"""
function l60_evaluate(name::AbstractString, column::Vector{Float64}, frame::DataFrame,
                      train_mask::AbstractVector{Bool}, test_mask::AbstractVector{Bool},
                      baseline_columns::Vector{Vector{Float64}};
                      response::Symbol = :supremacy)
    y_tr = Float64.(frame[train_mask, response]); y_te = Float64.(frame[test_mask, response])
    base_tr = [c[train_mask] for c in baseline_columns]
    base_te = [c[test_mask] for c in baseline_columns]
    x_tr = column[train_mask]; x_te = column[test_mask]

    oos_r2(Xtr, ytr, Xte, yte) = begin
        fit = eda_ols(Xtr, ytr)
        pred = hcat(ones(size(Xte, 1)), Xte) * fit.beta
        ss_res = sum(abs2, yte .- pred)
        ss_tot = sum(abs2, yte .- mean(yte))
        ss_tot < 1e-12 ? NaN : 1 - ss_res / ss_tot
    end
    base_r2 = isempty(base_tr) ? 0.0 :
              oos_r2(reduce(hcat, base_tr), y_tr, reduce(hcat, base_te), y_te)
    joint_r2 = oos_r2(reduce(hcat, vcat(base_tr, [x_tr])), y_tr,
                      reduce(hcat, vcat(base_te, [x_te])), y_te)
    solo_r2 = oos_r2(reshape(x_tr, length(x_tr), 1), y_tr, reshape(x_te, length(x_te), 1), y_te)
    joint_in = eda_ols(reduce(hcat, vcat(base_tr, [x_tr])), y_tr)

    return (
        name = String(name), n = length(x_te), n_live = count(!iszero, x_te),
        r = eda_pearson(x_te, y_te), rho = eda_spearman(x_te, y_te),
        auc = eda_auc(x_te, frame[test_mask, :home_win]),
        solo_r2 = solo_r2, joint_r2 = joint_r2, delta_r2 = joint_r2 - base_r2,
        t = joint_in.t[end],
    )
end

"""
    l60_bootstrap_delta_r2(column, frame, train_mask, test_mask, baseline_columns; draws)

Bootstrap of the OUT-OF-SAMPLE incremental R-squared. Coefficients are fitted once on the
history block and held fixed; only the held-out block is resampled. An interval spanning
zero means the candidate is indistinguishable from adding nothing.
"""
function l60_bootstrap_delta_r2(column::Vector{Float64}, frame::DataFrame,
                                train_mask::AbstractVector{Bool},
                                test_mask::AbstractVector{Bool},
                                baseline_columns::Vector{Vector{Float64}};
                                draws::Int = 3000, response::Symbol = :supremacy)
    y_tr = Float64.(frame[train_mask, response]); y_te = Float64.(frame[test_mask, response])
    base_tr = [c[train_mask] for c in baseline_columns]
    base_te = [c[test_mask] for c in baseline_columns]
    x_tr = column[train_mask]; x_te = column[test_mask]
    n = length(y_te)
    n < 40 && return (NaN, NaN, NaN)

    base_fit = isempty(base_tr) ? nothing : eda_ols(reduce(hcat, base_tr), y_tr)
    joint_fit = eda_ols(reduce(hcat, vcat(base_tr, [x_tr])), y_tr)
    base_pred = base_fit === nothing ? fill(mean(y_tr), n) :
                hcat(ones(n), reduce(hcat, base_te)) * base_fit.beta
    joint_pred = hcat(ones(n), reduce(hcat, vcat(base_te, [x_te]))) * joint_fit.beta

    deltas = Vector{Float64}(undef, draws)
    idx = Vector{Int}(undef, n)
    for b in 1:draws
        for i in 1:n
            idx[i] = rand(1:n)
        end
        yb = @view y_te[idx]
        ss_tot = sum(abs2, yb .- mean(yb))
        if ss_tot < 1e-12
            deltas[b] = 0.0; continue
        end
        rb = 1 - sum(abs2, yb .- @view(base_pred[idx])) / ss_tot
        rj = 1 - sum(abs2, yb .- @view(joint_pred[idx])) / ss_tot
        deltas[b] = rj - rb
    end
    keep = filter(isfinite, deltas)
    isempty(keep) && return (NaN, NaN, NaN)
    return (mean(keep), quantile(keep, 0.05), quantile(keep, 0.95))
end

"Standardise a column on its live entries so candidates on different scales compare."
function l60_standardise(column::Vector{Float64})
    live = filter(!iszero, column)
    length(live) < 10 && return column
    s = std(live)
    (isfinite(s) && s > 1e-12) ? column ./ s : column
end

# ==============================================================================
# 6. FAMILY F — DUAL-HORIZON DECAY KERNELS
# ==============================================================================
#
# HYPOTHESIS. A single decay half-life has to serve two different jobs at once: track
# short-run FORM (injuries, a new manager, a hot streak) and estimate long-run BASELINE
# quality. r95's sweep found a single optimum near a half-life of 8 matches, which is a
# compromise between the two and optimal for neither. If they are genuinely separate
# signals, a fast kernel and a slow kernel entered as TWO covariates should beat any one
# kernel — and the fast one should be the weaker of the two on its own.
#
# The natural test is not "is the pair better" but "does the fast kernel add anything
# once the slow one is in", which is exactly an incremental-R-squared question.

"""
    l60_dual_horizon(ds, records, extract; fast, slow, kappa) -> (fast_lookup, slow_lookup)

The same rolling walk run at two exponential half-lives. `fast` and `slow` are in
matches.
"""
function l60_dual_horizon(ds::L60_DATA.DataStore, records::Dict{Int, <:NamedTuple}, extract;
                          fast::Float64 = 2.0, slow::Float64 = 20.0, kappa::Float64 = 3.0,
                          require_covered::Bool = true, min_matches::Int = 3)
    return (l60_exponential(ds, records, extract; half_life = fast, kappa = kappa,
                            require_covered = require_covered, min_matches = min_matches),
            l60_exponential(ds, records, extract; half_life = slow, kappa = kappa,
                            require_covered = require_covered, min_matches = min_matches))
end

"Rolling walk with exponential match weights `2^(-j/h)`, j = 0 the most recent."
function l60_exponential(ds::L60_DATA.DataStore, records::Dict{Int, <:NamedTuple}, extract;
                         half_life::Float64 = 8.0, kappa::Float64 = 3.0,
                         require_covered::Bool = true, min_matches::Int = 3)
    schedule = l60_schedule(ds)
    scored = Dict{String, Vector{Float64}}()
    conceded = Dict{String, Vector{Float64}}()
    base_sum = 0.0; base_n = 0
    out = Dict{Int, NamedTuple{(:att_h, :def_h, :att_a, :def_a, :sup, :level, :available),
                               NTuple{7, Float64}}}()

    shrunk(vals, baseline) = begin
        n = length(vals)
        num = 0.0; den = 0.0
        for idx in 0:(n - 1)
            w = 0.5 ^ (idx / half_life)
            num += w * vals[n - idx]; den += w
        end
        (num + kappa * baseline) / (den + kappa)
    end

    i = 1; n_rows = length(schedule)
    while i <= n_rows
        j = i
        while j <= n_rows && schedule[j].kickoff == schedule[i].kickoff
            j += 1
        end
        baseline = base_n == 0 ? 0.0 : base_sum / base_n
        for t in i:(j - 1)
            r = schedule[t]
            hf = get(scored, r.home, Float64[]); ha = get(conceded, r.home, Float64[])
            af = get(scored, r.away, Float64[]); aa = get(conceded, r.away, Float64[])
            if length(hf) < min_matches || length(af) < min_matches
                out[r.id] = (att_h = 0.0, def_h = 0.0, att_a = 0.0, def_a = 0.0,
                             sup = 0.0, level = 0.0, available = 0.0); continue
            end
            att_h = shrunk(hf, baseline) - baseline; def_h = shrunk(ha, baseline) - baseline
            att_a = shrunk(af, baseline) - baseline; def_a = shrunk(aa, baseline) - baseline
            hs = att_h + def_a; as_ = att_a + def_h
            out[r.id] = (att_h = att_h, def_h = def_h, att_a = att_a, def_a = def_a,
                         sup = hs - as_, level = hs + as_, available = 1.0)
        end
        for t in i:(j - 1)
            r = schedule[t]
            rec = get(records, r.id, nothing)
            rec === nothing && continue
            (require_covered && !rec.covered) && continue
            vals = extract(rec); vals === nothing && continue
            all(isfinite, vals) || continue
            push!(get!(scored, r.home, Float64[]), vals[1])
            push!(get!(conceded, r.home, Float64[]), vals[2])
            push!(get!(scored, r.away, Float64[]), vals[3])
            push!(get!(conceded, r.away, Float64[]), vals[4])
            base_sum += vals[1] + vals[3]; base_n += 2
        end
        i = j
    end
    return out
end

# ==============================================================================
# 7. FAMILY G — BENCH DEPTH AND LATE-GAME DROP-OFF
# ==============================================================================
#
# HYPOTHESIS. Two sides with identical starting elevens are not equally strong if one
# can bring on a better bench. Squad wealth as currently built reads the STARTING XI
# only, so bench quality is invisible to every covariate in the builder. If depth
# matters it should show up twice: as a main effect, and concentrated in LATE goals.

"""
    l60_bench(ds) -> Dict{Int,NamedTuple}

`(depth_value, depth_count, available)` per match, home minus away.

`depth_value` is the log ratio of summed bench market value; `depth_count` is simply the
number of named substitutes, which is a proxy for squad size where valuations are thin.
"""
function l60_bench(ds::L60_DATA.DataStore)
    columns = propertynames(ds.lineups)
    value_col = :proposed_market_value in columns ? :proposed_market_value :
                (:market_value in columns ? :market_value : nothing)
    totals = Dict{Tuple{Int, Bool}, Float64}()
    counts = Dict{Tuple{Int, Bool}, Int}()
    for row in eachrow(ds.lineups)
        ismissing(row.team_side) && continue
        coalesce(row.is_substitute, false) || continue        # BENCH only
        side = lowercase(String(row.team_side))
        side in ("home", "away") || continue
        key = (Int(row.match_id), side == "home")
        counts[key] = get(counts, key, 0) + 1
        if value_col !== nothing && !ismissing(row[value_col])
            v = try Float64(row[value_col]) catch; 0.0 end
            isfinite(v) && v > 0 && (totals[key] = get(totals, key, 0.0) + v)
        end
    end
    out = Dict{Int, NamedTuple{(:depth_value, :depth_count, :available), NTuple{3, Float64}}}()
    for row in eachrow(ds.matches)
        mid = Int(row.match_id)
        hv = get(totals, (mid, true), 0.0); av = get(totals, (mid, false), 0.0)
        hc = get(counts, (mid, true), 0);   ac = get(counts, (mid, false), 0)
        (hc == 0 || ac == 0) && (out[mid] = (depth_value = 0.0, depth_count = 0.0,
                                             available = 0.0); continue)
        value_delta = (hv > 0 && av > 0) ? log(hv) - log(av) : 0.0
        out[mid] = (depth_value = value_delta, depth_count = Float64(hc - ac), available = 1.0)
    end
    return out
end

"""
    l60_late_share(ds) -> Dict{Int,NamedTuple}

Historical late-game strength: the rolling share of a side's pxG created after the
70th minute, home minus away. A side that fades late should carry a negative value.

Point-in-time by the same tie-group rule as every other walk here.
"""
function l60_late_share(ds::L60_DATA.DataStore; k::Int = 12, kappa::Float64 = 4.0,
                        cutoff::Float64 = 70.0)
    shots = L60_FEATURES.build_shots(ds)
    late = Dict{Int, NTuple{4, Float64}}()      # late_h, all_h, late_a, all_a
    if nrow(shots) > 0
        model = L60_FEATURES.fit_shot_xg(shots)
        predicted = L60_FEATURES.predict_xg(model, shots)
        for (i, r) in enumerate(eachrow(shots))
            (ismissing(r.is_home) || ismissing(r.time)) && continue
            xg = predicted[i]; isfinite(xg) || continue
            mid = Int(r.match_id)
            lh, ah, la, aa = get(late, mid, (0.0, 0.0, 0.0, 0.0))
            is_late = Float64(r.time) >= cutoff
            if r.is_home === true
                late[mid] = (lh + (is_late ? xg : 0.0), ah + xg, la, aa)
            else
                late[mid] = (lh, ah, la + (is_late ? xg : 0.0), aa + xg)
            end
        end
    end
    records = Dict{Int, NamedTuple}()
    for (mid, (lh, ah, la, aa)) in late
        (ah > 0 && aa > 0) || continue
        records[mid] = (pxg_h = lh / ah, pxg_a = la / aa, shots_h = 1.0, shots_a = 1.0,
                        goals_h = 0.0, goals_a = 0.0, covered = true)
    end
    return l60_rolling(ds, records, l60_pxg; k = k, kappa = kappa, min_matches = 3)
end

# ==============================================================================
# 8. FAMILY H — DUAL-TARGET RAPM AND WEALTH SYNERGY
# ==============================================================================
#
# HYPOTHESIS 1. `:y_shots` and `:y_xg` measure different player skills — creating
# volume and creating quality. r94 measured them correlated at 0.792 per stint but with
# very different reliabilities (0.555 vs 0.395). Entering both may beat either.
#
# HYPOTHESIS 2. Wealth and RAPM are near-orthogonal (r93: r = +0.336, both significant
# beyond the other). If their effects are not merely additive — if an expensive squad
# converts its money into results only when the players are also individually good —
# the product term carries signal that neither main effect does.

"""
    l60_rapm_column(ds, history; target, lambda, kappa, min_rated) -> (column, available)

A starting-XI RAPM differential fitted on `history` only, for an arbitrary target.
Mirrors the production feature; separated here so several targets can be built cheaply
from one store without going through the feature pipeline.
"""
function l60_rapm_column(ds::L60_DATA.DataStore, history::Set{Int}, ordered_ids;
                         target::Symbol = :y_xg, lambda::Float64 = 5000.0,
                         kappa::Float64 = 20.0, min_rated::Int = 3,
                         half_life_days::Float64 = 730.0)
    prep = L60_FEATURES.pm_prepared(ds)
    segments = prep.segments[in.(Int.(prep.segments.match_id), Ref(history)), :]
    nrow(segments) == 0 && return (zeros(length(ordered_ids)), zeros(length(ordered_ids)))
    fit = L60_FEATURES.fit_ratings(segments; target = target, λ = lambda, w_sim = 0.0,
                                   half_life = half_life_days,
                                   T_rating = maximum(segments.match_date),
                                   comp_sets = L60_FEATURES.competition_sets(ds; match_ids = history))
    fit === nothing && return (zeros(length(ordered_ids)), zeros(length(ordered_ids)))
    exposure = L60_FEATURES.player_exposure(segments)
    n_of = Dict(Int(r.player_id) => Float64(r.n_segments) for r in eachrow(exposure))
    adjusted = Dict{Int, Float64}(
        Int(r.player_id) => Float64(r.rapm) * (get(n_of, Int(r.player_id), 0.0) /
                                               (get(n_of, Int(r.player_id), 0.0) + kappa))
        for r in eachrow(fit))
    return l50_xi_column(ds, adjusted, ordered_ids; min_rated = min_rated)
end
