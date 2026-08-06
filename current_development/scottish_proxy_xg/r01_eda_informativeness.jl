#=
r01 — WP1 EDA / GO-NO-GO GATE. No MCMC; ~1 minute.

THE QUESTION: is commentary proxy xG worth putting in a model at all on 56/57, given the funnel
already has shot counts? Five experiments, run before spending ~25h of MCMC.

  E2  INFORMATIVENESS LADDER  (the gate).  A={goals} B={+shots} C={+pxG, NO shots} D={+both}.
      C vs B answers ARM A ("does xG BEAT shots?"); D vs B answers ARM B ("does xG ADD to shots?").
  E3  SPLIT-HALF RELIABILITY. Does pxG separate teams better per match than shots do?
  E4  VARIANCE LAW. Var ∝ mu (compound Poisson) or Var ∝ mu^2 (Gamma with constant nu)? Sets the
      nu prior and decides whether the linear-variance cell is needed.
  E5  EXTERNAL VALIDITY on 54/55, where SofaScore xG exists as a yardstick. Reports the SLOPE, not
      just the correlation — the Gamma pillar's MEAN is what matters, not its ranking.

Design follows current_development/bbc_xg_proxy/l02_informativeness.jl: walk-forward decayed form
over strictly prior matches, expanding-window by season, IDENTICAL eval sample across all sets.

    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_proxy_xg/r01_eda_informativeness.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using Dates
using GLM
using StatsModels
using SpecialFunctions: loggamma

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l01_proxy_xg_feature.jl"))

_r(x, d = 5) = isnan(x) ? NaN : round(x, digits = d)

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
team_df = proxy_team_rows(ds)
println("[INFO] $(nrow(team_df)) team-match rows over $(length(unique(team_df.match_id))) matches, " *
        "seasons $(join(sort(unique(team_df.season)), ", "))")

# ==========================================
# WALK-FORWARD DECAYED FORM
# ==========================================
const FORM_STREAMS = (
    form_gf = :goals,       form_ga = :goals_against,
    form_sf = :shots_bbc,   form_sa = :shots_bbc_against,
    form_xf = :pxg,         form_xa = :pxg_against,
)

"""
Per-team, per-stream decayed mean over STRICTLY PRIOR matches, w = 0.5^(dt / half_life).
Each stream is filtered independently so a match missing shots does not poison the goals form.
"""
function add_form!(df::DataFrame; half_life_days::Real = 365.0, min_prior::Int = 5)
    sort!(df, [:kickoff, :match_id])
    n = nrow(df)
    for col in keys(FORM_STREAMS)
        df[!, col] = Vector{Union{Missing, Float64}}(missing, n)
    end
    for g in groupby(df, :team)
        ts = g.kickoff
        for (form_col, src_col) in pairs(FORM_STREAMS)
            vals = g[!, src_col]
            for i in 1:nrow(g)
                num = 0.0; den = 0.0; nv = 0
                for j in 1:(i - 1)
                    v = vals[j]
                    (v === missing || (v isa AbstractFloat && isnan(v))) && continue
                    w = 0.5^(Float64(Dates.value(ts[i] - ts[j])) / half_life_days)
                    num += w * Float64(v); den += w; nv += 1
                end
                nv >= min_prior && (g[i, form_col] = num / den)
            end
        end
    end
    return df
end

"""Attach the opponent's DEFENSIVE forms onto each team row."""
function add_opp_form!(df::DataFrame)
    key = Dict((Int(r.match_id), String(r.team)) =>
               (ga = r.form_ga, sa = r.form_sa, xa = r.form_xa) for r in eachrow(df))
    df.opp_form_ga = [key[(Int(r.match_id), String(r.opponent))].ga for r in eachrow(df)]
    df.opp_form_sa = [key[(Int(r.match_id), String(r.opponent))].sa for r in eachrow(df)]
    df.opp_form_xa = [key[(Int(r.match_id), String(r.opponent))].xa for r in eachrow(df)]
    return df
end

add_form!(team_df; half_life_days = 365.0)
add_opp_form!(team_df)

const FORM_COLS = [:form_gf, :opp_form_ga, :form_sf, :opp_form_sa, :form_xf, :opp_form_xa]
usable = dropmissing(team_df, vcat([:goals, :is_home], FORM_COLS))
println("[INFO] usable team rows (all streams present): $(nrow(usable))")

# ==========================================
# E2 — THE INFORMATIVENESS LADDER
# ==========================================
println("\n", "="^78,
        "\nE2 — INFORMATIVENESS LADDER (the GATE)\n", "="^78)
println("""
    A = goals form only          B = A + shots           C = A + pxG (NO shots)      D = A + both
    C vs B  ->  ARM A: does xG REPLACE shots?
    D vs B  ->  ARM B: does xG ADD to shots?
""")

const GOALS_SETS = [
    :A => (term(:goals) ~ term(1) + term(:is_home) + term(:form_gf) + term(:opp_form_ga)),
    :B => (term(:goals) ~ term(1) + term(:is_home) + term(:form_gf) + term(:opp_form_ga) +
                          term(:form_sf) + term(:opp_form_sa)),
    :C => (term(:goals) ~ term(1) + term(:is_home) + term(:form_gf) + term(:opp_form_ga) +
                          term(:form_xf) + term(:opp_form_xa)),
    :D => (term(:goals) ~ term(1) + term(:is_home) + term(:form_gf) + term(:opp_form_ga) +
                          term(:form_sf) + term(:opp_form_sa) +
                          term(:form_xf) + term(:opp_form_xa)),
]

_pois_ll(y, λ) = y .* log.(λ) .- λ .- loggamma.(y .+ 1)

"""Expanding-window by season start. Returns (per-season summary, per-observation scores)."""
function eval_nested(df, sets, test_seasons; family, link, scorer, ycol)
    rows = NamedTuple[]
    per_obs = Dict{Symbol, Vector{Float64}}(name => Float64[] for (name, _) in sets)
    for s in test_seasons
        any(df.season .== s) || continue
        s0    = minimum(df.kickoff[df.season .== s])
        train = df[df.kickoff .< s0, :]
        test  = df[df.season .== s, :]
        (nrow(train) < 150 || nrow(test) == 0) && continue
        for (name, f) in sets
            m = glm(f, train, family, link)
            p = predict(m, test)
            sc = scorer(Float64.(test[!, ycol]), p)
            append!(per_obs[name], sc)
            push!(rows, (season = s, set = name, n = nrow(test), score = _r(mean(sc))))
        end
    end
    return DataFrame(rows), per_obs
end

"""Paired t on the per-observation score difference (x minus y)."""
function paired_t(x::Vector{Float64}, y::Vector{Float64})
    (length(x) != length(y) || isempty(x)) && return (Δ = NaN, t = NaN, n = 0)
    d = x .- y
    sd = std(d)
    return (Δ = mean(d), t = sd == 0 ? NaN : mean(d) / (sd / sqrt(length(d))), n = length(d))
end

TEST_SEASONS = filter(s -> s in ("24/25", "25/26"), sort(unique(usable.season)))
println("[INFO] test seasons: $(join(TEST_SEASONS, ", "))")

g_sum, g_obs = eval_nested(usable, GOALS_SETS, TEST_SEASONS;
                           family = Poisson(), link = LogLink(),
                           scorer = (y, p) -> _pois_ll(y, clamp.(p, 1e-6, 20.0)), ycol = :goals)
println("\n--- GOALS head: OOS Poisson log-lik per observation (HIGHER is better) ---")
show(unstack(g_sum, :season, :set, :score); allrows = true, allcols = true); println()

# --- match-level home-win head ---
function build_match_rows(df)
    byid = Dict((Int(r.match_id), String(r.team)) => r for r in eachrow(df))
    rows = NamedTuple[]
    for r in eachrow(df[df.is_home .== 1.0, :])
        haskey(byid, (Int(r.match_id), String(r.opponent))) || continue
        a = byid[(Int(r.match_id), String(r.opponent))]
        push!(rows, (match_id = r.match_id, season = r.season, kickoff = r.kickoff,
                     home_win = Float64(r.goals > r.goals_against),
                     d_gf = r.form_gf - a.form_gf, d_ga = r.form_ga - a.form_ga,
                     d_sf = r.form_sf - a.form_sf, d_sa = r.form_sa - a.form_sa,
                     d_xf = r.form_xf - a.form_xf, d_xa = r.form_xa - a.form_xa))
    end
    DataFrame(rows)
end

const RESULT_SETS = [
    :A => (term(:home_win) ~ term(1) + term(:d_gf) + term(:d_ga)),
    :B => (term(:home_win) ~ term(1) + term(:d_gf) + term(:d_ga) + term(:d_sf) + term(:d_sa)),
    :C => (term(:home_win) ~ term(1) + term(:d_gf) + term(:d_ga) + term(:d_xf) + term(:d_xa)),
    :D => (term(:home_win) ~ term(1) + term(:d_gf) + term(:d_ga) + term(:d_sf) + term(:d_sa) +
                             term(:d_xf) + term(:d_xa)),
]
_ll_obs(y, p) = -(y .* log.(clamp.(p, 1e-9, 1)) .+ (1 .- y) .* log.(clamp.(1 .- p, 1e-9, 1)))

mdf = build_match_rows(usable)
r_sum, r_obs = eval_nested(mdf, RESULT_SETS, TEST_SEASONS;
                           family = Binomial(), link = LogitLink(),
                           scorer = _ll_obs, ycol = :home_win)
println("\n--- HOME-WIN head: OOS log-loss per match (LOWER is better) ---")
show(unstack(r_sum, :season, :set, :score); allrows = true, allcols = true); println()

println("\n--- PAIRED COMPARISONS (pooled over test seasons) ---")
cmp_rows = NamedTuple[]
for (head, obs, sgn) in (("goals loglik (+ = better)", g_obs, 1.0),
                         ("homewin logloss (− = better)", r_obs, -1.0))
    for (a, b) in ((:B, :A), (:C, :A), (:C, :B), (:D, :B), (:D, :C))
        p = paired_t(obs[a], obs[b])
        push!(cmp_rows, (head = head, comparison = "$a − $b", n = p.n,
                         Δ = _r(p.Δ), t = _r(p.t, 2),
                         better = isnan(p.t) ? "—" : (sgn * p.Δ > 0 ? "$a" : "$b")))
    end
end
cmp = DataFrame(cmp_rows)
show(cmp; allrows = true, allcols = true, truncate = 0); println()

# The two comparisons the whole stream turns on.
_t(head_frag, comp) = (rows = cmp[occursin.(head_frag, cmp.head) .& (cmp.comparison .== comp), :t];
                       isempty(rows) ? NaN : Float64(rows[1]))
t_CB_goals = _t("goals",   "C − B");  t_CB_res = _t("homewin", "C − B")
t_DB_goals = _t("goals",   "D − B");  t_DB_res = _t("homewin", "D − B")
# goals head: positive t favours the first set. homewin head is a LOSS, so negative t favours it.
E2_ARM_A = (t_CB_goals > 1.5) || (t_CB_res < -1.5) || (abs(t_CB_goals) < 1.0 && abs(t_CB_res) < 1.0)
E2_ARM_B = (t_DB_goals > 1.5) || (t_DB_res < -1.5)

# ==========================================
# E3 — SPLIT-HALF RELIABILITY
# ==========================================
println("\n", "="^78, "\nE3 — SPLIT-HALF RELIABILITY (per-match signal-to-noise)\n", "="^78)
println("""
    Each team's matches are split odd/even. `self` correlates a team's half-1 mean against its
    half-2 mean (how reliably the metric measures the team). `predict` correlates half-1 mean
    against half-2 GOALS (how much of that reliability is about SCORING). Mirrors how the RAPM
    rating was validated (split-half 0.669 vs SofaScore's 0.660).
""")

function split_half(df, col; min_matches::Int = 30)
    rows = NamedTuple[]
    for g in groupby(sort(df, [:kickoff, :match_id]), :team)
        gg = dropmissing(g, [col, :goals])
        nrow(gg) < min_matches && continue
        odd  = 1:2:nrow(gg); even = 2:2:nrow(gg)
        push!(rows, (team = gg.team[1], n = nrow(gg),
                     h1 = mean(Float64.(gg[odd, col])),  h2 = mean(Float64.(gg[even, col])),
                     g1 = mean(Float64.(gg[odd, :goals])), g2 = mean(Float64.(gg[even, :goals]))))
    end
    t = DataFrame(rows)
    nrow(t) < 5 && return (n_teams = nrow(t), self = NaN, sb = NaN, predict = NaN)
    r = cor(t.h1, t.h2)
    return (n_teams = nrow(t),
            self    = r,
            sb      = 2r / (1 + r),                              # Spearman-Brown
            predict = (cor(t.h1, t.g2) + cor(t.h2, t.g1)) / 2)
end

rel_rows = NamedTuple[]
for (label, col) in (("goals", :goals), ("shots (ds.bbc)", :shots_bbc), ("proxy xG", :pxg))
    s = split_half(team_df, col)
    push!(rel_rows, (metric = label, teams = s.n_teams, self = _r(s.self, 3),
                     spearman_brown = _r(s.sb, 3), predicts_goals = _r(s.predict, 3)))
end
rel = DataFrame(rel_rows)
show(rel; allrows = true, allcols = true, truncate = 0); println()

_rel(m, c) = (rows = rel[rel.metric .== m, c]; isempty(rows) ? NaN : Float64(rows[1]))
E3_PASS = (_rel("proxy xG", :self) > _rel("shots (ds.bbc)", :self)) ||
          (_rel("proxy xG", :predicts_goals) > _rel("shots (ds.bbc)", :predicts_goals))

# ==========================================
# E4 — THE VARIANCE LAW
# ==========================================
println("\n", "="^78, "\nE4 — VARIANCE LAW: does Var(pxG) scale like mu or like mu^2?\n", "="^78)
println("""
    Proxy xG is a COMPOUND POISSON sum (sum over S shots of per-shot quality), whose variance is
    LINEAR in the mean. A Gamma with constant nu implies a QUADRATIC law. Regressing log Var on
    log mean across fitted-mean deciles recovers the exponent:
        slope ~ 1  ->  linear   : the Gamma(nu, mu/nu) form MIS-SPECIFIES the mean-variance link;
                                  run the linear-variance cell (Gamma(mu/theta, theta)).
        slope ~ 2  ->  quadratic: the Ireland form is right as written.
""")

E4_SLOPE = NaN
try
    fit = glm(term(:pxg) ~ term(1) + term(:is_home) + term(:form_xf) + term(:opp_form_xa) +
                           term(:form_sf) + term(:opp_form_sa),
              usable, Gamma(), LogLink())
    u = copy(usable); u.mu_hat = predict(fit, usable)
    qs = quantile(u.mu_hat, range(0, 1; length = 11))
    bins = NamedTuple[]
    for i in 1:10
        lo, hi = qs[i], qs[i + 1]
        sel = i == 10 ? (u.mu_hat .>= lo) : (u.mu_hat .>= lo) .& (u.mu_hat .< hi)
        b = u[sel, :]
        nrow(b) < 20 && continue
        push!(bins, (decile = i, n = nrow(b), mean = _r(mean(b.pxg), 3), var = _r(var(b.pxg), 4)))
    end
    bt = DataFrame(bins)
    show(bt; allrows = true, allcols = true); println()
    lawfit = lm(term(:logv) ~ term(1) + term(:logm),
                DataFrame(logv = log.(bt.var), logm = log.(bt.mean)))
    global E4_SLOPE = coef(lawfit)[2]
    println("    log Var = a + b·log mean   ->   b = $(_r(E4_SLOPE, 3))  " *
            "(1 = linear/compound-Poisson, 2 = quadratic/Gamma)")
    println("    implied constant-nu = mean(mean^2 / var) = " *
            "$(_r(mean(bt.mean .^ 2 ./ bt.var), 2))   <-- CENTRE THE nu PRIOR HERE")
catch e
    @warn "E4 variance-law fit failed" exception = e
end

# ==========================================
# E5 — EXTERNAL VALIDITY vs SofaScore xG (54/55)
# ==========================================
println("\n", "="^78, "\nE5 — EXTERNAL VALIDITY on 54/55 (SofaScore xG as yardstick)\n", "="^78)
println("""
    56/57 has no SofaScore xG at all, so the proxy is validated on the tiers that do. Correlation
    is the published number (0.817); the SLOPE is what this stream needs — the Gamma pillar anchors
    a MEAN, so a proxy that ranks well but sits 20% low would bias every rate.
""")

try
    dsu = Data.load_datastore_cached(Data.ScottishUpper())
    lut_u, _ = proxy_xg_table(dsu)
    st = dsu.statistics
    xg_cols = filter(c -> occursin("expectedgoals", lowercase(String(c))), names(st))
    if isempty(xg_cols) || isempty(lut_u)
        println("    ⚠ SKIPPED — no SofaScore xG columns found in ds.statistics " *
                "(saw: $(join(first(names(st), 12), ", ")))")
    else
        hc = first(filter(c -> occursin("home", lowercase(c)), xg_cols))
        ac = first(filter(c -> occursin("away", lowercase(c)), xg_cols))
        println("    using columns: $hc / $ac")
        pr = Float64[]; re = Float64[]
        for r in eachrow(st)
            mid = Int(r.match_id)
            haskey(lut_u, mid) || continue
            x_h, x_a, n_h, n_a = lut_u[mid]
            (n_h > 0 && n_a > 0) || continue
            for (p, v) in ((x_h, r[hc]), (x_a, r[ac]))
                (ismissing(v) || (v isa AbstractFloat && isnan(v))) && continue
                push!(pr, p); push!(re, Float64(v))
            end
        end
        if length(pr) < 100
            println("    ⚠ SKIPPED — only $(length(pr)) overlapping team-match rows")
        else
            f = lm(term(:real) ~ term(1) + term(:proxy), DataFrame(real = re, proxy = pr))
            println("    n=$(length(pr))  cor=$(_r(cor(pr, re), 3))  " *
                    "mean proxy=$(_r(mean(pr), 3))  mean real=$(_r(mean(re), 3))")
            println("    real = $(_r(coef(f)[1], 3)) + $(_r(coef(f)[2], 3))·proxy   " *
                    "(slope 1.0 and intercept 0.0 = perfectly calibrated level)")
        end
    end
catch e
    @warn "E5 external validity skipped" exception = e
end

# ==========================================
# THE GATE
# ==========================================
println("\n", "="^78, "\nWP1 GATE\n", "="^78)
println("  E2 Arm A (C vs B): t_goals=$(_r(t_CB_goals, 2)) t_homewin=$(_r(t_CB_res, 2))  " *
        "-> $(E2_ARM_A ? "xG at least MATCHES shots" : "xG is WORSE than shots")")
println("  E2 Arm B (D vs B): t_goals=$(_r(t_DB_goals, 2)) t_homewin=$(_r(t_DB_res, 2))  " *
        "-> $(E2_ARM_B ? "xG ADDS to shots" : "xG adds nothing to shots")")
println("  E3 reliability   : $(E3_PASS ? "pxG separates teams better than shots" : "pxG is no better than shots")")
println("  E4 variance slope: $(_r(E4_SLOPE, 3))  -> " *
        (isnan(E4_SLOPE) ? "unknown" :
         E4_SLOPE < 1.5 ? "LINEAR — schedule the linear-variance cell (cell 5)" :
                          "QUADRATIC — the Ireland Gamma form is correct as written"))

GO = E2_ARM_A || E2_ARM_B || E3_PASS
println("\n", GO ?
    ">> GO. Proceed to r02_smoke.jl. Record every number above in RESULTS_scottish_proxy_xg.md\n" *
    ">> BEFORE training, so the MCMC verdict cannot be reinterpreted after the fact." :
    ">> NO-GO. Proxy xG neither beats nor adds to shot counts, and separates teams no better.\n" *
    ">> Write it up as a null in RESULTS_scottish_proxy_xg.md and STOP — do not spend 25h.")
