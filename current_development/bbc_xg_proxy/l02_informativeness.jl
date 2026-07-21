#=
l02 — WP2 loader: forward informativeness of proxy-xG on League 1/2.

Question: do decayed form features built from proxy-xG predict NEXT-match goals /
home-win better than form built from goals and shots alone?

Pipeline:
  - apply_proxy!(team_df, artifact)      : proxy-xG for every team-match row with stats.
  - add_against!(team_df)                : opponent's SoT / proxy-xG as *_against.
  - add_form!(team_df; half_life_days)   : per-team exponentially-decayed
    walk-forward means (strictly prior matches) of goals for/against,
    SoT for/against, proxy-xG for/against. Per-stream validity filtering.
  - add_opp_form!(team_df)               : opponent's defensive forms on each row.
  - build_match_rows(team_df)            : home-minus-away form differences + home_win.
  - eval_goals_nested / eval_result_nested : expanding-window by season,
    nested sets A {goals} ⊂ B {+shots} ⊂ C {+proxy-xG}.
    Metrics: OOS Poisson log-lik (goals), log-loss + AUC (home win).
=#

include(joinpath(@__DIR__, "l01_xg_proxy.jl"))
using Serialization
using SpecialFunctions: loggamma

# ==========================================
# Proxy application
# ==========================================

"""Predict proxy-xG for every team row with usable stats (else missing)."""
function apply_proxy!(team_df::DataFrame, artifact)
    df = copy(team_df)
    df.sblock   = coalesce.(df.sblock, 0.0)
    df.woodwork = coalesce.(df.woodwork, 0.0)
    df.xg_floor = fill(1.0, nrow(df))          # dummy response for predict()
    ok = .!ismissing.(df.sot) .& .!ismissing.(df.soff) .&
         .!ismissing.(df.corners) .& .!ismissing.(df.poss) .& .!ismissing.(df.fouls_opp)
    pred = Vector{Union{Missing,Float64}}(missing, nrow(df))
    pred[findall(ok)] = predict_xg(artifact.model, df[findall(ok), :]; link = artifact.link)
    team_df.proxy_xg = pred
    return team_df
end

"""Attach opponent's sot/proxy from the paired row of the same match."""
function add_against!(team_df::DataFrame)
    key = Dict((Int(r.match_id), String(r.team)) => (sot = r.sot, px = r.proxy_xg)
               for r in eachrow(team_df))
    team_df.sot_against      = [key[(Int(r.match_id), String(r.opponent))].sot for r in eachrow(team_df)]
    team_df.proxy_xg_against = [key[(Int(r.match_id), String(r.opponent))].px  for r in eachrow(team_df)]
    return team_df
end

# ==========================================
# Walk-forward decayed form features
# ==========================================

const FORM_STREAMS = (
    form_gf   = :goals,            form_ga   = :goals_against,
    form_sotf = :sot,              form_sota = :sot_against,
    form_pxf  = :proxy_xg,         form_pxa  = :proxy_xg_against,
)

"""
Per-team, per-stream decayed means over strictly prior matches.
w = 0.5^(Δdays / half_life_days). Each stream needs ≥ min_prior valid prior
values, else missing (streams filtered independently — a match missing SoT
doesn't poison the goals form).
"""
function add_form!(team_df::DataFrame; half_life_days::Real = 365.0, min_prior::Int = 5)
    sort!(team_df, [:kickoff, :match_id])
    n = nrow(team_df)
    for col in keys(FORM_STREAMS)
        team_df[!, col] = Vector{Union{Missing,Float64}}(missing, n)
    end
    team_df[!, :n_prior] = Vector{Union{Missing,Float64}}(missing, n)
    for g in groupby(team_df, :team)
        ts   = g.kickoff
        for (form_col, src_col) in pairs(FORM_STREAMS)
            vals = g[!, src_col]
            for i in 1:nrow(g)
                num = 0.0; den = 0.0; nv = 0
                for j in 1:(i - 1)
                    v = vals[j]
                    (v === missing || (v isa Float64 && isnan(v))) && continue
                    w = 0.5^(Float64(Dates.value(ts[i] - ts[j])) / 86_400_000 / half_life_days)
                    num += w * Float64(v); den += w; nv += 1
                end
                nv >= min_prior && (g[i, form_col] = num / den)
                form_col === :form_gf && (g.n_prior[i] = Float64(i - 1))
            end
        end
    end
    return team_df
end

"""Attach opponent's defensive/against forms to each team row."""
function add_opp_form!(team_df::DataFrame)
    key = Dict((Int(r.match_id), String(r.team)) => (ga = r.form_ga, sota = r.form_sota, pxa = r.form_pxa)
               for r in eachrow(team_df))
    team_df.opp_form_ga   = [key[(Int(r.match_id), String(r.opponent))].ga   for r in eachrow(team_df)]
    team_df.opp_form_sota = [key[(Int(r.match_id), String(r.opponent))].sota for r in eachrow(team_df)]
    team_df.opp_form_pxa  = [key[(Int(r.match_id), String(r.opponent))].pxa  for r in eachrow(team_df)]
    return team_df
end

# ==========================================
# Nested evaluation — goals (team-level Poisson)
# ==========================================

const GOALS_SETS = [
    :A => (term(:goals) ~ term(1) + term(:is_home) + term(:form_gf) + term(:opp_form_ga)),
    :B => (term(:goals) ~ term(1) + term(:is_home) + term(:form_gf) + term(:opp_form_ga) +
                          term(:form_sotf) + term(:opp_form_sota)),
    :C => (term(:goals) ~ term(1) + term(:is_home) + term(:form_gf) + term(:opp_form_ga) +
                          term(:form_sotf) + term(:opp_form_sota) +
                          term(:form_pxf) + term(:opp_form_pxa)),
]

"""Rows usable by ALL nested sets — identical eval sample across A/B/C."""
usable_team_rows(df) = dropmissing(df, [:goals, :is_home, :form_gf, :opp_form_ga,
                                        :form_sotf, :opp_form_sota, :form_pxf, :opp_form_pxa])

_poisson_loglik(y, λ) = mean(y .* log.(λ) .- λ .- loggamma.(y .+ 1))

"""Expanding-window by season-start date: train strictly before season S, test = S."""
function eval_goals_nested(df::DataFrame; test_seasons::Vector{String})
    out = NamedTuple[]
    for s in test_seasons
        any(df.season .== s) || continue
        s_start = minimum(df.kickoff[df.season .== s])
        train = df[df.kickoff .< s_start, :]
        test  = df[df.season .== s, :]
        (nrow(train) < 200 || nrow(test) == 0) && continue
        for (name, f) in GOALS_SETS
            m = glm(f, train, Poisson(), LogLink())
            λ = clamp.(predict(m, test), 1e-6, 20.0)
            push!(out, (season = s, set = name, n = nrow(test),
                        loglik = round(_poisson_loglik(Float64.(test.goals), λ), digits=5)))
        end
    end
    DataFrame(out)
end

# ==========================================
# Nested evaluation — home win (match-level logit on form diffs)
# ==========================================

const RESULT_SETS = [
    :A => (term(:home_win) ~ term(1) + term(:d_gf) + term(:d_ga)),
    :B => (term(:home_win) ~ term(1) + term(:d_gf) + term(:d_ga) + term(:d_sotf) + term(:d_sota)),
    :C => (term(:home_win) ~ term(1) + term(:d_gf) + term(:d_ga) + term(:d_sotf) + term(:d_sota) +
                             term(:d_pxf) + term(:d_pxa)),
]

"""One row per match: home-minus-away form differences + home_win indicator."""
function build_match_rows(team_df::DataFrame)
    byid = Dict((Int(r.match_id), String(r.team)) => r for r in eachrow(team_df))
    rows = NamedTuple[]
    for r in eachrow(team_df[team_df.is_home .== 1.0, :])
        a = byid[(Int(r.match_id), String(r.opponent))]
        push!(rows, (
            match_id = r.match_id, season = r.season, kickoff = r.kickoff,
            home_win = Float64(r.goals > r.goals_against),
            d_gf   = passmissing(-)(r.form_gf,   a.form_gf),
            d_ga   = passmissing(-)(r.form_ga,   a.form_ga),
            d_sotf = passmissing(-)(r.form_sotf, a.form_sotf),
            d_sota = passmissing(-)(r.form_sota, a.form_sota),
            d_pxf  = passmissing(-)(r.form_pxf,  a.form_pxf),
            d_pxa  = passmissing(-)(r.form_pxa,  a.form_pxa),
        ))
    end
    DataFrame(rows)
end

_logloss(y, p) = -mean(y .* log.(clamp.(p, 1e-9, 1)) .+ (1 .- y) .* log.(clamp.(1 .- p, 1e-9, 1)))

function _auc(y::AbstractVector, p::AbstractVector)
    pos = p[y .== 1.0]; neg = p[y .== 0.0]
    (isempty(pos) || isempty(neg)) && return NaN
    gt = sum(pi > ni for pi in pos, ni in neg)
    eq = sum(pi == ni for pi in pos, ni in neg)
    (gt + 0.5eq) / (length(pos) * length(neg))
end

function eval_result_nested(mdf::DataFrame; test_seasons::Vector{String})
    mdf = dropmissing(mdf, [:d_gf, :d_ga, :d_sotf, :d_sota, :d_pxf, :d_pxa])
    out = NamedTuple[]
    for s in test_seasons
        any(mdf.season .== s) || continue
        s_start = minimum(mdf.kickoff[mdf.season .== s])
        train = mdf[mdf.kickoff .< s_start, :]
        test  = mdf[mdf.season .== s, :]
        (nrow(train) < 100 || nrow(test) == 0) && continue
        for (name, f) in RESULT_SETS
            m = glm(f, train, Binomial(), LogitLink())
            p = predict(m, test)
            push!(out, (season = s, set = name, n = nrow(test),
                        logloss = round(_logloss(test.home_win, p), digits=5),
                        auc = round(_auc(test.home_win, p), digits=4)))
        end
    end
    DataFrame(out)
end
