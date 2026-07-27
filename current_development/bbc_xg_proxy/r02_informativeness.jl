#=
r02 — WP2 runner: forward informativeness of proxy-xG on League One/Two (56/57).

Go/no-go gate for the Bayesian integration (WP3+): decayed proxy-xG form must
improve strictly-OOS prediction of next-match goals (Poisson log-lik) and
home-win (log-loss/AUC) over nested baselines {goals} ⊂ {goals+shots}.

Expanding-window by season pair (both divisions pooled): test 22/23…25/26.
Sensitivity: half_life_days ∈ {365 (Stage-A winner), 180}.

Run:  include(".../current_development/bbc_xg_proxy/r02_informativeness.jl")
Results in global `R2::Dict`.
=#

include(joinpath(@__DIR__, "l02_informativeness.jl"))
using Dates

R2 = Dict{Symbol,Any}()

artifact = deserialize(joinpath(@__DIR__, "proxy_model_v1.jls"))
println("[INFO] proxy artifact: ", artifact.formula, " / ", artifact.link,
        " (trained on ", artifact.n_rows, " rows, tiers ", artifact.train_tiers, ")")

conn = LibPQ.Connection(ENV["BF_DB_URL"])
wide = fetch_matches_wide(conn; tournaments = [56, 57])
team = to_team_rows(wide)
println("[INFO] League 1/2 matches=", nrow(wide), "  team rows=", nrow(team))

apply_proxy!(team, artifact)
add_against!(team)
println("[INFO] proxy coverage: ", count(!ismissing, team.proxy_xg), "/", nrow(team),
        "  mean=", round(mean(skipmissing(team.proxy_xg)), digits=3),
        "  (actual goals mean=", round(mean(skipmissing(team.goals)), digits=3), ")")

# season pair label "23/24" — unifies "League 1 23/24" / "League Two 23/24" etc.
season_label(s) = String(last(split(s)))
team.season = season_label.(team.season)

const TEST_SEASONS = ["22/23", "23/24", "24/25", "25/26"]

function run_block(team_df; half_life_days)
    t = copy(team_df)
    add_form!(t; half_life_days)
    add_opp_form!(t)
    tu = usable_team_rows(t)
    goals = eval_goals_nested(tu; test_seasons = TEST_SEASONS)
    mrows = build_match_rows(t)
    res   = eval_result_nested(mrows; test_seasons = TEST_SEASONS)
    (team = t, usable = tu, goals = goals, result = res)
end

pooled(df, metric) = combine(groupby(df, :set),
    [metric, :n] => ((m, n) -> round(sum(m .* n) / sum(n), digits=5)) => metric,
    :n => sum => :n)

for hl in (365.0, 180.0)
    println("\n", "="^70, "\nHALF-LIFE $(Int(hl)) DAYS\n", "="^70)
    blk = run_block(team; half_life_days = hl)
    R2[Symbol("hl", Int(hl))] = blk
    println("usable team rows: ", nrow(blk.usable), "  match rows evaluated: ",
            sum(unique(blk.result[!, [:season, :n]]).n))

    println("\n— GOALS (team-level Poisson, OOS mean log-lik; higher = better) —")
    show(unstack(blk.goals, :season, :set, :loglik), allrows=true); println()
    g = pooled(blk.goals, :loglik)
    show(g, allrows=true); println()
    ll = Dict(r.set => r.loglik for r in eachrow(g))
    println("Δ loglik: B−A = ", round(ll[:B] - ll[:A], digits=5),
            "   C−B = ", round(ll[:C] - ll[:B], digits=5),
            "   C−A = ", round(ll[:C] - ll[:A], digits=5))

    println("\n— HOME WIN (match-level logit, OOS log-loss; lower = better) —")
    show(unstack(blk.result, :season, :set, :logloss), allrows=true); println()
    r = pooled(blk.result, :logloss)
    a = pooled(blk.result, :auc)
    show(innerjoin(r, rename(a, :auc => :auc_pooled)[:, [:set, :auc_pooled]], on = :set),
         allrows=true); println()
    lo = Dict(x.set => x.logloss for x in eachrow(r))
    println("Δ logloss: B−A = ", round(lo[:B] - lo[:A], digits=5),
            "   C−B = ", round(lo[:C] - lo[:B], digits=5),
            "   C−A = ", round(lo[:C] - lo[:A], digits=5))
end

# Interpretation aid: set-C coefficients on all usable League 1/2 rows (hl365)
println("\n", "="^70, "\nSET-C GOALS COEFFICIENTS (all usable L1/2 rows, hl365)\n", "="^70)
tu = R2[:hl365].usable
mC = glm(last(GOALS_SETS[3]), tu, Poisson(), LogLink())
println(coeftable(mC))
R2[:setC_model] = mC

println("\n[INFO] r02 complete — record the verdict in NOTES.md / RESULTS.")
