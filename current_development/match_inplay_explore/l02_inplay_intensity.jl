#=
l02_inplay_intensity.jl  —  Parametric in-play scoring-intensity model.

Learns λ_inst(t, game_state, team_quality) — the per-90 instantaneous scoring rate — and predicts
REALIZED remaining goals via a Poisson GLM with a time-exposure offset:

    rem_goals_side ~ Poisson( λ_inst_side * rem_frac )
    log λ_inst_side = α + f(t_m) + β_home*is_home + β_trail*trailing + β_lead*leading
                        + β_man*man_adv_side + γ*log(pregame_λ_side)
    offset = log(rem_frac),  rem_frac = (90 - t_m)/90

Ground-truth target = actual remaining goals (final score − score at the bin), so the model is
INDEPENDENT of live odds and can be bet against the market. Validated on Ireland (held-out by
match): beats both the pregame-only baseline and the market's own λ_rem at predicting realized
remaining goals; score effects trailing +0.25 / leading −0.24 (log-rate) are highly significant.

Built on top of the l01 panel (`inplay_lambda_trace` / `build_panel`). Designed in LONG format
(one row per match×bin×side, with team_index/opp_index) so a future Turing.jl version can swap the
GLM for hierarchical pooling: global coefficients + delta[team_index].

Reuses: panel columns from l01 (gh, ga, t_m, λ_rem_*, residual), pre-game λ join (pg_λ_h/a),
        final scores from ds.matches.home_score/away_score.
=#

using DataFrames
using GLM
using Statistics
using Random
using SpecialFunctions: lgamma

# ---------------------------------------------------------------------------
# 1. Dataset construction (long format: one row per match × bin × side)
# ---------------------------------------------------------------------------

"Final scores per match from ds.matches (missing dropped downstream)."
function final_scores(ds)
    DataFrame(match_id = Int.(ds.matches.match_id),
              fh = [ismissing(x) ? missing : Int(x) for x in ds.matches.home_score],
              fa = [ismissing(x) ? missing : Int(x) for x in ds.matches.away_score])
end

"""
    build_intensity_dataset(panel, ds; tmax=80.0, remfloor=0.05, resid_max=0.08) -> DataFrame

Stack each clean panel bin into two rows (home side / away side). Covariates are defined from the
SIDE's perspective so a single model learns symmetric score effects with one `is_home` term.
Target `rem_goals` = realized remaining goals for that side; `logrem` = log time-exposure offset.
Drops clock-misassigned rows (`rem_goals < 0`) and bins with no pre-game λ.
"""
function build_intensity_dataset(panel, ds; tmax = 80.0, remfloor = 0.05, resid_max = 0.08)
    P = leftjoin(panel, final_scores(ds), on = :match_id)
    dropmissing!(P, [:fh, :fa, :pg_λ_h, :pg_λ_a])
    P = subset(P, :residual => ByRow(x -> !isnan(x) && x < resid_max),
                  :t_m      => ByRow(x -> 1 <= x <= tmax))
    rows = NamedTuple[]
    for r in eachrow(P)
        rf    = max((90.0 - r.t_m) / 90.0, remfloor)
        man_h = r.away_reds - r.home_reds                  # +ve => home has the extra man
        # (is_home, realized remaining, own−opp goal diff, man adv, log pregame λ, market λ_rem)
        sides = ((1, r.fh - r.gh, r.gh - r.ga,  man_h, log(r.pg_λ_h), r.λ_rem_h),
                 (0, r.fa - r.ga, r.ga - r.gh, -man_h, log(r.pg_λ_a), r.λ_rem_a))
        for (is_home, rem, gds, man, logpg, mkt) in sides
            rem < 0 && continue
            push!(rows, (
                match_id    = r.match_id,
                rem_goals   = rem,
                logrem      = log(rf),
                rem_frac    = rf,
                t_m         = r.t_m,
                t_m2        = r.t_m^2,
                is_home     = Float64(is_home),
                goal_diff   = gds,                     # signed own−opp (for game-state index)
                trailing    = Float64(gds < 0),
                leading     = Float64(gds > 0),
                man_adv     = Float64(man),
                log_pregame = logpg,
                mkt_lam     = mkt,
            ))
        end
    end
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# 2. Fit / predict
# ---------------------------------------------------------------------------

const INTENSITY_FORMULA =
    @formula(rem_goals ~ t_m + t_m2 + is_home + trailing + leading + man_adv + log_pregame)

"Fit the Poisson intensity GLM (log link, time-exposure offset)."
function fit_intensity_model(df; formula = INTENSITY_FORMULA)
    return glm(formula, df, Poisson(), LogLink(); offset = df.logrem)
end

"Predicted mean remaining goals (= λ_inst * rem_frac) for `newdata`."
predict_intensity(model, newdata) = predict(model, newdata; offset = newdata.logrem)

# ---------------------------------------------------------------------------
# 3. Evaluation helpers
# ---------------------------------------------------------------------------

poisson_logscore(y, μ) = y * log(max(μ, 1e-9)) - μ - lgamma(y + 1)
mean_logscore(y, μ)    = mean(poisson_logscore.(Float64.(y), μ))

"Split match ids into (train, test) sets (75/25 by default)."
function split_by_match(df; frac = 0.75, seed = 1)
    ms = shuffle(MersenneTwister(seed), unique(df.match_id))
    cut = round(Int, frac * length(ms))
    train = Set(ms[1:cut])
    return (subset(df, :match_id => ByRow(in(train))),
            subset(df, :match_id => ByRow(!in(train))))
end
