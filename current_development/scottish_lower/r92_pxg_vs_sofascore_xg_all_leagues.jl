# ==============================================================================
# r92 — Does the BBC live-text proxy actually measure xG?
# Cross-league validation against official SofaScore xG
# ==============================================================================
#
# WHAT THIS IS
#   An external-validity study of the zonal commentary proxy that `PxGFeature` and
#   `PxGRapmFeature(:y_xg)` are built on.
#
#   THE PROBLEM. The proxy exists BECAUSE Scottish tiers 56/57 have no xG — no
#   SofaScore statistics at all, no shot coordinates, nothing. That absence is
#   exactly why it cannot be validated where it is deployed.
#
#   THE OPPORTUNITY. Six tiers carry BOTH BBC live text and official SofaScore xG:
#   England 1/2/3/84 and Scotland 54/55 — 8,622 matches, 17,244 team-match
#   observations. The proxy can be scored there against a real reference.
#
#   THE QUESTION. Not "does it fit?" but "does it TRANSFER?" A cell table fitted and
#   scored on the same tier will always look good. The deployed table is fitted on
#   56/57's own shots, so the honest test is: fit where we deploy, score where we can
#   measure. Section 7 is that test; pooled and leave-one-tier-out bracket it.
#
#   THE CONTROL. Every metric is reported alongside a naive `shots × league mean xG
#   per shot` baseline. If parsing zone / body part / context does not beat counting
#   attempts, the parser is decoration and the proxy is a shot counter.
#
# WHAT THIS IS NOT
#   Not a claim about tiers 56/57. Nothing here measures the proxy on the tiers it is
#   deployed on; it measures whether a table fitted on those tiers behaves sensibly
#   on tiers where truth is observable. That is the strongest available evidence and
#   it is still indirect.
#
#   Not a betting study. Agreement with official xG is a measurement property, not an
#   edge.
#
# USAGE
#   source .env        # or otherwise export BF_DB_URL
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r92_pxg_vs_sofascore_xg_all_leagues.jl")
#
#   The database pull is cached to l92_pxg_validation_pull.jls. Set R92_FORCE_PULL=1
#   to refresh it after a re-scrape.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics

include(joinpath(@__DIR__, "l93_eda_toolkit.jl"))
include(joinpath(@__DIR__, "l92_pxg_validation.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R92_FORCE_PULL = get(ENV, "R92_FORCE_PULL", "0") == "1"
const R92_K = 25.0            # empirical-Bayes pseudo-count; the production default
const R92_DECILES = 10

eda_banner("r92 · BBC live-text pxG proxy vs official SofaScore xG")
println("  cell-table shrinkage k : ", R92_K)
println("  tiers                  : ",
        join(["$(t)=$(L92_TOURNAMENTS[t])" for t in L92_TIER_ORDER], "  "))

# %%
# ==============================================================================
# 3. Data snapshot
# ==============================================================================

eda_section("1/8", "Database pull and coverage")

pull = l92_fetch(; force = R92_FORCE_PULL)
@printf("  attempts pulled          : %d\n", nrow(pull.events))
@printf("  matches with official xG : %d\n", nrow(pull.official))
@printf("  match metadata rows      : %d\n", nrow(pull.matches))

shots = l92_shot_table(pull)
unattributed = count(ismissing, shots.is_home) / max(nrow(shots), 1)
@printf("  attempts with no side    : %d (%.2f%%)  [production expects ~2.4%%]\n",
        count(ismissing, shots.is_home), 100 * unattributed)
@printf("  parse coverage           : %.2f%% of attempts yielded a zone or a penalty\n",
        100 * count(shots.parsed) / max(nrow(shots), 1))

official_ids = Set(Int.(pull.official.match_id))
tier_of = Dict(Int(r.match_id) => Int(r.tournament_id) for r in eachrow(pull.matches))
livetext_ids = Set(Int.(shots.match_id))

println("\n  Coverage by tier (matches):")
@printf("  %-24s | %8s | %10s | %10s | %10s\n",
        "tier", "live text", "official xG", "BOTH", "attempts")
println("  " * repeat('-', 76))
for tier in L92_TIER_ORDER
    ids_lt = count(id -> get(tier_of, id, 0) == tier, livetext_ids)
    ids_xg = count(id -> get(tier_of, id, 0) == tier, official_ids)
    both = count(id -> get(tier_of, id, 0) == tier && id in official_ids, livetext_ids)
    n_shots = count(==(tier), shots.tournament_id)
    @printf("  %-24s | %8d | %10d | %10d | %10d\n",
            L92_TOURNAMENTS[tier], ids_lt, ids_xg, both, n_shots)
end
println("\n  [NOTE] Tiers 56/57 show zero official xG. That is the fact the proxy exists for,")
println("         and the reason this study can only ever validate it by transfer.")

# %%
# ==============================================================================
# 4. Shot descriptor distributions
# ==============================================================================

eda_section("2/8", "What the parser sees")

eda_freq_table(shots.zone; title = "Zone (all tiers):", top = 12)
println()
eda_freq_table(shots.body_part; title = "Body part:", top = 6)
println()
eda_freq_table(shots.context; title = "Set-piece context:", top = 6)
@printf("\n  penalties: %d (%.2f%% of attempts) | goals: %d | on target: %d\n",
        count(shots.is_penalty), 100 * count(shots.is_penalty) / max(nrow(shots), 1),
        count(shots.is_goal), count(shots.is_on_target))

println("\n  Parse coverage by tier (a drop here means a commentary-template drift):")
@printf("  %-24s | %9s | %9s | %9s\n", "tier", "attempts", "parsed %", "penalty %")
println("  " * repeat('-', 60))
for tier in L92_TIER_ORDER
    sub = shots[shots.tournament_id .== tier, :]
    nrow(sub) == 0 && continue
    @printf("  %-24s | %9d | %8.2f%% | %8.2f%%\n",
            L92_TOURNAMENTS[tier], nrow(sub),
            100 * count(sub.parsed) / nrow(sub),
            100 * count(sub.is_penalty) / nrow(sub))
end

# %%
# ==============================================================================
# 5. The fitted cell tables
# ==============================================================================

eda_section("3/8", "Fitting regimes")

regimes = l92_fit_regimes(shots; k = R92_K)
pooled = first(filter(r -> r.label == "pooled", regimes))
deployed_hits = filter(r -> r.label == "sco_lower", regimes)
deployed = isempty(deployed_hits) ? nothing : first(deployed_hits)

@printf("  regimes built: %d (pooled, sco_lower, and leave-one-tier-out per tier)\n",
        length(regimes))
@printf("  pooled base rate  : %.4f | penalty xG : %.4f | cells : %d\n",
        pooled.model.base_rate, pooled.model.penalty_xg, length(pooled.model.cells))
if deployed !== nothing
    @printf("  sco_lower (56/57) : %.4f | penalty xG : %.4f | cells : %d\n",
            deployed.model.base_rate, deployed.model.penalty_xg,
            length(deployed.model.cells))
end

println("\n  Highest-value cells under the pooled table:")
cells = l92_cell_table(pooled.model; top = 14)
@printf("  %-20s | %-12s | %-18s | %7s\n", "zone", "body", "context", "xG")
println("  " * repeat('-', 66))
for r in eachrow(cells)
    @printf("  %-20s | %-12s | %-18s | %7.4f\n", r.zone, r.body_part, r.context, r.xg)
end

# Do the two tables agree about which cells are dangerous? A rank disagreement here
# is the mechanism by which transfer would fail.
if deployed !== nothing
    shared = intersect(keys(pooled.model.cells), keys(deployed.model.cells))
    if length(shared) >= 5
        a = [pooled.model.cells[k] for k in shared]
        b = [deployed.model.cells[k] for k in shared]
        @printf("\n  pooled vs sco_lower cell agreement over %d shared cells: r = %.3f, rho = %.3f\n",
                length(shared), eda_pearson(a, b), eda_spearman(a, b))
    end
end

# %%
# ==============================================================================
# 6. Team-match agreement, pooled regime
# ==============================================================================

eda_section("4/8", "Team-match agreement — pooled table")

team_all = l92_team_frame(shots, pull; model = pooled.model, label = "pooled")
@printf("  team-match observations with both proxy and official xG: %d\n", nrow(team_all))

# ---- THE ZERO-FILL DEFECT -------------------------------------------------------
# Before any agreement number is believed: how much of the reference is real?
println("\n  Official-xG zero-fill audit (BOTH sides exactly 0.000):")
zero_report = l92_zero_fill_report(team_all)
@printf("  %-24s | %8s | %8s | %11s | %8s\n",
        "tier", "team-obs", "live", "zero-filled", "share")
println("  " * repeat('-', 70))
for r in eachrow(zero_report)
    @printf("  %-24s | %8d | %8d | %11d | %7.1f%%\n",
            r.tier, r.n, r.live, r.zero_filled, 100 * r.zero_share)
end
println()
println("  A match in this frame HAS live-text attempts, so an official 0.000/0.000 cannot")
println("  be a measurement — it is a placeholder row, and the tiers carrying them are named")
println("  above. Scoring against those rows drags a tier's apparent correlation from ~0.85 to")
println("  ~0.30 and invents a ~+0.85 xG bias. Every metric below excludes them.")
let worst = nrow(zero_report) == 0 ? nothing : argmax(zero_report.zero_share)
    worst === nothing || @printf("  worst affected: %s at %.1f%% zero-filled\n",
                                 zero_report.tier[worst], 100 * zero_report.zero_share[worst])
end

team_pooled = l92_live(team_all)
@printf("\n  usable team-match observations after the zero-fill filter: %d (dropped %d)\n",
        nrow(team_pooled), nrow(team_all) - nrow(team_pooled))

eda_print_describe([
    eda_describe("proxy pxG", team_pooled.pxg),
    eda_describe("official xG", team_pooled.official_xg),
    eda_describe("shot-count control", team_pooled.pxg_shot_count),
    eda_describe("goals", team_pooled.goals),
]; title = "\n  Distributions (team-match):")

println()
overall = [
    eda_agreement("ALL TIERS · proxy", team_pooled.pxg, team_pooled.official_xg),
    eda_agreement("ALL TIERS · control", team_pooled.pxg_shot_count, team_pooled.official_xg),
]
eda_print_agreement(overall; title = "Proxy vs the shot-count control:")

println()
by_tier = NamedTuple[]
for tier in L92_TIER_ORDER
    sub = team_pooled[team_pooled.tournament_id .== tier, :]
    nrow(sub) < 20 && continue
    push!(by_tier, eda_agreement(L92_TOURNAMENTS[tier], sub.pxg, sub.official_xg))
end
eda_print_agreement(by_tier; title = "By tier (pooled table):")

println()
control_by_tier = NamedTuple[]
for tier in L92_TIER_ORDER
    sub = team_pooled[team_pooled.tournament_id .== tier, :]
    nrow(sub) < 20 && continue
    push!(control_by_tier,
          eda_agreement(L92_TOURNAMENTS[tier], sub.pxg_shot_count, sub.official_xg))
end
eda_print_agreement(control_by_tier; title = "By tier (shot-count control):")

# %%
# ==============================================================================
# 7. Calibration and residuals
# ==============================================================================

eda_section("5/8", "Calibration and residual structure")

eda_print_decile(eda_decile_table(team_pooled.pxg, team_pooled.official_xg; k = R92_DECILES);
                 xlab = "proxy", ylab = "official",
                 title = "Decile calibration (proxy bin -> mean official xG):")

println()
eda_scatter(team_pooled.pxg, team_pooled.official_xg;
            xlab = "proxy pxG", ylab = "official xG",
            title = "Team-match scatter (all tiers, pooled table):")

residual = team_pooled.pxg .- team_pooled.official_xg
println()
eda_histogram(residual; label = "Residual (proxy - official), team-match:", bins = 22)

println()
eda_scatter(team_pooled.official_xg, residual;
            xlab = "official xG", ylab = "residual (proxy - official)",
            title = "Residual vs reference — a tilt here IS the calibration slope:")

@printf("\n  residual vs official xG correlation: %+.4f\n",
        eda_pearson(team_pooled.official_xg, residual))
println("  (a negative value means the proxy compresses: too high on weak performances,")
println("   too low on strong ones — i.e. a calibration slope above 1.)")

println("\n  Residual summary by tier:")
@printf("  %-24s | %7s | %8s | %8s | %8s | %8s\n",
        "tier", "n", "bias", "MAE", "RMSE", "sd(resid)")
println("  " * repeat('-', 74))
for tier in L92_TIER_ORDER
    sub = team_pooled[team_pooled.tournament_id .== tier, :]
    nrow(sub) < 20 && continue
    r = sub.pxg .- sub.official_xg
    @printf("  %-24s | %7d | %+8.4f | %8.4f | %8.4f | %8.4f\n",
            L92_TOURNAMENTS[tier], length(r), mean(r), mean(abs, r),
            sqrt(mean(abs2, r)), std(r))
end

# %%
# ==============================================================================
# 8. Transfer: the regime that licenses the production claim
# ==============================================================================

eda_section("6/8", "Transfer — fit where we deploy, score where we can measure")

println("  Each row scores a tier with a table that never saw that tier's shots.")
println("  `sco_lower` is the DEPLOYED table (fitted on 56/57 only).\n")

transfer = NamedTuple[]
for tier in L92_TIER_ORDER
    tier in (56, 57) && continue        # no official xG to score against
    loto_hits = filter(r -> r.label == "loto/$(tier)", regimes)
    isempty(loto_hits) && continue
    loto = first(loto_hits)

    sub_ids = Set(Int.(pull.matches[pull.matches.tournament_id .== tier, :match_id]))
    tier_shots = shots[in.(shots.match_id, Ref(sub_ids)), :]
    nrow(tier_shots) == 0 && continue

    frame_loto = l92_live(l92_team_frame(tier_shots, pull; model = loto.model, label = "loto"))
    nrow(frame_loto) < 20 && continue
    push!(transfer, eda_agreement("$(L92_TOURNAMENTS[tier]) [loto]",
                                  frame_loto.pxg, frame_loto.official_xg))

    if deployed !== nothing
        frame_dep = l92_live(l92_team_frame(tier_shots, pull; model = deployed.model, label = "sco_lower"))
        push!(transfer, eda_agreement("$(L92_TOURNAMENTS[tier]) [sco_lower]",
                                      frame_dep.pxg, frame_dep.official_xg))
    end
end
eda_print_agreement(transfer; title = "Held-out transfer:")

println("\n  Read the pairs: if [sco_lower] tracks [loto] closely, a table fitted on the")
println("  lowest two Scottish tiers carries the same information as one fitted on seven")
println("  other tiers — which is the claim the deployed feature rests on.")

# %%
# ==============================================================================
# 9. Match level and the supremacy difference
# ==============================================================================

eda_section("7/8", "Match total and the supremacy difference")

match_pooled = l92_live(l92_match_frame(team_pooled))
@printf("  matches with both sides resolved: %d\n\n", nrow(match_pooled))

level = [
    eda_agreement("match TOTAL", match_pooled.pxg_total, match_pooled.official_total),
    eda_agreement("match DIFFERENCE", match_pooled.pxg_diff, match_pooled.official_diff),
]
eda_print_agreement(level; title = "Aggregation level:")

println("\n  The DIFFERENCE row is the decision-relevant one: PxGCovariate's supremacy")
println("  role is built from exactly this quantity, and a proxy can track the total")
println("  well while tracking the difference badly.\n")

# Restrict to matches where BOTH differences are non-zero: a tie has no sign to agree with,
# and counting it as a miss understates the statistic.
let decided = (abs.(match_pooled.pxg_diff) .> 1e-9) .& (abs.(match_pooled.official_diff) .> 1e-9)
    agree = mean(sign.(match_pooled.pxg_diff[decided]) .== sign.(match_pooled.official_diff[decided]))
    @printf("  sign agreement on the xG difference : %.2f%%  (over %d decided matches)\n",
            100 * agree, count(decided))
end
@printf("  AUC, proxy diff -> official diff > 0 : %.4f\n",
        eda_auc(match_pooled.pxg_diff, match_pooled.official_diff .> 0))
@printf("  AUC, proxy diff -> actual home win   : %.4f\n",
        eda_auc(match_pooled.pxg_diff, match_pooled.goal_diff .> 0))
@printf("  AUC, official diff -> actual home win: %.4f   [the reference's own ceiling]\n",
        eda_auc(match_pooled.official_diff, match_pooled.goal_diff .> 0))

println()
eda_print_decile(eda_decile_table(match_pooled.pxg_diff, match_pooled.goal_diff; k = R92_DECILES);
                 xlab = "proxy diff", ylab = "goal diff",
                 title = "Does the proxy difference order real scorelines?")

# %%
# ==============================================================================
# 10. Verdict
# ==============================================================================

eda_section("8/8", "Verdict")

let
    all_proxy = overall[1]
    all_control = overall[2]
    gain_r = all_proxy.r - all_control.r
    gain_mae = all_control.mae - all_proxy.mae

    @printf("  Proxy   : r = %.3f, rho = %.3f, MAE = %.3f, RMSE = %.3f, bias = %+.3f, slope = %.3f, CCC = %.3f\n",
            all_proxy.r, all_proxy.rho, all_proxy.mae, all_proxy.rmse,
            all_proxy.bias, all_proxy.slope, all_proxy.ccc)
    @printf("  Control : r = %.3f, rho = %.3f, MAE = %.3f, RMSE = %.3f, bias = %+.3f, slope = %.3f, CCC = %.3f\n",
            all_control.r, all_control.rho, all_control.mae, all_control.rmse,
            all_control.bias, all_control.slope, all_control.ccc)
    @printf("\n  Parsing buys %+.3f correlation and %+.3f MAE over counting attempts.\n",
            gain_r, gain_mae)

    if gain_r < 0.02 && gain_mae < 0.01
        println("  [VERDICT] The parser is NOT earning its complexity — the proxy is a shot")
        println("            counter with extra steps. Prefer the control, or revisit the")
        println("            vocabulary in src/features/plus_minus/shot_parser.jl.")
    else
        println("  [VERDICT] The zonal parse carries information beyond attempt volume.")
    end

    if !isnan(all_proxy.slope) && abs(all_proxy.slope - 1) > 0.15
        @printf("  [NOTE]    Calibration slope %.3f is far from 1: the proxy is on a different\n",
                all_proxy.slope)
        println("            SCALE from official xG. Harmless for a covariate the model learns a")
        println("            weight for; it would matter if pxG were used as an absolute rate.")
    end
end

println("\n  Reminder: tiers 56/57 carry no official xG. Everything above is transfer")
println("  evidence, and the deployed claim rests on the [sco_lower] rows in section 6.")
eda_rule(100, '=')
