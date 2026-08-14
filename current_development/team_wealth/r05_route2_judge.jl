# current_development/team_wealth/r05_route2_judge.jl
#
# RUNNER: Route 2 Out-Of-Sample Benchmark Evaluation
#
# ==============================================================================
# THREE-WAY HEAD-TO-HEAD JUDGING:
# 1. Baseline Unanchored Engine (`l2_ire79_noanchor`)
# 2. Team Wealth Unanchored Engine (`l2_ire79_wealth`)
# 3. Market Anchored Engine (`l2_ire79_sup40_sw40`)
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const BT = BayesianFootball.BackTesting

include(joinpath(@__DIR__, "l01_wealth_data.jl"))
include(joinpath(@__DIR__, "l02_wealth_engine.jl"))
include(joinpath(@__DIR__, "l03_wealth_predict.jl"))

include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l01_l2_experiment.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l02_l2_ledger.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l03_l2_metrics.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l04_corpus_replay.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l05_curation.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l06_fullbook.jl"))
include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l07_route2.jl"))

const ENGINE_DIR = "./data/l2_ireland_engines"
const OUT_DIR    = "./data/l2_route2_wealth"
mkpath(OUT_DIR)

banner(s) = (println("\n", "="^95); println(s); println("="^95))
shw(t, d; n = 30) = (println("\n", t);
                     isempty(d) ? println("  (empty)") :
                         show(stdout, MIME"text/plain"(), first(d, min(n, nrow(d)))); println())

const WEALTH_METRICS = [BT.CumulativeWealth(), BT.SharpeRatio(), BT.CalmarRatio(), BT.SortinoRatio()]

function find_newest_experiment(prefix::String)
    dirs = filter(d -> startswith(basename(d), prefix),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR) if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r05: No experiment matching prefix '$prefix' found in $ENGINE_DIR")
    sorted = sort(dirs, by = mtime, rev = true)
    return EE.load_experiment(sorted[1])
end

# ==============================================================================
# 1. Load Pinned Data & Experiments
# ==============================================================================

banner("ROUTE 2 THREE-WAY BENCHMARK: LOADING ENGINES")

const PIN_PATH = joinpath(ENGINE_DIR, "ds_ire79.jls")
isfile(PIN_PATH) || error("r05: Missing pinned DataStore at $PIN_PATH")
ds = deserialize(PIN_PATH)

exp_noanchor = find_newest_experiment("l2_ire79_noanchor")
exp_wealth   = find_newest_experiment("l2_ire79_wealth")
exp_anchored = find_newest_experiment("l2_ire79_sup40_sw40")

println("✓ Loaded DataStore (Ireland 79, $(nrow(ds.matches)) matches)")
println("✓ Loaded Baseline Unanchored: $(exp_noanchor.config.name)")
println("✓ Loaded Team Wealth Engine : $(exp_wealth.config.name)")
println("✓ Loaded Market Anchored    : $(exp_anchored.config.name)")

# ==============================================================================
# 2. Build Route 2 Books & Frames
# ==============================================================================

banner("GENERATING OUT-OF-SAMPLE BOOKS & SELECTIONS")

function process_arm(ds, expr, label::String)
    println("Processing $label...")
    st = route2_setup(ds, expr; price = :close)
    frame = books_frame(st.books, st.ds1)
    base_policy = run_policy(st.books, reference_policy(); label = label, metrics = WEALTH_METRICS)
    return (label = label, expr = expr, st = st, books = st.books, frame = frame, base = base_policy)
end

arm_noanchor = process_arm(ds, exp_noanchor, "1. Baseline Unanchored")
arm_wealth   = process_arm(ds, exp_wealth,   "2. Team Wealth Unanchored")
arm_anchored = process_arm(ds, exp_anchored, "3. Market Anchored (sup40)")

# ==============================================================================
# 3. Supremacy Dispersion Analysis
# ==============================================================================

banner("[METRIC 1] SUPREMACY DISPERSION & EXPANSION RATIO (ρ)")

function compute_supremacy_dispersion(arm)
    f = filter(r -> r.family == "1X2" && r.selection in ("Home", "Away"), arm.frame)
    # Group by match to get home / away probs
    matches_df = combine(groupby(f, :match_id)) do sub
        h_row = filter(r -> r.selection == "Home", sub)
        a_row = filter(r -> r.selection == "Away", sub)
        if nrow(h_row) == 1 && nrow(a_row) == 1
            p_h_mod, p_a_mod = h_row.p_model[1], a_row.p_model[1]
            p_h_mkt, p_a_mkt = h_row.p_market[1], a_row.p_market[1]
            sup_mod = log(max(p_h_mod, 1e-6)) - log(max(p_a_mod, 1e-6))
            sup_mkt = log(max(p_h_mkt, 1e-6)) - log(max(p_a_mkt, 1e-6))
            return (sup_mod = sup_mod, sup_mkt = sup_mkt)
        else
            return (sup_mod = NaN, sup_mkt = NaN)
        end
    end
    matches_df = filter(r -> !isnan(r.sup_mod), matches_df)
    sd_mod = std(matches_df.sup_mod)
    sd_mkt = std(matches_df.sup_mkt)
    rho = sd_mod / sd_mkt
    return (arm = arm.label, n_matches = nrow(matches_df),
            sd_model_sup = sd_mod, sd_market_sup = sd_mkt, rho_dispersion = rho)
end

disp_rows = [
    compute_supremacy_dispersion(arm_noanchor),
    compute_supremacy_dispersion(arm_wealth),
    compute_supremacy_dispersion(arm_anchored)
]
disp_df = DataFrame(disp_rows)
shw("Supremacy Dispersion Comparison (Target: ρ ≈ 1.0)", disp_df)

# ==============================================================================
# 4. Out-Of-Sample Accuracy: Log-Loss, Brier Score, Market Blend w*
# ==============================================================================

banner("[METRIC 2] OUT-OF-SAMPLE ACCURACY & MARKET BLEND (w*)")

function compute_oos_accuracy(arm)
    w_res = w_star(arm.frame)
    b_res = book_skill(arm.frame, arm.label)
    
    # 1X2 subset accuracy
    f1x2 = filter(r -> r.family == "1X2", arm.frame)
    w_1x2 = isempty(f1x2) ? (ll_model = NaN, ll_market = NaN, w = NaN) : w_star(f1x2)
    
    # Brier Score on 1X2
    brier_mod = isempty(f1x2) ? NaN : mean((f1x2.p_model .- Float64.(f1x2.is_winner)).^2)
    brier_mkt = isempty(f1x2) ? NaN : mean((f1x2.p_market .- Float64.(f1x2.is_winner)).^2)

    return (arm = arm.label, n_selections = nrow(arm.frame),
            ll_model_all = w_res.ll_model, ll_market_all = w_res.ll_market,
            w_star_all = w_res.w, skill_all = b_res.skill,
            ll_model_1x2 = w_1x2.ll_model, brier_model_1x2 = brier_mod,
            brier_market_1x2 = brier_mkt, w_star_1x2 = w_1x2.w)
end

acc_rows = [
    compute_oos_accuracy(arm_noanchor),
    compute_oos_accuracy(arm_wealth),
    compute_oos_accuracy(arm_anchored)
]
acc_df = DataFrame(acc_rows)
shw("Accuracy & Market Blending Benchmark", acc_df)

# ==============================================================================
# 5. Market Family Breakdown
# ==============================================================================

banner("[METRIC 3] ACCURACY BY SELECTION FAMILY")

function family_breakdown(arm)
    combine(groupby(arm.frame, :family)) do sub
        ws = w_star(sub)
        bs = book_skill(sub, "")
        (arm = arm.label, n = nrow(sub), ll_model = ws.ll_model,
         ll_market = ws.ll_market, w_star = ws.w, skill = bs.skill)
    end
end

fam_all = vcat(family_breakdown(arm_noanchor),
               family_breakdown(arm_wealth),
               family_breakdown(arm_anchored))
shw("Family Breakdown (1X2, BTTS, O/U)", fam_all)

# ==============================================================================
# 6. Portfolio Allocation Simulation (Kelly / FixedCap Staking)
# ==============================================================================

banner("[METRIC 4] PORTFOLIO STAKING PERFORMANCE (SlateDrawdown + FixedCap)")

function run_portfolio_suite(arm)
    binding_policy = reference_policy(trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(23.0), cap = PF.FixedCap(0.25))
    slack_policy   = reference_policy(trust = PF.FlatTrust(0.25), risk = PF.NoRisk(), cap = PF.FixedCap(0.05))
    
    r_bind = run_policy(arm.books, binding_policy; label = "$(arm.label) [Binding Drawdown(23)]", metrics = WEALTH_METRICS)
    r_slack = run_policy(arm.books, slack_policy; label = "$(arm.label) [Slack FixedCap(0.05)]", metrics = WEALTH_METRICS)
    
    return [
        (arm = r_bind.label, slates = r_bind.n_slates, bets = r_bind.n_bets,
         final = r_bind.final, roi = r_bind.roi, roi_lo = r_bind.roi_lo, roi_hi = r_bind.roi_hi,
         growth = r_bind.growth, mdd = r_bind.mdd),
        (arm = r_slack.label, slates = r_slack.n_slates, bets = r_slack.n_bets,
         final = r_slack.final, roi = r_slack.roi, roi_lo = r_slack.roi_lo, roi_hi = r_slack.roi_hi,
         growth = r_slack.growth, mdd = r_slack.mdd)
    ]
end

port_rows = vcat(run_portfolio_suite(arm_noanchor),
                 run_portfolio_suite(arm_wealth),
                 run_portfolio_suite(arm_anchored))
port_df = DataFrame(port_rows)
shw("Portfolio Simulation Results", port_df)

# ==============================================================================
# 7. Serialize Artifacts & Summary
# ==============================================================================

out_artifact_path = joinpath(OUT_DIR, "route2_wealth_vs_baselines.jls")
serialize(out_artifact_path, (
    dispersion = disp_df,
    accuracy = acc_df,
    families = fam_all,
    portfolio = port_df,
    arms = (noanchor = arm_noanchor, wealth = arm_wealth, anchored = arm_anchored)
))

banner("ROUTE 2 BENCHMARK COMPLETE")
println("Artifact saved to: $out_artifact_path")
