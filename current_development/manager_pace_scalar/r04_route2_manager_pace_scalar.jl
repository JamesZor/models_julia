# current_development/manager_pace_scalar/r04_route2_manager_pace_scalar.jl
#
# RUNNER: 6-Way Route 2 Out-Of-Sample Benchmark Evaluation
#
# Comparing:
# 1. Baseline Unanchored (`l2_ire79_noanchor`)
# 2. Team Wealth Unanchored (`l2_ire79_wealth`)
# 3. Market Anchored (`l2_ire79_sup40_sw40`)
# 4. Team Wealth + Market Anchored (`l2_ire79_wealth_sup40_sw40`)
# 5. Hierarchical Manager + Wealth (38-param) (`l2_ire79_mgr_wealth`)
# 6. Scalar Manager Pace + Wealth (1-param) (`l2_ire79_mgr_pace_scalar`)

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const BT = BayesianFootball.BackTesting

include(joinpath(dirname(@__DIR__), "team_wealth", "l01_wealth_data.jl"))
include(joinpath(dirname(@__DIR__), "team_wealth", "l02_wealth_engine.jl"))
include(joinpath(dirname(@__DIR__), "team_wealth", "l03_wealth_predict.jl"))

include(joinpath(dirname(@__DIR__), "manager_wealth", "l01_manager_wealth_data.jl"))
include(joinpath(dirname(@__DIR__), "manager_wealth", "l02_manager_wealth_engine.jl"))
include(joinpath(dirname(@__DIR__), "manager_wealth", "l03_manager_wealth_predict.jl"))

include(joinpath(@__DIR__, "l01_manager_pace_data.jl"))
include(joinpath(@__DIR__, "l02_manager_pace_engine.jl"))
include(joinpath(@__DIR__, "l03_manager_pace_predict.jl"))

include(joinpath(dirname(@__DIR__), "orderbook_layer2", "l00_corpus.jl"))
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

function find_newest_experiment(pattern::Regex)
    dirs = filter(d -> occursin(pattern, basename(d)),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR) if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r04: No experiment matching pattern '$pattern' found in $ENGINE_DIR")
    sorted = sort(dirs, by = mtime, rev = true)
    return EE.load_experiment(sorted[1])
end

banner("6-WAY ROUTE 2 BENCHMARK: LOADING ENGINES")

const PIN_PATH = joinpath(ENGINE_DIR, "ds_ire79.jls")
ds = isfile(PIN_PATH) ? deserialize(PIN_PATH) : DD.load_datastore_cached(DD.IrelandPremier())

exp_noanchor        = find_newest_experiment(r"^l2_ire79_noanchor_\d+")
exp_wealth          = find_newest_experiment(r"^l2_ire79_wealth_\d+")
exp_anchored        = find_newest_experiment(r"^l2_ire79_sup40_sw40_\d+")
exp_wealth_anchor   = find_newest_experiment(r"^l2_ire79_wealth_sup40_sw40_\d+")
exp_mgr_wealth      = find_newest_experiment(r"^l2_ire79_mgr_wealth_\d+")
exp_mgr_pace_scalar = find_newest_experiment(r"^l2_ire79_mgr_pace_scalar_\d+")

println("✓ Loaded DataStore (Ireland 79, $(nrow(ds.matches)) matches)")
println("✓ Loaded [1] Baseline Unanchored            : $(exp_noanchor.config.name)")
println("✓ Loaded [2] Team Wealth Unanchored         : $(exp_wealth.config.name)")
println("✓ Loaded [3] Market Anchored                : $(exp_anchored.config.name)")
println("✓ Loaded [4] Team Wealth + Anchored         : $(exp_wealth_anchor.config.name)")
println("✓ Loaded [5] Manager + Wealth (38-param)    : $(exp_mgr_wealth.config.name)")
println("✓ Loaded [6] Scalar Manager Pace (1-param)  : $(exp_mgr_pace_scalar.config.name)")

banner("GENERATING OUT-OF-SAMPLE BOOKS & SELECTIONS")

function process_arm(ds, expr, label::String)
    println("Processing $label...")
    st = route2_setup(ds, expr; price = :close)
    frame = books_frame(st.books, st.ds1)
    base_policy = run_policy(st.books, reference_policy(); label = label, metrics = WEALTH_METRICS)
    return (label = label, expr = expr, st = st, books = st.books, frame = frame, base = base_policy)
end

arm1 = process_arm(ds, exp_noanchor,        "1. Baseline Unanchored")
arm2 = process_arm(ds, exp_wealth,          "2. Team Wealth Unanchored")
arm3 = process_arm(ds, exp_anchored,        "3. Market Anchored (sup40)")
arm4 = process_arm(ds, exp_wealth_anchor,   "4. Team Wealth + Anchored")
arm5 = process_arm(ds, exp_mgr_wealth,      "5. Manager + Wealth (38-param)")
arm6 = process_arm(ds, exp_mgr_pace_scalar, "6. Scalar Manager Pace (1-param)")

banner("[METRIC 1] SUPREMACY DISPERSION & EXPANSION RATIO (ρ)")

function compute_supremacy_dispersion(arm)
    f = filter(r -> r.market == "1X2" && Symbol(r.selection) in (:home, :away), arm.frame)
    matches_df = combine(groupby(f, :match_id)) do sub
        h_row = filter(r -> Symbol(r.selection) == :home, sub)
        a_row = filter(r -> Symbol(r.selection) == :away, sub)
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

disp_df = DataFrame([
    compute_supremacy_dispersion(arm1),
    compute_supremacy_dispersion(arm2),
    compute_supremacy_dispersion(arm3),
    compute_supremacy_dispersion(arm4),
    compute_supremacy_dispersion(arm5),
    compute_supremacy_dispersion(arm6)
])
shw("Supremacy Dispersion Comparison (Target: ρ ≈ 1.0)", disp_df)

banner("[METRIC 2] OUT-OF-SAMPLE ACCURACY & MARKET BLEND (w*)")

function compute_oos_accuracy(arm)
    w_res = w_star(arm.frame)
    b_res = book_skill(arm.frame, arm.label)
    
    f1x2 = filter(r -> r.market == "1X2", arm.frame)
    w_1x2 = isempty(f1x2) ? (ll_model = NaN, ll_market = NaN, w = NaN) : w_star(f1x2)
    brier_mod = isempty(f1x2) ? NaN : mean((f1x2.p_model .- Float64.(f1x2.is_winner)).^2)
    brier_mkt = isempty(f1x2) ? NaN : mean((f1x2.p_market .- Float64.(f1x2.is_winner)).^2)

    return (arm = arm.label, n_selections = nrow(arm.frame),
            ll_model_all = w_res.ll_model, ll_market_all = w_res.ll_market,
            w_star_all = w_res.w, skill_all = b_res.skill,
            ll_model_1x2 = w_1x2.ll_model, brier_model_1x2 = brier_mod,
            brier_market_1x2 = brier_mkt, w_star_1x2 = w_1x2.w)
end

acc_df = DataFrame([
    compute_oos_accuracy(arm1),
    compute_oos_accuracy(arm2),
    compute_oos_accuracy(arm3),
    compute_oos_accuracy(arm4),
    compute_oos_accuracy(arm5),
    compute_oos_accuracy(arm6)
])
shw("Out-of-Sample Accuracy & Skill Comparison:", acc_df)

banner("2. OUT-OF-SAMPLE KELLY COMPOUNDING SIMULATION (SLATE DRAWDOWN)")

pol_bind_1 = run_policy(arm1.st.books, PF.BindingSlateConstraint(); label = "1. Baseline Unanchored [Binding]", metrics = WEALTH_METRICS)
pol_slak_1 = run_policy(arm1.st.books, PF.SlackSlateConstraint(); label = "1. Baseline Unanchored [Slack]", metrics = WEALTH_METRICS)

pol_bind_2 = run_policy(arm2.st.books, PF.BindingSlateConstraint(); label = "2. Team Wealth Unanchored [Binding]", metrics = WEALTH_METRICS)
pol_slak_2 = run_policy(arm2.st.books, PF.SlackSlateConstraint(); label = "2. Team Wealth Unanchored [Slack]", metrics = WEALTH_METRICS)

pol_bind_3 = run_policy(arm3.st.books, PF.BindingSlateConstraint(); label = "3. Market Anchored (sup40) [Binding]", metrics = WEALTH_METRICS)
pol_slak_3 = run_policy(arm3.st.books, PF.SlackSlateConstraint(); label = "3. Market Anchored (sup40) [Slack]", metrics = WEALTH_METRICS)

pol_bind_4 = run_policy(arm4.st.books, PF.BindingSlateConstraint(); label = "4. Team Wealth + Anchored [Binding]", metrics = WEALTH_METRICS)
pol_slak_4 = run_policy(arm4.st.books, PF.SlackSlateConstraint(); label = "4. Team Wealth + Anchored [Slack]", metrics = WEALTH_METRICS)

pol_bind_5 = run_policy(arm5.st.books, PF.BindingSlateConstraint(); label = "5. Manager + Wealth (38-param) [Binding]", metrics = WEALTH_METRICS)
pol_slak_5 = run_policy(arm5.st.books, PF.SlackSlateConstraint(); label = "5. Manager + Wealth (38-param) [Slack]", metrics = WEALTH_METRICS)

pol_bind_6 = run_policy(arm6.st.books, PF.BindingSlateConstraint(); label = "6. Scalar Manager Pace (1-param) [Binding]", metrics = WEALTH_METRICS)
pol_slak_6 = run_policy(arm6.st.books, PF.SlackSlateConstraint(); label = "6. Scalar Manager Pace (1-param) [Slack]", metrics = WEALTH_METRICS)

all_policies = [
    pol_bind_1, pol_slak_1,
    pol_bind_2, pol_slak_2,
    pol_bind_3, pol_slak_3,
    pol_bind_4, pol_slak_4,
    pol_bind_5, pol_slak_5,
    pol_bind_6, pol_slak_6
]

port_comp = portfolio_comparison_table(all_policies)
shw("Comparative Out-of-Sample Kelly Compounding Results:", port_comp; n = 40)

# Save artifact
res_bundle = (;
    acc_df,
    port_comp,
    arm1 = (; frame = arm1.frame, st = arm1.st),
    arm2 = (; frame = arm2.frame, st = arm2.st),
    arm3 = (; frame = arm3.frame, st = arm3.st),
    arm4 = (; frame = arm4.frame, st = arm4.st),
    arm5 = (; frame = arm5.frame, st = arm5.st),
    arm6 = (; frame = arm6.frame, st = arm6.st),
)
out_file = joinpath(OUT_DIR, "route2_manager_pace_scalar_vs_baselines.jls")
serialize(out_file, res_bundle)
println("\n✓ Serialized 6-way Route 2 results to: $out_file")

banner("6-WAY ROUTE 2 BENCHMARK COMPLETE")
