# ==============================================================================
# SCOTTISH LOWER — ALL 4 ARMS COMPARATIVE EVALUATION (GATE 6 & GATE 7)
# ==============================================================================
using BayesianFootball
using DataFrames
using Statistics
using Printf

const SL_ROOT = let d = @__DIR__
    isfile(joinpath(d, "_protocol", "ScottishLowerProtocol.jl")) ? d :
        "current_development/scottish_lower"
end

include(joinpath(SL_ROOT, "_protocol/ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol

include(joinpath(SL_ROOT, "00_team_poisson/l01_model.jl"))
include(joinpath(SL_ROOT, "00_team_poisson/l02_equations.jl"))
include(joinpath(SL_ROOT, "00_team_poisson/l03_adapter.jl"))

include(joinpath(SL_ROOT, "02_poisson_wealth/l01_model.jl"))
include(joinpath(SL_ROOT, "02_poisson_wealth/l02_equations.jl"))
include(joinpath(SL_ROOT, "02_poisson_wealth/l03_adapter.jl"))

include(joinpath(SL_ROOT, "03_poisson_distance/l01_model.jl"))
include(joinpath(SL_ROOT, "03_poisson_distance/l02_equations.jl"))
include(joinpath(SL_ROOT, "03_poisson_distance/l03_adapter.jl"))

include(joinpath(SL_ROOT, "04_poisson_wealth_distance/l01_model.jl"))
include(joinpath(SL_ROOT, "04_poisson_wealth_distance/l02_equations.jl"))
include(joinpath(SL_ROOT, "04_poisson_wealth_distance/l03_adapter.jl"))

# 1. Setup contract and data
contract = sl_contract()
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower(); max_age_hours=100_000)
folds = sl_build_folds(ds, contract)

# Artifact paths on disk
arms = [
    ("00 Control (Pure Poisson)", TP00Adapter(half_life_days=180.0),
     "data/scottish_lower/00_team_poisson/f168cd23/00_team_poisson_grid_f168cd23_20260828_192218"),
    ("02 + Squad Wealth",         TP02Adapter(half_life_days=180.0),
     "data/scottish_lower/02_poisson_wealth/d4ad8fb1/02_poisson_wealth_grid_d4ad8fb1_20260828_193750"),
    ("03 + Travel Distance",      TP03Adapter(half_life_days=180.0),
     "data/scottish_lower/03_poisson_distance/385549d2/03_poisson_distance_grid_385549d2_20260828_195305"),
    ("04 + Joint Wealth & Dist",  TP04Adapter(half_life_days=180.0),
     "data/scottish_lower/04_poisson_wealth_distance/30568d91/04_poisson_wealth_distance_grid_30568d91_20260828_201019"),
]

betfair_odds = sl_betfair_odds_df(ds, contract)
spec = sl_book_spec(contract)

gate6_summary = DataFrame(
    arm = String[],
    ll_1x2 = Float64[],
    ll_ou25 = Float64[],
    beta_model_1x2 = Float64[],
    z_model_1x2 = Float64[],
    rqr_sd = Float64[],
    mean_lpd = Float64[]
)

gate7_summary = DataFrame(
    arm = String[],
    policy = String[],
    n_bets = Int[],
    final_wealth = Float64[],
    roi_pct = Float64[],
    roi_lo = Float64[],
    roi_hi = Float64[],
    growth_slate = Float64[],
    mdd_pct = Float64[]
)

println("=" ^ 100)
println("EVALUATING ALL 4 MODELS ACROSS 20 WALKING-FORWARD FOLDS")
println("=" ^ 100)

for (label, adapter, path) in arms
    println("\nEvaluating: ", label)
    loaded = sl_load_experiment(path)
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, loaded; force = true)
    oos_ids = Set(Int.(latents.df.match_id))
    
    # Gate 6 Books & Diagnostics
    bet365_odds = sl_market_book(ds.odds, contract; ids = oos_ids)
    model_b, fixtures = sl_model_book(adapter, latents, ds, contract)
    joined_b365 = innerjoin(model_b, bet365_odds, on=[:match_id, :market, :line, :selection])
    
    edge_tbl = ScottishLowerProtocol.sl_edge_table(joined_b365)
    score_tbl = sl_score_table(joined_b365)
    shape = sl_gate_shape(fixtures)
    
    # Extract 1X2 metrics
    e_1x2 = filter(r -> r.market == "1X2", edge_tbl)
    s_1x2 = filter(r -> r.market == "1X2", score_tbl)
    s_ou25 = filter(r -> r.market == "OverUnder" && r.line == 2.5, score_tbl)
    
    avg_beta = mean(filter(isfinite, e_1x2.β_model))
    avg_z = mean(filter(isfinite, e_1x2.z_model))
    ll_1x2_val = isempty(s_1x2) ? NaN : mean(s_1x2.ll_model)
    ll_ou25_val = isempty(s_ou25) ? NaN : mean(s_ou25.ll_model)
    
    rqr_sd_val = std(filter(isfinite, vcat(fixtures.rqr_h, fixtures.rqr_a)))
    lpd_val = mean(filter(isfinite, fixtures.lpd))
    
    push!(gate6_summary, (label, ll_1x2_val, ll_ou25_val, avg_beta, avg_z, rqr_sd_val, lpd_val))
    
    # Gate 7 Kelly Growth Simulation
    bf_odds = sl_betfair_odds_df(ds, contract; ids = oos_ids)
    books = BayesianFootball.Portfolio.build_books(spec, latents.df, loaded, bf_odds, ds)
    gt = sl_growth_table(books, contract; B=1500)
    for row in eachrow(gt)
        push!(gate7_summary, (label, row.policy, row.n_bets, row.final, row.roi_pct, row.roi_lo, row.roi_hi, row.growth, row.mdd_pct))
    end
end

println("\n" * "=" ^ 100)
println("GATE 6 SUMMARY: PROPER SCORES & GLM EDGE METRICS (vs Bet365 Close)")
println("=" ^ 100)
@printf("%-30s | %8s | %8s | %10s | %10s | %7s | %8s\n", "Model", "LL 1X2", "LL O2.5", "β_model", "z_model", "RQR SD", "Mean LPD")
println("-" ^ 100)
for r in eachrow(gate6_summary)
    @printf("%-30s | %8.4f | %8.4f | %+10.4f | %+10.2f | %7.4f | %8.4f\n",
            r.arm, r.ll_1x2, r.ll_ou25, r.beta_model_1x2, r.z_model_1x2, r.rqr_sd, r.mean_lpd)
end

println("\n" * "=" ^ 115)
println("GATE 7 SUMMARY: PORTFOLIO-KELLY GROWTH & DRAWDOWN (vs Betfair Close)")
println("=" ^ 115)
@printf("%-28s | %-14s | %6s | %7s | %8s [%-13s] | %10s | %8s\n", "Model", "Policy", "Bets", "Final", "ROI %", "95% CI", "Growth/Sl", "Max DD %")
println("-" ^ 115)
for r in eachrow(gate7_summary)
    ci_str = @sprintf("%+.1f, %+.1f", r.roi_lo, r.roi_hi)
    @printf("%-28s | %-14s | %6d | %6.3fx | %+7.2f%% [%-13s] | %+10.5f | %7.1f%%\n",
            r.arm, r.policy, r.n_bets, r.final_wealth, r.roi_pct, ci_str, r.growth_slate, r.mdd_pct)
end
println("=" ^ 115)
