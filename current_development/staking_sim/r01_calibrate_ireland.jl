#=
RUNNER — one-off calibration of SimConfig dials to the Ireland Premier reference.

Run on the kaimon session (BayesianFootball loaded, .cache/datastore_Ireland.jls present):
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    using BayesianFootball
    include("current_development/staking_sim/r01_calibrate_ireland.jl")

Prints a paste-block of calibrated values → bake into SimConfig defaults in
l01_sim_market_model.jl and record in experiments.md. Then sanity-simulates 5k matches at
those dials and compares league stats to the Ireland empirical values.

Lognormal/Jensen note: with zero-sum team effects α,β ~ N(0, σ0²) i.i.d.,
E[λ_away] = exp(μ + σ0²)  ⇒  μ  = log(mean_away_goals) − σ0²
E[λ_home] = exp(μ + ha + σ0²) ⇒ ha = log(mean_home / mean_away).
=#

using Statistics
using DataFrames

include(joinpath(@__DIR__, "l01_sim_market_model.jl"))

ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# ---- empirical league stats (completed matches with scores) ----
mdf = dropmissing(ds.matches, [:home_score, :away_score])
mh, ma = mean(mdf.home_score), mean(mdf.away_score)
emp_hw = mean(mdf.home_score .> mdf.away_score)
emp_dr = mean(mdf.home_score .== mdf.away_score)
emp_o25 = mean(mdf.home_score .+ mdf.away_score .> 2.5)

σ0_default = 0.20
μ_cal = log(ma) - σ0_default^2
ha_cal = log(mh / ma)

# ---- empirical closing overrounds per family ----
odf = ds.odds
ors = Dict{String,Float64}()
if :overround_close in propertynames(odf)
    for g in groupby(dropmissing(odf, :overround_close), :market_name)
        # one overround per (match, market, line) group; mean of per-row values is fine
        ors[string(g.market_name[1])] = mean(g.overround_close)
    end
end
O_1x2 = get(ors, "1X2", 1.06)
O_ou = get(ors, "OverUnder", 1.05)
O_btts = get(ors, "BTTS", 1.06)

println("="^70)
println("IRELAND PREMIER CALIBRATION (n matches = $(nrow(mdf)))")
println("  mean goals  home=$(round(mh,digits=3))  away=$(round(ma,digits=3))")
println("  home win %=$(round(100emp_hw,digits=1))  draw %=$(round(100emp_dr,digits=1))  over2.5 %=$(round(100emp_o25,digits=1))")
println("  overrounds: ", ors)
println()
println("PASTE INTO SimConfig (l01_sim_market_model.jl):")
println("    μ::Float64 = $(round(μ_cal, digits=4))")
println("    ha::Float64 = $(round(ha_cal, digits=4))")
println("    O_1x2::Float64 = $(round(O_1x2, digits=4))")
println("    O_ou::Float64 = $(round(O_ou, digits=4))")
println("    O_btts::Float64 = $(round(O_btts, digits=4))")
println("="^70)

# ---- sanity: simulate 15 CAMPAIGN-LENGTH chunks at the calibrated dials ----
# (one long continuous run is the wrong check: the zero-sum random walk's cross-team
#  spread grows without bound over ~1000 rounds and inflates goal means ~5–15%; the MC
#  uses fresh 330-match campaigns, so the honest sanity check does too.)
using Random
cfg = SimConfig(μ=μ_cal, ha=ha_cal, O_1x2=O_1x2, O_ou=O_ou, O_btts=O_btts,
                n_matches=330, n_prehist=0)
sh = Int[]; sa = Int[]
for s in 1:15
    for sm in simulate_campaign(cfg, Xoshiro(3000 + s); S=2)
        push!(sh, sm.score[1]); push!(sa, sm.score[2])
    end
end
println("SIM (15×330-match campaigns @ calibrated dials)  vs  EMPIRICAL")
println("  mean home goals  $(round(mean(sh),digits=3))  vs  $(round(mh,digits=3))")
println("  mean away goals  $(round(mean(sa),digits=3))  vs  $(round(ma,digits=3))")
println("  home win %       $(round(100mean(sh.>sa),digits=1))  vs  $(round(100emp_hw,digits=1))")
println("  draw %           $(round(100mean(sh.==sa),digits=1))  vs  $(round(100emp_dr,digits=1))")
println("  over 2.5 %       $(round(100mean(sh.+sa.>2.5),digits=1))  vs  $(round(100emp_o25,digits=1))")
