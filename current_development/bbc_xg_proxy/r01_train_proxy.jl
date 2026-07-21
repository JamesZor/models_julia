#=
r01 — WP1 runner: train + evaluate the bbc-stats xG proxy.

Ladder: m0 (SoT-only scaler) → m1 (full linear) → m2 (+sqrt terms & SoT×poss),
each under Gamma/log and lognormal links.
Gates:
  A. season-blocked CV — pooled OOS R²/MAE/Spearman per model.
  B. Champ↔Prem transfer — tier-invariance (League 1/2 sit outside training tiers).
  C. decile calibration of the winner.
Winner is frozen to proxy_model_v1.jls (+ feature spec) for WP2.

Run:  include(".../current_development/bbc_xg_proxy/r01_train_proxy.jl")
Results collected in global `R1::Dict`.
=#

include(joinpath(@__DIR__, "l01_xg_proxy.jl"))
using Serialization

R1 = Dict{Symbol,Any}()

conn = LibPQ.Connection(ENV["BF_DB_URL"])
wide = fetch_matches_wide(conn)
team = to_team_rows(wide)
tr   = training_rows(team[.!ismissing.(team.xg), :])
println("[INFO] matches=", nrow(wide), "  team rows=", nrow(team),
        "  training rows=", nrow(tr), " (tiers: ", sort(unique(tr.tournament_id)), ")")
R1[:wide] = wide; R1[:team] = team; R1[:tr] = tr

# ==========================================
# A. Season-blocked CV over the ladder
# ==========================================
println("\n", "="^70, "\nA. SEASON-BLOCKED CV (pooled OOS at bottom of each table)\n", "="^70)
ladder = [(:m0, F_M0), (:m1, F_M1), (:m2, F_M2)]
cv_summary = NamedTuple[]
for (name, f) in ladder, link in (:gamma, :lognormal)
    cv = blocked_cv(tr, f; link)
    R1[Symbol("cv_", name, "_", link)] = cv
    pooled = cv[cv.block .== "POOLED-OOS", :][1, :]
    push!(cv_summary, (model = name, link = link, r2 = pooled.r2,
                       mae = pooled.mae, spearman = pooled.spearman))
end
R1[:cv_summary] = DataFrame(cv_summary)
show(R1[:cv_summary], allrows=true); println()

# ==========================================
# B. Transfer gate on the best CV model
# ==========================================
println("\n", "="^70, "\nB. CHAMP↔PREM TRANSFER GATE\n", "="^70)
best = R1[:cv_summary][argmax(R1[:cv_summary].r2), :]
best_f = Dict(:m0 => F_M0, :m1 => F_M1, :m2 => F_M2)[best.model]
println("best by pooled-OOS R²: ", best.model, " / ", best.link)
R1[:transfer] = transfer_test(tr, best_f; link = best.link)
show(R1[:transfer], allrows=true); println()

# ==========================================
# C. Calibration of the winner (pooled OOS predictions)
# ==========================================
println("\n", "="^70, "\nC. DECILE CALIBRATION (winner, pooled OOS)\n", "="^70)
pooled_pred = similar(tr.xg, Union{Missing,Float64})
for (tid, seas) in sort(unique(collect(zip(tr.tournament_id, tr.season))))
    test = (tr.tournament_id .== tid) .& (tr.season .== seas)
    m = fit_proxy(tr[.!test, :], best_f; link = best.link)
    pooled_pred[test] = predict_xg(m, tr[test, :]; link = best.link)
end
R1[:calibration] = calibration_deciles(Float64.(tr.xg), Float64.(coalesce.(pooled_pred, NaN)))
show(R1[:calibration], allrows=true); println()

# ==========================================
# Freeze winner on ALL training data → artifact for WP2
# ==========================================
final_model = fit_proxy(tr, best_f; link = best.link)
R1[:final_model] = final_model
println("\nWinner coefficients (fit on all $(nrow(tr)) rows):")
println(coeftable(final_model))

artifact = (model = final_model, formula = string(best_f), link = best.link,
            stats = PROXY_STATS, trained = "2026-07-17",
            train_tiers = [54, 55], n_rows = nrow(tr))
serialize(joinpath(@__DIR__, "proxy_model_v1.jls"), artifact)
println("\n[INFO] frozen → proxy_model_v1.jls  (", best.model, "/", best.link, ")")
